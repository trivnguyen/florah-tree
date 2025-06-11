from typing import List, Optional, Tuple, Union

import os
import pickle

import ml_collections
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx, to_networkx

def read_dataset(
    dataset_name: str, dataset_root: str, max_num_files: int = 1,
    index_start: int=0, verbose: bool = True, prefix: str = "data",
    ignore_missing: bool = False
):
    data = []
    data_dir = os.path.join(dataset_root, dataset_name)
    for i in range(index_start, index_start + max_num_files):
        data_path = os.path.join(data_dir, "{}.{}.pkl".format(prefix, i))
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data += pickle.load(f)
            if verbose:
                print("Loading data from {}...".format(data_path))
        else:
            if verbose:
                print("Data file {} not found. Stopping".format(data_path))
            if ignore_missing:
                continue
            break
    if len(data) == 0:
        raise ValueError("No data found in the specified directory.")

    if not isinstance(data[0], Data):
        data = [from_networkx(d) for d in data]
    return data

def create_sampler(
    data: List[Data],
    bins: Union[int, Tuple] = 10,
    weighting_scheme: str = "inverse_frequency",
    replacement: bool = True,
):
    """ Create a WeightedRandomSampler for the dataset.
    By default, use the root mass as the label
    """

    print('Creating sampler with {} bins'.format(bins))
    print('Weighting scheme: {}'.format(weighting_scheme))
    print('Replacement: {}'.format(replacement))

    # Bins the labels, by default use the root mass as the label
    labels = np.array([d.x[0, 0].item() for d in data])
    hist, bin_edges = np.histogram(labels, bins=bins)
    bin_indices = np.digitize(labels, bin_edges[:-1])

    # Compute class weights (inverse frequency)
    counts = np.bincount(bin_indices)
    if weighting_scheme == "inverse_frequency":
        class_weights = np.where(counts > 0, 1.0 / counts, 0)
    elif weighting_scheme == "inverse_square_root_frequency":
        class_weights = np.where(counts > 0, 1.0 / np.sqrt(counts), 0)
    elif weighting_scheme == "inverse_log_frequency":
        class_weights = 1.0 / np.log1p(counts + 1)
    else:
        raise ValueError("Unknown weighting scheme: {}".format(
            weighting_scheme))
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    print("Bin edges: {}".format(bin_edges))
    print("Bin counts: {}".format(hist))
    print('Weights: {}'.format(class_weights))

    # Calculate sample weights
    sample_weights = torch.tensor([class_weights[bin_idx] for bin_idx in bin_indices])
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=replacement
    )

    return sampler


def prepare_dataloader(
    data: List,
    train_frac: float = 0.8,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 0,
    norm_dict: dict = None,
    reverse_time: bool = False,
    seed: Optional[int] = None,
    use_sampler: bool = False,
    sampler_args: dict = None,
):
    """
    Prepare the dataloader for training and evaluation.
    Args:
        data: list of PyTorch Geometric Data objects.
        train_frac: fraction of the data to use for training.
        train_batch_size: batch size for training.
        eval_batch_size: batch size for evaluation.
        num_workers: number of workers for data loading.
        norm_dict: dictionary containing normalization statistics.
        reverse_time: whether to reverse the time axis
        seed: random seed for shuffling the data.
        use_sampler: whether to use a sampler for the training set.
        sampler_args: arguments for the sampler.
    Returns:
        train_loader: PyTorch DataLoader for training.
        val_loader: PyTorch DataLoader for evaluation.
        norm_dict: dictionary containing normalization statistics.
    """

    rng = np.random.default_rng(seed)

    # create a sampler for the training set
    # because the weight is based on root mass, this needs to be done before
    # the normalization
    if use_sampler:
        # apply root cut on the data
        mroot = np.array([d.x[0, 0].item() for d in data])
        mroot_min, m_root_max = min(sampler_args.bins), max(sampler_args.bins)
        data = [data[i] for i in range(len(mroot)) if mroot_min <= mroot[i] <= m_root_max]
        rng.shuffle(data)
        num_total = len(data)
        num_train = int(num_total * train_frac)

        sampler = create_sampler(data[:num_train], **sampler_args)
        shuffle = False
    else:
        rng.shuffle(data)
        num_total = len(data)
        num_train = int(num_total * train_frac)

        sampler = None
        shuffle = True

    # calculate the normaliziation statistics
    if norm_dict is None:
        x = torch.cat([d.x[..., :-1] for d in data[:num_train]])
        t = torch.cat([d.x[..., -1:] for d in data[:num_train]])

        # standardize the input features and min-max normalize the time
        # x_loc = x.mean(dim=0)
        # x_scale = x.std(dim=0)

        # normalize x such that it is in range [-1, 1]
        x_min = x.min(dim=0).values
        x_max = x.max(dim=0).values
        x_loc = (x_min + x_max) / 2
        x_scale = (x_max - x_min) / 2

        t_loc = t.min()
        t_scale = t.max() - t_loc
        if reverse_time:
            t_loc = t_scale + t_loc
            t_scale = -t_scale

        norm_dict = {
            "x_loc": list(x_loc.numpy()),
            "x_scale": list(x_scale.numpy()),
            "t_loc": t_loc.numpy(),
            "t_scale": t_scale.numpy(),
        }
    else:
        x_loc = torch.tensor(norm_dict["x_loc"], dtype=torch.float32)
        x_scale = torch.tensor(norm_dict["x_scale"], dtype=torch.float32)
        t_loc = torch.tensor(norm_dict["t_loc"], dtype=torch.float32)
        t_scale = torch.tensor(norm_dict["t_scale"], dtype=torch.float32)
    for d in data:
        d.x[..., :-1] = (d.x[..., :-1] - x_loc) / x_scale
        d.x[..., -1:] = (d.x[..., -1:] - t_loc) / t_scale

    print("Number of training samples: {}".format(num_train))
    print("Number of validation samples: {}".format(num_total - num_train))

    print("Normalization statistics:")
    print("x_loc: {}".format(x_loc))
    print("x_scale: {}".format(x_scale))
    print("t_loc: {}".format(t_loc))
    print("t_scale: {}".format(t_scale))

    # create data loader
    train_loader = DataLoader(
        data[:num_train], batch_size=train_batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        data[num_train:], batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    return train_loader, val_loader, norm_dict
