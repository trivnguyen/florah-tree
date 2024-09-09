from typing import List, Optional, Tuple

import os
import pickle

import ml_collections
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx, to_networkx

def read_dataset(
    dataset_name: str, dataset_root: str, max_num_files: int = 1,
    verbose: bool = True
):
    data = []
    data_dir = os.path.join(dataset_root, dataset_name)
    for i in range(max_num_files):
        data_path = os.path.join(data_dir, "data.{}.pkl".format(i))
        if verbose:
            print("Loading data from {}...".format(data_path))
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data += pickle.load(f)
        else:
            break
    if not isinstance(data[0], Data):
        data = [from_networkx(d) for d in data]
    return data


def prepare_dataloader(
    data: List, train_frac: float = 0.8, train_batch_size: int = 32,
    eval_batch_size: int = 32, num_workers: int = 0, norm_dict: dict = None,
    seed: Optional[int] = None
):
    rng = np.random.default_rng(seed)
    rng.shuffle(data)

    num_total = len(data)
    num_train = int(num_total * train_frac)

    # calculate the normaliziation statistics
    if norm_dict is None:
        x = torch.cat([d.x[..., :-1] for d in data[:num_train]])
        t = torch.cat([d.x[..., -1:] for d in data[:num_train]])

        # standardize the input features and min-max normalize the time
        x_loc = x.mean(dim=0)
        x_scale = x.std(dim=0)
        t_loc = t.min()
        t_scale = t.max() - t_loc

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

    print("Normalization statistics:")
    print("x_loc: {}".format(x_loc))
    print("x_scale: {}".format(x_scale))
    print("t_loc: {}".format(t_loc))
    print("t_scale: {}".format(t_scale))

    # create data loader
    train_loader = DataLoader(
        data[:num_train], batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(
        data[num_train:], batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, norm_dict
