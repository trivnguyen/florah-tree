
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
    data: list, config: ml_collections.ConfigDict,
    norm_dict: dict = None
):
    train_frac = config.train_frac
    train_batch_size = config.train_batch_size
    eval_batch_size = config.eval_batch_size
    num_workers = config.num_workers

    np.random.shuffle(data) # shuffle the data

    num_total = len(data)
    num_train = int(num_total * train_frac)

    # calculate the normaliziation statistics
    if norm_dict is None:
        x = torch.cat([d.x for d in data[:num_train]])
        x_loc = x.mean(dim=0)
        x_scale = x.std(dim=0)
        norm_dict = {
            "x_loc": list(x_loc.numpy()),
            "x_scale": list(x_scale.numpy()),
        }
    else:
        x_loc = torch.tensor(norm_dict["x_loc"], dtype=torch.float32)
        x_scale = torch.tensor(norm_dict["x_scale"], dtype=torch.float32)
    for d in data:
        d.x = (d.x - x_loc) / x_scale

    # create data loader
    train_loader = DataLoader(
        data[:num_train], batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(
        data[num_train:], batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, norm_dict
