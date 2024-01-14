

import os
import h5py
import pickle
import time
import sys
from pathlib import Path

sys.path.append('../')

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch_geometric
from tqdm import tqdm
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.data import Data, Batch
from ml_collections import config_dict

from florah_analysis import utils
from analysis import analysis_utils


# ConsistentTree settings for SC-SAM
# DO NOT CHANGE UNLESS YOU WANT TO CHANGE THE SC-SAM CODE
PROPS_ORDER = (
    'scale_factor', 'snapshot', 'halo_id', 'root_halo_id', 'original_halo_id',
    'num_progenitors', 'mvir', 'cvir', 'X', 'Y', 'Z', 'vX', 'vY', 'vZ')
PROPS_TYPE = (
    float, int, int, int, int, int, float, float,
    float, float, float, float, float, float)
PROPS_FMT = (
    '%f', '%d', '%d', '%d', '%d', '%d', '%e', '%f',
    '%f', '%f', '%f', '%f', '%f', '%f')


def to_string(data, fmts, types):
    string = ''
    new_data = []
    for i in range(len(data)):
        new_data.append(types[i](data[i]))
    for i in range(len(data)):
        string += '{:' + fmts[i][1:] + '} '
    string += '\n'
    return string.format(*new_data)


if __name__ == '__main__':

    # settings
    box_name = 'vsmdpl'
    output_root = Path('/mnt/ceph/users/tnguyen/florah/sc-sam/florah-tree/vsmdpl-nanc2-B')
    output_name = 'gen'
    data_path = '/mnt/ceph/users/tnguyen/florah/generated_dataset/fixed-time-steps/VSMDPL-Nanc2-B'
    # data_path = '/mnt/ceph/users/tnguyen/florah/datasets/experiments/fixed-time-steps/VSMDPL-Nanc2'
    num_max_files = 10
    num_max_nodes = 5000
    num_max_trees = 10000
    prefix = 'trees_'
    is_dfs = False
    os.makedirs(output_root / "input", exist_ok=True)
    os.makedirs(output_root / "output", exist_ok=True)

    # check if data_path is directory or file
    if os.path.isdir(data_path):
        print(f'Reading {num_max_files} files from {data_path}')
        trees = []
        for i in range(num_max_files):
            tree_path = os.path.join(data_path, f'{prefix}{i}.pkl')
            if os.path.exists(tree_path):
                with open(tree_path, 'rb') as f:
                    trees += pickle.load(f)
    else:
        with open(data_path, 'rb') as f:
            trees = pickle.load(f)
    trees = trees[:num_max_trees]

    if isinstance(trees[0], nx.DiGraph):
        trees = [from_networkx(tree) for tree in trees]

    # remove trees with too many nodes
    trees = [tree for tree in trees if len(tree.x) <= num_max_nodes]

    # convert all trees to depth-first search order
    if not is_dfs:
        loop = tqdm(range(len(trees)), desc='Converting to DFS order')
        for itree in loop:
            loop.set_description(f'Converting to DFS order (tree {itree})')
            tree = trees[itree]
            edge_index = tree.edge_index
            order = analysis_utils.dfs(edge_index)
            tree.x = tree.x[order]
            tree.edge_index = torch.tensor(
                [[order.index(i) for i in edge_index[0]], [order.index(i) for i in edge_index[1]]])

    # get the snapshot times of the box
    snaps, aexp_snaps, redshift_snaps = utils.read_snapshot_times(box_name)

    # Start converting to ConsistentTree format
    # use PyG Batch to store all trees, very convenient for this application
    forest = Batch.from_data_list(trees)

    trees_data = {}

    trees_data['scale_factor'] = forest.x[:, 2].numpy()

    # assign snapshot number to each halo
    trees_data['snapshot'] = np.zeros(len(forest.x), dtype=int)
    for i in range(len(forest.x)):
        trees_data['snapshot'][i] = np.where(
            np.isclose(trees_data['scale_factor'][i], aexp_snaps, atol=1e-3))[0][0]
        trees_data['snapshot'][i] = snaps[0] - trees_data['snapshot'][i]

    trees_data['mvir'] = 10**forest.x[:, 0].numpy()
    trees_data['cvir'] = 10**forest.x[:, 1].numpy()
    trees_data['halo_id'] = np.arange(len(forest.x))
    trees_data['root_halo_id'] = forest.batch
    trees_data['original_halo_id'] = trees_data['halo_id'].copy()

    # get number of connections
    trees_data['num_progenitors'] = np.zeros(len(forest.x), dtype=int)
    for i in range(len(forest.x)):
        trees_data['num_progenitors'][i] = len(forest.edge_index[1][forest.edge_index[0] == i])

    # set position and velocity to zeros since FLORAH can't generate positions (yet?)
    trees_data['X'] = np.zeros(len(forest.x))
    trees_data['Y'] = np.zeros(len(forest.x))
    trees_data['Z'] = np.zeros(len(forest.x))
    trees_data['vX'] = np.zeros(len(forest.x))
    trees_data['vY'] = np.zeros(len(forest.x))
    trees_data['vZ'] = np.zeros(len(forest.x))

    # Convert everything into one big numpy array in the correct order
    ct_data = []
    for props in PROPS_ORDER:
        ct_data.append(trees_data[props])
    ct_data = np.stack(ct_data, axis=1)

    # Write to file
    num_trees_write = len(forest)
    output_path = output_root / "input" / f"{output_name}.dat"
    with open(output_path, 'w') as f:
        f.writelines(f"{str(num_trees_write)} \n")
        iline = 0

        loop = tqdm(range(num_trees_write), desc='Writing to file')
        for itree in loop:
            loop.set_description(f'Writing to file (tree {itree})')
            f.writelines(f"#tree {itree}\n")
            for iline in range(forest.ptr[itree], forest.ptr[itree+1]):
                string = to_string(ct_data[iline], PROPS_FMT, PROPS_TYPE)
                f.writelines(string)
