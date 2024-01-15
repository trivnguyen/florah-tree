

import os
import h5py
import pickle
import time
import glob
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
from absl import flags
from ml_collections import config_dict, config_flags

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

def convert_scsam(config: config_dict.ConfigDict):
    ''' Convert the trees to SC-SAM format '''

    print('Converting trees to SC-SAM format ...')

    # create a work directory and copy the config file there
    workdir = Path(os.path.join(config.workdir, config.name))
    input_dir = os.path.join(workdir, "input")
    output_dir = os.path.join(workdir, "output/")  # the slash is IMPORTANT. DO NOT REMOVE

    # check if data_path is directory or file
    data_path = os.path.join(config.data_root, config.data_name)
    if os.path.isdir(data_path):
        print(f'Reading {config.num_max_files} files from {data_path}')
        trees = []
        tree_paths = glob.glob(os.path.join(data_path, f'*.pkl'))
        tree_paths = sorted(tree_paths, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for i in range(config.num_max_files):
            if os.path.exists(tree_paths[i]):
                with open(tree_paths[i], 'rb') as f:
                    trees += pickle.load(f)
    else:
        with open(data_path, 'rb') as f:
            trees = pickle.load(f)

    if isinstance(trees[0], nx.DiGraph):
        trees = [from_networkx(tree) for tree in trees]

    # remove trees with too many nodes
    trees = [tree for tree in trees if len(tree.x) <= config.num_max_nodes]

    trees = trees[:config.num_max_trees]

    # convert all trees to depth-first search order
    if not config.is_dfs:
        loop = tqdm(range(len(trees)))
        for itree in loop:
            loop.set_description(f'Converting to DFS order (tree {itree})')
            tree = trees[itree]
            order = analysis_utils.dfs(tree.edge_index)

            # convert edge_index to DFS orders
            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}
            edge_index_dfs = torch.tensor(
                [old_to_new[node.item()] for node in tree.edge_index.view(-1)])
            edge_index_dfs = edge_index_dfs.view(2, -1)

            tree.x = tree.x[order]
            tree.edge_index = edge_index_dfs

    # get the snapshot times of the box
    snaps, aexp_snaps, redshift_snaps = utils.read_snapshot_times(config.box_name)

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
    num_trees_total = len(forest)
    num_trees_per_file = config.num_trees_per_file
    num_files = int(np.ceil(num_trees_total / num_trees_per_file))
    for ifile in range(num_files):
        output_path = os.path.join(input_dir, f"trees_{ifile}.dat")
        with open(output_path, 'w') as f:
            num_trees_write = min(num_trees_per_file, num_trees_total - ifile * num_trees_per_file)
            f.writelines(f"{str(num_trees_write)} \n")
            loop = tqdm(range(num_trees_write))
            for itree in loop:
                loop.set_description(f'Writing to file (tree {itree})')
                f.writelines(f"#tree {itree}\n")
                for iline in range(forest.ptr[itree], forest.ptr[itree+1]):
                    string = to_string(ct_data[iline], PROPS_FMT, PROPS_TYPE)
                    f.writelines(string)

    print(f"Converted {num_trees_total} trees to ConsistentTrees format")
    print(f"Output directory: {input_dir}")
    print(f"Done!")

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to SC-SAM configuration.",
        lock_config=True,
    )
    # Parse flags
    FLAGS(sys.argv)

    # Start training run
    convert_scsam(config=FLAGS.config)
