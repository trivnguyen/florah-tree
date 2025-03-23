
from typing import List, Tuple, Dict, Any, Optional

import os
import sys
import pickle
import glob
import re

sys.path.append('/mnt/home/tnguyen/projects/florah/florah-tree')

import numpy as np
import torch
import pytorch_lightning as pl
import ml_collections
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch.utils.data import DataLoader, TensorDataset
from absl import flags, logging
from ml_collections import config_flags
from tqdm import tqdm

import datasets
from models import infer_utils, training_utils, models_utils, analysis_utils
from models.atg import AutoregTreeGen

DEFAULT_METADATA_DIR = "/mnt/ceph/users/tnguyen/florah-tree/metadata"

def read_snapshot_times(box_name):
    """ Read in snapshot times from the simulations """
    default_dir = DEFAULT_METADATA_DIR
    if "GUREFT" in box_name:
        table_name = "snapshot_times_gureft.txt"
    else:
        table_name = "snapshot_times_{}.txt".format(box_name.lower())

    snapshot_times = np.genfromtxt(
        os.path.join(default_dir, table_name), delimiter=',', unpack=True)
    return snapshot_times

def generated_seeds(seed: int, num_jobs: int) -> List[List[int]]:
    """Generate a list of seeds for each job."""
    np.random.seed(seed)
    seeds = np.random.randint(0, 2**32 - 1, size=num_jobs).tolist()
    return seeds

def infer(config: ml_collections.ConfigDict):

    device = torch.device("cpu")

    # Load the model
    if config.checkpoint_infer in ['best', 'last']:
        # if checkpoint is not given, take the best checkpoint
        all_checkpoints = sorted(glob.glob(os.path.join(
            config.workdir, config.name, "lightning_logs/checkpoints/*")))
        val_losses = []
        steps = []
        for cp in all_checkpoints:
            print(cp)
            loss_match = re.search(r"val_loss=([-+]?\d*\.\d+|\d+)", cp)
            val_losses.append(float(loss_match.group(1)))
            step_match = re.search(r"step=(\d+)", cp)
            steps.append(int(step_match.group(1)))
        print(val_losses)
        print(steps)
        if config.checkpoint_infer == 'best':
            checkpoint_path = all_checkpoints[np.argmin(val_losses)]
        else:
            checkpoint_path = all_checkpoints[np.argmax(steps)]
    else:
        # check if checkpoint is absolute path
        if os.path.isabs(config.checkpoint_infer):
            checkpoint_path = config.checkpoint_infer
        else:
            checkpoint_path = os.path.join(
                config.workdir, config.name, "lightning_logs/checkpoints", config.checkpoint_infer)
        if not os.path.exists(checkpoint_path):
            print("Available checkpoints:")
            for cp in all_checkpoints:
                all_checkpoints = sorted(glob.glob(os.path.join(
                    config.workdir, config.name, "lightning_logs/checkpoints/*")))
                print(cp.split('/')[-1])
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")

    print(f'Loading model from checkpoint {checkpoint_path}')
    model = AutoregTreeGen.load_from_checkpoint(checkpoint_path, map_location=device).eval()

    # get the root features from the root data
    sim_data = datasets.read_dataset(
        dataset_name=config.data_infer.name,
        dataset_root=config.data_infer.root,
        index_start=config.data_infer.index_file_start,
        max_num_files=config.data_infer.num_files,
    )
    num_sim = len(sim_data)

    # read in the snapshot times from the simulation and get the times we want to infer
    snap_table, aexp_table, z_table = read_snapshot_times(config.data_infer.box)
    select = z_table <= config.data_infer.zmax
    snap_times_out = snap_table[select][::config.data_infer.step]
    aexp_times_out = aexp_table[select][::config.data_infer.step]
    zred_times_out = z_table[select][::config.data_infer.step]

    # create a tensor array for times
    x0 = torch.stack([sim_data[i].x[0, :-1] for i in range(num_sim)], dim=0)
    Zhist = torch.tensor(aexp_times_out, dtype=torch.float32).unsqueeze(1)
    snapshot_list = torch.tensor(snap_times_out, dtype=torch.long)

    # multiplicative factor
    print(x0.shape)
    x0 = x0.repeat(config.data_infer.multiplicative_factor, 1)
    print(x0[:1000])

    # job division
    job_size = len(x0) // config.data_infer.num_job
    start = job_size * config.data_infer.job_id
    end = job_size * (config.data_infer.job_id + 1)
    if config.data_infer.job_id == config.data_infer.num_job - 1:
        end = len(x0)
    x0 = x0[start:end]

    # Start generating tree
    # generate seed list
    seed = generated_seeds(config.seed.inference, config.data_infer.num_job)[config.data_infer.job_id]
    pl.seed_everything(seed)

    tree_list = infer_utils.generate_forest(
        model, x0, Zhist, norm_dict=model.norm_dict, device=device,
        batch_size=config.data_infer.batch_size, sort=True, snapshot_list=snapshot_list, verbose=True)

    # Write to file
    os.makedirs(config.data_infer.outdir, exist_ok=True)
    outfile = os.path.join(
        config.data_infer.outdir, f'halos.{config.data_infer.job_id}.pkl')

    print(outfile)
    with open(outfile, 'wb') as f:
        pickle.dump(tree_list, f)


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training or sampling hyperparameter configuration.",
        lock_config=True,
    )
    # Parse flags
    FLAGS(sys.argv)

    # Start training run
    infer(config=FLAGS.config)
