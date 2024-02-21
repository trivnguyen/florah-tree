
import os
import sys
import pickle

import numpy as np
import torch
import ml_collections
from torch_geometric.utils import from_networkx
from absl import flags, logging
from ml_collections import config_flags
from tqdm import tqdm

import datasets, analysis
from models import infer_utils
from models.classifier import SequenceClassifier
from models.regressor import SequenceRegressor, SequenceRegressorMultiFlows
from models.generator import TreeGenerator

logging.set_verbosity(logging.INFO)

def get_checkpoint(logdir, name, checkpoint):
    """ Get the checkpoint path """
    return os.path.join(logdir, name, "lightning_logs/checkpoints", checkpoint)


def infer(config: ml_collections.ConfigDict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.data_fmt == 'pickle':

        # Load in the dataset and extract the root features and times
        sim_data = datasets.read_dataset(
            dataset_name=config.data_name,
            dataset_root=config.data_root,
            max_num_files=config.get("max_num_files", 100),
        )

        # get the root features and times
        if config.fixed_time_steps:
            logging.info(
                "Using fixed time steps. Will assume all trees have the same time steps.")

            root_features = [sim_data[i].x[0, :-1] for i in range(len(sim_data))]

            # assume the time to be all the same
            sim_main_index = analysis.get_main_branch(
                sim_data[0].x[:, 0], sim_data[0].edge_index)
            sim_aexp = sim_data[0].x[sim_main_index, -1]
            times_out = [sim_aexp] * config.num_trees_per_sim * len(sim_data)
        else:
            logging.info(
                "Using variable time steps. Will assume each tree has different time steps.")

            root_features = []
            times_out = []

            # create progress bar
            loop = tqdm(
                range(len(sim_data)),
                desc="Loading the roots and main branches of the simulation")
            for i in loop:
                sim_tree = sim_data[i]
                sim_root_feat = sim_tree.x[0, :-1]
                sim_aexp = torch.unique(sim_tree.x[:, -1]).flip(0)
                root_features = root_features + [sim_root_feat] * config.num_trees_per_sim
                times_out = times_out + [sim_aexp] * config.num_trees_per_sim

    # elif config.data_fmt == 'txt':
    #     # read the snapshot times
    #     times_out = np.genfromtxt(
    #         os.path.join(config.data_root, config.data_name, "snapshot_times.txt"))
    #     data_path = os.path.join(
    #         config.data_root, config.data_name, "data.{}.txt".format(i))
    #     root_features = np.genfromtxt(data_path)

    #     # convert to torch tensors
    #     root_features = torch.tensor(root_features, dtype=torch.float32)
    #     times_out = torch.tensor(times_out, dtype=torch.float32)
    #     times_out = times_out.unsqueeze(0).repeat(len(root_features), 1)

    # Read in the model
    if config.mode.upper() == 'A':
        checkpoint_path = get_checkpoint(
            config.logdir, config.classifier.name, config.classifier.checkpoint)
        classifier =  SequenceClassifier.load_from_checkpoint(
            checkpoint_path, map_location=device)
        checkpoint_path = get_checkpoint(
            config.logdir, config.regressor.name, config.regressor.checkpoint)

        if config.multiflow:
            regressor = SequenceRegressorMultiFlows.load_from_checkpoint(
                checkpoint_path, map_location=device)
        else:
            regressor = SequenceRegressor.load_from_checkpoint(
                checkpoint_path, map_location=device)

        generate_args = dict(
            mode='A',
            classifier=classifier,
            regressor=regressor,
            classifier_norm_dict=classifier.norm_dict,
            regressor_norm_dict=regressor.norm_dict,
            n_max_iter=config.num_max_iter,
            multiflow=config.multiflow,
            device=device,
            verbose=config.verbose,
        )
    elif config.mode.upper() == 'B':
        checkpoint_path = get_checkpoint(
            config.logdir, config.generator.name, config.generator.checkpoint)
        generator = TreeGenerator.load_from_checkpoint(
            checkpoint_path, map_location=device)
        generate_args = dict(
            mode='B',
            generator=generator,
            norm_dict=generator.norm_dict,
            n_max_iter=config.num_max_iter,
            device=device,
            verbose=config.verbose,
        )
    else:
        raise ValueError("Invalid mode: {}".format(config.mode))

    # Generate trees
    # Divide the data into jobs then batches
    num_roots = len(root_features)
    num_roots_per_job = int(np.ceil(num_roots / config.num_jobs))
    job_start = config.job_id * num_roots_per_job
    job_end = min((config.job_id + 1) * num_roots_per_job, num_roots)
    output_start = int(np.ceil(job_start / config.batch_size))

    print('Total number of roots: ', num_roots)
    print('Number of roots per job: ', num_roots_per_job)
    print('Job start: ', job_start)
    print('Job end: ', job_end)
    print('Output start: ', output_start)

    root_features = root_features[job_start: job_end]
    times_out = times_out[job_start: job_end]

    num_batches = int(np.ceil(len(root_features) / config.batch_size))

    for i in range(num_batches):
        # Save the trees
        output_path = os.path.join(
            config.output_root, config.output_name, "data.{}.pkl".format(i + output_start))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if config.resume:
            if os.path.exists(output_path):
                logging.info("Skipping batch {}/{}...".format(i + 1, num_batches))
                continue
        logging.info("Generating batch {}/{}...".format(i + 1, num_batches))
        batch_root_features = root_features[i * config.batch_size: (i + 1) * config.batch_size]
        batch_times_out = times_out[i * config.batch_size: (i + 1) * config.batch_size]

        trees = infer_utils.generate_forest(
            root_features=batch_root_features,
            times_out=batch_times_out,
            **generate_args
        )
        logging.info("Saving trees to {}...".format(output_path))
        with open(output_path, 'wb') as f:
            pickle.dump(trees, f)


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
