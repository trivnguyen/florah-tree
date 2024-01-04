
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

from analysis import analysis_utils
from models import infer_utils
from models.classifier import SequenceClassifier
from models.regressor import SequenceRegressor
from models.generator import TreeGenerator

def infer(config: ml_collections.ConfigDict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load in the dataset and extract the root features and times
    data_path = os.path.join(config.data_root, config.data_name + ".pkl")
    logging.info("Loading data from {}...".format(data_path))

    with open(data_path, 'rb') as f:
        sim_data = pickle.load(f)
    sim_data = [from_networkx(d) for d in sim_data]

    root_features = []
    times_out = []

    # create progress bar
    loop = tqdm(len(sim_data), desc="Generating trees")
    for i in loop:
        sim_tree = sim_data[i]
        sim_main_index = analysis_utils.get_main_branch(
            sim_tree.x[:, 0], sim_tree.edge_index)
        sim_main_feats = sim_tree.x[sim_main_index]
        sim_aexp = sim_main_feats[..., -1]
        sim_root_feat = sim_tree.x[0, :-1]

        root_features = root_features + [sim_root_feat] * config.num_trees_per_sim
        times_out = times_out + [sim_aexp] * config.num_trees_per_sim

    # Read in the model
    if config.mode.upper() == 'A':
        classifier =  SequenceClassifier.load_from_checkpoint(
            config.classifier_path, map_location=device)
        regressor = SequenceRegressor.load_from_checkpoint(
            config.regressor_path, map_location=device)
        classifier_norm_dict = classifier.norm_dict
        regressor_norm_dict = regressor.norm_dict
        generate_args = dict(
            classifier=classifier, regressor=regressor,
            classifier_norm_dict=classifier_norm_dict,
            regressor_norm_dict=regressor_norm_dict,
            device=device, mode='A', n_max_iter=config.num_max_iter,
            verbose=config.verbose,
        )
    elif config.mode.upper() == 'B':
        generator = TreeGenerator.load_from_checkpoint(
            config.generator_path, map_location=device)
        norm_dict = generator.norm_dict
        generate_args = dict(
            generator=generator, norm_dict=norm_dict,
            device=device, mode='B', n_max_iter=config.num_max_iter,
            verbose=config.verbose,
        )
    else:
        raise ValueError("Invalid mode: {}".format(config.mode))

    # Generate trees
    num_batches = int(np.ceil(len(root_features) // config.batch_size))
    for i in range(num_batches):
        logging.info("Generating batch {}/{}...".format(i + 1, num_batches))
        batch_root_features = root_features[i * config.batch_size: (i + 1) * config.batch_size]
        batch_times_out = times_out[i * config.batch_size: (i + 1) * config.batch_size]

        trees = infer_utils.generate_forest(
            root_features=batch_root_features,
            times_out=batch_times_out,
            **generate_args
        )

        # Save the trees
        output_path = os.path.join(
            config.output_root, config.output_name, "trees_{}.pkl".format(i))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
