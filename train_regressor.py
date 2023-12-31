
import os
import pickle
import sys

import ml_collections
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
import torch_geometric
from absl import flags, logging
from ml_collections import config_flags
from models import models, regressor, training_utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx, to_networkx


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


def train(
    config: ml_collections.ConfigDict, workdir: str = "./logging/"
):
    # load dataset
    data_path = os.path.join(config.data_root, config.data_name + ".pkl")
    logging.info("Loading data from {}...".format(data_path))

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    data = [from_networkx(d) for d in data]

    # prepare dataloader
    logging.info("Preparing dataloader...")
    train_loader, val_loader, norm_dict = prepare_dataloader(
        data, config, norm_dict=None)

    # create model
    # logging.info("Creating model...")
    model = regressor.SequenceRegressor(
        input_size=config.input_size,
        featurizer_args=config.featurizer,
        rnn_args=config.rnn,
        flows_args=config.flows,
        optimizer_args=config.optimizer,
        scheduler_args=config.scheduler,
        d_time = config.d_time,
        d_time_projection = config.d_time_projection,
        d_feat_projection = config.d_feat_projection,
        num_samples_per_graph=config.num_samples_per_graph,
        norm_dict=norm_dict,
    )

    # create the trainer object
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor=config.monitor, patience=config.patience, mode=config.mode,
            verbose=True),
        pl.callbacks.ModelCheckpoint(
            monitor=config.monitor, save_top_k=config.save_top_k,
            mode=config.mode, save_weights_only=False),
        pl.callbacks.LearningRateMonitor("epoch"),
    ]
    train_logger = pl_loggers.TensorBoardLogger(workdir, name=config.name)
    trainer = pl.Trainer(
        default_root_dir=workdir,
        max_epochs=config.num_epochs,
        max_steps=config.num_steps,
        accelerator=config.accelerator,
        callbacks=callbacks,
        logger=train_logger,
        enable_progress_bar=config.get("enable_progress_bar", True),
    )

    # train the model
    logging.info("Training model...")
    trainer.fit(model, train_loader, val_loader)


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
    train(config=FLAGS.config, workdir=FLAGS.config.workdir)
