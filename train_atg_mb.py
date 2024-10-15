from typing import List, Optional

import os
import pickle
import shutil
import sys

import datasets
import ml_collections
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import yaml
from absl import flags, logging
from ml_collections import config_flags
from models import utils, training_utils
from models.atg_mb import AutoregTreeGenMB

logging.set_verbosity(logging.INFO)

def prepare_dataloader_mb(
    data: List,
    train_frac: float = 0.8,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 0,
    norm_dict: dict = None,
    reverse_time: bool = False,
    seed: Optional[int] = None
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
    Returns:
        train_loader: PyTorch DataLoader for training.
        val_loader: PyTorch DataLoader for evaluation.
        norm_dict: dictionary containing normalization statistics.
    """

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

    print("Normalization statistics:")
    print("x_loc: {}".format(x_loc))
    print("x_scale: {}".format(x_scale))
    print("t_loc: {}".format(t_loc))
    print("t_scale: {}".format(t_scale))

    # get the maxmimum length
    max_length = max([d.x.shape[0] for d in data])
    mb_data, mb_len = training_utils.pad_sequences(
        [d.x for d in data], max_len=max_length, padding_value=0)
    train_dataset = TensorDataset(mb_data[:num_train], mb_len[:num_train])
    val_dataset = TensorDataset(mb_data[num_train:], mb_len[num_train:])

    # # create data loader
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, norm_dict


def train(
    config: ml_collections.ConfigDict, workdir: str = "./logging/"
):
    # set up work directory
    if not hasattr(config, "name"):
        name = utils.get_random_name()
    else:
        name = config["name"]
    logging.info("Starting training run {} at {}".format(name, workdir))

    workdir = os.path.join(workdir, name)
    checkpoint_path = None
    if os.path.exists(workdir):
        if config.overwrite:
            shutil.rmtree(workdir)
        elif config.get('checkpoint', None) is not None:
            checkpoint_path = os.path.join(
                workdir, 'lightning_logs/checkpoints', config.checkpoint)
        else:
            raise ValueError(
                f"Workdir {workdir} already exists. Please set overwrite=True "
                "to overwrite the existing directory.")

    os.makedirs(workdir, exist_ok=True)

    # Save the configuration to a YAML file
    config_dict = config.to_dict()
    config_path = os.path.join(workdir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # load dataset and prepare dataloader
    logging.info("Preparing dataloader...")
    train_loader, val_loader, norm_dict = prepare_dataloader_mb(
        datasets.read_dataset(
            dataset_root=config.data.root,
            dataset_name=config.data.name,
            max_num_files=config.data.get("num_files", 1),
        ),
        train_frac=config.data.train_frac,
        train_batch_size=config.training.train_batch_size,
        eval_batch_size=config.training.eval_batch_size,
        num_workers=config.training.get("num_workers", 0),
        seed=config.seed.data,
        norm_dict=None,
        reverse_time=config.data.get("reverse_time", False),
    )

    # create the model
    logging.info("Creating model...")
    model_atg = AutoregTreeGenMB(
        d_in=config.model.d_in,
        encoder_args=config.model.encoder,
        npe_args=config.model.npe,
        optimizer_args=config.optimizer,
        scheduler_args=config.scheduler,
        norm_dict=norm_dict,
    )
    # Create call backs and trainer objects
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor=config.training.get('monitor', 'val_loss'),
            patience=config.training.get('patience', 1000),
            mode='min',
            verbose=True
        ),
        pl.callbacks.ModelCheckpoint(
            filename="{epoch}-{step}-{val_loss:.4f}",
            monitor=config.training.get('monitor', 'val_loss'),
            save_top_k=config.training.get('save_top_k', 1),
            mode='min',
            save_weights_only=False,
            save_last=True
        ),
        pl.callbacks.LearningRateMonitor("step"),
    ]
    train_logger = pl_loggers.TensorBoardLogger(workdir, version='')
    trainer = pl.Trainer(
        default_root_dir=workdir,
        max_epochs=config.training.max_epochs,
        max_steps=config.training.max_steps,
        accelerator=config.accelerator,
        callbacks=callbacks,
        logger=train_logger,
        gradient_clip_val=config.training.get('gradient_clip_val', 0),
        enable_progress_bar=config.get("enable_progress_bar", True),
        num_sanity_val_steps=0,
    )

    # train the model
    logging.info("Training model...")
    pl.seed_everything(config.seed.training)
    trainer.fit(
        model_atg,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=checkpoint_path,
    )

    # Done
    logging.info("Done!")


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
