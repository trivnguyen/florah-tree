
import os
import pickle
import shutil
import sys

import datasets
import ml_collections
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import yaml
from absl import flags, logging
from ml_collections import config_flags
from models import utils
from models.atg2 import AutoregTreeGen2

logging.set_verbosity(logging.INFO)

def train(
    config: ml_collections.ConfigDict, workdir: str = "./logging/"
):
    workdir = './debug'
    checkpoint_path = None
    logging.info("Debugging run at {}".format(workdir))
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir, exist_ok=True)

    # Save the configuration to a YAML file
    config_dict = config.to_dict()
    config_path = os.path.join(workdir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # load dataset and prepare dataloader
    logging.info("Preparing dataloader...")
    data = datasets.read_dataset(
        dataset_root=config.data.root,
        dataset_name=config.data.name,
        max_num_files=1,
    )
    data = data[:1000]
    train_loader, val_loader, norm_dict = datasets.prepare_dataloader(
        data,
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
    model_atg = AutoregTreeGen2(
        d_in=config.model.d_in,
        num_classes=config.model.num_classes,
        encoder_args=config.model.encoder,
        decoder_args=config.model.decoder,
        npe_args=config.model.npe,
        classifier_args=config.model.classifier,
        optimizer_args=config.optimizer,
        scheduler_args=config.scheduler,
        training_args=config.training,
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
            filename="best-{epoch}-{step}-{val_loss:.4f}",
            monitor=config.training.get('monitor', 'val_loss'),
            save_top_k=config.training.get('save_top_k', 1),
            mode='min',
            save_weights_only=False,
            save_last=False
        ),
        pl.callbacks.ModelCheckpoint(
            filename="last-{epoch}-{step}-{val_loss:.4f}",
            save_top_k=config.training.get('save_last_k', 1),
            monitor='epoch',
            mode='max',
            save_weights_only=False,
            save_last=False
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
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )

    # train the model
    logging.info("Training model...")
    pl.seed_everything(config.seed.training)
    trainer.fit(
        model_atg,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=None,
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
