import os
import pickle
import shutil
import sys

import datasets
import ml_collections
import numpy as np
import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import yaml
from absl import flags, logging
from ml_collections import config_flags
from models import utils
from models.atg import AutoregTreeGen
from models.atg3 import AutoregTreeGen3

logging.set_verbosity(logging.INFO)

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
            if not os.path.isabs(config.checkpoint):
                checkpoint_path = os.path.join(
                    workdir, 'lightning_logs/checkpoints', config.checkpoint)
            else:
                checkpoint_path = config.checkpoint
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

    # Configure environment for distributed training
    if "SLURM_JOB_ID" in os.environ:
        logging.info("Running in SLURM environment")
        # Set environment variables for stable distributed training
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

        # Get SLURM info if available
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
        gpus_per_node = int(os.environ.get("SLURM_GPUS_PER_NODE", 1))
        node_id = int(os.environ.get("SLURM_NODEID", 0))

        logging.info(f"SLURM configuration: {num_nodes} nodes, {gpus_per_node} GPUs per node, node ID: {node_id}")
    else:
        num_nodes = 1
        gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # load dataset and prepare dataloader
    logging.info("Preparing dataloader...")
    train_loader, val_loader, norm_dict = datasets.prepare_dataloader(
        datasets.read_dataset(
            dataset_root=config.data.root,
            dataset_name=config.data.name,
            max_num_files=config.data.get("num_files", 1),
        ),
        train_frac=config.data.train_frac,
        train_batch_size=config.training.train_batch_size,
        eval_batch_size=config.training.eval_batch_size,
        use_sampler=config.training.get("use_sampler", False),
        sampler_args=config.training.get("sampler_args"),
        num_workers=config.training.get("num_workers", 0),
        seed=config.seed.data,
        norm_dict=None,
        reverse_time=config.data.get("reverse_time", False),
    )

    # create the model
    logging.info("Creating model...")
    if config.model.get("name", "atg") in ("atg", "atg2"):
        logging.info("Creating ATG model.")
        model_atg = AutoregTreeGen(
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
    elif config.model.get("name", "atg") == "atg3":
        logging.info("Creating ATG3 model.")
        model_atg = AutoregTreeGen3(
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

    # Distributed training configuration
    # Default to DDP strategy for multi-GPU training
    strategy = "auto"
    if gpus_per_node > 1 or num_nodes > 1:
        strategy = "ddp"
        logging.info(f"Using {strategy} strategy for distributed training")

    # Define the trainer with distributed settings
    trainer = pl.Trainer(
        default_root_dir=workdir,
        max_epochs=config.training.max_epochs,
        max_steps=config.training.max_steps,
        accelerator=config.accelerator,
        devices="auto",  # Let Lightning automatically detect available devices
        num_nodes=num_nodes,
        strategy=strategy,
        callbacks=callbacks,
        logger=train_logger,
        gradient_clip_val=config.training.get('gradient_clip_val', 0),
        enable_progress_bar=config.get("enable_progress_bar", True),
        num_sanity_val_steps=0,
        # Add sync_batchnorm for multi-GPU training if batch norm is used
        sync_batchnorm=True if (gpus_per_node > 1 or num_nodes > 1) else False,
        # Avoid memory issues with persistent workers if using multiple GPUs
        persistent_workers=config.training.get("persistent_workers", True) if gpus_per_node > 1 else False,
    )

    # train the model
    pl.seed_everything(config.seed.training, workers=True)  # Set workers=True for reproducibility in distributed setting
    logging.info("Training model...")
    logging.info(f"Training with seed {config.seed.training}.")
    logging.info(f"Training for {config.training.max_epochs} epochs.")
    logging.info(f"Training for {config.training.max_steps} steps.")
    logging.info(f"Training with batch size {config.training.train_batch_size}.")
    logging.info(f"Training with eval batch size {config.training.eval_batch_size}.")
    logging.info(f"Training with gradient clip value {config.training.get('gradient_clip_val', 0)}.")
    logging.info('Freezing settings')
    logging.info(f"Freezing encoder: {config.training.freeze_args.encoder}")
    logging.info(f"Freezing classifier: {config.training.freeze_args.classifier}")
    logging.info(f"Freezing decoder: {config.training.freeze_args.decoder}")
    logging.info(f"Freezing npe: {config.training.freeze_args.npe}")

    if checkpoint_path is not None and config.get("reset_optimizer", False):
        logging.info("Resetting optimizer state from checkpoint.")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        model_atg.load_state_dict(ckpt['state_dict'])

        # Set checkpoint_path to None to avoid loading optimizer states in trainer.fit()
        checkpoint_path = None

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
