
import os
import pickle
import sys

import ml_collections
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from absl import flags, logging
from ml_collections import config_flags

import datasets
from models import generator, regressor, classifier
from models import models, utils, training_utils

logging.set_verbosity(logging.INFO)


def get_model(config, norm_dict=None):
    """ Get the model from config file """

    if config.model_name == "TreeGenerator":
        model = generator.TreeGenerator(
            input_size=config.input_size,
            num_classes=config.num_classes,
            featurizer_args=config.featurizer,
            rnn_args=config.rnn,
            flows_args=config.flows,
            classifier_args=config.classifier,
            optimizer_args=config.optimizer,
            scheduler_args=config.scheduler,
            d_time = config.d_time,
            d_time_projection = config.d_time_projection,
            d_feat_projection = config.d_feat_projection,
            classifier_loss_weight=config.classifier_loss_weight,
            num_samples_per_graph=config.num_samples_per_graph,
            norm_dict=norm_dict,
        )
    elif config.model_name == "SequenceClassifier":
        model = classifier.SequenceClassifier(
            input_size=config.input_size,
            num_classes=config.num_classes,
            num_samples_per_graph=config.num_samples_per_graph,
            d_time=config.d_time,
            d_time_projection=config.d_time_projection,
            featurizer_args=config.featurizer,
            classifier_args=config.classifier,
            optimizer_args=config.optimizer,
            scheduler_args=config.scheduler,
            norm_dict=norm_dict,
        )
    elif config.model_name == "SequenceRegressor":
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
    elif config.model_name == "SequenceRegressorMultiFlows":
        model = regressor.SequenceRegressorMultiFlows(
            input_size=config.input_size,
            num_classes=config.num_classes,
            featurizer_args=config.featurizer,
            flows_args=config.flows,
            optimizer_args=config.optimizer,
            scheduler_args=config.scheduler,
            d_time = config.d_time,
            d_time_projection = config.d_time_projection,
            num_samples_per_graph=config.num_samples_per_graph,
            norm_dict=norm_dict,
        )
    else:
        raise ValueError(f"Model {config.model_name} not recognized.")

    return model


def train(
    config: ml_collections.ConfigDict, workdir: str = "./logging/"
):
    # set up work directory
    if not hasattr(config, "name"):
        name = utils.get_random_name()
    else:
        name = config["name"]
    logging.info("Starting training run {} at {}".format(name, workdir))

    # set up random seed
    pl.seed_everything(config.seed)

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

    # load dataset and prepare dataloader
    logging.info("Preparing dataloader...")

    train_loader, val_loader, norm_dict = datasets.prepare_dataloader(
        datasets.read_dataset(
            dataset_name=config.data_name,
            dataset_root=config.data_root,
            max_num_files=config.get("max_num_files", 100),
        ), config, norm_dict=None)

    # create model
    logging.info("Creating model...")
    model = get_model(config, norm_dict=norm_dict)

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
    train_logger = pl_loggers.TensorBoardLogger(workdir, version='')
    trainer = pl.Trainer(
        default_root_dir=workdir,
        max_epochs=config.num_epochs,
        accelerator=config.accelerator,
        callbacks=callbacks,
        logger=train_logger,
        enable_progress_bar=config.get("enable_progress_bar", True),
    )

    # train the model
    logging.info("Training model...")
    trainer.fit(
        model, train_loader, val_loader,
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
