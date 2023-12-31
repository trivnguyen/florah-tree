

import torch
import torch.nn as nn
import pytorch_lightning as pl

from . import models, models_utils, training_utils


class SequenceClassifier(pl.LightningModule):
    """
    Sequence classifier based on the TransformerEncoder module from PyTorch.

    Attributes
    ----------
    input_size : int
        The size of the input
    num_classes : int
        The number of classes
    sum_features : bool
        Whether to sum the features of the featurizer in the time dimension.
    num_samples_per_graph : int
        The number of samples per graph.
    d_time : int
        The dimension of the time projection layer.
    d_time_projection : int
        The dimension of the time projection layer.
    batch_first : bool
        Whether the input is batch first.
    featurizer : nn.Module
        The featurizer.
    classifier : nn.Module
        The classifier.
    optimizer : torch.optim.Optimizer
        The optimizer.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        The scheduler.
    norm_dict : dict
        The normalization dictionary. For bookkeeping purposes only.
    """
    def __init__(
        self,
        input_size,
        num_classes,
        featurizer_args,
        classifier_args,
        optimizer_args=None,
        scheduler_args=None,
        sum_features=False,
        d_time=1,
        d_time_projection=128,
        num_samples_per_graph=1,
        norm_dict=None,
    ):
        """
        Parameters
        ----------
        input_size : int
            The size of the input
        num_classes : int
            The number of classes
        featurizer_args : dict
            Arguments for the featurizer
        classifier_args : dict
            Arguments for the classifier
        optimizer_args : dict, optional
            Arguments for the optimizer. Default: None
        scheduler_args : dict, optional
            Arguments for the scheduler. Default: None
        sum_features : bool, optional
            Whether to sum the features of the featurizer in the time dimension.
            Default: False
        d_time : int, optional
            The dimension of the time projection layer. Default: 1
        d_time_projection : int, optional
            The dimension of the time projection layer. Default: 1
        num_samples_per_graph : int, optional
            The number of samples per graph. Default: 1
        norm_dict : dict, optional
            The normalization dictionary. For bookkeeping purposes only.
            Default: None
        """
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.sum_features = sum_features
        self.num_samples_per_graph = num_samples_per_graph
        self.d_time = d_time
        self.d_time_projection = d_time_projection
        self.batch_first = True # always True
        self.featurizer_args = featurizer_args
        self.classifier_args = classifier_args
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.norm_dict = norm_dict or {}
        self.save_hyperparameters()

        self._setup_model()

    def _setup_model(self):

        # create the featurizer
        if self.featurizer_args.name == 'transformer':
            activation_fn = models_utils.get_activation(
                self.featurizer_args.activation)
            self.featurizer = models.TransformerFeaturizer(
                input_size=self.input_size,
                d_model=self.featurizer_args.d_model,
                nhead=self.featurizer_args.nhead,
                num_encoder_layers=self.featurizer_args.num_encoder_layers,
                dim_feedforward=self.featurizer_args.dim_feedforward,
                batch_first=self.batch_first,
                use_embedding=self.featurizer_args.use_embedding,
                activation_fn=activation_fn,
            )
        else:
            raise ValueError(
                f'Featurizer {featurizer_name} not supported')

        # create the time projection layer
        self.time_proj_layer = nn.Linear(self.d_time, self.d_time_projection)

        # create the classifier
        if self.classifier_args.name == 'mlp':
            activation_fn = models_utils.get_activation(
                self.classifier_args.activation)
            self.classifier = models.MLP(
                input_size=self.featurizer.d_model + self.d_time_projection,
                output_size=self.num_classes,
                hidden_sizes=self.classifier_args.hidden_sizes,
                activation_fn=activation_fn,
            )

    def _prepare_batch(self, batch):
        """ Wrapper around training_utils.prepare_batch """
        batch  = training_utils.prepare_batch(
            batch, num_samples_per_graph=self.num_samples_per_graph)
        padded_features = batch[0]  # [batch_size, seq_len1, feature_size]
        lengths = batch[1] # original lengths of the sequences
        padded_out_features = batch[2]  # [batch_size, seq_len2, feature_size]
        out_lengths = batch[3]  # original lengths of the sequences

        # Assuming padded_features is your input to the transformer
        # with shape [batch_size, seq_len, feature_size]
        batch_size, seq_len, _ = padded_features.size()

        # Create a mask for padding (assuming padding tokens are zero)
        # The mask should have the shape [seq_len, batch_size]
        transformer_padding_mask = training_utils.create_padding_mask(
            lengths, seq_len, batch_first=self.batch_first)

        # get the time of the output features
        # the output features are padded, but all the time should be the same
        t_out = padded_out_features[:, 0, -self.d_time].view(
            batch_size, self.d_time)

        # get the classifier labels, which is the original length of the output
        # features minus 1 (start from 0)
        classifier_labels = out_lengths - 1

        # make sure everything is on the same device
        padded_features = padded_features.to(self.device)
        transformer_padding_mask = transformer_padding_mask.to(self.device)
        t_out = t_out.to(self.device)
        classifier_labels = classifier_labels.to(self.device)

        # create an output dictionary
        return_dict = {
            'padded_features': padded_features,
            'transformer_padding_mask': transformer_padding_mask,
            't_out': t_out,
            'classifier_labels': classifier_labels,
            'batch_size': batch_size,
            'seq_len': seq_len,
        }
        return return_dict


    def forward(self, padded_features, t_out, transformer_padding_mask=None):

        # extract the features
        x = self.featurizer(
            padded_features, src_key_padding_mask=transformer_padding_mask)
        x = x.sum(dim=1) if self.sum_features else x[:, -1]

        # project the time and concatenate with the features
        t_out = self.time_proj_layer(t_out)
        x = torch.cat((x, t_out), dim=1)

        # apply classifier layer
        x = self.classifier(x)

        return x

    def training_step(self, batch, batch_idx):
        # prepare the batch
        batch_dict = self._prepare_batch(batch)

        # forward pass
        y_hat = self.forward(
            batch_dict['padded_features'], batch_dict['t_out'],
            batch_dict['transformer_padding_mask']
        )
        loss = torch.nn.CrossEntropyLoss()(
            y_hat, batch_dict['classifier_labels'])

        # calculate the accuracy
        acc = batch_dict['classifier_labels'].eq(y_hat.argmax(dim=1)).float().mean()

        # log the loss
        self.log(
            'train_loss', loss, on_step=True, on_epoch=True, logger=True,
            prog_bar=True, batch_size=batch_dit['batch_size'])
        self.log(
            'train_acc', acc, on_step=True, on_epoch=True, logger=True,
            prog_bar=True, batch_size=batch_dit['batch_size'])
        return loss

    def validation_step(self, batch, batch_idx):
        # prepare the batch
        batch_dict = self._prepare_batch(batch)

        # forward pass
        y_hat = self.forward(
            batch_dict['padded_features'], batch_dict['t_out'],
            batch_dict['transformer_padding_mask'])
        loss = torch.nn.CrossEntropyLoss()(
            y_hat, batch_dict['classifier_labels'])

        # calculate the accuracy
        acc = batch_dict['classifier_labels'].eq(y_hat.argmax(dim=1)).float().mean()

        # log the loss
        self.log(
            'val_loss', loss, on_step=True, on_epoch=True, logger=True,
            prog_bar=True, batch_size=batch_dict['batch_size'])
        self.log(
            'val_acc', acc, on_step=True, on_epoch=True, logger=True,
            prog_bar=True, batch_size=batch_dict['batch_size'])
        return loss


    def configure_optimizers(self):
        """ Initialize optimizer and LR scheduler """

        # setup the optimizer
        if self.optimizer_args.name == "Adam":
            return torch.optim.Adam(
                self.parameters(), lr=self.optimizer_args.lr,
                weight_decay=self.optimizer_args.weight_decay)
        elif self.optimizer_args.name == "AdamW":
            return torch.optim.AdamW(
                self.parameters(), lr=self.optimizer_args.lr,
                weight_decay=self.optimizer_args.weight_decay)
        else:
            raise NotImplementedError(
                "Optimizer {} not implemented".format(self.optimizer_args.name))

        # setup the scheduler
        if self.scheduler_args.get(name) is None:
            scheduler = None
        elif self.scheduler_args.name == 'ReduceLROnPlateau':
            scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', factor=self.scheduler_args.factor,
                patience=self.scheduler_args.patience)
        elif self.scheduler_args.name == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.scheduler_args.T_max,)
        else:
            raise NotImplementedError(
                "Scheduler {} not implemented".format(self.scheduler_args.name))

        if scheduler is None:
            return optimizer
        else:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'train_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }

