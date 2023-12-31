

import torch
import torch.nn as nn
import pytorch_lightning as pl

from . import models, models_utils, training_utils, flows_utils


class TreeGenerator(pl.LightningModule):
    def __init__(
        self,
        input_size,
        num_classes,
        featurizer_args,
        rnn_args,
        flows_args,
        classifier_args,
        optimizer_args=None,
        scheduler_args=None,
        sum_features=False,
        d_time=1,
        d_time_projection=128,
        d_feat_projection=128,
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
        rnn_args : dict
            Arguments for the RNN
        flows_args : dict
            Arguments for the normalizing flow
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
        d_feat_projection : int, optional
            The dimension of the feature projection layer. Default: 1
        classifier_loss_weight : float, optional
            The weight of the classifier loss. Default: 1.0
        num_samples_per_graph : int, optional
            The number of samples per graph. Default: 1
        norm_dict : dict, optional
            The normalization dictionary. For bookkeeping purposes only.
            Default: None
        """
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.featurizer_args = featurizer_args
        self.rnn_args = rnn_args
        self.flows_args = flows_args
        self.classifier_args = classifier_args
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.sum_features = sum_features
        self.d_time = d_time
        self.d_time_projection = d_time_projection
        self.d_feat_projection = d_feat_projection
        self.classifier_loss_weight = classifier_loss_weight
        self.num_samples_per_graph = num_samples_per_graph
        self.norm_dict = norm_dict
        self.batch_first = True # always True
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

        # create the rnn
        if self.rnn_args.name == 'gru':
            activation_fn = models_utils.get_activation(
                self.rnn_args.activation)
            self.rnn = models.GRUModel(
                input_size=self.d_feat_projection + self.d_time_projection,
                output_size=self.rnn_args.output_size,
                hidden_size=self.rnn_args.hidden_size,
                num_layers=self.rnn_args.num_layers,
                activation_fn=activation_fn,
            )
        else:
            raise ValueError(
                f'RNN {rnn_name} not supported')

        # create the flows
        self.flows = flows_utils.build_maf(
            features=self.input_size - self.d_time,
            hidden_features=self.flows_args.hidden_size,
            context_features=self.rnn_args.output_size + self.featurizer_args.d_model,
            num_layers=self.flows_args.num_layers,
            num_blocks=self.flows_args.num_blocks,
        )

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

        # create the projection layers
        self.time_proj_layer = nn.Linear(self.d_time, self.d_time_projection)
        self.feat_proj_layer = nn.Linear(
            self.input_size - self.d_time, self.d_feat_projection)

    def _prepare_batch(self, batch):
        """ Prepare the batch for training. """
        batch  = training_utils.prepare_batch(batch, num_samples_per_graph=1)
        padded_features = batch[0]
        lengths = batch[1]
        padded_out_features = batch[2]
        out_lengths = batch[3]

        # Separate the time and feature dimensions of the output
        t_out = padded_out_features[:, 0, -self.d_time:]
        f_out = padded_out_features[:, :, :-self.d_time]

        # add a starting token of all zeros to the first time step of padded_out_features
        padded_rnn_features = nn.functional.pad(f_out, (0, 0, 1, 0), value=0)

        # divide the rnn into the input and output component
        # the input will be feed into the RNN,
        # while the output will be used for the flow loss
        padded_rnn_input = padded_rnn_features[:, :-1]
        padded_rnn_output = padded_rnn_features[:, 1:]

        # Assuming padded_features is your input to the transformer
        # with shape [batch_size, seq_len, feature_size]
        batch_size, seq_len, _ = padded_features.size()
        out_seq_len = padded_out_features.size(1)

        # Create a mask for padding (assuming padding tokens are zero)
        # The mask should have the shape [seq_len, batch_size]
        transformer_padding_mask = training_utils.create_padding_mask(
            lengths, seq_len, batch_first=True)
        rnn_padding_mask = training_utils.create_padding_mask(
            out_lengths, out_seq_len, batch_first=True)

        # get the classifier labels, which is the original length of the output
        # features minus 1 (start from 0)
        classifier_labels = out_lengths - 1

        # Move to the same device as the model
        padded_features = padded_features.to(self.device)
        padded_rnn_input = padded_rnn_input.to(self.device)
        padded_rnn_output = padded_rnn_output.to(self.device)
        transformer_padding_mask = transformer_padding_mask.to(self.device)
        rnn_padding_mask = rnn_padding_mask.to(self.device)
        t_out = t_out.to(self.device)
        classifier_labels = classifier_labels.to(self.device)

        # return a dictionary of the inputs
        return_dict = {
            'padded_features': padded_features,
            'padded_rnn_input': padded_rnn_input,
            'padded_rnn_output': padded_rnn_output,
            'transformer_padding_mask': transformer_padding_mask,
            'rnn_padding_mask': rnn_padding_mask,
            'classifier_labels': classifier_labels,
            't_out': t_out,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'out_seq_len': out_seq_len,
        }
        return return_dict

    def forward(
        self, padded_features, padded_rnn_input, t_out,
        transformer_padding_mask=None, rnn_padding_mask=None
    ):
        # extract the features
        x = self.featurizer(
            padded_features, src_key_padding_mask=transformer_padding_mask)
        x = x.sum(dim=1) if self.sum_features else x[:, -1]

        # project the time and feature dimensions
        t_proj = self.time_proj_layer(t_out)
        f_proj = self.feat_proj_layer(padded_rnn_input)

        # RNN and flows
        out_seq_len = padded_rnn_input.size(1)  # lengths after padding
        out_lengths = rnn_padding_mask.eq(0).sum(-1).cpu() # original lengths
        x_rnn = torch.cat(
            [f_proj, t_proj.unsqueeze(1).repeat(1, out_seq_len, 1)], dim=-1)
        x_rnn = self.rnn(x_rnn, out_lengths)

        # create the context and input for the flows
        flow_context = torch.cat(
            [x_rnn, x.unsqueeze(1).repeat(1, out_seq_len, 1)], dim=-1)
        flow_context = flow_context[~rnn_padding_mask]

        # Classifier
        # project the time and concatenate with the features
        x_classifier = torch.cat((x, t_proj), dim=1)
        x_classifier = self.classifier(x_classifier)

        return flow_context, x_classifier

    def training_step(self, batch, batch_idx):
        batch_dict = self._prepare_batch(batch)

        # forward pass
        flow_context, yhat_classifier = self.forward(
            padded_features=batch_dict['padded_features'],
            padded_rnn_input=batch_dict['padded_rnn_input'],
            t_out=batch_dict['t_out'],
            transformer_padding_mask=batch_dict['transformer_padding_mask'],
            rnn_padding_mask=batch_dict['rnn_padding_mask'],
        )

        # compute the flow loss
        x_flow = batch_dict['padded_rnn_output'][~batch_dict['rnn_padding_mask']]
        log_prob = self.flows.log_prob(x_flow, context=flow_context)
        flow_loss = -log_prob.mean()

        # compute the classifier loss and accuracy
        classifier_loss = torch.nn.CrossEntropyLoss()(
            yhat_classifier, batch_dict['classifier_labels'])
        classifier_acc = batch_dict['classifier_labels'].eq(
            yhat_classifier.argmax(dim=1)).float().mean()

        # combine the losses
        loss = flow_loss + self.classifier_loss_weight * classifier_loss

        # log the loss and accuracy
        self.log(
            'train_flow_loss', flow_loss, on_step=True, on_epoch=True,
            logger=True, prog_bar=True, batch_size=batch_dict['batch_size'])
        self.log(
            'train_classifier_loss', classifier_loss, on_step=True,
            on_epoch=True, logger=True, prog_bar=True,
            batch_size=batch_dict['batch_size'])
        self.log(
            'train_classifier_acc', classifier_acc, on_step=True, on_epoch=True,
            logger=True, prog_bar=True, batch_size=batch_dict['batch_size'])
        self.log(
            'train_loss', loss, on_step=True, on_epoch=True, logger=True,
            prog_bar=True, batch_size=batch_dict['batch_size'])
        return loss

    def validation_step(self, batch, batch_idx):
        batch_dict = self._prepare_batch(batch)

        # forward pass
        flow_context, yhat_classifier = self.forward(
            padded_features=batch_dict['padded_features'],
            padded_rnn_input=batch_dict['padded_rnn_input'],
            t_out=batch_dict['t_out'],
            transformer_padding_mask=batch_dict['transformer_padding_mask'],
            rnn_padding_mask=batch_dict['rnn_padding_mask'],
        )

        # compute the flow loss
        x_flow = batch_dict['padded_rnn_output'][~batch_dict['rnn_padding_mask']]
        log_prob = self.flows.log_prob(x_flow, context=flow_context)
        flow_loss = -log_prob.mean()

        # compute the classifier loss and accuracy
        classifier_loss = torch.nn.CrossEntropyLoss()(
            yhat_classifier, batch_dict['classifier_labels'])
        classifier_acc = batch_dict['classifier_labels'].eq(
            yhat_classifier.argmax(dim=1)).float().mean()

        # combine the losses
        loss = flow_loss + self.classifier_loss_weight * classifier_loss

        # log the loss and accuracy
        self.log(
            'val_flow_loss', flow_loss, on_step=True, on_epoch=True,
            logger=True, prog_bar=True, batch_size=batch_dict['batch_size'])
        self.log(
            'val_classifier_loss', classifier_loss, on_step=True,
            on_epoch=True, logger=True, prog_bar=True,
            batch_size=batch_dict['batch_size'])
        self.log(
            'val_classifier_acc', classifier_acc, on_step=True, on_epoch=True,
            logger=True, prog_bar=True, batch_size=batch_dict['batch_size'])
        self.log(
            'val_loss', loss, on_step=True, on_epoch=True, logger=True,
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