

import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.transformer import Transformer

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
        d_time_projection=128,
        d_feat_projection=128,
        classifier_loss_weight=1.0,
        num_samples_per_graph=1,
        training_mode='all',
        norm_dict=None,
    ):
        """
        Parameters
        ----------
        input_size : int
            The size of the input, including the time dimension
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
        d_time_projection : int, optional
            The dimension of the time projection layer. Default: 1
        d_feat_projection : int, optional
            The dimension of the feature projection layer. Default: 1
        classifier_loss_weight : float, optional
            The weight of the classifier loss. Default: 1.0
        num_samples_per_graph : int, optional
            The number of samples per graph. Default: 1
        training_mode: str, optional
            The training mode. Can be 'all', 'classifier', or 'regressor'.
            Default: 'all'
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
        self.d_time_projection = d_time_projection
        self.d_feat_projection = d_feat_projection
        self.classifier_loss_weight = classifier_loss_weight
        self.num_samples_per_graph = num_samples_per_graph
        self.norm_dict = norm_dict
        self.batch_first = True # always True
        self.training_mode = training_mode
        if self.training_mode not in ['all', 'classifier', 'regressor']:
            raise ValueError(
                f'Training mode {self.training_mode} not supported')
        self.save_hyperparameters()

        self._setup_model()

    def _setup_model(self):

        self.transformer = Transformer(
            input_size=self.input_size,
            d_model=self.featurizer_args.d_model,
            nhead=self.featurizer_args.nhead,
            dim_feedforward=self.featurizer_args.dim_feedforward,
            num_layers=self.featurizer_args.num_layers,
            emb_size=self.featurizer_args.emb_size,
            emb_dropout=self.featurizer_args.emb_dropout,
            sum_features=self.featurizer_args.sum_features,
        )

        # create the rnn
        activation_fn = models_utils.get_activation(
            self.rnn_args.activation)
        self.rnn = models.GRUModel(
            input_size=self.d_feat_projection + self.d_time_projection,
            output_size=self.rnn_args.output_size,
            hidden_size=self.rnn_args.hidden_size,
            num_layers=self.rnn_args.num_layers,
            activation_fn=activation_fn
        )

        # create the flows
        self.flows = flows.NPE(
            in_dim=self.input_size - self.d_time,
            context_dim=self.rnn_args.output_size + self.featurizer_args.d_model,
            hidden_dims=self.flows_args.hidden_sizes,
            context_embedding_sizes=self.flows_args.context_embedding_sizes,
            num_transforms=self.num_transforms,
            dropout=self.flow_args.dropout,
        )

        # create the classifier
        # activation_fn = models_utils.get_activation(
            # self.classifier_args.activation)
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
        src_feat, src_len, tgt_feat, tgt_len  = training_utils.prepare_batch(
            batch, num_samples_per_graph=self.num_samples_per_graph)

        # Processing Transformer input and time steps
        src, src_t = src_feat[..., :-1], src_feat[..., -1:]
        src_padding_mask = training_utils.create_padding_mask(
            src_len, src_feat.size(1), batch_first=True)

        # Processing RNN/Flows input, output and timesteps
        tgt, tgt_t = tgt_feat[:, :, :-1], tgt_feat[:, 0, -1:]

        # add a starting token of all zeros to the first time step of tgt_feat
        # then, divide the rnn into the input and output component
        # the input will be feed into the RNN,
        # while the output will be used for the flow loss
        tgt = nn.functional.pad(tgt, (0, 0, 1, 0), value=0)
        tgt_in, tgt_out = tgt_inout[:, :-1], tgt_inout[:, 1:]
        tgt_padding_mask = training_utils.create_padding_mask(
            tgt_len, tgt_feat.size(1), batch_first=True)

        # Classifier target, which is the original length of the output
        # features minus 1 (start from 0)
        class_target = tgt_len - 1

        # Move to the same device as the model
        src = src.to(self.device)
        src_t = src_t.to(self.device)
        tgt_in = tgt_in.to(self.device)
        tgt_out = tgt_out.to(self.device)
        tgt_t = tgt_t.to(self.device)
        src_padding_mask = src_padding_mask.to(self.device)
        tgt_padding_mask = tgt_padding_mask.to(self.device)
        class_target = class_target.to(self.device)

        # return a dictionary of the inputs
        return {
            'src': src,
            'src_t': src_t,
            'tgt_in': tgt_in,
            'tgt_out': tgt_out,
            'tgt_t': tgt_t,
            'src_padding_mask': src_padding_mask,
            'tgt_padding_mask': tgt_padding_mask,
            'class_target': class_target,
            'batch_size': src.size(0),
            'src_len': src_len,
            'tgt_len': tgt_len,
        }

    def forward(
        self, src, src_t, tgt_in, tgt_t, src_padding_mask=None,
        tgt_padding_mask=None, tgt_len=None
    ):
        """ Forward pass of the model.
        Args:
            src: torch.Tensor, shape [batch_size, seq_len, input_size]
                Input of the Transformer.
            src_t: torch.Tensor, shape [batch_size, seq_len, 1]
                Time features of the Transformer.
            tgt_in: torch.Tensor, shape [batch_size, seq_len, input_size]
                Input of the RNN/Flows model.
            tgt_t: torch.Tensor, shape [batch_size, seq_len, 1]
                Time features of the RNN/Flows model.
            src_padding_mask: torch.Tensor, shape [batch_size, seq_len]
                Padding mask for the Transformer.
            tgt_padding_mask: torch.Tensor, shape [batch_size, seq_len]
                Padding mask for the RNN/Flows model.
            tgt_len: torch.Tensor, shape [batch_size]
                Original lengths of the RNN/Flows features.
        """
        # extract the features
        x_transf = self.transformer(src, src_t, padding_mask=src_padding_mask)

        # project the time and feature dimensions
        tgt_in_proj = self.feat_proj_layer(tgt_in)
        tgt_t_proj = self.time_proj_layer(tgt_t)

        # RNN and flows
        if tgt_len is None:
            tgt_len = tgt_padding_mask.eq(0).sum(-1).cpu() # original lengths
        x_rnn = torch.cat(
            [tgt_in_proj, tgt_t_proj.unsqueeze(1).repeat(1, tgt_in.size(1), 1)], dim=-1)
        x_rnn = self.rnn(x_rnn, tgt_len)

        # create the context and input for the flows
        flow_context = torch.cat(
            [x_rnn, x_transf.unsqueeze(1).repeat(1, tgt_in.size(1), 1)], dim=-1)
        flow_context = flow_context[~tgt_padding_mask]

        # Classifier
        # project the time and concatenate with the features
        yhat_class = self.classifier(torch.cat((x_transf, tgt_t_proj), dim=1))

        return flow_context, yhat_class

    def training_step(self, batch, batch_idx):
        batch_dict = self._prepare_batch(batch)
        batch_size = batch_size

        # forward pass
        flow_context, yhat_class = self.forward(
            src=batch_dict['src'],
            src_t_in=batch_dict['src_t'],
            tgt_in=batch_dict['tgt_in'],
            tgt_t=batch_dict['tgt_t'],
            src_padding_mask=batch_dict['src_padding_mask'],
            tgt_padding_mask=batch_dict['tgt_padding_mask'],
            tgt_len=batch_dict['tgt_len']
        )

        # compute the flow loss
        if self.training_mode == 'all' or self.training_mode == 'regressor':
            flow_target = batch_dict['tgt_out'][~batch_dict['tgt_padding_mask']]
            log_prob = self.flows(flow_context).log_prob(flow_target)
            flow_loss = -log_prob.mean()
        else:
            flow_loss = 0

        # compute the classifier loss and accuracy
        if self.training_mode == 'all' or self.training_mode == 'classifier':
            class_loss = torch.nn.CrossEntropyLoss()(
                yhat_class, batch_dict['class_target'])
            class_acc = batch_dict['class_target'].eq(
                yhat_class.argmax(dim=1)).float().mean()
        else:
            class_loss = 0
            class_acc = 0

        # combine the losses
        loss = flow_loss + self.classifier_loss_weight * class_loss

        # log the loss and accuracy
        self.log(
            'train_flow_loss', flow_loss, on_step=True, on_epoch=True,
            logger=True, prog_bar=True, batch_size=batch_size)
        self.log(
            'train_class_loss', class_loss, on_step=True, on_epoch=True,
            logger=True, prog_bar=True, batch_size=batch_size)
        self.log(
            'train_class_acc', class_acc, on_step=True, on_epoch=True,
            logger=True, prog_bar=True, batch_size=batch_size)
        self.log(
            'train_loss', loss, on_step=True, on_epoch=True,
            logger=True, prog_bar=True,  batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_dict = self._prepare_batch(batch)
        batch_size = batch_dict['batch_size']

   # forward pass
        flow_context, yhat_class = self.forward(
            src=batch_dict['src'],
            src_t_in=batch_dict['src_t'],
            tgt_in=batch_dict['tgt_in'],
            tgt_t=batch_dict['tgt_t'],
            src_padding_mask=batch_dict['src_padding_mask'],
            tgt_padding_mask=batch_dict['tgt_padding_mask'],
            tgt_len=batch_dict['tgt_len']
        )

        # compute the flow loss
        if self.training_mode == 'all' or self.training_mode == 'regressor':
            flow_target = batch_dict['tgt_out'][~batch_dict['tgt_padding_mask']]
            log_prob = self.flows(flow_context).log_prob(flow_target)
            flow_loss = -log_prob.mean()
        else:
            flow_loss = 0

        # compute the classifier loss and accuracy
        if self.training_mode == 'all' or self.training_mode == 'classifier':
            class_loss = torch.nn.CrossEntropyLoss()(
                yhat_class, batch_dict['class_target'])
            class_acc = batch_dict['class_target'].eq(
                yhat_class.argmax(dim=1)).float().mean()
        else:
            class_loss = 0
            class_acc = 0

        # combine the losses
        loss = flow_loss + self.classifier_loss_weight * class_loss

        # log the loss and accuracy
        self.log(
            'val_flow_loss', flow_loss, on_step=True, on_epoch=True,
            logger=True, prog_bar=True, batch_size=batch_size)
        self.log(
            'val_class_loss', class_loss, on_step=True, on_epoch=True,
            logger=True, prog_bar=True, batch_size=batch_size)
        self.log(
            'val_class_acc', class_acc, on_step=True, on_epoch=True,
            logger=True, prog_bar=True, batch_size=batch_size)
        self.log(
            'val_loss', loss, on_step=True, on_epoch=True,
            logger=True, prog_bar=True,  batch_size=batch_size)
        return loss


    def configure_optimizers(self):
        """ Initialize optimizer and LR scheduler """
        return models_utils.configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args)
