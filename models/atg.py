from typing import Union, Dict, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from ml_collections.config_dict import ConfigDict

from models import flows, models_utils, training_utils
from models.transformer import TransformerEncoder, TransformerDecoder
from models.models import MLP, GRUDecoder

class AutoregTreeGen(pl.LightningModule):
    """ Autoregressive tree generator model """
    def __init__(
        self,
        d_in: int,
        num_classes: int,
        encoder_args: Union[ConfigDict, Dict],
        decoder_args: Union[ConfigDict, Dict],
        npe_args: Union[ConfigDict, Dict],
        classifier_args: Union[ConfigDict, Dict],
        optimizer_args: Union[ConfigDict, Dict] = None,
        scheduler_args: Union[ConfigDict, Dict] = None,
        training_args: Union[ConfigDict, Dict] = None,
        norm_dict: Optional[Dict] = None,
        concat_npe_context: bool = False,
        concat_time: bool = False,
        use_sos_embedding: bool = False,
    ):
        """
        Parameters
        ----------
        d_in : int
            Input dimension, excluding time dimension
        num_classes : int
            The number of classes
        encoder_args : dict
            Arguments for the encoder
        decoder_args : dict
            Arguments for the decoder
        npe_args : dict
            Arguments for the NPE
        classifier_args : dict
            Arguments for the classifier
        optimizer_args : dict, optional
            Arguments for the optimizer. Default: None
        scheduler_args : dict, optional
            Arguments for the scheduler. Default: None
        training_args : dict, optional
            Arguments for the training process. Default: None
        norm_dict : dict, optional
            The normalization dictionary. For bookkeeping purposes only.
            Default: None
        concat_npe_context : bool, optional
            Whether to concatenate the context to the NPE. Default: True
        concat_time : bool, optional
            If True, concat the input and output time. Default: False
        use_sos_embedding : bool, optional
            If True, use an embedding layer for the start of sequence token, instead of
            zero-padding. The embedding layer takes in the encoder output. Only
            applicable for GRU decoder. Default: False.
        """
        super().__init__()
        self.d_in = d_in
        self.num_classes = num_classes
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        self.npe_args = npe_args
        self.classifier_args = classifier_args
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args
        self.norm_dict = norm_dict
        self.concat_npe_context = concat_npe_context
        self.concat_time = concat_time
        self.use_sos_embedding = use_sos_embedding

        # training args
        training_args_default = ConfigDict({
            'num_samples_per_graph': 1,
            'training_mode': 'all',
            'class_loss_weight': 1.0,
            'use_sample_weights': False,
            'max_weight': 100,
            'old_flows_loss': True,
            'all_nodes': False,
        })
        self.training_args = training_args_default
        if training_args is not None:
            self.training_args.update(training_args)
        if self.training_args.training_mode not in ('all', 'classifier', 'npe'):
            raise ValueError(
                f'Training mode {self.training_mode} not supported')
        if self.training_args.old_flows_loss and self.training_args.use_sample_weights:
            raise ValueError(
                f'Options old_flows_loss and use_sample_weights are not compatible')
        self.save_hyperparameters()

        self._setup_model()

    def _setup_model(self):

        # create the transformer
        d_time = 2 if self.concat_time else 1
        if self.encoder_args.name == 'transformer':
            self.encoder = TransformerEncoder(
                d_in=self.d_in,
                d_model=self.encoder_args.d_model,
                nhead=self.encoder_args.nhead,
                dim_feedforward=self.encoder_args.dim_feedforward,
                num_layers=self.encoder_args.num_layers,
                d_time=d_time,
                emb_size=self.encoder_args.emb_size,
                emb_dropout=self.encoder_args.emb_dropout,
                concat=self.encoder_args.concat
            )
        elif self.encoder_args.name == 'gru':
            self.encoder = GRUDecoder(
                d_in=self.d_in,
                d_model=self.encoder_args.d_model,
                d_out=self.encoder_args.d_out,
                dim_feedforward=self.encoder_args.dim_feedforward,
                num_layers=self.encoder_args.num_layers,
                d_time=d_time,
                activation_fn=nn.ReLU(),
                concat=self.encoder_args.concat,
            )
        else:
            raise NotImplementedError(
                f'Encoder {self.encoder_args.name} not implemented')

        # create the RNN and flows
        if self.decoder_args.name == 'transformer':
            self.decoder = TransformerDecoder(
                d_in=self.d_in,
                d_model=self.decoder_args.d_model,
                d_context=self.decoder_args.d_context,
                nhead=self.decoder_args.nhead,
                dim_feedforward=self.decoder_args.dim_feedforward,
                num_layers=self.decoder_args.num_layers,
                emb_size=self.decoder_args.emb_size,
                emb_dropout=self.decoder_args.emb_dropout,
                max_len=self.decoder_args.max_len,
            )
        elif self.decoder_args.name == 'gru':
            self.decoder = GRUDecoder(
                d_in=self.d_in,
                d_model=self.decoder_args.d_model,
                d_out=self.decoder_args.d_out,
                dim_feedforward=self.decoder_args.dim_feedforward,
                num_layers=self.decoder_args.num_layers,
                activation_fn=nn.ReLU(),
                concat=self.decoder_args.concat,
            )
        else:
            raise NotImplementedError(
                f'Decoder {self.decoder_args.name} not implemented')

        # create the NPE
        if self.concat_npe_context:
            context_size = self.encoder_args.d_model + self.decoder_args.d_model
        else:
            if self.decoder_args.d_model != self.encoder_args.d_model:
                raise ValueError(
                    'When not concatenating NPE context, encoder and decoder '
                    'dimension must be the same')
            context_size = self.decoder_args.d_model
        self.npe = flows.NPE(
            input_size=self.d_in,
            context_size=context_size,
            hidden_sizes=self.npe_args.hidden_sizes,
            context_embedding_sizes=self.npe_args.context_embedding_sizes,
            num_transforms=self.npe_args.num_transforms,
            dropout=self.npe_args.dropout,
        )

        # create the classifier
        self.classifier_context_embed = nn.Linear(
            self.classifier_args.d_context, self.encoder_args.d_model)
        self.classifier = MLP(
            d_in=self.encoder_args.d_model,
            hidden_sizes=self.classifier_args.hidden_sizes + [self.num_classes,],
        )

        # other layers
        if self.use_sos_embedding and self.decoder_args.name == 'gru':
            self.sos_embedding = MLP(
                d_in=self.encoder_args.d_model,
                hidden_sizes=[self.encoder_args.d_model, self.d_in],
            )
        else:
            self.sos_embedding = None

    def _prepare_batch(self, batch):
        """ Prepare the batch for training. """
        src_feat, src_len, tgt_feat, tgt_len, weights  = training_utils.prepare_batch(
            batch,
            num_samples_per_graph=self.training_args.num_samples_per_graph,
            return_weights=self.training_args.use_sample_weights,
            all_nodes=self.training_args.all_nodes,
        )

        # Processing Transformer input and time steps
        src = src_feat[..., :-1]
        if not self.concat_time:
            src_t = src_feat[..., -1:]
        else:
            src_t1 = src_feat[..., -1:]
            src_t2 = torch.cat([src_feat[:, 1:, -1:], tgt_feat[:, :1, -1:]], dim=1)
            src_t = torch.cat([src_t1, src_t2], dim=-1)
        src_padding_mask = training_utils.create_padding_mask(
            src_len, src_feat.size(1), batch_first=True)

        # Processing RNN/Flows input, output and timesteps
        tgt, tgt_t = tgt_feat[:, :, :-1], tgt_feat[:, 0, -1:]

        # add a starting token of all zeros to the first time step of tgt_feat
        # then, divide the rnn into the input and output component
        # the input will be feed into the RNN,
        # while the output will be used for the flow loss
        tgt = nn.functional.pad(tgt, (0, 0, 1, 0), value=0)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        tgt_padding_mask = training_utils.create_padding_mask(
            tgt_len, tgt_feat.size(1), batch_first=True)

        # Classifier target, which is the original length of the output
        # features minus 1 (start from 0)
        class_target = tgt_len - 1

        # Processing weights
        weights = torch.clamp(weights, min=None, max=self.training_args.max_weight)

        # Move to the same device as the model
        src = src.to(self.device)
        src_t = src_t.to(self.device)
        tgt_in = tgt_in.to(self.device)
        tgt_out = tgt_out.to(self.device)
        tgt_t = tgt_t.to(self.device)
        src_padding_mask = src_padding_mask.to(self.device)
        tgt_padding_mask = tgt_padding_mask.to(self.device)
        class_target = class_target.to(self.device)
        weights = weights.to(self.device)

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
            'weights': weights
        }

    def forward(
        self, src, src_t, tgt_in, tgt_t, src_padding_mask=None,
        tgt_padding_mask=None, tgt_len=None
    ):
        """ Forward pass of the model.
        Args:
            src: torch.Tensor, shape [batch_size, seq_len, d_in]
                Input of the Transformer.
            src_t: torch.Tensor, shape [batch_size, seq_len, 1]
                Time features of the Transformer.
            tgt_in: torch.Tensor, shape [batch_size, seq_len, d_in]
                Input of the RNN/Flows model.
            tgt_t: torch.Tensor, shape [batch_size, 1]
                Time features of the RNN/Flows model.
            src_padding_mask: torch.Tensor, shape [batch_size, seq_len]
                Padding mask for the Transformer.
            tgt_padding_mask: torch.Tensor, shape [batch_size, seq_len]
                Padding mask for the RNN/Flows model.
        """
        # 1. pass the input sequence through the encoder
        if self.encoder_args.name == 'transformer':
            x_enc = self.encoder(
                src,
                src_t,
                src_padding_mask=src_padding_mask
            )
            x_enc_reduced = models_utils.summarize_features(
                x_enc, reduction='last', padding_mask=src_padding_mask)
        elif self.encoder_args.name == 'gru':
            x_enc = self.encoder(
                x=src,
                t=src_t,
                lengths=src_padding_mask.eq(0).sum(-1).cpu(),
            )
            x_enc_reduced = models_utils.summarize_features(
                x_enc, reduction='last', padding_mask=src_padding_mask)

        # 2. pass the encoded features through the decoder
        # using the output time steps as the context
        if self.decoder_args.name == 'transformer':
            x_dec = self.decoder(
                tgt_in,
                memory=x_enc_reduced.unsqueeze(1),
                context=tgt_t,
                tgt_padding_mask=tgt_padding_mask,
                memory_padding_mask=None
            )
        elif self.decoder_args.name == 'gru':
            if self.use_sos_embedding:
                sos_embed = self.sos_embedding(x_enc_reduced)
                tgt_in[:, 0] = sos_embed

            x_dec = self.decoder(
                x=tgt_in,
                t=tgt_t.unsqueeze(1).expand(-1, tgt_in.size(1), -1),
                lengths=tgt_padding_mask.eq(0).sum(-1).cpu(),
            )

        # 3. create the flows context
        if self.concat_npe_context:
            context_flows = torch.cat([
                x_dec,
                x_enc_reduced.unsqueeze(1).expand(1, x_dec.size(1), 1)
            ], dim=-1)
        else:
            context_flows = x_dec + x_enc_reduced.unsqueeze(1).expand(-1, x_dec.size(1), -1)
        # context_flows = context_flows[~tgt_padding_mask]

        # 4. pass the encoded features through the classifier
        # add the output time steps to the encoded features
        x_class = x_enc_reduced + self.classifier_context_embed(tgt_t)
        x_class = self.classifier(x_class)

        return context_flows, x_class

    def training_step(self, batch, batch_idx):
        batch_dict = self._prepare_batch(batch)
        batch_size = batch_dict['batch_size']
        weights = batch_dict['weights'] / batch_dict['weights'].sum()
        weights = torch.ones_like(weights) / batch_size

        # forward pass
        context_flows, x_class = self.forward(
            src=batch_dict['src'],
            src_t=batch_dict['src_t'],
            tgt_in=batch_dict['tgt_in'],
            tgt_t=batch_dict['tgt_t'],
            src_padding_mask=batch_dict['src_padding_mask'],
            tgt_padding_mask=batch_dict['tgt_padding_mask'],
        )
        # compute the flow loss
        if self.training_args.training_mode in ('all', 'npe'):
            lp = self.npe.log_prob(batch_dict['tgt_out'], context=context_flows)
            lp = lp * batch_dict['tgt_padding_mask'].eq(0).float()
            if self.training_args.old_flows_loss:
                lp = torch.sum(lp) / batch_dict['tgt_padding_mask'].eq(0).sum()
            else:
                lp = torch.sum(torch.sum(lp, -1) * weights)
            flows_loss = -lp
        else:
            flows_loss = 0

        # compute the classifier loss and accuracy
        if self.training_args.training_mode in ('all', 'classifier'):
            class_loss = torch.nn.CrossEntropyLoss(reduction='none')(
                x_class, batch_dict['class_target'])
            class_loss = torch.sum(class_loss * weights)
            class_acc = batch_dict['class_target'].eq(
                x_class.argmax(dim=1)).float().mean()
        else:
            class_loss = 0
            class_acc = 0

        # combine the losses
        loss = flows_loss + self.training_args.class_loss_weight * class_loss

        # log the loss and accuracy
        self.log(
            'train_flows_loss', flows_loss, on_step=True, on_epoch=True,
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
        weights = batch_dict['weights'] / batch_dict['weights'].sum()
        weights = torch.ones_like(weights) / batch_size

        # forward pass
        context_flows, x_class = self.forward(
            src=batch_dict['src'],
            src_t=batch_dict['src_t'],
            tgt_in=batch_dict['tgt_in'],
            tgt_t=batch_dict['tgt_t'],
            src_padding_mask=batch_dict['src_padding_mask'],
            tgt_padding_mask=batch_dict['tgt_padding_mask'],
        )
        # compute the flow loss
        if self.training_args.training_mode in ('all', 'npe'):
            lp = self.npe.log_prob(batch_dict['tgt_out'], context=context_flows)
            lp = lp * batch_dict['tgt_padding_mask'].eq(0).float()
            if self.training_args.old_flows_loss:
                lp = torch.sum(lp) / batch_dict['tgt_padding_mask'].eq(0).sum()
            else:
                lp = torch.sum(torch.sum(lp, -1) * weights)
            flows_loss = -lp
        else:
            flows_loss = 0

        # compute the classifier loss and accuracy
        if self.training_args.training_mode in ('all', 'classifier'):
            class_loss = torch.nn.CrossEntropyLoss(reduction='none')(
                x_class, batch_dict['class_target'])
            class_loss = torch.sum(class_loss * weights)
            class_acc = batch_dict['class_target'].eq(
                x_class.argmax(dim=1)).float().mean()
        else:
            class_loss = 0
            class_acc = 0

        # combine the losses
        loss = flows_loss + self.training_args.class_loss_weight * class_loss

        # log the loss and accuracy
        self.log(
            'val_flows_loss', flows_loss, on_step=True, on_epoch=True,
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
