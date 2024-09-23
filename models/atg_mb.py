from typing import Union, Dict, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from ml_collections import ConfigDict

from models.models import GRUDecoder
from models.flows import NPE
from models import models_utils, training_utils

class AutoregTreeGenMB(pl.LightningModule):
    def __init__(
        self,
        d_in: int,
        encoder_args: Union[ConfigDict, Dict],
        npe_args: Union[ConfigDict, Dict],
        optimizer_args: Union[ConfigDict, Dict],
        scheduler_args: Union[ConfigDict, Dict],
        norm_dict: Optional[Dict] = None,
    ):
        super().__init__()
        self.d_in = d_in
        self.encoder_args = encoder_args
        self.npe_args = npe_args
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args
        self.norm_dict = norm_dict
        self.save_hyperparameters()

        self.encoder = GRUDecoder(
            d_in=self.d_in,
            d_model=self.encoder_args.d_model,
            d_out=self.encoder_args.d_out,
            dim_feedforward=self.encoder_args.dim_feedforward,
            num_layers=self.encoder_args.num_layers,
            d_time=2,
            activation_fn=nn.ReLU(),
            concat=self.encoder_args.concat,
        )
        self.npe = NPE(
            input_size=self.d_in,
            context_size=self.encoder_args.d_out,
            hidden_sizes=self.npe_args.hidden_sizes,
            context_embedding_sizes=self.npe_args.context_embedding_sizes,
            num_transforms=self.npe_args.num_transforms,
            dropout=self.npe_args.dropout,
        )

    def _prepare_batch(self, batch):
        feat, feat_len = batch
        max_len = feat_len.max().item()
        x, t = feat[:, :max_len, :self.d_in], feat[:, :max_len, self.d_in:]

        # divide the data into input and target, each shifted by one time step
        src, tgt = x[:, :-1], x[:, 1:]
        src_t = torch.cat([t[:, :-1], t[:, 1:]], dim=-1)
        src_len = feat_len - 1
        src_len = tgt_len = src_len.cpu()

        # create padding mask
        tgt_padding_mask = training_utils.create_padding_mask(
            tgt_len.cpu(), src.size(1), batch_first=True)

        # move the data to the correct device
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        src_t = src_t.to(self.device)
        tgt_padding_mask = tgt_padding_mask.to(self.device)

        return {
            "src": src,
            "src_t": src_t,
            "src_len": src_len,
            "tgt": tgt,
            "tgt_len": tgt_len,
            "tgt_padding_mask": tgt_padding_mask,
            "batch_size": x.size(0),
        }

    def forward(self, src, src_t, src_len):
        context = self.encoder(src, src_t, src_len)
        return context

    def training_step(self, batch, batch_idx):
        batch_dict = self._prepare_batch(batch)
        batch_size = batch_dict['batch_size']

        # forward pass
        context = self(
            src=batch_dict['src'],
            src_t=batch_dict['src_t'],
            src_len=batch_dict['src_len'],
        )
        lp = self.npe.log_prob(batch_dict['tgt'], context=context)
        lp = lp * batch_dict['tgt_padding_mask'].eq(0).float()
        loss = -lp.mean()
        self.log(
            'train_loss', loss, on_step=True, on_epoch=True,
            logger=True, prog_bar=True,  batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_dict = self._prepare_batch(batch)
        batch_size = batch_dict['batch_size']

        # forward pass
        context = self(
            src=batch_dict['src'],
            src_t=batch_dict['src_t'],
            src_len=batch_dict['src_len'],
        )
        lp = self.npe.log_prob(batch_dict['tgt'], context=context)
        lp = lp * batch_dict['tgt_padding_mask'].eq(0).float()
        loss = -lp.mean()
        self.log(
            'val_loss', loss, on_step=True, on_epoch=True,
            logger=True, prog_bar=True,  batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        """ Initialize optimizer and LR scheduler """
        return models_utils.configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args)
