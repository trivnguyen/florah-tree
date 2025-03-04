from typing import Union, Dict, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from ml_collections.config_dict import ConfigDict

from models import flows, models_utils, training_utils
from models.transformer import TransformerEncoder, TransformerDecoder
from models.models import MLP, GRU

class BinaryTreeGen(pl.LightningModule):
    """ Autoregressive tree generator model """
    def __init__(
        self,
        d_in: int,
        encoder_args: Union[ConfigDict, Dict],
        npe_args: Union[ConfigDict, Dict],
        classifier_args: Union[ConfigDict, Dict],
        optimizer_args: Union[ConfigDict, Dict] = None,
        scheduler_args: Union[ConfigDict, Dict] = None,
        norm_dict: Optional[Dict] = None,
    ):
        super().__init__()
        self.d_in = d_in
        self.encoder_args = encoder_args
        self.npe_args = npe_args
        self.classifier_args = classifier_args
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args
        self.norm_dict = norm_dict
        self.save_hyperparameters()

        self._setup_model()

    def _setup_model(self):

        # 1. Create the Encoder
        self.encoder = GRU(
            d_in=self.d_in + 2,
            d_model=self.encoder_args.d_model,
            d_out=self.encoder_args.d_out,
            dim_feedforward=self.encoder_args.dim_feedforward,
            num_layers=self.encoder_args.num_layers,
            d_time=1,
            activation_fn=nn.ReLU(),
            concat=self.encoder_args.concat,
        )

        # 2. Create NPE 1
        self.npe1_context_embed = nn.Linear(2, self.encoder_args.d_model)
        self.npe1 = flows.NPE(
            input_size=self.d_in,
            context_size=self.encoder_args.d_model,
            hidden_sizes=self.npe_args.hidden_sizes,
            context_embedding_sizes=self.npe_args.context_embedding_sizes,
            num_transforms=self.npe_args.num_transforms,
            dropout=self.npe_args.dropout,
        )

        # Craete NPE 2
        self.npe2_context_embed = nn.Linear(self.d_in, self.encoder_args.d_model)
        self.npe2 = flows.NPE(
            input_size=self.d_in,
            context_size=self.encoder_args.d_model,
            hidden_sizes=self.npe_args.hidden_sizes,
            context_embedding_sizes=self.npe_args.context_embedding_sizes,
            num_transforms=self.npe_args.num_transforms,
            dropout=self.npe_args.dropout,
        )

        # 4. Create the classifier
        self.classifier = MLP(
            d_in=self.encoder_args.d_model,
            hidden_sizes=self.classifier_args.hidden_sizes + [2,],
        )

    def _prepare_batch(self, batch):
        """ Prepare the batch for training. """
        src_feat, tgt_feat, num_prog, src_len, _  = training_utils.prepare_batch_branch(
            batch,
            max_split=2,
            return_weights=False,
            use_desc_mass_ratio=False,
            use_prog_position=False,
        )
        src = torch.cat([src_feat, tgt_feat[..., 0, -1:]], dim=-1)
        src_padding_mask = training_utils.create_padding_mask(
            src_len, src_feat.size(1), batch_first=True)

        # classifier
        class_target = num_prog[~src_padding_mask] - 1
        class_target_onehot = torch.nn.functional.one_hot(class_target, 2).float()

        # npe1
        tgt1_out = tgt_feat[..., 0, :-1]
        tgt1_out = tgt1_out[~src_padding_mask]

        # npe2
        npe2_mask = num_prog[~src_padding_mask] > 1
        tgt2_in = tgt1_out[npe2_mask]
        tgt2_out = tgt_feat[..., 1, :-1]
        tgt2_out = tgt2_out[~src_padding_mask][npe2_mask]

        # Move to the same device as the model
        src = src.to(self.device)
        src_padding_mask = src_padding_mask.to(self.device)
        tgt1_out = tgt1_out.to(self.device)
        tgt2_in = tgt2_in.to(self.device)
        tgt2_out = tgt2_out.to(self.device)
        class_target = class_target.to(self.device)
        npe2_mask = npe2_mask.to(self.device)

        # return a dictionary of the inputs
        return {
            'src': src,
            'src_padding_mask': src_padding_mask,
            'tgt1_out': tgt1_out,
            'tgt2_in': tgt2_in,
            'tgt2_out': tgt2_out,
            'class_target': class_target,
            'class_target_onehot': class_target_onehot,
            'npe2_mask': npe2_mask,
            'batch_size': src.size(0),
        }

    def training_step(self, batch, batch_idx):
        batch_dict =  self._prepare_batch(batch)
        src = batch_dict['src']
        src_padding_mask = batch_dict['src_padding_mask']
        tgt1_out = batch_dict['tgt1_out']
        tgt2_in = batch_dict['tgt2_in']
        tgt2_out = batch_dict['tgt2_out']
        class_target = batch_dict['class_target']
        class_target_onehot = batch_dict['class_target_onehot']
        npe2_mask = batch_dict['npe2_mask']
        batch_size = batch_dict['batch_size']

        x_enc = self.encoder(src, None, padding_mask=src_padding_mask)
        x_enc = x_enc[~src_padding_mask]

        # npe 1 loss
        x_enc1 = x_enc + self.npe1_context_embed(class_target_onehot)
        npe1_loss = self.npe1.log_prob(tgt1_out, context=x_enc1)
        npe1_loss = -npe1_loss.mean()

        # npe 2 loss
        x_enc2 = x_enc[npe2_mask] + self.npe2_context_embed(tgt2_in)
        npe2_loss = self.npe2.log_prob(tgt2_out, context=x_enc2)
        npe2_loss = -npe2_loss.mean()

        # classifier loss
        x_class = self.classifier(x_enc)
        class_loss = torch.nn.CrossEntropyLoss(reduction='mean')(x_class, class_target)
        class_acc = class_target.eq(x_class.argmax(dim=1)).float().mean()

        # total loss
        loss = npe1_loss + npe2_loss + class_loss

        # log the loss and accuracy
        self.log(
            'train_npe1_loss', npe1_loss, on_step=True, on_epoch=True,
            logger=True, prog_bar=True, batch_size=batch_size)
        self.log(
            'train_npe2_loss', npe2_loss, on_step=True, on_epoch=True,
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
        batch_dict =  self._prepare_batch(batch)
        src = batch_dict['src']
        src_padding_mask = batch_dict['src_padding_mask']
        tgt1_out = batch_dict['tgt1_out']
        tgt2_in = batch_dict['tgt2_in']
        tgt2_out = batch_dict['tgt2_out']
        class_target = batch_dict['class_target']
        class_target_onehot = batch_dict['class_target_onehot']
        npe2_mask = batch_dict['npe2_mask']
        batch_size = batch_dict['batch_size']

        x_enc = self.encoder(
            x=src, t=None, lengths=src_padding_mask.eq(0).sum(-1).cpu(),)
        x_enc = x_enc[~src_padding_mask]

        # npe 1 loss
        x_enc1 = x_enc + self.npe1_context_embed(class_target_onehot)
        npe1_loss = self.npe1.log_prob(tgt1_out, context=x_enc1)
        npe1_loss = -npe1_loss.mean()

        # npe 2 loss
        x_enc2 = x_enc[npe2_mask] + self.npe2_context_embed(tgt2_in)
        npe2_loss = self.npe2.log_prob(tgt2_out, context=x_enc2)
        npe2_loss = -npe2_loss.mean()

        # classifier loss
        x_class = self.classifier(x_enc)
        class_loss = torch.nn.CrossEntropyLoss(reduction='mean')(x_class, class_target)
        class_acc = class_target.eq(x_class.argmax(dim=1)).float().mean()

        # total loss
        loss = npe1_loss + npe2_loss + class_loss

        # log the loss and accuracy
        self.log(
            'val_npe1_loss', npe1_loss, on_step=True, on_epoch=True,
            logger=True, prog_bar=True, batch_size=batch_size)
        self.log(
            'val_npe2_loss', npe2_loss, on_step=True, on_epoch=True,
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
