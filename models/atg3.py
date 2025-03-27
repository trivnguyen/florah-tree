from typing import Union, Dict, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from ml_collections.config_dict import ConfigDict

from models import flows, models_utils, training_utils
from models.transformer import TransformerEncoder, TransformerDecoder
from models.models import MLP, GRU

class AutoregTreeGen3(pl.LightningModule):
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

        # training args
        training_args_default = ConfigDict({
            'training_mode': 'all',
            'class_loss_weight': 1.0,
            'use_sample_weight': False,
            'max_sample_weight': 1,
            'use_desc_mass_ratio': False,
            'num_branches_per_tree': None,
            'freeze_args': ConfigDict({
                'encoder': False,
                'decoder': False,
                'npe1': False,
                'npe2': False,
                'classifier': False,
            }),
        })
        self.training_args = training_args_default
        if training_args is not None:
            self.training_args.update(training_args)
        self.freeze_args = self.training_args.freeze_args
        if self.training_args.training_mode not in ('all', 'classifier', 'npe'):
            raise ValueError(
                f'Training mode {self.training_mode} not supported')
        self.save_hyperparameters()

        self._setup_model()

    def _freeze_component(self, component, freeze=True):
        for param in component.parameters():
            param.requires_grad = not freeze

    def _setup_model(self):

        # 1. Create the encoder network
        if self.encoder_args.name == 'transformer':
            self.encoder = TransformerEncoder(
                d_in=self.d_in,
                d_model=self.encoder_args.d_model,
                nhead=self.encoder_args.nhead,
                dim_feedforward=self.encoder_args.dim_feedforward,
                num_layers=self.encoder_args.num_layers,
                d_time=1,
                emb_size=self.encoder_args.emb_size,
                emb_dropout=self.encoder_args.emb_dropout,
                concat=self.encoder_args.concat
            )
        elif self.encoder_args.name == 'gru':
            self.encoder = GRU(
                d_in=self.d_in,
                d_model=self.encoder_args.d_model,
                d_out=self.encoder_args.d_out,
                dim_feedforward=self.encoder_args.dim_feedforward,
                num_layers=self.encoder_args.num_layers,
                d_time=1,
                activation_fn=nn.ReLU(),
                concat=self.encoder_args.concat,
            )
        else:
            raise NotImplementedError(
                f'Encoder {self.encoder_args.name} not implemented')

        # 2. Create the decoder network
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
            self.decoder = GRU(
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

        # 3. Create the NPE
        self.npe_class_embed = nn.Linear(self.num_classes, self.decoder_args.d_model)
        self.npe_position_embed = nn.Linear(self.num_classes, self.decoder_args.d_model)
        self.npe_time_embed = nn.Linear(1, self.decoder_args.d_model)
        self.npe1 = flows.NPE(
            input_size=self.d_in,
            context_size=self.decoder_args.d_model,
            hidden_sizes=self.npe_args.hidden_sizes,
            context_embedding_sizes=self.npe_args.context_embedding_sizes,
            num_transforms=self.npe_args.num_transforms,
            dropout=self.npe_args.dropout,
        )
        self.npe2 = flows.NPE(
            input_size=self.d_in,
            context_size=self.decoder_args.d_model,
            hidden_sizes=self.npe_args.hidden_sizes,
            context_embedding_sizes=self.npe_args.context_embedding_sizes,
            num_transforms=self.npe_args.num_transforms,
            dropout=self.npe_args.dropout,
        )

        # 4. Create the classifier
        self.classifier_context_embed = nn.Linear(
            self.classifier_args.d_context, self.encoder_args.d_model)
        self.classifier = MLP(
            d_in=self.encoder_args.d_model,
            hidden_sizes=self.classifier_args.hidden_sizes + [self.num_classes,],
        )

        # Freeze the components
        self._freeze_component(self.encoder, self.freeze_args.encoder)
        self._freeze_component(self.decoder, self.freeze_args.decoder)
        self._freeze_component(self.npe1, self.freeze_args.npe)
        self._freeze_component(self.npe2, self.freeze_args.npe)
        self._freeze_component(self.npe_class_embed, self.freeze_args.npe)
        self._freeze_component(self.npe_position_embed, self.freeze_args.npe)
        self._freeze_component(self.npe_time_embed, self.freeze_args.npe)
        self._freeze_component(self.classifier, self.freeze_args.classifier)
        self._freeze_component(self.classifier_context_embed, self.freeze_args.classifier)

        # Check if the freezing is working appropriately
        for name, component in zip(
            ['encoder', 'decoder', 'npe1', 'npe2', 'classifier'],
            [self.encoder, self.decoder, self.npe1, self.npe2, self.classifier]
        ):
            print(f'Checking {name} requires_grad')
            for param in component.parameters():
                assert param.requires_grad == (not self.freeze_args[name])
                print(f'{name} requires_grad: {param.requires_grad}')

    def _prepare_batch(self, batch):
        """ Prepare the batch for training. """
        src_feat, tgt_feat, num_prog, src_len, _ = training_utils.prepare_batch_branch(
            batch, self.num_classes,
            use_desc_mass_ratio=self.training_args.use_desc_mass_ratio,
            num_branches_per_tree=self.training_args.num_branches_per_tree,
            use_prog_position=False,
            return_weights=False,
        )
        # Processing Transformer input and time steps
        src, src_t = src_feat[..., :-1], src_feat[..., -1:]
        src_padding_mask = training_utils.create_padding_mask(
            src_len, src_feat.size(1), batch_first=True)

        # Processing RNN/Flows input, output and timesteps
        tgt_feat = tgt_feat[~src_padding_mask]
        num_prog = num_prog[~src_padding_mask]
        tgt, tgt_t = tgt_feat[:, :, :-1], tgt_feat[:, 0, -1:]

        # then, divide the rnn into the input and output component
        # the input will be feed into the RNN,
        # while the output will be used for the flow loss
        mask = num_prog > 1
        tgt_out1 = tgt[~mask, 0]
        tgt_in2, tgt_out2 = tgt[mask, :-1], tgt[mask, 1:]
        tgt_padding_mask2 = training_utils.create_padding_mask(
            num_prog[mask].cpu()-1, tgt_feat[mask].size(1)-1, batch_first=True)

        # Classifier target, which is the original length of the output
        # # features minus 1 (start from 0)
        class_target = num_prog - 1
        class_target_onehot = torch.nn.functional.one_hot(
            class_target, self.num_classes).float()
        class_target_onehot1 = class_target_onehot[~mask]
        class_target_onehot2 = class_target_onehot[mask]

        # create a one-hot position vector for tgt features
        tgt_pos = torch.arange(tgt.size(1)).unsqueeze(0).expand(tgt.size(0), -1)
        tgt_pos = torch.nn.functional.one_hot(tgt_pos, self.num_classes).float()
        tgt_pos2 = tgt_pos[mask]

        tgt_t1 = tgt_t[~mask]
        tgt_t2 = tgt_t[mask]

        # Move to the same device as the model
        src = src.to(self.device)
        tgt_out1 = tgt_out1.to(self.device)
        tgt_in2 = tgt_in2.to(self.device)
        tgt_out2 = tgt_out2.to(self.device)
        src_t = src_t.to(self.device)
        tgt_t1 = tgt_t1.to(self.device)
        tgt_t2 = tgt_t2.to(self.device)
        tgt_t = tgt_t.to(self.device)
        src_padding_mask = src_padding_mask.to(self.device)
        tgt_padding_mask2 = tgt_padding_mask2.to(self.device)
        class_target = class_target.to(self.device)
        class_target_onehot1 = class_target_onehot1.to(self.device)
        class_target_onehot2 = class_target_onehot2.to(self.device)
        tgt_pos2 = tgt_pos2.to(self.device)

        # return a dictionary of the inputs
        return {
            'src': src,
            'src_t': src_t,
            'tgt_t1': tgt_t1,
            'tgt_t2': tgt_t2,
            'tgt_t': tgt_t,
            'tgt_out1': tgt_out1,
            'tgt_in2': tgt_in2,
            'tgt_out2': tgt_out2,
            'tgt_pos2': tgt_pos2,
            'src_padding_mask': src_padding_mask,
            'tgt_padding_mask2': tgt_padding_mask2,
            'class_target': class_target,
            'class_target_onehot1': class_target_onehot1,
            'class_target_onehot2': class_target_onehot2,
            'batch_size': src.size(0),
            'src_len': src_len,
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
        x_enc = self.encoder(src, src_t, src_padding_mask)
        x_enc = x_enc[~src_padding_mask]

        # 2. pass the encoded features through the decoder
        # using the output time steps as the context
        if self.decoder_args.name == 'transformer':
            x_dec = self.decoder(
                tgt_in,
                memory=x_enc.unsqueeze(1),
                tgt_padding_mask=tgt_padding_mask,
                memory_padding_mask=None
            )
        elif self.decoder_args.name == 'gru':
            x_dec = self.decoder(
                x=tgt_in,
                padding_mask=tgt_padding_mask
            )

        # 3. pass the encoded features through the classifier
        # add the output time steps to the encoded features
        x_class = x_enc + self.classifier_context_embed(tgt_t)
        x_class = self.classifier(x_class)

        return x_enc, x_dec, x_class

    def training_step(self, batch, batch_idx):
        batch_dict = self._prepare_batch(batch)
        batch_size = batch_dict['batch_size']

        # forward pass
        x_enc, x_dec, x_class = self.forward(
            src=batch_dict['src'],
            src_t=batch_dict['src_t'],
            tgt_in=batch_dict['tgt_in2'],
            tgt_t=batch_dict['tgt_t'],
            src_padding_mask=batch_dict['src_padding_mask'],
            tgt_padding_mask=batch_dict['tgt_padding_mask2'],
        )

        # # compute the flow loss
        if self.training_args.training_mode in ('all', 'npe'):
            class_context1 = self.npe_class_embed(batch_dict['class_target_onehot1'])
            time_context1 = self.npe_time_embed(batch_dict['tgt_t1'])
            context1 = class_context1 + time_context1
            lp1 = self.npe1.log_prob(batch_dict['tgt_out1'], context=context1)


            class_context2 = self.npe_class_embed(batch_dict['class_target_onehot2'])
            time_context2 = self.npe_time_embed(batch_dict['tgt_t2'])
            pos_context2 = self.npe_position_embed(batch_dict['tgt_pos2'])
            class_context2 = class_context2.unsqueeze(1).expand_as(x_dec)
            time_context2 = time_context2.unsqueeze(1).expand_as(x_dec)

            context2 = class_context2 + time_context2 + pos_context2[:, 1:]
            lp2 = self.npe2.log_prob(batch_dict['tgt_out2'], context=context2)
            lp2 = lp2 * batch_dict['tgt_padding_mask2'].eq(0).float()

            flows_loss = -lp1.mean() - lp2.mean()
        else:
            flows_loss = 0

        # compute the classifier loss and accuracy
        if self.training_args.training_mode in ('all', 'classifier'):
            class_loss = torch.nn.CrossEntropyLoss(reduction='none')(
                x_class, batch_dict['class_target'])
            class_loss = class_loss.mean()
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

        # forward pass
        x_enc, x_dec, x_class = self.forward(
            src=batch_dict['src'],
            src_t=batch_dict['src_t'],
            tgt_in=batch_dict['tgt_in2'],
            tgt_t=batch_dict['tgt_t'],
            src_padding_mask=batch_dict['src_padding_mask'],
            tgt_padding_mask=batch_dict['tgt_padding_mask2'],
        )

        # # compute the flow loss
        if self.training_args.training_mode in ('all', 'npe'):
            class_context1 = self.npe_class_embed(batch_dict['class_target_onehot1'])
            time_context1 = self.npe_time_embed(batch_dict['tgt_t1'])
            context1 = class_context1 + time_context1
            lp1 = self.npe1.log_prob(batch_dict['tgt_out1'], context=context1)


            class_context2 = self.npe_class_embed(batch_dict['class_target_onehot2'])
            time_context2 = self.npe_time_embed(batch_dict['tgt_t2'])
            pos_context2 = self.npe_position_embed(batch_dict['tgt_pos2'])
            class_context2 = class_context2.unsqueeze(1).expand_as(x_dec)
            time_context2 = time_context2.unsqueeze(1).expand_as(x_dec)

            context2 = class_context2 + time_context2 + pos_context2[:, 1:]
            lp2 = self.npe2.log_prob(batch_dict['tgt_out2'], context=context2)
            lp2 = lp2 * batch_dict['tgt_padding_mask2'].eq(0).float()

            flows_loss = -lp1.mean() - lp2.mean()
        else:
            flows_loss = 0

        # compute the classifier loss and accuracy
        if self.training_args.training_mode in ('all', 'classifier'):
            class_loss = torch.nn.CrossEntropyLoss(reduction='none')(
                x_class, batch_dict['class_target'])
            class_loss = class_loss.mean()
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
