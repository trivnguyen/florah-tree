
import numpy as np
import torch
import torch.nn as nn
import math

def get_activation(activation):
    """ Get an activation function. """
    if activation.name.lower() == 'identity':
        return nn.Identity()
    elif activation.name.lower() == 'relu':
        return nn.ReLU(inplace=False)
    elif activation.name.lower() == 'tanh':
        return nn.Tanh()
    elif activation.name.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif activation.name.lower() == 'leaky_relu':
        return nn.LeakyReLU(activation.leaky_relu_alpha, inplace=False)
    elif activation.name.lower() == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f'Unknown activation function: {activation.name}')

class WarmUpCosineAnnealingLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, decay_steps, warmup_steps, eta_min=0, last_epoch=-1, restart=True):
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        self.restart = restart
        super().__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if self.restart:
            step = step % self.decay_steps
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return self.eta_min + (
            0.5 * (1 + math.cos(math.pi * (step - self.warmup_steps) / (self.decay_steps - self.warmup_steps))))


class WarmUpCosineDecayLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self, optimizer, init_value, peak_value, warmup_steps, decay_steps,
        end_value=0.0, exponent=1.0, last_epoch=-1):
        self.init_value = init_value
        self.peak_value = peak_value
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.end_value = end_value
        self.exponent = exponent

        super().__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def cosine_decay_schedule(self, step, init_value, decay_steps, alpha=0.0, exponent=1.0):
        step = min(step, self.decay_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / self.decay_steps))
        decayed = (1 - alpha) * cosine_decay ** exponent + alpha
        return init_value * decayed

    def linear_schedule(self, step, init_value, peak_value, warmup_steps):
        return init_value + (peak_value - init_value) * step / warmup_steps

    def lr_lambda(self, step):
        alpha = 0 if self.peak_value == 0 else self.end_value / self.peak_value
        if step < self.warmup_steps:
            return self.linear_schedule(
                step, self.init_value, self.peak_value, self.warmup_steps)
        return self.cosine_decay_schedule(
            step - self.warmup_steps,
            init_value=self.peak_value,
            decay_steps=self.decay_steps - self.warmup_steps,
            alpha=alpha,
            exponent=self.exponent,
        )

def configure_optimizers(parameters, optimizer_args, scheduler_args):
    """ Return optimizer and scheduler. """
    # setup the optimizer
    if optimizer_args.name == "Adam":
        optimizer =  torch.optim.Adam(
            parameters,
            lr=optimizer_args.lr,
            weight_decay=optimizer_args.weight_decay
        )
    elif optimizer_args.name == "AdamW":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=optimizer_args.lr,
            weight_decay=optimizer_args.weight_decay
        )
    else:
        raise NotImplementedError(
            "Optimizer {} not implemented".format(optimizer_args.name))

    # setup the scheduler
    if scheduler_args.get('name') is None:
        scheduler = None
    elif scheduler_args.name == 'ReduceLROnPlateau':
        scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            factor=scheduler_args.factor,
            patience=scheduler_args.patience
        )
    elif scheduler_args.name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_args.T_max,
            eta_min=scheduler_args.eta_min
        )
    elif scheduler_args.name == 'WarmUpCosineDecayLR':
        scheduler = WarmUpCosineDecayLR(
            optimizer,
            init_value=scheduler_args.init_value,
            peak_value=scheduler_args.peak_value,
            warmup_steps=scheduler_args.warmup_steps,
            decay_steps=scheduler_args.decay_steps,
            restart=scheduler_args.get('restart', True),
        )
    elif scheduler_args.name == 'WarmUpCosineAnnealingLR':
        scheduler = WarmUpCosineAnnealingLR(
            optimizer,
            decay_steps=scheduler_args.decay_steps,
            warmup_steps=scheduler_args.warmup_steps,
            eta_min=scheduler_args.eta_min)
    else:
        raise NotImplementedError(
            "Scheduler {} not implemented".format(scheduler_args.name))

    if scheduler is None:
        return optimizer
    else:
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
                'interval': scheduler_args.interval,
                'frequency': 1
            }
        }

def summarize_features(x, reduction='sum', padding_mask=None):
    if reduction == 'sum':
        if padding_mask is None:
            x = x.sum(dim=1)
        else:
            # set all the padding tokens to zero then sum over
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
    elif reduction == 'mean':
        if padding_mask is None:
            x = x.sum(dim=1) / x.size(1)
        else:
            # set all the padding tokens to zero then sum over
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
            x = x.sum(dim=1) / padding_mask.eq(0).sum(dim=1).unsqueeze(-1)
    elif reduction == 'last':
        if padding_mask is None:
            x = x[:, -1]
        else:
            lengths = padding_mask.eq(0).sum(dim=1)
            x = x[torch.arange(x.size(0), device=x.device), lengths-1]
    else:
        raise ValueError(f'Unknown reduction: {reduction}')

    return x

def get_casual_mask(size, device):
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    return mask.bool()
