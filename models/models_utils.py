
import torch
import torch.nn as nn
import math

def get_activation(activation):
    """ Get an activation function. """
    if activation.name.lower() == 'identity':
        return nn.Identity()
    elif activation.name.lower() == 'relu':
        return nn.ReLU()
    elif activation.name.lower() == 'tanh':
        return nn.Tanh()
    elif activation.name.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif activation.name.lower() == 'leaky_relu':
        return nn.LeakyReLU(activation.leaky_relu_alpha)
    elif activation.name.lower() == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f'Unknown activation function: {activation.name}')

class WarmUpCosineAnnealingLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, decay_steps, warmup_steps, eta_min=0, last_epoch=-1):
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        super().__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return self.eta_min + (
            0.5 * (1 + math.cos(math.pi * (step - self.warmup_steps) / (self.decay_steps - warmup_steps))))
