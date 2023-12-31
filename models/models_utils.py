
import torch
import torch.nn as nn

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