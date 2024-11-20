from typing import List

import torch
import torch.nn as nn

class MLP(nn.Module):
    """ A simple MLP. """

    def __init__(
        self,
        d_in: int,
        hidden_sizes: List[int],
        activation=nn.GELU()
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        self.layers = nn.ModuleList()
        for i, h in enumerate(hidden_sizes):
            h_in = d_in if i == 0 else hidden_sizes[i - 1]
            h_out = h
            self.layers.append(nn.Linear(h_in, h_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        # No activation on final layer
        x = self.layers[-1](x)
        return x

class GRU(nn.Module):
    """
    GRU model with a variable number of hidden layers.

    Attributes
    ----------
    gru_layers : nn.GRU
        The GRU layers.
    linear : nn.Linear
        The linear layer.
    activation_fn : callable
        The activation function to use.
    """
    def __init__(
        self,
        d_in: int,
        d_model: int,
        d_out: int,
        dim_feedforward: int,
        num_layers: int = 1,
        d_time: int = 1,
        activation_fn=nn.ReLU(),
        concat: bool = False
    ) -> None:
        """
        Parameters
        ----------
        d_in : int
            The size of the input
        d_out : int
            The number of classes
        dim_feedforward : int
            The size of the hidden layers
        num_layers : int, optional
            The number of hidden layers. Default: 1
        d_model: int : int, optional
            The size of the embedding. Default: 16
        activation_fn : callable, optional
            The activation function to use. Default: nn.ReLU()
        concat: bool, optional
        """
        super().__init__()

        self.linear_x_proj = nn.Linear(d_in, d_model)
        self.linear_t_proj = nn.Linear(d_time, d_model)
        if concat:
            self.gru_layers = nn.GRU(
                d_model * 2, dim_feedforward, num_layers=num_layers, batch_first=True)
        else:
            self.gru_layers = nn.GRU(
                d_model, dim_feedforward, num_layers=num_layers, batch_first=True)

        self.linear = nn.Linear(dim_feedforward, d_out)
        self.activation_fn = activation_fn
        self.concat = concat

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
        return_hidden_states: bool = False
    ) -> torch.Tensor:
        # project the input and time before passing through the GRU
        if t is not None:
            if self.concat:
                x = torch.cat([self.linear_x_proj(x), self.linear_t_proj(t)], dim=-1)
            else:
                x = self.linear_x_proj(x) + self.linear_t_proj(t)
        else:
            x = self.linear_x_proj(x)

        # pack the sequence and pass through the GRU
        lengths = padding_mask.eq(0).sum(-1).cpu() if padding_mask else x.size(1)
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        x, hout = self.gru_layers(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.activation_fn(x)
        x = self.linear(x)

        if return_hidden_states:
            return x, hout
        return x
