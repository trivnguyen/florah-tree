
import torch
import torch.nn as nn

class MLP(nn.Module):
    """ A simple MLP. """

    def __init__(self, d_in, hidden_sizes, activation=nn.GELU()):
        super().__init__()
        self.d_in = d_in
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        self.layers = nn.ModuleList()
        for i, h in enumerate(hidden_sizes):
            h_in = d_in if i == 0 else hidden_sizes[i - 1]
            h_out = h
            self.layers.append(nn.Linear(h_in, h_out))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        # No activation on final layer
        x = self.layers[-1](x)
        return x

class GRUDecoder(nn.Module):
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
    def __init__(self, input_size, output_size, hidden_size, num_layers=1,
                 activation_fn=nn.ReLU()):
        """
        Parameters
        ----------
        input_size : int
            The size of the input
        output_size : int
            The number of classes
        hidden_size : int
            The size of the hidden layers
        num_layers : int, optional
            The number of hidden layers. Default: 1
        activation_fn : callable, optional
            The activation function to use. Default: nn.ReLU()
        """
        super().__init__()

        self.gru_layers = nn.GRU(
            input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.activation_fn = activation_fn

    def forward(self, x, lengths, return_hidden_states=False):
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        x, hout = self.gru_layers(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.activation_fn(x)
        x = self.linear(x)

        if return_hidden_states:
            return x, hout
        return x
