
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerFeaturizer(nn.Module):
    """
    Featurizer based on the TransformerEncoder module from PyTorch.

    Attributes
    ----------
    embedding : nn.Linear
        The embedding layer.
    transformer_encoder : nn.TransformerEncoder
        The transformer encoder.
    """

    def __init__(
        self,
        input_size,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        sum_features=False,
        batch_first=True,
        use_embedding=True,
        activation_fn=None,
    ):
        """
        Parameters
        ----------
        input_size : int
            The size of the input
        d_model : int
            The number of expected features in the encoder/decoder inputs.
        nhead : int
            The number of heads in the multiheadattention models.
        num_encoder_layers : int
            The number of sub-encoder-layers in the encoder.
        sum_features : bool, optional
            Whether to sum the features along the sequence dimension. Default: False
        dim_feedforward : int
            The dimension of the feedforward network model.
        batch_first : bool, optional
            If True, then the input and output tensors are provided as
            (batch, seq, feature). Default: True
        use_embedding : bool, optional
            Whether to use an embedding layer. Default: True
        activation_fn : callable, optional
            The activation function to use for the embedding layer. Default: None
        """
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.batch_first = True
        self.use_embedding = use_embedding
        self.sum_features = sum_features
        self.activation_fn = activation_fn

        if use_embedding:
            self.embedding = nn.Linear(input_size, d_model)
        else:
            assert input_size == d_model, (
                "If not using embedding, input_size must be equal to d_model."
                f"Got input_size={input_size} and d_model={d_model}"
            )
            self.embedding = nn.Identity()

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers)

    def forward(self, src, src_key_padding_mask=None):
        src = self.embedding(src)
        if self.activation_fn is not None:
            src = self.activation_fn(src)
        output = self.transformer_encoder(
            src, src_key_padding_mask=src_key_padding_mask)

        # NOTE: only work when batch_first=True
        if self.sum_features:
            # set all the padding tokens to zero then sum over
            output = output.masked_fill(src_key_padding_mask.unsqueeze(-1), 0)
            output = output.sum(dim=1)
        else:
            lengths = src_key_padding_mask.eq(0).sum(dim=1)
            batch_size = output.shape[0]
            output = output[torch.arange(batch_size).to(src.device), lengths-1]

        return output


class MLP(nn.Module):
    """
    MLP with a variable number of hidden layers.

    Attributes
    ----------
    layers : nn.ModuleList
        The layers of the MLP.
    activation_fn : callable
        The activation function to use.
    """
    def __init__(self, input_size, output_size, hidden_sizes=[512],
                 activation_fn=nn.ReLU()):
        """
        Parameters
        ----------
        input_size : int
            The size of the input
        output_size : int
            The number of classes
        hidden_sizes : list of int, optional
            The sizes of the hidden layers. Default: [512]
        activation_fn : callable, optional
            The activation function to use. Default: nn.ReLU()
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation_fn = activation_fn

        # Create a list of all layer sizes: input, hidden, and output
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Create layers dynamically
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        # Store the activation function
        self.activation_fn = activation_fn

    def forward(self, x):
        # Apply layers and activation function
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation function to all but last layer
                x = self.activation_fn(x)
        return x

class GRUModel(nn.Module):
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