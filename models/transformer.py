
import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class FourierTimeEmbedding(nn.Module):
    def __init__(self, d_model, n_frequencies=16):
        super().__init__()
        # randomly initialize frequencies for the Fourier transformation
        self.B = nn.Parameter(torch.randn(n_frequencies, 1) * 2 * math.pi)

    def forward(self, time):
        # Apply sine and cosine functions for Fourier transformation
        time_proj = torch.matmul(time.unsqueeze(-1), self.B.T)
        time_emb = torch.cat([torch.sin(time_proj), torch.cos(time_proj)], dim=-1)
        return time_emb


class Transformer(nn.Module):
    def __init__(
        self,
        d_in,
        d_model,
        nhead,
        dim_feedforward,
        num_layers,
        emb_size=16,
        emb_dropout=0.1,
        sum_features=False,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.emb_size = emb_size
        self.emb_dropout = emb_dropout
        self._setup_model()

    def _setup_model(self):
        self.embedding = nn.Linear(self.d_in, self.d_model)
        self.time_embedding = FourierTimeEmbedding(self.d_model)
        self.dropout = nn.Dropout(self.emb_dropout)

        transformer_layer = TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.nhead,
            dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = Transformer(transformer_layer, self.num_layers)

    def forward(self, x, t, padding_mask=None):

        # embed the input and timestep
        x = self.embedding(x)
        t = self.time_embedding(t)
        x = x + t
        x = self.dropout(x)

        # pass through the transformer
        output = self.transformer(x, src_key_padding_mask=padding_mask)

        # NOTE: only work when batch_first=True
        if self.sum_features:
            if padding_mask is None:
                output = output.sum(dim=1)
            else:
                # set all the padding tokens to zero then sum over
                output = output.masked_fill(padding_mask.unsqueeze(-1), 0)
                output = output.sum(dim=1)
        else:
            if padding_mask is None:
                output = output[:, -1]
            else:
                lengths = padding_mask.eq(0).sum(dim=1)
                batch_size = output.shape[0]
                output = output[torch.arange(batch_size).to(x.device), lengths-1]

        return output
