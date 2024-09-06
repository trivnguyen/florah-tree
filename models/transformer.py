
import torch
import torch.nn as nn
import math

from models.models import MLP
from models import models_utils

class FourierTimeEmbedding(nn.Module):
    def __init__(self, emb_size: int):
        super().__init__()
        if emb_size % 2 != 0:
            raise ValueError("Embedding size must be even")

        # randomly initialize frequencies for the Fourier transformation
        self.B = nn.Parameter(torch.randn(emb_size // 2, 1) * 2 * math.pi)

    def forward(self, time):
        # Apply sine and cosine functions for Fourier transformation
        time_proj = torch.matmul(time.unsqueeze(-1), self.B.T)
        time_emb = torch.cat([torch.sin(time_proj), torch.cos(time_proj)], dim=-1)

        # reshape to match the input shape
        time_emb = time_emb.view(time.shape[0], time.shape[1], -1)
        return time_emb

class PositionalEmbedding(nn.Module):
    """ Positional embedding, adopted from PyTorch tutorial """
    def __init__(self, emb_size: int, maxlen: int = 5000):
        super().__init__()
        if emb_size % 2 != 0:
            raise ValueError("Embedding size must be even")

        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, seq_len):
        return self.pos_embedding[:, :seq_len]

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_in,
        d_model,
        nhead,
        dim_feedforward,
        num_layers,
        emb_size=16,
        emb_dropout=0.1,
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
        self.embedding = nn.Linear(self.d_in, self.emb_size)
        self.time_embedding = FourierTimeEmbedding(self.emb_size)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp = MLP(self.emb_size, [self.d_model, self.d_model * 4, self.d_model])

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=self.nhead,
                dim_feedforward=self.dim_feedforward, batch_first=True),
            self.num_layers)

    def forward(self, src, src_t, padding_mask=None):

        # embed the input and timestep
        x = self.embedding(src) + self.time_embedding(src_t)
        x = self.dropout(x)
        x = self.mlp(x)

        # pass through the transformer
        output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        return output

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_in,
        d_model,
        nhead,
        dim_feedforward,
        num_layers,
        emb_size=16,
        emb_dropout=0.1,
        max_len=3,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.emb_size = emb_size
        self.emb_dropout = emb_dropout
        self.max_len = max_len
        self._setup_model()

    def _setup_model(self):
        self.embedding = nn.Linear(self.d_in, self.emb_size)
        self.postional_embedding = PositionalEmbedding(self.emb_size, self.max_len)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp = MLP(self.emb_size, [self.d_model, self.d_model * 4, self.d_model])

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.d_model, nhead=self.nhead,
                dim_feedforward=self.dim_feedforward, batch_first=True),
            self.num_layers)

    def forward(self, tgt, cond, tgt_padding_mask=None, cond_padding_mask=None):

        x = self.embedding(tgt) + self.postional_embedding(tgt.size(1))
        x = self.dropout(x)
        x = self.mlp(x)

        # create casual mask
        tgt_mask = models_utils.get_casual_mask(x.size(1), device=x.device)

        # pass through the transformer_decoder
        output = self.transformer_decoder(
            x,
            cond,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=cond_padding_mask,
        )

        return output
