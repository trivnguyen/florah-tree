
from typing import Optional

import torch
import torch.nn as nn
import math

from models.models import MLP
from models import models_utils

class FourierTimeEmbedding(nn.Module):
    def __init__(self, emb_size: int) -> None:
        super().__init__()
        if emb_size % 2 != 0:
            raise ValueError("Embedding size must be even")

        # randomly initialize frequencies for the Fourier transformation
        self.B = nn.Parameter(torch.randn(emb_size // 2, 1) * 2 * math.pi)

    def forward(self, time) -> torch.Tensor:
        # Apply sine and cosine functions for Fourier transformation
        time_proj = torch.matmul(time.unsqueeze(-1), self.B.T)
        time_emb = torch.cat([torch.sin(time_proj), torch.cos(time_proj)], dim=-1)

        # reshape to match the input shape
        time_emb = time_emb.view(time.shape[0], time.shape[1], -1)
        return time_emb

class PositionalEmbedding(nn.Module):
    """ Positional embedding, adopted from PyTorch tutorial """
    def __init__(self, emb_size: int, maxlen: int = 5000) -> None:
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

    def forward(self, seq_len) -> torch.Tensor:
        return self.pos_embedding[:, :seq_len]

class TransformerEncoder(nn.Module):
    """ Transformer encoder with Fourier positional encoding """
    def __init__(
        self,
        d_in: int,
        d_model: int,
        nhead: int = 1,
        dim_feedforward: int = 128,
        num_layers: int = 1,
        d_time: int = 1,
        emb_size: int = 16,
        emb_dropout: float = 0.1,
        emb_type: str = 'fourier',
        concat: bool = False
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.d_time = d_time
        self.emb_size = emb_size
        self.emb_dropout = emb_dropout
        self.emb_type = emb_type
        self.concat = concat
        self._setup_model()

    def _setup_model(self) -> None:
        # embedding layers
        if self.emb_type == 'legacy':
            self.embedding = nn.Identity()
            self.time_embedding = nn.Identity()
            self.dropout = nn.Dropout(0.0)
            self.mlp = nn.Linear(self.d_in + self.d_time, self.d_model)

        else:
            self.embedding = nn.Linear(self.d_in, self.emb_size)
            if self.emb_type == 'fourier':
                if self.emb_size % self.d_time != 0:
                    raise ValueError("Embedding size must be divisible by time dimension")
                emb_size = self.emb_size // self.d_time
                self.time_embedding = FourierTimeEmbedding(emb_size)
            elif self.emb_type == 'linear':
                self.time_embedding = nn.Linear(self.d_time, self.emb_size)
            else:
                raise ValueError("Invalid embedding type")
            self.dropout = nn.Dropout(self.emb_dropout)
            if self.concat:
                self.mlp = nn.Linear(self.emb_size * 2, self.d_model)
            else:
                self.mlp = nn.Linear(self.emb_size, self.d_model)

        # transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=self.nhead,
                dim_feedforward=self.dim_feedforward, batch_first=True),
            self.num_layers)

    def forward(
        self,
        src: torch.Tensor,
        src_t: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # embed the input and timestep
        if self.concat or self.emb_type == 'legacy':
            x = torch.cat([self.embedding(src), self.time_embedding(src_t)], dim=-1)
        else:
            x = self.embedding(src) + self.time_embedding(src_t)
        x = self.dropout(x)
        x = self.mlp(x)

        # pass through the transformer
        output = self.transformer_encoder(
            x, src_key_padding_mask=src_padding_mask)

        return output

class TransformerDecoder(nn.Module):
    """ Transformer encoder with positional encoding and context """
    def __init__(
        self,
        d_in: int,
        d_model: int,
        d_context: Optional[int] = None,
        nhead: int = 1,
        dim_feedforward: int = 128,
        num_layers: int = 1,
        emb_size: int = 16,
        emb_dropout: float = 0.1,
        max_len: int = 3,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_context = d_context
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.emb_size = emb_size
        self.emb_dropout = emb_dropout
        self.max_len = max_len
        self._setup_model()

    def _setup_model(self) -> None:
        # embedding layers
        self.embedding = nn.Linear(self.d_in, self.emb_size)
        self.postional_embedding = PositionalEmbedding(self.emb_size, self.max_len)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp = MLP(self.emb_size, [self.d_model, self.d_model * 4, self.d_model])

        # context layers
        if self.d_context is not None:
            self.context_mlp = MLP(
                self.d_context, [self.d_model, self.d_model * 4, self.d_model])
        else:
            self.context_mlp = None

        # transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.d_model, nhead=self.nhead,
                dim_feedforward=self.dim_feedforward, batch_first=True),
            self.num_layers)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        x = self.embedding(tgt) + self.postional_embedding(tgt.size(1))
        x = self.dropout(x)
        x = self.mlp(x)

        if self.context_mlp is not None:
            context = self.context_mlp(context)
            context = context.unsqueeze(1).expand(-1, x.size(1), -1)
            x = x + context
        tgt_mask = models_utils.get_casual_mask(x.size(1), device=x.device)

        # pass through the transformer_decoder
        output = self.transformer_decoder(
            x,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )

        return output
