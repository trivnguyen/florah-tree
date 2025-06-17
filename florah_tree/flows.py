from typing import Optional

import torch
import torch.nn as nn
import zuko

class NPE(nn.Module):
    def __init__(
        self,
        input_size: int,
        context_size: int,
        hidden_sizes: int,
        num_transforms: int,
        context_embedding_sizes: Optional[int] = None,
        dropout: int = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.context_size = context_size
        self.hidden_sizes = hidden_sizes
        self.context_embedding_sizes = context_embedding_sizes
        self.num_transforms = num_transforms
        self.dropout = nn.Dropout(dropout)

        if context_embedding_sizes is None:
            self.lin_proj_layers = nn.Identity()
            self.flow = zuko.flows.NSF(
                input_size, context_size, transforms=num_transforms,
                hidden_features=hidden_sizes, randperm=True
            )
        else:
            self.lin_proj_layers = nn.ModuleList()
            for i in range(len(context_embedding_sizes)):
                in_proj_dim = context_size if i == 0 else context_embedding_sizes[i - 1]
                out_proj_dim = context_embedding_sizes[i]
                self.lin_proj_layers.append(nn.Linear(in_proj_dim, out_proj_dim))
                self.lin_proj_layers.append(nn.ReLU())
                self.lin_proj_layers.append(nn.BatchNorm1d(out_proj_dim))
                self.lin_proj_layers.append(nn.Dropout(dropout))
            self.lin_proj_layers = nn.Sequential(*self.lin_proj_layers)
            self.flow = zuko.flows.NSF(
                input_size, context_embedding_sizes[-1], transforms=num_transforms,
                hidden_features=hidden_sizes, randperm=True
            )

    def forward(
        self,
        context: torch.Tensor
    ) -> torch.Tensor:
        embed_context = self.lin_proj_layers(context)
        return embed_context

    def log_prob(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if context is not None:
            context = self.forward(context)
        return self.flow(context).log_prob(x)

    def sample(
        self,
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        if context is not None:
            context = self.forward(context)
        return self.flow(context).sample(**kwargs)