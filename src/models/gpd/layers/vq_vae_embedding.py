from typing import List

import torch
import torch.nn as nn

from src.models.gpd.layers.sinusoidal_position_encoding import SinusoidalPositionEncoding

class VQVAEEmbedding(nn.Module):
    def __init__(self, split_sizes: List, dividers: List, embedding_dim: int):
        super(VQVAEEmbedding, self).__init__()
        
        self.register_buffer('dividers', torch.tensor(dividers))
        self.embedding = nn.ModuleList(
            SinusoidalPositionEncoding(size, embedding_dim) for size in split_sizes
        )

    def forward(self, value):
        result = []
        for i, s in enumerate(self.dividers):
            value_idx = torch.div(value, s, rounding_mode='floor').long()
            result.append(self.embedding[i](value_idx))
            value -= value_idx * s
        
        return torch.cat(result, -1)