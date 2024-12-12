from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

class TransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 n_heads: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1) -> None:
        super().__init__()

        self.multihead_attn = torch.nn.MultiheadAttention(
            d_model,
            add_bias_kv=False,
            num_heads=n_heads,
            batch_first=True,
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, memory: Tensor, attn_mask: Optional[Tensor] = None, 
                key_padding_mask: Optional[Tensor] = None) -> Tensor:       
        x = self.norm1(x + self._mha_block(x, memory, attn_mask, key_padding_mask))
        x = self.norm2(x + self._ff_block(x))
        return x

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout1(x)
    
    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)