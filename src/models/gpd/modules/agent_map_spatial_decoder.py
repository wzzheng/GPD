import torch
from torch import Tensor
import torch.nn as nn
from src.utils.simple_utils import get_clones_module
from src.models.gpd.layers.transformer_decoder_layer import TransformerDecoderLayer

class AgentMapSpatialDecoder(nn.Module):
    def __init__(
        self, 
        dim=256,
        max_entity_num: int = 300,
        num_layers: int = 6,
        num_heads: int = 8,
    ) -> None:
        super().__init__()

        self.agent_map_spatial_attention_layers = get_clones_module(TransformerDecoderLayer(dim, num_heads), num_layers)
        self.spatial_attn_pos_emb = nn.Embedding(max_entity_num, dim)

    def forward(self, x: Tensor, agent_mask: Tensor, polygon_mask: Tensor) -> Tensor:
        
        agent_key_padding = agent_mask[:, 1:] & agent_mask[:, :-1]  # [bs, T, A]
        polygon_key_padding = polygon_mask.any(-1).unsqueeze(1).expand(-1, agent_key_padding.shape[1], -1)
        # key_padding_mask = ~torch.cat([polygon_key_padding, agent_key_padding], dim=-1)  # [bs, T, M+A]
        key_padding_mask = ~agent_key_padding  # [bs, T, M+A]
        
        x = x.flatten(0, 1)     # [bs*T, M+A, _]
        _, N, _ = x.shape
        x = x + self.spatial_attn_pos_emb.weight[None, :N]
        
        for layer in self.agent_map_spatial_attention_layers:
            x = layer(
                x, x,
                key_padding_mask=key_padding_mask.flatten(0, 1)
            )
        
        return x