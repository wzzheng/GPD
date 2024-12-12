import torch
from torch import Tensor
import torch.nn as nn
from einops import rearrange
from src.utils.simple_utils import get_clones_module
from src.models.gpd.layers.transformer_decoder_layer import TransformerDecoderLayer

class AutoregressiveDecoder(nn.Module):
    def __init__(
        self, 
        dim=256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_agent: int = 33,
        num_polygon: int = 64,
        max_window_T: int = 101,
    ) -> None:
        super().__init__()

        self.spatial_pos_emb = nn.Embedding(num_polygon + num_agent, dim)
        self.temporal_pos_emb = nn.Embedding(max_window_T, dim)

        num_entity = num_agent + num_polygon

        self.num_polygon = num_polygon

        self.decoder_layers = get_clones_module(TransformerDecoderLayer(dim, num_heads), num_layers)

        attn_shape = [max_window_T * num_entity, max_window_T * num_entity]
        attn_mask = torch.triu(torch.ones(attn_shape), diagonal=1)

        for t in range(max_window_T):
            attn_mask[t * num_entity : (t + 1) * num_entity, t * num_entity : (t + 1) * num_entity] = 0
            # # 设置map不能查看agent
            # for t2 in range(max_window_T):
            #     attn_mask[
            #         t * num_entity : t * num_entity + num_polygon, 
            #         t2 * num_entity + num_polygon + 1 : (t2 + 1) * num_entity   # 让每一个map都看不到过去所有时刻的agent
            #     ] = 1
        self.register_buffer("agent_map_attn_mask", attn_mask.bool())     # [M0+T*A, M0+T*A]

    def forward(self, x: Tensor) -> Tensor:
        
        B, T, E, _ = x.shape

        # temporal & spatial pos embedding
        x = rearrange(x, 'b t e c -> (b t) e c')
        x = x + self.spatial_pos_emb.weight.unsqueeze(0)
        x = rearrange(x, '(b t) e c -> (b e) t c', b=B, t=T)
        x = x + self.temporal_pos_emb.weight[None, :T]
        x = rearrange(x, '(b e) t c -> b (t e) c', b=B, e=E)

        # agent map attn mask
        agent_map_attn_mask = self.agent_map_attn_mask[:T*E, :T*E]
        
        for layer in self.decoder_layers:   
            x = layer(x, x, attn_mask=agent_map_attn_mask)          # [bs, T * (M + A), dim]

        x = rearrange(x, 'b (t e) c -> b t e c', t=T, e=E)
        
        return x