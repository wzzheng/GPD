import torch
import torch.nn as nn
import os

from src.models.gpd.layers.common_layers import build_mlp
from src.models.gpd.layers.sinusoidal_position_encoding import get_sinusoid_encoding_table

class MapLatentEncoder(nn.Module):
    def __init__(
        self,
        map_tokenizer_weight_path: str = '',
        dim=128,
    ) -> None:
        super().__init__()
        
        self.dim = dim

        self.agent_pos_emb = build_mlp(5, [self.dim] * 3, norm='ln')

        self.map_latent_adapter = build_mlp(self.dim, [self.dim * 4, self.dim], norm='ln')

        if os.path.exists(map_tokenizer_weight_path): 
            map_tokenizer_weight_path = torch.load(map_tokenizer_weight_path)
            self.register_buffer('map_tokenizer', map_tokenizer_weight_path)
        
        map_local_pos_emb_x = get_sinusoid_encoding_table(8, 128).unsqueeze(1).expand(-1, 8, -1)
        map_local_pos_emb_y = get_sinusoid_encoding_table(8, 128).unsqueeze(0).expand(8, -1, -1)
        map_local_pos_emb = torch.cat([map_local_pos_emb_x, map_local_pos_emb_y], dim=-1).reshape(-1, 256)
        self.register_buffer('map_local_pos_emb', map_local_pos_emb[None, None])
        self.map_pos_mlp = build_mlp(256, [128, 128], norm='ln')


    def forward(self, data) -> torch.Tensor:
        # 将indice转为具体值
        encoding_indices = data['map_latent']
        bs, T, M = encoding_indices.shape
        encoding_indices = encoding_indices.flatten().unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.dim, device=encoding_indices.device)
        encodings.scatter_(1, encoding_indices.to(torch.int64), 1)
        quantized = torch.matmul(encodings, self.map_tokenizer).view(bs, T, M, self.dim)

        # 添加local map pos embed
        map_pos_embed = self.map_pos_mlp(self.map_local_pos_emb)
        quantized += map_pos_embed

        # 添加全局agent pos embed
        ego_heading = data['agent']['heading'][:, 0].unsqueeze(-1)
        agent_pos = torch.cat(
            [
                data['agent']['position'][:, 0],
                ego_heading,
                torch.sin(ego_heading),
                torch.cos(ego_heading)
            ], dim=-1
        )
        agent_pos_embed = self.agent_pos_emb(agent_pos)
        map_feature = quantized + agent_pos_embed.unsqueeze(2)      # [bs, T, M, _]

        # 过一层MLP adapter
        map_feature = self.map_latent_adapter(map_feature)

        return map_feature