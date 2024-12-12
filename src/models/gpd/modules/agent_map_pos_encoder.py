from typing import Dict, List
from torch import Tensor

import torch
import torch.nn as nn

from src.models.gpd.layers.vq_vae_embedding import VQVAEEmbedding
from src.models.gpd.layers.common_layers import build_mlp
from src.utils.simple_utils import normalize_angle

class AgentMapPosEncoder(nn.Module):
    def __init__(
        self,
        dim=256,
        pos_xy_dim=24,
        pos_heading_dim=6,
        split_xy=[1, 0.01],
        max_pos_xy=[600, 100],
        min_pos_xy=-300,
        split_heading=[20, 1],
        max_pos_heading=[20, 20],
        min_pos_heading=-180,
    ) -> None:
        super().__init__()
        
        self.dim = dim
        
        # VQ-VAE embedding
        self.position_vq_vae_emb = VQVAEEmbedding(split_sizes=max_pos_xy, dividers=split_xy, embedding_dim=pos_xy_dim)
        self.heading_vq_vae_emb = VQVAEEmbedding(split_sizes=max_pos_heading, dividers=split_heading, embedding_dim=pos_heading_dim)
        self.min_pos_xy = min_pos_xy
        self.min_pos_heading = min_pos_heading

        pos_dim = pos_xy_dim * len(split_xy) * 2 + pos_heading_dim * len(split_heading)
        self.pos_emb = build_mlp(pos_dim, [dim] * 3, norm='ln')
        self.oob_emb = nn.Embedding(1, dim)     # 使不可见的agent可学习
        
    def forward(
        self, 
        data: dict,
        window_T: int = 91
    ) -> Tensor:

        agent_pos = data["agent"]["position"][:, :, 1 : window_T].transpose(1, 2)
        agent_heading = data["agent"]["heading"][:, :, 1 : window_T].transpose(1, 2)
        agent_mask = data["agent"]["valid_mask"][:, :, 1 : window_T].transpose(1, 2)
        
        polygon_center = data["map"]["polygon_center"].unsqueeze(1).expand(-1, window_T - 1, -1, -1)
        polygon_mask = data["map"]["valid_mask"].any(-1).unsqueeze(1).expand(-1, window_T - 1, -1)
        
        # position = torch.cat([polygon_center[..., :2], agent_pos], dim=2)     # [bs, T, M + A, 2]
        # heading = torch.cat([polygon_center[..., 2], agent_heading], dim=2)   # [bs, T, M + A]
        # valid_mask = torch.cat([polygon_mask, agent_mask], dim=2)             # [bs, T, M + A]
        position = agent_pos
        heading = agent_heading
        valid_mask = agent_mask
        
        # 执行vq-vae
        position = self.position_vq_vae_emb(position - self.min_pos_xy).flatten(-2, -1)
        heading = self.heading_vq_vae_emb(heading * 180 / torch.pi - self.min_pos_heading)
        
        # 组合feature
        pos = torch.cat([position, heading], dim=-1)
        
        # 映射pos到指定dim
        pos_embed = self.pos_emb(pos)       # [bs, T, M+A, dim]

        pos_embed[~valid_mask] = self.oob_emb.weight

        return pos_embed