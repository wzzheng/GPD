from typing import Dict, List
from torch import Tensor

import torch
import torch.nn as nn

from src.models.gpd.layers.transformer_decoder_layer import TransformerDecoderLayer
from src.models.gpd.layers.vq_vae_embedding import VQVAEEmbedding
from src.models.gpd.layers.common_layers import build_mlp
from src.utils.simple_utils import get_clones_module, normalize_angle

class AgentEncoderPos(nn.Module):
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
        self.agent_emb = build_mlp(pos_dim, [dim] * 3, norm='ln')
        
        # self.type_emb = nn.Embedding(4, dim)
        self.oob_emb = nn.Embedding(1, dim)     # 使不可见的agent可学习
        
    def forward(
        self, 
        data: Dict[str, Dict[str, Tensor]]
    ) -> Tensor:

        position = data["agent"]["position"]
        heading = data["agent"]["heading"]
        # velocity = data["agent"]["velocity"][:, :, :total_T]
        # shape = data["agent"]["shape"][:, :, :total_T]
        # category = data["agent"]["category"].long()
        valid_mask = data["agent"]["valid_mask"]
        
        # 执行vq-vae
        position_vec = self.position_vq_vae_emb(position - self.min_pos_xy).flatten(-2, -1)
        heading_vec = self.heading_vq_vae_emb(heading * 180 / torch.pi - self.min_pos_heading)
        
        # 组合feature
        agent_feature = torch.cat([position_vec, heading_vec], dim=-1)
        
        # 映射agent feature到指定dim
        agent_feature = self.agent_emb(agent_feature)   # [bs, A, T, dim]

        # TODO: 后续可以加入shape, category等属性的embedding
        
        agent_feature[~valid_mask] = self.oob_emb.weight

        return agent_feature                            # [bs, A, T, dim]