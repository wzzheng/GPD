from typing import Dict, List
from torch import Tensor

import torch
import torch.nn as nn

from src.models.gpd.layers.transformer_decoder_layer import TransformerDecoderLayer
from src.models.gpd.layers.vq_vae_embedding import VQVAEEmbedding
from src.models.gpd.layers.common_layers import build_mlp
from src.utils.simple_utils import get_clones_module, normalize_angle

class AgentTemporalEncoder(nn.Module):
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
        num_layers: int = 6,
        num_heads: int = 8,
        max_window_T: int = 101,    # 最大的T的长度，用于构建掩码
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
        
        self.temporal_attention_layers = get_clones_module(TransformerDecoderLayer(dim, num_heads), num_layers)
        self.register_buffer("agent_temporal_attn_mask", torch.triu(torch.ones(max_window_T, max_window_T), diagonal=1).bool()) # [T, T]
        # self.type_emb = nn.Embedding(4, dim)
        self.oob_emb = nn.Embedding(1, dim)     # 使不可见的agent可学习
        self.temporal_attn_pos_embed = nn.Embedding(max_window_T, dim)
        
    def forward(
        self, 
        data: Dict[str, Dict[str, Tensor]], 
        window_T: int = 91
    ) -> Tensor:

        position = data["agent"]["position"][:, :, :window_T]
        heading = data["agent"]["heading"][:, :, :window_T]
        # velocity = data["agent"]["velocity"][:, :, :total_T]
        # shape = data["agent"]["shape"][:, :, :total_T]
        # category = data["agent"]["category"].long()
        valid_mask = data["agent"]["valid_mask"][:, :, :window_T]
        
        valid_mask_vec = valid_mask[..., 1:] & valid_mask[..., :-1]
        heading_vec = normalize_angle(self.to_vector(heading, valid_mask_vec))   # 放到这里顺便mask，避免异常值影响 vq-vae
        position_vec = self.to_vector(position, valid_mask_vec)
        # velocity_vec = self.to_vector(velocity, valid_mask_vec)
        
        # 执行vq-vae
        position_vec = self.position_vq_vae_emb(position_vec - self.min_pos_xy).flatten(-2, -1)
        heading_vec = self.heading_vq_vae_emb(heading_vec * 180 / torch.pi - self.min_pos_heading)
        
        # 组合feature
        agent_feature = torch.cat([position_vec, heading_vec], dim=-1)
        
        # 映射agent feature到指定dim
        agent_feature = self.agent_emb(agent_feature)   # [bs, A, T, dim]

        # TODO: 后续可以加入shape, category等属性的embedding
        
        bs, A, T, _ = agent_feature.shape
        
        # 执行temporal attn
        # 注意：
        # 1. 我们对不可见的agent，我们使之可学习
        # 2. 我们在一个batch内padding，但多个batch间的agent和polygon是变长的

        agent_feature[~valid_mask_vec] = self.oob_emb.weight
        
        agent_feature = agent_feature.flatten(0, 1)
        
        temporal_attn_pos_embed = self.temporal_attn_pos_embed.weight[:T].unsqueeze(0)
        agent_feature += temporal_attn_pos_embed
        
        for layer in self.temporal_attention_layers:
            agent_feature = layer(
                agent_feature, agent_feature,
                attn_mask=self.agent_temporal_attn_mask[:T, :T]
            )

        # TODO 后续可以尝试
        # if not self.use_ego_history:
        
        # x_type = self.type_emb(category)

        # return x_agent + x_type
        return agent_feature.view(bs, A, T, -1)  # [bs, A, T, dim]

    @staticmethod
    def to_vector(feat, vec_mask):

        while len(vec_mask.shape) < len(feat.shape):
            vec_mask = vec_mask.unsqueeze(-1)

        return torch.where(
            vec_mask,
            feat[:, :, 1:, ...] - feat[:, :, :-1, ...],
            torch.zeros_like(feat[:, :, 1:, ...]),
        )