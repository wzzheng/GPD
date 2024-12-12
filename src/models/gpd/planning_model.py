import torch
import torch.nn as nn
from typing import Any, Dict
from torch import Tensor

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from src.feature_builders.empty_target_builder import EmptyTargetBuilder
from src.feature_builders.nuplan_feature_builder import NuplanFeatureBuilder
from src.feature_builders.map_latent_builder import MapLatentBuilder
from src.feature_builders.map_info_builder import MapInfoBuilder

from .layers.common_layers import build_mlp
from .modules.agent_encoder_pos import AgentEncoderPos
from .modules.map_latent_encoder import MapLatentEncoder
from .modules.autoregressive_decoder import AutoregressiveDecoder
from .modules.entity_decoder import EntityDecoder

# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)

class PlanningModel(TorchModuleWrapper):
    def __init__(
        self,
        map_tokenizer_weight_path: str = '',
        dim: int = 128,
        feature_builder_agent: NuplanFeatureBuilder = NuplanFeatureBuilder(),
        feature_builder_map_latent: MapLatentBuilder = None,
        feature_builder_map_info: MapInfoBuilder = MapInfoBuilder(),
        attn_num_layers: int = 6,
        attn_num_heads: int = 8
    ) -> None:
        feature_builders = [feature_builder_agent, feature_builder_map_info]
        # if feature_builder_map_info is not None:
        #     feature_builders.append(feature_builder_map_info)
        if feature_builder_map_latent is not None:
            feature_builders.append(feature_builder_map_latent)

        super().__init__(
            feature_builders=feature_builders,
            target_builders=[EmptyTargetBuilder()],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.dim = dim
        
        self.agent_encoder = AgentEncoderPos(
            dim=dim, pos_xy_dim=48, pos_heading_dim=32, split_xy=[1, 0.01], split_heading=[20, 1],
            max_pos_xy=[600, 100], max_pos_heading=[20, 20], min_pos_xy=-300, min_pos_heading=-180,
            )

        self.map_encoder = MapLatentEncoder(dim=dim, map_tokenizer_weight_path=map_tokenizer_weight_path)

        self.autoregressive_decoder = AutoregressiveDecoder(dim, num_layers=attn_num_layers, num_heads=attn_num_heads,
                                                            num_agent=33, num_polygon=64, max_window_T=101)

        self.entity_decoder = EntityDecoder(d_model=dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):     # TODO 更新用到的层
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.MultiheadAttention):
            # Xavier 初始化 for in_proj_weight and out_proj.weight
            torch.nn.init.xavier_uniform_(m.in_proj_weight)
            torch.nn.init.xavier_uniform_(m.out_proj.weight)
            # 将 bias 初始化为 0
            if m.in_proj_bias is not None:
                nn.init.constant_(m.in_proj_bias, 0)
            if m.out_proj.bias is not None:
                nn.init.constant_(m.out_proj.bias, 0)

    def forward(
            self, 
            data: Dict[str, Dict[str, Tensor]]
        ):

        x_agent = self.agent_encoder(data).transpose(1, 2)  # [bs, T, A, dim]
        x_map = self.map_encoder(data)      # [bs, T, M, dim]

        M = x_map.shape[2]

        x = torch.cat([x_map, x_agent], dim=2)

        x = self.autoregressive_decoder(x)
        
        x_map = x[:, :, :M]
        x_agent = x[:, :, M:]
        output = self.entity_decoder(x_agent, x_map)
        
        return output