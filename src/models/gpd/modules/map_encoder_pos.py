import torch
import torch.nn as nn

from src.models.gpd.layers.vq_vae_embedding import VQVAEEmbedding
from src.models.gpd.layers.common_layers import build_mlp


class PointsEncoder(nn.Module):
    """
        对polygon，先进行点级别编码，然后进行线级别编码
    """
    def __init__(self, dim=128, feat_channel=6):
        super().__init__()
        self.dim = dim
        self.first_mlp = nn.Sequential(
            nn.Linear(feat_channel, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
        )
        self.second_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, dim),
        )
        self.third_mlp = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),  # 取消BN in polygon level MLP，因为很可能只有一条路可见
            nn.ReLU(inplace=True),
            nn.Linear(256, dim),
        )

    def forward(self, polygon_feature):
        # polygon_feature [bs, M, 3, 20, 6]
        # mask [bs, M, 3, 20]
        
        # first embed，对最后一维进行一次MLP
        bs, M = polygon_feature.shape[:2]
        x_first_mlp = self.first_mlp(polygon_feature.flatten(0, 3))    # [bs, M, 3, 20, 256] 
        x_first_mlp = x_first_mlp.view(bs, M, 3, 20, -1)
        pooled_feature = x_first_mlp.max(dim=-2, keepdim=True)[0]  # [bs, M, 3, 1, 256]
        x_features = torch.cat( # [bs, M, 3, 20, 512]
            [x_first_mlp, pooled_feature.expand(-1, -1, -1, 20, -1)], dim=-1
        )

        # second embed, 再对x, y级别进行一次MLP
        x_second_mlp = self.second_mlp(x_features.flatten(0, 3))  # [bs, M, 3, 20, dim]
        x_second_mlp = x_second_mlp.view(bs, M, 3, 20, -1)
        res = x_second_mlp.max(dim=-2)[0]     # [bs, M, 3, dim]

        # third embed, 对一个polygon的左中右三条线进行一次MLP
        x_third_mlp = self.third_mlp(res.flatten(0, 2))     # [bs, M, 3, dim]
        x_third_mlp = x_third_mlp.view(bs, M, 3, -1)
        output = x_third_mlp.max(dim=-2)[0]  # [bs, M, dim]
        
        return output


class MapEncoderPos(nn.Module):
    def __init__(
        self,
        dim=128,
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

        self.polygon_encoder = PointsEncoder(dim)
        self.speed_limit_emb = nn.Sequential(
            nn.Linear(1, dim), nn.ReLU(), nn.Linear(dim, dim)
        )

        self.type_emb = nn.Embedding(3, dim)
        self.on_route_emb = nn.Embedding(2, dim)
        self.traffic_light_emb = nn.Embedding(4, dim)
        self.unknown_speed_emb = nn.Embedding(1, dim)
        
        # VQ-VAE embedding
        self.position_vq_vae_emb = VQVAEEmbedding(split_sizes=max_pos_xy, dividers=split_xy, embedding_dim=pos_xy_dim)
        self.heading_vq_vae_emb = VQVAEEmbedding(split_sizes=max_pos_heading, dividers=split_heading, embedding_dim=pos_heading_dim)
        self.min_pos_xy = min_pos_xy
        self.min_pos_heading = min_pos_heading

        pos_dim = pos_xy_dim * len(split_xy) * 2 + pos_heading_dim * len(split_heading)
        self.pos_emb = build_mlp(pos_dim, [dim] * 3, norm='ln')
        self.oob_emb = nn.Embedding(1, dim)     # 使不可见的agent可学习

    def forward(self, data) -> torch.Tensor:
        
        polygon_center = data["map"]["polygon_center"]
        polygon_type = data["map"]["polygon_type"].long()
        polygon_on_route = data["map"]["polygon_on_route"].long()
        polygon_tl_status = data["map"]["polygon_tl_status"].long()
        polygon_has_speed_limit = data["map"]["polygon_has_speed_limit"]
        polygon_speed_limit = data["map"]["polygon_speed_limit"]
        point_position = data["map"]["point_position"]
        point_vector = data["map"]["point_vector"]
        point_orientation = data["map"]["point_orientation"]
        valid_mask = data["map"]["valid_mask"].any(-1)

        polygon_feature = torch.cat(    # [bs, M, 3, 20, 6]
            [
                point_position - polygon_center[..., None, None, :2],
                point_vector,
                torch.stack(
                    [
                        point_orientation.cos(),
                        point_orientation.sin(),
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        
        # 对polygon级别进行一次MLP
        x_polygon = self.polygon_encoder(polygon_feature)   # [bs, M, dim]
        
        bs, M, _ = x_polygon.shape

        x_type = self.type_emb(polygon_type)
        x_on_route = self.on_route_emb(polygon_on_route)
        x_tl_status = self.traffic_light_emb(polygon_tl_status)
        x_speed_limit = torch.zeros(bs, M, self.dim, device=x_polygon.device)
        x_speed_limit[polygon_has_speed_limit] = self.speed_limit_emb(
            polygon_speed_limit[polygon_has_speed_limit].unsqueeze(-1)
        )
        x_speed_limit[~polygon_has_speed_limit] = self.unknown_speed_emb.weight

        x_polygon += x_type + x_on_route + x_tl_status + x_speed_limit

        # 编码polygon绝对位置
        position = self.position_vq_vae_emb(polygon_center[..., :2] - self.min_pos_xy).flatten(-2, -1)
        heading = self.heading_vq_vae_emb(polygon_center[..., 2] * 180 / torch.pi - self.min_pos_heading)
        
        # 组合feature
        pos = torch.cat([position, heading], dim=-1)
        
        # 映射pos到指定dim
        x_polygon += self.pos_emb(pos)       # [bs, M, dim]

        x_polygon[~valid_mask] = self.oob_emb.weight
        
        return x_polygon