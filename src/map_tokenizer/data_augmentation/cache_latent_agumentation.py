import logging
from typing import List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
import torch
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import (
    AbstractAugmentor,
)
from nuplan.planning.training.data_augmentation.data_augmentation_util import (
    ParameterToScale,
    ScalingDirection,
    UniformNoise,
)
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.common.actor_state.state_representation import StateSE2

from sledge.autoencoder.preprocessing.features.nuplan_feature import NuplanFeature
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVectorRaw, SledgeVectorElement, SledgeVector
from sledge.autoencoder.preprocessing.features.sledge_raster_feature import SledgeRaster, SledgeRasterIndex
from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig
from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import (
    sledge_raw_feature_processing,
)

logger = logging.getLogger(__name__)


class CacheLatentAgumenter(AbstractAugmentor):

    def __init__(
        self,
        config: RVAEConfig
    ) -> None:
        self._config = config

    def augment(
        self,
        features: FeaturesType,
        targets: TargetsType = None,
        scenario: Optional[AbstractScenario] = None,
    ) -> Tuple[FeaturesType, TargetsType]:
        # rand_idx = np.random.randint(20, 121)   # TODO 测一下是不是每次都不一样
        # 坐标转化
        line_states, line_masks = self._convert_line_to_relative(features)

        del features, targets

        # 绘制到画布，取得GT，逻辑过于复杂，反正cache latent就一次，不优化了
        # line_res = []
        # mask_res = []
        raster_res = []

        _ = np.full((1, 2, 2), -10)    # 避免修改大量代码，注入一个占位tensor
        for state, mask in zip(line_states, line_masks):
            sledge_vector_raw = SledgeVectorRaw(
                SledgeVectorElement(state, mask),
                SledgeVectorElement(_, _[:, 0]),
                SledgeVectorElement(_, _[:, 0]),
                SledgeVectorElement(_, _[:, 0]),
                SledgeVectorElement(_, _[:, 0]),
                SledgeVectorElement(_, _[:, 0]),
                SledgeVectorElement(_, _[:, 0]),
            )

            __, frame_raster = sledge_raw_feature_processing(sledge_vector_raw, self._config)
            
            raster_res.append(frame_raster.data)
            # line_res.append(frame_vector.lines.states)
            # mask_res.append(frame_vector.lines.mask)

        features = {}
        targets = {}
        
        # 我们cache latent时，不需要vector
        # targets["sledge_vector"] = SledgeVector(
        #     SledgeVectorElement(
        #         np.concatenate(line_res, axis=0),
        #         np.concatenate(mask_res, axis=0)
        #     ),
        #     SledgeVectorElement(_, _[:, 0, 0]),
        #     SledgeVectorElement(_, _[:, 0, 0]),
        #     SledgeVectorElement(_, _[:, 0, 0]),
        #     SledgeVectorElement(_, _[:, 0, 0]),
        #     SledgeVectorElement(_, _[:, 0, 0]),
        #     SledgeVectorElement(_, _[:, 0, 0]),
        # )
        features["sledge_raster"] = SledgeRaster(
            np.concatenate(raster_res, axis=-1)  # [256, 256, 12 * T]
        )

        return features, targets

    def _convert_line_to_relative(self, features):

        center_xy = features['map_cache'].data['agent']['position'][0]    # [T, 2]
        center_angle = features['map_cache'].data['agent']['heading'][0]  # [T]
        
        T = center_angle.shape[0]

        rotate_mat = self._rotate_matrix_from_pose(center_angle)    # [T, 2, 2]

        line_states = features['map_cache'].data['map']['lines']['states']
        line_mask = features['map_cache'].data['map']['lines']['mask']
        
        line_states = line_states[None, ...].repeat(T, axis=0)
        line_mask = line_mask[None, ...].repeat(T, axis=0)

        # 转化到当前帧ego坐标系
        line_states[..., :2] = np.matmul(   # [T, M, P, _]
            line_states[..., :2] - center_xy[:, None, None, :], rotate_mat[:, None, :, :]
        )
        line_states[..., 2] -= center_angle[:, None, None]

        return line_states, line_mask

    @staticmethod
    def _rotate_matrix_from_pose(heading):
        """
        Construct rotation matrices from heading angles.
        
        :param heading: [T] containing heading angles.
        :return: [T, 2, 2] representing rotation matrices.
        """
        cos_vals = np.cos(heading)
        sin_vals = np.sin(heading)
        
        # 构造旋转矩阵
        rotation_matrices = np.zeros((*heading.shape, 2, 2))  # [..., 2, 2]
        
        rotation_matrices[..., 0, 0] = cos_vals
        rotation_matrices[..., 0, 1] = -sin_vals
        rotation_matrices[..., 1, 0] = sin_vals
        rotation_matrices[..., 1, 1] = cos_vals
        
        return rotation_matrices

















    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return []

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return []

    @property
    def augmentation_probability(self) -> ParameterToScale:
        """Inherited, see superclass."""
        raise NotImplementedError
    
    @property
    def get_schedulable_attributes(self) -> List[ParameterToScale]:
        """Inherited, see superclass."""
        raise NotImplementedError