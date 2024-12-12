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

from sledge.autoencoder.preprocessing.features.nuplan_feature import NuplanFeature
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVectorRaw, SledgeVectorElement


logger = logging.getLogger(__name__)


class NuplanToSledgeAgumenter(AbstractAugmentor):


    def augment(
        self,
        features: FeaturesType,
        targets: TargetsType = None,
        scenario: Optional[AbstractScenario] = None,
    ) -> Tuple[FeaturesType, TargetsType]:
        
        # 将数据组装为：SledgeVectorRaw，后续处理
        _ = np.full((1, 2, 2), -10)    # 避免修改大量代码，注入一个占位tensor
        
        # 将数据转化到指定帧
        rand_idx = np.random.randint(0, 101)

        center_xy = features['map_cache'].data['ego']['position'][rand_idx]
        center_angle = features['map_cache'].data['ego']['heading'][rand_idx]
        
        rotate_mat = np.array(
            [
                [np.cos(center_angle), -np.sin(center_angle)],
                [np.sin(center_angle), np.cos(center_angle)],
            ],
            dtype=np.float64,
        )

        line_states = features['map_cache'].data['map']['lines']['states']
        line_mask = features['map_cache'].data['map']['lines']['mask']
        
        # 转化到当前帧ego坐标系
        line_states[..., :2] = np.matmul(
            line_states[..., :2] - center_xy, rotate_mat
        )
        line_states[..., 2] -= center_angle


        data = {}
        data["sledge_raw"] = SledgeVectorRaw(
            SledgeVectorElement(line_states, line_mask),
            SledgeVectorElement(_, _[:, 0]),
            SledgeVectorElement(_, _[:, 0]),
            SledgeVectorElement(_, _[:, 0]),
            SledgeVectorElement(_, _[:, 0]),
            SledgeVectorElement(_, _[:, 0]),
            SledgeVectorElement(_, _[:, 0]),
        )
        
        del features, targets

        return data, dict()




















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