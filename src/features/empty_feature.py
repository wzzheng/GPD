from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType, to_tensor

@dataclass
class EmptyFeature(AbstractModelFeature):

    placeholder: FeatureDataType

    def to_device(self, device: torch.device) -> EmptyFeature:
        """Implemented. See interface."""
        return EmptyFeature(self.placeholder.to(device=device))

    def to_feature_tensor(self) -> EmptyFeature:
        """Inherited, see superclass."""
        return EmptyFeature(to_tensor(self.placeholder))

    @classmethod
    def deserialize(cls, data) -> EmptyFeature:
        """Implemented. See interface."""
        return EmptyFeature(data)

    def unpack(self) -> List[EmptyFeature]:
        """Implemented. See interface."""
        return [EmptyFeature(placeholder) for placeholder in zip(self.placeholder)]

    # def torch_to_numpy(self) -> EmptyFeature:
    #     """Helper method to convert feature from torch tensor to numpy array."""
    #     return EmptyFeature(self.placeholder.detach().cpu().numpy())

    # def squeeze(self) -> EmptyFeature:
    #     """Helper method to apply .squeeze() on features."""
    #     return EmptyFeature(self.squeeze(0))
