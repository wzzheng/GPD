from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
)
from torch.nn.utils.rnn import pad_sequence

@dataclass
class MapLatentFeature(AbstractModelFeature):
    data: Dict[str, Any]

    @classmethod
    def collate(cls, feature_list: List[MapLatentFeature]) -> MapLatentFeature:
        batch_data = {}
        batch_data['encoding_indice'] = torch.stack(
            [f.data['encoding_indice'] for f in feature_list],
            dim=0
        )
            
        return MapLatentFeature(data=batch_data)

    def to_feature_tensor(self) -> MapLatentFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = torch.from_numpy(v)
        return MapLatentFeature(data=new_data)

    def to_numpy(self) -> MapLatentFeature:
        raise NotImplementedError

    def serialize(self) -> Dict[str, Any]:
        return self.data
    
    def to_device(self, device: torch.device) -> MapLatentFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = v.to(device)
        return MapLatentFeature(data=new_data)

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> MapLatentFeature:
        data['encoding_indice'] = data['encoding_indice'].reshape(101, 64)
        return MapLatentFeature(data=data)

    def unpack(self) -> List[AbstractModelFeature]:
        raise NotImplementedError