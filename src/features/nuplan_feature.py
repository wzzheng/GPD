from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
)
from torch.nn.utils.rnn import pad_sequence

from src.utils.conversion import to_device, to_numpy, to_tensor
from src.utils.simple_utils import normalize_angle

@dataclass
class NuplanFeature(AbstractModelFeature):
    data: Dict[str, Any]

    @classmethod
    def collate(cls, feature_list: List[NuplanFeature]) -> NuplanFeature:
        batch_data = {}

        batch_data["agent"] = {
            k: pad_sequence(
                [f.data["agent"][k] for f in feature_list], batch_first=True
            )
            for k in feature_list[0].data["agent"].keys()
        }
            
        return NuplanFeature(data=batch_data)

    def to_feature_tensor(self) -> NuplanFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_tensor(v)
        return NuplanFeature(data=new_data)

    def to_numpy(self) -> NuplanFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_numpy(v)
        return NuplanFeature(data=new_data)

    def to_device(self, device: torch.device) -> NuplanFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_device(v, device)
        return NuplanFeature(data=new_data)

    def serialize(self) -> Dict[str, Any]:
        return self.data

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> NuplanFeature:
        data['agent']['heading'] = normalize_angle(data['agent']['heading'])
        return NuplanFeature(data=data)

    def unpack(self) -> List[AbstractModelFeature]:
        raise NotImplementedError

    def is_valid(self) -> bool:
        return self.data["agent"]['heading'].shape[0] > 0

    @classmethod
    def normalize(
        self, data, first_time=False, radius=None, hist_steps=21,
        cur_step=0     # transform the coordinate system to be centered around the ego at a specific step
    ) -> NuplanFeature:
        center_xy = data['agent']['position'][0, cur_step].copy()
        center_angle = data['agent']['heading'][0, cur_step].copy()
        
        rotate_mat = np.array(
            [
                [np.cos(center_angle), -np.sin(center_angle)],
                [np.sin(center_angle), np.cos(center_angle)],
            ],
            dtype=np.float64,
        )

        data["agent"]["position"] = np.matmul( # [A, T, 2]，[bs, A, T, 2], [bs, 2]
            data["agent"]["position"] - center_xy, rotate_mat
        )
        # data["agent"]["velocity"] = np.matmul(data["agent"]["velocity"], rotate_mat)
        data["agent"]["heading"] -= center_angle

        return NuplanFeature(data=data)