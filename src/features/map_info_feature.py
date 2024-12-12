from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
)
from src.bean.sledge_vector_feature import SledgeVectorRaw, SledgeVectorElement, SledgeVector
from src.utils.sledge_utils.utils import process_lines_to_vector, process_lines_to_raster

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

def _convert_line_to_relative(features):

    center_xy = features['agent']['position'][0]    # [T, 2]
    center_angle = features['agent']['heading'][0]  # [T]
    
    T = center_angle.shape[0]

    rotate_mat = _rotate_matrix_from_pose(center_angle)    # [T, 2, 2]

    line_states = features['map']['lines']['states']
    line_mask = features['map']['lines']['mask']
    
    line_states = line_states[None, ...].repeat(T, axis=0)
    line_mask = line_mask[None, ...].repeat(T, axis=0)

    # 转化到当前帧ego坐标系
    line_states[..., :2] = np.matmul(   # [T, M, P, _]
        line_states[..., :2] - center_xy[:, None, None, :], rotate_mat[:, None, :, :]
    )
    line_states[..., 2] -= center_angle[:, None, None]

    return line_states, line_mask


@dataclass
class MapInfoFeature(AbstractModelFeature):
    data: Dict[str, Any]

    @classmethod
    def collate(cls, feature_list: List[MapInfoFeature]) -> MapInfoFeature:
        batch_data = {}
        batch_data['lines'] = torch.stack(
            [f.data['lines'] for f in feature_list],
            dim=0
        )
        batch_data['masks'] = torch.stack(
            [f.data['masks'] for f in feature_list],
            dim=0
        )
        return MapInfoFeature(data=batch_data)

    def to_feature_tensor(self) -> MapInfoFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = torch.from_numpy(v)
        return MapInfoFeature(data=new_data)

    def to_numpy(self) -> MapInfoFeature:
        raise NotImplementedError

    def serialize(self) -> Dict[str, Any]:
        return self.data
    
    def to_device(self, device: torch.device) -> MapInfoFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = v.to(device)
        return MapInfoFeature(data=new_data)

    @classmethod
    def deserialize(cls, data: Dict[str, Any], return_vector=True, retrun_raster=False) -> MapInfoFeature:
        # 坐标转化
        line_states, line_masks = _convert_line_to_relative(data)

        if return_vector:
            # 取得画布上的vector，测试自回归时用于获取GT
            line_res = []
            mask_res = []

            for state, mask in zip(line_states, line_masks):

                states, mask = process_lines_to_vector(SledgeVectorElement(state, mask))

                line_res.append(states)
                mask_res.append(mask)

            features = {
                'lines': np.stack(line_res, axis=0),
                'masks': np.stack(mask_res, axis=0)
            }

            return MapInfoFeature(data=features)
        
        if retrun_raster:
            # 取得画布上的raster，simulation时用于获取map latent
            raster_res = []

            for state, mask in zip(line_states, line_masks):    # 遍历T维度

                raster = process_lines_to_raster(SledgeVectorElement(state, mask))

                raster_res.append(raster)

            features = {
                'rasters': np.stack(raster_res, axis=0)
            }
            return MapInfoFeature(data=features)
        
        raise NotImplementedError

    def unpack(self) -> List[AbstractModelFeature]:
        raise NotImplementedError