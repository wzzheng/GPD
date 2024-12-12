from typing import Dict, Any, Callable

import torch
from src.utils.simple_utils import normalize_angle
from .greedy_decode_base import GreedyDecodeBase

class RegAgentEgoGreedyDecode(GreedyDecodeBase):
    """
        input   20 map agent ego, 80 map 
        output  80 ego agent
    """
    def __init__(self, init_T=21, pred_T=80):
        self.init_T = init_T
        self.pred_T = pred_T

    @staticmethod
    def _save_predicted_data(feature, output, curr_idx):
        """
        feature['agent']['position']: [bs, A, T, 2]
        output['agent']['position']: [bs, T, A, num_pred, 2]
        curr_idx 当前需要保存的帧
        直接保存最后一帧就可以
        """
        feature['agent']['position'][:, :, curr_idx] = output['agent']['position'][:, -1]    # [bs, A, dim]
        feature['agent']['heading'][:, :, curr_idx] = normalize_angle(output['agent']['heading'][:, -1, :, 0])
        feature['agent']['valid_mask'][:, :, curr_idx] = torch.gt(output['agent']['valid_mask'][:, -1, :, 0], 0)
        
    @staticmethod
    def _create_save_data(
            data: Dict[str, Any], 
            init_T: int = 21,   # init_T
            pred_T: int = 80,   # pred_T
        ) -> Dict[str, Any]:
        bs, A = data['agent']['position'].shape[:2]
        device = data['agent']['position'].device
        
        # 创建用于存储预测结果的变量
        predicted_data = { 
            "agent": {
                'position': torch.cat([data['agent']['position'][:, :, :init_T], torch.zeros(bs, A, pred_T, 2, device=device)], 2),
                'heading': torch.cat([data['agent']['heading'][:, :, :init_T], torch.zeros(bs, A, pred_T, device=device)], 2),
                'valid_mask': torch.cat([data['agent']['valid_mask'][:, :, :init_T], torch.zeros(bs, A, pred_T, device=device, dtype=torch.bool)], 2)
            },
            'map_latent': data['map_latent']
        }

        return predicted_data