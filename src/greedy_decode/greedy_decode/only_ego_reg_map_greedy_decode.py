from typing import Dict, Any, Callable

import torch
from src.utils.simple_utils import normalize_angle
from .greedy_decode_base import GreedyDecodeBase

class OnlyEgoRegMapGreedyDecode(GreedyDecodeBase):
    """
            input   20 map ego, 80 ego
            output  80 map
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
        feature['map_latent'][:, curr_idx] = torch.argmax(output['map_latent'][:, -1], dim=-1)
        
    @staticmethod
    def _create_save_data(
            data: Dict[str, Any], 
            init_T: int = 21,   # init_T
            pred_T: int = 80,   # pred_T
        ) -> Dict[str, Any]:
        bs, A, T, _ = data['agent']['position'].shape
        M = data['map_latent'].shape[2]
        device = data['agent']['position'].device
        
        # 创建用于存储预测结果的变量，将ego赋值为GT，将agent全部置0
        predicted_data = { 
            "agent": {
                'position': torch.cat([data['agent']['position'][:, 0:1], torch.zeros(bs, A-1, T, 2, device=device)], 1),
                'heading': torch.cat([data['agent']['heading'][:, 0:1], torch.zeros(bs, A-1, T, device=device)], 1),
                'valid_mask': torch.cat([data['agent']['valid_mask'][:, 0:1], torch.zeros(bs, A-1, T, device=device, dtype=torch.bool)], 1)
            },
            'map_latent': torch.cat([data['map_latent'][:, :init_T], torch.zeros(bs, pred_T, M, device=device)], 1)
        }

        return predicted_data