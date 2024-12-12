from typing import Dict, Any, Callable

import torch
from src.utils.simple_utils import normalize_angle
from .greedy_decode_base import GreedyDecodeBase
class RegMapGreedyDecode(GreedyDecodeBase):
    """
            input   20 map agent ego, 80 agent ego
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
        bs, A = data['agent']['position'].shape[:2]
        M = data['map_latent'].shape[2]
        device = data['agent']['position'].device
        
        # 创建用于存储预测结果的变量
        predicted_data = { 
            "agent": data['agent'],     # agent用真值
            'map_latent': torch.cat([data['map_latent'][:, :init_T], torch.zeros(bs, pred_T, M, device=device)], 1)
        }
        
        return predicted_data