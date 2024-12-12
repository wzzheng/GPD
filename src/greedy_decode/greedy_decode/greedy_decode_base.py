from typing import Dict, Any, Callable

import torch
from src.utils.simple_utils import normalize_angle
class GreedyDecodeBase:
    """
        对于不同的模拟方法，需要不同的decode，
        Base版本是:
            input   20 map agent ego
            output  80 map agent ego
    """
    def __init__(self, init_T=21, pred_T=80):
        self.init_T = init_T
        self.pred_T = pred_T

    def __call__(
            self,
            features: Dict[str, Any], 
            forward_fn: Callable
        ) -> Any:
        # 创建用于存储预测结果的变量
        predicted_data = self._create_save_data(features, self.init_T, self.pred_T)
        
        end_T = self.init_T + self.pred_T
        # 进行推断
        for t in range(self.init_T, end_T):
            output = forward_fn(self._get_feature_slice(predicted_data, 0, t))
            self._save_predicted_data(predicted_data, output, t)
            
        predict_output = self._get_feature_slice(predicted_data, self.init_T, end_T)

        return predict_output

    @staticmethod
    def _get_feature_slice(feature, t1, t2):
        return {
            'agent': {
                'position': feature['agent']['position'][:, :, t1:t2],
                'heading': feature['agent']['heading'][:, :, t1:t2],
                'valid_mask': feature['agent']['valid_mask'][:, :, t1:t2],
            },
            'map_latent': feature['map_latent'][:, t1:t2],
        }

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
            "agent": {
                'position': torch.cat([data['agent']['position'][:, :, :init_T], torch.zeros(bs, A, pred_T, 2, device=device)], 2),
                'heading': torch.cat([data['agent']['heading'][:, :, :init_T], torch.zeros(bs, A, pred_T, device=device)], 2),
                'valid_mask': torch.cat([data['agent']['valid_mask'][:, :, :init_T], torch.zeros(bs, A, pred_T, device=device, dtype=torch.bool)], 2)
            },
            'map_latent': torch.cat([data['map_latent'][:, :init_T], torch.zeros(bs, pred_T, M, device=device)], 1)
        }
        
        return predicted_data