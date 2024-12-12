from typing import Dict, Any, Callable

import torch

from src.bean.input_agent_bean import InputAgentBean
from src.bean.output_agent_bean import OutputAgentBean
from src.utils.simple_utils import normalize_angle

def greedy_decode(
        features: Dict[str, Any], 
        forward_fn: Callable,
        init_window_T: int = 21,
        pred_window_T: int = 80,
        next_autoregressive_input_frame: int = 1
    ) -> Any:
    # next_autoregressive_input_frame: 模型拿num_pred_frame中的几帧，作为下一次自回归的依据
    
    # 创建用于存储预测结果的变量
    predicted_data = _create_save_data(features, init_window_T, pred_window_T)
    
    # 进行推断
    for t in range(0, pred_window_T, next_autoregressive_input_frame):
        start_t = t + init_window_T
        if start_t > 101:
            # 我们模型的max_windowT等于101
            output = forward_fn(_get_feature_slice(predicted_data, start_t - 101, start_t))
        else:
            output = forward_fn(_get_feature_slice(predicted_data, 0, start_t))
        # 存储预测结果
        _save_predicted_data(predicted_data, output, start_t, start_t + next_autoregressive_input_frame)
        
    predict_output = _get_feature_slice(predicted_data, init_window_T, init_window_T + pred_window_T)

    return predict_output

def _get_feature_slice(feature, t1, t2):
    return {
        'agent': {
            'position': feature['agent']['position'][:, :, t1:t2],
            'heading': feature['agent']['heading'][:, :, t1:t2],
            'valid_mask': feature['agent']['valid_mask'][:, :, t1:t2],
        },
        'map_latent': feature['map_latent'][:, t1:t2],
    }

def _save_predicted_data(feature, output, t1, t2):
    """
    feature['agent']['position']: [bs, A, T, 2]
    output['agent']['position']: [bs, T, A, num_pred, 2]
    t1, t2: 需要保存的Output值的时间索引
    """
    # 安全措施，避免数组溢出
    T = feature['agent']['position'].shape[2]
    t2 = t2 if t2 <= T else T
    assert t2 > t1

    past_t1 = t1 - 1
    past_t2 = t2 - 1
    feature['agent']['position'][:, :, t1] = output['agent']['position'][:, past_t1]    # [bs, A, dim]
    feature['agent']['heading'][:, :, t1] = normalize_angle(output['agent']['heading'][:, past_t1, :, 0])
    feature['agent']['valid_mask'][:, :, t1] = torch.gt(output['agent']['valid_mask'][:, past_t1, :, 0], 0)
    feature['map_latent'][:, t1] = torch.argmax(output['map_latent'][:, past_t1], dim=-1)
    # feature['agent']['position'][:, :, t1: t2] = output['agent']['position'][:, past_t1: past_t2]    # [bs, A, dim]
    # feature['agent']['heading'][:, :, t1: t2] = normalize_angle(output['agent']['heading'][:, past_t1: past_t2])
    # feature['agent']['valid_mask'][:, :, t1: t2] = torch.gt(output['agent']['valid_mask'][:, past_t1: past_t2], 0.5)
    # feature['map_latent'][:, t1: t2] = torch.argmax(output['map_latent'][:, past_t1: past_t2], dim=-1)
    
def _create_save_data(
        data: Dict[str, Any], 
        init_T: int = 21,   # init_window_T
        pred_T: int = 80,   # pred_window_T
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