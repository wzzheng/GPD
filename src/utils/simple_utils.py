from typing import Any, Dict
import torch
import torch.nn.functional as F
import numpy as np
import copy

def mask_feature(feat, mask):
    # torch 版本
    while len(mask.shape) < len(feat.shape):
        mask = mask.unsqueeze(-1)
    return torch.where(mask.bool(), feat, torch.zeros_like(feat))


def generate_correct_agent_feature_indices(num_frame_pred: int, T: int, A: int, batch_size: int, dim: int = 10) -> torch.Tensor:
    """生成ego_loss，从target数据中提取未来帧时的T索引
    例子：num_frame_pred=3， T=5
    返回：[0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7]

    Args:
        num_frame_pred (int): 模型一次生成的未来帧的数量
        T (int): 模型总的输入时间
        A: agent数量
        dim: agent pred feature的dim: 10

    Returns:
        torch.Tensor: 生成的索引, [bs, indices, A, dim]
    """
    indices = []
    for t in range(1, T + 1):
        indices.append(torch.arange(t, t + num_frame_pred))
    re = torch.cat(indices)
    return re[None, :, None, None].expand(batch_size, -1, A, dim)

def normalize_angle(data):
    """
    将弧度制的角度张量归一化到 [-π, π] 之间
    :param data: 需要归一化的角度（弧度制）张量
    :return: 归一化后的角度张量（弧度制）
    """
    if isinstance(data, np.ndarray):
        return (data + np.pi) % (2 * np.pi) - np.pi
    elif isinstance(data, torch.Tensor):
        return (data + torch.pi) % (2 * torch.pi) - torch.pi
    else:
        raise TypeError("Input must be a numpy.ndarray or torch.Tensor")

def get_clones_module(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def data_premask(data: Dict[str, Any]) -> Dict[str, Any]:
    # 对数据进行一次mask，避免异常数据
    agent_mask = data["agent"]["valid_mask"]
    agent_level_mask = agent_mask.any(-1)
    for k in data['agent']:
        if k not in ['category', 'valid_mask']:     # category 需要level级别掩码
            data['agent'][k] = mask_feature(data['agent'][k], agent_mask)
        elif k == 'category':
            data['agent'][k] = mask_feature(data['agent']['category'], agent_level_mask)
        
    return data

def agent_padding(tensor, agent_num) -> Dict[str, Any]:
    A = tensor.shape[1]

    if A < agent_num:
        padding_size = agent_num - A

        if len(tensor.shape) == 3:
            return F.pad(tensor, (0, 0, 0, padding_size))
        else:
            return F.pad(tensor, (0, 0, 0, 0, 0, padding_size))
        
    return tensor