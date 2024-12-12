from typing import Dict
from torch import Tensor
import torch
from torch import nn

import torch.nn.functional as F

from src.bean.input_agent_bean import InputAgentBean
from src.bean.output_agent_bean import OutputAgentBean

class FocalLoss(nn.Module):
    """
    参考 https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, gamma=4.0, alpha=1, epsilon=1.e-9, device=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, device=device)
        else:
            self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
            alpha=[0.1, 0.2, 0.3, 0.15, 0.25]
        Returns:
            shape of [batch_size]
        """
        num_labels = input.size(-1)
        idx = target.view(-1, 1).long()
        one_hot_key = torch.zeros(idx.size(0), num_labels, dtype=torch.float32, device=idx.device)
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        one_hot_key[:, 0] = 0  # ignore 0 index.
        logits = torch.softmax(input, dim=-1)
        loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss = loss.sum(1)
        return loss.mean()

def reconstruction_agent_loss(pred: Tensor, label: Tensor) -> Dict[str, Tensor]:
    """
    Args:
        pred: [bs, T, num_frame_pred, A, _]
        label: [bs, T, num_frame_pred, A, _]
    """

    label_bean = InputAgentBean(label)
    pred_bean = OutputAgentBean(pred)
    label_mask = label_bean.valid_mask.bool()   # [bs, T, num_frame_pred, A]

    # 分别计算loss
    agent_position_loss = F.smooth_l1_loss(pred_bean.position[label_mask], label_bean.position[label_mask])
    agent_heading_loss = F.smooth_l1_loss(pred_bean.heading[label_mask], label_bean.heading[label_mask])
    agent_valid_mask_loss = F.binary_cross_entropy_with_logits(pred_bean.valid_mask, label_bean.valid_mask)    # 存在loss二分类

    return {
        "agent_position_loss": agent_position_loss * 5,     # 测试，调高loss的权重
        "agent_heading_loss": agent_heading_loss * 180 / 3.14 * 0.1,
        "agent_valid_mask_loss": agent_valid_mask_loss
    }

def reconstruction_map_loss(pred: Tensor, label: Tensor):
    return {
        # "map_latent_ce_loss": FocalLoss()(pred.flatten(0, 2), label.long().flatten(0, 1))
        "map_latent_ce_loss": F.cross_entropy(pred.permute(0, 3, 1, 2), label.long())
    }
