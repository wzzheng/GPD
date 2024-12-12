from typing import Dict

import torch
import torch.nn as nn
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType

from sledge.autoencoder.modeling.objectives.abstract_custom_objective import AbstractCustomObjective
from sledge.autoencoder.preprocessing.features.sledge_raster_feature import SledgeRasterIndex

class RVAERasterRestructionObjective(AbstractCustomObjective):
    """重建画布的loss"""
    def __init__(self, scenario_type_loss_weighting: Dict[str, float]) -> None:
        super().__init__()
        self._scenario_type_loss_weighting = scenario_type_loss_weighting
        self.map_info_indices = [
            SledgeRasterIndex._LINE_X
        ]
        self.ones_tensor = torch.ones(1, 1, 256, 256, dtype=torch.float32)
        self.ce_loss_weight = torch.tensor([2])

    def compute(
        self, predictions: FeaturesType, gt: FeaturesType, targets: TargetsType, matchings: TargetsType, scenarios: ScenarioListType
    ) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        # 处理gt
        label = gt["sledge_raster"].data[:, self.map_info_indices]
        label = torch.where(label > 0, self.ones_tensor.to(device=label.device), label)
        label = label.squeeze(1)    # [bs, 256, 256]
        
        pred = predictions['sledge_raster'].squeeze(1)  # [bs, 256, 256]

        return {
            "raster_Dice": self.dice_loss(label, pred),
            "raster_CE": self.weighted_ce_loss(label, pred, class_weights=self.ce_loss_weight)
        }

    @staticmethod
    def dice_loss(gt, pred, epsilon=1e-6):
        """
        计算 Dice Loss

        参数:
        - pred: [bs, H, W]，预测概率 (经过 sigmoid 或 softmax), 二分类
        - gt: [bs, H, W]，真实标签 (类别索引)

        返回:
        - loss: Dice 损失值
        """
        pred = torch.sigmoid(pred).view(-1)
        gt = gt.view(-1)
        
        intersection = (pred * gt).sum()
        union = (pred + gt).sum()
        dice = (2 * intersection + epsilon) / (union + epsilon)
        return 1 - dice
    
    @staticmethod
    def weighted_ce_loss(gt, pred, class_weights=None):
        """
        计算带权重的交叉熵损失 (Weighted Cross Entropy Loss)
        
        参数:
        - pred: [bs, num_classes, 256, 256]，预测 logits，未经过 softmax 的概率值
        - gt: [bs, 256, 256]，ground truth 分割标签 (值为 0, 1, 2, 3 表示 4 个类别)
        - class_weights: 每个类别的权重 (长度为 num_classes 的列表或张量)
        
        返回:
        - loss: 计算得到的加权交叉熵损失值
        """

        if class_weights is not None:
            class_weights = class_weights.to(pred.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        
        return criterion(pred, gt)
