from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from src.models.sledge.rvae_config import RVAEConfig

def lines_l1_loss(
    pred_states, pred_logits, gt_states, gt_mask, matching_indice, config: RVAEConfig = RVAEConfig()
) -> Dict[str, torch.Tensor]:
    """Inherited, see superclass."""

    # Arrange predictions and targets according to matching
    indices, permutation_indices = matching_indice, _get_src_permutation_idx(matching_indice)

    pred_states_idx = pred_states[permutation_indices]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_logits_idx = pred_logits[permutation_indices]
    gt_mask_idx = torch.cat([t[i] for t, (_, i) in zip(gt_mask, indices)], dim=0).float()

    # calculate CE and L1 Loss
    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1).mean(-1) * gt_mask_idx
    ce_weight, reconstruction_weight = config.line_ce_weight, config.line_reconstruction_weight

    ce_loss = F.binary_cross_entropy_with_logits(pred_logits_idx, gt_mask_idx, reduction="none")

    # 在时间维度，取平均
    l1_loss = l1_loss.view(-1, 80, 50).mean((0, 2))   # [bs, T, 50]
    ce_loss = ce_loss.view(-1, 80, 50).mean((0, 2))

    matched_map_feature = {     # 保留匹配好的值，后续使用
        "pred_lines": pred_states_idx,
        "pred_mask": pred_logits_idx > 0,
        "label_lines": gt_states_idx,
        "label_mask": gt_mask_idx
    }

    return reconstruction_weight * l1_loss, ce_weight * ce_loss, matched_map_feature

def _get_src_permutation_idx(indices) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper function for permutation of matched indices."""

    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])

    return batch_idx, src_idx
