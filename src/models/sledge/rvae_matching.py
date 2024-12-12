import torch
from scipy.optimize import linear_sum_assignment
from src.models.sledge.rvae_config import RVAEConfig
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType


@torch.no_grad()
def compute_line_matching(pred_states, pred_logits, gt_states, gt_mask, config: RVAEConfig = RVAEConfig()) -> TargetsType:
    """Inherited from superclass."""

    ce_cost = _get_ce_cost(gt_mask, pred_logits)

    l1_cost = _get_line_l1_cost(gt_states, pred_states, gt_mask)
    ce_weight, reconstruction_weight = config.line_ce_weight, config.line_reconstruction_weight

    cost = ce_weight * ce_cost + reconstruction_weight * l1_cost
    cost = cost.cpu()  # NOTE: This unfortunately is the runtime bottleneck

    indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
    matching = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    return matching


@torch.no_grad()
def _get_ce_cost(gt_mask: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Calculated cross-entropy matching cost based on numerically stable PyTorch version, see:
    https://github.com/pytorch/pytorch/blob/c64e006fc399d528bb812ae589789d0365f3daf4/aten/src/ATen/native/Loss.cpp#L214
    :param gt_mask: ground-truth binary existence labels, shape: (batch, num_gt)
    :param pred_logits: predicted (normalized) logits of existence, shape: (batch, num_pred)
    :return: cross-entropy cost tensor of shape (batch, num_pred, num_gt)
    """

    gt_mask_expanded = gt_mask[:, :, None].detach().float()  # (b, ng, 1)
    pred_logits_expanded = pred_logits[:, None, :].detach()  # (b, 1, np)

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val))
    ce_cost = (1 - gt_mask_expanded) * pred_logits_expanded + helper_term  # (b, ng, np)
    ce_cost = ce_cost.permute(0, 2, 1)  # (b, np, ng)

    return ce_cost


@torch.no_grad()
def _get_line_l1_cost(gt_states: torch.Tensor, pred_states: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """
    Calculates the L1 matching cost for line state tensors.
    :param gt_states: ground-truth line tensor, shape: (batch, num_gt, state_size)
    :param pred_states: predicted line tensor, shape: (batch, num_pred, state_size)
    :param gt_mask: ground-truth binary existence labels for masking, shape: (batch, num_gt)
    :return: L1 cost tensor of shape (batch, num_pred, num_gt)
    """

    gt_states_expanded = gt_states[:, :, None].detach()  # (b, ng, 1, *s)
    pred_states_expanded = pred_states[:, None].detach()  # (b, 1, np, *s)
    l1_cost = gt_mask[..., None] * (gt_states_expanded - pred_states_expanded).abs().sum(dim=-1).mean(dim=-1)
    l1_cost = l1_cost.permute(0, 2, 1)  # (b, np, ng)

    return l1_cost