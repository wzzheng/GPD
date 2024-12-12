import numpy as np
import torch

def compute_map_f1_metric(matched_map_feature, metrics, distance_threshold=1.5):
    """
    Compute F1 metric using distance threshold with matrix operations.
    
    :param pred_points: Predicted points, shape [bs, T, 50, 20, 2]
    :param gt_points: Ground truth points, shape [bs, T, 50, 20, 2]
    :param pred_mask: Predicted mask, shape [bs, T, 50]
    :param gt_mask: Ground truth mask, shape [bs, T, 50]
    :param distance_threshold: Distance threshold for matching points, default is 1.5m

    return: true_positives_mask, 后续compute_lateral_l2使用
    """
    # 预测这个地方有点 && 这个地方真值有点

    pred_points = matched_map_feature['pred_lines']
    pred_mask = matched_map_feature['pred_mask']
    gt_points = matched_map_feature['label_lines']
    gt_mask = matched_map_feature['label_mask']

    mask = pred_mask * gt_mask

    # 计算每个点之间的距离
    distances = torch.norm(pred_points - gt_points, dim=-1)

    # 这里，我们仅仅认定mask=1的位置的点，是有效的
    distances_masked = distances * mask.unsqueeze(-1)
    
    # true_positives: 这个地方预测有点，并且真实也有点，并且点合法
    true_positives_mask = (distances_masked <= distance_threshold) & (distances_masked > 0)
    true_positives = torch.sum(true_positives_mask.view(-1, 80, 50, 20), dim=(0, 2, 3))  # [80]

    # false_positives：这个地方预测有点，但真实没点
    # 理解：预测有点的集合里就两种情况，真实有点，以及真实没点，所以减去真实有点的，就是真实没点的
    # mask × 20是因为mask是线级别，我们统计的是点级别
    false_positives = torch.sum(pred_mask.view(-1, 80, 50), dim=(0, 2)) * 20 - true_positives
    
    # false_negatives：这个地方预测没点，但真实有点
    # 理解：真实有点的集合里就两种情况，预测有点，预测没点，所以减去预测有点的，就是预测没点的
    false_negatives = torch.sum(gt_mask.view(-1, 80, 50), dim=(0, 2)) * 20  - true_positives
    
    metrics['map_F1_TP'] += true_positives.cpu().numpy()
    metrics['map_F1_FP'] += false_positives.cpu().numpy()
    metrics['map_F1_FN'] += false_negatives.cpu().numpy()

    return true_positives_mask


def compute_lateral_l2(pred_points, gt_points, true_positives_mask, metrics):
    """
    Compute Lateral L2 metric using point-to-line distance formula (PyTorch version).
    
    :param pred_points: Predicted points, shape [bs, T, 50, 20, 2] (torch.Tensor)
    :param gt_points: Ground truth points, shape [bs, T, 50, 20, 2] (torch.Tensor)
    :param true_positives_mask: True positives mask, shape [bs, T, 50, 20] (torch.Tensor)
    :return: Lateral L2 metric
    """
    # 利用点到直线距离公式计算
    # Step 1: Calculate A, B, C for each segment between consecutive gt_points
    x1, y1 = gt_points[..., :-1, 0], gt_points[..., :-1, 1]  # First point in each segment
    x2, y2 = gt_points[..., 1:, 0], gt_points[..., 1:, 1]    # Second point in each segment
    
    # Compute A, B, C for the line passing through (x1, y1) and (x2, y2)
    A = y2 - y1
    B = -(x2 - x1)
    C = -(A * x1 + B * y1)
    
    # Repeat the last line parameters to match gt_points dimensions (for last point in each lane)
    A = torch.cat([A, A[..., -1:]], dim=-1)
    B = torch.cat([B, B[..., -1:]], dim=-1)
    C = torch.cat([C, C[..., -1:]], dim=-1)
    
    # Step 2: Normalize the line coefficients to avoid large values
    normalization_factor = torch.sqrt(A**2 + B**2) + 1e-6  # Avoid division by zero
    
    # Step 3: Calculate distance from each pred_point to its corresponding gt line
    # Formula: |Ax + By + C| / sqrt(A^2 + B^2)
    lateral_distances = torch.abs(A * pred_points[..., 0] + B * pred_points[..., 1] + C) / normalization_factor
    masked_lateral_distances = lateral_distances * true_positives_mask  # Mask out non-true positive points
    
    # Step 4: Sum lateral distances and count true positives
    metrics['map_lateral_l2_distance'] += torch.sum(masked_lateral_distances.reshape(-1, 80, 50, 20), dim=(0, 2, 3)).cpu().numpy()



def compute_chamfer(pred_points, gt_points, gt_mask, metrics, counts):
    """
    Compute Chamfer distance (squared distance) between predicted and ground truth points.
    
    :param pred_points: Predicted points, shape [bs*T, 50, 20, 2] (torch.Tensor)
    :param gt_points: Ground truth points, shape [bs*T, 50, 20, 2] (torch.Tensor)
    :param gt_mask: Ground truth mask, shape [bs*T, 50] (torch.Tensor)
    :return: Chamfer distance (torch.Tensor)
    """
    # Step 1: Apply gt_mask to filter out non-visible points in both sets
    pred_points_visible = pred_points * gt_mask[..., None, None]  # Shape: [bs*T, 50, 20, 2]
    gt_points_visible = gt_points * gt_mask[..., None, None]      # Shape: [bs*T, 50, 20, 2]
    
    # Step 2: Expand dimensions to compute pairwise squared distances between pred and gt points
    pred_expanded = pred_points_visible.unsqueeze(-3)  # Shape: [bs*T, 50, 1, 20, 2]
    gt_expanded = gt_points_visible.unsqueeze(-4)      # Shape: [bs*T, 1, 50, 20, 2]
    
    # Step 3: Compute squared L2 distance between each point in pred and gt sets
    squared_distances = torch.sum((pred_expanded - gt_expanded) ** 2, dim=-1)  # Shape: [bs*T, 50, 50, 20]
    
    # Step 4: For each point in pred, find the minimum squared distance to gt points (pred->gt)
    pred_to_gt_squared_distances = torch.min(squared_distances, dim=-2)[0]  # Shape: [bs*T, 50, 20]
    
    # Step 5: For each point in gt, find the minimum squared distance to pred points (gt->pred)
    gt_to_pred_squared_distances = torch.min(squared_distances, dim=-3)[0]  # Shape: [bs*T, 50, 20]
    
    # Step 6: Combine all distances and take the average over valid points
    total_pred_to_gt_squared_distance = torch.sum((pred_to_gt_squared_distances * gt_mask[..., None]).reshape(-1, 80, 50, 20), dim=(0, 2, 3))
    total_gt_to_pred_squared_distance = torch.sum((gt_to_pred_squared_distances * gt_mask[..., None]).reshape(-1, 80, 50, 20), dim=(0, 2, 3))
    
    metrics['map_chamfer_distance'] += (total_pred_to_gt_squared_distance + total_gt_to_pred_squared_distance).cpu().numpy()
    counts['map_chamfer_distance_gt_mask_count'] += torch.sum(gt_mask.reshape(-1, 80, 50), dim=(0, 2)).cpu().numpy() * 20 * 2     # 每条线20个点，pred_to_gt和gt_to_pred两种情况
