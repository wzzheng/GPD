from typing import Dict
from torch import Tensor
from src.bean.input_agent_bean import InputAgentBean
from src.bean.output_agent_bean import OutputAgentBean

import numpy as np

def metric_acc(pred, label, is_binary=False):
    def softmax(x, axis=-1):
        # 防止溢出，通过减去x的最大值来进行数值稳定性处理
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    if is_binary: # 二分类
        # 将其转化为0，1
        pred_sigmoid = pred > 0.5
        pred_flat = pred_sigmoid.flatten()
    else:   # 多分类
        # 对预测结果进行softmax
        pred_softmax = softmax(pred, axis=-1)

        # 将预测结果和目标结果展平为一维数组
        pred_flat = pred_softmax.argmax(axis=-1).flatten()
    label_flat = label.flatten()

    # 计算准确的个数
    return np.sum(pred_flat == label_flat)

def compute_agent_metrics(pred: Tensor, label: Tensor, metric: Dict[str, float], prefix: str):
    """
    统计pred 预测的10帧的 第1帧怎么样，前5帧平均怎么样，前10帧平均怎么样
       pred: [bs, T, num_frame_pred, _], ego / agent pred feature
       label: [bs, T, num_frame_pred, _], ego / agent label feature  
    """
    label_bean = InputAgentBean(label)
    label_mask = label_bean.valid_mask.cpu().numpy()   # [bs, T, num_frame_pred, A]
    label_bean = InputAgentBean(label.cpu().numpy())
    pred_bean = OutputAgentBean(pred.detach().cpu().numpy())

    position_x_abs = np.abs(label_bean.get_position_x() - pred_bean.get_position_x()) * label_mask
    position_y_abs = np.abs(label_bean.get_position_y() - pred_bean.get_position_y()) * label_mask
    heading_abs = np.abs(label_bean.heading - pred_bean.heading) * label_mask
    
    # metric[prefix + "_trajectory_x_80"] += np.sum(position_x_abs[:, :, :80])
    # metric[prefix + "_trajectory_x_10"] += np.sum(position_x_abs[:, :, :10])
    metric[prefix + "_trajectory_x_1"] += np.sum(position_x_abs)
    
    
    # metric[prefix + "_trajectory_y_80"] += np.sum(position_y_abs[:, :, :80])
    # metric[prefix + "_trajectory_y_10"] += np.sum(position_y_abs[:, :, :10])
    metric[prefix + "_trajectory_y_1"] += np.sum(position_y_abs)
    
    # metric[prefix + "_heading_80"] += np.sum(heading_abs[:, :, :80]) * 180 / 3.14
    # metric[prefix + "_heading_10"] += np.sum(heading_abs[:, :, :10]) * 180 / 3.14
    metric[prefix + "_heading_1"] += np.sum(heading_abs) * 180 / 3.14

    # metric[prefix + "_existence_80"] += metric_acc(pred_bean.valid_mask[:, :, :80], label_bean.valid_mask[:, :, :80], is_binary=True)
    # metric[prefix + "_existence_10"] += metric_acc(pred_bean.valid_mask[:, :, :10], label_bean.valid_mask[:, :, :10], is_binary=True)
    metric[prefix + "_existence_1"] += metric_acc(pred_bean.valid_mask, label_bean.valid_mask, is_binary=True)