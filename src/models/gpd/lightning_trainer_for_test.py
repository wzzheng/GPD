import logging
import os
from typing import Dict, Tuple, Union
import copy
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from src.utils.simple_utils import mask_feature, agent_padding
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from src.bean.input_agent_bean import InputAgentBean
from src.models.sledge.rvae_model import RVAEModel
from src.models.sledge.rvae_config import RVAEConfig
from src.models.sledge.rvae_matching import compute_line_matching
from src.models.sledge.rvae_objective import lines_l1_loss
from src.utils.sledge_utils.visualization.sledge_visualization_utils import simple_visualize_vector_in_sledge
from src.utils.visualization_utils import visualize_scene_in_navism
from src.metrics.agent_metrics import average_displacement_error, final_displacement_error, check_collisions, collision_by_radius_check, final_displacement_error_every_epoch
from src.metrics.map_metrics import compute_map_f1_metric, compute_lateral_l2, compute_chamfer
from src.utils.simple_utils import normalize_angle
from src.greedy_decode.data_preprocess.greedy_data_processor_base import GreedyDataProcessorBase
from src.greedy_decode.greedy_decode.greedy_decode_base import GreedyDecodeBase
logger = logging.getLogger(__name__)


class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        greedy_data_processor: GreedyDataProcessorBase,
        greedy_decode: GreedyDecodeBase,
        max_agent_num: int = 33,
        input_window_T: int = 21, 
        output_window_T: int = 80,
        is_visual: bool = False,
        need_map_restruction_metric = True,
        need_agent_restruction_metric = True,
        need_ego_restruction_metric = True,
    ) -> None:
        super().__init__()

        self.model = model
        self.greedy_data_processor = greedy_data_processor
        self.greedy_decode = greedy_decode
        self.max_agent_num = max_agent_num
        self.input_window_T = input_window_T
        self.output_window_T = output_window_T
        self.need_map_restruction_metric = need_map_restruction_metric
        self.need_agent_restruction_metric = need_agent_restruction_metric
        self.need_ego_restruction_metric = need_ego_restruction_metric

        # sledge
        sledge_config = RVAEConfig()
        sledge_decoder = RVAEModel(sledge_config)
        self.sledge_decoder = self.load_sledge_model(sledge_decoder, sledge_config)
        self.strict_loading = False     
        self.is_visual = is_visual


    def test_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        data_train, data_label = self.greedy_data_processor(
            batch[0], 
            input_window_T=self.input_window_T,
            output_window_T=self.output_window_T
        )

        agent_label = self._preprocess_agent(data_label['agent'])

        if self.is_visual:
            visualize_scene_in_navism(
                agent_label,
                data_label['map_vector'], 
                metas=batch[2],
                output_window_T=self.output_window_T
            )
    # def test_step(
    #     self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    # ) -> torch.Tensor:
    #     data_train, data_label = self.greedy_data_processor(
    #         batch[0], 
    #         input_window_T=self.input_window_T,
    #         output_window_T=self.output_window_T
    #     )

    #     data_pred = self.greedy_decode(data_train, self.model)

    #     # 将latent decode为vector
    #     map_pred_latent = data_pred['map_latent'].reshape(-1, 64)
    #     map_vector_pred = self.sledge_decoder(map_pred_latent)

    #     agent_label = self._preprocess_agent(data_label['agent'])
    #     agent_pred = self._preprocess_agent(data_pred['agent'])
    #     self.calculate_metric(
    #         agent_pred=agent_pred, 
    #         agent_label=agent_label, 
    #         map_latent_pred=data_pred['map_latent'], 
    #         map_latent_label=data_label['map_latent'], 
    #         map_vector_pred=map_vector_pred, 
    #         map_vector_label=data_label['map_vector']
    #     )

    #     if self.is_visual:
    #         visualize_scene_in_navism(
    #             agent_pred, 
    #             agent_label, 
    #             map_vector_pred, 
    #             data_label['map_vector'], 
    #             metas=batch[2],
    #             output_window_T=self.output_window_T
    #         )
    #         # simple_visualize_vector_in_sledge(map_pred_line, map_pred_line_mask, map_label_line, map_label_line_mask, metas=batch[2])


    @staticmethod
    def _preprocess_agent(data: Dict[str, Tensor]) -> Tensor:
        position = data["position"]
        heading = data["heading"][..., None]
        valid_mask = data["valid_mask"][..., None]
        
        agent_feature = torch.cat(
            [
                position,
                heading,
                valid_mask,
            ],
            -1,
        )

        return agent_feature    # [bs, A, T, _]

    
    @staticmethod
    def load_sledge_model(sledge_decoder, config: RVAEConfig):

        ckpt = torch.load(config.model_decode_ckpt_path)
        adapte_dict = {}

        for k, v in ckpt.items():
            if k == 'model._vector_quantization._embedding.weight':
                adapte_dict['_embedding.weight'] = v
            else:
                adapte_dict[k.replace('model.', '')] = v

        sledge_decoder.load_state_dict(adapte_dict)
        return sledge_decoder

    """
    ade, fde, cr, 以及map的几个metric
    """
    def calculate_metric(self, agent_pred, agent_label, map_latent_pred, map_latent_label, map_vector_pred, map_vector_label):
        self._calculate_agent_restruction_metric(agent_pred, agent_label)
        if self.need_map_restruction_metric:
            self._calculate_map_restruction_metric(map_latent_pred, map_latent_label, map_vector_pred, map_vector_label)

    def _calculate_agent_restruction_metric(self, agent_pred, agent_label):
        bs, A, T, agent_dim = agent_pred.shape

        self.test_count_collection["ego_full_count"] += bs
        self.test_count_collection["agent_full_count"] += bs * (A - 1)

        agent_pred = agent_pred.cpu().numpy()
        agent_label = agent_label.cpu().numpy()

        # 计算collision
        self._calculate_agent_collision(agent_pred[:, :, :, :3], self.test_metrics_collection, agent_label[..., -1])

        if self.need_ego_restruction_metric:
            absolute_ego_feature = InputAgentBean(agent_label[:, 0])  # [bs, T, _]
            absolute_ego_feature_pred = InputAgentBean(agent_pred[:, 0])  # [bs, T, _]
            self._calculate_agent_test_metric(absolute_ego_feature, absolute_ego_feature_pred, "ego_", self.test_metrics_collection, self.test_count_collection)

        if self.need_agent_restruction_metric:
            absolute_agent_synthesize_feature = InputAgentBean(     # 这里需要将agent的shape变成[bs*A, T, dim]的形式，输入函数统一计算
                agent_label[:, 1:].reshape(-1, T, agent_dim)
            )  # [bs*(A-1), T, _]
            absolute_agent_synthesize_feature_pred = InputAgentBean(
                agent_pred[:, 1:].reshape(-1, T, agent_dim)
            )  # [bs*(A-1), T, _]
            self._calculate_agent_test_metric(absolute_agent_synthesize_feature, absolute_agent_synthesize_feature_pred, "agent_", self.test_metrics_collection, self.test_count_collection)

    def _calculate_map_restruction_metric(self, map_latent_pred, map_latent_label, map_vector_pred, map_vector_label):
        bs, T, M = map_vector_label['lines'].shape[:3]
        self.test_count_collection["map_latent_tokens_count"] += bs * M
        # 计算map latent acc
        self.test_metrics_collection['map_latent_acc'] += (map_latent_pred == map_latent_label).sum(axis=(0, 2)).cpu().numpy()

        # 计算map vector metric
        map_pred_line = map_vector_pred.lines.states
        map_pred_line_mask = map_vector_pred.lines.mask

        map_label_line = map_vector_label['lines'].flatten(0, 1)
        map_label_line_mask = map_vector_label['mask'].flatten(0, 1)
        
        line_indices = compute_line_matching(map_pred_line, map_pred_line_mask, map_label_line, map_label_line_mask)

        l1_loss, ce_loss, matched_map_feature = lines_l1_loss(map_pred_line, map_pred_line_mask, map_label_line, map_label_line_mask, line_indices)

        self.test_metrics_collection['map_line_l1_loss'] += l1_loss.cpu().numpy()
        self.test_metrics_collection['map_line_ce_loss'] += ce_loss.cpu().numpy()

        # 计算F1
        true_positives_mask = compute_map_f1_metric(matched_map_feature, self.test_metrics_collection)

        # 计算lateral l2
        compute_lateral_l2(
            pred_points=matched_map_feature['pred_lines'],
            gt_points=matched_map_feature['label_lines'],
            true_positives_mask=true_positives_mask,
            metrics=self.test_metrics_collection
        )

        # 计算chamfer
        compute_chamfer(
            pred_points=matched_map_feature['pred_lines'].reshape(-1, 50, 20, 2),
            gt_points=matched_map_feature['label_lines'].reshape(-1, 50, 20, 2),
            gt_mask=matched_map_feature['label_mask'].reshape(-1, 50),
            metrics=self.test_metrics_collection,
            counts=self.test_count_collection
        )

    def on_test_start(self):
        def _get_agent_metric(prefix):
            return {    # 统计每一个时刻的误差变化
                prefix + "FDE": np.zeros(self.output_window_T), 
                prefix + "ADE": np.zeros(self.output_window_T), 
                prefix + "collision_sum": np.zeros(self.output_window_T),
                prefix + "nuplan_collision_sum": np.zeros(self.output_window_T),
                
                # 之前的指标，用于和之前的信息比较
                prefix + "trajectory_x": np.zeros(self.output_window_T),
                prefix + "trajectory_y": np.zeros(self.output_window_T),
                prefix + "heading": np.zeros(self.output_window_T),
                prefix + "heading_correct": np.zeros(self.output_window_T),
                prefix + "existence": np.zeros(self.output_window_T)
            }
        self.test_metrics_collection = {
            **_get_agent_metric('ego_'),
            **_get_agent_metric('agent_'),

            "map_latent_acc": np.zeros(self.output_window_T),
            "map_line_l1_loss": np.zeros(self.output_window_T),
            "map_line_ce_loss": np.zeros(self.output_window_T),

            # map单帧的F1 score
            "map_F1_TP": np.zeros(self.output_window_T),
            "map_F1_FP": np.zeros(self.output_window_T),
            "map_F1_FN": np.zeros(self.output_window_T),

            # map单帧的lateral_l2
            "map_lateral_l2_distance": np.zeros(self.output_window_T),

            # map单帧的chamfer
            "map_chamfer_distance": np.zeros(self.output_window_T)
        }

        self.test_count_collection = {    # 记录有多少例，最后用于求平均指标
            "agent_full_count": 0,
            "ego_full_count": 0,

            "agent_exist_count": 0,
            "ego_exist_count": 0,

            "map_latent_tokens_count": 0,

            "map_chamfer_distance_gt_mask_count": 0
        }

    def on_test_epoch_end(self, *args, **kwargs):
        if self.need_ego_restruction_metric:
            self._calculate_agent_metric_end('ego_', self.test_metrics_collection, self.test_count_collection)
        if self.need_agent_restruction_metric:
            self._calculate_agent_metric_end('agent_', self.test_metrics_collection, self.test_count_collection)

        if self.need_map_restruction_metric:
            self.test_metrics_collection['map_latent_acc'] /= self.test_count_collection["map_latent_tokens_count"]
            self.test_metrics_collection['map_line_l1_loss'] /= self.test_count_collection["ego_full_count"]
            self.test_metrics_collection['map_line_ce_loss'] /= self.test_count_collection["ego_full_count"]

            self._calculate_map_F1_score_end(self.test_metrics_collection)

            # 这里的count就是sum(true_positives_mask)
            self.test_metrics_collection['map_lateral_l2_distance'] /= self.test_metrics_collection['map_F1_TP'] if self.test_metrics_collection['map_F1_TP'].all() > 0 else np.zeros(self.output_window_T)

            self.test_metrics_collection['map_chamfer_distance'] /= self.test_count_collection['map_chamfer_distance_gt_mask_count']
            
        # 保存为csv文件
        metrics_data = {}
        single_metric_names = [
            # 'ego_FDE', 'agent_FDE', 'ego_ADE', 'agent_ADE', 'ego_collision_sum', 'agent_collision_sum', 
            # "map_f1_score", "map_lateral_l2_distance", "map_chamfer_distance"
        ]
        not_in_csv_metric_names = single_metric_names + [
            'map_F1_TP', 'map_F1_FP', 'map_F1_FN'
        ]

        # 计算80时刻总的平均值和最大值
        for key in self.test_metrics_collection.keys():
            if key not in not_in_csv_metric_names:
                metrics_data[key] = np.append(
                    [
                        self.test_metrics_collection[key].max(), 
                        self.test_metrics_collection[key].mean()
                    ],
                    self.test_metrics_collection[key]
                )

        columns = ["max", "average"] + [str(i) for i in range(self.output_window_T)]
        df = pd.DataFrame(metrics_data, index=columns).T
        df = df.round(3)    # 保留三位小数
        df.to_csv("test_metrics_results.csv", index=True)

        # 单独保存其他值
        with open('single_metrics.txt', 'w') as f:
            for k in single_metric_names:
                if self.test_metrics_collection.get(k) is not None:
                    f.write(f'{k}: {self.test_metrics_collection[k]}\n')

    @staticmethod
    def _calculate_agent_test_metric(target: InputAgentBean, predict: InputAgentBean, prefix: str, metric: dict, count: dict) -> None:
        label_mask = target.valid_mask  # [bs, T] 或 [bs * A, T]

        count[prefix + "exist_count"] += label_mask.any(-1).sum()    # 只要有一个可见，我们就认为其整体可见

        metric[prefix + "ADE"] += average_displacement_error(predict.position, target.position, label_mask)
        # metric[prefix + "FDE"] += final_displacement_error(predict.position, target.position, label_mask)   
        metric[prefix + "FDE"] += final_displacement_error_every_epoch(predict.position, target.position, label_mask)   
    
        # 之前的指标，用于和之前的信息比较
        metric[prefix + "trajectory_x"] += np.sum(np.abs(predict.get_position_x() - target.get_position_x()) * label_mask, axis=0)
        metric[prefix + "trajectory_y"] += np.sum(np.abs(predict.get_position_y() - target.get_position_y()) * label_mask, axis=0)   
        heading_abs = np.abs(predict.heading - target.heading) * label_mask
        metric[prefix + "heading"] += np.sum(heading_abs, axis=0) * 180 / 3.14
        metric[prefix + "heading_correct"] += np.sum(normalize_angle(heading_abs), axis=0) * 180 / 3.14
        metric[prefix + "existence"] += (predict.valid_mask == target.valid_mask).sum(axis=0)    # 存在loss二分类


    @staticmethod
    def _calculate_agent_collision(positions, metrics, gt_mask):

        # BUG？是否要使用pred的 exsit mask feature
        positions = np.where(gt_mask[..., np.newaxis], positions, np.full_like(positions, 999999))  # 看不见的就设为很大的数，下面计算碰撞率，就撞不到

        # 4.48 = np.hypot(1.85, 4.084) 参考occ world的ego长度
        nuplan_collisions_times = collision_by_radius_check(positions, radius_threshold=4.48)   # [bs, A]
        metrics['ego_nuplan_collision_sum'] += nuplan_collisions_times[:, 0].sum(0)
        metrics['agent_nuplan_collision_sum'] += nuplan_collisions_times[:, 1:].sum(axis=(0, 1))

        collisions_times = check_collisions(positions, gt_mask)   # [bs, A, T]
        metrics['ego_collision_sum'] += collisions_times[:, 0].sum(0)
        metrics['agent_collision_sum'] += collisions_times[:, 1:].sum(axis=(0, 1))

    @staticmethod
    def _calculate_agent_metric_end(prefix: str, metrics, counts):
        metrics[prefix + 'FDE'] /= counts[prefix + 'exist_count']
        metrics[prefix + 'ADE'] /= counts[prefix + 'exist_count']
        metrics[prefix + 'collision_sum'] /= counts[prefix + 'exist_count']
        metrics[prefix + 'nuplan_collision_sum'] /= counts[prefix + 'exist_count']

        # 修正之前的逐时间平均值
        metrics[prefix + 'trajectory_x_correct'] = metrics[prefix + 'trajectory_x'] / counts[prefix + 'exist_count']
        metrics[prefix + 'trajectory_y_correct'] = metrics[prefix + 'trajectory_y'] / counts[prefix + 'exist_count']
        metrics[prefix + 'heading_correct'] /= counts[prefix + 'exist_count']
        metrics[prefix + 'existence_correct'] = metrics[prefix + 'existence'] / counts[prefix + 'exist_count']

        # 之前的计算逐时间的平均值
        metrics[prefix + 'trajectory_x'] /= counts[prefix + 'full_count']
        metrics[prefix + 'trajectory_y'] /= counts[prefix + 'full_count']
        metrics[prefix + 'heading'] /= counts[prefix + 'full_count']
        metrics[prefix + 'existence'] /= counts[prefix + 'full_count']

    def _calculate_map_F1_score_end(self, metrics):

        TP = metrics['map_F1_TP']
        FP = metrics['map_F1_FP']
        FN = metrics['map_F1_FN']
        # 计算精度和召回率
        precision = TP / (TP + FP) if (TP + FP).all() > 0 else np.zeros(self.output_window_T)
        recall = TP / (TP + FN) if (TP + FN).all() > 0 else np.zeros(self.output_window_T)

        # 计算 F1 分数
        metrics['map_f1_score'] = 2 * (precision * recall) / (precision + recall) if (precision + recall).all() > 0 else np.zeros(self.output_window_T)
    