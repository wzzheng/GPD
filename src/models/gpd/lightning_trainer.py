from typing import Dict, Tuple, Union
from torch import Tensor

import copy
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

from src.optim.warmup_cos_lr import WarmupCosLR

from src.models.gpd.losses import reconstruction_agent_loss, reconstruction_map_loss
from src.bean.input_agent_bean import InputAgentBean
from src.bean.output_agent_bean import OutputAgentBean
from src.utils.simple_utils import generate_correct_agent_feature_indices, mask_feature, data_premask, agent_padding
from src.models.gpd.metrics import compute_agent_metrics
from src.utils.greedy_decode_utils import greedy_decode
# import logging
# import time
# logger = logging.getLogger(__name__)

class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
        batch_size=2,
        window_T=100,
        max_agent_num=33,
        strict_loading: bool = True,
        pred_window_T: int = 100        # 需要对其多少帧的loss
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.batch_size = batch_size
        self.window_T = window_T    # 模型的总时间长度大小
        self.strict_loading = strict_loading
        self.max_agent_num = max_agent_num
        self.pred_window_T = pred_window_T

    def _step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str
    ) -> torch.Tensor:
        data_train, data_label = self._data_prepare(batch[0])
        data_pred = self.forward(data_train)
        
        # concat到一起，方便计算loss和metric
        agent_label = self.preprocess_agent_label(data_label['agent'])
        agent_pred = self.preprocess_agent_pred(data_pred['agent'])

        losses = self._compute_objectives(agent_pred, agent_label, data_pred['map_latent'], data_label['map_latent'])

        if prefix == 'train':
            metrics = self.train_metrics
            counts = self.train_count
        elif prefix == 'val':
            metrics = self.val_metrics
            counts = self.val_count
        self._compute_metrics(agent_pred, agent_label, data_pred['map_latent'], data_label['map_latent'], metrics, counts)    # 不需要返回值，我们的指标都是一个epoch算一次
        
        self._log_step(losses["loss"], losses, prefix)  

        return losses["loss"]   # 这个必须传回loss，它会自己backward，自己outs.append(loss.detach())

    def _compute_objectives(self, agent_pred, agent_label, map_pred, map_label) -> Dict[str, torch.Tensor]:
        agent_loss_dict = reconstruction_agent_loss(agent_pred, agent_label)
        map_loss_dict = reconstruction_map_loss(map_pred, map_label)
        losses = {**agent_loss_dict, **map_loss_dict}
        losses['loss'] = sum(losses.values())
        return losses

    def _compute_metrics(self, agent_pred, agent_label, map_pred, map_label, metric, count):
        # ego metric
        compute_agent_metrics(agent_pred[:, :, 0], agent_label[:, :, 0], metric, 'ego')
        
        # agent metric
        compute_agent_metrics(agent_pred[:, :, 1:], agent_label[:, :, 1:], metric, 'agent')

        # map metric
        map_pred = torch.argmax(map_pred, dim=-1)
        metric['map_latent_acc'] += (map_label == map_pred).sum().item()
        
        # count
        bs, T, A, _ = agent_label.shape
        M = map_pred.shape[2]
        count["ego_count"] += bs * T
        count["agent_count"] += bs * T * (A - 1)
        count['map_count'] += bs * M * T

    def _log_step(
        self,
        loss: torch.Tensor,
        objectives: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = "loss",
    ) -> None:
        self.log(
            f"loss/{prefix}_{loss_name}",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size
        )

        for key, value in objectives.items():
            self.log(
                f"objectives/{prefix}_{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size
            )
        
    def training_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "train")

    def validation_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "val")

    def forward(self, *args, **kwargs) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        return self.model(*args, **kwargs)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        # Get optimizer
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        
        # Get lr_scheduler
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        return [optimizer], [scheduler]
    
    def on_train_epoch_end(self, *args, **kwargs):    
        # 计算train metric，         
        metrics_dict = self._compute_metric(self.train_metrics, self.train_count, "train_metric/")
        if metrics_dict is not None:
            self.log_dict(
                metrics_dict,
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # If on_epoch is True, the logger automatically logs the end of epoch metric value by calling .compute()
                batch_size=self.batch_size,
                sync_dist=True,
            )
        # 将所有的统计字典都重置，因为一个epoch结束，需要重新记录
        self.train_metrics = {k : 0 for k in self.train_metrics}
        self.train_count = {k : 0 for k in self.train_count}

    def on_validation_epoch_end(self, *args, **kwargs):    
        # 计算val metric，         
        metrics_dict = self._compute_metric(self.val_metrics, self.val_count, "val_metric/")
        if metrics_dict is not None:
            self.log_dict(
                metrics_dict,
                prog_bar=True,     # if True logs to the progress base.
                on_step=False,
                on_epoch=True,  # If on_epoch is True, the logger automatically logs the end of epoch metric value by calling .compute()
                batch_size=self.batch_size,
                sync_dist=True,
            )
        # 将所有的统计字典都重置，因为一个epoch结束，需要重新记录
        self.val_metrics = {k : 0 for k in self.val_metrics}
        self.val_count = {k : 0 for k in self.val_count}

    # Set gradients to `None` instead of zero to improve performance (not required on `torch>=2.0.0`).
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)


    # def on_load_checkpoint(self, checkpoint):
    #     # 删除不需要的键，这些键是用于快速抽取agent loss，存在register，但不需要记录他们的索引值
    #     keys_to_ignore = ['agent_feature_indices', 'agent_pred_feature_indices']
    #     for key in keys_to_ignore:
    #         if key in checkpoint["state_dict"]:
    #             del checkpoint["state_dict"][key]

    @staticmethod
    def preprocess_agent_label(data: Dict[str, Tensor]) -> Tensor:
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

        return agent_feature.transpose(1, 2)    # [bs, T, A, _]
    
    def preprocess_agent_pred(self, data: Dict[str, Tensor]) -> Tensor:
        position = data["position"]
        heading = data["heading"]
        valid_mask = data["valid_mask"]

        agent_feature = torch.cat(
            [
                position,
                heading,
                valid_mask,
            ],
            -1,
        )

        return agent_feature    # [bs, T, A, _]
    
    def on_fit_start(self) -> None:
        
        def _get_agent_metrics_collection(prefix):
            return {
                # prefix + "_trajectory_x_80": 0,
                # prefix + "_trajectory_x_10": 0,
                prefix + "_trajectory_x_1": 0,
                
                # prefix + "_trajectory_y_80": 0,
                # prefix + "_trajectory_y_10": 0,
                prefix + "_trajectory_y_1": 0,
                
                # prefix + "_heading_80": 0,
                # prefix + "_heading_10": 0,
                prefix + "_heading_1": 0,
                
                # prefix + "_existence_80": 0,
                # prefix + "_existence_10": 0,
                prefix + "_existence_1": 0,
            }

        metrics_collection = {
            **_get_agent_metrics_collection('ego'),
            **_get_agent_metrics_collection('agent'),
            "map_latent_acc": 0
        }

        count_collection = {    # 记录有多少例，最后用于求平均指标
            "agent_count": 0,
            "ego_count": 0,
            "map_count": 0
        }
         
        self.train_metrics = metrics_collection
        self.train_count = count_collection
        self.val_metrics = copy.deepcopy(metrics_collection)
        self.val_count = copy.deepcopy(count_collection)

    def _compute_metric(self, metric, count, prefix):
        
        def _get_agent_metric_result(prefix1, prefix2, count):
            prefix1 = prefix1 + prefix2
            return {
                # prefix1 + "_trajectory_x_80": metric[prefix2 + "_trajectory_x_80"] / count / 10,
                # prefix1 + "_trajectory_x_10": metric[prefix2 + "_trajectory_x_10"] / count / 5,
                prefix1 + "_trajectory_x_1": metric[prefix2 + "_trajectory_x_1"] / count,
                
                # prefix1 + "_trajectory_y_80": metric[prefix2 + "_trajectory_y_80"] / count / 10,
                # prefix1 + "_trajectory_y_10": metric[prefix2 + "_trajectory_y_10"] / count / 5,
                prefix1 + "_trajectory_y_1": metric[prefix2 + "_trajectory_y_1"] / count,
                
                # prefix1 + "_heading_80": metric[prefix2 + "_heading_80"] / count / 10,
                # prefix1 + "_heading_10": metric[prefix2 + "_heading_10"] / count / 5,
                prefix1 + "_heading_1": metric[prefix2 + "_heading_1"] / count,
                
                # prefix1 + "_existence_80": metric[prefix2 + "_existence_80"] / count / 10,
                # prefix1 + "_existence_10": metric[prefix2 + "_existence_10"] / count / 5,
                prefix1 + "_existence_1": metric[prefix2 + "_existence_1"] / count
            }
        
        return {
            **_get_agent_metric_result(prefix, 'ego', count["ego_count"]),
            **_get_agent_metric_result(prefix, 'agent', count["agent_count"]),
            prefix + "map_latent_acc": metric['map_latent_acc'] / count['map_count']
        }

    def _data_prepare(self, data):
        agent_data = data['feature'].data['agent']
        map_data = data['rvae_latent'].data['encoding_indice']

        agent_position = agent_data['position']
        agent_heading = agent_data['heading']
        agent_mask = agent_data['valid_mask']

        # 对agent进行一次掩码
        agent_position = mask_feature(agent_position, agent_mask)
        agent_heading = mask_feature(agent_heading, agent_mask)

        # padding agent to 33
        agent_position = agent_padding(agent_position, self.max_agent_num)
        agent_heading = agent_padding(agent_heading, self.max_agent_num)
        agent_mask = agent_padding(agent_mask, self.max_agent_num)


        # 组装train和label
        data_train = {
            'agent': {
                'position': agent_position[:, :, :self.window_T],
                'heading': agent_heading[:, :, :self.window_T],
                'valid_mask': agent_mask[:, :, :self.window_T]
            },
            'map_latent': map_data[:, :self.window_T]
        }

        data_label = {
            'agent': {
                'position': agent_position[:, :, 1: self.window_T + 1],
                'heading': agent_heading[:, :, 1: self.window_T + 1],
                'valid_mask': agent_mask[:, :, 1: self.window_T + 1]
            },
            'map_latent': map_data[:, 1: self.window_T + 1]
        }
        
        return data_train, data_label