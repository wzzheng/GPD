from typing import Dict, Tuple, Union
from torch import Tensor

import copy
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
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

class PlanningLightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
        batch_size=2,
        window_T=21,
        max_agent_num=33,
        strict_loading: bool = True,
        pred_window_T: int = 80,        # 需要对其多少帧的loss
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
        agent_position, ego_position_heading = self.forward(data_train)

        losses = self._compute_objectives(agent_position, ego_position_heading, data_label)

        self._log_step(losses["loss"], losses, prefix)  

        return losses["loss"]   # 这个必须传回loss，它会自己backward，自己outs.append(loss.detach())

    def _compute_objectives(self, agent_pred_position, ego_pred, data_label) -> Dict[str, torch.Tensor]:
        ego_label_position = data_label['agent']['position'][:, 0]
        ego_label_heading = data_label['agent']['heading'][:, 0]
        agent_label_position = data_label['agent']['position'][:, 1:]
        agent_label_mask = data_label['agent']['valid_mask'][:, 1:]
        
        # ego l1 loss
        ego_label = torch.cat(
            [
                ego_label_position,
                torch.stack(
                    [ego_label_heading.cos(), ego_label_heading.sin()], dim=-1
                )
            ],
            dim=-1
        )

        ego_xy_reg_loss = F.smooth_l1_loss(ego_pred[..., :2], ego_label[..., :2])
        ego_heading_reg_loss = F.smooth_l1_loss(ego_pred[..., 2:], ego_label[..., 2:])

        agent_reg_loss = F.smooth_l1_loss(
            agent_pred_position[agent_label_mask], agent_label_position[agent_label_mask]
        )

        loss = ego_xy_reg_loss + ego_heading_reg_loss + agent_reg_loss

        return {
            "loss": loss,
            "ego_xy_reg_loss": ego_xy_reg_loss,
            "ego_heading_reg_loss": ego_heading_reg_loss,
            "agent_reg_loss": agent_reg_loss,
        }


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


    # Set gradients to `None` instead of zero to improve performance (not required on `torch>=2.0.0`).
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

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
                'position': agent_position[:, :, self.window_T: self.window_T + self.pred_window_T],
                'heading': agent_heading[:, :, self.window_T: self.window_T + self.pred_window_T],
                'valid_mask': agent_mask[:, :, self.window_T: self.window_T + self.pred_window_T]
            }
        }
        
        return data_train, data_label