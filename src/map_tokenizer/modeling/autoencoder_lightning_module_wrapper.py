import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from hydra.utils import instantiate
from omegaconf import DictConfig
import gc
import time

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import pytorch_lightning as pl

from nuplan.planning.script.builders.lr_scheduler_builder import build_lr_scheduler
from nuplan.planning.training.modeling.objectives.abstract_objective import aggregate_objectives
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType

from sledge.autoencoder.modeling.autoencoder_torch_module_wrapper import AutoencoderTorchModuleWrapper
from sledge.autoencoder.modeling.matching.abstract_matching import AbstractMatching
from sledge.autoencoder.modeling.metrics.abstract_custom_metric import AbstractCustomMetric
from sledge.autoencoder.modeling.objectives.abstract_custom_objective import AbstractCustomObjective
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

logger = logging.getLogger(__name__)


class AutoencoderLightningModuleWrapper(pl.LightningModule):
    """
    Custom lightning module that wraps the training/validation/testing procedure and handles the objective/metric computation.
    """

    def __init__(
        self,
        model: AutoencoderTorchModuleWrapper,
        objectives: List[AbstractCustomObjective],
        metrics: Optional[List[AbstractCustomMetric]],
        matchings: Optional[List[AbstractMatching]],
        optimizer: Optional[DictConfig] = None,
        lr_scheduler: Optional[DictConfig] = None,
        warm_up_lr_scheduler: Optional[DictConfig] = None,
        objective_aggregate_mode: str = "sum",
        epochs: int = 50,
        lr_rate: int = 1
    ) -> None:
        """
        Initialize lightning autoencoder wrapper.
        :param model: autoencoder torch module wrapper.
        :param objectives: list of autoencoder objectives computed at each step
        :param metrics: optional list of metrics to track
        :param matchings: optional list of matching objects (e.g. for hungarian objectives)
        :param optimizer: config for instantiating optimizer. Can be 'None' for older models
        :param lr_scheduler: config for instantiating lr_scheduler. Can be 'None' for older models and when an lr_scheduler is not being used.
        :param warm_up_lr_scheduler: _description_, defaults to None
        :param objective_aggregate_mode: how should different objectives be combined, can be 'sum', 'mean', and 'max'.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.objectives = objectives
        self.metrics = metrics
        self.matchings = matchings
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.warm_up_lr_scheduler = warm_up_lr_scheduler
        self.objective_aggregate_mode = objective_aggregate_mode
        self.epochs = epochs
        self.lr_rate = lr_rate
        # self.cuda_index_train = 0
        # self.cuda_index_val = 0

    def _step(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str) -> torch.Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """
        features, targets, scenarios = batch

        predictions = self.forward(features)
        matchings = self._compute_matchings(predictions, targets)
        objectives = self._compute_objectives(predictions, features, targets, matchings, scenarios)
        metrics = self._compute_metrics(predictions, targets, matchings, scenarios)
        loss = aggregate_objectives(objectives, agg_mode=self.objective_aggregate_mode)

        self._log_step(loss, objectives, metrics, prefix)
        return loss

    def _compute_objectives(
        self, predictions: FeaturesType, gt: FeaturesType, targets: TargetsType, matchings: TargetsType, scenarios: ScenarioListType
    ) -> Dict[str, torch.Tensor]:
        """
        Computes a set of learning objectives used for supervision given the model's predictions and targets.

        :param predictions: dictionary of predicted dataclasses.
        :param gt: 输入模型的raster数据
        :param targets: dictionary of target dataclasses.
        :param matchings: dictionary of prediction-target matching.
        :param scenarios: list of scenario types (for adaptive weighting)
        :return: dictionary of objective names and values
        """
        objectives_dict: Dict[str, torch.Tensor] = {} 
        for objective in self.objectives:
            objectives_dict.update(objective.compute(predictions, gt, targets, matchings, scenarios))

        objectives_dict['embedding_loss'] = predictions['embedding_loss']   # vq-vae 码本loss
        return objectives_dict

    def _compute_metrics(
        self, predictions: FeaturesType, targets: TargetsType, matchings: TargetsType, scenarios: ScenarioListType
    ) -> Dict[str, torch.Tensor]:
        """
        Computes a set of metrics used for logging.

        :param predictions: dictionary of predicted dataclasses.
        :param targets: dictionary of target dataclasses.
        :param matchings: dictionary of prediction-target matching.
        :param scenarios: list of scenario types (for adaptive weighting)
        :return: dictionary of metrics names and values
        """
        metrics_dict: Dict[str, torch.Tensor] = {}
        if self.metrics:
            for metric in self.metrics:
                metrics_dict.update(metric.compute(predictions, targets, matchings, scenarios))
        return metrics_dict

    def _compute_matchings(self, predictions: FeaturesType, targets: TargetsType) -> FeaturesType:
        """
        Computes a the matchings (e.g. for hungarian loss) between prediction and targets.

        :param predictions: dictionary of predicted dataclasses.
        :param targets: dictionary of target dataclasses.
        :return: dictionary of matching names and matching dataclasses
        """
        matchings_dict: Dict[str, torch.Tensor] = {}
        if self.matchings:
            for matching in self.matchings:
                matchings_dict.update(matching.compute(predictions, targets))
        return matchings_dict

    def _log_step(
        self,
        loss: torch.Tensor,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = "loss",
    ) -> None:
        """
        Logs the artifacts from a training/validation/test step.

        :param loss: scalar loss value
        :type objectives: [type]
        :param metrics: dictionary of metrics names and values
        :param prefix: prefix prepended at each artifact's name
        :param loss_name: name given to the loss for logging
        """
        self.log(f"loss/{prefix}_{loss_name}", loss, sync_dist=True)

        for key, value in objectives.items():
            self.log(f"objectives/{prefix}_{key}", value, sync_dist=True)

        if self.metrics:
            for key, value in metrics.items():
                self.log(f"metrics/{prefix}_{key}", value, sync_dist=True)

    def training_step(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int) -> torch.Tensor:
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

    def test_step(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "test")

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        return self.model(features)

    def configure_optimizers(self):
        # 配置参数组：codebook 和其他层
        lr_rate = self.lr_rate     # 对于多机器，将学习率翻倍
        optimizer = AdamW([
            {
                'params': self.model._vector_quantization.parameters(),
                'lr': 5e-3 * lr_rate,  # 初始最大学习率为 1e-3
                "weight_decay": 0
            },
            {
                'params': [param for name, param in self.model.named_parameters() 
                           if not name.startswith('_vector_quantization')],
                'lr': 1e-4 * lr_rate,  # 其他层初始最大学习率为 1e-4
                "weight_decay": 0
            }
        ])

        # 配置学习率调度器：Warm-up + 余弦退火

        # 1. Linear warm-up (线性预热)
        total_epochs = self.epochs  # 总训练 epoch 数（示例）
        warmup_epochs = int(0.05 * total_epochs)  # 预热的 epoch 数量（可根据需求调整）

        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)

        # 2. Cosine annealing (余弦退火)
        cosine_scheduler= CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-5 * lr_rate)

        # 3. Sequential scheduler (按顺序运行预热和余弦退火)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

        # 配置 Lightning 的优化器和调度器
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "name": "scheduler"
                }
            }
    
    def on_train_epoch_end(self):
        gc.collect()

    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     # 获取所有优化器的学习率
    #     for i, optimizer in enumerate(self.trainer.optimizers):
    #         for j, param_group in enumerate(optimizer.param_groups):
    #             lr = param_group['lr']
    #             print(f"Batch {batch_idx+1}: Optimizer {i}, Param Group {j}, LR: {lr}")

    # def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
    #     if self.device.index == 0:
    #         self.cuda_index_val += 1
    #         self.on_validation_batch_start_time = time.time()
    
    # def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
    #     if self.device.index == 0:
    #         on_validation_batch_end_time = time.time()
    #         if self.cuda_index_val % 1 == 0:
    #             logger.info(f"on_val_batch_end: {on_validation_batch_end_time - self.on_validation_batch_start_time}")
    
    # def on_train_batch_start(self, batch, batch_idx):
    #     if self.device.index == 0:
    #         self.cuda_index_train += 1
    #         self.on_train_batch_start_time = time.time()

    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     if self.device.index == 0:
    #         on_train_epoch_end_time = time.time()
    #         if self.cuda_index_train % 1 == 0:
    #             logger.info(f"on_train_batch_end: {on_train_epoch_end_time - self.on_train_batch_start_time}")