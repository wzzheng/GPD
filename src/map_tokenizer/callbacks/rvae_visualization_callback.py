from typing import Any, List, Optional

import numpy as np
import numpy.typing as npt
import cv2

import torch
import torch.utils.data
import pytorch_lightning as pl

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate

from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVector
from sledge.autoencoder.preprocessing.features.sledge_raster_feature import SledgeRaster
from sledge.common.visualization.sledge_visualization_utils import get_sledge_raster, get_sledge_vector_as_raster, draw_decode_raster_img


class RVAEVisualizationCallback(pl.Callback):
    """
    Pytorch Lightning callback that visualizes the autoencoder inputs/outputs and logs them in Tensorboard.
    """

    def __init__(
        self,
        images_per_tile: int,
        num_train_tiles: int,
        num_val_tiles: int,
        config: RVAEConfig,
    ):
        """
        Initializes the RVAE visualization callback.
        :param images_per_tile: number of visualized samples (columns)
        :param num_train_tiles: number of visualized tiles from training set
        :param num_val_tiles: number of visualized tiles from validation set
        :param config: RVAEConfig object storting plotting parameters
        """
        super().__init__()

        self.custom_batch_size = images_per_tile
        self.num_train_images = num_train_tiles * images_per_tile
        self.num_val_images = num_val_tiles * images_per_tile
        self.config = config

        # lazy loaded
        self.train_dataloader: Optional[torch.utils.data.DataLoader] = None
        self.val_dataloader: Optional[torch.utils.data.DataLoader] = None
        self.test_dataloader: Optional[torch.utils.data.DataLoader] = None

    def _initialize_dataloaders(self, datamodule: pl.LightningDataModule) -> None:
        """
        Initialize the dataloaders. This makes sure that the same examples are sampled
        every time for comparison during visualization.
        :param datamodule: lightning datamodule
        """
        train_set = datamodule.train_dataloader().dataset
        val_set = datamodule.val_dataloader().dataset

        self.train_dataloader = self._create_dataloader(train_set, self.num_train_images)
        self.val_dataloader = self._create_dataloader(val_set, self.num_val_images)

    def _create_dataloader(self, dataset: torch.utils.data.Dataset, num_samples: int) -> torch.utils.data.DataLoader:
        """
        Creates torch dataloader given dataset.
        :param dataset: torch dataset
        :param num_samples: size of random subset for visualization
        :return: torch dataloader of sampler subset
        """
        dataset_size = len(dataset)
        num_keep = min(dataset_size, num_samples)
        sampled_indices = np.random.choice(dataset_size, num_keep, replace=False)
        subset = torch.utils.data.Subset(dataset=dataset, indices=sampled_indices)
        return torch.utils.data.DataLoader(
            dataset=subset,
            batch_size=self.custom_batch_size,
            collate_fn=FeatureCollate(),
        )

    def _log_from_dataloader(
        self,
        pl_module: pl.LightningModule,
        dataloader: torch.utils.data.DataLoader,
        loggers: List[Any],
        training_step: int,
        prefix: str,
    ) -> None:
        """
        Visualizes and logs all examples from the input dataloader.
        :param pl_module: lightning module used for inference
        :param dataloader: torch dataloader
        :param loggers: list of loggers from the trainer
        :param training_step: global step in training
        :param prefix: prefix to add to the log tag
        """
        for batch_idx, batch in enumerate(dataloader):
            features: FeaturesType = batch[0]
            targets: TargetsType = batch[1]
            predictions = self._infer_model(pl_module, move_features_type_to_device(features, pl_module.device))

            self._log_batch(loggers, features, targets, predictions, batch_idx, training_step, prefix)

    def _log_batch(
        self,
        loggers: List[Any],
        features: FeaturesType,
        targets: TargetsType,
        predictions: TargetsType,
        batch_idx: int,
        training_step: int,
        prefix: str,
    ) -> None:
        """
        Visualizes and logs a batch of data (features, targets, predictions) from the model.
        :param loggers: list of loggers from the trainer
        :param features: tensor of model features
        :param targets: tensor of model targets
        :param predictions: tensor of model predictions
        :param batch_idx: index of total batches to visualize
        :param training_step: global training step
        :param prefix: prefix to add to the log tag
        """

        image_batch = self._get_images_from_features(features, targets, predictions)
        tag = f"{prefix}_visualization_{batch_idx}"

        for logger in loggers:
            if isinstance(logger.experiment, torch.utils.tensorboard.SummaryWriter):
                logger.experiment.add_images(
                    tag=tag,
                    img_tensor=torch.from_numpy(image_batch),
                    global_step=training_step,
                    dataformats="NHWC",
                )

    def _get_images_from_features(
        self, features: FeaturesType, targets: TargetsType, predictions: TargetsType
    ) -> npt.NDArray[np.uint8]:
        """
        Create a list of RGB raster images from a batch of model data of features.
        :param features: tensor of model features
        :param targets: tensor of model targets
        :param predictions: tensor of model predictions
        :return: list of raster images
        """
        output_raster = []
        if predictions.get('sledge_vector') is not None:
            # 对向量实体进行可视化
            for gt_sledge_vector, pred_sledge_vector in zip(
                targets["sledge_vector"].unpack(),
                predictions["sledge_vector"].unpack(),
            ):
                column = []
                pred_sledge_vector: SledgeVector
                pred_sledge_vector_raster = get_sledge_vector_as_raster(
                    sledge_vector=pred_sledge_vector.torch_to_numpy(apply_sigmoid=True),
                    config=self.config
                )
                column.append(pred_sledge_vector_raster)

                gt_sledge_vector: SledgeVector
                gt_sledge_vector_raster = get_sledge_vector_as_raster(
                    sledge_vector=gt_sledge_vector.torch_to_numpy(apply_sigmoid=False),
                    config=self.config
                )
                column.append(gt_sledge_vector_raster)

                output_raster.append(np.concatenate(column, axis=0))
            output_raster = np.asarray(output_raster)

        output_restruction_raster = []
        if predictions.get('sledge_raster') is not None:
            # 对直接重建出来图片进行可视化
            for raster_gt, raster_pred in zip(features["sledge_raster"].unpack(), predictions['sledge_raster']):
                column = []
                map_raster_gt = get_sledge_raster(raster_gt, self.config.pixel_frame)
                column.append(map_raster_gt)

                map_raster_restruction = draw_decode_raster_img(raster_pred, self.config.pixel_frame)
                column.append(map_raster_restruction)

                output_restruction_raster.append(np.concatenate(column, axis=0))
            output_restruction_raster = np.asarray(output_restruction_raster)

        if len(output_restruction_raster) > 0:
            if len(output_raster) > 0:
                # 同时重建vector+raster
                return np.concatenate([output_raster, output_restruction_raster], axis=1)
            else:
                # 只重建raster
                return output_restruction_raster
        else:
            # 只重建vector
            return output_raster

    def _infer_model(self, pl_module: pl.LightningModule, features: FeaturesType) -> TargetsType:
        """
        Make an inference of the input batch features given a model.
        :param pl_module: lightning model
        :param features: model inputs
        :return: model predictions
        """
        with torch.no_grad():
            pl_module.eval()
            predictions = pl_module(features)
            predictions = move_features_type_to_device(predictions, torch.device("cpu"))
            pl_module.train()

        return predictions

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        unused: Optional = None,  # type: ignore
    ) -> None:
        """
        Visualizes and logs training examples at the end of the epoch.
        :param trainer: lightning trainer
        :param pl_module: lightning module
        """
        assert hasattr(trainer, "datamodule"), "Trainer missing datamodule attribute"
        assert hasattr(trainer, "global_step"), "Trainer missing global_step attribute"

        if self.train_dataloader is None:
            self._initialize_dataloaders(trainer.datamodule)

        self._log_from_dataloader(pl_module, self.train_dataloader, trainer.loggers, trainer.global_step, "train")

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        unused: Optional = None,  # type: ignore
    ) -> None:
        """
        Visualizes and logs validation examples at the end of the epoch.
        :param trainer: lightning trainer
        :param pl_module: lightning module
        """
        assert hasattr(trainer, "datamodule"), "Trainer missing datamodule attribute"
        assert hasattr(trainer, "global_step"), "Trainer missing global_step attribute"

        if self.val_dataloader is None:
            self._initialize_dataloaders(trainer.datamodule)

        self._log_from_dataloader(pl_module, self.val_dataloader, trainer.loggers, trainer.global_step, "val")

    def on_test_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        unused: Optional = None,  # type: ignore
    ) -> None:
        """
        Visualizes and logs test examples at the end of the epoch.
        :param trainer: lightning trainer
        :param pl_module: lightning module
        """
        assert hasattr(trainer, "datamodule"), "Trainer missing datamodule attribute"
        assert hasattr(trainer, "global_step"), "Trainer missing global_step attribute"

        if self.test_dataloader is None:
            test_set = trainer.datamodule.test_dataloader().dataset
            self.train_dataloader = self._create_dataloader(test_set, self.num_train_images)

        self._log_from_dataloader(pl_module, self.train_dataloader, trainer.loggers, trainer.global_step, "test")

def move_features_type_to_device(batch: FeaturesType, device: torch.device) -> FeaturesType:
    """
    Move all features to a device
    :param batch: batch of features
    :param device: new device
    :return: batch moved to new device
    """
    output = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):     # 可视化raster
            output[key] = value.to(device)
        else:
            output[key] = value.to_device(device)
    return output