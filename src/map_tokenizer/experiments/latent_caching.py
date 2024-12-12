import logging
from pathlib import Path
from omegaconf import DictConfig

import numpy as np
import torch
from tqdm import tqdm
import gzip
import pickle
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle

from sledge.script.builders.model_builder import build_autoencoder_torch_module_wrapper
from sledge.script.builders.autoencoder_builder import build_autoencoder_lightning_datamodule
from sledge.autoencoder.preprocessing.features.latent_feature import Latent
from sledge.autoencoder.modeling.autoencoder_lightning_module_wrapper import AutoencoderLightningModuleWrapper
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVectorElement, SledgeVector
from sledge.autoencoder.preprocessing.features.sledge_raster_feature import SledgeRaster, SledgeRasterIndex


logger = logging.getLogger(__name__)


def cache_latent(cfg: DictConfig, worker: WorkerPool) -> None:
    """
    Build the lightning datamodule and cache the latent of all training samples.
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    """

    assert cfg.autoencoder_checkpoint is not None, "cfg.autoencoder_checkpoint is not specified for latent caching!"

    # Create model
    logger.info("Building Autoencoder Module...")
    torch_module_wrapper = build_autoencoder_torch_module_wrapper(cfg)
    torch_module_wrapper = AutoencoderLightningModuleWrapper.load_from_checkpoint(
        cfg.autoencoder_checkpoint, model=torch_module_wrapper
    ).model
    logger.info("Building Autoencoder Module...DONE!")

    # Build the datamodule
    logger.info("Building Datamodule Module...")
    datamodule = build_autoencoder_lightning_datamodule(cfg, worker, torch_module_wrapper)
    datamodule.setup("cache_latent")
    dataloader = datamodule.test_dataloader()
    logger.info("Building Datamodule Module...DONE!")

    autoencoder_cache_path = Path(cfg.cache.autoencoder_cache_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    storing_mechanism = FeatureCachePickle()
    torch_module_wrapper = torch_module_wrapper.to(device)

    bs = cfg.data_loader.params.batch_size

    # Perform inference
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="Cache Latents (batch-wise)"):
            # Assuming batch is a tuple of (inputs, labels, indices) where indices track sample order
            features, targets, scenarios = datamodule.transfer_batch_to_device(batch, device, 0)

            features = _preprocess_data(features, device)

            predictions = torch_module_wrapper.forward(features, encode_only=True)
            # assert "quantized" in predictions

            encoding_indices = predictions["encoding_indices"].reshape(bs, -1).cpu().numpy()

            for encoding_indice, scenario in zip(encoding_indices, scenarios):
                file_name = (
                    autoencoder_cache_path
                    / scenario.log_name
                    / scenario.scenario_type
                    / scenario.token
                    / cfg.cache.latent_name
                ).with_suffix(".gz")
                with gzip.open(str(file_name), 'wb', compresslevel=1) as f:
                    pickle.dump({"encoding_indice": encoding_indice.astype(np.uint8)}, f)


    return None

def _preprocess_data(features, device):
    # bs = features['sledge_vector'].lines.states.shape[0] * 101
    # _ = torch.full((bs, 1, 2, 2), -10, device=device)    # 避免修改大量代码，注入一个占位tensor

    # features['sledge_vector'] = SledgeVector(
    #         SledgeVectorElement(
    #             features['sledge_vector'].lines.states.reshape(-1, 50, 20, 2),
    #             features['sledge_vector'].lines.mask.reshape(-1, 50)
    #         ),
    #         SledgeVectorElement(_, _[..., 0, 0]),
    #         SledgeVectorElement(_, _[..., 0, 0]),
    #         SledgeVectorElement(_, _[..., 0, 0]),
    #         SledgeVectorElement(_, _[..., 0, 0]),
    #         SledgeVectorElement(_, _[..., 0, 0]),
    #         SledgeVectorElement(_, _[..., 0, 0]),
    #     )
    features["sledge_raster"] = SledgeRaster(
        features["sledge_raster"].data.reshape(-1, 12, 256, 256)
    )
    return features