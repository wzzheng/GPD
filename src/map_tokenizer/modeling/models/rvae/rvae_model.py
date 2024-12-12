import torch
import torch.nn as nn

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from sledge.autoencoder.modeling.autoencoder_torch_module_wrapper import AutoencoderTorchModuleWrapper
from sledge.autoencoder.modeling.models.rvae.rvae_encoder import RVAEEncoder
from sledge.autoencoder.modeling.models.rvae.rvae_vector_decoder import RVAEVectorDecoder
from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig
from sledge.autoencoder.preprocessing.features.latent_feature import Latent
from sledge.autoencoder.preprocessing.feature_builders.feature_builder import FeatureBuilder
from sledge.autoencoder.modeling.models.rvae.quantizer import VectorQuantizer
from sledge.autoencoder.modeling.models.rvae.rvae_raster_resnet50_decoder import Bottleneck, ResNet
from sledge.autoencoder.preprocessing.target_builders.empty_target_builder import EmptyTargetBuilder

# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=16, time_horizon=16, interval_length=1)


class RVAEModel(AutoencoderTorchModuleWrapper):
    """Raster-Vector Autoencoder in of SLEDGE."""

    def __init__(self, config: RVAEConfig):
        """
        Initialize Raster-Vector Autoencoder.
        :param config: configuration dataclass of RVAE.
        """
        feature_builders = [FeatureBuilder()]
        target_builders = [EmptyTargetBuilder()]

        super().__init__(feature_builders=feature_builders, target_builders=target_builders)

        self._config = config

        self._raster_encoder = RVAEEncoder(config)

        self.pre_quantization_conv = nn.Conv2d(
            2 * config.latent_channel, config.latent_channel, kernel_size=3, stride=1, padding=1)
        self._vector_quantization = VectorQuantizer(config)

        # decoder
        self._use_vector_decode = config.use_vector_decode
        self._use_raster_decode = config.use_raster_decode
        if self._use_vector_decode:
            self._vector_decoder = RVAEVectorDecoder(config)

        if self._use_raster_decode:
            # self._raster_decoder = RVAERasterDecoder(config)
            self._raster_decoder = ResNet(
                Bottleneck,
                [3, 6, 4, 3],
                config=config
            )


    def forward(self, features: FeaturesType, encode_only: bool = False) -> TargetsType:
        """Inherited, see superclass."""
        predictions: TargetsType = {}

        # encoding
        latent = self._raster_encoder(features["sledge_raster"].data)

        latent = self.pre_quantization_conv(latent)
        embedding_loss, quantized, perplexity, encoding_indices = self._vector_quantization(latent)
        predictions["embedding_loss"] = embedding_loss
        
        if encode_only:
            predictions['encoding_indices'] = encoding_indices
            return predictions

        # decoding
        if self._use_vector_decode:
            predictions["sledge_vector"] = self._vector_decoder(quantized)
        
        if self._use_raster_decode:
            predictions["sledge_raster"] = self._raster_decoder(quantized)

        return predictions

    def get_encoder(self) -> nn.Module:
        """Inherited, see superclass."""
        return self._raster_encoder

    def get_decoder(self) -> nn.Module:
        """Inherited, see superclass."""
        return self._vector_decoder
