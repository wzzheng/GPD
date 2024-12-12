import torch
import torch.nn as nn

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType

from .rvae_encoder import RVAEEncoder
from .rvae_config import RVAEConfig

class RVAEEncodeToLatentModel(nn.Module):

    def __init__(self, config: RVAEConfig):

        super().__init__()

        self._config = config
        self._embedding_dim = config.latent_channel
        self._num_embeddings = config.num_embeddings

        self._raster_encoder = RVAEEncoder(config)

        self.pre_quantization_conv = nn.Conv2d(
            2 * config.latent_channel, config.latent_channel, kernel_size=3, stride=1, padding=1)
        
        self._linear = nn.Linear(self._embedding_dim, self._embedding_dim)
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        

    def forward(self, line_raster: torch.Tensor) -> TargetsType:
        """Inherited, see superclass."""
        
        # encoding
        latent = self._raster_encoder(line_raster)

        inputs = self.pre_quantization_conv(latent)

        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        inputs = self._linear(inputs)

        flat_input = inputs.view(-1, self._embedding_dim)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)

        return encoding_indices.view(-1, 64)
