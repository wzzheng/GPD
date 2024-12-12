import torch
import torch.nn as nn

from .rvae_config import RVAEConfig
from .rvae_vector_decoder import RVAEVectorDecoder

class RVAEModel(nn.Module):
    """Raster-Vector Autoencoder in of SLEDGE."""

    def __init__(self, config: RVAEConfig):

        super().__init__()

        self._config = config

        self._num_embeddings = config.num_embeddings

        self._embedding = nn.Embedding(config.num_embeddings, config.latent_channel)

        self._vector_decoder = RVAEVectorDecoder(config)

    def forward(self, encoding_indices):
        """Inherited, see superclass."""
        # 将indice转为具体值
        bs, M = encoding_indices.shape
        encoding_indices = encoding_indices.flatten().unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=encoding_indices.device)
        encodings.scatter_(1, encoding_indices.to(torch.int64), 1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(bs, 8, 8, self._num_embeddings)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # decoding
        sledge_vector = self._vector_decoder(quantized)

        return sledge_vector