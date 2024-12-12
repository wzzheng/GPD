import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import pickle
import numpy as np

from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig

class VectorQuantizer(nn.Module):
    def __init__(self, config: Union[RVAEConfig]):
        # 代码来自：https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = config.latent_channel
        self._num_embeddings = config.num_embeddings

        self._linear = nn.Linear(self._embedding_dim, self._embedding_dim)

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)

        if config.kmeans_weight_path is not None:
            kmeans_weight = np.load(config.kmeans_weight_path)  # shape should be (64, 64)
            self._embedding.weight.data = torch.tensor(kmeans_weight, dtype=torch.float32)

        self._commitment_cost = config.commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()

        # with open('50w_total_128codebook_1_discrete_input_latent.pkl', 'ab') as f:
        #     pickle.dump(inputs.detach().cpu().numpy(), f)

        inputs = self._linear(inputs)

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(inputs.shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encoding_indices