from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class RVAEConfig():
    """Configuration dataclass for Raster-Vector-Autoencoder."""

    # 1. features raw
    radius: int = 100
    pose_interval: int = 1.0

    # 2. features raster & vector
    frame: Tuple[int, int] = (64, 64)

    num_lines: int = 50
    num_vehicles: int = 50
    num_pedestrians: int = 20
    num_static_objects: int = 30
    num_green_lights: int = 20
    num_red_lights: int = 20

    num_line_poses: int = 20
    vehicle_max_velocity: float = 15
    pedestrian_max_velocity: float = 2

    pixel_size: float = 0.25
    line_dots_radius: int = 0

    # 3. raster encoder π
    model_name: str = "resnet50"
    local_weights_path: str = "/lpai/volumes/autopilot-end2end-lf/xzx/public_models/resnet50-11ad3fa6.pth"
    down_factor: int = 32  # NOTE: specific to resnet
    num_input_channels: int = 1
    latent_channel: int = 128

    # 3.5. vector quantizer
    num_embeddings: int = 128
    commitment_cost: float = .25
    kmeans_weight_path: str = "/lpai/volumes/autopilot-end2end-lf/xzx/public_models/50w_total_128codebook_1_discrete_input_latent_weight.npy"

    # 4. vector decoder φ
    use_vector_decode: bool = True
    num_encoder_layers: int = 0
    num_decoder_layers: int = 6

    patch_size: int = 1
    dropout: float = 0.1
    num_head: int = 8
    d_model: int = 512
    d_ffn: int = 2048
    activation: str = "relu"
    normalize_before: bool = False
    positional_embedding: str = "sine"
    split_latent: bool = False

    head_d_ffn: int = 1024
    head_num_layers: int = 1

    num_line_queries: int = 50
    num_vehicle_queries: int = 50
    num_pedestrian_queries: int = 20
    num_static_object_queries: int = 30
    num_green_light_queries: int = 20
    num_red_light_queries: int = 20
    model_decode_ckpt_path: str = '/lpai/volumes/autopilot-end2end-lf/xzx/public_models/slege_vq_950epoch_100w_decoder_weight.pt'

    # 5. raster decoder
    use_raster_decode: bool = False
    num_raster_decoder_layers: int = 4
    num_output_channels: int = 1

    # matching & loss
    line_reconstruction_weight: float = 2.0
    line_ce_weight: float = 5.0

    box_reconstruction_weight: float = 2.0
    box_ce_weight: float = 5.0

    ego_reconstruction_weight: float = 1.0
    kl_weight: float = 0.1

    norm_by_count: bool = False

    # output
    threshold: float = 0.3

    @property
    def pixel_frame(self) -> Tuple[int, int]:
        frame_width, frame_height = self.frame
        return int(frame_width / self.pixel_size), int(frame_height / self.pixel_size)

    @property
    def latent_frame(self) -> Tuple[int, int]:
        pixel_width, pixel_height = self.pixel_frame
        return int(pixel_width / self.down_factor), int(pixel_height / self.down_factor)

    @property
    def patches_frame(self) -> Tuple[int, int]:
        latent_width, latent_height = self.latent_frame
        return int(latent_width / self.patch_size), int(latent_height / self.patch_size)

    @property
    def num_patches(self) -> int:
        patches_width, patches_height = self.patches_frame
        return patches_width * patches_height

    @property
    def d_patches(self) -> int:

        num_channels = self.latent_channel // 2 if self.split_latent else self.latent_channel
        return int(self.patch_size**2) * num_channels

    @property
    def num_queries_list(self) -> List[int]:
        return [
            self.num_line_queries
        ]