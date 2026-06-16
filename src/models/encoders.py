"""Vision encoders: frozen DINOv2 for RGB, trainable PointNeXt for point clouds."""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from .pointnext import PointNeXt


class DINOv2Encoder(nn.Module):
    """Frozen DINOv2 encoder."""

    def __init__(self, model_name: str = "facebook/dinov2-with-registers-base"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch features from an RGB image.

        Args:
            x: (B, 3, H, W) RGB images.

        Returns:
            (B, N, D) patch tokens, where N = (H/P) * (W/P) and D = hidden_size.
        """
        outputs = self.model(x, return_dict=True)
        skip = 1 + self.num_register_tokens
        patch_tokens = outputs.last_hidden_state[:, skip:, :]
        return patch_tokens

    @property
    def output_dim(self) -> int:
        return self.hidden_size


class PointNeXtEncoder(nn.Module):
    """Trainable PointNeXt encoder.

    Wraps the PointNeXt U-Net to expose a consistent interface with
    `DINOv2Encoder`. Outputs feature tokens AND metric XYZ centroids;
    fusion uses the centroids for the 3D Fourier positional embed.
    """

    def __init__(
        self,
        width: int = 64,
        sa_radii: tuple[float, ...] = (0.025, 0.05, 0.10, 0.20),
        blocks: tuple[int, ...] = (1, 2, 1, 1),
        use_rgb: bool = True,
    ):
        super().__init__()
        self.use_rgb = use_rgb
        self.model = PointNeXt(
            c=width, sa_radii=sa_radii, blocks=blocks, use_rgb=use_rgb
        )

    def forward(self, xyz: torch.Tensor, rgb_pcl: Optional[torch.Tensor] = None):
        """Encode a point cloud into feature tokens and centroids.

        Args:
            xyz: (B, N, 3) metric meters.
            rgb_pcl: (B, N, 3) per-point RGB in [0, 1]. Required iff use_rgb.

        Returns:
            features: (B, 256, output_dim).
            centroids: (B, 256, 3).
        """
        return self.model(xyz, rgb_pcl=rgb_pcl)

    @property
    def output_dim(self) -> int:
        return self.model.out_dim
