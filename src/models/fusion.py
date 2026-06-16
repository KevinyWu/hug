"""Patch fusion.

Two fusion paths controlled by `use_pointpainting`:
- False (default): RGB patches (DINOv2) + PCL tokens (PointNeXt) as parallel
  streams, concatenated into a 512-token sequence with modality embeds.
- True: PointPainting (Vora et al., CVPR 2020). Each PCL centroid is projected
  to the RGB image plane and bilinearly samples the DINOv2 patch feature there.
  Concat painted RGB + PCL feature -> MLP -> 256 fused tokens. Requires K.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import CrossAttentionBlock, TransformerBlock


class FourierPosEmbed(nn.Module):
    """Random Fourier features + learned linear bypass.

    `scale` sets B's std (cycles per input unit); ~1/scale is the shortest
    wavelength the sin/cos basis can resolve. The additive `raw_proj` gives
    downstream layers a linear arithmetic path that Fourier alone makes
    nonlinear. Use in_dim=3 for metric XYZ (meters), in_dim=2 for normalized
    pixel coords (image fractions in [0, 1]).
    """

    def __init__(
        self,
        d_model: int,
        in_dim: int = 3,
        scale: float = 1.0,
        use_raw: bool = True,
    ):
        super().__init__()
        self.register_buffer("matrix", scale * torch.randn(in_dim, d_model // 2))
        self.raw_proj = nn.Linear(in_dim, d_model, bias=False) if use_raw else None

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """(..., in_dim) → (..., d_model)."""
        proj = 2 * math.pi * (coords @ self.matrix)
        out = torch.cat([proj.sin(), proj.cos()], dim=-1)
        if self.raw_proj is not None:
            out = out + self.raw_proj(coords)
        return out


class PatchFusion(nn.Module):
    """Fuse RGB patches (DINOv2) + PCL tokens (PointNeXt) with point conditioning."""

    def __init__(
        self,
        d_rgb_patch: int = 1024,
        d_depth_patch: int = 256,
        d_model: int = 1024,
        n_patches: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        patch_grid_size: int = 16,
        use_rgb: bool = True,
        use_depth: bool = True,
        use_pointpainting: bool = True,
        image_size: int = 224,
        fourier_scale: float = 1.0,
        use_2d_point: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_patches = n_patches
        self.patch_grid_size = patch_grid_size
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.use_pointpainting = use_pointpainting
        self.image_size = image_size
        self.use_2d_point = use_2d_point

        # 3D embed for depth centroids (and 3D point in default mode). When
        # use_2d_point=False, the same embed serves the point token so attention
        # can natively measure click-to-centroid proximity via shared sin/cos
        # + raw-linear features. When use_2d_point=True, point uses a separate
        # 2D embed below; centroids still need 3D.
        self.pos_embed_3d = FourierPosEmbed(d_model, in_dim=3, scale=fourier_scale)
        if use_2d_point:
            self.pos_embed_2d = FourierPosEmbed(d_model, in_dim=2, scale=fourier_scale)

        if use_pointpainting:
            if not (use_rgb and use_depth):
                raise ValueError("use_pointpainting requires use_rgb and use_depth")
            self.painting_proj = nn.Sequential(
                nn.Linear(d_rgb_patch + d_depth_patch, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
        else:
            if use_rgb:
                self.patch_proj = nn.Linear(d_rgb_patch, d_model, bias=False)
                self.rgb_pos_embed = nn.Parameter(
                    torch.randn(1, n_patches, d_model) * 0.02
                )
            if use_depth:
                self.depth_patch_proj = nn.Linear(d_depth_patch, d_model, bias=False)
            if use_rgb and use_depth:
                self.modality_embed = nn.Parameter(torch.randn(2, 1, d_model) * 0.02)

        self.point_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.point_cross_attn = CrossAttentionBlock(d_model, n_heads, dropout=dropout)

        self.transformer = nn.Sequential(
            *[
                TransformerBlock(d_model, n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def _project_centroids(
        self, centroids: torch.Tensor, camera_K: torch.Tensor
    ) -> torch.Tensor:
        """(B, N, 3) metric XYZ → (B, N, 2) normalized [-1, 1] image coords."""
        X, Y, Z = centroids.unbind(-1)
        Z_safe = Z.clamp(min=1e-3)
        fx = camera_K[:, 0, 0:1]
        fy = camera_K[:, 1, 1:2]
        cx = camera_K[:, 0, 2:3]
        cy = camera_K[:, 1, 2:3]
        u = fx * (X / Z_safe) + cx
        v = fy * (Y / Z_safe) + cy
        norm_u = 2.0 * u / self.image_size - 1.0
        norm_v = 2.0 * v / self.image_size - 1.0
        return torch.stack([norm_u, norm_v], dim=-1)

    def _paint(
        self,
        rgb_patches: torch.Tensor,
        depth_centroids: torch.Tensor,
        camera_K: torch.Tensor,
    ) -> torch.Tensor:
        """Bilinearly sample DINOv2 patch features at projected centroid pixels.

        Returns (B, N_pcl, d_rgb_patch).
        """
        B, N, _ = depth_centroids.shape
        G = self.patch_grid_size
        d_rgb = rgb_patches.shape[-1]
        rgb_map = rgb_patches.transpose(1, 2).reshape(B, d_rgb, G, G)
        grid = self._project_centroids(depth_centroids, camera_K).unsqueeze(1)
        painted = F.grid_sample(
            rgb_map,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return painted.squeeze(2).transpose(1, 2)

    def forward(
        self,
        point: torch.Tensor,
        rgb_patches: Optional[torch.Tensor] = None,
        depth_patches: Optional[torch.Tensor] = None,
        depth_centroids: Optional[torch.Tensor] = None,
        camera_K: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse RGB + PCL patches with point conditioning.

        Args:
            point: (B, 3) metric XYZ in camera frame, or (B, 2) normalized
                pixel coords in [0, 1] when use_2d_point.
            rgb_patches: (B, N_rgb, d_rgb_patch).
            depth_patches: (B, N_pcl, d_depth_patch).
            depth_centroids: (B, N_pcl, 3) metric XYZ of PCL token centroids.
            camera_K: (B, 3, 3) intrinsics at image_size resolution. Required
                when use_pointpainting is True; ignored otherwise.

        Returns (B, N_total, d_model) fused token sequence.
        """
        if self.use_2d_point:
            point_token = self.pos_embed_2d(point.unsqueeze(1))
        else:
            point_token = self.pos_embed_3d(point.unsqueeze(1))
        point_token = self.point_proj(point_token)

        if self.use_pointpainting:
            if camera_K is None:
                raise ValueError("camera_K required when use_pointpainting=True")
            painted = self._paint(rgb_patches, depth_centroids, camera_K)
            fused = torch.cat([painted, depth_patches], dim=-1)
            x = self.painting_proj(fused)
            x = x + self.pos_embed_3d(depth_centroids)
        elif self.use_rgb and self.use_depth:
            x_rgb = self.patch_proj(rgb_patches) + self.rgb_pos_embed
            x_depth = self.depth_patch_proj(depth_patches) + self.pos_embed_3d(
                depth_centroids
            )
            x_rgb = x_rgb + self.modality_embed[0]
            x_depth = x_depth + self.modality_embed[1]
            x = torch.cat([x_rgb, x_depth], dim=1)
        elif self.use_rgb:
            x = self.patch_proj(rgb_patches) + self.rgb_pos_embed
        else:
            x = self.depth_patch_proj(depth_patches) + self.pos_embed_3d(
                depth_centroids
            )

        x = self.point_cross_attn(x, context=point_token)

        x = self.transformer(x)
        return x
