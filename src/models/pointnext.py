"""PointNeXt point cloud encoder.

Slim in-repo reimpl using `torch_cluster` for FPS + kNN. Configurable
between PointNet++ U-Net (blocks=(0,0,0,0)) and PointNeXt-B/L/XL by
appending `InvResMLP` blocks after each SA stage. Relative positions
are normalized by query radius per the paper (Eq. 2). Reads features
at the FP3 stage (256 centroids) to match the existing fusion contract.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch_cluster


def _shared_mlp(dims: list[int]) -> nn.Sequential:
    """Per-channel Linear → LayerNorm → GeLU stack."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


def _fps_indices(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """Batched FPS: (B, N, 3) → (B, K) local indices."""
    B, N, _ = xyz.shape
    batch = torch.arange(B, device=xyz.device).repeat_interleave(N)
    ratio = k / N
    flat_idx = torch_cluster.fps(
        xyz.reshape(-1, 3), batch=batch, ratio=ratio, random_start=True
    )
    assert flat_idx.numel() == B * k, (
        f"FPS expected B*k={B * k} indices, got {flat_idx.numel()}"
    )
    flat_idx = flat_idx.reshape(B, k)
    offsets = torch.arange(B, device=xyz.device).unsqueeze(1) * N
    return flat_idx - offsets


def _knn_indices(query: torch.Tensor, ref: torch.Tensor, k: int) -> torch.Tensor:
    """Batched kNN: (B, M, 3) query in (B, N, 3) ref → (B, M, k) local indices."""
    B, M, _ = query.shape
    N = ref.shape[1]
    batch_q = torch.arange(B, device=query.device).repeat_interleave(M)
    batch_r = torch.arange(B, device=query.device).repeat_interleave(N)
    edge_index = torch_cluster.knn(
        ref.reshape(-1, 3),
        query.reshape(-1, 3),
        k=k,
        batch_x=batch_r,
        batch_y=batch_q,
    )
    # edge_index: (2, B*M*k). row[0]=flat query idx, row[1]=flat ref idx.
    # Sort by query idx so neighbors are grouped per centroid.
    order = edge_index[0].argsort(stable=True)
    flat_nbr = edge_index[1][order].reshape(B * M, k)
    offsets = (torch.arange(B * M, device=query.device) // M) * N
    return (flat_nbr - offsets.unsqueeze(1)).reshape(B, M, k)


def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather (B, N, C) by indices of shape (B, ...). Output: (B, ..., C)."""
    B, N, C = x.shape
    idx_flat = idx.reshape(B, -1)
    out = x.gather(1, idx_flat.unsqueeze(-1).expand(-1, -1, C))
    return out.reshape(*idx.shape, C)


class SetAbstraction(nn.Module):
    """FPS + kNN ball-grouping (with radius mask) + edge MLP + max-pool."""

    def __init__(
        self,
        n_centroids: int,
        radius: float,
        k_neighbors: int,
        in_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.n_centroids = n_centroids
        self.radius = radius
        self.k = k_neighbors
        self.mlp = _shared_mlp([3 + in_dim, out_dim])

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor):
        idx = _fps_indices(xyz, self.n_centroids)
        new_xyz = _gather(xyz, idx)
        nbr_idx = _knn_indices(new_xyz, xyz, self.k)
        nbr_xyz = _gather(xyz, nbr_idx)
        nbr_feat = _gather(feat, nbr_idx)
        rel_xyz = nbr_xyz - new_xyz.unsqueeze(2)
        # PointNeXt Eq. 2: divide relative position by query radius before
        # the MLP. Centroids `new_xyz` returned below remain metric meters.
        rel_xyz_norm = rel_xyz / self.radius
        edge_feat = torch.cat([rel_xyz_norm, nbr_feat], dim=-1)
        edge_feat = self.mlp(edge_feat)
        dist = rel_xyz.norm(dim=-1)
        valid = (dist < self.radius).unsqueeze(-1)
        # Ball-query fallback: if a centroid has zero neighbors in radius,
        # use its kNN result (closest k overall) instead of zero features.
        has_any = valid.any(dim=2, keepdim=True)
        keep = valid | ~has_any
        edge_feat = edge_feat.masked_fill(~keep, float("-inf"))
        new_feat = edge_feat.max(dim=2).values
        return new_xyz, new_feat


class LocalAggregation(nn.Module):
    """Same-resolution kNN + 1-layer edge MLP + max-pool.

    Used inside InvResMLP. Operates at a fixed xyz set (no downsampling),
    so query == ref. Relative positions are normalized by `radius`.
    """

    def __init__(self, dim: int, radius: float, k_neighbors: int):
        super().__init__()
        self.radius = radius
        self.k = k_neighbors
        self.mlp = _shared_mlp([3 + dim, dim])

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        nbr_idx = _knn_indices(xyz, xyz, self.k)
        nbr_xyz = _gather(xyz, nbr_idx)
        nbr_feat = _gather(feat, nbr_idx)
        rel_xyz = nbr_xyz - xyz.unsqueeze(2)
        rel_xyz_norm = rel_xyz / self.radius
        edge_feat = torch.cat([rel_xyz_norm, nbr_feat], dim=-1)
        edge_feat = self.mlp(edge_feat)
        dist = rel_xyz.norm(dim=-1)
        valid = (dist < self.radius).unsqueeze(-1)
        has_any = valid.any(dim=2, keepdim=True)
        keep = valid | ~has_any
        edge_feat = edge_feat.masked_fill(~keep, float("-inf"))
        return edge_feat.max(dim=2).values


class InvResMLP(nn.Module):
    """Inverted-residual MLP block (PointNeXt Sec. 3.2.2).

    LocalAggregation → 2-layer pointwise MLP with expansion ratio
    (inverted bottleneck) → residual add → activation. Operates at a
    fixed xyz set, so centroids passed in are returned untouched.
    """

    def __init__(
        self,
        dim: int,
        radius: float,
        k_neighbors: int,
        expansion: int = 4,
    ):
        super().__init__()
        self.local_aggr = LocalAggregation(dim, radius, k_neighbors)
        mid = dim * expansion
        self.pwconv = nn.Sequential(
            nn.Linear(dim, mid),
            nn.LayerNorm(mid),
            nn.GELU(),
            nn.Linear(mid, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        identity = feat
        f = self.local_aggr(xyz, feat)
        f = self.pwconv(f)
        return self.act(f + identity)


class FeaturePropagation(nn.Module):
    """3-NN inverse-distance interpolation + skip concat + MLP."""

    def __init__(self, in_dim_up: int, in_dim_skip: int, out_dim: int):
        super().__init__()
        self.mlp = _shared_mlp([in_dim_up + in_dim_skip, out_dim, out_dim])

    def forward(
        self,
        xyz_dst: torch.Tensor,
        xyz_src: torch.Tensor,
        feat_src: torch.Tensor,
        feat_skip: torch.Tensor,
    ):
        nbr_idx = _knn_indices(xyz_dst, xyz_src, k=3)
        nbr_xyz = _gather(xyz_src, nbr_idx)
        nbr_feat = _gather(feat_src, nbr_idx)
        dist = (nbr_xyz - xyz_dst.unsqueeze(2)).norm(dim=-1)
        weights = 1.0 / (dist + 1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        interp = (nbr_feat * weights.unsqueeze(-1)).sum(dim=2)
        combined = torch.cat([interp, feat_skip], dim=-1)
        return self.mlp(combined)


class PointNeXt(nn.Module):
    """Configurable PointNeXt encoder.

    blocks=(0,0,0,0): PointNet++ U-Net with paper-Eq.2 normalization (≈ -S).
    blocks=(1,2,1,1): PointNeXt-B.
    blocks=(2,4,2,2): PointNeXt-L.

    Output: 256 tokens × (8*c) features at the FP3 stage, with metric XYZ
    centroids returned alongside for 3D positional encoding in fusion.
    """

    def __init__(
        self,
        c: int = 64,
        sa_radii: tuple[float, ...] = (0.025, 0.05, 0.10, 0.20),
        blocks: tuple[int, ...] = (1, 2, 1, 1),
        radius_scaling: float = 2.0,
        expansion: int = 4,
        use_rgb: bool = True,
    ):
        super().__init__()
        assert len(sa_radii) == 4 and len(blocks) == 4
        self.use_rgb = use_rgb
        self.stem = _shared_mlp([6 if use_rgb else 3, c])

        r1, r2, r3, r4 = sa_radii
        ks = (32, 32, 32, 16)
        n_centroids = (1024, 256, 64, 16)
        dims = (2 * c, 4 * c, 8 * c, 16 * c)

        self.sa1 = SetAbstraction(n_centroids[0], r1, ks[0], c, dims[0])
        self.sa2 = SetAbstraction(n_centroids[1], r2, ks[1], dims[0], dims[1])
        self.sa3 = SetAbstraction(n_centroids[2], r3, ks[2], dims[1], dims[2])
        self.sa4 = SetAbstraction(n_centroids[3], r4, ks[3], dims[2], dims[3])

        # InvResMLP radii scaled (paper default x2) since SA already
        # downsampled — wider receptive field at lower resolution.
        inv_radii = tuple(r * radius_scaling for r in sa_radii)
        self.invres = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        InvResMLP(
                            dims[i],
                            inv_radii[i],
                            min(ks[i], n_centroids[i]),
                            expansion,
                        )
                        for _ in range(blocks[i])
                    ]
                )
                for i in range(4)
            ]
        )

        self.fp4 = FeaturePropagation(16 * c, 8 * c, 8 * c)
        self.fp3 = FeaturePropagation(8 * c, 4 * c, 8 * c)

        self.out_dim = 8 * c

    def forward(self, xyz: torch.Tensor, rgb_pcl: Optional[torch.Tensor] = None):
        """Encode a point cloud into FP3-stage feature tokens and centroids.

        Args:
            xyz: (B, N, 3) metric meters.
            rgb_pcl: (B, N, 3) per-point RGB in [0, 1]. Required iff use_rgb.

        Returns:
            features: (B, 256, out_dim).
            centroids: (B, 256, 3) metric XYZ of output tokens.
        """
        if self.use_rgb:
            assert rgb_pcl is not None, "use_rgb=True requires rgb_pcl"
            feat0 = self.stem(torch.cat([xyz, rgb_pcl], dim=-1))
        else:
            feat0 = self.stem(xyz)
        xyz1, feat1 = self.sa1(xyz, feat0)
        for blk in self.invres[0]:
            feat1 = blk(xyz1, feat1)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        for blk in self.invres[1]:
            feat2 = blk(xyz2, feat2)
        xyz3, feat3 = self.sa3(xyz2, feat2)
        for blk in self.invres[2]:
            feat3 = blk(xyz3, feat3)
        xyz4, feat4 = self.sa4(xyz3, feat3)
        for blk in self.invres[3]:
            feat4 = blk(xyz4, feat4)
        feat3_up = self.fp4(xyz3, xyz4, feat4, feat3)
        feat2_up = self.fp3(xyz2, xyz3, feat3_up, feat2)
        return feat2_up, xyz2
