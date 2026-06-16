"""Rectified flow matching model for MANO grasp prediction."""

import math

import torch
import torch.nn as nn

from .transformer import AdaLNCrossAttnBlock


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TokenPerGroupDiT(nn.Module):
    """DiT denoiser with separate tokens for translation, wrist, and fingers."""

    def __init__(
        self,
        d_cond: int = 768,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        norm_stats: dict = None,
    ):
        super().__init__()
        self.d_model = d_model

        # Per-group input projections
        self.proj_translation = nn.Linear(3, d_model, bias=False)
        self.proj_wrist = nn.Linear(6, d_model, bias=False)
        self.proj_fingers = nn.Linear(90, d_model, bias=False)

        # Learnable token-type embeddings
        self.token_type_emb = nn.Parameter(torch.randn(3, d_model) * 0.02)

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model // 4),
            nn.Linear(d_model // 4, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False),
        )

        # Condition projection
        self.cond_proj = nn.Linear(d_cond, d_model, bias=False)

        self.blocks = nn.ModuleList(
            [AdaLNCrossAttnBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # Per-group output projections
        self.out_translation = nn.Linear(d_model, 3, bias=False)
        self.out_wrist = nn.Linear(d_model, 6, bias=False)
        self.out_fingers = nn.Linear(d_model, 90, bias=False)

        # Final norm + modulation
        self.final_norm = nn.RMSNorm(d_model, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )

        # Zero-init outputs
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        for proj in [self.out_translation, self.out_wrist, self.out_fingers]:
            nn.init.zeros_(proj.weight)

        # Register normalization stats
        if norm_stats is not None:
            self._register_norm_stats(norm_stats)
        else:
            self.register_buffer("trans_mean", None)

    def _register_norm_stats(self, norm_stats: dict):
        self.register_buffer(
            "trans_mean", torch.tensor(norm_stats["translation"]["mean"])
        )
        self.register_buffer(
            "trans_std", torch.tensor(norm_stats["translation"]["std"])
        )
        self.register_buffer(
            "wrist_mean", torch.tensor(norm_stats["wrist_rot"]["mean"])
        )
        self.register_buffer("wrist_std", torch.tensor(norm_stats["wrist_rot"]["std"]))
        self.register_buffer(
            "finger_mean", torch.tensor(norm_stats["finger_rot"]["mean"])
        )
        self.register_buffer(
            "finger_std", torch.tensor(norm_stats["finger_rot"]["std"])
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize 99-dim MANO params per group."""
        trans = (x[:, :3] - self.trans_mean) / self.trans_std
        wrist = (x[:, 3:9] - self.wrist_mean) / self.wrist_std
        fingers = (x[:, 9:99] - self.finger_mean) / self.finger_std
        return torch.cat([trans, wrist, fingers], dim=-1)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize 99-dim MANO params per group."""
        trans = x[:, :3] * self.trans_std + self.trans_mean
        wrist = x[:, 3:9] * self.wrist_std + self.wrist_mean
        fingers = x[:, 9:99] * self.finger_std + self.finger_mean
        return torch.cat([trans, wrist, fingers], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity from noisy input (in normalized space).

        Args:
            x: (B, 99) noisy MANO params in normalized space.
            t: (B,) continuous timestep in [0, 1].
            cond: (B, N_patches, d_cond) patch sequence for cross-attention.
        """
        t_emb = self.time_mlp(t * 1000)  # (B, d_model)

        # Split and project to tokens
        trans_tok = self.proj_translation(x[:, :3])
        wrist_tok = self.proj_wrist(x[:, 3:9])
        finger_tok = self.proj_fingers(x[:, 9:99])

        # Stack tokens: (B, 3, d_model)
        h = torch.stack([trans_tok, wrist_tok, finger_tok], dim=1)
        h = h + self.token_type_emb.unsqueeze(0)

        context = self.cond_proj(cond)  # (B, N_patches, d_model)
        c = t_emb
        for block in self.blocks:
            h = block(h, c=c, context=context)

        # Final modulation
        mod = self.final_modulation(c)
        scale, shift = mod.chunk(2, dim=-1)
        h = self.final_norm(h)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # Per-group output projections
        out_trans = self.out_translation(h[:, 0])
        out_wrist = self.out_wrist(h[:, 1])
        out_fingers = self.out_fingers(h[:, 2])

        return torch.cat([out_trans, out_wrist, out_fingers], dim=-1)


class GraspFlowMatching(nn.Module):
    """Rectified flow matching model for MANO grasp prediction.

    Linear interpolation: x_t = (1-t) * x_0 + t * eps
    Velocity target: v = eps - x_0
    ODE integration: x_{t-dt} = x_t - v * dt (Euler, from t=1 to t=0)
    """

    def __init__(
        self,
        d_mano: int = 99,
        d_cond: int = 768,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        norm_stats: dict = None,
        sampling_steps: int = 50,
    ):
        super().__init__()
        self.d_mano = d_mano
        self.sampling_steps = sampling_steps

        if norm_stats is None:
            raise ValueError("norm_stats required")
        self.denoise_fn = TokenPerGroupDiT(
            d_cond=d_cond,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            norm_stats=norm_stats,
        )

    def recover_x0(self, output):
        """Recover x0 from flow matching training output.

        x_t = (1-t)*x0 + t*eps, v = eps - x0
        => x0 = x_t - t*v
        """
        pred_v = output["pred"]
        t = output["t"].unsqueeze(-1)  # (B, 1)
        x_noisy = output["x_noisy"]
        pred_x0 = x_noisy - t * pred_v
        return self.denoise_fn.denormalize(pred_x0)

    @torch.no_grad()
    def sample(self, cond, steps=None):
        """Euler ODE sampling: integrate from t=1 (noise) to t=0 (data)."""
        steps = steps or self.sampling_steps
        b = cond.shape[0]
        device = cond.device
        x = torch.randn(b, self.d_mano, device=device)
        dt = 1.0 / steps

        for i in reversed(range(steps)):
            t = torch.full((b,), (i + 1) * dt, device=device)
            v = self.denoise_fn(x, t, cond)
            x = x - v * dt

        x = self.denoise_fn.denormalize(x)
        return x

    def sample_t(self, b: int, device) -> torch.Tensor:
        """Sample training timesteps uniformly in [0, 1]."""
        return torch.rand(b, device=device)

    def forward(self, x_start, cond):
        """Training forward pass with rectified flow matching.

        Args:
            x_start: (B, d_mano) ground truth MANO params.
            cond: (B, N_patches, d_cond) scene condition.
        """
        b = x_start.shape[0]
        t = self.sample_t(b, x_start.device)

        x_norm = self.denoise_fn.normalize(x_start)
        eps = torch.randn_like(x_norm)
        t_ = t.unsqueeze(-1)  # (B, 1)
        x_t = (1 - t_) * x_norm + t_ * eps
        target = eps - x_norm  # velocity
        pred = self.denoise_fn(x_t, t, cond)
        return {"pred": pred, "target": target, "t": t, "x_noisy": x_t}
