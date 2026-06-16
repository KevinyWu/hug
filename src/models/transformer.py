"""Custom transformer building blocks: Flash Attention (SDPA), RMSNorm, GeLU MLP, QK-norm."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeLUMLP(nn.Module):
    """Standard GeLU feed-forward network (4x expansion)."""

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        hidden = 4 * d_model
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Attention(nn.Module):
    """Multi-head attention with fused QKV, QK-norm, and Flash Attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with RMSNorm + GeLU."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = GeLUMLP(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttention(nn.Module):
    """Cross-attention: Q from one stream, KV from another. QK-norm + Flash Attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        _, M, _ = context.shape
        q = (
            self.q_proj(x)
            .reshape(B, N, self.n_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        kv = self.kv_proj(context).reshape(B, M, 2, self.n_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, H, M, D)
        k, v = kv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        # attn_mask: (B, M) bool, True=valid → expand to (B, 1, 1, M) for SDPA
        if attn_mask is not None:
            attn_mask = attn_mask[:, None, None, :].expand(-1, -1, N, -1)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block: x = x + cross_attn(norm(x), context) + MLP residual."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.norm_ctx = nn.RMSNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = GeLUMLP(d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.cross_attn(
            self.norm1(x), self.norm_ctx(context), attn_mask=attn_mask
        )
        x = x + self.mlp(self.norm2(x))
        return x


class AdaLNCrossAttnBlock(nn.Module):
    """AdaLN self-attn → cross-attn to patches (zero-init gated) → AdaLN FFN."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # Self-attention (timestep modulated)
        self.norm1 = nn.RMSNorm(d_model, elementwise_affine=False)
        self.attn = Attention(d_model, n_heads, dropout=dropout)

        # Cross-attention to patch context
        self.norm_cross = nn.RMSNorm(d_model, elementwise_affine=False)
        self.norm_ctx = nn.RMSNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout=dropout)

        # FFN (timestep modulated)
        self.norm2 = nn.RMSNorm(d_model, elementwise_affine=False)
        self.mlp = GeLUMLP(d_model, dropout=dropout)

        # 7 params: scale1, shift1, gate1, gate_cross, scale2, shift2, gate2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 7 * d_model),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Self-attend, cross-attend to patches, then apply the FFN (all AdaLN-modulated).

        Args:
            x: (B, N_tokens, d_model) MANO tokens.
            c: (B, d_model) timestep embedding.
            context: (B, N_patches, d_model) patch features for cross-attn.
        """
        mod = self.adaLN_modulation(c)  # (B, 7*d_model)
        scale1, shift1, gate1, gate_cross, scale2, shift2, gate2 = mod.chunk(7, dim=-1)

        # Self-attention
        h = self.norm1(x)
        h = h * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        h = self.attn(h)
        x = x + gate1.unsqueeze(1) * h

        # Cross-attention to patches
        h = self.cross_attn(self.norm_cross(x), self.norm_ctx(context))
        x = x + gate_cross.unsqueeze(1) * h

        # FFN
        h = self.norm2(x)
        h = h * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate2.unsqueeze(1) * h

        return x
