"""
Loom Diffusion Transformer (DiT)
Spacetime patch-based transformer for latent video diffusion.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Text conditioning via pooled text embedding."""

    def __init__(self, text_embed_dim: int, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, text_embed: torch.Tensor) -> torch.Tensor:
        return self.mlp(text_embed)


class RoPE3D(nn.Module):
    """3D Rotary Position Embeddings for space-time."""

    def __init__(self, dim: int, max_t: int = 128, max_h: int = 64, max_w: int = 64):
        super().__init__()
        self.dim = dim
        # Precompute frequency grids
        self.max_t = max_t
        self.max_h = max_h
        self.max_w = max_w

    def get_rotary_embedding(self, t: int, h: int, w: int, device: torch.device):
        """Generate RoPE for a (t, h, w) grid. Returns complex tensor."""
        # Create position grids
        pos_t = torch.arange(t, device=device).view(t, 1, 1, 1).expand(t, h, w, 1)
        pos_h = torch.arange(h, device=device).view(1, h, 1, 1).expand(t, h, w, 1)
        pos_w = torch.arange(w, device=device).view(1, 1, w, 1).expand(t, h, w, 1)

        # Frequencies for each dimension
        dim_per_axis = self.dim // 3
        freqs_t = torch.exp(
            -math.log(10000)
            * torch.arange(0, dim_per_axis, 2, device=device).float()
            / dim_per_axis
        )
        freqs_h = torch.exp(
            -math.log(10000)
            * torch.arange(0, dim_per_axis, 2, device=device).float()
            / dim_per_axis
        )
        freqs_w = torch.exp(
            -math.log(10000)
            * torch.arange(0, dim_per_axis, 2, device=device).float()
            / dim_per_axis
        )

        # Apply to positions
        angles_t = pos_t * freqs_t.view(1, 1, 1, -1)
        angles_h = pos_h * freqs_h.view(1, 1, 1, -1)
        angles_w = pos_w * freqs_w.view(1, 1, 1, -1)

        # Interleave: [t0, h0, w0, t1, h1, w1, ...]
        emb = torch.zeros(t, h, w, self.dim, device=device)
        for i in range(dim_per_axis // 2):
            emb[..., 3 * i] = angles_t[..., i].sin()
            emb[..., 3 * i + 1] = angles_h[..., i].sin()
            emb[..., 3 * i + 2] = angles_w[..., i].sin()
            if 3 * (dim_per_axis // 2) + 3 * i + 2 < self.dim:
                emb[..., 3 * (dim_per_axis // 2) + 3 * i] = angles_t[..., i].cos()
                emb[..., 3 * (dim_per_axis // 2) + 3 * i + 1] = angles_h[..., i].cos()
                emb[..., 3 * (dim_per_axis // 2) + 3 * i + 2] = angles_w[..., i].cos()

        return emb

    def apply_rotary(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to attention queries/keys. x: (B, H, N, D), rope: (N, D)"""
        # x * cos + rotate(x) * sin
        x1, x2 = x[..., ::2], x[..., 1::2]
        rope_cos = rope.cos().unsqueeze(0).unsqueeze(1)
        rope_sin = rope.sin().unsqueeze(0).unsqueeze(1)
        y1 = x1 * rope_cos[..., ::2] - x2 * rope_sin[..., ::2]
        y2 = x1 * rope_sin[..., ::2] + x2 * rope_cos[..., ::2]
        y = torch.stack([y1, y2], dim=-1).flatten(-2)
        return y


class DiTBlock(nn.Module):
    """DiT block with adaptive layer norm, self-attention, and cross-attention to text."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 768,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        # AdaLN-Zero modulation: 6 groups for each sub-layer
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

        # Cross-attn modulation
        self.cross_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        text: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, N, D), c: (B, D) conditioning, text: (B, L, D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        # Self-attention with AdaLN
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * h

        # Cross-attention to text
        shift_cross, scale_cross = self.cross_adaLN_modulation(c).chunk(2, dim=1)
        h = modulate(self.norm2(x), shift_cross, scale_cross)
        h, _ = self.cross_attn(h, text, text, need_weights=False)
        x = x + h

        # MLP with AdaLN
        h = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class LoomDiT(nn.Module):
    """
    Diffusion Transformer for latent video generation.
    Input: noisy latent video patches + timestep + text embedding
    Output: predicted noise
    """

    def __init__(
        self,
        latent_dim: int = 16,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        hidden_size: int = 768,
        depth: int = 16,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 768,
        max_t: int = 128,
        max_h: int = 64,
        max_w: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.patch_dim = latent_dim * patch_size[0] * patch_size[1] * patch_size[2]

        # Patch embedding
        self.patch_embed = nn.Linear(self.patch_dim, hidden_size)

        # Position embedding (learned, simpler than RoPE for now)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_t * max_h * max_w, hidden_size)
        )
        nn.init.normal_(self.pos_embed, std=0.02)

        # Timestep and text embedders
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.text_embedder = LabelEmbedder(text_embed_dim, hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio, text_embed_dim)
                for _ in range(depth)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.head = nn.Linear(hidden_size, self.patch_dim)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.cross_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.cross_adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.head.weight, 0)
        nn.init.constant_(self.head.bias, 0)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W)
        return: (B, N, patch_dim) where N = num_patches
        """
        B, C, T, H, W = x.shape
        pt, ph, pw = self.patch_size
        assert T % pt == 0 and H % ph == 0 and W % pw == 0

        x = x.view(
            B,
            C,
            T // pt,
            pt,
            H // ph,
            ph,
            W // pw,
            pw,
        )
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.view(B, -1, self.patch_dim)
        return x

    def unpatchify(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """
        x: (B, N, patch_dim)
        return: (B, C, T, H, W)
        """
        B, N, _ = x.shape
        pt, ph, pw = self.patch_size
        C = self.latent_dim

        x = x.view(B, T // pt, H // ph, W // pw, C, pt, ph, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.view(B, C, T, H, W)
        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (B, C, T, H, W) noisy latent
        t: (B,) timestep
        text_embed: (B, L, D) text embeddings
        """
        B, C, T, H, W = x.shape

        # Patchify
        x = self.patchify(x)  # (B, N, patch_dim)
        x = self.patch_embed(x)  # (B, N, hidden_size)

        # Add positional embedding
        num_patches = x.shape[1]
        x = x + self.pos_embed[:, :num_patches, :]

        # Conditioning
        t_emb = self.t_embedder(t)  # (B, hidden_size)
        text_emb = self.text_embedder(text_embed.mean(dim=1))  # (B, hidden_size)
        c = t_emb + text_emb  # (B, hidden_size)

        # Causal attention mask for time
        # Build a mask where patches can only attend to same or earlier time positions
        pt, ph, pw = self.patch_size
        num_t = T // pt
        num_h = H // ph
        num_w = W // pw

        # Each patch index i corresponds to (t_i, h_i, w_i)
        # We want mask[i, j] = True if t_i >= t_j (causal in time, full in space)
        # For simplicity, use full attention for now (works for short clips)
        attn_mask = None

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, text_embed, attn_mask=attn_mask)

        # Final layer with AdaLN
        shift, scale = self.final_adaLN(c).chunk(2, dim=1)
        x = modulate(self.final_norm(x), shift, scale)
        x = self.head(x)

        # Unpatchify
        x = self.unpatchify(x, T, H, W)
        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
