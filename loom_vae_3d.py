"""
Loom 3D Causal VAE
Compresses video into spacetime latent patches.
Causal in time: frame t only sees frames <= t.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class CausalConv3d(nn.Module):
    """3D convolution with causal padding in the temporal dimension."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        # Causal padding: pad only at the start of time
        self.time_pad = kernel_size[0] - 1
        # Spatial padding handled by conv3d padding arg
        self.spatial_pad = (kernel_size[1] // 2, kernel_size[2] // 2)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, self.spatial_pad[0], self.spatial_pad[1]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        # Pad temporal dimension causally
        x = F.pad(x, (0, 0, 0, 0, self.time_pad, 0))
        return self.conv(x)


class ResBlock3D(nn.Module):
    """Residual block with GroupNorm and CausalConv3d."""

    def __init__(self, channels: int, kernel_size: Tuple[int, int, int] = (3, 3, 3)):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = CausalConv3d(channels, channels, kernel_size)
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = CausalConv3d(channels, channels, kernel_size)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class Attention3D(nn.Module):
    """Factorized space-time attention for efficiency."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Spatial attention: attend within each frame
        q_s = q.view(B, self.num_heads, C // self.num_heads, T, H * W)
        k_s = k.view(B, self.num_heads, C // self.num_heads, T, H * W)
        v_s = v.view(B, self.num_heads, C // self.num_heads, T, H * W)

        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(
            torch.einsum("b h t i, b h t j -> b h t i j", q_s, k_s) * scale, dim=-1
        )
        out_s = torch.einsum("b h t i j, b h t j -> b h t i", attn, v_s)
        out_s = out_s.view(B, C, T, H, W)

        # Temporal attention: attend across time at each spatial position
        q_t = q.view(B, self.num_heads, C // self.num_heads, H * W, T)
        k_t = k.view(B, self.num_heads, C // self.num_heads, H * W, T)
        v_t = v.view(B, self.num_heads, C // self.num_heads, H * W, T)

        # Causal mask for temporal attention
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, 1, T, T)
        attn_t = torch.einsum("b h s i, b h s j -> b h s i j", q_t, k_t) * scale
        attn_t = attn_t.masked_fill(causal_mask == 0, float("-inf"))
        attn_t = torch.softmax(attn_t, dim=-1)
        out_t = torch.einsum("b h s i j, b h s j -> b h s i", attn_t, v_t)
        out_t = out_t.view(B, C, T, H, W)

        out = out_s + out_t
        out = self.proj(out)
        return x + out


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        # Input: (B, 3, T, H, W)
        self.conv_in = CausalConv3d(3, 128, (3, 3, 3), stride=(1, 1, 1))

        # Down blocks
        self.down1 = nn.Sequential(
            ResBlock3D(128),
            CausalConv3d(128, 128, (3, 3, 3), stride=(1, 2, 2)),  # /2 spatial
        )
        self.down2 = nn.Sequential(
            ResBlock3D(128),
            CausalConv3d(
                128, 256, (3, 3, 3), stride=(2, 2, 2)
            ),  # /2 temporal, /2 spatial
        )
        self.down3 = nn.Sequential(
            ResBlock3D(256),
            CausalConv3d(
                256, 512, (3, 3, 3), stride=(2, 2, 2)
            ),  # /2 temporal, /2 spatial
        )
        self.down4 = nn.Sequential(
            ResBlock3D(512),
            CausalConv3d(512, 512, (3, 3, 3), stride=(1, 2, 2)),  # /2 spatial
        )

        # Mid
        self.mid = nn.Sequential(
            ResBlock3D(512),
            Attention3D(512),
            ResBlock3D(512),
        )

        # Out
        self.norm_out = nn.GroupNorm(32, 512)
        self.conv_out = CausalConv3d(512, latent_dim * 2, (3, 3, 3))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 3, T, H, W)
        h = self.conv_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down4(h)
        h = self.mid(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.conv_in = CausalConv3d(latent_dim, 512, (3, 3, 3))

        # Mid
        self.mid = nn.Sequential(
            ResBlock3D(512),
            Attention3D(512),
            ResBlock3D(512),
        )

        # Up blocks
        self.up1 = nn.Sequential(
            ResBlock3D(512),
            CausalConv3d(512, 512, (3, 3, 3), stride=(1, 1, 1)),
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),  # *2 spatial
        )
        self.up2 = nn.Sequential(
            ResBlock3D(512),
            CausalConv3d(512, 256, (3, 3, 3), stride=(1, 1, 1)),
            nn.Upsample(
                scale_factor=(2, 2, 2), mode="nearest"
            ),  # *2 temporal, *2 spatial
        )
        self.up3 = nn.Sequential(
            ResBlock3D(256),
            CausalConv3d(256, 128, (3, 3, 3), stride=(1, 1, 1)),
            nn.Upsample(
                scale_factor=(2, 2, 2), mode="nearest"
            ),  # *2 temporal, *2 spatial
        )
        self.up4 = nn.Sequential(
            ResBlock3D(128),
            CausalConv3d(128, 128, (3, 3, 3), stride=(1, 1, 1)),
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),  # *2 spatial
        )

        self.norm_out = nn.GroupNorm(32, 128)
        self.conv_out = CausalConv3d(128, 3, (3, 3, 3))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid(h)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


class LoomVAE3D(nn.Module):
    """Full 3D causal VAE for video compression."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim
        self.scaling_factor = 0.18215  # SD-style scaling

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode video to latent. x: (B, 3, T, H, W)"""
        mean, logvar = self.encoder(x)
        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z * self.scaling_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to video. z: (B, C, T, H, W)"""
        z = z / self.scaling_factor
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        recon = self.decoder(z)
        return recon, mean, logvar

    def get_latent_shape(
        self, video_shape: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """Given (B, C, T, H, W), return latent shape."""
        B, C, T, H, W = video_shape
        # Temporal: /4 (two stride-2 downsamples)
        # Spatial: /16 (four stride-2 downsamples)
        return (B, self.latent_dim, T // 4, H // 16, W // 16)
