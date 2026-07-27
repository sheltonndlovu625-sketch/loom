"""
Loom — Latent Video Diffusion Engine
=====================================
A Sora/Veo-style video generator using:
  1. 3D Causal VAE for video compression
  2. Diffusion Transformer (DiT) for latent denoising
  3. Text conditioning via T5/CLIP

This replaces the old non-parametric Loom engine.
"""

import numpy as np
from pathlib import Path
from typing import Optional

from loom_pipeline import LoomVideoPipeline


class LoomEngine:
    """Main synthesis engine. Loads VAE + DiT and generates video from text."""

    def __init__(
        self,
        vae_path: str = "checkpoints/vae/vae_final.pt",
        dit_path: str = "checkpoints/dit/dit_final.pt",
        latent_dim: int = 16,
        dit_hidden_size: int = 768,
        dit_depth: int = 16,
        dit_num_heads: int = 12,
        text_embed_dim: int = 768,
        device: Optional[str] = None,
    ):
        self.pipeline = LoomVideoPipeline(
            vae_path=vae_path,
            dit_path=dit_path,
            latent_dim=latent_dim,
            dit_hidden_size=dit_hidden_size,
            dit_depth=dit_depth,
            dit_num_heads=dit_num_heads,
            text_embed_dim=text_embed_dim,
            device=device,
        )
        print("Loom Engine initialized (Latent Diffusion Mode)")

    def synthesize(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        num_frames: int = 16,
        seed: Optional[int] = None,
        cfg_scale: float = 7.5,
    ) -> np.ndarray:
        """
        Synthesize video from text prompt.
        Returns: (T, H, W, 3) uint8 numpy array
        """
        # Round to valid dimensions (multiples of 16 for VAE)
        width = (width // 16) * 16
        height = (height // 16) * 16
        num_frames = (num_frames // 4) * 4

        return self.pipeline.generate(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            seed=seed,
            cfg_scale=cfg_scale,
        )
