"""
Loom Video Diffusion Pipeline
Full inference pipeline: Text -> Latent Diffusion -> VAE Decode -> Video
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path

from loom_vae_3d import LoomVAE3D
from loom_dit import LoomDiT


class TextEncoder:
    """
    Wrapper for text encoding. Tries T5-small first, falls back to CLIP,
    then to a simple learned embedding for testing.
    """

    def __init__(self, embed_dim: int = 768, max_length: int = 77):
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Try T5-small
        try:
            from transformers import T5Tokenizer, T5EncoderModel

            self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
            self.model = (
                T5EncoderModel.from_pretrained("t5-small").to(self.device).eval()
            )
            self.model_type = "t5"
            print("Text encoder: Loaded T5-small")
        except Exception as e:
            print(f"T5-small not available ({e}), trying CLIP...")
            try:
                from transformers import CLIPTokenizer, CLIPTextModel

                self.tokenizer = CLIPTokenizer.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                self.model = (
                    CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
                    .to(self.device)
                    .eval()
                )
                self.model_type = "clip"
                print("Text encoder: Loaded CLIP")
            except Exception as e2:
                print(f"CLIP not available ({e2}), using random embeddings")
                self.model_type = "random"
                self.random_proj = nn.Linear(768, embed_dim).to(self.device)

    def encode(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts to embeddings. Returns (B, L, D)"""
        if self.model_type == "random":
            # Fallback: hash-based random embeddings for testing
            hashes = [hash(p) % 100000 for p in prompts]
            torch.manual_seed(sum(hashes))
            dummy = torch.randn(len(prompts), self.max_length, 768, device=self.device)
            return self.random_proj(dummy)

        inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            if self.model_type == "t5":
                outputs = self.model(**inputs)
                # T5 returns (B, L, D) from last_hidden_state
                embeds = outputs.last_hidden_state
            else:  # clip
                outputs = self.model(**inputs)
                embeds = outputs.last_hidden_state

        # Project to target dim if needed
        if embeds.shape[-1] != self.embed_dim:
            if not hasattr(self, "proj"):
                self.proj = nn.Linear(embeds.shape[-1], self.embed_dim).to(self.device)
            embeds = self.proj(embeds)

        return embeds


class DDIMSampler:
    """DDIM sampler for fast deterministic generation."""

    def __init__(self, num_steps: int = 50):
        self.num_steps = num_steps
        self.timesteps = torch.linspace(999, 0, num_steps, dtype=torch.long)

    def set_noise_schedule(
        self,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        num_train_timesteps: int = 1000,
    ):
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        text_embed: torch.Tensor,
        device: torch.device,
        cfg_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Sample from the diffusion model using DDIM.
        model: LoomDiT
        shape: (B, C, T, H, W) latent shape
        text_embed: (B, L, D)
        """
        B = shape[0]
        # Initialize from noise
        x = torch.randn(shape, device=device, generator=generator)

        # Prepare unconditional embedding for CFG
        if cfg_scale > 1.0:
            uncond_embed = torch.zeros_like(text_embed)
            text_embed = torch.cat([uncond_embed, text_embed], dim=0)
            x = torch.cat([x, x], dim=0)

        self.set_noise_schedule()
        timesteps = self.timesteps.to(device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((x.shape[0],), t, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = model(x, t_batch, text_embed)

            # CFG
            if cfg_scale > 1.0:
                noise_uncond, noise_text = noise_pred.chunk(2, dim=0)
                noise_pred = noise_uncond + cfg_scale * (noise_text - noise_uncond)
                x = x[:B]  # Keep only conditional batch

            # DDIM step
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = (
                self.alphas_cumprod[timesteps[i + 1]]
                if i < len(timesteps) - 1
                else torch.tensor(1.0)
            )

            pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -10, 10)

            x = (
                torch.sqrt(alpha_prev) * pred_x0
                + torch.sqrt(1 - alpha_prev) * noise_pred
            )

        return x


class LoomVideoPipeline:
    """End-to-end video generation pipeline."""

    def __init__(
        self,
        vae_path: str,
        dit_path: str,
        latent_dim: int = 16,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        dit_hidden_size: int = 768,
        dit_depth: int = 16,
        dit_num_heads: int = 12,
        text_embed_dim: int = 768,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Pipeline device: {self.device}")

        # Load VAE
        self.vae = LoomVAE3D(latent_dim=latent_dim).to(self.device)
        if Path(vae_path).exists():
            self.vae.load_state_dict(torch.load(vae_path, map_location=self.device))
            print(f"Loaded VAE from {vae_path}")
        else:
            print(f"VAE checkpoint not found at {vae_path}, using random init")
        self.vae.eval()

        # Load DiT
        self.dit = LoomDiT(
            latent_dim=latent_dim,
            patch_size=patch_size,
            hidden_size=dit_hidden_size,
            depth=dit_depth,
            num_heads=dit_num_heads,
            text_embed_dim=text_embed_dim,
        ).to(self.device)
        if Path(dit_path).exists():
            self.dit.load_state_dict(torch.load(dit_path, map_location=self.device))
            print(f"Loaded DiT from {dit_path}")
        else:
            print(f"DiT checkpoint not found at {dit_path}, using random init")
        print(f"DiT parameters: {self.dit.get_num_params() / 1e6:.1f}M")
        self.dit.eval()

        # Text encoder
        self.text_encoder = TextEncoder(embed_dim=text_embed_dim)

        # Sampler
        self.sampler = DDIMSampler(num_steps=50)

        self.latent_dim = latent_dim

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        num_frames: int = 16,
        fps: float = 24.0,
        seed: Optional[int] = None,
        cfg_scale: float = 7.5,
    ) -> np.ndarray:
        """
        Generate video from text prompt.
        Returns: (T, H, W, 3) uint8 numpy array
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Encode text
        text_embed = self.text_encoder.encode([prompt])  # (1, L, D)

        # Compute latent shape
        # VAE compresses temporal by 4, spatial by 16
        latent_t = num_frames // 4
        latent_h = height // 16
        latent_w = width // 16
        latent_shape = (1, self.latent_dim, latent_t, latent_h, latent_w)

        print(f"Generating latent video: {latent_shape}")
        print(f'Prompt: "{prompt}"')

        # Diffusion sampling
        latent = self.sampler.sample(
            self.dit,
            latent_shape,
            text_embed,
            self.device,
            cfg_scale=cfg_scale,
            generator=generator,
        )

        # VAE decode
        print("Decoding with VAE...")
        video = self.vae.decode(latent)  # (1, 3, T, H, W)
        video = video[0].permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, 3)

        # Normalize to [0, 255]
        video = (video + 1.0) * 127.5
        video = np.clip(video, 0, 255).astype(np.uint8)

        return video
