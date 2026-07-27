#!/usr/bin/env python3
"""
Train Loom Diffusion Transformer
Learns to denoise spacetime latent patches conditioned on text.
Usage:
    python train_diffusion.py --data_dir data/videos --vae_path checkpoints/vae/vae_final.pt --output_dir checkpoints/dit
"""
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from loom_vae_3d import LoomVAE3D
from loom_dit import LoomDiT
from loom_pipeline import TextEncoder


class LatentVideoDataset(Dataset):
    """Dataset that precomputes VAE latents for efficient training."""
    def __init__(
        self,
        data_dir: str,
        vae: LoomVAE3D,
        text_encoder: TextEncoder,
        clip_frames: int = 16,
        resolution: int = 256,
        device: torch.device = torch.device("cuda"),
        cache_dir: str = "cache/latents",
    ):
        self.data_dir = Path(data_dir)
        self.clip_frames = clip_frames
        self.resolution = resolution
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Find videos
        self.video_files = []
        for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
            self.video_files.extend(list(self.data_dir.rglob(ext)))

        print(f"Found {len(self.video_files)} videos")

        # Build clips
        self.clips = []
        for vf in self.video_files:
            cap = cv2.VideoCapture(str(vf))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if total_frames >= clip_frames:
                # Extract multiple clips
                for start in range(0, total_frames - clip_frames, clip_frames):
                    self.clips.append((vf, start, fps))

        print(f"Total clips: {len(self.clips)}")

        # Precompute latents if not cached
        self.latent_files = []
        vae.eval()

        for i, (vf, start, fps) in enumerate(tqdm(self.clips, desc="Precomputing latents")):
            cache_path = self.cache_dir / f"{vf.stem}_{start}.pt"

            if cache_path.exists():
                self.latent_files.append((cache_path, vf.name))
                continue

            # Load video clip
            cap = cv2.VideoCapture(str(vf))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            for _ in range(clip_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (resolution, resolution))
                frames.append(frame)
            cap.release()

            if len(frames) < clip_frames:
                continue

            # Convert to tensor
            video = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0
            video = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0).to(device)  # (1, C, T, H, W)

            # Encode to latent
            with torch.no_grad():
                latent = vae.encode(video)  # (1, C_latent, T_latent, H_latent, W_latent)

            # Save
            torch.save(latent.cpu(), cache_path)
            self.latent_files.append((cache_path, vf.name))

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_path, video_name = self.latent_files[idx]
        latent = torch.load(latent_path)  # (1, C, T, H, W)
        latent = latent.squeeze(0)  # (C, T, H, W)

        # Use video name as pseudo-caption (in production, use actual captions)
        caption = video_name.replace("_", " ").replace(".mp4", "").replace(".avi", "")

        return latent, caption


def collate_fn(batch):
    """Collate latents and captions."""
    latents = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]
    return latents, captions


class DiffusionTrainer:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {self.device}")

        # Load VAE (frozen)
        self.vae = LoomVAE3D(latent_dim=args.latent_dim).to(self.device)
        self.vae.load_state_dict(torch.load(args.vae_path, map_location=self.device))
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        print(f"Loaded VAE from {args.vae_path}")

        # Text encoder
        self.text_encoder = TextEncoder(embed_dim=args.text_embed_dim)

        # DiT model
        self.model = LoomDiT(
            latent_dim=args.latent_dim,
            patch_size=(1, 2, 2),
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            text_embed_dim=args.text_embed_dim,
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"DiT parameters: {total_params / 1e6:.1f}M")

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.05)

        # Noise schedule
        self.num_train_timesteps = 1000
        self.betas = torch.linspace(1e-4, 2e-2, self.num_train_timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, latents, noise, timesteps):
        """Add noise to latents at given timesteps."""
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        return sqrt_alpha * latents + sqrt_one_minus_alpha * noise

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for latents, captions in pbar:
            latents = latents.to(self.device)  # (B, C, T, H, W)
            B = latents.shape[0]

            # Encode text
            text_embeds = self.text_encoder.encode(captions).to(self.device)  # (B, L, D)

            # Sample random timesteps
            timesteps = torch.randint(0, self.num_train_timesteps, (B,), device=self.device).long()

            # Sample noise
            noise = torch.randn_like(latents)

            # Add noise
            noisy_latents = self.add_noise(latents, noise, timesteps)

            # Predict noise
            noise_pred = self.model(noisy_latents, timesteps, text_embeds)

            # MSE loss
            loss = F.mse_loss(noise_pred, noise)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    def train(self, args):
        # Dataset
        dataset = LatentVideoDataset(
            args.data_dir,
            self.vae,
            self.text_encoder,
            clip_frames=args.clip_frames,
            resolution=args.resolution,
            device=self.device,
            cache_dir=args.cache_dir,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid issues with CUDA in multiprocessing
            collate_fn=collate_fn,
            drop_last=True,
        )

        # Training loop
        for epoch in range(args.epochs):
            self.train_epoch(dataloader, epoch)

            # Save checkpoint
            if (epoch + 1) % args.save_every == 0:
                save_path = Path(args.output_dir) / f"dit_epoch_{epoch+1}.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(f"\nSaved checkpoint to {save_path}")

        # Final save
        final_path = Path(args.output_dir) / "dit_final.pt"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), final_path)
        print(f"Training complete. Saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Loom Diffusion Transformer")
    parser.add_argument("--data_dir", required=True, help="Directory with training videos")
    parser.add_argument("--vae_path", required=True, help="Path to trained VAE checkpoint")
    parser.add_argument("--output_dir", default="checkpoints/dit", help="Checkpoint output directory")
    parser.add_argument("--cache_dir", default="cache/latents", help="Latent cache directory")
    parser.add_argument("--latent_dim", type=int, default=16, help="VAE latent dimension")
    parser.add_argument("--hidden_size", type=int, default=768, help="DiT hidden dimension")
    parser.add_argument("--depth", type=int, default=16, help="DiT depth")
    parser.add_argument("--num_heads", type=int, default=12, help="DiT attention heads")
    parser.add_argument("--text_embed_dim", type=int, default=768, help="Text embedding dimension")
    parser.add_argument("--resolution", type=int, default=256, help="Training resolution")
    parser.add_argument("--clip_frames", type=int, default=16, help="Frames per clip")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    args = parser.parse_args()

    trainer = DiffusionTrainer(args)
    trainer.train(args)


if __name__ == "__main__":
    main()
