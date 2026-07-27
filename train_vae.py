#!/usr/bin/env python3
"""
Train Loom 3D Causal VAE
Learns to compress video into spacetime latent patches.
Usage:
    python train_vae.py --data_dir data/videos --output_dir checkpoints/vae
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


class VideoDataset(Dataset):
    """Simple video dataset that loads clips."""
    def __init__(self, data_dir: str, clip_frames: int = 16, resolution: int = 256):
        self.data_dir = Path(data_dir)
        self.clip_frames = clip_frames
        self.resolution = resolution

        # Find all video files
        self.video_files = []
        for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
            self.video_files.extend(list(self.data_dir.rglob(ext)))

        print(f"Found {len(self.video_files)} videos")

        # Precompute clip indices
        self.clips = []
        for vf in self.video_files:
            cap = cv2.VideoCapture(str(vf))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total_frames >= clip_frames:
                for start in range(0, total_frames - clip_frames, clip_frames // 2):
                    self.clips.append((vf, start))

        print(f"Total clips: {len(self.clips)}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        video_path, start_frame = self.clips[idx]
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(self.clip_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.resolution, self.resolution))
            frames.append(frame)
        cap.release()

        # Pad if needed
        while len(frames) < self.clip_frames:
            frames.append(frames[-1] if frames else np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8))

        # Stack and normalize: (T, H, W, 3) -> (3, T, H, W), [-1, 1]
        video = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0
        video = torch.from_numpy(video).permute(3, 0, 1, 2)  # (C, T, H, W)
        return video


def train_vae(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Model
    model = LoomVAE3D(latent_dim=args.latent_dim).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"VAE parameters: {total_params / 1e6:.1f}M")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=0.0)

    # Dataset
    dataset = VideoDataset(args.data_dir, clip_frames=args.clip_frames, resolution=args.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            batch = batch.to(device)  # (B, 3, T, H, W)

            # Forward
            recon, mean, logvar = model(batch)

            # Reconstruction loss (L1 + L2)
            l1_loss = F.l1_loss(recon, batch)
            l2_loss = F.mse_loss(recon, batch)
            recon_loss = l1_loss + l2_loss

            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            kl_loss = kl_loss / batch.numel()

            # Perceptual loss (optional, using simple LPIPS-like gradient)
            # Skip for now to keep it simple

            # Total loss
            loss = recon_loss + args.kl_weight * kl_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Logging
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "kl": f"{kl_loss.item():.6f}",
            })

            global_step += 1

            # Save checkpoint
            if global_step % args.save_every == 0:
                save_path = Path(args.output_dir) / f"vae_step_{global_step}.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"\nSaved checkpoint to {save_path}")

    # Final save
    final_path = Path(args.output_dir) / "vae_final.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Loom 3D VAE")
    parser.add_argument("--data_dir", required=True, help="Directory with training videos")
    parser.add_argument("--output_dir", default="checkpoints/vae", help="Checkpoint output directory")
    parser.add_argument("--latent_dim", type=int, default=16, help="VAE latent dimension")
    parser.add_argument("--resolution", type=int, default=256, help="Training resolution")
    parser.add_argument("--clip_frames", type=int, default=16, help="Frames per clip")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--kl_weight", type=float, default=1e-6, help="KL loss weight")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--save_every", type=int, default=5000, help="Save checkpoint every N steps")
    args = parser.parse_args()

    train_vae(args)


if __name__ == "__main__":
    main()
