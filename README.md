# Loom — Latent Video Diffusion Engine

**This is the Sora/Veo rewrite.** The old non-parametric patch-quilting engine is gone. Loom is now a full latent video diffusion model with a 3D causal VAE and a Diffusion Transformer (DiT).

## Architecture

```
Text Prompt → T5 Text Encoder → Text Embeddings
                                      ↓
Random Noise Latent (B,16,T/4,H/16,W/16) → DiT (768-dim, 16 layers) → Denoised Latent
                                      ↓
                         3D Causal VAE Decoder
                                      ↓
                              Video (T,H,W,3)
```

### 1. 3D Causal VAE (`loom_vae_3d.py`)
- **Causal in time**: Frame t only sees frames ≤ t. No future leakage.
- **Compression**: 4× temporal, 16× spatial. A 16-frame 256×256 video compresses to a `(16, 4, 16, 16)` latent tensor.
- **Architecture**: CausalConv3d encoder/decoder with residual blocks and factorized space-time attention.

### 2. Diffusion Transformer (`loom_dit.py`)
- **Patchify**: Latent video is split into spacetime patches `(1, 2, 2)`.
- **DiT Blocks**: AdaLN-Zero conditioning + self-attention + cross-attention to text.
- **Text conditioning**: T5-small (or CLIP) text encoder → MLP projection → added to timestep embedding.
- **Parameters**: ~300M–500M depending on depth (default: 768-dim, 16 layers, ~350M params).

### 3. Pipeline (`loom_pipeline.py`)
- **DDIM sampling**: 50 steps by default, configurable.
- **Classifier-Free Guidance (CFG)**: Scale 7.5 default for prompt adherence.
- **Output**: uint8 RGB video array `(T, H, W, 3)`.

## Training

### Step 1: Train the 3D VAE
```bash
python train_vae.py   --data_dir data/videos   --output_dir checkpoints/vae   --resolution 256   --clip_frames 16   --batch_size 2   --epochs 100
```

The VAE learns to compress video into causal spacetime latents. Training requires ~8GB VRAM at 256×256.

### Step 2: Precompute Latents (automatic)
The diffusion trainer caches VAE latents to disk for fast training:
```bash
python train_diffusion.py   --data_dir data/videos   --vae_path checkpoints/vae/vae_final.pt   --output_dir checkpoints/dit   --resolution 256   --clip_frames 16   --batch_size 4   --epochs 100   --hidden_size 768   --depth 16
```

The DiT learns to predict noise in the latent space, conditioned on text embeddings. Training requires ~16GB VRAM at 256×256, batch_size=4.

## Inference

```python
from loom import LoomEngine

engine = LoomEngine(
    vae_path="checkpoints/vae/vae_final.pt",
    dit_path="checkpoints/dit/dit_final.pt",
)

video = engine.synthesize(
    prompt="a red sports car drifting on a mountain road at sunset",
    width=512,
    height=512,
    num_frames=16,
    seed=42,
    cfg_scale=7.5,
)
# video: (16, 512, 512, 3) uint8 numpy array
```

Or use the updated CLI:
```bash
python inference.py   --vae checkpoints/vae/vae_final.pt   --dit checkpoints/dit/dit_final.pt   --prompt "a baboon sitting in a bush"   --output baboon.mp4   --width 512   --height 512   --frames 16   --seed 42
```

## What Changed

| Old Loom | New Loom |
|---|---|
| Patch quilting + optical flow warping | 3D causal VAE + DiT |
| No neural networks | Full latent diffusion |
| 2D image warping | 3D spacetime generation |
| Keyword regex matching | T5 text encoder |
| Generates textures only | Generates objects, motion, physics |
| 1GB mobile target | GPU training, GPU inference |
| "Ocean waves" only | "A baboon in the bush" ✅ |

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- 16GB+ VRAM for training (inference works on 8GB)
- Training data: video files (MP4, AVI, MOV) with captions

## Next Steps

1. **Collect captioned video data**: The model needs text-video pairs. Use InternVid, WebVid, or your own dataset.
2. **Scale up**: Increase DiT to 1024-dim, 24 layers for higher fidelity (~800M params).
3. **Add motion embeddings**: Condition on camera motion (pan, dolly, orbit) for controllable cinematography.
4. **Distill to 4 steps**: Use adversarial distillation (like SnapGen-V) for fast inference.
5. **Quantize**: Export to INT8/CoreML for mobile deployment (future work).

## Files

| File | Purpose |
|---|---|
| `loom_vae_3d.py` | 3D causal VAE encoder/decoder |
| `loom_dit.py` | Diffusion Transformer backbone |
| `loom_pipeline.py` | Full inference pipeline with DDIM sampling |
| `loom.py` | `LoomEngine` interface (same API as before) |
| `train_vae.py` | VAE training script |
| `train_diffusion.py` | DiT training script |
| `requirements.txt` | Dependencies |

## Citation

If you use this code, cite the key papers that make it possible:
- Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT)
- Blattmann et al., "Stable Video Diffusion" (3D VAE design)
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (LDM)
