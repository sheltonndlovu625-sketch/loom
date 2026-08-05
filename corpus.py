"""
Loom Corpus Builder
Extract motion primitives, texture patches, and CA rules from raw video data.
No neural network training. Pure signal processing and clustering.
"""
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import json
import struct
import math

from loom import CARule

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    import zlib

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


class MotionExtractor:
    """Extract optical flow primitives from videos using classical methods."""

    def __init__(self, method: str = "farneback"):
        self.method = method

    def extract_from_video(self, video_path: str, sample_rate: int = 2) -> List[dict]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if len(frames) < 2:
            return []

        flows = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)

        for i in range(1, len(frames), sample_rate):
            gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            if self.method == "farneback":
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
            else:
                flow = np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)
            flows.append(flow)
            prev_gray = gray

        return self._segment_flows(flows, frames[0].shape[:2])

    def _segment_flows(self, flows: List[np.ndarray], shape: Tuple[int, int]) -> List[dict]:
        if not flows:
            return []

        flow_stack = np.stack(flows)
        mean_flow = np.mean(flow_stack, axis=(1, 2))
        std_flow = np.std(flow_stack, axis=(1, 2))

        dx_mean = np.mean(mean_flow[:, 0])
        dy_mean = np.mean(mean_flow[:, 1])
        dx_std = np.mean(std_flow[:, 0])
        dy_std = np.mean(std_flow[:, 1])

        if abs(dx_mean) > 2.0 and abs(dy_mean) < 1.0:
            category = "pan_right" if dx_mean > 0 else "pan_left"
        elif abs(dy_mean) > 2.0 and abs(dx_mean) < 1.0:
            category = "tilt_down" if dy_mean > 0 else "tilt_up"
        elif abs(dx_mean) > 1.0 and abs(dy_mean) > 1.0:
            category = "orbit" if dx_std > dy_std else "dolly"
        elif np.mean(std_flow) < 0.5:
            category = "static"
        else:
            category = "complex"

        flow_mag = np.sqrt(flow_stack[:, :, :, 0]**2 + flow_stack[:, :, :, 1]**2)
        temporal_mean = np.mean(flow_mag, axis=(1, 2))
        fft = np.fft.fft(temporal_mean)
        spectral = np.abs(fft[:min(64, len(fft))])
        if spectral.max() > 0:
            spectral = spectral / spectral.max()

        return [{
            "flow_field": flow_stack.transpose(0, 3, 1, 2),
            "category": category,
            "intensity": float(np.mean(flow_mag)),
            "spectral": spectral
        }]


class TextureExtractor:
    def __init__(self, patch_size: int = 64, stride: int = 32):
        self.patch_size = patch_size
        self.stride = stride

    def extract_from_video(self, video_path: str) -> List[dict]:
        cap = cv2.VideoCapture(video_path)
        patches = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 10 != 0:
                frame_count += 1
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            for y in range(0, h - self.patch_size, self.stride):
                for x in range(0, w - self.patch_size, self.stride):
                    patch = frame[y:y+self.patch_size, x:x+self.patch_size]
                    if np.std(patch) < 15:
                        continue

                    category = self._classify_patch(patch)
                    patch_flat = patch.reshape(-1, 3).astype(np.float32) / 255.0
                    mean = np.mean(patch_flat, axis=0)
                    centered = patch_flat - mean
                    cov = np.cov(centered.T)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    idx = np.argsort(eigvals)[::-1]
                    pca_basis = eigvecs[:, idx[:8]].flatten()

                    patches.append({
                        "patch": patch,
                        "pca_basis": pca_basis,
                        "category": category,
                        "scale_range": (0.5, 2.0)
                    })
            frame_count += 1

        cap.release()
        return self._deduplicate(patches, max_patches=500)

    def _classify_patch(self, patch: np.ndarray) -> str:
        mean_color = np.mean(patch, axis=(0, 1))
        if mean_color[2] > mean_color[0] + 20 and mean_color[2] > mean_color[1] + 20:
            return "sky"
        elif mean_color[0] > mean_color[1] + 20 and mean_color[0] > mean_color[2] + 20:
            return "ground"
        elif np.std(mean_color) < 20:
            return "neutral"
        elif mean_color[1] > mean_color[0] and mean_color[1] > mean_color[2]:
            return "vegetation"
        else:
            return "mixed"

    def _deduplicate(self, patches: List[dict], max_patches: int = 500) -> List[dict]:
        if len(patches) <= max_patches:
            return patches

        selected = [patches[0]]
        for patch in patches[1:]:
            if len(selected) >= max_patches:
                break

            patch_hist = cv2.calcHist([patch["patch"]], [0, 1, 2], None, [8, 8, 8], 
                                      [0, 256, 0, 256, 0, 256]).flatten()

            is_diverse = True
            for sel in selected:
                sel_hist = cv2.calcHist([sel["patch"]], [0, 1, 2], None, [8, 8, 8],
                                        [0, 256, 0, 256, 0, 256]).flatten()
                dist = cv2.compareHist(patch_hist.astype(np.float32), sel_hist.astype(np.float32),
                                      cv2.HISTCMP_BHATTACHARYYA)
                if dist < 0.3:
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(patch)

        return selected


class CARuleMiner:
    def __init__(self):
        self.known_rules = {
            "conway": CARule(name="conway", kernel=np.ones((3,3)), birth=[3], survive=[2,3]),
            "coral": CARule(name="coral", kernel=np.ones((3,3)), birth=[3,4,5,6,7,8], survive=[4,5,6,7,8]),
            "anneal": CARule(name="anneal", kernel=np.ones((3,3)), birth=[3,5,6,7,8], survive=[4,6,7,8]),
        }

    def mine_from_texture(self, texture: np.ndarray, num_generations: int = 50) -> Dict[str, CARule]:
        rules = {}
        for name, base_rule in self.known_rules.items():
            for i in range(3):
                variant_name = f"{name}_v{i}"
                birth = [b + np.random.randint(-1, 2) for b in base_rule.birth]
                survive = [s + np.random.randint(-1, 2) for s in base_rule.survive]
                birth = list(set([max(0, min(8, b)) for b in birth]))
                survive = list(set([max(0, min(8, s)) for s in survive]))

                rules[variant_name] = CARule(
                    name=variant_name,
                    kernel=base_rule.kernel,
                    birth=birth,
                    survive=survive
                )
        return rules


class CorpusPacker:
    def __init__(self):
        self.motion_primitives: List[dict] = []
        self.texture_patches: List[dict] = []
        self.ca_rules: Dict[str, CARule] = {}

    def add_video(self, video_path: str):
        print(f"Processing {video_path}...")
        motion_ext = MotionExtractor()
        motions = motion_ext.extract_from_video(video_path)
        self.motion_primitives.extend(motions)

        tex_ext = TextureExtractor()
        textures = tex_ext.extract_from_video(video_path)
        self.texture_patches.extend(textures)

        print(f"  -> {len(motions)} motion primitives, {len(textures)} texture patches")

    def add_ca_rules(self, rules: Dict[str, CARule]):
        self.ca_rules.update(rules)

    def pack(self, output_path: str, target_size_mb: float = 1000.0):
        print(f"\nPacking corpus into {output_path}...")

        # Flatten motion primitives
        motion_blocks = []
        for i, mp in enumerate(self.motion_primitives):
            flat = np.concatenate([
                [float(i)],
                mp["flow_field"].flatten(),
                [float(hash(mp["category"]) % 10000)],
                [mp["intensity"]],
                mp["spectral"]
            ]).astype(np.float32)
            motion_blocks.append(flat)

        motion_data = np.concatenate(motion_blocks) if motion_blocks else np.array([], dtype=np.float32)

        # Flatten texture patches
        texture_blocks = []
        for i, tp in enumerate(self.texture_patches):
            flat = np.concatenate([
                [float(i)],
                tp["patch"].flatten().astype(np.float32),
                tp["pca_basis"],
                [float(hash(tp["category"]) % 10000)],
                [tp["scale_range"][0], tp["scale_range"][1]]
            ]).astype(np.float32)
            texture_blocks.append(flat)

        texture_data = np.concatenate(texture_blocks) if texture_blocks else np.array([], dtype=np.float32)

        # Flatten CA rules
        ca_blocks = []
        for name, rule in self.ca_rules.items():
            flat = np.concatenate([
                rule.kernel.flatten(),
                [float(b) for b in rule.birth] + [0.0] * (3 - len(rule.birth)),
                [float(s) for s in rule.survive] + [0.0] * (3 - len(rule.survive))
            ]).astype(np.float32)
            ca_blocks.append(flat)

        ca_data = np.concatenate(ca_blocks) if ca_blocks else np.array([], dtype=np.float32)

        # Metadata
        metadata = {
            "version": 1,
            "num_motion_primitives": len(self.motion_primitives),
            "num_texture_patches": len(self.texture_patches),
            "num_ca_rules": len(self.ca_rules),
            "motion_primitive_size": len(motion_blocks[0]) if motion_blocks else 0,
            "texture_patch_size": len(texture_blocks[0]) if texture_blocks else 0,
            "ca_rule_size": len(ca_blocks[0]) if ca_blocks else 0,
            "flow_shape": list(self.motion_primitives[0]["flow_field"].shape) if self.motion_primitives else [0, 0, 0, 0],
            "patch_shape": list(self.texture_patches[0]["patch"].shape) if self.texture_patches else [0, 0, 0],
            "spectral_size": 64,
            "pca_components": 24,
            "sections": {
                "motion": {"id": 0},
                "texture": {"id": 1},
                "ca_rules": {"id": 2}
            }
        }

        # Compress sections
        if HAS_LZ4:
            motion_comp = lz4.frame.compress(motion_data.tobytes(), compression_level=9)
            texture_comp = lz4.frame.compress(texture_data.tobytes(), compression_level=9)
            ca_comp = lz4.frame.compress(ca_data.tobytes(), compression_level=9)
        else:
            motion_comp = zlib.compress(motion_data.tobytes(), level=9)
            texture_comp = zlib.compress(texture_data.tobytes(), level=9)
            ca_comp = zlib.compress(ca_data.tobytes(), level=9)

        # Write .vid file
        # Format:
        #   [Header: 32 bytes]
        #   [Section Directory: 24 bytes * num_sections]
        #   [Section Data 0]
        #   [Section Data 1]
        #   ...
        #   [Metadata JSON]
        #   [Metadata Length: 4 bytes]

        with open(output_path, 'wb') as f:
            # Header
            f.write(b"LOOM")
            f.write(struct.pack('<H', 1))  # version
            f.write(struct.pack('<H', 3))  # num_sections
            f.write(b'\x00' * 24)  # reserved

            # Reserve space for section directory
            dir_start = f.tell()
            f.write(b'\x00' * (3 * 24))

            # Write section data
            sections_info = []
            for sec_data in [motion_comp, texture_comp, ca_comp]:
                offset = f.tell()
                f.write(sec_data)
                sections_info.append((offset, len(sec_data), len(sec_data) * 2))

            # Write metadata
            meta_json = json.dumps(metadata).encode('utf-8')
            f.write(meta_json)
            f.write(struct.pack('<I', len(meta_json)))

            # Go back and write section directory
            f.seek(dir_start)
            for i, (offset, comp_len, decomp_len) in enumerate(sections_info):
                f.write(struct.pack('<I', i))      # section_id
                f.write(struct.pack('<Q', offset)) # offset
                f.write(struct.pack('<I', comp_len))   # comp_len
                f.write(struct.pack('<I', decomp_len)) # decomp_len
                f.write(b'\x00' * 4)               # reserved

        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"Packed: {output_path} ({file_size_mb:.1f} MB)")
        print(f"   Motion: {len(self.motion_primitives)} primitives")
        print(f"   Texture: {len(self.texture_patches)} patches")
        print(f"   CA Rules: {len(self.ca_rules)} rules")

        if file_size_mb > target_size_mb:
            print(f"Warning: Exceeds {target_size_mb}MB target")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build Loom corpus from video dataset")
    parser.add_argument("--input_dir", required=True, help="Directory containing training videos")
    parser.add_argument("--output", default="corpus.vid", help="Output .vid file")
    parser.add_argument("--max_videos", type=int, default=1000, help="Max videos to process")
    parser.add_argument("--target_size", type=float, default=1000.0, help="Target size in MB")
    args = parser.parse_args()

    packer = CorpusPacker()

    video_files = list(Path(args.input_dir).glob("*.mp4")) + \
                  list(Path(args.input_dir).glob("*.avi")) + \
                  list(Path(args.input_dir).glob("*.mov"))

    for video_path in tqdm(video_files[:args.max_videos], desc="Processing videos"):
        try:
            packer.add_video(str(video_path))
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    ca_miner = CARuleMiner()
    if packer.texture_patches:
        rules = ca_miner.mine_from_texture(packer.texture_patches[0]["patch"])
        packer.add_ca_rules(rules)

    packer.pack(args.output, target_size_mb=args.target_size)


if __name__ == "__main__":
    main()