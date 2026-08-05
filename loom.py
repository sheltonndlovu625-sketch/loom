"""
Loom — A Non-Parametric Video Synthesizer
=========================================
No neural networks. No gradient descent. No millions of parameters.

Video is woven from three threads:
  1. WARP   — Motion fields (optical flow primitives)
  2. WEFT   — Texture patches (reaction-diffusion grown)
  3. PATTERN — Cellular automata detail rules

Training builds a searchable corpus. Inference retrieves, warps, and weaves.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path
import json
import struct
import math
from collections import defaultdict

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    import zlib

# ---------------------------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------------------------

@dataclass
class MotionPrimitive:
    """A reusable motion field archetype."""
    id: int
    flow_field: np.ndarray
    category: str
    intensity: float
    spectral_signature: np.ndarray

    def warp_frame(self, frame: np.ndarray, t: int) -> np.ndarray:
        """Warp a single frame using this motion field at time t."""
        from scipy.ndimage import map_coordinates
        h, w = frame.shape[:2]
        flow = self.flow_field[t % len(self.flow_field)]

        y, x = np.mgrid[0:h, 0:w]
        coords = np.stack([y + flow[1], x + flow[0]])

        if frame.ndim == 3:
            warped = np.zeros_like(frame)
            for c in range(frame.shape[2]):
                warped[:, :, c] = map_coordinates(
                    frame[:, :, c], coords, order=1, mode='reflect'
                )
        else:
            warped = map_coordinates(frame, coords, order=1, mode='reflect')
        return warped


@dataclass  
class TexturePatch:
    """A reusable texture exemplar with synthesis metadata."""
    id: int
    patch: np.ndarray
    pca_basis: np.ndarray
    category: str
    scale_range: Tuple[float, float]

    def synthesize(self, target_shape: Tuple[int, int], seed: int = 0) -> np.ndarray:
        """Grow texture to target size using patch-based quilting."""
        from scipy.ndimage import zoom, rotate
        np.random.seed(seed)
        h, w = target_shape
        ph, pw = self.patch.shape[:2]

        output = np.zeros((h, w, 3), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)

        tile_h, tile_w = ph // 2, pw // 2
        for y in range(0, h, tile_h):
            for x in range(0, w, tile_w):
                scale = np.random.uniform(0.8, 1.2)
                angle = np.random.uniform(-5, 5)

                variant = self._transform_patch(scale, angle)
                vh, vw = variant.shape[:2]

                y_end = min(y + vh, h)
                x_end = min(x + vw, w)

                yy, xx = np.mgrid[0:y_end-y, 0:x_end-x]
                gw = np.exp(-((yy/(vh/2))**2 + (xx/(vw/2))**2))

                output[y:y_end, x:x_end] += variant[:y_end-y, :x_end-x] * gw[:, :, None]
                weights[y:y_end, x:x_end] += gw

        output = output / weights[:, :, None]
        return np.clip(output, 0, 255).astype(np.uint8)

    def _transform_patch(self, scale: float, angle: float) -> np.ndarray:
        """Apply random scale and rotation."""
        from scipy.ndimage import zoom, rotate
        scaled = zoom(self.patch, (scale, scale, 1), order=1)
        return rotate(scaled, angle, reshape=False, order=1)


@dataclass
class CARule:
    """Cellular Automata rule for organic detail generation."""
    name: str
    kernel: np.ndarray
    birth: List[int]
    survive: List[int]
    channels: int = 3

    def step(self, grid: np.ndarray) -> np.ndarray:
        """Apply one CA step."""
        from scipy.signal import convolve2d
        new_grid = grid.copy()

        for c in range(self.channels):
            neighbors = convolve2d(
                grid[:, :, c], self.kernel, mode='same', boundary='wrap'
            )
            born = np.isin(neighbors, self.birth)
            survive = np.isin(neighbors, self.survive)

            new_grid[:, :, c] = np.where(
                (grid[:, :, c] > 0.5) & survive,
                grid[:, :, c],
                np.where(born, 1.0, grid[:, :, c] * 0.95)
            )

        return np.clip(new_grid, 0, 1)


@dataclass
class SceneGraph:
    """Semantic composition plan for a video."""
    background: str
    foreground_objects: List[Dict]
    camera_motion: str
    lighting: str
    style: str
    duration_frames: int

    def to_prompt_vector(self) -> np.ndarray:
        """Convert to a searchable vector."""
        text = f"{self.background} {self.camera_motion} {self.lighting} {self.style}"
        hash_val = hash(text) % (2**20)
        vec = np.zeros(1024)
        vec[hash_val % 1024] = 1.0
        return vec


# ---------------------------------------------------------------------------
# The Loom Engine
# ---------------------------------------------------------------------------

class LoomEngine:
    """Main synthesis engine. No neural nets. Pure algorithms."""

    def __init__(self, vid_path: str):
        self.vid_path = vid_path
        self.corpus = self._load_corpus()
        self.motion_lib: Dict[str, List[MotionPrimitive]] = defaultdict(list)
        self.texture_lib: Dict[str, List[TexturePatch]] = defaultdict(list)
        self.ca_rules: Dict[str, CARule] = {}
        self.scene_templates: List[SceneGraph] = []
        self._index_corpus()

        # Load VAE imagination layer if present
        self.vae = None
        self.prototypes = {}
        self.texture_type = "raw"
        self._load_vae()


    def _load_corpus(self) -> dict:
        """Load the .vid corpus file."""
        with open(self.vid_path, 'rb') as f:
            # Header: LOOM + version(2) + num_sections(2) + reserved(24) = 32 bytes
            header = f.read(32)
            if len(header) < 32:
                raise ValueError("File too small for header")

            magic = header[:4]
            if magic != b"LOOM":
                raise ValueError(f"Invalid magic: {magic}")

            version = struct.unpack('<H', header[4:6])[0]
            num_sections = struct.unpack('<H', header[6:8])[0]

            # Section directory: each entry is 24 bytes
            # section_id(4) + offset(8) + comp_len(4) + decomp_len(4) + reserved(4)
            sections = {}
            for _ in range(num_sections):
                entry = f.read(24)
                sec_id = struct.unpack('<I', entry[0:4])[0]
                offset = struct.unpack('<Q', entry[4:12])[0]
                comp_len = struct.unpack('<I', entry[12:16])[0]
                decomp_len = struct.unpack('<I', entry[16:20])[0]
                sections[sec_id] = (offset, comp_len, decomp_len)

            # Read metadata from end of file
            f.seek(0, 2)  # end
            file_size = f.tell()

            # Metadata is last 4 bytes (length) + JSON string
            f.seek(file_size - 4)
            meta_len = struct.unpack('<I', f.read(4))[0]
            f.seek(file_size - 4 - meta_len)
            meta_json = f.read(meta_len)
            metadata = json.loads(meta_json.decode('utf-8'))

            # Decompress and load each section
            corpus = {'metadata': metadata, 'sections': {}}
            for sec_name, sec_info in metadata['sections'].items():
                sec_id = sec_info['id']
                if sec_id not in sections:
                    continue
                offset, comp_len, decomp_len = sections[sec_id]
                f.seek(offset)
                compressed = f.read(comp_len)

                if HAS_LZ4:
                    decompressed = lz4.frame.decompress(compressed)
                else:
                    decompressed = zlib.decompress(compressed)

                corpus['sections'][sec_name] = np.frombuffer(decompressed, dtype=np.float32)

            return corpus

    def _index_corpus(self):
        """Build searchable indexes from raw corpus data."""
        meta = self.corpus['metadata']

        motion_data = self.corpus['sections'].get('motion', np.array([]))
        if len(motion_data) > 0:
            mp_size = meta.get('motion_primitive_size', 0)
            if mp_size > 0:
                idx = 0
                while idx + mp_size <= len(motion_data):
                    block = motion_data[idx:idx+mp_size]
                    prim = self._decode_motion_primitive(block, meta)
                    self.motion_lib[prim.category].append(prim)
                    idx += mp_size

        tex_data = self.corpus['sections'].get('texture', np.array([]))
        if len(tex_data) > 0:
            tp_size = meta.get('texture_patch_size', 0)
            if tp_size > 0:
                idx = 0
                while idx + tp_size <= len(tex_data):
                    block = tex_data[idx:idx+tp_size]
                    patch = self._decode_texture_patch(block, meta)
                    self.texture_lib[patch.category].append(patch)
                    idx += tp_size

        ca_data = self.corpus['sections'].get('ca_rules', np.array([]))
        if len(ca_data) > 0:
            self.ca_rules = self._decode_ca_rules(ca_data, meta)

        total_motion = sum(len(v) for v in self.motion_lib.values())
        total_tex = sum(len(v) for v in self.texture_lib.values())
        print(f"Corpus loaded: {total_motion} motion, {total_tex} textures, {len(self.ca_rules)} CA rules")

    def _decode_motion_primitive(self, block: np.ndarray, meta: dict) -> MotionPrimitive:
        """Decode a motion primitive from flat array."""
        flow_size = np.prod(meta.get('flow_shape', [1, 2, 64, 64]))
        spectral_size = meta.get('spectral_size', 64)

        return MotionPrimitive(
            id=int(block[0]),
            flow_field=block[1:1+flow_size].reshape(meta.get('flow_shape', [1, 2, 64, 64])),
            category="motion",
            intensity=float(block[1+flow_size]) if 1+flow_size < len(block) else 0.5,
            spectral_signature=block[2+flow_size:2+flow_size+spectral_size] if 2+flow_size < len(block) else np.zeros(spectral_size)
        )

    def _decode_texture_patch(self, block: np.ndarray, meta: dict) -> TexturePatch:
        """Decode a texture patch from flat array."""
        patch_pixels = np.prod(meta.get('patch_shape', [64, 64, 3]))
        pca_components = meta.get('pca_components', 24)

        return TexturePatch(
            id=int(block[0]),
            patch=block[1:1+patch_pixels].reshape(meta.get('patch_shape', [64, 64, 3])).astype(np.uint8),
            pca_basis=block[1+patch_pixels:1+patch_pixels+pca_components] if 1+patch_pixels < len(block) else np.zeros(pca_components),
            category="texture",
            scale_range=(0.5, 2.0)
        )

    def _decode_ca_rules(self, data: np.ndarray, meta: dict) -> Dict[str, CARule]:
        """Decode CA rules."""
        rules = {}
        rule_size = meta.get('ca_rule_size', 15)
        idx = 0
        rule_id = 0
        while idx + rule_size <= len(data):
            block = data[idx:idx+rule_size]
            kernel = block[:9].reshape(3, 3)
            birth = [int(block[9]), int(block[10]), int(block[11])]
            survive = [int(block[12]), int(block[13]), int(block[14])]
            rules[f"rule_{rule_id}"] = CARule(
                name=f"rule_{rule_id}",
                kernel=kernel,
                birth=[b for b in birth if b > 0],
                survive=[s for s in survive if s > 0]
            )
            idx += rule_size
            rule_id += 1
        return rules

    def synthesize(
        self,
        prompt: str,
        width: int = 854,
        height: int = 480,
        num_frames: int = 120,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Synthesize video from text prompt. Returns (T, H, W, 3) uint8."""
        if seed is not None:
            np.random.seed(seed)

        scene = self._parse_prompt(prompt, num_frames)
        print(f"Scene: {scene.background} | Camera: {scene.camera_motion} | Style: {scene.style}")

        motion = self._retrieve_motion(scene.camera_motion, scene.duration_frames, height, width)
        bg_texture = self._synthesize_texture(scene.background, (height, width), seed)

        print("Weaving motion and texture...")
        frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)

        for t in range(num_frames):
            frame = motion.warp_frame(bg_texture, t)

            for obj in scene.foreground_objects:
                frame = self._composite_object(frame, obj, t, motion)

            if scene.style in self.ca_rules:
                frame = self._apply_ca(frame, scene.style, t)

            frame = self._color_grade(frame, scene.lighting)
            frames[t] = np.clip(frame, 0, 255).astype(np.uint8)

        return frames

    def _parse_prompt(self, prompt: str, duration: int) -> SceneGraph:
        """Parse text prompt into scene graph."""
        prompt = prompt.lower()

        backgrounds = ['ocean', 'sky', 'forest', 'city', 'desert', 'space', 'mountain', 'lake']
        bg = next((b for b in backgrounds if b in prompt), 'abstract')

        motions = {
            'pan left': 'pan_left', 'pan right': 'pan_right',
            'zoom in': 'zoom_in', 'zoom out': 'zoom_out',
            'orbit': 'orbit', 'static': 'static',
            'dolly': 'dolly', 'crane': 'crane'
        }
        cam = next((v for k, v in motions.items() if k in prompt), 'static')

        lights = ['sunset', 'sunrise', 'night', 'day', 'golden hour', 'neon', 'foggy']
        light = next((l for l in lights if l in prompt), 'day')

        styles = ['realistic', 'abstract', 'cartoon', 'dreamy', 'cinematic', 'vintage']
        style = next((s for s in styles if s in prompt), 'realistic')

        return SceneGraph(
            background=bg,
            foreground_objects=[],
            camera_motion=cam,
            lighting=light,
            style=style,
            duration_frames=duration
        )

    def _retrieve_motion(self, category: str, duration: int, h: int, w: int) -> MotionPrimitive:
        """Retrieve and adapt a motion primitive."""
        candidates = self.motion_lib.get(category, self.motion_lib.get('static', []))
        if not candidates:
            return MotionPrimitive(
                id=-1,
                flow_field=np.zeros((duration, 2, h, w), dtype=np.float32),
                category='static',
                intensity=0.0,
                spectral_signature=np.zeros(64)
            )

        prim = candidates[np.random.randint(len(candidates))]
        flow = prim.flow_field

        if flow.shape[1] != h or flow.shape[2] != w:
            from scipy.ndimage import zoom
            scale_y = h / flow.shape[1]
            scale_x = w / flow.shape[2]
            new_flow = np.zeros((flow.shape[0], h, w, 2), dtype=np.float32)
            for t in range(flow.shape[0]):
                new_flow[t, :, :, 0] = zoom(flow[t, 0], (scale_y, scale_x), order=1) * scale_x
                new_flow[t, :, :, 1] = zoom(flow[t, 1], (scale_y, scale_x), order=1) * scale_y
            flow = new_flow.transpose(0, 3, 1, 2)

        if flow.shape[0] < duration:
            repeats = math.ceil(duration / flow.shape[0])
            flow = np.tile(flow, (repeats, 1, 1, 1))[:duration]
        else:
            flow = flow[:duration]

        return MotionPrimitive(
            id=prim.id,
            flow_field=flow,
            category=prim.category,
            intensity=prim.intensity,
            spectral_signature=prim.spectral_signature
        )

    def _load_vae(self):
        """Load VAE decoder and prototypes from corpus if available."""
        if self.corpus['metadata'].get('version', 1) < 2:
            return  # Old corpus, no VAE

        texture_meta = self.corpus['metadata'].get('texture_meta', {})
        self.texture_type = texture_meta.get('type', 'raw')

        if self.texture_type != 'latent':
            return  # Raw textures, no imagination

        # Load VAE weights
        vae_data = self.corpus['sections'].get('vae_decoder', np.array([]))
        if len(vae_data) == 0:
            return

        # Parse VAE weights from flat bytes
        # (This is a simplified loader - in production you'd want a proper format)
        try:
            from loom_vae import TinyVAENumpy, sample_imagined_latent

            # Reconstruct weight dict from metadata hints
            # For now, we rely on the metadata to tell us the structure
            latent_dim = texture_meta.get('latent_dim', 256)
            self.vae = TinyVAENumpy(latent_dim=latent_dim)

            # Load prototypes
            proto_data = self.corpus['metadata'].get('prototypes', {})
            self.prototypes = {
                k: {
                    'mean': np.array(v['mean'], dtype=np.float32),
                    'std': np.array(v['std'], dtype=np.float32),
                }
                for k, v in proto_data.items()
            }

            print(f"  VAE loaded: {latent_dim}-dim imagination engine")
            print(f"  Prototypes: {list(self.prototypes.keys())}")
        except Exception as e:
            print(f"  VAE load failed: {e}")

    def _synthesize_texture(self, category: str, shape: Tuple[int, int], seed: int) -> np.ndarray:
        """Synthesize background texture — with imagination if VAE loaded."""
        # If VAE is available, use imagined textures
        if self.vae is not None and self.texture_type == "latent":
            return self._synthesize_texture_imagined([category], shape, seed)

        # Fallback: original retrieval-based synthesis
        candidates = self.texture_lib.get(category, self.texture_lib.get('abstract', []))
        if not candidates:
            np.random.seed(seed)
            return (np.random.rand(*shape, 3) * 255).astype(np.uint8)

        patch = candidates[np.random.randint(len(candidates))]
        return patch.synthesize(shape, seed)

    def _synthesize_texture_imagined(self, categories: List[str], shape: Tuple[int, int], seed: int) -> np.ndarray:
        """Use VAE to imagine textures from blended concepts."""
        from loom_vae import sample_imagined_latent

        np.random.seed(seed)

        # Sample imagined latent from prototypes
        z = sample_imagined_latent(self.prototypes, categories, seed)

        # Decode to patch
        patch_np = self.vae.decode(z)  # (3, 64, 64) in [-1, 1]
        patch_np = ((patch_np + 1) * 127.5).clip(0, 255).transpose(1, 2, 0).astype(np.uint8)

        # Use existing TexturePatch machinery to tile it to full size
        tp = TexturePatch(
            id=0,
            patch=patch_np,
            pca_basis=np.zeros(24),
            category="imagined",
            scale_range=(0.5, 2.0)
        )

        return tp.synthesize(shape, seed + 1)
    def _composite_object(self, frame: np.ndarray, obj: dict, t: int, motion: MotionPrimitive) -> np.ndarray:
        """Composite a foreground object onto the frame."""
        return frame

    def _apply_ca(self, frame: np.ndarray, style: str, t: int) -> np.ndarray:
        """Apply cellular automata detail layer."""
        rule = self.ca_rules.get(style, self.ca_rules.get('rule_0'))
        if rule is None:
            return frame

        grid = frame.astype(np.float32) / 255.0
        for _ in range(3):
            grid = rule.step(grid)

        result = frame * 0.85 + (grid * 255) * 0.15
        return result

    def _color_grade(self, frame: np.ndarray, lighting: str) -> np.ndarray:
        """Apply color grading based on lighting condition."""
        grading = {
            'sunset': {'tint': np.array([1.1, 0.9, 0.7])},
            'night': {'tint': np.array([0.7, 0.7, 1.2])},
            'day': {'tint': np.array([1.0, 1.0, 1.0])},
            'golden hour': {'tint': np.array([1.2, 1.0, 0.6])},
            'neon': {'tint': np.array([1.0, 0.8, 1.2])},
        }

        grade = grading.get(lighting, grading['day'])
        frame = frame.astype(np.float32)
        frame = frame * grade['tint']
        frame = np.clip(frame, 0, 255)
        return frame.astype(np.uint8)