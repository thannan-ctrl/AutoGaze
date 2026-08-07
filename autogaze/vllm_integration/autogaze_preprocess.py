"""
Full AutoGaze preprocessing for vLLM integration.

Runs the actual nvidia/AutoGaze learned model on raw video frames (before ViT)
to compute patch selection masks, then maps them to Qwen3-VL's patch grid.

Flow:
  raw video frames (any resolution)
    → resize to 224×224, ImageNet normalize
    → AutoGaze model (tiny LLaMA decoder, 4 layers, 192-dim)
    → per-frame patch masks at 4 scales (32+64+112+224 px)
    → take 224-scale mask (≈14×14 per frame)
    → bilinear upsample to target_grid_hw (e.g. 16×16 for Qwen3-VL)
    → flatten to (T × H × W,) retention mask for vLLM

Usage:
    from autogaze.vllm_integration.autogaze_preprocess import AutoGazePreprocessor

    prep = AutoGazePreprocessor.load("nvidia/AutoGaze")
    mask, K = prep.compute_retention_mask(
        raw_frames,            # (T, C, H, W) float32 [0,1]
        target_grid_hw=(16, 16),  # Qwen3-VL post-merge grid
        gazing_ratio=0.5,
    )
    # mask: (T*16*16,) bool tensor on CPU
    # K: int, number of True entries
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Mock training-only deps that autogaze imports at module level but doesn't
# need for inference. Do this before any autogaze import.
import types as _types

def _mock_if_missing(name: str, attrs: dict | None = None):
    if name not in sys.modules:
        m = _types.ModuleType(name)
        if attrs:
            for k, v in attrs.items():
                setattr(m, k, v)
        sys.modules[name] = m

_mock_if_missing("wandb", {"run": None, "log": lambda *a, **k: None, "init": lambda *a, **k: None})
_mock_if_missing("wandb.sdk", {})
_mock_if_missing("wandb.sdk.lib", {})


class AutoGazePreprocessor:
    """Wraps the AutoGaze model for vLLM retention-mask computation."""

    AUTOGAZE_GRID = (14, 14)   # native output at 224px with 16px patch
    IMG_SIZE = 224
    IMG_MEAN = (0.485, 0.456, 0.406)
    IMG_STD  = (0.229, 0.224, 0.225)
    MAX_FRAMES_PER_CHUNK = 16  # AutoGaze processes max 16 frames at once

    def __init__(self, model):
        self.model = model
        self.model.eval()

    @classmethod
    def load(cls, model_id: str = "nvidia/AutoGaze", device: str = "cuda") -> "AutoGazePreprocessor":
        from autogaze.models.autogaze import AutoGaze
        print(f"[AutoGazePreprocessor] Loading {model_id} ...", flush=True)

        # transformers 5.x renamed _tied_weights_keys → all_tied_weights_keys (as a dict).
        # Patch the class for backward compat with the local AutoGaze code.
        if not hasattr(AutoGaze, "all_tied_weights_keys"):
            AutoGaze.all_tied_weights_keys = property(lambda self: {})

        ag = AutoGaze.from_pretrained(model_id)
        ag = ag.to(device).eval()
        print("[AutoGazePreprocessor] Loaded.", flush=True)
        return cls(ag)

    def preprocess_frames(self, raw_frames: torch.Tensor) -> torch.Tensor:
        """
        Resize and normalize raw frames for AutoGaze.

        Args:
            raw_frames: (T, C, H, W) float32 [0, 1]
        Returns:
            (T, C, 224, 224) float32, ImageNet normalized
        """
        T = raw_frames.shape[0]
        # Resize to 224×224
        frames = F.interpolate(
            raw_frames,
            size=(self.IMG_SIZE, self.IMG_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        # ImageNet normalize
        mean = torch.tensor(self.IMG_MEAN, device=frames.device).view(1, 3, 1, 1)
        std  = torch.tensor(self.IMG_STD,  device=frames.device).view(1, 3, 1, 1)
        return (frames - mean) / std

    @torch.no_grad()
    def run_autogaze(
        self,
        frames_224: torch.Tensor,  # (T, C, 224, 224) normalized
        gazing_ratio: float = 0.5,
        task_loss_requirement: float | None = None,
    ) -> torch.Tensor:
        """
        Run AutoGaze on preprocessed frames.

        Returns:
            largest_scale_mask: (T, 14, 14) bool — True = keep this patch
        """
        device = next(self.model.parameters()).device
        frames_224 = frames_224.to(device)
        T = frames_224.shape[0]
        max_chunk = self.MAX_FRAMES_PER_CHUNK

        # AutoGaze processes in temporal chunks of max_chunk frames
        all_masks = []
        for start in range(0, T, max_chunk):
            chunk = frames_224[start:start + max_chunk]  # (t, C, H, W)
            # AutoGaze expects (B, T, C, H, W)
            video = chunk.unsqueeze(0)  # (1, t, C, H, W)
            output = self.model.forward(
                inputs={"video": video},
                gazing_ratio=gazing_ratio,
                task_loss_requirement=task_loss_requirement,
                generate_only=True,
            )
            # gazing_mask: list of (B, T_chunk, N_scale) per scale; last = 224-scale
            mask_224 = output["gazing_mask"][-1]  # (1, t, N_224) float 0/1
            all_masks.append(mask_224.squeeze(0))  # (t, N_224)

        mask_flat = torch.cat(all_masks, dim=0)  # (T, N_224) float 0/1

        # Reshape to spatial grid: N_224 ≈ 14×14 = 196 (may be 195 due to int rounding)
        N = mask_flat.shape[1]
        side = int(N ** 0.5)  # 14
        # Zero-pad to 14×14 if needed
        if N < side * side:
            pad = torch.zeros(T, side * side - N, device=mask_flat.device)
            mask_flat = torch.cat([mask_flat, pad], dim=1)
        mask_2d = mask_flat[:, :side * side].reshape(T, side, side)  # (T, 14, 14)
        return mask_2d.bool()

    def compute_retention_mask(
        self,
        raw_frames: torch.Tensor,      # (T, C, H, W) float32 [0,1]
        target_grid_hw: tuple[int, int],  # (H_grid, W_grid) of downstream ViT after merge
        gazing_ratio: float = 0.5,
        task_loss_requirement: float | None = None,
    ) -> tuple[torch.Tensor, int]:
        """
        Full pipeline: raw frames → AutoGaze → retention mask for vLLM.

        Args:
            raw_frames: (T, C, H, W) float32 [0,1]
            target_grid_hw: (H, W) of the downstream ViT's post-merge patch grid
            gazing_ratio: fraction of patches to keep per frame (0-1)
            task_loss_requirement: optional quality-driven stopping threshold

        Returns:
            retention_mask: (T * H_grid * W_grid,) bool, True = keep
            K: int, number of kept tokens
        """
        T = raw_frames.shape[0]
        H_grid, W_grid = target_grid_hw

        # Step 1: preprocess to 224×224
        device = next(self.model.parameters()).device
        frames_224 = self.preprocess_frames(raw_frames.to(device))

        # Step 2: run AutoGaze → (T, 14, 14) bool mask
        ag_mask_14 = self.run_autogaze(frames_224, gazing_ratio, task_loss_requirement)

        # Step 3: bilinear upsample (T, 14, 14) → (T, H_grid, W_grid)
        ag_mask_float = ag_mask_14.float().unsqueeze(0)  # (1, T, 14, 14)
        ag_mask_float = ag_mask_float.view(1, T, 14, 14)
        # Upsample each frame mask independently: (T, 1, 14, 14) → (T, 1, H, W)
        ag_mask_float = ag_mask_float.squeeze(0).unsqueeze(1)  # (T, 1, 14, 14)
        ag_mask_resized = F.interpolate(
            ag_mask_float, size=(H_grid, W_grid), mode="bilinear", align_corners=False
        ).squeeze(1)  # (T, H_grid, W_grid)

        # Threshold: keep patches where AutoGaze assigned > 0.5 weight
        retention_mask = (ag_mask_resized > 0.5).cpu()  # (T, H_grid, W_grid)
        retention_mask_flat = retention_mask.view(-1)    # (T * H_grid * W_grid,)
        K = int(retention_mask_flat.sum().item())

        print(
            f"[AutoGazePreprocessor] {T} frames × {H_grid}×{W_grid} grid → "
            f"K={K}/{T*H_grid*W_grid} tokens kept ({K/(T*H_grid*W_grid)*100:.1f}%)",
            flush=True,
        )
        return retention_mask_flat, K
