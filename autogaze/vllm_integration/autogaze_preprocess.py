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

# wandb is installed; no mock needed


class AutoGazePreprocessor:
    """Wraps the AutoGaze model for vLLM retention-mask computation."""

    AUTOGAZE_GRID = (14, 14)   # native output at 224px with 16px patch
    IMG_SIZE = 224
    IMG_MEAN = (0.485, 0.456, 0.406)
    IMG_STD  = (0.229, 0.224, 0.225)
    MAX_FRAMES_PER_CHUNK = 16  # AutoGaze processes max 16 frames at once
    # How many independent 16-frame chunks to run in one batched AutoGaze call.
    # Default 1 == original sequential-per-chunk behavior (byte-identical
    # output). Chunks don't share state, so batching them is a pure speed
    # lever, BUT batched vs. sequential matmul/attention kernels use a
    # different reduction order, which can flip an occasional greedy-argmax
    # gaze decision and cascade through the chunk's autoregressive per-frame
    # decode (verified: no cross-item state leakage, purely floating-point
    # kernel non-determinism -- see scripts/diagnose_autogaze_batch_divergence*.py).
    # Opt in explicitly (e.g. AutoGazePreprocessor.MAX_CHUNKS_PER_BATCH = 8)
    # once this drift has been validated against downstream accuracy.
    MAX_CHUNKS_PER_BATCH = 1

    def __init__(self, model):
        self.model = model
        self.model.eval()

    @classmethod
    def load(cls, model_id: str = "nvidia/AutoGaze", device: str = "cuda") -> "AutoGazePreprocessor":
        from autogaze.models.autogaze import AutoGaze
        print(f"[AutoGazePreprocessor] Loading {model_id} ...", flush=True)

        # transformers 5.x renamed _tied_weights_keys → all_tied_weights_keys (as a dict).
        # Patch the class for backward compat with the local AutoGaze code.
        # PreTrainedModel.post_init() assigns to this attribute, so the patched
        # property needs a setter (a read-only property raises AttributeError
        # there before the model can even be constructed).
        if not hasattr(AutoGaze, "all_tied_weights_keys"):
            AutoGaze.all_tied_weights_keys = property(
                lambda self: getattr(self, "_all_tied_weights_keys", {}),
                lambda self, value: setattr(self, "_all_tied_weights_keys", value),
            )

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
        seed: int = 42,
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

        # Fix random seed for reproducible K selection across runs
        torch.manual_seed(seed)

        # AutoGaze processes in temporal chunks of max_chunk frames. Chunks are
        # independent (no shared state/cache across them), so full-size chunks
        # are stacked into the batch dimension and run through AutoGaze together
        # instead of one sequential call per chunk -- this is the dominant cost
        # of AutoGaze preprocessing at high frame counts (see
        # nvila_hd_nvf16_experiments.md Experiment 2/4), and chunk independence
        # makes it a pure batching win with no change to per-chunk outputs.
        n_full_chunks = T // max_chunk
        full_chunk_starts = [i * max_chunk for i in range(n_full_chunks)]
        ragged_start = n_full_chunks * max_chunk
        has_ragged = ragged_start < T

        all_masks = [None] * (n_full_chunks + (1 if has_ragged else 0))

        def run_batch(chunk_list, order_indices):
            # chunk_list: list of (t, C, H, W) tensors, all with the same t
            video = torch.stack(chunk_list, dim=0)  # (Bc, t, C, H, W)
            output = self.model.forward(
                inputs={"video": video},
                gazing_ratio=gazing_ratio,
                task_loss_requirement=task_loss_requirement,
                generate_only=True,
            )
            # gazing_mask: list of (Bc, t, N_scale) per scale; last = 224-scale
            mask_224 = output["gazing_mask"][-1]  # (Bc, t, N_224) float 0/1
            for i, idx in enumerate(order_indices):
                all_masks[idx] = mask_224[i]  # (t, N_224)

        for batch_start in range(0, len(full_chunk_starts), self.MAX_CHUNKS_PER_BATCH):
            batch_starts = full_chunk_starts[batch_start:batch_start + self.MAX_CHUNKS_PER_BATCH]
            chunk_list = [frames_224[s:s + max_chunk] for s in batch_starts]
            order_indices = list(range(batch_start, batch_start + len(batch_starts)))
            run_batch(chunk_list, order_indices)

        if has_ragged:
            # Shorter trailing chunk can't be stacked with the fixed-size ones
            # (batching requires uniform T), so it runs on its own.
            run_batch([frames_224[ragged_start:T]], [n_full_chunks])

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
        seed: int = 42,
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
        ag_mask_14 = self.run_autogaze(frames_224, gazing_ratio, task_loss_requirement, seed=seed)

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
