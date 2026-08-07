"""
Retention mask implementations for AutoGaze × vLLM integration.

Provides drop-in replacements for vLLM's EVS compute_retention_mask.

Three modes:
  "evs"        — original EVS (cosine similarity between frames) — baseline
  "magnitude"  — magnitude-based importance (proof-of-concept, no extra model)
  "autogaze"   — full AutoGaze learned selection (requires nvidia/AutoGaze weights
                 AND raw frames stored in context via AutoGazeContext)
"""
from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

# Thread-local storage for raw frames (needed for full AutoGaze mode)
_ctx = threading.local()


class AutoGazeContext:
    """
    Context manager that passes pre-computed AutoGaze masks to the
    retention mask function during vLLM's model forward pass.

    Usage:
        prep = AutoGazePreprocessor.load("nvidia/AutoGaze")
        ag_mask, K = prep.compute_retention_mask(raw_frames, target_grid_hw=(rows, cols))

        with AutoGazeContext(ag_mask=ag_mask, K=K):
            outputs = llm.chat(messages, ...)
        # During llm.chat, compute_retention_mask will use ag_mask directly.
    """

    def __init__(
        self,
        ag_mask: torch.Tensor | None = None,
        K: int | None = None,
    ):
        # ag_mask: (T*H*W,) bool tensor with True = keep this patch
        self.payload = {"ag_mask": ag_mask, "K": K} if ag_mask is not None else None

    def __enter__(self):
        _ctx.raw_frames = self.payload
        return self

    def __exit__(self, *args):
        _ctx.raw_frames = None


def get_raw_frames() -> torch.Tensor | None:
    return getattr(_ctx, "raw_frames", None)


# ---------------------------------------------------------------------------
# EVS baseline (unchanged, re-exported for comparison)
# ---------------------------------------------------------------------------

def evs_retention_mask(
    video_embeds: torch.Tensor,
    video_size_thw: torch.LongTensor | tuple[int, int, int],
    spatial_merge_size: int,
    q: float,
) -> torch.Tensor:
    """Original EVS: cosine similarity between consecutive frames."""
    from vllm.multimodal.evs import compute_retention_mask as _orig
    return _orig(video_embeds, video_size_thw, spatial_merge_size, q)


# ---------------------------------------------------------------------------
# Magnitude-based retention (proof-of-concept, no extra model needed)
# ---------------------------------------------------------------------------

def magnitude_retention_mask(
    video_embeds: torch.Tensor,
    video_size_thw: torch.LongTensor | tuple[int, int, int],
    spatial_merge_size: int,
    q: float,
) -> torch.Tensor:
    """
    AutoGaze-inspired: rank patches by embedding magnitude (L2 norm).
    High-magnitude patches carry more information → keep the top-K.

    This is a proof-of-concept that plugs into the same hook without
    requiring the actual AutoGaze model weights or raw frames.

    Differences from EVS (cosine similarity):
      - EVS: measures temporal novelty (patch differs from previous frame)
      - Magnitude: measures absolute saliency (patch has large activation)
      - AutoGaze (production): learned importance from task signal
    """
    from vllm.multimodal.evs import compute_retained_tokens_count

    T, H, W = map(int, video_size_thw)
    rows, cols = H // spatial_merge_size, W // spatial_merge_size
    tokens_per_frame = rows * cols

    retain_num = compute_retained_tokens_count(tokens_per_frame, T, q)

    # Score = L2 norm of each patch embedding
    importance = video_embeds.norm(dim=-1)  # (T*rows*cols,)

    # Always include first frame (same guarantee as EVS / AutoGaze)
    first_frame_indices = torch.arange(tokens_per_frame, device=video_embeds.device)

    # For remaining patches: rank by magnitude
    rest_importance = importance.clone()
    rest_importance[:tokens_per_frame] = float("inf")  # force first frame in
    topk_indices = torch.topk(rest_importance, retain_num).indices

    mask = torch.zeros(video_embeds.shape[0], dtype=torch.bool, device=video_embeds.device)
    mask[topk_indices] = True
    # Also ensure first frame is always fully retained
    mask[:tokens_per_frame] = True

    # Clip to exact retain_num (first frame already accounted for)
    # Re-rank: importance of all selected
    selected = mask.nonzero(as_tuple=True)[0]
    if selected.numel() > retain_num:
        selected_importance = importance[selected]
        keep = torch.topk(selected_importance, retain_num).indices
        mask = torch.zeros_like(mask)
        mask[selected[keep]] = True

    return mask


# ---------------------------------------------------------------------------
# Full AutoGaze retention (uses nvidia/AutoGaze learned model + raw frames)
# ---------------------------------------------------------------------------

_autogaze_model = None
_autogaze_model_path = "nvidia/AutoGaze"


def _load_autogaze():
    global _autogaze_model
    if _autogaze_model is not None:
        return _autogaze_model

    try:
        # AutoGaze is loaded as part of the NVILA processor.
        # Here we load the standalone selector.
        from transformers import AutoModel
        _autogaze_model = AutoModel.from_pretrained(
            _autogaze_model_path,
            trust_remote_code=True,
        )
        _autogaze_model.eval()
        print(f"[AutoGaze-vLLM] Loaded AutoGaze model from {_autogaze_model_path}")
    except Exception as e:
        print(f"[AutoGaze-vLLM] Could not load AutoGaze model: {e}. Falling back to magnitude.")
        _autogaze_model = None

    return _autogaze_model


def autogaze_retention_mask(
    video_embeds: torch.Tensor,
    video_size_thw: torch.LongTensor | tuple[int, int, int],
    spatial_merge_size: int,
    q: float,
) -> torch.Tensor:
    """
    Full AutoGaze learned retention mask.

    If a pre-computed AutoGaze mask has been stored via AutoGazeContext,
    uses it directly (the actual nvidia/AutoGaze model output).

    Otherwise falls back to magnitude-based selection.

    For production use, pre-compute with AutoGazePreprocessor and store:
        prep = AutoGazePreprocessor.load("nvidia/AutoGaze")
        ag_mask, K = prep.compute_retention_mask(raw_frames, target_grid_hw=(rows, cols))
        with AutoGazeContext(ag_mask, K):
            outputs = llm.chat(...)
    """
    stored = get_raw_frames()

    # If we have a pre-computed AutoGaze mask stored in context, use it
    if stored is not None and isinstance(stored, dict) and "ag_mask" in stored:
        ag_mask = stored["ag_mask"]  # (T*H*W,) bool on CPU
        T, H, W = map(int, video_size_thw)
        rows, cols = H // spatial_merge_size, W // spatial_merge_size
        expected_len = T * rows * cols
        if ag_mask.numel() == expected_len:
            print(
                f"[AutoGaze-vLLM] Using pre-computed AutoGaze mask: "
                f"{ag_mask.sum().item()}/{expected_len} tokens kept",
                flush=True,
            )
            return ag_mask.to(video_embeds.device)
        else:
            print(
                f"[AutoGaze-vLLM] Mask size mismatch: got {ag_mask.numel()}, "
                f"expected {expected_len}. Falling back to magnitude.",
                flush=True,
            )

    return magnitude_retention_mask(video_embeds, video_size_thw, spatial_merge_size, q)
