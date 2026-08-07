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
    Context manager that stores raw video frames so the retention mask
    function can access them (AutoGaze needs raw frames, not ViT embeddings).

    Usage (in the vLLM model's _process_video_input before calling super):
        with AutoGazeContext(pixel_values_flat):
            embeddings = encode_with_vit(pixel_values_flat)
            # Inside the ViT call, compute_retention_mask will have access
            # to the raw frames via _ctx.raw_frames
    """

    def __init__(self, raw_frames: torch.Tensor):
        self.raw_frames = raw_frames

    def __enter__(self):
        _ctx.raw_frames = self.raw_frames
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

    If raw frames are available in AutoGazeContext, uses the actual
    nvidia/AutoGaze model to select patches.  Otherwise falls back to
    magnitude-based selection.

    For production use, call this inside AutoGazeContext:
        with AutoGazeContext(pixel_values_flat):
            ...vLLM model forward...
    """
    raw_frames = get_raw_frames()

    if raw_frames is None:
        # No raw frames in context → fall back to magnitude proxy
        return magnitude_retention_mask(video_embeds, video_size_thw, spatial_merge_size, q)

    ag_model = _load_autogaze()
    if ag_model is None:
        return magnitude_retention_mask(video_embeds, video_size_thw, spatial_merge_size, q)

    T, H, W = map(int, video_size_thw)
    rows, cols = H // spatial_merge_size, W // spatial_merge_size
    tokens_per_frame = rows * cols

    from vllm.multimodal.evs import compute_retained_tokens_count
    retain_num = compute_retained_tokens_count(tokens_per_frame, T, q)

    try:
        device = video_embeds.device
        with torch.no_grad():
            # AutoGaze forward pass on raw frames
            # The model outputs importance scores per patch
            raw_frames_dev = raw_frames.to(device)
            ag_output = ag_model(raw_frames_dev)
            # Expect (T, tokens_per_frame) importance scores
            if hasattr(ag_output, "logits"):
                scores = ag_output.logits.view(-1)
            else:
                scores = ag_output.view(-1)

        # Ensure first frame is always kept
        scores[:tokens_per_frame] = float("inf")

        topk_indices = torch.topk(scores, retain_num).indices
        mask = torch.zeros(video_embeds.shape[0], dtype=torch.bool, device=device)
        mask[topk_indices] = True
        return mask

    except Exception as e:
        print(f"[AutoGaze-vLLM] AutoGaze forward failed: {e}. Falling back to magnitude.")
        return magnitude_retention_mask(video_embeds, video_size_thw, spatial_merge_size, q)
