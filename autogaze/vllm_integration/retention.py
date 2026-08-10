"""
Retention mask implementations for AutoGaze × vLLM integration.

Provides drop-in replacements for vLLM's EVS compute_retention_mask.

Three post-ViT modes:
  "evs"        — original EVS (cosine similarity between frames) — baseline
  "magnitude"  — magnitude-based importance (proof-of-concept, no extra model)
  "autogaze"   — full AutoGaze learned selection (requires nvidia/AutoGaze weights
                 AND raw frames stored in context via AutoGazeContext)

Task 1 — Adaptive K:
  autogaze_retained_tokens_count replaces vLLM's fixed compute_retained_tokens_count.
  When AutoGazeContext carries K, vLLM uses that count instead of the fixed formula,
  enabling per-video token budgets rather than a uniform ratio across all videos.

Tasks 2+3 — Sparse ViT pass-through:
  When AutoGazeContext is used with K set but ag_mask=None (sparse ViT mode),
  autogaze_retention_mask returns an identity mask (all-True) because patch
  selection already happened inside the ViT via gather op — no post-ViT pruning.
"""
from __future__ import annotations

import threading

import torch

_ctx = threading.local()

# Capture original vLLM function before any monkey-patching so fallbacks
# never recurse into our patched version.
try:
    from vllm.multimodal.evs import compute_retained_tokens_count as _vllm_retained_count
except ImportError:
    _vllm_retained_count = None


class AutoGazeContext:
    """
    Context manager that passes AutoGaze data into vLLM's retention hooks.

    Two modes:

    1. Post-ViT AutoGaze (original):
        with AutoGazeContext(ag_mask=mask, K=K):
            outputs = llm.chat(...)
        → compute_retention_mask uses ag_mask directly (which K patches to keep).
        → compute_retained_tokens_count returns K (adaptive, not fixed formula).

    2. Sparse ViT pass-through (Tasks 2+3):
        with AutoGazeContext(K=K_merged):   # ag_mask=None
            outputs = llm.chat(...)
        → ViT already ran sparse (via SparseViTContext + patch_sparse_vit).
        → compute_retained_tokens_count returns K_merged for correct slot allocation.
        → compute_retention_mask returns identity (all-True, no post-ViT pruning).
    """

    def __init__(
        self,
        ag_mask: torch.Tensor | None = None,
        K: int | None = None,
    ):
        if ag_mask is not None or K is not None:
            self.payload = {"ag_mask": ag_mask, "K": K}
        else:
            self.payload = None

    def __enter__(self):
        _ctx.payload = self.payload
        return self

    def __exit__(self, *args):
        _ctx.payload = None


def get_raw_frames() -> dict | None:
    return getattr(_ctx, "payload", None)


# ---------------------------------------------------------------------------
# Task 1: Adaptive K — replace fixed compute_retained_tokens_count
# ---------------------------------------------------------------------------

def autogaze_retained_tokens_count(
    tokens_per_frame: int,
    T: int = 0,
    q: float = 0.0,
    *,
    num_frames: int | None = None,  # vLLM ≥0.24 passes this kwarg instead of positional T
    **_kwargs,                       # absorb any other future kwargs gracefully
) -> int:
    """
    Adaptive K: return the token count from AutoGazeContext when available,
    otherwise fall back to vLLM's fixed formula.

    This is monkey-patched over vllm.multimodal.evs.compute_retained_tokens_count
    by apply_autogaze_patch(). It ensures that the vLLM scheduler allocates the
    right number of KV-cache slots for the actual AutoGaze selection, not the
    rounded-down fixed-formula value.

    vLLM 0.24+ calls with num_frames=N as a keyword argument; earlier versions
    pass T as the second positional argument. Both are handled here.
    """
    if num_frames is not None:
        T = num_frames

    stored = get_raw_frames()
    if stored is not None and stored.get("K") is not None:
        K = stored["K"]
        print(
            f"[AutoGaze-vLLM] Adaptive K={K} "
            f"(from AutoGaze context, overrides fixed formula for T={T} frames)",
            flush=True,
        )
        return K

    if _vllm_retained_count is None:
        raise RuntimeError("[AutoGaze-vLLM] vllm.multimodal.evs not available")
    # Forward with the right calling convention for the installed vLLM version
    try:
        return _vllm_retained_count(tokens_per_frame, T, q)
    except TypeError:
        return _vllm_retained_count(tokens_per_frame, num_frames=T, q=q)


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

    Uses vllm.multimodal.evs.compute_retained_tokens_count via module reference
    so it respects the monkey-patch from apply_autogaze_patch() — meaning it
    also benefits from adaptive K when AutoGazeContext is active.
    """
    import vllm.multimodal.evs as _evs_mod  # module ref, not local bind → patchable

    T, H, W = map(int, video_size_thw)
    rows, cols = H // spatial_merge_size, W // spatial_merge_size
    tokens_per_frame = rows * cols

    retain_num = _evs_mod.compute_retained_tokens_count(tokens_per_frame, T, q)

    importance = video_embeds.norm(dim=-1)  # (T*rows*cols,)

    # Force first frame in (same guarantee as EVS)
    rest_importance = importance.clone()
    rest_importance[:tokens_per_frame] = float("inf")
    topk_indices = torch.topk(rest_importance, retain_num).indices

    mask = torch.zeros(video_embeds.shape[0], dtype=torch.bool, device=video_embeds.device)
    mask[topk_indices] = True
    mask[:tokens_per_frame] = True  # always keep first frame

    # Clip to exactly retain_num
    selected = mask.nonzero(as_tuple=True)[0]
    if selected.numel() > retain_num:
        keep = torch.topk(importance[selected], retain_num).indices
        mask = torch.zeros_like(mask)
        mask[selected[keep]] = True

    return mask


# ---------------------------------------------------------------------------
# Full AutoGaze retention (Tasks 1+2+3)
# ---------------------------------------------------------------------------

_autogaze_model = None
_autogaze_model_path = "nvidia/AutoGaze"


def _load_autogaze():
    global _autogaze_model
    if _autogaze_model is not None:
        return _autogaze_model
    try:
        from transformers import AutoModel
        _autogaze_model = AutoModel.from_pretrained(
            _autogaze_model_path, trust_remote_code=True,
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
    AutoGaze retention mask — handles three cases via AutoGazeContext:

    Case A (sparse ViT pass-through, Tasks 2+3):
        ag_mask is None, K is set.
        The ViT already ran sparse (patch selection before encoding).
        Return all-True of size video_embeds.shape[0] — identity, no post-ViT pruning.

    Case B (post-ViT AutoGaze, Task 1):
        ag_mask is a (T*H*W,) bool tensor.
        Use pre-computed AutoGaze mask directly.
        If size doesn't match current video, fall back to magnitude.

    Case C (no context):
        Fall back to magnitude-based selection.
    """
    stored = get_raw_frames()

    if stored is not None:
        ag_mask = stored.get("ag_mask")
        K_stored = stored.get("K")

        # ── Case A: sparse ViT pass-through ──────────────────────────────────
        if ag_mask is None and K_stored is not None:
            K_actual = video_embeds.shape[0]
            print(
                f"[AutoGaze-vLLM] Sparse ViT pass-through: identity mask "
                f"({K_actual} tokens, ViT already selected pre-encoding)",
                flush=True,
            )
            return torch.ones(K_actual, dtype=torch.bool, device=video_embeds.device)

        # ── Case B: pre-computed AutoGaze mask ────────────────────────────────
        if ag_mask is not None:
            T, H, W = map(int, video_size_thw)
            rows, cols = H // spatial_merge_size, W // spatial_merge_size
            expected_len = T * rows * cols
            if ag_mask.numel() == expected_len:
                print(
                    f"[AutoGaze-vLLM] Using AutoGaze mask: "
                    f"{ag_mask.sum().item()}/{expected_len} tokens kept",
                    flush=True,
                )
                return ag_mask.to(video_embeds.device)
            print(
                f"[AutoGaze-vLLM] Mask size mismatch: got {ag_mask.numel()}, "
                f"expected {expected_len}. Falling back to magnitude.",
                flush=True,
            )

    # ── Case C: magnitude fallback ────────────────────────────────────────────
    return magnitude_retention_mask(video_embeds, video_size_thw, spatial_merge_size, q)
