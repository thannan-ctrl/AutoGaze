"""
Patches vLLM's EVS hooks with AutoGaze.

Call apply_autogaze_patch() once at process startup, before loading the vLLM model.
After that, all vLLM models that use EVS (NanoNemotron, Qwen2.5-VL, Qwen3-VL)
will automatically use the AutoGaze retention mask and adaptive K.

Two functions are patched in vllm.multimodal.evs:
  compute_retention_mask      — which patches to keep (selection)
  compute_retained_tokens_count — how many patches to keep (adaptive K, Task 1)
"""
from __future__ import annotations

import vllm.multimodal.evs as _evs_module

from autogaze.vllm_integration.retention import (
    autogaze_retained_tokens_count,
    autogaze_retention_mask,
    evs_retention_mask,
    magnitude_retention_mask,
)

_ORIGINAL_MASK_FN = _evs_module.compute_retention_mask
_ORIGINAL_COUNT_FN = _evs_module.compute_retained_tokens_count
_PATCHED = False
_ACTIVE_MODE = "evs"


def apply_autogaze_patch(mode: str = "magnitude") -> None:
    """
    Replace vLLM's EVS hooks with AutoGaze.

    Args:
        mode: One of:
            "evs"       — keep original EVS (cosine similarity) — baseline
            "magnitude" — magnitude-based proxy (no extra model)
            "autogaze"  — full AutoGaze (requires AutoGazeContext with pre-computed mask)

    Both hooks are patched for "magnitude" and "autogaze" modes:
      - compute_retention_mask      → selects which patches to keep
      - compute_retained_tokens_count → adaptive K (Task 1): returns AutoGaze's K
                                        from context instead of fixed formula
    """
    global _PATCHED, _ACTIVE_MODE

    mask_fn_map = {
        "evs": _ORIGINAL_MASK_FN,
        "magnitude": magnitude_retention_mask,
        "autogaze": autogaze_retention_mask,
    }
    if mode not in mask_fn_map:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(mask_fn_map)}")

    _evs_module.compute_retention_mask = mask_fn_map[mode]

    # Task 1: also patch compute_retained_tokens_count so vLLM allocates
    # the right number of KV-cache slots for the actual AutoGaze selection.
    if mode in ("magnitude", "autogaze"):
        _evs_module.compute_retained_tokens_count = autogaze_retained_tokens_count
        print(f"[AutoGaze-vLLM] compute_retained_tokens_count → autogaze_retained_tokens_count (adaptive K)")
    else:
        _evs_module.compute_retained_tokens_count = _ORIGINAL_COUNT_FN

    _PATCHED = (mode != "evs")
    _ACTIVE_MODE = mode
    print(f"[AutoGaze-vLLM] compute_retention_mask → {mode}")


def restore_evs() -> None:
    """Restore original EVS behaviour (both hooks)."""
    global _PATCHED, _ACTIVE_MODE
    _evs_module.compute_retention_mask = _ORIGINAL_MASK_FN
    _evs_module.compute_retained_tokens_count = _ORIGINAL_COUNT_FN
    _PATCHED = False
    _ACTIVE_MODE = "evs"
    print("[AutoGaze-vLLM] Restored original EVS compute_retention_mask + compute_retained_tokens_count")


def active_mode() -> str:
    return _ACTIVE_MODE
