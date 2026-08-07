"""
Patches vLLM's EVS compute_retention_mask with AutoGaze.

Call apply_autogaze_patch() once at process startup, before loading
the vLLM model. After that, all vLLM models that use EVS (NanoNemotron,
Qwen2.5-VL, Qwen3-VL) will automatically use the AutoGaze retention mask.
"""
from __future__ import annotations

import vllm.multimodal.evs as _evs_module
from autogaze.vllm_integration.retention import (
    autogaze_retention_mask,
    evs_retention_mask,
    magnitude_retention_mask,
)

_ORIGINAL_FN = _evs_module.compute_retention_mask
_PATCHED = False
_ACTIVE_MODE = "evs"


def apply_autogaze_patch(mode: str = "magnitude") -> None:
    """
    Replace vLLM's EVS retention mask with AutoGaze.

    Args:
        mode: One of:
            "evs"       — keep original EVS (cosine similarity) — useful for baseline
            "magnitude" — magnitude-based proxy (no extra model)
            "autogaze"  — full AutoGaze (requires raw frames in AutoGazeContext)
    """
    global _PATCHED, _ACTIVE_MODE

    fn_map = {
        "evs": _ORIGINAL_FN,
        "magnitude": magnitude_retention_mask,
        "autogaze": autogaze_retention_mask,
    }
    if mode not in fn_map:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(fn_map)}")

    _evs_module.compute_retention_mask = fn_map[mode]
    _PATCHED = (mode != "evs")
    _ACTIVE_MODE = mode

    print(f"[AutoGaze-vLLM] compute_retention_mask → {mode}")


def restore_evs() -> None:
    """Restore original EVS behaviour."""
    global _PATCHED, _ACTIVE_MODE
    _evs_module.compute_retention_mask = _ORIGINAL_FN
    _PATCHED = False
    _ACTIVE_MODE = "evs"
    print("[AutoGaze-vLLM] Restored original EVS compute_retention_mask")


def active_mode() -> str:
    return _ACTIVE_MODE
