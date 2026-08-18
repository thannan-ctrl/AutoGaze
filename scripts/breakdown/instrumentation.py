"""Monkey-patches the vendored (trust_remote_code) NVILAProcessor class and
module at runtime -- no source edits -- to time each preprocessing stage and
to short-circuit AutoGaze's CPU transform when it's provably unused.

See README.md for what each timed stage means and why the short-circuit is
safe.
"""
import importlib

import torch

from . import timing

_processor_module_patched = False
_skip_autogaze_transform_state = {"skip": False}


def _make_transform_shortcircuit(orig_fn):
    """When the active processor's config means AutoGaze is skipped for both
    tiles and thumbnails (e.g. dense mode: gazing_ratio=1,
    task_loss_requirement=None), NVILAProcessor._get_gazing_info_from_videos's
    skip branches never read pixel_values_videos_{tiles,thumbnails}_autogaze
    -- confirmed by reading that method: the `videos_inputs.get(...)` calls
    for those keys only happen in the non-skip branch. So the real CPU
    resize/normalize/transform is dead work in that case; substitute a
    correctly-shaped zero tensor instead of computing it.
    transform_video_for_pytorch/AutoGazeImageProcessor.preprocess are pure
    functions (no side effects on `transform` or global state), so this
    substitution is behavior-preserving whenever the flag is set -- verified
    empirically too: dense-mode predictions were byte-identical across all 25
    questions with the short-circuit on vs off.
    """
    def shortcircuited(video_np, transform):
        if _skip_autogaze_transform_state["skip"]:
            n = video_np.shape[0]
            h, w = transform.size["height"], transform.size["width"]
            return torch.zeros((n, 3, h, w), dtype=torch.float32)
        return orig_fn(video_np, transform)
    return shortcircuited


def instrument(processor) -> None:
    """Instrument a freshly-built NVILAProcessor instance: patch its class
    and module once (idempotent across instances, since they're re-imported
    from the same dynamically-loaded module), then wrap this instance's own
    AutoGaze model and set the short-circuit flag from its config."""
    global _processor_module_patched
    cls = type(processor)
    proc_module = importlib.import_module(cls.__module__)

    if not _processor_module_patched:
        proc_module._load_video_frames = timing.wrap_cpu_time(proc_module._load_video_frames, "decode_ms")
        proc_module.transform_video_for_pytorch = timing.wrap_cpu_time(
            _make_transform_shortcircuit(proc_module.transform_video_for_pytorch), "autogaze_transform_ms"
        )
        cls._preprocess_videos = timing.wrap_cpu_time(cls._preprocess_videos, "preprocess_videos_total_ms")
        cls._get_gazing_info_from_videos = timing.wrap_cpu_time(
            cls._get_gazing_info_from_videos, "gazing_info_total_ms"
        )
        _processor_module_patched = True

    if processor._autogaze_model is not None:
        timing.wrap_cuda_forward(processor._autogaze_model, "autogaze_model_ms")

    skip_tiles = cls._should_gaze_all_patches(processor.gazing_ratio_tile, processor.task_loss_requirement_tile)
    skip_thumbs = cls._should_gaze_all_patches(
        processor.gazing_ratio_thumbnail, processor.task_loss_requirement_thumbnail
    )
    _skip_autogaze_transform_state["skip"] = skip_tiles and skip_thumbs
