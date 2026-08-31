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
_codec_state = {"enabled": False, "video_path": None}


def set_codec_video_context(video_path: str) -> None:
    """Called by runner.py right before `proc(...)` when mode == 'codec', so the
    patched _get_gazing_info_from_videos knows which video to score. See
    codec_selector.py and HEVC_Dump_Pipeline.md."""
    _codec_state["video_path"] = video_path


def _make_gazing_info_codec_override(orig_fn):
    """When codec mode is active, bypass the real AutoGaze selector entirely and
    substitute a codec-scored gazing_info dict for the single video set via
    set_codec_video_context -- built independently from video_path (see
    codec_selector.build_gazing_info), not from `videos_inputs` (which by this
    point has lost the original video path / tile crop-box bookkeeping).

    Reads num_video_frames/num_video_frames_thumbnail/max_tiles_video off the
    processor instance itself (`self`), not from the caller's kwargs dict --
    processor.build() mutates its own local copy of these per retry budget `nf`,
    so `self.<attr>` is the only value guaranteed to match what
    `_preprocess_videos` actually used for this exact call.
    """
    def overridden(self, videos_inputs):
        if not _codec_state["enabled"]:
            return orig_fn(self, videos_inputs)
        from . import codec_selector

        image_size = (
            self.image_processor.size.get("height", 392) if hasattr(self.image_processor, "size") else 392
        )
        return codec_selector.build_gazing_info(
            video_path=_codec_state["video_path"],
            num_video_frames=self.num_video_frames,
            num_video_frames_thumbnail=self.num_video_frames_thumbnail,
            max_tiles_video=self.max_tiles_video,
            autogaze_max_num_frames=self._autogaze_model.config.max_num_frames,
            image_size=image_size,
            scales=self.target_scales,
            patch_size=self.target_patch_size,
            gazing_ratio_tile=self.gazing_ratio_tile,
            gazing_ratio_thumbnail=self.gazing_ratio_thumbnail,
        )
    return overridden


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


def instrument(processor, mode: str | None = None) -> None:
    """Instrument a freshly-built NVILAProcessor instance: patch its class
    and module once (idempotent across instances, since they're re-imported
    from the same dynamically-loaded module), then wrap this instance's own
    AutoGaze model and set the short-circuit flags from its config / mode."""
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
            _make_gazing_info_codec_override(cls._get_gazing_info_from_videos), "gazing_info_total_ms"
        )
        _processor_module_patched = True

    if processor._autogaze_model is not None:
        timing.wrap_cuda_forward(processor._autogaze_model, "autogaze_model_ms")

    skip_tiles = cls._should_gaze_all_patches(processor.gazing_ratio_tile, processor.task_loss_requirement_tile)
    skip_thumbs = cls._should_gaze_all_patches(
        processor.gazing_ratio_thumbnail, processor.task_loss_requirement_thumbnail
    )
    _codec_state["enabled"] = mode == "codec"
    # codec mode never reads pixel_values_videos_{tiles,thumbnails}_autogaze (it
    # scores from the original video via codec_selector, not from these pixels),
    # so the CPU transform producing them is dead work here too.
    _skip_autogaze_transform_state["skip"] = (skip_tiles and skip_thumbs) or mode == "codec"
