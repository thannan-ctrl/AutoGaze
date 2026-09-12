"""Monkey-patches the vendored (trust_remote_code) NVILAProcessor class and
module at runtime -- no source edits -- to time each preprocessing stage and
to short-circuit AutoGaze's CPU transform when it's provably unused.

See README.md for what each timed stage means and why the short-circuit is
safe.
"""
import importlib

import torch
from transformers import LogitsProcessor

from . import config, timing

# Which scale (from the processor's own target_scales ladder) "autogaze_singlescale"
# restricts generation to -- see SingleScaleLogitsProcessor below for why this is a
# generation-time mask rather than a target_scales=[N] reconfiguration. 224 chosen as
# a literature-standard "default resolution" (ImageNet-style 224x224), not because
# anything else about it is special -- any of the model's other scales would work
# equally well as the ablation's single-scale choice.
SINGLE_SCALE_TARGET = 224


def _scale_token_range(scales, patch_size, target_scale):
    """[lo, hi) vocab-index range for one scale within AutoGaze's flat
    multi-scale vocabulary (scales concatenated in ladder order, patch-grid-major
    -- same convention as codec_selector.py's total_patches/grid_sizes)."""
    grid_sizes = [s // patch_size for s in scales]
    offset = 0
    for s, g in zip(scales, grid_sizes):
        if s == target_scale:
            return offset, offset + g * g
        offset += g * g
    raise ValueError(f"scale {target_scale} not in {scales}")


class SingleScaleLogitsProcessor(LogitsProcessor):
    """Restricts AutoGaze's autoregressive generation to one scale's token range
    within its flat multi-scale vocabulary -- Next Steps' "test AutoGaze with a
    single resolution scale" ablation.

    Why a generation-time mask instead of reconfiguring target_scales=[N]: AutoGaze
    is a *trained* model whose classification head has a fixed output dimension
    matching the full 4-scale vocab (56/112/224/448 at patch_size=16 by default)
    it was trained on. Passing a single-scale target_scales into the processor
    would change vocab_size and break at the model's output layer (or silently
    misalign token meanings even if shapes coincidentally matched) -- there's no
    way to get a faithful single-scale *trained* baseline without retraining.
    Masking at generation time keeps the model and its trained weights completely
    unchanged; it only removes 3 of the 4 scales' tokens as *choices* the existing
    autoregressive policy is allowed to make, at every generation step, alongside
    the pre-existing NoRepeatTokensLogitsProcessor/NoEosTokenLogitsProcessor this
    list already carries (see autogaze/models/autogaze/modeling_autogaze.py)."""

    def __init__(self, lo: int, hi: int):
        super().__init__()
        self.lo = lo
        self.hi = hi

    def __call__(self, input_ids, scores):
        mask = torch.ones(scores.shape[-1], dtype=torch.bool, device=scores.device)
        mask[self.lo:self.hi] = False
        if scores.ndim == 3:
            scores[:, :, mask] = -float("inf")
        else:
            scores[:, mask] = -float("inf")
        return scores

_processor_module_patched = False
_skip_autogaze_transform_state = {"skip": False}
# w_motion/skip_penalty/w_size/w_residual/full_first_frame/sampled_only come
# straight from config.CODEC_SCORE_KW (env-var-derived, read once at process
# start) -- these are static per-run settings, unlike video_path, so no
# per-question update. backend defaults to libde265 and is set per-mode in
# instrument() below (see codec_selector.BACKEND_FOR_MODE).
_codec_state = {"enabled": False, "video_path": None, "backend": "libde265", **config.CODEC_SCORE_KW}
_last_gazing_state = {"num_gazing_each_frame_tiles": None}


def set_codec_video_context(video_path: str, backend: str = "libde265") -> None:
    """Called by runner.py right before `proc(...)` for codec modes, so the
    patched _get_gazing_info_from_videos knows which video to score and which
    dump backend to use. See codec_selector.py."""
    _codec_state["video_path"] = video_path
    _codec_state["backend"] = backend


def reset_last_gazing_state() -> None:
    """Call before each proc(...) invocation so a stale capture from a prior
    question (e.g. one that raised before reaching gazing) can't leak into
    the next question's result."""
    _last_gazing_state["num_gazing_each_frame_tiles"] = None


def get_last_num_gazing_each_frame_tiles():
    """Returns the (num_tiles, T_tile) CPU tensor of *realized* per-tile,
    per-frame-position patch counts from the most recent
    _get_gazing_info_from_videos call (either mode), or None if it wasn't
    captured (e.g. the call returned None). Used to check whether AutoGaze's
    actual (possibly early-stopped) per-frame-position allocation matches
    the nominal gazing_ratio_tile schedule that codec mode statically fills
    -- see Codec_Patch_Selection_Theory.md's "ratio schedule is not
    codec-derived" caveat."""
    return _last_gazing_state["num_gazing_each_frame_tiles"]


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
            result = orig_fn(self, videos_inputs)
        else:
            from . import codec_selector

            image_size = (
                self.image_processor.size.get("height", 392) if hasattr(self.image_processor, "size") else 392
            )
            result = codec_selector.build_gazing_info(
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
                w_motion=_codec_state["w_motion"],
                skip_penalty=_codec_state["skip_penalty"],
                w_size=_codec_state["w_size"],
                w_residual=_codec_state["w_residual"],
                full_first_frame=_codec_state["full_first_frame"],
                sampled_only=_codec_state["sampled_only"],
                gop_restart=_codec_state["gop_restart"],
                backend=_codec_state["backend"],
            )
        # Capture the realized per-tile, per-frame-position patch counts (not
        # just the nominal ratio schedule) so callers can check whether
        # AutoGaze's actual early-stopped allocation matches the schedule
        # codec mode statically fills -- see reset_last_gazing_state()/
        # get_last_num_gazing_each_frame_tiles() above.
        if result is not None:
            ng = result.get("num_gazing_each_frame_tiles")
            if ng:
                _last_gazing_state["num_gazing_each_frame_tiles"] = ng[0].detach().cpu()
        return result
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
        if mode == "autogaze_singlescale":
            lo, hi = _scale_token_range(processor.target_scales, processor.target_patch_size, SINGLE_SCALE_TARGET)
            processor._autogaze_model.gazing_model.logits_processor.append(SingleScaleLogitsProcessor(lo, hi))

    skip_tiles = cls._should_gaze_all_patches(processor.gazing_ratio_tile, processor.task_loss_requirement_tile)
    skip_thumbs = cls._should_gaze_all_patches(
        processor.gazing_ratio_thumbnail, processor.task_loss_requirement_thumbnail
    )
    from . import codec_selector
    backend = codec_selector.BACKEND_FOR_MODE.get(mode)
    _codec_state["enabled"] = backend is not None
    _codec_state["backend"] = backend or "libde265"
    # codec modes never read pixel_values_videos_{tiles,thumbnails}_autogaze (they
    # score from the original video via codec_selector, not from these pixels),
    # so the CPU transform producing them is dead work here too.
    _skip_autogaze_transform_state["skip"] = (skip_tiles and skip_thumbs) or _codec_state["enabled"]
