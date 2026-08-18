"""Builds and instruments NVILAProcessor instances for a given mode."""
from transformers import AutoProcessor

from . import config
from .instrumentation import instrument


def build(mode: str, num_video_frames: int):
    kw = {**config.COMMON_KW, **config.CONFIGS[mode]}
    kw["num_video_frames"] = num_video_frames
    kw["num_video_frames_thumbnail"] = max(num_video_frames // 2, 1)
    kw["max_tiles_video"] = num_video_frames
    proc = AutoProcessor.from_pretrained(config.MODEL_PATH, trust_remote_code=True, **kw)
    instrument(proc)
    return proc
