"""Per-question timing accumulator plus the CPU/CUDA forward-wrapping hooks
used to instrument both the NVILA model (vit_ms, llm_prefill_ms/llm_decode_ms)
and the vendored AutoGaze processor (see instrumentation.py).
"""
import time

import torch

TIMING_KEYS = [
    "decode_ms", "preprocess_videos_total_ms", "autogaze_transform_ms",
    "gazing_info_total_ms", "autogaze_model_ms",
    "vit_ms", "llm_prefill_ms", "llm_decode_ms", "llm_calls",
]

_timing = {k: 0.0 for k in TIMING_KEYS}


def reset() -> None:
    for k in TIMING_KEYS:
        _timing[k] = 0.0


def snapshot() -> dict:
    return dict(_timing)


def wrap_cpu_time(fn, key: str):
    """Wrap a plain function; accumulate wall-clock ms into _timing[key]."""
    def wrapped(*args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        _timing[key] += (time.time() - t0) * 1000
        return out
    return wrapped


def wrap_cuda_forward(module, key: str) -> None:
    """Replace module.forward in place with a cuda-event-timed version;
    accumulate elapsed GPU ms into _timing[key]."""
    orig_forward = module.forward

    def timed_forward(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = orig_forward(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        _timing[key] += start.elapsed_time(end)
        return out

    module.forward = timed_forward


def wrap_llm_prefill_decode(module) -> dict:
    """Like wrap_cuda_forward, but splits llm_ms into llm_prefill_ms (the
    first forward per generate() call, i.e. the full prompt) vs
    llm_decode_ms (every subsequent forward, one new token each via KV
    cache). Returns a state dict; the caller resets
    state["calls_since_reset"] = 0 immediately before each generate() call."""
    orig_forward = module.forward
    state = {"calls_since_reset": 0}

    def timed_forward(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = orig_forward(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        key = "llm_prefill_ms" if state["calls_since_reset"] == 0 else "llm_decode_ms"
        _timing[key] += ms
        state["calls_since_reset"] += 1
        _timing["llm_calls"] += 1
        return out

    module.forward = timed_forward
    return state


def install_model_hooks(model) -> dict:
    """Instrument the NVILA model's vision tower (vit_ms) and LLM
    (llm_prefill_ms/llm_decode_ms). Returns the LLM call-count state dict --
    reset its "calls_since_reset" to 0 before each generate() call."""
    wrap_cuda_forward(model.vision_tower, "vit_ms")
    return wrap_llm_prefill_decode(model.llm)
