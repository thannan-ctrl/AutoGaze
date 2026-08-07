#!/usr/bin/env python3
"""
Approach 5 — AutoGaze × vLLM integration worker.
Runs INSIDE the vLLM Docker container with CLI args: --mode --pruning-rate

Modes:
  dense     — no compression, all visual tokens
  evs       — built-in vLLM EVS (cosine similarity)
  magnitude — magnitude-based proxy (no extra model)
  autogaze  — full nvidia/AutoGaze model (pre-ViT, learned selection)

Usage:
    python approach5_vllm_worker.py --mode autogaze --pruning-rate 0.5
"""
import argparse
import json
import os
import sys
import time

REPO_DIR = os.environ.get("REPO_DIR", "/workspace/AutoGaze")
HF_HOME = os.environ.get("HF_HOME", "/root/.cache/huggingface")
sys.path.insert(0, REPO_DIR)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["HF_HOME"] = HF_HOME

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
VIDEO_PATH = os.path.join(REPO_DIR, "assets", "example_input.mp4")
AUTOGAZE_MODEL_ID = "nvidia/AutoGaze"
PROMPT = (
    "Question: What does the white text on the green road sign say?\n"
    "A. Hampden St\nB. Hampden Ave\nC. HampdenBlvd\nD. Hampden Rd\n"
    "Please answer directly with the letter of the correct answer."
)

# Qwen3-VL post-merge patch grid for 448×448 input with 14px patches + merge_size=2
# 448/14=32 patches/side → /2 = 16 merged patches/side
QWEN_GRID_HW = (16, 16)


def load_video_frames(video_path: str, fps: float = 2.0, max_frames: int = 32):
    """Load video as (T, C, H, W) float32 [0,1] using best available backend."""
    import torch
    import numpy as np

    def _torchcodec(p):
        from torchcodec.decoders import VideoDecoder
        dec = VideoDecoder(p)
        meta = dec.get_metadata()
        src_fps = meta.average_fps or 30.0
        step = max(1, int(src_fps / fps))
        total = meta.num_frames or 1000
        indices = list(range(0, min(total, max_frames * step), step))[:max_frames]
        return dec.get_frames_at(indices=indices).data.float() / 255.0

    def _ffmpeg(p):
        import subprocess
        w, h = 448, 448
        cmd = ["ffmpeg", "-i", p, "-vf", f"fps={fps},scale={w}:{h}",
               "-frames:v", str(max_frames), "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-loglevel", "error", "-"]
        data = subprocess.run(cmd, capture_output=True).stdout
        n = len(data) // (h * w * 3)
        arr = np.frombuffer(data[:n * h * w * 3], dtype=np.uint8).reshape(n, h, w, 3)
        return torch.from_numpy(arr.copy()).permute(0, 3, 1, 2).float() / 255.0

    for name, fn in [("torchcodec", _torchcodec), ("ffmpeg", _ffmpeg)]:
        try:
            frames = fn(video_path)
            print(f"[approach5] Video loaded via {name}: {tuple(frames.shape)}", flush=True)
            return frames
        except Exception as e:
            print(f"[approach5] {name} failed: {e}", flush=True)
    raise RuntimeError("No video loader worked")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="dense",
                        choices=["dense", "evs", "magnitude", "autogaze"])
    parser.add_argument("--pruning-rate", type=float, default=0.5)
    args = parser.parse_args()

    mode = args.mode
    pruning_rate = args.pruning_rate

    print(f"[approach5] mode={mode} pruning_rate={pruning_rate}", flush=True)

    # ── Step 1: Apply retention mask patch BEFORE loading vLLM ────────────
    ag_mask = None
    ag_K = None

    if mode == "autogaze":
        # Install AutoGaze deps that may be missing from the vLLM container
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "omegaconf", "einops", "timm", "-q"],
            check=True
        )
        # Load the real AutoGaze model and preprocess the video
        from autogaze.vllm_integration.autogaze_preprocess import AutoGazePreprocessor
        from autogaze.vllm_integration.patch import apply_autogaze_patch
        apply_autogaze_patch(mode="autogaze")

        print(f"[approach5] Running AutoGaze preprocessing ...", flush=True)
        raw_frames = load_video_frames(VIDEO_PATH)
        prep = AutoGazePreprocessor.load(AUTOGAZE_MODEL_ID)
        ag_mask, ag_K = prep.compute_retention_mask(
            raw_frames,
            target_grid_hw=QWEN_GRID_HW,
            gazing_ratio=pruning_rate,
        )
        print(f"[approach5] AutoGaze selected K={ag_K} tokens", flush=True)
        del prep  # free GPU memory before loading vLLM

    elif mode == "magnitude":
        from autogaze.vllm_integration.patch import apply_autogaze_patch
        apply_autogaze_patch(mode="magnitude")

    # EVS: no patch needed (vLLM built-in, enabled via video_pruning_rate)
    # Dense: no patch

    # ── Step 2: Build vLLM engine ──────────────────────────────────────────
    from vllm import LLM, SamplingParams

    # video_pruning_rate activates EVS / retention mask in vLLM.
    # enforce_eager required: dynamic token reduction breaks CUDA graph capture.
    extra_kwargs = {}
    if mode != "dense":
        extra_kwargs["video_pruning_rate"] = pruning_rate
        extra_kwargs["enforce_eager"] = True

    print(f"[approach5] Loading {MODEL_ID} ...", flush=True)
    t_load = time.perf_counter()
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        gpu_memory_utilization=0.7,
        max_model_len=8192,
        limit_mm_per_prompt={"video": 1},
        allowed_local_media_path="/workspace",
        **extra_kwargs,
    )
    load_ms = (time.perf_counter() - t_load) * 1000
    print(f"[approach5] Model loaded in {load_ms:.0f} ms", flush=True)

    # ── Step 3: Run inference ──────────────────────────────────────────────
    video_url = f"file://{VIDEO_PATH}"
    messages = [{
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {
                "url": video_url, "fps": 2.0, "max_pixels": 448 * 448, "nframes": 32,
            }},
            {"type": "text", "text": PROMPT},
        ],
    }]

    sampling = SamplingParams(max_tokens=10, temperature=0.0)

    if mode == "autogaze" and ag_mask is not None:
        # Inject the real AutoGaze mask into the retention context
        from autogaze.vllm_integration.retention import AutoGazeContext
        print(f"[approach5] Injecting AutoGaze mask (K={ag_K}) into vLLM context ...", flush=True)
        t0 = time.perf_counter()
        with AutoGazeContext(ag_mask=ag_mask, K=ag_K):
            outputs = llm.chat(messages, sampling_params=sampling)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    else:
        print(f"[approach5] Running inference via llm.chat() ...", flush=True)
        t0 = time.perf_counter()
        outputs = llm.chat(messages, sampling_params=sampling)
        elapsed_ms = (time.perf_counter() - t0) * 1000

    answer = outputs[0].outputs[0].text.strip()
    num_prompt_tokens = len(outputs[0].prompt_token_ids)

    result = {
        "mode": mode,
        "pruning_rate": pruning_rate,
        "answer": answer,
        "num_prompt_tokens": num_prompt_tokens,
        "elapsed_ms": elapsed_ms,
        "load_ms": load_ms,
        "autogaze_K": ag_K,
    }
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
