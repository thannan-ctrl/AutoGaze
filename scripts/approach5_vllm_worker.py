#!/usr/bin/env python3
"""
Approach 5 — AutoGaze × vLLM integration worker.
Runs INSIDE the vLLM Docker container with CLI args: --mode --pruning-rate --reps

Modes:
  dense       — no compression, all visual tokens
  evs         — built-in vLLM EVS (cosine similarity)
  magnitude   — magnitude-based proxy (no extra model); adaptive K (Task 1)
  autogaze    — post-ViT AutoGaze mask + adaptive K (Task 1)
  sparse_vit  — pre-ViT gather op + sparse ViT blocks + adaptive K (Tasks 1+2+3)

Timing outputs (in RESULT_JSON):
  load_ms     — model load time
  elapsed_ms  — first rep total inference time  (vit + lm)
  vit_ms      — ViT encoder time  (CUDA events, from timing hook)
  lm_ms       — elapsed_ms - vit_ms
  reps        — list of per-rep {elapsed_ms, vit_ms, num_prompt_tokens, answer}

Usage:
    python approach5_vllm_worker.py --mode sparse_vit --pruning-rate 0.5 --reps 3
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
PROMPT = (
    "Question: What does the white text on the green road sign say?\n"
    "A. Hampden St\nB. Hampden Ave\nC. HampdenBlvd\nD. Hampden Rd\n"
    "Please answer directly with the letter of the correct answer."
)

QWEN_GRID_HW     = (16, 16)   # post-merge: for autogaze mask
QWEN_VIT_GRID_HW = (32, 32)   # pre-merge:  for sparse_vit mask
QWEN_MERGE_FACTOR = 4          # 2×2 spatial merge


def load_video_frames(video_path: str, fps: float = 2.0, max_frames: int = 32):
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
                        choices=["dense", "evs", "magnitude", "autogaze", "sparse_vit"])
    parser.add_argument("--pruning-rate", type=float, default=0.5)
    parser.add_argument("--reps", type=int, default=1,
                        help="Number of inference repetitions (first is warmup, rest are measured)")
    args = parser.parse_args()

    mode = args.mode
    pruning_rate = args.pruning_rate
    n_reps = max(1, args.reps)

    print(f"[approach5] mode={mode} pruning_rate={pruning_rate} reps={n_reps}", flush=True)

    # ── Apply retention mask patch BEFORE loading vLLM ────────────────────────
    ag_mask = None
    ag_K = None
    mask_vit = None
    K_vit = None
    K_merged = None

    if mode == "autogaze":
        mask_path = os.environ.get("AUTOGAZE_MASK_PATH", "/tmp/ag_mask.pt")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"AutoGaze mask not found at {mask_path}. "
                "Run run_autogaze_preprocess.py first."
            )
        import torch as _torch
        payload = _torch.load(mask_path, weights_only=True)
        ag_mask = payload["mask"]
        ag_K = int(payload["K"])
        print(f"[approach5] Loaded AutoGaze post-ViT mask: K={ag_K}", flush=True)
        from autogaze.vllm_integration.patch import apply_autogaze_patch
        apply_autogaze_patch(mode="autogaze")

    elif mode == "sparse_vit":
        mask_path = os.environ.get("AUTOGAZE_MASK_PATH", "/tmp/ag_mask_vit.pt")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"AutoGaze ViT-level mask not found at {mask_path}. "
                "Run run_autogaze_preprocess.py --grid-hw 32 32 first."
            )
        import torch as _torch
        payload = _torch.load(mask_path, weights_only=True)
        mask_vit = payload["mask"]
        K_vit = int(payload["K"])
        K_merged = K_vit // QWEN_MERGE_FACTOR
        print(f"[approach5] Loaded AutoGaze ViT-level mask: K_vit={K_vit}, K_merged≈{K_merged}", flush=True)
        from autogaze.vllm_integration.patch import apply_autogaze_patch
        apply_autogaze_patch(mode="autogaze")

    elif mode == "magnitude":
        from autogaze.vllm_integration.patch import apply_autogaze_patch
        apply_autogaze_patch(mode="magnitude")

    # ── Build vLLM engine ──────────────────────────────────────────────────────
    from vllm import LLM, SamplingParams

    # Class-level sparse ViT patch — apply AFTER `from vllm import LLM`
    # so vLLM's lazy model-module imports are already resolved, but BEFORE
    # LLM() instantiation so the class patch is inherited by EngineCore.
    if mode == "sparse_vit":
        from autogaze.vllm_integration.sparse_vit import patch_sparse_vit as _patch_cls
        _patch_cls(llm=None)

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

    # ── Tasks 2+3: instance-level patch attempt (works if executor is in-process) ─
    if mode == "sparse_vit":
        from autogaze.vllm_integration.sparse_vit import patch_sparse_vit
        encoder = patch_sparse_vit(llm)   # no-op if class patch already applied above
        if encoder is None:
            print("[approach5] WARNING: visual encoder patch unavailable — sparse_vit runs class-patched or falls back to post-ViT.", flush=True)

    # ── CUDA event timing hook on ViT (all non-dense modes) ──────────────────
    from autogaze.vllm_integration.sparse_vit import patch_vit_timing, get_vit_ms
    if mode != "dense":
        patch_vit_timing(llm)  # wraps current encoder.forward (after sparse patch if applied)

    # ── Build inference inputs ────────────────────────────────────────────────
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

    from autogaze.vllm_integration.retention import AutoGazeContext

    def _run_once():
        """Run one inference pass; returns (elapsed_ms, vit_ms, outputs)."""
        if mode == "autogaze":
            t0 = time.perf_counter()
            with AutoGazeContext(ag_mask=ag_mask, K=ag_K):
                outs = llm.chat(messages, sampling_params=sampling)
        elif mode == "sparse_vit":
            from autogaze.vllm_integration.sparse_vit import SparseViTContext
            T_frames = mask_vit.numel() // (QWEN_VIT_GRID_HW[0] * QWEN_VIT_GRID_HW[1])
            grid_thw = (T_frames, QWEN_VIT_GRID_HW[0], QWEN_VIT_GRID_HW[1])
            t0 = time.perf_counter()
            with SparseViTContext(mask=mask_vit, K=K_vit, grid_thw=grid_thw):
                with AutoGazeContext(ag_mask=None, K=K_merged):
                    outs = llm.chat(messages, sampling_params=sampling)
        else:
            t0 = time.perf_counter()
            outs = llm.chat(messages, sampling_params=sampling)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        vit_ms = get_vit_ms()
        return elapsed_ms, vit_ms, outs

    # ── Run reps ──────────────────────────────────────────────────────────────
    rep_results = []
    for rep in range(n_reps):
        label = "warmup" if rep == 0 and n_reps > 1 else f"rep {rep+1}/{n_reps}"
        print(f"[approach5] {label} ...", flush=True)
        elapsed_ms, vit_ms, outputs = _run_once()
        answer = outputs[0].outputs[0].text.strip()
        num_tokens = len(outputs[0].prompt_token_ids)
        rep_results.append({
            "rep": rep,
            "elapsed_ms": elapsed_ms,
            "vit_ms": vit_ms,
            "lm_ms": elapsed_ms - vit_ms if vit_ms is not None else None,
            "num_prompt_tokens": num_tokens,
            "answer": answer,
        })
        print(
            f"[approach5]   tokens={num_tokens} elapsed={elapsed_ms:.0f}ms "
            f"vit={vit_ms:.0f}ms answer={answer}"
            if vit_ms else
            f"[approach5]   tokens={num_tokens} elapsed={elapsed_ms:.0f}ms answer={answer}",
            flush=True,
        )

    # Use rep 0 as the primary (or only) result; measured reps are rep_results[1:]
    primary = rep_results[0]
    measured = rep_results[1:] if n_reps > 1 else rep_results

    def _avg(key):
        vals = [r[key] for r in measured if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    result = {
        "mode": mode,
        "pruning_rate": pruning_rate,
        "answer": primary["answer"],
        "num_prompt_tokens": primary["num_prompt_tokens"],
        "load_ms": load_ms,
        # First rep (or only rep) timing
        "elapsed_ms": primary["elapsed_ms"],
        "vit_ms": primary["vit_ms"],
        "lm_ms": primary["lm_ms"],
        # Averages over measured reps (excludes warmup if n_reps > 1)
        "avg_elapsed_ms": _avg("elapsed_ms"),
        "avg_vit_ms": _avg("vit_ms"),
        "avg_lm_ms": _avg("lm_ms"),
        # Sparse ViT specifics
        "autogaze_K": ag_K,
        "K_vit": K_vit,
        "K_merged": K_merged,
        # All reps for post-processing
        "reps": rep_results,
    }
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
