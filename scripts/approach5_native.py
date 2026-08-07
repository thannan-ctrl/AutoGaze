#!/usr/bin/env python3
"""
Approach 5 (native) — AutoGaze × vLLM in the auto_gaze conda env.

Runs all 4 modes in a single environment:
  dense      — no compression
  evs        — vLLM built-in EVS (cosine similarity)
  magnitude  — AutoGaze-inspired magnitude selection
  autogaze   — actual nvidia/AutoGaze learned model (pre-ViT)

Run with:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/scratch.thannan_wwfo/miniforge-aarch64/envs/auto_gaze/bin/python \
  scripts/approach5_native.py
"""
import json
import os
import subprocess
import sys
import time

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/home/scratch.thannan_wwfo/hf_cache")

PYTHON = "/home/scratch.thannan_wwfo/miniforge-aarch64/envs/auto_gaze/bin/python"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
VIDEO_PATH = os.path.join(REPO_DIR, "assets", "example_input.mp4")
AUTOGAZE_MODEL_ID = "nvidia/AutoGaze"
PRUNING_RATE = 0.5
QWEN_GRID_HW = (16, 16)
PROMPT = (
    "Question: What does the white text on the green road sign say?\n"
    "A. Hampden St\nB. Hampden Ave\nC. HampdenBlvd\nD. Hampden Rd\n"
    "Please answer directly with the letter of the correct answer."
)

WORKER_CODE = '''
import os, sys, json, time, subprocess, types
import torch

REPO_DIR = {repo_dir!r}
sys.path.insert(0, REPO_DIR)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/home/scratch.thannan_wwfo/hf_cache")

mode = {mode!r}
pruning_rate = {pruning_rate}
video_path = {video_path!r}
model_id = {model_id!r}

# ── Mock wandb (training dep) ────────────────────────────────────────────
def _mock(name, **attrs):
    if name not in sys.modules:
        m = types.ModuleType(name)
        for k, v in attrs.items(): setattr(m, k, v)
        sys.modules[name] = m
_mock("wandb", run=None, log=lambda *a,**k:None, init=lambda *a,**k:None)

# ── Apply AutoGaze patch ─────────────────────────────────────────────────
ag_mask = None
ag_K = None

if mode == "magnitude":
    from autogaze.vllm_integration.patch import apply_autogaze_patch
    apply_autogaze_patch(mode="magnitude")

elif mode == "autogaze":
    from autogaze.vllm_integration.patch import apply_autogaze_patch
    apply_autogaze_patch(mode="autogaze")

    # Run AutoGaze preprocessing
    print("[approach5] Running AutoGaze preprocessing ...", flush=True)
    # Load frames
    cmd = ["ffmpeg", "-i", video_path, "-vf", "fps=2,scale=448:448",
           "-frames:v", "32", "-f", "rawvideo", "-pix_fmt", "rgb24", "-loglevel", "error", "-"]
    import numpy as np
    data = subprocess.run(cmd, capture_output=True).stdout
    n = len(data) // (448 * 448 * 3)
    arr = np.frombuffer(data[:n*448*448*3], dtype=np.uint8).reshape(n,448,448,3)
    raw_frames = torch.from_numpy(arr.copy()).permute(0,3,1,2).float() / 255.0
    print(f"[approach5] Loaded {{raw_frames.shape[0]}} frames", flush=True)

    # Patch AutoGaze for transformers 5.x
    from autogaze.models.autogaze import AutoGaze
    if not hasattr(AutoGaze, "all_tied_weights_keys"):
        AutoGaze.all_tied_weights_keys = property(lambda self: {{}})

    from autogaze.vllm_integration.autogaze_preprocess import AutoGazePreprocessor
    prep = AutoGazePreprocessor.load({autogaze_model_id!r})
    ag_mask, ag_K = prep.compute_retention_mask(raw_frames, target_grid_hw={qwen_grid!r}, gazing_ratio=pruning_rate)
    print(f"[approach5] AutoGaze K={{ag_K}} tokens selected", flush=True)
    del prep
    torch.cuda.empty_cache()

# ── Build vLLM engine ────────────────────────────────────────────────────
from vllm import LLM, SamplingParams

extra = {{}}
if mode != "dense":
    extra["video_pruning_rate"] = pruning_rate
    extra["enforce_eager"] = True

print(f"[approach5] Loading {{model_id}} ...", flush=True)
t0 = time.perf_counter()
llm = LLM(
    model=model_id, dtype="bfloat16", gpu_memory_utilization=0.7,
    max_model_len=8192, limit_mm_per_prompt={{"video":1}},
    allowed_local_media_path=REPO_DIR,
    **extra,
)
load_ms = (time.perf_counter() - t0) * 1000
print(f"[approach5] Model loaded in {{load_ms:.0f}} ms", flush=True)

# ── Run inference ────────────────────────────────────────────────────────
messages = [{{
    "role": "user",
    "content": [
        {{"type": "video_url", "video_url": {{
            "url": f"file://{{video_path}}", "fps": 2.0, "max_pixels": 448*448, "nframes": 32,
        }}}},
        {{"type": "text", "text": {prompt!r}}},
    ],
}}]

sampling = SamplingParams(max_tokens=10, temperature=0.0)

t1 = time.perf_counter()
if mode == "autogaze" and ag_mask is not None:
    from autogaze.vllm_integration.retention import AutoGazeContext
    with AutoGazeContext(ag_mask=ag_mask, K=ag_K):
        outputs = llm.chat(messages, sampling_params=sampling)
else:
    outputs = llm.chat(messages, sampling_params=sampling)
elapsed_ms = (time.perf_counter() - t1) * 1000

answer = outputs[0].outputs[0].text.strip()
n_tok = len(outputs[0].prompt_token_ids)
result = {{"mode": mode, "pruning_rate": pruning_rate, "answer": answer,
           "num_prompt_tokens": n_tok, "elapsed_ms": elapsed_ms, "load_ms": load_ms,
           "autogaze_K": ag_K}}
print("RESULT_JSON:" + json.dumps(result))
'''


def run_mode(mode, pruning_rate):
    code = WORKER_CODE.format(
        repo_dir=REPO_DIR, mode=mode, pruning_rate=pruning_rate,
        video_path=VIDEO_PATH, model_id=MODEL_ID,
        autogaze_model_id=AUTOGAZE_MODEL_ID,
        qwen_grid=QWEN_GRID_HW, prompt=PROMPT,
    )
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    proc = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    full = proc.stdout + "\n" + proc.stderr
    for line in full.splitlines():
        if "RESULT_JSON" not in line:
            print(f"  [{mode}] {line}")
    for line in full.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    raise RuntimeError(f"No RESULT_JSON for mode={mode}\nstderr:\n{proc.stderr[-2000:]}")


def main():
    print("=" * 70)
    print("Approach 5 (native) — AutoGaze × vLLM in auto_gaze env")
    print("=" * 70)
    print(f"Model:  {MODEL_ID}")
    print(f"Video:  {VIDEO_PATH}")
    print(f"Pruning rate: {PRUNING_RATE}\n")

    modes = [("dense", None), ("evs", PRUNING_RATE), ("magnitude", PRUNING_RATE), ("autogaze", PRUNING_RATE)]
    results = []

    for mode, pr in modes:
        print(f"\n--- mode={mode} ---", flush=True)
        try:
            r = run_mode(mode, pr or 0.5)
            results.append(r)
            print(f"  ✓ tokens={r['num_prompt_tokens']}  elapsed={r['elapsed_ms']:.0f}ms  answer={r['answer']}")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results.append({"mode": mode, "pruning_rate": pr, "answer": "ERROR",
                            "num_prompt_tokens": -1, "elapsed_ms": -1, "autogaze_K": None})

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    dense_tok = next((r["num_prompt_tokens"] for r in results if r["mode"] == "dense"), None)
    print(f"{'Mode':<12}  {'Tokens':>8}  {'vs Dense':>10}  {'ms':>8}  {'AutoGaze K':>12}  Answer")
    print("-" * 70)
    for r in results:
        tok = r["num_prompt_tokens"]
        vs = f"(-{(1-tok/dense_tok)*100:.0f}%)" if dense_tok and tok > 0 and r["mode"] != "dense" else ""
        ag = str(r.get("autogaze_K") or "")
        print(f"{r['mode']:<12}  {tok:>8}  {vs:>10}  {r['elapsed_ms']:>8.0f}  {ag:>12}  {r['answer']}")

    with open(os.path.join(REPO_DIR, "approach5_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
