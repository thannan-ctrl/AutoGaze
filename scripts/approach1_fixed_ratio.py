#!/usr/bin/env python3
"""
Approach 1 — Fixed compression rate (EVS-style).

Sweeps gazing_ratio in {0.25, 0.5, 0.75, 1.0} with a uniform per-frame budget.
Reports token count, latency, GPU memory, and answer for each ratio.

token_count = int(gazing_ratio * patches_per_frame) * num_frames
This is the value that would be declared upfront to vLLM's scheduler.

Each ratio runs in its own subprocess to guarantee clean GPU state between runs
(large KV-cache activations from prior runs can fill the allocator cache).
"""
import json
import os
import subprocess
import sys
import time

REPO_DIR = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)

MODEL_PATH = "nvidia/NVILA-8B-HD-Video"
VIDEO_PATH = os.path.join(REPO_DIR, "assets", "example_input.mp4")
PROMPT = (
    "Question: What does the white text on the green road sign say?\n"
    "A. Hampden St\n"
    "B. Hampden Ave\n"
    "C. HampdenBlvd\n"
    "D. Hampden Rd\n"
    "Please answer directly with the letter of the correct answer."
)

RATIOS = [0.1, 0.25, 0.5]  # 0.75+ OOM with 128-frame 448x448 video on single-GPU config

NUM_VIDEO_FRAMES = 128
NUM_VIDEO_FRAMES_THUMBNAIL = 64
MAX_TILES_VIDEO = 48
MAX_BATCH_SIZE_AUTOGAZE = 16
MAX_BATCH_SIZE_SIGLIP = 32

_WORKER_SCRIPT = '''
import os, sys, json, time
import torch
from transformers import AutoModel, AutoProcessor
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
ratio = float(sys.argv[1])
model_path = sys.argv[2]
video_path = sys.argv[3]
num_frames = int(sys.argv[4])
num_thumbs = int(sys.argv[5])
max_tiles = int(sys.argv[6])
bs_ag = int(sys.argv[7])
bs_sgl = int(sys.argv[8])
prompt = sys.argv[9]

processor = AutoProcessor.from_pretrained(
    model_path, autogaze_model_id="nvidia/AutoGaze",
    num_video_frames=num_frames, num_video_frames_thumbnail=num_thumbs,
    max_tiles_video=max_tiles, gazing_ratio_tile=ratio, gazing_ratio_thumbnail=ratio,
    task_loss_requirement_tile=None, task_loss_requirement_thumbnail=None,
    max_batch_size_autogaze=bs_ag, trust_remote_code=True,
)
model = AutoModel.from_pretrained(
    model_path, trust_remote_code=True, device_map="auto", max_batch_size_siglip=bs_sgl,
)
model.eval()
vt = processor.tokenizer.video_token
t0 = time.perf_counter()
inputs = processor(text=f"{vt}\\n\\n{prompt}", videos=video_path, return_tensors="pt")
preprocess_ms = (time.perf_counter() - t0) * 1000
total_tokens = inputs["input_ids"].shape[1]
inputs_gpu = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
torch.cuda.synchronize()
t1 = time.perf_counter()
with torch.no_grad():
    out = model.generate(**inputs_gpu, max_new_tokens=10)
torch.cuda.synchronize()
gen_ms = (time.perf_counter() - t1) * 1000
answer = processor.batch_decode(out[:, inputs_gpu["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
mem_gb = torch.cuda.memory_allocated() / 1e9
result = {"ratio": ratio, "total_input_tokens": total_tokens, "preprocess_ms": preprocess_ms,
          "generate_ms": gen_ms, "total_ms": preprocess_ms + gen_ms, "mem_allocated_gb": mem_gb, "answer": answer}
print("RESULT_JSON:" + json.dumps(result))
'''


def run_ratio(ratio: float) -> dict:
    python = sys.executable
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    cmd = [
        python, "-c", _WORKER_SCRIPT,
        str(ratio), MODEL_PATH, VIDEO_PATH,
        str(NUM_VIDEO_FRAMES), str(NUM_VIDEO_FRAMES_THUMBNAIL), str(MAX_TILES_VIDEO),
        str(MAX_BATCH_SIZE_AUTOGAZE), str(MAX_BATCH_SIZE_SIGLIP), PROMPT,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    stdout = proc.stdout + proc.stderr
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        if not line.startswith("RESULT_JSON:"):
            print(" ", line)
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    raise RuntimeError(f"No RESULT_JSON found for ratio={ratio}. stderr:\n{proc.stderr[-2000:]}")


def main():
    print("=" * 70)
    print("Approach 1 — Fixed compression rate sweep")
    print("=" * 70)
    print(f"Ratios: {RATIOS}")
    print(f"Model:  {MODEL_PATH}")
    print(f"Video:  {VIDEO_PATH}")
    print("(Each ratio runs in a clean subprocess to avoid GPU memory accumulation)\n")

    results = []
    for ratio in RATIOS:
        print(f"\n--- gazing_ratio = {ratio} ---", flush=True)
        try:
            r = run_ratio(ratio)
        except RuntimeError as e:
            print(f"  SKIPPED (OOM): {e}")
            results.append({"ratio": ratio, "total_input_tokens": -1, "preprocess_ms": -1,
                             "generate_ms": -1, "total_ms": -1, "mem_allocated_gb": -1, "answer": "OOM"})
            continue
        results.append(r)
        print(f"  Answer:        {r['answer']}")
        print(f"  Input tokens:  {r['total_input_tokens']}")
        print(f"  Preprocess:    {r['preprocess_ms']:.0f} ms")
        print(f"  Generate:      {r['generate_ms']:.0f} ms")
        print(f"  Total:         {r['total_ms']:.0f} ms")
        print(f"  GPU mem alloc: {r['mem_allocated_gb']:.1f} GB")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Ratio':>8}  {'Tokens':>8}  {'Preproc ms':>11}  {'Gen ms':>8}  {'Total ms':>9}  {'Answer'}")
    print("-" * 70)
    dense_tokens = next((r["total_input_tokens"] for r in results if r["ratio"] == 1.0), None)
    for r in results:
        reduction = f" (-{(1-r['total_input_tokens']/dense_tokens)*100:.0f}%)" if dense_tokens and r["ratio"] < 1.0 else ""
        print(
            f"{r['ratio']:>8.2f}  {r['total_input_tokens']:>8}{reduction:<9}  "
            f"{r['preprocess_ms']:>11.0f}  {r['generate_ms']:>8.0f}  "
            f"{r['total_ms']:>9.0f}  {r['answer']}"
        )


if __name__ == "__main__":
    main()
