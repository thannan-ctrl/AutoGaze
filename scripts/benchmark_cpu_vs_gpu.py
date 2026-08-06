"""Benchmark AutoGaze selector inference: CPU vs GPU.

Saves results to assets/benchmark_cpu_vs_gpu.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import av
from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from autogaze.models.autogaze import AutoGazeImageProcessor, AutoGaze

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_WARMUP = 3
N_RUNS   = 10
GAZING_RATIO        = 0.75
TASK_LOSS_REQ       = 0.7

# ── Load input ────────────────────────────────────────────────────────────────
print("Loading model and video...")
transform = AutoGazeImageProcessor.from_pretrained("nvidia/AutoGaze")
model_cpu = AutoGaze.from_pretrained("nvidia/AutoGaze").eval()

video_path = os.path.join(REPO_DIR, "assets", "example_input.mp4")
container  = av.open(video_path)
raw_video  = read_video_pyav(container, list(range(model_cpu.config.max_num_frames)))
container.close()

video_cpu = transform_video_for_pytorch(raw_video, transform)[None]  # (1, T, C, H, W)
video_gpu = video_cpu.cuda()
model_gpu = AutoGaze.from_pretrained("nvidia/AutoGaze").cuda().eval()

num_frames   = video_cpu.shape[1]
num_patches  = model_cpu.num_vision_tokens_each_frame
gpu_name     = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
print(f"Input : {tuple(video_cpu.shape)}  ({num_frames} frames, {num_patches} patches/frame)")
print(f"GPU   : {gpu_name}\n")


def run(model, video, device_label):
    def forward():
        with torch.inference_mode():
            return model(
                {"video": video},
                gazing_ratio=GAZING_RATIO,
                task_loss_requirement=TASK_LOSS_REQ,
            )

    print(f"[{device_label}] warming up ({N_WARMUP} runs)...")
    for _ in range(N_WARMUP):
        forward()
    if device_label == "GPU":
        torch.cuda.synchronize()

    times = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        forward()
        if device_label == "GPU":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(round(elapsed, 2))
        print(f"  run {i+1:02d}: {elapsed:.1f} ms")

    mean = round(sum(times) / len(times), 2)
    mn   = round(min(times), 2)
    mx   = round(max(times), 2)
    std  = round((sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5, 2)
    print(f"[{device_label}] mean={mean:.1f}ms  min={mn:.1f}ms  max={mx:.1f}ms  std={std:.1f}ms\n")
    return {"mean_ms": mean, "min_ms": mn, "max_ms": mx, "std_ms": std, "runs_ms": times}


cpu_stats = run(model_cpu, video_cpu, "CPU")
gpu_stats = run(model_gpu, video_gpu, "GPU")

speedup = round(cpu_stats["mean_ms"] / gpu_stats["mean_ms"], 2)

print("=" * 45)
print(f"  CPU  mean : {cpu_stats['mean_ms']:7.1f} ms  (std {cpu_stats['std_ms']:.1f})")
print(f"  GPU  mean : {gpu_stats['mean_ms']:7.1f} ms  (std {gpu_stats['std_ms']:.1f})")
print(f"  Speedup   : {speedup:.1f}x  (CPU / GPU)")
print("=" * 45)

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "config": {
        "num_frames":        num_frames,
        "num_patches":       num_patches,
        "gazing_ratio":      GAZING_RATIO,
        "task_loss_req":     TASK_LOSS_REQ,
        "n_warmup":          N_WARMUP,
        "n_runs":            N_RUNS,
        "gpu":               gpu_name,
        "model":             "nvidia/AutoGaze",
        "model_params":      sum(p.numel() for p in model_cpu.parameters()),
    },
    "CPU": cpu_stats,
    "GPU": gpu_stats,
    "speedup_cpu_over_gpu": speedup,
}

out_path = os.path.join(REPO_DIR, "assets", "benchmark_cpu_vs_gpu.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved → {out_path}")
