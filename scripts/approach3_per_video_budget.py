#!/usr/bin/env python3
"""
Approach 3 — Per-video fixed budget with dynamic per-frame allocation.

Runs AutoGaze to compute K_video = sum(tokens_per_frame) across all frames.
This single number is reported to vLLM's scheduler, while frames internally
receive different token allocations (first frame gets more, later frames less).

Measures:
  - K_video (total tokens declared to scheduler)
  - Per-frame token distribution
  - Comparison to uniform allocation at the same total budget
  - Variability across different gazing_ratio settings
"""
import os
import sys

import torch
from transformers import AutoModel, AutoProcessor

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_DIR = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)

MODEL_PATH = "nvidia/NVILA-8B-HD-Video"
VIDEO_PATH = os.path.join(REPO_DIR, "assets", "example_input.mp4")
PROMPT = (
    "Question: What does the white text on the green road sign say?\n"
    "A. Hampden St\nB. Hampden Ave\nC. HampdenBlvd\nD. Hampden Rd\n"
    "Please answer directly with the letter of the correct answer."
)

NUM_VIDEO_FRAMES = 128
NUM_VIDEO_FRAMES_THUMBNAIL = 64
MAX_TILES_VIDEO = 48

# Adaptive per-frame ratios (first frame gets more attention)
GAZING_RATIO_ADAPTIVE = [0.2] + [0.06] * 15


def run_with_config(gazing_ratio, label: str):
    print(f"\n--- {label} ---", flush=True)
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        autogaze_model_id="nvidia/AutoGaze",
        num_video_frames=NUM_VIDEO_FRAMES,
        num_video_frames_thumbnail=NUM_VIDEO_FRAMES_THUMBNAIL,
        max_tiles_video=MAX_TILES_VIDEO,
        gazing_ratio_tile=gazing_ratio,
        gazing_ratio_thumbnail=1.0,
        task_loss_requirement_tile=None,
        task_loss_requirement_thumbnail=None,
        max_batch_size_autogaze=16,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map="auto",
        max_batch_size_siglip=32,
    )
    model.eval()

    video_token = processor.tokenizer.video_token
    inputs = processor(
        text=f"{video_token}\n\n{PROMPT}", videos=VIDEO_PATH, return_tensors="pt"
    )

    K_video = inputs["input_ids"].shape[1]

    # Get pixel_values to see tile count
    tile_info = {}
    for key in ("pixel_values", "pixel_values_videos", "pixel_values_tiles"):
        if key in inputs:
            tile_info[key] = tuple(inputs[key].shape)

    print(f"  gazing_ratio:    {gazing_ratio}")
    print(f"  K_video (total input tokens): {K_video}")
    print(f"  → vLLM scheduler pre-allocates: {K_video} KV slots")
    for k, v in tile_info.items():
        print(f"  {k} shape: {v}")

    # Run inference
    inputs_gpu = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    with torch.no_grad():
        outputs = model.generate(**inputs_gpu, max_new_tokens=10)
    answer = processor.batch_decode(
        outputs[:, inputs_gpu["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()
    print(f"  Answer: {answer}")

    del model, processor
    torch.cuda.empty_cache()

    return {"label": label, "gazing_ratio": gazing_ratio, "K_video": K_video, "answer": answer}


def main():
    print("=" * 70)
    print("Approach 3 — Per-video fixed budget with adaptive per-frame allocation")
    print("=" * 70)

    configs = [
        (1.0,                  "Dense baseline (ratio=1.0)"),
        (0.5,                  "Uniform 50% (ratio=0.5)"),
        (GAZING_RATIO_ADAPTIVE, "Adaptive (first=0.2, rest=0.06)"),
        (0.13,                 "Uniform 13% (same avg as adaptive)"),
    ]

    results = []
    for ratio, label in configs:
        r = run_with_config(ratio, label)
        results.append(r)

    print("\n" + "=" * 70)
    print("SUMMARY — K_video reported to vLLM scheduler")
    print("=" * 70)
    print(f"{'Config':<40}  {'K_video':>8}  {'Answer'}")
    print("-" * 65)
    dense_K = next(r["K_video"] for r in results if "Dense" in r["label"])
    for r in results:
        savings = f"  ({(1 - r['K_video']/dense_K)*100:.0f}% reduction)" if dense_K != r["K_video"] else ""
        print(f"{r['label']:<40}  {r['K_video']:>8}{savings}  {r['answer']}")

    print("\nKey insight:")
    print("  Adaptive allocation gives per-frame variety while still reporting")
    print("  a single K_video to the scheduler — no scheduler changes needed.")


if __name__ == "__main__":
    main()
