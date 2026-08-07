#!/usr/bin/env python3
"""
Approach 5 — AutoGaze × vLLM integration worker.
Runs INSIDE the vLLM Docker container with CLI args: --mode --pruning-rate

Usage:
    python approach5_vllm_worker.py --mode evs --pruning-rate 0.5
    python approach5_vllm_worker.py --mode magnitude --pruning-rate 0.5
    python approach5_vllm_worker.py --mode dense
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="dense", choices=["dense", "evs", "magnitude", "autogaze"])
    parser.add_argument("--pruning-rate", type=float, default=None)
    args = parser.parse_args()

    mode = args.mode
    pruning_rate = args.pruning_rate

    print(f"[approach5] mode={mode} pruning_rate={pruning_rate}", flush=True)

    # --- Apply AutoGaze patch BEFORE importing vllm model classes ---
    if mode in ("magnitude", "autogaze"):
        from autogaze.vllm_integration.patch import apply_autogaze_patch
        apply_autogaze_patch(mode=mode)

    # --- Build vLLM engine ---
    from vllm import LLM, SamplingParams

    mm_processor_kwargs = {}
    if pruning_rate is not None and mode != "dense":
        mm_processor_kwargs["video_pruning_rate"] = pruning_rate

    print(f"[approach5] Loading {MODEL_ID} ...", flush=True)
    t_load = time.perf_counter()
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        gpu_memory_utilization=0.7,
        max_model_len=8192,
        limit_mm_per_prompt={"video": 1},
        mm_processor_kwargs=mm_processor_kwargs if mm_processor_kwargs else None,
    )
    load_ms = (time.perf_counter() - t_load) * 1000
    print(f"[approach5] Model loaded in {load_ms:.0f} ms", flush=True)

    # --- Load video frames with av (PyAV) ---
    import av
    import numpy as np
    import torch

    container = av.open(VIDEO_PATH)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 30.0
    # Sample ~2 fps, max 32 frames
    step = max(1, int(fps / 2.0))
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i % step == 0:
            frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) >= 32:
            break
    container.close()

    # (T, H, W, 3) → (T, 3, H, W) float [0,1]
    video_np = np.stack(frames)  # (T, H, W, 3)
    sampled_frames = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0
    print(f"[approach5] Video: {sampled_frames.shape[0]} frames @ {fps:.1f} fps → shape {tuple(sampled_frames.shape)}", flush=True)

    # --- Build prompt using Qwen3-VL chat template ---
    from transformers import AutoProcessor as HFProcessor
    proc = HFProcessor.from_pretrained(MODEL_ID)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": sampled_frames,  # (T, C, H, W) tensor
                    "fps": 2.0,
                },
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    # Apply chat template → formatted text prompt
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Extract vision info (frames, images) for vLLM's multimodal input
    image_inputs, video_inputs = proc.process_vision_info(messages)

    sampling = SamplingParams(max_tokens=10, temperature=0.0)

    print("[approach5] Running inference ...", flush=True)
    mm_data = {}
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    if image_inputs is not None:
        mm_data["image"] = image_inputs

    t0 = time.perf_counter()
    outputs = llm.generate(
        {
            "prompt": text,
            "multi_modal_data": mm_data,
        },
        sampling_params=sampling,
    )
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
    }
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
