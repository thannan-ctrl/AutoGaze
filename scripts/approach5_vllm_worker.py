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

    # --- Build messages and run inference via llm.chat() ---
    # llm.chat() handles video file paths natively — no manual frame loading needed.
    assert os.path.exists(VIDEO_PATH), f"Video not found: {VIDEO_PATH}"
    video_url = f"file://{VIDEO_PATH}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_url,
                    "fps": 2.0,
                    "max_pixels": 448 * 448,
                    "nframes": 32,
                },
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    print(f"[approach5] Running inference via llm.chat() ...", flush=True)
    sampling = SamplingParams(max_tokens=10, temperature=0.0)
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
    }
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
