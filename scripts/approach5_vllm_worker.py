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

    # --- Load video frames (multi-fallback: torchcodec → VideoReader → av → ffmpeg) ---
    import numpy as np
    import torch

    def _load_frames_torchcodec(path, target_fps=2.0, max_frames=32):
        from torchcodec.decoders import VideoDecoder
        dec = VideoDecoder(path)
        meta = dec.get_metadata()
        src_fps = meta.average_fps or 30.0
        step = max(1, int(src_fps / target_fps))
        total = meta.num_frames or 1000
        indices = list(range(0, min(total, max_frames * step), step))[:max_frames]
        frames = dec.get_frames_at(indices=indices).data  # (T, C, H, W) uint8
        return frames.float() / 255.0

    def _load_frames_video_reader(path, target_fps=2.0, max_frames=32):
        from torchvision.io import VideoReader
        reader = VideoReader(path, "video")
        meta = reader.get_metadata()
        src_fps = meta["video"]["fps"][0] if meta and "video" in meta else 30.0
        step = max(1, int(src_fps / target_fps))
        frames, i = [], 0
        for frame in reader:
            if i % step == 0:
                frames.append(frame["data"])  # (C, H, W) uint8
            i += 1
            if len(frames) >= max_frames:
                break
        return torch.stack(frames).float() / 255.0

    def _load_frames_av(path, target_fps=2.0, max_frames=32):
        import av as _av
        container = _av.open(path)
        stream = container.streams.video[0]
        src_fps = float(stream.average_rate) if stream.average_rate else 30.0
        step = max(1, int(src_fps / target_fps))
        raw, i = [], 0
        for frame in container.decode(video=0):
            if i % step == 0:
                raw.append(frame.to_ndarray(format="rgb24"))
            i += 1
            if len(raw) >= max_frames:
                break
        container.close()
        arr = np.stack(raw)  # (T, H, W, 3)
        return torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0

    def _load_frames_ffmpeg(path, target_fps=2.0, max_frames=32, size=(448, 448)):
        import subprocess, struct
        w, h = size
        cmd = ["ffmpeg", "-i", path, "-vf", f"fps={target_fps},scale={w}:{h}",
               "-frames:v", str(max_frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        proc = subprocess.run(cmd, capture_output=True)
        data = proc.stdout
        n = len(data) // (h * w * 3)
        arr = np.frombuffer(data[:n * h * w * 3], dtype=np.uint8).reshape(n, h, w, 3)
        return torch.from_numpy(arr.copy()).permute(0, 3, 1, 2).float() / 255.0

    sampled_frames = None
    for loader_name, loader in [
        ("torchcodec", _load_frames_torchcodec),
        ("VideoReader", _load_frames_video_reader),
        ("av",         _load_frames_av),
        ("ffmpeg",     _load_frames_ffmpeg),
    ]:
        try:
            sampled_frames = loader(VIDEO_PATH)
            print(f"[approach5] Video loaded via {loader_name}: {tuple(sampled_frames.shape)}", flush=True)
            break
        except Exception as e:
            print(f"[approach5] {loader_name} failed: {e}", flush=True)

    if sampled_frames is None:
        raise RuntimeError("All video loaders failed")
    fps = 2.0  # effective sampling rate used by loaders above
    print(f"[approach5] Video: {sampled_frames.shape[0]} frames @ {fps:.1f} fps → shape {tuple(sampled_frames.shape)}", flush=True)

    # --- Build prompt and run inference ---
    # vLLM's Qwen3-VL processor expects: text prompt with <video> placeholder,
    # and multi_modal_data={"video": frames} where frames is (T, C, H, W) float32 [0,1].
    from transformers import AutoProcessor as HFProcessor
    proc = HFProcessor.from_pretrained(MODEL_ID)

    # Build chat-template text with a video placeholder
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "placeholder", "nframes": sampled_frames.shape[0]},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    try:
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback: manual Qwen3-VL chat format
        n = sampled_frames.shape[0]
        video_tokens = "<|vision_start|>" + "<|video_pad|>" * n + "<|vision_end|>"
        text = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{video_tokens}\n{PROMPT}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    print(f"[approach5] Prompt built ({len(text)} chars). Running inference ...", flush=True)

    sampling = SamplingParams(max_tokens=10, temperature=0.0)
    t0 = time.perf_counter()
    outputs = llm.generate(
        {
            "prompt": text,
            "multi_modal_data": {"video": sampled_frames},  # (T, C, H, W) float32
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
