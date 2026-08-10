#!/usr/bin/env python3
"""
Runs AutoGaze preprocessing OUTSIDE the vLLM Docker container,
using the auto_gaze conda env (transformers 4.x compatible).

Saves the retention mask to a .pt file for the Docker worker to load.

Usage — post-ViT AutoGaze mask (autogaze mode, default 16×16 post-merge grid):
    /path/to/auto_gaze/python run_autogaze_preprocess.py \
        --video /path/to/video.mp4 \
        --output /tmp/ag_mask.pt \
        --gazing-ratio 0.5

Usage — pre-ViT sparse-ViT mask (sparse_vit mode, 32×32 ViT patch grid):
    /path/to/auto_gaze/python run_autogaze_preprocess.py \
        --video /path/to/video.mp4 \
        --output /tmp/ag_mask_vit.pt \
        --gazing-ratio 0.5 \
        --grid-hw 32 32

Grid sizes for Qwen3-VL at 448×448 input (14px patches, merge_size=2):
    Post-merge (autogaze mode):  --grid-hw 16 16  [default]
    Pre-merge  (sparse_vit mode): --grid-hw 32 32
"""
import argparse
import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

import torch
import torch.nn.functional as F


def load_frames_ffmpeg(video_path: str, fps: float = 2.0, max_frames: int = 32, size: int = 448):
    import numpy as np
    import torch.nn.functional as F

    def _try_av(p):
        import av
        container = av.open(p)
        stream = container.streams.video[0]
        src_fps = float(stream.average_rate) if stream.average_rate else 30.0
        step = max(1, int(src_fps / fps))
        raw, i = [], 0
        for frame in container.decode(video=0):
            if i % step == 0:
                raw.append(frame.to_ndarray(format="rgb24"))
            i += 1
            if len(raw) >= max_frames:
                break
        container.close()
        return np.stack(raw)

    def _try_torchcodec(p):
        from torchcodec.decoders import VideoDecoder
        dec = VideoDecoder(p)
        meta = dec.get_metadata()
        src_fps = meta.average_fps or 30.0
        step = max(1, int(src_fps / fps))
        total = meta.num_frames or 1000
        indices = list(range(0, min(total, max_frames * step), step))[:max_frames]
        frames = dec.get_frames_at(indices=indices).data  # (T,C,H,W) uint8
        return frames.permute(0, 2, 3, 1).numpy()

    arr = None
    for name, fn in [("av", _try_av), ("torchcodec", _try_torchcodec)]:
        try:
            arr = fn(video_path)
            print(f"[preprocess] Loaded {arr.shape[0]} frames via {name}", flush=True)
            break
        except Exception as e:
            print(f"[preprocess] {name} failed: {e}", flush=True)

    if arr is None:
        raise RuntimeError("No video loader worked")

    t = torch.from_numpy(arr.copy()).permute(0, 3, 1, 2).float() / 255.0
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gazing-ratio", type=float, default=0.5)
    parser.add_argument("--target-grid-h", type=int, default=None,
                        help="Grid height (deprecated: use --grid-hw instead)")
    parser.add_argument("--target-grid-w", type=int, default=None,
                        help="Grid width (deprecated: use --grid-hw instead)")
    parser.add_argument("--grid-hw", type=int, nargs=2, default=None,
                        metavar=("H", "W"),
                        help="Grid size H W (e.g. 16 16 for post-merge, 32 32 for pre-merge/sparse-ViT)")
    parser.add_argument("--autogaze-model", default="nvidia/AutoGaze")
    args = parser.parse_args()

    # Resolve grid size: --grid-hw takes priority, then --target-grid-h/w, then default 16×16
    if args.grid_hw is not None:
        grid_h, grid_w = args.grid_hw
    elif args.target_grid_h is not None or args.target_grid_w is not None:
        grid_h = args.target_grid_h or 16
        grid_w = args.target_grid_w or 16
    else:
        grid_h, grid_w = 16, 16

    print(f"[autogaze_preprocess] Video: {args.video}", flush=True)
    print(f"[autogaze_preprocess] gazing_ratio: {args.gazing_ratio}", flush=True)
    print(f"[autogaze_preprocess] target_grid: ({grid_h}, {grid_w})", flush=True)

    from autogaze.vllm_integration.autogaze_preprocess import AutoGazePreprocessor

    raw_frames = load_frames_ffmpeg(args.video)
    print(f"[autogaze_preprocess] Loaded {raw_frames.shape[0]} frames", flush=True)

    prep = AutoGazePreprocessor.load(args.autogaze_model)
    mask, K = prep.compute_retention_mask(
        raw_frames,
        target_grid_hw=(grid_h, grid_w),
        gazing_ratio=args.gazing_ratio,
    )

    torch.save({"mask": mask, "K": K, "gazing_ratio": args.gazing_ratio}, args.output)
    print(f"[autogaze_preprocess] Saved mask to {args.output} (K={K})", flush=True)


if __name__ == "__main__":
    main()
