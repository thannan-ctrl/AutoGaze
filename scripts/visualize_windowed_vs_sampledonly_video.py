"""Side-by-side VIDEO comparing codec mode's two HEVC-encoding strategies --
windowed (_extract_and_encode_windows, real local-context frames) vs.
sampled-only (_encode_sampled_frames_only, only the sampled frames chained
back-to-back) -- across all sampled frames of one video. Both panels are
codec_selector.build_gazing_info() calls; no AutoGaze forward pass involved
(unlike visualize_codec_vs_autogaze_video.py), so this is fast and has no
batch-size confound.

Usage:
    REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 python3 scripts/visualize_windowed_vs_sampledonly_video.py \
        --video data/egoschema/videos/<q_uid>.mp4 \
        --out figures/windowed_vs_sampledonly_egoschema.mp4
"""
import argparse
import os
import sys

REPO_DIR = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from transformers import AutoProcessor

from scripts.breakdown import codec_selector

BLUE = "#2a78d6"
ORANGE = "#eb6834"
DIM_ALPHA = 0.55


def denormalize(chw: torch.Tensor, mean, std) -> np.ndarray:
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    img = (chw.float() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def flat_index_to_cell(idx: int, grid_sizes: list[int]):
    total_patches = sum(g * g for g in grid_sizes)
    idx = idx % total_patches
    offset = 0
    for g in grid_sizes:
        n = g * g
        if idx < offset + n:
            local = idx - offset
            return divmod(local, g) + (g,)
        offset += n
    raise ValueError(f"index {idx} out of range for grid_sizes={grid_sizes}")


def draw_selection(ax, base_img: np.ndarray, cells, color, title):
    h, w = base_img.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for row, col, g in cells:
        y0, y1 = int(row * h / g), int((row + 1) * h / g)
        x0, x1 = int(col * w / g), int((col + 1) * w / g)
        mask[y0:y1, x0:x1] = True
    bright = base_img.astype(np.float32)
    combined = np.where(mask[:, :, None], bright, bright * (1 - DIM_ALPHA)).astype(np.uint8)
    ax.imshow(combined)
    for row, col, g in cells:
        y0, y1 = row * h / g, (row + 1) * h / g
        x0, x1 = col * w / g, (col + 1) * w / g
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.0))
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(color); s.set_linewidth(2.5)
    ax.set_title(title, fontsize=11, color=color, fontweight="bold")


def render_frame_rgb(fig, axes, base_img, w_cells, s_cells, t, n_frames, subtitle):
    axes[0].cla(); axes[1].cla()
    draw_selection(axes[0], base_img, w_cells, BLUE, f"windowed -- {len(w_cells)} patches")
    draw_selection(axes[1], base_img, s_cells, ORANGE, f"sampled-only -- {len(s_cells)} patches")
    fig.suptitle(f"frame {t + 1}/{n_frames}{subtitle}", fontsize=10, color="#52514e")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--thumb-ratio", type=float, default=0.12,
                     help="nominal fraction of patches kept per frame, same value for both panels")
    ap.add_argument("--out", default="figures/windowed_vs_sampledonly.mp4")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--hold-frames", type=int, default=6)
    args = ap.parse_args()

    print("[setup] building processor (loads AutoGaze weights, no forward pass)...", flush=True)
    proc = AutoProcessor.from_pretrained(
        "nvidia/NVILA-8B-HD-Video", trust_remote_code=True,
        num_video_frames=args.num_frames,
        num_video_frames_thumbnail=args.num_frames,
        max_tiles_video=args.num_frames,
        max_batch_size_autogaze=32,
        autogaze_model_id="nvidia/AutoGaze",
        gazing_ratio_tile=1, task_loss_requirement_tile=None,
        gazing_ratio_thumbnail=args.thumb_ratio, task_loss_requirement_thumbnail=0.6,
    )

    image_size = proc.image_processor.size.get("height", 392) if hasattr(proc.image_processor, "size") else 392
    common_kw = dict(
        video_path=args.video,
        num_video_frames=proc.num_video_frames,
        num_video_frames_thumbnail=proc.num_video_frames_thumbnail,
        max_tiles_video=proc.max_tiles_video,
        autogaze_max_num_frames=proc._autogaze_model.config.max_num_frames,
        image_size=image_size,
        scales=proc.target_scales,
        patch_size=proc.target_patch_size,
        gazing_ratio_tile=0.01,  # unused (tiles not visualized), kept small to skip fast
        gazing_ratio_thumbnail=args.thumb_ratio,
    )

    print("[run] codec_selector.build_gazing_info (windowed)...", flush=True)
    gazing_windowed = codec_selector.build_gazing_info(**common_kw, sampled_only=False)
    print("[run] codec_selector.build_gazing_info (sampled-only)...", flush=True)
    gazing_sampled = codec_selector.build_gazing_info(**common_kw, sampled_only=True)

    grid_sizes = [s // proc.target_patch_size for s in proc.target_scales]
    mean = getattr(proc.image_processor, "image_mean", [0.5, 0.5, 0.5])
    std = getattr(proc.image_processor, "image_std", [0.5, 0.5, 0.5])

    import importlib
    proc_module = importlib.import_module(type(proc).__module__)
    frames = proc_module._load_video_frames(args.video, num_frames=proc.num_video_frames)
    videos_inputs = proc._preprocess_videos([frames])
    thumbs = videos_inputs["pixel_values_videos_thumbnails"][0]  # (T_thumb, 1, C, H, W)

    w_pos, w_counts = gazing_windowed["gazing_pos_thumbnails"][0], gazing_windowed["num_gazing_each_frame_thumbnails"][0]
    s_pos, s_counts = gazing_sampled["gazing_pos_thumbnails"][0], gazing_sampled["num_gazing_each_frame_thumbnails"][0]
    n_frames = thumbs.shape[0]
    print(f"[info] {n_frames} thumbnail frames, w_pos.shape={tuple(w_pos.shape)} s_pos.shape={tuple(s_pos.shape)}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.2), facecolor="#fcfcfb")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    subtitle = f"  (codec: windowed vs. sampled-only encoding, nvf={args.num_frames})"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    container = None
    stream = None
    for t in range(n_frames):
        base = denormalize(thumbs[t, 0], mean, std)
        k_w = int(w_counts[t, 0].item())
        k_s = int(s_counts[t, 0].item())
        w_cells = [flat_index_to_cell(i, grid_sizes) for i in w_pos[t, :k_w].tolist()]
        s_cells = [flat_index_to_cell(i, grid_sizes) for i in s_pos[t, :k_s].tolist()]
        frame_rgb = render_frame_rgb(fig, axes, base, w_cells, s_cells, t, n_frames, subtitle)
        if container is None:
            h, w = frame_rgb.shape[:2]
            w, h = w - (w % 2), h - (h % 2)
            container = av.open(args.out, mode="w")
            stream = container.add_stream("libx264", rate=args.fps)
            stream.width, stream.height = w, h
            stream.pix_fmt = "yuv420p"
            stream.options = {"movflags": "faststart"}
        for _ in range(args.hold_frames):
            av_frame = av.VideoFrame.from_ndarray(frame_rgb[:h, :w], format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
        print(f"[frame] {t + 1}/{n_frames}  windowed={k_w} sampled-only={k_s}", flush=True)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    plt.close(fig)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
