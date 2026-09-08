"""Side-by-side VIDEO comparing AutoGaze's real trained selector against a
codec variant built to match AutoGaze's actual design as closely as possible:
dense, consecutive real frames from frame 0 (no global sparse sampling),
cropped to the first N minutes, encoded in genuine non-overlapping 16-frame
windows with the HEVC coding *restarted* (forced I-frame) at every window
boundary -- codec_selector.py's new `gop_restart=16` -- and with every
patch kept (full multi-scale retention, no top-k) on the first frame of each
16-frame window, matching AutoGaze's own per-chunk anchor-frame treatment.

Usage:
    REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 python3 scripts/visualize_restarted_chunks_vs_autogaze.py \
        --video data/egoschema/videos/<q_uid>.mp4 \
        --out figures/restarted_chunks_vs_autogaze_egoschema.mp4
"""
import argparse
import importlib
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
GOP = 16


def native_frame_count(video_path: str):
    c = av.open(video_path)
    s = c.streams.video[0]
    n, fps = s.frames, float(s.average_rate)
    c.close()
    return n, fps


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


def render_frame_rgb(fig, axes, base_img, ag_cells, cd_cells, t, n_frames, subtitle):
    axes[0].cla(); axes[1].cla()
    draw_selection(axes[0], base_img, ag_cells, BLUE, f"AutoGaze -- {len(ag_cells)} patches")
    draw_selection(axes[1], base_img, cd_cells, ORANGE, f"codec (restart/16, full-1st) -- {len(cd_cells)} patches")
    fig.suptitle(f"frame {t + 1}/{n_frames}{subtitle}", fontsize=10, color="#52514e")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--duration-min", type=float, default=5.0,
                     help="crop to at most this many minutes of real video time, from frame 0")
    ap.add_argument("--thumb-ratio", type=float, default=0.12)
    ap.add_argument("--out", default="figures/restarted_chunks_vs_autogaze.mp4")
    ap.add_argument("--fps", type=int, default=8, help="output video fps")
    ap.add_argument("--hold-frames", type=int, default=3, help="video frames held per sampled frame")
    ap.add_argument("--render-stride", type=int, default=8,
                     help="render every Nth computed frame (subsamples the OUTPUT video only -- "
                          "at duration-min=5 on a ~30fps video, stride=1 renders ~9000 frames "
                          "and produces a 150MB+ file; 8 keeps it in the same ~10-15MB range as "
                          "the other whole-video figures)")
    args = ap.parse_args()

    n_native, fps = native_frame_count(args.video)
    n_cap = int(fps * 60 * args.duration_min)
    n_frames = min(n_native, n_cap)
    n_frames = (n_frames // GOP) * GOP  # exact whole number of 16-frame chunks
    print(f"[setup] video: {n_native} native frames @ {fps:.1f}fps -- "
          f"cropping to {n_frames} frames ({n_frames / fps:.1f}s, {n_frames // GOP} chunks of {GOP})", flush=True)

    print("[setup] building processor (loads AutoGaze weights, no 8B LLM -- never needed here)...", flush=True)
    proc = AutoProcessor.from_pretrained(
        "nvidia/NVILA-8B-HD-Video", trust_remote_code=True,
        num_video_frames=n_frames,
        num_video_frames_thumbnail=n_frames,  # dense -- keep every sampled frame, no thumbnail subsampling
        max_tiles_video=16,  # tiles unused/skipped here; kept small and NOT coupled to n_frames
        max_batch_size_autogaze=32,
        autogaze_model_id="nvidia/AutoGaze",
        gazing_ratio_tile=1, task_loss_requirement_tile=None,  # tiles unused, skip for speed
        gazing_ratio_thumbnail=args.thumb_ratio, task_loss_requirement_thumbnail=0.6,
    )
    proc_module = importlib.import_module(type(proc).__module__)
    frames = proc_module._load_video_frames(args.video, num_frames=n_frames)

    print("[run] _preprocess_videos + real AutoGaze forward (thumbnails)...", flush=True)
    videos_inputs = proc._preprocess_videos([frames])
    gazing_autogaze = proc._get_gazing_info_from_videos(videos_inputs)

    image_size = proc.image_processor.size.get("height", 392) if hasattr(proc.image_processor, "size") else 392
    print(f"[run] codec_selector.build_gazing_info(num_video_frames={n_frames}, "
          f"sampled_only=True, gop_restart={GOP}, full_first_frame applied per-chunk in-script)...", flush=True)
    gazing_codec = codec_selector.build_gazing_info(
        video_path=args.video,
        num_video_frames=n_frames,
        num_video_frames_thumbnail=n_frames,
        max_tiles_video=16,
        autogaze_max_num_frames=proc._autogaze_model.config.max_num_frames,
        image_size=image_size,
        scales=proc.target_scales,
        patch_size=proc.target_patch_size,
        gazing_ratio_tile=0.01,  # unused (tiles not visualized), kept small to skip fast
        gazing_ratio_thumbnail=args.thumb_ratio,
        sampled_only=True,
        gop_restart=GOP,
    )

    grid_sizes = [s // proc.target_patch_size for s in proc.target_scales]
    total_patches = sum(g * g for g in grid_sizes)
    mean = getattr(proc.image_processor, "image_mean", [0.5, 0.5, 0.5])
    std = getattr(proc.image_processor, "image_std", [0.5, 0.5, 0.5])

    thumbs = videos_inputs["pixel_values_videos_thumbnails"][0]  # (T_thumb, 1, C, H, W)
    ag_pos, ag_counts = gazing_autogaze["gazing_pos_thumbnails"][0], gazing_autogaze["num_gazing_each_frame_thumbnails"][0]
    cd_pos, cd_counts = gazing_codec["gazing_pos_thumbnails"][0], gazing_codec["num_gazing_each_frame_thumbnails"][0]
    n_thumbs = thumbs.shape[0]
    print(f"[info] {n_thumbs} thumbnail frames, ag_pos.shape={tuple(ag_pos.shape)} cd_pos.shape={tuple(cd_pos.shape)}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 5.2), facecolor="#fcfcfb")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    subtitle = f"  (codec: restart-every-{GOP}, full-first-frame/chunk, {args.duration_min:.0f}min cap)"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    container = None
    stream = None
    render_indices = list(range(0, n_thumbs, args.render_stride))
    for ri, t in enumerate(render_indices):
        base = denormalize(thumbs[t, 0], mean, std)
        k_ag = int(ag_counts[t, 0].item())
        ag_cells = [flat_index_to_cell(i, grid_sizes) for i in ag_pos[t, :k_ag].tolist()]
        if t % GOP == 0:
            # per-chunk full-first-frame anchor: keep every patch (all scales),
            # matching build_gazing_info's own full_first_frame semantics but
            # applied at EVERY chunk boundary, not just the video's global
            # first frame (see codec_vs_autogaze full-first-frame variant).
            k_cd = total_patches
            cd_cells = [flat_index_to_cell(i, grid_sizes) for i in range(total_patches)]
        else:
            k_cd = int(cd_counts[t, 0].item())
            cd_cells = [flat_index_to_cell(i, grid_sizes) for i in cd_pos[t, :k_cd].tolist()]
        frame_rgb = render_frame_rgb(fig, axes, base, ag_cells, cd_cells, t, n_thumbs, subtitle)
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
        if ri % 20 == 0:
            print(f"[frame] {ri + 1}/{len(render_indices)}  native_frame={t}  autogaze={k_ag} codec={k_cd}", flush=True)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    plt.close(fig)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
