"""Side-by-side VIDEO comparing which whole-frame patches AutoGaze's trained
selector keeps vs. codec mode, across all sampled frames of one video --
same real-pipeline approach as visualize_codec_vs_autogaze.py (real
_get_gazing_info_from_videos / codec_selector.build_gazing_info calls, no
benchmark harness), but using the *thumbnail* path (whole resized frame, no
spatial tile crop) so each output video frame shows the entire scene, and
animated over time instead of a single static figure.

Usage:
    REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 python3 scripts/visualize_codec_vs_autogaze_video.py \
        --video data/egoschema/videos/<q_uid>.mp4 \
        --out figures/codec_vs_autogaze_video.mp4
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
from transformers import AutoModel, AutoProcessor

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


def render_frame_rgb(fig, axes, base_img, ag_cells, cd_cells, t, n_frames, subtitle):
    axes[0].cla(); axes[1].cla()
    draw_selection(axes[0], base_img, ag_cells, BLUE, f"AutoGaze -- {len(ag_cells)} patches")
    draw_selection(axes[1], base_img, cd_cells, ORANGE, f"codec -- {len(cd_cells)} patches")
    fig.suptitle(f"frame {t + 1}/{n_frames}{subtitle}", fontsize=10, color="#52514e")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3].copy()  # drop alpha -> RGB uint8, contiguous


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--thumb-ratio", type=float, default=0.12,
                     help="nominal fraction of patches kept per frame, same value for both selectors")
    ap.add_argument("--task-loss-requirement", type=float, default=0.6)
    ap.add_argument("--out", default="figures/codec_vs_autogaze_video.mp4")
    ap.add_argument("--fps", type=int, default=12, help="output video fps")
    ap.add_argument("--hold-frames", type=int, default=18, help="video frames held per sampled frame")
    ap.add_argument("--w-size", type=float, default=1.0,
                     help="weight on codec's CU-size score term (score_cu's size_score); "
                          "0 disables it entirely, isolating motion+skip-penalty only")
    ap.add_argument("--w-motion", type=float, default=1.0, help="weight on codec's motion score term")
    ap.add_argument("--w-residual", type=float, default=0.0,
                     help="weight on codec's luma-residual-energy score term (LLaVA-OneVision-2-style "
                          "motion+residual saliency fusion; 0 = motion+CU-size only, prior default)")
    ap.add_argument("--full-first-frame", action="store_true",
                     help="keep every patch (no codec top-k) on the first sampled frame, "
                          "I-canvas-style anchor coverage per LLaVA-OneVision-2 sec 2.2")
    ap.add_argument("--sampled-only", action="store_true",
                     help="score motion only between the sampled frames themselves (chained "
                          "sequentially, no real-frame context window) instead of the default "
                          "windowed real-frame-context encoding -- see "
                          "codec_selector._encode_sampled_frames_only")
    args = ap.parse_args()

    device = os.environ.get("NVILA_DEVICE", "cuda:0")
    print("[setup] loading model...", flush=True)
    model = AutoModel.from_pretrained(
        "nvidia/NVILA-8B-HD-Video", trust_remote_code=True, dtype=torch.bfloat16, max_batch_size_siglip=32,
    ).to(device)
    model.eval()

    proc = AutoProcessor.from_pretrained(
        "nvidia/NVILA-8B-HD-Video", trust_remote_code=True,
        num_video_frames=args.num_frames,
        num_video_frames_thumbnail=args.num_frames,  # keep every sampled frame, no subsampling
        max_tiles_video=args.num_frames,
        max_batch_size_autogaze=64,
        autogaze_model_id="nvidia/AutoGaze",
        gazing_ratio_tile=1, task_loss_requirement_tile=None,  # tiles unused here, skip for speed
        gazing_ratio_thumbnail=args.thumb_ratio,
        task_loss_requirement_thumbnail=args.task_loss_requirement,
    )
    proc_module = importlib.import_module(type(proc).__module__)
    frames = proc_module._load_video_frames(args.video, num_frames=proc.num_video_frames)

    print("[run] _preprocess_videos + real AutoGaze forward (thumbnails)...", flush=True)
    videos_inputs = proc._preprocess_videos([frames])
    gazing_autogaze = proc._get_gazing_info_from_videos(videos_inputs)

    image_size = proc.image_processor.size.get("height", 392) if hasattr(proc.image_processor, "size") else 392
    print("[run] codec_selector.build_gazing_info...", flush=True)
    gazing_codec = codec_selector.build_gazing_info(
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
        w_motion=args.w_motion,
        w_size=args.w_size,
        w_residual=args.w_residual,
        full_first_frame=args.full_first_frame,
        sampled_only=args.sampled_only,
    )

    grid_sizes = [s // proc.target_patch_size for s in proc.target_scales]
    mean = getattr(proc.image_processor, "image_mean", [0.5, 0.5, 0.5])
    std = getattr(proc.image_processor, "image_std", [0.5, 0.5, 0.5])

    thumbs = videos_inputs["pixel_values_videos_thumbnails"][0]  # (T_thumb, 1, C, H, W)
    ag_pos, ag_counts = gazing_autogaze["gazing_pos_thumbnails"][0], gazing_autogaze["num_gazing_each_frame_thumbnails"][0]
    cd_pos, cd_counts = gazing_codec["gazing_pos_thumbnails"][0], gazing_codec["num_gazing_each_frame_thumbnails"][0]
    n_frames = thumbs.shape[0]
    print(f"[info] {n_frames} thumbnail frames, ag_pos.shape={tuple(ag_pos.shape)} cd_pos.shape={tuple(cd_pos.shape)}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 5.2), facecolor="#fcfcfb")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    subtitle = (
        f"  (codec score: w_size={args.w_size}, w_motion={args.w_motion}, w_residual={args.w_residual}"
        f"{', full first frame' if args.full_first_frame else ''})"
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    container = None
    stream = None
    for t in range(n_frames):
        base = denormalize(thumbs[t, 0], mean, std)
        k_ag = int(ag_counts[t, 0].item())
        k_cd = int(cd_counts[t, 0].item())
        ag_cells = [flat_index_to_cell(i, grid_sizes) for i in ag_pos[t, :k_ag].tolist()]
        cd_cells = [flat_index_to_cell(i, grid_sizes) for i in cd_pos[t, :k_cd].tolist()]
        frame_rgb = render_frame_rgb(fig, axes, base, ag_cells, cd_cells, t, n_frames, subtitle)
        if container is None:
            h, w = frame_rgb.shape[:2]
            # Even dimensions required by yuv420p (chroma subsampling halves each axis).
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
        print(f"[frame] {t + 1}/{n_frames}  autogaze={k_ag} codec={k_cd}", flush=True)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    plt.close(fig)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
