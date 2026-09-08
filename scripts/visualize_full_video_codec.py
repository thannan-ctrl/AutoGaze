"""Run codec mode the way AutoGaze's own QUICK_START.md actually prescribes:
back-to-back 16-frame chunks covering the WHOLE video starting at frame 0 --
no global sparse subsampling across the video's full length (that's what the
real NVILA-HD-Video model does instead, and what our benchmark harness
mirrors; see Sparse_Frame_Sampling_Concern.md).

Key trick: num_video_frames is set to the video's native frame count, so
`_sampled_frame_indices`'s np.linspace(0, frame_count-1, num_frames) becomes
the identity -- every consecutive real frame, no gaps. sampled_only=True is
the only sane encoding mode here (there's no gap between "sampled" frames
left to fill with windowed real-context, since every frame is already
included).

Critically, max_tiles_video is passed as a small constant, NOT coupled to
num_video_frames the way processor.py's real caller does it (`max_tiles_video
= num_video_frames`, see codec_selector.py's own module docstring / the
Feasibility doc's "root-caused" note) -- that coupling is what made high-nvf
runs combinatorially slow. Spatial tiling is a per-frame decision (depends on
aspect ratio/resolution) and has nothing to do with how many frames total
are in the video, so decoupling them here is a correctness fix for this
call site, not a shortcut.

Usage:
    REPO_DIR=$(pwd) python3 scripts/visualize_full_video_codec.py \
        --video data/egoschema/videos/<q_uid>.mp4 \
        --out figures/full_video_codec_egoschema.mp4
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

ORANGE = "#eb6834"
DIM_ALPHA = 0.55


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--duration-min", type=float, default=None,
                     help="crop to at most this many minutes of real video time, from frame 0 "
                          "(default: whole video)")
    ap.add_argument("--thumb-ratio", type=float, default=0.12)
    ap.add_argument("--max-tiles-video", type=int, default=16,
                     help="spatial-tiling budget, decoupled from num_video_frames "
                          "(see module docstring) -- 16 matches the standard nvf=16 setting")
    ap.add_argument("--out", default="figures/full_video_codec.mp4")
    ap.add_argument("--fps", type=int, default=12, help="output video fps")
    ap.add_argument("--render-stride", type=int, default=1,
                     help="render every Nth computed frame (subsamples the OUTPUT video "
                          "only, for practical render time -- the codec computation itself "
                          "always covers every native frame)")
    ap.add_argument("--full-first-frame", action="store_true",
                     help="keep every patch (all scales) on the first frame of EVERY "
                          "16-frame chunk, not just the very first frame of the whole "
                          "video -- codec_selector.build_gazing_info's own full_first_frame "
                          "flag only special-cases thumb_i==0 (global), so this is applied "
                          "here in the script instead, matching AutoGaze's actual per-chunk "
                          "anchor-frame design (each 16-frame chunk gets its own I-canvas)")
    args = ap.parse_args()

    n_total, fps = native_frame_count(args.video)
    n_native = n_total
    if args.duration_min is not None:
        n_native = min(n_native, int(fps * 60 * args.duration_min))
    # AutoGaze processes fixed 16-frame chunks (its own QUICK_START.md), so
    # num_video_frames must be a multiple of 16 -- round down to the nearest
    # whole chunk rather than padding/repeating frames.
    n_native = (n_native // 16) * 16
    print(f"[setup] video has {n_total} native frames @ {fps:.1f}fps -- "
          f"using the first {n_native} ({n_native / fps:.1f}s, nearest whole-16-chunk count), "
          f"from frame 0", flush=True)

    print("[setup] building processor (loads AutoGaze weights, no forward pass)...", flush=True)
    proc = AutoProcessor.from_pretrained(
        "nvidia/NVILA-8B-HD-Video", trust_remote_code=True,
        num_video_frames=n_native, num_video_frames_thumbnail=n_native,
        max_tiles_video=n_native,  # unused by our direct build_gazing_info call below; processor
                                    # construction just needs *a* value here
        max_batch_size_autogaze=32, autogaze_model_id="nvidia/AutoGaze",
        gazing_ratio_tile=1, task_loss_requirement_tile=None,
        gazing_ratio_thumbnail=args.thumb_ratio, task_loss_requirement_thumbnail=0.6,
    )

    image_size = proc.image_processor.size.get("height", 392) if hasattr(proc.image_processor, "size") else 392

    print(f"[run] codec_selector.build_gazing_info(num_video_frames={n_native}, "
          f"max_tiles_video={args.max_tiles_video}, sampled_only=True)...", flush=True)
    gazing = codec_selector.build_gazing_info(
        video_path=args.video,
        num_video_frames=n_native,
        num_video_frames_thumbnail=n_native,
        max_tiles_video=args.max_tiles_video,
        autogaze_max_num_frames=proc._autogaze_model.config.max_num_frames,
        image_size=image_size,
        scales=proc.target_scales,
        patch_size=proc.target_patch_size,
        gazing_ratio_tile=0.01,
        gazing_ratio_thumbnail=args.thumb_ratio,
        sampled_only=True,
    )

    grid_sizes = [s // proc.target_patch_size for s in proc.target_scales]
    total_patches = sum(g * g for g in grid_sizes)
    mean = getattr(proc.image_processor, "image_mean", [0.5, 0.5, 0.5])
    std = getattr(proc.image_processor, "image_std", [0.5, 0.5, 0.5])

    import importlib
    proc_module = importlib.import_module(type(proc).__module__)
    frames = proc_module._load_video_frames(args.video, num_frames=n_native)
    videos_inputs = proc._preprocess_videos([frames])
    thumbs = videos_inputs["pixel_values_videos_thumbnails"][0]

    pos, counts = gazing["gazing_pos_thumbnails"][0], gazing["num_gazing_each_frame_thumbnails"][0]
    n_frames = thumbs.shape[0]
    print(f"[info] {n_frames} thumbnail frames computed, rendering every {args.render_stride}", flush=True)

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.2), facecolor="#fcfcfb")
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    container = None
    stream = None
    render_indices = list(range(0, n_frames, args.render_stride))
    for ri, t in enumerate(render_indices):
        base = denormalize(thumbs[t, 0], mean, std)
        if args.full_first_frame and t % 16 == 0:
            k = total_patches
            cells = [flat_index_to_cell(i, grid_sizes) for i in range(total_patches)]
        else:
            k = int(counts[t, 0].item())
            cells = [flat_index_to_cell(i, grid_sizes) for i in pos[t, :k].tolist()]
        ax.cla()
        draw_selection(ax, base, cells, ORANGE, f"codec (dense, from frame 0) -- {k} patches")
        fig.suptitle(f"native frame {t}/{n_frames - 1}  (chunk {t // 16}, pos {t % 16})", fontsize=10, color="#52514e")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame_rgb = buf[:, :, :3].copy()
        if container is None:
            h, w = frame_rgb.shape[:2]
            w, h = w - (w % 2), h - (h % 2)
            container = av.open(args.out, mode="w")
            stream = container.add_stream("libx264", rate=args.fps)
            stream.width, stream.height = w, h
            stream.pix_fmt = "yuv420p"
            stream.options = {"movflags": "faststart"}
        av_frame = av.VideoFrame.from_ndarray(frame_rgb[:h, :w], format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)
        if ri % 50 == 0:
            print(f"[render] {ri + 1}/{len(render_indices)}  native_frame={t}  patches={k}", flush=True)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    plt.close(fig)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
