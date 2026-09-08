"""Side-by-side visualization: which patches AutoGaze's trained selector keeps
vs. which patches codec mode keeps, for the same video/frames/spatial tile.

Bypasses the scripts/breakdown benchmark harness entirely -- calls the
vendored NVILAProcessor's own _preprocess_videos / _get_gazing_info_from_videos
directly (so AutoGaze's real model runs, unshortcircuited) and calls
codec_selector.build_gazing_info() directly (so codec mode's real scoring
pipeline runs) against the exact same geometry, then overlays each selector's
kept patches on the same underlying SigLIP tile-crop image.

Usage:
    REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 python3 scripts/visualize_codec_vs_autogaze.py \
        --video data/egoschema/videos/<q_uid>.mp4 --tile 0 --frames 0,5,10 \
        --out figures/codec_vs_autogaze_patches.png
"""
import argparse
import importlib
import os
import sys

REPO_DIR = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from scripts.breakdown import codec_selector

BLUE = "#2a78d6"    # AutoGaze (trained selector)
ORANGE = "#eb6834"  # codec (heuristic selector)
DIM_ALPHA = 0.55     # darken-out factor for non-selected area


def denormalize(tile_chw: torch.Tensor, mean, std) -> np.ndarray:
    """(C,H,W) normalized tensor -> (H,W,C) uint8 array for display."""
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    img = (tile_chw.float() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def flat_index_to_cell(idx: int, grid_sizes: list[int]):
    """flat multiscale patch index -> (row, col, grid_size) of the scale it belongs to.

    codec's gazing_pos values are per-frame-local (< total_patches, see
    codec_selector.build_gazing_info's topk_ratio/argsort). AutoGaze's real
    trained selector instead emits a *global* index across the whole T_tile
    (e.g. 16) frame chunk -- frame_local_idx * total_patches + local_idx --
    confirmed empirically (max observed index ~= 15 * total_patches +
    local, for a 16-frame chunk). `% total_patches` recovers the per-frame
    local patch index in both cases (no-op when already local)."""
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


def selected_cells_for_frame(gazing_pos_tiles, num_gazing_each_frame_tiles, tile_idx, frame_idx, grid_sizes):
    """Returns list of (row, col, grid_size) cells selected for one (tile, frame)."""
    counts = num_gazing_each_frame_tiles[tile_idx]
    start = int(counts[:frame_idx].sum().item())
    k = int(counts[frame_idx].item())
    idxs = gazing_pos_tiles[tile_idx, start:start + k].tolist()
    return [flat_index_to_cell(i, grid_sizes) for i in idxs]


def draw_selection(ax, base_img: np.ndarray, cells, color, title):
    h, w = base_img.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for row, col, g in cells:
        y0, y1 = int(row * h / g), int((row + 1) * h / g)
        x0, x1 = int(col * w / g), int((col + 1) * w / g)
        mask[y0:y1, x0:x1] = True

    bright = base_img.astype(np.float32)
    dimmed = bright * (1 - DIM_ALPHA)
    combined = np.where(mask[:, :, None], bright, dimmed).astype(np.uint8)
    ax.imshow(combined)

    for row, col, g in cells:
        y0, y1 = row * h / g, (row + 1) * h / g
        x0, x1 = col * w / g, (col + 1) * w / g
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color,
                                          linewidth=1.1, zorder=3))
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(color); s.set_linewidth(2.5)
    ax.set_title(title, fontsize=10, color=color, fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tile", type=int, default=0, help="spatial tile index to visualize")
    ap.add_argument("--frames", default="0,5,10", help="comma-separated frame-within-chunk positions")
    ap.add_argument("--num-video-frames", type=int, default=16)
    ap.add_argument("--max-tiles-video", type=int, default=16)
    ap.add_argument("--codec-ratio-scale", type=float, default=0.28)
    ap.add_argument("--out", default="figures/codec_vs_autogaze_patches.png")
    args = ap.parse_args()

    frame_positions = [int(x) for x in args.frames.split(",")]
    device = os.environ.get("NVILA_DEVICE", "cuda:0")

    print("[setup] loading model...", flush=True)
    model = AutoModel.from_pretrained(
        "nvidia/NVILA-8B-HD-Video", trust_remote_code=True, dtype=torch.bfloat16, max_batch_size_siglip=32,
    ).to(device)
    model.eval()

    ratio_schedule = [0.2] + [0.06] * 15
    proc = AutoProcessor.from_pretrained(
        "nvidia/NVILA-8B-HD-Video", trust_remote_code=True,
        num_video_frames=args.num_video_frames,
        num_video_frames_thumbnail=max(args.num_video_frames // 2, 1),
        max_tiles_video=args.max_tiles_video,
        max_batch_size_autogaze=64,
        autogaze_model_id="nvidia/AutoGaze",
        gazing_ratio_tile=ratio_schedule,
        task_loss_requirement_tile=0.6,
        gazing_ratio_thumbnail=1,
        task_loss_requirement_thumbnail=None,
    )

    proc_module = importlib.import_module(type(proc).__module__)
    frames = proc_module._load_video_frames(args.video, num_frames=proc.num_video_frames)

    print("[run] _preprocess_videos + real AutoGaze forward...", flush=True)
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
        gazing_ratio_tile=[r * args.codec_ratio_scale for r in ratio_schedule],
        gazing_ratio_thumbnail=1,
    )

    grid_sizes = [s // proc.target_patch_size for s in proc.target_scales]
    mean = getattr(proc.image_processor, "image_mean", [0.5, 0.5, 0.5])
    std = getattr(proc.image_processor, "image_std", [0.5, 0.5, 0.5])

    siglip_tiles = videos_inputs["pixel_values_videos_tiles"][0]  # (num_tiles, T_tile, C, H, W)
    num_tiles = siglip_tiles.shape[0]
    if args.tile >= num_tiles:
        raise SystemExit(f"--tile {args.tile} out of range (video has {num_tiles} spatial x temporal tiles)")

    ag_pos, ag_counts = gazing_autogaze["gazing_pos_tiles"][0], gazing_autogaze["num_gazing_each_frame_tiles"][0]
    cd_pos, cd_counts = gazing_codec["gazing_pos_tiles"][0], gazing_codec["num_gazing_each_frame_tiles"][0]

    n = len(frame_positions)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 8.2), facecolor="#fcfcfb",
                               gridspec_kw={"hspace": 0.45, "wspace": 0.12})
    if n == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(
        "AutoGaze (trained selector) vs. codec (heuristic) patch selection",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.text(0.5, 0.935, f"{os.path.basename(args.video)}, spatial tile {args.tile}",
              fontsize=9.5, color="#52514e", ha="center")

    for col, f in enumerate(frame_positions):
        base = denormalize(siglip_tiles[args.tile, f], mean, std)
        ag_cells = selected_cells_for_frame(ag_pos, ag_counts, args.tile, f, grid_sizes)
        cd_cells = selected_cells_for_frame(cd_pos, cd_counts, args.tile, f, grid_sizes)
        draw_selection(axes[0, col], base, ag_cells, BLUE,
                        f"AutoGaze  frame {f}  ({len(ag_cells)} patches)")
        draw_selection(axes[1, col], base, cd_cells, ORANGE,
                        f"codec  frame {f}  ({len(cd_cells)} patches)")

    fig.tight_layout(rect=(0, 0, 1, 0.91))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=180, facecolor="#fcfcfb")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
