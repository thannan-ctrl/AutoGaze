"""Visualize AutoGaze gaze patches overlaid on video frames.

Saves:
  - assets/gaze_visualization.mp4   – annotated video (4 fps)
  - assets/gaze_grid.png            – all frames as a 4-column PNG grid
"""
import os
import sys
import math

import av
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import imageio

REPO_DIR = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)

from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from autogaze.models.autogaze import AutoGazeImageProcessor, AutoGaze

# ── Load model ───────────────────────────────────────────────────────────────
print("[viz] Loading model...")
autogaze_transform = AutoGazeImageProcessor.from_pretrained("nvidia/AutoGaze")
autogaze_model = AutoGaze.from_pretrained("nvidia/AutoGaze").cuda().eval()

N_PATCHES   = autogaze_model.num_vision_tokens_each_frame          # 196
INPUT_SIZE  = autogaze_model.gazing_model.input_img_size            # 224
GRID        = int(N_PATCHES ** 0.5)                                 # 14
PATCH_PX    = INPUT_SIZE // GRID                                    # 16

# ── Load video ────────────────────────────────────────────────────────────────
video_path = os.path.join(REPO_DIR, "assets", "example_input.mp4")
print(f"[viz] Loading video: {video_path}")
container = av.open(video_path)
sample_indices = list(range(autogaze_model.config.max_num_frames))
raw_video = read_video_pyav(container=container, indices=sample_indices)  # (T, H, W, 3) uint8
container.close()

video_input = transform_video_for_pytorch(raw_video, autogaze_transform)
video_input = video_input[None].cuda()   # (1, T, C, H, W)
T = video_input.shape[1]
print(f"[viz] {T} frames | patch grid {GRID}x{GRID} | patch size {PATCH_PX}px")

# ── Run AutoGaze ──────────────────────────────────────────────────────────────
print("[viz] Running AutoGaze...")
with torch.inference_mode():
    gaze_outputs = autogaze_model(
        {"video": video_input}, gazing_ratio=0.75, task_loss_requirement=0.7
    )

# Each scale has its own spatial resolution: N_s patches → sqrt(N_s)×sqrt(N_s) grid
scales          = gaze_outputs["scales"]                         # e.g. [56, 112, 168, 224]
n_per_scale     = autogaze_model.num_vision_tokens_each_scale_each_frame  # e.g. [4, 16, 36, 196]
n_scales        = len(scales)
print(f"[viz] scales: {scales}  |  patches/scale: {n_per_scale}")

# Build per-scale binary masks: list of (T_eff, N_s) bool tensors
scale_masks = []
T_eff = None
for s in range(n_scales):
    m = gaze_outputs["gazing_mask"][s][0].bool()   # (T_eff, N_s)
    scale_masks.append(m)
    if T_eff is None:
        T_eff = m.shape[0]
    print(f"  scale {s} ({scales[s]}px): mask {m.shape}, {m.sum().item():.0f} gazed patches total")

# Upsample to full T if frame_sampling_rate > 1
if T_eff < T:
    fsr = T // T_eff
    scale_masks = [m.repeat_interleave(fsr, dim=0) for m in scale_masks]

# ── Resize raw frames to model input size ────────────────────────────────────
frames_pil = [
    Image.fromarray(raw_video[t]).resize((INPUT_SIZE, INPUT_SIZE), Image.BICUBIC)
    for t in range(T)
]

# ── Annotate each frame ───────────────────────────────────────────────────────
OVERLAY_RGBA = (0, 220, 100, 110)   # green, semi-transparent
BORDER_RGB   = (0, 200, 80)
TEXT_RGB     = (255, 255, 255)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
except Exception:
    font = ImageFont.load_default()

annotated = []
for t in range(T):
    base = frames_pil[t].copy().convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)

    total_gazed = 0
    for s in range(n_scales):
        n_s       = n_per_scale[s]
        grid_s    = int(round(n_s ** 0.5))
        patch_px_s = INPUT_SIZE // grid_s
        alpha     = 60 + s * 15          # finer scales get slightly higher alpha
        color     = (0, 220, 100, alpha)

        gazed = scale_masks[s][t].nonzero(as_tuple=True)[0].cpu().numpy()
        total_gazed += len(gazed)
        for idx in gazed:
            r, c = divmod(int(idx), grid_s)
            x0, y0 = c * patch_px_s, r * patch_px_s
            x1, y1 = x0 + patch_px_s, y0 + patch_px_s
            draw_ov.rectangle([x0, y0, x1, y1], fill=color)

    blended = Image.alpha_composite(base, overlay).convert("RGB")
    draw = ImageDraw.Draw(blended)

    # Patch borders (finest scale only — coarser ones would clutter)
    finest_mask = scale_masks[-1][t]
    n_s_fine   = n_per_scale[-1]
    grid_fine  = int(round(n_s_fine ** 0.5))
    patch_fine = INPUT_SIZE // grid_fine
    for idx in finest_mask.nonzero(as_tuple=True)[0].cpu().numpy():
        r, c = divmod(int(idx), grid_fine)
        x0, y0 = c * patch_fine, r * patch_fine
        draw.rectangle([x0, y0, x0+patch_fine-1, y0+patch_fine-1], outline=BORDER_RGB, width=1)

    label = f"frame {t:02d}  |  {total_gazed}/{N_PATCHES} patches gazed"
    draw.rectangle([4, 4, INPUT_SIZE - 4, 20], fill=(0, 0, 0, 160))
    draw.text((6, 5), label, fill=TEXT_RGB, font=font)

    annotated.append(np.array(blended))
    print(f"  frame {t:02d}: {total_gazed} patches gazed")

# ── Save annotated video ──────────────────────────────────────────────────────
out_video = os.path.join(REPO_DIR, "assets", "gaze_visualization.mp4")
imageio.mimsave(out_video, annotated, fps=4, codec="libx264")
print(f"[viz] Saved video → {out_video}")

# ── Save frame grid PNG ───────────────────────────────────────────────────────
COLS = 4
ROWS = math.ceil(T / COLS)
grid = Image.new("RGB", (COLS * INPUT_SIZE, ROWS * INPUT_SIZE), (30, 30, 30))
for i, frame in enumerate(annotated):
    r, c = divmod(i, COLS)
    grid.paste(Image.fromarray(frame), (c * INPUT_SIZE, r * INPUT_SIZE))

out_grid = os.path.join(REPO_DIR, "assets", "gaze_grid.png")
grid.save(out_grid)
print(f"[viz] Saved grid  → {out_grid}")
print("[viz] Done.")
