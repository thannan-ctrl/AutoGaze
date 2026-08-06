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

gaze_model  = autogaze_model.gaze_model
N_PATCHES   = gaze_model.num_vision_tokens_each_frame   # 196
INPUT_SIZE  = gaze_model.input_img_size                 # 224
GRID        = int(N_PATCHES ** 0.5)                     # 14
PATCH_PX    = INPUT_SIZE // GRID                        # 16

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

gazing_pos     = gaze_outputs["gazing_pos"][0]          # (total_patches,)
if_padded      = gaze_outputs["if_padded_gazing"][0]    # (total_patches,)
num_each_frame = gaze_outputs["num_gazing_each_frame"]  # (T,)

pos_per_frame = gazing_pos.split(num_each_frame.tolist())
pad_per_frame = if_padded.split(num_each_frame.tolist())

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

    pos_t = pos_per_frame[t] - N_PATCHES * t          # within-frame indices
    real  = pos_t[~pad_per_frame[t]].cpu().numpy()    # non-padded

    # Semi-transparent overlay layer
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    for idx in real:
        r, c = divmod(int(idx), GRID)
        x0, y0 = c * PATCH_PX, r * PATCH_PX
        x1, y1 = x0 + PATCH_PX, y0 + PATCH_PX
        draw_ov.rectangle([x0, y0, x1, y1], fill=OVERLAY_RGBA)

    blended = Image.alpha_composite(base, overlay).convert("RGB")
    draw = ImageDraw.Draw(blended)

    # Patch borders
    for idx in real:
        r, c = divmod(int(idx), GRID)
        x0, y0 = c * PATCH_PX, r * PATCH_PX
        x1, y1 = x0 + PATCH_PX - 1, y0 + PATCH_PX - 1
        draw.rectangle([x0, y0, x1, y1], outline=BORDER_RGB, width=1)

    # Label
    label = f"frame {t:02d}  |  {len(real)}/{N_PATCHES} patches"
    draw.rectangle([4, 4, INPUT_SIZE - 4, 20], fill=(0, 0, 0, 160))
    draw.text((6, 5), label, fill=TEXT_RGB, font=font)

    annotated.append(np.array(blended))
    print(f"  frame {t:02d}: {len(real)} patches gazed")

# ── Save annotated video ──────────────────────────────────────────────────────
out_video = os.path.join(REPO_DIR, "assets", "gaze_visualization.mp4")
writer = imageio.get_writer(out_video, fps=4, codec="libx264", quality=8)
for frame in annotated:
    writer.append_data(frame)
writer.close()
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
