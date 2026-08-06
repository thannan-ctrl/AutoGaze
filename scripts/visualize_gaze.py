"""Visualize AutoGaze gaze patches overlaid on video frames.

Saves:
  - assets/gaze_visualization.mp4   – annotated video
  - assets/gaze_grid.png            – frame grid (first 8 frames)
"""
import os
import sys

import av
import numpy as np
import torch
import cv2

REPO_DIR = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)

from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from autogaze.models.autogaze import AutoGazeImageProcessor, AutoGaze

# ── Load model ──────────────────────────────────────────────────────────────
print("[viz] Loading model...")
autogaze_transform = AutoGazeImageProcessor.from_pretrained("nvidia/AutoGaze")
autogaze_model = AutoGaze.from_pretrained("nvidia/AutoGaze").cuda().eval()

gaze_model     = autogaze_model.gaze_model
N_PATCHES      = gaze_model.num_vision_tokens_each_frame   # 196
INPUT_SIZE     = gaze_model.input_img_size                 # 224
GRID           = int(N_PATCHES ** 0.5)                     # 14
PATCH_PX       = INPUT_SIZE // GRID                        # 16

# ── Load video ───────────────────────────────────────────────────────────────
video_path = os.path.join(REPO_DIR, "assets", "example_input.mp4")
print(f"[viz] Loading video: {video_path}")
container = av.open(video_path)
sample_indices = list(range(autogaze_model.config.max_num_frames))
raw_video = read_video_pyav(container=container, indices=sample_indices)  # (T, H, W, 3) uint8
container.close()

video_input = transform_video_for_pytorch(raw_video, autogaze_transform)
video_input = video_input[None].cuda()  # (1, T, C, H, W)
T = video_input.shape[1]
print(f"[viz] {T} frames  |  patch grid {GRID}x{GRID}  |  patch size {PATCH_PX}px")

# ── Run AutoGaze ─────────────────────────────────────────────────────────────
print("[viz] Running AutoGaze...")
with torch.inference_mode():
    gaze_outputs = autogaze_model(
        {"video": video_input}, gazing_ratio=0.75, task_loss_requirement=0.7
    )

gazing_pos       = gaze_outputs["gazing_pos"][0]          # (total_patches,)
if_padded        = gaze_outputs["if_padded_gazing"][0]    # (total_patches,)
num_each_frame   = gaze_outputs["num_gazing_each_frame"]  # (T,)

pos_per_frame    = gazing_pos.split(num_each_frame.tolist())
pad_per_frame    = if_padded.split(num_each_frame.tolist())

# ── Resize raw frames to model input size for annotation ─────────────────────
frames_resized = []
for t in range(T):
    f = cv2.resize(raw_video[t], (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    frames_resized.append(f)  # (224, 224, 3) uint8 RGB

# ── Build annotated frames ────────────────────────────────────────────────────
OVERLAY_COLOR = (0, 255, 128)   # bright green (RGB)
OVERLAY_ALPHA = 0.45

annotated = []
for t in range(T):
    frame = frames_resized[t].copy().astype(np.float32)

    pos_t  = pos_per_frame[t] - N_PATCHES * t          # within-frame indices
    real   = pos_t[~pad_per_frame[t]].cpu().numpy()    # non-padded

    overlay = frame.copy()
    for idx in real:
        r = int(idx) // GRID
        c = int(idx) % GRID
        y0, y1 = r * PATCH_PX, (r + 1) * PATCH_PX
        x0, x1 = c * PATCH_PX, (c + 1) * PATCH_PX
        overlay[y0:y1, x0:x1] = OVERLAY_COLOR

    blended = (OVERLAY_ALPHA * overlay + (1 - OVERLAY_ALPHA) * frame).clip(0, 255).astype(np.uint8)

    # Draw patch borders for gazed patches
    for idx in real:
        r = int(idx) // GRID
        c = int(idx) % GRID
        y0, y1 = r * PATCH_PX, (r + 1) * PATCH_PX
        x0, x1 = c * PATCH_PX, (c + 1) * PATCH_PX
        cv2.rectangle(blended, (x0, y0), (x1 - 1, y1 - 1), (0, 200, 80), 1)

    # Label: frame index and gazed patch count
    label = f"frame {t}  |  {len(real)}/{N_PATCHES} patches gazed"
    cv2.putText(blended, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    annotated.append(blended)

# ── Save annotated video ──────────────────────────────────────────────────────
out_video = os.path.join(REPO_DIR, "assets", "gaze_visualization.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_video, fourcc, 4.0, (INPUT_SIZE, INPUT_SIZE))
for frame in annotated:
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
writer.release()
print(f"[viz] Saved video → {out_video}")

# ── Save frame grid PNG ───────────────────────────────────────────────────────
import math

COLS = 4
show = min(T, 16)
ROWS = math.ceil(show / COLS)
grid_h = ROWS * INPUT_SIZE
grid_w = COLS * INPUT_SIZE
grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

for i in range(show):
    r, c = divmod(i, COLS)
    grid_img[r * INPUT_SIZE:(r + 1) * INPUT_SIZE, c * INPUT_SIZE:(c + 1) * INPUT_SIZE] = annotated[i]

out_grid = os.path.join(REPO_DIR, "assets", "gaze_grid.png")
cv2.imwrite(out_grid, cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
print(f"[viz] Saved grid  → {out_grid}")
print("[viz] Done.")
