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

# Debug: print raw output shapes and counts
print(f"[dbg] gazing_pos shape:        {gaze_outputs['gazing_pos'].shape}")
print(f"[dbg] if_padded_gazing shape:  {gaze_outputs['if_padded_gazing'].shape}")
print(f"[dbg] num_gazing_each_frame:   {gaze_outputs['num_gazing_each_frame']}")
print(f"[dbg] actual gazed (non-pad):  {(~gaze_outputs['if_padded_gazing']).sum().item()}")
print(f"[dbg] num gazing_mask scales:  {len(gaze_outputs['gazing_mask'])}")
print(f"[dbg] gazing_mask[0] shape:    {gaze_outputs['gazing_mask'][0].shape}")
print(f"[dbg] gazing_mask[0] sum:      {gaze_outputs['gazing_mask'][0].sum().item()}")
print(f"[dbg] per-frame sums:          {gaze_outputs['gazing_mask'][0][0].sum(dim=-1).tolist()}")

# gazing_mask[0]: (B, T_eff, N_patches) — 1 = gazed, 0 = not gazed
gaze_mask = gaze_outputs["gazing_mask"][0][0].bool()   # (T_eff, N_patches)
T_eff = gaze_mask.shape[0]
print(f"[dbg] gaze_mask shape (T_eff, N): {gaze_mask.shape}")
# If T_eff < T (frame_sampling_rate > 1), repeat each mask to cover all raw frames
if T_eff < T:
    frame_sampling_rate = T // T_eff
    gaze_mask = gaze_mask.repeat_interleave(frame_sampling_rate, dim=0)  # (T, N)
    print(f"[dbg] upsampled gaze_mask to: {gaze_mask.shape} (frame_sampling_rate={frame_sampling_rate})")

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

    gazed_indices = gaze_mask[t].nonzero(as_tuple=True)[0].cpu().numpy()
    n_gazed = len(gazed_indices)

    # Semi-transparent overlay layer
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    for idx in gazed_indices:
        r, c = divmod(int(idx), GRID)
        x0, y0 = c * PATCH_PX, r * PATCH_PX
        x1, y1 = x0 + PATCH_PX, y0 + PATCH_PX
        draw_ov.rectangle([x0, y0, x1, y1], fill=OVERLAY_RGBA)

    blended = Image.alpha_composite(base, overlay).convert("RGB")
    draw = ImageDraw.Draw(blended)

    # Patch borders
    for idx in gazed_indices:
        r, c = divmod(int(idx), GRID)
        x0, y0 = c * PATCH_PX, r * PATCH_PX
        x1, y1 = x0 + PATCH_PX - 1, y0 + PATCH_PX - 1
        draw.rectangle([x0, y0, x1, y1], outline=BORDER_RGB, width=1)

    # Label
    label = f"frame {t:02d}  |  {n_gazed}/{N_PATCHES} patches gazed"
    draw.rectangle([4, 4, INPUT_SIZE - 4, 20], fill=(0, 0, 0, 160))
    draw.text((6, 5), label, fill=TEXT_RGB, font=font)

    annotated.append(np.array(blended))
    print(f"  frame {t:02d}: {n_gazed} patches gazed")

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
