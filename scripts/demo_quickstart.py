"""Run the QUICK_START basic AutoGaze inference on assets/example_input.mp4."""
import os
import sys

import av
import torch

# Locate repo root from this script's path
REPO_DIR = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)

from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from autogaze.models.autogaze import AutoGazeImageProcessor, AutoGaze

print(f"[demo] Repo dir: {REPO_DIR}")
print(f"[demo] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[demo] GPU: {torch.cuda.get_device_name(0)}")

# Load model from HuggingFace
print("[demo] Loading AutoGaze model...")
autogaze_transform = AutoGazeImageProcessor.from_pretrained("bfshi/AutoGaze")
autogaze_model = AutoGaze.from_pretrained("bfshi/AutoGaze")
autogaze_model = autogaze_model.cuda().eval()

# Load video
video_path = os.path.join(REPO_DIR, "assets", "example_input.mp4")
print(f"[demo] Loading video: {video_path}")
container = av.open(video_path)
sample_indices = list(range(autogaze_model.config.max_num_frames))
raw_video = read_video_pyav(container=container, indices=sample_indices)
container.close()

# Preprocess
video_input = transform_video_for_pytorch(raw_video, autogaze_transform)
video_input = video_input[None].cuda()  # B * T * C * H * W
print(f"[demo] Video input shape: {video_input.shape}")

# Run AutoGaze
print("[demo] Running AutoGaze...")
with torch.inference_mode():
    gaze_outputs = autogaze_model({"video": video_input}, gazing_ratio=0.75, task_loss_requirement=0.7)

print(f"[demo] gazing_pos shape:       {gaze_outputs['gazing_pos'].shape}")
print(f"[demo] if_padded_gazing shape: {gaze_outputs['if_padded_gazing'].shape}")
print(f"[demo] actual gazed patches:   {(~gaze_outputs['if_padded_gazing']).sum(dim=-1).item()}")
print(f"[demo] num_gazing_each_frame:  {gaze_outputs['num_gazing_each_frame']}")
print("[demo] Done.")
