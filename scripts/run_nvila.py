import os
import sys

import torch
from transformers import AutoModel, AutoProcessor

model_path = "nvidia/NVILA-8B-HD-Video"
# video_path = "https://huggingface.co/datasets/bfshi/HLVid/resolve/main/example/clip_av_video_5_001.mp4"
# Locate repo root from this script's path
REPO_DIR = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)
video_path = os.path.join(REPO_DIR, "assets", "example_input.mp4")
prompt = "Question: What does the white text on the green road sign say?\n \
A. Hampden St\n \
B. Hampden Ave\n \
C. HampdenBlvd\n \
D. Hampden Rd\n \
Please answer directly with the letter of the correct answer."

# ----- Video processing args -----
num_video_frames = 128           # Total sampled frames for tiles
num_video_frames_thumbnail = 64  # Total sampled frames for thumbnails
max_tiles_video = 48             # Max spatial tiles per video (one tile is 392x392)

# ----- AutoGaze args (tiles) -----
gazing_ratio_tile = [0.2] + [0.06] * 15  # Per-frame max gazing ratios (single float or list)
task_loss_requirement_tile = 0.6

# ----- AutoGaze args (thumbnails) -----
gazing_ratio_thumbnail = 1       # Set to None to skip gazing on thumbnails
task_loss_requirement_thumbnail = None

# ----- Batching -----
max_batch_size_autogaze = 16
max_batch_size_siglip = 32

# Load processor and model
processor = AutoProcessor.from_pretrained(
    model_path,
    autogaze_model_id="nvidia/AutoGaze",
    num_video_frames=num_video_frames,
    num_video_frames_thumbnail=num_video_frames_thumbnail,
    max_tiles_video=max_tiles_video,
    gazing_ratio_tile=gazing_ratio_tile,
    gazing_ratio_thumbnail=gazing_ratio_thumbnail,
    task_loss_requirement_tile=task_loss_requirement_tile,
    task_loss_requirement_thumbnail=task_loss_requirement_thumbnail,
    max_batch_size_autogaze=max_batch_size_autogaze,
    trust_remote_code=True,
)

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    max_batch_size_siglip=max_batch_size_siglip,
)
model.eval()

# Run inference
video_token = processor.tokenizer.video_token
inputs = processor(text=f"{video_token}\n\n{prompt}", videos=video_path, return_tensors="pt")
inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

outputs = model.generate(**inputs)
response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
print(response)