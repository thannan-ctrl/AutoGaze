"""Environment-derived constants and per-mode processor kwargs.

Importing this module has one side effect: it puts REPO_DIR on sys.path, which
is required before `AutoProcessor.from_pretrained(..., trust_remote_code=True)`
can dynamically import the local `autogaze` package.
"""
import os
import sys

REPO_DIR = os.environ.get(
    "REPO_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, REPO_DIR)

MODEL_PATH = "nvidia/NVILA-8B-HD-Video"
DEVICE = os.environ.get("NVILA_DEVICE", "cuda:0")
DATA_DIR = os.path.join(REPO_DIR, "data", "egoschema")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "25"))
SEED = 0

LETTERS = "ABCDE"

_fixed_nvf = os.environ.get("FIXED_NUM_VIDEO_FRAMES")
FIXED_NUM_VIDEO_FRAMES = int(_fixed_nvf) if _fixed_nvf else None

COMMON_KW = dict(
    num_video_frames=FIXED_NUM_VIDEO_FRAMES or 128,
    num_video_frames_thumbnail=max((FIXED_NUM_VIDEO_FRAMES or 128) // 2, 1),
    max_tiles_video=FIXED_NUM_VIDEO_FRAMES or 48,
    max_batch_size_autogaze=int(os.environ.get("MAX_BATCH_SIZE_AUTOGAZE", "16")),
    autogaze_model_id="nvidia/AutoGaze",
)

# gazing_ratio=1 + task_loss_requirement=None means "keep all patches" -- the
# processor's _should_gaze_all_patches skip condition -- so dense never
# invokes the AutoGaze model at all.
CONFIGS = {
    "dense": dict(
        gazing_ratio_tile=1,
        task_loss_requirement_tile=None,
        gazing_ratio_thumbnail=1,
        task_loss_requirement_thumbnail=None,
    ),
    "autogaze": dict(
        gazing_ratio_tile=[0.2] + [0.06] * 15,
        task_loss_requirement_tile=0.6,
        gazing_ratio_thumbnail=1,
        task_loss_requirement_thumbnail=None,
    ),
}


def dense_frame_budgets() -> list:
    """Dense retries at halved frame budgets on OOM (see run.py); with a
    fixed budget and DENSE_RETRY unset, there's just the one budget."""
    if not FIXED_NUM_VIDEO_FRAMES:
        return [128, 64, 32, 16, 8]
    if not os.environ.get("DENSE_RETRY"):
        return [FIXED_NUM_VIDEO_FRAMES]
    budgets = []
    nf = FIXED_NUM_VIDEO_FRAMES
    while nf >= 8:
        budgets.append(nf)
        nf //= 2
    return budgets


def result_suffix() -> str:
    return f"_nvf{FIXED_NUM_VIDEO_FRAMES}" if FIXED_NUM_VIDEO_FRAMES else ""
