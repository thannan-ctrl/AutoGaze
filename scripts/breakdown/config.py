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
DATA_DIR = os.path.join(REPO_DIR, "data")
DATASET = os.environ.get("DATASET", "egoschema")  # "egoschema" | "video_mme"
_n_samples_env = os.environ.get("N_SAMPLES", "25")
N_SAMPLES = 0 if _n_samples_env.lower() == "full" else int(_n_samples_env)  # 0 = all usable items
SEED = 0

LETTERS = "ABCDE"

_fixed_nvf = os.environ.get("FIXED_NUM_VIDEO_FRAMES")
FIXED_NUM_VIDEO_FRAMES = int(_fixed_nvf) if _fixed_nvf else None

# codec always fills its fixed top-k budget (no EOS early-stop), so its raw
# average token count runs higher than AutoGaze's actual (often-early-stopped)
# average. Scale codec's gazing_ratio_tile by this factor to bring its average
# token count down to roughly match AutoGaze's, for a token-count-matched
# latency/accuracy comparison. 1.0 = original (unscaled) behavior.
_CODEC_RATIO_SCALE = float(os.environ.get("CODEC_RATIO_SCALE", "1.0"))

# codec_selector.build_gazing_info's scoring-weight knobs, env-driven so each
# LLaVA-OneVision-2-mimicry mechanism can be ablated in real benchmark runs
# without code changes -- see Codec_Selector_Feasibility.md's "Integration"
# section and the plan in .claude/plans for what each knob does. Defaults match
# build_gazing_info's own defaults exactly, so leaving these unset reproduces
# pre-existing behavior bit-for-bit.
CODEC_SCORE_KW = dict(
    w_motion=float(os.environ.get("CODEC_W_MOTION", "1.0")),
    skip_penalty=float(os.environ.get("CODEC_SKIP_PENALTY", "0.1")),
    w_size=float(os.environ.get("CODEC_W_SIZE", "1.0")),
    w_residual=float(os.environ.get("CODEC_W_RESIDUAL", "0.0")),
    full_first_frame=os.environ.get("CODEC_FULL_FIRST_FRAME", "0") == "1",
    sampled_only=os.environ.get("CODEC_SAMPLED_ONLY", "0") == "1",
    gop_restart=int(os.environ["CODEC_GOP_RESTART"]) if os.environ.get("CODEC_GOP_RESTART") else None,
)

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
    # Nominal budget same as "autogaze", but codec always fills it exactly
    # (fixed top-k, no EOS early-stop like AutoGaze's autoregressive selector),
    # so its actual average token count runs higher than AutoGaze's. Scale it
    # down via CODEC_RATIO_SCALE (see below) to match AutoGaze's *average*
    # token count for an apples-to-apples latency/accuracy comparison at equal
    # token budget -- see Codec_Selector_Feasibility.md's "Token-count-matched"
    # section for how the scale factor was calibrated.
    "codec": dict(
        gazing_ratio_tile=[r * _CODEC_RATIO_SCALE for r in ([0.2] + [0.06] * 15)],
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
    # EXTRA_SUFFIX lets codec-scoring ablation runs (Step 1+ of the
    # LLaVA-OneVision-2-mimicry plan) write to distinct result files instead of
    # clobbering the baseline/each other, e.g. EXTRA_SUFFIX=_wresidual1.
    nvf_suffix = f"_nvf{FIXED_NUM_VIDEO_FRAMES}" if FIXED_NUM_VIDEO_FRAMES else ""
    return nvf_suffix + os.environ.get("EXTRA_SUFFIX", "")
