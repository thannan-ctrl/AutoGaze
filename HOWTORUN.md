# How to run — AutoGaze × vLLM sparse ViT benchmark

Step-by-step guide to reproduce the dense vs EVS vs sparse\_vit comparison
from scratch on a new machine.

---

## 1. Setup

### Clone and checkout

```bash
git clone <repo-url> AutoGaze
cd AutoGaze
git checkout vllm-integration-experiments
```

### Install the autogaze package (host Python only — for the orchestrator)

```bash
pip install -e . --quiet
```

The orchestrator (`runtime_analysis.py`) runs on the host and just calls `docker run`.
It doesn't need GPU or heavy deps — only `torch` is needed for the result JSON parsing.

### HuggingFace cache

The Docker containers mount your HF cache so models download once and are reused.
Set `$HF_HOME` to wherever you keep your HF cache:

```bash
export HF_HOME=/path/to/your/hf_cache   # default: /home/scratch.thannan_wwfo/hf_cache
```

Models downloaded on first run (~4 GB for Qwen3-VL-2B, ~200 MB for AutoGaze).

---

## 2. Run the benchmark

One command. No conda, no separate preprocessing step, no env switching.

```bash
python3 scripts/runtime_analysis.py \
    --modes dense evs sparse_vit \
    --single-env \
    --gazing-ratio 0.245 \
    --pruning-rate 0.5 \
    --reps 3
```

**Arguments:**

| Flag | What it does | Value used |
|---|---|---|
| `--modes` | Which modes to benchmark | `dense evs sparse_vit` |
| `--single-env` | Run AutoGaze preprocessing inside Docker (no conda needed) | — |
| `--gazing-ratio` | AutoGaze patch selection target for sparse\_vit | `0.245` |
| `--pruning-rate` | EVS compression rate (also passed to vLLM for slot allocation) | `0.5` |
| `--reps` | Inference reps per mode (rep 1 = warmup, rest measured) | `3` |

**What runs:** Three Docker containers sequentially, one per mode.

```
dense      → no compression, measures baseline
evs        → vLLM's built-in EVS (cosine similarity post-ViT)
sparse_vit → AutoGaze selects patches pre-ViT (gather op before transformer blocks)
```

Each container:
1. Loads Qwen3-VL-2B-Instruct (~26 s, cached after first run)
2. For `sparse_vit`: installs missing deps + runs AutoGaze on the video (~10 s first run)
3. Runs 3 inference reps, reports ViT time / LM time / answer

Total wall time: ~25 min (mostly model loading, 3 containers).

---

## 3. Expected output

```
================================================================================
END-TO-END RUNTIME ANALYSIS  —  AutoGaze × vLLM
================================================================================
Model:         Qwen/Qwen3-VL-2B-Instruct
Video:         assets/example_input.mp4  (6 frames, 448×448)
GPU:           GB200
Reps:          3  (first=warmup, rest measured; avg reported)

Mode                      Tokens  vs Dense    ViT (ms)    LM (ms)   Infer (ms)  Answer
-----------------------  -------  --------  ----------  ---------  -----------  ------
  dense                      670         —         n/a        n/a       17,348       C
  evs                        376      -44%       3,148     13,692       16,840       C
  sparse_vit (ratio=0.245)   356      -47%         691     13,497       14,188       C

EVS vs sparse_vit:
  Tokens:    EVS=376  sparse_vit=356  (20 fewer)
  ViT:       3,148 ms → 691 ms  (4.56× faster)
  Inference: 16,840 ms → 14,188 ms  (2,652 ms / 16% faster)
================================================================================
```

`infer_ms` excludes model load (~26 s per container). All modes answer correctly (C).

`infer_ms` excludes model load. All modes give the same answer (C).

---

## 4. Tune the gazing ratio

After the benchmark saves `runtime_analysis.json` (dense + EVS cached), you can
sweep different ratios without re-running those two modes:

```bash
# ratio=0.245 matched EVS token count → ViT 4.6× faster, 15% faster end-to-end
python3 scripts/compare_sparse_vit_ratio.py --gazing-ratio 0.245 --reps 3

# more aggressive: ratio=0.1 → fewer tokens, bigger ViT speedup, verify accuracy holds
python3 scripts/compare_sparse_vit_ratio.py --gazing-ratio 0.1 --reps 3
```

---

## 5. Run a single mode manually

Run one mode directly in Docker without the orchestrator:

```bash
REPO=$(pwd)
HF=/path/to/hf_cache   # same as $HF_HOME
IMAGE=nvcr.io/nvidia/vllm:26.07-py3

# dense — baseline, no compression
docker run --rm --gpus all --shm-size 16g \
    -v $HF:/root/.cache/huggingface \
    -v $REPO:/workspace/AutoGaze \
    -e REPO_DIR=/workspace/AutoGaze \
    -e HF_HOME=/root/.cache/huggingface \
    $IMAGE \
    python /workspace/AutoGaze/scripts/worker.py \
        --mode dense --reps 3

# evs — built-in vLLM EVS
docker run --rm --gpus all --shm-size 16g \
    -v $HF:/root/.cache/huggingface \
    -v $REPO:/workspace/AutoGaze \
    -e REPO_DIR=/workspace/AutoGaze \
    -e HF_HOME=/root/.cache/huggingface \
    $IMAGE \
    python /workspace/AutoGaze/scripts/worker.py \
        --mode evs --pruning-rate 0.5 --reps 3

# sparse_vit — AutoGaze pre-ViT selection (single env, inline preprocessing)
docker run --rm --gpus all --shm-size 16g \
    -v $HF:/root/.cache/huggingface \
    -v $REPO:/workspace/AutoGaze \
    -e REPO_DIR=/workspace/AutoGaze \
    -e HF_HOME=/root/.cache/huggingface \
    $IMAGE \
    python /workspace/AutoGaze/scripts/worker.py \
        --mode sparse_vit \
        --video /workspace/AutoGaze/assets/example_input.mp4 \
        --gazing-ratio 0.245 \
        --pruning-rate 0.5 \
        --reps 3
```

---

## 6. What the numbers mean

**`gazing-ratio` vs EVS `pruning-rate` are not the same thing:**

| Parameter | Controls |
|---|---|
| `--pruning-rate 0.5` (EVS) | Keep the top 50% of post-merge tokens by cosine similarity → fixed 376 tokens |
| `--gazing-ratio 0.245` (sparse\_vit) | AutoGaze targets ~24.5% patch retention per frame, stops early when confident → variable K |

At `ratio=0.245`, AutoGaze selected 1,326 of 6,144 ViT patches (21.6%) → 331 merged
tokens → 356 prompt tokens. This closely matches EVS's 376, so LM cost is equal and
the 4.6× ViT speedup translates directly to end-to-end gain.

At `ratio=0.5`, AutoGaze selects ~44% → 681 merged tokens — nearly 2× more than EVS,
so the LM penalty outweighs the ViT saving. Always align `gazing-ratio` to your target
token count, not to EVS's `pruning-rate`.

**AutoGaze K is stochastic:** The model stops early when its `task_loss_prediction_head`
is confident. K can vary across runs at the same ratio (e.g. 1,326 or 433 at ratio=0.245).
Both give the correct answer. `seed=42` is set inside `compute_retention_mask` for
within-run reproducibility.

---

## 7. Files

```
scripts/
  runtime_analysis.py          orchestrator — runs all modes, prints comparison table
  compare_sparse_vit_ratio.py  tune ratio, reuse cached dense/EVS baseline
  worker.py                    vLLM inference worker (runs inside Docker)
  run_autogaze_preprocess.py   standalone AutoGaze preprocessing (legacy two-env flow)

autogaze/vllm_integration/
  autogaze_preprocess.py       AutoGazePreprocessor — video → bool patch mask
  patch.py                     apply_autogaze_patch() — replaces vLLM EVS hooks
  retention.py                 AutoGazeContext, autogaze_retained_tokens_count
  sparse_vit.py                SparseViTContext, patch_sparse_vit(), patch_vit_timing()
```

See `PIPELINE.md` for a line-by-line code trace of what happens during inference.
See `VLLM_INTEGRATION.md` for architecture, parameters, and implementation notes.
