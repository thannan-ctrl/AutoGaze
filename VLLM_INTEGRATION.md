# AutoGaze × vLLM Integration

Integrates `nvidia/AutoGaze` with vLLM to run the Qwen3-VL visual encoder sparsely —
selecting patches **before** the transformer blocks rather than pruning after.

## Architecture

```
outside Docker  (auto_gaze conda env)
  video frames
    → AutoGaze (tiny LLaMA decoder)
    → per-frame patch mask  (T × H_vit × W_vit)  bool
    → saved to /tmp/ag_mask_vit.pt

inside Docker  (nvcr.io/nvidia/vllm:26.07-py3)
  patch_embed(all N patches)    ← cheap conv, runs on everything
    → GATHER K selected patches  ← AutoGaze mask
    → transformer blocks(K)      ← O(K²) attention, not O(N²)
    → spatial merger             ← K → K/4 merged tokens
    → LLM                        ← K/4 visual tokens
```

**ViT speedup:** proportional to `(K/N)²` for attention. At 21.6% retention: **4.6× faster ViT**.  
**End-to-end:** 15% faster than EVS at matched token count (ratio=0.245 vs EVS q=0.5).

## Quick start

```bash
# 1. Compute AutoGaze mask (auto_gaze conda env, outside Docker)
/path/to/auto_gaze/python scripts/run_autogaze_preprocess.py \
    --video assets/example_input.mp4 \
    --output /tmp/ag_mask_vit.pt \
    --grid-hw 32 32 \
    --gazing-ratio 0.245

# 2. Run benchmark (launches Docker automatically)
python scripts/runtime_analysis.py --modes dense evs sparse_vit --reps 3

# 3. Tune gazing ratio
python scripts/compare_sparse_vit_ratio.py --gazing-ratio 0.245 --reps 3
```

## Files

```
autogaze/vllm_integration/
  autogaze_preprocess.py   AutoGazePreprocessor — runs AutoGaze, outputs (T×H×W) bool mask
  patch.py                 apply_autogaze_patch() — replaces vLLM's EVS hooks
  retention.py             AutoGazeContext, autogaze_retained_tokens_count (adaptive K)
  sparse_vit.py            SparseViTContext, patch_sparse_vit() — gather op before ViT blocks

scripts/
  run_autogaze_preprocess.py   run AutoGaze preprocessing (auto_gaze env, outside Docker)
  worker.py                    vLLM inference worker (runs inside Docker)
  runtime_analysis.py          benchmark: dense vs EVS vs sparse_vit
  compare_sparse_vit_ratio.py  sweep gazing_ratio, compare against cached EVS baseline
```

## Benchmark results

Model: `Qwen/Qwen3-VL-2B-Instruct` · Video: 6 frames · GB200 · reps=3 (warmup excluded)

| Mode | Tokens | ViT (ms) | LM (ms) | Infer (ms) |
|---|---:|---:|---:|---:|
| dense | 670 | n/a | n/a | 17,320 |
| evs (q=0.5) | 376 | 3,189 | 13,649 | 16,838 |
| **sparse_vit (ratio=0.245)** | **356** | **690** | **13,526** | **14,216** |

sparse_vit at ratio=0.245 beats EVS: **4.6× faster ViT**, same LM cost, **15% faster overall**.

## Key parameters

| Parameter | Where | What it controls |
|---|---|---|
| `gazing_ratio` | `run_autogaze_preprocess.py` | Fraction of patches AutoGaze targets per frame. Actual K may be lower due to early stopping. |
| `--grid-hw H W` | `run_autogaze_preprocess.py` | ViT patch grid size. Use `32 32` for sparse_vit (pre-merge), `16 16` for post-ViT autogaze. |
| `seed` | `compute_retention_mask()` | Random seed for reproducible K (default 42). AutoGaze generation is stochastic. |
| `--reps N` | `worker.py` | Inference repetitions. Rep 1 = warmup; reps 2+ are measured. |

## Implementation notes

**Adaptive K (Task 1):** `autogaze_retained_tokens_count` is monkey-patched over vLLM's
`compute_retained_tokens_count` so vLLM pre-allocates exactly K KV-cache slots per video,
not a fixed formula. This requires `AutoGazeContext(K=K_merged)` to be active.

**Class-level patch (vLLM ≥0.24 V1 engine):** The visual encoder runs in an `EngineCore`
subprocess. `patch_sparse_vit(llm=None)` patches the class before `LLM()` is called so the
patch is inherited by the subprocess. Must run after `from vllm import LLM`.

**Two ViT grid sizes:**
- `--grid-hw 32 32` (pre-merge): mask at ViT patch level; used by sparse_vit. K_merged = K_vit / 4.
- `--grid-hw 16 16` (post-merge): mask at merged-token level; used by post-ViT autogaze mode.

**AutoGaze stochasticity:** The model's `task_loss_prediction_head` stops patch selection
early when confident. K can vary across runs even at the same ratio. Pin `seed=42` for
reproducibility or use `task_loss_requirement` for a quality-floor stopping criterion.
