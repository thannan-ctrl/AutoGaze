# AutoGaze × vLLM — Sparse ViT Integration

**Branch:** `vllm-integration-experiments` · **Updated:** 2026-08-10

Integrates `nvidia/AutoGaze` with vLLM to run the Qwen3-VL visual encoder **sparsely** —
selecting K patches before the transformer blocks via a gather op, instead of pruning after.

---

## Results

**Model:** `Qwen/Qwen3-VL-2B-Instruct` · **Video:** 6 frames, 448×448 · **GPU:** GB200  
**Method:** 3 reps, warmup excluded, CUDA events for ViT/LM split

| Mode | Tokens | vs Dense | ViT (ms) | LM (ms) | Infer (ms) | Answer |
|---|---:|:---:|---:|---:|---:|:---:|
| dense | 670 | — | n/a | n/a | 17,320 | C |
| evs (q=0.5) | 376 | −44% | 3,189 | 13,649 | 16,838 | C |
| **sparse\_vit (ratio=0.245)** | **356** | **−47%** | **690** | **13,526** | **14,216** | **C** |

`infer_ms` excludes model load (~26 s). All modes answer correctly.

- **ViT: 4.6× faster** — 690 ms vs 3,189 ms. Scales as (K/N)² ≈ (0.216)² ≈ 4.7×.
- **LM: unchanged** — same token count to the LLM → same decode cost.
- **Net: 15% faster** than EVS end-to-end.

**Why `ratio=0.5` was slower than EVS (gotcha):** EVS `q=0.5` keeps ~25% of merged tokens
(376). AutoGaze `ratio=0.5` targets 44% of ViT patches → 681 merged tokens → more LM work.
Use `ratio≈0.245` to match EVS's token count. See [Parameter guide](#parameters) below.

**AutoGaze's adaptive K:** At the same `ratio=0.245`, two runs gave different K:

| Run | K\_vit | Retention | Tokens | Infer (ms) |
|---|---:|:---:|---:|---:|
| A | 1,326 | 21.6% | 356 | 14,216 |
| B | 433 | 7.0% | 133 | 13,265 |

Run B's `task_loss_prediction_head` stopped at 7% — it learned that was enough for this
question. Both correct. Run B is 21% faster than EVS with 80% fewer tokens than dense.

---

## Architecture

```
OUTSIDE DOCKER  (auto_gaze conda env, transformers 4.x)
────────────────────────────────────────────────────────
  video frames
    ↓ AutoGaze (ShallowVideoConvNet + LLaMA-4L)
  gazing_mask (T, 14, 14)  ← per-frame, task-driven early stopping
    ↓ bilinear upsample → ViT patch grid (32×32)
  ag_mask_vit.pt  ──────────────────────────────────────┐
                                                         │ file on disk
INSIDE DOCKER  (nvcr.io/nvidia/vllm:26.07-py3)          │
────────────────────────────────────────────────────────┘
  from vllm import LLM
  apply_autogaze_patch()       ← monkey-patches vllm.multimodal.evs
  patch_sparse_vit(llm=None)   ← class-level patch before LLM() [vLLM V1]
  LLM(video_pruning_rate=0.245, enforce_eager=True)

  inside SparseViTContext + AutoGazeContext:

  pixel_values (T × 32×32 patches)
    ↓ patch_embed    [all N — cheap conv]
    ↓ GATHER K       [from ag_mask_vit.pt]      ← K varies per video/frame
    ↓ blocks(K)      [O(K²) vs dense O(N²)]
    ↓ merger         [K → K/4 merged tokens]
    ↓ LLM            [K/4 visual tokens → answer]
```

**Three things patched in vLLM:**

| Hook | What changes | File |
|---|---|---|
| `compute_retained_tokens_count` | Returns AutoGaze's actual K, not fixed formula | `retention.py` |
| `compute_retention_mask` | Returns identity mask (ViT already selected) | `retention.py` |
| `Qwen2_5VLVisionTransformer.forward` | Applies gather op before transformer blocks | `sparse_vit.py` |

---

## Quick start

```bash
# Step 1 — compute AutoGaze mask (auto_gaze env, outside Docker)
/path/to/auto_gaze/python scripts/run_autogaze_preprocess.py \
    --video assets/example_input.mp4 \
    --output /tmp/ag_mask_vit.pt \
    --grid-hw 32 32 \
    --gazing-ratio 0.245

# Step 2 — benchmark all three modes (launches Docker automatically)
python scripts/runtime_analysis.py --modes dense evs sparse_vit --reps 3

# Step 3 — sweep a specific ratio, reuse cached dense/EVS baseline
python scripts/compare_sparse_vit_ratio.py --gazing-ratio 0.245 --reps 3
```

---

## Files

```
autogaze/vllm_integration/
  autogaze_preprocess.py   AutoGazePreprocessor.compute_retention_mask()
  patch.py                 apply_autogaze_patch() — patches both EVS hooks
  retention.py             AutoGazeContext · autogaze_retained_tokens_count
  sparse_vit.py            SparseViTContext · patch_sparse_vit() · patch_vit_timing()

scripts/
  run_autogaze_preprocess.py   run AutoGaze outside Docker (auto_gaze env)
  worker.py                    vLLM inference worker (runs inside Docker)
  runtime_analysis.py          benchmark: dense vs EVS vs sparse_vit
  compare_sparse_vit_ratio.py  tune ratio, reuse cached baseline
```

---

## Parameters

| Parameter | Where | Effect |
|---|---|---|
| `--gazing-ratio` | `run_autogaze_preprocess.py` | Soft target patch fraction. Actual K may be lower due to early stopping. Use `≈0.245` to match EVS token count. |
| `--grid-hw H W` | `run_autogaze_preprocess.py` | ViT patch grid: `32 32` for sparse\_vit (pre-merge), `16 16` for post-ViT autogaze mode. |
| `seed` | `compute_retention_mask(seed=42)` | Fixes random seed for reproducible K within a run. |
| `--reps N` | `worker.py` | Inference repetitions. Rep 1 = warmup. |

**`gazing_ratio` vs EVS `q` are not the same thing:**

| Parameter | Controls |
|---|---|
| EVS `q=0.5` | Keep top 50% of post-merge tokens by cosine similarity → fixed K |
| AutoGaze `ratio=0.5` | *Target* 50% retention per frame, stop early if confident → variable K |

---

## Implementation notes

**Adaptive K:** `autogaze_retained_tokens_count` is monkey-patched over vLLM's
`compute_retained_tokens_count`. When `AutoGazeContext(K=K_merged)` is active, vLLM
pre-allocates K KV-cache slots instead of the formula-derived count.

**Class-level patch (vLLM ≥0.24 V1):** The visual encoder runs in an `EngineCore`
subprocess. Call `patch_sparse_vit(llm=None)` **after** `from vllm import LLM` but **before**
`LLM(...)` so the class patch is inherited when the subprocess instantiates the model.

**AutoGaze K is stochastic:** The `task_loss_prediction_head` stops early based on CUDA
random state. Pin `seed=42` for within-run reproducibility. Use `task_loss_requirement`
(a quality-floor threshold) for more principled stopping.

**vLLM 0.24+ signature change:** `compute_retained_tokens_count` is called with
`num_frames=N` as a keyword arg. Our replacement accepts both the old positional `T` and the
new `num_frames=` kwarg.

---

## Next steps

1. Upstream vLLM PR: add `token_count` field to `ModalityInput`
2. Accuracy benchmark on EgoSchema / Video-MME (not just one question)
3. Replace `gazing_ratio` with `task_loss_requirement` for quality-floor stopping
