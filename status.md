# AutoGaze × vLLM — Sparse ViT Integration

**Branch:** `vllm-integration-experiments`  
**Last updated:** 2026-08-10

---

## What was built

Integrated `nvidia/AutoGaze` with vLLM to run the Qwen3-VL visual encoder **sparsely** —
selecting K patches before the transformer blocks via a gather op, instead of pruning the
full N-patch output after the ViT.

Three tasks implemented:

| Task | What it does | File |
|---|---|---|
| 1 — Adaptive K | `compute_retained_tokens_count` → AutoGaze's actual K per video (not fixed formula) | `retention.py`, `patch.py` |
| 2 — Pre-ViT selection | AutoGaze mask applied before transformer blocks | `sparse_vit.py` |
| 3 — Sparse ViT encoding | Transformer runs on K patches only (O(K²) attention) | `sparse_vit.py` |

---

## Benchmark results

**Model:** `Qwen/Qwen3-VL-2B-Instruct` · **Video:** `assets/example_input.mp4` (6 frames) · **GPU:** GB200  
**Method:** 3 reps per mode, warmup excluded, CUDA events for ViT/LM split

| Mode | Tokens | ViT (ms) | LM (ms) | Infer (ms) | Answer |
|---|---:|---:|---:|---:|:---:|
| dense | 670 | n/a | n/a | 17,320 | C |
| evs (q=0.5) | 376 | 3,189 | 13,649 | 16,838 | C |
| **sparse\_vit (ratio=0.245)** | **356** | **690** | **13,526** | **14,216** | **C** |

`infer_ms` excludes model load (~26 s). `gazing_ratio=0.245` targets ~21.6% patch retention
to match EVS's token count (~376 LLM tokens).

**Key numbers:**
- ViT: **4.6× faster** (3,189 ms → 690 ms) — scales as (K/N)² for attention
- LM: unchanged (same token count to LLM)
- End-to-end: **15% faster** than EVS (14,216 ms vs 16,838 ms)
- Accuracy: same answer (C) across all modes

---

## Architecture

```
 OUTSIDE DOCKER  (auto_gaze conda env, transformers 4.x)
 ───────────────────────────────────────────────────────
   video frames
     ↓ AutoGaze (ShallowVideoConvNet + LLaMA-4L)
   gazing_mask (T, 14, 14)  ← different K per frame, task-driven early stopping
     ↓ bilinear upsample to ViT patch grid (32×32)
   ag_mask_vit.pt  ─────────────────────────────────────┐
                                                         │ file on disk
 INSIDE DOCKER  (nvcr.io/nvidia/vllm:26.07-py3)         │
 ───────────────────────────────────────────────────────┘
   apply_autogaze_patch()         ← patches EVS hooks in vllm.multimodal.evs
   patch_sparse_vit(llm=None)     ← class-level patch before LLM() (vLLM V1)
   LLM(video_pruning_rate=0.245)

   llm.chat() inside SparseViTContext + AutoGazeContext:

   pixel_values (T×32×32 patches)
     ↓ patch_embed    [all N patches — cheap conv]
     ↓ GATHER K       [selected_idx from ag_mask_vit.pt]   ← Task 2
     ↓ blocks(K)      [O(K²) attention, not O(N²)]         ← Task 3
     ↓ merger         [K → K/4 merged tokens]
     ↓ LLM            [K/4 visual tokens]
     ↓ answer
```

---

## How AutoGaze selects K

AutoGaze generates gaze positions **autoregressively**, frame by frame. Its
`task_loss_prediction_head` stops early for each frame when it predicts the selected patches
are sufficient. This means:

- **K varies per frame** — static frames get far fewer patches than the first/most-informative frame
- **K varies per video** — harder videos use more patches
- `gazing_ratio` is a soft target, not a hard per-frame quota
- `seed=42` makes K reproducible within a run; across runs the CUDA random state can differ

At `ratio=0.245`, AutoGaze selected 1,326 of 6,144 patches (21.6%) in one run and
433 of 6,144 (7.0%) in another — both gave the correct answer.

---

## Files

```
autogaze/vllm_integration/
  autogaze_preprocess.py   AutoGazePreprocessor — runs AutoGaze, maps mask to ViT grid
  patch.py                 apply_autogaze_patch() — patches both EVS hooks
  retention.py             AutoGazeContext · autogaze_retained_tokens_count (Task 1)
  sparse_vit.py            SparseViTContext · patch_sparse_vit() · patch_vit_timing() (Tasks 2+3)

scripts/
  run_autogaze_preprocess.py   run AutoGaze outside Docker (auto_gaze env)
  worker.py                    vLLM inference worker (runs inside Docker)
  runtime_analysis.py          benchmark: dense vs EVS vs sparse_vit
  compare_sparse_vit_ratio.py  tune gazing_ratio, reuse cached dense/EVS baseline
```

---

## Next steps

1. Upstream vLLM PR: add `token_count` field to `ModalityInput`
2. Run accuracy benchmark on EgoSchema / Video-MME (not just single-question)
3. Tune `task_loss_requirement` as a quality-floor stopping criterion instead of fixed ratio
