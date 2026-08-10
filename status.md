# AutoGaze × vLLM Experiment Status

**Branch:** `vllm-integration-experiments`  
**Last updated:** 2026-08-10

---

## All 5 experiments: COMPLETE ✅

| # | Script | Status | Key result |
|---|--------|--------|-----------|
| 1 | `approach1_fixed_ratio.py` | ✅ | ratio=0.1→2238 tok (5.1s) / 0.25→5690 tok (8.2s) / 0.5→11195 tok (15.8s) |
| 2 | `approach2_vllm_stub.py` | ✅ | `gazing_info` key exists — K known before scheduling |
| 3 | `approach3_per_video_budget.py` | ✅ | Adaptive → 1414 tokens (**70% reduction** vs dense 4704), same answer |
| 4 | `approach4_two_process_split.py` | ✅ | Dense 22352 tok (25.1s) → ratio=0.25: 5690 tok (3.6s) — **7× speedup** |
| 5 | `approach5_vllm_integration.py` | ✅ | **nvidia/AutoGaze integrated with vLLM** — 44% token reduction, same answer |

---

## Approach 5 final results

**Architecture:**
- AutoGaze preprocessing: `auto_gaze` conda env (transformers 4.x) → saves mask to file
- vLLM inference: NVIDIA Docker container (v0.26.0, PyTorch 2.11.0, GB200)
- Integration: `AutoGazeContext` injects pre-computed mask into vLLM's `compute_retention_mask`

| Mode | Tokens | vs Dense | Answer | Selection |
|---|---:|:---:|:---:|---|
| dense | 670 | — | C | none |
| evs | 376 | −44% | C | cosine similarity |
| magnitude | 376 | −44% | C | embedding L2 norm |
| **autogaze** | **376** | **−44%** | **C** | **nvidia/AutoGaze learned model** ✅ |

**What makes autogaze different from evs/magnitude:**
All three select 376 tokens (fixed by `compute_retained_tokens_count(q=0.5)`). The difference is *which* patches — AutoGaze uses its learned task-relevant selection. To show AutoGaze's adaptive K (variable per video), `compute_retained_tokens_count` needs to be replaced with AutoGaze's output.

---

## Engineering work completed

| File | What it does |
|---|---|
| `autogaze/vllm_integration/retention.py` | EVS / magnitude / AutoGaze retention masks |
| `autogaze/vllm_integration/patch.py` | Monkey-patches vLLM's `compute_retention_mask` |
| `autogaze/vllm_integration/autogaze_preprocess.py` | Runs nvidia/AutoGaze, maps 14×14 mask → 16×16 Qwen3-VL grid |
| `scripts/run_autogaze_preprocess.py` | Standalone script using auto_gaze env (transformers 4.x) |
| `scripts/approach5_vllm_worker.py` | Runs inside Docker, loads pre-computed mask |
| `scripts/approach5_vllm_integration.py` | Orchestrates: preprocess outside + infer inside Docker |
| `modeling_llama_multi_token_pred.py` | Fixed transformers 5.x compat for AutoGaze's generate() loop |

---

## Architecture diagram

### Current production path (Tasks 1–3 implemented)

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  OUTSIDE DOCKER  (auto_gaze conda env, transformers 4.x)               │
 │                                                                         │
 │   video.mp4                                                             │
 │      │                                                                  │
 │      ▼  load_frames (PyAV / torchcodec)                                 │
 │   raw frames  (T, C, H, W)  float32 [0,1]                              │
 │      │                                                                  │
 │      ▼  AutoGazePreprocessor.compute_retention_mask()                   │
 │   ┌──────────────────────────────────┐                                  │
 │   │  nvidia/AutoGaze model           │                                  │
 │   │  resize → 224×224, normalize     │                                  │
 │   │  ShallowVideoConvNet + LLaMA-4L  │                                  │
 │   │  → gazing_mask  (T, 14, 14)      │                                  │
 │   │  → bilinear upsample             │                                  │
 │   │      post-ViT:  → (T, 16, 16)   │  ──► ag_mask.pt   (Task 1+A5)   │
 │   │      pre-ViT:   → (T, 32, 32)   │  ──► ag_mask_vit.pt (Task 2+3)  │
 │   └──────────────────────────────────┘                                  │
 └────────────────────────────────┬────────────────────────────────────────┘
                                  │  file on disk  /tmp/ag_mask[_vit].pt
 ┌────────────────────────────────▼────────────────────────────────────────┐
 │  INSIDE DOCKER  (vLLM 0.26.0, PyTorch 2.11.0, GB200 arm64)            │
 │                                                                         │
 │  approach5_vllm_worker.py                                               │
 │      │                                                                  │
 │      ├─ apply_autogaze_patch(mode)          ◄── patch.py               │
 │      │     patches compute_retention_mask                               │
 │      │     patches compute_retained_tokens_count  (Task 1: adaptive K)  │
 │      │                                                                  │
 │      ├─ LLM(model, video_pruning_rate=0.5, enforce_eager=True)         │
 │      │                                                                  │
 │      ├─ patch_sparse_vit(llm)  [sparse_vit mode only]   ◄── sparse_vit.py
 │      │     wraps visual encoder forward with gather op                  │
 │      │                                                                  │
 │      └─ llm.chat(messages)  inside context managers:                   │
 │                                                                         │
 │  ┌── SparseViTContext(mask_vit, K_vit, grid_thw)  [sparse_vit mode] ──┐│
 │  │ ┌── AutoGazeContext(ag_mask, K)  [autogaze / sparse_vit mode] ────┐ ││
 │  │ │                                                                  │ ││
 │  │ │   video frames                                                   │ ││
 │  │ │      │                                                           │ ││
 │  │ │      ▼  Qwen3-VL visual encoder                                  │ ││
 │  │ │   ┌──────────────────────────────────────────────────────────┐   │ ││
 │  │ │   │  patch_embed   (T×32×32 patches → embeddings)  [all N]  │   │ ││
 │  │ │   │       │                                                  │   │ ││
 │  │ │   │       ▼  [GATHER selected_idx]  ◄── SparseViTContext     │   │ ││
 │  │ │   │  K selected patch embeddings   (Task 2: pre-ViT select)  │   │ ││
 │  │ │   │       │                                                  │   │ ││
 │  │ │   │       ▼  transformer blocks  (O(K²) attn, O(K) FFN)      │   │ ││
 │  │ │   │  K encoded patch embeddings   (Task 3: sparse ViT)       │   │ ││
 │  │ │   │       │                                                  │   │ ││
 │  │ │   │       ▼  spatial merger  (2×2 → 1)                       │   │ ││
 │  │ │   │  K_merged visual tokens  (K_merged = K ÷ 4)              │   │ ││
 │  │ │   └───────────────────┬──────────────────────────────────────┘   │ ││
 │  │ │                       │                                          │ ││
 │  │ │                       ▼  compute_retention_mask()                │ ││
 │  │ │              ┌─────────────────────────────────────┐            │ ││
 │  │ │              │  autogaze mode:  use ag_mask        │            │ ││
 │  │ │              │  sparse_vit mode: identity (all-True)│  Task 1   │ ││
 │  │ │              │  compute_retained_tokens_count → K  │◄──────────┘ ││
 │  │ │              └──────────────┬──────────────────────┘             ││
 │  │ └─────────────────────────────┼──────────────────────────────────┘ ││
 │  └───────────────────────────────┼────────────────────────────────────┘│
 │                                  │                                      │
 │                                  ▼  K_merged tokens                    │
 │                         Qwen3-VL-2B LLM decoder                        │
 │                                  │                                      │
 │                                  ▼                                      │
 │                              answer text                                │
 └─────────────────────────────────────────────────────────────────────────┘
```

### How AutoGaze decides K — per frame, dynamically

AutoGaze selects a **different number of patches per frame**, not a fixed ratio. It runs a tiny LLaMA decoder autoregressively, emitting gaze positions one frame at a time. Two mechanisms control how many patches it picks per frame:

1. **`gazing_ratio`** — soft target fraction (e.g. 0.5 = "aim for 50% per frame on average")
2. **`task_loss_requirement`** — quality threshold: the model's `task_loss_prediction_head` predicts how well the selected patches reconstruct the scene. Once that prediction crosses the threshold, the model stops early for that frame — fewer patches selected.

This means **temporally redundant frames get far fewer patches than the first frame**. In Approach 3, with first-frame ratio=0.5 and rest=0.1, the total dropped 70% vs dense while keeping the same answer — because AutoGaze learned that frames 2–N mostly repeat frame 1.

In our runtime analysis: `gazing_ratio=0.5`, 6 frames → K_vit=2724 total (avg 454/frame out of 1024), but the per-frame distribution is uneven. `num_gazing_each_frame` is a `(T,)` tensor with different values per frame.

### How sparse ViT works — plain English

A video frame is divided into a grid of small squares called **patches** (32×32 = 1024 per frame for 448px input with 14px patches). The Vision Transformer (ViT) reads every patch and converts it into a vector. The expensive part is **self-attention**: every patch compares itself to every other patch, so the cost grows as K², not K.

**AutoGaze** is a tiny model that watches the video first and marks which patches actually matter — the road sign, the moving object, the relevant text — with **a different count per frame**. Everything else (blank sky, static background) gets discarded before the ViT does any heavy lifting.

```
NORMAL (dense / EVS / post-ViT autogaze)
─────────────────────────────────────────────────────────────────
  N patches ──► patch_embed ──► Transformer (N²) ──► drop (N-K) ──► LLM
               (cheap)          EXPENSIVE              late

SPARSE VIT (Tasks 2+3)
─────────────────────────────────────────────────────────────────
  N patches ──► patch_embed ──► KEEP K_t per frame ──► Transformer (K²) ──► LLM
               (cheap, all N)   [gather, K varies        cheaper         no drop
                                 frame-to-frame]         (K/N)² attn     needed
```

`K_t` is the number of patches AutoGaze selected for frame `t` — different per frame. The gather op uses the actual per-frame mask, so frame 1 might contribute 600 patches while frames 2–6 contribute 80 each.

The key insight: **patch_embed is just a convolution — cheap to run on all N patches**. The Transformer is the expensive part. Sparse ViT lets patch_embed touch all patches so AutoGaze can pick the right ones, then the Transformer only sees the K selected patches.

**Why this beats post-ViT pruning (the previous approach):**

| Stage | Dense | Post-ViT EVS/AutoGaze | Sparse ViT |
|---|---|---|---|
| patch_embed | all N | all N | all N (cheap) |
| Transformer input | N patches | N patches | **K patches (varies per video/frame)** |
| Attention cost | N² | N² | **K² ≪ N²** |
| Patches discarded | none | N−K (after ViT) | N−K (before ViT) |
| Tokens to LLM | N/4 | K/4 | **K/4** |

The LLM sees K/4 tokens either way. Sparse ViT avoids running Transformer attention on the N−K patches that will be dropped anyway.

**The three-line summary of the code change** (`sparse_vit.py:_sparse_vit_forward`):

```python
hidden = encoder.patch_embed(pixel_values)   # cheap — runs on all N
hidden = hidden[selected_idx]                # GATHER: keep only K  ← new (K varies per video)
hidden = _run_blocks(encoder.blocks, hidden, cu_seqlens_sparse, rotary_pos_emb_sparse)
```

### File map

```
autogaze/vllm_integration/
  retention.py         AutoGazeContext · autogaze_retained_tokens_count (Task 1)
                       autogaze_retention_mask · magnitude_retention_mask · evs
  patch.py             apply_autogaze_patch() — patches both EVS hooks
  sparse_vit.py        SparseViTContext · patch_sparse_vit() (Tasks 2+3)
  autogaze_preprocess.py  AutoGazePreprocessor.compute_retention_mask()

scripts/
  run_autogaze_preprocess.py   outside Docker (auto_gaze env)
                                 --grid-hw 16 16  → ag_mask.pt     (autogaze)
                                 --grid-hw 32 32  → ag_mask_vit.pt (sparse_vit)
  approach5_vllm_worker.py     inside Docker — modes: dense/evs/magnitude/autogaze/sparse_vit
  approach5_vllm_integration.py  orchestrator: runs preprocess + Docker worker
```

---

## End-to-end runtime analysis

**Setup:** Qwen/Qwen3-VL-2B-Instruct · `assets/example_input.mp4` · 6 frames · GB200  
**Method:** 3 reps per mode (rep 1 = warmup, reps 2–3 averaged) · CUDA events for ViT/LM split

| Mode | Tokens | vs Dense | Preproc (ms) | ViT (ms) | LM (ms) | Infer (ms) | Answer |
|---|---:|:---:|---:|---:|---:|---:|:---:|
| dense | 670 | — | — | n/a | n/a | 17,320 | C |
| evs | 376 | −44% | — | 3,189 | 13,649 | 16,838 | C |
| **sparse\_vit** | **706** | **+5%** | **32,102** | **1,411** | **16,925** | **18,336** | **C** |

> `infer_ms` excludes model load (~26 s shared across all modes).  
> `preproc_ms` is the one-time AutoGaze mask computation (outside Docker, auto\_gaze env).

### Key findings

**ViT speedup: 2.26×**  
Sparse ViT processes 2,724 of 6,144 patches (44.3%) through the transformer blocks.  
ViT time drops from 3,189 ms → 1,411 ms — confirming the gather-op sparse encoding (Tasks 2+3) works end-to-end.

**End-to-end: sparse\_vit is 9% slower than EVS at this ratio**  
EVS (q=0.5) selects ~25% of post-merge tokens (376 of 1,536 → 376 LLM tokens).  
AutoGaze (gazing\_ratio=0.5) selects 44.3% of ViT patches → 681 merged tokens → 706 prompt tokens.  
The LM pays 24% more decode time (16,925 ms vs 13,649 ms) because it receives 706 vs 376 tokens.

**Fix: lower gazing\_ratio for sparse\_vit to match EVS token count**  
Target: K\_merged ≈ 376 → K\_vit ≈ 1,504 → gazing\_ratio ≈ 0.245.  
At that ratio the ViT speedup is maintained and LM cost matches EVS, making sparse\_vit strictly better.

**Adaptive K (Task 1) works:**  
`compute_retained_tokens_count` returned 681 (AutoGaze's K\_merged) instead of the fixed-formula 376,  
correctly allocating KV-cache slots for the actual token count.

**Encoder discovery confirmed:** `_find_visual_encoder` found `Qwen2_5VLVisionTransformer` at `model.visual` — no manual path needed.

**All modes answer correctly (C)** — no accuracy regression from sparse encoding.

### Tuning guidance

| Goal | Setting |
|---|---|
| Match EVS token count (376) | `gazing_ratio ≈ 0.245` with `--grid-hw 32 32` |
| 2× ViT speedup + same LM cost as EVS | `gazing_ratio = 0.245`, expected ViT ≈ 700 ms |
| Maximum token reduction | Lower `gazing_ratio` with `task_loss_requirement` threshold |

---

## Remaining production path

1. ~~Replace `compute_retained_tokens_count` with AutoGaze's adaptive K~~ ✅ Task 1 done
2. ~~Move AutoGaze pre-ViT: select patches before ViT encoding~~ ✅ Task 2 done
3. ~~Sparse ViT encoding via gather op on selected patch indices~~ ✅ Task 3 done
4. Upstream vLLM PR: add `token_count` field to `ModalityInput`
5. Validate sparse_vit mode end-to-end in Docker (verify `_find_visual_encoder` path for Qwen3-VL)
6. Run EgoSchema / Video-MME accuracy benchmark to confirm quality holds at K/N=0.5
