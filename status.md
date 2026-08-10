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

### Mode comparison

```
Mode          ViT input    ViT attn       Post-ViT        LLM tokens   Task
─────────────────────────────────────────────────────────────────────────────
dense         N patches    O(N²)          none            N/4          —
evs           N patches    O(N²)          cosine sim→K    K/4          —
magnitude     N patches    O(N²)          L2 norm→K       K/4          —
autogaze      N patches    O(N²)          ag_mask→K       K/4          1
sparse_vit    K patches    O(K²)≈N²/4    identity        K/4          1+2+3
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

## Remaining production path

1. ~~Replace `compute_retained_tokens_count` with AutoGaze's adaptive K~~ ✅ Task 1 done
2. ~~Move AutoGaze pre-ViT: select patches before ViT encoding~~ ✅ Task 2 done
3. ~~Sparse ViT encoding via gather op on selected patch indices~~ ✅ Task 3 done
4. Upstream vLLM PR: add `token_count` field to `ModalityInput`
5. Validate sparse_vit mode end-to-end in Docker (verify `_find_visual_encoder` path for Qwen3-VL)
6. Run EgoSchema / Video-MME accuracy benchmark to confirm quality holds at K/N=0.5
