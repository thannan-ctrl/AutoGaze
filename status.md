# AutoGaze × vLLM Experiment Status

**Branch:** `vllm-integration-experiments`  
**Last updated:** 2026-08-07 (~16:30 UTC)

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

## Remaining production path

1. Replace `compute_retained_tokens_count` with AutoGaze's adaptive K (per-video budget)
2. Move AutoGaze pre-ViT: select patches before ViT encoding → save ViT compute
3. Sparse ViT encoding via gather op on selected patch indices
4. Upstream vLLM PR: add `token_count` field to `ModalityInput`
