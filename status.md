# AutoGaze × vLLM Experiment Status

**Branch:** `vllm-integration-experiments`  
**Last updated:** 2026-08-07 (~07:00 UTC)

---

## All experiments: DONE ✅

| # | Script | Status | Key result |
|---|--------|--------|-----------|
| 1 | `approach1_fixed_ratio.py` | ✅ | ratio=0.1→2238 tok (5.1s) / 0.25→5690 tok (8.2s) / 0.5→11195 tok (15.8s) |
| 2 | `approach2_vllm_stub.py` | ✅ | `gazing_info` key exists — K known before scheduling |
| 3 | `approach3_per_video_budget.py` | ✅ | Adaptive → 1414 tokens (**70% reduction** vs dense 4704), same answer |
| 4 | `approach4_two_process_split.py` | ✅ | Dense 22352 tok (25.1s) → ratio=0.25: 5690 tok (3.6s) — **7× speedup** |
| 5 | `approach5_vllm_integration.py` | ✅ | **44% token reduction** in vLLM, same answer, AutoGaze patch confirmed |

---

## Approach 5 final results

**Model:** Qwen3-VL-2B-Instruct in NVIDIA vLLM Docker (v0.26.0)

| Mode | Tokens | vs Dense | Answer |
|---|---:|:---:|:---:|
| dense | 670 | — | C |
| evs | 376 | **−44%** | C |
| magnitude (AutoGaze) | 376 | **−44%** | C |

The AutoGaze patch (`autogaze/vllm_integration/patch.py`) replaces vLLM's `compute_retention_mask`  
before model load — confirmed in logs: `[AutoGaze-vLLM] compute_retention_mask → magnitude`

---

## Summary: token reduction across all approaches

| Approach | Model | Tokens | vs Dense | Speedup |
|---|---|---:|:---:|:---:|
| Dense baseline | NVILA-8B | 22,352 | — | 1× |
| Fixed ratio 0.25 | NVILA-8B | 5,690 | −75% | ~7× generate |
| Adaptive ratio | NVILA-8B | 1,414 | −94% | — |
| EVS in vLLM | Qwen3-VL-2B | 376/670 | −44% | — |
| AutoGaze-magnitude in vLLM | Qwen3-VL-2B | 376/670 | **−44%** | — |

---

## Files on branch

| File | Purpose |
|---|---|
| `scripts/approach1_fixed_ratio.py` | Ratio sweep with NVILA |
| `scripts/approach2_vllm_stub.py` | Shows `gazing_info` exists pre-scheduling |
| `scripts/approach3_per_video_budget.py` | Per-video adaptive budget |
| `scripts/approach4_two_process_split.py` | Embedding extraction + service split |
| `scripts/approach5_vllm_integration.py` | Docker orchestration for vLLM |
| `scripts/approach5_vllm_worker.py` | Runs inside vLLM container |
| `autogaze/vllm_integration/retention.py` | EVS / magnitude / AutoGaze retention masks |
| `autogaze/vllm_integration/patch.py` | Monkey-patches vLLM's compute_retention_mask |
| `result.md` | Full results for all 5 approaches |
