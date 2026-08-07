# AutoGaze × vLLM Experiment Status

**Branch:** `vllm-integration-experiments`  
**Goal:** Adapt AutoGaze with vLLM for maximum efficiency gain in video processing.  
**Last updated:** 2026-08-07 (~04:50 UTC)

---

## Experiment status

| # | Script | Status | Key result |
|---|--------|--------|-----------|
| 1 | `approach1_fixed_ratio.py` | ✅ Done | ratio=0.1→2238 tokens (5.1s) / ratio=0.25→5690 tokens (8.2s) / ratio=0.5→11195 tokens (15.8s). OOM at ≥0.75 |
| 2 | `approach2_vllm_stub.py` | ✅ Done | `gazing_info` key already exists in processor output — K computable before scheduling |
| 3 | `approach3_per_video_budget.py` | ✅ Done | Dense=4704 tokens → adaptive allocation → 1414 tokens (70% reduction) same answer |
| 4 | `approach4_two_process_split.py` | 🔄 Running | Extracting visual embeddings at ratios {0.25, 0.5, 1.0} |

---

## Important: these are NOT vLLM runs

All experiments use **HuggingFace Transformers** to measure:
- How many tokens AutoGaze selects at each ratio
- Generate latency as a function of token count
- Whether reduced tokens maintain accuracy

**No vLLM server has been started.** vLLM integration doesn't exist yet — these experiments provide the data to justify it. The path to actual vLLM integration is documented in `VLLM_INTEGRATION.md` and summarized in `result.md`.

---

## Key findings so far

1. **70% token reduction** is achievable with adaptive ratio while keeping the same answer
2. **`gazing_info` already exists** in the processor output — the main engineering work for vLLM is plumbing K into `ModalityInput`
3. **Generate time scales ~linearly** with token count (1.2s → 3.6s → 9.5s for 2238→5690→11195 tokens)
4. **OOM boundary** at ratio≥0.75 with 128-frame 448px video on single GPU (max ~11K tokens per GPU with current config)

---

## OOM root cause

The SigLIP ViT processes ALL tiles (128 frames × 1 spatial tile) in batches of 32. Each batch = 32 tiles × 16 frames × 784 patches. At ratio≥0.75, the LLM attention forward pass with ~16K+ tokens fills GPU 0's 184 GB VRAM (model weights ~32 GB + peak activations ~152 GB+).

Fix: reduce `max_batch_size_siglip` or use tensor parallelism across GPUs.

---

## Next steps (after experiments complete)

1. ✅ Commit all 4 scripts + result.md + status.md to branch
2. Read `autogaze/` source to inspect `gazing_info` schema
3. Prototype `AutoGazeMultiModalProcessor` returning `(patch_indices, K)` for vLLM
4. Open vLLM RFC or PR with the token_count field in `ModalityInput`
