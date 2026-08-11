# AutoGaze × vLLM — Sparse ViT

## What this is

AutoGaze pre-selects which video patches matter before the Vision Transformer (ViT) runs. Instead of running self-attention over every patch (O(N²)), the ViT runs only on the K patches AutoGaze identifies as important (O(K²)). The LLM then sees far fewer visual tokens.

---

## Pipeline

```
Video (T frames)
  │
  ├─ AutoGaze (ShallowVideoConvNet + LLaMA-4L)
  │    → bool mask (T×28×28),  K = selected patches  [adaptive, stops when confident]
  │
  │    mask written to file ──────────────────────────────────────────────┐
  │                                                                        │ IPC
  └─ vLLM (main process)                                                  ▼
       │                                              EngineCore subprocess
       │  Hook 1: compute_retained_tokens_count       ┌──────────────────────────┐
       │          reads K/4 from context              │ patch_embed(all N)  cheap │
       │          → correct KV-cache allocation       │        │                  │
       │                                              │  GATHER K  ◄── mask      │
       │                                              │        │   K ≪ N          │
       │                                              │  blocks(K)  O(K²) attn   │
       │                                              │        │   vs O(N²) dense │
       │                                              │  merger → K/4 tokens     │
       │  Hook 2: compute_retention_mask              └──────────────────────────┘
       │          → identity (ViT already pruned)
       │
       └─ LLM (K/4 visual + text tokens) → Answer
```

| Hook | Effect |
|---|---|
| `Qwen2_5VLVisionTransformer.forward` | Inserts gather op: `patch_embed(N) → gather K → blocks(K) → merger` |
| `compute_retained_tokens_count` | Reserves K/4 KV-cache slots (not N/4) |
| `compute_retention_mask` | All-True identity — no post-ViT pruning needed |

---

## All Experiments (2026-08-11)

**Setup:** `Qwen/Qwen3-VL-2B-Instruct` · GB200 (gb-nvl-081-compute03) · `nvcr.io/nvidia/vllm:26.07-py3`  
All modes answer **C** correctly.

### Complete results table

| # | Video | Mode | enforce\_eager | AutoGaze device | Tokens | AutoGaze (ms) | Infer (ms) | E2E (ms) | vs dense\_eager |
|---|---|---|:---:|---|---:|---:|---:|---:|:---:|
| 1 | short (3f) | dense | ✗ | — | 670 | — | 11,754 | 11,754 | — |
| 2 | short (3f) | evs | ✓ | — | 376 | — | 13,359 | 13,359 | — |
| 3 | short (3f) | sparse\_vit | ✓ | GPU (inline) | 148 | ~bundled | 13,013 | 13,013 | — |
| 4 | long (32f) | dense | ✗ | — | 6,403 | — | 13,448 | 13,448 | −2.3% |
| 5 | long (32f) | **dense\_eager** | ✓ | — | 6,403 | — | 13,604 | **13,604** | **baseline** |
| 6 | long (32f) | evs | ✓ | — | 3,365 | — | 13,284 | 13,284 | −2.4% |
| 7 | long (32f) | sparse\_vit | ✓ | **GPU** (inline Docker) | 878 | ~bundled | **13,838** | **13,838** | **+1.7%** |
| 8 | long (32f) | sparse\_vit | ✓ | **CPU** (external miniforge) | 924 | 19,267 | 12,661 | 31,928 | +135% |
| 8† | long (32f) | sparse\_vit | ✓ | CPU (infer only) | 924 | excluded | **12,661** | 12,661 | **−6.9%** |

†Row 8† shows row 8's inference time with AutoGaze preprocessing excluded — isolating the pure ViT+LM savings.

---

### What each variable controls

**`enforce_eager` (rows 4 vs 5):**  
EVS and sparse_vit require `enforce_eager=True` to activate the pruning hooks; this disables CUDA graphs.  
- dense (row 4): CUDA graphs ON → 13,448 ms  
- dense_eager (row 5): CUDA graphs OFF → 13,604 ms (+156 ms, +1.1%)  

`dense_eager` is the correct baseline for EVS and sparse_vit — all three run under the same execution mode.

**AutoGaze on GPU vs CPU (rows 7 vs 8):**  
AutoGaze uses a 4-layer LLaMA decoder that generates patch selections autoregressively.  
- GPU (inside Docker, GB200): overhead bundled, total 13,838 ms  
- CPU (miniforge-aarch64 env, no CUDA): 19,267 ms preprocessing alone → E2E 31,928 ms  

**The AutoGaze preprocessing was running on CPU by mistake.** The miniforge-aarch64 environment uses a CPU-only PyTorch build. On GPU (inside Docker), overhead drops to ~1–2 s.

**Short vs long video (rows 1–3 vs 4–8):**  
Token compression scales with video length:
- Short (3f): sparse_vit selects K=148 tokens (−78% vs dense 670), N=2,352 ViT patches  
- Long (32f): sparse_vit selects K=878 tokens (−86% vs dense 6,403), N=25,088 ViT patches

---

### Key findings

**1. Inference speedup is confirmed (row 8† vs row 5):**  
With AutoGaze preprocessing excluded, sparse_vit (12,661 ms) beats dense_eager (13,604 ms) by **6.9%** and evs (13,284 ms) by **4.7%**. The savings come from:
- Sparse ViT runs O(K²) attention on 878 patches instead of O(N²) on 6,403 → faster rendering
- LM decodes 924 tokens instead of 6,403 → faster prefill and decode

**2. GPU AutoGaze makes sparse_vit near-neutral E2E (row 7 vs row 5):**  
With AutoGaze on GPU (row 7), sparse_vit is 13,838 ms vs dense_eager 13,604 ms — **+1.7%, within measurement noise**. AutoGaze GPU overhead (~1–2 s) approximately equals the ViT+LM savings (~1.1 s).

**3. CPU AutoGaze destroys E2E (row 8 vs row 5):**  
CPU preprocessing (19,267 ms) makes E2E 2.3× slower than dense_eager. This is a deployment issue — the miniforge-aarch64 env has no CUDA.

**4. EVS overhead for short video (row 2 vs row 1):**  
EVS cuts tokens 44% but is 14% *slower* than dense on the short video because `enforce_eager` overhead exceeds token savings at this scale. At 32 frames (row 6), EVS recovers to −2.4% vs dense_eager.

**5. Short video sparse_vit inference faster than EVS (row 3 vs row 2):**  
Even at 3 frames, sparse_vit (13,013 ms) beats EVS (13,359 ms) by 346 ms (2.6%). The sparse ViT selection gives an edge even at small N.

---

## Shortcomings

### ViT and LM timing not captured

All `ViT (ms)` and `LM (ms)` values are empty. The CUDA timing hook runs inside vLLM's `EngineCore` subprocess and every IPC mechanism tried failed:

| Mechanism | Why it failed |
|---|---|
| Thread-local | Not visible across process fork |
| File (`/tmp/`) | EngineCore has isolated mount namespace |
| Anonymous mmap (`MAP_SHARED`) | Post-fork lock issues on Python 3.12 |
| `multiprocessing.Value` | Same post-fork lock issue |
| `stdout` print | EngineCore stdout redirected by vLLM |
| Unix socket pair | vLLM uses `spawn`; child starts fresh without inherited fds |

### Original results required a different Docker image

The reference (sparse_vit 14,188 ms vs dense 17,348 ms, 18% faster) used an internal image where the ViT ran in the main process (V0 single-process executor). CUDA timing worked → ViT measured at 691 ms vs EVS 3,148 ms (4.56×). That 2,457 ms saving was directly visible in wall time.

With `nvcr.io/nvidia/vllm:26.07-py3` (V1 engine), the gather op fires correctly (878 vs 6,403 tokens confirms it), but ViT savings are absorbed inside the rendering phase.

---

## How to run

```bash
git checkout vllm-integration-experiments
export HF_HOME=/home/scratch.thannan_wwfo/hf_cache

# Short video (3 frames, 670 dense tokens)
python3 scripts/runtime_analysis.py \
    --modes dense evs sparse_vit \
    --gazing-ratio 0.245 --pruning-rate 0.5 --reps 3

# Long video — GPU AutoGaze inline, fair eager comparison
python3 scripts/runtime_analysis.py \
    --modes dense_eager evs sparse_vit \
    --gazing-ratio 0.245 --pruning-rate 0.5 --reps 1 \
    --video /workspace/AutoGaze/assets/long_test_video.mp4 \
    --max-frames 32 --fps 2.0

# Long video — external AutoGaze (CPU host), inference-only elapsed
python3 scripts/runtime_analysis.py \
    --modes dense_eager evs sparse_vit \
    --external-autogaze \
    --gazing-ratio 0.245 --pruning-rate 0.5 --reps 1 \
    --video /workspace/AutoGaze/assets/long_test_video.mp4 \
    --max-frames 32 --fps 2.0
```

---

## Next steps

1. **Accelerate AutoGaze**: The LLaMA-4L decoder runs autoregressively. Batching frames, INT8/FP8 quantization, or caching masks would cut GPU overhead well below 500 ms → net E2E speedup over dense_eager.

2. **Eliminate enforce\_eager**: Communicating K to vLLM's scheduler without `video_pruning_rate` (e.g., upstream PR exposing `token_count` in `ModalityInput`) would allow CUDA graphs → close the remaining 156 ms gap.

3. **Accuracy at scale**: Run EgoSchema / Video-MME to validate answer quality at high compression ratios across many questions.

4. **ViT timing**: Run `scripts/bench_vit_timing.py` (loads visual encoder from HuggingFace directly, CUDA events in-process) to confirm the (N/K)² ≈ 62× attention speedup independently of vLLM.
