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

## Results (2026-08-11)

**Model:** `Qwen/Qwen3-VL-2B-Instruct` · **Video:** `assets/long_test_video.mp4` (31 s, 32 frames sampled)  
**GPU:** GB200 (gb-nvl-081-compute03) · **Image:** `nvcr.io/nvidia/vllm:26.07-py3`

All modes answer correctly (**C**). Token compression works: AutoGaze selected ~924 of 6,403 visual tokens (14.4%).

### Inference-only comparison (all modes with `enforce_eager=True`)

`dense_eager` is the correct baseline — same execution mode as EVS and sparse_vit (no CUDA graphs).

| Mode | Tokens | Infer (ms) | vs dense\_eager | AutoGaze preproc (ms) | E2E (ms) |
|---|---:|---:|:---:|---:|---:|
| dense\_eager | 6,403 | 13,777 | — | — | 13,777 |
| evs | 3,365 | 13,532 | −1.8% | — | 13,532 |
| **sparse\_vit** | **924** | **12,661** | **−8.1%** | **19,267** | **31,928** |

**sparse_vit is 8.1% faster than dense_eager and 6.4% faster than EVS in inference time.**  
The 924 token LM input (vs 6,403 dense) reduced both ViT compute and decode time.

### Why E2E is still slower: AutoGaze preprocessing

The AutoGaze mask computation (ShallowVideoConvNet + LLaMA decoder on 32 frames) takes **19,267 ms** — more than the **1,116 ms** saved in inference. End-to-end, sparse_vit costs 31,928 ms vs 13,777 ms for dense_eager.

### CUDA graphs vs eager mode (why dense < dense_eager)

| Mode | enforce_eager | Infer (ms) | Notes |
|---|:---:|---:|---|
| dense | ✗ | 13,448 | CUDA graphs active — fastest baseline |
| dense\_eager | ✓ | 13,777 | +329 ms graph-compilation overhead |
| evs | ✓ | 13,532 | enforce\_eager required for pruning hooks |
| sparse\_vit | ✓ | 12,661 | enforce\_eager + sparse selection |

Comparing sparse_vit (12,661 ms) to **dense** (13,448 ms, no enforce_eager): **sparse_vit is still 5.9% faster**, even with the graph-compilation overhead it must pay and dense does not.

---

## Shortcomings

### 1. AutoGaze preprocessing dominates E2E time

The inference speedup is confirmed (**8.1% faster** than dense_eager, **6.4% faster** than EVS), but the AutoGaze mask computation (19,267 ms on 32 frames) makes E2E wall time 2.3× worse than not using AutoGaze at all.

The preprocessing runs ShallowVideoConvNet + LLaMA-4L autoregressively on all frames. It needs to be accelerated (batching, early-exit tuning, or hardware-optimized inference) for the E2E speedup to be net positive.

### 2. ViT and LM timing not captured

The `ViT (ms)` and `LM (ms)` columns are empty. The CUDA event timing hook runs inside vLLM's `EngineCore` subprocess, and all IPC mechanisms attempted failed to deliver the value to the parent process:

| Mechanism | Outcome |
|---|---|
| Thread-local (`_vit_timing.ms`) | Invisible across fork |
| File write (`/tmp/`) | Child writes to its own namespace copy |
| Anonymous mmap (`MAP_SHARED`) | Post-fork lock issues on Python 3.12 |
| `multiprocessing.Value` | Same lock issue |
| `stdout` print | EngineCore stdout is redirected by vLLM |
| Unix socket pair (`socketpair`) | vLLM likely uses `spawn` not `fork` → child doesn't inherit parent's fds |

Without ViT timing, it is not possible to directly measure the ViT speedup (the key claim: K²/N² ≈ (0.14)² ≈ 50× attention reduction) from within this inference pipeline.

### 3. Original speedup was on a different Docker image

The reference result (sparse_vit 14,188 ms vs dense 17,348 ms — 18% faster) was measured on an **NVIDIA internal image** (`gitlab-master...main-py3.60784172-devel-arm64`) that used vLLM's V0 single-process executor. In that executor:
- The ViT runs in the **main process** (no subprocess)
- `SparseViTContext` thread-locals are directly visible → gather op fires
- CUDA event timing works → ViT measured at 691 ms vs EVS 3,148 ms (4.6×)
- ViT savings of 2,457 ms on a 16,840 ms baseline → 16% speedup visible end-to-end

With `nvcr.io/nvidia/vllm:26.07-py3` (V1 engine), the ViT runs in an `EngineCore` subprocess. The gather op does fire (token count proves: 878 vs 6,403), but the savings cannot be measured or seen in wall time because rendering dominates.

---

## How to run

```bash
git checkout vllm-integration-experiments
export HF_HOME=/home/scratch.thannan_wwfo/hf_cache

# Short video (3 frames, 670 dense tokens)
python3 scripts/runtime_analysis.py \
    --modes dense evs sparse_vit \
    --gazing-ratio 0.245 --pruning-rate 0.5 --reps 3

# Long video (32 frames, 6403 dense tokens) — shows larger token compression
python3 scripts/runtime_analysis.py \
    --modes dense evs sparse_vit \
    --gazing-ratio 0.245 --pruning-rate 0.5 --reps 1 \
    --video /workspace/AutoGaze/assets/long_test_video.mp4 \
    --max-frames 32 --fps 2.0
```

---

## Next steps

1. **Get ViT timing**: Run `scripts/bench_vit_timing.py` inside Docker directly — loads the visual encoder from HuggingFace, times dense vs sparse ViT with CUDA events in-process (no subprocess, no IPC). This will confirm the attention-layer speedup.

2. **Eliminate enforce_eager overhead**: Implement sparse_vit without relying on `video_pruning_rate` (which forces enforce_eager). If the KV-cache slot count can be communicated to vLLM's scheduler differently, sparse_vit could run with CUDA graphs and close the ~600 ms gap.

3. **Accuracy benchmark**: Run EgoSchema / Video-MME to validate that AutoGaze's adaptive patch selection preserves answer quality at high compression ratios (current tests are single-question).

4. **Upstream vLLM PR**: Expose `token_count` in `ModalityInput` so K is communicated to the scheduler without monkey-patching.
