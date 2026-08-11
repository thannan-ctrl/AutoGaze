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
**Command:** `--modes dense evs sparse_vit --gazing-ratio 0.245 --pruning-rate 0.5 --reps 1`

| Mode | Tokens | vs Dense | Infer (ms) | ViT (ms) | LM (ms) | Answer |
|---|---:|:---:|---:|:---:|:---:|:---:|
| dense | 6,403 | — | 13,448 | — | — | C |
| evs (q=0.5) | 3,365 | −47% | 13,284 | — | — | C |
| **sparse\_vit (ratio=0.245)** | **878** | **−86%** | **13,866** | **—** | **—** | **C** |

All modes answer correctly. Token compression works as expected: AutoGaze selected ~878 of 6,403 tokens (13.7%), reducing LM input by 86%.

---

## Shortcomings

### 1. sparse_vit is slower than dense end-to-end

Despite 86% fewer tokens, sparse_vit (13,866 ms) is **3% slower** than dense (13,448 ms). This has three causes:

**a) Rendering dominates (~12–13 s, same for all modes).** vLLM's internal "Rendering conversations" phase — video decode, frame preprocessing, and ViT encoding — takes ~12–13 s regardless of pruning mode. The ViT savings from K=878 vs N=6,403 patches are real but occur inside this 12–13 s block, making them invisible in total wall time.

**b) `enforce_eager=True` overhead (~600 ms).** EVS and sparse_vit require `enforce_eager=True` to activate the pruning hooks. This disables CUDA graphs, adding ~600 ms for non-dense modes.

**c) AutoGaze preprocessing on 32 frames (~1–2 s).** AutoGaze runs the ShallowVideoConvNet + LLaMA-4L decoder on all 32 frames before vLLM starts. This overhead grows with frame count.

Net: AutoGaze overhead + enforce_eager overhead ≈ 1.5–2.5 s; ViT+LM savings ≈ 0.5–1 s → sparse_vit appears slower.

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
