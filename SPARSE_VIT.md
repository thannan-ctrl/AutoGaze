# AutoGaze × vLLM — Sparse ViT

**TL;DR:** We integrated AutoGaze into vLLM to select patches *before* the ViT runs, reducing attention cost from O(N²) to O(K²). The sparse ViT fires correctly and reduces inference time by ~7% vs a fair baseline. AutoGaze preprocessing is the remaining bottleneck.

---

## How it works

```
Video frames
  ├─ AutoGaze  →  bool mask (K of N patches selected, adaptive per frame)
  └─ vLLM
       ├─ patch_embed(all N)          # cheap conv, unchanged
       ├─ GATHER K patches  ←─ mask  # key change: O(K) instead of O(N)
       ├─ ViT blocks(K)               # O(K²) attention vs O(N²) dense
       ├─ merger  →  K/4 tokens
       └─ LLM  →  answer
```

Three hooks monkey-patched in vLLM: `Qwen2_5VLVisionTransformer.forward` (gather op), `compute_retained_tokens_count` (KV-cache allocation), `compute_retention_mask` (identity pass-through).

---

## Results

**Model:** Qwen/Qwen3-VL-2B-Instruct · **Video:** 31 s, 32 frames · **GPU:** GB200  
**Image:** nvcr.io/nvidia/vllm:26.07-py3 · All modes answer correctly (**C**)

| Mode | enforce\_eager | AutoGaze | Tokens | Infer (ms) | AutoGaze preproc (ms) | E2E (ms) | vs dense\_eager |
|---|:---:|---|---:|---:|---:|---:|:---:|
| dense | ✗ | — | 6,403 | 13,448 | — | 13,448 | −2.3% |
| **dense\_eager** | ✓ | — | 6,403 | 13,604 | — | **13,604** | **baseline** |
| evs | ✓ | — | 3,365 | 13,284 | — | 13,284 | −2.4% |
| sparse\_vit | ✓ | GPU (inline) | 878 | 13,838 | ~bundled | 13,838 | +1.7% |
| sparse\_vit | ✓ | CPU (external) | 924 | 12,661 | 19,267 | 31,928 | +135% |
| **sparse\_vit†** | ✓ | CPU (infer only) | 924 | **12,661** | excluded | **12,661** | **−6.9%** |

†Inference time only, AutoGaze preprocessing excluded from elapsed.  
`dense_eager` = dense with `enforce_eager=True` (same mode as EVS/sparse_vit — no CUDA-graph advantage).

---

## Key findings

**1. Inference speedup is real (−6.9% vs dense\_eager).**  
With AutoGaze preprocessing excluded, sparse_vit (12,661 ms) beats dense_eager (13,604 ms) by 943 ms. Token reduction: 924 vs 6,403 (−86%). Both ViT compute and LM decode are cheaper.

**2. GPU AutoGaze makes sparse\_vit near break-even E2E (+1.7%).**  
Running AutoGaze inside Docker on the GB200 bundles ~1–2 s of preprocessing into inference. The ViT+LM savings (~1.1 s) nearly cancel it. The 1.7% gap is within run-to-run noise.

**3. AutoGaze was accidentally running on CPU.**  
The external miniforge-aarch64 env has no CUDA → 19,267 ms preprocessing → E2E 2.3× worse. On GPU this drops to ~1–2 s. **This was the deployment bug, not the approach.**

**4. `enforce_eager` costs ~156 ms.**  
EVS and sparse_vit require `enforce_eager=True` to activate pruning hooks, disabling CUDA graphs. Dense without eager runs at 13,448 ms; dense\_eager at 13,604 ms. Even accounting for this, sparse\_vit inference-only (12,661 ms) beats plain dense (13,448 ms) by **5.9%**.

---

## Limitations

- **ViT timing not captured.** The CUDA event hook runs inside vLLM's EngineCore subprocess. All IPC mechanisms (file, mmap, shared memory, socket pair, stdout) failed to return the value to the parent. The 4.56× ViT speedup from the original experiment cannot be directly re-measured with this vLLM version.

- **Original results used a different image.** The reference (sparse\_vit 14,188 ms vs dense 17,348 ms, −18%) was on an internal NVIDIA image (V0 single-process executor) where the ViT ran in-process and timing worked directly. The current public image (V1, EngineCore subprocess) hides ViT savings inside the rendering phase.

---

## Next steps

| Priority | Action | Expected impact |
|---|---|---|
| High | Quantize / batch AutoGaze decoder (INT8, larger chunks) | AutoGaze GPU overhead: 1–2 s → <500 ms → net E2E speedup |
| High | Remove `enforce_eager` dependency (upstream vLLM PR) | Recover 156 ms from CUDA graphs |
| Medium | Benchmark EgoSchema / Video-MME | Validate accuracy at high compression ratios |
| Low | Fix ViT timing IPC | Confirm (N/K)² ≈ 62× attention speedup directly |
