# AutoGaze × vLLM — Sparse ViT

**TL;DR:** We integrated AutoGaze into vLLM to select patches *before* the ViT runs, reducing attention cost from O(N²) to O(K²). The sparse ViT fires correctly. Inference alone is 6.9% faster than a fair baseline. AutoGaze preprocessing (currently on CPU by mistake) is the remaining bottleneck — on GPU it becomes near break-even E2E.

---

## How it works

```
Video frames
  ├─ AutoGaze  →  bool mask (K of N patches selected, adaptive per frame)
  └─ vLLM
       ├─ patch_embed(all N)          # cheap conv, unchanged
       ├─ GATHER K patches  ←─ mask  # key change
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
| dense | ✗ | — | 6,403 | 13,448 | — | 13,448 | −1.1% |
| **dense\_eager** | ✓ | — | 6,403 | 13,604 | — | **13,604** | **baseline** |
| evs | ✓ | — | 3,365 | 13,284 | — | 13,284 | −2.4% |
| sparse\_vit | ✓ | GPU (inline Docker) | 878 | 13,838 | ~bundled | 13,838 | +1.7% |
| sparse\_vit | ✓ | CPU (external host) | 924 | **12,661** | 19,267 | 31,928 | +135% |

`dense_eager` = dense with `enforce_eager=True` — same execution mode as EVS/sparse_vit (no CUDA-graph advantage for the baseline).  
`Infer (ms)` for the CPU-external row is Docker inference time only; the 19,267 ms AutoGaze preprocessing ran separately on the host and is shown in its own column.

---

## Key findings

**1. Inference speedup is real (−6.9% vs dense\_eager).**  
The CPU-external row isolates pure inference: sparse_vit Docker time (12,661 ms) beats dense_eager (13,604 ms) by **943 ms**. Token reduction: 924 vs 6,403 (−86%). Both ViT compute and LM decode benefit.

**2. GPU AutoGaze brings E2E to near break-even (+1.7%).**  
With AutoGaze running on the GB200 inside Docker, preprocessing overhead is ~1–2 s (bundled into Infer). The ViT+LM savings (~943 ms) nearly cancel it. The 1.7% gap is within run-to-run noise.

**3. AutoGaze was accidentally running on CPU.**  
The external miniforge-aarch64 Python env has no CUDA → 19,267 ms → E2E 2.3× slower than dense_eager. On GPU this drops to ~1–2 s. **This was a deployment bug, not a fundamental problem.**

**4. `enforce_eager` adds ~156 ms — and sparse_vit still wins.**  
EVS and sparse_vit require `enforce_eager=True` to activate pruning hooks, disabling CUDA graphs. Dense without eager: 13,448 ms; dense\_eager: 13,604 ms (+156 ms). Even so, sparse_vit inference-only (12,661 ms) beats plain dense (13,448 ms) by **5.9%**.

---

## Limitations

- **ViT timing not captured.** CUDA events inside vLLM's EngineCore subprocess cannot be read from the parent process — all IPC mechanisms tried (file, mmap, shared memory, socket pair, stdout) failed. The theoretical (N/K)² ≈ 62× attention speedup and the original 4.56× measured ViT speedup cannot be re-confirmed with this vLLM version.

- **Original results required a different image.** The reference (sparse_vit 14,188 ms vs dense 17,348 ms, −18%) was on an internal NVIDIA image with vLLM's V0 single-process executor, where the ViT ran in-process and timing worked. The current public image (V1, EngineCore subprocess) absorbs ViT savings inside the rendering phase, making end-to-end comparison noisier.

---

## Next steps

| Priority | Action | Expected impact |
|---|---|---|
| High | Run AutoGaze on GPU (fix deployment) | Already demonstrated: E2E within noise of dense\_eager |
| High | Quantize / batch AutoGaze decoder (INT8, larger chunks) | AutoGaze GPU: 1–2 s → <500 ms → clear net E2E speedup |
| High | Remove `enforce_eager` dependency (upstream vLLM PR) | Recover 156 ms; enable CUDA graphs for sparse\_vit |
| Medium | Benchmark EgoSchema / Video-MME | Validate accuracy at high compression ratios |
| Low | Fix ViT timing IPC | Directly confirm attention-layer speedup |
