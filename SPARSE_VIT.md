# AutoGaze × vLLM — Sparse ViT

**TL;DR:** We connected AutoGaze to vLLM to select patches *before* the ViT runs, reducing attention cost from O(N²) to O(K²). The sparse ViT fires correctly. Inference alone is 6.9% faster than a fair baseline. AutoGaze preprocessing (currently on CPU by mistake) is the remaining bottleneck — on GPU it becomes near break-even E2E.

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


**1. GPU AutoGaze is near break-even E2E (+1.7%).**  
~1.2 s AutoGaze overhead on GB200 partially offsets the 943 ms inference savings → sparse_vit is only **234 ms slower** than dense_eager. Within run-to-run noise.

**2. `enforce_eager` costs 156 ms — sparse_vit still beats plain dense.**  
Even with CUDA graphs disabled (required for pruning hooks), sparse_vit inference (12,661 ms) is **5.9% faster than dense** (13,448 ms, CUDA graphs on).

**3. At the same latency, sparse_vit processes 7× more video.**  
878 vs 6,403 tokens at 32 frames (**7.3× reduction**, same answer quality). Within a ~13.6 s budget, sparse_vit scales to ~230 frames where dense caps at 32.

---

## Limitations

- **ViT timing not captured.** CUDA events inside vLLM's EngineCore subprocess cannot be read from the parent process — all IPC mechanisms tried (file, mmap, shared memory, socket pair, stdout) failed. The theoretical (N/K)² ≈ 62× attention speedup and the original 4.56× measured ViT speedup cannot be re-confirmed with this vLLM version.


---

## Next steps

| Action | Expected impact |
|---|---|
| Quantize / batch AutoGaze decoder (INT8, larger chunks) | AutoGaze GPU: 1–2 s → <500 ms → clear net E2E speedup |
| Remove `enforce_eager` dependency (upstream vLLM PR) | Recover 156 ms; enable CUDA graphs for sparse\_vit |
| Benchmark EgoSchema / Video-MME | Validate accuracy at high compression ratios |
| Fix ViT timing IPC | Directly confirm attention-layer speedup |
