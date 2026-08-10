# AutoGaze × vLLM — Sparse ViT

## Motivation

Video understanding models spend most of their compute in the Vision Transformer (ViT).
The ViT runs self-attention over every patch in every frame — cost grows as N² (number of patches squared). After the ViT, vLLM applies EVS to drop ~50% of tokens before the LLM. But the ViT already paid the full N² cost on patches that get discarded.

**The idea:** run `nvidia/AutoGaze` first to identify which patches matter, then pass only those K patches into the ViT. The ViT runs at O(K²) instead of O(N²). The LLM sees the same K tokens either way — the savings come purely from the ViT.

AutoGaze is a lightweight model (ShallowVideoConvNet + 4-layer LLaMA) that predicts gaze positions autoregressively, stopping early per frame when its `task_loss_prediction_head` is confident the selected patches are sufficient. This makes K **adaptive** — static frames get far fewer patches than novel ones.

---

## Results

**Model:** `Qwen/Qwen3-VL-2B-Instruct` · **Video:** 6 frames, 448×448 · **GPU:** GB200  
**Method:** 3 reps, warmup excluded, CUDA events for ViT/LM split

| Mode | Tokens | vs Dense | ViT (ms) | LM (ms) | Infer (ms) | Answer |
|---|---:|:---:|---:|---:|---:|:---:|
| dense | 670 | — | n/a | n/a | 17,348 | C |
| evs (q=0.5) | 376 | −44% | 3,148 | 13,692 | 16,840 | C |
| **sparse\_vit (ratio=0.245)** | **356** | **−47%** | **691** | **13,497** | **14,188** | **C** |

`infer_ms` excludes model load (~26 s). All modes answer correctly.

**sparse\_vit vs EVS:**
- ViT: **4.56× faster** (691 ms vs 3,148 ms) — scales as (K/N)² = (0.216)² ≈ 4.7×
- LM: **unchanged** — same ~356 tokens to the LLM → same decode cost
- End-to-end: **16% faster** (14,188 ms vs 16,840 ms)

**AutoGaze's adaptive K:** At ratio=0.245, AutoGaze selected 1,326 of 6,144 ViT patches (21.6%) in one run and 433 (7.0%) in another. Both gave the correct answer. The model stops when it's confident — no fixed quota per frame.

**The ratio gotcha:** `gazing_ratio=0.5` and EVS `q=0.5` are not the same. EVS keeps ~25% of merged tokens (376). AutoGaze at ratio=0.5 targets 44% of ViT patches → 681 merged tokens → more LM work → slower than EVS overall. Use `ratio≈0.245` to match EVS's token count.

---

## How it works

```
video frames (T × 32×32 ViT patches = 6,144 patches)
  │
  ├─ AutoGaze (tiny model, runs inline)
  │    ShallowVideoConvNet + LLaMA-4L decoder
  │    → gazing_mask (T, 14, 14) — different K per frame
  │    → bilinear upsample to (T, 32, 32)
  │    → flat bool mask: 1,326 True out of 6,144
  │
  ├─ patch_embed (Conv2D on ALL 6,144 patches — cheap)
  │
  ├─ GATHER K=1,326 embeddings  ◄── the key change
  │
  ├─ transformer blocks on K patches only
  │    attention: O(1,326²) vs dense O(6,144²) → 4.6× cheaper
  │
  ├─ spatial merger  →  K/4 = 331 merged tokens
  │
  └─ LLM  →  answer
```

Three things are monkey-patched in vLLM to make this work:

| Hook | What changes |
|---|---|
| `vllm.multimodal.evs.compute_retained_tokens_count` | Returns AutoGaze's actual K instead of a fixed formula, so vLLM allocates the right KV-cache slots |
| `vllm.multimodal.evs.compute_retention_mask` | Returns all-True identity mask — ViT already selected K tokens, no post-ViT pruning needed |
| `Qwen2_5VLVisionTransformer.forward` | Inserts the gather op: patch_embed(all N) → gather K → blocks(K) → merger |

---

## How to run

```bash
git clone <repo> AutoGaze && cd AutoGaze
git checkout vllm-integration-experiments
export HF_HOME=/path/to/hf_cache

python3 scripts/runtime_analysis.py \
    --modes dense evs sparse_vit \
    --single-env \
    --gazing-ratio 0.245 \
    --pruning-rate 0.5 \
    --reps 3
```

`--single-env` runs AutoGaze preprocessing inside the same Docker container as vLLM — no conda environment needed. On first run, the container installs a few missing Python packages (~10 s). The HF cache is mounted so models download once.

Total wall time: ~25 min (model loading dominates; 3 containers × ~8 min each).

**Tune the ratio** (reuses cached dense/EVS baseline):
```bash
python3 scripts/compare_sparse_vit_ratio.py --gazing-ratio 0.1 --reps 3
```

---

## Technical details

### Why class-level patching

vLLM ≥0.24 (V1 engine) runs the visual encoder in a separate `EngineCore` subprocess.
Instance-level monkey-patching in the main process doesn't reach it. The fix: patch the
class's `forward` method *before* `LLM()` is called, so the subprocess inherits the patched
class when it instantiates the model:

```python
from vllm import LLM                          # triggers lazy module imports
patch_sparse_vit(llm=None)                    # patches Qwen2_5VLVisionTransformer.forward
llm = LLM(..., video_pruning_rate=0.245)      # EngineCore inherits patched class
```

### What happens during `llm.chat()`

Three hooks fire in sequence:

**1. `compute_retained_tokens_count`** — called before the ViT to reserve KV-cache slots.
Our version reads K from `AutoGazeContext` (331) instead of computing it from a formula.

**2. `Qwen2_5VLVisionTransformer.forward`** — the ViT forward. Our patch intercepts here:
```python
hidden = encoder.patch_embed(pixel_values)      # (6144, D) — all N, cheap
hidden = hidden[selected_idx]                   # (1326, D) — gather K
hidden = _run_blocks(encoder.blocks, hidden, cu_seqlens_sparse, rotary_sparse)
hidden = encoder.merger(hidden)                 # (331, D)
```
`cu_seqlens` is recomputed from the per-frame mask counts so flash-attention flash-attn-2
knows where each frame's tokens begin and end after sparse selection.

**3. `compute_retention_mask`** — normally EVS prunes post-ViT. Since the ViT already
output only 331 tokens, this returns an all-True identity mask and the 331 tokens pass
directly to the LLM.

### Single environment

The original split between `auto_gaze` conda (transformers 4.x) and Docker (transformers 5.x)
existed because three files had dead `from omegaconf import OmegaConf` module-level imports.
Removing those dead imports and making `wandb`/`loguru` lazy inside the training-only
functions that actually use them was sufficient. AutoGaze now loads cleanly inside the
vLLM Docker image.

---

## Files

```
autogaze/vllm_integration/
  autogaze_preprocess.py   AutoGazePreprocessor — video → (T*H*W,) bool mask
  patch.py                 apply_autogaze_patch() — replaces both vllm.multimodal.evs hooks
  retention.py             AutoGazeContext · autogaze_retained_tokens_count
  sparse_vit.py            SparseViTContext · patch_sparse_vit() · patch_vit_timing()

scripts/
  worker.py                vLLM inference worker (runs inside Docker)
  runtime_analysis.py      orchestrator: dense vs EVS vs sparse_vit
  compare_sparse_vit_ratio.py  tune ratio, reuse cached baseline
```

---

## Next steps

1. Upstream vLLM PR: expose `token_count` in `ModalityInput` so K is communicated without monkey-patching
2. Accuracy benchmark on EgoSchema / Video-MME (current results are single-question)
3. Replace `gazing_ratio` with `task_loss_requirement` for quality-floor stopping rather than ratio-floor
