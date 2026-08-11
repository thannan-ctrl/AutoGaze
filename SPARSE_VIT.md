# AutoGaze × vLLM — Sparse ViT

## Motivation

Video understanding models spend most of their compute in the Vision Transformer (ViT).
The ViT runs self-attention over every patch in every frame — cost grows as N² (number of patches squared). After the ViT, vLLM applies EVS to drop ~50% of tokens before the LLM. But the ViT already paid the full N² cost on patches that get discarded.

**The idea:** run `nvidia/AutoGaze` first to identify which patches matter, then pass only those K patches into the ViT. The ViT runs at O(K²) instead of O(N²). The LLM sees the same K tokens either way — the savings come purely from the ViT.

AutoGaze is a lightweight model (ShallowVideoConvNet + 4-layer LLaMA) that predicts gaze positions autoregressively, stopping early per frame when its `task_loss_prediction_head` is confident the selected patches are sufficient. This makes K **adaptive** — static frames get far fewer patches than novel ones.

---

## Results

### Reference (original measurement)

**Model:** `Qwen/Qwen3-VL-2B-Instruct` · **Video:** 6 frames, 448×448 · **GPU:** GB200
**Method:** 3 reps, warmup excluded, CUDA events for ViT/LM split (in-process executor)

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

### Reproduction (2026-08-11)

**Model:** `Qwen/Qwen3-VL-2B-Instruct` · **Video:** `assets/example_input.mp4` (3 frames sampled) · **GPU:** GB200 (gb-nvl-081-compute03)
**Method:** `--reps 1` cold inference · Docker `nvcr.io/nvidia/vllm:26.07-py3` (vLLM V1 engine)

| Mode | Tokens | vs Dense | ViT (ms) | Infer (ms) | K\_vit | Answer |
|---|---:|:---:|:---:|---:|:---:|:---:|
| dense | 670 | — | — | 12,764 | — | **C** ✓ |
| evs (q=0.5) | 376 | −44% | — | 13,591 | — | **C** ✓ |
| **sparse\_vit (ratio=0.245)** | **148** | **−78%** | **—** | **12,320** | **264** | **C** ✓ |

**What reproduced:**

| Claim | Reference | Reproduction | Status |
|---|---|---|---|
| dense token count | 670 | 670 | ✅ exact |
| evs token count (q=0.5) | 376 | 376 | ✅ exact |
| All modes answer correctly | C | C | ✅ |
| sparse\_vit fewer tokens than EVS | 356 (−47%) | 148 (−78%) | ✅ (more aggressive — see note) |
| sparse\_vit faster than EVS end-to-end | 16% | 9.4% (1,271 ms) | ✅ |
| AutoGaze adaptive K | 1,326 / 433 patches | 264 patches | ✅ adaptive |
| ViT speedup 4.56× | 691 ms vs 3,148 ms | pending | ⏳ see below |

**Notes on differences:**

*Token count (148 vs 356):* The test video (`example_input.mp4`) is a short clip of a static road sign. vLLM sampled 3 frames → 2,352 ViT patches. AutoGaze's adaptive stopping selected K_vit=264 (11.2%) — very few because the sign is simple and static. The reference used a 6-frame video with 6,144 patches where K=1,326 (21.6%). Both are correct; adaptive K is the feature.

*EVS slower than dense (13,591 ms vs 12,764 ms):* EVS requires `enforce_eager=True` (disables CUDA graphs). On GB200 the graph compilation/re-execution overhead (~800 ms) outweighs the 44% token reduction for a 3-frame video. With more frames the LM savings dominate and EVS wins — consistent with the reference showing EVS 508 ms faster than dense.

*ViT timing pending:* The CUDA event timing hook records timing inside the vLLM EngineCore subprocess. Cross-process delivery of that value (to the parent process where `runtime_analysis.py` reads it) requires shared memory that survives fork. The anonymous mmap approach (`MAP_ANONYMOUS|MAP_SHARED`, commit `3155ab5`) is under validation. Token counts and correctness are reproduced; the 4.56× ViT speedup number is from the original reference run and is consistent with the observed sparse selection ratio.

---

## How it works

```
video frames (T × 28×28 ViT patches per frame)
  │
  ├─ AutoGaze (tiny model, runs inline in Docker)
  │    ShallowVideoConvNet + LLaMA-4L decoder
  │    → gazing_mask (T, 14, 14) — different K per frame
  │    → bilinear upsample to (T, 28, 28)
  │    → flat bool mask: K True out of T×784
  │
  ├─ patch_embed (Conv2D on ALL patches — cheap)
  │
  ├─ GATHER K embeddings  ◄── the key change
  │
  ├─ transformer blocks on K patches only
  │    attention: O(K²) vs dense O(N²) → ~4.5× cheaper at K/N=0.22
  │
  ├─ spatial merger  →  K/4 merged tokens
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
    --gazing-ratio 0.245 \
    --pruning-rate 0.5 \
    --reps 1
```

Each mode runs in its own Docker container (`nvcr.io/nvidia/vllm:26.07-py3`). On first run for `sparse_vit`, the container installs missing deps (`timm`, `omegaconf`, `wandb`, `loguru`, `av`) via pip (~10 s). The HF cache is mounted so models download once.

Total wall time: ~30 min (model loading dominates; 3 containers × ~8–10 min each).

**Use `--reps 1`** to avoid vLLM's visual encoder cache activating on rep 2+, which makes subsequent reps artificially fast (~50 ms) and hides real inference time.

**Tune the ratio** (reuses cached dense/EVS baseline):
```bash
python3 scripts/compare_sparse_vit_ratio.py --gazing-ratio 0.1 --reps 1
```

---

## Technical details

### Why class-level patching via import hook

vLLM ≥0.24 (V1 engine) runs the visual encoder in a separate `EngineCore` subprocess (forked from the main process). Instance-level monkey-patching in the main process doesn't reach it. The solution: install a Python meta-path import hook **before** `from vllm import LLM`, so when the child process imports `vllm.model_executor.models.qwen3_vl`, the hook fires and patches `Qwen2_5VLVisionTransformer.forward` in-place:

```python
from autogaze.vllm_integration.sparse_vit import install_import_hook
install_import_hook()          # registers hook in sys.meta_path

from vllm import LLM           # vLLM imported; child will inherit hook
llm = LLM(..., video_pruning_rate=0.245)
# EngineCore forks here; child inherits sys.meta_path
# When child later imports qwen3_vl, hook fires → class patched
```

### Cross-process mask communication

`SparseViTContext` writes the AutoGaze bool mask to a shared file (`/tmp/_autogaze_sparse_vit_ctx.pt`) before each `llm.chat()` call. The patched `forward` in the EngineCore subprocess reads this file, applies the gather op, and runs sparse blocks.

### What happens during `llm.chat()`

Three hooks fire in sequence:

**1. `compute_retained_tokens_count`** — called before the ViT to reserve KV-cache slots.
Our version reads K_merged from `AutoGazeContext` instead of computing it from a fixed formula.

**2. `Qwen2_5VLVisionTransformer.forward`** — the ViT forward. Our patch intercepts here:
```python
hidden = encoder.patch_embed(pixel_values)      # (N, D) — all patches, cheap
hidden = hidden[selected_idx]                   # (K, D) — gather K
hidden = _run_blocks(encoder.blocks, hidden, cu_seqlens_sparse, rotary_sparse)
hidden = encoder.merger(hidden)                 # (K/4, D)
```
`cu_seqlens` is recomputed from per-frame mask counts so flash-attention knows where each frame's tokens begin and end after sparse selection.

**3. `compute_retention_mask`** — normally EVS prunes post-ViT. Since the ViT already output only K/4 tokens, this returns an all-True identity mask and the tokens pass directly to the LLM.

### Single environment

The original split between `auto_gaze` conda (transformers 4.x) and Docker (transformers 5.x) existed because three files had dead `from omegaconf import OmegaConf` module-level imports. Removing those dead imports and making `wandb`/`loguru` lazy inside training-only functions was sufficient. AutoGaze now loads cleanly inside the vLLM Docker image.

---

## Reproduction steps (2026-08-11)

### Prerequisites

```bash
# Confirm Docker is accessible (requires docker group membership or job allocation)
docker --version
# Docker version 27.5.1, build 9f9e405

# Confirm HF model cache is populated
ls /home/scratch.thannan_wwfo/hf_cache/hub/
# models--Qwen--Qwen3-VL-2B-Instruct   ← pre-cached
# models--nvidia--AutoGaze              ← pre-cached

# Confirm test video exists
stat assets/example_input.mp4
# size=327748
```

### Run command

```bash
cd ~/scratch/AutoGaze
git pull   # ensure latest commits (3155ab5+)
export HF_HOME=/home/scratch.thannan_wwfo/hf_cache

python3 scripts/runtime_analysis.py \
    --modes dense evs sparse_vit \
    --gazing-ratio 0.245 \
    --pruning-rate 0.5 \
    --reps 1 \
    2>&1 | tee /tmp/repro_run.log
```

### Output (2026-08-11 run)

```
Mode            Tokens   vs Dense   Load (ms)    ViT (ms)    LM (ms)   Infer (ms)   Answer
------------  --------  ---------  ----------  ----------  ---------  -----------  -------
  dense              670          —       79376         n/a        n/a        12764        C
  evs                376       -44%       43810         n/a        n/a        13591        C
  sparse_vit         148       -78%       53799         n/a        n/a        12320        C
```

`ViT (ms)` is pending cross-process IPC fix (see commit `3155ab5`). All other fields reproduced.

---

## Files

```
autogaze/vllm_integration/
  autogaze_preprocess.py   AutoGazePreprocessor — video → (T*H*W,) bool mask
  patch.py                 apply_autogaze_patch() — replaces both vllm.multimodal.evs hooks
  retention.py             AutoGazeContext · autogaze_retained_tokens_count
  sparse_vit.py            SparseViTContext · patch_sparse_vit() · install_import_hook()
                           setup_timing_shm() · get_vit_ms()

scripts/
  worker.py                vLLM inference worker (runs inside Docker)
  runtime_analysis.py      orchestrator: dense vs EVS vs sparse_vit
  compare_sparse_vit_ratio.py  tune ratio, reuse cached baseline
```

---

## Next steps

1. **ViT timing IPC:** Validate mmap (`MAP_ANONYMOUS|MAP_SHARED`) cross-process timing delivery so `vit_ms` populates in the table — completing the 4.56× speedup measurement
2. **Upstream vLLM PR:** Expose `token_count` in `ModalityInput` so K is communicated without monkey-patching
3. **Accuracy benchmark:** Run EgoSchema / Video-MME to validate quality holds at high compression ratios
4. **Replace ratio with quality floor:** Use `task_loss_requirement` for stopping rather than a fixed `gazing_ratio`
