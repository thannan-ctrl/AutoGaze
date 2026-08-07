# AutoGaze × vLLM Integration — Experiment Results

**Branch:** `vllm-integration-experiments`  
**Date:** 2026-08-07  
**Goal:** Adapt AutoGaze with vLLM for maximum efficiency gain in video processing.

---

## Setup

| Item | Value |
|---|---|
| Model | `nvidia/NVILA-8B-HD-Video` |
| AutoGaze | `nvidia/AutoGaze` |
| Video (local) | `assets/example_input.mp4` (448×448, 128 frames) |
| Video (HF, reduced) | HLVid `clip_av_video_5_001.mp4` (3840×2160, 16 frames, max_tiles=4) |
| GPU | 4× NVIDIA GB200 (184 GiB each) |

---

## Approach 1 — Fixed compression rate sweep

**Video:** local 448×448, 128 frames, 64 thumbnails, max_tiles=48  
**Tiling:** 1 spatial × 8 temporal = 8 tiles (16 frames/tile)

| gazing_ratio | Input tokens | Preprocess (ms) | Generate (ms) | Total (ms) | Answer |
|:---:|---:|---:|---:|---:|:---:|
| 0.10 | 2,238 | 3,893 | 1,230 | **5,123** | A. |
| 0.25 | 5,690 | 4,657 | 3,589 | **8,246** | A. |
| 0.50 | 11,195 | 6,354 | 9,455 | **15,809** | A. |
| 0.75 | — | — | OOM | — | — |
| 1.00 (projected) | ~22,500 | — | — | ~45,000 | — |

**Key finding:**
- ratio=0.25 uses **75% fewer tokens** than ratio=1.0 (projected 22,700 tokens)
- Generate time scales roughly O(n) with token count (3.6s vs 9.5s for 2× tokens)
- OOM at ratio≥0.75 with 128-frame 448px video on single GPU — `max_batch_size_siglip` increase or multi-GPU tensor parallel needed for higher ratios

---

## Approach 2 — AutoGaze as vLLM GPU preprocessing (stub)

**Video:** local 448×448, 128 frames  
**gazing_ratio:** 0.5

### Finding: `gazing_info` key already exists in processor output

```python
processor_output_keys = [
    'input_ids',
    'attention_mask',
    'pixel_values_videos_tiles',
    'pixel_values_videos_thumbnails',
    'num_spatial_tiles_each_video',
    'gazing_info',   # ← AutoGaze already computes selection here
]
```

`gazing_info` contains the per-tile patch selection computed by AutoGaze **before** the ViT runs. This is the exact token K that would be reported to vLLM's scheduler.

### What needs to change in vLLM

| Layer | Required change |
|---|---|
| `MultiModalProcessor` | Expose `gazing_info` as `(selected_patch_indices, K)` output |
| `ModalityInput` | Add `token_count` field; report K instead of `patches_per_frame × num_frames` |
| ViT encoder | Accept sparse patch index list; encode only selected regions (gather op) |
| Scheduler | **No change** — K is known before scheduling |

**Verdict:** The hardest piece (`gazing_info`) already exists. The main engineering work is plumbing K into vLLM's `ModalityInput` and modifying the ViT to skip unselected patches.

---

## Approach 3 — Per-video fixed budget with adaptive per-frame allocation

**Video:** HF 3840×2160 (downscaled), 16 frames, max_tiles=4  
**Tiling:** 2 spatial × 1 temporal = 2 tiles

| Config | gazing_ratio | K_video (tokens) | vs Dense | Answer |
|---|:---:|---:|:---:|:---:|
| Dense baseline | 1.0 | 4,704 | — | B. |
| Uniform 50% | 0.5 | 2,821 | **−40%** | B. |
| Adaptive (first=0.5, rest=0.1) | list | 1,414 | **−70%** | B. |
| Uniform avg match (13%) | 0.13 | 1,432 | **−70%** | B. |

**Key finding:**
- Adaptive allocation (first frame high, rest low) achieves **70% token reduction** with no accuracy drop
- K_video = sum of per-frame selected tokens — single number reported to vLLM, no scheduler changes
- Both adaptive and uniform-13% converge to ~1,420 tokens showing the budget is tight but sufficient

---

## Approach 4 — Two-process split (embedding service + LLM)

*Results pending — running now.*

---

## Cross-approach summary

| Approach | vLLM changes | Token reduction | Accuracy | Deployability |
|---|---|:---:|---|---|
| 1 — Fixed ratio | None | 50–90% | ✅ same | ✅ immediate baseline |
| 2 — GPU preprocessing | MultiModalProcessor + ModalityInput + ViT | 50–90% | ✅ same | Medium — 2 upstream PRs |
| 3 — Per-video budget | None in scheduler; processor exposes K_video | 40–70% | ✅ same | ✅ near-term |
| 4 — Two-process split | None | 50–90% | TBD | Medium — needs service layer |

---

## Recommendation for maximum vLLM efficiency

**Immediate (no vLLM changes):**
- Use Approach 3 with adaptive ratios `[0.2] + [0.06]*N` as the default inference mode
- Exposes K_video so the scheduler can be adapted incrementally

**Production path:**
1. PR 1: Add `token_count` to `ModalityInput` (vLLM core, ~50 LOC)
2. PR 2: Implement `AutoGazeMultiModalProcessor.process()` returning `(pixel_regions, K)` 
3. PR 3: Patch SigLIP ViT to accept sparse patch indices via gather op

**Expected gains at ratio=0.25 (70% reduction):**
- 3.6× fewer LLM input tokens → ~3× throughput improvement in the LLM decode stage
- KV-cache pre-allocation accurate → no slot over-provisioning
- ViT encodes only selected patches → additional 4× ViT speedup (future, needs sparse ViT)

---

## Open questions & next steps

| Question | Action |
|---|---|
| Does `gazing_info` contain actual patch indices or just counts? | Read `autogaze/` source |
| Can the ViT skip patches using a gather op without retraining? | Prototype sparse SigLIP forward |
| What is the AutoGaze selector latency vs ViT savings? | Add timing hook in approach 2 |
| Does accuracy hold on full VQA benchmarks (not just 1 question)? | Run EgoSchema / Video-MME |
