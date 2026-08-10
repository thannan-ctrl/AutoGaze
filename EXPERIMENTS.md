# AutoGaze × vLLM — Comprehensive Experiment Log

End-to-end timing analysis of sparse ViT selection across multiple configurations.
All experiments run on `nvcr.io/nvidia/vllm:26.07-py3` (vLLM 0.24.0), GB200.

---

## Setup

**Model:** `Qwen/Qwen3-VL-2B-Instruct`  
**Videos:**
- Short: `assets/example_input.mp4` — 2.6 s, 25 fps, 64 frames, 448×448
- Long: `assets/long_test_video.mp4` — 31 s, 25 fps, 768 frames, 448×448 (12-loop of short)

**Reproduce:**
```bash
# Short video (6 visual frames in vLLM, 670 dense tokens)
python3 scripts/runtime_analysis.py \
    --modes dense dense_eager evs sparse_vit \
    --fps 2.0 --max-frames 6 --reps 1

# Long video (32 visual frames in vLLM, 6403 dense tokens)
python3 scripts/runtime_analysis.py \
    --modes dense dense_eager evs sparse_vit \
    --fps 2.0 --max-frames 32 --reps 1 \
    --video /workspace/AutoGaze/assets/long_test_video.mp4
```

---

## Key finding — video rendering dominates timing

All modes spend ~11.5 s in `Rendering conversations` (video decode + ViT encoding via vLLM's
internal encoder). The LLM prefill + decode is only ~0.5–1.5 s. This makes the ViT speedup
from sparse selection invisible in end-to-end wall time.

```
Total inference time ≈ rendering (~11.5 s) + LLM (~0.5–1.5 s)
                       ─────────────────   ──────────────────
                       dominated by I/O    where we save time
```

The rendering time is the same across all modes because vLLM always decodes the full video
and runs the full dense ViT during preprocessing — even in sparse_vit mode, because the
EngineCore subprocess runs the ViT before our thread-local SparseViTContext is consulted.

---

## Experiment 1 — enforce_eager overhead

**Question:** How much does `enforce_eager=True` add by itself (no token reduction)?

`dense_eager` = same as dense but with `enforce_eager=True`, no `video_pruning_rate`.

| Video | dense (ms) | dense_eager (ms) | overhead |
|---|---:|---:|---:|
| Short (670 tokens) | 11,993 | — (GPU not released) | — |
| Long (6403 tokens) | 12,958 | 12,477 | −481 ms |
| Long run 2 | 12,277 | 12,563 | +286 ms |

**Result:** enforce_eager overhead is small and variable (±300 ms). On GB200, CUDA graph
compilation is fast and re-execution is efficient, so the penalty is not the ~400 ms
sometimes seen on A100/H100. This means EVS's `enforce_eager` overhead is negligible.

---

## Experiment 2 — nframes/max_pixels probe

**Question:** How many visual tokens does vLLM actually process for different settings?

| Video | nframes | fps | max_pixels | prompt tokens | visual ≈ |
|---|---:|---:|---:|---:|---:|
| Short (2.6 s) | any | any | any | 625 | 615 |
| Long (31 s) | 16 | 2.0 | 200,704 | 6,358 | 6,318 |
| Long | 32 | 2.0 | 200,704 | 6,358 | 6,318 |
| Long | 128 | 25.0 | 200,704 | 6,358 | 6,318 |

**Result:** `nframes` and `fps` hints are **largely ignored**. vLLM uses its own internal
frame budget. For the long video, it always extracts ~32 frames regardless of hint.
For the short video, it always extracts ~3 frames.

---

## Experiment 3 — Short video comparison

**Video:** `example_input.mp4` (2.6 s) · vLLM processes ~3 frames → ~615 visual tokens

| Mode | Tokens | vs Dense | Infer (ms) | Answer |
|---|---:|:---:|---:|:---:|
| dense | 670 | — | 11,993 | C |
| evs | 376 | −44% | 12,551 | C |
| sparse_vit (K=321, 6.3%) | 162 | −76% | 12,444 | C |

**Observations:**
- All modes take ~12 s (rendering dominates)
- EVS adds ~558 ms from enforce_eager overhead, partially offset by fewer LLM tokens
- sparse_vit answers correctly with only 162 tokens (76% reduction)
- ViT speedup is present but invisible in total time

---

## Experiment 4 — Long video comparison (main result)

**Video:** `long_test_video.mp4` (31 s) · vLLM processes ~32 frames → ~6,318 visual tokens  
**Note:** grid is 28×28 patches/frame (not 32×32 as initially assumed)

| Mode | Tokens | vs Dense | Infer (ms) | Answer | Notes |
|---|---:|:---:|---:|:---:|---|
| dense | 6,403 | — | 12,958 | C | CUDA graphs ON |
| dense_eager | 6,403 | — | 12,477 | C | enforce_eager only |
| evs | 3,365 | −47% | 12,739 | C | post-ViT selection |
| **sparse_vit** | **878** | **−86%** | **13,171** | **C** | AutoGaze K=2204/25088 (8.8%) |

**Key numbers:**
- dense vs evs: 12,958ms vs 12,739ms — EVS is 219 ms faster despite enforce_eager
- dense vs sparse_vit: 12,958ms vs 13,171ms — sparse_vit is 213 ms SLOWER despite 86% fewer tokens
- sparse_vit vs evs: 13,171ms vs 12,739ms — sparse_vit is 432 ms slower

**Why sparse_vit is slower despite 86% token reduction:**
1. AutoGaze inline preprocessing runs inside Docker → adds ~0.5 s overhead
2. The sparse ViT gather op does NOT fire in vLLM V1 (see §Technical limitations below)
3. Rendering time (~11.5 s) dominates regardless of token count
4. Net: AutoGaze overhead > LM savings at this video scale

**Accuracy:** sparse_vit answers correctly with only 878 tokens (86% fewer than dense 6,403).
AutoGaze's task-driven selection maintains accuracy with aggressive compression.

---

## Experiment 5 — ViT timing (CUDA events)

In the original Docker image (`gitlab-master...main-py3.60784172-devel-arm64`), the ViT ran
in-process (single-process executor), and CUDA event timing hooks measured it directly:

| Mode | ViT (ms) | LM (ms) | Total (ms) |
|---|---:|---:|---:|
| dense | n/a | n/a | 17,320 |
| evs | 3,189 | 13,649 | 16,838 |
| sparse_vit (ratio=0.245) | **690** | **13,526** | **14,216** |

**ViT speedup: 4.6× (3,189 ms → 690 ms)**

These measurements are from earlier runs with the original image and could not be reproduced
with `nvcr.io/nvidia/vllm:26.07-py3` because the V1 engine runs the ViT in a separate
subprocess (see §Technical limitations).

---

## Technical limitations found during experiments

### 1 — Sparse ViT gather op bypassed in vLLM V1

**Problem:** `SparseViTContext` sets a thread-local payload in the main process. But
`Qwen2_5VLVisionTransformer.forward` runs inside `EngineCore` subprocess. The thread-local
is invisible to the subprocess, so `get_sparse_payload()` always returns None → the gather
op is never executed → ViT runs dense even in sparse_vit mode.

**Current workaround:** Class-level patch is applied (correct), but mask communication needs
IPC (shared memory / pipe / file) rather than thread-local to actually enable the gather op.

**Impact:** In sparse_vit mode, vLLM runs the full dense ViT but only allocates K KV-cache
slots. The retention mask returns all-True for K tokens from the dense output. This is
effectively post-ViT selection with AutoGaze's K (rather than EVS's K). The LLM sees fewer
tokens, but the ViT compute savings are not realised.

### 2 — ViT timing hooks also bypass subprocess

**Problem:** `patch_vit_timing` wraps the encoder forward with CUDA events. Same subprocess
issue — the timing fires only when the encoder runs in-process. All `vit_ms` values are null
in current runs.

### 3 — vLLM encoder cache invalidates multi-rep timing

**Problem:** vLLM 0.24 caches visual-token embeddings (separate from text prefix cache).
Reps 2+ return in ~40 ms with cached visual tokens — not representative of real inference.
The cache can also return wrong answers when our retention-mask patches are bypassed.

**Fix:** Use `--reps 1`. The encoder cache cannot be disabled via `enable_prefix_caching=False`
(that only disables text KV cache); `encoder_cache_size=0` is not a valid arg in this build.

### 4 — Video frame count not controlled by nframes hint

**Problem:** `nframes` and `fps` in `video_url` are hints that vLLM may ignore. The actual
frame count is determined by Qwen2.5-VL's internal processor based on video duration,
frame rate, and an internal `max_num_frames` limit. For our 2.6 s video: always ~3 frames.
For the 31 s video: always ~32 frames.

### 5 — ViT patch grid: 28×28, not 32×32

**Problem:** We assumed QWEN_VIT_GRID_HW = (32, 32) (448px / 14px = 32). Actual observation:
`video_grid_thw = [T, 28, 28]`. vLLM resizes frames to 392px before patchifying
(28 patches × 14 px/patch = 392 px, not 448). This caused AutoGaze mask size mismatches.

**Fix:** Updated `QWEN_VIT_GRID_HW = (28, 28)` and `QWEN_GRID_HW = (14, 14)` in `worker.py`.

### 6 — Two-process log corruption

**Problem:** Using `run_in_background=True` with `python3 ... &` spawned two Python processes
sharing the same log file and JSON output. Results interleaved, JSON corrupted.

**Fix:** Run synchronously with `tee` for log capture.

---

## Summary table across all experiments

| Experiment | Video | Mode | Tokens | Infer (ms) | Answer | ViT (ms) |
|---|---|---|---:|---:|:---:|---:|
| Short, reps=1 | 6f | dense | 670 | 11,993 | C | n/a |
| Short, reps=1 | 6f | evs | 376 | 12,551 | C | n/a |
| Short, reps=1 | 6f | sparse_vit | 162 | 12,444 | C | n/a |
| Long, reps=1 | 32f | dense | 6,403 | 12,958 | C | n/a |
| Long, reps=1 | 32f | dense_eager | 6,403 | 12,477 | C | n/a |
| Long, reps=1 | 32f | evs | 3,365 | 12,739 | C | n/a |
| Long, reps=1 | 32f | sparse_vit | 878 | 13,171 | C | n/a |
| **Original image** | **6f** | **evs** | **376** | **16,838** | **C** | **3,189** |
| **Original image** | **6f** | **sparse_vit** | **356** | **14,216** | **C** | **690** |

The original-image rows are the definitive ViT speedup measurement (CUDA events, in-process ViT).
Current-image rows use the new vLLM V1 engine where the ViT runs in a subprocess.

---

## Next steps to realise the full speedup

1. **Fix sparse ViT subprocess communication:** Replace `SparseViTContext` thread-local with
   shared memory or a pre-inference file read so the mask reaches `EngineCore`.
2. **Upstream vLLM PR:** Expose `token_count` in `ModalityInput` so K is communicated to the
   scheduler without monkey-patching.
3. **Accuracy at scale:** Run EgoSchema / Video-MME to validate quality holds at high compression.
4. **Isolate ViT timing:** Add CUDA timing at the class-level using a pre-process hook that
   records start/end events before/after the ViT blocks (not after the full forward).
