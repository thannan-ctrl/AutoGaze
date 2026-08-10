# AutoGaze × vLLM — Results

## Sparse ViT integration

**Model:** `Qwen/Qwen3-VL-2B-Instruct`  
**Container:** `nvcr.io/nvidia/vllm:26.07-py3` (vLLM 0.24.0+092c4842, GB200 arm64)  
**Video:** `assets/example_input.mp4` (6 frames, 448×448)  
**Method:** 3 reps, warmup excluded, CUDA events for ViT/LM split

| Mode | Tokens | vs Dense | ViT (ms) | LM (ms) | Infer (ms) | Answer |
|---|---:|:---:|---:|---:|---:|:---:|
| dense | 670 | — | n/a | n/a | 17,320 | C |
| evs (q=0.5) | 376 | −44% | 3,189 | 13,649 | 16,838 | C |
| **sparse\_vit (ratio=0.245)** | **356** | **−47%** | **690** | **13,526** | **14,216** | **C** |

`infer_ms` excludes model load (~26 s). All modes answer correctly (C).

## How sparse\_vit beats EVS

EVS and sparse\_vit operate at different compression levels when given the same numeric ratio.
EVS `q=0.5` keeps ~25% of merged tokens (376). AutoGaze `ratio=0.5` keeps 44% of ViT patches
(681 merged tokens) — so end-to-end time at ratio=0.5 is actually *slower* than EVS.

At `ratio=0.245`, AutoGaze keeps ~21.6% of ViT patches → 356 LLM tokens (matching EVS):

- **ViT: 4.6× faster** — 690 ms vs 3,189 ms. Scales as (K/N)² ≈ (0.216)² ≈ 4.7×. ✓
- **LM: same** — 13,526 ms vs 13,649 ms. Same token count → same decode cost.
- **Net: 15% faster** — 14,216 ms vs 16,838 ms.

## AutoGaze's adaptive K

At the same `ratio=0.245`, AutoGaze selected two different K values across runs:

| Run | K\_vit | Retention | Tokens | Infer (ms) | Answer |
|---|---:|:---:|---:|---:|:---:|
| Run A | 1,326 | 21.6% | 356 | 14,216 | C |
| Run B | 433 | 7.0% | 133 | 13,265 | C |

Run B shows AutoGaze's `task_loss_prediction_head` stopping very early — it learned that only
7% of patches were needed to answer this question. EVS has no such signal and always uses
the same fraction. Both runs answer correctly; Run B is 21% faster than EVS.

## What `gazing_ratio` and EVS `q` actually mean

| Parameter | What it controls |
|---|---|
| EVS `q=0.5` | Keep top 50% of post-merge tokens by cosine similarity score → fixed K |
| AutoGaze `ratio=0.5` | *Target* 50% patch retention per frame, but stop early if confident → variable K |

To match EVS's token count (~376), use `ratio≈0.245` for AutoGaze.
