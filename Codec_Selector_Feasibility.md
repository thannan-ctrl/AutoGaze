# Codec-Based Selector for AutoGaze: Results

`"codec"` mode (`scripts/breakdown`) vs. AutoGaze's trained selector, same base model
(**NVILA-8B-HD-Video**). Token-count-matched via `CODEC_RATIO_SCALE=0.28`. Caches
cleared before every run below except the videomme one (~3 questions/video, so most
items there hit a legitimate warm cache — split into cold/warm rows).

`avg selector cost` = `autogaze_ops_ms + autogaze_model_ms`. `avg VLM cost` =
`avg_e2e_ms − avg_preproc_ms`. `avg tokens` = final `input_ids` length.

## Concept

`codec_selector.build_gazing_info()` is a drop-in substitute for AutoGaze's trained
selector — same integration point, same output tensor schema. Instead of a trained
autoregressive model, it scores HEVC coding units (small + motion-heavy + non-skip →
high) and keeps a fixed top-k per frame.

![From HEVC codec layout to AutoGaze patch indices](figures/codec_to_patches_diagram.svg)

## N=500 (full), egoschema, nvf=16

| Mode | Accuracy | avg e2e | avg selector cost | avg VLM cost | avg tokens |
|---|---|---|---|---|---|
| `codec` (scale=0.28) | 307/500 = 61.4% | 5,214ms | 3,927ms | 315ms | 1,572 |
| `autogaze` | 302/500 = 60.4% | 8,328ms | 6,936ms | 409ms | 1,598 |
| `dense` (baseline) | 272/500 = 54.4% | 5,350ms | — | 4,294ms | 23,784 |

## N=1395 (full), videomme, nvf=16

| Mode | Accuracy | avg e2e | avg selector cost | avg VLM cost | avg tokens |
|---|---|---|---|---|---|
| `codec`, all (33% cold / 67% warm) | 780/1395 = 55.9% | 4,021ms | 2,709ms | 311ms | 1,547 |
| `codec`, cold only (n=465) | 55.7% | 6,135ms | 4,771ms | 320ms | 1,546 |
| `codec`, warm only (n=930) | 56.0% | 2,963ms | 1,678ms | 306ms | 1,548 |
| `autogaze`, all | 775/1395 = 55.6% | 9,571ms | 8,101ms | 352ms | 1,526 |
| `dense`, all (baseline) | 766/1395 = 54.9% | 5,666ms | — | 4,576ms | 28,318 |

---

## Reproducing

```bash
source /home/scratch.thannan_wwfo/miniforge-aarch64/etc/profile.d/conda.sh
conda activate auto_gaze  # GB200 = aarch64; build hevc_dump/cmake_build_aarch64 first if missing

run() {  # $1=dataset $2=nvf
  rm -rf data/hevc_dump_cache/*
  rm -f benchmark_results/nvila_hd_accuracy_breakdown_codec_${1}_nvf${2}.jsonl
  CUDA_VISIBLE_DEVICES=0 REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    FIXED_NUM_VIDEO_FRAMES=$2 N_SAMPLES=full MAX_BATCH_SIZE_AUTOGAZE=64 MODES=codec,autogaze,dense \
    CODEC_RATIO_SCALE=0.28 DATASET=$1 python3 scripts/nvila_hd_accuracy_breakdown_test.py
}
run egoschema 16
run video_mme 16

# Results: benchmark_results/nvila_hd_accuracy_breakdown_summary_{egoschema,video_mme}_nvf16.json
```
