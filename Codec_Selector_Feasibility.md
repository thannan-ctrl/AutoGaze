# Codec-Based Selector for AutoGaze: Results

`"codec"` mode (`scripts/breakdown`) vs. AutoGaze's trained selector, same base model
(**NVILA-8B-HD-Video**). Token-count-matched via `CODEC_RATIO_SCALE=0.28`. Caches
cleared before every run below except the videomme one (videomme has ~3
questions/video, so most items there hit a legitimate warm cache — split into
cold/warm rows).

`avg selector cost` = `autogaze_ops_ms + autogaze_model_ms`. `avg VLM cost` =
`avg_e2e_ms − avg_preproc_ms`. `avg tokens` = final `input_ids` length.

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

Resumable — skips `item_id`s already in the output `.jsonl`. Delete a mode's `.jsonl`
under `benchmark_results/` to force a clean re-run, and `data/hevc_dump_cache/*` for
a genuinely-cold `codec` re-run.

```bash
# GB200 nodes are aarch64 -- the x86_64 auto_gaze conda env won't run there.
source /home/scratch.thannan_wwfo/miniforge-aarch64/etc/profile.d/conda.sh
conda activate auto_gaze

# One-time native build of hevc_dump for aarch64 (skip if
# scripts/hevc_dump/cmake_build_aarch64/dump_stats already exists):
#   cmake -S scripts/hevc_dump -B scripts/hevc_dump/cmake_build_aarch64 \
#     -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_C_FLAGS=-fPIC -DCMAKE_CXX_FLAGS=-fPIC
#   cmake --build scripts/hevc_dump/cmake_build_aarch64
# codec_selector.py picks this build (vs. the x86_64 cmake_build) via platform.machine().

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

