# Quickstart: `dense`, `codec` (optimized selector), `codec_nvdec` (optimized selector)

Bare-minimum steps to reproduce these three results from scratch. See
[`Codec_Selector_Feasibility.md`](Codec_Selector_Feasibility.md) for the full methodology
and results table.

| Mode | What it is | EgoSchema acc. | Avg E2E |
|---|---|--:|--:|
| `dense` | No patch selection at all (baseline) | 58.6% | 4.38s |
| `codec` (optimized selector) | HEVC motion/size heuristic, CPU (`libde265`) decode | 61.6% | 4.50s |
| `codec_nvdec` (optimized selector) | Same heuristic, GPU (NVDEC) decode | 60.6% | **2.44s** |

`codec_nvdec` additionally needs GB200-class hardware with NVIDIA driver ≥595.84.01 /
Video Codec SDK 13.1+ (confirmed on `gb200nvl4`-class nodes; an older-driver node such as
`gb200nvl72_preprod` does not support NVDEC — `dense` and `codec` don't have this
requirement).

## 1. Clone (with the `hevc_dump` submodule)

```bash
git clone --recurse-submodules --branch share/codec-nvdec-quickstart \
  git@github.com:thannan-ctrl/AutoGaze.git AutoGaze
cd AutoGaze
```

If you cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

## 2. Environment

```bash
conda create -n autogaze python=3.11 && conda activate autogaze
conda install -c nvidia cuda-toolkit=12.8   # match your installed torch's CUDA version if different
pip install uv
uv pip install -e .
pip install PyNvVideoCodec   # only needed for codec_nvdec
```

If that fails on your CUDA/architecture, install `torch`/`transformers` manually first,
then `pip install -e . --no-deps`. On aarch64 (GB200), `transformers~=4.51` in
`pyproject.toml` is too old — use `transformers==5.14.1` instead.

## 3. Build `hevc_dump` (only needed for `codec`; skip for `dense`/`codec_nvdec`-only)

```bash
cd scripts/hevc_dump
mkdir cmake_build && cd cmake_build
cmake ..
make -j4
cd ../../..
```

Details: [`scripts/hevc_dump/README.md`](scripts/hevc_dump/README.md).

## 4. Data

```bash
mkdir -p data/egoschema
huggingface-cli download VLM2Vec/egoschema-rawvideo --repo-type dataset \
  --local-dir data/egoschema/videos
```

Then place `data/egoschema/subset.json` (`{q_uid, question, "option 0".."option 4",
answer, ...}` per question) — source from the official EgoSchema repo
(github.com/egoschema/EgoSchema); it isn't part of the video download above.

No model download step needed — `nvidia/NVILA-8B-HD-Video` and `nvidia/AutoGaze`
auto-download from Hugging Face Hub on first run (`trust_remote_code=True`, neither gated
nor private).

## 5. Run

```bash
CUDA_VISIBLE_DEVICES=0 REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 \
  FIXED_NUM_VIDEO_FRAMES=16 N_SAMPLES=full MAX_BATCH_SIZE_AUTOGAZE=32 \
  MODES=dense,codec,codec_nvdec DATASET=egoschema \
  python3 scripts/nvila_hd_accuracy_breakdown_test.py
```

Drop `codec` from `MODES=` if you skipped step 3. Results land in
`benchmark_results/nvila_hd_accuracy_breakdown_{mode}_egoschema_nvf16.jsonl` (per-question)
and the matching `_summary_*.json` (averaged) — expected values are the table above.
