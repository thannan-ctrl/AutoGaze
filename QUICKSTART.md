# Quickstart: dense / codec / codec_nvdec

3 modes, ranked by speed. Full writeup: [`Codec_Selector_Feasibility.md`](Codec_Selector_Feasibility.md).

| Mode | N | Acc | Tokens | Encode | Decode | Selector | ViT | LLM | E2E |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| dense (baseline, no selection) | 500 | 58.6% | 24,176 | — | — | — | 2.70s | 0.80s | 4.38s |
| codec (CPU decode) | 500 | 61.6% | 2,716 | 0.87s | 1.11s | 1.40s | 0.16s | 0.29s | 4.50s |
| codec_nvdec (GPU decode) | 500 | 60.6% | 2,716 | 0.86s | 0.09s | 0.37s | 0.16s | 0.28s | **2.44s** |

`codec_nvdec` needs a GB200-class node, driver ≥595.84.01. `dense`/`codec` run anywhere.

## 0. Get a node

```bash
srun --partition=gb200nvl4 --time=08:00:00 --gres=gpu:1 --pty bash
```
(8h was the cluster's max per job — plan around it. No shared FS on these nodes, so use `/tmp`.)

## 1. Clone

```bash
git clone --recurse-submodules --branch share/codec-nvdec-quickstart \
  git@github.com:thannan-ctrl/AutoGaze.git AutoGaze
cd AutoGaze
```

## 2. Environment

```bash
conda create -n autogaze python=3.11 && conda activate autogaze
conda install -c nvidia cuda-toolkit=12.8
pip install uv && uv pip install -e .
pip install PyNvVideoCodec   # for codec_nvdec only
```
GB200/aarch64: use `transformers==5.14.1` if `uv pip install -e .` pulls in something older.

## 3. Build hevc_dump (codec only — skip for dense/codec_nvdec)

```bash
cd scripts/hevc_dump && mkdir cmake_build && cd cmake_build && cmake .. && make -j4 && cd ../../..
```

## 4. Get the data

```bash
mkdir -p data/egoschema
huggingface-cli download VLM2Vec/egoschema-rawvideo --repo-type dataset --local-dir data/egoschema/videos

git clone https://github.com/egoschema/EgoSchema.git /tmp/EgoSchema_official
cp /tmp/EgoSchema_official/questions.json /tmp/EgoSchema_official/subset_answers.json data/egoschema/

python3 scripts/build_egoschema_subset.py
```
Models download themselves on first run — nothing to do there.

## 5. Run

```bash
CUDA_VISIBLE_DEVICES=0 REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 \
  FIXED_NUM_VIDEO_FRAMES=16 N_SAMPLES=full MAX_BATCH_SIZE_AUTOGAZE=32 \
  MODES=dense,codec,codec_nvdec DATASET=egoschema \
  python3 scripts/nvila_hd_accuracy_breakdown_test.py
```
Drop `codec` from `MODES=` if you skipped step 3.

## 6. Print the table

```bash
python3 scripts/print_table.py
```
