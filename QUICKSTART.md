# Quickstart: `dense`, `codec` (optimized selector), `codec_nvdec` (optimized selector)

Bare-minimum steps to reproduce these three results from scratch — node allocation through
printing the final table. Everything below is a plain command meant to be typed by hand, no
tooling assumed. See [`Codec_Selector_Feasibility.md`](Codec_Selector_Feasibility.md) for
the full methodology.

| Mode | What it is | EgoSchema acc. | Avg E2E |
|---|---|--:|--:|
| `dense` | No patch selection at all (baseline) | 58.6% | 4.38s |
| `codec` (optimized selector) | HEVC motion/size heuristic, CPU (`libde265`) decode | 61.6% | 4.50s |
| `codec_nvdec` (optimized selector) | Same heuristic, GPU (NVDEC) decode | 60.6% | **2.44s** |

`codec_nvdec` needs GB200-class hardware with NVIDIA driver ≥595.84.01 / Video Codec SDK
13.1+ (confirmed on `gb200nvl4`-class nodes; an older-driver node does not support NVDEC —
`dense` and `codec` don't have this requirement and will run on any CUDA GPU).

## 0. Allocate a node

```bash
srun --partition=gb200nvl4 --time=08:00:00 --gres=gpu:1 --pty bash
```

Adjust `--partition`/`--time` for your cluster's naming and limits (8h was this cluster's
enforced maximum per job at the time of writing — a longer single job will be rejected
outright, not silently capped). If your node has no shared filesystem (true for
`gb200nvl4` on this cluster — local `/tmp` only, wiped when the allocation ends), do
everything below under `/tmp` on that node, not your home/scratch directory, and expect to
redo steps 1-4 on every fresh allocation.

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

Drop `codec` from `MODES=` if you skipped step 3. This prints a running per-question log
line as it goes, and a final `===== Accuracy + Profiling Summary =====` block (one line per
mode) when done — that alone is enough to sanity-check against the table above.

## 6. Print the table

Each mode writes two files under `benchmark_results/`:
`nvila_hd_accuracy_breakdown_{mode}_egoschema_nvf16.jsonl` (one line per question) and
`nvila_hd_accuracy_breakdown_summary_egoschema_nvf16.json` (the averaged summary — gets
overwritten per mode, so copy it aside between runs if running modes separately, or use the
combined `MODES=dense,codec,codec_nvdec` run from step 5, which appends per mode instead).

Read the per-question JSONL directly and average the fields that make up the table's
columns (Encode/Decode = `codec_encode_ms`/`codec_decode_ms`, Selector =
`selector_score_ms` + `selector_rank_ms`, ViT = `vit_ms`, LLM = `llm_ms`):

```bash
python3 -c "
import json, glob

for mode in ['dense', 'codec', 'codec_nvdec']:
    path = f'benchmark_results/nvila_hd_accuracy_breakdown_{mode}_egoschema_nvf16.jsonl'
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    acc = sum(r.get('correct', False) for r in rows) / n
    avg = lambda k: sum(r.get(k, 0) or 0 for r in rows) / n
    print(f'{mode:12s} n={n:4d} acc={acc:.1%} '
          f'tokens={avg(\"num_tokens\"):6.0f} '
          f'encode={avg(\"codec_encode_ms\")/1000:5.2f}s '
          f'decode={avg(\"codec_decode_ms\")/1000:5.2f}s '
          f'selector={(avg(\"selector_score_ms\")+avg(\"selector_rank_ms\"))/1000:5.2f}s '
          f'vit={avg(\"vit_ms\")/1000:5.2f}s '
          f'llm={avg(\"llm_ms\")/1000:5.2f}s '
          f'e2e={avg(\"e2e_ms\")/1000:5.2f}s')
"
```

That prints one line per mode with the same numbers as the table at the top of this file.
