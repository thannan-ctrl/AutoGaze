# Quickstart: `dense`, `codec` (optimized selector), `codec_nvdec` (optimized selector)

Bare-minimum steps to reproduce these three results from scratch — node allocation through
printing the final table. Everything below is a plain command meant to be typed by hand, no
tooling assumed. See [`Codec_Selector_Feasibility.md`](Codec_Selector_Feasibility.md) for
the full methodology.

| Mode | What it is | N | Acc | Tokens | Encode | Decode | Selector | ViT | LLM | E2E |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `dense` | No patch selection at all (baseline) | 500 | 58.6% | 24,176 | — | — | — | 2.70s | 0.80s | 4.38s |
| `codec` (optimized selector) | HEVC motion/size heuristic, CPU (`libde265`) decode | 500 | 61.6% | 2,716 | 0.87s | 1.11s | 1.40s | 0.16s | 0.29s | 4.50s |
| `codec_nvdec` (optimized selector) | Same heuristic, GPU (NVDEC) decode | 500 | 60.6% | 2,716 | 0.86s | 0.09s | 0.37s | 0.16s | 0.28s | **2.44s** |

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

**Videos:**

```bash
mkdir -p data/egoschema
huggingface-cli download VLM2Vec/egoschema-rawvideo --repo-type dataset \
  --local-dir data/egoschema/videos
```

**Questions + answers**, from the official EgoSchema repo:

```bash
git clone https://github.com/egoschema/EgoSchema.git /tmp/EgoSchema_official
cp /tmp/EgoSchema_official/questions.json data/egoschema/questions.json
cp /tmp/EgoSchema_official/subset_answers.json data/egoschema/subset_answers.json
```

`questions.json` is the full 5,031-question set (no answers, held out for leaderboard
submission); `subset_answers.json` is `{q_uid: answer_idx}` for just the 500-question
public Subset this repo uses. `data/egoschema/subset.json` (what `dataset.py` actually
reads) is the two merged — filtered to the 500 Subset `q_uid`s, with `answer` added:

```bash
python3 -c "
import json

questions = json.load(open('data/egoschema/questions.json'))
answers = json.load(open('data/egoschema/subset_answers.json'))

by_quid = {q['q_uid']: q for q in questions}
subset = []
for q_uid, answer in answers.items():
    item = dict(by_quid[q_uid])
    item['answer'] = answer
    subset.append(item)

json.dump(subset, open('data/egoschema/subset.json', 'w'))
print(f'wrote {len(subset)} questions to data/egoschema/subset.json')
"
```

Should print `wrote 500 questions to data/egoschema/subset.json`.

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

Read the per-question JSONL directly and average **every** timing/count field the harness
records per question (`runner.py`'s full return dict — not just the headline columns from
the table at the top):

```bash
python3 -c "
import json

FIELDS = [
    'num_tokens',
    'preproc_ms', 'decode_ms', 'image_preproc_ms', 'autogaze_ops_ms', 'autogaze_model_ms',
    'codec_encode_ms', 'codec_decode_ms',
    'selector_score_ms', 'selector_rank_ms',
    'selector_csvparse_ms', 'selector_scorecu_ms', 'selector_paintmap_ms',
    'gazing_info_total_ms', 'selector_glue_ms',
    'other_ms', 'cpu_ms', 'gpu_ms',
    'generate_ms', 'vit_ms', 'llm_prefill_ms', 'llm_decode_ms', 'llm_ms', 'llm_calls',
    'e2e_ms',
]

modes = ['dense', 'codec', 'codec_nvdec']
rows_by_mode = {}
for mode in modes:
    path = f'benchmark_results/nvila_hd_accuracy_breakdown_{mode}_egoschema_nvf16.jsonl'
    rows_by_mode[mode] = [json.loads(l) for l in open(path)]

header = ['mode', 'n', 'acc'] + FIELDS
print(' | '.join(header))
print(' | '.join('---' for _ in header))
for mode in modes:
    rows = rows_by_mode[mode]
    n = len(rows)
    acc = sum(r.get('correct', False) for r in rows) / n
    avg = lambda k: sum((r.get(k) or 0) for r in rows) / n
    cells = [mode, str(n), f'{acc:.1%}']
    for f in FIELDS:
        v = avg(f)
        cells.append(f'{v:.0f}' if f in ('num_tokens', 'llm_calls') else f'{v/1000:.3f}s')
    print(' | '.join(cells))
"
```

Every field is 0 where a mode genuinely doesn't touch that code path — e.g. `dense` has no
`codec_*`/`selector_*` fields (no patch selection at all), and `codec_nvdec` has no
`selector_csvparse_ms`/`selector_scorecu_ms`/`selector_paintmap_ms` (those are `libde265`
CSV-parsing-only; the NVDEC backend gets its data from `.npz` decode stats instead, never
touching that code path) — that's expected, not a bug.
