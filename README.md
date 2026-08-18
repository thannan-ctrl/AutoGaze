# AutoGaze: Internal Testing

[![Website](https://img.shields.io/badge/Website-76b900?style=for-the-badge&logo=safari&labelColor=555555)](https://autogaze.github.io/)

AutoGaze (Autoregressive Gazing) is a model that automatically selects informative patches and removes redundant ones in any video, such that downstream ViTs/MLLMs can process fewer patches without information loss. This makes downstream ViTs/MLLMs much more scalable to high-resolution, high-FPS, long-form videos (e.g., 4K-resolution 1K-frame videos).

## NVILA-HD-Video @ nvf=16: Fine-Grained Latency Breakdown

Dense vs AutoGaze (chunked-batched, `MAX_BATCH_SIZE_AUTOGAZE=64`), 25 EgoSchema questions.

### Results (n=25, avg/question)

![Latency breakdown: Dense vs AutoGaze](assets/nvf16_summary_plots/latency_breakdown_dense_vs_autogaze.png)

#### Deatailed Table:

| | dense | autogaze (batched) |
|---|---:|---:|
| accuracy | 52.0% (13/25) | 68.0% (17/25) |
| avg tokens to LLM | 24,145 | 1,535 |
| avg e2e | 4,915 ms | 8,257 ms |
| decode (CPU) | 31 | 105 |
| image_preproc (CPU) | 867 | 867 |
| autogaze_ops (CPU) | 0 | 818 |
| autogaze_model (GPU) | 0 | 6,007 |
| other (CPU) | 69 | 5 |
| vit (GPU) | 2,893 | 188 |
| llm_prefill (GPU) | 872 | 114 |
| llm_decode (GPU) | 126 | 108 |

### Findings

1. **Dense preproc is CPU-bound; AutoGaze preproc is GPU-bound** — its gazing model (6,007ms) is
   ~77% of its 7.8s preproc.
2. **AutoGaze's win is on the LLM side, not preproc.** ~16x fewer tokens cuts `vit` 15x and
   `prefill` 7x, but its 6s gazing-model cost outweighs those savings — AutoGaze is currently
   *slower* end-to-end (8.3s vs 4.9s).

## Full-dataset results: EgoSchema (n=500) + VideoMME (n=1395)

Same methodology as above, scaled up to the full EgoSchema subset and the locally-available
VideoMME subset (see [Data](#data) for why VideoMME is 1395 of 2700 questions). Both datasets run
mode-major (`MODES=autogaze,dense`), `N_SAMPLES=full`, otherwise identical settings.

| | EgoSchema dense | EgoSchema autogaze | VideoMME dense | VideoMME autogaze |
|---|---:|---:|---:|---:|
| accuracy | 54.4% (272/500) | 60.4% (302/500) | 54.9% (766/1395) | 55.6% (775/1395) |
| avg tokens to LLM | 23,784 | 1,598 | 28,318 | 1,526 |
| avg e2e | 5,350 ms | 8,328 ms | 5,666 ms | 9,572 ms |
| decode (CPU) | 67 | 110 | 52 | 54 |
| image_preproc (CPU) | 884 | 868 | 934 | 1,059 |
| autogaze_ops (CPU) | 9 | 822 | 4 | 1,060 |
| autogaze_model (GPU) | 0 | 6,114 | 0 | 7,041 |
| other (CPU) | 95 | 5 | 99 | 5 |
| vit (GPU) | 3,325 | 166 | 3,472 | 146 |
| llm_prefill (GPU) | 823 | 98 | 973 | 77 |
| llm_decode (GPU) | 92 | 94 | 78 | 86 |

Raw per-question data and averaged summaries: `benchmark_results/nvila_hd_accuracy_breakdown_{autogaze,dense}_{egoschema,video_mme}_nvf16.jsonl` / `..._summary_{egoschema,video_mme}_nvf16.json`. Full run log/status: `EXPERIMENT_LOG.md`.

### Findings

1. **Same qualitative pattern holds at full scale, on a second dataset.** AutoGaze wins accuracy
   on both datasets (EgoSchema +6.0pp, VideoMME +0.6pp) and cuts LLM input tokens 15-19x, but is
   net *slower* end-to-end on both at nvf=16 (EgoSchema 8.3s vs 5.4s, 1.56x; VideoMME 9.6s vs
   5.7s, 1.69x).
2. **The gazing model's own cost is the bottleneck, not the LLM/ViT savings.** `autogaze_model`
   averages 6.1-7.0s across both datasets — more than the entire dense-mode pipeline (5.4-5.7s) —
   so the 20-24x `vit` and 7-13x `prefill` savings it enables downstream don't close the gap.
3. **VideoMME's accuracy gap is much smaller than EgoSchema's** (+0.6pp vs +6.0pp), suggesting
   the accuracy benefit is dataset-dependent and not guaranteed to offset the latency cost.

### Installation

```bash
conda create -n autogaze python=3.11 && conda activate autogaze
conda install -c nvidia cuda-toolkit=12.8   # match your installed torch's CUDA version if different
pip install uv
uv pip install -e .
```

If that fails on your CUDA/architecture, install torch/transformers for your platform manually
first, then `pip install -e . --no-deps`. On aarch64 (GB200), `transformers~=4.51` in
`pyproject.toml` is too old — use `transformers==5.14.1` instead.

### Data

```
data/egoschema/
  subset.json          # {q_uid, question, "option 0".."option 4", answer, ...}
  videos/<q_uid>.mp4
```

```bash
huggingface-cli download VLM2Vec/egoschema-rawvideo --repo-type dataset \
  --local-dir data/egoschema/videos
```

`subset.json`/`questions.json` follow the official EgoSchema question format
(github.com/egoschema/EgoSchema) — source them from there.

### Models

No manual download step — `nvidia/NVILA-8B-HD-Video` (~16.2GB) and `nvidia/AutoGaze` (tiny) are
neither gated nor private, so `trust_remote_code=True` auto-downloads both from Hugging Face Hub
on first run and caches them (`HF_HOME`, or `~/.cache/huggingface` by default). Requires internet
access on first run; every run after that is offline/cached. No HF token/login needed.

### How to Run
```bash
CUDA_VISIBLE_DEVICES=1 REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  FIXED_NUM_VIDEO_FRAMES=16 N_SAMPLES=25 MAX_BATCH_SIZE_AUTOGAZE=64 MODES=autogaze,dense \
  DATASET=egoschema \
  python3 scripts/nvila_hd_accuracy_breakdown_test.py
python3 scripts/plot_latency_breakdown.py
```

`DATASET` selects `egoschema` or `video_mme`. `N_SAMPLES` also accepts `full` to use every
available item — this is what was used for the [full-dataset results](#full-dataset-results-egoschema-n500--videomme-n1395)
above:

```bash
conda activate auto_gaze
cd /home/thannan/scratch/AutoGaze

# EgoSchema (500 questions, the full answerable subset)
CUDA_VISIBLE_DEVICES=<gpu> REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  FIXED_NUM_VIDEO_FRAMES=16 N_SAMPLES=full MAX_BATCH_SIZE_AUTOGAZE=64 \
  MODES=autogaze,dense DATASET=egoschema \
  python3 scripts/nvila_hd_accuracy_breakdown_test.py

# VideoMME (1395 questions -- only ones whose video is downloaded locally, see Data above)
CUDA_VISIBLE_DEVICES=<gpu> REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  FIXED_NUM_VIDEO_FRAMES=16 N_SAMPLES=full MAX_BATCH_SIZE_AUTOGAZE=64 \
  MODES=autogaze,dense DATASET=video_mme \
  python3 scripts/nvila_hd_accuracy_breakdown_test.py
```

Set `CUDA_VISIBLE_DEVICES` to a free GPU (check with `nvidia-smi`); run each dataset on a separate
GPU if launching both at once. The runner is resumable — it skips any `item_id` already present
in the output jsonl, so if an allocation dies mid-run (e.g. an `srun` wall-clock limit), just
re-launch the same command. See `RESTART.md` for the full resume procedure and `EXPERIMENT_LOG.md`
for this run's status/history.

`scripts/nvila_hd_accuracy_breakdown_test.py` monkey-patches the vendored (`trust_remote_code`)
`NVILAProcessor` at runtime (no source edits) to split `preproc_ms` into `decode`, `image_preproc`,
`autogaze_ops`, `autogaze_model` (GPU), and `other`, and `llm_ms` into `prefill`/`decode`. It also
short-circuits the AutoGaze CPU transform whenever a processor's config skips AutoGaze for both
tiles and thumbnails (dense mode) — safe because that output is never read in the skip path.

### Raw outputs

- `benchmark_results/nvila_hd_accuracy_breakdown_{autogaze,dense}_egoschema_nvf16.jsonl` — per-question (this README's n=25 numbers)
- `benchmark_results/nvila_hd_accuracy_breakdown_summary_egoschema_nvf16.json` — averaged summary
