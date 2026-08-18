# AutoGaze: Internal Testing

[![Website](https://img.shields.io/badge/Website-76b900?style=for-the-badge&logo=safari&labelColor=555555)](https://autogaze.github.io/)

AutoGaze (Autoregressive Gazing) is a model that automatically selects informative patches and removes redundant ones in any video, such that downstream ViTs/MLLMs can process fewer patches without information loss. This makes downstream ViTs/MLLMs much more scalable to high-resolution, high-FPS, long-form videos (e.g., 4K-resolution 1K-frame videos).

## NVILA-HD-Video @ nvf=16: Fine-Grained Latency Breakdown

Dense vs AutoGaze (chunked-batched, `MAX_BATCH_SIZE_AUTOGAZE=64`), 25 EgoSchema questions.

### Results (n=25, avg/question)

![Latency breakdown: Dense vs AutoGaze](assets/nvf16_summary_plots/latency_breakdown_dense_vs_autogaze.png)

Dense's AutoGaze CPU ops are short-circuited (unused output — see "How to Run"); this is the
already-fixed number.

| | dense | autogaze (batched) |
|---|---:|---:|
| accuracy | 52.0% (13/25) | 68.0% (17/25) |
| avg tokens to LLM | 24,145 | 1,535 |
| avg e2e | 4,915 ms | 8,257 ms |
| decode (CPU) | 31 | 105 |
| image_preproc (CPU) | 867 | 867 |
| autogaze_ops (CPU) | ~0 (short-circuited) | 818 |
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

### How to Run
```bash
CUDA_VISIBLE_DEVICES=1 REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  FIXED_NUM_VIDEO_FRAMES=16 N_SAMPLES=25 MAX_BATCH_SIZE_AUTOGAZE=64 MODES=autogaze,dense \
  python3 scripts/nvila_hd_accuracy_breakdown_test.py
python3 scripts/plot_latency_breakdown.py
```

`scripts/nvila_hd_accuracy_breakdown_test.py` monkey-patches the vendored (`trust_remote_code`)
`NVILAProcessor` at runtime (no source edits) to split `preproc_ms` into `decode`, `image_preproc`,
`autogaze_ops`, `autogaze_model` (GPU), and `other`, and `llm_ms` into `prefill`/`decode`. It also
short-circuits the AutoGaze CPU transform whenever a processor's config skips AutoGaze for both
tiles and thumbnails (dense mode) — safe because that output is never read in the skip path.

### Raw outputs

- `benchmark_results/nvila_hd_accuracy_breakdown_{autogaze,dense}_nvf16.jsonl` — per-question
- `benchmark_results/nvila_hd_accuracy_breakdown_summary_nvf16.json` — averaged summary
