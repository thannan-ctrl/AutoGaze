# Full EgoSchema + VideoMME latency-breakdown run — status & restart guide

**Why this file exists**: this runs under a `salloc`/`srun` allocation with an 8-hour wall-clock
limit. If the allocation is reclaimed mid-run, this file plus the resumable JSONL outputs let a
new session pick up exactly where it left off, with zero re-derivation.

## How to resume

**See [RESTART.md](RESTART.md)** for the full step-by-step (new `salloc`, environment, and what
to tell Claude Code). Short version: `scripts/breakdown/runner.py` is resumable — it reads
`benchmark_results/nvila_hd_accuracy_breakdown_<mode>_<dataset>_nvf16.jsonl`, skips any
`item_id` already in it, and appends new results. Killing and re-launching is always safe.

## Commands

```bash
# EgoSchema (500 questions, the full answerable subset)
CUDA_VISIBLE_DEVICES=<gpu> REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  FIXED_NUM_VIDEO_FRAMES=16 N_SAMPLES=full MAX_BATCH_SIZE_AUTOGAZE=64 \
  MODES=autogaze,dense DATASET=egoschema \
  python3 scripts/nvila_hd_accuracy_breakdown_test.py

# VideoMME (1395 questions -- only ones whose video is downloaded locally;
# 465/900 videos present in data/video_mme/videos/, see README/Data)
CUDA_VISIBLE_DEVICES=<gpu> REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  FIXED_NUM_VIDEO_FRAMES=16 N_SAMPLES=full MAX_BATCH_SIZE_AUTOGAZE=64 \
  MODES=autogaze,dense DATASET=video_mme \
  python3 scripts/nvila_hd_accuracy_breakdown_test.py
```

Both datasets launched as background processes on separate GPUs (`nohup ... & disown`), each
running `MODES=autogaze,dense` mode-major (all autogaze questions, then all dense questions —
not interleaved). Logs: `EXPERIMENT_LOG_egoschema.log`, `EXPERIMENT_LOG_video_mme.log`.

**Launched**: 2026-08-18 ~06:04 UTC, job 1880436, ~3h46m remaining on the allocation at launch.

## Status table

| Dataset | Mode | GPU | Target n | Progress | Status |
|---|---|---|---:|---:|---|
| egoschema | autogaze | 1 | 500 | 500/500 | **done** |
| egoschema | dense | 1 | 500 | 500/500 | **done** |
| video_mme | autogaze | 2 | 1395 | 1395/1395 | **done** |
| video_mme | dense | 2 | 1395 | 1395/1395 | **done** |

**Run complete.** Job 1880436 (original launch) was killed by the 8h wall-clock limit after
EgoSchema finished both modes. Resumed video_mme under job 1884278 per RESTART.md; autogaze
picked up at 1366/1395 and finished immediately, dense mode then ran to completion.

## Results summary

| Dataset | Mode | n | Accuracy | Avg e2e latency | Avg tokens to LLM |
|---|---|---:|---:|---:|---:|
| egoschema | autogaze | 500 | 60.4% | 8328 ms | 1598 |
| egoschema | dense | 500 | 54.4% | 5350 ms | 23784 |
| video_mme | autogaze | 1395 | 55.6% | 9572 ms | 1526 |
| video_mme | dense | 1395 | 54.9% | 5666 ms | 28318 |

AutoGaze wins accuracy on both datasets (EgoSchema +6.0pp, VideoMME +0.6pp) and cuts tokens fed
to the LLM by 15-19x, but is net **slower end-to-end** on both (EgoSchema 1.56x, VideoMME 1.69x)
because the AutoGaze gazing-selection model itself (~6.1-7.0s avg) costs more than the
LLM/ViT time it saves downstream. See per-dataset summary JSONs for full metric breakdowns:
`benchmark_results/nvila_hd_accuracy_breakdown_summary_{egoschema,video_mme}_nvf16.json`.

Update this table (and re-check `wc -l` on the jsonl files below) each time you check in.

## Progress check

```bash
wc -l benchmark_results/nvila_hd_accuracy_breakdown_*_egoschema_nvf16.jsonl 2>/dev/null
wc -l benchmark_results/nvila_hd_accuracy_breakdown_*_video_mme_nvf16.jsonl 2>/dev/null
```
(counts should climb toward 500 / 1395 respectively; each line is one completed question)

## Known limitation

VideoMME questions.json has 2700 questions across 900 unique videos, but only 465 videos are
downloaded locally (`data/video_mme/videos/`) — 1395 of the 2700 questions are actually
answerable here. "Full" VideoMME in this run means those 1395, not all 2700. See README.md's
Data section for how to fetch the rest if a complete run is needed later.

## Timing (measured via 2-question smoke test on each dataset before launch)

| | dense | autogaze |
|---|---:|---:|
| egoschema (per q) | ~4.9s | ~8.3s |
| video_mme (per q) | ~5.8s | ~10.1s |

| Dataset | Mode | n | Est. wall time |
|---|---|---:|---:|
| egoschema | autogaze | 500 | ~69 min |
| egoschema | dense | 500 | ~41 min |
| egoschema | **total** | | **~110 min** (fits in the remaining window) |
| video_mme | autogaze | 1395 | ~236 min |
| video_mme | dense | 1395 | ~135 min |
| video_mme | **total** | | **~371 min (~6.2h)** — will NOT finish in one 8h allocation from this launch point |

**Feasibility**: EgoSchema (both modes) should complete within this session. VideoMME will not —
at launch there was ~3h46m left, autogaze alone needs ~236min, so dense for video_mme likely
won't even start before the job ends. This is expected; the run is resumable (see RESTART.md)
and will keep making progress across sessions until both hit their target counts.
