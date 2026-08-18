# Restart mechanism — full EgoSchema + VideoMME latency-breakdown run

Read this first if the `srun` job died (8h wall-clock limit, or manually killed) and you're
starting a new Claude Code session to continue.

## What actually happens when the job ends

This run's two background processes (EgoSchema on GPU 1, VideoMME on GPU 2) were started with
`nohup ... & disown` inside the current `srun --pty --overlap bash` shell. That only protects
them from the *terminal/Claude Code session* disconnecting — it does **not** protect them from
SLURM. When the job's wall-clock limit hits (or `scancel`), SLURM kills the entire job's cgroup,
which kills every process in it, nohup or not. **Both background runs die with the job.**

What survives: the per-question result files on disk
(`benchmark_results/nvila_hd_accuracy_breakdown_<mode>_<dataset>_nvf16.jsonl`), which are
appended to and `flush()`ed after every single question. Resuming means re-launching the exact
same commands in a fresh allocation — `scripts/breakdown/runner.py` reads each output file at
startup, skips any `item_id` already present, and appends new results. **Nothing is lost, no
re-derivation needed** — you just lose the wall-clock time between the job dying and the next
`salloc` going through.

## Step 1 — new allocation

```bash
salloc --partition=gb200nvl72_preprod --nodes=1 --gres=gpu:1 --time=08:00:00 srun --pty --overlap bash
```

(4 physical GPUs are typically available on this node type — check with `nvidia-smi
--query-gpu=index,memory.used --format=csv` and pick free indices; this run used GPU 1 for
EgoSchema and GPU 2 for VideoMME.)

## Step 2 — environment

```bash
conda activate auto_gaze
cd /home/thannan/scratch/AutoGaze
```

## Step 3 — open Claude Code and say

Just point it at this file — that's the whole handoff:

> Resume the experiment described in RESTART.md.

A fresh Claude Code session has no memory of this conversation, but this file plus
`EXPERIMENT_LOG.md`'s status table is a complete, self-contained handoff — it doesn't need
anything else from you.

## Step 4 — what Claude Code (or you, manually) should do

1. Check `EXPERIMENT_LOG.md`'s status table for which dataset/mode rows aren't done yet.
2. Check progress so far: `wc -l benchmark_results/nvila_hd_accuracy_breakdown_*_egoschema_nvf16.jsonl benchmark_results/nvila_hd_accuracy_breakdown_*_video_mme_nvf16.jsonl` (target: 500 for egoschema, 1395 for video_mme, per mode).
3. Re-launch whichever dataset(s) aren't at target count yet, with the **exact same commands**
   (same `N_SAMPLES=full`, same `FIXED_NUM_VIDEO_FRAMES=16`, same `MODES=autogaze,dense`, same
   `DATASET=`) — changing any of those changes the output filename or sampling and breaks
   resumption:

```bash
# EgoSchema
CUDA_VISIBLE_DEVICES=<free_gpu> REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  FIXED_NUM_VIDEO_FRAMES=16 N_SAMPLES=full MAX_BATCH_SIZE_AUTOGAZE=64 \
  MODES=autogaze,dense DATASET=egoschema \
  nohup python3 scripts/nvila_hd_accuracy_breakdown_test.py > EXPERIMENT_LOG_egoschema.log 2>&1 &
disown

# VideoMME
CUDA_VISIBLE_DEVICES=<free_gpu> REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  FIXED_NUM_VIDEO_FRAMES=16 N_SAMPLES=full MAX_BATCH_SIZE_AUTOGAZE=64 \
  MODES=autogaze,dense DATASET=video_mme \
  nohup python3 scripts/nvila_hd_accuracy_breakdown_test.py > EXPERIMENT_LOG_video_mme.log 2>&1 &
disown
```

Note: `MODES=autogaze,dense` runs *all* autogaze questions first, then *all* dense questions
(not interleaved). If autogaze isn't finished yet, dense for that dataset hasn't started at all
— check the jsonl line counts, not just "is the process running", to know true progress.

4. Update `EXPERIMENT_LOG.md`'s status table with the new GPU indices / start time.
5. Once both datasets hit their target counts for both modes, regenerate summaries + write up
   results (the existing `benchmark_results/nvila_hd_accuracy_breakdown_summary_<dataset>_nvf16.json`
   files are rewritten after each mode completes, so check those directly rather than
   re-deriving from the jsonl by hand).

## Files involved

- `EXPERIMENT_LOG.md` — live status table + timing estimates, update as you go
- `RESTART.md` — this file, static, shouldn't need edits
- `EXPERIMENT_LOG_egoschema.log` / `EXPERIMENT_LOG_video_mme.log` — full stdout/stderr of each
  run (persistent, under the repo, not `/tmp`)
- `benchmark_results/nvila_hd_accuracy_breakdown_{autogaze,dense}_{egoschema,video_mme}_nvf16.jsonl`
  — the actual resumable per-question results
- `benchmark_results/nvila_hd_accuracy_breakdown_summary_{egoschema,video_mme}_nvf16.json`
  — per-dataset summary, rewritten after each mode finishes
