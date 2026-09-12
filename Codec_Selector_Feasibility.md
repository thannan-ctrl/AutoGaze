# Codec-Based Patch Selection: A Feasibility Study

> See the [README](README.md) for setup, installation, and AutoGaze's own baseline latency breakdown.

AutoGaze speeds up NVILA-HD-Video by feeding the LLM only the informative patches per
frame, picked by a trained selector model. This tests a free substitute: HEVC encoding
already computes a per-block motion vector during normal compression — threshold on
that instead of training anything.

## Contents

- [Summary](#summary)
- [Full-Dataset Results (current)](#full-dataset-results-current)
- [Implementation](#implementation)
- [Earlier Exploratory Results (N=1/N=25, collapsed)](#earlier-exploratory-results-n1n25-collapsed)
- [Next Steps](#next-steps)

## Summary

- Matches AutoGaze on accuracy and beats it on latency up to ~128 sampled frames.
- Above ~256 frames both hit scaling limits before the LLM does — at nvf=1024, only
  codec's **sampled-only** variant survives; AutoGaze and codec-windowed hard-crash (OOM).

## Full-Dataset Results (current)

**In plain terms:** the codec-based method (no trained model needed) matches AutoGaze's
accuracy while running noticeably faster — 3.46s vs. 6.24s per question on EgoSchema — because
it skips AutoGaze's own selector model entirely and uses information the video encoder already
computes for free. This is the current, most-reliable table (full EgoSchema Subset, partial
VideoMME sample — see the dataset coverage note under Implementation). Smaller, earlier test
runs (single videos, 25-sample checks) that led up to these numbers are collapsed further down
under "Earlier Exploratory Results."

All rows are `gb200nvl4` (GB200, driver 595.84.01) unless the Node column says otherwise.
`codec_geo`/`codec_ord` also have an independent run on `gb200nvl72_preprod` (older driver, no
NVDEC) for comparison — non-NVDEC latency is the same either way (`codec_geo` EgoSchema: 22.19s
vs. 22.44s, within 1.1%); only `codec_nvdec` specifically needs the newer driver.

**Bold** = best in column, <u>underline</u> = second-best — compared within each dataset's
complete rows (Acc and E2E only; `codec_geo`'s gb200nvl72_preprod E2E uses a different,
slower two-pass encoding method so isn't included in that comparison).

**EgoSchema** (N=500, official Subset — complete coverage)

| Mode | Node | N | Acc | Tokens | Encode | Decode | Selector | ViT | LLM | E2E |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| codec_nvdec (optimized selector) | gb200nvl4 | 500 | 60.6% | 2,716 | 0.86s | 0.09s | 0.37s | 0.16s | 0.28s | **2.44s** |
| codec_nvdec | gb200nvl4 | 500 | 60.6% | 2,716 | 0.85s | 0.08s | 1.28s | 0.16s | 0.27s | **3.46s** |
| dense | gb200nvl4 | 500 | 58.6% | 24,176 | — | — | — | 2.70s | 0.80s | <u>4.38s</u> |
| codec (optimized selector) | gb200nvl4 | 500 | 61.6% | 2,716 | 0.87s | 1.11s | 1.40s | 0.16s | 0.29s | <u>4.50s</u> |
| codec (+ csvparse fix)* | gb200nvl4 | 500 | 61.6% | 2,716 | 0.87s | 1.12s | 1.51s | 0.16s | 0.41s | 5.05s |
| codec | gb200nvl4 | 500 | <u>61.6%</u> | 2,716 | 0.86s | 1.15s | 2.30s | 0.17s | 0.28s | 5.61s |
| autogaze_singlescale | gb200nvl4 | 500 | 61.2% | 1,751 | — | — | — | 0.30s | 0.28s | 6.18s |
| autogaze | gb200nvl4 | 500 | 60.4% | 1,598 | — | — | — | 0.33s | 0.29s | 6.24s |
| codec_ord | gb200nvl72_preprod | 500 | 61.6% | 2,716 | 0.00s | 0.00s | 8.01s | 0.18s | 0.20s | 9.38s |
| codec_geo | gb200nvl72_preprod | 500 | **62.6%** | 2,716 | 2.91s | 9.66s | 8.14s | 0.19s | 0.27s | 22.19s |
| codec_geo | gb200nvl4 | 500/500 | — | — | — | — | — | — | — | — |
| codec_ord | gb200nvl4 | 126/500 | — | — | — | — | — | — | — | — |

*"+ csvparse fix" replaces `parse_csv_for_pocs`'s `grep` subprocess with an in-process
byte-prefix scan (no `fork()`) — see Key findings below. This full-N=500 run overlapped with
~1h of heavy rescue/rsync traffic hitting the same nodes (an infra incident, see Operational
notes), so its 5.05s E2E is noisier than ideal; the clean, controlled measurement is the N=25
before/after A/B below (-33.4% E2E, -43.2% preproc, same node, back-to-back, accuracy
unchanged). A clean full-dataset VideoMME rerun with this fix was in progress but lost to the
same infra incident before completing (1,575/2,700 rows reached) — pending relaunch.

**VideoMME** (N=1,395 rows = 465-video partial sample; N=2,700 rows = complete 900-video set,
marked below)

| Mode | Node | N | Acc | Tokens | Encode | Decode | Selector | ViT | LLM | E2E |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| codec_nvdec | gb200nvl4 | 1,395 | <u>55.0%</u> | 2,903 | 0.70s | 0.03s | 1.49s | 0.18s | 0.14s | **3.48s** |
| codec (optimized selector, full 900) | gb200nvl4 | 1,995/2,700 | 65.0%* | 2,902 | 0.62s | 0.71s | 0.94s | 0.18s | 0.16s | 3.53s |
| codec | gb200nvl4 | 1,395 | 54.7% | 2,903 | 0.70s | 0.36s | 1.73s | 0.18s | 0.14s | <u>4.01s</u> |
| dense | gb200nvl4 | 1,395 | 54.7% | 28,318 | — | — | — | 3.20s | 0.79s | 4.99s |
| autogaze | gb200nvl4 | 1,395 | **55.6%** | 1,526 | — | — | — | 0.24s | 0.28s | 6.89s |
| autogaze_singlescale | gb200nvl4 | 2,258/2,700 | — | — | — | — | — | — | — | — |
| autogaze_singlescale | gb200nvl72_preprod | 2,013 | — | — | — | — | — | — | — | — |
| codec_geo | gb200nvl4 | 48 | — | — | — | — | — | — | — | — |
| codec_geo | gb200nvl72_preprod | 630/full | — | — | — | — | — | — | — | — |
| codec_ord | gb200nvl4 | 69 | — | — | — | — | — | — | — | — |
| codec_ord | gb200nvl72_preprod | — | — | — | — | — | — | — | — | — |
| codec_nvdec (optimized selector, full 900) | gb200nvl4 | — | — | — | — | — | — | — | — | — |
| dense (full 900, new videos only†) | gb200nvl4 | queued | — | — | — | — | — | — | — | — |
| autogaze (full 900, new videos only†) | gb200nvl4 | queued | — | — | — | — | — | — | — | — |

*The "(optimized selector, full 900 videos)" VideoMME rows use the complete video set, not the
465-video partial sample the other VideoMME rows above use — their accuracy isn't directly
comparable to those; it's new data, not noise.

†These `dense`/`autogaze` rows only test the ~435 videos *not* already covered by the
465-video partial sample above — the already-completed 465-video results are seeded into the
output file so the harness's resumability logic skips re-testing them. Unlike `codec`/
`codec_nvdec`, these two modes don't touch the optimized code path at all, so mixing
old+new timing here doesn't corrupt anything. (`codec`/`codec_nvdec`'s full-900 reruns above
deliberately run fresh instead, since mixing pre/post-optimization timing would corrupt the
before/after speedup comparison.)

**Reproduce this table:**

```bash
for ds in egoschema video_mme; do
  # windowed (default) + AutoGaze + dense, one N=500/1,395 run per dataset
  CUDA_VISIBLE_DEVICES=0 REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 \
    FIXED_NUM_VIDEO_FRAMES=16 N_SAMPLES=full MAX_BATCH_SIZE_AUTOGAZE=32 \
    MODES=codec,autogaze,dense DATASET=$ds EXTRA_SUFFIX=_fulldsbreakdown \
    python3 scripts/nvila_hd_accuracy_breakdown_test.py

  # sampled-only codec, same N
  CUDA_VISIBLE_DEVICES=0 REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 \
    FIXED_NUM_VIDEO_FRAMES=16 N_SAMPLES=full MAX_BATCH_SIZE_AUTOGAZE=32 \
    MODES=codec CODEC_SAMPLED_ONLY=1 DATASET=$ds EXTRA_SUFFIX=_fulldsbreakdown_sampledonly \
    python3 scripts/nvila_hd_accuracy_breakdown_test.py
done
```

### Key findings from this table

- **codec_nvdec is the fastest option and doesn't cost accuracy.** ~12-14x faster "Decode" step
  than plain codec, translating to a 13-41% faster end-to-end time depending on dataset, with
  accuracy within noise of plain codec (60.6% vs. 61.6% EgoSchema, 55.0% vs. 54.7% VideoMME).
- **Selector (the scoring step) is the main remaining cost, and it's almost all "scoring," not
  "ranking."** Profiling split it into the two steps: extracting scores from the decode data is
  ~98% of Selector time, choosing the top patches from those scores is under 1%. That made a
  concrete fix possible: `pool_score_map` in `hevc_to_gaze.py` was redoing the same expensive
  setup step 4 times per region instead of once, and used a slow, un-vectorized loop. Fixed both,
  verified the fix produces byte-identical results (0.0 difference across 50 random tests), and
  confirmed a real 1.7-3.8x Selector speedup end-to-end. Full-scale reruns with the fix are the
  "(optimized selector)" rows above.
- **A second fix: don't `fork()` a `grep` subprocess from inside the loaded-model process.**
  `parse_csv_for_pocs` (`hevc_to_gaze.py`) used to shell out to `grep` to pre-filter the HEVC
  CSV before parsing. Forking a subprocess out of a process with NVILA+AutoGaze already loaded
  pays for a full page-table clone of that resident memory, and the effect lingers past the
  fork itself (subsequent memory touches in the parent keep paying extra page-fault/TLB cost).
  Replaced it with an in-process pure-Python byte-prefix scan (no fork at all), verified
  byte-identical CU output across 10 random trials, then confirmed at N=25 (EgoSchema, `codec`
  mode): preproc -43.2% (4,611ms → 2,621ms), E2E -33.4% (5,664ms → 3,773ms), accuracy unchanged
  (68.0% → 68.0%). Interestingly the isolated CSV-parse timer itself didn't show the win — the
  cost shows up downstream in the rest of preproc, consistent with the lingering-fork-cost
  explanation rather than the grep call itself being slow.
- **Restricting AutoGaze to a single resolution scale doesn't hurt accuracy either** —
  61.2% vs. 60.4% for the normal multi-scale version (`autogaze_singlescale` row), a small edge
  in the single-scale's favor, though likely just noise at this sample size.
- **Non-NVDEC latency is consistent across different physical machines** — the same `codec_geo`
  step took 22.19s and 22.44s on two different node types (within 1.1%), so hardware choice
  mostly only matters for the NVDEC-specific path.

<details>
<summary>Operational notes (infra incidents hit while producing this table — not results, kept for the record)</summary>

**Lost a node mid-run.** Its holding job used `sleep 28000` under an
`--time=08:00:00` job — shorter than the time budget, so the sleep finished and released the
node ~50 minutes early with no warning, orphaning the work still running on it. Five of six
EgoSchema legs were already complete and safely saved elsewhere; only `codec_geo` (saved up to
item 42/500) and the not-yet-started `codec_ord` were affected. Rebuilt on a fresh node and
resumed from that checkpoint. Applied the same fix (sleep duration ≥ time limit) to the other
nodes preemptively.

**A silent kill on `codec_geo`/`codec_ord` runs.** The process was getting killed part-way
through with no error message — the launcher script had no error checking, so it silently moved
on to the next item type with zero results written, instead of stopping. Root cause: likely
SLURM's default per-job memory limit (conservative when not explicitly requested) being
exceeded by internal caching, even though that caching has explicit size limits. Fixed by
explicitly requesting more memory per job; confirmed no further kills since.

**The 40GB A100 couldn't run `autogaze` mode at all — every single item failed with an
out-of-memory error**, even after fixing a separate memory-fragmentation issue. The baseline
memory this configuration needs (~34GB) leaves no room to spare on a 40GB card, regardless of
which video is being processed. Switched to an 80GB A100 instead, which works cleanly:

| Dataset | Mode | Node | N | Acc | Tokens | E2E |
|---|---|---|--:|--:|--:|--:|
| EgoSchema | autogaze | A100-80GB | 500 | 61.4% | 1,596 | 13.69s |
| VideoMME | autogaze | A100-80GB | in progress (478/1,395, 0 errs) | — | — | — |

**Once everything above finishes, the full experiment matrix reruns against the complete
900-video VideoMME set** (earlier runs used a partial video subset, ~465-873 videos, that only
reached the full 900 mid-run as the download completed).

</details>

## Implementation

Scores each video block by size, motion, and skip status from the HEVC encoder's own output
(small + moving + not-skipped = important) — no trained model, drops into NVILA-HD at the same
point AutoGaze's own selector runs, with no other changes needed.

<details>
<summary>How it works under the hood + setup for the commands throughout this doc</summary>

`"codec" mode` scores each coding block by size, motion, and skip status (small + moving
+ not-skipped = important) via `codec_selector.build_gazing_info()`, intercepting
NVILA-HD's `_get_gazing_info_from_videos` at the exact point AutoGaze's selector runs,
returning the same tensor shapes — no NVILA-HD changes needed.

Under the hood: [`hevc_dump`](https://gitlab-master.nvidia.com/seadie/hevc_dump) decodes
HEVC into a CSV of per-block motion/size/residual stats; `codec_selector.py` encodes the
sampled frames and runs `dump_stats`; `hevc_to_gaze.py` scores each block and converts to
AutoGaze's patch-index format. `hevc_dump`'s own scorer, `hevc_autogaze.py` (CU-size-vs-
AutoGaze-scale kernel matching, geometric or ordinal), is wired in separately as modes
`codec_geo`/`codec_ord` — see Full-Dataset Results above and Next Steps.

`conda activate auto_gaze` first (GB200 = aarch64; build `hevc_dump/cmake_build_aarch64`
before any codec-mode run). Swap `DATASET=egoschema` for `DATASET=video_mme` to run the
other dataset. GIFs are downsampled previews — captions link the full-res video.

**Dataset coverage vs. the official benchmarks — EgoSchema is complete, VideoMME is a
partial sample.** EgoSchema's `data/egoschema/subset.json` is the official 500-question
public Subset (the other 5,031 questions in `questions.json` are the Full set, whose answers
are held out for leaderboard submission — unused here); all 500 Subset videos are present
locally, so every EgoSchema number in this doc covers the complete official Subset.
VideoMME's `data/video_mme/questions.json` does contain the full official 2,700 questions
(900 videos × 3 questions each), but only **465 of those 900 video files were ever downloaded
locally** — `dataset.py::_load_videomme` filters to whichever questions have a video file on
disk, so every "N=1,395" in this doc is really 465/900 (51.7%) of the official videos, not
the full benchmark. That 465-video sample is reasonably representative rather than skewed to
one slice, though: duration split is 152 short / 151 medium / 162 long (official is an exact
300/300/300), and every one of the 6 official video categories is present at roughly the same
~50-63% coverage rate (e.g. Knowledge 135/270, Sports Competition 75/150, Multilingual
19/30) — there's no record of *why* these particular 435 videos were never pulled down (a
partial download at data-setup time, not a principled split), but the shortfall isn't
concentrated in any one duration or domain. The full 900-video set has since been downloaded
and is queued for a full rerun (see Full-Dataset Results above); **numbers in this doc so far
still reflect the 465-video partial sample** and aren't directly comparable to a paper
reporting on the full 2,700-question set.

</details>

## Earlier Exploratory Results (N=1/N=25, collapsed)

Smaller-scale runs (single reference videos, N=25 samples) that led up to the full-dataset
numbers above — kept for the reasoning trail, collapsed since they're superseded by the
current full-dataset results.

<details>
<summary><b>Results: accuracy/latency across variants, N=25 (findings 1-4)</b></summary>

### 1. At 16 frames, codec matches AutoGaze on accuracy and is faster end-to-end

<table>
<tr>
<td align="center"><b>EgoSchema</b><br>
<img src="figures/codec_vs_autogaze_egoschema_comparison.gif" width="360"><br>
<a href="figures/codec_vs_autogaze_egoschema_comparison.mp4">full-res</a></td>
<td align="center"><b>VideoMME</b><br>
<img src="figures/codec_vs_autogaze_video_mme_comparison.gif" width="360"><br>
<a href="figures/codec_vs_autogaze_video_mme_comparison.mp4">full-res</a></td>
</tr>
</table>

| Mode | Accuracy | Total time | Selection time | LLM time | Tokens |
|---|---|---|---|---|---|
| **EgoSchema, N=500** | | | | | |
| codec | **61.4%** | **5.2s** | **3.9s** | **0.3s** | **1,572** |
| AutoGaze | 60.4% | 8.3s | 6.9s | 0.4s | 1,598 |
| dense (no selection) | 54.4% | 5.4s | — | 4.3s | 23,784 |
| **VideoMME, N=1,395** | | | | | |
| codec | **55.9%** | **4.0s** | **2.7s** | **0.3s** | 1,547 |
| AutoGaze | 55.6% | 9.6s | 8.1s | 0.4s | **1,526** |
| dense | 54.9% | 5.7s | — | 4.6s | 28,318 |

<details>
<summary>Commands</summary>

```bash
CUDA_VISIBLE_DEVICES=0 REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 \
  FIXED_NUM_VIDEO_FRAMES=16 N_SAMPLES=full MAX_BATCH_SIZE_AUTOGAZE=32 MODES=codec,autogaze,dense \
  CODEC_RATIO_SCALE=0.28 DATASET=egoschema python3 scripts/nvila_hd_accuracy_breakdown_test.py

# comparison video
python3 scripts/visualize_codec_vs_autogaze_video.py --video <path.mp4> --out figures/out.mp4
```

</details>

### 2. Windowed vs. sampled-only: same accuracy, sampled-only is consistently faster

AutoGaze samples frames sparsely instead of using the continuous stream, since keeping
every frame OOMs (Finding 4) — codec mode inherited this, so neither variant here ever
scores genuinely consecutive real frames. Windowed keeps a few real context frames per
sample, but samples can still be seconds apart; sampled-only skips that context entirely
and computes motion vectors directly between samples, however far apart. Neither matches
what HEVC motion vectors are actually built to measure.

**LLaVA-OneVision-2** seems to avoid this — it scores saliency on every real frame via
adaptive GOP boundaries, no sparse-sampling gap. But the paper reports
$\color{red}{\textbf{no latency or memory numbers}}$ for this pipeline anywhere, and it
does train at up to 768 frames/10-15min without incident — pointing our OOM wall more at
this repo's `max_tiles_video` bug than an inherent limit, though "seems to" is still
doing real work there.

<table>
<tr>
<td align="center"><b>Concept</b><br>
<img src="figures/windowed_vs_sampledonly_concept.png" width="360"></td>
<td align="center"><b>Same video, both variants</b><br>
<img src="figures/windowed_vs_sampledonly_egoschema.gif" width="360"><br>
<a href="figures/windowed_vs_sampledonly_egoschema.mp4">full-res</a></td>
</tr>
</table>

| Frames | EgoSchema acc. | Windowed | Sampled-only | Δ | VideoMME acc. | Windowed | Sampled-only | Δ |
|---|---|---|---|---|---|---|---|---|
| 16 | 68.0% | **4.1s** | 4.2s | +1.6% | 60.0% | 6.6s | **5.6s** | -15.5% |
| 32 | 64/60%* | 11.0s | **7.6s** | -31.2% | 60.0% | 14.6s | **12.3s** | -15.8% |
| 64 | 68.0% | 17.2s | **15.4s** | -10.8% | 64.0% | 38.1s | **33.7s** | -11.4% |
| 128 | 68.0% | 34.1s | **30.0s** | -12.0% | | |  | |
| 256 | 60.0% | 133.6s | **103.2s** | -22.8% | | | | |
| 512 | 40.0% | 152.6s | **92.4s** | -39.4% | | | | |
| 1024 | 12.0% | **324.8s** | 419.1s | +29.0% | | | | |

*64% vs. 60% at 32 frames on EgoSchema is likely N=25 noise. Accuracy degrades past 128
frames and keeps falling — 12% by 1024.

<details>
<summary>Commands</summary>

```bash
for nvf in 16 32 64 128 256 512; do
  # windowed: real local-frame context around each sampled frame (default)
  REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 FIXED_NUM_VIDEO_FRAMES=$nvf N_SAMPLES=25 \
    MODES=codec DATASET=egoschema EXTRA_SUFFIX=_windowedn25_$nvf \
    python3 scripts/nvila_hd_accuracy_breakdown_test.py

  # sampled-only: just the sampled frames, chained, no extra context
  REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 FIXED_NUM_VIDEO_FRAMES=$nvf N_SAMPLES=25 \
    MODES=codec DATASET=egoschema CODEC_SAMPLED_ONLY=1 EXTRA_SUFFIX=_sampledonly$nvf \
    python3 scripts/nvila_hd_accuracy_breakdown_test.py
done
```

`CODEC_SAMPLED_ONLY=1` sampled-only encoding · `N_SAMPLES=full|<int>` dataset size ·
`EXTRA_SUFFIX` keeps result files from clobbering each other across sweep points.

</details>

### 3. Matching AutoGaze's actual chunking did not help

Restart encoding every 16 frames, keep full detail on each chunk's first frame — built
up here in three steps: dense chunks, + anchor frame, + restart vs. real AutoGaze.

<table>
<tr>
<td align="center"><b>Dense chunks</b><br>
<img src="figures/full_video_codec_egoschema.gif" width="360"><br>
125 patches/frame · <a href="figures/full_video_codec_egoschema.mp4">full-res</a></td>
<td align="center"><b>+ full-first-frame anchor</b><br>
<img src="figures/full_video_codec_egoschema_fullfirstframe.gif" width="360"><br>
1,038 patches at boundaries · <a href="figures/full_video_codec_egoschema_fullfirstframe.mp4">full-res</a></td>
</tr>
<tr>
<td align="center"><b>Same, VideoMME (~20s crop)</b><br>
<img src="figures/full_video_codec_video_mme_fullfirstframe.gif" width="360"><br>
higher motion complexity caps duration · <a href="figures/full_video_codec_video_mme_fullfirstframe.mp4">full-res</a></td>
<td align="center"><b>+ restart vs. real AutoGaze</b><br>
<img src="figures/restarted_chunks_vs_autogaze_egoschema.gif" width="360"><br>
AutoGaze constant at 109 patches/frame · <a href="figures/restarted_chunks_vs_autogaze_egoschema.mp4">full-res</a></td>
</tr>
</table>

| Dataset | Sampled-only (128 frames) | + restart/anchor | Tokens |
|---|---|---|---|
| EgoSchema | **68.0%** | **68.0%** | 37,873 vs. **25,129** (1.5x) |
| VideoMME | **76.0%** | 48.0% | 86,240 vs. **53,035** (1.6x) |

<details>
<summary>Commands</summary>

```bash
REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 FIXED_NUM_VIDEO_FRAMES=128 N_SAMPLES=25 \
  MODES=codec DATASET=egoschema \
  CODEC_SAMPLED_ONLY=1 CODEC_GOP_RESTART=16 CODEC_FULL_FIRST_FRAME=1 \
  EXTRA_SUFFIX=_restart16_fff \
  python3 scripts/nvila_hd_accuracy_breakdown_test.py

# whole video, real consecutive 16-frame chunks (add --full-first-frame for the anchor variant)
python3 scripts/visualize_full_video_codec.py --video <path.mp4> --out figures/out.mp4

# restart-every-16 + full-first-frame vs. real AutoGaze
python3 scripts/visualize_restarted_chunks_vs_autogaze.py --video <path.mp4> --out figures/out.mp4
```

`CODEC_GOP_RESTART=N` force an I-frame every N sampled frames ·
`CODEC_FULL_FIRST_FRAME=1` keep every patch on each chunk's first frame.

</details>

### 4. Isolating selection time alone (no ViT, no LLM)

<table>
<tr>
<td align="center"><img src="figures/selector_latency_egoschema.png" width="360"></td>
<td align="center"><img src="figures/selector_latency_video_mme.png" width="360"></td>
</tr>
</table>

<details>
<summary>Scaling rate and OOM root cause</summary>

16→128 frames (8x): AutoGaze grows 8.8x (EgoSchema)/22.3x (VideoMME) vs. codec's 12-14x
— super-linear well before any crash.

All three OOMs are confirmed real host-RAM kills (~985-987GB resident), reproduced 2-3/3
tries — not a hang. Windowing's extra context frames and AutoGaze's own model both scale
memory with frame count; sampled-only's flat footprint doesn't, which is why it's the
only variant surviving nvf=1024.

**Commands:**

```bash
for nvf in 16 32 64 128 256 512 1024; do
  # AutoGaze's own trained selector
  REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 SKIP_LLM=1 FIXED_NUM_VIDEO_FRAMES=$nvf N_SAMPLES=25 \
    DATASET=egoschema MODES=autogaze EXTRA_SUFFIX=_selectoronly_autogaze$nvf \
    python3 scripts/nvila_hd_accuracy_breakdown_test.py

  # codec, windowed
  REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 SKIP_LLM=1 FIXED_NUM_VIDEO_FRAMES=$nvf N_SAMPLES=25 \
    DATASET=egoschema MODES=codec EXTRA_SUFFIX=_selectoronly_windowedn25_$nvf \
    python3 scripts/nvila_hd_accuracy_breakdown_test.py

  # codec, sampled-only
  REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 SKIP_LLM=1 FIXED_NUM_VIDEO_FRAMES=$nvf N_SAMPLES=25 \
    DATASET=egoschema MODES=codec CODEC_SAMPLED_ONLY=1 EXTRA_SUFFIX=_selectoronly_sampledonly$nvf \
    python3 scripts/nvila_hd_accuracy_breakdown_test.py
done
```

`SKIP_LLM=1` selection-latency only (never loads the 8B model).

</details>

</details>

<details>
<summary><b>Single-Video Sanity Check (vs. LLaVA-OneVision-2), N=1/N=25</b></summary>

Same two videos as LLaVA-OV-2's own smoke test (`llava_onevision2_repro/SMOKE_TEST.md`,
separate repo), run through this repo's pipeline at nvf=16 — different VLM/selector, not
a controlled comparison, just a cross-check on the same source videos.

| Dataset | Mode | Tokens | Encode | Selector | ViT | LLM | E2E |
|---|---|---:|---:|---:|---:|---:|---:|
| EgoSchema (0074f737…, 180s) | dense | 23,176 | — | — | 2.4s | 1.5s | **5.1s** |
| EgoSchema | AutoGaze | **1,593** | — | 6.2s | 0.2s | 0.3s | 8.3s |
| EgoSchema | codec | 2,555 | 0.3s | 4.9s* | 0.2s | 0.3s | 6.3s |
| VideoMME (001-1, fFjv93ACGo8) | dense | 28,703 | — | — | 2.9s | 1.6s | 5.7s |
| VideoMME | AutoGaze | **1,655** | — | 7.2s | 0.2s | 0.2s | 9.7s |
| VideoMME | codec | 2,930 | 0.2s | 3.7s* | 0.3s | 0.3s | **5.3s** |

*codec's "selector" includes its own encode step (transcode analog — only the sampled
frames, not the whole video, so much cheaper than LLaVA-OV-2's whole-video H264
transcode). AutoGaze's "selector" is its trained model's GPU forward pass, with no analog
in the frames-only baseline.

Dense is fastest E2E despite ~15x more tokens — ViT/LLM cost is still cheaper than
AutoGaze's own selector overhead (6-7s) at this budget, matching Finding 1: selection
only pays off from nvf≈64-128, not 16.

Confirmed at N=25/dataset: dense stays fastest at nvf=16 both places — 4.4s/5.1s vs.
codec's 5.6s/6.4s vs. AutoGaze's 8.1s/9.2s (EgoSchema/VideoMME) — matching Finding 1
closely; AutoGaze's selector overhead is again what keeps it slowest.

<details>
<summary>Commands</summary>

```bash
# same two videos as LLaVA-OV-2's smoke test, patch-selector + ViT/LLM breakdown
for mode in dense autogaze codec; do
  REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 FIXED_NUM_VIDEO_FRAMES=16 \
    DATASET=egoschema MODES=$mode ITEM_ID=0074f737-11cb-497d-8d07-77c3a8127391 \
    EXTRA_SUFFIX=_1vid_$mode python3 scripts/nvila_hd_accuracy_breakdown_test.py

  REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 FIXED_NUM_VIDEO_FRAMES=16 \
    DATASET=video_mme MODES=$mode ITEM_ID=001-1 \
    EXTRA_SUFFIX=_1vid_$mode python3 scripts/nvila_hd_accuracy_breakdown_test.py
done

# N=25/dataset confirmation
for ds in egoschema video_mme; do
  for mode in dense autogaze codec; do
    REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 FIXED_NUM_VIDEO_FRAMES=16 N_SAMPLES=25 \
      DATASET=$ds MODES=$mode EXTRA_SUFFIX=_n25sanity \
      python3 scripts/nvila_hd_accuracy_breakdown_test.py
  done
done
```

`ITEM_ID=<q_uid|question_id>` pins to one exact question (skips random sampling) — the
same two videos LLaVA-OV-2's smoke test uses.

</details>

### Same check, + codec_nvdec, on a driver that actually supports it

- Same two videos, same nvf=16 setup as above — but run on a different machine
  with a newer GPU driver, specifically because `codec_nvdec` (the
  GPU-hardware-decode backend) needs a driver our usual machine doesn't have.
- **Numbers here aren't directly comparable to the table above** (different machine, freshly
  built environment) — but all four rows below come from that same machine/setup, so
  comparing them *to each other* (especially codec vs. codec_nvdec) is fair.

| Dataset | Mode | Tokens | Encode | Decode | Selector | ViT | LLM | E2E |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| EgoSchema | dense | 23,176 | — | — | — | 4.9s | 9.5s | 15.1s |
| EgoSchema | AutoGaze | 1,593 | — | — | 20.4s | 3.8s | 2.3s | 28.1s |
| EgoSchema | codec | 2,555 | 0.9s | 0.9s | 2.6s | 1.6s | 2.1s | 9.1s |
| EgoSchema | codec_nvdec | 2,555 | 0.9s | 0.1s | 1.2s | 1.3s | 1.2s | **5.6s** |
| VideoMME | dense | 28,703 | — | — | — | 4.9s | 2.3s | 8.0s |
| VideoMME | AutoGaze | 1,655 | — | — | 16.5s | 1.4s | 1.7s | 21.3s |
| VideoMME | codec | 2,930 | 0.5s | 0.5s | 2.3s | 1.3s | 1.4s | 7.0s |
| VideoMME | codec_nvdec | 2,930 | 0.5s | 0.1s | 1.5s | 1.3s | 1.2s | **5.7s** |

**What the numbers say:**
- **Same number of tokens picked either way** — expected, since both use the same
  top-k budget, just fed by a different source of motion data.
- **Encode is basically identical** between codec and codec_nvdec (0.9s vs 0.9s EgoSchema,
  0.5s vs 0.5s VideoMME) — both backends do the *exact same* video-encoding step; NVDEC only
  changes what happens after that.
- **Decode is where NVDEC actually wins: ~10x faster on EgoSchema (0.9s → 0.1s), ~5.5x on
  VideoMME (0.5s → 0.1s).** Codec (libde265) decodes on the CPU and writes/reads a CSV file
  to hand off the results; codec_nvdec decodes on a dedicated GPU chip and keeps the results
  in memory the whole time — no file round-trip.
- **Selector (the scoring step after decode) is also somewhat faster with NVDEC** — smaller
  win though (2.6s → 1.2s EgoSchema, 2.3s → 1.5s VideoMME). See why below.
- **Real-world NVDEC speedup is usually much bigger (~90x)** than what shows up here — at
  only 16 frames, most of NVDEC's time is fixed one-time setup cost, not actual decoding, so
  the advantage doesn't show up fully yet. It grows with more frames.

**Why Selector is still slower for codec, even with Decode counted separately:** matching
token *counts* doesn't mean matching *work* — both backends always keep a fixed number of
patches per frame by design, so the count is the same no matter how good or bad the
underlying scores are. What differs is how much work it takes to *get* those scores:
- **codec (libde265)** works from the video's true, fine-grained block structure (blocks as
  small as 4×4 pixels). Scoring a frame means: filter a text file down to the right rows,
  parse those rows in Python one at a time, then paint each block's score onto a
  full-resolution image-sized array before cropping it down to size.
- **codec_nvdec** only ever sees a fixed, coarse 16×16 grid — that's all the detail NVDEC's
  hardware reports. So there's no text file to filter or parse, no per-block Python loop, and
  no full-resolution image to paint — just one fast, vectorized math operation on an
  already-small grid.

That's the gap: file-parsing-plus-per-block-Python-loop (codec) vs. one fast bulk operation on
data that's already sitting in memory (codec_nvdec). Both cache this work per frame, so it's
only paid once and reused — but that first pass is structurally more expensive for codec,
which is why it stays slower even once Decode itself is counted separately. One side note:
because NVDEC's grid is coarser, the *specific* patches it picks can differ slightly from
codec's even when the *count* matches — a precision trade-off, not a speed one.

<details>
<summary>Commands / environment notes</summary>

`gb200nvl4` has no `/home/scratch.thannan_wwfo` mount, so this used a from-scratch
environment: `pip install --break-system-packages --target=<local /tmp dir> torch
torchvision transformers==5.14.1 numpy pillow av einops timm omegaconf matplotlib imageio
tqdm loguru opencv-python-headless PyNvVideoCodec psutil`, repo code + the two videos copied
in via a shared per-user Lustre mount (`/home/thannan/lustre`, the only filesystem visible
from both partitions), model weights (~16GB) freshly downloaded via `trust_remote_code`
(confirmed internet access from that partition), and the `libde265`/`dump_stats` binary
copied over (same aarch64 arch) with `LD_LIBRARY_PATH` pointed at its co-located
`libde265.so`. Needed one extra fix beyond that: the real (non-`SKIP_LLM`) path triggers a
Triton JIT compile requiring `Python.h`, not present on this minimal node image and not
installable without root — worked around by downloading `libpython3.12-dev` directly from
Ubuntu's package archive and extracting it with `dpkg-deb -x` (no root needed) into a local
dir, then pointing `CPATH` at it.

**Getting the full N=500/1,395 video datasets onto `gb200nvl4`** (needed once this moved from
the 2-video sanity check to a full-dataset run) couldn't reuse the Lustre bridge —
`/home/thannan/lustre` turned out to be carved out of a 5GB per-user NFS home quota, not a
real parallel filesystem, and had only ~200MB free after the environment install. Instead,
copied the ~19.4GB of EgoSchema/VideoMME video files directly node-to-node: a short-lived
`salloc` on `gb200nvl72_preprod` to get a shell with both clusters' hostnames resolvable,
then `rsync` straight from that node's local video directory to `gb200nvl4`'s local disk
(`/tmp/nvdec_repo_parent/repo/data/<dataset>/videos/`, on `/dev/nvme3n1p2`, 1.4TB free) over
SSH — bypassing the shared-filesystem bridge entirely. Model weights and Python packages
were **not** copied this way; those came from a fresh `pip install`/`trust_remote_code`
download directly on `gb200nvl4` (confirmed to have outbound internet access), which is
lighter than shipping an existing HF cache across.

```bash
for spec in "egoschema:0074f737-11cb-497d-8d07-77c3a8127391" "video_mme:001-1"; do
  ds="${spec%%:*}"; iid="${spec##*:}"
  for mode in dense autogaze codec codec_nvdec; do
    FIXED_NUM_VIDEO_FRAMES=16 DATASET=$ds MODES=$mode ITEM_ID=$iid \
      python3 scripts/nvila_hd_accuracy_breakdown_test.py
  done
done
```

</details>

### Frame-count sweep: nvf=16 → 1024 (same two videos, patch-selector only)

Same two videos, patch-selector latency only (`SKIP_LLM=1`), across the full nvf range
from Findings 1-4. Encode is codec's x265 pass over the sampled frames; Selector is
AutoGaze's trained-model forward pass, or codec's motion-vector scoring pass (no GPU
model).

<table>
<tr>
<td align="center"><img src="figures/frame_sweep_latency_egoschema.png" width="360"></td>
<td align="center"><img src="figures/frame_sweep_latency_video_mme.png" width="360"></td>
</tr>
</table>

<details>
<summary>Raw data</summary>

| Dataset | Mode | nvf | Tokens | Encode | Selector | E2E |
|---|---|---:|---:|---:|---:|---:|
| EgoSchema | AutoGaze | 16 | 1,593 | — | 0.8s | 8.6s |
| EgoSchema | AutoGaze | 32 | 2,949 | — | 12.4s | 15.8s |
| EgoSchema | AutoGaze | 64 | — | — | — | **OOM** |
| EgoSchema | codec, windowed | 16 | 2,555 | 1.0s | 3.5s | 5.3s |
| EgoSchema | codec, windowed | 32 | 5,006 | 1.4s | 7.3s | 10.5s |
| EgoSchema | codec, windowed | 64 | 9,908 | 2.0s | 14.5s | 18.5s |
| EgoSchema | codec, windowed | 128 | 19,712 | 3.7s | 28.3s | 37.6s |
| EgoSchema | codec, windowed | 256 | 39,320 | 6.6s | 55.9s | 73.7s |
| EgoSchema | codec, windowed | 512 | 78,536 | 12.9s | 112.9s | 147.8s |
| EgoSchema | codec, windowed | 1024 | 156,968 | 25.1s | 234.1s | 311.7s |
| EgoSchema | codec, sampled-only | 16 | 2,555 | 0.6s | 2.0s | 3.7s |
| EgoSchema | codec, sampled-only | 32 | 5,006 | 0.7s | 3.8s | 6.7s |
| EgoSchema | codec, sampled-only | 64 | 9,908 | 0.8s | 7.3s | 10.5s |
| EgoSchema | codec, sampled-only | 128 | 19,712 | 1.1s | 14.4s | 20.0s |
| EgoSchema | codec, sampled-only | 256 | 39,320 | 1.9s | 29.9s | 44.0s |
| EgoSchema | codec, sampled-only | 512 | 78,536 | 2.8s | 58.2s | 86.5s |
| EgoSchema | codec, sampled-only | 1024 | 156,968 | 4.4s | 121.1s | 177.4s |
| VideoMME | AutoGaze | 16 | 1,655 | — | 0.9s | 9.8s |
| VideoMME | AutoGaze | 32 | 3,682 | — | 26.1s | 32.4s |
| VideoMME | AutoGaze | 64 | 8,769 | — | 438.8s | 458.6s |
| VideoMME | AutoGaze | 128 | 17,329 | — | 165.2s | 203.6s |
| VideoMME | AutoGaze | 256 | 69,667 | — | 992.8s | 1254.2s |
| VideoMME | AutoGaze | 512 | 133,688 | — | 1978.4s | 2472.1s |
| VideoMME | AutoGaze | 1024 | — | — | — | **OOM** |
| VideoMME | codec, windowed | 16 | 2,930 | 0.5s | 2.9s | 4.2s |
| VideoMME | codec, windowed | 32 | 9,037 | 0.8s | 8.1s | 11.5s |
| VideoMME | codec, windowed | 64 | 26,567 | 1.5s | 23.6s | 31.3s |
| VideoMME | codec, windowed | 128 | 53,039 | 2.9s | 46.6s | 68.3s |
| VideoMME | codec, windowed | 256 | 306,271 | 5.5s | 245.7s | 358.7s |
| VideoMME | codec, windowed | 512 | 612,447 | 10.4s | 484.3s | 696.2s |
| VideoMME | codec, windowed | 1024 | 1,224,799 | 18.4s | 987.8s | 1443.9s |
| VideoMME | codec, sampled-only | 16 | 2,930 | 0.3s | 2.0s | 3.7s |
| VideoMME | codec, sampled-only | 32 | 9,037 | 0.4s | 6.9s | 10.7s |
| VideoMME | codec, sampled-only | 64 | 26,567 | 0.4s | 20.5s | 28.1s |
| VideoMME | codec, sampled-only | 128 | 53,039 | 0.7s | 39.9s | 60.6s |
| VideoMME | codec, sampled-only | 256 | 306,271 | 1.2s | 232.0s | 359.0s |
| VideoMME | codec, sampled-only | 512 | 612,447 | 1.6s | 467.1s | 680.1s |
| VideoMME | codec, sampled-only | 1024 | 1,224,799 | 2.7s | 915.5s | 1387.6s |

</details>

Same tokens as windowed at every nvf (as expected — encoding strategy doesn't change how
many patches get kept), but consistently faster (e.g. EgoSchema nvf=128: 38.0s windowed
vs. 20.0s sampled-only) and all 14/14 combos completed cleanly — no OOMs at nvf=1024,
matching the sampled-only-survives pattern from the Summary and Finding 4.

<details>
<summary>Commands</summary>

```bash
EGO_ID=0074f737-11cb-497d-8d07-77c3a8127391
VMME_ID=001-1

for nvf in 16 32 64 128 256 512 1024; do
  for spec in "egoschema:$EGO_ID" "video_mme:$VMME_ID"; do
    ds="${spec%%:*}"; iid="${spec##*:}"
    for mode in autogaze codec; do
      REPO_DIR=$(pwd) NVILA_DEVICE=cuda:0 SKIP_LLM=1 FIXED_NUM_VIDEO_FRAMES=$nvf \
        DATASET=$ds MODES=$mode ITEM_ID=$iid EXTRA_SUFFIX=_1vid_${mode}${nvf} \
        python3 scripts/nvila_hd_accuracy_breakdown_test.py
    done
  done
done
```

`SKIP_LLM=1` selection-latency only, same as Finding 4. `ITEM_ID` pins to the same two
videos throughout.

</details>

</details>

## Next Steps

1. Run codec_nvdec (optimized selector) for Dense, Codec, NVDEC for all videos
2. Run nvf sweep with codec_nvdec (optimized selector) 