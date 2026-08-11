#!/usr/bin/env python3
"""
End-to-end runtime analysis: dense vs EVS vs sparse_vit.

Runs three modes sequentially in NVIDIA vLLM Docker containers,
with REPS inference passes each (first is warmup, rest are measured).

Timing breakdown per mode:
  load_ms        — vLLM model load (inside Docker)
  vit_ms         — Visual encoder forward (CUDA events, includes gather op for sparse_vit)
  lm_ms          — LLM prefill + decode  (elapsed_ms - vit_ms)
  elapsed_ms     — total inference (vit + lm, excludes model load)

Results are printed as a formatted table and saved to runtime_analysis.json.

Usage:
    python scripts/runtime_analysis.py [--reps N] [--pruning-rate R]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE = os.environ.get("HF_HOME", "/home/scratch.thannan_wwfo/hf_cache")
RESULTS_FILE = os.path.join(REPO_DIR, "runtime_analysis.json")

VLLM_IMAGE = "nvcr.io/nvidia/vllm:26.07-py3"


def run_docker(mode: str, pruning_rate: float, reps: int,
               gazing_ratio: float = 0.245,
               max_frames: int = 32, fps: float = 2.0,
               video_path: str | None = None) -> dict:
    """
    Run one mode in a Docker container.
    Returns the parsed RESULT_JSON dict augmented with wall_ms.
    """
    # Resolve video path: default to example_input.mp4, allow override
    docker_video = video_path or "/workspace/AutoGaze/assets/example_input.mp4"

    env_vars = [
        "-e", f"REPO_DIR=/workspace/AutoGaze",
        "-e", f"HF_HOME=/root/.cache/huggingface",
        "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "-e", f"VIDEO_PATH={docker_video}",
    ]
    vol_mounts = [
        "-v", f"{HF_CACHE}:/root/.cache/huggingface",
        "-v", f"{REPO_DIR}:/workspace/AutoGaze",
    ]

    worker_args = [
        "--mode", mode,
        "--pruning-rate", str(pruning_rate),
        "--reps", str(reps),
        "--max-frames", str(max_frames),
        "--fps", str(fps),
    ]

    if mode == "sparse_vit":
        worker_args += [
            "--video", docker_video,
            "--gazing-ratio", str(gazing_ratio),
        ]

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--shm-size", "16g",
        *vol_mounts,
        *env_vars,
        VLLM_IMAGE,
        "python", "/workspace/AutoGaze/scripts/worker.py",
        *worker_args,
    ]

    print(f"\n--- Docker: mode={mode} reps={reps} ---", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    wall_ms = (time.perf_counter() - t0) * 1000

    full = proc.stdout + "\n" + proc.stderr
    for line in full.splitlines():
        if "RESULT_JSON" not in line and "VIT_TIMING_MS" not in line:
            print(f"  {line}")

    # Collect VIT_TIMING_MS lines emitted by _hook_forward inside the
    # EngineCore subprocess.  The subprocess inherits the container's stdout
    # (shared file descriptor after fork), so these lines appear in the
    # captured Docker output in inference-order.
    vit_times = []
    for line in full.splitlines():
        s = line.strip()
        if s.startswith("VIT_TIMING_MS:"):
            try:
                vit_times.append(float(s[len("VIT_TIMING_MS:"):]))
            except ValueError:
                pass

    for line in full.splitlines():
        if line.startswith("RESULT_JSON:"):
            r = json.loads(line[len("RESULT_JSON:"):])
            r["wall_ms"] = wall_ms

            # Inject per-rep ViT timing into the result.
            # vit_times[i] corresponds to rep i (the i-th llm.chat() call).
            # With the vLLM encoder cache, reps 1+ may bypass the ViT entirely,
            # so vit_times may have fewer entries than total reps.
            if vit_times:
                r["rep_vit_ms"] = vit_times
                # Assign to individual rep records
                for i, rep in enumerate(r.get("reps", [])):
                    if i < len(vit_times):
                        rep["vit_ms"] = vit_times[i]
                        if rep.get("elapsed_ms") and vit_times[i]:
                            rep["lm_ms"] = rep["elapsed_ms"] - vit_times[i]

                # Primary (rep 0) timings
                r["vit_ms"] = vit_times[0]
                if r.get("elapsed_ms") and vit_times[0]:
                    r["lm_ms"] = r["elapsed_ms"] - vit_times[0]

                # Averages over measured reps (excluding warmup rep 0)
                measured_vit = vit_times[1:] if len(vit_times) > 1 else []
                if measured_vit:
                    r["avg_vit_ms"] = sum(measured_vit) / len(measured_vit)
                    if r.get("avg_elapsed_ms") and r["avg_vit_ms"]:
                        r["avg_lm_ms"] = r["avg_elapsed_ms"] - r["avg_vit_ms"]

            return r

    raise RuntimeError(
        f"No RESULT_JSON in output for mode={mode}\n"
        f"returncode={proc.returncode}\n"
        f"stderr (last 2000 chars):\n{proc.stderr[-2000:]}"
    )


def _fmt(val, unit="ms", precision=0):
    if val is None:
        return "  n/a"
    return f"{val:{6+precision}.{precision}f}{unit}"


def _ms(val) -> str:
    return f"{val:,.0f}" if val is not None else "n/a"


def print_report(results: list[dict], pruning_rate: float, reps: int) -> None:
    W = 90
    print("\n" + "=" * W)
    print("END-TO-END RUNTIME ANALYSIS  —  AutoGaze × vLLM")
    print("=" * W)
    print(f"Model:         Qwen/Qwen3-VL-2B-Instruct")
    print(f"Video:         assets/example_input.mp4")
    print(f"Pruning rate:  {pruning_rate}")
    print(f"Reps:          {reps}  (rep 0 = cold; reps 1+ = warm / encoder-cached)")
    print()

    dense = next((r for r in results if r["mode"] == "dense"), None)
    dense_tok     = dense["num_prompt_tokens"] if dense and dense.get("num_prompt_tokens", -1) > 0 else None
    dense_cold_ms = dense.get("elapsed_ms")    if dense else None
    dense_cold_vit= dense.get("vit_ms")        if dense else None

    # ── COLD table (rep 0: fresh model load, no GPU warmup) ──────────────────
    print("COLD INFERENCE  (rep 0 — fresh container, no warmup, no encoder cache)")
    cols = [
        ("Mode",        "<13"),
        ("Tokens",      ">7"),
        ("vs Dense",    ">9"),
        ("Load (ms)",   ">10"),
        ("ViT (ms)",    ">10"),
        ("LM (ms)",     ">9"),
        ("Total (ms)",  ">11"),
        ("Answer",      ">7"),
    ]
    hdr = "  ".join(f"{h:{f}}" for h, f in cols)
    sep = "  ".join("-" * int(f.lstrip("<>^")) for _, f in cols)
    print(hdr)
    print(sep)

    for r in results:
        tok   = r.get("num_prompt_tokens", -1)
        vit   = r.get("vit_ms")
        lm    = r.get("lm_ms")
        total = r.get("elapsed_ms")
        load  = r.get("load_ms")
        vs    = (f"-{(1 - tok/dense_tok)*100:.0f}%"
                 if dense_tok and tok > 0 and r["mode"] != "dense" else "—")
        print(
            f"  {r['mode']:<13}  {tok:>7}  {vs:>9}  "
            f"{_ms(load):>10}  {_ms(vit):>10}  "
            f"{_ms(lm):>9}  {_ms(total):>11}  {r.get('answer','?'):>7}"
        )

    # ── WARM table (avg reps 1+: GPU warm, but vLLM encoder cache active) ────
    warm_available = any(r.get("avg_elapsed_ms") for r in results
                         if (r.get("avg_elapsed_ms") or 0) < (r.get("elapsed_ms") or 1e9))
    if warm_available and reps > 1:
        print()
        print("WARM INFERENCE  (avg reps 1+ — GPU warm; NOTE: vLLM encoder cache active → fast)")
        print(hdr)
        print(sep)
        for r in results:
            tok   = r.get("num_prompt_tokens", -1)
            vit   = r.get("avg_vit_ms")
            lm    = r.get("avg_lm_ms")
            total = r.get("avg_elapsed_ms")
            load  = r.get("load_ms")
            vs    = (f"-{(1 - tok/dense_tok)*100:.0f}%"
                     if dense_tok and tok > 0 and r["mode"] != "dense" else "—")
            print(
                f"  {r['mode']:<13}  {tok:>7}  {vs:>9}  "
                f"{'—':>10}  {_ms(vit):>10}  "
                f"{_ms(lm):>9}  {_ms(total):>11}  {r.get('answer','?'):>7}"
            )

    # ── Speedup summary ───────────────────────────────────────────────────────
    print()
    if dense_cold_ms:
        print("Speedup over dense  (cold inference, rep 0):")
        for r in results:
            if r["mode"] == "dense":
                continue
            inf = r.get("elapsed_ms")
            if inf:
                speedup = dense_cold_ms / inf
                sign = "×" if speedup >= 1 else "× (slower)"
                print(f"  {r['mode']:<13}  {speedup:.2f}{sign}  "
                      f"({_ms(inf)} ms vs {_ms(dense_cold_ms)} ms)")

    evs = next((r for r in results if r["mode"] == "evs"), None)
    evs_vit = evs.get("vit_ms") if evs else None
    if dense_cold_vit and evs_vit:
        print()
        print("ViT timing breakdown  (cold, from CUDA events in EngineCore subprocess):")
        print(f"  dense      ViT: {_ms(dense_cold_vit)} ms")
        print(f"  evs        ViT: {_ms(evs_vit)} ms")
        for r in results:
            if r["mode"] == "sparse_vit":
                svit = r.get("vit_ms")
                if svit:
                    print(f"  sparse_vit ViT: {_ms(svit)} ms")
                    print(f"  sparse_vit vs EVS ViT speedup: {evs_vit/svit:.2f}×  "
                          f"({_ms(svit)} ms vs {_ms(evs_vit)} ms)")
                    print(f"  (K/N)² theoretical: "
                          f"{(r.get('K_vit',0) or 0) and f\"{(r['K_vit']/max(r.get('K_vit',1),1))**2:.2f}×\" or 'n/a'}")

    elif not (dense_cold_vit or evs_vit):
        print()
        print("ViT timing: n/a — hook did not fire or VIT_TIMING_MS not found in Docker output.")
        print("  Check that git pull got the latest commit and Docker container output is not suppressed.")

    # ── Per-rep detail ────────────────────────────────────────────────────────
    print()
    print("Per-rep detail:")
    for r in results:
        reps_data = r.get("reps", [])
        if not reps_data:
            continue
        print(f"  {r['mode']}:")
        for rep in reps_data:
            label = "warmup" if rep["rep"] == 0 and len(reps_data) > 1 else f"rep {rep['rep']}"
            vit_s = f"  vit={rep['vit_ms']:.0f}ms" if rep.get("vit_ms") else ""
            lm_s  = f"  lm={rep['lm_ms']:.0f}ms"   if rep.get("lm_ms")  else ""
            print(f"    [{label}]  elapsed={_ms(rep.get('elapsed_ms'))} ms{vit_s}{lm_s}  "
                  f"tokens={rep.get('num_prompt_tokens','?')}  answer={rep.get('answer','?')}")

    print()
    print(f"Results saved to: {RESULTS_FILE}")
    print("=" * W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3,
                        help="Inference reps per mode (first=warmup)")
    parser.add_argument("--pruning-rate", type=float, default=0.5)
    parser.add_argument("--gazing-ratio", type=float, default=0.245)
    parser.add_argument("--modes", nargs="+",
                        default=["dense", "evs", "sparse_vit"],
                        choices=["dense", "dense_eager", "evs", "magnitude", "autogaze", "sparse_vit"],
                        help="Modes to benchmark")
    parser.add_argument("--max-frames", type=int, default=32,
                        help="Max video frames to sample")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Frame sampling rate")
    parser.add_argument("--video", default=None,
                        help="Docker-side video path (default: assets/example_input.mp4)")
    args = parser.parse_args()

    pruning_rate = args.pruning_rate
    reps = args.reps
    modes = args.modes

    print("=" * 70)
    print("Runtime Analysis — dense vs EVS vs sparse_vit")
    print("=" * 70)
    print(f"Modes: {modes}  |  pruning_rate={pruning_rate}  |  reps={reps}")
    print(f"Image: {VLLM_IMAGE}\n")

    # ── Run each mode in Docker ───────────────────────────────────────────────
    results = []

    for i, mode in enumerate(modes):
        if i > 0:
            time.sleep(15)  # pause so GPU fully releases between containers
        try:
            r = run_docker(
                mode=mode,
                pruning_rate=pruning_rate,
                reps=reps,
                gazing_ratio=args.gazing_ratio,
                max_frames=args.max_frames,
                fps=args.fps,
                video_path=args.video,
            )
            results.append(r)
            tok  = r.get("num_prompt_tokens", "?")
            inf  = r.get("avg_elapsed_ms") or r.get("elapsed_ms", "?")
            ans  = r.get("answer", "?")
            print(f"\n  ✓ {mode}: tokens={tok}  infer={f'{inf:.0f}ms' if inf else 'n/a'}  answer={ans}")
        except Exception as exc:
            print(f"\n  ✗ {mode}: FAILED — {exc}")
            results.append({
                "mode": mode,
                "pruning_rate": pruning_rate,
                "answer": "ERROR",
                "num_prompt_tokens": -1,
                "elapsed_ms": None,
                "vit_ms": None,
                "lm_ms": None,
                "avg_elapsed_ms": None,
                "avg_vit_ms": None,
                "avg_lm_ms": None,
                "load_ms": None,
                "reps": [],
            })

    # ── Report ────────────────────────────────────────────────────────────────
    print_report(results, pruning_rate, reps)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
