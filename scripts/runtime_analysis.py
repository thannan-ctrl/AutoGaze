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
               gazing_ratio: float = 0.245) -> dict:
    """
    Run one mode in a Docker container.
    Returns the parsed RESULT_JSON dict augmented with wall_ms.
    """
    env_vars = [
        "-e", f"REPO_DIR=/workspace/AutoGaze",
        "-e", f"HF_HOME=/root/.cache/huggingface",
        "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
    ]
    vol_mounts = [
        "-v", f"{HF_CACHE}:/root/.cache/huggingface",
        "-v", f"{REPO_DIR}:/workspace/AutoGaze",
    ]

    worker_args = [
        "--mode", mode,
        "--pruning-rate", str(pruning_rate),
        "--reps", str(reps),
    ]

    if mode == "sparse_vit":
        worker_args += [
            "--video", "/workspace/AutoGaze/assets/example_input.mp4",
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
        if "RESULT_JSON" not in line:
            print(f"  {line}")

    for line in full.splitlines():
        if line.startswith("RESULT_JSON:"):
            r = json.loads(line[len("RESULT_JSON:"):])
            r["wall_ms"] = wall_ms
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


def print_report(results: list[dict], pruning_rate: float, reps: int) -> None:
    print("\n" + "=" * 80)
    print("END-TO-END RUNTIME ANALYSIS  —  AutoGaze × vLLM")
    print("=" * 80)
    print(f"Model:         Qwen/Qwen3-VL-2B-Instruct")
    print(f"Video:         assets/example_input.mp4")
    print(f"Pruning rate:  {pruning_rate}")
    print(f"Reps:          {reps}  (first=warmup, rest measured; avg reported)")
    print()

    dense = next((r for r in results if r["mode"] == "dense"), None)
    dense_tok  = dense["num_prompt_tokens"] if dense else None
    dense_vit  = dense.get("avg_vit_ms") or dense.get("vit_ms")
    dense_lm   = dense.get("avg_lm_ms")  or dense.get("lm_ms")
    dense_inf  = dense.get("avg_elapsed_ms") or dense.get("elapsed_ms")

    # Header
    cols = [
        ("Mode",          "<12"),
        ("Tokens",        ">8"),
        ("vs Dense",      ">9"),
        ("Load (ms)",     ">10"),
        ("ViT (ms)",      ">10"),
        ("LM (ms)",       ">9"),
        ("Infer (ms)",    ">11"),
        ("Answer",        ">7"),
    ]
    header = "  ".join(f"{h:{fmt}}" for h, fmt in cols)
    sep    = "  ".join("-" * int(fmt.lstrip("<>^")) for _, fmt in cols)
    print(header)
    print(sep)

    for r in results:
        tok   = r.get("num_prompt_tokens", -1)
        vit   = r.get("avg_vit_ms")   or r.get("vit_ms")
        lm    = r.get("avg_lm_ms")    or r.get("lm_ms")
        infer = r.get("avg_elapsed_ms") or r.get("elapsed_ms")
        load  = r.get("load_ms")

        vs = (f"-{(1 - tok/dense_tok)*100:.0f}%"
              if dense_tok and tok > 0 and r["mode"] != "dense"
              else "—")
        vit_s  = f"{vit:.0f}" if vit else "n/a"
        lm_s   = f"{lm:.0f}"  if lm  else "n/a"
        inf_s  = f"{infer:.0f}" if infer else "n/a"
        load_s = f"{load:.0f}" if load else "n/a"

        print(
            f"  {r['mode']:<12}  {tok:>8}  {vs:>9}  "
            f"{load_s:>10}  {vit_s:>10}  "
            f"{lm_s:>9}  {inf_s:>11}  {r.get('answer','?'):>7}"
        )

    print()

    # Speedup summary
    if dense_inf:
        print("Speedup over dense (inference only, excluding model load):")
        for r in results:
            if r["mode"] == "dense":
                continue
            inf = r.get("avg_elapsed_ms") or r.get("elapsed_ms")
            if inf:
                speedup = dense_inf / inf
                print(f"  {r['mode']:<12}  {speedup:.2f}×  ({inf:.0f} ms vs {dense_inf:.0f} ms)")

    if dense_vit:
        print("\nViT speedup (sparse_vit only):")
        for r in results:
            if r["mode"] != "sparse_vit":
                continue
            vit = r.get("avg_vit_ms") or r.get("vit_ms")
            if vit:
                vs_speedup = dense_vit / vit
                print(f"  sparse_vit ViT:  {vit:.0f} ms vs dense {dense_vit:.0f} ms  →  {vs_speedup:.2f}×")

    # Per-rep breakdown for sparse_vit (if available)
    for r in results:
        rep_data = r.get("reps")
        if rep_data and len(rep_data) > 1:
            measured = rep_data[1:]
            elapsed_vals = [x["elapsed_ms"] for x in measured if x["elapsed_ms"]]
            vit_vals     = [x["vit_ms"]     for x in measured if x.get("vit_ms")]
            if elapsed_vals:
                print(f"\n  {r['mode']} per-rep (excluding warmup):")
                print(f"    elapsed: min={min(elapsed_vals):.0f}  "
                      f"mean={statistics.mean(elapsed_vals):.0f}  "
                      f"max={max(elapsed_vals):.0f} ms")
            if vit_vals:
                print(f"    vit:     min={min(vit_vals):.0f}  "
                      f"mean={statistics.mean(vit_vals):.0f}  "
                      f"max={max(vit_vals):.0f} ms")

    print()
    print(f"Results saved to: {RESULTS_FILE}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3,
                        help="Inference reps per mode (first=warmup)")
    parser.add_argument("--pruning-rate", type=float, default=0.5)
    parser.add_argument("--gazing-ratio", type=float, default=0.245)
    parser.add_argument("--modes", nargs="+",
                        default=["dense", "evs", "sparse_vit"],
                        choices=["dense", "evs", "magnitude", "autogaze", "sparse_vit"],
                        help="Modes to benchmark")
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
            time.sleep(5)  # brief pause so GPU state resets between containers
        try:
            r = run_docker(
                mode=mode,
                pruning_rate=pruning_rate,
                reps=reps,
                gazing_ratio=args.gazing_ratio,
            )
            results.append(r)
            tok  = r.get("num_prompt_tokens", "?")
            inf  = r.get("avg_elapsed_ms") or r.get("elapsed_ms", "?")
            ans  = r.get("answer", "?")
            print(f"\n  ✓ {mode}: tokens={tok}  infer={inf:.0f}ms  answer={ans}")
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
