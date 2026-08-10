#!/usr/bin/env python3
"""
Run sparse_vit at a target gazing_ratio and compare against cached dense/EVS results.

Usage:
    python scripts/compare_sparse_vit_ratio.py --gazing-ratio 0.245 --reps 3
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE  = os.environ.get("HF_HOME", "/home/scratch.thannan_wwfo/hf_cache")
CACHED_RESULTS = os.path.join(REPO_DIR, "runtime_analysis.json")
VLLM_IMAGE = "nvcr.io/nvidia/vllm:26.07-py3"
AUTOGAZE_PYTHON = "/home/scratch.thannan_wwfo/miniforge-aarch64/envs/auto_gaze/bin/python"


def precompute_mask(gazing_ratio: float, output: str) -> float:
    video = os.path.join(REPO_DIR, "assets", "example_input.mp4")
    cmd = [
        AUTOGAZE_PYTHON,
        os.path.join(REPO_DIR, "scripts", "run_autogaze_preprocess.py"),
        "--video", video,
        "--output", output,
        "--gazing-ratio", str(gazing_ratio),
        "--grid-hw", "32", "32",
    ]
    print(f"\n[preprocess] gazing_ratio={gazing_ratio} grid=32×32 → {output}", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=False, text=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if r.returncode != 0:
        raise RuntimeError(f"Preprocessing failed (exit {r.returncode})")
    print(f"[preprocess] Done in {elapsed_ms:.0f} ms", flush=True)
    return elapsed_ms


def run_sparse_vit(mask_path: str, gazing_ratio: float, reps: int) -> dict:
    env_vars = [
        "-e", f"REPO_DIR=/workspace/AutoGaze",
        "-e", f"HF_HOME=/root/.cache/huggingface",
        "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "-e", f"AUTOGAZE_MASK_PATH={mask_path}",
    ]
    vol_mounts = [
        "-v", f"{HF_CACHE}:/root/.cache/huggingface",
        "-v", f"{REPO_DIR}:/workspace/AutoGaze",
        "-v", f"{mask_path}:{mask_path}",
    ]
    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all", "--shm-size", "16g",
        *vol_mounts, *env_vars,
        VLLM_IMAGE,
        "python", "/workspace/AutoGaze/scripts/approach5_vllm_worker.py",
        "--mode", "sparse_vit",
        "--pruning-rate", str(gazing_ratio),
        "--reps", str(reps),
    ]
    print(f"\n--- Docker: sparse_vit gazing_ratio={gazing_ratio} reps={reps} ---", flush=True)
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
        f"No RESULT_JSON in output\nreturncode={proc.returncode}\n"
        f"stderr:\n{proc.stderr[-2000:]}"
    )


def print_comparison(cached: list[dict], new_result: dict, gazing_ratio: float) -> None:
    # Pull dense and evs from cache
    dense = next((r for r in cached if r["mode"] == "dense"), None)
    evs   = next((r for r in cached if r["mode"] == "evs"),   None)

    rows = []
    if dense: rows.append(("dense",                    dense))
    if evs:   rows.append(("evs",                      evs))
    rows.append((f"sparse_vit (ratio={gazing_ratio})", new_result))

    dense_tok = dense["num_prompt_tokens"] if dense else None
    dense_inf = (dense.get("avg_elapsed_ms") or dense.get("elapsed_ms")) if dense else None

    print("\n" + "=" * 82)
    print(f"COMPARISON — dense vs EVS vs sparse_vit @ gazing_ratio={gazing_ratio}")
    print("=" * 82)
    print(f"{'Mode':<30}  {'Tokens':>7}  {'vs Dense':>9}  {'ViT (ms)':>9}  {'LM (ms)':>8}  {'Infer (ms)':>11}  {'Answer':>6}")
    print("-" * 82)

    for label, r in rows:
        tok   = r.get("num_prompt_tokens", -1)
        vit   = r.get("avg_vit_ms")   or r.get("vit_ms")
        lm    = r.get("avg_lm_ms")    or r.get("lm_ms")
        infer = r.get("avg_elapsed_ms") or r.get("elapsed_ms")
        vs    = (f"-{(1-tok/dense_tok)*100:.0f}%"
                 if dense_tok and tok > 0 and "dense" not in label
                 else "—")
        vit_s   = f"{vit:.0f}"   if vit   else "n/a"
        lm_s    = f"{lm:.0f}"    if lm    else "n/a"
        infer_s = f"{infer:.0f}" if infer else "n/a"
        print(f"  {label:<28}  {tok:>7}  {vs:>9}  {vit_s:>9}  {lm_s:>8}  {infer_s:>11}  {r.get('answer','?'):>6}")

    print()
    if dense_inf:
        print("Inference speedup vs dense:")
        for label, r in rows:
            if "dense" in label:
                continue
            inf = r.get("avg_elapsed_ms") or r.get("elapsed_ms")
            if inf:
                print(f"  {label:<30}  {dense_inf/inf:.2f}×")

    # EVS vs sparse_vit at matched tokens
    if evs and new_result:
        evs_inf   = evs.get("avg_elapsed_ms")   or evs.get("elapsed_ms")
        evs_vit   = evs.get("avg_vit_ms")        or evs.get("vit_ms")
        sv_inf    = new_result.get("avg_elapsed_ms") or new_result.get("elapsed_ms")
        sv_vit    = new_result.get("avg_vit_ms")     or new_result.get("vit_ms")
        sv_tok    = new_result.get("num_prompt_tokens")
        evs_tok   = evs.get("num_prompt_tokens")

        print(f"\nEVS vs sparse_vit @ ratio={gazing_ratio} (token-matched comparison):")
        print(f"  Tokens:     EVS={evs_tok}   sparse_vit={sv_tok}  "
              f"({'matched' if abs(sv_tok-evs_tok) < 30 else f'delta={sv_tok-evs_tok:+d}'})")
        if evs_vit and sv_vit:
            print(f"  ViT:        EVS={evs_vit:.0f}ms → sparse_vit={sv_vit:.0f}ms  "
                  f"({evs_vit/sv_vit:.2f}× faster)")
        if evs_inf and sv_inf:
            sign = "faster" if sv_inf < evs_inf else "slower"
            print(f"  Inference:  EVS={evs_inf:.0f}ms → sparse_vit={sv_inf:.0f}ms  "
                  f"({abs(evs_inf-sv_inf):.0f}ms {sign})")
    print("=" * 82)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gazing-ratio", type=float, default=0.245)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    ratio = args.gazing_ratio
    mask_path = f"/tmp/ag_mask_vit_{ratio:.3f}.pt"

    # Load cached dense/EVS results
    cached = []
    if os.path.exists(CACHED_RESULTS):
        with open(CACHED_RESULTS) as f:
            cached = json.load(f)
        print(f"Loaded cached results for: {[r['mode'] for r in cached]}")
    else:
        print(f"WARNING: {CACHED_RESULTS} not found — dense/EVS rows will be absent")

    # Precompute mask at new ratio
    preprocess_ms = precompute_mask(ratio, mask_path)

    # Run sparse_vit
    result = run_sparse_vit(mask_path, ratio, args.reps)
    result["preprocess_ms"] = preprocess_ms
    result["gazing_ratio_used"] = ratio

    # Save
    out_file = os.path.join(REPO_DIR, f"sparse_vit_{ratio:.3f}_results.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_file}")

    print_comparison(cached, result, ratio)


if __name__ == "__main__":
    main()
