#!/usr/bin/env python3
"""
Approach 5 — AutoGaze × vLLM Integration (orchestrator).

Launches the NVIDIA vLLM Docker container for each mode and collects results.
Each mode runs in its own container to guarantee clean GPU state.

Modes:
  dense      — no compression, all visual tokens
  evs        — EVS cosine-similarity (vLLM built-in) at pruning_rate=0.5
  magnitude  — AutoGaze-inspired magnitude selection (proof-of-concept)

Results are saved to approach5_results.json and summarised in result.md.

Usage:
    python scripts/approach5_vllm_integration.py
"""
import json
import os
import subprocess
import sys
import time

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE = os.environ.get("HF_HOME", "/home/scratch.thannan_wwfo/hf_cache")
RESULTS_FILE = os.path.join(REPO_DIR, "approach5_results.json")

VLLM_IMAGE = "gitlab-master.nvidia.com:5005/dl/dgx/vllm:main-py3.60784172-devel-arm64"
PRUNING_RATE = 0.5

MODES = [
    ("dense",     None),
    ("evs",       PRUNING_RATE),
    ("magnitude", PRUNING_RATE),
    ("autogaze",  PRUNING_RATE),   # actual nvidia/AutoGaze model, pre-ViT
]


def run_mode_in_docker(mode: str, pruning_rate: float | None) -> dict:
    pr_args = ["--pruning-rate", str(pruning_rate)] if pruning_rate is not None else []
    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--shm-size", "16g",
        "-v", f"{HF_CACHE}:/root/.cache/huggingface",
        "-v", f"{REPO_DIR}:/workspace/AutoGaze",
        "-e", f"REPO_DIR=/workspace/AutoGaze",
        "-e", f"HF_HOME=/root/.cache/huggingface",
        "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        VLLM_IMAGE,
        "python", "/workspace/AutoGaze/scripts/approach5_vllm_worker.py",
        "--mode", mode,
    ] + pr_args

    print(f"\n--- Docker run: mode={mode} pruning_rate={pruning_rate} ---", flush=True)
    print("  CMD:", " ".join(cmd[:6]), "... [truncated]", flush=True)

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    wall_s = time.perf_counter() - t0

    full_output = proc.stdout + "\n" + proc.stderr
    for line in full_output.splitlines():
        if "RESULT_JSON" not in line:
            print(f"  {line}")

    for line in full_output.splitlines():
        if line.startswith("RESULT_JSON:"):
            r = json.loads(line[len("RESULT_JSON:"):])
            r["wall_s"] = wall_s
            return r

    raise RuntimeError(
        f"No RESULT_JSON found for mode={mode}.\n"
        f"returncode={proc.returncode}\n"
        f"stderr (last 2000):\n{proc.stderr[-2000:]}"
    )


def update_result_md(results: list[dict]) -> None:
    result_md = os.path.join(REPO_DIR, "result.md")
    with open(result_md, "r") as f:
        content = f.read()

    section = "\n---\n\n## Approach 5 — AutoGaze × vLLM Integration\n\n"
    section += f"**Model:** `Qwen/Qwen3-VL-2B-Instruct`  \n"
    section += f"**Container:** `{VLLM_IMAGE}`  \n"
    section += f"**Video:** `assets/example_input.mp4`  \n"
    section += f"**Pruning rate:** {PRUNING_RATE} ({int(PRUNING_RATE*100)}% tokens removed when active)\n\n"

    dense_tok = next((r["num_prompt_tokens"] for r in results if r["mode"] == "dense"), None)
    section += "| Mode | Tokens | vs Dense | Latency (ms) | Answer |\n"
    section += "|---|---:|:---:|---:|:---:|\n"
    for r in results:
        tok = r["num_prompt_tokens"]
        vs = f"-{(1 - tok/dense_tok)*100:.0f}%" if dense_tok and tok > 0 and r["mode"] != "dense" else "—"
        section += f"| {r['mode']} | {tok} | {vs} | {r['elapsed_ms']:.0f} | {r['answer']} |\n"

    section += "\n**Key findings:**\n"
    if len(results) >= 3:
        evs_r = next((r for r in results if r["mode"] == "evs"), None)
        mag_r = next((r for r in results if r["mode"] == "magnitude"), None)
        if evs_r and mag_r and dense_tok:
            evs_red = (1 - evs_r["num_prompt_tokens"] / dense_tok) * 100
            mag_red = (1 - mag_r["num_prompt_tokens"] / dense_tok) * 100
            section += f"- EVS achieves {evs_red:.0f}% token reduction; answer: `{evs_r['answer']}`\n"
            section += f"- AutoGaze-magnitude achieves {mag_red:.0f}% token reduction; answer: `{mag_r['answer']}`\n"
            if evs_r["answer"] == mag_r["answer"]:
                section += "- Both compression methods give the **same answer** as dense baseline\n"

    section += "\n**Production path (not yet implemented):**\n"
    section += "1. Move AutoGaze to run pre-ViT (in the processor)\n"
    section += "2. Store raw frames in `AutoGazeContext` before ViT encoding\n"
    section += "3. `autogaze_retention_mask()` uses the full trained model on raw frames\n"
    section += "4. Report actual K to vLLM scheduler (not fixed `(1-q)*N`)\n"

    if "## Approach 5" in content:
        idx = content.index("## Approach 5")
        content = content[:idx] + section[2:]
    else:
        content = content + section

    with open(result_md, "w") as f:
        f.write(content)
    print(f"\nUpdated {result_md}")


def main():
    print("=" * 70)
    print("Approach 5 — AutoGaze × vLLM Integration")
    print("=" * 70)
    print(f"Image:  {VLLM_IMAGE}")
    print(f"Modes:  {[m for m, _ in MODES]}")
    print(f"Video:  {REPO_DIR}/assets/example_input.mp4\n")

    results = []
    for mode, pr in MODES:
        try:
            r = run_mode_in_docker(mode, pr)
            results.append(r)
            print(f"\n  ✓ {mode}: {r['num_prompt_tokens']} tokens, {r['elapsed_ms']:.0f} ms, answer={r['answer']}")
        except Exception as e:
            print(f"\n  ✗ {mode}: FAILED — {e}")
            results.append({
                "mode": mode, "pruning_rate": pr,
                "answer": "ERROR", "num_prompt_tokens": -1,
                "elapsed_ms": -1, "load_ms": -1,
            })

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    dense_tok = next((r["num_prompt_tokens"] for r in results if r["mode"] == "dense"), None)
    print(f"{'Mode':<12}  {'Tokens':>8}  {'vs Dense':>10}  {'ms':>8}  {'Answer'}")
    print("-" * 60)
    for r in results:
        tok = r["num_prompt_tokens"]
        vs = f"(-{(1-tok/dense_tok)*100:.0f}%)" if dense_tok and tok > 0 and r["mode"] != "dense" else ""
        print(f"{r['mode']:<12}  {tok:>8}  {vs:>10}  {r['elapsed_ms']:>8.0f}  {r['answer']}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {RESULTS_FILE}")

    try:
        update_result_md(results)
    except Exception as e:
        print(f"Warning: Could not update result.md: {e}")


if __name__ == "__main__":
    main()
