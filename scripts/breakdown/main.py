"""Loads the model once, runs each requested mode, writes the summary json."""
import json
import os

import torch
from transformers import AutoModel

from . import config, dataset, runner, summary, timing


def main() -> None:
    print(f"[setup] CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"[setup] Device: {config.DEVICE}", flush=True)

    samples = dataset.load_samples()
    print(f"[setup] Loaded {len(samples)} {config.DATASET} questions (seed={config.SEED})", flush=True)

    # SKIP_LLM=1: patch-selection-latency-only probe -- never load the 8B LLM
    # (which is what actually OOMs at high nvf/high token counts, see
    # Codec_Selector_Feasibility.md's VideoMME nvf=512 sampled-only finding),
    # only build each mode's processor (which still runs the real selector --
    # AutoGaze model or codec_selector -- since that's the thing being timed).
    skip_llm = os.environ.get("SKIP_LLM", "0") == "1"
    if skip_llm:
        print("[setup] SKIP_LLM=1 -- not loading the 8B model, patch-selection-only latency probe", flush=True)
        model, llm_call_state = None, None
    else:
        print("[setup] Loading model (once, reused for both modes)...", flush=True)
        model = AutoModel.from_pretrained(
            config.MODEL_PATH, trust_remote_code=True, dtype=torch.bfloat16, max_batch_size_siglip=32,
        ).to(config.DEVICE)
        model.eval()
        llm_call_state = timing.install_model_hooks(model)

    modes = os.environ.get("MODES", "autogaze,dense").split(",")
    print(f"[setup] Modes to run: {modes}", flush=True)

    summary_path = os.path.join(
        config.REPO_DIR, "benchmark_results",
        f"nvila_hd_accuracy_breakdown_summary_{config.DATASET}{config.result_suffix()}.json",
    )
    summaries = {}
    for mode in modes:
        results = runner.run_mode(mode, model, llm_call_state, samples, skip_llm=skip_llm)
        summaries[mode] = summary.summarize(results, len(samples))
        # Write after each mode (not just at the end) so a killed run still
        # leaves a summary for whatever modes did finish.
        with open(summary_path, "w") as f:
            json.dump(summaries, f, indent=2)

    print("\n===== Accuracy + Profiling Summary =====")
    for mode, s in summaries.items():
        print(summary.format_line(mode, s))
    print(f"\n[done] Summary written to {summary_path}", flush=True)
