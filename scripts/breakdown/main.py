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
        results = runner.run_mode(mode, model, llm_call_state, samples)
        summaries[mode] = summary.summarize(results, len(samples))
        # Write after each mode (not just at the end) so a killed run still
        # leaves a summary for whatever modes did finish.
        with open(summary_path, "w") as f:
            json.dump(summaries, f, indent=2)

    print("\n===== Accuracy + Profiling Summary =====")
    for mode, s in summaries.items():
        print(summary.format_line(mode, s))
    print(f"\n[done] Summary written to {summary_path}", flush=True)
