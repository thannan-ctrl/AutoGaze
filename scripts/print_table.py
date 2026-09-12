"""Print every timing/count field the harness records per question, averaged per mode,
as a markdown table -- see QUICKSTART.md step 6.

Usage: python3 scripts/print_table.py [--dataset egoschema|video_mme] [mode ...]
"""
import argparse
import json
import os

FIELDS = [
    "num_tokens",
    "preproc_ms", "decode_ms", "image_preproc_ms", "autogaze_ops_ms", "autogaze_model_ms",
    "codec_encode_ms", "codec_decode_ms",
    "selector_score_ms", "selector_rank_ms",
    "selector_csvparse_ms", "selector_scorecu_ms", "selector_paintmap_ms",
    "gazing_info_total_ms", "selector_glue_ms",
    "other_ms", "cpu_ms", "gpu_ms",
    "generate_ms", "vit_ms", "llm_prefill_ms", "llm_decode_ms", "llm_ms", "llm_calls",
    "e2e_ms",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="egoschema", choices=["egoschema", "video_mme"])
    ap.add_argument("modes", nargs="*", default=["dense", "codec", "codec_nvdec"])
    args = ap.parse_args()

    header = ["mode", "n", "acc"] + FIELDS
    print(" | ".join(header))
    print(" | ".join("---" for _ in header))
    for mode in args.modes:
        path = f"benchmark_results/nvila_hd_accuracy_breakdown_{mode}_{args.dataset}_nvf16.jsonl"
        if not os.path.exists(path):
            print(f"{mode} | (missing: {path})")
            continue
        rows = [json.loads(l) for l in open(path)]
        n = len(rows)
        acc = sum(r.get("correct", False) for r in rows) / n
        avg = lambda k: sum((r.get(k) or 0) for r in rows) / n
        cells = [mode, str(n), f"{acc:.1%}"]
        for f in FIELDS:
            v = avg(f)
            cells.append(f"{v:.0f}" if f in ("num_tokens", "llm_calls") else f"{v / 1000:.3f}s")
        print(" | ".join(cells))


if __name__ == "__main__":
    main()
