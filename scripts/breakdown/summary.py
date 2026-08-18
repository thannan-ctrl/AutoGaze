"""Averages a mode's per-question results into the summary dict shape written
to benchmark_results/nvila_hd_accuracy_breakdown_summary<suffix>.json."""

_AVG_FIELDS = [
    "e2e_ms", "preproc_ms", "decode_ms", "image_preproc_ms", "autogaze_ops_ms",
    "autogaze_model_ms", "other_ms", "cpu_ms", "gpu_ms", "vit_ms", "llm_ms",
    "llm_prefill_ms", "llm_decode_ms", "llm_calls", "num_tokens",
]


def summarize(results: list, n_samples: int) -> dict:
    n_correct = sum(1 for r in results if r.get("correct"))
    ok_results = [r for r in results if "e2e_ms" in r]
    avg = lambda k: sum(r[k] for r in ok_results) / len(ok_results) if ok_results else float("nan")

    out = {
        "n": n_samples,
        "correct": n_correct,
        "accuracy": n_correct / n_samples,
    }
    out.update({f"avg_{k}": avg(k) for k in _AVG_FIELDS})
    return out


def format_line(mode: str, s: dict) -> str:
    return (
        f"{mode:>8}: acc={s['correct']}/{s['n']}={s['accuracy']:.1%}  "
        f"avg_tokens={s['avg_num_tokens']:.0f}  avg_e2e={s['avg_e2e_ms']:.0f}ms  "
        f"avg_preproc={s['avg_preproc_ms']:.0f}ms "
        f"(decode={s['avg_decode_ms']:.0f} imgprep={s['avg_image_preproc_ms']:.0f} "
        f"agops={s['avg_autogaze_ops_ms']:.0f} agmodel={s['avg_autogaze_model_ms']:.0f} "
        f"other={s['avg_other_ms']:.0f} | cpu={s['avg_cpu_ms']:.0f} gpu={s['avg_gpu_ms']:.0f})  "
        f"avg_vit={s['avg_vit_ms']:.0f}ms  avg_llm={s['avg_llm_ms']:.0f}ms "
        f"(prefill={s['avg_llm_prefill_ms']:.0f}/decode={s['avg_llm_decode_ms']:.0f}, "
        f"{s['avg_llm_calls']:.1f} calls)"
    )
