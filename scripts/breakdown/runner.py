"""Per-question and per-mode benchmark loop.

Resumable: each result is appended to the output JSONL immediately (not
buffered to end-of-run), and on startup any item_ids already present in an
existing output file are loaded and skipped. Safe to kill and re-launch with
the same command -- it picks up where it left off.
"""
import gc
import json
import os
import time

import torch

from . import config, dataset, instrumentation, processor, timing


def output_path(mode: str) -> str:
    return os.path.join(
        config.REPO_DIR, "benchmark_results",
        f"nvila_hd_accuracy_breakdown_{mode}_{config.DATASET}{config.result_suffix()}.jsonl",
    )


def load_done_ids(mode: str) -> dict:
    """Returns {item_id: result_dict} for already-completed items."""
    path = output_path(mode)
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done[r["item_id"]] = r
    return done


def run_question(model, llm_call_state, proc, item: dict, mode: str = None) -> dict:
    text = f"{proc.tokenizer.video_token}\n\n{dataset.build_prompt(item)}"

    if mode == "codec":
        instrumentation.set_codec_video_context(item["video_path"])

    timing.reset()
    t0 = time.time()
    inputs = proc(text=text, videos=item["video_path"], return_tensors="pt")
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    num_tokens = inputs["input_ids"].shape[1]
    torch.cuda.synchronize()
    preproc_ms = (time.time() - t0) * 1000
    preproc_timing = timing.snapshot()

    decode_ms = preproc_timing["decode_ms"]
    image_preproc_ms = preproc_timing["preprocess_videos_total_ms"] - preproc_timing["autogaze_transform_ms"]
    autogaze_ops_ms = preproc_timing["autogaze_transform_ms"] + max(
        preproc_timing["gazing_info_total_ms"] - preproc_timing["autogaze_model_ms"], 0.0
    )
    autogaze_model_ms = preproc_timing["autogaze_model_ms"]
    other_ms = max(preproc_ms - decode_ms - image_preproc_ms - autogaze_ops_ms - autogaze_model_ms, 0.0)

    timing.reset()
    llm_call_state["calls_since_reset"] = 0
    gen_t0 = time.time()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=8)
    torch.cuda.synchronize()
    gen_ms = (time.time() - gen_t0) * 1000
    gen_timing = timing.snapshot()

    response = proc.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()
    elapsed = time.time() - t0
    pred = dataset.parse_letter(response, len(item["options"]))

    return {
        "item_id": item["item_id"],
        "pred": dataset.LETTERS[pred] if pred >= 0 else None,
        "answer": dataset.LETTERS[item["answer_idx"]],
        "preproc_ms": preproc_ms,
        "decode_ms": decode_ms,
        "image_preproc_ms": image_preproc_ms,
        "autogaze_ops_ms": autogaze_ops_ms,
        "autogaze_model_ms": autogaze_model_ms,
        "other_ms": other_ms,
        "cpu_ms": decode_ms + image_preproc_ms + autogaze_ops_ms + other_ms,
        "gpu_ms": autogaze_model_ms,
        "generate_ms": gen_ms,
        "vit_ms": gen_timing["vit_ms"],
        "llm_prefill_ms": gen_timing["llm_prefill_ms"],
        "llm_decode_ms": gen_timing["llm_decode_ms"],
        "llm_ms": gen_timing["llm_prefill_ms"] + gen_timing["llm_decode_ms"],
        "llm_calls": gen_timing["llm_calls"],
        "e2e_ms": elapsed * 1000,
        "correct": pred == item["answer_idx"],
        "raw_text": response,
        "num_tokens": num_tokens,
        "elapsed_s": elapsed,
    }


def _log_result(mode: str, i: int, total: int, r: dict, correct: int, n_scored: int) -> None:
    if "error" in r:
        print(f"[{mode}] {i}/{total} id={r['item_id'][:12]} ERROR: {r['error']}", flush=True)
        return
    nf_note = f" nvf={r['num_video_frames_used']}" if mode == "dense" else ""
    print(
        f"[{mode}] {i}/{total} id={r['item_id'][:12]} pred={r['pred']} "
        f"answer={r['answer']} correct={r['correct']} tokens={r['num_tokens']}{nf_note} "
        f"e2e={r['elapsed_s']:.1f}s preproc={r['preproc_ms']:.0f}ms "
        f"[decode={r['decode_ms']:.0f} imgprep={r['image_preproc_ms']:.0f} "
        f"agops={r['autogaze_ops_ms']:.0f} agmodel={r['autogaze_model_ms']:.0f} "
        f"other={r['other_ms']:.0f} | cpu={r['cpu_ms']:.0f} gpu={r['gpu_ms']:.0f}]ms "
        f"vit={r['vit_ms']:.0f}ms llm={r['llm_ms']:.0f}ms"
        f"(prefill={r['llm_prefill_ms']:.0f}/decode={r['llm_decode_ms']:.0f}, {r['llm_calls']} calls) "
        f"(running acc={correct}/{n_scored}={correct / max(n_scored, 1):.1%})",
        flush=True,
    )


def run_mode(mode: str, model, llm_call_state, samples: list) -> list:
    kw = {**config.COMMON_KW, **config.CONFIGS[mode]}
    print(f"\n[{mode}] kwargs: {kw}", flush=True)

    done = load_done_ids(mode)
    if done:
        print(f"[{mode}] Resuming: {len(done)}/{len(samples)} items already done, skipping those.", flush=True)

    static_proc = processor.build(mode, kw["num_video_frames"]) if mode != "dense" else None

    out_path = output_path(mode)
    results = [done[item["item_id"]] for item in samples if item["item_id"] in done]
    correct = sum(1 for r in results if r.get("correct"))
    n_scored = sum(1 for r in results if "error" not in r)

    with open(out_path, "a") as out_f:
        for item in samples:
            if item["item_id"] in done:
                continue
            i = len(results) + 1
            budgets = config.dense_frame_budgets() if mode == "dense" else [kw["num_video_frames"]]
            r = None
            for bi, nf in enumerate(budgets):
                try:
                    proc = static_proc or processor.build(mode, nf)
                    r = run_question(model, llm_call_state, proc, item, mode=mode)
                    r["num_video_frames_used"] = nf
                    break
                except torch.cuda.OutOfMemoryError as e:
                    gc.collect()
                    torch.cuda.empty_cache()
                    if bi == len(budgets) - 1:
                        r = {"item_id": item["item_id"], "error": f"OOM at all frame budgets {budgets}: {str(e).splitlines()[0]}"}
                    else:
                        print(
                            f"[{mode}] {i}/{len(samples)} id={item['item_id'][:12]} OOM at "
                            f"num_video_frames={nf}, retrying at {budgets[bi + 1]}",
                            flush=True,
                        )
                except Exception as e:
                    r = {"item_id": item["item_id"], "error": str(e)}
                    break
                finally:
                    torch.cuda.empty_cache()

            results.append(r)
            out_f.write(json.dumps(r) + "\n")
            out_f.flush()
            if "error" not in r:
                n_scored += 1
                correct += int(r["correct"])
            _log_result(mode, i, len(samples), r, correct, n_scored)

    print(f"\n[{mode}] Final accuracy: {correct}/{n_scored} = {correct / max(n_scored, 1):.1%}", flush=True)
    print(f"[{mode}] Wrote {out_path}", flush=True)

    return results
