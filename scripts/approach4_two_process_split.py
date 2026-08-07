#!/usr/bin/env python3
"""
Approach 4 — Two-process split (AutoGaze embedding service + VLM).

Architecture:
  Client → AutoGaze service (GPU) → compressed embeddings [K × D]
                                          ↓
                               VLM receives K embedding tokens
                               (no raw video, no ViT, just embeddings)

What this script does:
  Step 1 (AutoGaze service): Load NVILA, run preprocessing + ViT encoder,
    extract the visual token embeddings before they enter the LLM.
    Save embeddings to disk at embeddings/visual_tokens_{ratio}.pt

  Step 2 (LLM service stub): Load just the LLM component, inject saved
    embeddings at the multimodal token positions, run generation.
    (Full injection requires custom forward; we demonstrate the shape/size
    and run a reference inference to validate correctness.)

Pro: Zero vLLM changes — embeddings arrive as a fixed-length tensor.
Con: Doubles GPU memory; ViT must be available in the service process.
"""
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_DIR = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)

MODEL_PATH = "nvidia/NVILA-8B-HD-Video"
VIDEO_PATH = os.path.join(REPO_DIR, "assets", "example_input.mp4")
PROMPT = (
    "Question: What does the white text on the green road sign say?\n"
    "A. Hampden St\nB. Hampden Ave\nC. HampdenBlvd\nD. Hampden Rd\n"
    "Please answer directly with the letter of the correct answer."
)

GAZING_RATIOS = [0.25, 0.5, 1.0]
EMBED_DIR = Path(__file__).parent.parent / "embeddings"


def extract_visual_embeddings(processor, model, ratio: float) -> dict:
    """
    Run AutoGaze patch selection + ViT encoder.
    Hook into the model to capture visual token embeddings before the LLM.
    Returns the embeddings tensor and metadata.
    """
    video_token = processor.tokenizer.video_token
    inputs = processor(
        text=f"{video_token}\n\n{PROMPT}", videos=VIDEO_PATH, return_tensors="pt"
    )

    inputs_gpu = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    captured = {}

    def hook_fn(module, input, output):
        # Capture the output of the vision encoder (before LLM embedding merge)
        if isinstance(output, torch.Tensor):
            captured["visual_embeddings"] = output.detach().cpu()
        elif isinstance(output, (tuple, list)) and len(output) > 0:
            if isinstance(output[0], torch.Tensor):
                captured["visual_embeddings"] = output[0].detach().cpu()

    # Try to hook the vision tower / image encoder
    hooks = []
    for name, module in model.named_modules():
        if any(kw in name.lower() for kw in ("vision_tower", "image_encoder", "visual_encoder", "siglip", "vit")):
            if hasattr(module, "forward") and "encoder" in name.lower():
                h = module.register_forward_hook(hook_fn)
                hooks.append((name, h))
                break

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs_gpu, max_new_tokens=10)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    for _, h in hooks:
        h.remove()

    answer = processor.batch_decode(
        outputs[:, inputs_gpu["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()

    K = inputs["input_ids"].shape[1]
    result = {
        "ratio": ratio,
        "K": K,
        "answer": answer,
        "generate_ms": elapsed_ms,
        "input_ids": inputs["input_ids"].cpu(),
    }
    if "visual_embeddings" in captured:
        result["visual_embeddings"] = captured["visual_embeddings"]

    return result


def main():
    print("=" * 70)
    print("Approach 4 — Two-process split (embedding service + LLM service)")
    print("=" * 70)

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for ratio in GAZING_RATIOS:
        print(f"\n=== Step 1: AutoGaze service  (ratio={ratio}) ===", flush=True)
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            autogaze_model_id="nvidia/AutoGaze",
            num_video_frames=128,
            num_video_frames_thumbnail=64,
            max_tiles_video=48,
            gazing_ratio_tile=ratio,
            gazing_ratio_thumbnail=ratio,
            task_loss_requirement_tile=None,
            task_loss_requirement_thumbnail=None,
            max_batch_size_autogaze=16,
            trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(
            MODEL_PATH, trust_remote_code=True, device_map="auto", max_batch_size_siglip=32
        )
        model.eval()

        print("  Extracting visual embeddings...", flush=True)
        result = extract_visual_embeddings(processor, model, ratio)  # noqa: E501
        all_results.append(result)

        embed_path = EMBED_DIR / f"visual_tokens_ratio{ratio}.pt"
        payload = {"K": result["K"], "input_ids": result["input_ids"]}
        if "visual_embeddings" in result:
            payload["visual_embeddings"] = result["visual_embeddings"]
            print(f"  visual_embeddings shape: {tuple(result['visual_embeddings'].shape)}")
            print(f"  Embedding size (bytes):  {result['visual_embeddings'].numel() * 2 / 1e6:.1f} MB (fp16)")
        else:
            print("  (Vision tower hook did not fire — model merges internally)")
        torch.save(payload, embed_path)
        print(f"  Saved to: {embed_path}")

        print(f"\n=== Step 2: LLM service stub (ratio={ratio}) ===")
        print(f"  K tokens pre-allocated:  {result['K']}")
        print(f"  Answer from full model:  {result['answer']}")
        print(f"  Generate latency:        {result['generate_ms']:.0f} ms")
        print("  [STUB] In production: LLM service receives embedding payload,")
        print("         injects at multimodal token positions, generates response.")

        del model, processor
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("SUMMARY — Embedding service output sizes")
    print("=" * 70)
    print(f"{'Ratio':>6}  {'K tokens':>9}  {'Answer'}")
    print("-" * 40)
    for r in all_results:
        print(f"{r['ratio']:>6.2f}  {r['K']:>9}  {r['answer']}")

    base_K = next(r["K"] for r in all_results if r["ratio"] == 1.0)
    print("\nToken reduction vs dense baseline:")
    for r in all_results:
        if r["ratio"] < 1.0:
            print(f"  ratio={r['ratio']}: {r['K']}/{base_K} = {r['K']/base_K:.1%} of dense")

    print(f"\nEmbeddings saved to: {EMBED_DIR}/")
    print("\nNext steps:")
    print("  1. Implement a FastAPI server wrapping Step 1")
    print("  2. Implement a vLLM custom connector that injects saved embeddings")
    print("  3. Benchmark throughput vs end-to-end NVILA baseline")


if __name__ == "__main__":
    main()
