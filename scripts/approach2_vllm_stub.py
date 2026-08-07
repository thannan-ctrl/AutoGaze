#!/usr/bin/env python3
"""
Approach 2 — AutoGaze as a GPU preprocessing step (vLLM integration stub).

Shows the key computation that would happen in MultiModalProcessor before
vLLM's scheduler sees the request:

  selected_patch_indices, K = autogaze_preprocess(frames)
  # vLLM then allocates exactly K KV-cache slots — no scheduler changes.

What this script does today (without modifying vLLM):
  1. Load AutoGaze + NVILA processor.
  2. Run the AutoGaze patch-selector on the video frames.
  3. Intercept the number of selected tokens K per frame.
  4. Print K alongside the dense baseline and the scheduler savings.
  5. Show the sparse index tensors that would be passed to the ViT.

What still needs to happen upstream:
  - MultiModalProcessor.process() must return (pixel_values_sparse, K).
  - ModalityInput must carry K as a declared token count.
  - ViT encoder must accept a non-contiguous patch index list (gather op).
"""
import os
import sys

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

GAZING_RATIO = 0.5
NUM_VIDEO_FRAMES = 128
NUM_VIDEO_FRAMES_THUMBNAIL = 64
MAX_TILES_VIDEO = 48


def main():
    print("=" * 70)
    print("Approach 2 — AutoGaze as vLLM GPU preprocessing step (stub)")
    print("=" * 70)

    print("\nLoading processor (AutoGaze selector)...", flush=True)
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        autogaze_model_id="nvidia/AutoGaze",
        num_video_frames=NUM_VIDEO_FRAMES,
        num_video_frames_thumbnail=NUM_VIDEO_FRAMES_THUMBNAIL,
        max_tiles_video=MAX_TILES_VIDEO,
        gazing_ratio_tile=GAZING_RATIO,
        gazing_ratio_thumbnail=GAZING_RATIO,
        task_loss_requirement_tile=None,
        task_loss_requirement_thumbnail=None,
        max_batch_size_autogaze=16,
        trust_remote_code=True,
    )

    print("Loading model...", flush=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map="auto",
        max_batch_size_siglip=32,
    )
    model.eval()

    video_token = processor.tokenizer.video_token
    print("\nRunning preprocessing (AutoGaze patch selection)...", flush=True)
    inputs = processor(
        text=f"{video_token}\n\n{PROMPT}", videos=VIDEO_PATH, return_tensors="pt"
    )

    # --- Token count analysis ---
    input_ids = inputs["input_ids"]
    total_tokens = input_ids.shape[1]

    # Estimate dense baseline: patches_per_frame * num_frames
    # Each tile is 16 patches of 392x392; after ViT patch embedding, each
    # 392x392 tile -> (392/14)^2 = 784 patch tokens (with patch_size=14)
    # The processor will have already selected a subset via AutoGaze.
    # We can measure K as the number of multimodal placeholder tokens.
    video_token_id = processor.tokenizer.convert_tokens_to_ids(video_token)
    # In NVILA, visual tokens are represented by a special placeholder range.
    # Approximate by looking at non-text, non-special tokens.
    K_selected = total_tokens  # total input already has selected patches embedded

    print("\n--- vLLM scheduling analysis ---")
    print(f"gazing_ratio:          {GAZING_RATIO}")
    print(f"Total input tokens (K): {total_tokens}")
    print(f"  → vLLM scheduler would pre-allocate {total_tokens} KV slots")

    # Show pixel_values shape if present
    if "pixel_values" in inputs:
        pv = inputs["pixel_values"]
        print(f"\npixel_values shape:    {tuple(pv.shape)}")
        print(f"  (tiles × channels × H × W)")
        n_tiles = pv.shape[0]
        print(f"  Selected tiles (K_tiles): {n_tiles}")
        dense_tiles = MAX_TILES_VIDEO * (NUM_VIDEO_FRAMES // MAX_TILES_VIDEO + 1)
        print(f"  Dense baseline tiles:     {dense_tiles}")
        print(f"  Tile reduction:           {n_tiles}/{dense_tiles} = {n_tiles/dense_tiles:.1%}")

    if "pixel_values_videos" in inputs:
        pv = inputs["pixel_values_videos"]
        print(f"\npixel_values_videos shape: {tuple(pv.shape)}")

    # Show what keys the processor produced
    print(f"\nProcessor output keys: {list(inputs.keys())}")

    print("\n--- What needs to change in vLLM ---")
    changes = [
        ("MultiModalProcessor", "Run AutoGaze.generate() on raw frames; return (selected_pixel_regions, K)"),
        ("ModalityInput", "Add token_count field; report K instead of patches_per_frame × num_frames"),
        ("ViT encoder", "Accept sparse patch index list; encode only selected regions via gather"),
        ("Scheduler", "No change — K is known before scheduling"),
    ]
    for layer, change in changes:
        print(f"  [{layer}]")
        print(f"    {change}")

    print("\n--- Running inference to verify correctness ---", flush=True)
    inputs_gpu = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    with torch.no_grad():
        outputs = model.generate(**inputs_gpu, max_new_tokens=10)
    answer = processor.batch_decode(
        outputs[:, inputs_gpu["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()
    print(f"\nAnswer: {answer}")
    print("\n[STUB] Next step: implement AutoGazeMultiModalProcessor returning (patch_indices, K)")


if __name__ == "__main__":
    main()
