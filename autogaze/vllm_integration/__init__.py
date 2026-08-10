"""
AutoGaze × vLLM integration package.

Patches vLLM's EVS (Efficient Video Sampling) hooks with AutoGaze's learned
patch selection, enabling better-quality token reduction at the same ratio.

Task 1 — Adaptive K (per-video budget):
    apply_autogaze_patch() now also patches compute_retained_tokens_count so vLLM
    allocates the correct number of KV-cache slots for each video's AutoGaze K,
    rather than the same fixed formula across all videos.

Tasks 2+3 — Sparse ViT encoding:
    patch_sparse_vit(llm) patches the visual encoder to run the transformer blocks
    only on K selected patch embeddings (gathered after patch_embed), saving
    O(K²/N²) attention FLOPs and O(K/N) FFN FLOPs at inference time.

Usage — post-ViT AutoGaze (Tasks 1):
    from autogaze.vllm_integration.patch import apply_autogaze_patch
    from autogaze.vllm_integration.retention import AutoGazeContext

    apply_autogaze_patch(mode="autogaze")   # before LLM(...)
    llm = LLM(..., video_pruning_rate=0.5)

    with AutoGazeContext(ag_mask=mask, K=K):
        outputs = llm.chat(messages)

Usage — sparse ViT (Tasks 1+2+3):
    from autogaze.vllm_integration.patch import apply_autogaze_patch
    from autogaze.vllm_integration.retention import AutoGazeContext
    from autogaze.vllm_integration.sparse_vit import SparseViTContext, patch_sparse_vit

    apply_autogaze_patch(mode="autogaze")   # before LLM(...)
    llm = LLM(..., video_pruning_rate=0.5, enforce_eager=True)
    patch_sparse_vit(llm)                   # after LLM(...)

    with SparseViTContext(mask=mask_vit, K=K_vit, grid_thw=(T, 32, 32)):
        with AutoGazeContext(K=K_merged):   # ag_mask=None → identity retention
            outputs = llm.chat(messages)
"""
