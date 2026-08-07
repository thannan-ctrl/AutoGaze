"""
AutoGaze × vLLM integration package.

Patches vLLM's EVS (Efficient Video Sampling) with AutoGaze's learned
patch selection, enabling better quality at the same compression ratio.

Usage:
    from autogaze.vllm_integration.patch import apply_autogaze_patch
    apply_autogaze_patch(mode="magnitude")   # proof-of-concept
    apply_autogaze_patch(mode="autogaze")    # full learned model (requires AutoGaze weights)
"""
