# Pipeline walkthrough

End-to-end code trace from raw video to LLM answer.

---

## Why two environments

AutoGaze requires `transformers 4.x`; vLLM requires `transformers 5.x`. They conflict,
so the pipeline splits across two processes with a file as the handoff:

```
auto_gaze conda env                     vLLM Docker container
(transformers 4.x)    ag_mask_vit.pt   (transformers 5.x)
  AutoGaze model    ──────────────────►  Qwen3-VL + sparse ViT
```

---

## Step 1 — AutoGaze preprocessing

**Script:** `scripts/run_autogaze_preprocess.py`  
**Runs in:** `auto_gaze` conda env, outside Docker

```python
prep = AutoGazePreprocessor.load("nvidia/AutoGaze")
mask, K = prep.compute_retention_mask(
    raw_frames,               # (T, C, H, W) float32 [0,1]
    target_grid_hw=(32, 32),  # ViT patch grid: 448px / 14px patch = 32/side
    gazing_ratio=0.245,
    seed=42,
)
torch.save({"mask": mask, "K": K}, "/tmp/ag_mask_vit.pt")
```

Three sub-steps inside `compute_retention_mask` (`autogaze_preprocess.py:143`):

**1a. Resize to 224×224** (`preprocess_frames`, line 71)

AutoGaze's input resolution is fixed at 224×224 regardless of original video size.

```python
frames = F.interpolate(raw_frames, size=(224, 224), mode="bilinear")
frames = (frames - IMG_MEAN) / IMG_STD   # ImageNet normalize
```

**1b. Run AutoGaze model** (`run_autogaze`, line 94)

```python
torch.manual_seed(seed)   # reproducible K
output = self.model.forward(
    inputs={"video": video},    # (1, T, C, 224, 224)
    gazing_ratio=gazing_ratio,
    generate_only=True,
)
mask_224 = output["gazing_mask"][-1]   # (1, T, 196)  — 14×14 per frame
```

AutoGaze is a `ShallowVideoConvNet` encoder + 4-layer LLaMA decoder. It generates
gaze positions **autoregressively** — for each frame it emits patch indices one at a time
and stops when its `task_loss_prediction_head` is confident the selected set is sufficient.
This is why **K varies per frame and per video**, not just per `gazing_ratio`.

**1c. Upsample 14×14 → 32×32** (lines 174–186)

AutoGaze's native output is 14×14 (224px / 16px patch). Qwen3-VL's ViT uses 32×32
(448px / 14px patch). Bilinear upsample bridges the grids:

```python
ag_mask_resized = F.interpolate(ag_mask_14, size=(32, 32), mode="bilinear")
retention_mask = (ag_mask_resized > 0.5).cpu()  # (T, 32, 32) bool
retention_mask_flat = retention_mask.view(-1)    # (T*32*32,)  saved to .pt
K = retention_mask_flat.sum().item()             # number of True entries
```

---

## Step 2 — vLLM startup

**Script:** `scripts/worker.py`  
**Runs in:** vLLM Docker container

Three hooks are installed in order before the model loads.

### Hook A — patch the two EVS functions (`patch.py:29`)

```python
from autogaze.vllm_integration.patch import apply_autogaze_patch
apply_autogaze_patch(mode="autogaze")
```

This replaces two functions in `vllm.multimodal.evs` at the module attribute level:

```python
# patch.py lines 54–59
_evs_module.compute_retention_mask        = autogaze_retention_mask
_evs_module.compute_retained_tokens_count = autogaze_retained_tokens_count
```

Module-level replacement is the only hook point — vLLM calls both through the module
reference, so swapping the attribute redirects all future calls.

### Hook B — patch the ViT class (`sparse_vit.py:patch_sparse_vit`)

Must run **after** `from vllm import LLM` (so vLLM's lazy module imports resolve) but
**before** `LLM(...)` (so the class is patched before the subprocess instantiates it).

```python
from vllm import LLM, SamplingParams
from autogaze.vllm_integration.sparse_vit import patch_sparse_vit
patch_sparse_vit(llm=None)   # None → class-level, not instance-level
```

With `llm=None`, `_find_visual_encoder` skips instance-level discovery and imports
the encoder class directly (`sparse_vit.py:191`):

```python
mod = importlib.import_module("vllm.model_executor.models.qwen3_vl")
cls = mod.Qwen2_5VLVisionTransformer
```

`patch_sparse_vit` then wraps `cls.forward` as an unbound method (`sparse_vit.py:287`):

```python
original_forward = encoder.forward   # encoder is the class

def _patched_class_forward(self, *args, **kwargs):
    payload = get_sparse_payload()          # check thread-local SparseViTContext
    if payload is None:
        return original_forward(self, *args, **kwargs)   # no-op when inactive
    return _sparse_vit_forward(self, bound_orig, payload, *args, **kwargs)

encoder.forward = _patched_class_forward
```

When vLLM's `EngineCore` subprocess later instantiates `Qwen2_5VLVisionTransformer`,
that instance inherits the patched `forward` through Python's normal class → instance
method lookup. This is why the patch must be on the class, not the instance: vLLM ≥0.24
runs the model in a separate subprocess that the main process cannot reach.

### LLM creation

```python
llm = LLM(
    model="Qwen/Qwen3-VL-2B-Instruct",
    video_pruning_rate=0.245,   # activates the EVS hook path in vLLM
    enforce_eager=True,         # required: dynamic K breaks CUDA graph capture
)
```

`video_pruning_rate` is the flag that tells vLLM to call both EVS hooks during
preprocessing. Without it, neither patched function is ever invoked.

---

## Step 3 — Inference

Context managers write to thread-locals that the hooks read during the forward pass:

```python
with SparseViTContext(mask=mask_vit, K=K_vit, grid_thw=(6, 32, 32)):
    # _sparse_ctx.payload = {"mask": ..., "K": 1326, "grid_thw": (6, 32, 32)}
    with AutoGazeContext(ag_mask=None, K=K_merged):   # K_merged = K_vit // 4 = 331
        # _ctx.payload = {"ag_mask": None, "K": 331}
        outputs = llm.chat(messages, sampling_params=sampling)
```

Inside `llm.chat()`, vLLM calls the hooks in this sequence:

### Call 1 — `compute_retained_tokens_count` (slot pre-allocation)

vLLM calls this **before** running the model to know how many KV-cache slots to reserve.
Our replacement reads K directly from context (`retention.py:107`):

```python
stored = get_raw_frames()                      # reads _ctx.payload
if stored is not None and stored.get("K"):
    return stored["K"]                         # returns 331
```

vLLM now allocates exactly 331 slots instead of whatever the formula would give.

### Call 2 — `Qwen2_5VLVisionTransformer.forward` (the ViT)

Our class-level patch intercepts this and calls `_sparse_vit_forward`
(`sparse_vit.py:232`):

```python
# 1. patch_embed on ALL N patches — cheap conv, negligible cost
hidden_states = encoder.patch_embed(pixel_values)   # (6144, D)

# 2. rotary position embeddings for all positions
rotary_pos_emb = encoder.rot_pos_emb(grid_thw_arg) # (6144, pos_dim)

# 3. GATHER — select K of N using the AutoGaze mask
selected_idx   = mask.nonzero().view(-1)            # (1326,)
hidden_sparse  = hidden_states[selected_idx]         # (1326, D)
rotary_sparse  = rotary_pos_emb[selected_idx]        # (1326, pos_dim)

# 4. recompute cu_seqlens for flash-attention (per-frame counts changed)
k_per_frame = mask.view(T, H*W).sum(dim=1).int()    # different per frame
cu_seqlens  = F.pad(k_per_frame.cumsum(0), (1, 0))  # (T+1,)

# 5. transformer blocks on K patches only
#    attention: O(K²) = O(1326²)  vs  dense O(N²) = O(6144²)  → 4.6× cheaper
hidden_sparse = _run_blocks(encoder.blocks, hidden_sparse, cu_seqlens, rotary_sparse)

# 6. spatial merger: K patches → K/4 merged tokens
hidden_sparse = encoder.merger(hidden_sparse)       # (331, hidden_dim)
```

`patch_embed` still runs on all 6,144 patches because it is a single Conv2D — its cost
is negligible. Every transformer block (self-attention + FFN) only touches the 1,326
selected patches.

### Call 3 — `compute_retention_mask` (post-ViT selection)

Normally EVS calls this to prune the ViT output. In sparse_vit mode the ViT already
output only 331 tokens, so this returns an identity mask (`retention.py:242`):

```python
# Case A: ag_mask is None and K is set → sparse ViT pass-through
if ag_mask is None and K_stored is not None:
    K_actual = video_embeds.shape[0]    # 331 — ViT already selected
    return torch.ones(K_actual, dtype=torch.bool, device=video_embeds.device)
```

All 331 tokens pass to the LLM, which generates the answer.

---

## What each file does

| File | Change | Why |
|---|---|---|
| `autogaze_preprocess.py` | Runs AutoGaze, produces `(T*H*W,)` bool mask | AutoGaze needs transformers 4.x — must run outside Docker |
| `patch.py` | Replaces two `vllm.multimodal.evs` functions at module level | vLLM calls both through the module; attribute swap is the only hook point |
| `retention.py:autogaze_retained_tokens_count` | Returns K from context instead of fixed formula | vLLM pre-allocates KV-cache slots before the ViT runs — wrong K crashes or wastes memory |
| `retention.py:autogaze_retention_mask` | Identity in sparse_vit mode; AutoGaze bool mask in post-ViT mode | In sparse_vit mode the ViT already selected K tokens — no further pruning |
| `sparse_vit.py:patch_sparse_vit` | Wraps `Qwen2_5VLVisionTransformer.forward` at class level | vLLM ≥0.24 runs ViT in subprocess — instance patching doesn't reach it; class patching does |
| `sparse_vit.py:_sparse_vit_forward` | patch_embed(all N) → gather K → blocks(K) → merger | This is where FLOPs are saved: O(K²) attention instead of O(N²) |
