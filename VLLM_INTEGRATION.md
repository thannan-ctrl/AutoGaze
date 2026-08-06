# AutoGaze × vLLM Integration — Design Brainstorm

## Problem statement

AutoGaze selects a _dynamic_ number of visual patches per frame at inference time.
vLLM's scheduler requires the token count to be known **before** it schedules a request, so it can pre-allocate KV-cache slots.
If the token count changes mid-flight (after multimodal embeddings are computed), vLLM crashes.

This is the single blocking constraint for a native integration.

---

## Approach 1 — Fixed compression rate (EVS-style) ✅ lowest friction

Run AutoGaze with a CLI-configurable `gazing_ratio` (e.g. `0.5`) applied uniformly to every frame.

```
token_count = int(gazing_ratio × patches_per_frame) × num_frames
```

The multimodal processor declares this count upfront; vLLM allocates accordingly.

**Pro**
- Zero scheduler changes; mirrors how EVS was upstreamed.
- Immediate, auditable baseline to measure benefit vs. dense tokens.

**Con**
- Loses the per-frame adaptivity that is AutoGaze's main value proposition (Mike: "more tokens is essentially always better in aggregate").
- Still adds AutoGaze model latency to every request.

**Verdict**: Use this _only_ as a benchmarking baseline, not a production target.

---

## Approach 2 — AutoGaze as a GPU preprocessing step 🏆 recommended starting point

Run the AutoGaze patch-selector during multimodal preprocessing (already runs on a GPU worker in the right vLLM flow) and return the selected patch indices alongside their count.

```
Video frames
  └─► AutoGaze patch-selector   (GPU, preprocessing worker)
        └─► selected_patch_indices, token_count=K   ← known before scheduling
              └─► vLLM scheduler allocates K slots
                    └─► ViT encodes only the K selected patches
                          └─► LLM generates
```

**What needs to change in vLLM**

| Layer | Change |
|---|---|
| `MultiModalProcessor` | Run `AutoGazeModel.generate()` on raw frames; return `(selected_pixel_regions, K)` |
| `ModalityInput` / placeholder counting | Report `K` tokens instead of `patches_per_frame × num_frames` |
| ViT encoder | Accept a non-contiguous patch index list; encode only selected regions |
| Scheduler | No change — token count is known before scheduling |

**Subtlety**: The ViT must accept a sparse index list, not a dense grid.
Most HuggingFace ViTs can be patched to skip unselected patch embeddings with a gather op.

**Pro**
- Preserves full dynamic adaptivity; the budget is just determined _earlier_.
- Doesn't require scheduler changes.
- GPU preprocessing precedent already exists (hardware video decode, keep-frames-on-GPU).
- AutoGaze patch-selector is small (~lightweight LLaMA decoder); runs fast relative to the main ViT.

**Con**
- Requires the AutoGaze model to be co-resident with the preprocessing worker.
- Adds a new GPU model to the preprocessing stage — conceptually unfamiliar to vLLM maintainers.
- ViT must be modified to encode only selected patches.

---

## Approach 3 — Per-video fixed budget with dynamic per-frame allocation

Run AutoGaze once per video to determine total token count `K_video` (sum across frames), then tell the scheduler `K_video`.
Within the video, frames can still get different numbers of tokens.

```
K_video = sum(num_gazing_each_frame)   # from AutoGaze.generate()
```

This gives vLLM a single fixed number to pre-allocate while preserving AutoGaze's adaptive per-frame distribution.

**Pro**
- Single pre-computed number satisfies the scheduler.
- Retains the per-frame adaptive nature (the first frame can get 198 tokens, subsequent frames 10 each).

**Con**
- `K_video` still varies across videos — scheduler must handle variable-length multimodal sequences (which it already does, just not dynamically).
- Requires full AutoGaze forward pass before scheduling.
- For batched inference, each video has a different `K_video`, complicating batching.

---

## Approach 4 — Two-process split (AutoGaze service + vLLM)

Run a separate AutoGaze microservice that accepts raw video frames and returns compressed token embeddings.
vLLM receives pre-computed embeddings of known length `K` and treats them like any fixed-length multimodal input.

```
Client → AutoGaze service (GPU) → compressed embeddings [K × D]
                                          ↓
                               vLLM receives K embedding tokens
                               (no video, no ViT, just embeddings)
```

**Pro**
- Zero vLLM changes — embeddings arrive like any tensor input.
- AutoGaze service can be scaled independently.
- Can be validated as a "frankenstein" prototype immediately without upstream PRs.

**Con**
- Doubles GPU memory: AutoGaze service holds its own model + ViT; vLLM holds the LLM.
- Network/IPC overhead for embedding tensors (mitigated if co-located on same node).
- Requires the downstream LLM to understand sparse-patch embedding representations.

---

## Recommended experiment plan

### Phase 1 — Prove the benefit (no vLLM changes)

Use the two-process split (Approach 4) as a prototype:

1. Wrap `scripts/demo_quickstart.py` into an embedding server.
2. Pass compressed embeddings to an off-the-shelf VLM (e.g. Qwen2-VL or LLaVA-Next).
3. Measure: accuracy on captioning / VQA benchmarks at `gazing_ratio ∈ {0.25, 0.5, 0.75, 1.0}`.
4. Measure: wall-clock latency and throughput vs. dense-token baseline.

**Key question**: Does the quality–compute tradeoff justify the extra AutoGaze pass?

### Phase 2 — Upstream to vLLM (Approach 2)

If Phase 1 shows a net win:

1. Open a vLLM RFC: "GPU-side patch selection as a preprocessing hook."
2. Implement `AutoGazeMultiModalProcessor` that returns `(patch_indices, K)`.
3. Patch the target ViT to accept a sparse index list.
4. Propose `token_count` as a field in `ModalityInput` returned by processors.

**Talking points for vLLM maintainers**
- Token count is known before scheduling — no scheduler changes.
- Follows existing GPU preprocessing precedent (hardware decode, on-GPU frame retention).
- AutoGaze selector is cheap; its cost is amortized against ViT + LLM savings.
- Fixed-ratio mode (Approach 1) is available as a fallback for simplicity.

---

## Open questions

| Question | Notes |
|---|---|
| Does Qwen2-VL / LLaVA handle sparse patch embeddings gracefully? | Need to test; positional embeddings may assume dense grid |
| Can AutoGaze selector run on CPU for CPU-only preprocessing workers? | Selector is a small LLaMA; CPU inference is slow but feasible for offline use |
| What is the breakeven latency? | AutoGaze adds ~Xms; ViT savings must exceed this |
| Does `task_loss_requirement` provide a better knob than `gazing_ratio`? | Likely yes — quality-driven stopping is more principled than ratio-driven |
| Multi-image / interleaved video support? | AutoGaze is per-video; multi-turn or interleaved inputs need per-clip runs |
