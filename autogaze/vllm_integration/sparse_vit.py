"""
Sparse ViT encoding via gather op for AutoGaze × vLLM (Tasks 2 + 3).

Task 2 — Move AutoGaze pre-ViT:
    Select patches BEFORE transformer encoding rather than pruning post-ViT.
    Saves ViT compute proportional to the selection ratio.

Task 3 — Sparse ViT via gather op:
    After patch embedding (a cheap conv/linear), gather only the K selected
    patch embeddings, then run the transformer blocks on K tokens instead of N.

Architecture after patching:
    pixel_values (all N patches)
      │
      ├─ patch_embed (lightweight conv — still runs on all N)
      │
      ├─ [GATHER selected_idx]  ← Task 2+3: select K of N
      │
      ├─ transformer blocks     ← runs on K only (O(K²) attn vs O(N²))
      │
      └─ merger (2×2 → 1)       ← K → K_merged tokens
             ↓
         LLM (K_merged visual tokens)

vs. current post-ViT EVS:
    pixel_values (N)
      ├─ patch_embed → blocks (O(N²)) → merger (N_merged tokens)
      └─ EVS retention mask: keep K_merged of N_merged
             ↓
         LLM (K_merged tokens)

Speedup (at 50% selection):
    Attention FLOPs: ~4× reduction  (K² / N² = 0.25)
    FFN/MLP FLOPs:  ~2× reduction  (K / N = 0.5)

Architecture target:
    Qwen2.5/3-VL visual encoder (Qwen2VLVisionTransformer) in vLLM ≥ 0.24.
    Expected attributes: patch_embed, rot_pos_emb, blocks, merger.
    Falls back gracefully to dense ViT if architecture does not match.

Usage:
    # 1. Compute mask at ViT patch resolution (before merge)
    VIT_GRID_HW = (32, 32)  # 448px / 14px_patch = 32 patches/side
    mask_vit, K_vit = prep.compute_retention_mask(
        raw_frames, target_grid_hw=VIT_GRID_HW, gazing_ratio=0.5
    )
    MERGE_FACTOR = 4  # 2×2 spatial merge in Qwen3-VL
    K_merged = K_vit // MERGE_FACTOR

    # 2. Load vLLM model with video_pruning_rate so EVS hook is active
    llm = LLM(..., video_pruning_rate=0.5, enforce_eager=True)

    # 3. Patch visual encoder after model load
    patch_sparse_vit(llm)

    # 4. Run inference with both contexts:
    #    - SparseViTContext drives the pre-ViT gather op
    #    - AutoGazeContext(K=K_merged) drives adaptive K + identity retention mask
    with SparseViTContext(mask=mask_vit, K=K_vit, grid_thw=(T, 32, 32)):
        with AutoGazeContext(K=K_merged):   # identity retention mask (sparse-ViT pass-through)
            outputs = llm.chat(messages, ...)
"""
from __future__ import annotations

import os
import socket
import struct
import threading
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

_sparse_ctx = threading.local()
_PATCHED_ENCODERS: dict = {}   # id(encoder) → original_forward

# File used for cross-process mask communication.
# vLLM V1 runs the ViT in an EngineCore subprocess; thread-locals set in the
# main process are invisible there.  Writing the payload to this shared file
# (inside the same Docker container) lets _sparse_vit_forward read it.
_IPC_PATH = os.environ.get("SPARSE_VIT_IPC_PATH", "/tmp/_autogaze_sparse_vit_ctx.pt")

# Socket pair for cross-process ViT timing.
#
# Approach: create a connected Unix socket pair BEFORE LLM() forks the
# EngineCore subprocess.  After fork both processes share the same
# underlying sockets (file descriptors are inherited).  The child writes
# the timing float via the "send" socket; the parent reads it from the
# "recv" socket.  Unlike mmap, stdout-print, or filesystem approaches,
# socketpair() is unaffected by:
#   - vLLM's internal stdout/stderr redirection in the EngineCore
#   - mount-namespace isolation (AF_UNIX sockets are in the abstract
#     namespace, not the filesystem)
#   - Python 3.12 post-fork lock issues (no locking involved)
#
# Layout: _timing_socks = (recv_sock, send_sock)
#   parent → recv from index 0
#   child  → send  via index 1
_timing_socks: Optional[tuple] = None


def setup_timing_shm():
    """Create the socket pair for timing IPC (call once, BEFORE LLM())."""
    global _timing_socks
    if _timing_socks is None:
        recv_s, send_s = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        recv_s.setblocking(False)   # parent: non-blocking read
        _timing_socks = (recv_s, send_s)
    return _timing_socks

# ---------------------------------------------------------------------------
# CUDA event timing hook (used by runtime_analysis, independent of sparse ViT)
# ---------------------------------------------------------------------------
_vit_timing = threading.local()
_TIMING_ENCODERS: dict = {}    # id(encoder) → forward before timing wrap


def patch_vit_timing(llm) -> Optional[object]:
    """
    Wrap the visual encoder's current forward with CUDA event timing.

    Records start/end events around the encoder's full forward pass
    (including patch_embed + blocks + merger) so wall-clock ViT time
    can be separated from LLM decode time.

    Must be called AFTER patch_sparse_vit() so timing wraps the complete
    (possibly sparse) forward, not the original dense one.

    Returns the encoder module, or None if not found.
    """
    encoder = _find_visual_encoder(llm)
    if encoder is None:
        return None

    enc_id = id(encoder)
    if enc_id in _TIMING_ENCODERS:
        return encoder  # already wrapped

    current_forward = encoder.forward
    _TIMING_ENCODERS[enc_id] = current_forward

    def _timed_forward(*args, **kwargs):
        import torch as _torch
        start = _torch.cuda.Event(enable_timing=True)
        end = _torch.cuda.Event(enable_timing=True)
        start.record()
        out = current_forward(*args, **kwargs)
        end.record()
        _torch.cuda.synchronize()
        _vit_timing.ms = start.elapsed_time(end)
        return out

    encoder.forward = _timed_forward
    print(f"[TimingHook] CUDA event timing added to {type(encoder).__name__}.forward", flush=True)
    return encoder


def get_vit_ms() -> Optional[float]:
    """Return the ViT forward time (ms) from the last inference, or None."""
    # Fast path: thread-local (works when ViT runs in-process)
    ms = getattr(_vit_timing, "ms", None)
    if ms is not None:
        return ms
    # Subprocess path: read from socket pair written by _hook_forward in child.
    if _timing_socks is not None:
        try:
            data = _timing_socks[0].recv(8)
            if len(data) == 8:
                return struct.unpack("d", data)[0]
        except BlockingIOError:
            pass   # nothing written yet — child did not send timing
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class SparseViTContext:
    """
    Stores the pre-ViT patch selection mask for sparse encoding.

    Fields:
        mask   (total_vit_patches,) bool — True = selected patch
        K      int — number of True entries in mask
        grid_thw (T, H_vit, W_vit) — ViT patch grid dimensions (pre-merge)
    """

    def __init__(
        self,
        mask: torch.Tensor,
        K: int,
        grid_thw: Tuple[int, int, int],
    ):
        self.payload = {"mask": mask, "K": K, "grid_thw": grid_thw}

    def __enter__(self):
        _sparse_ctx.payload = self.payload
        # Write to file so the EngineCore subprocess can read it.
        # The mask tensor is already CPU; grid_thw is a plain tuple.
        try:
            torch.save(self.payload, _IPC_PATH)
        except Exception as e:
            print(f"[SparseViT] Warning: could not write IPC file {_IPC_PATH}: {e}", flush=True)
        return self

    def __exit__(self, *args):
        _sparse_ctx.payload = None
        try:
            if os.path.exists(_IPC_PATH):
                os.unlink(_IPC_PATH)
        except Exception:
            pass


def get_sparse_payload() -> Optional[dict]:
    # Fast path: thread-local (works when ViT runs in-process)
    p = getattr(_sparse_ctx, "payload", None)
    if p is not None:
        return p
    # Subprocess path: read from file written by main process
    if os.path.exists(_IPC_PATH):
        try:
            return torch.load(_IPC_PATH, weights_only=True)
        except Exception as e:
            print(f"[SparseViT] Warning: could not read IPC file {_IPC_PATH}: {e}", flush=True)
    return None


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def _find_visual_encoder(llm) -> Optional[object]:
    """
    Locate the visual encoder (ViT) inside a vLLM LLM object.

    Tries instance-level paths (vLLM ≤0.23 / single-process executor) and
    class-level paths (vLLM ≥0.24 V1 engine where the model runs in the
    EngineCore subprocess and is not reachable from the main process).

    For V1 (class-level), we patch the class's forward method directly so the
    patch is already in place when vLLM instantiates the model in EngineCore.
    Call patch_sparse_vit() BEFORE LLM(...) to take advantage of this.
    """
    # ── Instance-level discovery (vLLM ≤0.23, single-process executor) ────────
    raw_model = None
    try:
        driver = llm.llm_engine.model_executor.driver_worker
        runner = getattr(driver, "model_runner", None)
        if runner is None:
            runner = getattr(getattr(driver, "worker", None), "model_runner", None)
        if runner is not None:
            raw_model = runner.model
    except Exception as e:
        print(f"[SparseViT] Instance path unavailable ({e}), trying class-level.", flush=True)

    if raw_model is not None:
        for path in [
            "model.visual",
            "visual",
            "model.vision_model",
            "vision_model",
            "model.model.visual",
        ]:
            obj = raw_model
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                print(f"[SparseViT] Found visual encoder at '{path}': {type(obj).__name__}", flush=True)
                return obj
        top = [a for a in dir(raw_model) if not a.startswith("_")][:30]
        print(f"[SparseViT] Instance found but visual encoder not at known paths. Attrs: {top}", flush=True)

    # ── Class-level discovery (vLLM ≥0.24 V1, model in EngineCore subprocess) ─
    # Patch the class before the model is loaded so the patch is inherited by
    # the EngineCore process.  Caller must invoke patch_sparse_vit() BEFORE LLM().
    #
    # NOTE: as of vllm==0.26.0 (this environment), Qwen3VL ships its own vision
    # transformer class -- Qwen3_VisionTransformer -- distinct from the Qwen2.5VL
    # one. The two have different forward() signatures (see _sparse_vit_forward
    # docstring), so Qwen3_VisionTransformer must be tried first / matched
    # exactly, not treated as interchangeable with the Qwen2.5VL classes below.
    for cls_path in [
        ("vllm.model_executor.models.qwen3_vl",   "Qwen3_VisionTransformer"),
        ("vllm.model_executor.models.qwen3_vl",   "Qwen2_5VLVisionTransformer"),
        ("vllm.model_executor.models.qwen2_5_vl", "Qwen2_5VLVisionTransformer"),
        ("vllm.model_executor.models.qwen2_vl",   "Qwen2VLVisionTransformer"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(cls_path[0])
            cls = getattr(mod, cls_path[1], None)
            if cls is not None:
                print(
                    f"[SparseViT] Class-level encoder: {cls_path[0]}.{cls_path[1]}  "
                    f"(patch applied to class, active in EngineCore subprocess)",
                    flush=True,
                )
                return cls  # caller will patch cls.forward as an unbound method
        except ImportError:
            continue

    print("[SparseViT] Visual encoder class not found in known vLLM module paths.", flush=True)
    return None


# ---------------------------------------------------------------------------
# Sparse ViT forward (Task 3: gather op)
# ---------------------------------------------------------------------------

def _run_blocks_qwen3(
    blocks,
    deepstack_visual_indexes: list,
    deepstack_merger_list,
    hidden_states: torch.Tensor,
    encoder_metadata: dict,
) -> Tuple[torch.Tensor, list]:
    """
    Run Qwen3VL transformer blocks with the current (vllm==0.26.0) block
    signature: blk(hidden_states, cu_seqlens=, rotary_pos_emb_cos=,
    rotary_pos_emb_sin=, max_seqlen=, sequence_lengths=). Collects deepstack
    intermediate features at the configured layer indices, mirroring
    Qwen3_VisionTransformer.forward() exactly (vllm/model_executor/models/
    qwen3_vl.py lines ~800-841 in this environment's installed version).
    """
    deepstack_feature_lists = []
    for layer_num, blk in enumerate(blocks):
        hidden_states = blk(
            hidden_states,
            cu_seqlens=encoder_metadata["cu_seqlens"],
            rotary_pos_emb_cos=encoder_metadata["rotary_pos_emb_cos"],
            rotary_pos_emb_sin=encoder_metadata["rotary_pos_emb_sin"],
            max_seqlen=encoder_metadata["max_seqlen"],
            sequence_lengths=encoder_metadata.get("sequence_lengths"),
        )
        if layer_num in deepstack_visual_indexes:
            idx = deepstack_visual_indexes.index(layer_num)
            deepstack_feature_lists.append(deepstack_merger_list[idx](hidden_states))
    return hidden_states, deepstack_feature_lists


def _dense_from_embedded(
    encoder,
    hidden_states: torch.Tensor,
    grid_thw_list: list,
) -> torch.Tensor:
    """Dense ViT forward starting from already-embedded patches (skip patch_embed)."""
    encoder_metadata = encoder.prepare_encoder_metadata(grid_thw_list)
    hidden_states = hidden_states + encoder_metadata["pos_embeds"]
    hidden_states = hidden_states.unsqueeze(1)
    hidden_states, deepstack_feature_lists = _run_blocks_qwen3(
        encoder.blocks, encoder.deepstack_visual_indexes, encoder.deepstack_merger_list,
        hidden_states, encoder_metadata,
    )
    hidden_states = encoder.merger(hidden_states)
    return torch.cat([hidden_states] + deepstack_feature_lists, dim=1)


def _sparse_vit_forward(
    encoder,
    original_forward,
    payload: dict,
    *args,
    **kwargs,
) -> torch.Tensor:
    """
    Sparse ViT forward for Qwen3VL (vllm==0.26.0's Qwen3_VisionTransformer —
    see vllm/model_executor/models/qwen3_vl.py, class Qwen3_VisionTransformer,
    for the reference dense forward() this mirrors).

    Architecture this targets differs from the older Qwen2.5VL-style gather
    this function previously assumed:
      - Positional embeddings come from `fast_pos_embed_interpolate` (a
        bicubic-style interpolation over a learned (num_grid_per_side**2, D)
        table), not a `rot_pos_emb`-only scheme — must be gathered at the
        same selected indices as the patch embeddings.
      - Rotary embeddings are two separate tensors (`rotary_pos_emb_cos`,
        `rotary_pos_emb_sin`), not one combined tensor.
      - `cu_seqlens`/`sequence_lengths`/`max_seqlen` are backend-specific
        (computed via `MMEncoderAttention` static methods keyed off
        `encoder.attn_backend`) rather than a hand-built flash-attn cu_seqlens.
      - Deepstack: intermediate features at `encoder.deepstack_visual_indexes`
        each go through their own `deepstack_merger_list[i]`, then get
        concatenated onto the final merged output along the feature dim.

    Steps:
    1. patch_embed  — runs on ALL N patches (lightweight conv/linear)
    2. [GATHER]     — select K patches via AutoGaze mask (patch embeddings,
                       positional embeddings, and rotary cos/sin all gathered
                       at the same indices)
    3. blocks       — run transformer on K patches only  (Task 3)
    4. merger + deepstack — spatial merge + multi-scale feature concat

    NOT independently verified end-to-end in this environment: the installed
    vllm (0.26.0, this conda env) cannot load Qwen3VL at all here — importing
    vllm.model_executor.models.qwen3_vl fails with "vllm.vllm_flash_attn
    requires the CUDA flash attention extensions (_vllm_fa2_C or _vllm_fa3_C)",
    which are not built for this environment/arch (aarch64, torch 2.13+cu130).
    This rewrite was done by reading the installed vllm source directly (Read
    tool, not import) and matching its exact API. Before trusting any
    throughput/accuracy number from this path, run the numerical sanity check
    from the plan this replaces: compare sparse_vit output embeddings against
    dense on a small input where all patches are kept (should match near-
    exactly) — in an environment where vLLM's CUDA kernels actually load
    (e.g. the nvcr.io/nvidia/vllm:26.07-py3 container used for the original
    batch_throughput_test_results.md benchmarks).
    """
    mask: torch.Tensor = payload["mask"]       # (T*H*W,) bool CPU
    K: int = payload["K"]
    T, H_vit, W_vit = payload["grid_thw"]
    total_vit = T * H_vit * W_vit

    if not (hasattr(encoder, "patch_embed") and hasattr(encoder, "blocks")
            and hasattr(encoder, "prepare_encoder_metadata")):
        print(
            f"[SparseViT] {type(encoder).__name__} missing patch_embed/blocks/"
            "prepare_encoder_metadata — dense fallback (unsupported architecture).",
            flush=True,
        )
        return original_forward(*args, **kwargs)

    # Unpack positional args: (pixel_values, grid_thw)
    if args:
        pixel_values = args[0]
        grid_thw_arg = args[1] if len(args) > 1 else kwargs.get("grid_thw")
    else:
        pixel_values = kwargs.get("x") or kwargs.get("pixel_values")
        grid_thw_arg = kwargs.get("grid_thw")

    if pixel_values is None:
        return original_forward(*args, **kwargs)

    device = pixel_values.device
    mask = mask.to(device)
    grid_thw_list = [[T, H_vit, W_vit]]

    # ── Step 1: Patch embedding (all N patches — cheap) ───────────────────────
    hidden_states = encoder.patch_embed(
        pixel_values.to(device=device, dtype=encoder.dtype, non_blocking=True)
    )  # (actual_total, D)

    actual_total = hidden_states.shape[0]
    if actual_total != total_vit:
        print(
            f"[SparseViT] Patch count mismatch: got {actual_total}, "
            f"expected {total_vit} (T={T}×{H_vit}×{W_vit}). Dense fallback.",
            flush=True,
        )
        return _dense_from_embedded(encoder, hidden_states, grid_thw_arg or grid_thw_list)

    # ── Step 2: Dense encoder metadata for the FULL grid (pos_embeds + rotary
    #            cos/sin), computed via the model's own helper for correctness —
    #            then gathered at the selected indices below. ──────────────────
    dense_metadata = encoder.prepare_encoder_metadata(grid_thw_list)
    pos_embeds = dense_metadata["pos_embeds"]                 # (total_vit, D)
    rotary_cos = dense_metadata["rotary_pos_emb_cos"]          # (total_vit, pos_dim)
    rotary_sin = dense_metadata["rotary_pos_emb_sin"]          # (total_vit, pos_dim)

    # ── Step 3: GATHER — select K of N patch embeddings (Task 2+3) ───────────
    selected_idx = mask.nonzero(as_tuple=False).view(-1)  # (K,)

    if selected_idx.numel() == 0 or selected_idx.numel() != K:
        print(
            f"[SparseViT] Mask gives {selected_idx.numel()} patches, expected K={K}. Dense fallback.",
            flush=True,
        )
        return _dense_from_embedded(encoder, hidden_states, grid_thw_list)

    hidden_sparse = hidden_states[selected_idx]        # (K, D)
    pos_embeds_sparse = pos_embeds[selected_idx]        # (K, D)
    rotary_cos_sparse = rotary_cos[selected_idx]        # (K, pos_dim)
    rotary_sin_sparse = rotary_sin[selected_idx]        # (K, pos_dim)
    print(
        f"[SparseViT] Gather: {actual_total} → {K} patches "
        f"({K / actual_total * 100:.1f}% retained before ViT blocks)",
        flush=True,
    )

    # ── Step 4: cu_seqlens/sequence_lengths/max_seqlen for the SPARSE per-frame
    #            counts, via the same backend-specific helpers prepare_encoder_
    #            metadata uses (MMEncoderAttention), not a hand-rolled cu_seqlens
    #            — keeps this in sync with whatever attention backend is active. ─
    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        MMEncoderAttention,
    )
    import numpy as np

    mask_per_frame = mask.view(T, H_vit * W_vit)                      # (T, H*W)
    k_per_frame = mask_per_frame.sum(dim=1).cpu().numpy().astype(np.int32)  # (T,)
    cu_seqlens_np = np.concatenate([[0], np.cumsum(k_per_frame)]).astype(np.int32)

    sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(
        encoder.attn_backend, cu_seqlens_np, device
    )
    max_seqlen_val = MMEncoderAttention.compute_max_seqlen(encoder.attn_backend, cu_seqlens_np)
    cu_seqlens_sparse = MMEncoderAttention.maybe_recompute_cu_seqlens(
        encoder.attn_backend, cu_seqlens_np, encoder.hidden_size, encoder.tp_size, device,
        fp8_padded_hidden_size=encoder.fp8_padded_hidden_size,
    )
    sparse_metadata = {
        "cu_seqlens": cu_seqlens_sparse,
        "rotary_pos_emb_cos": rotary_cos_sparse,
        "rotary_pos_emb_sin": rotary_sin_sparse,
        "max_seqlen": torch.tensor(max_seqlen_val, dtype=torch.int32),
        "sequence_lengths": sequence_lengths,
    }

    # ── Step 5: Transformer blocks on K sparse patches (Task 3) ──────────────
    hidden_sparse = hidden_sparse + pos_embeds_sparse
    hidden_sparse = hidden_sparse.unsqueeze(1)
    hidden_sparse, deepstack_feature_lists = _run_blocks_qwen3(
        encoder.blocks, encoder.deepstack_visual_indexes, encoder.deepstack_merger_list,
        hidden_sparse, sparse_metadata,
    )

    # ── Step 6: Spatial merger + deepstack concat: K patches → K_merged tokens ─
    hidden_sparse = encoder.merger(hidden_sparse)
    hidden_sparse = torch.cat([hidden_sparse] + deepstack_feature_lists, dim=1)

    K_merged = hidden_sparse.shape[0]
    print(
        f"[SparseViT] Output: {actual_total} patches → {K} sparse → {K_merged} merged tokens "
        f"(vs dense {actual_total // (encoder.spatial_merge_unit)} merged tokens)",
        flush=True,
    )
    return hidden_sparse


# ---------------------------------------------------------------------------
# Patch / restore
# ---------------------------------------------------------------------------

def patch_sparse_vit(llm=None) -> Optional[object]:
    """
    Patch the vLLM visual encoder for sparse ViT encoding (Tasks 2+3).

    Two modes depending on vLLM version:

    Instance mode (vLLM ≤0.23, single-process executor):
        Call AFTER ``llm = LLM(...)`` — patches the specific encoder instance.

    Class mode (vLLM ≥0.24 V1 engine, model in EngineCore subprocess):
        Call BEFORE ``llm = LLM(...)`` — patches the encoder class so the
        patch is inherited when EngineCore instantiates the model.
        Pass llm=None: ``patch_sparse_vit()``

    The patch is a no-op when no SparseViTContext is active, so the same
    LLM instance can be used for both dense and sparse inference.

    Returns the patched encoder object (instance or class), or None on failure.
    """
    encoder = _find_visual_encoder(llm)
    if encoder is None:
        print("[SparseViT] Visual encoder not found — sparse ViT unavailable.", flush=True)
        return None

    enc_id = id(encoder)
    if enc_id in _PATCHED_ENCODERS:
        print("[SparseViT] Visual encoder already patched.", flush=True)
        return encoder

    import inspect
    is_class = inspect.isclass(encoder)

    if is_class:
        # Class-level patch: wrap the unbound forward method
        original_forward = encoder.forward

        def _patched_class_forward(self, *args, **kwargs):
            payload = get_sparse_payload()
            if payload is None:
                return original_forward(self, *args, **kwargs)
            bound_orig = original_forward.__get__(self, type(self))
            return _sparse_vit_forward(self, bound_orig, payload, *args, **kwargs)

        encoder.forward = _patched_class_forward
        _PATCHED_ENCODERS[enc_id] = original_forward
        print(
            f"[SparseViT] Class-level patch applied to {encoder.__name__}.forward "
            "(active in EngineCore subprocess when SparseViTContext is set).",
            flush=True,
        )
    else:
        # Instance-level patch
        original_forward = encoder.forward
        _PATCHED_ENCODERS[enc_id] = original_forward

        def _patched_instance_forward(*args, **kwargs):
            payload = get_sparse_payload()
            if payload is None:
                return original_forward(*args, **kwargs)
            return _sparse_vit_forward(encoder, original_forward, payload, *args, **kwargs)

        encoder.forward = _patched_instance_forward
        print(
            f"[SparseViT] Instance patch applied to {type(encoder).__name__}.forward "
            "(sparse when SparseViTContext active, dense otherwise).",
            flush=True,
        )

    return encoder


def install_import_hook() -> None:
    """
    Install a Python meta-path import hook that patches Qwen2_5VLVisionTransformer
    the moment its module is imported — in ANY process that inherits this hook.

    On Linux, vLLM forks the EngineCore subprocess after LLM() is called.
    The child inherits sys.meta_path, so when the child later imports
    vllm.model_executor.models.qwen3_vl it gets the patched class automatically,
    without needing any class-level patching from the parent after fork.

    Must be called BEFORE ``from vllm import LLM`` so that the fork occurs
    after the hook is installed.

    This is the reliable approach for vLLM V1 (EngineCore subprocess):
        import_hook installed → LLM() forks → child imports qwen3_vl
                                            → hook fires → class patched
                                            → instance.forward() uses patched class
    """
    import sys

    _TARGET_MODULES = {
        "vllm.model_executor.models.qwen3_vl":   "Qwen3_VisionTransformer",
        "vllm.model_executor.models.qwen2_5_vl": "Qwen2_5VLVisionTransformer",
        "vllm.model_executor.models.qwen2_vl":   "Qwen2VLVisionTransformer",
    }

    class _SparseViTImportHook:
        def find_module(self, fullname, path=None):
            return self if fullname in _TARGET_MODULES else None

        def load_module(self, fullname):
            import importlib
            # Avoid infinite recursion: remove self, import normally, re-add.
            sys.meta_path.remove(self)
            try:
                mod = importlib.import_module(fullname)
            finally:
                sys.meta_path.insert(0, self)

            # Patch the visual encoder class in-place
            cls_name = _TARGET_MODULES[fullname]
            cls = getattr(mod, cls_name, None)
            if cls is not None and id(cls) not in _PATCHED_ENCODERS:
                original_forward = cls.forward
                _PATCHED_ENCODERS[id(cls)] = original_forward

                def _hook_forward(self_enc, *args, **kwargs):
                    import torch as _torch
                    start = _torch.cuda.Event(enable_timing=True)
                    end = _torch.cuda.Event(enable_timing=True)
                    start.record()

                    payload = get_sparse_payload()
                    if payload is None:
                        result = original_forward(self_enc, *args, **kwargs)
                    else:
                        bound = original_forward.__get__(self_enc, type(self_enc))
                        result = _sparse_vit_forward(self_enc, bound, payload, *args, **kwargs)

                    end.record()
                    _torch.cuda.synchronize()
                    ms = start.elapsed_time(end)
                    _vit_timing.ms = ms

                    # Send timing via socket pair — the send socket is
                    # inherited across fork, bypasses stdout redirection
                    # and filesystem namespace isolation.
                    if _timing_socks is not None:
                        import struct as _struct
                        try:
                            _timing_socks[1].send(_struct.pack("d", ms))
                        except Exception:
                            pass
                    return result

                cls.forward = _hook_forward
                print(
                    f"[SparseViT] Import hook patched {cls_name}.forward "
                    f"(module={fullname})",
                    flush=True,
                )
            return mod

    # Only install once
    if not any(type(h).__name__ == "_SparseViTImportHook" for h in sys.meta_path):
        sys.meta_path.insert(0, _SparseViTImportHook())
        print("[SparseViT] Import hook installed — will patch on first qwen3_vl import.", flush=True)


def restore_sparse_vit(llm) -> None:
    """Restore the original visual encoder forward."""
    encoder = _find_visual_encoder(llm)
    if encoder is None:
        return
    enc_id = id(encoder)
    if enc_id in _PATCHED_ENCODERS:
        encoder.forward = _PATCHED_ENCODERS.pop(enc_id)
        print("[SparseViT] Restored original visual encoder forward.", flush=True)
