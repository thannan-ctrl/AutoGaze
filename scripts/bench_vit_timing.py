#!/usr/bin/env python3
"""
Standalone ViT timing benchmark — dense vs sparse forward pass.

Loads Qwen2.5/3-VL visual encoder directly from HuggingFace (no vLLM, no
subprocess).  CUDA events run in the main process so there is no IPC issue.

Prints:
    VIT_BENCH_JSON:<json>

Usage (inside the vLLM Docker container or any env with transformers+cuda):
    python3 scripts/bench_vit_timing.py \
        --model Qwen/Qwen3-VL-2B-Instruct \
        --k-vit 264 \
        --n-frames 3 \
        --reps 10 --warmup 3
"""
import argparse
import json
import os
import sys
import statistics

REPO_DIR = os.environ.get("REPO_DIR", "/workspace/AutoGaze")
HF_HOME  = os.environ.get("HF_HOME",  "/root/.cache/huggingface")
sys.path.insert(0, REPO_DIR)
os.environ["HF_HOME"] = HF_HOME


def _ensure_deps():
    for pkg, imp in [("timm", "timm"), ("omegaconf", "omegaconf"),
                     ("wandb", "wandb"), ("loguru", "loguru"), ("av", "av")]:
        try:
            __import__(imp)
        except ImportError:
            import subprocess as _sp
            _sp.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)


def _cuda_time(fn, n_warmup: int, n_reps: int) -> list[float]:
    import torch
    for _ in range(n_warmup):
        with torch.no_grad():
            fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(n_reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--k-vit",   type=int, default=264,
                        help="Sparse K (selected ViT patches) from AutoGaze run")
    parser.add_argument("--n-frames",type=int, default=3,
                        help="Number of video frames (sets N = n_frames * 28 * 28)")
    parser.add_argument("--h-patches",type=int, default=28)
    parser.add_argument("--w-patches",type=int, default=28)
    parser.add_argument("--patch-size",type=int, default=14,
                        help="ViT patch size in pixels (14 for Qwen2.5/3-VL)")
    parser.add_argument("--reps",    type=int, default=10)
    parser.add_argument("--warmup",  type=int, default=3)
    args = parser.parse_args()

    _ensure_deps()

    import torch
    import torch.nn.functional as F

    T, H, W = args.n_frames, args.h_patches, args.w_patches
    N = T * H * W
    K = min(args.k_vit, N)
    px = args.patch_size

    print(f"[bench_vit] Loading encoder from {args.model} ...", flush=True)
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    encoder = model.visual
    encoder.eval()
    print(f"[bench_vit] Encoder loaded. N={N} ({T}×{H}×{W} patches), K={K} "
          f"(K/N={K/N:.3f})", flush=True)

    # Dummy pixel_values: (N, 3, px, px) — same format as vLLM feeds the encoder
    pv = torch.randn(N, 3, px, px, dtype=torch.bfloat16, device="cuda") * 0.5
    grid_thw = torch.tensor([[T, H, W]], dtype=torch.int32, device="cuda")

    # ── Dense forward ─────────────────────────────────────────────────────────
    print(f"[bench_vit] Timing dense ViT (N={N}) ...", flush=True)
    dense_times = _cuda_time(
        lambda: encoder(pv, grid_thw=grid_thw),
        args.warmup, args.reps,
    )

    # ── Sparse forward (gather op) ────────────────────────────────────────────
    # Simulate AutoGaze mask: select K patches deterministically
    torch.manual_seed(42)
    selected = torch.randperm(N, device="cuda")[:K].sort().values

    # Mask per-frame for cu_seqlens (even split across frames for synthetic mask)
    k_per_frame = torch.full((T,), K // T, dtype=torch.int32, device="cuda")
    k_per_frame[:K % T] += 1
    cu_seqlens = F.pad(k_per_frame.cumsum(0, dtype=torch.int32), (1, 0))

    def sparse_fn():
        # Step 1: patch embedding on ALL N (cheap conv)
        hidden = encoder.patch_embed(pv)          # (N, D)
        # Step 2: gather K
        sparse = hidden[selected]                  # (K, D)
        # Step 3: rotary pos emb for selected positions
        rot = encoder.rot_pos_emb(grid_thw) if hasattr(encoder, "rot_pos_emb") else None
        rot_s = (rot[selected] if rot is not None and rot.dim() == 2
                 else rot[:, selected, :] if rot is not None
                 else None)
        # Step 4: transformer blocks on K tokens
        for blk in encoder.blocks:
            try:
                sparse = blk(sparse, cu_seqlens=cu_seqlens,
                             rotary_pos_emb=rot_s)
            except TypeError:
                sparse = blk(sparse)
        # Step 5: merger
        if hasattr(encoder, "merger"):
            sparse = encoder.merger(sparse)
        return sparse

    print(f"[bench_vit] Timing sparse ViT (K={K}) ...", flush=True)
    sparse_times = _cuda_time(sparse_fn, args.warmup, args.reps)

    # ── Results ───────────────────────────────────────────────────────────────
    def _stats(times):
        return {
            "min":    min(times),
            "median": statistics.median(times),
            "mean":   statistics.mean(times),
            "max":    max(times),
        }

    dense_stats  = _stats(dense_times)
    sparse_stats = _stats(sparse_times)
    speedup      = dense_stats["median"] / sparse_stats["median"]
    attn_theory  = (N / K) ** 2
    ffn_theory   = N / K

    print(f"\n{'='*60}")
    print(f"ViT Standalone Benchmark  —  {args.model}")
    print(f"{'='*60}")
    print(f"  N={N} ({T}×{H}×{W})   K={K}   K/N={K/N:.3f}")
    print(f"  Reps={args.reps}  Warmup={args.warmup}")
    print()
    print(f"  Dense  ViT:  median={dense_stats['median']:.0f} ms  "
          f"[{dense_stats['min']:.0f}–{dense_stats['max']:.0f}]")
    print(f"  Sparse ViT:  median={sparse_stats['median']:.0f} ms  "
          f"[{sparse_stats['min']:.0f}–{sparse_stats['max']:.0f}]")
    print(f"  Measured speedup:       {speedup:.2f}×")
    print(f"  Theoretical attn (N/K)²: {attn_theory:.1f}×")
    print(f"  Theoretical FFN  (N/K):  {ffn_theory:.1f}×")
    print(f"{'='*60}")

    result = {
        "model":         args.model,
        "N":             N, "K": K, "K_over_N": K / N,
        "n_frames":      T, "h_patches": H, "w_patches": W,
        "reps":          args.reps,
        "dense":         dense_stats,
        "sparse":        sparse_stats,
        "speedup":       speedup,
        "attn_theory":   attn_theory,
        "ffn_theory":    ffn_theory,
    }
    print("VIT_BENCH_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
