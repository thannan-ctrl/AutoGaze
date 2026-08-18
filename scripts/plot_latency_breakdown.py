"""Nsight-style stacked horizontal timeline: dense vs AutoGaze (batched)
latency breakdown, from benchmark_results/nvila_hd_accuracy_breakdown_summary_nvf16.json
(see README.md for how each bucket is measured).
"""
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_PATH = os.path.join(REPO_DIR, "benchmark_results", "nvila_hd_accuracy_breakdown_summary_nvf16.json")
OUT_PATH = os.path.join(REPO_DIR, "assets", "nvf16_summary_plots", "latency_breakdown_dense_vs_autogaze.png")

# Palette: fixed categorical order (validated adjacent-pair CVD-safe ordering)
STAGES = [
    ("avg_decode_ms", "Video decode (CPU)", "#2a78d6"),
    ("avg_image_preproc_ms", "Image preproc: tile/resize/SigLIP (CPU)", "#eb6834"),
    ("avg_autogaze_ops_ms", "AutoGaze ops: transform+bookkeeping (CPU)", "#1baf7a"),
    ("avg_autogaze_model_ms", "AutoGaze gazing model (GPU)", "#eda100"),
    ("avg_other_ms", "Other: tokenize/bookkeeping (CPU)", "#e87ba4"),
    ("avg_vit_ms", "NVILA ViT (GPU)", "#008300"),
    ("avg_llm_prefill_ms", "LLM prefill (GPU)", "#4a3aa7"),
    ("avg_llm_decode_ms", "LLM decode (GPU)", "#e34948"),
]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
GAP_LW_PT = 2.2  # surface-color divider between segments, in points (screen space, not data)
MIN_VISIBLE_FRAC = 0.008  # segments narrower than this fraction of max_total are drawn at
                          # this minimum width so they stay visible (e.g. dense's decode);
                          # the *next* segment still starts at the true cumulative x and is
                          # drawn on top, so it covers the overshoot -- true widths/totals
                          # are unaffected, only the tiny segment's own rendered sliver is
                          # widened for legibility.


def load_summary():
    with open(SUMMARY_PATH) as f:
        return json.load(f)


def main():
    summary = load_summary()
    modes = [("dense", "Dense"), ("autogaze", "AutoGaze (batched, max_batch_size_autogaze=64)")]

    fig, ax = plt.subplots(figsize=(11, 3.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    bar_h = 0.5
    y_positions = [1, 0]

    max_total = max(sum(summary[m][k] for k, _, _ in STAGES) for m, _ in modes)

    min_visible_ms = max_total * MIN_VISIBLE_FRAC

    for (mode, label), y in zip(modes, y_positions):
        s = summary[mode]
        x = 0.0
        segments = []  # (x_start, width, color) at TRUE proportional geometry
        for key, stage_label, color in STAGES:
            width = s[key]
            if width <= 0:
                continue
            segments.append((x, width, color))
            rect = FancyBboxPatch(
                (x, y - bar_h / 2), width, bar_h,
                boxstyle="round,pad=0,rounding_size=2",
                linewidth=0, facecolor=color, mutation_aspect=1,
            )
            ax.add_patch(rect)
            # Inline label only for segments wide enough to hold the text comfortably.
            if width > max_total * 0.09:
                ax.text(
                    x + width / 2, y, f"{width:,.0f}",
                    ha="center", va="center", fontsize=8.5, color="white", fontweight="bold",
                    zorder=5,
                )
            x += width

        # Segments too thin to see get a "boost" patch of fixed minimum width,
        # centered on their true midpoint, drawn in its own higher-zorder pass
        # (after all the true-geometry rects above) so it isn't immediately
        # painted over by the next segment. Purely a legibility overlay --
        # dividers/labels/totals below still use true widths.
        boosted = [width < min_visible_ms for _, width, _ in segments]
        for (seg_x, width, color), is_boosted in zip(segments, boosted):
            if not is_boosted:
                continue
            # Plain (unrounded) rectangle: FancyBboxPatch's corner rounding is a
            # fixed display-unit radius that swallows a patch this thin, collapsing
            # it to ~1px regardless of min_visible_ms.
            boost_x = max(seg_x + width / 2 - min_visible_ms / 2, 0.0)
            ax.add_patch(plt.Rectangle(
                (boost_x, y - bar_h / 2), min_visible_ms, bar_h,
                linewidth=0, facecolor=color, zorder=3,
            ))

        total = x
        # Surface-color divider lines between segments -- fixed screen-space width
        # (points), not data units, so thin segments aren't visually eaten by the gap.
        # Skipped around a boosted segment: the boost patch is only a few px wide,
        # and the divider's own linewidth would wipe most or all of it out.
        ymin_frac = (y - bar_h / 2 - (-0.65)) / 2.3
        ymax_frac = (y + bar_h / 2 - (-0.65)) / 2.3
        cum_x = 0.0
        for i, (seg_x, width, _) in enumerate(segments[:-1]):
            cum_x = seg_x + width
            if boosted[i] or boosted[i + 1]:
                continue
            ax.axvline(cum_x, ymin=ymin_frac, ymax=ymax_frac, color=SURFACE, linewidth=GAP_LW_PT, zorder=4)

        ax.text(
            total + max_total * 0.012, y, f"{total:,.0f} ms (Σ stages)",
            ha="left", va="center", fontsize=9.5, color=INK_PRIMARY, fontweight="bold",
        )
        ax.text(
            -max_total * 0.012, y, label,
            ha="right", va="center", fontsize=10.5, color=INK_PRIMARY, fontweight="bold",
        )

    ax.set_xlim(0, max_total * 1.16)
    ax.set_ylim(-0.65, 1.65)
    ax.set_yticks([])
    ax.set_xlabel("Latency per question (ms), avg over n=25 EgoSchema questions @ nvf=16", color=INK_SECONDARY, fontsize=9.5)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8.5)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    # matplotlib fills a multi-column legend column-major; reorder entries so the
    # rendered grid reads row-major left-to-right, matching stack order.
    ncol = 4
    nrow = -(-len(STAGES) // ncol)
    order = [r * ncol + c for c in range(ncol) for r in range(nrow) if r * ncol + c < len(STAGES)]
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=STAGES[i][2], edgecolor="none")
        for i in order
    ]
    legend_labels = [STAGES[i][1] for i in order]
    fig.legend(
        handles, legend_labels, loc="lower center", ncol=ncol, frameon=False,
        bbox_to_anchor=(0.5, -0.1), fontsize=8.5, labelcolor=INK_SECONDARY,
        handlelength=1.2, handleheight=1.2, columnspacing=1.3,
    )

    fig.suptitle(
        "NVILA-HD-Video preprocessing + LLM latency breakdown: Dense vs AutoGaze",
        fontsize=13, color=INK_PRIMARY, fontweight="bold", y=1.04,
    )
    fig.text(
        0.5, -0.16,
        "Bars sum the measured stages above; actual avg end-to-end (incl. batch_decode, misc python "
        f"overhead) was {summary['dense']['avg_e2e_ms']:,.0f} ms dense / {summary['autogaze']['avg_e2e_ms']:,.0f} ms "
        f"AutoGaze. Accuracy: {summary['dense']['accuracy']:.0%} dense / {summary['autogaze']['accuracy']:.0%} AutoGaze (n=25).",
        ha="center", va="top", fontsize=8, color=INK_MUTED,
    )

    fig.tight_layout(rect=[0.02, 0.14, 0.98, 0.92])
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
