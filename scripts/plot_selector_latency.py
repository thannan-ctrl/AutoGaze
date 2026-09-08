"""Selection-only latency (SKIP_LLM=1) vs. frame count: AutoGaze's trained
selector vs. codec (windowed / sampled-only), per dataset.

Reads benchmark_results/nvila_hd_accuracy_breakdown_summary_{dataset}_nvf{N}_selectoronly_{variant}{N}.json
and plots avg_e2e_ms per nvf. See Codec_Selector_Feasibility.md, Finding 4.
"""
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_DIR, "benchmark_results")
FIGURES_DIR = os.path.join(REPO_DIR, "figures")

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"

SERIES = [
    ("autogaze", "autogaze{nvf}", "AutoGaze (trained selector)", "#eda100", "o"),
    ("codec", "windowedn25_{nvf}", "Codec, windowed", "#2a78d6", "s"),
    ("codec", "sampledonly{nvf}", "Codec, sampled-only", "#1baf7a", "^"),
]
NVFS = [16, 32, 64, 128, 256, 512, 1024]

# Points confirmed (not just "not yet run") to crash -- reproduced 3/3 with a real
# host-RAM OOM (~985GB RSS, see dmesg) right after model load, independent of
# batch size. Mark these explicitly; leave everything else as a plain gap.
CONFIRMED_OOM = {
    ("egoschema", "AutoGaze (trained selector)", 1024),
    ("video_mme", "AutoGaze (trained selector)", 1024),
    ("video_mme", "Codec, windowed", 1024),  # sampled-only completes fine at 1024 on both
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK_PRIMARY,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
})


def load_series(dataset: str, mode_key: str, variant_template: str):
    xs, ys = [], []
    for nvf in NVFS:
        variant = variant_template.format(nvf=nvf)
        path = os.path.join(
            RESULTS_DIR, f"nvila_hd_accuracy_breakdown_summary_{dataset}_nvf{nvf}_selectoronly_{variant}.json"
        )
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        entry = data.get(mode_key)
        if not entry:
            continue
        xs.append(nvf)
        ys.append(entry["avg_e2e_ms"] / 1000.0)
    return xs, ys


def plot_dataset(dataset: str, title: str, out_name: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    any_plotted = False
    for mode_key, variant_template, label, color, marker in SERIES:
        xs, ys = load_series(dataset, mode_key, variant_template)
        if not xs:
            continue
        any_plotted = True
        ax.plot(
            xs, ys, marker=marker, markersize=5.5, linewidth=2.0, color=color,
            label=label, solid_capstyle="round", zorder=3,
        )
        next_i = NVFS.index(xs[-1]) + 1
        if next_i < len(NVFS) and (dataset, label, NVFS[next_i]) in CONFIRMED_OOM:
            ax.plot(
                NVFS[next_i], ys[-1], marker="x", markersize=9, markeredgewidth=2.2,
                color=color, zorder=4,
            )
            ax.annotate(
                "OOM", (NVFS[next_i], ys[-1]), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8.5, color=color, fontweight="bold",
            )

    if not any_plotted:
        plt.close(fig)
        return

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(NVFS)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Sampled frames (nvf)", fontsize=10.5)
    ax.set_ylabel("Selection time per question (s)", fontsize=10.5)
    ax.set_title(title, fontsize=12.5, color=INK_PRIMARY, fontweight="bold", pad=12)

    ax.grid(True, which="major", axis="both", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", axis="y", color=GRID, linewidth=0.4, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)

    ax.tick_params(axis="both", labelsize=9.5, length=0)

    legend = ax.legend(
        loc="upper left", frameon=False, fontsize=9.5, handlelength=1.6, labelcolor=INK_SECONDARY,
    )

    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, out_name)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_dataset("egoschema", "Selection-only latency — EgoSchema", "selector_latency_egoschema.png")
    plot_dataset("video_mme", "Selection-only latency — VideoMME", "selector_latency_video_mme.png")
