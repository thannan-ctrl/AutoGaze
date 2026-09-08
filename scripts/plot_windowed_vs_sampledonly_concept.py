"""Concept diagram: how codec mode's two encoding strategies differ.

Windowed (_extract_and_encode_windows): for each sampled frame, decodes it plus
WINDOW=4 real, temporally-adjacent frames before it, and encodes that 5-frame
window on its own (forced I-frame at the window start) -- motion vectors come
from genuine local motion, but ~5x more frames are decoded/encoded than are
actually scored.

Sampled-only (_encode_sampled_frames_only): decodes only the exact sampled
frames and chains them into one continuous stream in sampled order (first
frame I, rest P) -- motion vectors are computed directly between consecutive
*sampled* points, however far apart in real video time, using no extra frames.

See Codec_Selector_Feasibility.md, Finding 2.
"""
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(REPO_DIR, "figures", "windowed_vs_sampledonly_concept.png")

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
UNUSED = "#e1e0d9"
SAMPLED = "#eb6834"
WINDOWED_BLUE = "#2a78d6"
SAMPLEDONLY_GREEN = "#1baf7a"

N_FRAMES = 24
SAMPLE_IDX = [2, 8, 14, 20]
WINDOW = 4

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK_PRIMARY})


def frame_box(ax, x, y, w=0.8, h=0.8, color=UNUSED, edge=None, lw=1.2, zorder=2):
    r = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=color, edgecolor=edge or color, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(r)


def draw_timeline(ax, y, highlight_windows=False):
    for i in range(N_FRAMES):
        in_window = any(i in range(max(0, s - WINDOW), s + 1) for s in SAMPLE_IDX)
        if i in SAMPLE_IDX:
            frame_box(ax, i, y, color=SAMPLED, zorder=3)
        elif highlight_windows and in_window:
            frame_box(ax, i, y, color=WINDOWED_BLUE, zorder=3)
        else:
            frame_box(ax, i, y, color=UNUSED, zorder=2)


def main():
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.6), facecolor=SURFACE)

    # ---------- Panel 1: windowed ----------
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    draw_timeline(ax, y=2.6, highlight_windows=True)
    ax.text(-1.6, 3.0, "real video\ntimeline", fontsize=9, color=INK_MUTED, ha="left", va="center")

    for s in SAMPLE_IDX:
        lo = max(0, s - WINDOW)
        # bracket under the window
        ax.plot([lo, lo, s + 0.8, s + 0.8], [2.35, 2.25, 2.25, 2.35], color=WINDOWED_BLUE, linewidth=1.4, zorder=1)
        ax.text((lo + s + 0.8) / 2, 1.75, "5-frame\nwindow", fontsize=7.5, color=WINDOWED_BLUE,
                 ha="center", va="top")
        # local MV arrows within the window (real motion, adjacent decoded frames)
        for i in range(lo, s):
            ax.add_patch(FancyArrowPatch((i + 0.8, 2.95), (i + 1.05, 2.95), arrowstyle="-|>",
                                          mutation_scale=8, color=INK_SECONDARY, linewidth=1.0, zorder=4))

    ax.text(N_FRAMES / 2 - 0.8, 4.0, "Windowed: encode each sampled frame + 4 real frames before it, separately",
             fontsize=11.5, fontweight="bold", color=WINDOWED_BLUE, ha="center")
    ax.text(N_FRAMES / 2 - 0.8, 0.75,
             f"decodes/encodes ~{len(SAMPLE_IDX) * (WINDOW + 1)} frames to score {len(SAMPLE_IDX)} "
             "-- MVs from genuine local motion, but most decoded frames are thrown away",
             fontsize=9, color=INK_MUTED, ha="center")
    ax.set_xlim(-2.2, N_FRAMES + 0.5)
    ax.set_ylim(0.2, 4.3)
    ax.axis("off")

    # ---------- Panel 2: sampled-only ----------
    ax = axes[1]
    ax.set_facecolor(SURFACE)
    draw_timeline(ax, y=2.6, highlight_windows=False)
    ax.text(-1.6, 3.0, "real video\ntimeline", fontsize=9, color=INK_MUTED, ha="left", va="center")

    # extraction lines down to the chained stream
    chain_y = 0.9
    for k, s in enumerate(SAMPLE_IDX):
        ax.plot([s + 0.4, k * 1.1 + 0.4], [2.5, chain_y + 0.9], color=SAMPLEDONLY_GREEN,
                 linewidth=1.0, linestyle=(0, (2, 2)), zorder=1)
        frame_box(ax, k * 1.1, chain_y, color=SAMPLED, zorder=3)
        if k > 0:
            ax.add_patch(FancyArrowPatch(((k - 1) * 1.1 + 0.8, chain_y + 0.4), (k * 1.1, chain_y + 0.4),
                                          arrowstyle="-|>", mutation_scale=10, color=INK_SECONDARY,
                                          linewidth=1.2, zorder=4))

    ax.text(len(SAMPLE_IDX) * 1.1 / 2 - 0.15, chain_y - 0.35, "chained encode, sampled order",
             fontsize=7.5, color=SAMPLEDONLY_GREEN, ha="center", va="top")

    ax.text(N_FRAMES / 2 - 0.8, 4.0,
             "Sampled-only: decode just the sampled frames, chain them into one stream",
             fontsize=11.5, fontweight="bold", color=SAMPLEDONLY_GREEN, ha="center")
    ax.text(N_FRAMES / 2 - 0.8, -0.55,
             f"decodes/encodes exactly {len(SAMPLE_IDX)} frames -- MVs computed between consecutive "
             "*sampled* frames, however far apart in real time",
             fontsize=9, color=INK_MUTED, ha="center")
    ax.set_xlim(-2.2, N_FRAMES + 0.5)
    ax.set_ylim(-0.9, 4.3)
    ax.axis("off")

    handles = [
        mpatches.Patch(facecolor=SAMPLED, label="sampled frame"),
        mpatches.Patch(facecolor=WINDOWED_BLUE, label="extra context frame (windowed only)"),
        mpatches.Patch(facecolor=UNUSED, label="not decoded"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Codec mode: two ways to build the motion signal", fontsize=14, fontweight="bold",
                  color=INK_PRIMARY, y=0.99)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
