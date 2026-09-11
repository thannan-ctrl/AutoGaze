"""Convert HEVC CU partition/motion-vector statistics into AutoGaze scores.

The production codec selector consumes hevc_dump's compact binary stream;
the command-line converter below retains support for legacy YUView CSV. Its
JSON output is consumed by autogaze.datasets.video_folder.VideoFolder:

    {video_path: {"gazing_pos": [[patch_idx, ...] per frame],
                  "task_losses": [[float, ...] per frame]}}

hevc_dump CSV schema (';'-delimited, header lines start with '%'):
    map/range rows  (7 fields): POC;x;y;w;h;typeID;value
    vector rows     (8 fields): POC;x;y;w;h;typeID;mvX;mvY
    typeID: 0=PredMode (0 intra/1 inter/2 skip), 1=PartMode, 2=QP,
            3=IntraPredMode, 4=MotionVectorL0, 5=MotionVectorL1
    x, y, w, h are pixel-space (luma plane); MVs are quarter-pel.

Scoring heuristic (OneVision-Encoder-style "codec-aligned sparsity": small,
motion-heavy, non-skip blocks are salient; large skip blocks are redundant):

    score = (1 - area / max_cu_area) + w_motion * norm(mv_magnitude)
    score *= skip_penalty   if PredMode == skip

This is a first-pass heuristic, not validated against ground-truth reconstruction
loss -- treat the task_losses output as an approximation until checked against
real AutoGaze training signal.
"""

import argparse
import csv
import json
import os
import struct
import subprocess
from collections import defaultdict

import numpy as np

PRED_MODE, PART_MODE, QP, INTRA_PRED_MODE, MV_L0, MV_L1 = range(6)
SKIP = 2
MAX_CU_SIZE = 64  # HEVC max CTU side length
NVDEC_CU_SIZE = 16  # NVDEC decode-stats grid (CUVIDDECODESTATS is per 16x16)

# Compact output produced by scripts/hevc_dump/dump_stats.cc --binary.  Keep
# this format fixed-width so Python can view all blocks with np.frombuffer,
# without text conversion or one dictionary allocation per CSV row.
BLOCK_STATS_MAGIC = b"AGCUBIN1"
BLOCK_STATS_VERSION = 1
BLOCK_STATS_FILE_HEADER = struct.Struct("<8sIII")  # magic, version, width, height
BLOCK_STATS_FRAME_HEADER = struct.Struct("<iI")   # POC, number of blocks
BLOCK_STATS_DTYPE = np.dtype(
    [
        ("x", "<i4"), ("y", "<i4"), ("w", "<i4"), ("h", "<i4"),
        ("mv0_x", "<i2"), ("mv0_y", "<i2"),
        ("mv1_x", "<i2"), ("mv1_y", "<i2"),
        ("pred_mode", "u1"), ("flags", "u1"), ("reserved", "<u2"),
    ],
    align=False,
)
MV0_VALID = 1
MV1_VALID = 2


def parse_block_stats_binary(blob):
    """Return ``(width, height, {poc: structured_block_array})``.

    Arrays are zero-copy views into ``blob``.  Callers that retain the arrays
    must therefore retain the returned mapping (whose arrays keep a reference
    to the backing immutable bytes object).
    """
    view = memoryview(blob)
    if len(view) < BLOCK_STATS_FILE_HEADER.size:
        raise ValueError("truncated block-stats binary header")
    magic, version, width, height = BLOCK_STATS_FILE_HEADER.unpack_from(view, 0)
    if magic != BLOCK_STATS_MAGIC:
        raise ValueError(f"bad block-stats magic {magic!r}")
    if version != BLOCK_STATS_VERSION:
        raise ValueError(
            f"unsupported block-stats version {version}; expected {BLOCK_STATS_VERSION}"
        )

    offset = BLOCK_STATS_FILE_HEADER.size
    by_poc = {}
    while offset < len(view):
        if len(view) - offset < BLOCK_STATS_FRAME_HEADER.size:
            raise ValueError("truncated block-stats frame header")
        poc, n_blocks = BLOCK_STATS_FRAME_HEADER.unpack_from(view, offset)
        offset += BLOCK_STATS_FRAME_HEADER.size
        nbytes = int(n_blocks) * BLOCK_STATS_DTYPE.itemsize
        if len(view) - offset < nbytes:
            raise ValueError(
                f"truncated block-stats frame {poc}: need {nbytes} bytes, "
                f"have {len(view) - offset}"
            )
        blocks = np.frombuffer(view, dtype=BLOCK_STATS_DTYPE, count=n_blocks, offset=offset)
        offset += nbytes
        by_poc[poc] = blocks
    return int(width), int(height), by_poc


def _rows_to_cus(rows):
    cus = {}
    for row in rows:
        if not row or row[0].startswith("%"):
            continue
        poc, x, y, w, h, type_id = (int(v) for v in row[:6])
        key = (poc, x, y, w, h)
        cu = cus.setdefault(
            key, {"poc": poc, "x": x, "y": y, "w": w, "h": h, "mvs": []}
        )
        if type_id == PRED_MODE:
            cu["pred_mode"] = int(row[6])
        elif type_id == QP:
            cu["qp"] = int(row[6])
        elif type_id in (MV_L0, MV_L1):
            cu["mvs"].append((int(row[6]), int(row[7])))
        # PART_MODE / INTRA_PRED_MODE currently unused by the scoring heuristic
    return list(cus.values())


def parse_csv(csv_path):
    """Group hevc_dump CSV rows into one record per (POC, x, y, w, h) CU."""
    with open(csv_path, newline="") as f:
        return _rows_to_cus(csv.reader(f, delimiter=";"))


def parse_csv_for_pocs(csv_path, pocs):
    """Like parse_csv, but only load CU records for the given POC set.

    A full-video CSV can be tens of millions of lines (e.g. a 5400-frame video
    at ~2-3k CU rows/frame), while a benchmark query only samples ~16 of those
    frames. Scanning every line in the Python interpreter to then discard all
    but 16 POCs was the dominant cost of a codec-mode query (~78s of the ~88s
    end-to-end latency) -- see HEVC_Dump_Pipeline.md's performance-finding
    section. `grep` (C-speed) pre-filters to just the needed POCs' lines
    before any Python-level parsing happens.
    """
    pocs = sorted({int(p) for p in pocs})
    if not pocs:
        return []
    pattern = "|".join(f"^{p};" for p in pocs)
    proc = subprocess.run(
        ["grep", "-E", pattern, csv_path], capture_output=True, text=True, check=False
    )
    if proc.returncode > 1:  # 0 = matches found, 1 = no matches (valid), >1 = error
        raise RuntimeError(f"grep failed on {csv_path} (exit {proc.returncode}): {proc.stderr}")
    return _rows_to_cus(csv.reader(proc.stdout.splitlines(), delimiter=";"))


def score_cu(cu, w_motion, skip_penalty):
    area = cu["w"] * cu["h"]
    size_score = 1.0 - min(area, MAX_CU_SIZE**2) / (MAX_CU_SIZE**2)
    mv_mag = max((mx**2 + my**2) ** 0.5 for mx, my in cu["mvs"]) if cu["mvs"] else 0.0
    motion_score = w_motion * min(mv_mag / 256.0, 1.0)  # quarter-pel; cap at 64px
    score = size_score + motion_score
    if cu.get("pred_mode") == SKIP:
        score *= skip_penalty
    return score


def score_block_array(blocks, w_motion, skip_penalty):
    """Vectorized equivalent of :func:`score_cu` for compact binary blocks."""
    if not len(blocks):
        return np.empty(0, dtype=np.float32)
    area = blocks["w"].astype(np.float32) * blocks["h"].astype(np.float32)
    size_score = 1.0 - np.minimum(area, float(MAX_CU_SIZE**2)) / float(MAX_CU_SIZE**2)

    flags = blocks["flags"]
    mv0 = np.hypot(blocks["mv0_x"].astype(np.float32), blocks["mv0_y"].astype(np.float32))
    mv1 = np.hypot(blocks["mv1_x"].astype(np.float32), blocks["mv1_y"].astype(np.float32))
    mv0 = np.where((flags & MV0_VALID) != 0, mv0, 0.0)
    mv1 = np.where((flags & MV1_VALID) != 0, mv1, 0.0)
    score = size_score + np.float32(w_motion) * np.minimum(np.maximum(mv0, mv1) / 256.0, 1.0)
    return np.where(
        blocks["pred_mode"] == SKIP, score * np.float32(skip_penalty), score
    ).astype(np.float32, copy=False)


def score_cu_grid(cu_type, mv0_x, mv0_y, mv1_x, mv1_y, w_motion, skip_penalty, cu_size=NVDEC_CU_SIZE):
    """Vectorized `score_cu` for a regular `cu_size` x `cu_size` grid (NVDEC's
    flattened 16x16 decode-stats layout). Arrays are (mh, mw); output is too.

    Size term is constant on a regular grid (every cell has the same area), so
    ranking within a frame is motion + skip only -- same formula as `score_cu`,
    just no per-CU partition geometry.
    """
    size_score = 1.0 - min(cu_size * cu_size, MAX_CU_SIZE**2) / (MAX_CU_SIZE**2)
    mv0 = np.hypot(np.asarray(mv0_x, dtype=np.float32), np.asarray(mv0_y, dtype=np.float32))
    mv1 = np.hypot(np.asarray(mv1_x, dtype=np.float32), np.asarray(mv1_y, dtype=np.float32))
    motion_score = np.float32(w_motion) * np.minimum(np.maximum(mv0, mv1) / 256.0, 1.0)
    score = np.float32(size_score) + motion_score
    return np.where(
        np.asarray(cu_type) == SKIP, score * np.float32(skip_penalty), score
    ).astype(np.float32, copy=False)


def upsample_cu_grid(grid, frame_w, frame_h, cu_size=NVDEC_CU_SIZE):
    """Nearest-neighbor expand a (mh, mw) CU-grid score map to (frame_h, frame_w)
    pixels, matching `build_frame_score_map` painting each CU as a solid rectangle.
    Cropped to the picture size so right/bottom edge cells that overhang (when
    width/height aren't multiples of `cu_size`) don't extend past the frame."""
    import numpy as np

    up = np.repeat(np.repeat(np.asarray(grid, dtype=np.float64), cu_size, axis=0), cu_size, axis=1)
    return up[:frame_h, :frame_w]


def pool_score_map(score_map, grid_size):
    """Area-average-pool a 2D score map to grid_size x grid_size via an integral
    image (exact even when the map's dimensions don't divide evenly into
    grid_size)."""
    h, w = score_map.shape
    # Keep the original implementation's float64 integral-image storage so
    # this loop removal does not introduce an additional precision loss.
    integral = np.zeros((h + 1, w + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(score_map, axis=0), axis=1)

    # All grid cells can be read from the integral image in one advanced-index
    # operation.  This is exactly the former nested loop's rounding/math.
    edges_y = np.rint(np.arange(grid_size + 1) * h / grid_size).astype(np.intp)
    edges_x = np.rint(np.arange(grid_size + 1) * w / grid_size).astype(np.intp)
    y0, y1 = edges_y[:-1], edges_y[1:]
    x0, x1 = edges_x[:-1], edges_x[1:]
    total = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    area = np.maximum((y1 - y0)[:, None] * (x1 - x0)[None, :], 1)
    return total / area


def build_frame_score_map(cus, frame_w, frame_h):
    """Paint each CU's score onto a pixel-resolution (frame_h x frame_w) map,
    once per frame. Factored out so callers that need the same frame's map for
    multiple crop boxes (e.g. codec_selector's per-tile rasterization) can build
    it once and reuse it, instead of re-looping over the CU list per box."""
    import numpy as np

    score_map = np.zeros((frame_h, frame_w), dtype=np.float64)
    for cu, score in cus:
        y0, y1 = cu["y"], min(cu["y"] + cu["h"], frame_h)
        x0, x1 = cu["x"], min(cu["x"] + cu["w"], frame_w)
        score_map[y0:y1, x0:x1] = score
    return score_map


def build_frame_score_map_from_blocks(blocks, scores, frame_w, frame_h):
    """Paint compact structured block records without constructing CU dicts."""
    score_map = np.zeros((frame_h, frame_w), dtype=np.float32)
    for block, score in zip(blocks, scores):
        y0 = int(block["y"])
        x0 = int(block["x"])
        y1 = min(y0 + int(block["h"]), frame_h)
        x1 = min(x0 + int(block["w"]), frame_w)
        score_map[y0:y1, x0:x1] = score
    return score_map


def topk_sorted_indices(scores, k):
    """Select top-k scores in O(n), then return indices in model order."""
    flat = np.asarray(scores).reshape(-1)
    k = min(max(int(k), 0), flat.size)
    if k == 0:
        return np.empty(0, dtype=np.int64)
    if k == flat.size:
        return np.arange(flat.size, dtype=np.int64)
    chosen = np.argpartition(-flat, k - 1)[:k]
    return np.sort(chosen.astype(np.int64, copy=False))


def rasterize_to_grid(cus, frame_w, frame_h, grid_size):
    """Paint each CU's score onto a pixel-resolution map, then area-average-pool
    to a grid_size x grid_size grid."""
    return pool_score_map(build_frame_score_map(cus, frame_w, frame_h), grid_size)


def crop_and_resize_map(score_map, box_x0, box_y0, box_w, box_h, canvas_size):
    """Crop a native-pixel-space box out of a precomputed full-resolution score
    map and nearest-neighbor-resize it to canvas_size x canvas_size (i.e. the box
    resampled to canvas_size, matching how AutoGaze resizes a tile to its largest
    target scale before patchifying). Vectorized replacement for
    cus_to_local_canvas when the same frame's map is reused across many boxes."""
    import numpy as np

    h, w = score_map.shape
    ix0, iy0 = max(int(box_x0), 0), max(int(box_y0), 0)
    ix1 = min(int(round(box_x0 + box_w)), w)
    iy1 = min(int(round(box_y0 + box_h)), h)
    if ix1 <= ix0 or iy1 <= iy0:
        return np.zeros((canvas_size, canvas_size), dtype=np.float64)
    crop = score_map[iy0:iy1, ix0:ix1]
    row_idx = np.clip((np.arange(canvas_size) * crop.shape[0] // canvas_size), 0, crop.shape[0] - 1)
    col_idx = np.clip((np.arange(canvas_size) * crop.shape[1] // canvas_size), 0, crop.shape[1] - 1)
    return crop[row_idx][:, col_idx]


def cus_to_local_canvas(cus, box_x0, box_y0, box_w, box_h, canvas_size):
    """Paint CU scores that intersect a native-pixel-space crop box (box_x0,
    box_y0, box_w, box_h) onto a canvas_size x canvas_size local canvas (i.e.
    the box resampled to canvas_size, matching how AutoGaze resizes a tile to
    its largest target scale before patchifying)."""
    import numpy as np

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float64)
    sx = canvas_size / box_w
    sy = canvas_size / box_h
    for cu, score in cus:
        ix0, iy0 = max(cu["x"], box_x0), max(cu["y"], box_y0)
        ix1, iy1 = min(cu["x"] + cu["w"], box_x0 + box_w), min(cu["y"] + cu["h"], box_y0 + box_h)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        cx0, cx1 = int((ix0 - box_x0) * sx), int(round((ix1 - box_x0) * sx))
        cy0, cy1 = int((iy0 - box_y0) * sy), int(round((iy1 - box_y0) * sy))
        cx1, cy1 = max(cx1, cx0 + 1), max(cy1, cy0 + 1)
        canvas[cy0:cy1, cx0:cx1] = score
    return canvas


def rasterize_multiscale(cus, box_x0, box_y0, box_w, box_h, scales, patch_size):
    """Score a native-pixel-space crop box across AutoGaze's multiscale patch
    pyramid (one grid per scale, ascending-scale-then-row-major flat index,
    matching modeling_video_mae.py's patch ordering). Returns a flat
    sum((s//patch_size)**2 for s in scales)-length score array."""
    import numpy as np

    canvas = cus_to_local_canvas(cus, box_x0, box_y0, box_w, box_h, canvas_size=scales[-1])
    grid_sizes = [s // patch_size for s in scales]
    flat = np.zeros(sum(g * g for g in grid_sizes), dtype=np.float64)
    offset = 0
    for g in grid_sizes:
        flat[offset:offset + g * g] = pool_score_map(canvas, g).flatten()
        offset += g * g
    return flat


def rasterize_multiscale_from_map(score_map, box_x0, box_y0, box_w, box_h, scales, patch_size):
    """Same output as rasterize_multiscale, but takes a precomputed full-resolution
    per-frame score map (from build_frame_score_map) instead of a raw CU list --
    lets callers that rasterize many boxes per frame (e.g. one per spatial tile)
    build the frame's map once and reuse it, instead of re-looping over CUs and
    repainting a canvas from scratch for every box."""
    import numpy as np

    canvas = crop_and_resize_map(score_map, box_x0, box_y0, box_w, box_h, canvas_size=scales[-1])
    grid_sizes = [s // patch_size for s in scales]
    flat = np.zeros(sum(g * g for g in grid_sizes), dtype=np.float64)
    offset = 0
    for g in grid_sizes:
        flat[offset:offset + g * g] = pool_score_map(canvas, g).flatten()
        offset += g * g
    return flat


def rasterize_multiscale_from_cu_grid(
    grid, box_x0, box_y0, box_w, box_h, scales, patch_size, cu_size=NVDEC_CU_SIZE
):
    """Token scores from an NVDEC 16x16 score grid.

    `grid` is (mh, mw) in CU-cell space (native pixels / `cu_size`). Pixel crop
    boxes are converted to that space so we never expand to frame resolution:
    nearest-neighbor onto the AutoGaze canvas, then the same area-pool pyramid
    as `rasterize_multiscale_from_map`. Tile edges that fall inside a 16x16 cell
    snap to cell indices (same as `crop_and_resize_map`); they are not
    area-weighted sub-cell crops.
    """
    s = float(cu_size)
    return rasterize_multiscale_from_map(
        grid, box_x0 / s, box_y0 / s, box_w / s, box_h / s, scales, patch_size
    )


def convert(csv_path, frame_w, frame_h, grid_size, w_motion, skip_penalty, topk):
    cus = parse_csv(csv_path)
    by_frame = defaultdict(list)
    for cu in cus:
        by_frame[cu["poc"]].append((cu, score_cu(cu, w_motion, skip_penalty)))

    gazing_pos, task_losses = [], []
    for poc in sorted(by_frame):  # NOTE: assumes POC order == display/decode order
        grid = rasterize_to_grid(by_frame[poc], frame_w, frame_h, grid_size)
        flat = grid.flatten()
        ranked = sorted(range(len(flat)), key=lambda i: -flat[i])
        if topk is not None:
            ranked = ranked[:topk]
        gazing_pos.append(ranked)
        task_losses.append([float(flat[i]) for i in ranked])
    return gazing_pos, task_losses


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="hevc_dump output CSV")
    ap.add_argument("--video-path", required=True, help="key used in the output JSON (path AutoGaze loads the video from)")
    ap.add_argument("--frame-width", type=int, required=True)
    ap.add_argument("--frame-height", type=int, required=True)
    ap.add_argument("--grid-size", type=int, default=14, help="patch grid side length (default 14 = AutoGaze's 224px/16px scale)")
    ap.add_argument("--topk", type=int, default=None, help="keep only the top-K scored patches per frame (default: keep all grid_size**2, ranked)")
    ap.add_argument("--w-motion", type=float, default=1.0, help="weight on motion-magnitude score term")
    ap.add_argument("--skip-penalty", type=float, default=0.1, help="multiplier applied to skip-coded CUs")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--merge", action="store_true", help="merge into an existing --out JSON instead of overwriting it")
    args = ap.parse_args()

    gazing_pos, task_losses = convert(
        args.csv, args.frame_width, args.frame_height, args.grid_size,
        args.w_motion, args.skip_penalty, args.topk,
    )

    data = {}
    if args.merge and os.path.exists(args.out):
        with open(args.out) as f:
            data = json.load(f)
    data[args.video_path] = {"gazing_pos": gazing_pos, "task_losses": task_losses}

    with open(args.out, "w") as f:
        json.dump(data, f)
    print(f"Wrote {len(gazing_pos)} frames for {args.video_path} -> {args.out}")


if __name__ == "__main__":
    main()
