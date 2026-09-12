"""Codec-based (HEVC CU partition/motion) substitute for AutoGaze's autoregressive
selector, wired into the scripts/breakdown benchmark as modes "codec" and
"codec_nvdec".

Rather than intercepting NVILAProcessor mid-pipeline (crop-box bookkeeping isn't
retained by the time `_get_gazing_info_from_videos` runs), this module independently
replicates NVILA-HD's own frame-sampling and spatial-tiling math -- imported directly
from the loaded `processing_nvila` module, not reimplemented -- so patch indices line
up with what the real pipeline would produce for the same video_path/config. See
Codec_Selector_Feasibility.md for the full architecture writeup.

WINDOWED ENCODING (not full-video): only ~num_video_frames * (WINDOW+1) real frames
are ever decoded/encoded/dumped, not the whole video. Each needed frame gets a short
window of real, temporally-adjacent context (WINDOW frames before it) so motion
vectors stay meaningful, with an explicit I-frame forced at each window's start so
no motion/prediction data crosses window boundaries. This replaced an earlier
full-video re-encode (needed, it was believed, for POC alignment) that cost
~140-160s/video; windowed encoding costs ~2s/video by only touching the ~1-2% of
frames actually scored. Since we now choose exactly which real frame maps to which
artificial-stream POC (`poc_map`), the old "does POC == cv2-frame-index" assumption
this module used to carry is no longer a risk -- the mapping is explicit and correct
by construction rather than assumed.

Backends:
  libde265 ("codec")     -- dump_stats walks the true CU quad-tree, writes a
                            YUView CSV, then we grep/parse it back. Ground truth
                            partition geometry, slow plumbing.
  nvdec ("codec_nvdec")  -- same windowed x265 Annex-B encode, then sequential
                            NVDEC decode-stats (CreateDemuxer + CreateDecoder)
                            on a regular 16x16 grid, kept as in-memory numpy
                            (cached as .npz). Scores and token selection stay
                            on that grid (pixel crop boxes are mapped into
                            CU-cell space)
"""
import functools
import hashlib
import os
import platform
import sys
import time

import cv2
import numpy as np
import torch

from . import timing

WINDOW = 4  # real frames of context before each scored frame (5 frames/window total)

_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)
from scripts import hevc_to_gaze as h2g  # noqa: E402
# `nvdec_dump` (PyNvVideoCodec wrapper) is imported lazily inside
# `_nvdec_dump_grids`, not here -- it's an optional dependency only needed by
# backend="nvdec", and as of this merge scripts/nvdec_dump.py doesn't exist in
# this repo yet (see Codec_Selector_Feasibility.md Next Steps). A hard
# top-level import would break every "codec"/libde265 use of this module too.

# Two separate native builds exist -- cmake_build (x86_64, built on the login
# node) and cmake_build_aarch64 (aarch64, built natively on GB200 compute
# nodes; libde265's encoder subtree needed -fPIC forced via
# -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_C_FLAGS=-fPIC
# -DCMAKE_CXX_FLAGS=-fPIC, see HEVC_Dump_Pipeline.md). Pick the one matching
# this process's own architecture.
_BUILD_DIR = "cmake_build_aarch64" if platform.machine() == "aarch64" else "cmake_build"
DUMP_STATS_BIN = os.path.join(_REPO_DIR, "scripts", "hevc_dump", _BUILD_DIR, "dump_stats")
CACHE_DIR = os.path.join(_REPO_DIR, "data", "hevc_dump_cache")
NVDEC_CACHE_DIR = os.path.join(_REPO_DIR, "data", "hevc_nvdec_cache")
# Separate from CACHE_DIR: hevc_geo/hevc_ord encode at AutoGaze's own resized
# resolution (see _build_gazing_info_hevc_autogaze), not the source video's
# native resolution -- a different encode of the same video, so it needs its
# own cache namespace rather than sharing keys with the native-res "codec"/
# "codec_nvdec" cache.
HEVC_AUTOGAZE_CACHE_DIR = os.path.join(_REPO_DIR, "data", "hevc_autogaze_cache")

# Benchmark mode name -> dump backend. "codec" keeps the original libde265/CSV
# path; "codec_nvdec" is the NVDEC 16x16 variant; "codec_geo"/"codec_ord" score
# via hevc_autogaze.py's CU-size-vs-AutoGaze-scale kernel (geometric/ordinal).
BACKEND_FOR_MODE = {
    "codec": "libde265", "codec_nvdec": "nvdec",
    "codec_geo": "hevc_geo", "codec_ord": "hevc_ord",
}


def _video_key(video_path: str, frame_indices, sampled_only: bool = False, gop_restart: int | None = None) -> str:
    st = os.stat(video_path)
    frames_key = hashlib.sha1(str(sorted(set(frame_indices))).encode()).hexdigest()[:8]
    # sampled_only/gop_restart must be part of the key -- each changes what's
    # encoded for the SAME (video_path, frame_indices), so a shared key would
    # let one mode silently serve another's cached CSV.
    suffix = "_so" if sampled_only else ""
    if gop_restart:
        suffix += f"_gop{gop_restart}"
    return hashlib.sha1(f"{video_path}:{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:16] + "_" + frames_key + suffix


def _extract_and_encode_windows(video_path: str, frame_indices, hevc_path: str, container: str = "hevc"):
    """Encode only small real-frame windows around each needed frame, not the
    whole video. For each target frame index, grabs it plus WINDOW real
    preceding frames (via a single sequential decode pass over the source --
    seeking isn't needed since we just skip frames outside the needed set),
    then encodes all windows back-to-back into one short HEVC stream with an
    explicit I-frame forced at every window's start (`pict_type = I`, not
    reliance on periodic `keyint` -- windows have variable length near the
    start of the video, so periodic keyint doesn't reliably land on window
    boundaries and can let motion data leak across windows).

    `container` is the PyAV output format. Both dump backends consume Annex-B
    (`hevc`); mp4 is not required.

    Returns (width, height, poc_map) where poc_map maps original cv2 frame
    index -> POC in the encoded stream (sequential 0..N-1 in window order).
    """
    import av
    from av.video.frame import PictureType

    windows = [list(range(max(0, idx - WINDOW), idx + 1)) for idx in frame_indices]
    needed = sorted({i for win in windows for i in win})

    src = av.open(video_path)
    vs = src.streams.video[0]
    w, h = vs.codec_context.width, vs.codec_context.height
    needed_set = set(needed)
    max_needed = max(needed)
    frames_by_idx = {}
    frame_i = 0
    for frame in src.decode(vs):
        if frame_i in needed_set:
            frames_by_idx[frame_i] = frame.to_ndarray(format="rgb24")
        if frame_i >= max_needed:
            break
        frame_i += 1
    src.close()

    out = av.open(hevc_path, mode="w", format=container)
    enc = out.add_stream("libx265", rate=25)
    enc.width, enc.height = w, h
    enc.pix_fmt = "yuv420p"
    # pools=8 (explicit, not auto-detected) + preset=superfast: the x265
    # NUMA-topology auto-detection bug on this many-core ARM box (garbage
    # thread counts, indefinite hang) is specifically in *auto-detection* --
    # an explicit pools count sidesteps it while still getting real
    # multi-threading. superfast preserves min-CU-size=8 (same as the default
    # medium preset), unlike ultrafast which coarsens it to 16 and would blunt
    # the size-based saliency signal score_cu depends on. scenecut=0 disables
    # automatic I-frame insertion so only our explicit per-window I-frames
    # (below) create GOP boundaries.
    enc.options = {"x265-params": "qp=27:pools=8:scenecut=0", "preset": "superfast"}
    poc_map = {}
    new_poc = 0
    for win in windows:
        for j, i in enumerate(win):
            vf = av.VideoFrame.from_ndarray(frames_by_idx[i], format="rgb24")
            if j == 0:
                vf.pict_type = PictureType.I
            for packet in enc.encode(vf):
                out.mux(packet)
            poc_map[i] = new_poc
            new_poc += 1
    for packet in enc.encode():
        out.mux(packet)
    out.close()
    return w, h, poc_map


def _probe_video_dims(video_path: str) -> tuple:
    """Cheap (width, height) probe -- reads stream metadata only, no frame
    decode. Used by the hevc_geo/hevc_ord backends to compute AutoGaze's tile
    grid *before* encoding, so the encode itself can happen at that resized
    resolution (see _build_gazing_info_hevc_autogaze)."""
    import av

    src = av.open(video_path)
    vs = src.streams.video[0]
    w, h = vs.codec_context.width, vs.codec_context.height
    src.close()
    return w, h


def _extract_and_encode_windows_resized(
    video_path: str, frame_indices, hevc_path: str, target_w: int, target_h: int, container: str = "hevc"
):
    """Like `_extract_and_encode_windows`, but each real frame is resized to
    (target_w, target_h) *before* encoding, so the encoded stream's native HEVC
    CU sizes are directly comparable, pixel-for-pixel, to AutoGaze's patch
    coverage -- hevc_autogaze.py's kernel table assumes exactly this (see
    Codec_Selector_Feasibility.md's "hevc_autogaze integration" section). The
    plain (native-resolution) encode used by "codec"/"codec_nvdec" doesn't
    satisfy that assumption whenever AutoGaze's own resize factor isn't 1:1.

    Returns (target_w, target_h, poc_map) -- same shape as
    `_extract_and_encode_windows`, so callers can treat target_w/target_h as
    this stream's "orig_w/orig_h" downstream.
    """
    import av
    from av.video.frame import PictureType

    windows = [list(range(max(0, idx - WINDOW), idx + 1)) for idx in frame_indices]
    needed = sorted({i for win in windows for i in win})

    src = av.open(video_path)
    vs = src.streams.video[0]
    needed_set = set(needed)
    max_needed = max(needed)
    frames_by_idx = {}
    frame_i = 0
    for frame in src.decode(vs):
        if frame_i in needed_set:
            arr = frame.to_ndarray(format="rgb24")
            if arr.shape[1] != target_w or arr.shape[0] != target_h:
                arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_AREA)
            frames_by_idx[frame_i] = arr
        if frame_i >= max_needed:
            break
        frame_i += 1
    src.close()

    out = av.open(hevc_path, mode="w", format=container)
    enc = out.add_stream("libx265", rate=25)
    enc.width, enc.height = target_w, target_h
    enc.pix_fmt = "yuv420p"
    enc.options = {"x265-params": "qp=27:pools=8:scenecut=0", "preset": "superfast"}
    poc_map = {}
    new_poc = 0
    for win in windows:
        for j, i in enumerate(win):
            vf = av.VideoFrame.from_ndarray(frames_by_idx[i], format="rgb24")
            if j == 0:
                vf.pict_type = PictureType.I
            for packet in enc.encode(vf):
                out.mux(packet)
            poc_map[i] = new_poc
            new_poc += 1
    for packet in enc.encode():
        out.mux(packet)
    out.close()
    return target_w, target_h, poc_map


def get_or_build_hevc_autogaze_csv(video_path: str, frame_indices, target_w: int, target_h: int, tag: str):
    """Encode `frame_indices`' windows resized to (target_w, target_h) and dump
    libde265 decode-stats to CSV, cached under HEVC_AUTOGAZE_CACHE_DIR (separate
    from CACHE_DIR -- a different resize target is a different encode of the
    same video, so it needs its own cache key space). `tag` distinguishes the
    tile-grid-resolution stream from the single-canvas thumbnail stream for the
    same video -- both cover the same frame_indices in general but at different
    resolutions, so they can't share a cache key even with target_w/target_h
    already folded in (keeps the two easy to tell apart when debugging)."""
    os.makedirs(HEVC_AUTOGAZE_CACHE_DIR, exist_ok=True)
    st = os.stat(video_path)
    frames_key = hashlib.sha1(str(sorted(set(frame_indices))).encode()).hexdigest()[:8]
    key = (
        hashlib.sha1(f"{video_path}:{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:16]
        + f"_{frames_key}_{target_w}x{target_h}_{tag}"
    )
    csv_path = os.path.join(HEVC_AUTOGAZE_CACHE_DIR, f"{key}.csv")
    pocmap_path = os.path.join(HEVC_AUTOGAZE_CACHE_DIR, f"{key}.pocmap")
    if os.path.exists(csv_path) and os.path.exists(pocmap_path):
        with open(pocmap_path) as f:
            poc_map = dict(tuple(int(x) for x in line.split(",")) for line in f if line.strip())
        return csv_path, poc_map

    hevc_path = os.path.join(HEVC_AUTOGAZE_CACHE_DIR, f"{key}.hevc")
    t_encode0 = time.time()
    _, _, poc_map = _extract_and_encode_windows_resized(video_path, frame_indices, hevc_path, target_w, target_h)
    # Only runs on a cache miss -- see get_or_build_stats's identical comment.
    timing.add("codec_encode_ms", (time.time() - t_encode0) * 1000)

    yuv_path = os.path.join(HEVC_AUTOGAZE_CACHE_DIR, f"{key}.yuv")
    t_decode0 = time.time()
    ret = os.system(f'"{DUMP_STATS_BIN}" "{hevc_path}" "{yuv_path}" "{csv_path}"')
    timing.add("codec_decode_ms", (time.time() - t_decode0) * 1000)
    if ret != 0:
        raise RuntimeError(f"dump_stats failed on {video_path} ({tag}, exit code {ret})")
    if os.path.exists(yuv_path):
        os.remove(yuv_path)
    with open(pocmap_path, "w") as f:
        for real_idx, stream_poc in poc_map.items():
            f.write(f"{real_idx},{stream_poc}\n")
    return csv_path, poc_map


@functools.lru_cache(maxsize=8)
def _hevc_autogaze_scorer(image_size: int, patch_size: int, scales: tuple):
    from scripts.hevc_dump.hevc_autogaze import HevcAutogazePatchScorer

    return HevcAutogazePatchScorer(scales=list(scales), tile_size=image_size, patch_size=patch_size)


@functools.lru_cache(maxsize=16)
def _hevc_autogaze_result(csv_path: str, image_size: int, patch_size: int, scales: tuple):
    scorer = _hevc_autogaze_scorer(image_size, patch_size, scales)
    return scorer.score_csv(csv_path)


def _build_gazing_info_hevc_autogaze(
    video_path: str,
    num_video_frames: int,
    num_video_frames_thumbnail: int,
    max_tiles_video: int,
    autogaze_max_num_frames: int,
    image_size: int,
    scales: list,
    patch_size: int,
    gazing_ratio_tile,
    gazing_ratio_thumbnail,
    full_first_frame: bool,
    kind: str,
):
    """backend in ("hevc_geo", "hevc_ord"): score via hevc_autogaze.py's
    CU-size-vs-AutoGaze-scale kernel, which assumes the encoded stream *is*
    the same pixel grid AutoGaze tiles (see Codec_Selector_Feasibility.md).
    To make that true, this encodes TWO separate resized streams per video --
    one at the real tile-grid resolution (cols*image_size x rows*image_size)
    for spatial tiles, one at a single image_size x image_size canvas (whole
    frame, 1x1 grid) for thumbnails -- rather than reusing the native-
    resolution encode/cache shared by "codec"/"codec_nvdec". Roughly 2x the
    Encode cost of "codec" as a result (two encode passes instead of one).

    Once encoded this way, hevc_autogaze.py's own tile_grid() exactly matches
    NVILA's cols x rows spatial tiling (target_w/target_h are exact multiples
    of image_size), and its per-tile vocab layout (four scales, row-major)
    exactly matches this codebase's total_patches ordering -- so scores are
    read directly off HevcAutogazePatchScorer's own PatchScoreResult, no
    crop/rasterize step needed (contrast with the libde265/nvdec backends'
    score_region, which cross native pixel space via crop_and_resize_map).
    """
    find_closest_aspect_ratio = _find_closest_aspect_ratio_fn()
    frame_indices = _sampled_frame_indices(video_path, num_video_frames)

    orig_w, orig_h = _probe_video_dims(video_path)
    aspect_ratio = orig_w / orig_h
    max_spatial_tiles = max(max_tiles_video, 1)
    target_ratios = sorted(
        {
            (i, j)
            for n in range(1, max_spatial_tiles + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if 1 <= i * j <= max_spatial_tiles
        },
        key=lambda x: x[0] * x[1],
    )
    cols, rows = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_w, orig_h, image_size)
    target_w, target_h = image_size * cols, image_size * rows
    num_spatial_tiles = cols * rows

    temporal_chunks = num_video_frames // autogaze_max_num_frames
    assert temporal_chunks >= 1 and num_video_frames % autogaze_max_num_frames == 0, (
        f"num_video_frames ({num_video_frames}) must be divisible by "
        f"autogaze_max_num_frames ({autogaze_max_num_frames})"
    )
    T_tile = autogaze_max_num_frames
    grid_sizes = [s // patch_size for s in scales]
    total_patches = sum(g * g for g in grid_sizes)
    kind_name = "geometric" if kind == "hevc_geo" else "ordinal"

    def topk_ratio(ratio, index):
        r = ratio[index] if isinstance(ratio, (list, tuple)) else ratio
        return max(1, int(round(total_patches * r)))

    def frame_scores(result, poc):
        return result.frame(poc) if kind_name == "geometric" else result.ordinal_frame(poc)

    # --- tiles: encode resized to the real tile-grid resolution ---
    tile_csv, tile_poc_map = get_or_build_hevc_autogaze_csv(video_path, frame_indices, target_w, target_h, tag="tiles")
    tile_result = _hevc_autogaze_result(tile_csv, image_size, patch_size, tuple(scales))

    tile_pos, tile_counts = [], []
    for t_chunk in range(temporal_chunks):
        for spatial_idx in range(num_spatial_tiles):
            frame_pos, frame_counts = [], []
            for f_local in range(T_tile):
                real_idx = frame_indices[t_chunk * T_tile + f_local]
                poc = tile_poc_map[real_idx]
                t0 = time.time()
                scores = frame_scores(tile_result, poc)[spatial_idx]
                timing.add("selector_score_ms", (time.time() - t0) * 1000)
                k = total_patches if (full_first_frame and f_local == 0) else topk_ratio(gazing_ratio_tile, f_local)
                t0 = time.time()
                ranked = np.sort(np.argsort(-scores)[:k])
                timing.add("selector_rank_ms", (time.time() - t0) * 1000)
                frame_pos.append(torch.as_tensor(ranked, dtype=torch.long))
                frame_counts.append(k)
            tile_pos.append(torch.cat(frame_pos))
            tile_counts.append(torch.tensor(frame_counts, dtype=torch.long))

    gazing_pos_tiles = torch.nn.utils.rnn.pad_sequence(tile_pos, batch_first=True, padding_value=0)
    if_padded_gazing_tiles = torch.zeros_like(gazing_pos_tiles, dtype=torch.bool)
    num_gazing_each_frame_tiles = torch.stack(tile_counts)

    # --- thumbnails: encode resized to a single image_size x image_size canvas ---
    if len(frame_indices) > num_video_frames_thumbnail:
        step = len(frame_indices) // num_video_frames_thumbnail
        thumb_indices = frame_indices[::step][:num_video_frames_thumbnail]
    else:
        thumb_indices = frame_indices

    thumb_csv, thumb_poc_map = get_or_build_hevc_autogaze_csv(video_path, thumb_indices, image_size, image_size, tag="thumb")
    thumb_result = _hevc_autogaze_result(thumb_csv, image_size, patch_size, tuple(scales))

    thumb_pos, thumb_counts = [], []
    for thumb_i, real_idx in enumerate(thumb_indices):
        poc = thumb_poc_map[real_idx]
        t0 = time.time()
        scores = frame_scores(thumb_result, poc)[0]  # single 1x1-grid tile
        timing.add("selector_score_ms", (time.time() - t0) * 1000)
        if full_first_frame and thumb_i == 0:
            k = total_patches
        else:
            k = topk_ratio(gazing_ratio_thumbnail if gazing_ratio_thumbnail is not None else 1.0, 0)
        t0 = time.time()
        ranked = np.sort(np.argsort(-scores)[:k])
        timing.add("selector_rank_ms", (time.time() - t0) * 1000)
        thumb_pos.append(torch.as_tensor(ranked, dtype=torch.long))
        thumb_counts.append(k)

    gazing_pos_thumbnails = torch.nn.utils.rnn.pad_sequence(thumb_pos, batch_first=True, padding_value=0)
    if_padded_gazing_thumbnails = torch.zeros_like(gazing_pos_thumbnails, dtype=torch.bool)
    num_gazing_each_frame_thumbnails = torch.tensor(thumb_counts, dtype=torch.long).unsqueeze(1)

    return {
        "gazing_pos_tiles": [gazing_pos_tiles],
        "num_gazing_each_frame_tiles": [num_gazing_each_frame_tiles],
        "if_padded_gazing_tiles": [if_padded_gazing_tiles],
        "gazing_pos_thumbnails": [gazing_pos_thumbnails],
        "num_gazing_each_frame_thumbnails": [num_gazing_each_frame_thumbnails],
        "if_padded_gazing_thumbnails": [if_padded_gazing_thumbnails],
    }


def _encode_sampled_frames_only(video_path: str, frame_indices, hevc_path: str):
    """Encode ONLY the exact sampled frames, chained into one continuous stream
    -- no real-frame context window. The first sampled frame is forced I; every
    subsequent sampled frame is left as a natural P-frame, so x265's own motion
    estimation computes MVs between CONSECUTIVE SAMPLED FRAMES directly --
    however far apart they actually are in the source video's real timeline --
    rather than from genuinely-adjacent real frames like
    `_extract_and_encode_windows` does.

    This is an ablation/speed variant: it skips decoding/encoding the WINDOW
    real context frames per sampled point entirely (the dominant cost of the
    windowed approach at high frame counts -- see
    Codec_Selector_Feasibility.md's frame-count-sweep section), at the cost of
    motion vectors that may span large real-time gaps and thus be less
    physically meaningful.

    Returns (width, height, poc_map) where poc_map maps original cv2 frame
    index -> POC in the encoded stream (sequential 0..N-1, matching sorted
    frame_indices order -- no window-relative offsetting needed since there's
    no overlap/duplication).
    """
    import av
    from av.video.frame import PictureType

    sorted_indices = sorted(set(frame_indices))
    needed_set = set(sorted_indices)
    max_needed = sorted_indices[-1]

    src = av.open(video_path)
    vs = src.streams.video[0]
    w, h = vs.codec_context.width, vs.codec_context.height
    frames_by_idx = {}
    frame_i = 0
    for frame in src.decode(vs):
        if frame_i in needed_set:
            frames_by_idx[frame_i] = frame.to_ndarray(format="rgb24")
        if frame_i >= max_needed:
            break
        frame_i += 1
    src.close()

    out = av.open(hevc_path, mode="w", format="hevc")
    enc = out.add_stream("libx265", rate=25)
    enc.width, enc.height = w, h
    enc.pix_fmt = "yuv420p"
    # Same encoder settings as _extract_and_encode_windows (see comments
    # there); scenecut=0 here means the ONLY I-frame is the explicit first one
    # below -- everything else stays a natural P-frame referencing the
    # previous sampled frame, whatever the real gap between them.
    enc.options = {"x265-params": "qp=27:pools=8:scenecut=0", "preset": "superfast"}
    poc_map = {}
    for new_poc, idx in enumerate(sorted_indices):
        vf = av.VideoFrame.from_ndarray(frames_by_idx[idx], format="rgb24")
        if new_poc == 0:
            vf.pict_type = PictureType.I
        for packet in enc.encode(vf):
            out.mux(packet)
        poc_map[idx] = new_poc
    for packet in enc.encode():
        out.mux(packet)
    out.close()
    return w, h, poc_map


def _encode_periodic_restart(video_path: str, frame_indices, hevc_path: str, gop_restart: int):
    """Like `_encode_sampled_frames_only` (chains the exact sampled frames,
    no real-frame context window), but forces an I-frame every `gop_restart`
    frames instead of only at position 0 -- x265 never references across an
    I-frame boundary, so this "restarts the coding" at each chunk start,
    matching AutoGaze's own real design of independent, non-overlapping
    16-frame chunks (each with its own anchor) rather than one continuous
    autoregressive stream. Motion vectors are computed between consecutive
    sampled frames *within* the same chunk only; the first frame of each
    chunk has no predecessor to reference, same as a real chunk boundary.

    Returns (width, height, poc_map), same shape as `_encode_sampled_frames_only`.
    """
    import av
    from av.video.frame import PictureType

    sorted_indices = sorted(set(frame_indices))
    needed_set = set(sorted_indices)
    max_needed = sorted_indices[-1]

    src = av.open(video_path)
    vs = src.streams.video[0]
    w, h = vs.codec_context.width, vs.codec_context.height
    frames_by_idx = {}
    frame_i = 0
    for frame in src.decode(vs):
        if frame_i in needed_set:
            frames_by_idx[frame_i] = frame.to_ndarray(format="rgb24")
        if frame_i >= max_needed:
            break
        frame_i += 1
    src.close()

    out = av.open(hevc_path, mode="w", format="hevc")
    enc = out.add_stream("libx265", rate=25)
    enc.width, enc.height = w, h
    enc.pix_fmt = "yuv420p"
    # scenecut=0 so the ONLY I-frames are the explicit periodic ones below --
    # x265 won't insert its own extra I-frames on top of our restart cadence.
    enc.options = {"x265-params": "qp=27:pools=8:scenecut=0", "preset": "superfast"}
    poc_map = {}
    for new_poc, idx in enumerate(sorted_indices):
        vf = av.VideoFrame.from_ndarray(frames_by_idx[idx], format="rgb24")
        if new_poc % gop_restart == 0:
            vf.pict_type = PictureType.I
        for packet in enc.encode(vf):
            out.mux(packet)
        poc_map[idx] = new_poc
    for packet in enc.encode():
        out.mux(packet)
    out.close()
    return w, h, poc_map


def get_or_build_stats(video_path: str, frame_indices, sampled_only: bool = False, gop_restart: int | None = None):
    """Return (csv_path, width, height, poc_map) for a video's hevc_dump CSV,
    encoding + dumping it once and caching by (path, size, mtime,
    frame_indices, sampled_only, gop_restart). See `_extract_and_encode_windows`
    (default) vs. `_encode_sampled_frames_only` (sampled_only=True) vs.
    `_encode_periodic_restart` (gop_restart=N, implies sampled_only-style
    chaining otherwise) for what each encoding strategy does."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _video_key(video_path, frame_indices, sampled_only, gop_restart)
    csv_path = os.path.join(CACHE_DIR, f"{key}.csv")
    meta_path = os.path.join(CACHE_DIR, f"{key}.meta")
    pocmap_path = os.path.join(CACHE_DIR, f"{key}.pocmap")
    if os.path.exists(csv_path) and os.path.exists(meta_path) and os.path.exists(pocmap_path):
        with open(meta_path) as f:
            w, h = (int(x) for x in f.read().split(","))
        with open(pocmap_path) as f:
            poc_map = dict(tuple(int(x) for x in line.split(",")) for line in f if line.strip())
        return csv_path, w, h, poc_map

    hevc_path = os.path.join(CACHE_DIR, f"{key}.hevc")
    t_encode0 = time.time()
    if gop_restart:
        w, h, poc_map = _encode_periodic_restart(video_path, frame_indices, hevc_path, gop_restart)
    else:
        encode_fn = _encode_sampled_frames_only if sampled_only else _extract_and_encode_windows
        w, h, poc_map = encode_fn(video_path, frame_indices, hevc_path)
    # Only runs on a cache miss -- a hit falls through the early `return`
    # above, leaving this key at whatever timing.reset() left it (0 for the
    # current question), which correctly reads as "no fresh encode happened".
    timing.add("codec_encode_ms", (time.time() - t_encode0) * 1000)

    yuv_path = os.path.join(CACHE_DIR, f"{key}.yuv")
    t_decode0 = time.time()
    ret = os.system(f'"{DUMP_STATS_BIN}" "{hevc_path}" "{yuv_path}" "{csv_path}"')
    timing.add("codec_decode_ms", (time.time() - t_decode0) * 1000)
    if ret != 0:
        raise RuntimeError(f"dump_stats failed on {video_path} (exit code {ret})")
    if os.path.exists(yuv_path):
        os.remove(yuv_path)
    with open(meta_path, "w") as f:
        f.write(f"{w},{h}")
    with open(pocmap_path, "w") as f:
        for real_idx, stream_poc in poc_map.items():
            f.write(f"{real_idx},{stream_poc}\n")
    return csv_path, w, h, poc_map


def _gpu_id() -> int:
    """Visible CUDA device index for PyNvVideoCodec. NVILA_DEVICE is already
    the index *within* CUDA_VISIBLE_DEVICES (e.g. cuda:0), matching gpu_id=0
    when the benchmark sets CUDA_VISIBLE_DEVICES to a single GPU."""
    dev = os.environ.get("NVILA_DEVICE", "cuda:0")
    if ":" in dev:
        return int(dev.split(":")[-1])
    return int(dev) if str(dev).isdigit() else 0


def _nvdec_dump_grids(hevc_path: str, width: int, height: int):
    """Decode `hevc_path` sequentially with NVDEC and return per-display-frame
    dicts {cu_type, mv0_x, mv0_y, mv1_x, mv1_y}, each (mh, mw). Pixels are
    discarded -- only the decode-stats buffer is kept."""
    try:
        from scripts import nvdec_dump as nvd
    except ImportError as e:
        raise RuntimeError(
            "backend='nvdec' requires scripts/nvdec_dump.py (PyNvVideoCodec wrapper), "
            "which isn't present in this checkout -- see Codec_Selector_Feasibility.md "
            "Next Steps. Use backend='libde265' (the default) instead."
        ) from e
    return nvd.dump_nvdec_grids(hevc_path, width, height, gpu_id=_gpu_id())


def get_or_build_nvdec_stats(video_path: str, frame_indices):
    """Windowed-encode then NVDEC-dump to a cached .npz of 16x16 grids.

    Returns (npz_path, width, height, poc_map). The npz stores raw cu_type/MVs
    (not scored maps) so w_motion/skip_penalty stay runtime parameters, matching
    how the CSV backend defers scoring until parse time.
    """
    os.makedirs(NVDEC_CACHE_DIR, exist_ok=True)
    key = _video_key(video_path, frame_indices)
    npz_path = os.path.join(NVDEC_CACHE_DIR, f"{key}.npz")
    if os.path.exists(npz_path):
        w, h, poc_map, _n = _nvdec_npz_meta(npz_path)
        return npz_path, w, h, poc_map

    hevc_path = os.path.join(NVDEC_CACHE_DIR, f"{key}.hevc")
    t_encode0 = time.time()
    w, h, poc_map = _extract_and_encode_windows(video_path, frame_indices, hevc_path)
    timing.add("codec_encode_ms", (time.time() - t_encode0) * 1000)
    t_decode0 = time.time()
    try:
        grids = _nvdec_dump_grids(hevc_path, w, h)
    finally:
        timing.add("codec_decode_ms", (time.time() - t_decode0) * 1000)
        if os.path.exists(hevc_path):
            os.remove(hevc_path)

    n_expected = (max(poc_map.values()) + 1) if poc_map else 0
    if len(grids) != n_expected:
        raise RuntimeError(
            f"NVDEC decoded {len(grids)} frames from windowed stream, expected {n_expected} "
            f"(display-order POCs 0..{n_expected - 1})"
        )

    np.savez_compressed(
        npz_path,
        width=np.int32(w),
        height=np.int32(h),
        poc_real=np.array(list(poc_map.keys()), dtype=np.int32),
        poc_stream=np.array(list(poc_map.values()), dtype=np.int32),
        cu_type=np.stack([g["cu_type"] for g in grids]),
        mv0_x=np.stack([g["mv0_x"] for g in grids]),
        mv0_y=np.stack([g["mv0_y"] for g in grids]),
        mv1_x=np.stack([g["mv1_x"] for g in grids]),
        mv1_y=np.stack([g["mv1_y"] for g in grids]),
    )
    return npz_path, w, h, poc_map


@functools.lru_cache(maxsize=8)
def _nvdec_npz_meta(npz_path: str):
    with np.load(npz_path) as data:
        w, h = int(data["width"]), int(data["height"])
        poc_pairs = zip(data["poc_real"].tolist(), data["poc_stream"].tolist())
        n = int(data["cu_type"].shape[0])
    return w, h, dict(poc_pairs), n


@functools.lru_cache(maxsize=8)
def _load_nvdec_npz(npz_path: str):
    """Keep the raw 16x16 stacks in process memory (same role as _cached_by_poc)."""
    with np.load(npz_path) as data:
        return {
            "cu_type": np.asarray(data["cu_type"]),
            "mv0_x": np.asarray(data["mv0_x"]),
            "mv0_y": np.asarray(data["mv0_y"]),
            "mv1_x": np.asarray(data["mv1_x"]),
            "mv1_y": np.asarray(data["mv1_y"]),
        }


@functools.lru_cache(maxsize=8)
def _cached_by_poc(csv_path: str, pocs: tuple, w_motion: float, skip_penalty: float, w_size: float = 1.0, w_residual: float = 0.0):
    """Parse only the needed POCs' rows out of a hevc_dump CSV once per process,
    and keep the (POC -> [(cu, score)]) grouping in memory.

    A full-video CSV can be tens of millions of lines while a query only needs
    ~16 sampled frames; grep-filtering to just those POCs before any Python
    parsing (h2g.parse_csv_for_pocs) turns an ~O(24M-line) scan into an
    ~O(16-frames-worth-of-lines) one. This in turn also means repeat queries
    against the same video within a process hit this cache directly. Both were
    needed to fix the ~78s/query codec-mode latency documented in
    HEVC_Dump_Pipeline.md's performance-finding section -- the CSV *file* being
    cached on disk was not, by itself, enough."""
    t0 = time.time()
    cus = h2g.parse_csv_for_pocs(csv_path, pocs)
    _dt = (time.time() - t0) * 1000
    if os.environ.get("DEBUG_CSVPARSE_CALLS"):
        print(f"[DEBUG_CSVPARSE] call n_pocs={len(pocs)} dt_ms={_dt:.1f} csv_path={csv_path}", flush=True)
    timing.add("selector_csvparse_ms", _dt)
    t0 = time.time()
    by_poc = {}
    for cu in cus:
        by_poc.setdefault(cu["poc"], []).append((cu, h2g.score_cu(cu, w_motion, skip_penalty, w_size, w_residual)))
    timing.add("selector_scorecu_ms", (time.time() - t0) * 1000)
    return by_poc


@functools.lru_cache(maxsize=1024)
def _cached_frame_score_map(csv_path: str, poc: int, pocs: tuple, w_motion: float, skip_penalty: float, orig_w: int, orig_h: int, w_size: float = 1.0, w_residual: float = 0.0):
    """Full-resolution per-frame score map, built once per (video, POC) and reused
    across every spatial tile that needs a crop of it -- replaces re-looping over
    the frame's CU list (and repainting a canvas from scratch) once per tile."""
    by_poc = _cached_by_poc(csv_path, pocs, w_motion, skip_penalty, w_size, w_residual)
    t0 = time.time()
    result = h2g.build_frame_score_map(by_poc.get(poc, []), orig_w, orig_h)
    timing.add("selector_paintmap_ms", (time.time() - t0) * 1000)
    return result


@functools.lru_cache(maxsize=1024)
def _cached_nvdec_cu_scores(npz_path: str, poc: int, w_motion: float, skip_penalty: float):
    """Per-frame 16x16 score grid (not upsampled). Token rasterization maps
    pixel crop boxes into this CU-cell space."""
    packed = _load_nvdec_npz(npz_path)
    return h2g.score_cu_grid(
        packed["cu_type"][poc],
        packed["mv0_x"][poc], packed["mv0_y"][poc],
        packed["mv1_x"][poc], packed["mv1_y"][poc],
        w_motion, skip_penalty,
    )


def _sampled_frame_indices(video_path: str, num_frames: int):
    """Mirrors processing_nvila.py::_load_video_frames's frame-index selection
    exactly (same cv2 frame-count probing + np.linspace), without decoding frames."""
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_count > 0:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        if vidcap.grab():
            break
        frame_count -= 1
    vidcap.release()
    if frame_count <= 0:
        raise ValueError(f"Video '{video_path}' has no frames.")
    return np.round(np.linspace(0, frame_count - 1, num_frames)).astype(int).tolist()


def _find_closest_aspect_ratio_fn():
    """Import _find_closest_aspect_ratio from the already-loaded processing_nvila
    module (via sys.modules) instead of duplicating its selection logic."""
    for name, mod in sys.modules.items():
        if name.endswith("processing_nvila") and hasattr(mod, "_find_closest_aspect_ratio"):
            return mod._find_closest_aspect_ratio
    raise RuntimeError(
        "processing_nvila module not found in sys.modules -- build a processor first "
        "(e.g. via scripts.breakdown.processor.build) so its trust_remote_code module is loaded."
    )


def build_gazing_info(
    video_path: str,
    num_video_frames: int,
    num_video_frames_thumbnail: int,
    max_tiles_video: int,
    autogaze_max_num_frames: int,
    image_size: int,
    scales: list,
    patch_size: int,
    gazing_ratio_tile,
    gazing_ratio_thumbnail,
    w_motion: float = 1.0,
    skip_penalty: float = 0.1,
    w_size: float = 1.0,
    w_residual: float = 0.0,
    full_first_frame: bool = False,
    sampled_only: bool = False,
    gop_restart: int | None = None,
    backend: str = "libde265",
):
    """Build a codec-scored gazing_info dict for one video, matching the schema
    NVILAProcessor._get_gazing_info_from_videos produces for a single video:
    gazing_pos_tiles/num_gazing_each_frame_tiles/if_padded_gazing_tiles (each a
    (num_tiles, T_tile[, N]) tensor) and the *_thumbnails analogs.

    Unlike the autoregressive selector (which can emit a variable, EOS-terminated
    count per frame), this always selects a fixed top-k = round(total_patches *
    ratio) per frame, so if_padded is always False -- there's no padding to signal.

    full_first_frame: LLaVA-OneVision-2-style I-canvas treatment (An et al. 2026,
    sec 2.2) -- the anchor frame carries the video's global context, so instead of
    codec-scored top-k it keeps every patch (k = total_patches) on the first frame
    of each thumbnail sequence and of each tile's temporal chunk, leaving only the
    remaining (P-canvas-like) frames codec-scored.

    sampled_only: when True, motion/residual signal comes only from the sampled
    frames themselves (chained sequentially, no real-frame context window) --
    see `_encode_sampled_frames_only`. Much cheaper at high num_video_frames,
    at the cost of motion vectors possibly spanning large real-time gaps.

    gop_restart: when set (e.g. 16), forces an I-frame every `gop_restart`
    sampled frames instead of only at position 0 -- see
    `_encode_periodic_restart`. Combined with dense/consecutive frame_indices
    (num_video_frames == native frame count), this "restarts the coding" at
    each real AutoGaze-style chunk boundary, matching its actual independent,
    non-overlapping-chunk design rather than one continuous stream. Takes
    precedence over `sampled_only`'s encode-function choice (implies chained,
    no-real-context encoding either way).

    backend: "libde265" (true CU tree via dump_stats CSV; supports sampled_only/
    gop_restart) or "nvdec" (PyNvVideoCodec 16x16 decode-stats, no CSV --
    windowed encoding only so far, sampled_only/gop_restart not yet wired up),
    or "hevc_geo"/"hevc_ord" (hevc_autogaze.py's CU-size-vs-AutoGaze-scale
    kernel -- routed to a dedicated builder, see
    _build_gazing_info_hevc_autogaze, since it needs a resized encode rather
    than this function's native-resolution crop/rasterize flow; sampled_only/
    gop_restart aren't supported there either).
    """
    if backend in ("hevc_geo", "hevc_ord"):
        if sampled_only or gop_restart:
            raise NotImplementedError(
                "hevc_geo/hevc_ord backends only support the default windowed "
                "encoding so far -- sampled_only/gop_restart are libde265-only"
            )
        return _build_gazing_info_hevc_autogaze(
            video_path=video_path,
            num_video_frames=num_video_frames,
            num_video_frames_thumbnail=num_video_frames_thumbnail,
            max_tiles_video=max_tiles_video,
            autogaze_max_num_frames=autogaze_max_num_frames,
            image_size=image_size,
            scales=scales,
            patch_size=patch_size,
            gazing_ratio_tile=gazing_ratio_tile,
            gazing_ratio_thumbnail=gazing_ratio_thumbnail,
            full_first_frame=full_first_frame,
            kind=backend,
        )

    find_closest_aspect_ratio = _find_closest_aspect_ratio_fn()

    frame_indices = _sampled_frame_indices(video_path, num_video_frames)
    if backend == "nvdec":
        if sampled_only or gop_restart:
            raise NotImplementedError(
                "codec_nvdec backend only supports the default windowed encoding so "
                "far -- sampled_only/gop_restart are libde265-only"
            )
        npz_path, orig_w, orig_h, poc_map = get_or_build_nvdec_stats(video_path, frame_indices)

        def score_region(poc, box_x0, box_y0, box_w, box_h):
            t0 = time.time()
            grid = _cached_nvdec_cu_scores(npz_path, poc, w_motion, skip_penalty)
            result = h2g.rasterize_multiscale_from_cu_grid(
                grid, box_x0, box_y0, box_w, box_h, scales, patch_size
            )
            timing.add("selector_score_ms", (time.time() - t0) * 1000)
            return result
    elif backend == "libde265":
        csv_path, orig_w, orig_h, poc_map = get_or_build_stats(video_path, frame_indices, sampled_only, gop_restart)
        pocs = tuple(sorted(set(poc_map.values())))

        def score_region(poc, box_x0, box_y0, box_w, box_h):
            t0 = time.time()
            score_map = _cached_frame_score_map(
                csv_path, poc, pocs, w_motion, skip_penalty, orig_w, orig_h, w_size, w_residual
            )
            result = h2g.rasterize_multiscale_from_map(
                score_map, box_x0, box_y0, box_w, box_h, scales, patch_size
            )
            timing.add("selector_score_ms", (time.time() - t0) * 1000)
            return result
    else:
        raise ValueError(
            f"unknown codec backend {backend!r} (expected 'libde265', 'nvdec', 'hevc_geo', or 'hevc_ord')"
        )

    # --- replicate spatial tiling decision (processing_nvila.py::_preprocess_videos) ---
    aspect_ratio = orig_w / orig_h
    max_spatial_tiles = max(max_tiles_video, 1)
    target_ratios = sorted(
        {
            (i, j)
            for n in range(1, max_spatial_tiles + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if 1 <= i * j <= max_spatial_tiles
        },
        key=lambda x: x[0] * x[1],
    )
    cols, rows = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_w, orig_h, image_size)
    target_w, target_h = image_size * cols, image_size * rows
    num_spatial_tiles = cols * rows
    sx, sy = orig_w / target_w, orig_h / target_h  # resized-frame px -> original-video px

    temporal_chunks = num_video_frames // autogaze_max_num_frames
    assert temporal_chunks >= 1 and num_video_frames % autogaze_max_num_frames == 0, (
        f"num_video_frames ({num_video_frames}) must be divisible by "
        f"autogaze_max_num_frames ({autogaze_max_num_frames})"
    )
    T_tile = autogaze_max_num_frames

    grid_sizes = [s // patch_size for s in scales]
    total_patches = sum(g * g for g in grid_sizes)

    def topk_ratio(ratio, index):
        # NOTE: gazing_ratio_tile, when a list, is indexed by frame-WITHIN-tile
        # (length == autogaze's max_num_frames, e.g. 16), matching
        # AutoGazeModel.generate()'s `max_gaze_tokens_each_frame: int | (T,)`
        # convention -- NOT per-spatial-tile. See config.py's
        # `[0.2] + [0.06]*15` (keyframe-heavy schedule).
        r = ratio[index] if isinstance(ratio, (list, tuple)) else ratio
        return max(1, int(round(total_patches * r)))

    # --- tiles ---
    tile_pos, tile_counts = [], []
    for t_chunk in range(temporal_chunks):
        for spatial_idx in range(num_spatial_tiles):
            col, row = spatial_idx % cols, spatial_idx // cols
            box_x0, box_y0 = col * image_size * sx, row * image_size * sy
            box_w, box_h = image_size * sx, image_size * sy

            frame_pos, frame_counts = [], []
            for f_local in range(T_tile):
                poc = poc_map[frame_indices[t_chunk * T_tile + f_local]]
                scores = score_region(poc, box_x0, box_y0, box_w, box_h)
                k = total_patches if (full_first_frame and f_local == 0) else topk_ratio(gazing_ratio_tile, f_local)
                t0 = time.time()
                ranked = np.sort(np.argsort(-scores)[:k])  # ascending, matching _sort_gazing_pos_per_frame
                timing.add("selector_rank_ms", (time.time() - t0) * 1000)
                frame_pos.append(torch.as_tensor(ranked, dtype=torch.long))
                frame_counts.append(k)
            tile_pos.append(torch.cat(frame_pos))
            tile_counts.append(torch.tensor(frame_counts, dtype=torch.long))

    gazing_pos_tiles = torch.nn.utils.rnn.pad_sequence(tile_pos, batch_first=True, padding_value=0)
    if_padded_gazing_tiles = torch.zeros_like(gazing_pos_tiles, dtype=torch.bool)
    num_gazing_each_frame_tiles = torch.stack(tile_counts)

    # --- thumbnails (whole-frame region, no spatial cropping) ---
    if len(frame_indices) > num_video_frames_thumbnail:
        step = len(frame_indices) // num_video_frames_thumbnail
        thumb_indices = frame_indices[::step][:num_video_frames_thumbnail]
    else:
        thumb_indices = frame_indices

    thumb_pos, thumb_counts = [], []
    for thumb_i, real_idx in enumerate(thumb_indices):
        poc = poc_map[real_idx]
        scores = score_region(poc, 0, 0, orig_w, orig_h)
        if full_first_frame and thumb_i == 0:
            k = total_patches
        else:
            k = topk_ratio(gazing_ratio_thumbnail if gazing_ratio_thumbnail is not None else 1.0, 0)
        t0 = time.time()
        ranked = np.sort(np.argsort(-scores)[:k])
        timing.add("selector_rank_ms", (time.time() - t0) * 1000)
        thumb_pos.append(torch.as_tensor(ranked, dtype=torch.long))
        thumb_counts.append(k)

    gazing_pos_thumbnails = torch.nn.utils.rnn.pad_sequence(thumb_pos, batch_first=True, padding_value=0)
    if_padded_gazing_thumbnails = torch.zeros_like(gazing_pos_thumbnails, dtype=torch.bool)
    num_gazing_each_frame_thumbnails = torch.tensor(thumb_counts, dtype=torch.long).unsqueeze(1)

    return {
        "gazing_pos_tiles": [gazing_pos_tiles],
        "num_gazing_each_frame_tiles": [num_gazing_each_frame_tiles],
        "if_padded_gazing_tiles": [if_padded_gazing_tiles],
        "gazing_pos_thumbnails": [gazing_pos_thumbnails],
        "num_gazing_each_frame_thumbnails": [num_gazing_each_frame_thumbnails],
        "if_padded_gazing_thumbnails": [if_padded_gazing_thumbnails],
    }
