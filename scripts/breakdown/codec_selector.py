"""Codec-based (HEVC CU partition/motion) substitute for AutoGaze's autoregressive
selector, wired into the scripts/breakdown benchmark as mode "codec".

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
"""
import functools
import hashlib
import os
import platform
import sys

import cv2
import numpy as np
import torch

WINDOW = 4  # real frames of context before each scored frame (5 frames/window total)

_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)
from scripts import hevc_to_gaze as h2g  # noqa: E402

# Two separate native builds exist -- cmake_build (x86_64, built on the login
# node) and cmake_build_aarch64 (aarch64, built natively on GB200 compute
# nodes; libde265's encoder subtree needed -fPIC forced via
# -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_C_FLAGS=-fPIC
# -DCMAKE_CXX_FLAGS=-fPIC, see HEVC_Dump_Pipeline.md). Pick the one matching
# this process's own architecture.
_BUILD_DIR = "cmake_build_aarch64" if platform.machine() == "aarch64" else "cmake_build"
DUMP_STATS_BIN = os.path.join(_REPO_DIR, "scripts", "hevc_dump", _BUILD_DIR, "dump_stats")
CACHE_DIR = os.path.join(_REPO_DIR, "data", "hevc_dump_cache")


def _video_key(video_path: str, frame_indices) -> str:
    st = os.stat(video_path)
    frames_key = hashlib.sha1(str(sorted(set(frame_indices))).encode()).hexdigest()[:8]
    return hashlib.sha1(f"{video_path}:{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:16] + "_" + frames_key


def _extract_and_encode_windows(video_path: str, frame_indices, hevc_path: str):
    """Encode only small real-frame windows around each needed frame, not the
    whole video. For each target frame index, grabs it plus WINDOW real
    preceding frames (via a single sequential decode pass over the source --
    seeking isn't needed since we just skip frames outside the needed set),
    then encodes all windows back-to-back into one short HEVC stream with an
    explicit I-frame forced at every window's start (`pict_type = I`, not
    reliance on periodic `keyint` -- windows have variable length near the
    start of the video, so periodic keyint doesn't reliably land on window
    boundaries and can let motion data leak across windows).

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

    out = av.open(hevc_path, mode="w", format="hevc")
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


def get_or_build_stats(video_path: str, frame_indices):
    """Return (csv_path, width, height, poc_map) for a video's hevc_dump CSV,
    windowed-encoding + dumping it once and caching by (path, size, mtime,
    frame_indices)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _video_key(video_path, frame_indices)
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
    w, h, poc_map = _extract_and_encode_windows(video_path, frame_indices, hevc_path)

    yuv_path = os.path.join(CACHE_DIR, f"{key}.yuv")
    ret = os.system(f'"{DUMP_STATS_BIN}" "{hevc_path}" "{yuv_path}" "{csv_path}"')
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


@functools.lru_cache(maxsize=8)
def _cached_by_poc(csv_path: str, pocs: tuple, w_motion: float, skip_penalty: float):
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
    cus = h2g.parse_csv_for_pocs(csv_path, pocs)
    by_poc = {}
    for cu in cus:
        by_poc.setdefault(cu["poc"], []).append((cu, h2g.score_cu(cu, w_motion, skip_penalty)))
    return by_poc


@functools.lru_cache(maxsize=1024)
def _cached_frame_score_map(csv_path: str, poc: int, pocs: tuple, w_motion: float, skip_penalty: float, orig_w: int, orig_h: int):
    """Full-resolution per-frame score map, built once per (video, POC) and reused
    across every spatial tile that needs a crop of it -- replaces re-looping over
    the frame's CU list (and repainting a canvas from scratch) once per tile."""
    by_poc = _cached_by_poc(csv_path, pocs, w_motion, skip_penalty)
    return h2g.build_frame_score_map(by_poc.get(poc, []), orig_w, orig_h)


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
):
    """Build a codec-scored gazing_info dict for one video, matching the schema
    NVILAProcessor._get_gazing_info_from_videos produces for a single video:
    gazing_pos_tiles/num_gazing_each_frame_tiles/if_padded_gazing_tiles (each a
    (num_tiles, T_tile[, N]) tensor) and the *_thumbnails analogs.

    Unlike the autoregressive selector (which can emit a variable, EOS-terminated
    count per frame), this always selects a fixed top-k = round(total_patches *
    ratio) per frame, so if_padded is always False -- there's no padding to signal.
    """
    find_closest_aspect_ratio = _find_closest_aspect_ratio_fn()

    frame_indices = _sampled_frame_indices(video_path, num_video_frames)
    csv_path, orig_w, orig_h, poc_map = get_or_build_stats(video_path, frame_indices)
    pocs = tuple(sorted(set(poc_map.values())))

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

    def score_region(poc, box_x0, box_y0, box_w, box_h):
        score_map = _cached_frame_score_map(csv_path, poc, pocs, w_motion, skip_penalty, orig_w, orig_h)
        return h2g.rasterize_multiscale_from_map(score_map, box_x0, box_y0, box_w, box_h, scales, patch_size)

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
                k = topk_ratio(gazing_ratio_tile, f_local)
                ranked = np.sort(np.argsort(-scores)[:k])  # ascending, matching _sort_gazing_pos_per_frame
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
    for real_idx in thumb_indices:
        poc = poc_map[real_idx]
        scores = score_region(poc, 0, 0, orig_w, orig_h)
        k = topk_ratio(gazing_ratio_thumbnail if gazing_ratio_thumbnail is not None else 1.0, 0)
        ranked = np.sort(np.argsort(-scores)[:k])
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
