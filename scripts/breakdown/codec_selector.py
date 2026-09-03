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

import cv2
import numpy as np
import torch

WINDOW = 4  # real frames of context before each scored frame (5 frames/window total)

_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)
from scripts import hevc_to_gaze as h2g  # noqa: E402
from scripts import nvdec_dump as nvd  # noqa: E402

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

# Benchmark mode name -> dump backend. "codec" keeps the original libde265/CSV
# path; "codec_nvdec" is the NVDEC 16x16 variant.
BACKEND_FOR_MODE = {"codec": "libde265", "codec_nvdec": "nvdec"}


def _video_key(video_path: str, frame_indices) -> str:
    st = os.stat(video_path)
    frames_key = hashlib.sha1(str(sorted(set(frame_indices))).encode()).hexdigest()[:8]
    return hashlib.sha1(f"{video_path}:{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:16] + "_" + frames_key


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
    w, h, poc_map = _extract_and_encode_windows(video_path, frame_indices, hevc_path)
    try:
        grids = _nvdec_dump_grids(hevc_path, w, h)
    finally:
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
    backend: str = "libde265",
):
    """Build a codec-scored gazing_info dict for one video, matching the schema
    NVILAProcessor._get_gazing_info_from_videos produces for a single video:
    gazing_pos_tiles/num_gazing_each_frame_tiles/if_padded_gazing_tiles (each a
    (num_tiles, T_tile[, N]) tensor) and the *_thumbnails analogs.

    Unlike the autoregressive selector (which can emit a variable, EOS-terminated
    count per frame), this always selects a fixed top-k = round(total_patches *
    ratio) per frame, so if_padded is always False -- there's no padding to signal.

    `backend`: "libde265" (true CU tree via dump_stats CSV) or "nvdec"
    (PyNvVideoCodec 16x16 decode-stats, no CSV).
    """
    find_closest_aspect_ratio = _find_closest_aspect_ratio_fn()

    frame_indices = _sampled_frame_indices(video_path, num_video_frames)
    if backend == "nvdec":
        npz_path, orig_w, orig_h, poc_map = get_or_build_nvdec_stats(video_path, frame_indices)

        def score_region(poc, box_x0, box_y0, box_w, box_h):
            grid = _cached_nvdec_cu_scores(npz_path, poc, w_motion, skip_penalty)
            return h2g.rasterize_multiscale_from_cu_grid(
                grid, box_x0, box_y0, box_w, box_h, scales, patch_size
            )
    elif backend == "libde265":
        csv_path, orig_w, orig_h, poc_map = get_or_build_stats(video_path, frame_indices)
        pocs = tuple(sorted(set(poc_map.values())))

        def score_region(poc, box_x0, box_y0, box_w, box_h):
            score_map = _cached_frame_score_map(
                csv_path, poc, pocs, w_motion, skip_penalty, orig_w, orig_h
            )
            return h2g.rasterize_multiscale_from_map(
                score_map, box_x0, box_y0, box_w, box_h, scales, patch_size
            )
    else:
        raise ValueError(f"unknown codec backend {backend!r} (expected 'libde265' or 'nvdec')")

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
