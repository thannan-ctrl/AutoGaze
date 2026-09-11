"""Codec-based (HEVC CU partition/motion) substitute for AutoGaze's autoregressive
selector, wired into the scripts/breakdown benchmark as modes "codec" and
"codec_nvdec".

Rather than intercepting NVILAProcessor mid-pipeline (crop-box bookkeeping isn't
retained by the time `_get_gazing_info_from_videos` runs), this module independently
replicates NVILA-HD's own frame-sampling and spatial-tiling math -- imported directly
from the loaded `processing_nvila` module, not reimplemented -- so patch indices line
up with what the real pipeline would produce for the same video_path/config. See
Codec_Selector_Feasibility.md for the full architecture writeup.

SOURCE HANDLING: ``codec_nvdec`` decodes H.264/HEVC inputs directly by default, so
it performs no CPU pixel decode or x265 transcode. Other codecs fall back to the
controlled windowed HEVC stream used by ``codec``. That fallback makes one
sequential source-decode pass but keeps only WINDOW+1 native-YUV frames live and
encodes only ~num_video_frames * (WINDOW+1) frames. Each target gets real temporal
context and an explicit I-frame at the window boundary. ``poc_map`` records the
target POC in this artificial stream explicitly.

Backends:
  libde265 ("codec")     -- dump_stats walks the true CU quad-tree and returns
                            only target POCs in a compact binary stream. NumPy
                            views that stream without CSV/text conversion.
  nvdec ("codec_nvdec")  -- sequential NVDEC decode-stats on a regular 16x16
                            grid. Decoded pixels remain in device memory and are
                            discarded; only selected, small stats grids cross to
                            NumPy. Optional .npz persistence runs asynchronously.
"""
import functools
import hashlib
import os
import platform
import subprocess
import sys
import threading
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import cv2
import numpy as np
import torch

WINDOW = 4  # real frames of context before each scored frame (5 frames/window total)
CACHE_FORMAT_VERSION = 2
_MEMORY_CACHE_SIZE = 8

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

# Benchmark mode name -> dump backend.
BACKEND_FOR_MODE = {"codec": "libde265", "codec_nvdec": "nvdec"}

_CACHE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="codec-cache")
_LIB_MEMORY_CACHE = OrderedDict()
_NVDEC_MEMORY_CACHE = OrderedDict()


def _remember(cache, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _MEMORY_CACHE_SIZE:
        cache.popitem(last=False)


def _atomic_write_bytes(path, payload):
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(payload)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _persist_bytes(path, payload):
    if os.environ.get("CODEC_PERSIST_CACHE", "1") != "0":
        _CACHE_EXECUTOR.submit(_atomic_write_bytes, path, payload)


def _video_key(video_path: str, frame_indices, variant: str = "libde265") -> str:
    st = os.stat(video_path)
    normalized_indices = sorted(set(int(i) for i in frame_indices))
    frames_key = hashlib.sha1(str(normalized_indices).encode()).hexdigest()[:8]
    source = f"v{CACHE_FORMAT_VERSION}:{variant}:{video_path}:{st.st_size}:{st.st_mtime_ns}"
    return hashlib.sha1(source.encode()).hexdigest()[:16] + "_" + frames_key


def _window_poc_map(frame_indices):
    """Deterministic target POCs for the concatenated context windows."""
    poc_map = {}
    next_poc = 0
    for index in sorted(set(int(i) for i in frame_indices)):
        next_poc += min(int(index), WINDOW) + 1
        poc_map[int(index)] = next_poc - 1
    return poc_map, next_poc


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

    Returns ``(width, height, poc_map, encoded_frame_count)``. ``poc_map`` maps
    each target source index to its display POC in the encoded stream.
    """
    import av
    from av.video.frame import PictureType

    frame_indices = [int(i) for i in frame_indices]
    if not frame_indices:
        raise ValueError("frame_indices must not be empty")
    if frame_indices != sorted(frame_indices):
        raise ValueError("frame_indices must be in nondecreasing display order")
    if frame_indices[0] < 0:
        raise ValueError("frame_indices must be non-negative")
    # np.linspace sampling can repeat indices for very short inputs. One stats
    # record is sufficient for every repeated occurrence downstream.
    frame_indices = sorted(set(frame_indices))
    windows = [list(range(max(0, idx - WINDOW), idx + 1)) for idx in frame_indices]

    src = av.open(video_path)
    vs = src.streams.video[0]
    w, h = vs.codec_context.width, vs.codec_context.height
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
    target_i = 0
    ring = deque(maxlen=WINDOW + 1)
    try:
        for frame_i, frame in enumerate(src.decode(vs)):
            # Retain the decoder's native YUV frame rather than converting every
            # selected frame to RGB and back to YUV for x265.  Only WINDOW+1
            # frames are live at a time.
            ring.append(frame.reformat(format="yuv420p"))
            while target_i < len(windows) and frame_i == frame_indices[target_i]:
                win = windows[target_i]
                local_frames = list(ring)[-len(win):]
                if len(local_frames) != len(win):
                    raise RuntimeError(
                        f"missing context for frame {frame_indices[target_i]}: "
                        f"wanted {len(win)}, got {len(local_frames)}"
                    )
                for j, vf in enumerate(local_frames):
                    vf.pts = new_poc
                    vf.time_base = Fraction(1, 25)
                    vf.pict_type = PictureType.I if j == 0 else PictureType.NONE
                    for packet in enc.encode(vf):
                        out.mux(packet)
                    new_poc += 1
                # Only target POCs are needed downstream. Context frames remain
                # in the bitstream solely to establish meaningful prediction.
                poc_map[frame_indices[target_i]] = new_poc - 1
                target_i += 1
            if target_i == len(windows):
                break
        if target_i != len(windows):
            raise RuntimeError(
                f"video ended after satisfying {target_i}/{len(windows)} requested frames"
            )
        for packet in enc.encode():
            out.mux(packet)
    finally:
        out.close()
        src.close()
    return w, h, poc_map, new_poc


def get_or_build_stats(video_path: str, frame_indices):
    """Return compact libde265 block stats, without YUV or CSV plumbing."""
    target_indices = sorted(set(int(i) for i in frame_indices))
    if not target_indices:
        raise ValueError("frame_indices must not be empty")
    if target_indices[0] < 0:
        raise ValueError("frame_indices must be non-negative")
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _video_key(video_path, target_indices, variant="libde265-binary")
    stats_path = os.path.join(CACHE_DIR, f"{key}.agcu")
    if stats_path in _LIB_MEMORY_CACHE or os.path.exists(stats_path):
        w, h, _by_poc = _load_lib_stats(stats_path)
        poc_map, _n_stream_frames = _window_poc_map(target_indices)
        return stats_path, w, h, poc_map

    hevc_path = os.path.join(CACHE_DIR, f"{key}.hevc")
    try:
        w, h, poc_map, n_stream_frames = _extract_and_encode_windows(
            video_path, target_indices, hevc_path
        )
        expected_poc_map, expected_n_frames = _window_poc_map(target_indices)
        if poc_map != expected_poc_map or n_stream_frames != expected_n_frames:
            raise RuntimeError("windowed encoder produced an inconsistent POC mapping")
        target_pocs = sorted(set(poc_map.values()))
        threads = int(os.environ.get("CODEC_LIBDE265_THREADS", min(os.cpu_count() or 1, 8)))
        proc = subprocess.run(
            [
                DUMP_STATS_BIN,
                hevc_path,
                "--binary",
                "--pocs", ",".join(str(p) for p in target_pocs),
                "--threads", str(max(threads, 0)),
            ],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"dump_stats failed on {video_path} (exit {proc.returncode}): {err}"
            )
        blob = proc.stdout
        parsed_w, parsed_h, by_poc = h2g.parse_block_stats_binary(blob)
        if (parsed_w, parsed_h) != (w, h):
            raise RuntimeError(
                f"libde265 reported {parsed_w}x{parsed_h}, expected {w}x{h}"
            )
        missing = set(target_pocs) - set(by_poc)
        if missing:
            raise RuntimeError(f"libde265 did not return target POCs {sorted(missing)}")
        _remember(_LIB_MEMORY_CACHE, stats_path, (blob, w, h, by_poc))
        _persist_bytes(stats_path, blob)
        return stats_path, w, h, poc_map
    finally:
        if os.path.exists(hevc_path):
            os.remove(hevc_path)


def _load_lib_stats(stats_path):
    cached = _LIB_MEMORY_CACHE.get(stats_path)
    if cached is not None:
        _LIB_MEMORY_CACHE.move_to_end(stats_path)
        return cached[1], cached[2], cached[3]
    with open(stats_path, "rb") as f:
        blob = f.read()
    w, h, by_poc = h2g.parse_block_stats_binary(blob)
    _remember(_LIB_MEMORY_CACHE, stats_path, (blob, w, h, by_poc))
    return w, h, by_poc


def _gpu_id() -> int:
    """Visible CUDA device index for PyNvVideoCodec. NVILA_DEVICE is already
    the index *within* CUDA_VISIBLE_DEVICES (e.g. cuda:0), matching gpu_id=0
    when the benchmark sets CUDA_VISIBLE_DEVICES to a single GPU."""
    dev = os.environ.get("NVILA_DEVICE", "cuda:0")
    if ":" in dev:
        return int(dev.split(":")[-1])
    return int(dev) if str(dev).isdigit() else 0


def _nvdec_dump_grids(path: str, width: int, height: int, **kwargs):
    """Decode ``path`` sequentially with NVDEC and return per-display-frame
    dicts {cu_type, mv0_x, mv0_y, mv1_x, mv1_y}, each (mh, mw). Pixels are
    discarded -- only the decode-stats buffer is kept."""
    return nvd.dump_nvdec_grids(path, width, height, gpu_id=_gpu_id(), **kwargs)


def _pack_nvdec_grids(grids, pocs):
    pocs = sorted(set(int(p) for p in pocs))
    missing = [p for p in pocs if p >= len(grids) or grids[p] is None]
    if missing:
        raise RuntimeError(f"NVDEC did not return statistics for target frames {missing}")
    selected = [grids[p] for p in pocs]
    packed = {
        "stream_poc": np.asarray(pocs, dtype=np.int32),
        "cu_type": np.stack([g["cu_type"] for g in selected]),
        "mv0_x": np.stack([g["mv0_x"] for g in selected]),
        "mv0_y": np.stack([g["mv0_y"] for g in selected]),
        "mv1_x": np.stack([g["mv1_x"] for g in selected]),
        "mv1_y": np.stack([g["mv1_y"] for g in selected]),
    }
    packed["poc_to_row"] = {poc: row for row, poc in enumerate(pocs)}
    return packed


def _write_nvdec_npz(npz_path, width, height, poc_map, n_frames, packed):
    tmp = f"{npz_path}.{os.getpid()}.{threading.get_ident()}.tmp.npz"
    save = np.savez_compressed if os.environ.get("CODEC_COMPRESS_CACHE", "0") == "1" else np.savez
    try:
        save(
            tmp,
            format_version=np.int32(CACHE_FORMAT_VERSION),
            width=np.int32(width),
            height=np.int32(height),
            n_frames=np.int32(n_frames),
            poc_real=np.asarray(list(poc_map.keys()), dtype=np.int32),
            poc_stream=np.asarray(list(poc_map.values()), dtype=np.int32),
            stream_poc=packed["stream_poc"],
            cu_type=packed["cu_type"],
            mv0_x=packed["mv0_x"],
            mv0_y=packed["mv0_y"],
            mv1_x=packed["mv1_x"],
            mv1_y=packed["mv1_y"],
        )
        os.replace(tmp, npz_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _persist_nvdec_npz(npz_path, width, height, poc_map, n_frames, packed):
    if os.environ.get("CODEC_PERSIST_CACHE", "1") != "0":
        _CACHE_EXECUTOR.submit(
            _write_nvdec_npz, npz_path, width, height, dict(poc_map), n_frames, packed
        )


def _existing_nvdec_cache(video_path, frame_indices, variant):
    key = _video_key(video_path, frame_indices, variant=variant)
    path = os.path.join(NVDEC_CACHE_DIR, f"{key}.npz")
    cached = _NVDEC_MEMORY_CACHE.get(path)
    if cached is not None:
        _NVDEC_MEMORY_CACHE.move_to_end(path)
        _packed, w, h, poc_map, _n = cached
        return path, w, h, dict(poc_map)
    if os.path.exists(path):
        _load_nvdec_npz(path)
        _packed, w, h, poc_map, _n = _NVDEC_MEMORY_CACHE[path]
        return path, w, h, poc_map
    return None


def _finish_nvdec_stats(npz_path, width, height, poc_map, grids):
    target_pocs = sorted(set(poc_map.values()))
    packed = _pack_nvdec_grids(grids, target_pocs)
    _remember(
        _NVDEC_MEMORY_CACHE,
        npz_path,
        (packed, int(width), int(height), dict(poc_map), len(grids)),
    )
    _persist_nvdec_npz(npz_path, width, height, poc_map, len(grids), packed)
    return npz_path, width, height, poc_map


def get_or_build_nvdec_stats(video_path: str, frame_indices, source_width=None, source_height=None):
    """Return selected NVDEC statistics, using source H.264/HEVC when possible.

    ``CODEC_NVDEC_SOURCE_MODE=auto`` (default) avoids the CPU decode/x265
    transcode for H.264/HEVC sources. ``off`` retains controlled windowed-x265
    semantics; ``force`` rejects unsupported source codecs instead of falling
    back. Only target frames' stats are copied to NumPy in either path.
    """
    os.makedirs(NVDEC_CACHE_DIR, exist_ok=True)
    source_mode = os.environ.get("CODEC_NVDEC_SOURCE_MODE", "auto").lower()
    if source_mode not in {"auto", "off", "force"}:
        raise ValueError("CODEC_NVDEC_SOURCE_MODE must be auto, off, or force")

    target_indices = sorted(set(int(i) for i in frame_indices))
    if not target_indices:
        raise ValueError("frame_indices must not be empty")
    if target_indices[0] < 0:
        raise ValueError("frame_indices must be non-negative")

    if source_mode != "off":
        cached = _existing_nvdec_cache(video_path, frame_indices, "nvdec-source")
        if cached is not None:
            return cached

    # A fallback cache can only have been produced after a previous direct
    # source attempt found this codec unsupported (or source mode was off).
    # Reuse it without reopening and reprobeing the input on every auto query.
    if source_mode != "force":
        fallback_cached = _existing_nvdec_cache(
            video_path, frame_indices, "nvdec-x265-windowed"
        )
        if fallback_cached is not None:
            return fallback_cached

    if source_mode != "off":
        if source_width is None or source_height is None:
            source_width, source_height = _video_dimensions(video_path)
        poc_map = {i: i for i in target_indices}
        key = _video_key(video_path, frame_indices, variant="nvdec-source")
        npz_path = os.path.join(NVDEC_CACHE_DIR, f"{key}.npz")
        try:
            grids = _nvdec_dump_grids(
                video_path,
                source_width,
                source_height,
                max_frames=target_indices[-1] + 1,
                keep_frames=target_indices,
                require_stats_codec=True,
            )
            return _finish_nvdec_stats(
                npz_path, source_width, source_height, poc_map, grids
            )
        except nvd.UnsupportedDecodeStatsCodec:
            if source_mode == "force":
                raise

    key = _video_key(video_path, frame_indices, variant="nvdec-x265-windowed")
    npz_path = os.path.join(NVDEC_CACHE_DIR, f"{key}.npz")
    hevc_path = os.path.join(NVDEC_CACHE_DIR, f"{key}.hevc")
    try:
        w, h, poc_map, n_stream_frames = _extract_and_encode_windows(
            video_path, target_indices, hevc_path
        )
        target_pocs = sorted(set(poc_map.values()))
        grids = _nvdec_dump_grids(
            hevc_path,
            w,
            h,
            max_frames=n_stream_frames,
            keep_frames=target_pocs,
            require_stats_codec=True,
        )
        if len(grids) != n_stream_frames:
            raise RuntimeError(
                f"NVDEC decoded {len(grids)} frames from windowed stream, "
                f"expected {n_stream_frames}"
            )
        return _finish_nvdec_stats(npz_path, w, h, poc_map, grids)
    finally:
        if os.path.exists(hevc_path):
            os.remove(hevc_path)


def _load_nvdec_npz(npz_path: str):
    """Load metadata and selected stacks in one pass, once per process."""
    cached = _NVDEC_MEMORY_CACHE.get(npz_path)
    if cached is not None:
        _NVDEC_MEMORY_CACHE.move_to_end(npz_path)
        return cached[0]
    with np.load(npz_path) as data:
        w, h = int(data["width"]), int(data["height"])
        poc_pairs = zip(data["poc_real"].tolist(), data["poc_stream"].tolist())
        poc_map = dict(poc_pairs)
        n_frames = int(data["n_frames"])
        packed = {
            "stream_poc": np.asarray(data["stream_poc"]),
            "cu_type": np.asarray(data["cu_type"]),
            "mv0_x": np.asarray(data["mv0_x"]),
            "mv0_y": np.asarray(data["mv0_y"]),
            "mv1_x": np.asarray(data["mv1_x"]),
            "mv1_y": np.asarray(data["mv1_y"]),
        }
    packed["poc_to_row"] = {
        int(poc): row for row, poc in enumerate(packed["stream_poc"].tolist())
    }
    _remember(_NVDEC_MEMORY_CACHE, npz_path, (packed, w, h, poc_map, n_frames))
    return packed


@functools.lru_cache(maxsize=1024)
def _cached_frame_score_map(stats_path: str, poc: int, w_motion: float, skip_penalty: float, orig_w: int, orig_h: int):
    """Full-resolution per-frame score map, built once per (video, POC) and reused
    across every spatial tile that needs a crop of it -- replaces re-looping over
    the frame's CU list (and repainting a canvas from scratch) once per tile."""
    _w, _h, by_poc = _load_lib_stats(stats_path)
    blocks = by_poc.get(poc, np.empty(0, dtype=h2g.BLOCK_STATS_DTYPE))
    scores = h2g.score_block_array(blocks, w_motion, skip_penalty)
    return h2g.build_frame_score_map_from_blocks(blocks, scores, orig_w, orig_h)


@functools.lru_cache(maxsize=1024)
def _cached_nvdec_cu_scores(npz_path: str, poc: int, w_motion: float, skip_penalty: float):
    """Per-frame 16x16 score grid (not upsampled). Token rasterization maps
    pixel crop boxes into this CU-cell space."""
    packed = _load_nvdec_npz(npz_path)
    row = packed["poc_to_row"][poc]
    return h2g.score_cu_grid(
        packed["cu_type"][row],
        packed["mv0_x"][row], packed["mv0_y"][row],
        packed["mv1_x"][row], packed["mv1_y"][row],
        w_motion, skip_penalty,
    )


def _sampled_frame_info(video_path: str, num_frames: int):
    """Mirrors processing_nvila.py::_load_video_frames's frame-index selection
    exactly and returns ``(indices, width, height)``."""
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_count > 0:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        if vidcap.grab():
            break
        frame_count -= 1
    vidcap.release()
    if frame_count <= 0:
        raise ValueError(f"Video '{video_path}' has no frames.")
    indices = np.round(np.linspace(0, frame_count - 1, num_frames)).astype(int).tolist()
    return indices, width, height


def _sampled_frame_indices(video_path: str, num_frames: int):
    return _sampled_frame_info(video_path, num_frames)[0]


def _video_dimensions(video_path: str):
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    try:
        return (
            int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        vidcap.release()


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

    ``backend`` is ``libde265`` (true CU tree via compact dump_stats output) or
    ``nvdec`` (PyNvVideoCodec 16x16 decode stats).
    """
    find_closest_aspect_ratio = _find_closest_aspect_ratio_fn()

    frame_indices, source_w, source_h = _sampled_frame_info(video_path, num_video_frames)
    if backend == "nvdec":
        npz_path, orig_w, orig_h, poc_map = get_or_build_nvdec_stats(
            video_path, frame_indices, source_width=source_w, source_height=source_h
        )

        def score_region(poc, box_x0, box_y0, box_w, box_h):
            grid = _cached_nvdec_cu_scores(npz_path, poc, w_motion, skip_penalty)
            return h2g.rasterize_multiscale_from_cu_grid(
                grid, box_x0, box_y0, box_w, box_h, scales, patch_size
            )
    elif backend == "libde265":
        stats_path, orig_w, orig_h, poc_map = get_or_build_stats(video_path, frame_indices)

        def score_region(poc, box_x0, box_y0, box_w, box_h):
            score_map = _cached_frame_score_map(
                stats_path, poc, w_motion, skip_penalty, orig_w, orig_h
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
                ranked = h2g.topk_sorted_indices(scores, k)
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
        ranked = h2g.topk_sorted_indices(scores, k)
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
