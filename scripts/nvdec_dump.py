"""Sequential NVDEC decode-stats dump (Annex-B elementary streams or containers).

SimpleDecoder is a seek/index wrapper: it calls SetPTSOffset, requires a
frame count, and rejects elementary streams. This module uses the core
CreateDemuxer + CreateDecoder path instead, which walks packets forward and
does not assume a seekable container.

Tried in order:
  1. CreateDemuxer(callback) -- documented non-seekable / streaming path
  2. CreateDemuxer(filename=path) -- FFmpeg sequential demux (works for
     Annex-B .hevc as well as mp4); used if the callback constructor fails
"""
from __future__ import annotations

import subprocess
import sys
import time

import numpy as np

from scripts import hevc_to_gaze as h2g

_STATS_KEYS = ("qp_luma", "cu_type", "mv0_x", "mv0_y", "mv1_x", "mv1_y")


def import_pynv():
    try:
        import PyNvVideoCodec as nvc
    except ImportError as e:
        raise ImportError(
            "PyNvVideoCodec is required (pip install PyNvVideoCodec) plus a driver "
            "that supports NVDEC decode stats (Video Codec SDK 13.1 / driver 590+)."
        ) from e
    return nvc


class _ForwardFile:
    """Feed a file to CreateDemuxer without seeking."""

    def __init__(self, path):
        self._fh = open(path, "rb")

    def close(self):
        fh = getattr(self, "_fh", None)
        if fh is not None:
            fh.close()
            self._fh = None

    def feed_chunk(self, demuxer_buffer):
        if self._fh is None:
            return 0
        n = len(demuxer_buffer)
        data = self._fh.read(n)
        if not data:
            return 0
        demuxer_buffer[: len(data)] = data
        return len(data)


def as_mb_grid(arr, frame_w, frame_h, cu_size=h2g.NVDEC_CU_SIZE):
    """Reshape a ParseDecodeStats field (flat raster or already 2D) to (mh, mw)."""
    mw = (frame_w + cu_size - 1) // cu_size
    mh = (frame_h + cu_size - 1) // cu_size
    a = np.asarray(arr)
    if a.ndim == 2:
        if a.shape != (mh, mw):
            raise RuntimeError(
                f"decode-stats grid {a.shape} != {(mh, mw)} for frame {frame_w}x{frame_h}"
            )
        return a
    if a.size != mh * mw:
        raise RuntimeError(
            f"decode-stats length {a.size} != {mh}*{mw} 16x16 cells for frame {frame_w}x{frame_h}"
        )
    return a.reshape(mh, mw)


def _as_stats_dict(parsed):
    if isinstance(parsed, dict):
        return parsed
    if parsed is None:
        return {}
    out = {}
    for i, key in enumerate(_STATS_KEYS):
        if i < len(parsed):
            out[key] = parsed[i]
    return out


def _iter_frames(result):
    if result is None:
        return
    if hasattr(result, "ParseDecodeStats"):
        yield result
        return
    try:
        frames = iter(result)
    except TypeError:
        return
    for frame in frames:
        if frame is not None:
            yield frame


def _create_decoder(nvc, codec, gpu_id):
    attempts = [
        dict(gpuid=gpu_id, codec=codec, usedevicememory=True, enableDecodeStats=True),
        dict(
            gpuid=gpu_id,
            codec=codec,
            cudacontext=0,
            cudastream=0,
            usedevicememory=True,
            enableDecodeStats=True,
        ),
        dict(gpuid=gpu_id, codec=codec, use_device_memory=True, enableDecodeStats=True),
        dict(gpu_id=gpu_id, codec=codec, usedevicememory=True, enableDecodeStats=True),
        dict(gpuid=gpu_id, codec=codec, usedevicememory=True, enable_decode_stats=True),
    ]
    last = None
    for kw in attempts:
        try:
            return nvc.CreateDecoder(**kw)
        except TypeError as e:
            last = e
    raise TypeError(
        f"PyNvVideoCodec.CreateDecoder did not accept decode-stats kwargs; last error: {last}"
    ) from last


def _create_demuxer(nvc, path):
    """Prefer a non-seekable callback demuxer; fall back to filename demux.

    CreateDemuxer is FFmpeg-side (no GPU). Any callback constructor failure
    is retried with a path so Annex-B files can still probe via extension.
    """
    feeder = _ForwardFile(path)
    try:
        return nvc.CreateDemuxer(feeder.feed_chunk), feeder
    except Exception:
        feeder.close()

    last = None
    for args, kwargs in (((), {"filename": path}), ((path,), {})):
        try:
            return nvc.CreateDemuxer(*args, **kwargs), None
        except TypeError as e:
            last = e
    raise TypeError(f"CreateDemuxer rejected path {path!r}; last error: {last}") from last


def _codec_id(demuxer):
    for name in ("GetNvCodecId", "getNvCodecId", "GetCodecId"):
        fn = getattr(demuxer, name, None)
        if callable(fn):
            return fn()
    raise RuntimeError("demuxer has no GetNvCodecId()")


def _loaded_extension():
    mod = sys.modules.get("_PyNvVideoCodec")
    return getattr(mod, "__file__", None) or "unknown"


def _nvidia_smi():
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if proc.returncode == 0:
            return proc.stdout.strip() or "(empty nvidia-smi)"
        err = (proc.stderr or proc.stdout or "").strip()
        return f"nvidia-smi exit {proc.returncode}: {err}"
    except Exception as e:
        return f"unavailable ({e})"


def _try_decoder_caps(nvc, gpu_id, codec):
    fn = getattr(nvc, "GetDecoderCaps", None)
    if not callable(fn):
        return "GetDecoderCaps not in this PyNvVideoCodec build"
    chroma_enum = getattr(nvc, "cudaVideoChromaFormat", None)
    chroma = 1
    if chroma_enum is not None:
        chroma = getattr(chroma_enum, "YUV420", getattr(chroma_enum, "Chroma420", 1))
    attempts = [
        lambda: fn(gpuid=gpu_id, codec=codec, chromaformat=chroma, bitdepth=8),
        lambda: fn(gpu_id, codec, chroma, 8),
    ]
    last = None
    for call in attempts:
        try:
            caps = call()
            fields = []
            for name in (
                "codec_id", "chromaformat_id", "bitdepth", "num_decoder_engines",
                "width_min", "width_max", "height_min", "height_max", "mb_num_max",
            ):
                if hasattr(caps, name):
                    fields.append(f"{name}={getattr(caps, name)}")
            return ", ".join(fields) if fields else repr(caps)
        except TypeError as e:
            last = e
        except Exception as e:
            return f"GetDecoderCaps failed: {e}"
    return f"GetDecoderCaps signature mismatch: {last}"


def describe_nvdec_env(nvc, gpu_id, codec=None):
    lines = [
        f"PyNvVideoCodec {getattr(nvc, '__version__', '?')}",
        f"extension={_loaded_extension()}",
        f"gpu_id={gpu_id}",
        f"nvidia-smi: {_nvidia_smi()}",
        "decode-stats need Video Codec SDK 13.1 / display driver 590+ "
        "(SDK 13.0 HEVC decode on driver 570 is not enough)",
    ]
    if codec is not None:
        lines.append(f"codec={codec}")
        lines.append(f"GetDecoderCaps: {_try_decoder_caps(nvc, gpu_id, codec)}")
    return "\n".join(lines)


def _no_stats_error(path, nvc, gpu_id, codec, demuxer):
    return RuntimeError(
        "NVDEC decoded the bitstream but returned no decode stats.\n"
        "PyNvVideoCodec logs 'Decode statistics requested but not supported, "
        "disabling decode stats' when cuvidGetDecoderCaps() has the feature off. "
        "That is a GPU/driver capability check, not a container or seek issue "
        f"(sequential demux of {path!r} already produced frames).\n"
        f"demuxer={type(demuxer).__name__}\n"
        f"{describe_nvdec_env(nvc, gpu_id, codec)}\n"
        "Check nvidia-smi driver_version; if it is below 590, codec_nvdec cannot "
        "work on this node. Use mode 'codec' (libde265) instead."
    )


def dump_nvdec_grids(
    path,
    width,
    height,
    max_frames=None,
    gpu_id=0,
    include_qp=False,
    verbose=False,
    timings=None,
):
    """Decode `path` sequentially and return per-display-frame 16x16 grids.

    Each dict has cu_type / mv0_x / mv0_y / mv1_x / mv1_y (and qp_luma if
    include_qp and the field is present). Pixels are discarded.

    If `timings` is a dict, it is filled with demuxer_ms / decoder_ms /
    decode_ms / teardown_ms (CreateDecoder is usually the 4-frame bottleneck).
    """
    def _stamp(key, t0):
        if timings is not None:
            timings[key] = (time.perf_counter() - t0) * 1000.0

    nvc = import_pynv()
    t0 = time.perf_counter()
    demuxer, feeder = _create_demuxer(nvc, path)
    codec = _codec_id(demuxer)
    _stamp("demuxer_ms", t0)

    t0 = time.perf_counter()
    decoder = _create_decoder(nvc, codec, gpu_id)
    _stamp("decoder_ms", t0)
    if verbose:
        print(describe_nvdec_env(nvc, gpu_id, codec))
        print(f"demuxer={type(demuxer).__name__}")
    zero = np.zeros(
        (
            (height + h2g.NVDEC_CU_SIZE - 1) // h2g.NVDEC_CU_SIZE,
            (width + h2g.NVDEC_CU_SIZE - 1) // h2g.NVDEC_CU_SIZE,
        ),
        dtype=np.int16,
    )
    grids = []
    printed = False

    def _take_frame(frame):
        nonlocal printed
        size = getattr(frame, "decode_stats_size", 0)
        if size <= 0:
            raise _no_stats_error(path, nvc, gpu_id, codec, demuxer)
        stats = _as_stats_dict(frame.ParseDecodeStats())
        if verbose and not printed:
            printed = True
            keys = {k: (np.asarray(v).shape, str(np.asarray(v).dtype)) for k, v in stats.items()}
            print(f"ParseDecodeStats fields (frame 0): {keys}")
            print(f"  decode_stats_size={size}  demux={type(demuxer).__name__}")

        def _grid(name, dtype):
            if name not in stats:
                return zero.astype(dtype, copy=True)
            return as_mb_grid(stats[name], width, height).astype(dtype, copy=False)

        rec = {
            "cu_type": _grid("cu_type", np.uint8),
            "mv0_x": _grid("mv0_x", np.int16),
            "mv0_y": _grid("mv0_y", np.int16),
            "mv1_x": _grid("mv1_x", np.int16),
            "mv1_y": _grid("mv1_y", np.int16),
        }
        if include_qp:
            rec["qp_luma"] = (
                as_mb_grid(stats["qp_luma"], width, height) if "qp_luma" in stats else None
            )
        grids.append(rec)
        return max_frames is not None and len(grids) >= max_frames

    t0 = time.perf_counter()
    try:
        done = False
        for packet in demuxer:
            for frame in _iter_frames(decoder.Decode(packet)):
                if _take_frame(frame):
                    done = True
                    break
            if done:
                break
        if not done:
            flush = getattr(decoder, "Flush", None)
            if callable(flush):
                for frame in _iter_frames(flush()):
                    if _take_frame(frame):
                        break
    finally:
        _stamp("decode_ms", t0)
        t1 = time.perf_counter()
        if feeder is not None:
            feeder.close()
        for obj in (decoder, demuxer):
            close = getattr(obj, "close", None)
            if callable(close):
                close()
        del decoder, demuxer
        _stamp("teardown_ms", t1)
    return grids
