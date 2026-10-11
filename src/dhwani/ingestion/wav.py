"""WAV (RIFF) reader with defensive parsing of the non-standard 'auxi'
chunk that SDR recorders (SDRuno, HDSDR, SDR-Console) use to store centre
frequency and timestamp.

scipy.io.wavfile ignores auxi and struggles with some files, so we walk the
RIFF chunks ourselves for metadata and fall back to soundfile for awkward
sample formats.
"""
from __future__ import annotations

import os
import struct

import numpy as np


def walk_riff_chunks(path: str) -> dict:
    """Return {chunk_id: (offset, size)} for the top-level RIFF chunks."""
    chunks = {}
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        hdr = f.read(12)
        if len(hdr) < 12 or hdr[:4] not in (b"RIFF", b"RF64"):
            raise ValueError("not a RIFF/RF64 file")
        pos = 12
        while pos + 8 <= size:
            f.seek(pos)
            cid, csize = struct.unpack("<4sI", f.read(8))
            chunks[cid.decode("latin1").strip()] = (pos + 8, csize)
            pos += 8 + csize + (csize & 1)
    return chunks


def parse_auxi(path: str, offset: int, size: int) -> dict:
    """Parse the SDRuno/SDR-Console 'auxi' chunk. Layout (rfsoapyfile):
    two SYSTEMTIME structs (16 bytes each) then int32 centre frequency Hz.
    Parsed defensively: on any inconsistency return {}."""
    out = {}
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            raw = f.read(size)
        if len(raw) >= 36:
            st = struct.unpack("<8H", raw[:16])
            if 1900 < st[0] < 2100 and 1 <= st[1] <= 12 and 1 <= st[3] <= 31:
                out["timestamp"] = (f"{st[0]:04d}-{st[1]:02d}-{st[3]:02d}"
                                    f"T{st[4]:02d}:{st[5]:02d}:{st[6]:02d}Z")
            cf = struct.unpack("<i", raw[32:36])[0]
            if 0 < cf < 40_000_000_000:
                out["center_frequency"] = float(cf)
    except Exception:
        return {}
    return out


def read_wav(path: str) -> dict:
    """Read a WAV file; returns dict with samples (complex64 when 2-channel
    IQ), sample_rate, and metadata."""
    chunks = walk_riff_chunks(path)
    if "fmt" not in chunks:
        raise ValueError("WAV file has no fmt chunk")
    with open(path, "rb") as f:
        off, sz = chunks["fmt"]
        f.seek(off)
        fmt = f.read(min(sz, 16))
    audio_fmt, n_ch, rate, _, block_align, bits = struct.unpack("<HHIIHH", fmt)

    meta = {"sample_rate": float(rate), "channels": n_ch, "bits": bits,
            "audio_format": audio_fmt}
    if "auxi" in chunks:
        meta.update(parse_auxi(path, *chunks["auxi"]))

    if "data" not in chunks:
        raise ValueError("WAV file has no data chunk")
    doff, dsize = chunks["data"]
    dsize = min(dsize, os.path.getsize(path) - doff)

    dtype_map = {(1, 8): np.uint8, (1, 16): np.int16, (1, 32): np.int32,
                 (3, 32): np.float32, (3, 64): np.float64}
    key = (audio_fmt, bits)
    if key in dtype_map:
        dt = np.dtype(dtype_map[key])
        count = dsize // dt.itemsize
        raw = np.memmap(path, dtype=dt, mode="r", offset=doff, shape=(count,))
        if n_ch > 1:
            raw = raw[: (count // n_ch) * n_ch].reshape(-1, n_ch)
        meta["datatype"] = str(dt)
    elif key == (1, 24):
        # 24-bit PCM: no numpy dtype; decode via soundfile
        import soundfile as sf
        data, rate2 = sf.read(path, dtype="float32", always_2d=True)
        raw = data
        meta["datatype"] = "int24"
        meta["sample_rate"] = float(rate2)
    else:
        raise ValueError(f"unsupported WAV format {audio_fmt}/{bits}bit")

    meta["raw"] = raw
    return meta


def wav_to_complex(meta: dict) -> tuple:
    """Convert WAV channel data to complex64 baseband.

    2 channels -> I + jQ ("iq").  1 channel: baseband audio (energy from
    DC, e.g. an FM receiver's output) stays real at the file rate
    ("baseband"); anything else -> real_to_complex (rate halves, centre
    moves to fs/4; the caller adjusts the metadata) ("real").
    Returns (samples, kind)."""
    raw = meta["raw"]
    if raw.ndim == 2 and raw.shape[1] >= 2:
        i = _norm(raw[:, 0])
        q = _norm(raw[:, 1])
        return (i + 1j * q).astype(np.complex64), "iq"
    x = _norm(raw.ravel())
    if is_baseband(x):
        return x.astype(np.complex64), "baseband"
    return real_to_complex(x), "real"


def is_baseband(x: np.ndarray) -> bool:
    """Real signal whose band starts at DC: the lowest 1% of its power lies
    below fs/100. A data waveform out of an FM discriminator does; an RF or
    audio-tone carrier (SSB-received PSK, AFSK, the IF tests) does not."""
    from scipy.signal import welch
    x = x[: 1 << 20]
    f, p = welch(x - x.mean(), nperseg=min(4096, len(x)))
    c = np.cumsum(p) / (p.sum() + 1e-30)
    return bool(f[np.searchsorted(c, 0.01)] < 0.01)


def real_to_complex(x: np.ndarray) -> np.ndarray:
    """Real signal at fs -> complex baseband at fs/2: mix by -fs/4 so the
    0..fs/2 band sits around DC, half-band low-pass to drop the mirrored
    image, decimate by 2. The band fills the output, so CFAR never sees an
    empty half-spectrum. web/analyze.js realToComplex is the same filter."""
    from scipy.signal import firwin
    mix = np.array([1, -1j, -1, 1j], dtype=np.complex64)
    y = x.astype(np.float32) * mix[np.arange(len(x)) % 4]
    y = np.convolve(y, firwin(63, 0.5), mode="same")[::2]
    return y.astype(np.complex64)


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return (x.astype(np.float32) - 127.5) / 127.5
    if x.dtype == np.int16:
        return x.astype(np.float32) / 32768.0
    if x.dtype == np.int32:
        return x.astype(np.float32) / 2147483648.0
    return x.astype(np.float32)
