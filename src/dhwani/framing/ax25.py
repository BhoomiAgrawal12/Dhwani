"""AX.25 over G3RUH: descramble, NRZI decode, HDLC flags, bit unstuffing,
CRC-16/X.25.  Works on a demodulated bit stream of unknown polarity."""
from __future__ import annotations

import numpy as np

from .crc import crc_compute

FLAG = np.array([0, 1, 1, 1, 1, 1, 1, 0], np.uint8)
MIN_FRAME_BYTES = 17            # 2 addresses + control + CRC


def nrzi_decode(bits: np.ndarray) -> np.ndarray:
    """bit = 1 if b[n] == b[n-1] else 0."""
    b = np.asarray(bits, np.uint8)
    return (b[1:] == b[:-1]).astype(np.uint8)


def g3ruh_descramble(bits: np.ndarray) -> np.ndarray:
    """Multiplicative x^17 + x^12 + 1: y[n] = x[n] ^ x[n-12] ^ x[n-17]."""
    x = np.asarray(bits, np.uint8)
    return (x[17:] ^ x[5:-12] ^ x[:-17]).astype(np.uint8)


def _unstuff(bits: np.ndarray):
    out, run = [], 0
    for b in bits.tolist():
        if run == 5:
            if b:
                return None             # 6 ones inside a frame: abort
            run = 0
            continue
        out.append(b)
        run = run + 1 if b else 0
    return np.array(out, np.uint8)


def hdlc_frames(bits: np.ndarray) -> list:
    """Byte strings between flags, unstuffed, LSB-first."""
    bits = np.asarray(bits, np.uint8)
    if len(bits) < 8:
        return []
    win = np.lib.stride_tricks.sliding_window_view(bits, 8)
    starts = np.flatnonzero((win == FLAG).all(axis=1))
    frames = []
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a < 8 + MIN_FRAME_BYTES * 8:
            continue
        body = _unstuff(bits[a + 8:b])
        if body is None or len(body) % 8:
            continue
        frames.append(np.packbits(body, bitorder="little").tobytes())
    return frames


def crc_ok(frame: bytes) -> bool:
    if len(frame) < MIN_FRAME_BYTES:
        return False
    calc = crc_compute(frame[:-2], 16, 0x1021, 0xFFFF, True, True, 0xFFFF)
    return calc == int.from_bytes(frame[-2:], "little")


def callsigns(frame: bytes) -> list:
    """Address field: 7-byte entries, characters are byte >> 1, the last
    entry has bit 0 set."""
    out = []
    for i in range(0, len(frame) - 6, 7):
        e = frame[i:i + 7]
        name = "".join(chr(c >> 1) for c in e[:6]).strip()
        if not name.isprintable():
            break
        out.append(f"{name}-{(e[6] >> 1) & 0xF}")
        if e[6] & 1:
            break
    return out


def decode_ax25(bits: np.ndarray) -> list:
    """CRC-valid AX.25 frames in a bit stream, trying both polarities and
    both descramble/NRZI orders.  [{"data", "callsigns"}, ...]"""
    bits = np.asarray(bits, np.uint8)
    if len(bits) < 64:
        return []
    best = []
    for pol in (bits, 1 - bits):
        for stream in (nrzi_decode(g3ruh_descramble(pol)),
                       g3ruh_descramble(nrzi_decode(pol))):
            good = [f for f in hdlc_frames(stream) if crc_ok(f)]
            if len(good) > len(best):
                best = good
        if best:
            break
    return [{"data": f, "callsigns": callsigns(f)} for f in best]
