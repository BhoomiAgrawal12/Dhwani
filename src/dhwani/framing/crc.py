"""Parametric CRC engine (width, poly, init, refin, refout, xorout) and a
staged CRC hunter.

The hunter is deliberately bounded: it tests the configured candidate list
over plausible (data_start, crc_position) pairs rather than brute-forcing
the full parameter space (staged candidate pruning).
"""
from __future__ import annotations

import numpy as np


from functools import lru_cache


def _reflect(v: int, width: int) -> int:
    r = 0
    for i in range(width):
        if (v >> i) & 1:
            r |= 1 << (width - 1 - i)
    return r


_REFLECT8 = [_reflect(b, 8) for b in range(256)]


@lru_cache(maxsize=None)
def _crc_table(width: int, poly: int) -> tuple:
    """Byte-wise MSB-first lookup table (the CRC hunt calls crc_compute
    hundreds of thousands of times; bit loops dominated the bit layer)."""
    top, mask = 1 << (width - 1), (1 << width) - 1
    table = []
    for i in range(256):
        crc = i << (width - 8)
        for _ in range(8):
            crc = ((crc << 1) ^ poly) if crc & top else (crc << 1)
        table.append(crc & mask)
    return tuple(table)


def crc_compute(data: bytes, width: int, poly: int, init: int,
                refin: bool, refout: bool, xorout: int) -> int:
    crc = init
    mask = (1 << width) - 1
    if width >= 8:
        table, shift = _crc_table(width, poly), width - 8
        for byte in data:
            if refin:
                byte = _REFLECT8[byte]
            crc = ((crc << 8) & mask) ^ table[((crc >> shift) ^ byte) & 0xFF]
    else:
        topbit = 1 << (width - 1)
        for byte in data:
            if refin:
                byte = _REFLECT8[byte]
            crc ^= byte
            for _ in range(8):
                crc = ((crc << 1) ^ poly) if (crc & topbit) else (crc << 1)
                crc &= mask
    if refout:
        crc = _reflect(crc, width)
    return crc ^ xorout


CRC_PRESETS = [
    {"name": "CRC-16-CCITT-FALSE", "width": 16, "poly": 0x1021, "init": 0xFFFF, "refin": False, "refout": False, "xorout": 0x0000},
    {"name": "CRC-16-XMODEM", "width": 16, "poly": 0x1021, "init": 0x0000, "refin": False, "refout": False, "xorout": 0x0000},
    {"name": "CRC-16-ARC", "width": 16, "poly": 0x8005, "init": 0x0000, "refin": True, "refout": True, "xorout": 0x0000},
    {"name": "CRC-32", "width": 32, "poly": 0x04C11DB7, "init": 0xFFFFFFFF, "refin": True, "refout": True, "xorout": 0xFFFFFFFF},
    {"name": "CRC-8", "width": 8, "poly": 0x07, "init": 0x00, "refin": False, "refout": False, "xorout": 0x00},
    {"name": "CRC-16-X25", "width": 16, "poly": 0x1021, "init": 0xFFFF, "refin": True, "refout": True, "xorout": 0xFFFF},
]


def crc_hunt(frames: np.ndarray, candidates: list = None,
             min_pass_fraction: float = 0.9) -> list:
    """Search for a CRC in stacked frames.

    frames: (n_frames, frame_bits) 0/1 matrix (frame_bits % 8 == 0 assumed
    for byte-aligned CRCs; non-aligned positions are skipped).
    Tests: CRC over bytes [start:crc_pos) equals bytes [crc_pos:crc_pos+w).
    Returns list of hits sorted by pass fraction.
    """
    if candidates is None:
        candidates = CRC_PRESETS
    n_frames, frame_bits = frames.shape
    if frame_bits % 8 or n_frames < 2:
        return []
    frame_bytes = frame_bits // 8
    as_bytes = np.packbits(frames.astype(np.uint8), axis=1)
    hits = []
    for cand in candidates:
        w = cand["width"] // 8
        # CRC assumed at the end of a region ending at frame end or before
        for crc_end in (frame_bytes,):
            crc_pos = crc_end - w
            if crc_pos <= 0:
                continue
            for start in range(0, min(crc_pos, 8)):
                passes = 0
                total = min(n_frames, 64)
                # stop once min_pass_fraction is out of reach
                need = int(np.ceil(total * min_pass_fraction - 1e-9))
                max_fail = total - need
                for f in range(total):
                    if f - passes > max_fail:
                        break
                    row = as_bytes[f]
                    calc = crc_compute(row[start:crc_pos].tobytes(),
                                       cand["width"], cand["poly"], cand["init"],
                                       cand["refin"], cand["refout"], cand["xorout"])
                    stored = int.from_bytes(row[crc_pos:crc_pos + w].tobytes(), "big")
                    if calc == stored:
                        passes += 1
                frac = passes / total
                if frac >= min_pass_fraction:
                    hits.append({**{k: v for k, v in cand.items()},
                                 "data_start_byte": start,
                                 "crc_byte_offset": crc_pos,
                                 "passes": passes, "total": total,
                                 "pass_fraction": frac})
    hits.sort(key=lambda h: -h["pass_fraction"])
    return hits
