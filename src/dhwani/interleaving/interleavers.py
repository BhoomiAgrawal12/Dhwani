"""Interleaver encoders + exact inverses (used by the synthetic factory and
by the blind identifier to test hypotheses).

Bits come back as uint8 whatever integer or bool type goes in; floating
input (an LLR array) keeps its dtype, so it goes through the same
permutation as the bits it belongs to."""
from __future__ import annotations

import numpy as np


def _as_stream(bits) -> np.ndarray:
    a = np.asarray(bits)
    return a if np.issubdtype(a.dtype, np.floating) else a.astype(np.uint8, copy=False)


def block_interleave(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Write row-wise, read column-wise, block by block."""
    P = rows * cols
    n = (len(bits) // P) * P
    m = _as_stream(bits[:n]).reshape(-1, rows, cols)
    return m.transpose(0, 2, 1).reshape(-1)


def block_deinterleave(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    P = rows * cols
    n = (len(bits) // P) * P
    m = _as_stream(bits[:n]).reshape(-1, cols, rows)
    return m.transpose(0, 2, 1).reshape(-1)


def _conv_perm(n: int, branches: int, delay: int) -> np.ndarray:
    """Output position for each input position of a multiplexed convolutional
    interleaver (branch b delays by b*delay symbols); realised as a fixed
    permutation over a stream of length n (positions that would come from
    'before the start' wrap around, which keeps it invertible for testing)."""
    idx = np.arange(n)
    b = idx % branches
    out = idx + b * branches * delay
    return np.argsort(out % n, kind="stable")


def conv_interleave(bits: np.ndarray, branches: int, delay: int) -> np.ndarray:
    n = len(bits)
    perm = _conv_perm(n, branches, delay)
    return _as_stream(bits)[perm]


def conv_deinterleave(bits: np.ndarray, branches: int, delay: int) -> np.ndarray:
    bits = _as_stream(bits)
    n = len(bits)
    perm = _conv_perm(n, branches, delay)
    out = np.empty(n, dtype=bits.dtype)
    out[perm] = bits
    return out


def helical_interleave(bits: np.ndarray, rows: int, cols: int, step: int) -> np.ndarray:
    """Block interleaver composed with a per-column cyclic shift of `step`
    (diagonal/helical read-out)."""
    P = rows * cols
    n = (len(bits) // P) * P
    # a copy: the column shifts below are in place
    m = np.array(_as_stream(bits[:n])).reshape(-1, rows, cols)
    for c in range(cols):
        m[:, :, c] = np.roll(m[:, :, c], (c * step) % rows, axis=1)
    return m.transpose(0, 2, 1).reshape(-1)


def helical_deinterleave(bits: np.ndarray, rows: int, cols: int, step: int) -> np.ndarray:
    P = rows * cols
    n = (len(bits) // P) * P
    m = _as_stream(bits[:n]).reshape(-1, cols, rows).transpose(0, 2, 1).copy()
    for c in range(cols):
        m[:, :, c] = np.roll(m[:, :, c], -((c * step) % rows), axis=1)
    return m.reshape(len(m), -1).reshape(-1)


# 802.11a/g OFDM modes: (coded bits per symbol, coded bits per subcarrier)
IEEE80211_MODES = ((48, 1), (96, 2), (192, 4), (288, 6))


def ieee80211_permutation(ncbps: int, nbpsc: int) -> np.ndarray:
    """802.11a/g bit interleaver over one OFDM symbol (IEEE 802.11 OFDM PHY,
    two-step permutation): k -> i = (N/16)(k mod 16) + floor(k/16) spreads
    adjacent bits over non-adjacent subcarriers; i -> j = s*floor(i/s) +
    (i + N - floor(16 i / N)) mod s, s = max(nbpsc/2, 1), alternates bit
    significance. Returned in pn_interleave's convention (out[p] = in[perm[p]])."""
    k = np.arange(ncbps)
    i = (ncbps // 16) * (k % 16) + k // 16
    s = max(nbpsc // 2, 1)
    j = s * (i // s) + (i + ncbps - (16 * i) // ncbps) % s
    return np.argsort(j)


def pn_interleave(bits: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    P = len(permutation)
    n = (len(bits) // P) * P
    m = _as_stream(bits[:n]).reshape(-1, P)
    return m[:, permutation].reshape(-1)


def pn_deinterleave(bits: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    P = len(permutation)
    inv = np.argsort(permutation)
    n = (len(bits) // P) * P
    m = _as_stream(bits[:n]).reshape(-1, P)
    return m[:, inv].reshape(-1)
