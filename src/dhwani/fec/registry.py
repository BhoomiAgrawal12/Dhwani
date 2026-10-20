"""Code Registry (stage S9): every standard code compiled once, at import,
into the parity checks the Code Probe scores.

Convolutional codes become their lightest dual-code vectors.  Over a window
of W periods the transmitted bits are spanned by the impulse responses of
the K-1 state bits and of the window's W*L inputs (L = input steps per
period; an impulse at input q reaches v_j[t] through bit K-1-(t-q) of g_j;
punctured positions are dropped).  The GF(2) null space of that span is
every check of span <= W periods that starts on a period boundary.  W is
chosen so the null space has dimension <= MAX_NULL_DIM, and all 2^d - 1
nonzero combinations are enumerated (bit-packed, vectorised), so the
minimum weight is exact for the window, not a search heuristic.  Each
minimum-weight vector is shifted by whole periods until its first tap
falls in the first period, and the shifted copies collapse to one.  The
lightest odd-weight vector is kept on its own: an even-weight check is
blind to a whole-stream inversion, an odd one flips sign under it.  Bit
convention as in conv.py: bit K-1 of a generator taps the current input,
and the encoder emits out[t*n + j] = v_j[t].

Reed-Solomon codes become their root exponents (generator^(fcr+j)) plus
GF(2^m) tables shared per field polynomial; the probe tests every
alignment from them in one pass.  LDPC codes join in a later commit.
Everything is built in memory here: no file I/O at analysis time.
"""
from __future__ import annotations

import numpy as np

from ..common.config import FECConfig
from .conv import ConvCode, conv_encode

# 2^12 combinations per code: for every entry below, dimensions 12..20 give
# the same minimum and odd-check weights and 12..18 the same kept checks
# (measured), at 1/16 the import cost of 16
MAX_NULL_DIM = 12
MAX_CHECKS = 4      # distinct minimum-weight checks kept per code

RATE_HALF = ((3, (0o7, 0o5)), (4, (0o17, 0o15)), (5, (0o35, 0o23)),
             (6, (0o75, 0o53)), (7, (0o171, 0o133)), (8, (0o371, 0o247)),
             (9, (0o753, 0o561)))
RATE_THIRD = ((3, (0o5, 0o7, 0o7)), (4, (0o13, 0o15, 0o17)),
              (5, (0o25, 0o33, 0o37)), (6, (0o47, 0o53, 0o75)),
              (7, (0o133, 0o145, 0o175)), (8, (0o225, 0o331, 0o367)),
              (9, (0o557, 0o663, 0o711)),
              (7, (0o133, 0o171, 0o165)))           # LTE
# DVB-S / CCSDS punctures of the K=7 (171,133) mother: row j = generator j,
# column = step in the period
PUNCTURES = (("2/3", ((1, 0), (1, 1))),
             ("3/4", ((1, 0, 1), (1, 1, 0))),
             ("5/6", ((1, 0, 1, 0, 1), (1, 1, 0, 1, 0))),
             ("7/8", ((1, 0, 0, 0, 1, 0, 1), (1, 1, 1, 1, 0, 1, 0))))

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def basic_check(K: int, g1: int, g2: int) -> np.ndarray:
    """Taps of g2(D)v1 + g1(D)v2 = 0 on the interleaved stream (v1 at even
    offsets): the one check per generator pair the exhaustive sweep uses."""
    i = np.arange(K)
    v1 = 2 * i[((g2 >> i) & 1).astype(bool)]
    v2 = 2 * i[((g1 >> i) & 1).astype(bool)] + 1
    return np.sort(np.concatenate([v1, v2])).astype(np.int64)


def _transmitted(n_out: int, puncture, steps: int) -> tuple:
    """(step, output) of every transmitted bit over `steps` input steps, in
    transmission order: generator order inside a step, punctured skipped."""
    t = np.repeat(np.arange(steps), n_out)
    j = np.tile(np.arange(n_out), steps)
    if puncture is not None:
        P = np.asarray(puncture, dtype=bool)
        keep = P[j, t % P.shape[1]]
        t, j = t[keep], j[keep]
    return t, j


def _window_span(K: int, gens: tuple, puncture, W: int) -> np.ndarray:
    """Generator matrix of a W-period window: one row per input that reaches
    it (the K-1 state bits, then the window's inputs), one column per
    transmitted bit."""
    L = 1 if puncture is None else len(puncture[0])
    t, j = _transmitted(len(gens), puncture, W * L)
    q = np.arange(-(K - 1), W * L)
    d = t[None, :] - q[:, None]                 # delay from input q to bit
    g = np.asarray(gens, dtype=np.int64)[j][None, :]
    taps = (g >> np.clip(K - 1 - d, 0, K - 1)) & 1
    return np.where((d >= 0) & (d < K), taps, 0).astype(np.uint8)


def _null_space(G: np.ndarray) -> np.ndarray:
    """Basis (rows) of {h : G h^T = 0} over GF(2), by Gauss-Jordan."""
    A = G.copy()
    rows, cols = A.shape
    pivots, r = [], 0
    for c in range(cols):
        if r == rows:
            break
        nz = np.flatnonzero(A[r:, c])
        if nz.size == 0:
            continue
        p = r + int(nz[0])
        A[[r, p]] = A[[p, r]]
        hit = np.flatnonzero(A[:, c])
        A[hit[hit != r]] ^= A[r]
        pivots.append(c)
        r += 1
    free = np.setdiff1d(np.arange(cols), pivots)
    basis = np.zeros((len(free), cols), dtype=np.uint8)
    basis[np.arange(len(free)), free] = 1
    if pivots:
        basis[:, pivots] = A[:len(pivots)][:, free].T
    return basis


def _canonical(rows: np.ndarray, n_bits: int, period: int) -> list:
    """Unpack checks, shift each by whole periods so taps[0] < period,
    drop the shifted copies, order by span then taps."""
    seen = {}
    for b in np.unpackbits(rows, axis=1)[:, :n_bits]:
        taps = np.flatnonzero(b)
        taps -= (taps[0] // period) * period
        seen.setdefault(tuple(taps.tolist()), None)
    return [{"taps": np.array(tp, dtype=np.int64), "span": tp[-1] + 1,
             "weight": len(tp)} for tp in sorted(seen, key=lambda x: (x[-1], x))]


def _lightest(basis: np.ndarray, period: int) -> tuple:
    """Every nonzero combination of the basis rows (doubling, bit-packed)
    -> (distinct minimum-weight checks, lightest odd-weight check)."""
    packed = np.packbits(basis, axis=1)
    combos = np.zeros((1, packed.shape[1]), dtype=np.uint8)
    for row in packed:
        combos = np.concatenate([combos, combos ^ row])
    combos = combos[1:]
    w = _POP8[combos].sum(axis=1, dtype=np.int64)
    checks = _canonical(combos[w == w.min()], basis.shape[1], period)
    odd = (w % 2) == 1
    odd_check = None
    if odd.any():
        lightest_odd = combos[odd & (w == w[odd].min())]
        odd_check = _canonical(lightest_odd, basis.shape[1], period)[0]
    return checks[:MAX_CHECKS], odd_check


def _conv_entry(K: int, gens: tuple, puncture=None,
                rate: str | None = None) -> dict:
    n = len(gens)
    L = 1 if puncture is None else len(puncture[0])
    period = n * L if puncture is None else int(np.sum(puncture))
    growth = period - L                      # null-space dimension per period
    W = max(1, (MAX_NULL_DIM + K - 1) // growth)
    while True:
        basis = _null_space(_window_span(K, gens, puncture, W))
        if len(basis) <= MAX_NULL_DIM or W == 1:
            break
        W -= 1
    if not len(basis):
        raise ValueError(f"no dual vector within {W} periods: K={K} {gens}")
    checks, odd_check = _lightest(basis, period)
    rate = rate or f"1/{n}"
    return {"name": f"conv K{K} r{rate} ({','.join(format(g, 'o') for g in gens)})",
            "family": "convolutional", "K": K, "generators": tuple(gens),
            "rate": rate, "n_out": n,
            "puncture": None if puncture is None else tuple(map(tuple, puncture)),
            "period": period, "steps": L, "checks": checks,
            "odd_check": odd_check}


def encode(entry: dict, info_bits: np.ndarray) -> np.ndarray:
    """The entry's own transmitter: conv_encode (no tail) then puncturing."""
    coded = conv_encode(info_bits, ConvCode(entry["K"], entry["generators"]),
                        terminate=False)
    if entry["puncture"] is None:
        return coded
    v = coded.reshape(-1, entry["n_out"])
    P = np.asarray(entry["puncture"], dtype=bool)
    return v[P[:, np.arange(len(v)) % P.shape[1]].T]


_GF = {}
_RS = {}


def _gf_tables(prim: int) -> tuple:
    """(m, exp, log) of GF(2^m) for one field polynomial, built once."""
    if prim not in _GF:
        m = prim.bit_length() - 1
        size = 1 << m
        exp = np.zeros(2 * size, dtype=np.int64)
        log = np.zeros(size, dtype=np.int64)
        x = 1
        for i in range(size - 1):
            exp[i] = x
            log[x] = i
            x <<= 1
            if x & size:
                x ^= prim
        exp[size - 1:2 * size - 2] = exp[:size - 1]
        _GF[prim] = (m, exp, log)
    return _GF[prim]


def rs_entry(cand: dict) -> dict:
    """Registry entry for any RS candidate dict (config or bundled), cached
    by (n, k, prim, fcr, generator)."""
    key = (int(cand["n"]), int(cand["k"]), int(cand.get("prim", 0x11D)),
           int(cand.get("fcr", 1)), int(cand.get("generator", 2)))
    if key not in _RS:
        n, k, prim, fcr, gen = key
        m, exp, log = _gf_tables(prim)
        roots = (int(log[gen]) * (fcr + np.arange(n - k))) % ((1 << m) - 1)
        _RS[key] = {"name": f"RS({n},{k}) fcr {fcr} gen {gen} prim {prim:#x}",
                    "family": "reed_solomon", "n": n, "k": k, "prim": prim,
                    "fcr": fcr, "generator": gen, "m": m,
                    "roots": roots.astype(np.int64), "exp": exp, "log": log}
    return _RS[key]


def _build() -> dict:
    reg = {}

    def add(e):
        reg[e["name"]] = e
    for K, g in RATE_HALF:
        for gens in (g, g[::-1]):
            add(_conv_entry(K, gens))
    for K, g in RATE_THIRD:
        for gens in (g, g[::-1]):
            add(_conv_entry(K, gens))
    # the swapped order transmits the second generator first (802.11a)
    mother = (0o171, 0o133)
    for rate, rows in PUNCTURES:
        add(_conv_entry(7, mother, rows, rate))
        add(_conv_entry(7, mother[::-1], rows[::-1], rate))
    # the config's candidates include DVB's RS(204,188)
    for cand in FECConfig().rs_candidates:
        add(rs_entry(cand))
    return reg


REGISTRY: dict = _build()


def conv_entries() -> list:
    return [e for e in REGISTRY.values() if e["family"] == "convolutional"]


def rs_entries() -> list:
    return [e for e in REGISTRY.values() if e["family"] == "reed_solomon"]


def ldpc_entries() -> list:
    """Bundled LDPC matrices are registered in a later commit."""
    return []
