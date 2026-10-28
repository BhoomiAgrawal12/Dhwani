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
alignment from them in one pass.

LDPC codes are the matrices bundled under fec/codes (index.json is the
catalogue, with sources): 802.11n n = 648 at rate 1/2 and 3/4, and CCSDS
AR4JA with k = 1024.  The "AR4JA n = 1280" of the plan is the rate-4/5
code (H 384 x 1408, the last 128 columns punctured); the rate-1/2 code
sends n = 2048 (H 1536 x 2560, the last 512 punctured), so both are
bundled.  Each becomes its sparse edge list (rows, cols), its decoder
layers, its punctured columns and its probe checks (the lightest checks
on sent bits, grouped by weight as (checks, weight) tap arrays).  They
live in LDPC_REGISTRY beside REGISTRY: the LDPC Pre-screen scores them
per codeword alignment, not probe_entries' per-phase sweep.  The
configured seeded candidates and any other matrix file are registered on
demand (ldpc_entry) and cached.  Everything bundled is read and built
here, at import: no file I/O at analysis time unless a candidate names a
matrix file.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np

from ..common.config import FECConfig
from .conv import ConvCode, conv_encode
from .ldpc import LDPCCode, read_code

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


CODES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codes")


def _ldpc_entry(code: LDPCCode, **extra) -> dict:
    s = code.structure()
    return {"name": code.name, "family": "ldpc", "n": code.n, "k": code.k,
            "n_full": code.n_full, "m": code.m, "punctured": s["punctured"],
            "rows": s["rows"], "cols": s["cols"], "checks": s["checks"],
            "check_weight": s["check_weight"], "layers": s["layers"],
            "seed": code.seed, "code": code, **extra}


def _bundled_ldpc() -> dict:
    """The catalogue's codes; "punctured" there counts the last columns of
    H that are never sent.  Without the data files (a copy installed
    without its package data) there are none, and every other code family
    still works."""
    try:
        with open(os.path.join(CODES_DIR, "index.json")) as f:
            index = json.load(f)
    except OSError as exc:
        logging.getLogger(__name__).warning("no bundled LDPC codes: %s", exc)
        return {}
    out = {}
    for item in index:
        code = read_code(os.path.join(CODES_DIR, item["file"]), item["name"],
                         item["punctured"])
        if (code.n, code.k) != (item["n"], item["k"]):
            raise ValueError(f"{item['file']}: n, k = {code.n}, {code.k}, the "
                             f"catalogue says {item['n']}, {item['k']}")
        out[item["name"]] = _ldpc_entry(code, standard=item["standard"],
                                        source=item["source"], file=item["file"])
    return out


LDPC_REGISTRY: dict = _bundled_ldpc()
_LDPC_CACHE = {}       # seeded codes and matrix files, registered on demand


def ldpc_entries() -> list:
    """The bundled LDPC codes, in catalogue order."""
    return list(LDPC_REGISTRY.values())


def ldpc_entry(cand: dict) -> dict:
    """Registry entry for an LDPC candidate, built once and cached: a
    bundled code by "name"; a matrix file by "path" (.qc or .alist, with
    an optional "name" and "punctured", the count of last columns never
    sent); otherwise the seeded construction {"n", "k", "seed"}, where a
    "name" is only a label (as before)."""
    name = cand.get("name")
    if isinstance(name, str) and name in LDPC_REGISTRY:
        return LDPC_REGISTRY[name]
    if "path" in cand:
        key = ("path", os.path.abspath(cand["path"]), name,
               int(cand.get("punctured", 0)))
        if key not in _LDPC_CACHE:
            _LDPC_CACHE[key] = _ldpc_entry(read_code(cand["path"], name, key[3]))
        return _LDPC_CACHE[key]
    if "n" not in cand or "k" not in cand:
        raise ValueError(f"LDPC candidate {cand!r}: give n and k (seeded), a "
                         f"path, or a bundled name: {', '.join(LDPC_REGISTRY)}")
    key = (int(cand["n"]), int(cand["k"]), int(cand.get("seed", 1)))
    if key not in _LDPC_CACHE:
        _LDPC_CACHE[key] = _ldpc_entry(LDPCCode(*key))
    return _LDPC_CACHE[key]
