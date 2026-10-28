"""LDPC codes (stage S9): matrices, Pre-screen and Layered Decoder.

Blind reconstruction of an arbitrary sparse parity-check matrix from a
noisy stream is research-grade, so LDPC codes are identified from a set of
known matrices, and the result says so.

Matrices.  Two plain formats: .alist (MacKay: n m, the largest column and
row degrees, every degree, then the column lists and the row lists,
1-based, zero padding allowed) and .qc (a base matrix of shifts with a
lifting size Z: -1 is an empty Z x Z block, s >= 0 the identity shifted
right by s, so row r of the block has its one in column (r + s) mod Z).
The Code Registry reads the bundled ones (fec/codes/index.json, with
sources) at import; any other matrix loads by path (read_code, or a
{"path": ...} candidate).  The seeded construction H = [P | I_m] stays for
the configured candidates and the synthetic factory.

Punctured codes.  CCSDS AR4JA sends n = 2048 of its 2560 columns at rate
1/2 (1280 of 1408 at rate 4/5): the decoder runs on the full H with the
punctured bits at LLR 0.  The probe needs checks on sent bits only, and
every AR4JA row touches a punctured bit, but each punctured column has a
row touching no other punctured bit: adding that row once per punctured
bit of a row cancels them all (weight 3 + 2 x 5 - 2 = 11 at rate 1/2).
The same elimination, pivoting on the rows with the fewest punctured bits,
gives a basis of the sent code's checks for any punctured H.

Pre-screen.  Before any decode, every registered code is scored at every
alignment 0..n-1 over a few codewords (about SCAN_VALUES check values per
alignment) by the Code Probe: a check's value
is prod tanh(L/2) over its taps (+-1 without LLRs), divided by a^w so
hard and soft input share one scale.  Window start q = j*n + a is
codeword j at alignment a, so each of a code's lightest checks is read
once over q < J*n, one contiguous slice per tap, and the alignments come
out as the minor axis.  Both polarities are tested: a wholly inverted
codeword (a 180-degree phase ambiguity) fails exactly the odd-weight
checks, so the inverted hypothesis only re-signs the sums of odd-weight
checks.  The Early Verdict (an SPRT planned at a mean of (1 - 2p)^w per
check, p = EXPECTED_BER or, where lower, the BER the code's rate can
still correct) decides each (code, polarity, alignment), and may start
at any of the first LEAD_CODEWORDS + 1 codewords (a capture that opens
with a preamble has junk there); only the best "code" that also stands
DECOY_Z spreads above the other hypotheses of its check weight on the
same stream (structure in uncoded data biases wrong checks too, and the
decoys measure by how much) and keeps its verdict over more codewords
goes on to decoding.

Alignment of a quasi-cyclic code.  Every bundled matrix is built of
circulants (Z = 27 for 802.11n; 128 and 32 for AR4JA), and a window d bits
off a codeword is a codeword with each circulant rotated by d, except at
the circulant boundaries: one bit off, 97% of AR4JA r1/2's weight-11
checks still hold (92% of 802.11n r1/2's), so the scan's best alignment
is often a neighbour of the true one, and the decoder converges there
too, onto the rotated codewords with the data slid.  The confirmation
therefore compares the alignments SETTLE_BITS either side over more
codewords and every probe check; their z differ only through the checks
that cross a boundary.  For a noisy stream settle_offset then decodes
the neighbours: on a codeword both decode, the rotated one also flips
half of the boundary bits (8 of AR4JA r1/2's 16, 12 of 802.11n's 24).

Layered Decoder.  Layered offset min-sum: the rows are split into layers
that share no column (the block rows of a QC matrix, a greedy colouring
otherwise); per layer and row the old check message is subtracted from
the posterior LLRs, the two smallest magnitudes and the sign product of
the rest give the new message (magnitude scale * max(min - offset, 0):
offset first, then scale), and it is added back at once, so later layers
of the same iteration already use it.  Every codeword of the batch is
checked after each iteration and frozen once its syndrome is zero.
"""
from __future__ import annotations

import functools
import os

import numpy as np
from scipy import sparse

from .probe import (ALPHA, BETA, BLOCK, DECOY_Z, EXPECTED_BER, LLR_CLIP,
                    _VERDICT, _agree, _bounds, decoy_null, planned_m1,
                    soft_values, stream_stats)

PRESCREEN_CODEWORDS = 8    # codewords per alignment in the scan, at least (the old quick sweep)
CONFIRM_CODEWORDS = 64     # the winner keeps its verdict over this many (the old confirm)
# probe checks per code: the lightest weight classes, topped up to at least
# MIN_PROBE_CHECKS (802.11n rate 3/4 has only 54 of weight 14), at most
# MAX_PROBE_CHECKS
MIN_PROBE_CHECKS = 128
MAX_PROBE_CHECKS = 512
# check values per alignment the scan reads (its cost grows with them): a
# code with few probe checks reads more codewords to get there, and the
# lightest weight class is always read whole.  802.11n rate 1/2 at Eb/N0
# 2.5 dB (FER < 1e-3) reads z 13.8 soft, 10.7 hard over 8 codewords of its
# 216 lightest checks, against the decoy gate of 7; AR4JA's heavier checks
# carry less per value (r1/2: 512 of weight 11; r4/5: 128 of weight 33 and
# 35), and without all of them it went undetected 1 dB above the Eb/N0
# where it decodes
SCAN_VALUES = 2048
# the Early Verdict may start at any of the first LEAD_CODEWORDS + 1
# codewords: a capture that opens with a preamble or inside a codeword has
# junk there, and the old sweep's 5-of-8 clean-codeword gate tolerated 3
LEAD_CODEWORDS = 3
# alignments compared either side of the scan's best, and moves while the
# best sits at the edge (see the module docstring: near its threshold the
# scan lands up to 3 bits off on AR4JA r1/2)
SETTLE_BITS = 4
SETTLE_MOVES = 4
# codewords settle_offset decodes per alignment: one codeword both
# decode already separates them by ~6 flips (sd ~2)
SETTLE_BLOCKS = 16
DECODE_BATCH = 256         # codewords per decoder call (bounds the message memory)
# BER range for turning hard bits into +-L0: finite on clean blocks,
# still informative on bad ones
HARD_BER = (1e-3, 0.2)


# ------------------------------------------------------------- readers ---
def read_alist(path: str) -> np.ndarray:
    """(m, n) uint8 parity-check matrix from a MacKay .alist file.  The
    column lists define H; the degree lines and the row lists must agree
    with them.  Zero-padded lists (every list as long as the largest
    degree) are read too."""
    with open(path) as f:
        tok = np.array(f.read().split(), dtype=np.int64)
    n, m = int(tok[0]), int(tok[1])
    col_deg, row_deg = tok[4:4 + n], tok[4 + n:4 + n + m]
    lists = tok[4 + n + m:]
    dc, dr = int(tok[2]), int(tok[3])
    if len(lists) == col_deg.sum() + row_deg.sum():
        split = int(col_deg.sum())
        col_cols = np.repeat(np.arange(n), col_deg)
        col_rows = lists[:split] - 1
        row_rows = np.repeat(np.arange(m), row_deg)
        row_cols = lists[split:] - 1
    elif len(lists) == n * dc + m * dr:
        C = lists[:n * dc].reshape(n, dc)
        R = lists[n * dc:].reshape(m, dr)
        col_cols, j = np.nonzero(C > 0)
        col_rows = C[col_cols, j] - 1
        row_rows, j = np.nonzero(R > 0)
        row_cols = R[row_rows, j] - 1
    else:
        raise ValueError(f"{path}: list lengths match neither the degrees "
                         "nor zero padding")
    by_cols, by_rows = col_rows * n + col_cols, row_rows * n + row_cols
    if not (np.array_equal(np.unique(by_cols), np.sort(by_rows))
            and len(by_cols) == len(by_rows)
            and np.array_equal(np.bincount(col_cols, minlength=n), col_deg)
            and np.array_equal(np.bincount(row_rows, minlength=m), row_deg)):
        raise ValueError(f"{path}: column lists, row lists and degrees disagree")
    H = np.zeros((m, n), dtype=np.uint8)
    H[col_rows, col_cols] = 1
    return H


def write_alist(path: str, H: np.ndarray) -> None:
    """Write H as a MacKay .alist file (1-based lists, no padding)."""
    H = np.asarray(H) != 0
    m, n = H.shape
    cols = [np.flatnonzero(c) + 1 for c in H.T]
    rows = [np.flatnonzero(r) + 1 for r in H]
    out = [f"{n} {m}", f"{max(map(len, cols), default=0)} "
                       f"{max(map(len, rows), default=0)}",
           " ".join(str(len(c)) for c in cols),
           " ".join(str(len(r)) for r in rows)]
    out += [" ".join(map(str, c)) for c in cols]
    out += [" ".join(map(str, r)) for r in rows]
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(out) + "\n")


def qc_base(path: str) -> tuple:
    """(base matrix, Z) of a .qc file: '#' comment lines, then 'rows cols
    Z', then the base matrix of shifts."""
    with open(path) as f:
        lines = [ln.split() for ln in f
                 if ln.strip() and not ln.lstrip().startswith("#")]
    r, c, Z = (int(v) for v in lines[0])
    base = np.array(lines[1:1 + r], dtype=np.int64)
    if base.shape != (r, c) or base.max() >= Z or base.min() < -1:
        raise ValueError(f"{path}: base matrix does not match 'rows cols Z'")
    return base, Z


def expand_qc(base: np.ndarray, Z: int) -> np.ndarray:
    """Expand a base matrix of shifts: block (i, j) with shift s >= 0 puts
    row i*Z + r's one at column j*Z + (r + s) mod Z; -1 is an empty block."""
    base = np.asarray(base, dtype=np.int64)
    H = np.zeros((base.shape[0] * Z, base.shape[1] * Z), dtype=np.uint8)
    i, j = np.nonzero(base >= 0)
    r = np.arange(Z)
    H[(i[:, None] * Z + r).ravel(),
      (j[:, None] * Z + (r + base[i, j][:, None]) % Z).ravel()] = 1
    return H


def read_qc(path: str) -> np.ndarray:
    """(rows*Z, cols*Z) uint8 parity-check matrix of a .qc file."""
    return expand_qc(*qc_base(path))


def read_code(path: str, name: str = None, punctured: int = 0) -> "LDPCCode":
    """The code of a .qc or .alist file (by its extension); `punctured`
    counts the last columns of H that are never sent (as index.json
    does).  A .qc file keeps its lifting size, so the decoder's layers are
    its block rows."""
    lift = None
    if path.lower().endswith(".qc"):
        base, lift = qc_base(path)
        H = expand_qc(base, lift)
    else:
        H = read_alist(path)
    n_full = H.shape[1]
    code = LDPCCode.from_parity_check(
        H, name or os.path.basename(path),
        punctured=np.arange(n_full - int(punctured), n_full))
    code.lift = lift
    return code


# ---------------------------------------------------------- the code ---
class LDPCCode:
    """Binary LDPC code.  Seeded: H = [P | I_m], G = [I_k | P^T] with a
    sparse deterministic P, so the factory and the identifier share the
    matrix.  from_parity_check: any H, optionally with punctured columns
    (decoded at LLR 0, never sent).  n is the sent length, n_full the
    columns of H, k = n_full - m (as before; the true dimension when H has
    full row rank, as every bundled H does).  decode returns the first k
    bits of the decoded word, as it always has (a punctured code: the
    encoder's information bits), so the encoder puts the information
    there and needs the last m columns of H independent."""

    lift = None    # QC lifting size Z when H was expanded from a base matrix

    def __init__(self, n: int, k: int, seed: int = 1, column_weight: int = 3):
        if not 0 < k < n:
            raise ValueError("need 0 < k < n")
        self.n, self.k, self.seed = n, k, seed
        self.m = n - k
        rng = np.random.default_rng(seed)
        P = np.zeros((self.m, k), dtype=np.uint8)
        for col in range(k):
            rows = rng.choice(self.m, size=min(column_weight, self.m),
                              replace=False)
            P[rows, col] = 1
        # every check row must involve at least one information bit
        for row in range(self.m):
            if not P[row].any():
                P[row, rng.integers(0, k)] = 1
        self.P = P
        self.H = np.concatenate([P, np.eye(self.m, dtype=np.uint8)], axis=1)
        self.name = f"LDPC({n},{k}) seed {seed}"
        self._setup(np.zeros(0, dtype=np.int64))

    @classmethod
    def from_parity_check(cls, H: np.ndarray, name: str = "custom",
                          punctured=None):
        """Code of any (m, n_full) parity-check matrix; `punctured` columns
        are never sent (the decoder starts them at LLR 0).  `seed` holds
        the name, as before."""
        obj = cls.__new__(cls)
        H = np.asarray(H, dtype=np.uint8)
        obj.H = H
        obj.m, n_full = H.shape
        punct = np.unique(np.asarray([] if punctured is None else punctured,
                                     dtype=np.int64))
        if punct.size and (punct[0] < 0 or punct[-1] >= n_full):
            raise ValueError("punctured column out of range")
        obj.n = n_full - len(punct)
        obj.k = n_full - obj.m
        obj.P = None
        obj.seed = name
        obj.name = name
        obj._setup(punct)
        return obj

    def _setup(self, punctured: np.ndarray):
        self.n_full = self.H.shape[1]
        self.punctured = punctured
        self.transmitted = np.setdiff1d(np.arange(self.n_full), punctured)
        self._struct_cache = None
        self._enc_cache = None

    # -- structure, built once on first use (the registry builds it at import)
    def structure(self) -> dict:
        """Edges (rows, cols), decoder layers, the checks on sent bits and
        their lightest classes (probe checks), sparse syndrome matrices."""
        if self._struct_cache is None:
            self._struct_cache = _structure(self)
        return self._struct_cache

    def _encoder(self) -> tuple:
        if self._enc_cache is None:
            info, parity, G = _systematic(self.H)
            if not len(self.punctured) and not np.array_equal(info, np.arange(self.k)):
                raise ValueError("the last m columns of H are singular, so the "
                                 "first k bits (what decode returns) are no "
                                 "information set; permute the columns of H")
            self._enc_cache = (info, parity, G)
        return self._enc_cache

    def info_columns(self) -> np.ndarray:
        """Columns of the decoded word that decode() returns: the first k,
        as before, unless the code is punctured (then the encoder's
        information set, 0..k-1 for AR4JA; the first k again when H has
        redundant rows and so no encoder)."""
        if self.P is not None or not len(self.punctured):
            return np.arange(self.k)
        try:
            return self._encoder()[0]
        except ValueError:
            return np.arange(self.k)

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """Encode blocks of k bits (sent bits only); pads the tail block
        with zeros."""
        bits = np.asarray(info_bits, dtype=np.uint8)
        pad = (-len(bits)) % self.k
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        blocks = bits.reshape(-1, self.k)
        if self.P is not None:
            parity = (blocks @ self.P.T) % 2
            return np.concatenate([blocks, parity], axis=1).reshape(-1).astype(np.uint8)
        return self.encode_full(blocks)[:, self.transmitted].reshape(-1)

    def encode_full(self, blocks: np.ndarray) -> np.ndarray:
        """(B, n_full) full codewords (punctured bits included) of (B, k)
        information blocks: systematic, parity = G @ info mod 2."""
        blocks = np.asarray(blocks, dtype=np.uint8).reshape(-1, self.k)
        info, parity, G = self._encoder()
        full = np.zeros((len(blocks), self.n_full), dtype=np.uint8)
        full[:, info] = blocks
        # 0/1 sums below 2^24 are exact in float32 (BLAS instead of int loops)
        full[:, parity] = (blocks.astype(np.float32) @ G.T) % 2
        return full

    def syndromes(self, codewords: np.ndarray) -> np.ndarray:
        """(n_cw, rows) syndrome matrix for stacked codewords: full words
        (n_full bits) against H, the sent words of a punctured code (n
        bits) against its checks on sent bits."""
        cw = np.asarray(codewords, dtype=np.uint8)
        s = self.structure()
        M = s["H_csr"] if cw.shape[-1] == self.n_full else s["T_csr"]
        if cw.shape[-1] != M.shape[1]:
            raise ValueError(f"codewords of {cw.shape[-1]} bits; the code has "
                             f"{self.n} (sent) / {self.n_full} (full)")
        # any stack of codewords, as the dense product took
        flat = cw.reshape(-1, cw.shape[-1])
        syn = (M @ flat.T.astype(np.float32)).T % 2
        return syn.astype(np.uint8).reshape(cw.shape[:-1] + (M.shape[0],))

    def syndrome_zero_rate(self, bits: np.ndarray, offset: int = 0,
                           max_codewords: int = 64) -> float:
        b = np.asarray(bits[offset:], dtype=np.uint8)
        n_cw = min(len(b) // self.n, max_codewords)
        if n_cw < 2:
            return 0.0
        cw = b[: n_cw * self.n].reshape(n_cw, self.n)
        return float((self.syndromes(cw) == 0).all(axis=1).mean())

    def channel_llrs(self, blocks: np.ndarray, llrs: np.ndarray = None) -> np.ndarray:
        """Decoder input for (B, n) received blocks (or (B, n_full)): the
        LLR magnitudes with the signs of the bits (a misaligned LLR source
        costs reliability, never a decision), or +-L0 for hard bits - and
        for LLRs that are all zero, which say nothing - with L0 from the
        BER the blocks' probe checks read."""
        blocks = np.atleast_2d(np.asarray(blocks, dtype=np.uint8))
        if blocks.shape[1] not in (self.n, self.n_full):
            raise ValueError(f"blocks of {blocks.shape[1]} bits; the code has "
                             f"{self.n} (sent) / {self.n_full} (full)")
        sign = 1.0 - 2.0 * blocks.astype(np.float32)
        if llrs is not None:
            mag = np.abs(np.nan_to_num(np.asarray(llrs, dtype=np.float32), nan=0.0,
                                       posinf=LLR_CLIP, neginf=-LLR_CLIP))
            if mag.size != blocks.size:
                raise ValueError(f"{mag.size} LLRs for {blocks.size} bits")
            if mag.any():
                return sign * np.minimum(mag.reshape(blocks.shape), LLR_CLIP)
        if not len(blocks):
            return sign
        # the probe checks index the sent bits (full words: drop the punctured)
        sent = blocks if blocks.shape[1] == self.n else blocks[:, self.transmitted]
        values, weights = [], []
        for taps in self.structure()["checks"]:
            values.append(1.0 - 2.0 * (sent[:, taps].sum(axis=2) & 1))
            weights.append(taps.shape[1])
        p = _ber_from_mean(float(np.mean(np.concatenate([v.ravel() for v in values]))),
                           weights, [v.size for v in values])
        p = float(np.clip(p, *HARD_BER))
        return sign * np.float32(np.log((1.0 - p) / p))

    def decode(self, codeword: np.ndarray, max_iters: int = 30,
               llrs: np.ndarray = None) -> tuple:
        """Layered offset min-sum decode of one sent block (hard bits, or
        their LLRs).  Returns (info_bits, converged)."""
        c = np.asarray(codeword, dtype=np.uint8)
        L = self.channel_llrs(c[None, :], None if llrs is None
                              else np.asarray(llrs)[None, :])
        hard, ok = layered_decode(L, self, max_iter=max_iters)
        return hard[0, self.info_columns()], bool(ok[0])

    def decode_blocks(self, bits: np.ndarray, offset: int = 0,
                      max_blocks: int = 4096, llrs: np.ndarray = None) -> tuple:
        """Layered decode of consecutive sent blocks from `offset`: the
        received blocks (B, n), the decoded words (B, n_full) and which
        converged (B,).  `llrs` (aligned with `bits`, > 0 => bit 0) make
        it a soft decode."""
        b = np.asarray(bits[offset:], dtype=np.uint8)
        n_blocks = max(0, min(len(b) // self.n, max_blocks))
        blocks = b[: n_blocks * self.n].reshape(n_blocks, self.n)
        if not n_blocks:
            return blocks, np.zeros((0, self.n_full), np.uint8), np.zeros(0, bool)
        soft = None
        if llrs is not None and len(llrs) == len(bits):
            soft = np.asarray(llrs)[offset:offset + n_blocks * self.n]
        L = self.channel_llrs(blocks, soft)
        words, ok = [], []
        for s in range(0, n_blocks, DECODE_BATCH):
            w, c = layered_decode(L[s:s + DECODE_BATCH], self)
            words.append(w)
            ok.append(c)
        return blocks, np.concatenate(words), np.concatenate(ok)

    def decode_stream(self, bits: np.ndarray, offset: int = 0,
                      max_blocks: int = 4096, llrs: np.ndarray = None) -> tuple:
        """Decode consecutive blocks; returns (info_bits, converged_rate).
        `llrs` (aligned with `bits`, > 0 => bit 0) make it a soft decode."""
        _, words, ok = self.decode_blocks(bits, offset, max_blocks, llrs)
        if not len(ok):
            return np.zeros(0, dtype=np.uint8), 0.0
        return words[:, self.info_columns()].reshape(-1), float(ok.mean())


_BISECT = 50                 # bisection steps: 0.5 / 2^50, below float precision


def _ber_from_mean(mean: float, weights, counts=None) -> float:
    """BER p with mean_i (1 - 2p)^w_i = mean (a hard check mean over checks
    of the given weights, `counts` checks of each when given); 0.5 when the
    mean says nothing."""
    if not mean > 0:
        return 0.5
    if mean >= 1:
        return 0.0
    w = np.asarray(weights, dtype=np.float64)
    cnt = np.ones(len(w)) if counts is None else np.asarray(counts, dtype=np.float64)
    lo, hi = 0.0, 0.5
    for _ in range(_BISECT):         # monotone in p
        p = (lo + hi) / 2
        if np.dot(cnt, (1 - 2 * p) ** w) / cnt.sum() > mean:
            lo = p
        else:
            hi = p
    return (lo + hi) / 2


# ------------------------------------------------------- structure ---
def _systematic(H: np.ndarray) -> tuple:
    """(info columns, parity columns, G float32) with codeword[parity] =
    G @ codeword[info] mod 2.  Gauss-Jordan over GF(2) on bit-packed rows
    visiting the last m columns first, so the information bits come first
    whenever that parity part is invertible; otherwise the pivot columns
    found become the parity positions."""
    m, n = H.shape
    order = np.concatenate([np.arange(n - m, n), np.arange(n - m)])
    A = np.packbits(H[:, order], axis=1, bitorder="little")
    pad = (-A.shape[1]) % 8
    # 64 columns per little-endian word; the column fancy index above leaves
    # a Fortran layout, which a wider view cannot take
    A = np.ascontiguousarray(np.concatenate([A, np.zeros((m, pad), np.uint8)],
                                            axis=1)).view("<u8")
    pivots, r = [], 0
    for j in range(n):
        if r == m:
            break
        word, bit = j >> 6, np.uint64(1) << np.uint64(j & 63)
        col = (A[:, word] & bit) != 0
        cand = np.flatnonzero(col[r:])
        if not cand.size:
            continue
        p = r + int(cand[0])
        if p != r:
            A[[r, p]] = A[[p, r]]
            col[[r, p]] = col[[p, r]]
        col[r] = False
        A[col] ^= A[r]
        pivots.append(j)
        r += 1
    if r < m:
        raise ValueError(f"H has redundant rows (rank {r} < {m}); drop them first")
    full = np.unpackbits(A.view(np.uint8), axis=1, bitorder="little")[:, :n]
    free = np.setdiff1d(np.arange(n), pivots)
    free = free[np.argsort(order[free])]            # info bits in column order
    return (order[free], order[np.array(pivots)],
            full[:, free].astype(np.float32))


def _csr(rows: np.ndarray, cols: np.ndarray, shape: tuple):
    return sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                             shape=shape)


def _sent_checks(H: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                 punctured: np.ndarray, transmitted: np.ndarray) -> tuple:
    """Edges (rows, positions among the sent bits) and count of checks
    spanning the code on sent bits only: each punctured column is
    eliminated with a pivot row, which is then dropped.  When every
    punctured column has a row touching no other punctured bit (AR4JA),
    the lightest such row is the pivot and every other row gains it once
    per punctured bit: one sparse product.  Otherwise Gaussian elimination
    pivoting on the row with the fewest punctured bits."""
    m, n_full = H.shape
    if not len(punctured):
        return rows, cols, m
    is_p = np.zeros(n_full, dtype=bool)
    is_p[punctured] = True
    pos = np.full(n_full, -1)
    pos[transmitted] = np.arange(len(transmitted))
    pe = is_p[cols]
    pcount = np.bincount(rows[pe], minlength=m)
    weight = np.bincount(rows, minlength=m)
    single = pe & (pcount[rows] == 1)
    # per punctured column the lightest row (then the first) touching no
    # other punctured bit
    best = np.full(n_full, np.iinfo(np.int64).max)
    np.minimum.at(best, cols[single], weight[rows[single]] * m + rows[single])
    if (best[punctured] < np.iinfo(np.int64).max).all():
        owner = np.full(n_full, -1)
        owner[punctured] = best[punctured] % m
        others = np.setdiff1d(np.arange(m), owner[punctured])
        idx = np.full(m, -1)
        idx[others] = np.arange(len(others))
        e = pe & (idx[rows] >= 0)
        X = _csr(np.concatenate([np.arange(len(others)), idx[rows[e]]]),
                 np.concatenate([others, owner[cols[e]]]), (len(others), m))
        T = (X @ _csr(rows, cols, H.shape)).tocoo()
        odd = (np.rint(T.data).astype(np.int64) & 1).astype(bool)
        t_rows, t_cols = T.row[odd], T.col[odd]
        if is_p[t_cols].any():
            raise RuntimeError("a punctured column survived the elimination")
    else:
        A = H.copy()
        pcount, weight = pcount.astype(np.int64), weight.astype(np.int64)
        used = np.zeros(m, dtype=bool)
        for c in punctured:
            hit = np.flatnonzero(A[:, c])
            cand = hit[~used[hit]]
            if not cand.size:
                continue
            p = cand[np.lexsort((weight[cand], pcount[cand]))[0]]
            others = hit[hit != p]
            A[others] ^= A[p]
            pcount[others] = A[others][:, punctured].sum(axis=1)
            weight[others] = A[others].sum(axis=1)
            used[p] = True
        t_rows, t_cols = np.nonzero(A[~used])
    # sent positions, rows renumbered without the empty ones, row-major
    order = np.lexsort((t_cols, t_rows))
    t_rows, t_cols = t_rows[order], pos[t_cols[order]]
    kept, t_rows = np.unique(t_rows, return_inverse=True)
    return t_rows, t_cols, len(kept)


def _layers(H: np.ndarray, rows: np.ndarray, cols: np.ndarray,
            lift) -> list:
    """Conflict-free row groups, each split by row degree into dense
    (rows, degree) column-index arrays: the block rows of a QC matrix (Z
    rows of shifted identities share no column), else a greedy colouring
    in row order."""
    m = H.shape[0]
    deg = np.bincount(rows, minlength=m)
    start = np.concatenate([[0], np.cumsum(deg)])
    groups = None
    if lift and m % lift == 0:
        groups = [np.arange(b, b + lift) for b in range(0, m, lift)]
        if any(H[g].sum(axis=0).max() > 1 for g in groups):
            groups = None
    if groups is None:
        busy, colour = [], np.zeros(m, dtype=np.int64)
        for r in range(m):
            c = cols[start[r]:start[r + 1]].tolist()
            k = next((k for k, b in enumerate(busy) if b.isdisjoint(c)), len(busy))
            if k == len(busy):
                busy.append(set())
            busy[k].update(c)
            colour[r] = k
        groups = [np.flatnonzero(colour == k) for k in range(len(busy))]
    layers = []
    for g in groups:
        for d in np.unique(deg[g]):
            if d == 0:
                continue
            r = g[deg[g] == d]
            layers.append({"rows": r,
                           "cols": cols[start[r][:, None] + np.arange(d)]})
    return layers


def _structure(code: LDPCCode) -> dict:
    H = code.H
    rows, cols = np.nonzero(H)                   # row-major: rows sorted
    t_rows, t_cols, n_t = _sent_checks(H, rows, cols, code.punctured,
                                       code.transmitted)
    w = np.bincount(t_rows, minlength=n_t)
    start = np.concatenate([[0], np.cumsum(w)])
    # probe checks: whole weight classes, lightest first, until there are
    # MIN_PROBE_CHECKS; at most MAX_PROBE_CHECKS
    checks, total = [], 0
    for wc in np.unique(w[w > 0]):
        if total >= MIN_PROBE_CHECKS:
            break
        sel = np.flatnonzero(w == wc)[:MAX_PROBE_CHECKS - total]
        checks.append(t_cols[start[sel][:, None] + np.arange(wc)])
        total += len(sel)
    if not checks:
        raise ValueError(f"{code.name}: no parity check on the sent bits")
    return {"rows": rows, "cols": cols, "punctured": code.punctured,
            "layers": _layers(H, rows, cols, code.lift),
            "checks": checks, "check_weight": int(checks[0].shape[1]),
            "H_csr": _csr(rows, cols, H.shape),
            "T_csr": _csr(t_rows, t_cols, (n_t, code.n))}


# ------------------------------------------------- layered decoder ---
def layered_decode(llrs: np.ndarray, code: LDPCCode, max_iter: int = 30,
                   scale: float = 0.8, offset: float = 0.15) -> tuple:
    """Layered offset min-sum over a batch of codewords.

    llrs: (B, n) channel LLRs of the sent bits (> 0 => bit 0; punctured
    bits enter at 0) or (B, n_full).  Per layer and row: T = Q - R_old,
    new R = (sign product of the other edges) * scale * max(min of the
    other edges' |T| - offset, 0), Q = T + R_new.  A codeword stops once
    its hard decisions satisfy every row (checked before the first and
    after every iteration), after at most max_iter iterations.
    Returns (hard bits (B, n_full) uint8, converged (B,) bool)."""
    L = np.atleast_2d(np.asarray(llrs, dtype=np.float32))
    B = len(L)
    s = code.structure()
    if L.shape[1] == code.n_full:
        Q = L.copy()
    elif L.shape[1] == code.n:
        Q = np.zeros((B, code.n_full), dtype=np.float32)
        Q[:, code.transmitted] = L
    else:
        raise ValueError(f"LLRs of {L.shape[1]} bits for a code of "
                         f"{code.n} (sent) / {code.n_full} (full)")
    Q = np.clip(np.nan_to_num(Q, nan=0.0), -LLR_CLIP, LLR_CLIP)
    layers = s["layers"]
    R = [np.zeros((B,) + lay["cols"].shape, dtype=np.float32) for lay in layers]
    hard = np.zeros((B, code.n_full), dtype=np.uint8)
    ok = np.zeros(B, dtype=bool)
    live = np.arange(B)
    for it in range(max_iter + 1):
        dec = (Q < 0).astype(np.uint8)
        good = ~((s["H_csr"] @ dec.T.astype(np.float32)) % 2).any(axis=0)
        if good.any():
            hard[live[good]] = dec[good]
            ok[live[good]] = True
            keep = ~good
            Q, live = Q[keep], live[keep]
            R = [r[keep] for r in R]
        if not len(live) or it == max_iter:
            break
        for lay, Rl in zip(layers, R):
            cols = lay["cols"]
            T = Q[:, cols]
            T -= Rl
            if cols.shape[1] == 1:       # a one-bit check pins its bit to 0
                Rl[...] = LLR_CLIP
            else:
                mag = np.abs(T)
                m1 = mag.min(axis=2, keepdims=True)
                is_min = mag == m1
                # second smallest: the minimum again if it is tied
                m2 = np.where(is_min, np.inf, mag).min(axis=2, keepdims=True)
                m2 = np.where(is_min.sum(axis=2, keepdims=True) > 1, m1, m2)
                ext = np.where(is_min, m2, m1)
                ext -= offset
                np.maximum(ext, 0.0, out=ext)
                ext *= scale
                neg = T < 0
                # sign of the other edges' product: the row's parity of
                # negatives, flipped by the edge's own sign
                odd = (neg.sum(axis=2, keepdims=True) & 1).astype(bool)
                np.negative(ext, out=ext, where=odd ^ neg)
                Rl[...] = ext
            Q[:, cols] = T + Rl
    if len(live):
        hard[live] = Q < 0
    return hard, ok


# ------------------------------------------------------- Pre-screen ---
def _sprt(inc: np.ndarray, used: np.ndarray, alpha: float, beta: float,
          per_cw: int = 1, starts: int = 1) -> tuple:
    """Gaussian SPRT over blocks: inc (blocks, H) LLR increments in
    codeword-major order, per_cw blocks per codeword; used (blocks, H)
    values per block.  The test may start at any of the first `starts`
    codewords: the first start that accepts "code" stands, otherwise the
    test from codeword 0.  Returns per column (state 1 code / -1 no code /
    0 open, LLR at the first crossing or the end, values used, the
    codeword the test started at)."""
    upper, lower = _bounds(alpha, beta)
    H = inc.shape[1]
    out = [np.zeros(H, np.int8), np.zeros(H), np.zeros(H, np.int64),
           np.zeros(H, np.int64)]
    if not len(inc):
        return tuple(out)
    path, n = np.cumsum(inc, axis=0), np.cumsum(used, axis=0)
    # a later start can only accept where its path ever reaches the upper
    # bound: the suffix maximum finds those columns without walking the rest
    top = np.maximum.accumulate(path[::-1], axis=0)[::-1] if starts > 1 else None
    for j0 in range(min(starts, -(-len(inc) // per_cw))):
        r0 = j0 * per_cw
        if not r0:
            cols, p, u = np.arange(H), path, n
        else:
            cols = np.flatnonzero((out[0] != 1) & (top[r0] - path[r0 - 1] >= upper))
            if not len(cols):
                continue
            p = path[r0:, cols] - path[r0 - 1, cols]
            u = n[r0:, cols] - n[r0 - 1, cols]
        up, lo = p >= upper, p <= lower
        hit = up | lo
        crossed = hit.any(axis=0)
        first = np.where(crossed, hit.argmax(axis=0), len(p) - 1)
        k = np.arange(len(cols))
        state = np.where(crossed, np.where(up[first, k], 1, -1), 0)
        keep = state == 1 if r0 else np.ones(len(cols), bool)
        c = cols[keep]
        out[0][c] = state[keep]
        out[1][c] = p[first, k][keep]
        out[2][c] = np.rint(u[first, k][keep])
        out[3][c] = j0
    return tuple(out)


class _Probe:
    """Per-stream state shared by every code: soft values (float32), bits,
    idle runs, the normalisers a = mean|t| and b = mean t^2."""

    def __init__(self, bits: np.ndarray, llrs):
        self.st = stream_stats(soft_values(bits, llrs), bits)
        if not self.st.a > 0:        # LLRs all zero or NaN: they say nothing
            self.st = stream_stats(soft_values(bits), bits)
        self.bits = self.st.bits
        self.odd_bits = self.bits != 0
        self.t = self.st.t.astype(np.float32)
        self.N = self.st.N
        self.idle = bool(self.st.idle.any())

    def norm(self, w: np.ndarray) -> tuple:
        """Per check of weight w: (scale a^-w, null variance (b/a^2)^w)."""
        w = np.asarray(w, dtype=np.float64)
        return self.st.a ** -w, (self.st.b / self.st.a ** 2) ** w

    def informative(self, first: np.ndarray, last: np.ndarray) -> np.ndarray:
        """Checks whose span [first, last] is not wholly inside idle runs."""
        cum = self.st.cum_idle
        return (cum[last + 1] - cum[first]) < (last - first + 1)

    def informative_run(self, first: int, last: int, width: int) -> np.ndarray:
        """informative() for the spans [first + q, last + q], q < width."""
        cum = self.st.cum_idle
        return (cum[last + 1:last + 1 + width] - cum[first:first + width]) \
            < (last - first + 1)


@functools.lru_cache(maxsize=None)
def _planned_ber(rate: float) -> float:
    """The BER H1 is planned at for a code of this rate: EXPECTED_BER, or
    the BSC capacity limit 1 - H2(p) = rate where that is lower (3.1% at
    rate 4/5, 4.2% at 3/4: planning a rate-4/5 code's weight-33 checks at
    10% would make them wait for evidence a decodable stream never lacks)."""
    lo, hi = 0.0, 0.5
    for _ in range(_BISECT):
        p = (lo + hi) / 2
        if 1 + p * np.log2(p) + (1 - p) * np.log2(1 - p) > rate:
            lo = p
        else:
            hi = p
    return min(EXPECTED_BER, (lo + hi) / 2)


def _check_values(pr: _Probe, taps: list, width: int) -> np.ndarray:
    """The check with these taps at window starts 0..width-1, one
    contiguous slice per tap (a cache-sized working set per check): the
    product of the soft values, or for hard input the odd parities (bool:
    XOR as not_equal, several times faster than on uint8)."""
    if pr.st.hard:
        odd = pr.odd_bits[taps[0]:taps[0] + width].copy()
        for k in taps[1:]:
            np.not_equal(odd, pr.odd_bits[k:k + width], out=odd)
        return odd
    v = pr.t[taps[0]:taps[0] + width].copy()
    for k in taps[1:]:
        v *= pr.t[k:k + width]
    return v


def _scan_checks(entry: dict, J: int) -> list:
    """The probe checks the scan reads: the whole lightest weight class
    (at most MAX_PROBE_CHECKS), then the next classes, lightest first, up
    to about SCAN_VALUES values per alignment over J codewords."""
    budget = max(BLOCK, SCAN_VALUES // J, len(entry["checks"][0]))
    out = []
    for taps in entry["checks"]:
        if budget <= 0:
            break
        out.append(taps[:budget])
        budget -= len(out[-1])
    return out


def _scan(pr: _Probe, entry: dict, codewords: int, alpha: float,
          beta: float):
    """Every alignment 0..n-1 of one code over at least `codewords`
    codewords: per polarity (index 0 as received, 1 inverted: every
    odd-weight check flips sign; tested when the checks have odd weights)
    the SPRT state, its LLR, values used and start codeword, and the z of
    the whole scan, shaped (polarities, n); per alignment the check values
    read.  Window start q = j*n + a is codeword j at alignment a, so each
    check is evaluated once over q in [0, J*n) and the alignments are the
    minor axis; the SPRT reads blocks of BLOCK checks of one weight,
    codeword after codeword, so the inverted polarity only re-signs block
    sums."""
    n = entry["n"]
    # a code with few probe checks reads more codewords, so that every code
    # reads about SCAN_VALUES values per alignment (AR4JA r4/5 has 128
    # checks of weight 33 and 35: 16 codewords)
    n_checks = sum(len(t) for t in entry["checks"])
    J = min(max(codewords, -(-SCAN_VALUES // n_checks)), (pr.N - n + 1) // n)
    if J < 1:
        return None
    width = J * n
    checks = _scan_checks(entry, J)
    pol = np.array([1.0, -1.0] if any(t.shape[1] % 2 for t in checks) else [1.0])
    P = len(pol)
    p_plan = _planned_ber(entry["k"] / entry["n"])
    inc, used = [], []
    zsum, vsum, nsum = np.zeros((P, width)), np.zeros(width), np.zeros(width)
    for taps in checks:
        w = taps.shape[1]
        flip = pol if w % 2 else np.ones_like(pol)
        scale, var0 = pr.norm(w)
        m1 = float(planned_m1(w, p_plan))
        for g in range(0, len(taps), BLOCK):
            group = taps[g:g + BLOCK].tolist()
            # odd parities count in int32, soft products sum in float32
            s = np.zeros(width, np.int32 if pr.st.hard else np.float32)
            cnt = np.zeros(width) if pr.idle else np.full(width, float(len(group)))
            for tp in group:
                v = _check_values(pr, tp, width)
                if pr.idle:
                    keep = pr.informative_run(tp[0], tp[-1], width)
                    v = v & keep if pr.st.hard else np.where(keep, v, np.float32(0))
                    cnt += keep
                s += v
            # +-1 values: count - 2 * odd
            s = cnt - 2.0 * s if pr.st.hard else s.astype(np.float64)
            s *= scale
            sp = flip[:, None] * s           # (polarities, width)
            # a block moves the LLR by (m1 * sum - m1^2/2 * count) / var0
            inc.append(((m1 * sp - m1 * m1 / 2 * cnt) / var0).reshape(P, J, n))
            used.append(np.broadcast_to(cnt.reshape(J, n), (P, J, n)))
            zsum += sp
            vsum += var0 * cnt
            nsum += cnt
    zs = zsum.reshape(P, J, n).sum(axis=1)
    vs = vsum.reshape(J, n).sum(axis=0)
    z = zs / np.sqrt(np.maximum(vs, 1e-300))

    def blocks(x: list) -> np.ndarray:
        """Codeword-major blocks; columns = polarity x alignment."""
        return np.stack(x, axis=2).transpose(1, 2, 0, 3).reshape(-1, P * n)

    state, llr, n_used, start = _sprt(blocks(inc), blocks(used), alpha, beta,
                                      per_cw=len(inc), starts=LEAD_CODEWORDS + 1)
    shape = (P, n)
    return {"entry": entry, "codewords": J, "z": z, "llr": llr.reshape(shape),
            "used": n_used.reshape(shape), "start": start.reshape(shape),
            "state": _agree(state.reshape(shape), z),
            "n_values": nsum.reshape(J, n).sum(axis=0)}


def _confirm(pr: _Probe, entry: dict, a: int, inverted: bool, codewords: int,
             alpha: float, beta: float) -> dict:
    """The Early Verdict of one alignment and polarity over up to
    `codewords` codewords and every probe check, read like the scan
    (blocks of BLOCK checks of one weight, codeword after codeword, from
    any of the first LEAD_CODEWORDS + 1), the z of all of them, and the
    hard parities' mean and BER from the codeword the test started at."""
    n = entry["n"]
    J = min(codewords, (pr.N - a) // n)
    if J < 1:
        return {"verdict": "undecided", "llr": 0.0, "checks_used": 0,
                "false_alarm": float(alpha), "ber_estimate": 0.5, "z": 0.0,
                "codewords": 0, "hard_mean": 0.0}
    starts = a + n * np.arange(J)
    p_plan = _planned_ber(entry["k"] / entry["n"])
    inc, used, classes = [], [], []
    zsum = vsum = 0.0
    for taps in entry["checks"]:
        w = taps.shape[1]
        flip = bool(inverted and w % 2)       # an inverted word fails odd checks
        scale, var0 = pr.norm(w)
        m1 = float(planned_m1(w, p_plan))
        pos = starts[:, None, None] + taps[None, :, :]       # (J, C, w)
        par = (pr.bits[pos].sum(axis=2) + flip) & 1
        if pr.st.hard:
            s = 1.0 - 2.0 * par
        else:
            s = pr.t[pos].prod(axis=2, dtype=np.float64) * (-1.0 if flip else 1.0)
        s = s * scale
        keep = pr.informative(pos[:, :, 0], pos[:, :, -1]) if pr.idle \
            else np.ones(s.shape, dtype=bool)
        v = np.where(keep, (m1 * s - m1 * m1 / 2) / var0, 0.0)
        for g in range(0, len(taps), BLOCK):
            inc.append(v[:, g:g + BLOCK].sum(axis=1))
            used.append(keep[:, g:g + BLOCK].sum(axis=1))
        zsum += float(np.where(keep, s, 0.0).sum())
        vsum += float(keep.sum()) * var0
        # per codeword: hard parity sum and count, for the BER
        classes.append((w, np.where(keep, 1.0 - 2.0 * par, 0.0).sum(axis=1),
                        keep.sum(axis=1)))
    state, llr, n_used, start = _sprt(
        np.stack(inc, axis=1).reshape(-1, 1),
        np.stack(used, axis=1).reshape(-1, 1).astype(np.float64), alpha, beta,
        per_cw=len(inc), starts=LEAD_CODEWORDS + 1)
    z = zsum / np.sqrt(max(vsum, 1e-300))
    j0 = int(start[0])
    counts = [int(c[j0:].sum()) for _, _, c in classes]
    hard_mean = sum(float(h[j0:].sum()) for _, h, _ in classes) / max(sum(counts), 1)
    ber = _ber_from_mean(hard_mean, [w for w, _, _ in classes], counts) \
        if sum(counts) else 0.5
    st = _agree(state, np.array([z]))[0]
    return {"verdict": _VERDICT[int(st)], "llr": float(llr[0]),
            "checks_used": int(n_used[0]), "false_alarm": float(alpha),
            "ber_estimate": float(ber), "z": float(z), "codewords": int(J),
            "hard_mean": float(hard_mean)}


def prescreen(bits: np.ndarray, llrs: np.ndarray = None, entries: list = None,
              codewords: int = PRESCREEN_CODEWORDS,
              confirm_codewords: int = CONFIRM_CODEWORDS,
              alpha: float = ALPHA, beta: float = BETA) -> list:
    """LDPC Pre-screen: which registered code, at which alignment, before
    any decode.  prescreen(bits, llrs=...) takes the bits (uint8, e.g. a
    de-interleaved window) with optional aligned llrs (> 0 => bit 0);
    prescreen(llrs) takes LLRs alone (a floating array; the bits are their
    signs).  entries: registry LDPC entries (default: the bundled codes;
    identify_ldpc passes its candidates).

    Every code is scored at every alignment 0..n-1 and both polarities
    (an inverted codeword fails exactly the odd-weight checks) over
    `codewords` codewords and judged by the Early Verdict; the best (code,
    polarity, alignment) whose verdict is "code" must also stand DECOY_Z
    above the scan's other hypotheses of its check weight (decoy_null).
    The alignment is then settled among its neighbours by the z of a
    confirmation over `confirm_codewords` codewords and every probe check
    (see the module docstring), whose verdict must be "code" too.  Returns
    [hit] or []: hit = {"name", "entry", "code", "n", "k", "offset" (bits
    to skip to the first whole codeword), "inverted" (decode the
    complemented bits), "early_verdict" (the confirmation's, with the BER
    of its hard parities), "z" (the scan's at the offset), "decoy_z" (how
    far the code's best scan alignment stands above the decoys),
    "check_weight", "n_checks" (check values the scan read at the
    offset), "codewords" (the scan's), "confirm" (the confirmation's
    verdict, z, codewords, hard_mean), "check_zero_rate"
    ((1 + hard_mean)/2)}."""
    # ponytail: the scan reads the stream's first few codewords (as the old
    # sweep did) and the Early Verdict may skip LEAD_CODEWORDS of them; a
    # longer preamble hides the code - slide the scan window if real
    # captures need it
    first = np.asarray(bits)
    if first.dtype.kind == "f" and llrs is None:
        llrs, bits = first, first < 0
    if entries is None:
        from .registry import ldpc_entries      # the registry imports this module
        entries = ldpc_entries()
    bits = np.asarray(bits, dtype=np.uint8)
    soft = None
    if llrs is not None and len(llrs) == len(bits):
        soft = np.asarray(llrs, dtype=np.float64)
    if not len(bits) or not entries:
        return []
    pr = _Probe(bits, soft)
    scans = [s for s in (_scan(pr, e, codewords, alpha, beta) for e in entries)
             if s is not None]
    if not scans:
        return []
    # every (code, polarity, alignment) of the scan is a decoy for the others
    null = decoy_null(np.concatenate([s["z"].ravel() for s in scans]),
                      np.concatenate([np.full(s["z"].size, s["entry"]["check_weight"])
                                      for s in scans]))
    best = None
    for s in scans:
        cand = np.argwhere(s["state"] == 1)
        if not len(cand):
            continue
        p, a = (int(v) for v in cand[np.argmax(s["z"][cand[:, 0], cand[:, 1]])])
        rel = null(s["z"][p, a], s["entry"]["check_weight"])
        if best is None or rel > best[0]:
            best = (rel, s, p, a)
    if best is None or best[0] < DECOY_Z:
        return []
    rel, s, p, a = best
    e = s["entry"]
    confs = {}

    def conf(b: int) -> dict:
        b %= e["n"]
        if b not in confs:
            confs[b] = _confirm(pr, e, b, bool(p), confirm_codewords, alpha, beta)
        return confs[b]

    for _ in range(SETTLE_MOVES):
        near = [(a + d) % e["n"] for d in range(-SETTLE_BITS, SETTLE_BITS + 1)]
        b = max(near, key=lambda c: conf(c)["z"])
        if conf(b)["z"] <= conf(a)["z"]:
            break
        a = b
    c = conf(a)
    if c["verdict"] != "code":
        return []
    ev = {k: c[k] for k in ("verdict", "llr", "checks_used", "false_alarm",
                            "ber_estimate")}
    return [{"name": e["name"], "entry": e, "code": e["code"], "n": e["n"],
             "k": e["k"], "offset": a, "inverted": bool(p), "early_verdict": ev,
             "z": float(s["z"][p, a]), "decoy_z": float(rel),
             "check_weight": e["check_weight"],
             "n_checks": int(round(s["n_values"][a])), "codewords": s["codewords"],
             "confirm": c,
             "check_zero_rate": (1.0 + float(np.clip(c["hard_mean"], -1, 1))) / 2}]


def settle_offset(code: LDPCCode, bits: np.ndarray, offset: int,
                  llrs: np.ndarray = None, blocks: int = SETTLE_BLOCKS) -> int:
    """The alignment near `offset` whose layered decode explains the stream
    best: on the first `blocks` codewords, the decode one bit to either
    side replaces the centre when, over the codewords both converge on, it
    flips fewer received bits (a quasi-cyclic window one bit off converges
    onto the rotated codewords and also flips the circulant boundaries'
    bits); repeated while it moves, at most SETTLE_BITS times.  Without a
    codeword both converge on it stays."""
    n = code.n
    base = offset + n if offset <= SETTLE_BITS else offset   # room to step left
    runs = {}

    def flips(b: int) -> tuple:
        if b not in runs:
            blk, words, ok = code.decode_blocks(bits, b, blocks, llrs)
            runs[b] = ((words[:, code.transmitted] != blk).sum(axis=1), ok)
        return runs[b]

    def gain(a: int, b: int):
        """Bits the decode at a flips beyond the one at b (None: no
        codeword both converge on)."""
        (fa, oa), (fb, ob) = flips(a), flips(b)
        m = min(len(oa), len(ob))
        both = oa[:m] & ob[:m]
        return int(fa[:m][both].sum() - fb[:m][both].sum()) if both.any() else None

    a = base
    for _ in range(SETTLE_BITS):
        g = [(gain(a, b), b) for b in (a - 1, a + 1)]
        g = [(v, b) for v, b in g if v is not None and v > 0]
        if not g:
            break
        a = max(g)[1]
    return a % n


def identify_ldpc(bits: np.ndarray, candidates: list,
                  quick_codewords: int = 8, min_rate: float = 0.5,
                  llrs: np.ndarray = None) -> list:
    """The Pre-screen over the candidates: seeded codes {"n", "k", "seed"},
    bundled ones {"name": ...} (registry.ldpc_entries) and matrix files
    {"path": ...} (registry.ldpc_entry).  At most one code passes, the
    Pre-screen's (code, alignment); as before there is one hit per
    candidate naming it (a code listed twice is scored once and reported
    twice), each with HEAD's keys - the candidate's own, n, k, seed (a
    parity-check code's is its name, as from_parity_check always set
    it), offset, syndrome_zero_rate (the clean fraction of up to
    CONFIRM_CODEWORDS codewords as received; of the complemented bits
    when the hit is inverted), code - plus name, inverted, early_verdict,
    z, decoy_z, check_weight, confirm; [] when nothing passes.  min_rate
    is the old sweep's gate and still applies: the clean fraction of the
    first quick_codewords codewords must exceed it.  None turns it off: a
    noisy stream that decodes has no clean codeword (identify_fec)."""
    from .registry import ldpc_entry
    bits = np.asarray(bits, dtype=np.uint8)
    entries, owners = [], {}
    for cand in candidates or []:
        e = ldpc_entry(cand)
        if id(e) not in owners:
            owners[id(e)] = []
            entries.append(e)
        owners[id(e)].append(cand)
    hits = []
    for h in prescreen(bits, llrs, entries, codewords=quick_codewords) if entries else []:
        code = h["code"]
        src = bits ^ np.uint8(h["inverted"])
        if min_rate is not None and \
                not code.syndrome_zero_rate(src, h["offset"], quick_codewords) > min_rate:
            continue
        szr = code.syndrome_zero_rate(src, h["offset"], CONFIRM_CODEWORDS)
        hits += [{**cand, "n": code.n, "k": code.k, "seed": code.seed,
                  "offset": h["offset"], "syndrome_zero_rate": szr, "code": code,
                  "name": h["name"], "inverted": h["inverted"],
                  "early_verdict": h["early_verdict"], "z": h["z"],
                  "decoy_z": h["decoy_z"], "check_weight": h["check_weight"],
                  "confirm": h["confirm"]} for cand in owners[id(h["entry"])]]
    return hits
