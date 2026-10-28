"""Pattern Scan (stage S8): the interleaver's geometry and the code under
it, found together.

The rank profile shows an interleaver only through its period, and that
period is gone at about 0.5% BER (it never shows over Reed-Solomon).  The
Pattern Scan looks for the code itself, through the interleaver.

Dilation scan.  Under a block interleaver with R rows, neighbouring bits of
a code row sit R apart, so a code check with taps T holds as the dilated
check R*T wherever its window stays inside one row.  The lightest check of
every rate-1/2 registry code is tested at every dilation D = 1..max_dim+8
on hard parities (bit-packed: one XOR covers 8 windows); the z of each
global mean finds the spacing.  D = 1 strongest (apart from dilations the
stream's own repetition explains) means plainly coded, not interleaved.  A
spacing D gives block R = D, helical R = D - step and convolutional
B*M = D - 1.  It must stand out from every other dilation and from the
other codes at the same D (an interleaved code moves its own checks only),
and its taps must not pair up modulo a lag at which the stream repeats
itself (on repeated frames or fill such a check holds whatever the code).

Fold and template.  Folded by P = R*C, the dilated check has a known shape:
it holds in the columns c <= C - span whose code position has the check's
phase, and (helical) in the rows whose rotation does not wrap inside the
window.  The column means rank C against the valid-column pattern with its
mean removed: a check that holds in every column (a uniform bias, or a
linear recurrence such as an LFSR sequence decimated by D) fits no C.  The
best few C are then correlated with the full template over all P residues
(one FFT each), which gives the block start.

Period candidates.  When the rank profile does show a period P, its block,
helical, convolutional and 802.11 geometries are screened at every offset
on short de-interleaved windows (bit-packed parities of the same checks),
taken mid-way through the scanned stretch like the selection windows below
(a lead-in of non-code bits then costs nothing); the 802.11 modes are
screened on flat profiles too (a short capture hides their period).  Block offsets run over 1.5 periods, so neighbouring offsets
meet on shared bits even across the wrap.  Helical goes last, as in the
rank-profile sweep.  The work is capped by counts, not time, so a slow or
busy machine finds what a fast one finds.

Confirm.  The candidates' de-interleaved windows, mid-way through the
scanned stretch, are scored together on hard parities, best by mean then z
(the offsets of one geometry read the same stream bits); the winner's
window is checked against every check of
every registry code (rate 1/3 and punctured codes included: they veto a
rate-1/2 claim they fit better) and against the raw stream, then its full
de-interleaved stream, from where the code starts to hold, against the
rate-1/2 codes with the Code Probe (soft when LLRs come along) and the
Early Verdict.

Reed-Solomon.  A code symbol that starts a row's byte is 8 bits of one row:
a byte dilated by R in the interleaved stream.  For a ladder of common
geometries every offset's symbol stream gets one prefix-XOR pass per root
family (four shared roots packed in a uint32), which tests every codeword
alignment at once, as the Code Probe's RS scan does; the windows come from
the middle of the stream, and the survivors go to identify_rs on their
de-interleaved stream.

Acceptance is calibrated like identify_convolutional's: a geometry is
reported only if (i) its Early Verdict is "code" on the full de-interleaved
stream, with every check of the code holding (with the signs of one
substream inversion pattern), (ii) it stands DECOY_Z above the decoys of
this stream (the scan's other dilations or windows, every other check on
the de-interleaved window and stream, each against its own check weight;
for RS the clean rate of every other alignment), the claimed check at its
other phase reads as a decoy (a code check holds at one phase; one that
holds at both is a recurrence of the data) and the raw stream shows no code
half as strongly (a geometry that is the identity on this stream, such as
convolutional B=2 M=1 on any rate-1/2 code, is no interleaver), and (iii)
identify_convolutional / identify_rs on the de-interleaved stream name the
same code, whose decode re-encodes within sampling error of the BER its
checks read (identify_fec's _decode_agrees, for every claim).  Offsets
follow the engine's convention: skip `offset` bits, then de-interleave.

ponytail: the scan, the screening and the selection read hard parities;
LLRs enter the confirmation only (soft probe, identify_convolutional,
Viterbi), so they do not widen the BER range (K=7 under 8x16 is found up to
about 6% BER at 64k bits; soft dilated products would reach ~8%).
"""
from __future__ import annotations

import functools
import math

import numpy as np

from ..common.config import FECConfig
from ..common.models import InterleaverHypothesis
from ..fec import probe
from ..fec.conv import ConvCode, conv_encode, viterbi_decode
from ..fec.detect import _decode_agrees, identify_convolutional, identify_rs
from ..fec.probe import DECOY_Z, Z_CONFIRM, _idle_positions, decoy_null
from ..fec.registry import conv_entries, rs_entry
from .interleavers import IEEE80211_MODES, ieee80211_permutation

# shortest capture searched: below it the decoys, the Early Verdict and the
# decode check get too few checks to tell a code from structured data
# (ponytail: clean wide rows are found from ~2000 bits; shorter captures
# are left to the rank-profile sweep)
MIN_BITS = 4096
SCAN_BITS = 1 << 16      # bits the dilation scan reads: K=7 under 8x16 at 3% BER gives z ~ 16
DILATION_EXTRA = 8       # D runs to max_dim + 8: helical R + step, convolutional B*M + 1
MIN_SCAN_WINDOWS = 512   # dilated windows a z needs (8 per packed byte)
# a spacing or window worth a closer look: the ~10^3 dilations and ~10^5
# window tests of one scan stay below ~5 on independent data
SCAN_Z = 6.0
MAX_SPACINGS = 2         # strongest dilations followed up
GEOMETRIES_PER_SPACING = 2   # best template fits per spacing (block, helical steps, C)
TOP_C = 3                # column counts matched at full resolution (2C also fits, worse)
TOP_STARTS = 3           # block starts per fit: a start one row off scores ~0.9 of the true one
MIN_FOLD_ROWS = 8        # blocks a fold averages over, at least
MAX_LAG = 2048           # repetition lags tested: a dilated check spans <= 72 x 22 bits
REPEAT_LAGS = 4          # strongest repetition lags a spacing is checked against
HELICAL_STEPS = (1, 2, 3, 4, 5)   # the rank-profile sweep's helical steps...
HELICAL_PAIRS = 6        # ...tried on the first six factor pairs of a period
BRANCHES = range(2, 17)  # the rank-profile sweep's convolutional branch counts...
DELAYS = (1, 2, 4, 8)    # ...and delays (plus P // B)
SCREEN_BITS = 512        # shortest window per screened offset: z ~ 14 at 0.5% BER
SCREEN_MAX_BITS = 4096   # longest: z ~ 24 for K=7 at 3% BER, well past DECOY_Z
SCREEN_SHARE = 4         # a screened window takes at most 1/4 of the stream (every offset's must fit)
# all screened windows together: few geometries get long windows, many get
# SCREEN_BITS each
SCREEN_TOTAL_BITS = 1 << 20
# hard cap per screening pass (~0.3 s of parities): geometries past it are
# not screened, in their order (802.11, then by rank-profile period)
SCREEN_CAP_BITS = 1 << 22
SCREEN_KEEP = 8          # offsets kept per screened geometry, at most
SCREEN_TIE = 2.0         # ...those within this z of its best (a row-off start scores close)
SCREEN_NEAR = 4          # ...and the best one's neighbours this far either side
MAX_SCREENED = 16        # screened candidates the code selection probes
# de-interleaved bits the code selection probes: at 3% BER this separates
# a block start from its neighbour one row off (0.9 of the mean) by ~5 sigma
SELECT_BITS = 16384
CONFIRM_BITS = 100000    # the winner's full-stream probe: identify_convolutional's max_test_bits
IDENTIFY_BITS = 32768    # the window identify_convolutional confirms on: z ~ 70 for K=7 at 3% BER
DECODE_BITS = 16384      # coded bits the decode check decodes (~50 ms of Viterbi at K=7)
# the raw stream's best check may reach this share of the winner's z at
# most: an interleaved code shows nothing undilated (|z| ~ 3 at worst over
# the scan checks), a geometry that is the identity on the stream the same z
NONE_RATIO = 0.5
TRIM_BLOCK = 256         # check values per block when finding where the code starts
# mean |tanh(L/2)| below this: the LLRs carry no reliability (all zero, NaN
# or vanishing), and the probe's a^-w normalisation would overflow
SOFT_MIN = 1e-3
# bit block sizes common enough to try under RS without a period: squares
# and 2:1 rectangles; a 64-row or 64-column geometry costs 4-8x one of these
RS_LADDER = ((8, 8), (8, 16), (16, 8), (16, 16), (16, 32), (32, 16), (32, 32))
RS_WINDOWS = 3           # codeword windows per alignment the RS prefilter reads
RS_ROOTS = 4             # roots of a family tested at once, packed in a uint32
RS_PRE_P0 = 2.0 ** -32   # chance a window of independent symbols passes four roots
RS_PERIODS = (1, 3, 5)   # symbol periods d | 255 that make a window a codeword of most root sets
RS_CHUNK = 4096          # ladder offsets per prefilter pass (memory)
RS_MAX_COLUMNS = 1 << 14  # offsets the prefilter tests in all (~0.3 s), smallest periods first
RS_MAX_PASSES = 4096     # windows passing one pass; independent data gives ~2.7e6 x 2^-32
MAX_RS_CONFIRM = 4       # prefilter survivors handed to identify_rs
RS_CONFIRM_BITS = 66 * 255 * 8   # identify_rs reads 64 + 2 codewords of 255 bytes at most
_LOG_DECOY_P = math.log(0.5 * math.erfc(DECOY_Z / math.sqrt(2.0)))  # one-sided N(0,1) tail
_IDLE_FILL = 0x5EED      # seed of the fixed fill that replaces idle runs in the scan
_TINY = 1e-12            # energy floor: an empty template or a constant stream scores 0
_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
# per-byte popcount: numpy >= 2 has it built in, older numpy looks it up
_popcount = getattr(np, "bitwise_count", None) or (lambda a: _POP8[a])
_EVEN, _ODD = 0xAA, 0x55  # packbits is MSB first: even window starts are the odd bits
_GEOMETRY_KEYS = ("rows", "cols", "step", "branches", "delay", "ncbps", "nbpsc")
_INVERSIONS = ((0, 0), (1, 0), (0, 1), (1, 1))  # flips of the even, odd stream bits


@functools.lru_cache(maxsize=1)
def _scan_entries() -> tuple:
    """The codes the scan looks for: the registry's rate-1/2 unpunctured
    codes, the ones identify_convolutional can confirm.  Each is scanned
    by its first check (the registry orders them by span)."""
    # ponytail: rate-1/3 and punctured codes only veto a claim (window
    # decoys); none is reported, as no commit-1 identifier confirms them
    # (identify_fec decodes rate 1/2 only)
    return tuple(e for e in conv_entries()
                 if e["rate"] == "1/2" and e["puncture"] is None)


def _packs(bits2d: np.ndarray) -> list:
    """bits2d[..., k:] bit-packed for k = 0..7: a tap at offset d is the
    slice [d // 8:] of pack d % 8, each byte holding 8 window starts."""
    return [np.packbits(bits2d[..., k:], axis=-1) for k in range(8)]


def _parity(packs: list, taps: np.ndarray, n_bytes: int) -> np.ndarray:
    """Packed parities of the check `taps` at window starts 0..8*n_bytes-1."""
    d = int(taps[0])
    acc = packs[d & 7][..., d >> 3:(d >> 3) + n_bytes].copy()
    for d in taps[1:]:
        d = int(d)
        np.bitwise_xor(acc, packs[d & 7][..., d >> 3:(d >> 3) + n_bytes],
                       out=acc)
    return acc


def _check_values(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """The check's +-1 value at every window start of x (hard parities)."""
    n_bytes = (len(x) - int(taps[-1])) // 8
    if n_bytes <= 0:
        return np.zeros(0)
    return 1.0 - 2.0 * np.unpackbits(_parity(_packs(x[None, :]), taps, n_bytes)[0])


def _stands_out(z_abs: np.ndarray, i: int) -> float:
    """|z| of code i in units of the other codes' |z| on the same windows
    (median, 1.4826 MAD, spread at least 1).  An interleaved code makes its
    own checks hold and leaves every other code's at N(0, 1); periodic data
    (frames, a fixed header, text) biases many codes' checks at once."""
    others = np.delete(z_abs, i)
    c = float(np.median(others))
    s = max(1.0, 1.4826 * float(np.median(np.abs(others - c))))
    return float((z_abs[i] - c) / s)


def _window_z(X: np.ndarray, taps: list) -> tuple:
    """Each row of X is a de-interleaved window: (z, mean) of every check's
    hard parities per phase, shaped (rows, checks, 2), all rows at once."""
    W = X.shape[1]
    packs = _packs(X)
    Z = np.empty((len(X), len(taps), 2))
    M = np.empty_like(Z)
    for h, tp in enumerate(taps):
        n_bytes = (W - int(tp[-1])) // 8
        par = _parity(packs, tp, n_bytes)
        m = 4 * n_bytes                       # windows per phase
        for ph, mask in ((0, _EVEN), (1, _ODD)):
            odd = _popcount(par & mask).sum(axis=1, dtype=np.int64)
            M[:, h, ph] = 1.0 - 2.0 * odd / m
            Z[:, h, ph] = M[:, h, ph] * np.sqrt(m)
    return Z, M


# --------------------------------------------------------- geometries ---
def _geometry(kind: str, **params) -> dict:
    """An interleaver hypothesis: its parameters (the engine's names), its
    period and, for the block permutations (block, helical, 802.11), q[i] =
    the position inside an interleaved block of de-interleaved bit i."""
    g = {"kind": kind, "params": params}
    if kind == "convolutional":
        B, M = params["branches"], params["delay"]
        g.update(period=B, reach=(B - 1) * B * M, offsets=B)
        return g
    if kind == "ieee80211":
        P = params["ncbps"]
        q = np.argsort(ieee80211_permutation(P, params["nbpsc"]))
    else:
        R, C = params["rows"], params["cols"]
        P = R * C
        r, c = np.divmod(np.arange(P), C)
        q = c * R + (r + c * params.get("step", 0)) % R
    g.update(period=P, reach=0, offsets=P, q=q)
    return g


# the 802.11 BPSK and QPSK modes are exactly the block interleavers
# (ncbps/16) x 16 (checked here): {(rows, cols): (ncbps, nbpsc)}
_IEEE_BLOCKS = {(n // 16, 16): (n, b) for n, b in IEEE80211_MODES
                if np.array_equal(_geometry("ieee80211", ncbps=n, nbpsc=b)["q"],
                                  _geometry("block", rows=n // 16, cols=16)["q"])}


def _key(g: dict) -> tuple:
    return (g["kind"],) + tuple(g["params"].get(k) for k in _GEOMETRY_KEYS)


def _positions(g: dict, n: int) -> np.ndarray:
    """Where the first n de-interleaved bits sit, counted from the offset:
    exactly what the engine's _deinterleave_with reads."""
    k = np.arange(n)
    if g["kind"] == "convolutional":
        B, M = g["params"]["branches"], g["params"]["delay"]
        return k + (k % B) * (B * M)
    P = g["period"]
    return (k // P) * P + g["q"][k % P]


def _usable(g: dict, N: int, offset: int) -> int:
    """De-interleaved bits a stream of N bits yields after the offset:
    whole blocks, or (convolutional) the bits whose sources are inside."""
    if g["kind"] == "convolutional":
        return max(0, N - offset - g["reach"])
    return max(0, (N - offset) // g["period"]) * g["period"]


def _window(g: dict, offset: int, n: int, straddle: bool = False) -> np.ndarray:
    """Stream positions of a de-interleaved window of n bits.  A
    convolutional window starts `reach` (even) bits in: the first output of
    a convolutional interleaver that started inside the capture, or of one
    realised as a wrap-around permutation (conv_interleave), holds no code
    bits of this stretch.  `straddle` centres a block window on the first
    block boundary, where an offset a few rows off breaks its rows (inside
    the first rows of a block it reads as clean as the true one)."""
    k0 = g["reach"] + g["reach"] % 2
    if straddle and not g["reach"]:
        k0 = max(0, g["period"] - n // 2)
        k0 -= k0 % 2                      # keep the code phase
    return offset + _positions(g, k0 + n)[k0:]


def _windowed(g: dict, N: int, offset: int) -> int:
    """_usable after the window start."""
    return max(0, _usable(g, N, offset) - g["reach"] - g["reach"] % 2)


def _centred(g: dict, start: int, extent: int, N: int) -> int:
    """How far to move windows spanning `extent` stream bits from `start`
    so they sit mid-way through the bits the dilation scan reads, as far as
    the stream allows: a whole, even number of the geometry's offset
    periods (offsets and the code phase keep their meaning).  A lead-in or
    a tail of non-code bits (pre-lock bits, a preamble) then costs the
    screening and the selection nothing."""
    step = g["offsets"] * (1 + g["offsets"] % 2)
    target = min(min(N, SCAN_BITS) // 2 - extent // 2, N - extent)
    return max(0, (target - start) // step * step)


def _period_geometries(periods, max_dim: int) -> list:
    """The geometries the rank-profile sweep tries for each period (block
    and helical factor pairs as detect._factor_pairs, branch x delay), the
    802.11 modes always and first (a fixed cost the screening cap must not
    drop); helical last, the costliest to screen."""
    first = [_geometry("ieee80211", ncbps=n, nbpsc=b) for n, b in IEEE80211_MODES]
    helical = []
    for P in dict.fromkeys(int(p) for p in periods):
        pairs = [(r, P // r) for r in range(2, min(P, max_dim) + 1)
                 if P % r == 0 and 2 <= P // r <= max_dim]
        first += [_geometry("block", rows=r, cols=c) for r, c in pairs]
        for B in BRANCHES:
            if P % B == 0:
                for M in dict.fromkeys(DELAYS + (max(1, P // B),)):
                    first.append(_geometry("convolutional", branches=B, delay=M))
        helical += [_geometry("helical", rows=r, cols=c, step=s)
                    for r, c in pairs[:HELICAL_PAIRS] for s in HELICAL_STEPS if s < r]
    return first + helical


def _reported(g: dict) -> tuple:
    """(kind, parameters) as reported: an 802.11 BPSK or QPSK mode found as
    its block interleaver is named by the standard, as the rank-profile
    sweep names it (the de-interleaving is the same)."""
    p = g["params"]
    mode = _IEEE_BLOCKS.get((p.get("rows"), p.get("cols"))) \
        if g["kind"] == "block" else None
    if mode:
        return "ieee80211", {"ncbps": mode[0], "nbpsc": mode[1]}
    return g["kind"], dict(p)


def _reported_period(g: dict, periods) -> int:
    """InterleaverHypothesis.period as the rank-profile sweep sets it: the
    block period, and for a convolutional geometry the profile period the
    sweep tries it at (None on a flat profile: the sweep never does)."""
    if not g["reach"]:
        return g["period"]
    B, M = g["params"]["branches"], g["params"]["delay"]
    return next((P for P in periods
                 if P % B == 0 and M in DELAYS + (max(1, P // B),)), None)


# ----------------------------------------------------- dilation scan ---
def _idle_filled(bits: np.ndarray) -> np.ndarray:
    """A copy with every idle run (IDLE_RUN equal bits or more) replaced by
    a fixed random fill: a constant run satisfies every even check at every
    dilation, and its all-zero symbols are a codeword of every RS code."""
    b = np.array(bits, dtype=np.uint8)
    idle = _idle_positions(b)
    if idle.any():
        fill = np.random.default_rng(_IDLE_FILL).integers(0, 2, len(b), dtype=np.uint8)
        b[idle] ^= fill[idle]
    return b


class _Scan:
    """Every scan check at every dilation D: the z of its global hard mean
    over the first SCAN_BITS bits (idle runs filled, _idle_filled)."""

    def __init__(self, bits: np.ndarray, d_max: int):
        n = min(len(bits), SCAN_BITS)
        b = _idle_filled(bits[:n])
        self.n, self.packs = n, _packs(b)
        self.entries = _scan_entries()
        self.taps = [e["checks"][0]["taps"] for e in self.entries]
        self.span = np.array([int(t[-1]) + 1 for t in self.taps])
        self.weight = np.array([len(t) for t in self.taps])
        self.z = np.zeros((len(self.taps), d_max + 1))
        done = np.zeros(self.z.shape, dtype=bool)
        for i, taps in enumerate(self.taps):
            for D in range(1, d_max + 1):
                n_bytes = (n - D * int(taps[-1])) // 8
                if 8 * n_bytes < MIN_SCAN_WINDOWS:
                    break
                odd = int(_popcount(_parity(self.packs, D * taps, n_bytes))
                          .sum(dtype=np.int64))
                m = 8 * n_bytes
                self.z[i, D] = (1.0 - 2.0 * odd / m) * np.sqrt(m)
                done[i, D] = True
        # every check at every interleaver spacing is a decoy for the others
        done[:, :2] = False
        self.null = decoy_null(self.z[done],
                               np.broadcast_to(self.weight[:, None], done.shape)[done])
        # the stream's own repetition lags (frames, fill): |autocorrelation|
        # z over the lags a dilated check can span, mean removed (unbalanced
        # bits correlate at every lag)
        t = 1.0 - 2.0 * b
        t -= t.mean()
        F = np.fft.rfft(t, 2 * n)
        L = np.arange(1, min(MAX_LAG, n // 2) + 1)
        za = np.abs(np.fft.irfft(F * np.conj(F))[L]) / \
            (np.sqrt(n - L) * max(float((t * t).mean()), _TINY))
        rep = np.flatnonzero(za >= SCAN_Z)
        self.lags = L[rep[np.argsort(-za[rep], kind="stable")][:REPEAT_LAGS]]
        # plainly coded: the undilated check is the strongest, apart from
        # dilations the repetition explains (on repeated frames a check
        # whose dilated taps pair up modulo the frame holds there too)
        top1 = float(np.abs(self.z[:, 1]).max(initial=0.0))
        stronger = np.argwhere(np.abs(self.z[:, 2:]) > top1)
        self.plain = top1 >= SCAN_Z and all(
            self.repeats((D + 2) * self.taps[i]) for i, D in stronger)

    def relative(self, i: int, D: int) -> float:
        z = self.z[i, D]
        return self.null(z, self.weight[i], 1 if z >= 0 else -1)

    def spacings(self) -> list:
        """(D, check index) of the strongest dilations D >= 2 that stand out
        from every other dilation and from the other codes at the same D,
        and that the stream's own repetition does not explain."""
        out = []
        for D in range(2, self.z.shape[1]):
            col = np.abs(self.z[:, D])
            i = int(np.argmax(col))
            if col[i] >= SCAN_Z and self.relative(i, D) >= DECOY_Z and \
                    _stands_out(col, i) >= DECOY_Z and \
                    not self.repeats(D * self.taps[i]):
                out.append((col[i], D, i))
        out.sort(reverse=True)
        return [(D, i) for _, D, i in out[:MAX_SPACINGS]]

    def repeats(self, taps: np.ndarray) -> bool:
        """The check's taps pair up (all but at most one) modulo a lag at
        which the stream repeats itself: on repeated frames or fill each
        pair cancels, so it holds there whatever the code (K=4's taps x32
        fall pairwise on one column of a 96-bit frame)."""
        for L in self.lags:
            _, counts = np.unique(taps % L, return_counts=True)
            if (counts % 2).sum() <= 1:
                return True
        return False

    def series(self, i: int, D: int) -> np.ndarray:
        """The dilated check's +-1 value at every window start, signed so a
        constant-1 syndrome (inverted substream) reads +1 too."""
        n_bytes = (self.n - D * int(self.taps[i][-1])) // 8
        par = np.unpackbits(_parity(self.packs, D * self.taps[i], n_bytes))
        return (1.0 - 2.0 * par) * (1.0 if self.z[i, D] >= 0 else -1.0)


def _template(R: int, C: int, step: int, span: int, phase: int) -> np.ndarray:
    """1 where a dilated window may start inside an interleaved block (q =
    c*R + r'): the code window lies inside its row (c <= C - span), does
    not wrap a helical rotation, and starts on the check's phase (code
    bit r*C + c, r the code row)."""
    c, rp = np.divmod(np.arange(R * C), R)
    r = (rp - c * step) % R
    return ((c <= C - span) & (rp + (span - 1) * step < R)
            & ((r * C + c) % 2 == phase)).astype(np.float64)


def _circular_z(prof: np.ndarray, v: np.ndarray, var: float) -> np.ndarray:
    """z of the correlation of a fold with a 0..1 template at every
    circular shift (one FFT): corr[o] = sum_q v[q] prof[(q + o) mod len];
    each fold value has null variance `var`."""
    corr = np.fft.irfft(np.conj(np.fft.rfft(v)) * np.fft.rfft(prof), n=len(v))
    return corr / np.sqrt(max((v * v).sum(), _TINY) * var)


def _fit(s: np.ndarray, R: int, step: int, span: int, max_dim: int) -> list:
    """[(z, C, block starts)] of the best column counts.  The fold of s by
    P = R*C, averaged per column, ranks C against the column template
    without the phase (a block start mid-column mixes two neighbouring
    columns of opposite phase) and with its mean removed (a wrong C smears
    the pattern into a flat level); a column profile without that contrast
    (the check holds in every column: a uniform bias, or an LFSR sequence
    decimated by D, which satisfies a check at every bit) fits no C.  The
    TOP_C counts are then matched at full resolution, every residue and
    both phases, which also gives the block start.  A fold averages `rows`
    blocks: null variance 1/rows."""
    if (span - 1) * step >= R:
        return []
    ranked = []
    for C in range(span, max_dim + 1):
        P = R * C
        rows = len(s) // P
        if rows < MIN_FOLD_ROWS:
            break
        prof = s[:rows * P].reshape(rows, P).mean(axis=0)
        # without the phase every column keeps the same share of its rows
        # (the helical wrap removes the same rows in each): the column
        # template is the valid-column indicator, up to a constant
        v = (np.arange(C) <= C - span).astype(np.float64)
        z = _circular_z(prof.reshape(C, R).mean(axis=1), v - v.mean(),
                        1.0 / (rows * R))
        ranked.append((float(z.max()), C, prof, rows))
    ranked.sort(key=lambda x: -x[0])
    out = []
    for z_col, C, prof, rows in ranked[:TOP_C]:
        if z_col < SCAN_Z:
            break
        z = np.stack([_circular_z(prof, _template(R, C, step, span, ph), 1.0 / rows)
                      for ph in (0, 1)]).max(axis=0)
        top = np.argsort(-z, kind="stable")[:TOP_STARTS]
        out.append((float(z[top[0]]), C, [int(o) for o in top]))
    return sorted(out, key=lambda x: -x[0])


def _spacing_candidates(scan: _Scan, D: int, i: int, max_dim: int) -> tuple:
    """The geometries a significant spacing D of check i stands for: the
    best template fits (block R = D, helical R = D - step), each with its
    likeliest block starts, and the convolutional B*M = D - 1 (screened
    over their B offsets)."""
    s, span = scan.series(i, D), int(scan.span[i])
    fits = []
    for step in (0,) + HELICAL_STEPS:
        R = D - step
        if R < 2 or R > max_dim:
            continue
        # ponytail: helical is a pure dilation only while (span-1)*step < R
        # (13*step < R for K=7); typical small helical geometries come from
        # the rank profile's period instead, at low BER
        for z, C, starts in _fit(s, R, step, span, max_dim):
            if z >= SCAN_Z:
                kind = "helical" if step else "block"
                extra = {"step": step} if step else {}
                fits.append((z, _geometry(kind, rows=R, cols=C, **extra), starts))
    fits.sort(key=lambda f: -f[0])
    # ponytail: a convolutional interleaver is a pure dilation B*M + 1 only
    # when B >= the check span; smaller B come from the rank profile
    conv = [_geometry("convolutional", branches=B, delay=(D - 1) // B)
            for B in BRANCHES if B >= span and (D - 1) % B == 0]
    return fits[:GEOMETRIES_PER_SPACING], conv


# ---------------------------------------------------------- screening ---
def _screen(bits: np.ndarray, geos: list, taps: list) -> tuple:
    """Every offset of every geometry: z of each scan check's hard mean per
    phase over a de-interleaved window, all windows of one length and in
    one pass (few geometries get long windows, many get SCREEN_BITS; past
    SCREEN_CAP_BITS no further geometry is screened).  A block
    permutation's offsets run over 1.5 periods, r and r + P being one
    offset: every circular run of offsets then meets on shared bits
    somewhere (offsets P-1 and 0 read different blocks, P-1 and P do not),
    and each offset keeps its better alias (extra windows, not shorter
    ones).  Returns the kept candidates [(z, the best's for its
    neighbours; geometry index; offset; z array (checks, phases))], best
    first, and the null of every value (the decoys)."""
    N = len(bits)

    def n_offsets(g):
        return g["offsets"] + (0 if g["reach"] else g["offsets"] // 2)

    n_win = sum(g["offsets"] for g in geos
                if _windowed(g, N, n_offsets(g) - 1) >= SCREEN_BITS)
    W = int(np.clip(SCREEN_TOTAL_BITS // max(1, n_win), SCREEN_BITS,
                    max(SCREEN_BITS, min(SCREEN_MAX_BITS, N // SCREEN_SHARE))))
    windows, owner, total = [], [], 0
    for gi, g in enumerate(geos):
        idx = _window(g, 0, W, straddle=True)
        # every offset needs the whole window inside the stream (the
        # aliases past one period only where they fit)
        n_off = min(n_offsets(g), N - int(idx.max()))
        if n_off < g["offsets"]:
            continue
        if total + n_off * W > SCREEN_CAP_BITS:
            break
        total += n_off * W
        offs = _centred(g, 0, n_off + int(idx.max()), N) + np.arange(n_off)
        windows.append(bits[offs[:, None] + idx[None, :]])
        owner.append((gi, n_off))
    if not windows:
        return [], None
    Z, _ = _window_z(np.concatenate(windows), taps)
    weight = np.array([len(t) for t in taps])
    null = decoy_null(Z.ravel(), np.broadcast_to(weight[None, :, None], Z.shape).ravel())
    found, row = [], 0
    for gi, n_off in owner:
        Zg = Z[row:row + n_off]
        row += n_off
        score = np.abs(Zg).reshape(n_off, -1).max(axis=1)
        P = geos[gi]["offsets"]
        if n_off > P:
            alias = np.arange(P, n_off)
            up = alias[score[alias] > score[alias - P]]
            s1, Z1 = score[:P].copy(), Zg[:P].copy()
            s1[up - P], Z1[up - P] = score[up], Zg[up]
            score, Zg = s1, Z1
        top = np.argsort(-score, kind="stable")[:SCREEN_KEEP]
        best = float(score[top[0]])
        keep = {int(o): float(score[o]) for o in top
                if score[o] >= max(SCAN_Z, best - SCREEN_TIE)}
        if keep:
            # on a short window a block start's row-off neighbours (offsets
            # +-1, +-2, ...) read within noise of it and can crowd it out:
            # the best offset's neighbours go along, ranked with it, and the
            # selection compares them on shared bits
            for o in (top[0] + np.arange(-SCREEN_NEAR, SCREEN_NEAR + 1)) % P:
                keep[int(o)] = best
        found += [(s, gi, o, Zg[o]) for o, s in keep.items()]
    found.sort(key=lambda f: -f[0])
    return found, null


# ------------------------------------------------------- code probes ---
def _probe_conv(t: np.ndarray, bits: np.ndarray, entries) -> tuple:
    """probe_entries for convolutional entries, keeping what it summarises
    away: each result's raw z, and the z and weight of every check at every
    phase (this stream's decoys)."""
    st = probe.stream_stats(t, bits)
    out, zs, ws = {}, [], []
    by_period = {}
    for e in entries:
        by_period.setdefault(e["period"], []).append(e)
    for period, group in by_period.items():
        taps, owner = [], []
        for gi, e in enumerate(group):
            for c in e["checks"]:
                taps.append(c["taps"])
                owner.append((gi, "main"))
            if e["odd_check"] is not None:
                taps.append(e["odd_check"]["taps"])
                owner.append((gi, "odd"))
        res = probe.probe_checks(t, bits, taps, period, _st=st)
        zs.append(res["z"].ravel())
        ws.append(res["weight"].ravel())
        for gi, e in enumerate(group):
            rows = [i for i, o in enumerate(owner) if o == (gi, "main")]
            odd = [i for i, o in enumerate(owner) if o == (gi, "odd")]
            r = probe.summarize(res, rows, odd[0] if odd else None)
            r["z_raw"] = float(res["z"][rows[r["check"]], r["phase"]])
            # a code satisfies all its checks, an accident of the data
            # rarely more than the one it happens to fit.  Under a
            # substream inversion each check's syndrome sign is (-1)^(its
            # taps on the flipped bits), so the checks hold with the signs
            # of one pattern; full-stream z (an SPRT frozen on a lead-in
            # says nothing about the rest)
            ph = r["phase"]
            on = np.array([[np.count_nonzero((ph + taps[i]) % 2 == k)
                            for k in (0, 1)] for i in rows])
            signs = (-1.0) ** (on @ np.array(_INVERSIONS).T)    # (checks, patterns)
            z = res["z"][rows, ph][:, None]
            r["all_hold"] = bool((signs * z >= Z_CONFIRM).all(axis=0).any())
            out[e["name"]] = r
    return out, np.concatenate(zs), np.concatenate(ws)


def _window_decoys(x: np.ndarray) -> tuple:
    """One de-interleaved window against every convolutional registry code
    (rate 1/3 and punctured included: periodic data biases checks of its
    own period): the hard z of each check, main and odd-weight, at every
    phase.  Returns their decoy_null and each code's best |z|."""
    packs = _packs(x[None, :])
    zs, ws, best = [], [], {}
    for e in conv_entries():
        for c in e["checks"] + ([e["odd_check"]] if e["odd_check"] else []):
            n_bytes = (len(x) - int(c["taps"][-1])) // 8
            par = np.unpackbits(_parity(packs, c["taps"], n_bytes)[0])
            par = par[:len(par) // e["period"] * e["period"]].reshape(-1, e["period"])
            z = (1.0 - 2.0 * par.mean(axis=0)) * np.sqrt(len(par))
            zs.append(z)
            ws.append(np.full(e["period"], c["weight"]))
            if c is not e["odd_check"]:
                best[e["name"]] = max(best.get(e["name"], 0.0), float(np.abs(z).max()))
    return decoy_null(np.concatenate(zs), np.concatenate(ws)), best


def _reliable(bits: np.ndarray, llrs: np.ndarray):
    """The LLRs (NaN -> 0, clipped), or None when they are unusable to the
    probe (probe.usable_llrs) or their soft values carry no reliability
    where it normalises (mean |t| off idle runs below SOFT_MIN)."""
    llrs = probe.usable_llrs(llrs, len(bits))
    if llrs is None:
        return None
    t = probe.soft_values(bits, llrs)
    rest = ~_idle_positions(bits, t)
    a = float(np.abs(t[rest]).mean()) if rest.any() else 1.0
    return llrs if a >= SOFT_MIN else None


def _code_start(x: np.ndarray, taps: np.ndarray, phase: int, sign: int) -> int:
    """Bits to skip so a de-interleaved stretch starts where its code does
    (pre-lock bits, a preamble): the leading blocks of TRIM_BLOCK values
    of the check whose mean stays below half the stretch's, at most half
    the stretch; even, so the phase holds."""
    v = sign * _check_values(x, taps)[phase::2]
    nb = len(v) // TRIM_BLOCK
    if nb < 2 or not v.mean() > 0:
        return 0
    m = v[:nb * TRIM_BLOCK].reshape(nb, TRIM_BLOCK).mean(axis=1)
    low = m < v.mean() / 2
    k = nb if low.all() else int(np.argmin(low))
    return 2 * TRIM_BLOCK * min(k, nb // 2)


def _decode_explains(b: np.ndarray, llrs, hit: dict) -> bool:
    """identify_fec's decode check, held to its sub-min-rate standard for
    every claim: decode DECODE_BITS (soft when LLRs come along) under the
    inversion patterns the syndrome sign admits; a re-encode must sit
    within sampling error of the BER the checks read (_decode_agrees).  A
    check that holds by accident of the data reads a BER no decode
    reaches."""
    off = hit["offset"]
    raw = b[off:off + DECODE_BITS]
    raw = raw[:2 * (len(raw) // 2)]
    # a NaN LLR carries no reliability, as in the probe's soft_values
    rel = None if llrs is None else np.clip(np.nan_to_num(
        np.abs(llrs[off:off + len(raw)]), nan=0.0, posinf=probe.LLR_CLIP),
        0.0, probe.LLR_CLIP)
    code = ConvCode(hit["K"], (hit["g1"], hit["g2"]))
    for inv in (((1, 0), (0, 1)) if hit["syndrome_is_one"] else ((0, 0), (1, 1))):
        cand = raw.copy()
        cand[0::2] ^= inv[0]
        cand[1::2] ^= inv[1]
        dec = viterbi_decode(cand, code, soft=None if rel is None else
                             np.where(cand == 0, rel, -rel))
        re_enc = conv_encode(dec, code, terminate=False)
        m = min(len(re_enc), len(cand))
        if m and _decode_agrees(hit, float((re_enc[:m] != cand[:m]).mean()), m):
            return True
    return False


def _rank(r: dict) -> tuple:
    """Best first: an accepted code, then the normalised mean (2 dp), z."""
    return (r["early_verdict"]["verdict"] != "code", -round(r["mean"], 2),
            -r["z"])


def _shared_bases(cands: list) -> list:
    """Each candidate with its window base: its offset, or for a block
    permutation the offset nearest (mod P) its geometry's first candidate,
    so the offsets of one geometry are compared on the same stream bits
    (offsets P-1 and 0 read different blocks, P-1 and P the same ones)."""
    first, out = {}, []
    for c in cands:
        g, o = c["geometry"], c["offset"]
        if g["reach"]:              # convolutional offsets are neighbours already
            out.append({**c, "base": o})
            continue
        P = g["period"]
        a = first.setdefault(_key(g), o)
        out.append({**c, "base": a + (o - a + P // 2) % P - P // 2})
    low = {}
    for c in out:
        k = _key(c["geometry"])
        low[k] = min(low.get(k, 0), c["base"])
    for c in out:
        if low[_key(c["geometry"])] < 0:    # a base before the stream: one block on
            c["base"] += c["geometry"]["period"]
    return out


def _conv_hit(bits, llrs, cands, scan: _Scan, cfg: FECConfig, periods) -> list:
    """Code selection over the candidates (hard parities of every scan
    check on a SELECT_BITS window of each, all in one pass; best by mean,
    then z), then the winner's acceptance gates and its confirmation on
    the full de-interleaved stream."""
    N = len(bits)
    entries = _scan_entries()
    cands = [c for c in _shared_bases(cands)
             if _windowed(c["geometry"], N, c["base"]) >= SCREEN_BITS]
    if not cands:
        return []
    n = min([SELECT_BITS] + [_windowed(c["geometry"], N, c["base"]) for c in cands])
    # the offsets of one geometry move together, so they still share bits
    lo, hi = {}, {}
    for c in cands:
        k = _key(c["geometry"])
        lo[k] = min(lo.get(k, c["base"]), c["base"])
        hi[k] = max(hi.get(k, c["base"]), c["base"])
    for c in cands:
        g, k = c["geometry"], _key(c["geometry"])
        c["select"] = c["base"] + _centred(
            g, lo[k], hi[k] - lo[k] + int(_window(g, 0, n).max()) + 1, N)
    X = np.stack([bits[_window(c["geometry"], c["select"], n)] for c in cands])
    taps = [e["checks"][0]["taps"] for e in entries]
    Z, M = _window_z(X, taps)
    flat = np.abs(Z).reshape(len(cands), -1).argmax(axis=1)
    hs, phs = np.divmod(flat, 2)
    j = np.arange(len(cands))
    mean, z = np.abs(M[j, hs, phs]), np.abs(Z[j, hs, phs])
    win = min(j, key=lambda i: (-round(float(mean[i]), 2), -float(z[i]), i))
    c, h, ph = cands[win], int(hs[win]), int(phs[win])
    g, base, name = c["geometry"], c["base"], entries[h]["name"]
    wt, zw = int(scan.weight[h]), float(Z[win, h, ph])
    sign = 1 if zw >= 0 else -1
    # (ii) the decoys of the scan itself: the other codes on this window,
    # and the claimed code at every other spacing (dilation) or window
    # (screen) of the same check weight
    if z[win] < SCAN_Z or _stands_out(np.abs(Z[win]).max(axis=1), h) < DECOY_Z:
        return []
    if c["origin"] == "dilation":
        stage = scan.relative(h, c["spacing"])
    else:
        # the screened window's own phase of the check (an alias an odd
        # period away reads it at the other phase)
        zs = c["screen_z"][h, int(np.argmax(np.abs(c["screen_z"][h])))]
        stage = c["null"](zs, wt, 1 if zs >= 0 else -1)
    if stage < DECOY_Z:
        return []
    # every check of every registry code on the window: decoys, and a veto
    # when a code the scan does not report (rate 1/3, punctured) fits better
    null, codes = _window_decoys(X[win])
    if null(zw, wt, sign) < DECOY_Z or \
            max(v for k, v in codes.items() if k != name) >= z[win]:
        return []
    # a code check holds at one phase of its period; one that holds at both
    # is a recurrence of the data (an LFSR sequence, PRBS or scrambled idle,
    # decimated by D satisfies every check its polynomial divides, at every
    # bit): the other phase must read as a decoy
    if abs(null(float(Z[win, h, 1 - ph]), wt)) >= Z_CONFIRM:
        return []
    # an interleaver must make its code hold where the raw stream does not:
    # a geometry that is the identity on this stream (convolutional B=2 M=1
    # turns any rate-1/2 code into its generator-swapped twin; B*M a
    # multiple of a frame period) reads about the same z without it
    s0 = min(int(_window(g, c["select"], n).min()), N - n)
    raw_z, _ = _window_z(bits[s0:s0 + n][None, :], taps)
    if float(np.abs(raw_z).max()) >= NONE_RATIO * z[win]:
        return []
    # the confirmation stretch, from where the winning check starts to hold
    # (pre-lock bits or a preamble ahead of the code would freeze the Early
    # Verdict on "no code" and dilute the decode check)
    idx = _window(g, base, min(CONFIRM_BITS, _windowed(g, N, base)))
    idx = idx[_code_start(bits[idx], taps[h], ph, sign):]
    b = bits[idx]
    ll = None if llrs is None else _reliable(b, llrs[idx])
    res, z_all, w_all = _probe_conv(probe.soft_values(b, ll), b, entries)
    best_name, best = min(res.items(), key=lambda kv: _rank(kv[1]))
    # (i) "code" on the full stream with every check of the code (soft,
    # Early Verdict and Z_CONFIRM), and no other rate-1/2 code fits better
    if best_name != name or best["early_verdict"]["verdict"] != "code" or \
            not best["all_hold"]:
        return []
    # (ii) every other check of the de-interleaved stream is a decoy
    rel = decoy_null(z_all, w_all)(best["z_raw"], best["weight"], best["sign"])
    if rel < DECOY_Z:
        return []
    # (iii) the commit-1 identifier on a de-interleaved window, with its
    # whole gate set, names the same code, and that code explains the stream
    e, head = entries[h], b[:IDENTIFY_BITS]
    hits = identify_convolutional(
        head, tuple(cfg.conv_constraint_lengths), cfg.conv_exhaustive_max_k,
        cfg.max_test_bits, cfg.min_syndrome_zero_rate,
        llrs=None if ll is None else _reliable(head, ll[:IDENTIFY_BITS]))
    if not hits or (hits[0]["K"], hits[0]["g1"], hits[0]["g2"]) != \
            (e["K"],) + tuple(e["generators"]) or not _decode_explains(b, ll, hits[0]):
        return []
    ev = best["early_verdict"]
    kind, params = _reported(g)
    return [InterleaverHypothesis(
        kind=kind, period=_reported_period(g, periods),
        parameters={**params, "offset": int(base % g["period"]),
                    "method": "pattern scan", "code": name,
                    "early_verdict": ev, "z": round(best["z"], 2),
                    "decoy_z": round(rel, 2), "scan_decoy_z": round(stage, 2)},
        score=float(min(100.0, ev["llr"])))]


# ------------------------------------------------------ Reed-Solomon ---
def _rs_families(candidates) -> list:
    """GF(2^8) candidates grouped so each group shares RS_ROOTS roots; per
    group a table MUL[j, v] = the RS_ROOTS products v * x^-j packed in a
    uint32 (j < 255; x^255 = 1).  Built once per candidate set."""
    return _families(tuple((int(c["n"]), int(c["k"]), int(c.get("prim", 0x11D)),
                            int(c.get("fcr", 1)), int(c.get("generator", 2)))
                           for c in candidates))


@functools.lru_cache(maxsize=4)
def _families(keys: tuple) -> list:
    fams = []
    for n, k, prim, fcr, gen in keys:
        e = rs_entry({"n": n, "k": k, "prim": prim, "fcr": fcr, "generator": gen})
        if e["m"] != 8:
            continue            # ponytail: GF(2^8) only (dilated bytes)
        roots = set(int(r) for r in e["roots"])
        for f in fams:
            if f["prim"] == e["prim"] and len(f["roots"] & roots) >= RS_ROOTS:
                f["roots"] &= roots
                f["n"].add(e["n"])
                break
        else:
            fams.append({"prim": e["prim"], "roots": roots, "n": {e["n"]},
                         "exp": e["exp"], "log": e["log"]})
    for f in fams:
        j = np.arange(255)[:, None]
        v = np.arange(256)[None, :]
        mul = np.zeros((255, 256), dtype=np.uint32)
        for slot, r in enumerate(sorted(f["roots"])[:RS_ROOTS]):
            val = f["exp"][(f["log"][v] - r * j) % 255]
            val[:, 0] = 0
            mul |= val.astype(np.uint32) << np.uint32(8 * slot)
        f["mul"] = mul
        f["n"] = tuple(sorted(f["n"]))
    return fams


def _dilated_bytes(bits: np.ndarray, R: int) -> np.ndarray:
    """db[p] = the byte bits[p], bits[p+R], ..., bits[p+7R] (MSB first)."""
    n = len(bits) - 7 * R
    out = np.zeros(max(n, 0), dtype=np.uint8)
    for i in range(8):
        out |= bits[i * R:i * R + n] << np.uint8(7 - i)
    return out


def _rs_prefilter(bits, geos, fams) -> tuple:
    """Every offset of every block geometry, every codeword alignment: the
    windows whose RS_ROOTS syndromes are zero, read from one stretch in the
    middle of the stream (a lead-in or a preamble then costs nothing).
    Returns the survivors {(geometry index, offset from the stretch's
    start): count}, the windows tested, the windows passed (for the decoys'
    clean rate) and the stretch's start."""
    # ponytail: symbols are row-aligned bytes (cols a multiple of 8, the
    # block starting on a byte of the code stream, as when whole codewords
    # or bytes are interleaved); other byte phases cross row ends and need
    # 8x the work
    N = len(bits)
    n_sym = (RS_WINDOWS + 1) * max(n for f in fams for n in f["n"])
    s = np.arange(n_sym)
    plan, columns = [], 0
    for gi, g in enumerate(geos):
        R, C = g["params"]["rows"], g["params"]["cols"]
        P = R * C
        need = (-(-8 * n_sym // P) + 1) * P + 7 * R
        if C % 8 or need > N or columns + P > RS_MAX_COLUMNS:
            continue
        columns += P
        plan.append((gi, R, C, P, need))
    if not plan:
        return {}, 0, 0, 0
    span = max(p[-1] for p in plan)
    c0 = (N - span) // 2
    x = _idle_filled(bits[c0:c0 + span])
    db, chunks, cur = {}, [], []
    for gi, R, C, P, _ in plan:
        if R not in db:
            db[R] = _dilated_bytes(x, R)
        spb, spr = P // 8, C // 8
        start = (s // spb) * P + 8 * R * (s % spr) + (s % spb) // spr
        cur.append((gi, start, R, P))
        if sum(p for *_, p in cur) >= RS_CHUNK:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    survivors, tested, passed = {}, 0, 0
    view = np.lib.stride_tricks.sliding_window_view
    for chunk in chunks:
        # column = (geometry, offset), row = code symbol: every offset's
        # symbol stream side by side, one prefix-XOR pass for all
        sym = np.concatenate([view(db[R], P)[start] for _, start, R, P in chunk],
                             axis=1)
        col_geo = np.concatenate([np.full(P, gi) for gi, _, _, P in chunk])
        col_off = np.concatenate([np.arange(P) for *_, P in chunk])
        for f in fams:
            T = np.zeros((n_sym + 1, sym.shape[1]), dtype=np.uint32)
            for i in range(n_sym):
                np.bitwise_xor(T[i], f["mul"][i % 255][sym[i]], out=T[i + 1])
            for n in f["n"]:
                ss, cc = np.nonzero(T[n:] == T[:-n])
                tested += (n_sym + 1 - n) * sym.shape[1]
                if len(ss) > RS_MAX_PASSES:
                    passed += len(ss)       # codeword-like everywhere: structure
                    continue
                for a, c in zip(ss.tolist(), cc.tolist()):
                    w = sym[a:a + n, c]
                    # a repeated symbol, or a period-3 or -5 run, is a
                    # codeword of most root sets: fill, not evidence
                    if any(np.array_equal(w[d:], w[:-d]) for d in RS_PERIODS):
                        continue
                    passed += 1
                    key = (int(col_geo[c]), int(col_off[c]))
                    survivors[key] = survivors.get(key, 0) + 1
    return survivors, tested, passed, c0


def _rs_hit(bits, geos, candidates) -> list:
    """The prefilter's survivors, most windows first, each de-interleaved
    (from the prefilter's stretch on) and handed to identify_rs (its SPRT
    and refutation step); a hit must also beat the decoys' clean rate at
    the DECOY_Z level."""
    fams = _rs_families(candidates)
    if not fams:
        return []
    survivors, tested, passed, c0 = _rs_prefilter(bits, geos, fams)
    for (gi, o), count in sorted(survivors.items(),
                                 key=lambda kv: -kv[1])[:MAX_RS_CONFIRM]:
        g, base = geos[gi], c0 + o
        n = min(_windowed(g, len(bits), base), RS_CONFIRM_BITS)
        hits = identify_rs(bits[_window(g, base, n)], candidates)
        if not hits or hits[0]["early_verdict"]["verdict"] != "code":
            continue
        hit = hits[0]
        m = int(hit["n_codewords_tested"])
        k = int(round(hit["syndrome_zero_rate"] * m))
        # (ii) the decoys: every other alignment's clean rate in the
        # prefilter (four roots: at least as easy as the full root set);
        # P(k of m clean) <= C(m, k) p0^k must reach the DECOY_Z tail
        p0 = max(RS_PRE_P0, (passed - count) / max(1, tested))
        if k < 1 or math.lgamma(m + 1) - math.lgamma(k + 1) - \
                math.lgamma(m - k + 1) + k * math.log(p0) > _LOG_DECOY_P:
            continue
        ev = hit["early_verdict"]
        kind, params = _reported(g)
        return [InterleaverHypothesis(
            kind=kind, period=g["period"],
            parameters={**params, "offset": int(base % g["period"]),
                        "method": "pattern scan", "code": rs_entry(hit)["name"],
                        "early_verdict": ev,
                        "clean_codewords": k, "codewords_tested": m},
            score=float(min(100.0, ev["llr"])))]
    return []


# --------------------------------------------------------------- scan ---
def pattern_scan(bits: np.ndarray, llrs: np.ndarray = None, periods=(),
                 max_dim: int = 64, fec_config: FECConfig = None,
                 rs_ladder=RS_LADDER) -> list:
    """The interleaver geometry and the code under it, or [] when nothing
    passes the acceptance gates (see the module docstring).  `periods`: the
    rank profile's candidate periods, when it shows any; `llrs` (aligned
    with `bits`, > 0 => bit 0) feed the code probes; `max_dim` bounds rows
    and columns; `fec_config` (FECConfig's defaults when None) gives the RS
    candidates and identify_convolutional's settings; `rs_ladder` are the
    (rows, cols) tried under RS without a period.  Returns a list of
    InterleaverHypothesis with parameters method="pattern scan", code (the
    registry name), early_verdict and the offset in the engine's
    convention (bits to skip before de-interleaving)."""
    bits = np.asarray(bits, dtype=np.uint8)
    N = len(bits)
    if N < MIN_BITS:
        return []
    soft = None
    if llrs is not None and np.ndim(llrs) == 1 and len(llrs) == N:
        soft = np.asarray(llrs, dtype=np.float64)
    cfg = FECConfig() if fec_config is None else fec_config
    periods = [int(p) for p in periods]
    scan = _Scan(bits, max_dim + DILATION_EXTRA)
    if scan.plain:
        return []                       # the check holds undilated
    cands, seen, screened = [], set(), []
    for D, i in scan.spacings():
        fits, conv = _spacing_candidates(scan, D, i, max_dim)
        for _, g, starts in fits:
            for o in starts:
                if (_key(g), o) not in seen:
                    seen.add((_key(g), o))
                    cands.append({"geometry": g, "offset": o,
                                  "origin": "dilation", "spacing": D})
        screened += conv
    screened += _period_geometries(periods, max_dim)
    uniq = {}
    for g in screened:
        uniq.setdefault(_key(g), g)
    geos = list(uniq.values())
    taps = [e["checks"][0]["taps"] for e in _scan_entries()]
    # helical last, as in the rank-profile sweep: the costliest to screen
    # (every step, every offset) and the least common
    for group in ([g for g in geos if g["kind"] != "helical"],
                  [g for g in geos if g["kind"] == "helical"]):
        found, null = _screen(bits, group, taps)
        for _, gi, o, Z in found[:MAX_SCREENED]:
            if (_key(group[gi]), o) not in seen:
                seen.add((_key(group[gi]), o))
                cands.append({"geometry": group[gi], "offset": o,
                              "origin": "screen", "screen_z": Z, "null": null})
        hit = _conv_hit(bits, soft, cands, scan, cfg, periods)
        if hit:
            return hit
        cands = []
    # ponytail: LDPC under an interleaver is not searched; with registry
    # LDPC entries its probe would run here on the same block geometries
    ladder = {_key(g): g for g in
              [_geometry("block", rows=r, cols=c) for r, c in rs_ladder] +
              [g for g in geos if g["kind"] == "block"]}
    rs_geos = sorted(ladder.values(), key=lambda g: g["period"])
    return _rs_hit(bits, rs_geos, cfg.rs_candidates)
