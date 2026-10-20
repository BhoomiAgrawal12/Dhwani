"""Code Probe and Early Verdict (stage S9).

Code Probe: does this stream satisfy code X at alignment a?  Every bit
becomes a soft value t = (1-2b)|tanh(L/2)| (+-1 without LLRs).  The sign
always comes from the hard bit, so a misaligned LLR source can only cost
reliability, never flip a decision.  A parity check with taps T placed at
window start p has value s(p) = prod t[p + T]: its mean is about
(1-2 BER)^w for a true check of weight w and 0 for a false one.  Dividing
by a^w (a = mean|t|) puts hard and soft input on one scale; under "no
code" the variance per check is then (b/a^2)^w (b = mean t^2), which
gives a z-score.  The soft values are the evidence, but the BER and the
syndrome-zero rate are read from the hard parities of the same windows:
errors sit on the less confident bits, which lifts the normalised soft
mean above (1-2 BER)^w.  A negative mean is a constant-1 syndrome
(substream inversion), so both signs are tested.  Idle or padding runs
(long, confident, constant) satisfy every even-weight check whatever the
code, so windows inside them are skipped.

Many checks of one span are scored in one pass: sliding windows of the
bits (and of log|t|) times the 0/1 tap matrix give every check's parity
(and log-magnitude) at every position, one matrix product per chunk.  A
generator sweep is cheaper still: each rate-1/2 check is (even taps) x
(odd taps), so all pairs come out of one small matrix product per block.

Early Verdict: a sequential probability ratio test on those values, fed
in blocks of 64.  H1 = "code, mean m1" with m1 = (1-2 EXPECTED_BER)^w
(floored at M1_MIN), H0 = "no code, mean 0"; accept at log((1-beta)/alpha),
reject at log(beta/(1-alpha)).  Once every hypothesis of a sweep has
settled the test costs nothing more, so a thousand codes cost a few
thousand checks each; the means and z still cover the whole stream, and
a verdict the whole stream contradicts (a "code" whose z fell below
Z_CONFIRM, a "no code" whose z rose above it) is reported undecided.

H0 is independent data.  Uncoded data is not: text, repeated frames and
fixed headers give wrong checks small persistent biases, and a long
stream turns any bias into a large z.  A sweep is its own control:
decoy_null measures, per check weight, how far the z of its hypotheses
(nearly all of them wrong) spread on this very stream, and a candidate
has to stand out from that, not just from N(0, 1).

Reed-Solomon: syndromes live in GF(2^m), so the per-alignment values are
clean-codeword indicators, judged by the same test with a Bernoulli
likelihood.  For root x = alpha^r the syndrome of the n symbols starting
at s is S(s) = x^(n-1+s) (T(s+n) xor T(s)) with the prefix XOR
T(j) = xor_{i<j} c_i x^-i, so S(s) = 0 <=> T(s+n) = T(s): one prefix-XOR
pass per root and bit offset covers every symbol alignment, shortened
codes included (no x^n = 1 needed).
"""
from __future__ import annotations

import numpy as np

from .rs import bits_to_symbols

ALPHA = 1e-6          # false alarm per hypothesis and sign: a 2000-hypothesis sweep stays < 1%
BETA = 1e-3           # missed code per hypothesis
BLOCK = 64            # checks per SPRT step
EXPECTED_BER = 0.1    # H1 is planned at this channel BER; cleaner streams decide sooner
M1_MIN = 0.05         # keeps "no code" decidable within ~6k checks for heavy checks
RELIABLE = 0.9        # |t| at or above this is a confident bit (idle mask)
IDLE_RUN = 32         # constant runs shorter than this occur by chance (2^-31 per bit)
# full-stream z a verdict has to keep: an SPRT acceptance always sits at
# z >= sqrt(2 log((1-BETA)/ALPHA)) = 5.26, and 5 sigma keeps a sweep of
# ~5000 columns near a 1e-3 false alarm (Bonferroni)
Z_CONFIRM = 5.0
DECOY_MIN = 30        # decoys per weight class for a stable median and MAD
# how far a candidate must stand above the decoys of its weight: z >= 7 on
# independent data; over 122 structured uncoded streams (beacons, mostly
# repeated frames, text and header frames) the worst wrong check sat at
# 5.95 decoy spreads, a true K=7 code at 10% BER over 6000 bits at 7.6
DECOY_Z = 7.0
RS_EXPECTED_BER = 5e-4  # planned bit error rate inside RS codewords (p1 ~ 0.36, n = 255)
RS_P0_FLOOR = 1e-12   # floor of the clean-window chance under "no code"
LLR_CLIP = 30.0       # tanh(15) is 1 to float64 precision anyway
SOFT_MIN = 1e-3       # mean |t| below this: the LLRs carry no usable reliability
SATURATED = 1.0 - 1e-9  # |t| above this everywhere: hard decisions, parity-only path
MIN_CHECKS = 64       # fewer checks than this say nothing, even undecided
# window positions per stacked pass: small first, so the many checks that
# settle within a few hundred values stop costing SPRT work early
_CHUNKS = (256, 512, 1024, 2048, 4096, 8192, 16384)
_TINY = 1e-12         # |t| floor so log|t| stays finite in the matrix product
_RANK = 1e12          # ranking keys: a verdict outranks any z
_VERDICT = {1: "code", -1: "no code", 0: "undecided"}


def usable_llrs(llrs: np.ndarray, n: int):
    """The LLRs as float64 (NaN -> 0, clipped to +-LLR_CLIP), or None when
    they are missing, belong to a stream of another length, or carry no
    reliability anywhere (all zero, all NaN, scaled to nothing: they tell no
    more than the bits do, and the a^w normalisation would blow up)."""
    if llrs is None or len(llrs) != n:
        return None
    L = np.nan_to_num(np.asarray(llrs, dtype=np.float64), nan=0.0,
                      posinf=LLR_CLIP, neginf=-LLR_CLIP)
    L = np.clip(L, -LLR_CLIP, LLR_CLIP)
    if not np.abs(np.tanh(L / 2.0)).mean() >= SOFT_MIN:
        return None
    return L


def soft_values(bits: np.ndarray, llrs: np.ndarray = None) -> np.ndarray:
    """t = (1-2b)|tanh(L/2)|, or +-1 when there are no usable (aligned)
    LLRs (usable_llrs); a NaN LLR carries no reliability (t = 0)."""
    sign = 1.0 - 2.0 * (np.asarray(bits) != 0)
    L = usable_llrs(llrs, len(sign))
    if L is None:
        return sign
    return sign * np.abs(np.tanh(L / 2.0))


# ---------------------------------------------------------------- SPRT ---
def _bounds(alpha: float, beta: float) -> tuple:
    return float(np.log((1 - beta) / alpha)), float(np.log(beta / (1 - alpha)))


def planned_m1(weight, expected_ber: float = EXPECTED_BER):
    """The H1 mean a check of this weight is planned at."""
    return np.maximum((1.0 - 2.0 * expected_ber) ** np.asarray(weight, float),
                      M1_MIN)


class _Walk:
    """SPRT state of many columns (hypothesis x sign).  Each block moves a
    column's LLR by c1*sum + c0*count; a column freezes at its first
    threshold crossing (state 1 = code, -1 = no code, 0 = still open)."""

    def __init__(self, c1, c0, alpha: float, beta: float):
        self.c1 = np.asarray(c1, dtype=np.float64)
        self.c0 = np.asarray(c0, dtype=np.float64)
        self.upper, self.lower = _bounds(alpha, beta)
        self.llr = np.zeros(len(self.c1))
        self.used = np.zeros(len(self.c1), dtype=np.int64)
        self.state = np.zeros(len(self.c1), dtype=np.int8)

    def feed(self, sums: np.ndarray, counts: np.ndarray, cols: np.ndarray):
        """sums: (B, len(cols)) block sums; counts: (B, 1) or (B, len(cols))."""
        open_ = self.state[cols] == 0
        if not open_.any() or not len(sums):
            return
        cols = cols[open_]
        s = sums[:, open_]
        n = np.broadcast_to(counts, sums.shape)[:, open_]
        path = self.llr[cols] + np.cumsum(s * self.c1[cols] + n * self.c0[cols],
                                          axis=0)
        used = self.used[cols] + np.cumsum(n, axis=0)
        up, lo = path >= self.upper, path <= self.lower
        hit = up | lo
        crossed = hit.any(axis=0)
        first = np.where(crossed, hit.argmax(axis=0), len(path) - 1)
        k = np.arange(len(cols))
        self.llr[cols] = path[first, k]
        self.used[cols] = np.rint(used[first, k]).astype(np.int64)
        self.state[cols] = np.where(crossed, np.where(up[first, k], 1, -1), 0)


def _gauss_ber(mean: float, weight: float) -> float:
    """Invert mean = (1-2p)^w; a non-positive mean says nothing (0.5)."""
    if not mean > 0:
        return 0.5
    return float((1.0 - min(mean, 1.0) ** (1.0 / weight)) / 2.0)


def ber_spread(mean: float, weight: float, n: float) -> float:
    """Standard error of _gauss_ber(mean, weight) when mean is a hard +-1
    mean over n checks (delta method: dp/dm = m^(1/w - 1) / 2w, var m =
    (1 - m^2)/n); 0.5 when the mean says nothing."""
    if not (mean > 0 and n > 0):
        return 0.5
    m = min(float(mean), 1.0)
    return float(m ** (1.0 / weight - 1.0) / (2.0 * weight)
                 * np.sqrt((1.0 - m * m) / n))


def _agree(state: np.ndarray, z: np.ndarray,
           z_min: float = Z_CONFIRM) -> np.ndarray:
    """SPRT verdicts checked against the full-stream z of their own sign: a
    "code" whose z fell below z_min, or a "no code" whose z reached it, is
    undecided (the test froze on the stream's start, the rest disagrees)."""
    state = np.array(state, dtype=np.int8)
    state[(state == 1) & (z < z_min)] = 0
    state[(state == -1) & (z >= z_min)] = 0
    return state


def _verdict(state, llr, used, alpha, ber) -> dict:
    return {"verdict": _VERDICT[int(state)], "llr": float(llr),
            "checks_used": int(used), "false_alarm": float(alpha),
            "ber_estimate": None if ber is None else float(ber)}


def early_verdicts(values, weight=1, var0=1.0, expected_ber: float = EXPECTED_BER,
                   m1=None, alpha: float = ALPHA, beta: float = BETA,
                   block: int = BLOCK, model: str = "gauss", p0: float = None,
                   p1: float = None, ber_model=None) -> list:
    """early_verdict over the columns of a (n, H) matrix; weight, var0 and
    m1 may be per-column arrays."""
    v = np.asarray(values, dtype=np.float64)
    v = v[:, None] if v.ndim == 1 else v
    n, H = v.shape
    w = np.broadcast_to(np.asarray(weight, dtype=np.float64), (H,))
    if model == "gauss":
        m = planned_m1(w, expected_ber) if m1 is None else m1
        m = np.broadcast_to(np.asarray(m, dtype=np.float64), (H,))
        var = np.broadcast_to(np.asarray(var0, dtype=np.float64), (H,))
        c1, c0 = m / var, -m * m / (2.0 * var)
    elif model == "bernoulli":
        c0 = np.full(H, np.log((1 - p1) / (1 - p0)))
        c1 = np.full(H, np.log(p1 / p0)) - c0
    else:
        raise ValueError(f"unknown model {model!r}")
    walk = _Walk(c1, c0, alpha, beta)
    if n:
        B = -(-n // block)
        pad = np.zeros((B * block - n, H))
        sums = np.concatenate([v, pad]).reshape(B, block, H).sum(axis=1)
        counts = np.minimum(block, n - block * np.arange(B))[:, None]
        walk.feed(sums, counts.astype(np.float64), np.arange(H))
    mean = v.mean(axis=0) if n else np.zeros(H)
    state = walk.state
    if model == "gauss" and n:
        state = _agree(state, mean * np.sqrt(n / var))
    out = []
    for h in range(H):
        if model == "gauss":
            ber = _gauss_ber(mean[h], w[h])
        else:
            ber = None if ber_model is None else ber_model(float(mean[h]))
        out.append(_verdict(state[h], walk.llr[h], walk.used[h], alpha, ber))
    return out


def early_verdict(values, weight=1, var0=1.0, expected_ber: float = EXPECTED_BER,
                  m1=None, alpha: float = ALPHA, beta: float = BETA,
                  block: int = BLOCK, model: str = "gauss", p0: float = None,
                  p1: float = None, ber_model=None) -> dict:
    """SPRT over per-check values (gauss: normalised products s/a^w;
    bernoulli: 0/1 clean-codeword indicators).  Returns {"verdict": "code" |
    "no code" | "undecided", "llr", "checks_used", "false_alarm",
    "ber_estimate"}; the BER comes from all values given, and a gauss
    verdict that all values together contradict (z = mean sqrt(n/var0)
    against Z_CONFIRM) is undecided."""
    return early_verdicts(np.asarray(values, dtype=np.float64).ravel(), weight,
                          var0, expected_ber, m1, alpha, beta, block, model,
                          p0, p1, ber_model)[0]


# ----------------------------------------------------------- the probe ---
def _idle_positions(bits: np.ndarray, t: np.ndarray = None) -> np.ndarray:
    """Bits inside a constant run of at least IDLE_RUN bits (and confident,
    when soft values are given): idle or padding, not data."""
    N = len(bits)
    if not N:
        return np.zeros(0, dtype=bool)
    starts = np.flatnonzero(np.concatenate([[True], bits[1:] != bits[:-1]]))
    lengths = np.diff(np.concatenate([starts, [N]]))
    idle = np.repeat(lengths, lengths) >= IDLE_RUN
    if t is not None:
        idle &= np.abs(t) >= RELIABLE
    return idle


def stream_stats(t: np.ndarray, bits: np.ndarray) -> "_Stream":
    """Precompute what every probe call on this stream shares (masks,
    normalisers); pass it as _st= to probe_checks / probe_pairs."""
    return _Stream(bits, t)


class _Stream:
    """Per-stream quantities every check shares: the idle positions, the
    normalisers a = mean|t| and b = mean t^2 over the rest, log|t|."""

    def __init__(self, bits: np.ndarray, t: np.ndarray):
        self.bits = np.asarray(bits, dtype=np.uint8)
        self.t = np.asarray(t, dtype=np.float64)
        self.N = len(self.bits)
        # saturated LLRs (high SNR) are hard decisions: take the cheaper
        # parity-only path
        self.hard = bool(np.all(np.abs(self.t) > SATURATED))
        self.b32 = self.bits.astype(np.float32)
        self.idle = _idle_positions(self.bits, self.t)
        self.cum_idle = np.concatenate([[0], np.cumsum(self.idle)])
        rest = ~self.idle
        self.a = float(np.abs(self.t[rest]).mean()) if rest.any() else 1.0
        self.b = float((self.t[rest] ** 2).mean()) if rest.any() else 1.0
        # log|t| sums in float32 are exact to ~1e-6: plenty for a mean
        self.l32 = None if self.hard else \
            np.log(np.maximum(np.abs(self.t), _TINY)).astype(np.float32)

    def informative(self, span: int) -> np.ndarray:
        """Window starts 0..N-span not wholly inside idle runs."""
        n_win = self.N - span + 1
        if n_win <= 0:
            return np.zeros(0, dtype=bool)
        return (self.cum_idle[span:span + n_win] - self.cum_idle[:n_win]) < span


def _block_sums(x: np.ndarray, block: int) -> np.ndarray:
    """Column sums over consecutive blocks of rows (the last may be short),
    in float64."""
    full = (len(x) // block) * block
    out = x[:full].reshape(-1, block, x.shape[1]).sum(axis=1, dtype=np.float64)
    if full < len(x):
        out = np.concatenate([out, x[full:].sum(axis=0, keepdims=True,
                                                dtype=np.float64)])
    return out


def _score_stack(st: _Stream, M: np.ndarray, w: np.ndarray, period: int,
                 phase: int, informative: np.ndarray, m1: np.ndarray,
                 alpha: float, beta: float, block: int) -> dict:
    """Checks of one span (rows of the 0/1 tap matrix M) at one phase.  The
    SPRT stops moving once every column has settled; the means (soft and
    hard) cover every window."""
    H, span = M.shape
    n_win = max(0, (st.N - span - phase) // period + 1)
    var0 = (st.b / st.a ** 2) ** w
    scale = st.a ** -w.astype(np.float64)
    walk = _Walk(np.tile(m1 / var0, 2), np.tile(-m1 * m1 / (2 * var0), 2),
                 alpha, beta)
    total, hard_total, count = np.zeros(H), np.zeros(H), 0.0
    view = np.lib.stride_tricks.sliding_window_view
    view_b = view(st.b32, span) if n_win else None
    view_l = None if (st.hard or not n_win) else view(st.l32, span)
    keep = informative[phase::period][:n_win]
    Mt = M.astype(np.float32).T
    cols = np.arange(2 * H)
    k0, step = 0, 0
    while k0 < n_win:
        k1 = min(n_win, k0 + _CHUNKS[min(step, len(_CHUNKS) - 1)])
        step += 1
        rows = slice(phase + period * k0, phase + period * (k1 - 1) + 1, period)
        # parity of every check at every window (exact: weights < 128)
        odd = (view_b[rows] @ Mt).astype(np.int8)
        odd &= 1
        m = keep[k0:k1]
        if not m.all():
            odd[~m] = 0
        counts = _block_sums(m[:, None], block)
        # +-1 values: a block sums to (its count) - 2 (its odd parities)
        hard = counts - 2.0 * _block_sums(odd, block)
        if view_l is None:
            sums = hard * scale
        else:
            s = view_l[rows] @ Mt                    # log|prod t| per window
            np.exp(s, out=s)
            np.negative(s, out=s, where=odd.view(bool))
            if not m.all():
                s[~m] = 0.0
            sums = _block_sums(s, block) * scale
        total += sums.sum(axis=0)
        hard_total += hard.sum(axis=0)
        count += counts.sum()
        walk.feed(np.concatenate([sums, -sums], axis=1), counts, cols)
        k0 = k1
    mean = total / max(count, 1.0)
    z = mean * np.sqrt(count / var0)
    state = _agree(walk.state.reshape(2, H).T, np.stack([z, -z], axis=1))
    return {"mean": mean, "hard_mean": hard_total / max(count, 1.0),
            "n": np.full(H, count), "z": z, "var0": var0,
            "llr": walk.llr.reshape(2, H).T, "used": walk.used.reshape(2, H).T,
            "state": state}


def probe_checks(t: np.ndarray, bits: np.ndarray, checks: list, period: int,
                 phases=None, expected_ber: float = EXPECTED_BER,
                 alpha: float = ALPHA, beta: float = BETA,
                 block: int = BLOCK, _st: "_Stream" = None) -> dict:
    """Score checks (tap arrays, all repeating every `period` bits) at every
    phase and both signs.  Checks of equal span share one pass.

    Returns arrays: mean, hard_mean, n, z, var0, weight shaped (checks,
    phases); llr, used, state shaped (checks, phases, 2), sign index 0 =
    syndrome 0, 1 = constant-1 syndrome; state 1 = code, -1 = no code,
    0 = undecided (the SPRT's verdict, kept where the full-stream z agrees).
    `mean` (normalised soft values), `hard_mean` (+-1 parities) and `z`
    cover the whole stream, for sign +1 (negate them for the other)."""
    st = _Stream(bits, t) if _st is None else _st
    phases = list(range(period)) if phases is None else list(phases)
    taps = [np.asarray(c, dtype=np.int64) for c in checks]
    spans = np.array([int(tp[-1]) + 1 for tp in taps], dtype=np.int64)
    weights = np.array([len(tp) for tp in taps], dtype=np.int64)
    C, P = len(taps), len(phases)
    out = {"mean": np.zeros((C, P)), "hard_mean": np.zeros((C, P)),
           "n": np.zeros((C, P)), "z": np.zeros((C, P)),
           "var0": np.zeros((C, P)), "weight": np.repeat(weights[:, None], P, 1),
           "llr": np.zeros((C, P, 2)), "used": np.zeros((C, P, 2), np.int64),
           "state": np.zeros((C, P, 2), np.int8)}
    m1_all = planned_m1(weights, expected_ber)
    for span in np.unique(spans):
        idx = np.flatnonzero(spans == span)
        M = np.zeros((len(idx), int(span)))
        for r, i in enumerate(idx):
            M[r, taps[i]] = 1.0
        informative = st.informative(int(span))
        for pi, ph in enumerate(phases):
            res = _score_stack(st, M, weights[idx], period, ph, informative,
                               m1_all[idx], alpha, beta, block)
            for key in ("mean", "hard_mean", "n", "z", "var0", "llr", "used",
                        "state"):
                out[key][idx, pi] = res[key]
    return out


def _part_products(st: _Stream, parts: list, span: int, period: int,
                   phase: int, n_win: int) -> tuple:
    """(values, parities): (n_win, len(parts)) float32 signed products
    prod t[p + taps] / a^w of each tap set at the windows p = phase +
    k*period, and the same products of the hard +-1 bits."""
    M = np.zeros((len(parts), span), dtype=np.float32)
    for r, tp in enumerate(parts):
        M[r, tp] = 1.0
    rows = slice(phase, phase + period * (n_win - 1) + 1, period)
    view = np.lib.stride_tricks.sliding_window_view
    odd = (view(st.b32, span)[rows] @ M.T).astype(np.int8)
    odd &= 1
    signs = (1 - 2 * odd).astype(np.float32)
    if st.hard:
        out = signs.copy()
    else:
        out = view(st.l32, span)[rows] @ M.T
        np.exp(out, out=out)
        np.negative(out, out=out, where=odd.view(bool))
    out *= (st.a ** -M.sum(axis=1).astype(np.float64)).astype(np.float32)
    return out, signs


def probe_pairs(t: np.ndarray, bits: np.ndarray, first: list, second: list,
                span: int, period: int = 2, phases=None,
                expected_ber: float = EXPECTED_BER, alpha: float = ALPHA,
                beta: float = BETA, block: int = BLOCK,
                _st: "_Stream" = None) -> dict:
    """Every check made of one tap set from `first` and one from `second`
    (disjoint positions inside one window of `span`), all pairs at once.
    A check's value is the product of its two parts' values, so per block
    of windows the block sums of all |first| x |second| checks are one small
    matrix product (parts_1^T @ parts_2): a rate-1/2 generator sweep costs
    two passes of 2^(K-2) columns instead of 4^(K-2).  Blocks are fed to
    the Early Verdict until every pair has settled; one product over the
    remaining windows then completes the means (soft and hard).  Returns
    probe_checks' arrays, one row per pair: row a * len(second) + b."""
    st = _Stream(bits, t) if _st is None else _st
    phases = list(range(period)) if phases is None else list(phases)
    wa = np.array([len(tp) for tp in first])
    wb = np.array([len(tp) for tp in second])
    w = (wa[:, None] + wb[None, :]).ravel()              # (A*B,) pair weights
    A, Bn, P = len(first), len(second), len(phases)
    var0 = (st.b / st.a ** 2) ** w
    m1 = planned_m1(w, expected_ber)
    informative = st.informative(span)
    out = {k: np.zeros((A * Bn, P)) for k in ("mean", "hard_mean", "n", "z")}
    out["var0"] = np.repeat(var0[:, None], P, 1)
    out["weight"] = np.repeat(w[:, None], P, 1)
    for k, dt in (("llr", np.float64), ("used", np.int64), ("state", np.int8)):
        out[k] = np.zeros((A * Bn, P, 2), dtype=dt)
    cols = np.arange(2 * A * Bn)
    for pi, ph in enumerate(phases):
        n_win = max(0, (st.N - span - ph) // period + 1)
        walk = _Walk(np.tile(m1 / var0, 2), np.tile(-m1 * m1 / (2 * var0), 2),
                     alpha, beta)
        if n_win:
            keep = informative[ph::period][:n_win]
            pa, ha = _part_products(st, first, span, period, ph, n_win)
            pb, hb = _part_products(st, second, span, period, ph, n_win)
            pa[~keep] = 0.0
            ha[~keep] = 0.0
            total = np.zeros(A * Bn)
            r0, step = 0, 0
            while r0 < n_win and (walk.state == 0).any():
                r1 = min(n_win, r0 + _CHUNKS[min(step, len(_CHUNKS) - 1)])
                step += 1
                full = r0 + ((r1 - r0) // block) * block
                sums = np.matmul(
                    pa[r0:full].reshape(-1, block, A).transpose(0, 2, 1),
                    pb[r0:full].reshape(-1, block, Bn)).reshape(-1, A * Bn)
                if full < r1:            # the stream's last, short block
                    sums = np.concatenate([sums, (pa[full:r1].T @ pb[full:r1])
                                           .reshape(1, -1)])
                sums = sums.astype(np.float64)
                walk.feed(np.concatenate([sums, -sums], axis=1),
                          _block_sums(keep[r0:r1, None], block), cols)
                total += sums.sum(axis=0)
                r0 = r1
            if r0 < n_win:               # all settled: the rest only feeds means
                total += (pa[r0:].T.astype(np.float64)
                          @ pb[r0:].astype(np.float64)).ravel()
            # +-1 products sum to integers < 2^24: exact in float32
            hard = total * st.a ** w if st.hard else \
                (ha.T @ hb).ravel().astype(np.float64)
            n = float(keep.sum())
            mean = total / max(n, 1.0)
            out["mean"][:, pi] = mean
            out["hard_mean"][:, pi] = hard / max(n, 1.0)
            out["n"][:, pi] = n
            out["z"][:, pi] = mean * np.sqrt(n) / np.sqrt(var0)
        out["llr"][:, pi] = walk.llr.reshape(2, -1).T
        out["used"][:, pi] = walk.used.reshape(2, -1).T
        z = out["z"][:, pi]
        out["state"][:, pi] = _agree(walk.state.reshape(2, -1).T,
                                     np.stack([z, -z], axis=1))
    return out


def _keys(res: dict, rows) -> tuple:
    """Per (row, phase): the sign to report (an accepted one, else the
    mean's) and the ranking key (state of that sign, then signed z)."""
    st, mean, z = res["state"][rows], res["mean"][rows], res["z"][rows]
    si = np.where((st[..., 0] == 1) | ((st[..., 1] != 1) & (mean >= 0)), 0, 1)
    state = np.where(si == 0, st[..., 0], st[..., 1])
    return state * _RANK + (1 - 2 * si) * z, si, state


def row_best(res: dict) -> tuple:
    """For every check row of a probe_checks result: (key, phase, sign
    index, state) of its best phase; a hypothesis made of several rows
    takes its row with the largest key (as summarize does)."""
    key, si, state = _keys(res, slice(None))
    ph = key.argmax(axis=1)
    r = np.arange(len(key))
    return key[r, ph], ph, si[r, ph], state[r, ph]


def _pick(res: dict, rows: list) -> tuple:
    """Best (row, phase, sign index) among `rows`: an accepted code first,
    then undecided, then by signed z (first of equals)."""
    rows = np.asarray(rows, dtype=np.intp)
    key, si, _ = _keys(res, rows)
    r, p = divmod(int(np.argmax(key)), key.shape[1])
    return int(rows[r]), p, int(si[r, p])


def summarize(res: dict, rows, odd_row=None, alpha: float = ALPHA) -> dict:
    """One hypothesis (its checks = `rows` of a probe_checks result): the
    best (check, phase, sign) with its Early Verdict, mean and hard_mean
    (for that sign), z, n, the inversion flag its odd-weight check reveals
    (None without one, or while it is undecided), and from the hard
    parities the syndrome_zero_rate = (1 + hard_mean)/2 and the verdict's
    BER (unbiased for soft input too)."""
    rows = [int(r) for r in rows]
    r, p, si = _pick(res, rows)
    sign = 1 - 2 * si
    w = int(res["weight"][r, p])
    mean = float(sign * res["mean"][r, p])
    hard = float(sign * res["hard_mean"][r, p])
    inverted = None
    if odd_row is not None:
        ost = res["state"][odd_row, p]
        inverted = False if ost[0] == 1 else (True if ost[1] == 1 else None)
    ev = _verdict(res["state"][r, p, si], res["llr"][r, p, si],
                  res["used"][r, p, si], alpha, _gauss_ber(hard, w))
    return {"check": rows.index(r), "phase": int(p), "sign": sign,
            "weight": w, "mean": mean, "hard_mean": hard,
            "z": float(sign * res["z"][r, p]), "n": int(res["n"][r, p]),
            "syndrome_zero_rate": (1.0 + float(np.clip(hard, -1, 1))) / 2.0,
            "inverted": inverted, "early_verdict": ev}


def decoy_null(z: np.ndarray, weight: np.ndarray):
    """The empirical null of a sweep: from the z of all its checks (every
    hypothesis and phase, nearly all of them wrong), per check weight the
    median and the spread 1.4826 MAD (never below 1, the independent-data
    value).  Structure in uncoded data moves wrong checks by weight (on
    unbalanced bits the bias falls as E[t]^w), so each weight is compared
    with its own decoys; a weight with fewer than DECOY_MIN borrows the
    nearest ones.  Returns relative(z, weight, sign): sign*z in units of
    that null, sign*(z - median)/spread."""
    z = np.asarray(z, dtype=np.float64).ravel()
    w = np.asarray(weight, dtype=np.int64).ravel()
    cache = {}

    def relative(zc: float, wc: int, sign: int = 1) -> float:
        wc = int(wc)
        if wc not in cache:
            if not len(z):
                cache[wc] = (0.0, 1.0)
            else:
                d = np.abs(w - wc)
                sel = d <= np.sort(d)[min(DECOY_MIN, len(d)) - 1]
                c = float(np.median(z[sel]))
                mad = float(np.median(np.abs(z[sel] - c)))
                cache[wc] = (c, max(1.0, 1.4826 * mad))
        c, s = cache[wc]
        return float(sign * (zc - c) / s)
    return relative


def _conv_result(entry: dict, res: dict, rows: list, odd_row, alpha: float) -> dict:
    return {"entry": entry["name"], **summarize(res, rows, odd_row, alpha)}


def probe_entries(t: np.ndarray, bits: np.ndarray, entries: list,
                  alpha: float = ALPHA, beta: float = BETA, **kw) -> dict:
    """Best (check, phase, sign) per registry entry, with its Early Verdict.
    Convolutional entries sharing a period are stacked into one pass; RS
    entries share one prefix-XOR scan.  Returns {name: result}."""
    # ponytail: each entry is judged against independent data only; on
    # structured uncoded data (frame headers, text) an entry can read "code"
    # - weigh it against decoys (decoy_null) before acting on it, as
    # identify_convolutional does with its sweep
    out = {}
    by_period = {}
    for e in entries:
        if e["family"] == "convolutional":
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
        res = probe_checks(t, bits, taps, period, alpha=alpha, beta=beta, **kw)
        for gi, e in enumerate(group):
            rows = [i for i, o in enumerate(owner) if o == (gi, "main")]
            odd = [i for i, o in enumerate(owner) if o == (gi, "odd")]
            out[e["name"]] = _conv_result(e, res, rows, odd[0] if odd else None,
                                          alpha)
    rs = [e for e in entries if e["family"] == "reed_solomon"]
    for e, r in zip(rs, probe_rs_many(bits, rs, alpha=alpha, beta=beta)):
        out[e["name"]] = r
    return {e["name"]: out[e["name"]] for e in entries if e["name"] in out}


def probe_entry(t: np.ndarray, bits: np.ndarray, entry: dict, **kw) -> dict:
    """probe_entries for one entry."""
    return probe_entries(t, bits, [entry], **kw)[entry["name"]]


# ------------------------------------------------------- Reed-Solomon ---
def _rs_symbols(bits: np.ndarray, m: int, n_sym: int) -> np.ndarray:
    """(m, n_sym) symbols at every bit offset, zero past the stream's end."""
    sym = np.zeros((m, n_sym), dtype=np.int64)
    for off in range(m):
        s = bits_to_symbols(bits[off:off + n_sym * m], m)
        sym[off, :len(s)] = s
    return sym


def _rs_prefix_tables(sym: np.ndarray, m: int, exp: np.ndarray,
                      log: np.ndarray, roots: np.ndarray) -> np.ndarray:
    """T[off, root, j] = xor_{i<j} c_i x^-i for every bit offset and root."""
    order = (1 << m) - 1
    i = np.arange(sym.shape[1])
    small = np.uint8 if m <= 8 else np.uint16
    exp_s = exp.astype(small)
    T = np.zeros((len(sym), len(roots), sym.shape[1] + 1), dtype=small)
    for off, s in enumerate(sym):
        e = (log[s][None, :] - roots[:, None] * i[None, :]) % order
        vals = np.where(s[None, :] != 0, exp_s[e], small(0))
        T[off, :, 1:] = np.bitwise_xor.accumulate(vals, axis=1)
    return T


def rs_zero_indicators(bits: np.ndarray, entry: dict, _T=None) -> tuple:
    """(clean, informative), bool arrays (bit offset, symbol start): clean =
    every root's syndrome is zero for the n symbols starting there;
    informative = the n symbols lie inside the stream at that bit offset,
    not every bit of them is idle (idle satisfies RS codes: the all-zero
    word always, all-ones and run boundaries for many root sets), and they
    are not one symbol repeated (n equal symbols s give c(x) = s(x^n-1)/
    (x-1), zero at every root x != 1 of a full-length code: preamble or
    flag fill)."""
    bits = np.asarray(bits, dtype=np.uint8)
    m, n = entry["m"], entry["n"]
    n_sym = len(bits) // m                 # at bit offset 0; fewer further in
    n_starts = n_sym - n + 1
    if n_starts <= 0:
        return np.zeros((m, 0), bool), np.zeros((m, 0), bool)
    sym = _rs_symbols(bits, m, n_sym)
    roots = np.unique(np.asarray(entry["roots"], dtype=np.int64))
    if _T is None:
        T = _rs_prefix_tables(sym, m, entry["exp"], entry["log"], roots)
    else:
        lookup, table = _T
        T = table[:, [lookup[int(r)] for r in roots], :n_sym + 1]
    clean = np.all(T[:, :, n:n + n_starts] == T[:, :, :n_starts], axis=1)
    first = np.arange(m)[:, None] + m * np.arange(n_starts)[None, :]
    end = first + m * n
    inside = end <= len(bits)
    cum_idle = np.concatenate([[0], np.cumsum(_idle_positions(bits))])
    lo, hi = np.minimum(first, len(bits)), np.minimum(end, len(bits))
    not_idle = cum_idle[hi] - cum_idle[lo] < m * n
    changes = np.concatenate([np.zeros((m, 1), np.int64),
                              np.cumsum(sym[:, 1:] != sym[:, :-1], axis=1)], axis=1)
    varied = changes[:, n - 1:n - 1 + n_starts] > changes[:, :n_starts]
    return clean, inside & not_idle & varied


def probe_rs_many(bits: np.ndarray, entries: list, alpha: float = ALPHA,
                  beta: float = BETA) -> list:
    """RS entries over one shared prefix-XOR table per field; each entry's
    best (bit offset, symbol alignment) with its Early Verdict."""
    bits = np.asarray(bits, dtype=np.uint8)
    shared = {}
    for e in entries:
        shared.setdefault((e["prim"], e["m"]), set()).update(int(r) for r in e["roots"])
    tables = {}
    for (prim, m), roots in shared.items():
        e0 = next(e for e in entries if (e["prim"], e["m"]) == (prim, m))
        roots = np.array(sorted(roots), dtype=np.int64)
        n_sym = len(bits) // m
        if n_sym:
            tables[(prim, m)] = ({int(r): k for k, r in enumerate(roots)},
                                 _rs_prefix_tables(_rs_symbols(bits, m, n_sym), m,
                                                   e0["exp"], e0["log"], roots))
    return [probe_rs(bits, e, alpha, beta, tables.get((e["prim"], e["m"])))
            for e in entries]


def probe_rs(bits: np.ndarray, entry: dict, alpha: float = ALPHA,
             beta: float = BETA, _T=None) -> dict:
    """Early Verdict per alignment (bit offset, symbol start mod n) on the
    clean-codeword indicators, block = 1; returns the best alignment (by
    verdict, then clean fraction) plus every alignment's state
    (alignment_state: (m, n), 1/0/-1).  A full-length code is cyclic, so a
    window j symbols off a codeword is itself one whenever the j symbols
    it trades are equal (chance 2^-mj): next to the true alignment a
    neighbour can read "code" too, at a far lower clean fraction."""
    m, n, k = entry["m"], entry["n"], entry["k"]
    clean, inform = rs_zero_indicators(bits, entry, _T)
    p0 = max(float(2 ** m) ** -(n - k), RS_P0_FLOOR)
    p1 = (1 - RS_EXPECTED_BER) ** (m * n)
    c0 = np.log((1 - p1) / (1 - p0))
    walk = _Walk(np.full(m * n, np.log(p1 / p0) - c0), np.full(m * n, c0),
                 alpha, beta)
    n_starts = clean.shape[1]
    rows = -(-n_starts // n) if n_starts else 0
    pad = rows * n - n_starts
    z = np.concatenate([clean & inform, np.zeros((m, pad), bool)], axis=1)
    v = np.concatenate([inform, np.zeros((m, pad), bool)], axis=1)
    # column (off, a) holds the codewords starting at symbols a, a+n, ...
    z = z.reshape(m, rows, n).transpose(1, 0, 2).reshape(rows, m * n)
    v = v.reshape(m, rows, n).transpose(1, 0, 2).reshape(rows, m * n)
    if rows:
        walk.feed(z.astype(np.float64), v.astype(np.float64), np.arange(m * n))
    n_cw = v.sum(axis=0)
    frac = np.divide(z.sum(axis=0), n_cw, out=np.zeros(m * n), where=n_cw > 0)
    order = np.lexsort((walk.used, -frac, -walk.state))
    best = int(order[0])
    r = float(frac[best])
    ber = 0.5 if r <= 0 else float(1 - r ** (1.0 / (m * n)))
    seq = z[v[:, best], best]
    return {"entry": entry["name"], "bit_offset": best // n,
            "symbol_offset": best % n, "clean_fraction": r,
            "n": int(n_cw[best]), "clean_sequence": seq,
            "early_verdict": _verdict(walk.state[best], walk.llr[best],
                                      walk.used[best], alpha, ber),
            "alignment_state": walk.state.reshape(m, n).copy()}
