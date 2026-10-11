"""Stage S6: synchronisation and demodulation to soft bits.

Chain: coarse CFO removal -> RRC matched filter -> Gardner
timing-recovery loop -> decision-directed carrier PLL -> AGC -> slicer
with max-log LLRs.  FSK takes the instantaneous-frequency path instead.

The auto-driver configures everything from the S4/S5 estimates; every
loop publishes a lock metric so failure is visible, not silent.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sig

from ..common.models import DemodulationResult
from .constellations import (CONSTELLATIONS, MOD_FAMILY, slice_symbols,
                             symmetry_order)
from .filters import rrc_taps
from ..params.estimators import occupied_bandwidth


def _coarse_cfo(x: np.ndarray, order: int, limiter: bool = True,
                max_abs: float = None) -> float:
    """M-power CFO estimate.

    The hard limiter sharpens the spectral line for constant-modulus (PSK)
    signals but destroys the amplitude structure that the x^4 line of QAM
    depends on, so QAM callers pass limiter=False. max_abs limits the
    search to |CFO| <= max_abs (x^M also carries cyclic side-lines at
    M*CFO +- k*Rs, which unscrambled coded data can make the strongest).
    """
    n = min(len(x), 1 << 18)
    base = (x[:n] / (np.abs(x[:n]) + 1e-12)) if limiter else x[:n]
    y = base ** order
    Y = np.abs(np.fft.fft(y * np.hanning(n)))
    f = np.fft.fftfreq(n) / order
    Y[0] = 0
    if max_abs is not None:
        Y[np.abs(f) > max_abs] = 0
    k = int(np.argmax(Y))
    return float(f[k])


def _timing_recover(x: np.ndarray, sps: float,
                    max_symbols: int = 200000) -> tuple:
    """Feedforward Oerder&Meyr timing recovery.

    The squared envelope |x|^2 of a shaped linear modulation contains a
    spectral tone at the symbol rate whose phase encodes the timing offset.
    We measure that phase per block, unwrap it, fit a line (offset + clock
    drift) and interpolate the signal at the fitted symbol instants.
    Fully vectorised and deterministic - no feedback loop to diverge.

    Returns (symbols, tone_strength, locked).
    """
    # bring the stream to an integer number of samples/symbol
    Q = int(round(sps))
    if Q < 4:
        Q = 4
    if abs(sps - Q) / sps > 0.003:
        from fractions import Fraction
        fr = Fraction(Q / sps).limit_denominator(500)
        x = sig.resample_poly(x, fr.numerator, fr.denominator)
    n = len(x)
    if n < 8 * Q:
        return np.zeros(0, dtype=np.complex128), 0.0, False

    env = np.abs(x) ** 2
    # Pin the envelope tone frequency before the block-phase fit: the
    # nominal rate can be off by a few percent (near-Nyquist bins, line
    # competition at low SNR), which drifts more than Q/2 per block and
    # aliases the unwrap below. A parabolic peak fit near 1/Q measures
    # the true tone to a fraction of a bin.
    T = float(Q)
    nwin = int(min(n, 1 << 18))
    nfft = int(min(1 << 18, 1 << (int(np.ceil(np.log2(max(nwin, 16)))) + 2)))
    if nwin >= 32 * Q:
        E = np.abs(np.fft.rfft((env[:nwin] - env[:nwin].mean()) *
                               np.hanning(nwin), nfft))
        lo = int(np.floor(0.94 * nfft / Q))
        hi = min(int(np.ceil(1.06 * nfft / Q)) + 1, len(E) - 2)
        if lo >= 2 and hi - lo > 4:
            k0 = lo + int(np.argmax(E[lo:hi]))
            med = float(np.median(E[lo:hi])) + 1e-12
            if E[k0] > 4 * med:
                am, bm, cm = float(E[k0 - 1]), float(E[k0]), float(E[k0 + 1])
                den = am - 2 * bm + cm
                delta = 0.5 * (am - cm) / den if abs(den) > 1e-12 else 0.0
                f_ref = (k0 + float(np.clip(delta, -0.5, 0.5))) / nfft
                if abs(f_ref * Q - 1.0) < 0.08:
                    T = 1.0 / f_ref
    block = max(int(T * 64), 512)
    n_blocks = max(1, n // block)
    cs, weights = [], []
    k = np.exp(-2j * np.pi * np.arange(block) / T)
    for b in range(n_blocks):
        seg = env[b * block:(b + 1) * block]
        if len(seg) < Q * 8:
            break
        c = (seg * k[:len(seg)]).sum()
        # re-reference the block phase to global time zero: the block
        # length is generally NOT a multiple of the (fractional) symbol
        # period, and per-block references would masquerade as a huge
        # fake clock drift of (block mod T)/block per sample
        c = c * np.exp(-2j * np.pi * (b * block) / T)
        weights.append(np.abs(c) / (seg.sum() + 1e-12))
        cs.append(c)
    cs = np.asarray(cs, dtype=np.complex128)
    weights = np.asarray(weights)
    tone = float(weights.mean()) if len(weights) else 0.0
    if len(cs) == 0:
        return np.zeros(0, dtype=np.complex128), 0.0, False
    # complex-domain offset + drift: averaging c_b * conj(c_{b-1}) gives
    # the mean per-block phase increment without an unwrap chain, so one
    # noisy block cannot corrupt every later block's timing estimate
    if len(cs) >= 2:
        # coarse slope from lag-1 products (wrap-safe), then a weighted
        # regression on the detrended residual phases: lag-1 alone has
        # regression-grade bias-freedom but not regression-grade
        # variance, and dense QAM cannot afford the accumulated timing
        # error of a noisy slope
        dphi = float(np.angle((cs[1:] * np.conj(cs[:-1])).sum()))
        idx = np.arange(len(cs), dtype=np.float64)
        det = cs * np.exp(-1j * dphi * idx)
        mean_ang = float(np.angle(det.sum()))
        phi_r = np.angle(det * np.exp(-1j * mean_ang))
        w = weights / (weights.sum() + 1e-12)
        i_m = float((w * idx).sum())
        p_m = float((w * phi_r).sum())
        var = float((w * (idx - i_m) ** 2).sum())
        slope = (float((w * (idx - i_m) * (phi_r - p_m)).sum()) / var
                 if var > 0 else 0.0)
        phi0 = mean_ang + p_m - slope * i_m
        dphi_t = dphi + slope
        drift = -dphi_t / (2 * np.pi) * T / block   # timing samples/sample
        mu0 = float(-phi0 / (2 * np.pi) * T)
    else:
        mu0, drift = float(-np.angle(cs[0]) / (2 * np.pi) * T), 0.0
    # symbol sampling instants, cubic Lagrange interpolation (linear
    # interpolation leaves an ISI floor of several percent EVM, which is
    # irrelevant for QPSK but fatal for 128/256-QAM decisions)
    n_sym = min(int((n - Q) // T), max_symbols)
    kk = np.arange(n_sym)
    t = mu0 % T + kk * (T + drift * T)
    t = t[(t >= 1) & (t < n - 2)]
    i0 = t.astype(int)
    f = t - i0
    xm1, x0, x1, x2 = x[i0 - 1], x[i0], x[i0 + 1], x[i0 + 2]
    c0 = -f * (f - 1) * (f - 2) / 6
    c1 = (f + 1) * (f - 1) * (f - 2) / 2
    c2 = -(f + 1) * f * (f - 2) / 2
    c3 = (f + 1) * f * (f - 1) / 6
    syms = c0 * xm1 + c1 * x0 + c2 * x1 + c3 * x2
    locked = bool(tone > 0.02)
    return syms, tone, locked


def _dd_pll(symbols: np.ndarray, modulation: str, loop_bw: float = 0.02) -> tuple:
    """Decision-directed carrier phase/frequency tracking.

    Returns (corrected symbols, phase_error_rms, locked)."""
    table, _ = CONSTELLATIONS[modulation]
    zeta = 0.707
    theta = loop_bw
    d = 1 + 2 * zeta * theta + theta * theta
    alpha = 4 * zeta * theta / d
    beta = 4 * theta * theta / d
    phase = 0.0
    freq = 0.0
    # 4th-power feedforward warm start anchored to the constellation's OWN
    # fourth moment: rotate so angle(E[y^4]) matches angle(E[table^4]).
    # This is exact for square QAM, cross QAM and APSK alike (the old
    # assumption of angle pi only held for square constellations).
    ref4 = (table ** 4).mean()
    if np.abs(ref4) > 0.05:
        c4 = (symbols[: 4096] ** 4).mean()
        if np.abs(c4) > 1e-6:
            phase = float((np.angle(c4) - np.angle(ref4)) / 4.0)
    else:
        # constellations with a vanishing 4th moment (APSK rings, dense
        # PSK): probe a grid of rotations inside the constellation's own
        # symmetry sector and start from the best-fitting one
        sym_sector = np.pi / 2
        probe = symbols[: 2048]
        best_phi, best_d = 0.0, np.inf
        for kk in range(16):
            phi = kk * sym_sector / 16
            rot = probe * np.exp(-1j * phi)
            d = np.abs(rot[:, None] - table[None, :]).min(axis=1).mean()
            if d < best_d:
                best_d, best_phi = d, phi
        phase = float(best_phi)
    out = np.empty_like(symbols)
    errs = np.empty(len(symbols))
    for i, s in enumerate(symbols):
        y = s * np.exp(-1j * phase)
        dpt = table[np.argmin(np.abs(y - table))]
        e = float(np.angle(y * np.conj(dpt)))
        freq += beta * e
        freq = float(np.clip(freq, -0.2, 0.2))
        phase += freq + alpha * e
        out[i] = y
        errs[i] = e
    tail = errs[len(errs) // 2:]
    return out, float(np.sqrt((tail ** 2).mean())), bool(np.abs(tail).mean() < 0.35)


# CFO estimation strategy per family: (nonlinearity order, use limiter).
# The hard limiter sharpens PSK lines but destroys amplitude structure, so
# QAM/APSK/ASK run without it. High-order PSK lines (x^16, x^32) are weak
# but usable at the SNRs where those modulations decode at all.
_CFO_STRATEGY = {
    "psk": lambda order: (min(order, 32), True),
    "oqpsk": lambda order: (4, True),
    "qam": lambda order: (4, False),
    "apsk": lambda order: (4, False),
    "ask": lambda order: (2, False),
}

# EVM thresholds (percent) for the quality gate: (good, degraded) per
# modulation; anything worse is FAILED. Denser constellations need lower
# EVM to decode, so their gates are tighter.
_EVM_GATES = {
    "BPSK": (25, 45), "QPSK": (18, 32), "OQPSK": (18, 32), "8PSK": (12, 22),
    "16PSK": (7, 13), "32PSK": (7.5, 12),
    "OOK": (30, 50), "4ASK": (15, 28), "8ASK": (10, 18),
    "16QAM": (12, 20), "32QAM": (9, 16), "64QAM": (7, 12),
    "128QAM": (5, 9), "256QAM": (3.5, 7),
    "16APSK": (10, 18), "32APSK": (8, 14), "64APSK": (7, 11),
    "128APSK": (6, 10),
}


def _order_of(modulation: str) -> int:
    table, k = CONSTELLATIONS[modulation]
    return len(table)


def _apply_status(res: DemodulationResult, modulation: str):
    """Mandatory S6 quality gate: GOOD / DEGRADED / FAILED from EVM and
    lock metrics. A FAILED demodulation must not feed the bit layer."""
    gate = _EVM_GATES.get(modulation)
    if gate is None:                      # fsk/gmsk/analog set it themselves
        return
    evm = res.evm_percent
    locked = res.timing_locked and res.carrier_locked
    if evm is None or not locked:
        res.demodulation_status = "FAILED"
        res.warnings.append("no synchronisation lock: bit stream withheld")
        return
    good, degraded = gate
    if evm <= good:
        res.demodulation_status = "GOOD"
    elif evm <= degraded:
        res.demodulation_status = "DEGRADED"
        res.warnings.append(
            f"EVM {evm}% is marginal for {modulation}: downstream results "
            "are speculative")
    else:
        res.demodulation_status = "FAILED"
        res.warnings.append(
            f"EVM {evm}% is too poor to slice {modulation} reliably: "
            "bit stream withheld from the bit layer")
    # The symbol-domain carrier estimate is ambiguous by 1/m cycles/symbol
    # (m = rotational symmetry): the wrong alias still lands every symbol on
    # the grid, so EVM looks fine while the bits are random. Near the edge
    # of the unambiguous range +-1/(2m) the choice is a coin flip.
    # ponytail: flag, do not resolve; resolving needs a carrier estimate
    # better than Rs/(2m) (band centre and oversampled lines were not)
    f = res.lock_metrics.get("symbol_cfo_per_symbol")
    m = symmetry_order(modulation)
    if (res.demodulation_status == "GOOD" and f is not None and m >= 8
            and abs(f) > 0.75 / (2 * m)):
        res.demodulation_status = "DEGRADED"
        res.lock_metrics["cfo_alias_ambiguous"] = True
        res.warnings.append(
            f"carrier frequency ambiguous: the {modulation} estimate "
            f"({f:+.4f} cycles/symbol) is near the alias boundary "
            f"+-{1 / (2 * m):.4f}, where a false lock keeps a good EVM but "
            "scrambles the bits")
    res.lock_metrics["evm_gate_good"] = good
    res.lock_metrics["evm_gate_degraded"] = degraded
    # EVM-implied SNR: honest quality figure alongside the M2M4 estimate
    if evm and evm > 0:
        res.lock_metrics["snr_from_evm_db"] = round(
            -20 * np.log10(evm / 100.0), 1)


def _ring_info(table: np.ndarray) -> list:
    """[(radius, n_points)] rings of a constellation (for APSK checks)."""
    radii = np.abs(table)
    rings = []
    for r in sorted(set(np.round(radii, 3))):
        rings.append((float(r), int(np.sum(np.abs(radii - r) < 1e-3))))
    return rings


def _ring_gate_width(table: np.ndarray, radius: float) -> float:
    """Half-width for gating symbols onto one ring: at most 30% of the
    radius but never past the midpoint to the neighbouring ring (dense
    APSK rings sit close and cross-ring leakage biases the phase
    statistic)."""
    radii = sorted(r for r, _n in _ring_info(table))
    gaps = [abs(radius - r) for r in radii if abs(radius - r) > 1e-6]
    min_gap = min(gaps) if gaps else radius
    return float(min(0.3 * radius, 0.45 * min_gap))


def _nearest_distance(syms: np.ndarray, table: np.ndarray,
                      n_probe: int = 2048) -> float:
    """Mean distance to the nearest constellation point (unit-power
    domain). Independent of any spectral-line statistic, so it can
    arbitrate between candidate CFO lines without circularity."""
    z = syms[: n_probe]
    return float(np.abs(z[:, None] - table[None, :]).min(axis=1).mean())


def _rotational_concentration(syms: np.ndarray, modulation: str,
                              family: str) -> float:
    """True carrier-lock evidence: nearest-point EVM is blind to a
    spinning dense-PSK ring, but |E[(s/|s|)^M]| collapses to ~0 when the
    constellation rotates. For APSK the dominant ring's point count sets
    M; for QAM the 4th moment is compared to the table's own."""
    table, _k = CONSTELLATIONS[modulation]
    u = syms / (np.abs(syms) + 1e-12)
    if family in ("psk", "oqpsk"):
        M = len(table) if family == "psk" else 4
        return float(np.abs((u ** M).mean()))
    if family == "apsk":
        rings = _ring_info(table)
        radius, n_pts = max(rings, key=lambda rn: rn[1])
        band = np.abs(np.abs(syms) - radius) < \
            _ring_gate_width(table, radius)
        if band.sum() < 32:
            return 0.0
        return float(np.abs((u[band] ** n_pts).mean()))
    # QAM: a moment-based check is circular with the CFO line selection,
    # so lock evidence comes from geometry: on-grid symbols sit much
    # closer to the table than the same symbols rotated off-grid
    d_meas = _nearest_distance(syms, table)
    d_off = _nearest_distance(syms * np.exp(1j * np.pi / 8), table)
    return float(np.clip((d_off - d_meas) / (d_off + 1e-12), 0.0, 1.0) /
                 0.6)


def _vv_feedforward(syms: np.ndarray, modulation: str, family: str,
                    block: int = 192) -> np.ndarray:
    """Viterbi&Viterbi-style block feedforward carrier recovery.

    Per block, the constellation's rotational statistic (u^M for PSK,
    s^4 against the table reference for QAM/APSK) yields a phase estimate;
    estimates are unwrapped across blocks and interpolated. Being
    feedforward, it cannot cycle-slip - the residual constant M-fold
    ambiguity is exactly what the S7 fan-out enumerates."""
    table, _k = CONSTELLATIONS[modulation]
    order = len(table)
    if family in ("psk", "oqpsk"):
        M = order if family == "psk" else 4
        z = (syms / (np.abs(syms) + 1e-12)) ** M
        ref_angle = float(np.angle((table / np.abs(table)) [0] ** 0)) * 0
        zref = ((table / np.abs(table)) ** M).mean()
        ref_angle = float(np.angle(zref)) if np.abs(zref) > 1e-6 else 0.0
    else:
        M = 4
        z = syms ** 4
        zref = (table ** 4).mean()
        if np.abs(zref) < 0.02:
            # no usable 4th-moment reference (dense APSK): use the
            # dominant ring's point count on ring-gated symbols
            rings = _ring_info(table)
            radius, n_pts = max(rings, key=lambda rn: rn[1])
            M = n_pts
            gate = np.abs(np.abs(syms) - radius) < \
                _ring_gate_width(table, radius)
            u = syms / (np.abs(syms) + 1e-12)
            z = np.where(gate, u ** M, 0)
            tri = table[np.abs(np.abs(table) - radius) < 1e-3]
            zref = ((tri / np.abs(tri)) ** M).mean()
        ref_angle = float(np.angle(zref)) if np.abs(zref) > 1e-6 else 0.0

    n_blocks = max(1, len(z) // block)
    centers, angles = [], []
    for b in range(n_blocks):
        zz = z[b * block:(b + 1) * block]
        m = zz[zz != 0].mean() if np.any(zz != 0) else 0
        if np.abs(m) < 1e-9:
            continue
        centers.append(b * block + block / 2)
        angles.append(np.angle(m) - ref_angle)
    if len(angles) < 1:
        return syms
    ang = np.array(angles, dtype=float)
    # unwrap in the M-fold sector
    for i in range(1, len(ang)):
        while ang[i] - ang[i - 1] > np.pi:
            ang[i] -= 2 * np.pi
        while ang[i] - ang[i - 1] < -np.pi:
            ang[i] += 2 * np.pi
    phi = np.interp(np.arange(len(syms)), centers, ang / M)
    return syms * np.exp(-1j * phi)


def _symbol_domain_cfo(syms: np.ndarray, modulation: str,
                       family: str) -> tuple:
    """Residual CFO estimation on symbol-spaced samples.

    After timing recovery the samples are ISI-free, so the M-power
    spectral line is far cleaner than in the oversampled domain - this is
    what makes 16/32-PSK and APSK lockable at all (their sample-domain
    x^16/x^32 lines drown in shaping sidelobes).
    Returns (corrected symbols, applied freq cycles/symbol, line quality).
    """
    table, _k = CONSTELLATIONS[modulation]
    order = len(table)
    if family in ("psk", "oqpsk"):
        Ms = [order if family == "psk" else 4]
    elif family == "apsk":
        Ms = [n for _r, n in _ring_info(table)] + [4]
    elif family == "ask":
        Ms = [2]
    else:
        Ms = [4]
    u = syms / (np.abs(syms) + 1e-12)
    n = len(u)
    nfft = 1 << int(np.ceil(np.log2(max(1024, 4 * n))))
    w = np.hanning(n)
    # Candidate lines can be DATA lines (frame periodicity puts spectral
    # lines into u^M as well), so every candidate is validated by the
    # rotational concentration it produces after correction - the true
    # CFO line freezes the constellation, a data line does not.
    kk = np.arange(n)
    cands = [(0.0, 0.0)]
    for M in sorted(set(Ms)):
        if M < 2 or M > 64:
            continue
        base = u if family in ("psk", "oqpsk", "apsk") else syms
        z = (base ** M) * w
        Z = np.abs(np.fft.fft(z, nfft))
        med = float(np.median(Z))
        # the concentration validation below rejects junk candidates, so
        # even modest lines are worth evaluating (dense QAM's s^4 line is
        # real but only a few times the noise median)
        order_idx = np.argsort(Z)[::-1][:10]
        for k in order_idx:
            quality = float(Z[k] / (med + 1e-12))
            if quality < 2.5:
                break
            f = float(np.fft.fftfreq(nfft)[int(k)] / M)
            if abs(f) < 0.45:
                cands.append((f, quality))
    # arbitrate candidates by independent geometric evidence: the true
    # correction puts the probe symbols closest to the constellation
    # (validating with the same M-power statistic would be circular)
    table_full, _kk2 = CONSTELLATIONS[modulation]
    best_f, best_d, best_q = 0.0, np.inf, 0.0
    for f, q in cands:
        trial = syms[: 2048] * np.exp(-2j * np.pi * f * kk[: 2048]) if f \
            else syms[: 2048]
        # constant rotation must not penalise a correct frequency: align
        # the probe's best constant phase first (cheap grid search)
        d = min(_nearest_distance(trial * np.exp(-1j * ph), table_full,
                                  1024)
                for ph in np.linspace(0, np.pi / 2, 12, endpoint=False))
        if d < best_d - 1e-6:
            best_f, best_d, best_q = f, d, q
    if best_f:
        syms = syms * np.exp(-2j * np.pi * best_f * kk)
    return syms, best_f, best_q


def _phase_polish(symbols: np.ndarray, modulation: str,
                  window: int = 129) -> np.ndarray:
    """Feedforward fine phase correction: decision-directed error smoothed
    over a sliding window. Removes the residual jitter feedback loops
    leave on dense constellations (128/256-QAM, high-order APSK)."""
    table, _ = CONSTELLATIONS[modulation]
    d2 = np.abs(symbols[:, None] - table[None, :])
    dec = table[np.argmin(d2, axis=1)]
    err = np.angle(symbols * np.conj(dec))
    kernel = np.ones(window) / window
    trend = np.convolve(err, kernel, mode="same")
    return symbols * np.exp(-1j * trend)


def demodulate(x: np.ndarray, modulation: str, sps: float,
               config, cfo_norm: float = None) -> DemodulationResult:
    """Auto-configured demodulation, dispatched by modulation family.

    Families: psk / oqpsk / qam / apsk / ask (linear chain with
    family-specific CFO and carrier settings), fsk, gmsk (frequency
    discriminator chain), analog (AM/FM/SSB detectors producing audio,
    not bits). Every path ends in the mandatory quality gate that sets
    res.demodulation_status."""
    x = np.asarray(x, dtype=np.complex128)
    family = MOD_FAMILY.get(modulation)
    if family in ("psk", "oqpsk", "qam", "apsk", "ask") and 0 < sps < 3.0:
        # Oerder&Meyr timing needs about 3+ samples per symbol; near
        # Nyquist-rate recordings are upsampled 2x first (CFO is in
        # cycles/sample, so the normalised offset halves with the rate)
        x = sig.resample_poly(x, 2, 1)
        sps *= 2.0
        if cfo_norm is not None:
            cfo_norm = cfo_norm / 2.0
    res = DemodulationResult(modulation=modulation, samples_per_symbol=sps)
    res.demodulation_status = "FAILED"

    if family == "analog":
        return _demod_analog(x, modulation, res)
    if family == "gmsk":
        return _demod_gmsk(x, sps, res)
    if family == "fsk" or (modulation.endswith("FSK") and
                           modulation[0].isdigit()):
        out = _demod_fsk(x, modulation, sps, res)
        out.demodulation_status = ("GOOD" if out.timing_locked else "DEGRADED")
        if out.hard_bits is None or not len(out.hard_bits):
            out.demodulation_status = "FAILED"
        return out
    if family == "oqpsk":
        return _demod_oqpsk(x, sps, config, res)
    if family not in ("psk", "qam", "apsk", "ask"):
        res.warnings.append(f"unsupported modulation for demodulation: "
                            f"{modulation}")
        return res
    return _demod_linear(x, modulation, family, sps, config, cfo_norm, res)


def _demod_linear(x, modulation, family, sps, config, cfo_norm, res):
    order = _order_of(modulation)
    m_order, limiter = _CFO_STRATEGY[family](order)

    # 1. coarse CFO: upstream estimate for LOW-order constant-modulus
    # families only (the S4 estimator guesses the nonlinearity order and
    # its spurious lines would poison high-order PSK), then a
    # family-appropriate M-power refinement; the precise correction
    # happens later in the symbol domain where the line is clean.
    n = np.arange(len(x))
    cfo = 0.0
    if family in ("psk", "oqpsk") and cfo_norm and order <= 8:
        cfo = float(cfo_norm)
        x = x * np.exp(-2j * np.pi * cfo * n)
    # Sample-domain M-power refinement only where the line is reliable:
    # for dense PSK and APSK the x^M line drowns in shaping sidelobes and
    # a wrong "correction" here poisons the whole chain (timing itself is
    # CFO-invariant, and the symbol-domain estimator after timing is far
    # more sensitive), so those families skip it entirely.
    sample_cfo_reliable = (family in ("psk", "oqpsk") and order <= 8) or \
        family == "ask"
    if sample_cfo_reliable:
        # a residual: the line must sit inside +-Rs/2 of the x^M carrier
        # line, clear of the cyclic side-lines at +-Rs
        resid = _coarse_cfo(x, m_order, limiter=limiter,
                            max_abs=0.5 / (m_order * sps))
        if abs(resid) < 0.05:
            x = x * np.exp(-2j * np.pi * resid * n)
            cfo += resid
        res.lock_metrics["residual_cfo_est"] = round(float(resid), 6)
    res.cfo_applied_norm = float(cfo)

    # 2. matched filter
    n_int = max(2, int(round(sps)))
    taps = rrc_taps(n_int, config.rrc_span_symbols, config.rrc_rolloff)
    x = sig.fftconvolve(x, taps, mode="same")

    # eye-diagram trace for the display
    q_eye = max(2, n_int)
    res.lock_metrics["eye_sps"] = q_eye
    res.eye_trace = np.asarray(x[: 200 * q_eye], dtype=np.complex64)

    # 3. timing recovery (feedforward Oerder&Meyr)
    syms, tone, tlock = _timing_recover(x, sps, config.max_symbols)
    res.timing_locked = tlock
    res.lock_metrics["timing_tone_strength"] = round(tone, 4)
    if len(syms) < 32:
        res.warnings.append("timing recovery produced too few symbols")
        return res

    # 4. AGC
    syms = syms / (np.sqrt((np.abs(syms) ** 2).mean()) + 1e-12)

    # 4b. symbol-domain residual CFO (clean M-power line on ISI-free
    # samples; essential for 16/32-PSK and APSK)
    syms, f_resid, line_q = _symbol_domain_cfo(syms, modulation, family)
    res.lock_metrics["symbol_cfo_per_symbol"] = round(f_resid, 7)
    res.lock_metrics["symbol_cfo_line_quality"] = round(line_q, 1)

    # 5. carrier recovery.
    # Low-order constellations use the decision-directed PLL (fast, tracks
    # residual frequency). Dense PSK / APSK / dense QAM use slip-free
    # feedforward Viterbi&Viterbi blocks followed by a decision-directed
    # polish; a feedback loop WILL cycle-slip on these and scramble the
    # bit stream even while the EVM still looks fine.
    dense = (family == "psk" and order >= 16) or family == "apsk" or \
        (family in ("qam",) and order >= 128)
    if family == "ask":
        # align the PAM line onto the real axis via the second moment
        c2 = (syms[:4096] ** 2).mean()
        syms = syms * np.exp(-1j * np.angle(c2) / 2)
        syms, perr, clock = _dd_pll(syms, modulation, 0.005)
    elif dense and family == "qam":
        # dense QAM: after the symbol-domain CFO the only unknown is a
        # constant phase. Decision-directed loops and per-block VV both
        # ADD noise here (the s^4 statistic and the decisions are equally
        # unreliable per block), so the phase is found by a fine grid
        # search on nearest-constellation distance and only a very slow
        # feedforward polish tracks residual drift.
        table, _ktab = CONSTELLATIONS[modulation]
        grid = np.linspace(0, np.pi / 2, 96, endpoint=False)
        probe = syms[: 1024]
        best_ph = min(grid, key=lambda ph: _nearest_distance(
            probe * np.exp(-1j * ph), table, 1024))
        syms = syms * np.exp(-1j * best_ph)
        syms = _phase_polish(syms, modulation, 401)
        syms = _phase_polish(syms, modulation, 201)
        d2 = np.abs(syms[:, None] - table[None, :])
        dec = table[np.argmin(d2, axis=1)]
        errs = np.angle(syms * np.conj(dec))
        perr = float(np.sqrt(np.mean(errs[len(errs) // 2:] ** 2)))
        clock = bool(np.abs(errs).mean() < 0.35)
    elif dense:
        table, _ktab = CONSTELLATIONS[modulation]
        syms = _vv_feedforward(syms, modulation, family)
        if family == "apsk":
            # the VV anchor rides the dominant ring, which fixes phase
            # only modulo that ring's own symmetry (e.g. 30 deg for a
            # 12-point ring) - finer than the full constellation's 90 deg
            # symmetry, so the residual k*(2pi/M_ring) rotation must be
            # resolved against the WHOLE table
            rings = _ring_info(table)
            m_dom = max(rings, key=lambda rn: rn[1])[1]
            probe = syms[: 20000]
            n_sym_full = 2 * np.pi / m_dom
            # the dominant ring is invariant under exactly these rotations
            # and carries NO coset information while dominating the symbol
            # count: gate it OUT and let the other rings discriminate
            radius_dom = max(rings, key=lambda rn: rn[1])[0]
            off_dom = np.abs(np.abs(probe) - radius_dom) > \
                _ring_gate_width(table, radius_dom)
            probe_d = probe[off_dom] if off_dom.sum() > 256 else probe
            cand_ds = []
            # rotations inside the constellation's own symmetry group are
            # identical: only the distinct cosets are candidates
            from .constellations import symmetry_order
            sym = symmetry_order(modulation)
            for kk2 in range(max(1, m_dom // sym)):
                phi = kk2 * n_sym_full
                rot = probe_d * np.exp(-1j * phi)
                d = np.abs(rot[:, None] - table[None, :]).min(axis=1)
                # trimmed mean: outliers (decision-boundary symbols) carry
                # no rotation information and only blur the margin
                d = np.sort(d)[: int(0.8 * len(d))].mean()
                cand_ds.append((float(d), phi))
            cand_ds.sort()
            best_d, best_phi = cand_ds[0]
            margin = (cand_ds[1][0] - best_d) / (best_d + 1e-12)
            res.lock_metrics["apsk_snap_margin"] = round(float(margin), 3)
            if margin < 0.02:
                res.warnings.append(
                    "APSK ring-rotation ambiguity is marginal (snap margin "
                    f"{margin:.3f}); the bit mapping may be rotated")
            if best_phi:
                syms = syms * np.exp(-1j * best_phi)
        syms = _phase_polish(syms, modulation, 201)
        syms = _phase_polish(syms, modulation, 65)
        # residual phase error metric from decisions
        d2 = np.abs(syms[:, None] - table[None, :])
        dec = table[np.argmin(d2, axis=1)]
        errs = np.angle(syms * np.conj(dec))
        perr = float(np.sqrt(np.mean(errs[len(errs) // 2:] ** 2)))
        clock = bool(np.abs(errs).mean() < 0.35)
    else:
        loop_bw = config.carrier_loop_bw
        syms, perr, clock = _dd_pll(syms, modulation, loop_bw)
        if order >= 16:
            syms, perr2, clock2 = _dd_pll(syms, modulation, loop_bw / 4)
            perr = min(perr, perr2)
            clock = clock or clock2
        if order >= 64:
            syms = _phase_polish(syms, modulation)
    # rotational concentration: the lock metric nearest-point EVM cannot
    # fake (a spinning ring reads as concentration ~0)
    conc = _rotational_concentration(syms, modulation, family)
    res.lock_metrics["rotational_concentration"] = round(conc, 3)
    if conc < 0.2:
        clock = False
    res.carrier_locked = clock
    res.lock_metrics["phase_error_rms"] = round(perr, 4)

    settle = min(len(syms) // 10, 500)
    syms = syms[settle:]

    # 6. slice to hard bits + LLRs
    noise_var = max(1e-4, float(np.var(np.abs(syms)) * 0.5))
    hard, llrs, evm = slice_symbols(syms, modulation, noise_var)
    res.symbols = syms.astype(np.complex64)
    res.hard_bits = hard
    res.llrs = llrs
    res.evm_percent = round(evm, 1)
    _apply_status(res, modulation)
    return res


def _demod_oqpsk(x, sps, config, res):
    """OQPSK: the staggered rails mix under any carrier rotation, so the
    carrier must be recovered FIRST (x^4 carries a clean line at 4fc for
    OQPSK because the offset removes the 180-degree transitions), then the
    half-symbol stagger is undone and the signal decodes as QPSK. The
    90-degree ambiguity swaps the rails and thereby flips the stagger
    direction, so both de-offset directions are tried and the one whose
    constellation concentrates wins."""
    x = np.asarray(x, dtype=np.complex128)
    n = np.arange(len(x))

    # carrier frequency from the x^4 line
    m = min(len(x), 1 << 16)
    z = (x[:m] ** 4) * np.hanning(m)
    Z = np.abs(np.fft.fft(z))
    k = int(np.argmax(Z))
    quality = float(Z[k] / (np.median(Z) + 1e-12))
    fc = float(np.fft.fftfreq(m)[k] / 4)
    if quality > 15 and abs(fc) < 0.05:
        x = x * np.exp(-2j * np.pi * fc * n)
        res.cfo_applied_norm = float(fc)
    res.lock_metrics["oqpsk_x4_line_quality"] = round(quality, 1)
    # carrier phase from the time-domain 4th moment (pi/2 ambiguous)
    c4 = (x[: 1 << 16] ** 4).mean()
    if np.abs(c4) > 1e-9:
        phi = (np.angle(c4) - np.pi) / 4.0
        x = x * np.exp(-1j * phi)

    # symbol rate: the x^2 spectrum carries a line PAIR at 2fc +- Rs.
    # Data periodicity produces additional strong lines, so only a pair
    # SYMMETRIC about 2fc (known from the x^4 stage; ~0 after the
    # correction above) is accepted, and only at a rate the bandwidth
    # allows (Rs = OBW/(1+rolloff), rolloff <= 1): repeated frames put
    # stronger symmetric lines near DC, and S4's |x|^2 rate is weak here
    # because the stagger flattens the envelope.
    obw = occupied_bandwidth(x)["obw99"] or 1.0
    z2 = (x[:m] ** 2) * np.hanning(m)
    Z2 = np.abs(np.fft.fft(z2))
    freqs2 = np.fft.fftfreq(m)
    med2 = float(np.median(Z2))
    peak_idx = np.argsort(Z2)[::-1][:24]
    best_pair = None
    for a in range(len(peak_idx)):
        for b in range(a + 1, len(peak_idx)):
            fa, fb = freqs2[peak_idx[a]], freqs2[peak_idx[b]]
            centre = (fa + fb) / 2.0
            rs_c = abs(fa - fb) / 2.0
            if abs(centre) < 0.002 and 0.45 * obw < rs_c < 1.05 * obw:
                q = float(min(Z2[peak_idx[a]], Z2[peak_idx[b]]) / (med2 + 1e-12))
                if q > 10 and (best_pair is None or q > best_pair[1]):
                    best_pair = (rs_c, q)
    if best_pair:
        sps = 1.0 / best_pair[0]
        res.lock_metrics["oqpsk_x2_rs_norm"] = round(best_pair[0], 6)
        res.lock_metrics["oqpsk_x2_pair_quality"] = round(best_pair[1], 1)
    n_int = max(2, int(round(sps)))
    if not (2 <= n_int <= 64):
        n_int = 8
        sps = 8.0

    taps = rrc_taps(n_int, config.rrc_span_symbols, config.rrc_rolloff)
    mf = sig.fftconvolve(x, taps, mode="same")
    half = n_int // 2

    best = None
    for direction in ("advance_q", "advance_i"):
        if direction == "advance_q":
            rail = np.empty_like(mf.imag)
            rail[:-half] = mf.imag[half:]
            rail[-half:] = mf.imag[-1]
            cand = mf.real + 1j * rail
        else:
            rail = np.empty_like(mf.real)
            rail[:-half] = mf.real[half:]
            rail[-half:] = mf.real[-1]
            cand = rail + 1j * mf.imag
        syms, tone, tlock = _timing_recover(cand, sps, config.max_symbols)
        if len(syms) < 32:
            continue
        syms = syms / (np.sqrt((np.abs(syms) ** 2).mean()) + 1e-12)
        syms, f_r, lq = _symbol_domain_cfo(syms, "OQPSK", "oqpsk")
        syms, perr, clock = _dd_pll(syms, "OQPSK", config.carrier_loop_bw)
        conc = float(np.abs(((syms / (np.abs(syms) + 1e-12)) ** 4).mean()))
        if best is None or conc > best["conc"]:
            best = {"syms": syms, "tone": tone, "tlock": tlock,
                    "perr": perr, "clock": clock, "conc": conc,
                    "direction": direction}
    if best is None:
        res.warnings.append("OQPSK timing recovery failed on both rail "
                            "hypotheses")
        return res
    syms = best["syms"]
    settle = min(len(syms) // 10, 500)
    syms = syms[settle:]
    res.timing_locked = best["tlock"]
    res.carrier_locked = best["clock"] and best["conc"] > 0.2
    res.lock_metrics.update({
        "timing_tone_strength": round(best["tone"], 4),
        "phase_error_rms": round(best["perr"], 4),
        "rotational_concentration": round(best["conc"], 3),
        "stagger_direction": best["direction"],
    })
    res.samples_per_symbol = float(sps)
    noise_var = max(1e-4, float(np.var(np.abs(syms)) * 0.5))
    hard, llrs, evm = slice_symbols(syms, "OQPSK", noise_var)
    res.symbols = syms.astype(np.complex64)
    res.hard_bits = hard
    res.llrs = llrs
    res.evm_percent = round(evm, 1)
    _apply_status(res, "OQPSK")
    return res


def _msk_squaring_lines(x: np.ndarray) -> tuple:
    """Classic squaring estimator for h=0.5 CPM (MSK/GMSK): x^2 carries
    two spectral lines at 2fc +- Rs/2, so their spacing IS the symbol
    rate and their midpoint locates the residual carrier.
    Returns (rs_norm, fc_resid_norm, quality)."""
    n = min(len(x), 1 << 17)
    z = (x[:n] ** 2) * np.hanning(n)
    Z = np.abs(np.fft.fftshift(np.fft.fft(z)))
    freqs = np.fft.fftshift(np.fft.fftfreq(n))
    med = float(np.median(Z))
    k1 = int(np.argmax(Z))
    guard = max(4, n // 2048)
    Z2 = Z.copy()
    Z2[max(0, k1 - guard):k1 + guard] = 0
    k2 = int(np.argmax(Z2))
    quality = float(min(Z[k1], Z2[k2]) / (med + 1e-12))
    rs = float(abs(freqs[k1] - freqs[k2]))
    fc = float(0.5 * (freqs[k1] + freqs[k2]) / 2)
    return rs, fc, quality


def _demod_gmsk(x, sps, res):
    """GMSK (h = 0.5): squaring-line symbol rate and carrier, frequency
    discriminator, mid-symbol sign decisions. The discriminator sign IS
    the bit for MSK-family signals, so no carrier loop is needed."""
    rs, fc, quality = _msk_squaring_lines(x)
    if quality > 20 and 1e-3 < rs < 0.5:
        n_int = int(round(1.0 / rs))
        x = x * np.exp(-2j * np.pi * fc * np.arange(len(x)))
        res.lock_metrics["squaring_rs_norm"] = round(rs, 6)
        res.lock_metrics["squaring_line_quality"] = round(quality, 1)
    else:
        n_int = int(round(sps))
        res.warnings.append("GMSK squaring lines not found: falling back "
                            "to the upstream symbol-rate estimate")
    if not (2 <= n_int <= 64):
        n_int = 8

    inst = np.diff(np.unwrap(np.angle(x))) / (2 * np.pi)
    inst = inst - np.median(inst)

    def sample_at(n_i):
        w = max(2, n_i // 2)
        sm = np.convolve(inst, np.ones(w) / w, mode="same")
        best_p, best_score = 0, -1.0
        for p2 in range(n_i):
            sc = float(np.abs(sm[p2::n_i]).mean())
            if sc > best_score:
                best_score, best_p = sc, p2
        return sm[best_p::n_i], best_score / (0.25 / n_i)

    samp, margin0 = sample_at(n_int)
    dev = 0.25 / n_int                     # theoretical peak deviation
    res.samples_per_symbol = float(n_int)
    res.hard_bits = (samp > 0).astype(np.uint8)
    res.llrs = (-samp / (dev + 1e-12) * 4).astype(np.float32)
    res.symbols = (samp / (dev + 1e-12)).astype(np.complex64)
    margin = float(np.abs(samp).mean() / (dev + 1e-12))
    res.timing_locked = res.carrier_locked = bool(margin > 0.5)
    res.lock_metrics["discriminator_margin"] = round(margin, 3)
    res.demodulation_status = ("GOOD" if margin > 0.7 else
                               "DEGRADED" if margin > 0.4 else "FAILED")
    # clean GMSK sits at 0.75-1.05x the deviation; noise drives the
    # discriminator far past it (white noise 2.7-7x, BER 0.24 at 1.3x)
    if margin > 1.25:
        res.demodulation_status = "FAILED"
        res.timing_locked = res.carrier_locked = False
    if res.demodulation_status == "FAILED":
        res.warnings.append("GMSK discriminator margin outside 0.4-1.25 "
                            "(weak or noise-dominated): bit stream withheld")
    return res


def _demod_analog(x, modulation, res):
    """Analog demodulators: audio out, no bit stream. The bit layer is
    skipped for these signals by design."""
    n = len(x)
    audio = None
    extra = {}
    if modulation == "FM":
        inst = np.diff(np.unwrap(np.angle(x))) / (2 * np.pi)
        audio = inst - np.median(inst)
        extra["peak_deviation_norm"] = round(float(np.percentile(
            np.abs(audio), 99)), 5)
    elif modulation in ("AM-DSB-WC", "AM-DSB-SC"):
        env = np.abs(x)
        if modulation == "AM-DSB-WC":
            audio = env - env.mean()
            carrier = float(env.mean() / (env.std() + 1e-12))
            extra["carrier_to_modulation"] = round(carrier, 2)
        else:
            # suppressed carrier: recover the carrier from the squared
            # spectrum line, then take the coherent (real) component
            cfo = _coarse_cfo(x, 2, limiter=False)
            y = x * np.exp(-2j * np.pi * cfo * np.arange(n))
            rot = np.angle((y[:65536] ** 2).mean()) / 2
            audio = (y * np.exp(-1j * rot)).real
            extra["recovered_carrier_norm"] = round(float(cfo), 6)
    else:                                   # SSB
        # centre the occupied band; absolute audio pitch is ambiguous for
        # suppressed-carrier SSB and reported as such
        X = np.abs(np.fft.fft(x[:min(n, 1 << 17)]))
        k = int(np.argmax(X))
        f0 = np.fft.fftfreq(len(X))[k]
        y = x * np.exp(-2j * np.pi * f0 * np.arange(n))
        audio = y.real
        extra["shifted_by_norm"] = round(float(f0), 6)
        if modulation.endswith("SC"):
            res.warnings.append("suppressed-carrier SSB: absolute audio "
                                "pitch is ambiguous without the carrier")
    # post-detection audio low-pass: the discriminator/envelope output
    # carries wideband noise (FM click noise especially) far above the
    # message band; audio lives well below ~0.05 cycles/sample here
    lpf = sig.firwin(129, 0.05)
    audio = sig.fftconvolve(audio, lpf, mode="same")
    audio = audio / (np.abs(audio).max() + 1e-12)
    # audio quality: tonal concentration of the demodulated spectrum
    A = np.abs(np.fft.rfft(audio[:min(len(audio), 1 << 16)]))
    conc = float(np.sort(A)[-8:].sum() / (A.sum() + 1e-12))
    res.lock_metrics.update({"audio_tonal_concentration": round(conc, 3),
                             **extra})
    res.audio = audio.astype(np.float32)
    res.demodulation_status = ("GOOD" if conc > 0.15 else
                               "DEGRADED" if conc > 0.05 else "FAILED")
    res.timing_locked = res.carrier_locked = res.demodulation_status == "GOOD"
    res.warnings.append("analog transmission: audio demodulated, no bit "
                        "layer applies")
    return res


def _demod_fsk(x: np.ndarray, modulation: str, sps: float,
               res: DemodulationResult) -> DemodulationResult:
    order = int(modulation[0])
    k = int(np.log2(order))
    inst_f = np.diff(np.unwrap(np.angle(x))) / (2 * np.pi)
    # smooth over roughly half a symbol
    w = max(1, int(sps) // 2)
    inst_s = np.convolve(inst_f, np.ones(w) / w, mode="same")
    # tone centres from histogram
    lo, hi = np.percentile(inst_s, [1, 99])
    hist, edges = np.histogram(inst_s, bins=256, range=(lo, hi))
    hist = sig.medfilt(hist.astype(float), 5)
    peaks, _ = sig.find_peaks(hist, height=0.25 * hist.max(), distance=8)
    centers = np.sort(0.5 * (edges[peaks] + edges[peaks + 1]))
    if len(centers) != order:
        res.warnings.append(
            f"expected {order} FSK tones, found {len(centers)}; "
            "using theoretical spacing")
        rs = 1.0 / sps
        centers = (np.arange(order) - (order - 1) / 2) * rs
    # pick the sampling phase that minimises distance to tone centres
    n_int = max(1, int(round(sps)))
    best_phase, best_cost = 0, np.inf
    for p in range(n_int):
        sampled = inst_s[p::n_int]
        d = np.abs(sampled[:, None] - centers[None, :]).min(axis=1)
        c = float(d.mean())
        if c < best_cost:
            best_cost, best_phase = c, p
    sampled = inst_s[best_phase::n_int]
    dists = np.abs(sampled[:, None] - centers[None, :])
    idx = np.argmin(dists, axis=1)
    # tone index -> bits (MSB first)
    bits = np.zeros((len(idx), k), dtype=np.uint8)
    for j in range(k):
        bits[:, j] = (idx >> (k - 1 - j)) & 1
    # LLR per bit from tone-distance margin
    scale = np.diff(centers).mean() if len(centers) > 1 else 1.0
    llrs = np.zeros((len(idx), k), dtype=np.float32)
    for j in range(k):
        mask0 = np.array([(t >> (k - 1 - j)) & 1 == 0 for t in range(order)])
        d0 = dists[:, mask0].min(axis=1)
        d1 = dists[:, ~mask0].min(axis=1)
        llrs[:, j] = ((d1 - d0) / (scale + 1e-12)) * 4
    res.hard_bits = bits.ravel()
    res.llrs = llrs.ravel()
    res.symbols = (sampled / (scale + 1e-12)).astype(np.complex64)
    res.timing_locked = res.carrier_locked = bool(best_cost < 0.25 * scale)
    res.lock_metrics["tone_fit_cost"] = round(float(best_cost), 5)
    res.evm_percent = None
    return res
