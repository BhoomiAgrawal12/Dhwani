"""Stage S3: frequency-shift a signal of interest to baseband, low-pass
filter and decimate. SciPy polyphase resampling (correct + fast enough for
offline files; a liquid-dsp fast path can be slotted behind this interface
later without touching callers)."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal as sig


def channelize(x: np.ndarray, segment, target_oversample: float = 8.0,
               max_samples: int = 1 << 23, trim: bool = True) -> dict:
    """Extract `segment` from wideband stream `x`.

    Returns dict with baseband samples plus a record of every operation so
    later measurements can be mapped back to absolute Hz."""
    s = np.asarray(x[segment.start_sample: segment.end_sample][:max_samples],
                   dtype=np.complex64)
    fc = segment.center_norm
    bw = max(segment.bandwidth_norm, 1e-4)

    # frequency shift to zero
    n = np.arange(len(s), dtype=np.float64)
    s = (s * np.exp(-2j * np.pi * fc * n)).astype(np.complex64)

    # decimate so that the occupied bandwidth is ~1/target_oversample of the
    # new rate (leaves room for pulse shaping skirts)
    decim_target = max(1.0, 1.0 / (bw * target_oversample))
    frac = Fraction(1, 1) / Fraction(decim_target).limit_denominator(64)
    up, down = frac.numerator, frac.denominator
    if down > 1 or up > 1:
        s = sig.resample_poly(s, up, down).astype(np.complex64)
    new_rate_norm = up / down            # new_rate = original * up/down

    # trim leading/trailing noise: the detector's time resolution is one
    # waterfall row, so the box usually includes noise-only padding that
    # would poison EVM and bit alignment downstream
    s, kept = _trim_burst(s) if trim else (s, (0, len(s)))

    return {"samples": s, "trimmed": kept,
            "freq_shift_norm": fc,
            "resample_up": up, "resample_down": down,
            "rate_ratio": new_rate_norm,
            "bandwidth_norm_at_new_rate": min(0.95, bw / new_rate_norm),
            "sample_rate": (segment.sample_rate * new_rate_norm
                            if segment.sample_rate else None)}


def _trim_burst(s: np.ndarray, margin: int = 32) -> tuple:
    """Keep the strongest contiguous high-envelope region of the extracted
    channel (dips shorter than 8 smoothing windows are bridged)."""
    if len(s) < 1024:
        return s, (0, len(s))
    p = np.abs(s) ** 2
    win = max(64, len(s) // 512)
    k = np.ones(win) / win
    sm = np.convolve(p, k, mode="same")
    hi = np.median(sm[sm > np.median(sm)])
    # halfway (in dB) between the noise floor and the burst, never below
    # -6 dB: a weak burst's noise lead-in sits near -6 dB and was kept.
    # Only with real contrast: a flat envelope would split on its wiggles.
    lo = np.percentile(sm, 10)
    thr = max(0.25 * hi, np.sqrt(lo * hi)) if hi > 2 * lo else 0.25 * hi
    on = np.r_[False, sm > thr, False]
    st = np.flatnonzero(~on[:-1] & on[1:])
    en = np.flatnonzero(on[:-1] & ~on[1:])
    if st.size == 0:
        return s, (0, len(s))
    # two packets in one detection box (EntrySat: different carriers, 0.56
    # s apart) cannot share a carrier lock: keep the stronger one
    # ponytail: one burst per box, split boxes in S2 if frames are lost
    runs = [[st[0], en[0]]]
    for a, b in zip(st[1:], en[1:]):
        if a - runs[-1][1] < 8 * win:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    a, b = max(runs, key=lambda r: sm[r[0]:r[1]].sum())
    a, b = max(0, int(a) - margin), min(len(s), int(b) + margin)
    return s[a:b], (a, b)
