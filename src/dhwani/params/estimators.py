"""Stage S4: physical parameter estimation.

Tiered strategy: cheap estimators first (spectral line, OBW,
M2M4), then a time-smoothed cyclic periodogram sweep for symbol rate
confirmation. Every estimate carries a confidence and its method name.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sig

from ..common.models import SignalParameters, Confidence, Verdict


# --------------------------------------------------------------------------
def occupied_bandwidth(x: np.ndarray, fraction: float = 0.99) -> dict:
    """ITU-R SM.443 99%-power bandwidth + -3dB bandwidth, normalised."""
    n = min(len(x), 1 << 20)
    nper = min(8192, n)
    f, p = sig.welch(x[:n], fs=1.0, nperseg=nper, return_onesided=False,
                     detrend=False)
    idx = np.argsort(f)
    f, p = f[idx], p[idx]
    # noise floor subtraction for a defensible OBW on noisy captures;
    # the floor tracks the bin-to-bin variability (few Welch averages
    # leave exponential noise residue above the bare median that would
    # otherwise dominate the 99% integral at low SNR)
    med = np.median(p)
    mad = np.median(np.abs(p - med))
    floor = med + 2.0 * 1.4826 * mad
    ps = np.clip(p - floor, 0, None)
    total = ps.sum()
    if total <= 0:
        return {"obw99": None, "obw3db": None, "f_center": 0.0}
    c = np.cumsum(ps) / total
    lo = f[np.searchsorted(c, (1 - fraction) / 2)]
    hi = f[min(np.searchsorted(c, 1 - (1 - fraction) / 2), len(f) - 1)]
    # smooth before the -3dB read: a single noisy bin must not define
    # the peak nor the crossing points
    ps_s = np.convolve(ps, np.ones(9) / 9.0, mode="same")
    peak = ps_s.max()
    above = f[ps_s > peak / 2]
    obw3 = float(above[-1] - above[0]) if len(above) else None
    centroid = float((f * ps).sum() / total)
    return {"obw99": float(hi - lo), "obw3db": obw3, "f_center": centroid,
            "psd_freq": f, "psd": 10 * np.log10(p + 1e-20)}


def m2m4_snr(x: np.ndarray) -> float:
    """Moment-based blind SNR estimate (GNU Radio snr_est_m2m4 form) for a
    constant-modulus signal (signal kurtosis 1); QAM is slightly
    underestimated - acceptable for a blind first pass."""
    x = x[: 1 << 20]
    m2 = float((np.abs(x) ** 2).mean())
    m4 = float((np.abs(x) ** 4).mean())
    # standard M2M4 for ka=1 (CM signal), kw=2 (complex Gaussian noise):
    inner = 2 * m2 * m2 - m4
    if inner <= 0:
        return -5.0
    s = np.sqrt(inner)
    n = m2 - s
    if n <= 0:
        return 40.0
    return float(10 * np.log10(s / n))


def carrier_offset(x: np.ndarray, orders=(2, 4, 8)) -> dict:
    """CFO from the strongest spectral line of x^M (classic M-power method).
    Also reports which order produced the line - a modulation-order hint."""
    n = min(len(x), 1 << 18)
    xx = x[:n] / (np.abs(x[:n]) + 1e-12)      # limiter improves the line
    best = {"cfo": 0.0, "order": 0, "line_snr": 0.0}
    for M in orders:
        y = xx ** M
        Y = np.abs(np.fft.fft(y * np.hanning(n)))
        Y[0] = 0
        k = int(np.argmax(Y))
        freqs = np.fft.fftfreq(n)
        med = np.median(Y)
        line_snr = float(Y[k] / (med + 1e-12))
        if line_snr > best["line_snr"]:
            best = {"cfo": float(freqs[k] / M), "order": M,
                    "line_snr": line_snr}
    best["confident"] = best["line_snr"] > 8.0
    return best


def symbol_rate(x: np.ndarray, min_rate: float = 1e-4,
                lags=(0, 1, 2, 4), obw99: float = None) -> dict:
    """Blind symbol-rate estimation via time-smoothed cyclic periodograms.

    The magnitude nonlinearity |x|^2 exposes a spectral line at the symbol
    rate for shaped linear modulations; summing cyclic periodograms of
    x(t)x*(t-d) over several delays d strengthens the line for
    low-excess-bandwidth signals. Returns candidates ranked by prominence.
    """
    n = min(len(x), 1 << 19)
    xx = np.asarray(x[:n], dtype=np.complex64)
    acc = None
    for d in lags:
        y = xx[d:] * np.conj(xx[:len(xx) - d]) if d else (np.abs(xx) ** 2).astype(np.complex64)
        m = len(y)
        Y = np.abs(np.fft.fft((y - y.mean()) * np.hanning(m), n))
        acc = Y if acc is None else acc[:len(Y)] + Y[:len(acc)]
    freqs = np.fft.fftfreq(len(acc))
    # the symbol rate of a linear modulation cannot sit far below its
    # occupied bandwidth; this guard rejects low-frequency envelope spurs
    # (prominent for QAM) without excluding real candidates
    if obw99:
        min_rate = max(min_rate, 0.25 * obw99)
    pos = (freqs > min_rate) & (freqs < 0.5)
    f_pos, a_pos = freqs[pos], acc[pos]
    if len(a_pos) == 0:
        return {"candidates": [], "confidence": 0.0}
    # peak prominence against a smoothed background; the median filter is
    # one-sided at the mask edge, which inflates prominence there and used
    # to elect a spurious "rate" exactly at 0.25*OBW - so the edge bins
    # are excluded from candidacy
    bg = sig.medfilt(a_pos, kernel_size=51)
    prom = a_pos / (bg + 1e-12)
    edge = min(26, len(prom) // 4)
    prom[:edge] = 0.0
    prom[-3:] = 0.0
    order = np.argsort(prom)[::-1]
    cands = []
    for k in order[:20]:
        f = float(f_pos[k])
        # skip harmonics of an already-kept candidate
        if any(abs(f - m * c["rate_norm"]) < 2.0 / n
               for c in cands for m in (2, 3, 4)):
            continue
        cands.append({"rate_norm": f, "prominence": float(prom[k])})
        if len(cands) >= 5:
            break
    conf = 0.0
    if cands:
        top = cands[0]["prominence"]
        conf = float(min(1.0, max(0.0, (top - 3) / 20)))
    return {"candidates": cands, "confidence": conf}


def fsk_tones(x: np.ndarray, max_tones: int = 8) -> dict:
    """Instantaneous-frequency histogram: tone count, spacing, and whether
    the signal looks like FSK at all.

    The decisive FSK evidence is the constant envelope (measured on our
    synthetic corpus: RRC-shaped PSK/QAM cv >= 0.27, FSK cv <= 0.13 down to
    ~12 dB SNR). The tone histogram then only determines the order; its
    smoothing window is chosen adaptively because the symbol rate is not
    yet known at this stage.
    """
    n = min(len(x), 1 << 18)
    xx = x[:n]
    amp_all = np.abs(xx)
    cv = float(amp_all.std() / (amp_all.mean() + 1e-12))
    if cv > 0.17:
        return {"is_fsk": False, "amplitude_cv": cv}
    ph = np.unwrap(np.angle(xx))
    inst_raw = np.diff(ph) / (2 * np.pi)
    amp = amp_all[1:]
    good = amp > 0.3 * np.median(amp)
    inst_raw = inst_raw[good]
    if len(inst_raw) < 1000:
        return {"is_fsk": False, "amplitude_cv": cv}

    best = None
    for w in (2, 4, 8, 16, 32):
        inst_f = np.convolve(inst_raw, np.ones(w) / w, mode="valid")
        lo, hi = np.percentile(inst_f, [1, 99])
        span = hi - lo
        lo -= 0.15 * span
        hi += 0.15 * span
        hist, edges = np.histogram(inst_f, bins=256, range=(lo, hi))
        hist_s = sig.medfilt(hist.astype(float), 5)
        peaks, _ = sig.find_peaks(hist_s, height=0.25 * hist_s.max(),
                                  distance=8)
        # rank only windows that resolve a valid M-FSK tone count: a window
        # longer than the symbol (w >= sps) merges neighbouring symbols
        # into spurious intermediate peaks
        if len(peaks) not in (2, 4, 8) or len(peaks) > max_tones:
            continue
        width = max(2, len(hist_s) // (4 * len(peaks)))
        in_peak = np.zeros(len(hist_s), dtype=bool)
        for pk in peaks:
            in_peak[max(0, pk - width):pk + width] = True
        frac = float(hist_s[in_peak].sum() / hist_s.sum())
        centers = 0.5 * (edges[peaks] + edges[peaks + 1])
        cand = {"n_tones": int(len(peaks)), "frac": frac,
                "centers": centers, "hist": hist_s, "edges": edges, "w": w}
        if best is None or frac > best["frac"]:
            best = cand
    if best is None or best["frac"] < 0.5:
        return {"is_fsk": False, "amplitude_cv": cv,
                "note": "constant envelope but no clean tone structure"}
    centers = np.sort(best["centers"])
    diffs = np.diff(centers)
    return {"is_fsk": True, "tone_count": best["n_tones"],
            "amplitude_cv": cv,
            "tone_freqs_norm": [float(c) for c in centers],
            "deviation_norm": float(diffs.mean()) if len(diffs) else 0.0,
            "mass_fraction": best["frac"],
            "smoothing_window": best["w"],
            "histogram": best["hist"].tolist(),
            "bin_centers": (0.5 * (best["edges"][:-1] + best["edges"][1:])).tolist()}


def fsk_symbol_rate(x: np.ndarray, min_rate: float = 1e-4) -> dict:
    """Symbol rate of an FSK signal from the spectral line of the tone
    *transition* process |d/dt inst_freq|."""
    n = min(len(x), 1 << 18)
    inst_f = np.diff(np.unwrap(np.angle(x[:n]))) / (2 * np.pi)
    d = np.abs(np.diff(inst_f))
    d -= d.mean()
    D = np.abs(np.fft.fft(d * np.hanning(len(d))))
    freqs = np.fft.fftfreq(len(d))
    pos = (freqs > min_rate) & (freqs < 0.45)
    if not pos.any():
        return {"rate_norm": None, "confidence": 0.0}
    f_pos, a_pos = freqs[pos], D[pos]
    bg = sig.medfilt(a_pos, kernel_size=51)
    prom = a_pos / (bg + 1e-12)
    # exclude the mask edges where the one-sided median filter inflates
    # prominence (same artifact as in symbol_rate)
    edge = min(26, len(prom) // 4)
    prom[:edge] = 0.0
    prom[-26:] = 0.0
    k = int(np.argmax(prom))
    # the transition process is an impulse train: harmonics can outscore
    # the fundamental after background normalisation, so prefer a
    # sub-harmonic when it carries a comparable line
    best_k = k
    for div in (4, 3, 2):
        f_sub = f_pos[k] / div
        j = int(np.argmin(np.abs(f_pos - f_sub)))
        lo, hi = max(0, j - 2), min(len(prom), j + 3)
        jj = lo + int(np.argmax(prom[lo:hi]))
        if prom[jj] > 0.4 * prom[k]:
            best_k = jj
            break
    conf = float(min(1.0, max(0.0, (prom[best_k] - 3) / 20)))
    if conf <= 0:
        return {"rate_norm": None, "confidence": 0.0}
    return {"rate_norm": float(f_pos[best_k]), "confidence": conf}


def ofdm_detect(x: np.ndarray, fft_sizes=(64, 128, 256, 512, 1024, 2048)) -> dict:
    """Multi-evidence OFDM detector.

    Cyclic-prefix correlation alone false-positives on oversampled
    single-carrier signals and slow FSK, so a detection requires ALL of:
    1. strong CP correlation (99th percentile of the lag-N product),
    2. a peaky (bursty) correlation profile,
    3. the peaks recurring at the OFDM symbol period nfft+cp
       (autocorrelation of the correlation profile),
    4. the same detection in both halves of the recording.
    Anything less is reported as insufficient evidence.
    """
    n = min(len(x), 1 << 18)
    xx = x[:n]
    p = float((np.abs(xx) ** 2).mean())
    best = {"detected": False, "corr": 0.0, "evidence": {}}
    for nfft in fft_sizes:
        if n < 6 * nfft:
            continue
        c = xx[nfft:] * np.conj(xx[:-nfft])
        for cp in (nfft // 4, nfft // 8):
            if cp < 16:
                continue
            k = np.ones(cp) / cp
            m = np.abs(np.convolve(c, k, mode="valid")) / (p + 1e-12)
            med = float(np.median(m))
            p99 = float(np.percentile(m, 99))
            peaky = (p99 - med) / (med + 1e-9)
            corr = p99
            # peakiness is reported but is NOT a discriminator: continuous
            # OFDM keeps the lag product elevated everywhere (low peaky),
            # while oversampled single carriers can be very peaky - the
            # symbol-period recurrence below separates them
            if corr < 0.45:
                continue
            # evidence 3: peak recurrence at the OFDM symbol period
            mz = m - m.mean()
            period = nfft + cp
            if len(mz) < 3 * period:
                continue
            num = float(np.mean(mz[:-period] * mz[period:]))
            # compare against an off-period lag as the null
            off_lag = period + period // 3
            den = float(np.mean(mz[:-off_lag] * mz[off_lag:]))
            var = float(np.mean(mz * mz)) + 1e-15
            period_score = (num - den) / var
            # evidence 4: both halves agree
            half = len(m) // 2
            c1 = float(np.percentile(m[:half], 99))
            c2 = float(np.percentile(m[half:], 99))
            halves = min(c1, c2) > 0.35
            detected = period_score > 0.3 and halves
            if corr > best["corr"] and detected:
                best = {"detected": True, "fft_size": nfft, "cp_length": cp,
                        "corr": corr,
                        "evidence": {"cp_correlation": round(corr, 3),
                                     "peakiness": round(peaky, 2),
                                     "period_score": round(period_score, 3),
                                     "halves_consistent": bool(halves)}}
            elif corr > best.get("corr", 0) and not best["detected"]:
                best = {"detected": False, "fft_size": nfft, "cp_length": cp,
                        "corr": corr,
                        "evidence": {"cp_correlation": round(corr, 3),
                                     "peakiness": round(peaky, 2),
                                     "period_score": round(period_score, 3),
                                     "halves_consistent": bool(halves),
                                     "verdict": "insufficient evidence"}}
    return best


# --------------------------------------------------------------------------
# ponytail: small fixed tables, extend when an off-air capture misses
SDR_SAMPLE_RATES = (32e3, 40e3, 48e3, 96e3, 192e3, 250e3, 1e6, 1.024e6,
                    1.2e6, 1.92e6, 2e6, 2.048e6, 2.4e6, 2.56e6, 3.2e6, 8e6,
                    10e6, 20e6)
STANDARD_SYMBOL_RATES = (300, 600, 1200, 2400, 4800, 9600, 19200, 38400,
                         57600, 76800, 115200, 50e3, 100e3, 125e3, 250e3,
                         270833, 500e3, 1e6)


def sample_rate_candidates(symbols_per_sample: float,
                           tol: float = 0.005) -> list:
    """Headerless IQ: the common SDR sample rates at which the measured
    symbol rate (per *recorded* sample) lands within `tol` of a standard
    baud rate. Several usually match (rates come in powers of two), so
    these are PROBABLE candidates with evidence, never the sample rate."""
    out = []
    for fs in SDR_SAMPLE_RATES:
        rs = symbols_per_sample * fs
        std = min(STANDARD_SYMBOL_RATES, key=lambda s: abs(rs - s) / s)
        err = abs(rs - std) / std
        if err <= tol:
            out.append({"sample_rate_hz": fs, "symbol_rate_hz": round(rs, 1),
                        "standard_symbol_rate": std,
                        "rel_error": round(err, 5), "verdict": "PROBABLE",
                        "method": "measured symbol rate equals a standard "
                                  "baud rate at a common SDR sample rate"})
    return sorted(out, key=lambda c: c["rel_error"])


def estimate_parameters(x: np.ndarray, config, sample_rate=None) -> SignalParameters:
    """Run the full S4 battery on a channelised baseband signal."""
    p = SignalParameters(sample_rate=sample_rate)

    bw = occupied_bandwidth(x, config.obw_fraction)
    p.obw99_norm = bw["obw99"]
    p.obw3db_norm = bw["obw3db"]
    p.confidences["obw"] = Confidence(0.9 if bw["obw99"] else 0.0,
                                      "cumulative PSD (ITU-R SM.443)").to_dict()

    cfo = carrier_offset(x)
    p.carrier_offset_norm = cfo["cfo"] if cfo["confident"] else bw["f_center"]
    p.confidences["cfo"] = Confidence(
        min(1.0, cfo["line_snr"] / 20) if cfo["confident"] else 0.3,
        f"x^{cfo['order']} spectral line" if cfo["confident"]
        else "spectral centroid").to_dict()

    p.snr_db = round(m2m4_snr(x), 1)
    p.confidences["snr"] = Confidence(0.7, "M2M4 moment estimator").to_dict()

    sr = symbol_rate(x, config.symbol_rate_min_norm, obw99=bw["obw99"])
    if sr["candidates"]:
        p.symbol_rate_norm = sr["candidates"][0]["rate_norm"]
        p.samples_per_symbol = 1.0 / p.symbol_rate_norm
        cands = [c["rate_norm"] for c in sr["candidates"][:5]]
        p.symbol_rate_candidates = list(cands)
        # bandwidth-implied candidates: a shaped linear modulation's
        # rate sits near OBW99/(1+rolloff) (clean SNR) and near the -3dB
        # bandwidth (noise-robust; OBW99 integrates the noise floor at
        # low SNR). A near-full band suggests a Nyquist-rate recording.
        # At low SNR data lines can outrank the true rate line, so
        # runner-up lines near an implied rate are promoted and the
        # implied rates themselves are appended as hypotheses of last
        # resort for the downstream receiver trials.
        implied = []
        if bw.get("obw99"):
            implied.append(bw["obw99"] / 1.35)
            if bw["obw99"] > 0.45:
                implied.append(0.5)
        if bw.get("obw3db"):
            implied.append(min(bw["obw3db"], 0.5))
        promoted, extras = [], []
        for imp in implied:
            near = [r for r in cands[1:]
                    if abs(r - imp) <= 0.12 * imp and r not in promoted]
            promoted.extend(near)
            if not near and all(abs(imp - e) > 0.05 * imp
                                for e in extras):
                extras.append(imp)
        rest = [r for r in cands[1:] if r not in promoted]
        seen, ordered = set(), []
        for r in cands[:1] + promoted + extras + rest:
            key = round(r, 4)
            if key not in seen:
                seen.add(key)
                ordered.append(r)
        p.symbol_rate_candidates = ordered
        p.confidences["symbol_rate"] = Confidence(
            sr["confidence"], "cyclic periodogram (delay-product sweep)").to_dict()
    else:
        p.confidences["symbol_rate"] = Confidence(
            0.0, "no cyclic feature found", Verdict.UNKNOWN).to_dict()

    fsk = fsk_tones(x)
    if fsk.get("is_fsk"):
        p.fsk_tone_count = fsk["tone_count"]
        p.fsk_deviation_norm = fsk["deviation_norm"]
        p.confidences["fsk"] = Confidence(
            fsk["mass_fraction"], "instantaneous-frequency histogram").to_dict()
        # FSK has a constant envelope, so the |x|^2 cyclic estimator above
        # is unreliable; measure the tone dwell rate instead.
        fr = fsk_symbol_rate(x, config.symbol_rate_min_norm)
        if fr["rate_norm"]:
            p.symbol_rate_norm = fr["rate_norm"]
            p.samples_per_symbol = 1.0 / fr["rate_norm"]
            p.confidences["symbol_rate"] = Confidence(
                fr["confidence"],
                "FSK transition-rate spectral line").to_dict()

    ofdm = ofdm_detect(x)
    if ofdm.get("detected"):
        p.ofdm_detected = True
        p.ofdm_fft_size = ofdm["fft_size"]
        p.ofdm_cp_length = ofdm["cp_length"]
        p.confidences["ofdm"] = Confidence(
            min(1.0, ofdm["corr"]), "cyclic-prefix autocorrelation").to_dict()

    # absolute values when the rate is known
    if sample_rate:
        if p.carrier_offset_norm is not None:
            p.carrier_offset_hz = p.carrier_offset_norm * sample_rate
        if p.obw99_norm is not None:
            p.obw99_hz = p.obw99_norm * sample_rate
        if p.symbol_rate_norm is not None:
            p.symbol_rate_hz = p.symbol_rate_norm * sample_rate
    # excess bandwidth (rolloff estimate) from OBW vs symbol rate
    if p.obw99_norm and p.symbol_rate_norm:
        p.excess_bandwidth = round(max(0.0, min(1.0,
            p.obw99_norm / p.symbol_rate_norm - 1.0)), 2)
    return p
