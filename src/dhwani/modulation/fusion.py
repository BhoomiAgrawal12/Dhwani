"""Stage S5 driver: physical pre-checks -> two engines -> fusion voter.

Order of authority:
1. Hard physical constraints from S4 (FSK tone histogram, OFDM detection,
   constant-envelope test) can exclude classes outright.
2. Cumulant engine (explainable, no training data).
3. CVNet-RF (advisory; its 24 RadioML classes are mapped onto our class
   list and out-of-distribution mass is reported, not hidden).
Disagreement between engines is surfaced, never averaged away.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sig

from ..common.models import ModulationHypothesis
from ..demod.constellations import MOD_FAMILY
from ..demod.filters import rrc_taps
from ..demod.receiver import (_EVM_GATES, _coarse_cfo, _timing_recover,
                              demodulate)
from .cumulants import classify_cumulants
from .cvnet import classify_cvnet

_LINEAR_FAMILIES = ("psk", "oqpsk", "qam", "apsk", "ask")
# family representatives tried when the fused shortlist itself fails to
# lock: smeared clouds most often steal probability from these sparse
# tables, so they must always get a hearing
_TRIAL_FALLBACK = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM",
                   "4ASK", "16APSK"]
# M-PSK points sit on the 2M-PSK grid, so a 2M-PSK receiver also locks on
# M-PSK data at the same EVM; the lower order is trialled against it and
# wins when it explains the symbols as well (true 2M-PSK data fails the
# M-PSK trial: about twice the EVM)
_NESTED = {"32PSK": "16PSK", "16PSK": "8PSK", "8PSK": "QPSK"}


def _occupancy(symbols, candidate) -> float:
    """Normalised entropy of the constellation hit histogram.

    Sparse tables embed in dense ones up to gain (QPSK corners ARE four
    16QAM points), so nearest-point EVM alone cannot reject the denser
    table; a true signal exercises all of its points, an embedded subset
    leaves most of them empty."""
    from ..demod.constellations import CONSTELLATIONS
    table, _k = CONSTELLATIONS[candidate]
    t = np.asarray(table, dtype=np.complex128)
    t = t / (np.sqrt((np.abs(t) ** 2).mean()) + 1e-12)
    sset = np.asarray(symbols[:8192], dtype=np.complex128)
    sset = sset / (np.sqrt((np.abs(sset) ** 2).mean()) + 1e-12)
    idx = np.argmin(np.abs(sset[:, None] - t[None, :]), axis=1)
    hist = np.bincount(idx, minlength=len(t)).astype(np.float64)
    pr = hist / hist.sum()
    pr = pr[pr > 0]
    h = float(-(pr * np.log2(pr)).sum())
    return h / max(np.log2(len(t)), 1.0)


def _grid_structure(symbols, candidate) -> float:
    """Decision-directed residual-bias structure score in [0, 1].

    A true lock leaves residuals CENTRED on each constellation point
    (per-point mean near zero); a structureless cloud quantised onto the
    same table leaves residuals biased along the cloud's local density
    gradient inside every decision region. Unlike an off-grid-rotation
    baseline this cannot be fooled by dense rings whose angular pitch
    divides the rotation angle."""
    from ..demod.constellations import CONSTELLATIONS
    table, _k = CONSTELLATIONS[candidate]
    t = np.asarray(table, dtype=np.complex128)
    t = t / (np.sqrt((np.abs(t) ** 2).mean()) + 1e-12)
    sset = np.asarray(symbols[:4096], dtype=np.complex128)
    sset = sset / (np.sqrt((np.abs(sset) ** 2).mean()) + 1e-12)
    idx = np.argmin(np.abs(sset[:, None] - t[None, :]), axis=1)
    resid = sset - t[idx]
    # project out the common linear error (residual gain, rotation, DC):
    # those are receiver imperfections that scale with the point, not
    # evidence of a cloud; a cloud's density-gradient bias is nonlinear
    # in position and survives this projection
    A = np.vstack([t[idx], np.ones_like(idx, dtype=np.complex128)]).T
    coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
    resid = resid - A @ coef
    rms = float(np.sqrt((np.abs(resid) ** 2).mean())) + 1e-12
    biases = []
    for pt in np.unique(idx):
        r = resid[idx == pt]
        if len(r) >= 6:
            # noise-debiased: a centred Gaussian still shows |mean| of
            # about rms/sqrt(n), which must not count as cloud bias
            raw = float(np.abs(r.mean()))
            floor = float(np.sqrt((np.abs(r) ** 2).mean())) / np.sqrt(len(r))
            biases.append(max(0.0, raw - floor))
    if not biases:
        return 0.0
    bias_ratio = float(np.median(biases)) / rms
    return max(0.0, 1.0 - bias_ratio)


def _trial_one(x, candidate, sps, demod_cfg, cfo_norm=None):
    """Trial-demodulate one candidate with the real S6 receiver and
    score the lock against that constellation's own calibrated EVM gate.
    Raw nearest-point EVM always improves with table density, so EVM is
    only comparable across candidates after normalising by the gate; the
    occupancy factor rejects subset embeddings the EVM cannot see."""
    try:
        res = demodulate(x, candidate, sps, demod_cfg)
        # S6 is given S4's carrier estimate; a trial that fails on its own
        # retries with it (EntrySat: a 0.124 cycles/sample residual failed
        # BPSK, then S6 locked it). Only a retry: on a locking signal the
        # estimate can be a frame-structure line that invites an alias.
        if (res.demodulation_status == "FAILED" and cfo_norm and
                abs(cfo_norm) > 0.01):
            res2 = demodulate(x, candidate, sps, demod_cfg, cfo_norm=cfo_norm)
            if res2.demodulation_status != "FAILED":
                res = res2
    except Exception:
        return None
    evm = res.evm_percent
    gate = _EVM_GATES.get(candidate)
    locked = res.timing_locked and res.carrier_locked
    if (evm is None or gate is None or not locked or
            res.symbols is None or len(res.symbols) < 256):
        return {"status": res.demodulation_status, "evm": evm,
                "sps": round(float(sps), 3), "score": None}
    margin = gate[0] / max(float(evm), 0.1)
    bonus = {"GOOD": 1.0, "DEGRADED": 0.5}.get(res.demodulation_status, 0.0)
    occ = _occupancy(res.symbols, candidate)
    # off-grid-rotation baseline: a Gaussian blob sits at blob-level
    # nearest-point EVM on any dense grid (with full occupancy), so the
    # measured EVM only counts as evidence if rotating the symbols half
    # a symmetry sector off the grid makes the fit clearly worse
    structure = _grid_structure(res.symbols, candidate)
    if structure < 0.5:
        return {"status": res.demodulation_status, "evm": evm,
                "occupancy": round(occ, 3),
                "structure": round(structure, 3),
                "sps": round(float(sps), 3), "score": None}
    score = (bonus + min(3.0, margin)) * occ * min(1.0, structure / 0.8)
    return {"status": res.demodulation_status, "evm": evm,
            "occupancy": round(occ, 3), "structure": round(structure, 3),
            "sps": round(float(sps), 3), "score": round(float(score), 3)}

# CVNet (RadioML) label -> our label; None = out of our supported set
CVNET_MAP = {
    "BPSK": "BPSK", "QPSK": "QPSK", "8PSK": "8PSK",
    "16QAM": "16QAM", "64QAM": "64QAM", "OQPSK": "QPSK",
    "GMSK": "2FSK", "OOK": "OOK",
}


def _prepare_symbols(x: np.ndarray, sps: float) -> np.ndarray:
    """CFO-correct + matched filter + blind timing so the cumulant engine
    sees symbol-spaced samples. Tries the plausible M-power CFO
    corrections and keeps the most structured result."""
    x = np.asarray(x, dtype=np.complex128)
    best_syms, best_metric = None, -1.0
    for order, limiter in ((2, True), (4, True), (4, False), (8, True)):
        cfo = _coarse_cfo(x, order, limiter=limiter)
        y = x * np.exp(-2j * np.pi * cfo * np.arange(len(x)))
        mf = sig.fftconvolve(y, rrc_taps(max(2, int(round(sps))), 10, 0.35),
                             mode="same")
        syms, tone, _ = _timing_recover(mf, sps, max_symbols=20000)
        if len(syms) < 64:
            continue
        syms = syms / (np.sqrt((np.abs(syms) ** 2).mean()) + 1e-12)
        # structure metric: strongest M-power concentration
        sn = syms / (np.abs(syms) + 1e-12)
        metric = max(np.abs((sn ** m).mean()) for m in (2, 4, 8))
        metric += float(np.abs((syms ** 4).mean()))
        if metric > best_metric:
            best_metric, best_syms = metric, syms
    return best_syms


def classify_modulation(x: np.ndarray, params, config) -> ModulationHypothesis:
    """x: channelised baseband signal; params: SignalParameters from S4."""
    constraints = []
    engines = {}
    amp = np.abs(np.asarray(x[: 1 << 18]))
    env_cv = float(amp.std() / (amp.mean() + 1e-12))

    # ---- hard physical constraints -------------------------------------
    # FSK first: a slow FSK's tone dwell also produces bursty lag
    # correlation that can mimic an OFDM cyclic prefix.
    if params.fsk_tone_count:
        # h=0.5 CPM (MSK/GMSK) leaves a decisive x^2 line pair at
        # 2fc +- Rs/2; rectangular M-FSK does not (its pair sits at the
        # full tone spacing and the histogram already resolves it)
        from ..demod.receiver import _msk_squaring_lines
        rs2, _fc2, q2 = _msk_squaring_lines(np.asarray(x[: 1 << 17]))
        # h-index disambiguation: the x^2 line spacing measures h*Rs
        # products, so Rs must come independently from the transition
        # line. h=1 rectangular FSK has a sharp transition line and
        # rs2 = 2*Rs; h=0.5 GMSK has rs2 = Rs and a smeared transition
        # line. Only call GMSK when the x^2 pair is strong AND the h=1
        # relationship does not hold.
        rs_t = params.symbol_rate_norm
        is_h1_fsk = bool(rs_t and rs2 > 0 and
                         abs(rs2 - 2 * rs_t) < 0.3 * rs2)
        if q2 > 60 and not is_h1_fsk:
            return ModulationHypothesis(
                prediction="GMSK", confidence=round(min(1.0, q2 / 120), 3),
                alternatives=[["GMSK", round(min(1.0, q2 / 120), 3)],
                              ["2FSK", 0.2]],
                engine_predictions={"squaring_lines": ["GMSK", round(q2, 1)]},
                constraints_applied=[
                    "constant envelope with x^2 line pair at 2fc +- Rs/2 "
                    "(h=0.5 continuous-phase signature)"],
                in_distribution=True)
        label = f"{params.fsk_tone_count}FSK"
        conf = params.confidences.get("fsk", {}).get("value", 0.5)
        constraints.append(f"instantaneous-frequency histogram shows "
                           f"{params.fsk_tone_count} tones")
        return ModulationHypothesis(
            prediction=label, confidence=round(min(1.0, conf), 3),
            alternatives=[[label, round(min(1.0, conf), 3)]],
            engine_predictions={"fsk_histogram": [label, round(min(1.0, conf), 3)]},
            constraints_applied=constraints,
            in_distribution=label in ("2FSK", "4FSK"))

    sps = params.samples_per_symbol or 8.0

    # ---- engine A: cumulants -------------------------------------------
    syms = _prepare_symbols(x, sps)
    cum = None
    if syms is not None:
        cum = classify_cumulants(syms, params.snr_db)
        engines["cumulant"] = cum

    # ---- engine B: CVNet-RF --------------------------------------------
    cv = None
    if config.cvnet_enabled:
        cv = classify_cvnet(x, config.cvnet_checkpoint, config.cvnet_variant,
                            config.device, config.frame_size, config.max_frames)
        if cv and "probabilities" in cv:
            engines["cvnet"] = cv

    # ---- OFDM: only when no single-carrier structure was found ----------
    # (slow FSK tone dwell and single-carrier pulse autocorrelation both
    # mimic a cyclic prefix, so CP correlation alone is not sufficient)
    cum_top = max(cum["probabilities"].values()) if cum else 0.0
    # a strong single-carrier symbol-rate line is decisive AGAINST OFDM
    # (frame periodicity in single-carrier data can mimic cyclic-prefix
    # recurrence, but OFDM has no |x|^2 symbol-rate tone)
    sr_conf = params.confidences.get("symbol_rate", {}).get("value", 0)
    if params.ofdm_detected and env_cv > 0.35 and cum_top < 0.75 and \
            sr_conf < 0.5 and \
            (params.confidences.get("ofdm", {}).get("value", 0) > 0.6):
        return ModulationHypothesis(
            prediction="OFDM", confidence=params.confidences["ofdm"]["value"],
            alternatives=[["OFDM", params.confidences["ofdm"]["value"]]],
            engine_predictions={"cumulant": ["no clear constellation",
                                             round(cum_top, 3)]} if cum else {},
            constraints_applied=["cyclic-prefix autocorrelation",
                                 f"envelope cv {env_cv:.2f} consistent with OFDM",
                                 "no single-carrier cumulant structure"],
            in_distribution=False)

    # ---- fusion ---------------------------------------------------------
    classes = list(config.classes)
    fused = {c: 0.0 for c in classes}
    w_cum = config.fusion_weights.get("cumulant", 0.5)
    w_cv = config.fusion_weights.get("cvnet", 0.35)
    total_w = 0.0
    if cum:
        for c, p in cum["probabilities"].items():
            if c in fused:
                fused[c] += w_cum * p
        total_w += w_cum
    ood_mass = 0.0
    if cv and "probabilities" in cv:
        mapped = {c: 0.0 for c in classes}
        for label, p in cv["probabilities"].items():
            ours = CVNET_MAP.get(label)
            if ours and ours in mapped:
                mapped[ours] += p
            else:
                ood_mass += p
        s = sum(mapped.values())
        if s > 0.05:
            for c in mapped:
                fused[c] += w_cv * mapped[c] / s
            total_w += w_cv
    # constant envelope excludes QAM outright
    if env_cv < 0.15:
        for c in list(fused):
            if c.endswith("QAM"):
                fused[c] *= 0.05
        constraints.append(f"envelope cv {env_cv:.2f}: QAM excluded "
                           "(constant-envelope signal)")
    if total_w == 0:
        return ModulationHypothesis(prediction="UNKNOWN", confidence=0.0,
                                    constraints_applied=constraints,
                                    in_distribution=False)
    for c in fused:
        fused[c] /= total_w
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])
    top, top_p = ranked[0]

    # ---- receiver-trial arbitration -------------------------------------
    # Cumulant features are measured on crudely synchronised symbols, so a
    # smeared cloud can sit nearest to a lookalike dense constellation.
    # The decisive evidence is whether the real S6 receiver LOCKS: trial
    # the shortlist, score each candidate's EVM against its own gate, and
    # let lock quality overrule feature distance.
    trial_detail = {}
    if MOD_FAMILY.get(top) in _LINEAR_FAMILIES:
        from ..common.config import DemodConfig
        demod_cfg = DemodConfig()
        xs = np.asarray(x[: 1 << 16], dtype=np.complex128)
        # same carrier hint as S6, or a trial can fail what S6 then locks
        cfo = getattr(params, "carrier_offset_norm", None)
        shortlist = [c for c, _p in ranked
                     if MOD_FAMILY.get(c) in _LINEAR_FAMILIES][:3]

        def _sc(c):
            t = trial_detail.get(c)
            return -1.0 if not t or t["score"] is None else t["score"]

        for c in shortlist:
            trial_detail[c] = _trial_one(xs, c, sps, demod_cfg, cfo)
        best = max(trial_detail, key=_sc) if trial_detail else None
        if best is None or _sc(best) < 2.0:
            # nothing locked cleanly: give the family representatives a
            # hearing before concluding anything
            for c in _TRIAL_FALLBACK:
                if c in classes and c not in trial_detail:
                    trial_detail[c] = _trial_one(xs, c, sps, demod_cfg, cfo)
            best = max(trial_detail, key=_sc) if trial_detail else None
        if best is None or _sc(best) < 2.0:
            # representatives did not lock either: every remaining linear
            # class gets a hearing in fused-probability order (the true
            # class of a high-order signal, e.g. 32APSK, may sit outside
            # both the shortlist and the representative set)
            for c, _p in ranked:
                if MOD_FAMILY.get(c) in _LINEAR_FAMILIES and \
                        c not in trial_detail:
                    trial_detail[c] = _trial_one(xs, c, sps, demod_cfg, cfo)
                    if _sc(c) >= 2.0:
                        break
            best = max(trial_detail, key=_sc) if trial_detail else None
        if best is None or _sc(best) < 2.0:
            # still nothing: at low SNR the elected symbol-rate line is
            # sometimes a data line, so revisit the runner-up rate
            # candidates before concluding anything
            alt_sps = []
            for r in getattr(params, "symbol_rate_candidates", [])[1:6]:
                cand_sps = 1.0 / max(r, 1e-6)
                if 1.9 <= cand_sps <= 64 and \
                        abs(cand_sps - sps) > 0.25 and \
                        all(abs(cand_sps - a) > 0.25 for a in alt_sps):
                    alt_sps.append(cand_sps)
            for s_alt in alt_sps[:3]:
                for c in _TRIAL_FALLBACK[:5]:
                    if c not in classes:
                        continue
                    t = _trial_one(xs, c, s_alt, demod_cfg, cfo)
                    if t and (t["score"] or 0) > max(
                            0.0, _sc(c) if c in trial_detail else 0.0):
                        trial_detail[c] = t
                best = max(trial_detail, key=_sc) if trial_detail else None
                if best is not None and _sc(best) >= 2.0:
                    break
        while best in _NESTED and _NESTED[best] in classes and \
                _NESTED[best] not in trial_detail:
            sub = _NESTED[best]
            t_sub = trial_detail[sub] = _trial_one(xs, sub, sps, demod_cfg,
                                                   cfo)
            t_best = trial_detail[best]
            if not (t_sub and t_sub["score"] is not None and
                    t_sub["evm"] <= 1.05 * t_best["evm"]):
                break
            constraints.append(f"{sub} explains the symbols as well as "
                               f"{best} (EVM {t_sub['evm']}% vs "
                               f"{t_best['evm']}%): fewer points preferred")
            best = sub
        # OOK and BPSK share one bipolar table after S1, so the trial ties
        # on EVM and only OOK's looser gate separates them: name BPSK
        # whenever it passes its own trial.
        # ponytail: OOK is named only when BPSK fails; add an on/off
        # envelope test when a real OOK capture needs the name
        if best == "OOK" and "BPSK" in classes:
            if "BPSK" not in trial_detail:
                trial_detail["BPSK"] = _trial_one(xs, "BPSK", sps, demod_cfg,
                                                  cfo)
            if _sc("BPSK") >= 0.8:
                constraints.append("OOK and BPSK share one bipolar "
                                   "constellation after DC removal: BPSK "
                                   "named, OOK kept as the alternative")
                best = "BPSK"
        if best is not None and _sc(best) >= 0.8:
            if best != top:
                t_best = trial_detail[best]
                t_top = trial_detail.get(top)
                loser = ("fails its own EVM gate" if not t_top or
                         t_top["score"] is None else
                         f"locks worse relative to its gate "
                         f"(EVM {t_top['evm']}%)")
                constraints.append(
                    f"receiver trial overruled {top}: {best} locks at "
                    f"EVM {t_best['evm']}% ({t_best['status']}) while "
                    f"{top} {loser}")
            # probability mass only over candidates that clear the
            # acceptance bar: sub-threshold locks stay visible in the
            # trial detail but must not dilute the winner below the
            # UNKNOWN cutoff by sheer headcount
            scored = sorted((c for c in trial_detail if _sc(c) >= 0.8),
                            key=lambda c: (c != best, -_sc(c),
                                           -fused.get(c, 0.0)))
            exps_t = {c: float(np.exp(_sc(c))) for c in scored}
            z_t = sum(exps_t.values())
            trial_probs = {c: min(0.95, exps_t[c] / z_t) for c in scored}
            ranked = ([(c, trial_probs[c]) for c in scored] +
                      [(c, 0.0) for c, _p in ranked if c not in scored])
            top, top_p = ranked[0]
        else:
            constraints.append(
                "no shortlisted constellation locks in the receiver "
                "trial: single-carrier claim withheld")
            top_p = min(top_p, 0.2)

    engine_preds = {}
    if cum:
        cb = max(cum["probabilities"].items(), key=lambda kv: kv[1])
        engine_preds["cumulant"] = [cb[0], round(cb[1], 3)]
    if cv and "probabilities" in cv:
        vb = max(cv["probabilities"].items(), key=lambda kv: kv[1])
        engine_preds["cvnet"] = [vb[0], round(vb[1], 3)]
    agreement = True
    if len(engine_preds) == 2:
        a = engine_preds["cumulant"][0]
        b = CVNET_MAP.get(engine_preds["cvnet"][0], engine_preds["cvnet"][0])
        agreement = (a == b)

    hyp = ModulationHypothesis(
        prediction=top if top_p > 0.25 else "UNKNOWN",
        confidence=round(top_p if top_p > 0.25 else 0.0, 3),
        alternatives=[[c, round(p, 3)] for c, p in ranked[:5] if p > 0.01],
        engine_predictions=engine_preds,
        classifier_agreement=agreement,
        constraints_applied=constraints,
        in_distribution=(ood_mass < 0.5))
    if cum:
        hyp.engine_predictions["cumulant_features"] = dict(
            zip(cum["feature_names"], cum["features"]))
    if trial_detail:
        hyp.engine_predictions["receiver_trial"] = {
            c: t for c, t in trial_detail.items() if t is not None}
    return hyp
