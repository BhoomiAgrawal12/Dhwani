"""Ground-truth validation matrix across every supported modulation.

For each modulation x SNR (x impairments), a signal with full ground
truth is generated and measured on:
  1. blind classification (S5) - top-1 and top-2
  2. demodulation with the true modulation (S6) - status, EVM, BER
  3. the fully blind chain (S0-S11) - frame detection, CRC, payload trust

Verdicts per case:
  PASS      blind class top-1 correct AND BER < 2%
  DEGRADED  BER < 2% but classification missed top-1, or demod DEGRADED
  FAIL      unreliable bits (and honestly gated) or chain failure

Run:  python scripts/validate_matrix.py [--quick] [--seeds 1]
Writes results/validation_report.json (synthetic data; provenance,
per-case rows, confusion matrix, per-class accuracy).
"""
import argparse
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
logging.disable(logging.INFO)

from dhwani.common.config import Config
from dhwani.conditioning import condition
from dhwani.detection import detect_signals
from dhwani.channelization import channelize
from dhwani.params import estimate_parameters
from dhwani.modulation import classify_modulation
from dhwani.demod import demodulate
from dhwani.demod.constellations import CONSTELLATIONS, MOD_FAMILY, slice_symbols
from dhwani.pipeline import Analyzer
from dhwani.synth.factory import WaveformFactory
from provenance import write_result

# modulation -> SNR levels to test (family-appropriate: denser needs more)
MATRIX = {
    "BPSK": [8, 15], "QPSK": [12, 20], "OQPSK": [15, 22],
    "8PSK": [18, 25], "16PSK": [25, 32], "32PSK": [30, 37],
    "OOK": [10, 15], "4ASK": [18, 25], "8ASK": [24, 30],
    "16QAM": [18, 25], "32QAM": [21, 28], "64QAM": [24, 30],
    "128QAM": [28, 34], "256QAM": [31, 37],
    "16APSK": [20, 27], "32APSK": [24, 30], "64APSK": [27, 33],
    "128APSK": [32, 38],
    "2FSK": [10, 15], "4FSK": [14, 20], "GMSK": [10, 15],
    "OFDM": [20, 25],
    "FM": [15], "AM-DSB-WC": [15], "AM-DSB-SC": [15],
    "AM-SSB-WC": [15], "AM-SSB-SC": [15],
}
ANALOG = {"FM", "AM-DSB-WC", "AM-DSB-SC", "AM-SSB-WC", "AM-SSB-SC"}


def mid_window_ber(rx_bits, tx_bits, step, win=1200):
    if rx_bits is None or len(rx_bits) < 400:
        return 1.0
    start = (max(0, len(rx_bits) // 4) // step) * step
    seg = rx_bits[start:start + win]
    if len(seg) < 400 or len(tx_bits) <= len(seg):
        return 1.0
    best = 1.0
    for off in range(0, len(tx_bits) - len(seg), step):
        b = float((tx_bits[off:off + len(seg)] != seg).mean())
        if b < best:
            best = b
        if best < 0.003:
            break
    return best


def known_mod_ber(res, mod, tx_bits):
    fam = MOD_FAMILY.get(mod)
    if fam in ("fsk", "gmsk"):
        k = 2 if mod == "4FSK" else 1
        if res.hard_bits is None or not len(res.hard_bits):
            return 1.0
        return min(mid_window_ber(res.hard_bits, tx_bits, k),
                   mid_window_ber((1 - res.hard_bits).astype(np.uint8),
                                  tx_bits, k))
    if res.symbols is None or not len(res.symbols):
        return 1.0
    from dhwani.demod.constellations import symmetry_order
    table, k = CONSTELLATIONS[mod]
    n_rot = symmetry_order(mod)
    best = 1.0
    syms = res.symbols.astype(np.complex128)
    # OQPSK stagger pairing is an S7 ambiguity, resolved downstream
    variants = [syms] + ([syms.real + 1j * np.roll(syms.imag, k)
                          for k in (-1, 1)] if mod == "OQPSK" else [])
    for conj, v in [(c, v) for v in variants for c in (False, True)]:
        base = np.conj(v) if conj else v
        for r in range(n_rot):
            hard, _, _ = slice_symbols(base * np.exp(2j * np.pi * r / n_rot),
                                       mod, 0.05)
            best = min(best, mid_window_ber(hard, tx_bits, k))
            if best < 0.003:
                return best
    return best


CFO_NORM, PHASE, SPS = 0.004, 0.3, 8.0   # digital cases; analog: no offset
# stress axes at the flagship SNR (classification + demodulation only):
# a well-tuned capture has ~zero CFO; FSK is often sampled near 4 sps
AXES = {"CFO 0": lambda mod: mod not in ANALOG and mod != "OFDM",
        "4 sps": lambda mod: mod in ("2FSK", "4FSK", "GMSK")}
# --impairments: real-channel effects at the flagship SNR (digital only)
IMPAIRMENTS = {
    "multipath": {"multipath_taps": [1.0, 0.0, 0.35 + 0.2j]},
    "drift": {"cfo_drift_norm": 0.002},
    "clock 50ppm": {"clock_ppm": 50.0},
    "phase noise": {"phase_noise_std": 0.01},
    "interferer": {"interferer": {"freq_norm": 0.2, "rel_db": -10.0}},
}
SWEEP_SNRS = list(range(0, 31, 3))
FAMILY = {"psk": "PSK", "oqpsk": "PSK", "qam": "QAM", "apsk": "APSK",
          "ask": "ASK", "fsk": "FSK", "gmsk": "FSK"}


def wilson(k: int, n: int, z: float = 1.96) -> list:
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return [0.0, 0.0]
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def analog_verdict(top1: bool, status: str) -> tuple:
    """Analog cases demodulate with the true modulation supplied, so the
    demodulator alone cannot earn a PASS: the blind name must be right too.
    Returns (verdict, reason or None)."""
    if status == "FAILED":
        return "FAIL", "demodulation FAILED"
    if not top1:
        return "DEGRADED", ("blind classification missed (demodulates only "
                            "when the modulation is supplied)")
    return ("PASS", None) if status == "GOOD" else \
        ("DEGRADED", f"demod status {status}")


def run_case(mod, snr, seed, cfg, analyzer, full_chain, axis=None):
    fac = WaveformFactory(seed=seed)
    sps = 4.0 if axis == "4 sps" else SPS
    cfo = 0.0 if axis == "CFO 0" else CFO_NORM
    kwargs = dict(modulation=mod, sps=sps, snr_db=snr, n_frames=60)
    if mod not in ANALOG:
        kwargs.update(cfo_norm=cfo, phase_offset=PHASE)
    if axis in IMPAIRMENTS:
        kwargs["impair"] = IMPAIRMENTS[axis]
    iq, gt = fac.generate(**kwargs)
    row = {"modulation": mod, "snr_db": snr, "seed": seed, "sps": sps,
           "cfo_norm": kwargs.get("cfo_norm", 0.0), "axis": axis}
    t0 = time.perf_counter()

    x, _ = condition(iq)
    segs, _ = detect_signals(x, cfg.cfar)
    if not segs:
        row.update(verdict="FAIL", reason="not detected")
        return row
    ch = channelize(x, segs[0])
    p = estimate_parameters(ch["samples"], cfg.params)

    # 1. blind classification
    m = classify_modulation(ch["samples"], p, cfg.modulation)
    # informationally inseparable pairs: OQPSK is QPSK with a stagger the
    # classifier cannot see pre-carrier; OOK collapses onto bipolar BPSK
    # once S1 removes the DC term (so OOK->BPSK is accepted, but BPSK->OOK
    # is a real miss); GMSK/2FSK share the discriminator view
    equiv = {("QPSK", "OQPSK"), ("OQPSK", "QPSK"), ("GMSK", "2FSK"),
             ("2FSK", "GMSK"), ("OOK", "BPSK")}
    top1 = m.prediction == mod or (mod, m.prediction) in equiv
    top2 = top1 or any(a[0] == mod for a in m.alternatives[:2])
    row["classified"] = m.prediction
    row["confidence"] = round(float(m.confidence), 4)
    row["class_top1"] = bool(top1)
    row["class_top2"] = bool(top2)

    # analog and OFDM: classification + (for analog) demod status is the test
    if mod == "OFDM":
        row["verdict"] = "PASS" if top1 else "FAIL"
        row["elapsed_s"] = round(time.perf_counter() - t0, 1)
        return row
    if mod in ANALOG:
        res = demodulate(ch["samples"], mod, 8.0, cfg.demod)
        row["demod_status"] = res.demodulation_status
        row["audio_quality"] = res.lock_metrics.get(
            "audio_tonal_concentration")
        row["verdict"], reason = analog_verdict(top1, res.demodulation_status)
        if reason:
            row["reason"] = reason
        row["elapsed_s"] = round(time.perf_counter() - t0, 1)
        return row

    # 2. demodulation with the true modulation
    sps_est = 1.0 / p.symbol_rate_norm if p.symbol_rate_norm else sps
    if not (2 <= sps_est <= 64):
        sps_est = sps
    fam = MOD_FAMILY.get(mod)
    res = demodulate(ch["samples"], mod, sps_est, cfg.demod,
                     cfo_norm=p.carrier_offset_norm
                     if fam in ("psk", "oqpsk") else None)
    tx = np.array(gt.info_bits, dtype=np.uint8)
    ber = known_mod_ber(res, mod, tx)
    row.update(demod_status=res.demodulation_status,
               evm_percent=res.evm_percent, ber=round(float(ber), 4),
               cfo_error=abs((res.cfo_applied_norm or 0) -
                             cfo + segs[0].center_norm))

    # 3. full blind chain (frames/CRC/payload) on the flagship SNR only
    if full_chain:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            pth = os.path.join(td, "m.iq")
            iq.astype(np.complex64).tofile(pth)
            fr = analyzer.analyze(pth, sample_rate=1e6, datatype="complex64",
                                  overrides={"modulation": mod})
            row["frame_detected"] = bool(fr.frames)
            row["crc_pass"] = bool(fr.frames and fr.frames.crc and
                                   fr.frames.crc["pass_fraction"] > 0.9)
            pi = fr.payload_intelligence or {}
            row["payload_trust"] = (pi.get("summary") or {}).get("trust")
            # ground-truth payload equality: the extracted bytes must be
            # a contiguous substring of the transmitted payload (blind
            # acquisition legitimately loses frames at the burst edges)
            truth_payload = (bytes.fromhex("".join(gt.payloads))
                             if getattr(gt, "payloads", None) else b"")
            extracted = fr.payload.data if fr.payload else b""
            row["payload_match"] = bool(
                truth_payload and extracted and
                extracted in truth_payload and
                len(extracted) >= len(truth_payload) // 2)

    good_ber = ber < 0.02
    if good_ber and top1 and res.demodulation_status == "GOOD":
        row["verdict"] = "PASS"
    elif good_ber:
        row["verdict"] = "DEGRADED"
        row["reason"] = ("classification missed" if not top1 else
                         f"demod status {res.demodulation_status}")
    else:
        row["verdict"] = "FAIL" if res.demodulation_status != "FAILED" \
            else "FAIL(gated)"
        row["reason"] = f"BER {ber:.3f}"
    row["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="one SNR per modulation, no full-chain runs")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--impairments", action="store_true",
                    help="add the multipath / drift / clock / phase-noise / "
                         "interferer axes at the flagship SNR")
    ap.add_argument("--snr-sweep", action="store_true",
                    help="also sweep every digital modulation over 0-30 dB "
                         "(classification + demodulation), per-family curves")
    args = ap.parse_args()
    axes = dict(AXES)
    if args.impairments:
        axes.update({k: (lambda mod: mod not in ANALOG and mod != "OFDM")
                     for k in IMPAIRMENTS})
    cfg = Config()
    cfg.modulation.cvnet_enabled = False       # deterministic + fast
    analyzer = Analyzer(cfg, use_cache=False)
    rows = []
    for mod, snrs in MATRIX.items():
        snr_list = snrs[-1:] if args.quick else snrs
        for snr in snr_list:
            for seed in range(7, 7 + args.seeds):
                full = (not args.quick) and snr == snrs[-1] and \
                    mod not in ANALOG and mod != "OFDM"
                cases = [(None, full)] + [(a, False) for a, on in axes.items()
                                          if on(mod) and snr == snrs[-1]]
                for axis, f in cases:
                    row = run_case(mod, snr, seed, cfg, analyzer, f, axis)
                    rows.append(row)
                    _print_row(row)
    n = len(rows)

    n_pass = sum(r["verdict"] == "PASS" for r in rows)
    n_deg = sum(r["verdict"] == "DEGRADED" for r in rows)
    n_fail = n - n_pass - n_deg
    summary = {"total": n, "pass": n_pass, "degraded": n_deg, "fail": n_fail}
    print(f"\nTOTAL: {n_pass} PASS, {n_deg} DEGRADED, {n_fail} FAIL of {n}")
    sweep = []
    if args.snr_sweep:
        for mod in MATRIX:
            if mod in ANALOG or mod == "OFDM":
                continue
            for snr in SWEEP_SNRS:
                for seed in range(7, 7 + args.seeds):
                    r = run_case(mod, snr, seed, cfg, analyzer, False)
                    sweep.append({k: r.get(k) for k in (
                        "modulation", "snr_db", "seed", "class_top1", "confidence",
                        "ber", "demod_status")})
                print(f"sweep {mod:8s} {snr:3d} dB done", flush=True)
    path = write_result("validation_report", {
        "settings": {"quick": args.quick, "seeds": list(range(7, 7 + args.seeds)),
                     "sps": SPS, "n_frames": 60, "cvnet": False,
                     "cfo_norm": CFO_NORM, "phase_offset": PHASE,
                     "axes": list(axes), "snr_sweep": SWEEP_SNRS
                     if args.snr_sweep else None},
        "summary": summary, **classification_scores(rows),
        **({"snr_curves": snr_curves(sweep), "sweep_cases": sweep}
           if sweep else {}),
        "cases": rows})
    print("report:", path)


def _print_row(row):
    print(f"{row['modulation']:10s} SNR{row['snr_db']:3.0f} "
          f"{(row['axis'] or ''):6s}: "
          f"class={str(row.get('classified')):9s}"
          f"{'+' if row.get('class_top1') else '-'} "
          f"status={str(row.get('demod_status', '-')):8s} "
          f"BER={str(row.get('ber', '-')):8s} "
          f"CRC={str(row.get('crc_pass', '-')):5s} "
          f"pay={str(row.get('payload_match', '-')):5s} "
          f"-> {row['verdict']}", flush=True)


def classification_scores(rows: list) -> dict:
    """4.D: confusion matrix (true -> predicted counts) and per-class top-1 /
    top-2 accuracy of blind S5 classification."""
    confusion, per_class = {}, {}
    for r in rows:
        if "classified" not in r:
            continue
        t, p = r["modulation"], r["classified"]
        if r.get("axis"):
            t = f"{t} ({r['axis']})"
        confusion.setdefault(t, {})
        confusion[t][p] = confusion[t].get(p, 0) + 1
        c = per_class.setdefault(t, {"n": 0, "top1": 0, "top2": 0})
        c["n"] += 1
        c["top1"] += bool(r.get("class_top1"))
        c["top2"] += bool(r.get("class_top2"))
    for c in per_class.values():
        c["top1_accuracy"] = round(c["top1"] / c["n"], 3)
        c["top2_accuracy"] = round(c["top2"] / c["n"], 3)
        c["top1_ci95"] = wilson(c["top1"], c["n"])
    return {"confusion_matrix": confusion, "per_class": per_class}


def snr_curves(sweep: list) -> dict:
    """Per modulation family and SNR: blind top-1 accuracy with its 95%
    interval, and the median known-modulation BER."""
    from dhwani.demod.constellations import MOD_FAMILY
    cells = {}
    for r in sweep:
        fam = FAMILY.get(MOD_FAMILY.get(r["modulation"]), "other")
        cells.setdefault(fam, {}).setdefault(r["snr_db"], []).append(r)
    out = {}
    for fam, by_snr in cells.items():
        out[fam] = []
        for snr in sorted(by_snr):
            rs = by_snr[snr]
            k = sum(bool(r["class_top1"]) for r in rs)
            bers = [r["ber"] for r in rs if r.get("ber") is not None]
            out[fam].append({"snr_db": snr, "n": len(rs),
                             "top1_accuracy": round(k / len(rs), 3),
                             "top1_ci95": wilson(k, len(rs)),
                             "median_ber": round(float(np.median(bers)), 4)
                             if bers else None})
    return out


if __name__ == "__main__":
    main()
