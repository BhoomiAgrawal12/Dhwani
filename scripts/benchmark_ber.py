"""Decoding proof: bit error rate vs Eb/N0 through Dhwani's own receiver
-> results/ber_report.json (synthetic data).

For every Eb/N0 point (0 to 10 dB in 1 dB steps) and every chain

  uncoded     QPSK, hard decisions
  conv_k7     convolutional K=7 (171,133) rate 1/2, soft Viterbi
  ldpc_648    LDPC 802.11n n=648 rate 1/2, the Layered Decoder

each frame is built and received like this:

  random information bits -> encode -> burst = [random preamble | 64-bit
  sync word | coded payload | random tail] -> QPSK (S6's Gray table) ->
  RRC pulse shaping (8 samples/symbol, rolloff 0.35: the factory's filter)
  -> carrier offset, random carrier phase, random fractional timing offset
  -> AWGN -> noise-only padding before and after, so S2 has a burst to find.

The noise is set from the burst's measured energy per symbol Es (mean
|x|^2 times the samples per symbol) so that Es/N0 = Eb/N0 + 10 log10(2 R)
exactly, R = information bits / coded bits of the payload (the preamble,
sync word and tail are overhead and carry no Eb).

The receiver is the stage calls pipeline.py makes: S1 condition, S2
detect_signals (segment 0, the strongest), S3 channelize, S4
estimate_parameters, S6 demodulate with S4's samples per symbol and
carrier offset.  The modulation is pinned to QPSK: S5 classification is
not what this measures.  Frame sync then uses the known sync word: it is
correlated at every bit offset of the S7 ambiguity streams (4 rotations x
conjugation, enumerate_ambiguities without differential decoding) and the
best stream/offset is accepted when at most a quarter of its bits are
wrong.  That stream's bits and LLRs for the payload go to the decoder, so
the curve measures demodulation plus decoding only, not blind
identification.

A frame is a LOCK FAILURE, not bit errors, when S2 finds nothing, S6
says FAILED (its quality gate withholds the bits, as in the pipeline) or
the sync word is not found.  BER counts the information bits of locked
frames only (zero errors: BER 0 with the rule-of-three 95% upper bound
3/bits); p_lock = locked / frames with its Wilson 95% interval.  "theory"
is uncoded Gray QPSK, 0.5 erfc(sqrt(Eb/N0)).

Every frame has its own seed, (seed, chain, Eb/N0 in tenths of a dB,
frame), so a frame is the same whatever --frames or --ebn0 says.

Run:  python scripts/benchmark_ber.py [--frames N] [--quick] [--seed S]
                                      [--ebn0 DB ...] [--out-dir DIR]
"""
import argparse
import logging
import math
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
logging.disable(logging.INFO)

from dhwani.bits.ambiguity import enumerate_ambiguities
from dhwani.channelization import channelize
from dhwani.common.config import Config
from dhwani.conditioning import condition
from dhwani.demod import demodulate
from dhwani.demod.constellations import bits_to_iq_symbols
from dhwani.detection import detect_signals
from dhwani.fec.conv import ConvCode, conv_encode, viterbi_decode
from dhwani.fec.registry import ldpc_entry
from dhwani.params import estimate_parameters
from dhwani.synth import impairments as imp
from dhwani.synth.factory import WaveformFactory
import provenance
from provenance import write_result
from validate_matrix import wilson

EBN0_DB = list(range(0, 11))
QUICK_EBN0_DB = [4, 10]
QUICK_FRAMES = 2
# 11 points x 3 chains x 80 frames at the measured 0.33-0.46 s per frame
# (frames S2 misses take ~0.07 s) is a full run of about 15 minutes, and
# ~400k uncoded bits per point resolve BER down to ~1e-5
DEFAULT_FRAMES = 80
SPS = 8                 # the factory's and the bit-layer matrix's default
ROLLOFF = 0.35          # the factory's RRC rolloff (S6's matched filter too)
CFO_NORM = 0.004        # the carrier offset validate_matrix/bitlayer use
# S6 drops its first min(n/10, 500) symbols while its loops settle; 640
# keeps the sync word clear of them with room for S3's burst-edge trim
PREAMBLE_SYMBOLS = 640
# a guard after the payload: the burst edge S3 trims and the matched
# filter's tail fall on it, not on payload symbols
TAIL_SYMBOLS = 64
PAD_SAMPLES = 4000      # noise-only samples each side (S2's floor and box)
# the CCSDS 64-bit attached sync marker (rate-1/2 turbo codes): a fixed
# word with low autocorrelation sidelobes
SYNC_WORD = 0x034776C7272895B0
SYNC_BITS = np.array([(SYNC_WORD >> (63 - i)) & 1 for i in range(64)],
                     dtype=np.uint8)
SYNC_MAX_ERRORS = len(SYNC_BITS) // 4
LDPC_NAME = "802.11n 648 r1/2"
# every chain sends about the same payload in channel symbols (8 LDPC
# codewords at rate 1/2), so S2-S6 see bursts of one length: a shorter
# burst reads a worse EVM from the noise S3's trim box keeps around it
PAYLOAD_SYMBOLS = 2592
CONV = ConvCode(7, (0o171, 0o133))


def chains() -> dict:
    """id -> label, code, decoder, info_bits (per frame), encode(info),
    decode(bits, llrs, n)."""
    ldpc = ldpc_entry({"name": LDPC_NAME})["code"]
    return {
        "uncoded": {
            "label": "uncoded QPSK", "code": None,
            "decoder": "hard decisions", "info_bits": 2 * PAYLOAD_SYMBOLS,
            "encode": lambda u: u,
            "decode": lambda b, llr, n: b[:n]},
        "conv_k7": {
            "label": "conv K=7 (171,133), soft Viterbi",
            "code": "conv K7 r1/2 (171,133), terminated",
            "decoder": "soft-decision Viterbi on the S6 LLRs",
            "info_bits": PAYLOAD_SYMBOLS,      # + K-1 tail: 2598 symbols
            "encode": lambda u: conv_encode(u, CONV, terminate=True),
            "decode": lambda b, llr, n: viterbi_decode(b, CONV, soft=llr)[:n]},
        "ldpc_648": {
            "label": "LDPC 802.11n n=648 r1/2, layered",
            "code": LDPC_NAME,
            "decoder": "Layered Decoder (offset min-sum) on the S6 LLRs",
            "info_bits": 2 * PAYLOAD_SYMBOLS // ldpc.n * ldpc.k,  # 8 words
            "encode": ldpc.encode,
            "decode": lambda b, llr, n: ldpc.decode_stream(b, llrs=llr)[0][:n]},
    }


def make_frame(chain: dict, ebn0_db: float, rng, n_info: int) -> tuple:
    """(capture iq, information bits, coded payload length in bits,
    Es/N0 dB, code rate)."""
    info = rng.integers(0, 2, n_info).astype(np.uint8)
    coded = np.asarray(chain["encode"](info), dtype=np.uint8)
    rate = n_info / len(coded)
    rand = lambda n: rng.integers(0, 2, n).astype(np.uint8)
    bits = np.concatenate([rand(2 * PREAMBLE_SYMBOLS), SYNC_BITS, coded,
                           rand(2 * TAIL_SYMBOLS + len(coded) % 2)])
    # the factory's pulse shaping, so the filter is the one every other
    # synthetic result was measured with
    x = WaveformFactory(seed=0)._pulse_shape(
        bits_to_iq_symbols(bits, "QPSK"), SPS, ROLLOFF)
    x = imp.freq_offset(x, CFO_NORM, rng.uniform(0, 2 * np.pi))
    x = imp.timing_offset(x, rng.uniform(0, 1))
    es = float(np.mean(np.abs(x) ** 2)) * SPS
    esn0_db = ebn0_db + 10 * math.log10(2 * rate)
    n0 = es / 10 ** (esn0_db / 10)
    noise = lambda n: rng.normal(0, math.sqrt(n0 / 2), (n, 2)) @ np.array([1, 1j])
    iq = np.concatenate([noise(PAD_SAMPLES), x + noise(len(x)),
                         noise(PAD_SAMPLES)])
    return iq.astype(np.complex64), info, len(coded), esn0_db, rate


def receive(iq: np.ndarray, cfg) -> tuple:
    """S1-S6 as pipeline.py calls them, S5 pinned to QPSK.  Returns (the
    DemodulationResult, None) or (None, lock-failure reason)."""
    x, _ = condition(iq)
    segs, _ = detect_signals(x, cfg.cfar)
    if not segs:
        return None, "no_detection"
    ch = channelize(x, segs[0])
    p = estimate_parameters(ch["samples"], cfg.params,
                            sample_rate=ch["sample_rate"])
    demod = demodulate(ch["samples"], "QPSK", p.samples_per_symbol or 8.0,
                       cfg.demod, cfo_norm=p.carrier_offset_norm)
    if (demod.demodulation_status == "FAILED" or demod.symbols is None
            or not len(demod.symbols)):
        return None, "demod_failed"
    return demod, None


def find_sync(streams: list, sync: np.ndarray, n_payload: int):
    """Known-word frame sync over the S7 streams: the word is compared at
    every bit offset that leaves room for the whole payload after it, and
    the stream/offset with the fewest disagreeing bits wins.  Returns
    (stream index, payload start, sync bit errors), or None when even the
    best has more than a quarter of the word wrong."""
    L, best = len(sync), None
    for i, s in enumerate(streams):
        n_pos = len(s.bits) - L - n_payload + 1
        if n_pos <= 0:
            continue
        win = np.lib.stride_tricks.sliding_window_view(
            s.bits[:n_pos + L - 1], L)
        errors = np.count_nonzero(win != sync, axis=1)
        j = int(np.argmin(errors))
        if best is None or errors[j] < best[2]:
            best = (i, j + L, int(errors[j]))
    if best is None or best[2] > L // 4:
        return None
    return best


def run_frame(chain: dict, ebn0_db: float, rng, cfg) -> dict:
    n_info = chain["info_bits"]
    iq, info, n_coded, esn0_db, rate = make_frame(chain, ebn0_db, rng, n_info)
    demod, failure = receive(iq, cfg)
    out = {"esn0_db": esn0_db, "rate": rate, "n_coded": n_coded,
           "failure": failure, "errors": 0}
    if demod is None:
        return out
    streams = enumerate_ambiguities(demod.symbols, "QPSK",
                                    include_differential=False)
    hit = find_sync(streams, SYNC_BITS, n_coded)
    if hit is None:
        out["failure"] = "no_sync"
        return out
    i, start, _ = hit
    s = streams[i]
    decoded = chain["decode"](s.bits[start:start + n_coded],
                              s.llrs[start:start + n_coded], n_info)
    out["errors"] = int(np.count_nonzero(decoded != info))
    return out


def _sig(v: float, digits: int = 6) -> float:
    return float(f"{v:.{digits}g}")


def point_summary(ebn0_db: float, frames: list, n_info: int,
                  seconds: float) -> dict:
    locked = [f for f in frames if f["failure"] is None]
    reasons = {"no_detection": 0, "demod_failed": 0, "no_sync": 0}
    for f in frames:
        if f["failure"]:
            reasons[f["failure"]] += 1
    bits = n_info * len(locked)
    errors = sum(f["errors"] for f in locked)
    return {"ebn0_db": ebn0_db, "esn0_db": round(frames[0]["esn0_db"], 3),
            "frames": len(frames), "locked": len(locked),
            "lock_failures": len(frames) - len(locked),
            "lock_failure_reasons": reasons,
            "p_lock": round(len(locked) / len(frames), 4),
            "p_lock_ci95": wilson(len(locked), len(frames)),
            "bits": bits, "bit_errors": errors,
            "ber": _sig(errors / bits) if bits else None,
            "ber_upper_95": _sig(3 / bits) if bits and not errors else None,
            "elapsed_s": round(seconds, 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=None,
                    help=f"frames per chain and Eb/N0 point "
                         f"(default {DEFAULT_FRAMES})")
    ap.add_argument("--quick", action="store_true",
                    help=f"Eb/N0 {QUICK_EBN0_DB} dB only, {QUICK_FRAMES} "
                         "frames per chain (smoke test)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ebn0", type=float, nargs="+", default=None,
                    help="Eb/N0 points in dB (default 0..10 in 1 dB steps)")
    ap.add_argument("--out-dir", default=None,
                    help="directory to write ber_report.json (default results)")
    args = ap.parse_args(argv)
    if args.out_dir is not None:
        out = (args.out_dir if os.path.isabs(args.out_dir)
               else os.path.join(ROOT, args.out_dir))
        provenance.RESULTS = os.path.abspath(out)
    points = args.ebn0 or (QUICK_EBN0_DB if args.quick else EBN0_DB)
    points = [int(x) if float(x).is_integer() else float(x) for x in points]
    frames = args.frames or (QUICK_FRAMES if args.quick else DEFAULT_FRAMES)
    cfg = Config()
    t_start = time.perf_counter()
    report = {}
    for ci, (cid, chain) in enumerate(chains().items()):
        pts, total_s, rate, n_coded = [], 0.0, None, None
        for x in points:
            t0 = time.perf_counter()
            rows = []
            for k in range(frames):
                rng = np.random.default_rng(
                    [args.seed, ci, int(round(10 * x)) % 2 ** 32, k])
                with warnings.catch_warnings():
                    # S4's median filter warns on short spectra; not a result
                    warnings.simplefilter("ignore", UserWarning)
                    rows.append(run_frame(chain, x, rng, cfg))
            dt = time.perf_counter() - t0
            total_s += dt
            rate, n_coded = rows[0]["rate"], rows[0]["n_coded"]
            p = point_summary(x, rows, chain["info_bits"], dt)
            pts.append(p)
            ber = ("-" if p["ber"] is None else
                   f"< {p['ber_upper_95']:.1e} (0 errors)" if not p["bit_errors"]
                   else f"{p['ber']:.2e}")
            print(f"{chain['label']:36s} {x:5.1f} dB: locked "
                  f"{p['locked']}/{p['frames']}, BER {ber} ({dt:.1f} s)",
                  flush=True)
        report[cid] = {
            "label": chain["label"], "code": chain["code"],
            "decoder": chain["decoder"], "code_rate": _sig(rate),
            "info_bits_per_frame": chain["info_bits"],
            "coded_bits_per_frame": n_coded,
            "seconds_per_frame": round(total_s / (frames * len(points)), 3),
            "points": pts}
    theory = [{"ebn0_db": x,
               "ber": _sig(0.5 * math.erfc(math.sqrt(10 ** (x / 10))))}
              for x in points]
    path = write_result("ber_report", {
        "settings": {
            "ebn0_db": points, "frames": frames, "quick": args.quick,
            "seed": args.seed,
            "frame_seed": "numpy default_rng([seed, chain index, "
                          "round(10 * Eb/N0) mod 2^32, frame])",
            "modulation": "QPSK (pinned at S6)", "samples_per_symbol": SPS,
            "rolloff": ROLLOFF, "cfo_norm": CFO_NORM,
            "carrier_phase": "uniform [0, 2 pi) per frame",
            "timing_offset_samples": "uniform [0, 1) per frame",
            "payload_symbols": PAYLOAD_SYMBOLS,
            "preamble_symbols": PREAMBLE_SYMBOLS,
            "sync_word_hex": f"{SYNC_WORD:016x}",
            "sync_max_errors": SYNC_MAX_ERRORS,
            "tail_symbols": TAIL_SYMBOLS, "noise_pad_samples": PAD_SAMPLES,
            "esn0": "Eb/N0 + 10 log10(2 R) from the burst's measured "
                    "energy per symbol",
            "receiver": "S1 condition, S2 detect_signals (segment 0), S3 "
                        "channelize, S4 estimate_parameters, S6 demodulate "
                        "(QPSK, S4 samples/symbol and carrier offset)",
            "frame_sync": "known sync word over the 8 S7 streams "
                          "(rotations x conjugation, no differential)",
            "lock_failure": "S2 detected nothing, S6 FAILED, or no sync word "
                            f"with <= {SYNC_MAX_ERRORS} of {len(SYNC_BITS)} "
                            "bits wrong"},
        "chains": report,
        "theory": {"chain": "uncoded", "formula": "0.5 erfc(sqrt(Eb/N0))",
                   "points": theory},
        "elapsed_s": round(time.perf_counter() - t_start, 1)},
        data="synthetic (QPSK frames through Dhwani's S1-S6 receiver, "
             "scripts/benchmark_ber.py)")
    print("report:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
