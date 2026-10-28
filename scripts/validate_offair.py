"""Score real (off-air) recordings against analyst-known parameters.

The recordings are not in the repo: supply licensed captures and a
manifest (default data/offair/manifest.json), a JSON list of entries:

  {"file": "data/offair/pocsag_1200.wav",     relative to the repo root
   "license": "CC-BY-4.0, <author>",          required: where it came from
   "source": "<url or description>",
   "sample_rate": null, "datatype": null,     optional overrides
   "expect": {"modulation": "2FSK",           any subset of these keys
              "symbol_rate_hz": 1200,         (within 1%)
              "fec": "none", "interleaver": "none",
              "frame_length_bits": 544, "crc": "CRC-16-CCITT-FALSE",
              "sync_word_hex": "7cd215d8",    (prefix match)
              "sample_rate_hz": 48000}}       (within 0.1%; see below)

sample_rate_hz scores PS task (i): leave the entry's "sample_rate" null.
It passes when the rate the tool applied (WAV header, SigMF, file name)
is right, or, when it applied none (headerless IQ), when the true rate
is among its PROBABLE candidates; "sample_rate_source" says which. A
rate the manifest supplies is listed under "given", never scored. For a
WAV the header's rate is scored (a mono IF WAV is analysed at fs/2).

"signal": false expects no detection (a capture whose source says the
transmitter was not received); true expects at least one.

Run:  python scripts/validate_offair.py [--manifest PATH] [--ml]
Writes results/offair_report.json, labelled off-air. Each case also
records its signal_quality (S6 EVM, the SNR it implies, S6 status), which
places it on the Proof Chart (scripts/plots.py).
"""
import argparse
import json
import logging
import math
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
logging.disable(logging.INFO)

from dhwani.common.config import load_config
from dhwani.pipeline import Analyzer
import provenance
from provenance import write_result



def observed(res) -> dict:
    p, fr = res.parameters, res.frames
    meta = res.recording_meta
    return {
        "signal": bool(res.segments),
        # a mono IF WAV is analysed at fs/2; the file's own rate is scored
        "sample_rate_hz": (meta.get("file_sample_rate")
                           or meta.get("sample_rate")),
        "sample_rate_source": meta.get("sample_rate_source"),
        "sample_rate_candidates": [c["sample_rate_hz"] for c in
                                   (p.sample_rate_candidates if p else [])],
        "modulation": res.modulation.prediction if res.modulation else None,
        "symbol_rate_hz": p.symbol_rate_hz if p else None,
        "fec": res.fec.family if res.fec else None,
        "interleaver": res.interleaver.kind if res.interleaver else None,
        "frame_length_bits": fr.frame_length_bits if fr else None,
        "crc": (fr.crc or {}).get("name") if fr else None,
        "crc_frames": (fr.crc or {}).get("passes") if fr else None,
        "callsigns": (fr.crc or {}).get("callsigns") if fr else None,
        "sync_word_hex": fr.sync_word_hex if fr else None,
    }


def signal_quality(res) -> dict:
    """What the Proof Chart places a recording by: S6's EVM, the SNR it
    implies (EVM^2 = N0/Es after the matched filter, so -20 log10(EVM)
    reads Es/N0; Eb/N0 for uncoded BPSK), the S6 status and the modulation
    S6 ran.  None where S6 did not run or measured no EVM (FSK, analog)."""
    d = res.demodulation or {}
    evm = d.get("evm_percent")
    return {"evm_percent": evm,
            "snr_db_evm": (round(-20 * math.log10(evm / 100.0), 1)
                           if evm else None),
            "demodulation_status": d.get("demodulation_status"),
            "modulation_demodulated": d.get("modulation")}


def matches(key, want, got) -> bool:
    if got is None:
        return False
    if key == "symbol_rate_hz":
        return abs(got - want) <= 0.01 * want
    if key == "sample_rate_hz":
        return abs(got - want) <= 0.001 * want
    if key == "sync_word_hex":
        return str(got).lower().startswith(str(want).lower())
    return got == want


def score(entry: dict, analyzer, no_ml: bool) -> dict:
    t0 = time.perf_counter()
    res = analyzer.analyze(os.path.join(ROOT, entry["file"]),
                           sample_rate=entry.get("sample_rate"),
                           datatype=entry.get("datatype"), no_ml=no_ml)
    got = observed(res)
    expect = dict(entry["expect"])
    given = []
    if "sample_rate_hz" in expect and got["sample_rate_source"] == "user":
        given.append("sample_rate_hz")     # echoed back, not identified
        del expect["sample_rate_hz"]
    checks = {k: matches(k, v, got[k]) for k, v in expect.items()}
    want_fs = expect.get("sample_rate_hz")
    if want_fs and got["sample_rate_hz"] is None:
        # headerless: a hit among the candidates; the rate stays unapplied
        checks["sample_rate_hz"] = any(
            matches("sample_rate_hz", want_fs, c)
            for c in got["sample_rate_candidates"])
        got["sample_rate_source"] = "candidates (not applied)"
    shown = list(entry["expect"]) + (
        ["sample_rate_candidates"] if want_fs else [])
    return {"file": entry["file"], "license": entry["license"],
            "source": entry.get("source"), "expected": entry["expect"],
            "observed": {k: got[k] for k in shown},
            "decoded": {"crc": got["crc"], "frames": got["crc_frames"],
                        "callsigns": got["callsigns"]},
            "sample_rate_source": got["sample_rate_source"],
            "given": given,
            "checks": checks,
            "all_correct": bool(checks) and all(checks.values()),
            "signal_quality": signal_quality(res),
            "warnings": res.warnings,
            "elapsed_s": round(time.perf_counter() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default=os.path.join(ROOT, "data", "offair", "manifest.json"))
    ap.add_argument("--out-dir", default=None,
                    help="directory to write offair_report.json (default results)")
    ap.add_argument("--ml", action="store_true",
                    help="enable CVNet-RF (may download the checkpoint)")
    args = ap.parse_args(argv)
    if args.out_dir is not None:
        out = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
        provenance.RESULTS = os.path.abspath(out)
    if not os.path.exists(args.manifest):
        print(f"no manifest at {args.manifest}; see this script's docstring")
        return 1
    with open(args.manifest) as f:
        entries = json.load(f)
    analyzer = Analyzer(load_config(), use_cache=False)
    rows = []
    for e in entries:
        r = score(e, analyzer, no_ml=not args.ml)
        rows.append(r)
        print(f"{e['file']}: {'PASS' if r['all_correct'] else 'FAIL'} "
              f"{r['checks']}", flush=True)
    keys = sorted({k for r in rows for k in r["checks"]})
    per_key = {k: {"n": sum(k in r["checks"] for r in rows),
                   "correct": sum(r["checks"].get(k, False) for r in rows)}
               for k in keys}
    path = write_result("offair_report", {
        "settings": {"manifest": os.path.relpath(args.manifest, ROOT),
                     "cvnet": args.ml},
        "summary": {"recordings": len(rows),
                    "all_correct": sum(r["all_correct"] for r in rows),
                    "per_parameter": per_key},
        "cases": rows}, data="off-air (licensed recordings, see cases)")
    print("report:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
