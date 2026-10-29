"""Render the README results block from results/*.json, so every number
in the README comes from a committed, script-generated file.

Run:  python scripts/render_readme.py            rewrite the block
      python scripts/render_readme.py --check    exit 1 if it is stale (CI)
"""
import json
import os
import sys
from fractions import Fraction

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
README = os.path.join(ROOT, "README.md")
BEGIN, END = "<!-- results:begin -->", "<!-- results:end -->"


def _load(name):
    if name == "offair_report":
        p_offair = os.path.join(ROOT, "results", "offair", name + ".json")
        if os.path.exists(p_offair):
            with open(p_offair, encoding="utf-8") as f:
                return json.load(f)
    p = os.path.join(ROOT, "results", name + ".json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _source(d, name):
    p = d["provenance"]
    return (f"_Source: `results/{name}.json` ({p['data']}), commit "
            f"`{p['git_commit']}`{' (dirty)' if p.get('git_dirty') else ''}, "
            f"{p['cpu']}, Python {p['python']}, {p['generated_utc']}._")


def _missing(cmd):
    return f"_Not generated yet: run `{cmd}` and commit `results/`._"


def _chart(name, alt):
    """Embed results/charts/<name>.svg when scripts/plots.py has drawn it."""
    if os.path.exists(os.path.join(ROOT, "results", "charts", name + ".svg")):
        return [f"\n![{alt}](results/charts/{name}.svg)"]
    return []


def render() -> str:
    out = []
    d = _load("benchmark")
    out.append("### Speed (synthetic inputs)\n")
    if d:
        out.append("| operation | measured | design target |")
        out.append("|---|---:|---|")
        out += [f"| {t['operation']} | {t['seconds']:.2f} s | {t['target']} |"
                for t in d["timings"]]
        out.append("\n" + _source(d, "benchmark"))
    else:
        out.append(_missing("python scripts/benchmark.py --big"))

    d = _load("validation_report")
    out.append("\n### Modulation ID and demodulation (synthetic)\n")
    if d:
        s = d["summary"]
        out.append(f"{s['pass']} PASS, {s['degraded']} DEGRADED, "
                   f"{s['fail']} FAIL of {s['total']} cases.\n")
        out += _chart("modulation_top1", "Blind modulation ID top-1 accuracy "
                      "per modulation, CFO 0.004 vs CFO 0")
        out += ["", "| modulation | cases | blind top-1 | 95% CI | blind top-2 |",
                "|---|---:|---:|---:|---:|"]
        out += [f"| {m} | {c['n']} | {c['top1_accuracy']:.0%} | "
                + (f"{c['top1_ci95'][0]:.0%}-{c['top1_ci95'][1]:.0%}"
                   if c.get("top1_ci95") else "-")
                + f" | {c['top2_accuracy']:.0%} |"
                for m, c in d.get("per_class", {}).items()]
        if d.get("snr_curves"):
            curves = d["snr_curves"]
            snrs = sorted({p["snr_db"] for ps in curves.values() for p in ps})
            out += _chart("snr_curves", "Blind modulation ID accuracy versus "
                          "SNR per modulation family")
            out += ["", "| family | " + " | ".join(f"{s} dB" for s in snrs)
                    + " |", "|---|" + "---:|" * len(snrs)]
            for fam, ps in curves.items():
                by = {p["snr_db"]: p["top1_accuracy"] for p in ps}
                out.append(f"| {fam} | " + " | ".join(
                    f"{by[s]:.0%}" if s in by else "-" for s in snrs) + " |")
        out.append("\n" + _source(d, "validation_report"))
    else:
        out.append(_missing("python scripts/validate_matrix.py"))

    d = _load("bitlayer_report")
    out.append("\n### Interleaver and FEC identification (synthetic)\n")
    if d:
        out += _chart("bitlayer_grid", "Bit-layer frames recovered per "
                      "interleaver and FEC family")
        out.append("")
        for key, title in (("per_interleaver", "interleaver"),
                           ("per_fec", "FEC")):
            out.append(f"| {title} | cases | interleaver ID | FEC ID | CRC pass |")
            out.append("|---|---:|---:|---:|---:|")
            out += [f"| {k} | {c['n']} | {c['interleaver_correct_rate']:.0%} | "
                    f"{c['fec_correct_rate']:.0%} | {c['crc_pass_rate']:.0%} |"
                    for k, c in d[key].items()]
            out.append("")
        out.append(_source(d, "bitlayer_report"))
    else:
        out.append(_missing("python scripts/validate_bitlayer.py"))

    d = _load("ber_report")
    out.append("\n### Decoding proof (synthetic BER + off-air)\n")
    if d:
        out += _proof_table(d)
        out.append("\n" + _source(d, "ber_report"))
    else:
        out.append(_missing("python scripts/benchmark_ber.py"))

    d = _load("samplerate_report")
    out.append("\n### Headerless sample-rate candidates (synthetic)\n")
    if d:
        s = d["summary"]
        out.append(f"True sample rate among the PROBABLE candidates (with a "
                   f"correct symbol rate) in {s['true_rate_found']} of "
                   f"{s['cases']} headerless files; symbol rate right in "
                   f"{s['symbol_rate_ok']}; {s['mean_candidates']} candidates "
                   f"on average; sample rate applied: "
                   f"{'yes' if s['sample_rate_ever_applied'] else 'never'}.\n")
        out.append(_source(d, "samplerate_report"))
    else:
        out.append(_missing("python scripts/validate_samplerate.py"))

    d = _load("offair_report")
    out.append("\n### Off-air recordings\n")
    if d:
        s = d["summary"]
        out.append(f"{s['all_correct']} of {s['recordings']} recordings "
                   "fully correct. Off-air set: recordings with "
                   "source-verified parameters only.\n")
        out.append("| parameter | correct |")
        out.append("|---|---:|")
        out += [f"| {k} | {v['correct']}/{v['n']} |"
                for k, v in s["per_parameter"].items()]
        out += _offair_table(d["cases"])
        out.append("\n" + _source(d, "offair_report"))
    else:
        out.append("_Measured on synthetic recordings; to score real "
                   "captures run `python scripts/validate_offair.py`._")
    # Real recordings first; synthetic sections follow, folded and labelled.
    secs = ("\n".join(out)).split("### ")[1:]
    by = {x.split("\n", 1)[0]: "### " + x.rstrip() for x in secs}
    real = [by.pop("Off-air recordings"), by.pop("Decoding proof (synthetic BER + off-air)")]
    return ("\n\n".join(real) + "\n\nSynthetic-signal validation (simulator, "
            "not off-air) is in `results/*.json`.")


LOCK_LEVEL = 0.9        # "locks from": the lowest Eb/N0 with p_lock >= this
BER_LEVEL = 1e-3        # "BER from": the lowest Eb/N0 with BER <= this


def _proof_table(d) -> list:
    """Per chain: code rate, the lowest Eb/N0 that locks >= 90% of frames,
    the lowest with BER <= 1e-3 among locked frames, and the BER at the
    highest point (an upper bound when it saw no errors)."""
    s, chains = d["settings"], d["chains"]
    top = max(s["ebn0_db"])
    db = lambda x: f"{x:g} dB"
    rows = [f"{s['frames']} frames per chain and Eb/N0 point through Dhwani's "
            "own S1-S6 receiver; BER over the information bits of locked "
            "frames only (a frame S2 does not detect, S6 gates as FAILED or "
            "whose sync word is not found is a lock failure, not bit "
            "errors)."]
    chart = _chart("proof", "Bit error rate and probability of lock versus "
                   "Eb/N0 per decoding chain, with off-air recordings at "
                   "their EVM SNR")
    if chart:
        rows += chart
        rows.append("\nOff-air markers: the recordings below at their EVM "
                    "SNR; filled when fully correct, ringed when a CRC-valid "
                    "frame was decoded.")
    ber_txt = f"{BER_LEVEL:.0e}".replace("e-0", "e-")      # 1e-03 -> 1e-3
    rows += ["", f"| chain | code rate | locks ≥ {LOCK_LEVEL:.0%} from | "
             f"BER ≤ {ber_txt} from | BER at {db(top)} |",
             "|---|---:|---:|---:|---:|"]
    for c in chains.values():
        pts = sorted(c["points"], key=lambda p: p["ebn0_db"])
        lock = next((p["ebn0_db"] for p in pts if p["p_lock"] >= LOCK_LEVEL),
                    None)
        good = next((p["ebn0_db"] for p in pts if p.get("ber") is not None
                     and p["ber"] <= BER_LEVEL), None)
        last = pts[-1]
        if last.get("ber") is None:
            at_top = "no frame locked"
        elif not last["bit_errors"]:
            at_top = f"< {last['ber_upper_95']:.1e} (0 errors)"
        else:
            at_top = f"{last['ber']:.1e}"
        rate = Fraction(c["code_rate"]).limit_denominator(16)
        rows.append(f"| {c['label']} | {rate} | "
                    f"{db(lock) if lock is not None else '-'} | "
                    f"{db(good) if good is not None else '-'} | {at_top} |")
    return rows


def _offair_table(cases) -> list:
    """One row per recording: expected, Dhwani's result, decoded frames,
    and a tick per scored parameter."""
    mark = lambda ok: "✓" if ok else "✗"
    rows = ["", "| recording | expected | Dhwani | frames / CRC | "
            "mod | baud | fs |", "|---|---|---|---|:-:|:-:|:-:|"]
    for c in cases:
        name = os.path.splitext(os.path.basename(c["file"]))[0]
        url = (c.get("source") or "").split(";")[0].strip()
        link = f"[{name}]({url})" if url.startswith("http") else name
        e, o, k = c["expected"], c["observed"], c["checks"]
        if e.get("signal") is False:
            rows.append(f"| {link} | no signal | "
                        f"{'no signal' if k.get('signal') else 'signal'} "
                        f"| - | - | - | - |")
            continue
        got_rate = o.get("symbol_rate_hz")
        dec = c.get("decoded") or {}
        if dec.get("crc") and dec.get("frames"):
            calls = " → ".join(dec.get("callsigns") or [])
            frames = f"{dec['frames']} frame(s), {dec['crc']} ✓; {calls}"
        else:
            frames = "-"
        rows.append(
            f"| {link} | {e.get('modulation')} {e.get('symbol_rate_hz')} Bd "
            f"| {o.get('modulation')} {got_rate:.0f} Bd | {frames} | "
            f"{mark(k.get('modulation'))} | {mark(k.get('symbol_rate_hz'))} "
            f"| {mark(k.get('sample_rate_hz'))} |"
            if got_rate else
            f"| {link} | {e.get('modulation')} {e.get('symbol_rate_hz')} Bd "
            f"| {o.get('modulation')} | {frames} | "
            f"{mark(k.get('modulation'))} | {mark(k.get('symbol_rate_hz'))} "
            f"| {mark(k.get('sample_rate_hz'))} |")
    return rows


def main(argv=None):
    check = "--check" in (argv if argv is not None else sys.argv[1:])
    # UTF-8 explicitly: the block carries non-ASCII marks, and a platform
    # default (cp1252 on Windows) misreads or cannot write them
    with open(README, encoding="utf-8") as f:
        text = f.read()
    i, j = text.index(BEGIN) + len(BEGIN), text.index(END)
    new = text[:i] + "\n" + render() + "\n" + text[j:]
    if check:
        if new != text:
            print("README results block is stale: run "
                  "python scripts/render_readme.py")
            return 1
        return 0
    with open(README, "w", encoding="utf-8") as f:
        f.write(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
