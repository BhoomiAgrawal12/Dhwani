"""Render the README results block from results/*.json, so every number
in the README comes from a committed, script-generated file.

Run:  python scripts/render_readme.py            rewrite the block
      python scripts/render_readme.py --check    exit 1 if it is stale (CI)
"""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
README = os.path.join(ROOT, "README.md")
BEGIN, END = "<!-- results:begin -->", "<!-- results:end -->"


def _load(name):
    if name == "offair_report":
        p_offair = os.path.join(ROOT, "results", "offair", name + ".json")
        if os.path.exists(p_offair):
            with open(p_offair) as f:
                return json.load(f)
    p = os.path.join(ROOT, "results", name + ".json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
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
                   "fully correct.\n")
        out.append("| parameter | correct |")
        out.append("|---|---:|")
        out += [f"| {k} | {v['correct']}/{v['n']} |"
                for k, v in s["per_parameter"].items()]
        out.append("\n" + _source(d, "offair_report"))
    else:
        out.append("_Measured on synthetic recordings; to score real "
                   "captures run `python scripts/validate_offair.py`._")
    return "\n".join(out)


def main(argv=None):
    check = "--check" in (argv if argv is not None else sys.argv[1:])
    with open(README) as f:
        text = f.read()
    i, j = text.index(BEGIN) + len(BEGIN), text.index(END)
    new = text[:i] + "\n" + render() + "\n" + text[j:]
    if check:
        if new != text:
            print("README results block is stale: run "
                  "python scripts/render_readme.py")
            return 1
        return 0
    with open(README, "w") as f:
        f.write(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
