"""Charts for the README, drawn from results/*.json as plain SVG (stdlib only).

  results/charts/modulation_top1.svg  blind S5 top-1 accuracy per modulation,
                                      standard CFO vs zero CFO
  results/charts/bitlayer_grid.svg    interleaver x FEC: frames recovered,
                                      CRC pass written in each cell
  results/charts/snr_curves.svg       blind top-1 accuracy vs SNR per
                                      modulation family (--snr-sweep runs)
  results/charts/proof.svg            Proof Chart: BER and probability of
                                      lock vs Eb/N0 per decoding chain
                                      (benchmark_ber.py), with the off-air
                                      recordings at their EVM SNR

Colours follow the dataviz reference palette (categorical slots 1-2 and the
blue sequential ramp, validated light + dark); every value is also written
as text, and the README tables are the table view. Each mark carries an SVG
<title> (hover tooltip when the file is opened directly).

Run:  python scripts/plots.py   (after the result scripts)
"""
import json
import math
import os
from xml.sax.saxutils import escape

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "charts")

STYLE = """<style>
  svg { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
        --grid:#e1e0d9; --axis:#c3c2b7; --s1:#2a78d6; --s2:#eb6834;
        --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4;
        font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  @media (prefers-color-scheme: dark) {
    svg { --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
          --grid:#2c2c2a; --axis:#383835; --s1:#3987e5; --s2:#d95926;
          --s3:#199e70; --s4:#c98500; --s5:#d55181; } }
  .bg { fill: var(--surface); } .title { fill: var(--ink); font-size: 15px; font-weight: 600; }
  .sub { fill: var(--ink2); font-size: 12px; } .lab { fill: var(--ink2); font-size: 12px; }
  .tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
  .val { fill: var(--ink2); font-size: 11px; font-variant-numeric: tabular-nums; }
  .grid { stroke: var(--grid); stroke-width: 1; } .axis { stroke: var(--axis); stroke-width: 1; }
  .s1 { fill: var(--s1); } .s2 { fill: var(--s2); }
  .s3 { fill: var(--s3); } .s4 { fill: var(--s4); } .s5 { fill: var(--s5); }
  .l1 { stroke: var(--s1); } .l2 { stroke: var(--s2); } .l3 { stroke: var(--s3); }
  .l4 { stroke: var(--s4); } .l5 { stroke: var(--s5); }
  .line { fill: none; stroke-width: 2; stroke-linejoin: round; }
  .dot { stroke: var(--surface); stroke-width: 2; }
  .cell-dark { fill: #ffffff; font-size: 12px; } .cell-light { fill: #0b0b0b; font-size: 12px; }
</style>"""

# blue sequential ramp, light -> dark (reference palette steps 100..650)
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]


def _bar(x, y, w, h, cls, tip, r=4):
    """Horizontal bar anchored at x, rounded only at its data end."""
    if w <= 0:
        return ""
    r = min(r, w, h / 2)
    d = (f"M{x},{y}h{w - r}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r}"
         f"a{r},{r} 0 0 1 {-r},{r}h{-(w - r)}z")
    return f'<path class="{cls}" d="{d}"><title>{escape(tip)}</title></path>'


def _svg(w, h, body, label):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img" aria-label="{escape(label)}">'
            f'{STYLE}<rect class="bg" width="{w}" height="{h}" rx="6"/>'
            f'{body}</svg>\n')


def modulation_chart(d):
    pc = d["per_class"]
    mods = [m for m in pc if "(" not in m]
    rows = [(m, pc[m]["top1_accuracy"], pc.get(f"{m} (CFO 0)", {})
             .get("top1_accuracy")) for m in mods]
    left, top, plot_w, bar_h, gap, row_gap = 96, 76, 440, 9, 2, 9
    row_h = 2 * bar_h + gap + row_gap
    h = top + len(rows) * row_h + 34
    w = left + plot_w + 70
    out = ['<text class="title" x="16" y="26">Blind modulation ID, top-1 accuracy</text>',
           f'<text class="sub" x="16" y="44">synthetic, '
           f'{escape(d["provenance"]["git_commit"])}; one seed per case</text>']
    # legend (>= 2 series)
    for i, (cls, name) in enumerate((("s1", "CFO 0.004"), ("s2", "CFO 0 (stress)"))):
        lx = 16 + i * 150
        out.append(f'<rect class="{cls}" x="{lx}" y="54" width="12" height="9" rx="2"/>'
                   f'<text class="lab" x="{lx + 18}" y="63">{name}</text>')
    bottom = top + len(rows) * row_h - row_gap
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        gx = left + frac * plot_w
        out.append(f'<line class="{"axis" if frac == 0 else "grid"}" '
                   f'x1="{gx}" y1="{top - 4}" x2="{gx}" y2="{bottom + 4}"/>'
                   f'<text class="tick" x="{gx}" y="{bottom + 20}" '
                   f'text-anchor="middle">{int(frac * 100)}%</text>')
    for i, (m, a, b) in enumerate(rows):
        y = top + i * row_h
        out.append(f'<text class="lab" x="{left - 8}" y="{y + bar_h + 5}" '
                   f'text-anchor="end">{escape(m)}</text>')
        for j, (v, cls, tag) in enumerate(((a, "s1", "CFO 0.004"),
                                           (b, "s2", "CFO 0"))):
            if v is None:
                continue
            by = y + j * (bar_h + gap)
            out.append(_bar(left, by, v * plot_w, bar_h, cls,
                            f"{m}, {tag}: {v:.0%}"))
            if v < 1:                       # label only the misses
                out.append(f'<text class="val" x="{left + v * plot_w + 6}" '
                           f'y="{by + bar_h - 1}">{v:.0%}</text>')
    return _svg(w, h, "".join(out), "Blind modulation ID top-1 accuracy per "
                "modulation at CFO 0.004 and CFO 0")


def bitlayer_chart(d):
    # one cell per (interleaver, FEC) at the highest SNR, averaged over seeds
    top_snr = max(c["snr_db"] for c in d["cases"])
    cases = [c for c in d["cases"] if c["snr_db"] == top_snr]
    ils = list(dict.fromkeys(c["interleaver"] for c in cases))
    fecs = list(dict.fromkeys(c["fec"] for c in cases))
    cell = {}
    for (il, f) in {(c["interleaver"], c["fec"]) for c in cases}:
        cs = [c for c in cases if (c["interleaver"], c["fec"]) == (il, f)]
        cell[(il, f)] = {
            "frames_recovered": sum(c["frames_recovered"] for c in cs) / len(cs),
            "crc_pass": sum(c["crc_pass"] for c in cs) / len(cs) >= 0.5,
            "crc_rate": sum(c["crc_pass"] for c in cs) / len(cs),
            "interleaver_correct": all(c["interleaver_correct"] for c in cs),
            "fec_correct": all(c["fec_correct"] for c in cs), "n": len(cs)}
    left, top, cw, ch, g = 118, 92, 104, 46, 2
    w = left + len(fecs) * (cw + g) + 16
    h = top + len(ils) * (ch + g) + 24
    out = ['<text class="title" x="16" y="26">Bit layer: frames recovered '
           '(interleaver x FEC)</text>',
           f'<text class="sub" x="16" y="44">synthetic, PN9-whitened QPSK, '
           f'{top_snr} dB, '
           f'{escape(d["provenance"]["git_commit"])}; darker = more frames</text>']
    for j, f in enumerate(fecs):
        out.append(f'<text class="lab" x="{left + j * (cw + g) + cw / 2}" '
                   f'y="{top - 10}" text-anchor="middle">{escape(f)}</text>')
    for i, il in enumerate(ils):
        y = top + i * (ch + g)
        out.append(f'<text class="lab" x="{left - 8}" y="{y + ch / 2 + 4}" '
                   f'text-anchor="end">{escape(il)}</text>')
        for j, f in enumerate(fecs):
            c = cell.get((il, f))
            if not c:
                continue
            v = c["frames_recovered"]
            fill = RAMP[min(len(RAMP) - 1, int(round(v * (len(RAMP) - 1))))]
            txt = "cell-dark" if v >= 0.5 else "cell-light"
            line1 = "CRC ok" if c["crc_pass"] else "no frames"
            line2 = f"{v:.0%} frames" if c["crc_pass"] else ""
            x = left + j * (cw + g)
            tip = (f"{il} x {f}: CRC pass in {c['crc_rate']:.0%} of "
                   f"{c['n']} runs, {v:.0%} of frames, interleaver ID "
                   f"{'right' if c['interleaver_correct'] else 'wrong'}, FEC ID "
                   f"{'right' if c['fec_correct'] else 'wrong'}")
            y1 = y + 20 if line2 else y + ch / 2 + 4      # centre one line
            out.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="4" '
                       f'fill="{fill}"><title>{escape(tip)}</title></rect>'
                       f'<text class="{txt}" x="{x + cw / 2}" y="{y1}" '
                       f'text-anchor="middle">{line1}</text>'
                       f'<text class="{txt}" x="{x + cw / 2}" y="{y + 35}" '
                       f'text-anchor="middle">{line2}</text>')
    return _svg(w, h, "".join(out), "Bit-layer frames recovered per "
                "interleaver and FEC family")


def snr_chart(d):
    curves = d["snr_curves"]
    fams = [f for f in ("PSK", "QAM", "APSK", "ASK", "FSK") if f in curves]
    snrs = sorted({p["snr_db"] for f in fams for p in curves[f]})
    left, top, pw, ph = 56, 70, 460, 240
    w, h = left + pw + 70, top + ph + 48
    lo, hi = snrs[0], snrs[-1]
    X = lambda s: left + (s - lo) / max(1, hi - lo) * pw
    Y = lambda a: top + (1 - a) * ph
    out = ['<text class="title" x="16" y="26">Blind modulation ID vs SNR</text>',
           f'<text class="sub" x="16" y="44">synthetic, top-1 accuracy per '
           f'family, {escape(d["provenance"]["git_commit"])}; hover a point '
           f'for its 95% interval</text>']
    for a in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line class="{"axis" if a == 0 else "grid"}" x1="{left}" '
                   f'y1="{Y(a)}" x2="{left + pw}" y2="{Y(a)}"/>'
                   f'<text class="tick" x="{left - 8}" y="{Y(a) + 4}" '
                   f'text-anchor="end">{int(a * 100)}%</text>')
    for s in snrs:
        out.append(f'<text class="tick" x="{X(s)}" y="{top + ph + 18}" '
                   f'text-anchor="middle">{s}</text>')
    out.append(f'<text class="tick" x="{left + pw / 2}" y="{top + ph + 36}" '
               f'text-anchor="middle">SNR (dB)</text>')
    for i, f in enumerate(fams, start=1):
        pts = curves[f]
        path = " ".join(f"{'M' if j == 0 else 'L'}{X(p['snr_db']):.1f},"
                        f"{Y(p['top1_accuracy']):.1f}" for j, p in enumerate(pts))
        out.append(f'<path class="line l{i}" d="{path}"/>')
        for p in pts:
            ci = p["top1_ci95"]
            out.append(f'<circle class="s{i} dot" cx="{X(p["snr_db"]):.1f}" '
                       f'cy="{Y(p["top1_accuracy"]):.1f}" r="4"><title>'
                       f'{f} at {p["snr_db"]} dB: {p["top1_accuracy"]:.0%} '
                       f'(95% CI {ci[0]:.0%}-{ci[1]:.0%}, n={p["n"]})'
                       f'</title></circle>')
        end = pts[-1]                       # direct label (contrast relief)
        out.append(f'<text class="lab" x="{X(end["snr_db"]) + 8}" '
                   f'y="{Y(end["top1_accuracy"]) + 4 + (i - 3) * 11}">{f}</text>')
    for i, f in enumerate(fams, start=1):   # legend
        lx = 16 + (i - 1) * 90
        out.append(f'<rect class="s{i}" x="{lx}" y="52" width="12" height="9" rx="2"/>'
                   f'<text class="lab" x="{lx + 18}" y="61">{f}</text>')
    return _svg(w, h, "".join(out), "Blind modulation ID accuracy versus SNR "
                "per modulation family")


# Proof Chart marks on top of STYLE (kept out of STYLE so the other charts'
# files stay byte-identical): hollow = an upper bound, not a measurement
PROOF_STYLE = """<style>
  .hollow { fill: var(--surface); stroke-width: 2; }
  .stem { stroke-width: 1.5; } .ring { fill: none; stroke-width: 1.5; }
  .theory { fill: none; stroke: var(--ink2); stroke-width: 1.5;
            stroke-dasharray: 5 4; }
  .key { fill: none; stroke: var(--ink2); stroke-width: 1.5; }
  .keyfill { fill: var(--ink2); }
  .hit { fill: var(--ink); fill-opacity: 0; }
</style>"""
CHAR_W = 6.4            # px per character of 12px system-ui text (legend wrap)
OFFAIR_STACK = 11       # px between off-air markers that would overlap
# px between the chains' markers at one Eb/N0: chains that share a value
# (all 0% below lock, equal upper bounds) would otherwise hide each other
DODGE = 6
PAD_X = 14              # px inside each end of the x axis for dodged marks


def _offair_points(offair) -> list:
    """Off-air recordings with an EVM: x = their EVM SNR (Es/N0; Eb/N0 for
    the uncoded BPSK/2FSK recordings), y = 1 unless S6 said FAILED.  No
    signal, analog and FSK (no EVM) and older reports without
    signal_quality are left out."""
    pts = []
    for c in (offair or {}).get("cases", []):
        q = c.get("signal_quality") or {}
        if q.get("snr_db_evm") is None or q.get("demodulation_status") is None:
            continue
        dec = c.get("decoded") or {}
        pts.append({"name": os.path.splitext(os.path.basename(c["file"]))[0],
                    "snr": float(q["snr_db_evm"]), "evm": q.get("evm_percent"),
                    "status": q["demodulation_status"],
                    "mod": q.get("modulation_demodulated"),
                    "locked": q["demodulation_status"] != "FAILED",
                    "correct": bool(c.get("all_correct")),
                    "crc": dec.get("crc") if dec.get("frames") else None})
    return pts


def _legend(items, x0, y0, width) -> tuple:
    """Lay (glyph svg at (0,0), text) items left to right, wrapping at
    width.  Returns (svg, y below the last row)."""
    out, x, y = [], x0, y0
    for glyph, text in items:
        w = 26 + len(text) * CHAR_W + 18
        if x > x0 and x + w > x0 + width:
            x, y = x0, y + 18
        out.append(f'<g transform="translate({x:.0f},{y})">{glyph}</g>'
                   f'<text class="lab" x="{x + 26:.0f}" y="{y + 4}">'
                   f'{escape(text)}</text>')
        x += w
    return "".join(out), y + 18


def _marker(x, y, tip, shape) -> str:
    """A hover group: a >= 20px transparent hit target under the mark."""
    return (f'<g><title>{escape(tip)}</title>'
            f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="10"/>'
            f'{shape}</g>')


def _diamond(x, y, cls, r=5.5) -> str:
    return (f'<path class="{cls}" d="M{x:.1f},{y - r:.1f}L{x + r:.1f},{y:.1f}'
            f'L{x:.1f},{y + r:.1f}L{x - r:.1f},{y:.1f}z"/>')


def proof_chart(ber, offair=None):
    """Top: BER of locked frames vs Eb/N0 per chain (log axis), uncoded
    theory dashed; a point with 0 errors is a hollow marker at its 95%
    upper bound 3/bits with a stem pointing down.  Bottom: probability
    of lock per chain, with the off-air recordings at their EVM SNR.
    Each chain's marks sit DODGE px beside the next one's (a fraction of
    a dB; the tooltips carry the exact values)."""
    chains = ber["chains"]
    theory = (ber.get("theory") or {}).get("points", [])
    off = _offair_points(offair)
    xs = [p["ebn0_db"] for c in chains.values() for p in c["points"]]
    xs += [o["snr"] for o in off]
    lo_x, hi_x = math.floor(min(xs)), math.ceil(max(xs))
    hi_x = max(hi_x, lo_x + 1)
    span = hi_x - lo_x
    step = 1 if span <= 12 else 2 if span <= 24 else 5
    vals = [v for c in chains.values() for p in c["points"]
            for v in (p.get("ber"), p.get("ber_upper_95")) if v]
    vals += [p["ber"] for p in theory if p["ber"] > 0]
    lo_dec = max(-9, math.floor(math.log10(min(vals)))) if vals else -6
    lo_dec = min(lo_dec, -1)

    w, left, pw = 640, 64, 540
    names = [(cid, c, i) for i, (cid, c) in enumerate(chains.items(), start=1)]
    frames = sorted({p["frames"] for c in chains.values() for p in c["points"]})
    commit = ber.get("provenance", {}).get("git_commit")
    out = ['<text class="title" x="16" y="26">Decoding proof: bit error rate '
           'and lock vs Eb/N0</text>',
           f'<text class="sub" x="16" y="44">synthetic QPSK through '
           f'Dhwani\'s own S1-S6 receiver, {"/".join(map(str, frames))} '
           f'frames per point, {escape(str(commit))}</text>']
    if off:
        out.append('<text class="sub" x="16" y="60">off-air recordings at '
                   'their EVM SNR (Es/N0, = Eb/N0 for uncoded BPSK); hover '
                   'any mark for its numbers</text>')
    leg = [(f'<line class="line l{i}" x1="0" y1="0" x2="18" y2="0"/>'
            f'<circle class="s{i} dot" cx="9" cy="0" r="4"/>', c["label"])
           for cid, c, i in names]
    if theory:
        leg.append(('<line class="theory" x1="0" y1="0" x2="18" y2="0"/>',
                    "theory, uncoded"))
    leg.append(('<circle class="hollow key" cx="9" cy="0" r="4"/>'
                '<line class="key" x1="9" y1="5" x2="9" y2="10"/>',
                "0 errors: 95% upper bound 3/bits"))
    if off:
        leg += [(_diamond(9, 0, "keyfill", 5), "off-air, fully correct"),
                (_diamond(9, 0, "key", 5), "off-air, not fully correct"),
                ('<circle class="key" cx="9" cy="0" r="7"/>',
                 "ring: a CRC-valid frame decoded")]
    svg_leg, y = _legend(leg, 16, 82 if off else 66, w - 32)
    out.append(svg_leg)

    top, ph = y + 22, 230
    top2, ph2 = top + ph + 52, 110
    bottom = top2 + ph2
    X = lambda v: left + PAD_X + (v - lo_x) / span * (pw - 2 * PAD_X)
    Yb = lambda b: top + (-math.log10(min(1.0, max(b, 10.0 ** lo_dec)))
                          / -lo_dec) * ph
    Yp = lambda p: top2 + (1 - p) * ph2
    out.append(f'<text class="lab" x="{left}" y="{top - 10}">Bit error rate '
               f'(information bits of locked frames)</text>'
               f'<text class="lab" x="{left}" y="{top2 - 10}">Probability of '
               f'lock (S2 detects, S6 not FAILED, sync word found)</text>')
    v = math.ceil(lo_x / step) * step
    while v <= hi_x:
        out.append(f'<line class="grid" x1="{X(v):.1f}" y1="{top}" '
                   f'x2="{X(v):.1f}" y2="{top + ph}"/>'
                   f'<line class="grid" x1="{X(v):.1f}" y1="{top2}" '
                   f'x2="{X(v):.1f}" y2="{bottom}"/>'
                   f'<text class="tick" x="{X(v):.1f}" y="{bottom + 18}" '
                   f'text-anchor="middle">{v}</text>')
        v += step
    out.append(f'<text class="tick" x="{left + pw / 2}" y="{bottom + 36}" '
               f'text-anchor="middle">Eb/N0 (dB)</text>')
    for d in range(0, lo_dec - 1, -1):
        out.append(f'<line class="{"axis" if d == lo_dec else "grid"}" '
                   f'x1="{left}" y1="{Yb(10.0 ** d):.1f}" x2="{left + pw}" '
                   f'y2="{Yb(10.0 ** d):.1f}"/>'
                   f'<text class="tick" x="{left - 8}" y="{Yb(10.0 ** d) + 4:.1f}" '
                   f'text-anchor="end">{"1" if d == 0 else f"1e{d}"}</text>')
    for a in (0, 0.5, 1.0):
        out.append(f'<line class="{"axis" if a == 0 else "grid"}" x1="{left}" '
                   f'y1="{Yp(a):.1f}" x2="{left + pw}" y2="{Yp(a):.1f}"/>'
                   f'<text class="tick" x="{left - 8}" y="{Yp(a) + 4:.1f}" '
                   f'text-anchor="end">{int(a * 100)}%</text>')

    if theory:
        path = " ".join(f"{'M' if j == 0 else 'L'}{X(p['ebn0_db']):.1f},"
                        f"{Yb(p['ber']):.1f}" for j, p in enumerate(theory))
        out.append(f'<path class="theory" d="{path}"><title>theory, uncoded '
                   f'QPSK: ' + escape(", ".join(
                       f"{p['ebn0_db']} dB {p['ber']:.1e}" for p in theory))
                   + '</title></path>')
    for cid, c, i in names:
        label = c["label"]
        dx = (i - (len(names) + 1) / 2) * DODGE
        Xc = lambda v: X(v) + dx
        meas = [p for p in c["points"] if p.get("bits") and p.get("bit_errors")]
        if len(meas) > 1:
            path = " ".join(f"{'M' if j == 0 else 'L'}{Xc(p['ebn0_db']):.1f},"
                            f"{Yb(p['ber']):.1f}" for j, p in enumerate(meas))
            out.append(f'<path class="line l{i}" d="{path}"/>')
        path = " ".join(f"{'M' if j == 0 else 'L'}{Xc(p['ebn0_db']):.1f},"
                        f"{Yp(p['p_lock']):.1f}"
                        for j, p in enumerate(c["points"]))
        out.append(f'<path class="line l{i}" d="{path}"/>')
        for p in c["points"]:
            x = Xc(p["ebn0_db"])
            locked = f"locked {p['locked']}/{p['frames']}"
            if p.get("bits") and p.get("bit_errors"):
                yb = Yb(p["ber"])
                out.append(_marker(x, yb, (
                    f"{label} at {p['ebn0_db']} dB: BER {p['ber']:.2e} "
                    f"({p['bit_errors']} errors in {p['bits']} bits), {locked}"),
                    f'<circle class="s{i} dot" cx="{x:.1f}" cy="{yb:.1f}" r="4"/>'))
            elif p.get("bits"):
                yb = Yb(p["ber_upper_95"])
                out.append(_marker(x, yb, (
                    f"{label} at {p['ebn0_db']} dB: 0 errors in {p['bits']} "
                    f"bits, BER < {p['ber_upper_95']:.1e} (95% upper bound), "
                    f"{locked}"),
                    f'<line class="l{i} stem" x1="{x:.1f}" y1="{yb + 5:.1f}" '
                    f'x2="{x:.1f}" y2="{yb + 11:.1f}"/>'
                    f'<circle class="hollow l{i}" cx="{x:.1f}" cy="{yb:.1f}" r="4"/>'))
            ci = p.get("p_lock_ci95") or [0, 0]
            why = ", ".join(f"{k.replace('_', ' ')} {n}" for k, n in
                            (p.get("lock_failure_reasons") or {}).items() if n)
            yp = Yp(p["p_lock"])
            out.append(_marker(x, yp, (
                f"{label} at {p['ebn0_db']} dB: {locked} = {p['p_lock']:.0%} "
                f"(95% CI {ci[0]:.0%}-{ci[1]:.0%})"
                + (f"; lock failures: {why}" if why else "")),
                f'<circle class="s{i} dot" cx="{x:.1f}" cy="{yp:.1f}" r="4"/>'))

    # off-air: overlapping markers on the same line step inward (y is a
    # yes/no, so the stack stays readable as "locked" / "not locked")
    placed = []
    for o in sorted(off, key=lambda o: o["snr"]):
        x = X(o["snr"])
        k = sum(1 for (px, py) in placed if py == o["locked"] and abs(px - x) < 10)
        placed.append((x, o["locked"]))
        y = Yp(1.0) + k * OFFAIR_STACK if o["locked"] else \
            Yp(0.0) - k * OFFAIR_STACK
        shape = _diamond(x, y, "s4 dot" if o["correct"] else "hollow l4")
        if o["crc"]:
            shape += f'<circle class="ring l4" cx="{x:.1f}" cy="{y:.1f}" r="8.5"/>'
        evm = f", EVM {o['evm']:.1f}%" if o["evm"] is not None else ""
        out.append(_marker(x, y, (
            f"{o['name']}: SNR {o['snr']:.1f} dB{evm}, {o['mod']} {o['status']}"
            f"; {'fully correct' if o['correct'] else 'not fully correct'}"
            + (f"; CRC-valid frame decoded ({o['crc']})" if o["crc"] else "")),
            shape))
    return _svg(w, bottom + 48, PROOF_STYLE + "".join(out),
                "Bit error rate and probability of lock versus Eb/N0 per "
                "decoding chain, with off-air recordings at their EVM SNR")


def _load_offair():
    """results/offair/offair_report.json first, then results/ (as the
    README renderer reads it)."""
    for p in (os.path.join(RESULTS, "offair", "offair_report.json"),
              os.path.join(RESULTS, "offair_report.json")):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    written = []
    for name, fn, src in (("modulation_top1", modulation_chart,
                           "validation_report"),
                          ("snr_curves", snr_chart, "validation_report"),
                          ("bitlayer_grid", bitlayer_chart, "bitlayer_report")):
        p = os.path.join(RESULTS, src + ".json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            data = json.load(f)
        if name == "snr_curves" and "snr_curves" not in data:
            continue
        svg = fn(data)
        path = os.path.join(OUT, name + ".svg")
        with open(path, "w") as f:
            f.write(svg)
        written.append(path)
    p = os.path.join(RESULTS, "ber_report.json")
    if os.path.exists(p):               # the off-air overlay is optional
        with open(p, encoding="utf-8") as f:
            ber = json.load(f)
        path = os.path.join(OUT, "proof.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(proof_chart(ber, _load_offair()))
        written.append(path)
    print("\n".join(written) or "no results to plot")


if __name__ == "__main__":
    main()
