"""S8 Pattern Scan: interleaver geometry and code found together.

The rank profile loses an interleaver's period at about 0.5% BER and never
shows one over Reed-Solomon; the Pattern Scan looks for the code itself
through the interleaver (dilated checks, de-interleaved windows, an RS
scan per offset) and reports a geometry only with its code confirmed."""
import json
import time

import numpy as np
import pytest

from dhwani.fec import registry
from dhwani.fec.conv import ConvCode, conv_encode
from dhwani.fec.detect import identify_convolutional, identify_fec, identify_rs
from dhwani.fec.rs import RSCode
from dhwani.interleaving import (block_deinterleave, block_interleave,
                                 conv_deinterleave, conv_interleave,
                                 helical_interleave, ieee80211_permutation,
                                 pn_interleave)
from dhwani.interleaving import detect
from dhwani.interleaving.detect import identify_interleaver
from dhwani.interleaving.pattern_scan import pattern_scan
from dhwani.scrambling import lfsr

K7 = "conv K7 r1/2 (171,133)"
RS223 = registry.rs_entry({"n": 255, "k": 223, "prim": 0x11d, "fcr": 1,
                           "generator": 2})["name"]


def _k7(rng, n_info):
    return conv_encode(rng.integers(0, 2, n_info).astype(np.uint8),
                       ConvCode(7, (0o171, 0o133)), terminate=False)


def _flip(bits, ber, rng):
    out = bits.copy()
    out[rng.random(len(out)) < ber] ^= 1
    return out


def _is_scan(h):
    return h.parameters.get("method") == "pattern scan"


def _block_hit(hits, rows, cols, offset):
    return bool(hits) and hits[0].kind == "block" and (
        hits[0].parameters["rows"], hits[0].parameters["cols"],
        hits[0].parameters["offset"]) == (rows, cols, offset)


def test_block_8x16_over_k7_at_3pct_ber(config):
    """The rank profile is flat at 3% BER; the code's check dilated by the
    row count still holds inside every row."""
    rng = np.random.default_rng(7)
    coded = _k7(rng, 30000)
    x = _flip(block_interleave(coded, 8, 16)[37:], 0.03, rng)
    hyps = identify_interleaver(x, max_L=256)
    top = hyps[0]
    assert top.kind == "block" and _is_scan(top)
    p = top.parameters
    assert (p["rows"], p["cols"], p["offset"]) == (8, 16, 91)
    assert p["code"] == K7
    assert p["early_verdict"]["verdict"] == "code"
    assert abs(p["early_verdict"]["ber_estimate"] - 0.03) < 0.01
    assert top.period == 128 and top.score > 12
    assert top.rank_profile is not None and hyps[-1].kind == "none"
    json.dumps(top.to_dict())               # the report dumps parameters as JSON
    fec = identify_fec(block_deinterleave(x[91:], 8, 16), config.fec,
                       try_rs=False, try_ldpc=False)[0]
    assert fec.family == "convolutional"
    assert (fec.parameters["K"], fec.parameters["g1_octal"],
            fec.parameters["g2_octal"]) == (7, oct(0o171), oct(0o133))


def test_block_32x32_over_rs_255_223(config):
    """New: RS under a bit block interleaver (flat rank profile at any
    BER) is found by an RS scan over every offset of a common geometry."""
    rng = np.random.default_rng(11)
    rs = RSCode(255, 223)
    data = rng.integers(0, 256, 223 * 40, dtype=np.uint8)
    cws = b"".join(rs.encode(bytes(data[i:i + 223]))
                   for i in range(0, len(data), 223))
    bits = np.unpackbits(np.frombuffer(cws, dtype=np.uint8))
    x = block_interleave(bits, 32, 32)[500:]           # cut mid-block
    hyps = identify_interleaver(x, max_L=256)
    top = hyps[0]
    assert top.kind == "block" and _is_scan(top)
    p = top.parameters
    assert (p["rows"], p["cols"], p["offset"]) == (32, 32, 524)
    assert p["code"] == RS223
    assert p["early_verdict"]["verdict"] == "code"
    json.dumps(top.to_dict())
    hit = identify_rs(block_deinterleave(x[524:], 32, 32),
                      config.fec.rs_candidates)[0]
    assert (hit["n"], hit["k"], hit["fcr"]) == (255, 223, 1)


def test_convolutional_interleaver_b8_m4():
    """B = 8 branches is shorter than the K=7 check (span 14), so no single
    dilation holds; the rank profile's period gives the candidates and the
    de-interleaved windows give the code."""
    rng = np.random.default_rng(3)
    coded = _k7(rng, 20000)
    x = conv_interleave(coded, 8, 4)[37:]
    top = identify_interleaver(x, max_L=256)[0]
    assert top.kind == "convolutional" and _is_scan(top)
    p = top.parameters
    assert (p["branches"], p["delay"], p["offset"]) == (8, 4, 3)
    assert p["code"] == K7 and p["early_verdict"]["verdict"] == "code"
    # `period` keeps its meaning: the rank-profile period the sweep tried
    assert top.period == 200
    # the engine's convention: skip `offset` bits, then de-interleave
    de = conv_deinterleave(x[3:], 8, 4)
    assert np.array_equal(de[:30000], coded[40:30040])


def _hdlc_fill(seed=21, n_seg=40):
    """HDLC-like flag fill (0x7E) around stretches of random bits."""
    rng = np.random.default_rng(seed)
    flag = np.unpackbits(np.array([0x7E], np.uint8))
    parts = []
    for _ in range(n_seg):
        parts.append(np.tile(flag, int(rng.integers(4, 40))))
        parts.append(rng.integers(0, 2, int(rng.integers(300, 1500))))
    return np.concatenate(parts).astype(np.uint8)


def test_structured_uncoded_streams_find_nothing(monkeypatch):
    """The commit-1 lesson: text, repeated frames, a beacon and flag fill
    bias wrong checks, and a long stream turns any bias into a large z.
    The scan must report nothing on them, so identify_interleaver returns
    what the old path returns.  The old hypothesis sweep (40-90 s on these
    streams) is not under test: a stub keeps it cheap, the same in both
    runs."""
    from dhwani.synth.factory import WaveformFactory
    streams = {
        "frames": WaveformFactory(seed=99).build_frames(480)[0],
        "text": WaveformFactory(seed=5).build_frames(160,
                                                     payload_mode="text")[0],
        "text400": WaveformFactory(seed=3).build_frames(400,
                                                        payload_mode="text")[0],
        "beacon": np.tile(np.random.default_rng(1).integers(0, 2, 1024)
                          .astype(np.uint8), 24),
        "hdlc": _hdlc_fill(),
        "random": np.random.default_rng(2).integers(0, 2, 60000)
                    .astype(np.uint8),
    }
    calls, periods, real = [], [], detect.pattern_scan

    def spy(bits, **kw):
        calls.append((kw.get("periods", ()), real(bits, **kw)))
        return calls[-1][1]
    monkeypatch.setattr(detect, "_align_and_test", lambda *a, **k: (0, 0.0))
    for name, bits in streams.items():
        calls.clear()
        monkeypatch.setattr(detect, "pattern_scan", spy)
        hyps = identify_interleaver(bits, max_L=128)
        assert len(calls) == 1 and calls[0][1] == [], name
        periods.append(calls[0][0])
        assert not any(_is_scan(h) for h in hyps), name
        monkeypatch.setattr(detect, "pattern_scan", lambda *a, **k: [])
        old = identify_interleaver(bits, max_L=128)
        assert [h.kind for h in hyps] == [h.kind for h in old], name
    # both paths were exercised: flat profiles and detected periods
    assert any(len(p) for p in periods) and any(not len(p) for p in periods)


def test_lfsr_sequences_find_nothing():
    """An m-sequence decimated by D satisfies every check its polynomial
    divides, at every bit: PRBS7 and 802.11-scrambled idle at 2-6% BER
    once read as a block interleaver over K=4 (15,17) (HEAD: 'none').  A
    code check holds in its valid columns, at one phase."""
    w = lfsr.KNOWN_WHITENERS["IEEE802.11"]
    seqs = {"PRBS7": (0xC1, 0x7F), "PRBS7 reciprocal": (0x83, 0x7F),
            "802.11 idle": (w["poly"], w["seed"])}
    for name, (poly, seed) in seqs.items():
        base = lfsr.additive_scramble(np.zeros(102100, np.uint8), poly, seed)
        for ber, sd in ((0.02, 4), (0.04, 2), (0.06, 5600)):
            rng = np.random.default_rng(sd)
            n, ph = int(rng.choice([30000, 60000, 100000])), int(rng.integers(0, 2047))
            x = _flip(base[ph:ph + n], ber, rng)
            assert pattern_scan(x) == [], (name, ber)
    # soft input: BPSK over AWGN, about 6% BER
    rng = np.random.default_rng(900)
    x0 = lfsr.additive_scramble(np.zeros(60000, np.uint8), 0xC1, 0x7F)
    rx = (1 - 2.0 * x0) + rng.normal(0, 0.65, len(x0))
    assert pattern_scan((rx < 0).astype(np.uint8), llrs=2 * rx / 0.65 ** 2) == []


def test_plainly_coded_repeated_frames_are_not_interleaved():
    """Identical frames, plainly coded: a check whose dilated taps pair up
    modulo the frame holds at every dilation, and convolutional B=2 M=1 is
    the identity on any rate-1/2 code (it reads as the generator-swapped
    twin).  An interleaver must make its code hold where the raw stream
    does not."""
    from dhwani.synth.factory import WaveformFactory
    frame = WaveformFactory(seed=99).build_frames(1)[0]
    for code in (ConvCode(7, (0o171, 0o133)), ConvCode(5, (0o35, 0o23))):
        for ber, sd in ((0.0, 1), (0.01, 1300)):
            x = _flip(conv_encode(np.tile(frame, 300), code, terminate=False),
                      ber, np.random.default_rng(sd))
            for periods in ((), (96, 192), (240, 224, 256, 24)):
                assert pattern_scan(x, periods=periods) == [], (code.K, ber, periods)


def test_inverted_substream_and_lead_in():
    """One inverted substream makes half the checks hold with the other
    sign; random bits ahead of the code (pre-lock bits) freeze an SPRT on
    'no code' and would fill short windows taken from the start."""
    rng = np.random.default_rng(31)
    for substream in (0, 1):
        y = _k7(rng, 30000)
        y[substream::2] ^= 1
        hits = pattern_scan(_flip(block_interleave(y, 8, 16)[37:], 0.02, rng))
        assert _block_hit(hits, 8, 16, 91), substream
    lead = rng.integers(0, 2, 2000).astype(np.uint8)
    x = np.concatenate([lead, _flip(block_interleave(_k7(rng, 30000), 8, 16)[37:],
                                    0.01, rng)])
    assert _block_hit(pattern_scan(x), 8, 16, (2000 + 91) % 128)
    x = np.concatenate([lead, conv_interleave(_k7(rng, 30000), 8, 4)[37:]])
    hits = pattern_scan(x, periods=(200,))
    assert hits and hits[0].kind == "convolutional"
    assert (hits[0].parameters["branches"], hits[0].parameters["delay"],
            hits[0].parameters["offset"]) == (8, 4, 3)
    assert hits[0].period == 200


def test_capture_on_a_block_boundary():
    """Offsets 0 and P-1 read different blocks from the capture's start;
    they are compared on the same bits (P-1 against P)."""
    for cut, offset in ((0, 0), (1, 2047)):
        rng = np.random.default_rng(1177)
        coded = _k7(rng, 30000)
        x = _flip(block_interleave(coded, 64, 32)[cut:], 0.03, rng)
        assert _block_hit(pattern_scan(x), 64, 32, offset), cut


def test_80211_bpsk_qpsk_keep_their_name():
    """For BPSK and QPSK the 802.11 interleaver is exactly the block
    (ncbps/16) x 16; it is reported by the standard's name, as the
    rank-profile sweep named these captures."""
    for ncbps, nbpsc in ((48, 1), (96, 2)):
        rng = np.random.default_rng(3)
        x = pn_interleave(_k7(rng, 4000), ieee80211_permutation(ncbps, nbpsc))[29:]
        top = identify_interleaver(x, max_L=256)[0]
        assert top.kind == "ieee80211" and _is_scan(top), ncbps
        p = top.parameters
        assert (p["ncbps"], p["nbpsc"], p["offset"]) == (ncbps, nbpsc, ncbps - 29)
        assert top.period == ncbps and "rows" not in p


@pytest.mark.parametrize("case", ["helical", "convolutional", "ieee80211"])
def test_scan_offsets_follow_the_engine_convention(case):
    """The engine skips `offset` bits, then de-interleaves: on every kind
    the scan reports, that gives back the code."""
    from dhwani.hypothesis.engine import _deinterleave_with
    rng = np.random.default_rng(5)
    coded = _k7(rng, 30000)
    if case == "helical":
        x = _flip(helical_interleave(coded, 32, 16, 1)[100:], 0.02, rng)
        want = {"rows": 32, "cols": 16, "step": 1, "offset": 412}
    elif case == "convolutional":
        x = _flip(conv_interleave(coded, 16, 2)[37:], 0.02, rng)
        want = {"branches": 16, "delay": 2, "offset": 11}
    else:
        x = pn_interleave(coded, ieee80211_permutation(192, 4))[37:]
        want = {"ncbps": 192, "nbpsc": 4, "offset": 155}
    hits = pattern_scan(x)
    assert hits and hits[0].kind == case
    assert {k: hits[0].parameters[k] for k in want} == want
    top = identify_convolutional(_deinterleave_with(x, hits[0]))[0]
    assert (top["K"], top["g1"], top["g2"]) == (7, 0o171, 0o133)
    assert top["syndrome_zero_rate"] > 0.8


@pytest.mark.parametrize("n,k,fcr,rows,cols,ber", [
    (204, 188, 0, 16, 16, 0.0), (255, 223, 1, 32, 32, 3e-4)])
def test_rs_under_other_geometries_and_errors(n, k, fcr, rows, cols, ber):
    rng = np.random.default_rng(n + rows)
    rs = RSCode(n, k, fcr=fcr)
    data = rng.integers(0, 256, k * 40, dtype=np.uint8)
    cws = b"".join(rs.encode(bytes(data[i:i + k])) for i in range(0, len(data), k))
    bits = np.unpackbits(np.frombuffer(cws, dtype=np.uint8))
    x = _flip(block_interleave(bits, rows, cols)[500:], ber, rng)
    hits = pattern_scan(x)
    P = rows * cols
    assert _block_hit(hits, rows, cols, (P - 500) % P)
    assert hits[0].parameters["code"].startswith(f"RS({n},{k})")


def test_degenerate_llrs_are_ignored():
    """All-zero or all-NaN LLRs carry no reliability: the scan reads the
    bits alone instead of dividing by a zero mean |t|."""
    rng = np.random.default_rng(12)
    x = _flip(block_interleave(_k7(rng, 30000), 8, 16)[37:], 0.02, rng)
    for soft in (np.zeros(len(x)), np.full(len(x), np.nan)):
        assert _block_hit(pattern_scan(x, llrs=soft), 8, 16, 91)
    top = identify_interleaver(x, max_L=256, llrs=np.zeros(len(x)))[0]
    assert top.kind == "block" and top.parameters["offset"] == 91


def test_old_sweep_still_finds_a_block_without_the_scan(monkeypatch):
    """The rank-profile sweep stays the fallback for what the scan misses
    (codes outside the registry, small helicals at low BER)."""
    monkeypatch.setattr(detect, "pattern_scan", lambda *a, **k: [])
    x = block_interleave(_k7(np.random.default_rng(1234), 8000), 8, 16)[37:]
    top = identify_interleaver(x, max_L=256)[0]
    assert top.kind == "block" and not _is_scan(top)
    assert (top.parameters["rows"], top.parameters["cols"]) == (8, 16)
    assert top.period == 128


def test_flat_profile_streams_keep_the_old_verdict():
    """No stubs: flag fill and random bits (flat rank profiles) are 'none',
    as before."""
    for bits in (_hdlc_fill(),
                 np.random.default_rng(2).integers(0, 2, 30000).astype(np.uint8)):
        assert pattern_scan(bits) == []
        hyps = identify_interleaver(bits, max_L=256)
        assert hyps[0].kind == "none" and not any(_is_scan(h) for h in hyps)


def test_plain_k7_at_3pct_is_not_interleaved():
    """A plainly coded stream: its check holds undilated (D = 1)."""
    rng = np.random.default_rng(7)
    x = _flip(_k7(rng, 30000), 0.03, rng)
    assert pattern_scan(x) == []
    hyps = identify_interleaver(x, max_L=256)
    assert hyps[0].kind == "none" and not any(_is_scan(h) for h in hyps)


def test_soft_values_follow_the_bits(rng):
    """LLRs are optional and must line up with the bits; misaligned ones
    are ignored rather than trusted."""
    coded = _k7(rng, 20000)
    x = _flip(block_interleave(coded, 8, 16)[37:], 0.02, rng)
    llrs = (1 - 2.0 * x) * rng.uniform(0.5, 4, len(x))
    for soft in (llrs, llrs[:-5], None):
        hits = pattern_scan(x, llrs=soft)
        assert hits and hits[0].kind == "block"
        assert (hits[0].parameters["rows"], hits[0].parameters["cols"],
                hits[0].parameters["offset"]) == (8, 16, 91)


def test_short_and_empty_inputs():
    for n in (0, 7, 100, 3000):
        bits = np.random.default_rng(n).integers(0, 2, n).astype(np.uint8)
        assert pattern_scan(bits) == []
        assert pattern_scan(bits, periods=(128, 96)) == []


def test_engine_passes_soft_values_to_the_interleaver_search(config,
                                                              monkeypatch):
    """The bit layer hands identify_interleaver the de-scrambled LLRs of the
    bits it searches: signs flipped wherever the descrambler flips a bit,
    magnitudes kept.  Whitened sync-word frames put a de-whitened option
    among the searched ones."""
    import copy
    from dhwani.common.models import BitStream, InterleaverHypothesis
    from dhwani.hypothesis import engine
    from dhwani.synth.factory import WaveformFactory
    rng = np.random.default_rng(4)
    w = lfsr.KNOWN_WHITENERS["IEEE802.11"]
    bits = lfsr.additive_scramble(WaveformFactory(seed=3).build_frames(400)[0],
                                  w["poly"], w["seed"])
    llrs = ((1 - 2.0 * bits) * rng.uniform(0.3, 5, len(bits))).astype(np.float32)
    seen = []

    def spy(b, **kw):
        seen.append((np.array(b), kw.get("llrs")))
        return [InterleaverHypothesis(kind="none", score=1.0)]
    monkeypatch.setattr(engine, "identify_interleaver", spy)
    monkeypatch.setattr(lfsr, "KNOWN_WHITENERS", {"IEEE802.11": w})
    cfg = copy.deepcopy(config)
    cfg.bitlayer.beam_width = 1
    engine.bitlayer_search([BitStream(bits=bits, llrs=llrs)], cfg.bitlayer,
                           cfg.fec, cfg.framing)
    assert any((b != bits).any() for b, _ in seen)
    for b, soft in seen:
        assert soft is not None and len(soft) == len(b)
        assert np.array_equal(soft < 0, b == 1)
        assert np.array_equal(np.abs(soft), np.abs(llrs))


def test_pattern_scan_cost():
    """It runs on every beam stream: random bits (the common case) and a
    coded interleaved stream.  The bounds are on CPU time (wall time on a
    shared machine swings 2-6x with the load), about 3x the measured cost;
    both are printed."""
    rng = np.random.default_rng(4)
    rand = rng.integers(0, 2, 200000).astype(np.uint8)
    coded = _flip(block_interleave(_k7(rng, 30000), 8, 16), 0.03, rng)
    cpu, wall = {}, {}
    for name, bits in (("random 200k", rand), ("8x16 K7 3% 60k", coded)):
        cpu[name] = wall[name] = np.inf
        for _ in range(3):
            t0, c0 = time.perf_counter(), time.process_time()
            out = pattern_scan(bits)
            wall[name] = min(wall[name], time.perf_counter() - t0)
            cpu[name] = min(cpu[name], time.process_time() - c0)
        assert (out == []) == (name == "random 200k")
    print("pattern_scan cost (cpu / wall):",
          {k: f"{cpu[k] * 1e3:.0f} / {wall[k] * 1e3:.0f} ms" for k in cpu})
    assert cpu["random 200k"] < 1.0 and cpu["8x16 K7 3% 60k"] < 2.0
