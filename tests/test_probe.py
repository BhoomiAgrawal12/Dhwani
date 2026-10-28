"""S9 Code Registry, Code Probe and Early Verdict.

The registry compiles every library code into its parity checks once; the
probe scores "does this stream satisfy code X at alignment a?" from soft
bits; the early verdict (an SPRT) turns the per-check values into
code / no code / undecided."""
import numpy as np

from dhwani.bits.ambiguity import enumerate_ambiguities
from dhwani.common.models import BitStream, DemodulationResult
from dhwani.demod.constellations import bits_to_iq_symbols, slice_symbols
from dhwani.fec import probe, registry
from dhwani.fec.conv import ConvCode, conv_encode
from dhwani.fec.detect import identify_convolutional, identify_fec, identify_rs
from dhwani.fec.rs import RSCode

K7 = "conv K7 r1/2 (171,133)"


def _k7(rng, n_info):
    msg = rng.integers(0, 2, n_info).astype(np.uint8)
    return msg, conv_encode(msg, ConvCode(7, (0o171, 0o133)), terminate=False)


def _flip(bits, ber, rng):
    out = bits.copy()
    out[rng.random(len(out)) < ber] ^= 1
    return out


def _qpsk_llrs(coded, esn0, rng):
    """S6 slicer output for Gray QPSK at the given Es/N0 (linear), with the
    noise variance S7 hands the slicer (var(|syms|)/2)."""
    syms = bits_to_iq_symbols(coded, "QPSK")
    sigma = np.sqrt(1.0 / (2.0 * esn0))
    rx = syms + rng.normal(0, sigma, len(syms)) + 1j * rng.normal(0, sigma, len(syms))
    hard, llrs, _ = slice_symbols(rx, "QPSK",
                                  max(1e-4, float(np.var(np.abs(rx)) * 0.5)))
    return hard, llrs


def _syndromes(bits, taps, period):
    starts = np.arange(0, len(bits) - taps[-1], period)
    return np.bitwise_xor.reduce(bits[starts[:, None] + taps[None, :]], axis=1)


# 1 --------------------------------------------------------------------
def test_registry_checks_hold_on_own_encoder_output(rng):
    convs = registry.conv_entries()
    # K3..K9 rate 1/2 and 1/3 (+ LTE) in both orders, 4 punctures x 2 orders
    assert len(convs) == 14 + 16 + 8
    names = {e["name"] for e in convs}
    assert {K7, "conv K7 r1/2 (133,171)", "conv K7 r3/4 (171,133)",
            "conv K7 r7/8 (133,171)", "conv K7 r1/3 (133,171,165)",
            "conv K3 r1/3 (7,7,5)"} <= names
    msg = rng.integers(0, 2, 2000).astype(np.uint8)
    for e in convs:
        coded = registry.encode(e, msg)
        assert e["checks"], e["name"]
        assert len({c["weight"] for c in e["checks"]}) == 1    # all minimum
        extra = [e["odd_check"]] if e["odd_check"] else []
        for c in e["checks"] + extra:
            taps = c["taps"]
            assert taps[0] < e["period"] and c["span"] == taps[-1] + 1
            assert c["weight"] == len(taps)
            assert not _syndromes(coded, taps, e["period"]).any(), e["name"]
        if e["odd_check"]:
            assert e["odd_check"]["weight"] % 2 == 1
            assert e["odd_check"]["weight"] >= e["checks"][0]["weight"]
    k7 = registry.REGISTRY[K7]
    assert k7["period"] == 2 and k7["checks"][0]["weight"] == 10
    # both generators have odd weight: no dual vector sees an inversion
    assert k7["odd_check"] is None
    assert any(np.array_equal(c["taps"], registry.basic_check(7, 0o171, 0o133))
               for c in k7["checks"])
    k3 = registry.REGISTRY["conv K3 r1/2 (7,5)"]
    assert k3["odd_check"] is not None and k3["checks"][0]["weight"] == 5
    # g2 = g3 makes v2 + v3 = 0 a weight-2 check inside one step
    k3t = registry.REGISTRY["conv K3 r1/3 (5,7,7)"]
    assert np.array_equal(k3t["checks"][0]["taps"], [1, 2])
    assert registry.REGISTRY["conv K7 r3/4 (171,133)"]["period"] == 4
    # the bundled LDPC codes are registered beside REGISTRY: the LDPC
    # Pre-screen scores them per codeword alignment, not probe_entries
    assert [e["name"] for e in registry.ldpc_entries()] == [
        "802.11n 648 r1/2", "802.11n 648 r3/4", "AR4JA k1024 r1/2",
        "AR4JA k1024 r4/5"]
    assert not set(registry.LDPC_REGISTRY) & set(registry.REGISTRY)


def test_registry_rs_entries():
    rs = {(e["n"], e["k"], e["fcr"], e["generator"]): e
          for e in registry.rs_entries()}
    assert (204, 188, 0, 2) in rs and (255, 223, 112, 11) in rs
    dvb = rs[(204, 188, 0, 2)]
    assert dvb["m"] == 8 and list(dvb["roots"]) == list(range(16))
    assert registry.rs_entry({"n": 204, "k": 188, "prim": 0x11d, "fcr": 0,
                              "generator": 2}) is dvb


# 2 --------------------------------------------------------------------
def test_k7_at_5pct_ber_is_code_only_for_its_own_entry(rng):
    _, coded = _k7(rng, 60000)
    noisy = _flip(coded, 0.05, rng)
    t = probe.soft_values(noisy)
    results = probe.probe_entries(t, noisy, list(registry.REGISTRY.values()))
    assert set(results) == set(registry.REGISTRY)
    for name, r in results.items():
        if name == K7:
            assert r["early_verdict"]["verdict"] == "code"
            assert (r["phase"], r["sign"]) == (0, 1)
            assert abs(r["early_verdict"]["ber_estimate"] - 0.05) < 0.01
            assert r["z"] > 20
        else:
            assert r["early_verdict"]["verdict"] == "no code", name
    # the right generators at the wrong phase are no code either
    res = probe.probe_checks(t, noisy, [registry.basic_check(7, 0o171, 0o133)], 2)
    assert res["state"][0, 0, 0] == 1 and (res["state"][0, 1] == -1).all()


# 3 --------------------------------------------------------------------
def test_soft_probe_decides_code_at_10pct_ber(rng):
    msg, coded = _k7(rng, 6000)
    hard, llrs = _qpsk_llrs(coded, 1.2816 ** 2, rng)   # Q(sqrt(Es/N0)) = 10%
    assert 0.085 < (hard != coded).mean() < 0.115
    # the old fixed threshold: hard syndrome-zero rate below min_rate
    syn = _syndromes(hard, registry.basic_check(7, 0o171, 0o133), 2)
    assert 1 - syn.mean() < 0.7
    soft = probe.probe_entry(probe.soft_values(hard, llrs), hard,
                             registry.REGISTRY[K7])
    hard_only = probe.probe_entry(probe.soft_values(hard), hard,
                                  registry.REGISTRY[K7])
    assert soft["early_verdict"]["verdict"] == "code" and soft["phase"] == 0
    assert soft["z"] > hard_only["z"]
    # the BER is read from the hard parities: errors sit on the less
    # confident bits, so the soft mean alone would read it ~20% low
    ber = float((hard != coded).mean())
    assert abs(soft["early_verdict"]["ber_estimate"] - ber) < 0.012
    hits = identify_convolutional(hard, llrs=llrs)
    assert hits and (hits[0]["K"], hits[0]["g1"], hits[0]["g2"],
                     hits[0]["offset"]) == (7, 0o171, 0o133, 0)
    top = hits[0]
    assert top["early_verdict"]["verdict"] == "code"
    assert top["check_weight"] == 10 and top["z"] > 5
    assert top["syndrome_zero_rate"] < 0.7 and not top["syndrome_is_one"]


def test_soft_viterbi_used_when_llrs_given(config, rng):
    msg, coded = _k7(rng, 6000)
    hard, llrs = _qpsk_llrs(coded, 10 ** (4.0 / 10), rng)  # Eb/N0 1 dB
    hard_dec = identify_fec(hard, config.fec, try_rs=False, try_ldpc=False)[0]
    soft_dec = identify_fec(hard, config.fec, try_rs=False, try_ldpc=False,
                            llrs=llrs)[0]
    assert soft_dec.family == hard_dec.family == "convolutional"
    assert soft_dec.parameters["early_verdict"]["verdict"] == "code"
    errs = [int((h.decoded_bits[:len(msg)] != msg).sum())
            for h in (hard_dec, soft_dec)]
    assert errs[1] < errs[0]


# 4 --------------------------------------------------------------------
def test_random_bits_are_no_code_for_every_entry():
    rng = np.random.default_rng(5)
    # weight-24/32 punctured checks are planned at M1_MIN: ~6k checks each
    # per phase and sign, and period 8 needs 8 bits per check
    bits = rng.integers(0, 2, 400000).astype(np.uint8)
    t = probe.soft_values(bits)
    by_period = {}
    for e in registry.conv_entries():
        by_period.setdefault(e["period"], []).append(e)
    for period, group in by_period.items():
        taps = [c["taps"] for e in group for c in
                e["checks"] + ([e["odd_check"]] if e["odd_check"] else [])]
        res = probe.probe_checks(t, bits, taps, period)
        # every check x phase x sign
        assert res["state"].shape == (len(taps), period, 2)
        assert (res["state"] == -1).all(), [e["name"] for e in group]
    for e in registry.rs_entries():
        res = probe.probe_rs(bits[:120000], e)
        assert res["alignment_state"].shape == (8, e["n"])
        assert (res["alignment_state"] == -1).all(), e["name"]
        assert res["early_verdict"]["verdict"] == "no code"
    assert identify_convolutional(bits) == []
    assert identify_rs(bits, [{"n": 204, "k": 188, "prim": 0x11d, "fcr": 0,
                               "generator": 2}]) == []


# 5 --------------------------------------------------------------------
def test_early_verdict_thresholds_and_keys():
    upper = np.log((1 - probe.BETA) / probe.ALPHA)
    lower = np.log(probe.BETA / (1 - probe.ALPHA))
    ev = probe.early_verdict(np.ones(1000), weight=1)     # m1 = 0.8
    assert set(ev) == {"verdict", "llr", "checks_used", "false_alarm",
                       "ber_estimate"}
    assert ev["verdict"] == "code" and ev["checks_used"] == 64
    assert ev["llr"] >= upper and ev["false_alarm"] == probe.ALPHA
    assert ev["ber_estimate"] == 0.0
    ev = probe.early_verdict(np.zeros(1000), weight=1)
    assert ev["verdict"] == "no code" and ev["checks_used"] == 64
    assert ev["llr"] <= lower and ev["ber_estimate"] == 0.5
    # the LLR moves by (m1*x - m1^2/2)/var0 per value: x = m1/2 never moves
    ev = probe.early_verdict(np.full(5000, 0.4), weight=1)
    assert ev["verdict"] == "undecided" and ev["checks_used"] == 5000
    # exact crossing point: +0.5 per value, upper threshold 13.82 -> 28th
    ev = probe.early_verdict(np.ones(100), m1=1.0, block=1)
    assert ev["verdict"] == "code" and ev["checks_used"] == 28
    # a larger null variance needs proportionally more evidence
    ev = probe.early_verdict(np.ones(100), m1=1.0, block=1, var0=2.0)
    assert ev["checks_used"] == 56
    # a partial final block is used
    ev = probe.early_verdict(np.ones(70), weight=1, block=100)
    assert ev["verdict"] == "code" and ev["checks_used"] == 70
    # m1 floor: a weight-40 check is planned at M1_MIN, not 0.8**40, so a
    # mean of M1_MIN still decides (0.08 per block of 64 -> 173 blocks)
    ev = probe.early_verdict(np.full(20000, probe.M1_MIN), weight=40)
    assert ev["verdict"] == "code" and ev["checks_used"] == 173 * 64
    # BER inversion from all values: mean (1-2p)^w
    ev = probe.early_verdict(np.full(3000, 0.9 ** 4), weight=4)
    assert ev["verdict"] == "code" and abs(ev["ber_estimate"] - 0.05) < 1e-9
    assert probe.early_verdict(np.full(500, -0.3), weight=2)["ber_estimate"] == 0.5
    # Bernoulli model (RS clean-codeword indicators, one value per step)
    p1 = (1 - probe.RS_EXPECTED_BER) ** (8 * 255)
    ev = probe.early_verdict(np.array([0, 0, 1, 0, 0]), model="bernoulli",
                             block=1, p0=1e-12, p1=p1,
                             ber_model=lambda r: 1 - r ** (1 / 2040))
    assert ev["verdict"] == "code" and ev["checks_used"] == 3
    assert abs(ev["ber_estimate"] - (1 - 0.2 ** (1 / 2040))) < 1e-12
    ev = probe.early_verdict(np.zeros(40), model="bernoulli", block=1,
                             p0=1e-12, p1=p1)
    assert ev["verdict"] == "no code" and ev["checks_used"] == 16
    # the vectorised form agrees column by column
    cols = np.stack([np.ones(300), np.zeros(300), np.full(300, 0.4)], axis=1)
    many = probe.early_verdicts(cols, weight=1)
    assert [m["verdict"] for m in many] == ["code", "no code", "undecided"]
    assert many[0] == probe.early_verdict(cols[:, 0], weight=1)
    # the test stops at its first crossing; a verdict that all the values
    # contradict (z against Z_CONFIRM) is undecided
    noise = np.random.default_rng(0).choice([-1.0, 1.0], 10000)
    ev = probe.early_verdict(np.concatenate([np.ones(64), noise]), weight=1)
    assert ev["verdict"] == "undecided" and ev["llr"] >= upper
    ev = probe.early_verdict(np.concatenate([np.zeros(2000), np.ones(10000)]),
                             weight=1)
    assert ev["verdict"] == "undecided" and ev["llr"] <= lower
    assert probe.early_verdict(np.zeros(0))["verdict"] == "undecided"


# 6 --------------------------------------------------------------------
def _rs_stream(rng, n, k, fcr, generator, n_cw):
    rs = RSCode(n, k, fcr=fcr, generator=generator)
    data = [bytes(rng.integers(0, 256, k, dtype=np.uint8)) for _ in range(n_cw)]
    stream = b"".join(rs.encode(d) for d in data)
    return np.unpackbits(np.frombuffer(stream, dtype=np.uint8)), data


def test_rs_alignment_and_dvb_204_188(config, rng):
    assert RSCode(204, 188).m == 8
    assert any((c["n"], c["k"]) == (204, 188) for c in config.fec.rs_candidates)
    bits, data = _rs_stream(rng, 204, 188, 0, 2, 20)
    cut = bits[8 * (2 * 204 + 37) + 3:]      # mid-codeword, mid-byte start
    hits = identify_rs(cut, config.fec.rs_candidates)
    top = hits[0]
    assert (top["n"], top["k"], top["fcr"]) == (204, 188, 0)
    # bits[5:] starts on original symbol 446; codeword 3 starts at 612
    assert (top["bit_offset"], top["symbol_offset"]) == (5, 166)
    assert top["syndrome_zero_rate"] == 1.0
    assert top["early_verdict"]["verdict"] == "code"
    hyp = identify_fec(cut, config.fec)[0]
    assert hyp.family == "reed_solomon"
    assert (hyp.parameters["n"], hyp.parameters["k"]) == (204, 188)
    assert hyp.parameters["early_verdict"]["verdict"] == "code"
    got = np.packbits(hyp.decoded_bits).tobytes()
    assert got == b"".join(data[3:])


def test_rs_255_223_alignment_with_errors(config, rng):
    bits, _ = _rs_stream(rng, 255, 223, 1, 2, 12)
    bits = bits.copy()
    bits[rng.choice(len(bits), 12, replace=False)] ^= 1   # a few symbol errors
    cut = bits[8 * 100 + 6:]
    top = identify_rs(cut, config.fec.rs_candidates)[0]
    # (255,239) fcr=1 has a subset of the roots, so it matches too; the
    # code with more roots satisfied is the stronger (and right) claim
    assert (top["n"], top["k"], top["fcr"]) == (255, 223, 1)
    assert (top["bit_offset"], top["symbol_offset"]) == (2, 154)
    assert top["early_verdict"]["verdict"] == "code"
    assert 0 < top["early_verdict"]["ber_estimate"] < 0.01


def test_rs_candidate_sharing_most_roots_is_refuted(config, rng):
    """fcr 0 shares 31 of fcr 1's 32 roots: an fcr-1 codeword whose symbols
    XOR to zero (c(1) = 0) passes it too, and one clean window decides the
    Early Verdict.  The fcr-1 codewords that fail fcr 0 refute it.  (This
    guards identify_rs's refutation step; the old quick-pass sweep never
    took one clean window as proof.)"""
    rs = RSCode(255, 223, fcr=1)
    for _ in range(4096):            # 1 in 256 codewords has c(1) = 0
        lucky = rs.encode(bytes(rng.integers(0, 256, 223, dtype=np.uint8)))
        if np.bitwise_xor.reduce(np.frombuffer(lucky, np.uint8)) == 0:
            break
    else:
        raise AssertionError("no codeword with c(1) = 0")
    rest = [rs.encode(bytes(rng.integers(0, 256, 223, dtype=np.uint8)))
            for _ in range(9)]
    bits = np.unpackbits(np.frombuffer(lucky + b"".join(rest), np.uint8))
    fcr0 = registry.rs_entry({"n": 255, "k": 223, "prim": 0x11d, "fcr": 0,
                              "generator": 2})
    assert probe.probe_rs(bits, fcr0)["early_verdict"]["verdict"] == "code"
    hits = identify_rs(bits, config.fec.rs_candidates)
    assert hits and all(h["fcr"] == 1 for h in hits)
    assert (hits[0]["n"], hits[0]["k"]) == (255, 223)


def test_rs_hit_keys_and_codeword_count(config, rng):
    """Every codeword is tested, the last one too (it ends where the stream
    does), and the hit keeps the old keys."""
    bits, _ = _rs_stream(rng, 255, 239, 1, 2, 12)
    top = identify_rs(bits, config.fec.rs_candidates)[0]
    assert (top["n"], top["k"], top["fcr"]) == (255, 239, 1)
    assert (top["bit_offset"], top["symbol_offset"]) == (0, 0)
    assert top["n_codewords_tested"] == 12
    assert top["rate"] == top["syndrome_zero_rate"] == 1.0


# 7 --------------------------------------------------------------------
def test_idle_runs_do_not_make_a_code(config):
    rng = np.random.default_rng(3)
    parts = []
    for _ in range(6):
        parts += [np.zeros(4000, np.uint8),
                  rng.integers(0, 2, 4000).astype(np.uint8),
                  np.ones(3000, np.uint8)]
    bits = np.concatenate(parts)
    for llrs in (None, (1 - 2.0 * bits) * 9.0):          # hard, confident soft
        t = probe.soft_values(bits, llrs)
        for name, r in probe.probe_entries(
                t, bits, list(registry.REGISTRY.values())).items():
            assert r["early_verdict"]["verdict"] != "code", name
        assert identify_convolutional(bits, llrs=llrs) == []
    assert identify_rs(bits, config.fec.rs_candidates) == []
    assert identify_fec(bits, config.fec, try_ldpc=False)[0].family == "none"
    # n equal bytes are a codeword of every fcr >= 1 full-length RS code:
    # preamble or flag fill is not evidence either
    rand = rng.integers(0, 2, 40000).astype(np.uint8)
    fill = np.unpackbits(np.frombuffer(bytes([0x55]) * 300, np.uint8))
    assert identify_rs(np.concatenate([rand[:20000], fill, rand[20000:]]),
                       config.fec.rs_candidates) == []


def test_structured_uncoded_data_is_no_code(config, monkeypatch):
    """Uncoded data with structure: frames whose sync word and header recur,
    readable text payloads, a beacon repeating one frame.  Wrong checks
    pick up small persistent biases that a long stream turns into a large
    z, and the Viterbi guard cannot reject them (a wrong K=7 code
    re-encodes uncoded bits at ~13% < 20%); the sweep's own decoys can."""
    from dhwani.fec import detect
    from dhwani.synth.factory import WaveformFactory
    frames = WaveformFactory(seed=99).build_frames(480)[0]
    text = WaveformFactory(seed=5).build_frames(160, payload_mode="text")[0]
    beacon = np.tile(np.random.default_rng(1).integers(0, 2, 1024)
                     .astype(np.uint8), 24)
    for bits in (frames, text, beacon):
        assert identify_convolutional(bits) == []
        assert identify_fec(bits, config.fec, try_ldpc=False)[0].family == "none"
    # judged against N(0, 1) instead of the decoys they would read "code"
    monkeypatch.setattr(detect, "decoy_null",
                        lambda z, w: lambda zc, wc, sign=1: sign * zc)
    assert len(identify_convolutional(frames)) > 10
    assert len(identify_convolutional(beacon)) > 10


def test_rate_third_code_is_not_a_rate_half_code(config):
    """LTE's rate-1/3 K=7 code: one rate-1/2 check is a real dual vector at
    one window alignment in three, so it holds on a third of the windows
    (syndrome-zero rate 2/3, a huge z) - and no decode reaches the BER it
    reads.  Under the old 0.7 rate gate this was fec "none"; it stays so."""
    info = np.random.default_rng(3).integers(0, 2, 10000).astype(np.uint8)
    lte = conv_encode(info, ConvCode(7, (0o133, 0o171, 0o165)), terminate=False)
    hits = identify_convolutional(lte)
    assert hits and all(h["syndrome_zero_rate"] < 0.7 for h in hits)
    assert identify_fec(lte, config.fec, try_ldpc=False)[0].family == "none"


def test_early_code_verdicts_must_hold_on_the_whole_stream():
    """Over ~4000 sequential tests an early upward fluctuation crosses the
    SPRT bound now and then (each of these random QPSK streams gave a false
    code); the whole stream then disagrees."""
    for seed in (12, 265):
        r = np.random.default_rng(10 ** 6 + seed)
        hard, llrs = _qpsk_llrs(r.integers(0, 2, 20000).astype(np.uint8),
                                10 ** 0.4, r)
        assert identify_convolutional(hard) == []
        assert identify_convolutional(hard, llrs=llrs) == []


def test_random_preamble_does_not_hide_a_code():
    """A capture that starts on random bits: the SPRT settles "no code" on
    the preamble, but over the whole stream the code clears min_rate, as
    the old fixed-rate sweep found."""
    rng = np.random.default_rng(8)
    _, coded = _k7(rng, 20000)
    noisy = _flip(coded, 0.005, rng)
    bits = np.concatenate([rng.integers(0, 2, 8000).astype(np.uint8), noisy])
    top = identify_convolutional(bits)[0]
    assert (top["K"], top["g1"], top["g2"], top["offset"]) == (7, 0o171, 0o133, 0)
    assert top["syndrome_zero_rate"] > 0.85


def test_single_substream_inversion_is_seen(config):
    """A rotation ambiguity inverts one substream.  The basic dual check
    has odd tap counts on both substreams (K=7 and K=9 generators have odd
    weight), so its syndrome turns constant 1; the lighter checks u(D)(g2,
    g1) with u(1) = 0 have even counts on both and stay 0, so the sign must
    not come from them (it picked the wrong inversion patterns)."""
    for K, gens in ((7, (0o171, 0o133)), (9, (0o753, 0o561))):
        for seed in range(3):
            rng = np.random.default_rng(seed)
            msg = rng.integers(0, 2, 10000).astype(np.uint8)
            coded = conv_encode(msg, ConvCode(K, gens), terminate=False)
            for j in (0, 1):
                x = coded.copy()
                x[j::2] ^= 1
                x = _flip(x, 0.02, rng)
                top = identify_convolutional(x)[0]
                assert (top["K"], top["g1"], top["g2"]) == (K, *gens)
                assert top["syndrome_is_one"], (K, seed, j)
                if (K, seed, j) == (7, 2, 0):
                    hyp = identify_fec(x, config.fec, try_rs=False,
                                       try_ldpc=False)[0]
                    assert hyp.parameters["substream_inversion"] in ([1, 0], [0, 1])
                    d, m = hyp.decoded_bits[100:9000], msg[100:9000]
                    assert min((d != m).sum(), (d != 1 - m).sum()) < 20
    # an odd-weight check sees a whole-stream inversion
    k3 = registry.REGISTRY["conv K3 r1/2 (7,5)"]
    c = registry.encode(k3, np.random.default_rng(9).integers(0, 2, 4000)
                        .astype(np.uint8))
    for inv in (0, 1):
        r = probe.probe_entry(probe.soft_values(c ^ inv), c ^ inv, k3)
        assert r["early_verdict"]["verdict"] == "code"
        assert r["inverted"] is bool(inv)


def test_repeated_coded_frames_keep_sign_and_ber(config):
    """A beacon: one CRC'd frame sent 300 times, K=7, 1% BER.  The check
    values repeat with the frame and only windows holding an error are
    unique, so nothing may weigh repeats differently from fresh data (an
    exact-repeat mask kept just the errored windows: a negative mean, the
    wrong inversion, a garbage decode)."""
    from dhwani.synth.factory import WaveformFactory
    msg = np.tile(WaveformFactory(seed=99).build_frames(1)[0], 300)
    coded = conv_encode(msg, ConvCode(7, (0o171, 0o133)), terminate=False)
    noisy = _flip(coded, 0.01, np.random.default_rng(5))
    hyp = identify_fec(noisy, config.fec, try_rs=False, try_ldpc=False)[0]
    assert hyp.family == "convolutional"
    assert hyp.parameters["substream_inversion"] == [0, 0]
    assert np.array_equal(hyp.decoded_bits[:len(msg) - 10], msg[:len(msg) - 10])
    ber = float((noisy != coded).mean())
    assert abs(hyp.parameters["early_verdict"]["ber_estimate"] - ber) < 0.002


def test_nan_or_misaligned_llrs_carry_no_reliability(rng):
    bits = rng.integers(0, 2, 40000).astype(np.uint8)
    llrs = (1 - 2.0 * bits) * rng.uniform(0.2, 4, len(bits))
    llrs[1234] = np.nan
    t = probe.soft_values(bits, llrs)
    assert np.isfinite(t).all() and t[1234] == 0
    assert identify_convolutional(bits, llrs=llrs) == []
    # LLRs of another length belong to another stream: unused, even where
    # both arrays are longer than max_test_bits
    _, coded = _k7(rng, 3000)
    other = rng.normal(0, 3, len(coded) - 1000)
    hard = identify_convolutional(coded, max_test_bits=4000)[0]
    assert identify_convolutional(coded, max_test_bits=4000,
                                  llrs=other)[0]["z"] == hard["z"]


def test_llrs_without_reliability_fall_back_to_the_bits(rng, config):
    """All-zero, all-NaN or vanishingly small LLRs carry no reliability: the
    probe uses the hard decisions instead of dividing by a^w = 0."""
    _, coded = _k7(rng, 4000)
    hard = identify_convolutional(coded)[0]
    for llrs in (np.zeros(len(coded)), np.full(len(coded), np.nan),
                 (1 - 2.0 * coded) * 1e-9):
        assert np.array_equal(probe.soft_values(coded, llrs), 1 - 2.0 * coded)
        top = identify_convolutional(coded, llrs=llrs)[0]
        assert (top["K"], top["g1"], top["g2"]) == (hard["K"], hard["g1"],
                                                    hard["g2"])
        assert identify_fec(coded, config.fec, llrs=llrs)[0].family == \
            "convolutional"


def test_short_and_empty_inputs(config):
    k7 = [registry.basic_check(7, 0o171, 0o133)]
    rs = registry.rs_entry(config.fec.rs_candidates[1])
    for n in (0, 7, 100):
        bits = np.random.default_rng(n).integers(0, 2, n).astype(np.uint8)
        assert identify_convolutional(bits) == []
        assert identify_rs(bits, config.fec.rs_candidates) == []
        res = probe.probe_checks(probe.soft_values(bits), bits, k7, 2)
        assert (res["state"] == 0).all()           # too few checks to decide
        assert probe.probe_rs(bits, rs)["early_verdict"]["verdict"] == "undecided"
        assert identify_fec(bits, config.fec)[0].family == "none"


# 8 --------------------------------------------------------------------
def _signs_agree(s):
    assert s.llrs is not None and len(s.llrs) == len(s.bits)
    nz = s.llrs != 0
    assert nz.mean() > 0.99
    assert np.array_equal(s.llrs[nz] < 0, s.bits[nz] == 1)


def test_differential_and_fsk_llrs_agree_with_their_bits(rng):
    syms = bits_to_iq_symbols(rng.integers(0, 2, 4000).astype(np.uint8), "QPSK")
    rx = syms + 0.35 * (rng.normal(size=len(syms)) + 1j * rng.normal(size=len(syms)))
    streams = enumerate_ambiguities(rx, "QPSK")
    assert any(s.hypothesis["differential"] for s in streams)
    for s in streams:
        _signs_agree(s)
    base = streams[0]
    diff = next(s for s in streams if s.hypothesis["differential"]
                and s.hypothesis["rotation"] == base.hypothesis["rotation"]
                and s.hypothesis["iq_swap"] == base.hypothesis["iq_swap"])
    assert diff is streams[1]
    a, b = base.llrs[1:].astype(float), base.llrs[:-1].astype(float)
    ok = (np.abs(a) < 20) & (np.abs(b) < 20)      # atanh stays finite here
    assert ok.sum() > 100
    boxplus = 2 * np.arctanh(np.tanh(a[ok] / 2) * np.tanh(b[ok] / 2))
    assert np.allclose(diff.llrs[1:][ok], boxplus, atol=1e-4)
    assert (np.abs(diff.llrs[1:]) <= np.minimum(np.abs(a), np.abs(b)) + 1e-4).all()
    assert diff.llrs[0] == base.llrs[0]

    from dhwani.pipeline import _fsk_streams
    bits = rng.integers(0, 2, 3000).astype(np.uint8)
    llrs = ((1 - 2.0 * bits) * rng.uniform(0.1, 6, len(bits))).astype(np.float32)
    fsk = _fsk_streams(DemodulationResult(modulation="2FSK", hard_bits=bits,
                                          llrs=llrs))
    assert len(fsk) == 4
    for s in fsk:
        _signs_agree(s)


def test_bitlayer_search_hands_aligned_llrs_to_fec(config, monkeypatch, rng):
    """Soft values follow the bits through descrambling and de-interleaving.
    The interleaver and whitener searches are not under test: a fixed
    interleaver hypothesis and one short library whitener keep it fast."""
    import copy
    from dhwani.common.models import InterleaverHypothesis
    from dhwani.hypothesis import engine
    from dhwani.interleaving import block_interleave
    from dhwani.scrambling import lfsr
    _, coded = _k7(rng, 8000)
    w = lfsr.KNOWN_WHITENERS["IEEE802.11"]
    bits = lfsr.additive_scramble(block_interleave(coded, 8, 16), w["poly"],
                                  w["seed"])
    llrs = ((1 - 2.0 * bits) * rng.uniform(0.5, 4, len(bits))).astype(np.float32)
    seen, real = [], engine.identify_fec

    def spy(b, cfg, *a, llrs=None, **kw):
        seen.append((b, llrs))
        return real(b, cfg, *a, llrs=llrs, **kw)
    monkeypatch.setattr(engine, "identify_fec", spy)
    monkeypatch.setattr(engine, "identify_interleaver", lambda b, **kw: [
        InterleaverHypothesis(kind="block", score=20.0,
                              parameters={"rows": 8, "cols": 16, "offset": 0}),
        InterleaverHypothesis(kind="none", score=0.1)])
    monkeypatch.setattr(lfsr, "KNOWN_WHITENERS", {"IEEE802.11": w})
    cfg = copy.deepcopy(config)
    cfg.bitlayer.beam_width = 1
    out = engine.bitlayer_search([BitStream(bits=bits, llrs=llrs)],
                                 cfg.bitlayer, cfg.fec, cfg.framing)
    # (whitener, none) x (block, none), each with its own LLRs
    assert len(seen) == 4
    for b, soft in seen:
        _signs_agree(BitStream(bits=b, llrs=soft))
    best = out["best"]
    assert best["candidate"]["whitener"]["name"] == "IEEE802.11"
    assert best["interleaver"].kind == "block"
    assert best["fec"].family == "convolutional"
    assert best["fec"].parameters["soft_decision"]
