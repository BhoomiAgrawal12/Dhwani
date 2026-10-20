"""Cross-check Gate: the accepted code's BER against the BER the EVM implies.

S6 reads the channel BER off the EVM through the constellation's Gaussian
error model; the bit layer reads it off the accepted code's parity checks
(its Early Verdict ber_estimate).  A real code agrees with its signal; a
code claim far weaker than the signal allows is a likely false detection
and is downgraded before S11 reads the demodulation status."""
import logging
import math

import numpy as np
import pytest
from scipy.special import erfc

from dhwani import pipeline
from dhwani.common.models import (AnalysisHypothesis, AnalysisResult,
                                  DemodulationResult, FECHypothesis)
from dhwani.demod.constellations import (CONSTELLATIONS, bits_to_iq_symbols,
                                         slice_symbols)
from dhwani.demod.cross_check import (BER_RATIO, BER_SLACK, DOWNGRADE,
                                      ber_from_evm, cross_check_gate)
from dhwani.fec import probe, registry
from dhwani.fec.conv import ConvCode, conv_encode
from dhwani.fec.detect import identify_fec
from dhwani.pipeline import Analyzer
from dhwani.synth.factory import WaveformFactory

logging.disable(logging.INFO)

K7 = "conv K7 r1/2 (171,133)"
K7_FEC = {"family": "convolutional", "K": 7, "generators": (0o171, 0o133)}
# Gray-coded tables: every point's nearest distance is d_min and every
# nearest neighbour differs from it in one label bit
GRAY = ("8PSK", "16PSK", "32PSK", "OOK", "4ASK", "8ASK", "16QAM", "64QAM",
        "256QAM")
REAL = ("BPSK", "OOK", "4ASK", "8ASK")       # every point on the real axis


def _q(x):
    return 0.5 * erfc(np.asarray(x, dtype=np.float64) / math.sqrt(2.0))


def _channel(bits, modulation, evm, rng, axes=2):
    """Symbols of `bits` plus Gaussian noise of the given EVM (fraction),
    circular (axes=2, an RF carrier) or real-valued (axes=1, FM-receiver
    audio), sliced as S6 slices: (rx, hard, llrs, measured EVM %)."""
    syms = bits_to_iq_symbols(bits, modulation)
    s = evm / math.sqrt(axes)
    rx = syms + rng.normal(0, s, len(syms))
    if axes == 2:
        rx = rx + 1j * rng.normal(0, s, len(syms))
    rx = np.asarray(rx, dtype=np.complex128)
    hard, llrs, evm_pct = slice_symbols(
        rx, modulation, max(1e-4, float(np.var(np.abs(rx)) * 0.5)))
    return rx, hard, llrs, evm_pct


def _demod(bits, modulation, evm, rng, status="GOOD"):
    rx, hard, llrs, evm_pct = _channel(bits, modulation, evm, rng)
    return DemodulationResult(modulation=modulation,
                              symbols=rx.astype(np.complex64), hard_bits=hard,
                              llrs=llrs, evm_percent=round(evm_pct, 1),
                              carrier_locked=True, timing_locked=True,
                              demodulation_status=status)


def _claim(bits, llrs, name=K7, score=0.8):
    """An FEC hypothesis claiming registry code `name` on these bits, with
    the Code Probe's own Early Verdict for it (as identify_fec records)."""
    e = registry.REGISTRY[name]
    r = probe.probe_entry(probe.soft_values(bits, llrs), bits, e)
    g1, g2 = e["generators"]
    return FECHypothesis(
        family="convolutional", code_rate=0.5,
        parameters={"K": e["K"], "g1_octal": oct(g1), "g2_octal": oct(g2),
                    "rate": "1/2", "stream_offset": r["phase"],
                    "early_verdict": r["early_verdict"]},
        syndrome_zero_rate=r["syndrome_zero_rate"], pre_fec_ber=0.0,
        score=score)


def _node(*evidence):
    node = AnalysisHypothesis(stage="fec")
    for name, value in evidence:
        node.add_evidence(name, value)
    return node


def _k7(bits):
    return conv_encode(bits, ConvCode(7, (0o171, 0o133)), terminate=False)


# ------------------------------------------------------------- ber_evm ---
def test_ber_from_evm_qpsk_is_q_of_inverse_evm():
    for evm in (2.0, 5.0, 10.0, 18.0, 25.0, 32.0, 45.0):
        for mod in ("QPSK", "OQPSK", "BPSK"):
            assert ber_from_evm(evm, mod) == pytest.approx(
                float(_q(100.0 / evm)), rel=1e-9), (mod, evm)
    # conservative for BPSK on an RF carrier: its decisions see only the
    # in-phase half of circular noise, the exact AWGN value Q(sqrt(2)/EVM)
    for evm in (20.0, 35.0, 45.0):
        assert ber_from_evm(evm, "BPSK") > \
            float(_q(math.sqrt(2.0) * 100.0 / evm))
    assert ber_from_evm(0.0, "QPSK") == 0.0


def test_ber_from_evm_is_monotone_in_evm():
    evms = np.linspace(0.5, 60.0, 120)
    for mod in CONSTELLATIONS:
        v = np.array([ber_from_evm(e, mod) for e in evms])
        assert np.all(np.diff(v) >= 0), mod
        assert v[0] < 1e-6 < v[-1] <= 0.5, mod


def test_ber_from_evm_gray_tables_are_the_nearest_neighbour_formula():
    """(N_nn / k) Q(d_min / (2 sigma)) from the unit-power table, N_nn the
    mean number of neighbours at d_min, sigma the noise per decision axis:
    EVM / sqrt(2) on a complex table, all of the EVM on a real one."""
    for mod in GRAY:
        table, k = CONSTELLATIONS[mod]
        d = np.abs(table[:, None] - table[None, :])
        np.fill_diagonal(d, np.inf)
        d_min = d.min()
        n_nn = (d <= d_min * (1 + 1e-9)).sum(axis=1).mean()
        for evm in (3.0, 8.0, 15.0, 25.0):
            sigma = evm / 100.0 / (1.0 if mod in REAL else math.sqrt(2.0))
            want = n_nn / k * _q(d_min / (2.0 * sigma))
            assert ber_from_evm(evm, mod) == pytest.approx(float(want),
                                                           rel=1e-9), mod


def test_ber_from_evm_real_tables_take_all_the_noise(rng):
    """A real table (BPSK, OOK, ASK) decides on the in-phase axis alone and
    charges it the whole EVM: exact for real-valued FM-receiver audio,
    which S6 slices with these tables, and conservative for an RF carrier
    (as Q(1/EVM) is for BPSK).  Split over two axes, a true code on OOK
    audio at 44% EVM read 19x the BER that model implied and was vetoed."""
    for evm in (10.0, 30.0, 45.0):                # OOK is BPSK's +-1 table
        assert ber_from_evm(evm, "OOK") == pytest.approx(
            ber_from_evm(evm, "BPSK"), rel=1e-9)
    for mod, evm in (("OOK", 0.44), ("4ASK", 0.24), ("8ASK", 0.13)):
        table, k = CONSTELLATIONS[mod]
        bits = rng.integers(0, 2, k * 100000).astype(np.uint8)
        _rx, hard, _llrs, evm_pct = _channel(bits, mod, evm, rng, axes=1)
        ber = float((hard != bits).mean())
        model = ber_from_evm(evm_pct, mod)
        assert ber / 2 < model < ber * 2, (mod, ber, model)
        # a true code passes, even on a differential stream (2x the BER)
        assert 2 * ber <= BER_RATIO * model + BER_SLACK, (mod, ber, model)
        # on an RF carrier the quadrature half never reaches a decision
        _rx, hard, _llrs, evm_pct = _channel(bits, mod, evm, rng, axes=2)
        assert ber_from_evm(evm_pct, mod) >= float((hard != bits).mean())


def test_ber_from_evm_16qam_close_to_exact_awgn(rng):
    # exact Gray 16QAM in AWGN: (3 Q(x) + 2 Q(3x) - Q(5x)) / 4,
    # x = sqrt(Es / (5 N0)) = 1 / (sqrt(5) EVM)
    for evm in (8.0, 12.0, 16.0, 20.0):
        x = 1.0 / (math.sqrt(5.0) * evm / 100.0)
        exact = (3 * _q(x) + 2 * _q(3 * x) - _q(5 * x)) / 4
        assert ber_from_evm(evm, "16QAM") == pytest.approx(float(exact),
                                                           rel=0.01)
    # and a simulated channel read through the slicer's own EVM (decision
    # directed, as S6 reports it) at moderate SNR (Es/N0 16 dB)
    bits = rng.integers(0, 2, 4 * 100000).astype(np.uint8)
    _rx, hard, _llrs, evm_pct = _channel(bits, "16QAM", 0.16, rng)
    ber = float((hard != bits).mean())
    assert 1e-3 < ber < 4e-3
    assert ber_from_evm(evm_pct, "16QAM") == pytest.approx(ber, rel=0.25)


def test_ber_from_evm_non_gray_tables_track_simulation(rng):
    """The ring-major APSK and cross-QAM labels are not Gray, and APSK rings
    set different nearest distances (one 128APSK point in sixteen sits at
    the global d_min): per point, its own nearest neighbours weighted by the
    label bits they flip stay within 2x of the simulated BER (asserted at
    2.5x for the sampling noise), where the global (N_nn / k)
    Q(d_min / ...) form reads 128APSK ~40x low."""
    for mod, evm in (("128APSK", 0.08), ("32QAM", 0.125), ("16APSK", 0.14)):
        table, k = CONSTELLATIONS[mod]
        bits = rng.integers(0, 2, k * 40000).astype(np.uint8)
        _rx, hard, _llrs, evm_pct = _channel(bits, mod, evm, rng)
        ber = float((hard != bits).mean())
        model = ber_from_evm(evm_pct, mod)
        assert ber / 2.5 < model < ber * 2.5, (mod, ber, model)


def test_ber_from_evm_unknown_is_none():
    for mod in ("2FSK", "4FSK", "8FSK", "GMSK", "FM", "AM-DSB-WC", "UNKNOWN",
                None):
        assert ber_from_evm(10.0, mod) is None, mod
    assert ber_from_evm(None, "QPSK") is None
    assert ber_from_evm(float("nan"), "QPSK") is None
    assert ber_from_evm(float("inf"), "QPSK") is None
    assert ber_from_evm(-1.0, "QPSK") is None


# ------------------------------------------------------------- to_dict ---
def test_fec_to_dict_keeps_every_key_and_adds_cross_check():
    # positional construction is unchanged: the new field sits at the end
    h = FECHypothesis("convolutional", {"K": 7}, 0.5, 0.99,
                      np.zeros(8, np.uint8), 0.01, 0.9)
    d = h.to_dict()
    assert set(d) == {"family", "parameters", "code_rate",
                      "syndrome_zero_rate", "n_decoded_bits", "pre_fec_ber",
                      "score", "cross_check"}
    assert (d["family"], d["parameters"], d["code_rate"],
            d["syndrome_zero_rate"], d["n_decoded_bits"], d["pre_fec_ber"],
            d["score"]) == ("convolutional", {"K": 7}, 0.5, 0.99, 8, 0.01, 0.9)
    assert d["cross_check"] is None
    h.cross_check = {"ber_evm": 1e-5, "ber_code": 2e-5, "consistent": True}
    assert h.to_dict()["cross_check"] == {"ber_evm": 1e-5, "ber_code": 2e-5,
                                          "consistent": True}


# ---------------------------------------------------------------- gate ---
def test_code_forced_on_random_bits_fails_and_is_downgraded(rng):
    bits = rng.integers(0, 2, 20000).astype(np.uint8)
    demod = _demod(bits, "QPSK", 0.05, rng)              # a clean QPSK
    assert 4.0 < demod.evm_percent < 6.0
    fec = _claim(demod.hard_bits, demod.llrs)            # K=7 forced on it
    # random bits hold no K=7 check: a hard mean near 0 reads 0.15-0.5
    assert fec.parameters["early_verdict"]["ber_estimate"] > 0.1
    result = AnalysisResult(demodulation=demod.to_dict(), fec=fec)
    node = _node(("pre_structure", 0.9), ("interleaver", 0.5),
                 ("fec_syndrome", fec.syndrome_zero_rate))
    score0 = node.score

    check = cross_check_gate(result, demod, node)
    assert check is fec.cross_check
    assert check["consistent"] is False
    assert check["ber_code"] == fec.parameters["early_verdict"]["ber_estimate"]
    assert check["ber_evm"] == ber_from_evm(demod.evm_percent, "QPSK") < 1e-9
    assert check["ber_code"] > BER_RATIO * check["ber_evm"] + BER_SLACK
    # downgraded: score halved, a zero cross-check evidence on the node
    assert fec.score == pytest.approx(0.4)
    assert node.evidence["cross_check"] == 0.0 and node.score < score0
    # capped at DEGRADED in the dict AND the object S11's provenance reads
    assert demod.demodulation_status == "DEGRADED"
    assert result.demodulation["demodulation_status"] == "DEGRADED"
    warn = [w for w in result.warnings if "Cross-check Gate" in w]
    assert len(warn) == 1
    assert f"{check['ber_code']:.3g}" in warn[0]
    assert f"{check['ber_evm']:.3g}" in warn[0]
    assert "likely false detection" in warn[0]
    assert "GOOD capped at DEGRADED" in warn[0]
    assert result.to_dict()["fec"]["cross_check"] == check
    # a second pass changes nothing: the downgrade applies once
    evidence = dict(node.evidence)
    assert cross_check_gate(result, demod, node) is check
    assert fec.score == pytest.approx(0.4) and node.evidence == evidence
    assert len(result.warnings) == 1


def test_inconsistency_never_raises_the_status(rng):
    bits = rng.integers(0, 2, 8000).astype(np.uint8)
    for status, capped in (("GOOD", "DEGRADED"), ("DEGRADED", "DEGRADED"),
                           ("FAILED", "FAILED")):
        demod = _demod(bits, "QPSK", 0.05, rng, status=status)
        result = AnalysisResult(demodulation=demod.to_dict(),
                                fec=_claim(demod.hard_bits, demod.llrs))
        assert cross_check_gate(result, demod)["consistent"] is False
        assert demod.demodulation_status == capped
        assert result.demodulation["demodulation_status"] == capped
        # "capped" only when the status moved
        assert ("capped at" in result.warnings[0]) == (status == "GOOD")


def test_gate_reads_the_dict_without_a_demodulation_result(rng):
    """No DemodulationResult given: the EVM, modulation and status come
    from result.demodulation, and only the dict is capped."""
    bits = rng.integers(0, 2, 8000).astype(np.uint8)
    demod = _demod(bits, "QPSK", 0.05, rng)
    result = AnalysisResult(demodulation=demod.to_dict(),
                            fec=_claim(demod.hard_bits, demod.llrs))
    check = cross_check_gate(result)
    assert check["ber_evm"] == ber_from_evm(demod.evm_percent, "QPSK")
    assert check["consistent"] is False
    assert result.demodulation["demodulation_status"] == "DEGRADED"
    assert demod.demodulation_status == "GOOD"            # not handed over
    assert "QPSK EVM of" in result.warnings[0]


def test_crc_validated_chain_is_not_called_a_false_detection(rng):
    """A CRC pass through this very decode proves the code: the downgrade
    stands (dhwani.md), but the warning blames errors the model does not
    see rather than a false detection."""
    bits = rng.integers(0, 2, 8000).astype(np.uint8)
    demod = _demod(bits, "QPSK", 0.05, rng)
    result = AnalysisResult(demodulation=demod.to_dict(),
                            fec=_claim(demod.hard_bits, demod.llrs))
    node = _node(("crc", 1.0))
    node.status = "validated"
    assert cross_check_gate(result, demod, node)["consistent"] is False
    assert result.fec.score == pytest.approx(0.8 * DOWNGRADE)
    assert "CRC validates" in result.warnings[0]
    assert "false detection" not in result.warnings[0]


def test_real_code_through_noisy_16qam_passes(rng):
    """The two estimates agree on a real code: K=7 behind 16QAM at ~1%
    channel BER (EVM 19%, inside the 16QAM DEGRADED band)."""
    msg = rng.integers(0, 2, 12000).astype(np.uint8)
    coded = _k7(msg)
    demod = _demod(coded, "16QAM", 0.19, rng, status="DEGRADED")
    ber = float((demod.hard_bits != coded[:len(demod.hard_bits)]).mean())
    assert 0.003 < ber < 0.02
    fec = _claim(demod.hard_bits, demod.llrs)
    assert fec.parameters["early_verdict"]["verdict"] == "code"
    result = AnalysisResult(demodulation=demod.to_dict(), fec=fec)
    node = _node(("fec_syndrome", fec.syndrome_zero_rate))
    evidence0 = dict(node.evidence)

    check = cross_check_gate(result, demod, node)
    assert check["consistent"] is True
    # both read the same channel
    assert check["ber_code"] == pytest.approx(ber, rel=0.5)
    assert check["ber_evm"] == pytest.approx(ber, rel=0.5)
    # a consistent claim is left as it was
    assert fec.score == 0.8 and node.evidence == evidence0
    assert demod.demodulation_status == "DEGRADED"
    assert not result.warnings


@pytest.mark.parametrize("mod,evm", [("32PSK", 0.10), ("64QAM", 0.115),
                                     ("128APSK", 0.085)])
def test_real_code_at_a_dense_tables_degraded_edge_passes(rng, mod, evm):
    """Near a dense table's DEGRADED gate a true code reads 1.5-4% BER,
    above BER_SLACK: there the BER_RATIO x ber_evm term is what passes it
    (with 2.8-6x to spare)."""
    coded = _k7(rng.integers(0, 2, 12000).astype(np.uint8))
    demod = _demod(coded, mod, evm, rng, status="DEGRADED")
    result = AnalysisResult(demodulation=demod.to_dict(),
                            fec=_claim(demod.hard_bits, demod.llrs))
    check = cross_check_gate(result, demod)
    assert check["ber_code"] > BER_SLACK
    assert check["consistent"] is True and not result.warnings


@pytest.mark.parametrize("coded", [False, True], ids=["uncoded", "K7"])
@pytest.mark.parametrize("pattern", ["text frames", "beacon", "flag fill"])
def test_structured_data_through_the_detector(rng, config, pattern, coded):
    """Uncoded structured data biases wrong checks (DESIGN lesson): on a
    repeated text beacon identify_fec claims a K=7 code at szr 0.75, on
    HDLC-like flag fill one at szr 0.92 whose checks fail on one window in
    twelve (a code BER of 0.0113, just over BER_SLACK).  A clean QPSK EVM
    explains neither, so no such claim leaves the gate undowngraded; the
    same data behind a real K=7 code passes untouched."""
    if pattern == "text frames":
        bits = WaveformFactory(seed=5).build_frames(
            100, payload_mode="text")[0]
    elif pattern == "beacon":
        bits = np.tile(WaveformFactory(seed=5).build_frames(
            1, payload_mode="text")[0], 100)
    else:
        bits = np.tile(np.unpackbits(np.frombuffer(b"\x7e" * 4 + b"\x11\x22",
                                                   np.uint8)), 200)
    demod = _demod(_k7(bits) if coded else bits, "QPSK", 0.05, rng)
    fec = identify_fec(demod.hard_bits, config.fec, llrs=demod.llrs)[0]
    score0 = fec.score
    result = AnalysisResult(demodulation=demod.to_dict(), fec=fec)
    check = cross_check_gate(result, demod)
    if coded:
        assert fec.family == "convolutional" and check["consistent"] is True
        assert demod.demodulation_status == "GOOD" and not result.warnings
    elif fec.family != "none":
        assert check["consistent"] is False
        assert fec.score == pytest.approx(score0 * DOWNGRADE)
        assert demod.demodulation_status == "DEGRADED"
        assert len(result.warnings) == 1


def test_missing_number_is_no_verdict(rng):
    """No EVM (FSK) or no code estimate ('none'): consistent is None and
    nothing is touched, never a guess."""
    bits = rng.integers(0, 2, 8000).astype(np.uint8)
    fsk = DemodulationResult(modulation="2FSK", hard_bits=bits,
                             evm_percent=None, carrier_locked=True,
                             timing_locked=True, demodulation_status="GOOD")
    fec = _claim(bits, None)
    result = AnalysisResult(demodulation=fsk.to_dict(), fec=fec)
    check = cross_check_gate(result, fsk)
    assert check["ber_evm"] is None and check["consistent"] is None
    assert check["ber_code"] == fec.parameters["early_verdict"]["ber_estimate"]
    assert fec.score == 0.8 and fsk.demodulation_status == "GOOD"
    none = FECHypothesis(family="none", score=0.2,
                         parameters={"reason": "no code identified above "
                                               "threshold"})
    qpsk = _demod(bits, "QPSK", 0.05, rng)
    result = AnalysisResult(demodulation=qpsk.to_dict(), fec=none)
    check = cross_check_gate(result, qpsk)
    assert check["ber_code"] is None and check["consistent"] is None
    assert check["ber_evm"] is not None
    assert none.score == 0.2 and qpsk.demodulation_status == "GOOD"
    assert not result.warnings
    assert cross_check_gate(AnalysisResult(), qpsk) is None   # no FEC at all


def test_concatenated_reads_the_inner_code(rng):
    bits = rng.integers(0, 2, 8000).astype(np.uint8)
    inner = _claim(bits, None)
    cat = FECHypothesis(
        family="concatenated", code_rate=0.5 * 223 / 255,
        parameters={"inner": {"family": "convolutional", **inner.parameters},
                    "outer": {"family": "reed_solomon", "n": 255, "k": 223,
                              "early_verdict": {"ber_estimate": 0.0}}},
        syndrome_zero_rate=0.99, score=0.9)
    qpsk = _demod(bits, "QPSK", 0.05, rng)
    result = AnalysisResult(demodulation=qpsk.to_dict(), fec=cat)
    check = cross_check_gate(result, qpsk)
    assert check["ber_code"] == \
        inner.parameters["early_verdict"]["ber_estimate"]
    assert check["consistent"] is False and cat.score == pytest.approx(0.45)


def test_any_family_with_an_early_verdict_is_checked(rng):
    """The gate reads parameters['early_verdict'] whatever the family: an
    LDPC hypothesis that records one is checked like a convolutional one."""
    ldpc = FECHypothesis(family="ldpc", code_rate=0.5, score=0.9,
                         parameters={"n": 648, "k": 324, "early_verdict": {
                             "verdict": "code", "ber_estimate": 0.004}})
    bits = rng.integers(0, 2, 4000).astype(np.uint8)
    qpsk = _demod(bits, "QPSK", 0.05, rng)
    result = AnalysisResult(demodulation=qpsk.to_dict(), fec=ldpc)
    check = cross_check_gate(result, qpsk)
    assert check["ber_code"] == 0.004 and check["consistent"] is True


# ------------------------------------------------------------ pipeline ---
def _run(tmp_path, config, gen_kwargs):
    iq, _gt = WaveformFactory(seed=99).generate(**gen_kwargs)
    p = str(tmp_path / "sig.iq")
    iq.astype(np.complex64).tofile(p)
    return Analyzer(config, use_cache=False).analyze(p, sample_rate=1e6)


def _trace(res, stage):
    return next(t for t in res.pipeline_trace if t["stage"] == stage)["summary"]


def _s6_status(res):
    """The status S6 itself reported ("status GOOD, ...")."""
    return _trace(res, "S6").split(",")[0].split()[-1]


QPSK_K7 = dict(modulation="QPSK", sps=8.0, snr_db=16.0, cfo_norm=0.004,
               phase_offset=0.3, fec=K7_FEC, n_frames=80)


def test_conv_coded_qpsk_passes_the_gate(tmp_path, config):
    res = _run(tmp_path, config, QPSK_K7)
    assert res.fec.family == "convolutional" and res.fec.parameters["K"] == 7
    cc = res.fec.cross_check
    assert cc["consistent"] is True
    assert cc["ber_evm"] is not None and cc["ber_code"] is not None
    assert cc["ber_code"] <= BER_RATIO * cc["ber_evm"] + BER_SLACK
    assert res.to_dict()["fec"]["cross_check"] == cc
    # demodulation status unchanged, also where S11 reads it
    assert res.demodulation["demodulation_status"] == _s6_status(res) == "GOOD"
    assert res.payload_intelligence["provenance"]["demod_status"] == "GOOD"
    assert not any("Cross-check Gate" in w for w in res.warnings)
    assert not any("cross_check" in h.evidence for h in res.hypotheses)
    assert "Cross-check Gate" not in _trace(res, "S9")


def test_2fsk_pipeline_is_unaffected(tmp_path, config):
    """No EVM for FSK: the code's estimate is recorded, no verdict drawn."""
    res = _run(tmp_path, config, dict(modulation="2FSK", sps=8.0,
                                      snr_db=17.0, fec=K7_FEC, n_frames=80))
    cc = res.fec.cross_check
    assert cc["ber_evm"] is None and cc["consistent"] is None
    assert cc["ber_code"] is not None
    status = _s6_status(res)
    assert res.demodulation["demodulation_status"] == status
    assert res.payload_intelligence["provenance"]["demod_status"] == status
    assert not any("Cross-check Gate" in w for w in res.warnings)
    assert not any("cross_check" in h.evidence for h in res.hypotheses)


def test_gate_runs_before_s11(tmp_path, config, monkeypatch):
    """A false code claim (K=5 forced on a K=7 stream) reaches S11 already
    capped: the gate sits between the bit layer and the payload provenance."""
    real = pipeline.bitlayer_search

    def forced(streams, *args, **kwargs):
        out = real(streams, *args, **kwargs)
        s = streams[0]
        out["best"]["fec"] = _claim(s.bits, s.llrs, "conv K5 r1/2 (35,23)",
                                    score=0.9)
        return out

    monkeypatch.setattr(pipeline, "bitlayer_search", forced)
    res = _run(tmp_path, config, QPSK_K7)
    cc = res.fec.cross_check
    assert cc["consistent"] is False and res.fec.score == pytest.approx(0.45)
    assert _s6_status(res) == "GOOD"                    # what S6 measured
    assert res.demodulation["demodulation_status"] == "DEGRADED"
    assert res.payload_intelligence["provenance"]["demod_status"] == "DEGRADED"
    assert "Cross-check Gate" in _trace(res, "S9")
    warn = [w for w in res.warnings if "Cross-check Gate" in w]
    assert len(warn) == 1
    winner = [h for h in res.hypotheses if "cross_check" in h.evidence]
    assert len(winner) == 1
    # the frame score is the winning node's: it follows the downgrade
    assert res.frames is not None and res.frames.score == winner[0].score
    # the true K=7 decode still passes its CRC: not a false detection
    assert winner[0].status == "validated" and "CRC validates" in warn[0]
