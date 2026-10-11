"""End-to-end regression tests over the synthetic waveform factory.

Each test generates a signal with full known ground truth, writes it as a
raw .iq file, and checks the blind pipeline recovers the structure."""
import logging

import numpy as np
import pytest

from dhwani.pipeline import Analyzer
from dhwani.synth.factory import WaveformFactory

logging.disable(logging.INFO)


def _run(tmp_path, config, gen_kwargs, sample_rate=1e6):
    fac = WaveformFactory(seed=99)
    iq, gt = fac.generate(**gen_kwargs)
    p = str(tmp_path / "sig.iq")
    iq.astype(np.complex64).tofile(p)
    an = Analyzer(config, use_cache=False)
    return an.analyze(p, sample_rate=sample_rate), gt


def test_e2e_full_stack_qpsk(tmp_path, config):
    """QPSK + conv FEC + block interleaver + PN9 + framed CRC payload."""
    res, gt = _run(tmp_path, config, dict(
        modulation="QPSK", sps=8.0, snr_db=22.0, cfo_norm=0.006,
        phase_offset=0.5,
        fec={"family": "convolutional", "K": 7, "generators": (0o171, 0o133)},
        interleaver={"kind": "block", "rows": 8, "cols": 16},
        scrambler={"kind": "known_whitening", "name": "PN9-CC1101"},
        n_frames=80))
    assert res.modulation.prediction == "QPSK"
    assert abs(res.parameters.symbol_rate_norm - 0.125) < 0.002
    assert res.scrambler.name == "PN9-CC1101"
    assert res.interleaver.kind == "block"
    assert (res.interleaver.parameters["rows"],
            res.interleaver.parameters["cols"]) == (8, 16)
    assert res.fec.family == "convolutional"
    assert res.fec.parameters["K"] == 7
    assert res.fec.syndrome_zero_rate > 0.98
    assert res.frames.frame_length_bits == 96
    assert res.frames.sync_word_hex.startswith("eb90")
    assert res.frames.crc["name"] == "CRC-16-CCITT-FALSE"
    assert res.frames.crc["pass_fraction"] > 0.9


def test_e2e_plain_bpsk(tmp_path, config):
    """Uncoded, unscrambled BPSK: verdicts must honestly be 'none'."""
    res, gt = _run(tmp_path, config, dict(
        modulation="BPSK", sps=8.0, snr_db=18.0, cfo_norm=0.004,
        n_frames=80))
    assert res.modulation.prediction == "BPSK"
    assert res.interleaver is None or res.interleaver.kind in ("none",)
    assert res.fec is None or res.fec.family == "none"
    assert res.frames is not None and res.frames.frame_length_bits == 96
    assert res.frames.crc is not None
    assert res.frames.crc["pass_fraction"] > 0.9


def test_e2e_2fsk(tmp_path, config):
    res, gt = _run(tmp_path, config, dict(
        modulation="2FSK", sps=8.0, snr_db=17.0, n_frames=80))
    assert res.modulation.prediction == "2FSK"
    assert res.frames is not None
    assert res.frames.frame_length_bits == 96


def test_e2e_rs_coded(tmp_path, config):
    res, gt = _run(tmp_path, config, dict(
        modulation="QPSK", sps=8.0, snr_db=25.0, cfo_norm=0.003,
        fec={"family": "reed_solomon", "n": 255, "k": 223},
        n_frames=60))
    assert res.modulation.prediction == "QPSK"
    assert res.fec is not None and res.fec.family == "reed_solomon"
    assert res.fec.parameters["n"] == 255


def test_e2e_concatenated_rs_conv(tmp_path, config):
    """3.A: RS(255,223) outer + conv K=7 inner, both identified, CRC passes
    and the recovered payload bytes are the transmitted ones."""
    res, gt = _run(tmp_path, config, dict(
        modulation="QPSK", sps=8.0, snr_db=22.0, cfo_norm=0.004,
        fec={"family": "concatenated",
             "outer": {"family": "reed_solomon", "n": 255, "k": 223},
             "inner": {"family": "convolutional", "K": 7,
                       "generators": (0o171, 0o133)}},
        n_frames=80))
    assert res.fec is not None and res.fec.family == "concatenated"
    inner, outer = res.fec.parameters["inner"], res.fec.parameters["outer"]
    assert (inner["family"], inner["K"]) == ("convolutional", 7)
    assert (outer["family"], outer["n"], outer["k"]) == ("reed_solomon", 255, 223)
    assert res.frames is not None and res.frames.crc["pass_fraction"] > 0.9
    # every recovered 6-byte payload is a transmitted one; codewords cut
    # by the stream edges (first ~19 frames, padded tail) are lost
    sent = [bytes.fromhex(p) for p in gt.payloads]
    d = res.payload.data
    got = [d[i:i + 6] for i in range(0, len(d), 6)]
    assert all(g in sent for g in got)
    assert len(got) >= len(sent) // 2


def test_e2e_diagonal_helical(tmp_path, config):
    """3.B: conv code behind a helical ('diagonal') interleaver, full chain."""
    res, gt = _run(tmp_path, config, dict(
        modulation="QPSK", sps=8.0, snr_db=22.0, cfo_norm=0.004,
        fec={"family": "convolutional", "K": 7, "generators": (0o171, 0o133)},
        interleaver={"kind": "helical", "rows": 8, "cols": 16, "step": 3},
        n_frames=80))
    assert res.interleaver.kind == "helical"
    assert res.interleaver.label == "diagonal/helical"
    assert res.fec.family == "convolutional"
    assert res.frames is not None and res.frames.crc["pass_fraction"] > 0.9


def test_e2e_ieee80211_interleaver(tmp_path, config):
    """R3: 16QAM + conv K=7 behind the 802.11a/g bit interleaver (192
    coded bits per symbol, 4 per subcarrier): identified by name, CRC ok.
    Unwhitened: a whitener after this interleaver is the open R2 coupling
    (its phase alignment needs a period the permutation hides).

    The search reaches the right node late (118 s on a 4-core x86 laptop,
    over the 90 s default budget, which then reports the stop in a
    warning), so this correctness test lifts the budget."""
    import copy
    config = copy.deepcopy(config)
    config.bitlayer.time_budget_s = 600.0
    res, gt = _run(tmp_path, config, dict(
        modulation="16QAM", sps=8.0, snr_db=25.0, cfo_norm=0.004,
        phase_offset=0.3,
        fec={"family": "convolutional", "K": 7, "generators": (0o171, 0o133)},
        interleaver={"kind": "ieee80211", "ncbps": 192, "nbpsc": 4},
        n_frames=80))
    assert res.interleaver.kind == "ieee80211"
    assert res.interleaver.label == "802.11 bit interleaver"
    assert res.frames is not None and res.frames.crc["pass_fraction"] > 0.9


def test_e2e_pseudo_random_stays_unknown(tmp_path, config):
    """3.B: a PN interleaver's period is reported, its permutation is not
    guessed, and no frame structure is minted from the scrambled bits.
    Nothing validates, so every beam node is explored: a narrow beam keeps
    this fast (74 s at the default width, 469 s before A4)."""
    import copy
    config = copy.deepcopy(config)
    config.bitlayer.beam_width = 2
    res, gt = _run(tmp_path, config, dict(
        modulation="QPSK", sps=8.0, snr_db=22.0, cfo_norm=0.004,
        fec={"family": "convolutional", "K": 7, "generators": (0o171, 0o133)},
        interleaver={"kind": "pseudo_random", "period": 128},
        n_frames=80))
    assert res.interleaver.kind == "pseudo_random"
    assert res.interleaver.period == 128
    assert res.frames is None or not res.frames.crc


@pytest.mark.parametrize("fec", [{"family": "none"},
                                 {"family": "reed_solomon", "n": 255, "k": 223}])
def test_e2e_whitened_without_conv_code(tmp_path, config, fec):
    """A1: PN9-whitened frames with no convolutional code. Too few frames for
    the GF(2) rank scan to expose a period, so the whitener used to be
    skipped and no frame was ever found."""
    res, gt = _run(tmp_path, config, dict(
        modulation="QPSK", sps=8.0, snr_db=20.0, cfo_norm=0.004,
        phase_offset=0.3, fec=fec,
        scrambler={"kind": "known_whitening", "name": "PN9-CC1101"},
        n_frames=80))
    assert res.scrambler is not None and res.scrambler.name == "PN9-CC1101"
    assert res.frames is not None and res.frames.crc["pass_fraction"] > 0.9
    if fec["family"] == "reed_solomon":
        assert res.fec.family == "reed_solomon"


def test_e2e_noise_only(tmp_path, config):
    rng = np.random.default_rng(5)
    noise = (rng.normal(size=40000) + 1j * rng.normal(size=40000)).astype(np.complex64)
    p = str(tmp_path / "noise.iq")
    noise.tofile(p)
    an = Analyzer(config, use_cache=False)
    res = an.analyze(p, sample_rate=1e6)
    # noise may produce marginal detections, but never confident ones
    assert all(s.confidence < 0.5 for s in res.segments)
    if res.modulation is not None:
        assert res.modulation.confidence < 0.9 or \
            res.modulation.prediction == "UNKNOWN"


def test_long_noise_detects_nothing(tmp_path, config):
    """Off-air: 70 s noise-only captures (AstroBio, MTCube2) gave 53-56
    CFAR 'signals' (3-bin, 2-frame noise clumps) and nonsense S4 values.
    A long white-noise file must stop at S2 with no detections."""
    rng = np.random.default_rng(6)
    n = 1 << 21
    noise = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    p = str(tmp_path / "noise.iq")
    noise.tofile(p)
    res = Analyzer(config, use_cache=False).analyze(p, sample_rate=40e3)
    assert res.segments == []
    assert any("no signals detected" in w for w in res.warnings)


def test_uncoded_crc_chain_exits_early(tmp_path, config):
    """A4: a CRC-validated uncoded chain is conclusive; the search used to
    explore all 40 nodes (105 s) because early exit required an FEC
    syndrome rate that 'none' cannot have."""
    res, gt = _run(tmp_path, config, dict(
        modulation="BPSK", sps=8.0, snr_db=18.0, cfo_norm=0.004,
        n_frames=80))
    assert res.frames is not None and res.frames.crc["pass_fraction"] > 0.9
    # exits after the first validated node (3 explored); the old behaviour
    # went through all 40. Counting nodes does not depend on CPU speed.
    assert len(res.hypotheses) < 10


def test_search_budget_is_reported(tmp_path, config):
    """A4: when the bit-layer budget runs out the result says so rather
    than presenting a partial search as complete."""
    import copy
    config = copy.deepcopy(config)
    config.bitlayer.time_budget_s = 0.0
    res, _ = _run(tmp_path, config, dict(
        modulation="QPSK", sps=8.0, snr_db=20.0, cfo_norm=0.004,
        n_frames=40))
    assert any("budget" in w for w in res.warnings)
    assert res.frames is None
