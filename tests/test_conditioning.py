import numpy as np

from dhwani.conditioning import condition


def test_truncation_warns_and_records_span(rng):
    """1.F: an 8M-sample file whose burst starts after 5M samples."""
    n = 8 << 20
    x = (0.01 * rng.standard_normal(n)).astype(np.float32).astype(np.complex64)
    x[5 << 20:] += np.exp(2j * np.pi * 0.1 * np.arange(n - (5 << 20)))
    _, rep = condition(x)
    assert rep.analysed_samples == 1 << 22
    assert any("only the first 4,194,304 of 8,388,608" in w for w in rep.warnings)


def test_short_file_is_analysed_whole(rng):
    _, rep = condition(rng.standard_normal(5000).astype(np.complex64))
    assert rep.analysed_samples == 5000
    assert not any("only the first" in w for w in rep.warnings)


def test_dead_air_warning_does_not_claim_exclusion(rng):
    """1.G: dead air is reported, not excluded; the text must say so."""
    x = rng.standard_normal(1 << 16).astype(np.complex64)
    x[: 1 << 15] *= 1e-4
    _, rep = condition(x)
    w = [m for m in rep.warnings if "dead air" in m]
    assert w and "exclude" not in w[0]


def _classify_zero_cfo(tmp_path, config, modulation, seed=99, phase=0.0):
    from dhwani.pipeline import Analyzer
    from dhwani.synth.factory import WaveformFactory
    iq, _ = WaveformFactory(seed=seed).generate(
        modulation=modulation, sps=8.0, snr_db=22.0, cfo_norm=0.0,
        phase_offset=phase, n_frames=60)
    p = str(tmp_path / "z.iq")
    iq.astype(np.complex64).tofile(p)
    res = Analyzer(config, use_cache=False).analyze(
        p, sample_rate=1e6, no_ml=True, stop_after="S5")
    return res.modulation.prediction


def test_zero_cfo_constellations_survive_conditioning(tmp_path, config):
    """A well-tuned capture has ~zero carrier offset. Blind Gram-Schmidt
    'IQ correction' read the structure of short framed bursts as 4-8 deg
    of imbalance and its correction broke classification (16QAM ->
    UNKNOWN, 8PSK -> 64APSK)."""
    assert _classify_zero_cfo(tmp_path, config, "16QAM") == "16QAM"
    assert _classify_zero_cfo(tmp_path, config, "8PSK", phase=0.4) == "8PSK"


import pytest  # noqa: E402


@pytest.mark.parametrize("mod,snr", [("64QAM", 30), ("16PSK", 32),
                                     ("32QAM", 28)])
def test_zero_cfo_dense_constellations(tmp_path, config, mod, snr):
    """At zero CFO constant frame fields give the burst a real mean (11-17%
    of RMS); removing it as 'DC offset' skewed every point and S6 FAILED."""
    from dhwani.pipeline import Analyzer
    from dhwani.synth.factory import WaveformFactory
    iq, _ = WaveformFactory(seed=7).generate(
        modulation=mod, sps=8.0, snr_db=snr, cfo_norm=0.0, phase_offset=0.3,
        n_frames=60)
    p = str(tmp_path / "d.iq")
    iq.astype(np.complex64).tofile(p)
    res = Analyzer(config, use_cache=False).analyze(
        p, sample_rate=1e6, stop_after="S5")
    assert res.modulation.prediction == mod


def test_hardware_dc_removed_signal_mean_kept(rng):
    """Hardware DC is in every sample, noise-only stretches included; the
    burst's own mean is not. Only the former is subtracted."""
    n, burst = 1 << 16, slice(20000, 50000)
    x = (0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n)))
    x[burst] += 1.0 + 0.3j                     # burst with a strong own mean
    dc = 0.05 - 0.02j                          # LO leakage
    y, rep = condition((x + dc).astype(np.complex64))
    assert abs(rep.dc_offset - dc) < 0.005
    seg = y[burst] * np.sqrt((np.abs(x) ** 2).mean())
    assert abs(seg.mean() - (1.0 + 0.3j)) < 0.05


def test_zero_cfo_bpsk(tmp_path, config):
    """Was OOK: removing the burst's own mean degraded the PSK trial (EVM
    11% vs 3%) until the ASK receiver won the tie."""
    assert _classify_zero_cfo(tmp_path, config, "BPSK") == "BPSK"


def test_gate_never_calls_wrong_bits_good(config):
    """16PSK at zero CFO: the channeliser leaves a residual near Rs/32, where
    the symbol-domain carrier estimate is ambiguous by Rs/16; the wrong
    alias still lands on the grid (EVM 6%, bits random). GOOD must mean
    correct bits, so the gate has to say otherwise."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from validate_matrix import known_mod_ber
    from dhwani.channelization import channelize
    from dhwani.demod import demodulate
    from dhwani.detection import detect_signals
    from dhwani.params import estimate_parameters
    from dhwani.synth.factory import WaveformFactory
    iq, gt = WaveformFactory(seed=7).generate(
        modulation="16PSK", sps=8.0, snr_db=32, cfo_norm=0.0,
        phase_offset=0.3, n_frames=60)
    x, _ = condition(iq)
    ch = channelize(x, detect_signals(x, config.cfar)[0][0])
    p = estimate_parameters(ch["samples"], config.params)
    r = demodulate(ch["samples"], "16PSK", 1 / p.symbol_rate_norm,
                   config.demod, cfo_norm=p.carrier_offset_norm)
    ber = known_mod_ber(r, "16PSK", np.array(gt.info_bits, dtype=np.uint8))
    if r.demodulation_status == "GOOD":
        assert ber < 0.02
    else:
        assert any("ambiguous" in w for w in r.warnings)


def test_gmsk_gate_does_not_pass_noise(config, rng):
    """Noise demodulated as GMSK was gated GOOD, because a noise
    discriminator output is large and the gate only had a floor.
    Clean GMSK sits near 1x the theoretical deviation; noise far above."""
    from dhwani.demod import demodulate
    from dhwani.synth.factory import WaveformFactory
    w = (rng.standard_normal(1 << 17) + 1j * rng.standard_normal(1 << 17))
    for sps in (4, 8, 28):
        assert demodulate(w, "GMSK", sps, config.demod).demodulation_status \
            == "FAILED"
    iq, _ = WaveformFactory(seed=2).generate("GMSK", sps=8, snr_db=20)
    assert demodulate(iq, "GMSK", 8, config.demod).demodulation_status == "GOOD"


def _late_burst_file(tmp_path, rng, start=5_000_000, n=6_000_000):
    from dhwani.synth.factory import WaveformFactory
    iq, _ = WaveformFactory(seed=4).generate(
        modulation="QPSK", sps=8.0, snr_db=25.0, cfo_norm=0.004,
        phase_offset=0.3, n_frames=60)
    x = (0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
         ).astype(np.complex64)
    x[start:start + len(iq)] += iq.astype(np.complex64)
    p = str(tmp_path / "late.iq")
    x.tofile(p)
    return p


def test_burst_after_first_window_is_found(tmp_path, config, rng):
    """A5: S1/S2 used to see only the first 2^22 samples, so a burst at
    sample 5M was never detected; now the strongest window is analysed."""
    from dhwani.pipeline import Analyzer
    # datatype pinned: the sniffer reads this mostly-noise complex64 file as
    # int8 (a separate, recorded gap)
    res = Analyzer(config, use_cache=False).analyze(
        _late_burst_file(tmp_path, rng), sample_rate=1e6,
        datatype="complex64", stop_after="S5")
    a0 = res.recording_meta["analysed_start"]
    assert a0 > 0 and a0 <= 5_000_000 < a0 + res.recording_meta["analysed_samples"]
    assert res.modulation.prediction == "QPSK"
    assert any("window" in w for w in res.warnings)


def test_explicit_start_sample(tmp_path, config, rng):
    from dhwani.pipeline import Analyzer
    res = Analyzer(config, use_cache=False).analyze(
        _late_burst_file(tmp_path, rng), sample_rate=1e6,
        datatype="complex64", start_sample=4_900_000, stop_after="S5")
    assert res.recording_meta["analysed_start"] == 4_900_000
    assert res.modulation.prediction == "QPSK"
