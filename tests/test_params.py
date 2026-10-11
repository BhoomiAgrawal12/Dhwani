import numpy as np

from dhwani.params.estimators import sample_rate_candidates
from dhwani.pipeline import Analyzer
from dhwani.synth.factory import WaveformFactory


def test_sample_rate_candidates_table():
    c = sample_rate_candidates(0.2)          # 9600 Bd at 48 kHz, 5 sps
    assert [x["sample_rate_hz"] for x in c] == [48e3, 96e3, 192e3, 250e3]
    assert all(x["verdict"] == "PROBABLE" and x["method"] for x in c)
    assert sample_rate_candidates(0.1234567) == []


def test_channelised_rates_are_candidates():
    """Off-air: SatNOGS/Zenodo captures are often channelised to 32 or
    40 kHz (HADES-D, the Vega-C cubesats); neither was in the table."""
    hz = lambda c: [x["sample_rate_hz"] for x in c]
    assert 40e3 in hz(sample_rate_candidates(2400 / 40e3))
    assert 32e3 in hz(sample_rate_candidates(1200 / 32e3))


def test_headerless_iq_gets_candidates_not_a_rate(tmp_path, config):
    """3.D: a 9600 Bd burst recorded at 48 kHz, written without metadata."""
    iq, _ = WaveformFactory(seed=4).generate(modulation="QPSK", sps=5.0,
                                             snr_db=25.0, n_frames=40)
    p = str(tmp_path / "x.iq")
    iq.astype(np.complex64).tofile(p)
    res = Analyzer(config, use_cache=False).analyze(p, no_ml=True,
                                                    stop_after="S5")
    assert res.recording_meta["sample_rate"] is None
    assert res.parameters.sample_rate is None
    rates = [c["sample_rate_hz"] for c in res.parameters.sample_rate_candidates]
    assert 48e3 in rates


import pytest  # noqa: E402


@pytest.mark.parametrize("mod,snr", [("2FSK", 15), ("4FSK", 20)])
def test_fsk_at_4_samples_per_symbol(tmp_path, config, mod, snr):
    """FSK is often sampled near 4 sps (and a mono WAV halves the rate).
    Smoothing windows >= the symbol length smear the tones; the detector
    must still keep the window that resolves a valid tone count."""
    iq, _ = WaveformFactory(seed=7).generate(
        modulation=mod, sps=4.0, snr_db=snr, cfo_norm=0.004,
        phase_offset=0.3, n_frames=60)
    p = str(tmp_path / "f.iq")
    iq.astype(np.complex64).tofile(p)
    res = Analyzer(config, use_cache=False).analyze(p, stop_after="S5")
    assert res.modulation.prediction == mod


def _sps40(tmp_path):
    iq, _ = WaveformFactory(seed=3).generate(modulation="QPSK", sps=40.0,
                                             snr_db=20.0, cfo_norm=0.002,
                                             phase_offset=0.3, n_frames=60)
    p = str(tmp_path / "s40.iq")
    iq.astype(np.complex64).tofile(p)
    return p


def test_symbol_rate_reported_per_recorded_sample(tmp_path, config):
    """The channeliser decimates a narrowband burst (rate_ratio 0.28 at 40
    sps); the normalised symbol rate a user reads must be per RECORDED
    sample (0.025), not per channelised sample (0.0895)."""
    res = Analyzer(config, use_cache=False).analyze(
        _sps40(tmp_path), datatype="complex64", stop_after="S5")
    assert abs(res.parameters.symbol_rate_norm_recording / 0.025 - 1) < 0.01


def test_symbol_rate_override_is_in_recording_units(tmp_path, config):
    res = Analyzer(config, use_cache=False).analyze(
        _sps40(tmp_path), datatype="complex64", stop_after="S5",
        overrides={"symbol_rate_norm": 0.025})
    p = res.parameters
    assert abs(p.symbol_rate_norm_recording - 0.025) < 1e-9
    assert abs(p.symbol_rate_norm * p.rate_ratio - 0.025) < 1e-9


@pytest.mark.parametrize("seed,hint_sps", [(0, 8.0), (9, 19.2)])
def test_oqpsk_rate_from_x2_pair_within_bandwidth(config, seed, hint_sps):
    """R5: repeated frames put symmetric x^2 lines near DC that outscore
    the +-Rs pair (seed 0 locked to Rs 0.018), and S4's |x|^2 rate can be
    far off for OQPSK (seed 9: 19.2 sps). Only a pair the occupied
    bandwidth allows is accepted, whatever the hint."""
    from dhwani.channelization import channelize
    from dhwani.conditioning import condition
    from dhwani.demod import demodulate
    from dhwani.detection import detect_signals
    iq, _ = WaveformFactory(seed=seed).generate(
        modulation="OQPSK", sps=8.0, snr_db=22.0, cfo_norm=0.004,
        phase_offset=0.3, n_frames=60)
    x, _ = condition(iq)
    seg = detect_signals(x, config.cfar)[0][0]
    r = demodulate(channelize(x, seg)["samples"], "OQPSK", hint_sps,
                   config.demod)
    assert abs(r.samples_per_symbol - 8.0) < 0.05
    assert r.demodulation_status == "GOOD"


def test_bpsk_at_low_snr_not_named_ook(tmp_path, config):
    """R5: OOK and BPSK share one bipolar table after S1, so the receiver
    trial ties on EVM and OOK's looser gate named every 8 dB BPSK burst
    OOK (5/5 seeds). BPSK is named whenever it passes its own trial."""
    iq, _ = WaveformFactory(seed=0).generate(
        modulation="BPSK", sps=8.0, snr_db=8.0, cfo_norm=0.004,
        phase_offset=0.3, n_frames=60)
    p = str(tmp_path / "b.iq")
    iq.astype(np.complex64).tofile(p)
    m = Analyzer(config, use_cache=False).analyze(
        p, sample_rate=1e6, datatype="complex64", no_ml=True,
        stop_after="S5").modulation
    assert m.prediction == "BPSK"
    assert "OOK" in [a[0] for a in m.alternatives]


def test_unscrambled_conv_qpsk_carrier_refinement(config):
    """R5/A8: unscrambled conv-coded QPSK puts an x^4 side-line at Rs that
    beat the carrier line; the 'residual' Rs/4 (0.031) shifted the burst
    out of the matched filter and timing failed. The refinement now only
    searches inside +-Rs/2 of the x^M carrier line."""
    from dhwani.channelization import channelize
    from dhwani.conditioning import condition
    from dhwani.demod import demodulate
    from dhwani.detection import detect_signals
    from dhwani.params import estimate_parameters
    iq, _ = WaveformFactory(seed=0).generate(
        modulation="QPSK", sps=8.0, snr_db=20.0, cfo_norm=0.004,
        phase_offset=0.3, n_frames=60,
        fec={"family": "convolutional", "K": 7, "generators": (0o171, 0o133)})
    x, _ = condition(iq)
    ch = channelize(x, detect_signals(x, config.cfar)[0][0])["samples"]
    p = estimate_parameters(ch, config.params)
    r = demodulate(ch, "QPSK", 8.0, config.demod,
                   cfo_norm=p.carrier_offset_norm)
    assert abs(r.cfo_applied_norm) < 0.005
    assert r.demodulation_status == "GOOD"


def test_receiver_trial_uses_the_s4_carrier(config):
    """Off-air (EntrySat): an off-centre detection box left a residual
    carrier of 0.124 cycles/sample after S3. S6 is given S4's carrier
    estimate and locks BPSK at EVM 11.7%; the S5 trials were not, failed
    BPSK at 67.9% and named QPSK. Both stages must use the same estimate."""
    from dhwani.modulation import classify_modulation
    from dhwani.params import estimate_parameters
    iq, _ = WaveformFactory(seed=1).generate(
        "BPSK", sps=2.5, snr_db=15, cfo_norm=0.12, phase_offset=0.3)
    p = estimate_parameters(iq, config.params)
    assert abs(p.carrier_offset_norm - 0.12) < 0.005
    m = classify_modulation(iq, p, config.modulation)
    assert m.prediction == "BPSK"
