import numpy as np

from dhwani.ingestion import load_recording, sniff_raw_iq


def _tone(n=20000, f=0.1):
    t = np.arange(n)
    return (np.exp(2j * np.pi * f * t) +
            0.05 * (np.random.default_rng(0).normal(size=n) +
                    1j * np.random.default_rng(1).normal(size=n))).astype(np.complex64)


def test_raw_iq_complex64(tmp_path):
    p = str(tmp_path / "a.iq")
    _tone().tofile(p)
    rec = load_recording(p)
    assert rec.datatype == "complex64"
    assert rec.sample_rate is None
    assert any("sample rate unknown" in w for w in rec.warnings)
    assert rec.samples.dtype == np.complex64


def test_raw_iq_int16_sniff(tmp_path):
    p = str(tmp_path / "b.iq")
    x = _tone()
    interleaved = np.empty(2 * len(x), dtype=np.int16)
    interleaved[0::2] = (x.real * 8000).astype(np.int16)
    interleaved[1::2] = (x.imag * 8000).astype(np.int16)
    interleaved.tofile(p)
    cands = sniff_raw_iq(p)
    assert cands[0]["dtype"] == "int16"
    rec = load_recording(p, sample_rate=48000)
    assert rec.sample_rate == 48000
    assert rec.datatype == "int16"


def test_wav_stereo_iq(tmp_path):
    import scipy.io.wavfile as wavfile
    p = str(tmp_path / "c.wav")
    x = _tone()
    stereo = np.stack([(x.real * 20000).astype(np.int16),
                       (x.imag * 20000).astype(np.int16)], axis=1)
    wavfile.write(p, 96000, stereo)
    rec = load_recording(p)
    assert rec.format == "wav"
    assert rec.sample_rate == 96000
    assert rec.channels == 2
    # spectrum should peak near +0.1 normalised
    X = np.abs(np.fft.fft(rec.samples[:8192]))
    peak = np.fft.fftfreq(8192)[np.argmax(X)]
    assert abs(peak - 0.1) < 0.01


def test_sigmf_sidecar(tmp_path):
    import json
    p = str(tmp_path / "d.sigmf-data")
    _tone().tofile(p)
    with open(str(tmp_path / "d.sigmf-meta"), "w") as f:
        json.dump({"global": {"core:datatype": "cf32_le",
                              "core:sample_rate": 250000.0},
                   "captures": [{"core:sample_start": 0,
                                 "core:frequency": 14.1e6}]}, f)
    rec = load_recording(p)
    assert rec.sample_rate == 250000.0
    assert rec.center_frequency == 14.1e6


def test_wav_mono_real_is_made_analytic(tmp_path, config):
    """1.A: a real-valued mono WAV must not reach CFAR with a mirrored image."""
    import scipy.io.wavfile as wavfile
    from dhwani.conditioning import condition
    from dhwani.detection import detect_signals
    rng = np.random.default_rng(3)
    n, sps = 1 << 17, 32
    bits = rng.integers(0, 2, n // sps + 1).repeat(sps)[:n]
    f = 0.15 + 0.01 * (2 * bits - 1)                      # real 2FSK burst
    x = np.cos(2 * np.pi * np.cumsum(f)) + 0.1 * rng.normal(size=n)
    p = str(tmp_path / "mono.wav")
    wavfile.write(p, 48000, x.astype(np.float32))
    rec = load_recording(p)
    assert rec.channels == 1
    assert any("complex baseband" in w for w in rec.warnings)
    assert rec.sample_rate == 24000 and rec.center_frequency == 12000
    segs, _ = detect_signals(condition(rec.samples)[0], config.cfar,
                             sample_rate=rec.sample_rate)
    top = max(segs, key=lambda s: s.snr_db)
    # 0.15*fs real -> (0.15 - 0.25) / 0.5 = -0.2 of the new rate, i.e.
    # centre frequency 12 kHz + offset -4.8 kHz = 7.2 kHz = 0.15 * 48 kHz
    assert abs(top.center_norm + 0.2) < 0.02
    assert abs(rec.center_frequency + top.absolute("center") - 7200) < 500
    # no mirrored image of comparable power anywhere else in the band
    # (FSK sidelobes still yield weak extra CFAR segments, as for IQ input)
    strong = [s for s in segs if s.snr_db > top.snr_db - 10]
    assert all(abs(s.center_norm + 0.2) < 0.1 for s in strong)


def test_wav_mono_baseband_audio_keeps_its_rate(tmp_path, config):
    """A mono WAV from an FM receiver holds the baseband data waveform,
    not an RF carrier. Mixing it by -fs/4 and halving the rate halves the
    samples per symbol and breaks the symbol rate; read as real at the
    file rate, S4 finds 2400 Bd."""
    import scipy.io.wavfile as wavfile
    from scipy.signal import firwin, lfilter
    from dhwani.pipeline import Analyzer
    rng = np.random.default_rng(4)
    n, sps = 1 << 17, 20                                  # 2400 Bd at 48 kHz
    nrz = (2.0 * rng.integers(0, 2, n // sps + 1) - 1).repeat(sps)[:n]
    x = lfilter(firwin(41, 0.12), 1, nrz) + 0.1 * rng.normal(size=n)
    p = str(tmp_path / "audio.wav")
    wavfile.write(p, 48000, (0.5 * x).astype(np.float32))
    rec = load_recording(p)
    assert rec.sample_rate == 48000 and rec.file_sample_rate == 48000
    assert any("baseband audio" in w for w in rec.warnings)
    res = Analyzer(config, use_cache=False).analyze(p, stop_after="S5")
    assert abs(res.parameters.symbol_rate_hz - 2400) < 24


def test_two_level_baseband_audio_is_named_2fsk(tmp_path, config):
    """Off-air (TY-2, TRISAT): FM-receiver audio of an FSK downlink was
    named BPSK, since two baseband levels pass the bipolar receiver. Out of
    an FM discriminator two levels are binary FSK (FSK/GFSK/GMSK alike)."""
    import scipy.io.wavfile as wavfile
    from scipy.signal import firwin, lfilter
    from dhwani.pipeline import Analyzer
    rng = np.random.default_rng(4)
    n, sps = 1 << 17, 20
    nrz = (2.0 * rng.integers(0, 2, n // sps + 1) - 1).repeat(sps)[:n]
    x = lfilter(firwin(41, 0.12), 1, nrz) + 0.1 * rng.normal(size=n)
    p = str(tmp_path / "audio.wav")
    wavfile.write(p, 48000, (0.5 * x).astype(np.float32))
    res = Analyzer(config, use_cache=False).analyze(p)
    assert res.modulation.prediction == "2FSK"
    assert res.demodulation["demodulation_status"] in ("GOOD", "DEGRADED")


def test_bpsk_on_a_low_audio_carrier_is_not_baseband():
    """Off-air (ITASAT 1): 1200 Bd BPSK on a 1.6 kHz SSB audio carrier has
    1% of its power below fs/100, so it was read as FM-receiver baseband
    audio. Baseband data peaks near DC; this band is >10 dB down there."""
    from dhwani.ingestion.wav import is_baseband
    rng = np.random.default_rng(5)
    n, sps, fs = 1 << 17, 40, 48000                      # 1200 Bd
    sym = np.sign(rng.standard_normal(n // sps + 1)).repeat(sps)[:n]
    x = sym * np.cos(2 * np.pi * 1600 / fs * np.arange(n))
    x += 0.05 * rng.standard_normal(n)
    assert not is_baseband(x)


import pytest  # noqa: E402


@pytest.mark.parametrize("name,rate,cf", [
    ("gqrx_20231010_153045_145800000_2400000_fc.raw", 2.4e6, 145.8e6),
    ("pass_2.048Msps_437.5MHz.iq", 2.048e6, 437.5e6),
    ("capture_250ksps.cfile", 250e3, None),
    ("capture.iq", None, None),
])
def test_rate_from_filename(tmp_path, name, rate, cf):
    """R4: recording software writes the sample rate / frequency into the
    file name; it is used (labelled 'filename') when nothing better exists."""
    p = str(tmp_path / name)
    _tone().tofile(p)
    rec = load_recording(p, datatype="complex64")
    assert rec.sample_rate == rate
    assert rec.center_frequency == cf
    if rate:
        assert rec.sample_rate_source == "filename"
        assert any("file name" in w for w in rec.warnings)


def test_user_rate_beats_filename(tmp_path):
    p = str(tmp_path / "x_2400000sps.iq")
    _tone().tofile(p)
    rec = load_recording(p, sample_rate=1e6, datatype="complex64")
    assert (rec.sample_rate, rec.sample_rate_source) == (1e6, "user")
