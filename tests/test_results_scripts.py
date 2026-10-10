"""Phase 4: measurement scripts write results/*.json with provenance."""
import json
import os
import sys

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, SCRIPTS)

import provenance  # noqa: E402


def test_write_result_has_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "RESULTS", str(tmp_path))
    path = provenance.write_result("x", {"timings": [1]})
    d = json.load(open(path))
    assert d["timings"] == [1]
    for k in ("generated_utc", "git_commit", "cpu", "python", "numpy", "data"):
        assert k in d["provenance"]
    assert d["provenance"]["data"].startswith("synthetic")


def test_confusion_matrix_and_per_class_accuracy():
    """4.D: raw predictions in the matrix, accuracy per true class."""
    import validate_matrix as vm
    rows = [{"modulation": "BPSK", "classified": "OOK", "class_top1": True,
             "class_top2": True},
            {"modulation": "8PSK", "classified": "16PSK", "class_top1": False,
             "class_top2": True},
            {"modulation": "8PSK", "classified": "8PSK", "class_top1": True,
             "class_top2": True},
            {"modulation": "FM", "verdict": "PASS"}]
    sc = vm.classification_scores(rows)
    assert sc["confusion_matrix"] == {"BPSK": {"OOK": 1},
                                      "8PSK": {"16PSK": 1, "8PSK": 1}}
    assert sc["per_class"]["8PSK"]["top1_accuracy"] == 0.5
    assert sc["per_class"]["8PSK"]["top2_accuracy"] == 1.0


def test_bitlayer_case_scores_demo_chain(config):
    """4.C: one fast case (block + conv, 20 dB) scores fully correct."""
    import validate_bitlayer as vb
    from dhwani.pipeline import Analyzer
    r = vb.run_case("block", "convolutional", 20, 11,
                    Analyzer(config, use_cache=False))
    assert r["interleaver_correct"] and r["fec_correct"] and r["crc_pass"]
    assert r["payload_ber"] == 0.0 and r["frames_recovered"] >= 0.5


def test_offair_scorer_on_a_known_recording(tmp_path, monkeypatch):
    """4.F: the scorer itself, on a synthetic stand-in (the real
    recordings are supplied by the user, never fabricated here)."""
    import validate_offair as vo
    from dhwani.synth.factory import WaveformFactory
    from dhwani.synth.writers import write_recording
    monkeypatch.setattr(provenance, "RESULTS", str(tmp_path))
    monkeypatch.setattr(vo, "ROOT", str(tmp_path))
    iq, gt = WaveformFactory(seed=99).generate(
        modulation="QPSK", sps=8.0, snr_db=22.0, cfo_norm=0.006,
        phase_offset=0.5, fec={"family": "convolutional", "K": 7},
        interleaver={"kind": "block", "rows": 8, "cols": 16},
        scrambler={"kind": "known_whitening", "name": "PN9-CC1101"},
        n_frames=80)
    write_recording(iq, gt, str(tmp_path / "r.wav"), sample_rate=1e6)
    (tmp_path / "m.json").write_text(json.dumps([{
        "file": "r.wav", "license": "synthetic test stand-in",
        "expect": {"modulation": "QPSK", "symbol_rate_hz": 125000,
                   "fec": "convolutional", "crc": "CRC-16-CCITT-FALSE",
                   "interleaver": "pseudo_random"}}]))
    assert vo.main(["--manifest", str(tmp_path / "m.json")]) == 0
    d = json.load(open(tmp_path / "offair_report.json"))
    assert d["provenance"]["data"].startswith("off-air")
    checks = d["cases"][0]["checks"]
    assert checks == {"modulation": True, "symbol_rate_hz": True,
                      "fec": True, "crc": True, "interleaver": False}


def test_readme_block_renders_from_results(tmp_path, monkeypatch):
    """5.A/5.B: README numbers come only from results/*.json; --check
    fails when the block is stale."""
    import render_readme as rr
    (tmp_path / "results").mkdir()
    readme = tmp_path / "README.md"
    readme.write_text(f"x\n{rr.BEGIN}\n{rr.END}\ny\n")
    monkeypatch.setattr(rr, "ROOT", str(tmp_path))
    monkeypatch.setattr(rr, "README", str(readme))
    assert rr.main([]) == 0 and rr.main(["--check"]) == 0
    assert "Not generated yet" in readme.read_text()
    prov = {"data": "synthetic", "git_commit": "abc", "cpu": "cpu",
            "python": "3", "generated_utc": "t"}
    (tmp_path / "results" / "benchmark.json").write_text(json.dumps(
        {"provenance": prov, "timings": [
            {"operation": "op", "seconds": 1.234, "target": "< 2 s"}]}))
    assert rr.main(["--check"]) == 1
    rr.main([])
    assert "| op | 1.23 s | < 2 s |" in readme.read_text()


def test_analog_verdict_counts_classification():
    """A13: an analog signal demodulated only because the true modulation
    was supplied must not PASS when blind classification named it wrong."""
    import validate_matrix as vm
    assert vm.analog_verdict(True, "GOOD") == ("PASS", None)
    v, why = vm.analog_verdict(False, "GOOD")
    assert v == "DEGRADED" and "classification" in why
    assert vm.analog_verdict(True, "FAILED")[0] == "FAIL"


def test_samplerate_case_scores_and_never_applies(config):
    """A12: a headerless 9600 Bd file at 48 kHz: right symbol rate, true
    rate among the candidates, sample rate itself left unknown."""
    import validate_samplerate as vs
    from dhwani.pipeline import Analyzer
    r = vs.run_case(48e3, 9600, 3, Analyzer(config, use_cache=False))
    assert r["symbol_rate_ok"] and r["true_rate_found"]
    assert r["reported_sample_rate"] is None


def test_plots_from_results(tmp_path, monkeypatch):
    """A14: charts are drawn from result files, well-formed, and every
    value is also present as text (not colour alone)."""
    import xml.dom.minidom
    import plots
    res = tmp_path / "results"
    res.mkdir()
    prov = {"git_commit": "abc"}
    (res / "validation_report.json").write_text(json.dumps({
        "provenance": prov, "per_class": {
            "BPSK": {"top1_accuracy": 0.5}, "BPSK (CFO 0)": {"top1_accuracy": 1.0},
            "FM": {"top1_accuracy": 0.0}},
        "snr_curves": {"PSK": [
            {"snr_db": 0, "n": 10, "top1_accuracy": 0.3, "top1_ci95": [0.1, 0.6]},
            {"snr_db": 30, "n": 10, "top1_accuracy": 1.0, "top1_ci95": [0.7, 1.0]}]}}))
    (res / "bitlayer_report.json").write_text(json.dumps({
        "provenance": prov, "settings": {"snrs": [20]}, "cases": [
            {"interleaver": "none", "fec": "ldpc", "crc_pass": True,
             "snr_db": 20,
             "frames_recovered": 0.89, "interleaver_correct": True,
             "fec_correct": True}]}))
    monkeypatch.setattr(plots, "RESULTS", str(res))
    monkeypatch.setattr(plots, "OUT", str(res / "charts"))
    plots.main()
    for name, text in (("modulation_top1", "50%"), ("bitlayer_grid", "89% frames"),
                       ("snr_curves", "95% CI 10%-60%")):
        svg = (res / "charts" / f"{name}.svg").read_text()
        xml.dom.minidom.parseString(svg)
        assert text in svg


def test_dirty_flag_ignores_results_dir():
    """Writing results/ must not mark the next run of the batch dirty."""
    path = os.path.join(provenance.RESULTS, "benchmark.json")
    if not os.path.exists(path):
        return
    before = provenance.provenance()["git_dirty"]
    with open(path) as f:
        original = f.read()
    try:
        with open(path, "a") as f:
            f.write(" ")
        assert provenance.provenance()["git_dirty"] == before
    finally:
        with open(path, "w") as f:
            f.write(original)


def test_wilson_interval_and_snr_curves():
    """R6: confidence intervals and per-family SNR curves."""
    import validate_matrix as vm
    assert vm.wilson(0, 0) == [0.0, 0.0]
    lo, hi = vm.wilson(9, 10)
    assert lo < 0.9 < hi and hi <= 1.0 and lo > 0.5
    sweep = [{"modulation": "QPSK", "snr_db": 3, "class_top1": i < 6,
              "ber": 0.1} for i in range(10)] + \
            [{"modulation": "16QAM", "snr_db": 3, "class_top1": True,
              "ber": 0.0}]
    c = vm.snr_curves(sweep)
    assert c["PSK"][0]["top1_accuracy"] == 0.6 and c["PSK"][0]["n"] == 10
    assert c["QAM"][0]["median_ber"] == 0.0


def test_bitlayer_scorer_gives_no_credit_when_chain_stopped():
    """A run that stops before S8 reports no interleaver and no FEC; that
    must not score as a correct 'none' (the 12 dB rows read il=+ fec=+
    after 0.1 s)."""
    import validate_bitlayer as vb
    from dhwani.common.models import AnalysisResult
    res = AnalysisResult()
    assert not vb.interleaver_correct(res, vb.INTERLEAVERS["none"])
    assert not vb.fec_correct(res, vb.FECS["none"])


def test_offair_scores_sample_rate_applied_or_candidate():
    """PS (i) off-air: a rate the tool applied must be right; with none
    applied (headerless IQ) the true rate must be among the candidates,
    and the report says which."""
    import validate_offair as vo
    from dhwani.common.models import AnalysisResult, SignalParameters

    class Stub:
        def __init__(self, applied, cands):
            self.applied, self.cands = applied, cands

        def analyze(self, *a, **k):
            r = AnalysisResult(recording_meta={
                "sample_rate": self.applied,
                "sample_rate_source": "wav_header" if self.applied else "unknown"})
            r.parameters = SignalParameters(
                sample_rate_candidates=[{"sample_rate_hz": c}
                                        for c in self.cands])
            return r

    e = {"file": "x.iq", "license": "test",
         "expect": {"sample_rate_hz": 48000}}
    assert vo.score(e, Stub(48000.0, []), True)["checks"] == \
        {"sample_rate_hz": True}
    assert not vo.score(e, Stub(96000.0, [48000]), True)["all_correct"]
    hit = vo.score(e, Stub(None, [96000, 48000]), True)
    assert hit["all_correct"]
    assert hit["sample_rate_source"] == "candidates (not applied)"
    assert not vo.score(e, Stub(None, [96000]), True)["all_correct"]


def test_demo_wav_matches_its_seed(tmp_path):
    """The committed examples/demo_chain.wav is what make_demo_report.py
    regenerates (within 1 LSB), so the demo cannot drift from the code."""
    import wave
    import numpy as np
    import make_demo_report as mdr
    fresh = str(tmp_path / "demo_chain.wav")
    mdr.make_wav(fresh)

    def read(path):
        with wave.open(path) as w:
            assert (w.getnchannels(), w.getframerate()) == (2, mdr.SAMPLE_RATE)
            return np.frombuffer(w.readframes(w.getnframes()), np.int16)
    a, b = read(fresh), read(mdr.WAV)
    assert a.shape == b.shape
    assert np.abs(a.astype(int) - b).max() <= 1


def test_validate_offair_out_dir(tmp_path, monkeypatch):
    """--out-dir directs offair_report.json to the specified folder."""
    import validate_offair as vo
    monkeypatch.setattr(provenance, "RESULTS", provenance.RESULTS)
    monkeypatch.setattr(vo, "ROOT", str(tmp_path))
    (tmp_path / "m.json").write_text("[]")
    custom_out = tmp_path / "custom_results"
    assert vo.main(["--manifest", str(tmp_path / "m.json"), "--out-dir", str(custom_out)]) == 0
    assert (custom_out / "offair_report.json").exists()


def test_readme_loads_offair_report_from_offair_dir(tmp_path, monkeypatch):
    """render_readme._load reads results/offair/offair_report.json first,
    falling back to results/offair_report.json."""
    import render_readme as rr
    monkeypatch.setattr(rr, "ROOT", str(tmp_path))
    res = tmp_path / "results"
    res_offair = res / "offair"
    res_offair.mkdir(parents=True)

    # 1. Fallback when only results/offair_report.json exists
    (res / "offair_report.json").write_text(json.dumps({"source": "fallback"}))
    assert rr._load("offair_report") == {"source": "fallback"}

    # 2. Priority: results/offair/offair_report.json takes precedence
    (res_offair / "offair_report.json").write_text(json.dumps({"source": "offair_dir"}))
    assert rr._load("offair_report") == {"source": "offair_dir"}
