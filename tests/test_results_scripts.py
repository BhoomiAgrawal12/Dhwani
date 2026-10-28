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


def test_offair_scoring_rules_from_first_real_run():
    """First off-air run: mono WAVs were scored at the analysis rate fs/2
    (24 kHz) against the file's 48 kHz; rates the manifest itself supplied
    were counted as identified (6/10); 'nothing to detect' had no key."""
    import validate_offair as vo
    from dhwani.common.models import AnalysisResult, SignalParameters

    class Stub:
        def __init__(self, **meta):
            self.meta, self.segments = meta, meta.pop("segments", [])

        def analyze(self, *a, **k):
            r = AnalysisResult(recording_meta=dict(self.meta))
            r.parameters, r.segments = SignalParameters(), self.segments
            return r

    e = {"file": "x.wav", "license": "test",
         "expect": {"sample_rate_hz": 48000}}
    mono = Stub(sample_rate=24000.0, sample_rate_source="wav_header",
                file_sample_rate=48000.0)
    assert vo.score(e, mono, True)["checks"] == {"sample_rate_hz": True}
    given = vo.score(e, Stub(sample_rate=48000.0, sample_rate_source="user"),
                     True)
    assert given["checks"] == {} and given["given"] == ["sample_rate_hz"]
    assert not given["all_correct"]          # nothing was actually scored
    quiet = {"file": "n.iq", "license": "test", "expect": {"signal": False}}
    assert vo.score(quiet, Stub(sample_rate_source="user"), True)["all_correct"]
    assert not vo.score(quiet, Stub(sample_rate_source="user",
                                    segments=["seg"]), True)["all_correct"]


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


def test_matrix_case_records_s5_confidence(config):
    """The precision-recall chart needs S5's confidence on every case."""
    import validate_matrix as vm
    from dhwani.pipeline import Analyzer
    row = vm.run_case("BPSK", 25, 7, config, Analyzer(config, use_cache=False),
                      False)
    assert 0.0 <= row["confidence"] <= 1.0


def test_benchmark_ber_frame_sync_by_known_word():
    """Proof Chart data: the known sync word is found in whichever S7
    stream the carrier lock left it (here a conjugated quarter turn), and
    refused once more than a quarter of its bits are wrong."""
    import numpy as np
    import benchmark_ber as bb
    from dhwani.bits.ambiguity import enumerate_ambiguities
    from dhwani.demod.constellations import bits_to_iq_symbols
    rng = np.random.default_rng(5)
    pre = rng.integers(0, 2, 200).astype(np.uint8)
    pay = rng.integers(0, 2, 400).astype(np.uint8)
    sync = bb.SYNC_BITS
    syms = np.conj(bits_to_iq_symbols(np.concatenate([pre, sync, pay]),
                                      "QPSK")) * 1j
    streams = enumerate_ambiguities(syms, "QPSK", include_differential=False)
    hit = bb.find_sync(streams, sync, len(pay))
    assert hit is not None
    i, start, errors = hit
    assert errors == 0
    assert np.array_equal(streams[i].bits[start:start + len(pay)], pay)
    bad = sync.copy()
    bad[:len(sync) // 4 + 1] ^= 1
    syms = bits_to_iq_symbols(np.concatenate([pre, bad, pay]), "QPSK")
    assert bb.find_sync(enumerate_ambiguities(
        syms, "QPSK", include_differential=False), sync, len(pay)) is None


def test_benchmark_ber_quick_writes_report(tmp_path, monkeypatch):
    """Proof Chart data: benchmark_ber --quick writes ber_report.json with
    provenance, every chain at every point with its lock bookkeeping, and
    the uncoded theory curve; uncoded QPSK at 10 dB locks every frame."""
    import math
    import benchmark_ber as bb
    monkeypatch.setattr(provenance, "RESULTS", str(tmp_path))
    assert bb.main(["--quick"]) == 0
    d = json.load(open(tmp_path / "ber_report.json"))
    assert d["provenance"]["data"].startswith("synthetic")
    s = d["settings"]
    assert s["quick"] and s["frames"] == 2 and len(s["ebn0_db"]) == 2
    assert "seed" in s
    assert set(d["chains"]) == {"uncoded", "conv_k7", "ldpc_648"}
    keys = {"ebn0_db", "esn0_db", "frames", "locked", "lock_failures",
            "lock_failure_reasons", "p_lock", "p_lock_ci95", "bits",
            "bit_errors", "ber", "ber_upper_95"}
    for c in d["chains"].values():
        assert c["label"] and 0 < c["code_rate"] <= 1
        assert c["seconds_per_frame"] > 0
        assert [p["ebn0_db"] for p in c["points"]] == s["ebn0_db"]
        for p in c["points"]:
            assert keys <= set(p)
            assert p["locked"] + p["lock_failures"] == p["frames"] == 2
            assert sum(p["lock_failure_reasons"].values()) == p["lock_failures"]
            if p["bits"] and not p["bit_errors"]:
                assert p["ber"] == 0.0
                assert math.isclose(p["ber_upper_95"], 3 / p["bits"],
                                    rel_tol=1e-4)
    for p in d["theory"]["points"]:
        assert math.isclose(p["ber"], 0.5 * math.erfc(
            math.sqrt(10 ** (p["ebn0_db"] / 10))), rel_tol=1e-4)
    top = d["chains"]["uncoded"]["points"][-1]
    assert top["ebn0_db"] == 10 and top["locked"] == 2 and top["ber"] < 1e-2


def _ber_report_fixture():
    def pt(x, frames, locked, bits, errors):
        return {"ebn0_db": x, "esn0_db": x, "frames": frames,
                "locked": locked, "lock_failures": frames - locked,
                "lock_failure_reasons": {"no_detection": frames - locked,
                                         "demod_failed": 0, "no_sync": 0},
                "p_lock": locked / frames, "p_lock_ci95": [0.0, 1.0],
                "bits": bits, "bit_errors": errors,
                "ber": errors / bits if bits else None,
                "ber_upper_95": 3 / bits if bits and not errors else None}
    return {
        "provenance": {"data": "synthetic (benchmark_ber)", "git_commit": "abc",
                       "cpu": "cpu", "python": "3", "generated_utc": "t"},
        "settings": {"ebn0_db": [0, 5, 10], "frames": 4,
                     "info_bits_per_frame": 2592},
        "chains": {
            "uncoded": {"label": "uncoded QPSK", "code_rate": 1.0, "points": [
                pt(0, 4, 0, 0, 0), pt(5, 4, 2, 5184, 30),
                pt(10, 4, 4, 10368, 0)]},
            "ldpc_648": {"label": "LDPC 802.11n n=648 r1/2", "code_rate": 0.5,
                         "points": [pt(0, 4, 0, 0, 0), pt(5, 4, 4, 10368, 0),
                                    pt(10, 4, 4, 10368, 0)]}},
        "theory": {"chain": "uncoded", "points": [
            {"ebn0_db": 0, "ber": 0.0786}, {"ebn0_db": 5, "ber": 0.00596},
            {"ebn0_db": 10, "ber": 3.87e-06}]}}


def test_proof_chart_from_result_files(tmp_path, monkeypatch):
    """Proof Chart: BER and lock panels drawn from ber_report.json with the
    off-air recordings placed at their EVM SNR; zero-error points are upper
    bounds and say so; recordings without an EVM are left out."""
    import xml.dom.minidom
    import plots
    res = tmp_path / "results"
    (res / "offair").mkdir(parents=True)
    (res / "ber_report.json").write_text(json.dumps(_ber_report_fixture()))
    q = lambda evm, snr, status, mod: {
        "evm_percent": evm, "snr_db_evm": snr, "demodulation_status": status,
        "modulation_demodulated": mod}
    (res / "offair" / "offair_report.json").write_text(json.dumps({
        "provenance": {"git_commit": "abc"}, "cases": [
            {"file": "results/offair/recordings/entrysat.wav",
             "all_correct": True, "decoded": {"crc": "CRC-16-X25", "frames": 1},
             "signal_quality": q(10.0, 20.0, "GOOD", "BPSK")},
            {"file": "results/offair/recordings/lilacsat1.wav",
             "all_correct": False, "decoded": {"crc": None, "frames": None},
             "signal_quality": q(50.0, 6.0, "FAILED", "BPSK")},
            {"file": "results/offair/recordings/ty_2.wav", "all_correct": True,
             "decoded": {"crc": None, "frames": None},
             "signal_quality": q(None, None, "GOOD", "2FSK")},
            {"file": "older_report_entry.wav", "all_correct": True,
             "decoded": {}}]}))
    monkeypatch.setattr(plots, "RESULTS", str(res))
    monkeypatch.setattr(plots, "OUT", str(res / "charts"))
    plots.main()
    svg = (res / "charts" / "proof.svg").read_text()
    xml.dom.minidom.parseString(svg)
    for text in ("uncoded QPSK", "LDPC 802.11n n=648 r1/2", "theory",
                 "0 errors", "entrysat", "20.0 dB", "lilacsat1"):
        assert text in svg, text
    assert "ty_2" not in svg and "older_report_entry" not in svg
    # without an off-air report the chart still draws
    xml.dom.minidom.parseString(plots.proof_chart(_ber_report_fixture(), None))


def test_readme_renders_decoding_proof_section(tmp_path, monkeypatch):
    """Proof Chart in the README: its own section after the bit-layer one,
    numbers only from ber_report.json, 'Not generated yet' without it."""
    import render_readme as rr
    (tmp_path / "results" / "charts").mkdir(parents=True)
    monkeypatch.setattr(rr, "ROOT", str(tmp_path))
    text = rr.render()
    assert (text.index("### Interleaver and FEC") <
            text.index("### Decoding proof") < text.index("### Headerless"))
    section = text.split("### Decoding proof")[1].split("\n### ")[0]
    assert "Not generated yet: run `python scripts/benchmark_ber.py`" in section
    (tmp_path / "results" / "ber_report.json").write_text(
        json.dumps(_ber_report_fixture()))
    (tmp_path / "results" / "charts" / "proof.svg").write_text("<svg/>")
    section = rr.render().split("### Decoding proof")[1].split("\n### ")[0]
    assert "Not generated yet" not in section
    assert "(results/charts/proof.svg)" in section
    assert "| uncoded QPSK | 1 | 10 dB | 10 dB |" in section
    assert "| LDPC 802.11n n=648 r1/2 | 1/2 | 5 dB | 5 dB |" in section
    assert "`results/ber_report.json`" in section


def test_offair_records_signal_quality(tmp_path, monkeypatch):
    """Proof Chart off-air points: each recording also records its signal
    quality (EVM, the SNR it implies, S6 status, the modulation S6 ran) as
    new keys beside the existing ones."""
    import math
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
        "expect": {"modulation": "QPSK", "symbol_rate_hz": 125000}}]))
    assert vo.main(["--manifest", str(tmp_path / "m.json")]) == 0
    case = json.load(open(tmp_path / "offair_report.json"))["cases"][0]
    for k in ("observed", "checks", "decoded", "all_correct"):
        assert k in case
    assert case["checks"] == {"modulation": True, "symbol_rate_hz": True}
    q = case["signal_quality"]
    assert q["modulation_demodulated"] == "QPSK"
    assert q["demodulation_status"] in ("GOOD", "DEGRADED")
    assert 0 < q["evm_percent"] < 20
    assert math.isclose(q["snr_db_evm"],
                        -20 * math.log10(q["evm_percent"] / 100), abs_tol=0.05)


def test_offair_signal_quality_without_evm():
    """No demodulation (nothing detected) or no EVM (FSK, analog): the
    signal-quality keys are there with None, nothing is invented."""
    import validate_offair as vo
    from dhwani.common.models import AnalysisResult

    class Stub:
        def __init__(self, demod):
            self.demod = demod

        def analyze(self, *a, **k):
            r = AnalysisResult(recording_meta={
                "sample_rate": 48000.0, "sample_rate_source": "wav_header"})
            r.demodulation = self.demod
            return r

    e = {"file": "x.wav", "license": "test", "expect": {"modulation": "2FSK"}}
    assert vo.score(e, Stub(None), True)["signal_quality"] == {
        "evm_percent": None, "snr_db_evm": None,
        "demodulation_status": None, "modulation_demodulated": None}
    q = vo.score(e, Stub({"modulation": "2FSK", "demodulation_status": "GOOD",
                          "evm_percent": None}), True)["signal_quality"]
    assert q["snr_db_evm"] is None and q["demodulation_status"] == "GOOD"
    q = vo.score(e, Stub({"modulation": "BPSK", "demodulation_status": "GOOD",
                          "evm_percent": 10.0}), True)["signal_quality"]
    assert q["snr_db_evm"] == 20.0 and q["modulation_demodulated"] == "BPSK"
