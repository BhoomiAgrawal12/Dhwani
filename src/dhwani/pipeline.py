"""Analyzer: the single analysis engine behind the CLI and the GUI.

Runs S0..S11 with stage caching, analyst overrides and honest uncertainty
propagation. Overrides re-run only downstream stages (the cache keys of
untouched stages are unchanged, so their results are reused).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .common.cache import StageCache, content_hash
from .common.config import Config, load_config
from .common.logging import StageTimer, get_logger, new_run_id
from .common.models import (AnalysisResult, ModulationHypothesis, Payload,
                            FrameHypothesis, ScramblerHypothesis)
from .conditioning import condition
from .detection import detect_signals, compute_psd
from .channelization import channelize
from .params import estimate_parameters
from .params.estimators import sample_rate_candidates
from .modulation import classify_modulation
from .demod import demodulate
from .bits.ambiguity import enumerate_ambiguities
from .common.models import BitStream
from .framing.frames import binary_autocorrelation, payload_stats
from .hypothesis import bitlayer_search
from .ingestion import load_recording

STAGE_VERSION = 10   # bump to invalidate caches when algorithms change

# stage catalogue for the pipeline trace (drives the flow visualisation)
STAGES = [
    ("S0", "Ingestion & format detection"),
    ("S1", "Conditioning"),
    ("S2", "Signal detection (CFAR)"),
    ("S3", "Channelisation"),
    ("S4", "Parameter estimation"),
    ("S5", "Modulation classification"),
    ("S6", "Demodulation"),
    ("S7", "Ambiguity fan-out & descrambling"),
    ("S8", "Blind de-interleaving"),
    ("S9", "Blind FEC identification"),
    ("S10", "Framing & CRC"),
    ("S11", "Payload intelligence"),
    ("S12", "Reporting & export"),
]


class _Trace:
    """Collects one entry per pipeline stage for the flow visualisation."""

    def __init__(self, timings: dict):
        self.entries = {}
        self.timings = timings

    def mark(self, stage: str, status: str, summary: str):
        self.entries[stage] = {"status": status, "summary": summary}

    def finalize(self, stop_reason: str = None) -> list:
        out = []
        for sid, title in STAGES:
            e = self.entries.get(sid)
            if e is None:
                e = {"status": "skipped",
                     "summary": stop_reason or "not reached"}
            timing_key = {
                "S0": "S0_ingest", "S1": "S1_condition", "S2": "S2_detect",
                "S3": "S3_channelize", "S4": "S4_parameters",
                "S5": "S5_modulation", "S6": "S6_demodulate",
                "S7": "S7_ambiguity", "S8": "S8_S10_bitlayer",
                "S9": "S8_S10_bitlayer", "S10": "S8_S10_bitlayer",
                "S11": "S11_payload_intel",
            }.get(sid)
            out.append({"stage": sid, "title": title, **e,
                        "elapsed_s": round(self.timings.get(timing_key, 0), 3)
                        if timing_key and timing_key in self.timings else None})
        return out


class Analyzer:
    def __init__(self, config: Optional[Config] = None,
                 use_cache: bool = True):
        self.config = config or load_config()
        self.log = get_logger("dhwani", self.config.log_level)
        self.cache = StageCache(self.config.cache_dir, enabled=use_cache)

    # ------------------------------------------------------------------
    def analyze(self, path: str, sample_rate: float = None,
                center_frequency: float = None, datatype: str = None,
                signal_index: int = 0, overrides: dict = None,
                no_ml: bool = False,
                progress: Callable[[str, float], None] = None,
                stop_after: str = None,
                start_sample: int = None) -> AnalysisResult:
        """Full-chain analysis of one file.

        overrides: analyst-pinned values, e.g. {"modulation": "BPSK",
        "symbol_rate_norm": 0.125, "signal_index": 1}. Every stage re-runs;
        only S2 detection is served from the cache. stop_after: "S2" or
        "S5" ends the run there (CLI detect / classify share this engine).
        start_sample pins the analysis window (default: the strongest of
        the scanned windows)."""
        overrides = overrides or {}
        result = AnalysisResult(run_id=new_run_id())
        timer = StageTimer(self.log, result.stage_timings)
        trace = _Trace(result.stage_timings)
        prog = progress or (lambda stage, frac: None)

        # ---------------- S0 ingest -----------------------------------
        with timer.stage("S0_ingest", path=path):
            rec = load_recording(path, sample_rate, center_frequency, datatype)
            result.recording_meta = rec.meta_dict()
            result.warnings.extend(rec.warnings)
            trace.mark("S0", "executed",
                       f"{rec.format} / {rec.datatype or 'n/a'}, "
                       f"{rec.n_samples:,} samples, sample rate "
                       f"{'{:,.0f} Hz'.format(rec.sample_rate) if rec.sample_rate else 'unknown'}")
        prog("ingest", 1.0)
        data_hash = content_hash(rec.samples)

        # ---------------- S1 + S2 per analysis window -------------------
        # S1/S2 work on WINDOW samples at a time. A longer file is scanned
        # window by window (up to cfar.scan_windows) and the window holding
        # the strongest detection is analysed; start_sample pins it.
        WINDOW = 1 << 22
        n_total = rec.n_samples
        if start_sample is not None:
            starts = [int(start_sample)]
        else:
            starts = list(range(0, max(1, n_total), WINDOW))
            if len(starts) > 1 and n_total - starts[-1] < WINDOW:
                starts[-1] = n_total - WINDOW      # full-length tail window
            starts = starts[: self.config.cfar.scan_windows]
        chosen = None
        for st in starts:
            with timer.stage("S1_condition"):
                x_w, cond_w = condition(rec.samples[st:st + WINDOW])
            with timer.stage("S2_detect"):
                key = self.cache.key(data_hash, "detect", STAGE_VERSION,
                                     {"cfar": vars(self.config.cfar),
                                      "start": st})
                cached = self.cache.get(key)
                if cached is None:
                    segs_w, dbg = detect_signals(
                        x_w, self.config.cfar, sample_rate=rec.sample_rate)
                    wf = dbg["waterfall"]
                    plots_w = {
                        "psd": _psd_plot(x_w),
                        "waterfall": {
                            "db": _quantize(wf["waterfall_db"]),
                            "freq_norm": wf["freq_norm"].tolist(),
                        },
                    }
                    self.cache.put(key, (segs_w, plots_w))
                else:
                    segs_w, plots_w = cached
            strength = max((sg.snr_db for sg in segs_w), default=-np.inf)
            if chosen is None or strength > chosen[0]:
                chosen = (strength, st, x_w, cond_w, segs_w, plots_w)
        _s, start, x, cond, segments, plots = chosen
        result.conditioning = cond
        result.recording_meta["analysed_start"] = start
        result.recording_meta["analysed_samples"] = cond.analysed_samples
        result.warnings.extend(cond.warnings)
        if start > 0 or start + len(x) < n_total:
            scanned = min(n_total, starts[-1] + WINDOW) - starts[0]
            result.warnings.append(
                f"analysed samples {start:,}-{start + len(x):,} of {n_total:,}"
                + (" (pinned by start_sample)" if start_sample is not None
                   else f": the {WINDOW:,}-sample window with the strongest "
                        f"detection among {len(starts)} scanned")
                + (f"; samples after {starts[0] + scanned:,} were not "
                   "scanned" if starts[0] + scanned < n_total
                   and start_sample is None else ""))
        trace.mark("S1", "executed",
                   f"DC {'removed' if cond.dc_removed else 'negligible'}, "
                   f"clipping {cond.clipping_fraction*100:.2f}%, "
                   f"IQ imbalance {cond.iq_gain_imbalance_db:.2f} dB / "
                   f"{cond.iq_phase_error_deg:.1f} deg (measured, "
                   f"{'corrected' if cond.iq_corrected else 'not corrected'})")
        prog("condition", 1.0)

        for s in segments:
            s.sample_rate = rec.sample_rate
        result.segments = segments
        result.plots.update(plots)
        trace.mark("S2", "executed",
                   f"{len(segments)} signal(s) above threshold" +
                   (f"; best SNR {segments[0].snr_db:.1f} dB"
                    if segments else ""))
        if not segments:
            result.warnings.append("no signals detected above the CFAR "
                                   "threshold; analysis stopped at S2")
            result.pipeline_trace = trace.finalize(
                "stopped at S2: no signals detected")
            return result
        prog("detect", 1.0)
        if stop_after == "S2":
            result.pipeline_trace = trace.finalize("stopped after S2 (detect)")
            return result

        # ---------------- segment selection ----------------------------
        sel = overrides.get("signal_index", signal_index)
        sel = min(sel, len(segments) - 1)
        seg = segments[sel]
        result.selected_segment = seg.to_dict()

        # ---------------- S3 channelize --------------------------------
        with timer.stage("S3_channelize"):
            # FM-receiver audio is louder in hiss than with a carrier (FM
            # quieting): its envelope cannot locate the burst
            ch = channelize(x, seg, trim=not rec.baseband_audio)
            base = ch["samples"]
            trace.mark("S3", "executed",
                       f"signal {sel}: shifted {seg.center_norm:+.4f}, "
                       f"{len(base):,} samples after trim")
        prog("channelize", 1.0)

        # ---------------- S4 parameters --------------------------------
        with timer.stage("S4_parameters"):
            params = estimate_parameters(base, self.config.params,
                                         sample_rate=ch["sample_rate"])
            params.rate_ratio = ch["rate_ratio"]
            if "symbol_rate_norm" in overrides:
                # the analyst gives symbols per RECORDED sample
                params.symbol_rate_norm = (overrides["symbol_rate_norm"]
                                           / ch["rate_ratio"])
                params.samples_per_symbol = 1.0 / params.symbol_rate_norm
                if ch["sample_rate"]:
                    params.symbol_rate_hz = (params.symbol_rate_norm
                                             * ch["sample_rate"])
                params.confidences["symbol_rate"] = {
                    "value": 1.0, "method": "analyst override",
                    "verdict": "detected"}
            if params.symbol_rate_norm:
                params.symbol_rate_norm_recording = (params.symbol_rate_norm
                                                     * ch["rate_ratio"])
            if rec.sample_rate is None and params.symbol_rate_norm_recording:
                params.sample_rate_candidates = sample_rate_candidates(
                    params.symbol_rate_norm_recording)
            result.parameters = params
            trace.mark("S4", "executed",
                       f"Rs {params.symbol_rate_norm_recording or 'unknown'}"
                       f" (symbols/recorded sample), SNR {params.snr_db} dB, "
                       f"OBW {params.obw99_norm}" +
                       ("; sample-rate candidates (PROBABLE): " +
                        ", ".join(f"{c['sample_rate_hz']:,.0f} Hz"
                                  for c in params.sample_rate_candidates[:4])
                        if params.sample_rate_candidates else ""))
        prog("parameters", 1.0)

        # ---------------- S5 modulation --------------------------------
        with timer.stage("S5_modulation"):
            if "modulation" in overrides:
                modulation = overrides["modulation"]
                result.modulation = ModulationHypothesis(
                    prediction=modulation, confidence=1.0,
                    alternatives=[[modulation, 1.0]],
                    engine_predictions={"analyst": [modulation, 1.0]},
                    constraints_applied=["analyst override"])
            else:
                mcfg = self.config.modulation
                if no_ml:
                    import copy
                    mcfg = copy.copy(mcfg)
                    mcfg.cvnet_enabled = False
                result.modulation = classify_modulation(base, params, mcfg)
                modulation = result.modulation.prediction
        prog("modulation", 1.0)

        trace.mark("S5", "executed",
                   f"{modulation} (confidence "
                   f"{result.modulation.confidence})")
        if stop_after == "S5":
            result.pipeline_trace = trace.finalize("stopped after S5 (classify)")
            return result
        if modulation in ("UNKNOWN", "OFDM"):
            result.warnings.append(
                f"modulation '{modulation}' is outside the demodulation set; "
                "stopping after parameter estimation (analyst may override)")
            result.pipeline_trace = trace.finalize(
                f"stopped after S5: '{modulation}' not demodulated")
            return result

        # ---------------- S6 demodulate --------------------------------
        with timer.stage("S6_demodulate", modulation=modulation):
            sps = params.samples_per_symbol or 8.0
            demod = demodulate(base, modulation, sps, self.config.demod,
                               cfo_norm=(params.carrier_offset_norm
                                         if not modulation.endswith("FSK") else None))
            result.demodulation = demod.to_dict()
            result.warnings.extend(demod.warnings)
            if demod.symbols is not None and len(demod.symbols):
                result.plots["constellation"] = _const_plot(demod.symbols)
            # EVM-based SNR refinement: M2M4 assumes a symbol-spaced
            # constant-modulus signal and reads low on oversampled or QAM
            # inputs; once the demodulator locks, EVM gives the better
            # estimate and the method is recorded honestly.
            if (demod.carrier_locked and demod.timing_locked and
                    demod.evm_percent and demod.evm_percent > 0.1):
                import math
                params.snr_db = round(-20 * math.log10(
                    demod.evm_percent / 100.0), 1)
                params.confidences["snr"] = {
                    "value": 0.9, "method": "EVM after synchronisation",
                    "verdict": "estimated"}
            if getattr(demod, "eye_trace", None) is not None and \
                    len(demod.eye_trace):
                q = int(demod.lock_metrics.get("eye_sps", 8))
                tr = demod.eye_trace
                n_tr = min(100, len(tr) // (2 * q))
                result.plots["eye"] = {
                    "sps": q,
                    "traces": [[round(float(v), 4) for v in
                                tr[i * 2 * q:(i + 1) * 2 * q].real]
                               for i in range(n_tr)]}
            trace.mark("S6", "executed",
                       f"status {demod.demodulation_status}, "
                       f"{0 if demod.symbols is None else len(demod.symbols):,}"
                       f" symbols, EVM {demod.evm_percent}%, locks "
                       f"T={demod.timing_locked} C={demod.carrier_locked}")
            if getattr(demod, "audio", None) is not None:
                # analog transmission: audio out, the bit layer does not apply
                a = demod.audio
                step = max(1, len(a) // 4000)
                result.plots["audio"] = {
                    "samples": [round(float(v), 4) for v in a[::step][:4000]],
                    "decimation": step}
                result.pipeline_trace = trace.finalize(
                    "analog transmission: no bit layer")
                return result
            if demod.demodulation_status == "FAILED":
                result.warnings.append(
                    "S6 quality gate: demodulation FAILED "
                    f"(EVM {demod.evm_percent}%, locks T={demod.timing_locked} "
                    f"C={demod.carrier_locked}); the bit layer is withheld "
                    "rather than fed unreliable bits")
                result.pipeline_trace = trace.finalize(
                    "stopped by the S6 quality gate: unreliable demodulation")
                return result
            if demod.hard_bits is None or len(demod.hard_bits) < 256:
                result.warnings.append("demodulation produced too few bits "
                                       "for bit-layer analysis")
                result.pipeline_trace = trace.finalize(
                    "stopped after S6: too few bits recovered")
                return result
        prog("demodulate", 1.0)

        # ---------------- S7 ambiguity fan-out --------------------------
        with timer.stage("S7_ambiguity"):
            if modulation.endswith("FSK"):
                streams = _fsk_streams(demod)
            else:
                streams = enumerate_ambiguities(
                    demod.symbols, modulation,
                    self.config.bitlayer.max_ambiguity_streams)
                for s in streams:
                    if s.llrs is None:
                        s.llrs = demod.llrs
            trace.mark("S7", "executed",
                       f"{len(streams)} candidate bit streams")
        prog("ambiguity", 1.0)

        # ---------------- S8-S10 bit-layer search -----------------------
        frames_matrix = None
        with timer.stage("S8_S10_bitlayer", n_streams=len(streams)):
            search = bitlayer_search(streams, self.config.bitlayer,
                                     self.config.fec, self.config.framing,
                                     logger=self.log,
                                     progress=lambda f: prog("bitlayer", f))
            result.hypotheses = search["hypotheses"]
            if search.get("budget_exhausted"):
                result.warnings.append(
                    f"bit-layer search stopped at its "
                    f"{self.config.bitlayer.time_budget_s:.0f} s budget before "
                    "every candidate stream was examined; the verdicts below "
                    "come from the streams explored")
            best = search["best"]
            if best:
                _fill_result_from_best(result, best)
                fr = best.get("frames")
                if fr:
                    frames_matrix = fr["analysis"]["frames"]
            il, fec, frh = result.interleaver, result.fec, result.frames
            scr = result.scrambler
            s7_extra = (f"; descrambler {scr.name or scr.kind}"
                        if scr else "")
            trace.mark("S7", "executed",
                       f"{len(streams)} candidate bit streams{s7_extra}")
            trace.mark("S8", "executed",
                       f"interleaver: {il.label} {il.parameters if il.kind != 'none' else ''}"
                       if il else "no interleaver hypothesis")
            trace.mark("S9", "executed",
                       (f"FEC: {fec.family}"
                        + (f", syndrome-zero rate {fec.syndrome_zero_rate}"
                           if fec.syndrome_zero_rate is not None else ""))
                       if fec else "no FEC hypothesis")
            trace.mark("S10", "executed",
                       (f"frames {frh.frame_length_bits} bits, sync "
                        f"{frh.sync_word_hex}, CRC "
                        f"{frh.crc['name'] if frh.crc else 'none found'}")
                       if frh else "no frame structure found")
        prog("bitlayer", 1.0)

        # ---------------- S11 payload intelligence ----------------------
        if self.config.payload_intelligence.enabled and result.payload and \
                result.payload.data:
            with timer.stage("S11_payload_intel",
                             n_bytes=len(result.payload.data)):
                from .payload import analyze_payload
                crc = result.frames.crc if result.frames else None
                provenance = {
                    "crc_validated": bool(crc and
                                          crc.get("pass_fraction", 0) > 0.9),
                    "fec_syndrome_rate": (result.fec.syndrome_zero_rate
                                          if result.fec else None),
                    "demod_locked": bool(demod.carrier_locked and
                                         demod.timing_locked),
                    "demod_status": demod.demodulation_status,
                    "frame_validated": bool(result.frames is not None),
                }
                result.payload_intelligence = analyze_payload(
                    result.payload.data, frames=frames_matrix,
                    provenance=provenance,
                    config=self.config.payload_intelligence)
                pi = result.payload_intelligence
                trace.mark("S11", "executed",
                           f"{pi['summary']['classification']} "
                           f"({pi['summary']['strength']}, "
                           f"{pi['summary']['confidence']})"
                           if pi.get("available") else "no payload to analyse")
        else:
            trace.mark("S11", "skipped",
                       "disabled in config" if not
                       self.config.payload_intelligence.enabled
                       else "no payload bytes recovered")
        prog("payload_intel", 1.0)

        trace.mark("S12", "ready",
                   "exports: JSON, CSV, payload, SigMF, PDF, GRC "
                   "(rendered by CLI/GUI/web)")
        result.pipeline_trace = trace.finalize()
        return result


# ----------------------------------------------------------------------
def _fsk_streams(demod) -> list:
    bits = demod.hard_bits
    inv = (1 - bits).astype(np.uint8)
    def diff(b):
        d = np.bitwise_xor(b[1:], b[:-1])
        return np.concatenate([[b[0]], d]).astype(np.uint8)
    return [
        BitStream(bits=bits, llrs=demod.llrs, hypothesis={"mapping": "normal"}),
        BitStream(bits=inv, hypothesis={"mapping": "inverted"}),
        BitStream(bits=diff(bits), hypothesis={"mapping": "normal", "differential": True}),
        BitStream(bits=diff(inv), hypothesis={"mapping": "inverted", "differential": True}),
    ]


def _fill_result_from_best(result: AnalysisResult, best: dict):
    cand = best["candidate"]
    wh = cand["whitener"]
    bm = best.get("scrambler_bm") or {}
    if wh["name"] != "none":
        from .scrambling.lfsr import KNOWN_WHITENERS
        winfo = KNOWN_WHITENERS.get(wh["name"], {})
        result.scrambler = ScramblerHypothesis(
            kind="known_whitening", name=wh["name"],
            polynomial=winfo.get("poly"), degree=winfo.get("degree"),
            seed=wh.get("phase"), score=1.0)
    elif bm.get("found"):
        result.scrambler = ScramblerHypothesis(
            kind="additive_lfsr", polynomial=bm["poly"], degree=bm["degree"],
            score=bm["match"])
    else:
        result.scrambler = ScramblerHypothesis(kind="none", score=0.5)
        if not best.get("crc_hits"):
            from .scrambling.berlekamp import MAX_WHITENER_PHASES
            from .scrambling.lfsr import KNOWN_WHITENERS
            long = [n for n, w in KNOWN_WHITENERS.items()
                    if (1 << w["degree"]) - 1 > MAX_WHITENER_PHASES]
            if long:
                result.warnings.append(
                    f"no scrambler found, but {', '.join(long)} was only "
                    f"phase-searched over its first {MAX_WHITENER_PHASES} "
                    "phases; a stream scrambled at another phase would "
                    "look unscrambled")

    result.interleaver = best["interleaver"]
    result.fec = best["fec"]
    if result.fec is not None:
        result.fec.decoded_bits = None   # keep result JSON-sized
    fr = best.get("frames")
    if fr and not fr.get("valid", True):
        result.warnings.append(
            "frame structure candidate rejected: insufficient evidence (" +
            ", ".join(f"{k}={v}" for k, v in
                      fr.get("validity_evidence", {}).items()) +
            "); periodicity alone does not make a frame")
        fr = None
    if fr:
        analysis = fr["analysis"]
        sync = analysis["sync"]
        crc = best["crc_hits"][0] if best["crc_hits"] else None
        fh = FrameHypothesis(
            frame_length_bits=fr["frame_length"],
            sync_word_hex=sync.get("hex") if sync.get("found") else None,
            sync_offset=sync.get("offset_bits"),
            column_entropy=analysis["entropy"],
            field_map=analysis["fields"],
            crc={k: (hex(v) if isinstance(v, int) and k in
                     ("poly", "init", "xorout") else v)
                 for k, v in crc.items()} if crc else None,
            score=best["hypothesis"].score)
        result.frames = fh
        # payload: bytes of the non-sync, non-crc region of each frame
        frames_mat = analysis["frames"]
        payload = _extract_payload(frames_mat, analysis, crc)
        stats = payload_stats(payload)
        result.payload = Payload(data=payload, **stats)
        if stats["likely_encrypted"]:
            result.warnings.append(
                "payload entropy is near 1 bit/bit: content is likely "
                "encrypted or compressed; no further recovery attempted")
        result.plots["entropy_map"] = [round(float(e), 3)
                                       for e in analysis["entropy"]]
    result.plots["rank_profile"] = (result.interleaver.rank_profile
                                    if result.interleaver else None)
    # bit-stream correlation view (PS v): peaks at the frame period
    dec = best.get("decoded_bits")
    if dec is not None and len(dec) > 256:
        fl = (best.get("frames") or {}).get("frame_length") or 0
        ac = binary_autocorrelation(dec[:100000],
                                    min(2048, 4 * fl) if fl else 1024)
        result.plots["bit_autocorr"] = [round(float(v), 4) for v in ac]


def _extract_payload(frames_mat: np.ndarray, analysis: dict, crc) -> bytes:
    """Payload = the CRC-protected data section minus sync and header.

    Per-byte-column behaviour across frames separates the fields: the
    sync run and counter-like bytes are header; a constant byte inside
    the data section still belongs to the payload when it matches the
    payload's content class (text payloads legitimately contain constant
    label characters, which a pure entropy rule would misfile as sync
    and silently drop from every frame)."""
    sync = analysis["sync"]
    n_bits = frames_mat.shape[1]
    start_bit = 0
    if sync.get("found"):
        start_bit = sync["offset_bits"] + sync["length_bits"]
        # keep the payload byte-aligned: a constant run can end mid-byte
        # (constant MSBs of a following counter field) and a bit-shifted
        # slice would garble every downstream byte analysis
        start_bit -= start_bit % 8
    end_bit = n_bits
    if crc:
        end_bit = min(end_bit, crc["crc_byte_offset"] * 8)
    if end_bit <= start_bit:
        start_bit, end_bit = 0, n_bits
    n_bytes = (end_bit - start_bit) // 8
    if n_bytes <= 0:
        out = bytearray()
        for row in frames_mat:
            seg = row[start_bit:end_bit]
            n8 = (len(seg) // 8) * 8
            if n8:
                out.extend(np.packbits(seg[:n8]).tobytes())
        return bytes(out)
    vals = np.packbits(
        frames_mat[:, start_bit:start_bit + n_bytes * 8], axis=1)

    def _is_counter(v):
        if len(v) < 4:
            return False
        d = np.diff(v.astype(np.int16)) % 256
        top, cnt = np.unique(d, return_counts=True)
        best = int(top[np.argmax(cnt)])
        return best != 0 and cnt.max() >= 0.8 * len(d)

    def _printable_frac(v):
        return float(np.mean(((v >= 0x20) & (v < 0x7f)) |
                             (v == 9) | (v == 10) | (v == 13)))

    # core = varying, non-counter byte columns. Structured text varies
    # only in the low bits of each character, so a mean-bit-entropy
    # threshold would miss it entirely.
    core = [i for i in range(n_bytes)
            if not np.all(vals[:, i] == vals[0, i])
            and not _is_counter(vals[:, i])]
    core_text = bool(core) and float(
        np.mean([_printable_frac(vals[:, i]) for i in core])) > 0.9
    first = 0
    while first < n_bytes:
        v = vals[:, first]
        if _is_counter(v):
            first += 1
            continue
        if np.all(v == v[0]):
            if core_text and _printable_frac(v) == 1.0:
                break        # constant text character: payload label
            first += 1
            continue
        break                # varying non-counter byte: payload begins
    if first >= n_bytes:
        first = 0
    return vals[:, first:].tobytes()


def _psd_plot(x: np.ndarray) -> dict:
    p = compute_psd(x)
    step = max(1, len(p["freq_norm"]) // 2048)
    return {"freq_norm": p["freq_norm"][::step].tolist(),
            "psd_db": [round(float(v), 2) for v in p["psd_db"][::step]]}


def _const_plot(symbols: np.ndarray, max_points: int = 3000) -> dict:
    s = symbols[:max_points]
    return {"i": [round(float(v), 4) for v in s.real],
            "q": [round(float(v), 4) for v in s.imag]}


def _quantize(wf: np.ndarray, max_rows: int = 256, max_cols: int = 512) -> list:
    r_step = max(1, wf.shape[0] // max_rows)
    c_step = max(1, wf.shape[1] // max_cols)
    small = wf[::r_step, ::c_step]
    return [[round(float(v), 1) for v in row] for row in small]
