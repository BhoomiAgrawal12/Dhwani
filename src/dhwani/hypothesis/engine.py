"""The bit-layer hypothesis search (beam with retry loop).

Blind analysis is a search, not a forward pass: each demodulator ambiguity
x descrambler option is a candidate bit stream; candidates are pre-scored
with cheap objective tests (GF(2) rank structure), the best survivors get
the expensive treatment (interleaver ID -> FEC ID -> framing -> CRC), and
every step contributes evidence to the hypothesis score.  A CRC pass is
treated as near-conclusive validation.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from ..common.models import AnalysisHypothesis, InterleaverHypothesis
from ..framing.ax25 import decode_ax25
from ..framing.crc import crc_hunt
from ..framing.frames import analyze_frames, find_frame_length
from ..fec.detect import identify_fec, identify_outer
from ..gf2.rank import rank_profile
from ..interleaving.detect import identify_interleaver
from ..interleaving.interleavers import (block_deinterleave, conv_deinterleave,
                                         helical_deinterleave,
                                         ieee80211_permutation, pn_deinterleave)
from ..scrambling.berlekamp import detect_additive_scrambler


_PRESCORE_LADDER = np.array(sorted(set(
    list(range(2, 17)) +
    [20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192,
     204, 224, 240, 255, 256, 288, 320, 352, 384, 416, 448, 480, 512])))


def _cheap_structure_score(bits: np.ndarray) -> float:
    """Fast pre-score: strongest rank-deficiency significance over a sparse
    ladder of trial widths (small L for plain codes, typical interleaver
    periods up to 512 for interleaved ones). ~60 ms for 24k bits."""
    bits = bits[:24000]
    if len(bits) < 2000:
        return 0.0
    ladder = _PRESCORE_LADDER[_PRESCORE_LADDER <= len(bits) // 8]
    prof = rank_profile(bits, max_bits=24000, L_values=ladder)
    sig = float(prof["significance"].max()) if len(prof["significance"]) else 0.0
    bias = abs(float(bits.mean()) - 0.5)
    return sig + bias


def _deinterleave_with(bits: np.ndarray, hyp: InterleaverHypothesis) -> np.ndarray:
    off = hyp.parameters.get("offset", 0)
    b = bits[off:] if off else bits
    if hyp.kind == "block":
        return block_deinterleave(b, hyp.parameters["rows"], hyp.parameters["cols"])
    if hyp.kind == "helical":
        return helical_deinterleave(b, hyp.parameters["rows"],
                                    hyp.parameters["cols"], hyp.parameters["step"])
    if hyp.kind == "convolutional":
        return conv_deinterleave(b, hyp.parameters["branches"],
                                 hyp.parameters["delay"])
    if hyp.kind == "ieee80211":
        return pn_deinterleave(b, ieee80211_permutation(
            hyp.parameters["ncbps"], hyp.parameters["nbpsc"]))
    return b


def _ax25_result(bits: np.ndarray):
    """AX.25/G3RUH path: CRC-valid HDLC frames make a validated frame
    structure (frames of the most common length are stacked)."""
    found = decode_ax25(bits)
    if not found:
        return None
    lengths = [len(f["data"]) for f in found]
    n = max(set(lengths), key=lengths.count)
    same = [f for f in found if len(f["data"]) == n]
    frames = np.array([np.unpackbits(np.frombuffer(f["data"], np.uint8))
                       for f in same], np.uint8)
    fl = n * 8
    fr = analyze_frames(frames.reshape(-1), fl)
    hit = {"name": "CRC-16-X25", "width": 16, "poly": 0x1021,
           "init": 0xFFFF, "refin": True, "refout": True, "xorout": 0xFFFF,
           "data_start_byte": 0, "crc_byte_offset": n - 2,
           "passes": len(same), "total": len(same), "pass_fraction": 1.0,
           "callsigns": same[0]["callsigns"]}
    frame_result = {"frame_length": fl, "analysis": fr,
                    "candidates": [{"length": fl, "score": 1.0,
                                    "stability": 1.0}],
                    "bits_inverted": False, "ax25_frames": len(found)}
    return frame_result, [hit], frames.reshape(-1)


def _frame_and_crc(decoded: np.ndarray, framing_config) -> tuple:
    """Frame length, sync re-stacking and CRC hunt on both polarities.
    Returns (frame_result or None, crc_hits, bits the frames were cut from)."""
    ax = _ax25_result(decoded)
    if ax is not None:
        return ax
    frame_result, crc_hits = None, []
    # the Viterbi output polarity can be flipped by ambiguity
    # transforms the structural tests are blind to; framing and
    # CRC are polarity-sensitive, so try both
    for pol_bits, pol in ((decoded, False),
                          ((1 - decoded).astype(np.uint8), True)):
        frame_cands = find_frame_length(pol_bits,
                                        framing_config.min_frame_bits,
                                        framing_config.max_frame_bits)
        if not frame_cands:
            continue
        fl = frame_cands[0]["length"]
        fr = analyze_frames(pol_bits, fl)
        # re-stack so the frame starts at the sync word: the
        # decoded stream begins at an arbitrary point inside a
        # frame and the CRC hunter anchors to the frame end
        sync = fr["sync"]
        if sync.get("found") and sync["offset_bits"] > 0:
            pol_bits = pol_bits[sync["offset_bits"]:]
            fr = analyze_frames(pol_bits, fl)
        hits = crc_hunt(fr["frames"], framing_config.crc_candidates) \
            if fl % 8 == 0 else []
        # the longest-constant-run sync heuristic can anchor on
        # constant PAYLOAD bytes (e.g. repetitive text), leaving
        # the CRC off the frame end; search the remaining byte
        # rotations for a CRC-consistent boundary
        if not hits and fl % 8 == 0:
            for shift in range(1, fl // 8):
                cand_bits = pol_bits[shift * 8:]
                if len(cand_bits) < 4 * fl:
                    break
                fr2 = analyze_frames(cand_bits, fl)
                h2 = crc_hunt(fr2["frames"],
                              framing_config.crc_candidates)
                if h2:
                    pol_bits, fr, hits = cand_bits, fr2, h2
                    break
        # sync-anchored bit-level rotations: the true frame
        # phase need not be byte-aligned relative to where the
        # decoded stream happens to start, but every strong
        # constant run is a candidate frame anchor
        if not hits and fl % 8 == 0:
            # wraparound-aware: a sync word split across the
            # current stacking boundary appears as two short
            # runs at the frame edges, so runs are scanned on
            # the doubled entropy array
            ent = fr["entropy"]
            ent2 = np.concatenate([ent, ent])
            runs, i2 = [], 0
            while i2 < len(ent2):
                if ent2[i2] < 0.15:
                    j2 = i2
                    while j2 < len(ent2) and ent2[j2] < 0.15:
                        j2 += 1
                    if j2 - i2 >= 6 and i2 < fl:
                        runs.append((min(j2 - i2, fl), i2 % fl))
                    i2 = j2
                else:
                    i2 += 1
            runs = sorted(set(runs), reverse=True)
            for _rl, anchor in runs[:4]:
                if anchor == 0:
                    continue
                cand_bits = pol_bits[anchor:]
                if len(cand_bits) < 4 * fl:
                    continue
                fr2 = analyze_frames(cand_bits, fl)
                h2 = crc_hunt(fr2["frames"],
                              framing_config.crc_candidates)
                if h2:
                    pol_bits, fr, hits = cand_bits, fr2, h2
                    break
        if frame_result is None or hits:
            frame_result = {"frame_length": fl, "analysis": fr,
                            "candidates": frame_cands,
                            "bits_inverted": pol}
            crc_hits = hits
            decoded = pol_bits
        if hits:
            break
    return frame_result, crc_hits, decoded


def bitlayer_search(streams: list, config, fec_config, framing_config,
                    logger=None, progress=None) -> dict:
    """streams: list[BitStream] from the ambiguity fan-out.

    Flow: pre-score the raw ambiguity streams with a
    sparse rank scan (additive scrambling does NOT hide rank structure, so
    no descrambling is needed yet) -> for the beam survivors find the
    period P, align each library whitener's phase by maximising the rank
    deficiency at P, then run interleaver ID -> FEC ID -> framing -> CRC.
    Returns {"best": {...}, "hypotheses": [AnalysisHypothesis...]}.
    """
    from ..scrambling.berlekamp import (align_whitener, raw_word_repeats,
                                        whitener_by_repeats)
    from ..scrambling.lfsr import KNOWN_WHITENERS

    log = logger.info if logger else (lambda *a: None)
    all_hyps = []

    # ---- stage 1: cheap pre-scoring of ambiguity streams -----------------
    for s in streams:
        s.score = _cheap_structure_score(s.bits)
    # The ambiguity transforms are linear maps, so GF(2) structure scores
    # tie across variants; prefer the simplest hypothesis among ties
    # (non-differential, no swap, rotation 0) - the identifiers absorb
    # rotation effects as equivalent code descriptions.
    def order(s):
        h = s.hypothesis
        simplicity = (bool(h.get("differential")), bool(h.get("iq_swap")),
                      str(h.get("rotation", h.get("mapping", ""))))
        return (-round(s.score, 0), simplicity)
    ranked = sorted(streams, key=order)
    beam = ranked[: config.beam_width]
    log("bit-layer search: %d streams, beam %d, best pre-score %.2f",
        len(streams), len(beam), beam[0].score if beam else 0.0)

    best = None
    t_start = time.perf_counter()
    fec_cache = {}      # identical input bits (e.g. 'none' and fallback
                        # interleaver branches) share one FEC search
    budget_hit = False
    for ci, stream in enumerate(beam):
        if progress:
            progress(ci / max(1, len(beam)))
        if time.perf_counter() - t_start > config.time_budget_s:
            budget_hit = True
            log("bit-layer search budget (%.0f s) spent after %d of %d "
                "streams", config.time_budget_s, ci, len(beam))
            break
        bits = stream.bits

        # period from the rank profile of the RAW stream
        prof = rank_profile(bits, 2, config.rank_scan_max_L,
                           rows_factor=config.rank_scan_rows_factor,
                           max_bits=config.rank_scan_max_bits)
        sig, L = prof["significance"], prof["L"]
        signif = L[sig > config.deficiency_significance]
        P = int(signif[0]) if signif.size else None

        # descrambler candidates: none + phase-aligned library whiteners
        descr_options = [{"name": "none", "bits": bits, "phase": None,
                          "sig": float(sig.max()) if len(sig) else 0.0}]
        if P is not None:
            base_sig = descr_options[0]["sig"]
            for wname in KNOWN_WHITENERS:
                al = align_whitener(bits, wname, P,
                                    stop_sig=1.8 * base_sig + 5)
                descr_options.append({"name": wname, "bits": al["bits"],
                                      "phase": al["phase"], "sig": al["sig"]})
            descr_options.sort(key=lambda d: -d["sig"])
        # no code period needed: a library whitener whose removal makes the
        # sync word repeat once per frame (uncoded or systematic RS/LDPC
        # frames, too few rows for the rank scan) goes first
        raw_rep = raw_word_repeats(bits)
        for wname in KNOWN_WHITENERS:
            wr = whitener_by_repeats(bits, wname)
            if wr["repeats"] >= max(16, 3 * raw_rep):
                descr_options.insert(0, {
                    "name": wname, "bits": wr["bits"], "phase": wr["phase"],
                    "sig": float(wr["repeats"])})

        bm = detect_additive_scrambler(bits)
        # S6 LLRs ride along with the bits: a descrambler flips the sign
        # wherever it flips a bit, an interleaver inverse moves both alike
        soft = stream.llrs if (stream.llrs is not None and
                               len(stream.llrs) == len(bits)) else None

        for opt in descr_options[:2]:
            dbits = opt["bits"]
            dsoft = None if soft is None else \
                np.where(dbits != bits, -soft, soft)
            h = AnalysisHypothesis(stage="bitlayer", assumptions={
                "ambiguity": stream.hypothesis,
                "descrambler": opt["name"],
                "descrambler_phase": opt["phase"]})
            h.add_evidence("pre_structure", min(1.0, stream.score / 20))
            h.add_evidence("descramble_structure", min(1.0, opt["sig"] / 40))

            il_hyps = identify_interleaver(
                dbits, max_L=config.rank_scan_max_L,
                sig_thr=config.deficiency_significance,
                rows_factor=config.rank_scan_rows_factor,
                max_bits=config.rank_scan_max_bits)
            # frame-repetition structure can masquerade as an interleaver
            # period, so the plain (no-interleaver) branch is always tried
            # and the CRC/frame evidence decides between them
            tried = il_hyps[:2]
            if not any(h.kind == "none" for h in tried):
                tried.append(InterleaverHypothesis(kind="none", score=0.1,
                                                   parameters={"reason": "always-tried baseline"}))
            for il in tried:
                de = _deinterleave_with(dbits, il)
                de_soft = None if dsoft is None else _deinterleave_with(dsoft, il)
                # RS syndromes are not invariant under the constellation
                # ambiguity transforms (non-binary code), so RS is only
                # worth trying on plain (non-differential) streams
                try_rs = not stream.hypothesis.get("differential", False)
                key = (hashlib.sha1(np.ascontiguousarray(de).tobytes())
                       .hexdigest(), try_rs, de_soft is not None)
                if key not in fec_cache:
                    fec_cache[key] = identify_fec(de, fec_config,
                                                  try_rs=try_rs, llrs=de_soft)
                fec_hyps = fec_cache[key]
                fec = fec_hyps[0]
                decoded = fec.decoded_bits if (fec.decoded_bits is not None and
                                               len(fec.decoded_bits)) else de

                node = AnalysisHypothesis(
                    parent_id=h.id, stage="fec",
                    assumptions={**h.assumptions,
                                 "interleaver": {"kind": il.kind, **il.parameters},
                                 "fec": fec.family})
                node.evidence.update(h.evidence)
                node.add_evidence("interleaver",
                                  min(1.0, il.score / 10) if il.kind != "none" else 0.5)
                node.add_evidence("fec_syndrome", fec.syndrome_zero_rate or
                                  (0.5 if fec.family == "none" else 0.0))

                frame_result, crc_hits, decoded = _frame_and_crc(
                    decoded, framing_config)
                # concatenated codes: a confident inner convolutional decode
                # whose output carries no valid CRC may hide an RS outer code
                # ponytail: one RS sweep (~2-3 s) per such node; cache by
                # decoded-stream hash if unframed conv captures get slow
                if fec.family == "convolutional" and \
                        (fec.syndrome_zero_rate or 0) > 0.9 and \
                        not (crc_hits and crc_hits[0]["pass_fraction"] > 0.9):
                    cat = identify_outer(fec, fec_config)
                    if cat is not fec:
                        fec = cat
                        node.assumptions["fec"] = fec.family
                        frame_result, crc_hits, decoded = _frame_and_crc(
                            fec.decoded_bits, framing_config)
                if frame_result:
                    fr0 = frame_result["analysis"]
                    sync0 = fr0["sync"]
                    n_fr = len(fr0["frames"])
                    stab = frame_result["candidates"][0].get("stability", 0)
                    crc_ok = bool(crc_hits and
                                  crc_hits[0]["pass_fraction"] > 0.9)
                    # evidence-gated frame validity: a CRC pass is
                    # conclusive; otherwise demand a real sync word, high
                    # column stability and enough frames - periodicity
                    # alone must never mint a frame structure
                    valid = crc_ok or (sync0.get("found") and
                                       sync0.get("length_bits", 0) >= 16 and
                                       stab > 0.35 and n_fr >= 8)
                    frame_result["valid"] = bool(valid)
                    frame_result["validity_evidence"] = {
                        "crc_pass": crc_ok,
                        "sync_found": bool(sync0.get("found")),
                        "sync_bits": int(sync0.get("length_bits", 0) or 0),
                        "column_stability": round(float(stab), 3),
                        "n_frames": n_fr,
                    }
                    node.add_evidence("frame_stability",
                                      min(1.0, frame_result["candidates"][0]["score"]))
                    node.add_evidence("crc", crc_hits[0]["pass_fraction"]
                                      if crc_hits else 0.0)
                all_hyps.append(node)
                node.status = "validated" if (crc_hits and
                                              crc_hits[0]["pass_fraction"] > 0.9) else "open"

                # 'none' branches (fallback / always-tried baseline) let CRC
                # evidence overrule a spurious period; unvalidated, they must
                # not hide the identifier's verdict (e.g. an unrecovered PN
                # period would otherwise be reported as 'no interleaver')
                il_report = il_hyps[0] if (il.kind == "none" and
                                           il is not il_hyps[0] and
                                           node.status != "validated") else il
                summary = {"candidate": {"stream": stream, "whitener": opt},
                           "interleaver": il_report, "fec": fec,
                           "scrambler_bm": bm, "frames": frame_result,
                           "crc_hits": crc_hits, "decoded_bits": decoded,
                           "hypothesis": node}
                if best is None or node.score > best["hypothesis"].score:
                    best = summary
                # a CRC pass over many frames is conclusive; with a code it
                # must also be cleanly decodable (uncoded has no syndrome)
                if node.status == "validated" and (
                        fec.family == "none" or
                        (fec.syndrome_zero_rate or 0) > 0.95):
                    log("early exit: CRC-validated chain (score %.2f)", node.score)
                    _mark_rejected(all_hyps, node.id)
                    return {"best": best, "hypotheses": all_hyps,
                            "budget_exhausted": False}
            all_hyps.append(h)
    if best:
        _mark_rejected(all_hyps, best["hypothesis"].id)
    return {"best": best, "hypotheses": all_hyps,
            "budget_exhausted": budget_hit}


def _mark_rejected(hyps: list, winner_id: str):
    for h in hyps:
        if h.id != winner_id and h.status == "open":
            h.status = "rejected"
