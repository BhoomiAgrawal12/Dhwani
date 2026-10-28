"""Blind interleaver identification (stage S8) - the project's core IP.

Method (after Sicot & Houcke 2009):
1. GF(2) rank-deficiency profile over trial widths L.  Linear code
   structure makes the profile dip at multiples of the (interleaver x code)
   period; a flat profile means no linear structure is visible.
2. The *fundamental* significant L is the working period P.
3. If structure exists at many small L already (dense profile), the stream
   is plain-coded: verdict "none".
4. Otherwise test hypotheses: block (r x c factorisations of P), helical
   (block + step sweep), convolutional (branch/delay sweep), PN (period
   detected, permutation not recovered - reported honestly).  Each hypothesis is scored by how much *fine-grained*
   structure its de-interleaver reveals: after the correct inverse, code
   structure reappears at small L (the "second, deeper dip").
Before step 4, and on a flat profile too, the Pattern Scan (pattern_scan.py)
looks for a registry code through the interleaver; a geometry it confirms
together with its code is returned at once, otherwise the steps above run
as they always did.
"""
from __future__ import annotations

import numpy as np

from ..common.models import InterleaverHypothesis
from ..gf2.rank import (expected_random_deficiency, gf2_rank_bits,
                        rank_profile)
from .interleavers import (IEEE80211_MODES, block_deinterleave,
                           conv_deinterleave, helical_deinterleave,
                           ieee80211_permutation, pn_deinterleave)
from .pattern_scan import pattern_scan


def _structure_score(bits: np.ndarray, L_max: int, sig_thr: float) -> tuple:
    """(number of significant L below L_max, total significance, smallest
    significant L)."""
    prof = rank_profile(bits, 2, L_max, max_bits=60000)
    sig = prof["significance"]
    L = prof["L"]
    mask = sig > sig_thr
    n_sig = int(mask.sum())
    total = float(sig[mask].sum())
    smallest = int(L[mask][0]) if n_sig else 0
    return n_sig, total, smallest


def _fine_structure_sig(bits: np.ndarray, test_Ls=(16, 32)) -> float:
    """Cheap single-shot evidence of fine-grained linear structure: max
    rank-deficiency significance over a few small trial widths."""
    best = 0.0
    for L in test_Ls:
        n_rows = min(len(bits) // L, 2 * L + 32)
        if n_rows < L // 2:
            continue
        r, used = gf2_rank_bits(bits, L, n_rows)
        sig = (min(used, L) - r) - expected_random_deficiency(used, L)
        best = max(best, sig)
    return best


def _align_and_test(bits: np.ndarray, deinterleave, P: int,
                    sig_thr: float, max_offsets: int = None,
                    stride: int = 2) -> tuple:
    """Sweep the block-phase offset (the demodulated stream almost never
    starts on an interleaver boundary); returns (best_offset, best_sig).

    Coarse-to-fine: for block/convolutional inverses the fine-structure
    score decays gradually around the true offset, so a stride-2 scan
    followed by a +-1 refinement finds the same maximum at half the cost,
    and a decisively strong dip stops the scan immediately. The helical
    inverse peaks at exactly one offset, so its caller passes stride=1."""
    n_off = min(P, max_offsets or P)

    def test(off):
        return _fine_structure_sig(deinterleave(bits[off:])[:20000])

    best_off, best_sig = 0, 0.0
    stride = stride if n_off >= 16 else 1
    for off in range(0, n_off, stride):
        sig = test(off)
        if sig > best_sig:
            best_off, best_sig = off, sig
        if best_sig > 30:
            break
    if stride > 1 and best_sig > 0:
        for off in (best_off - 1, best_off + 1):
            if 0 <= off < n_off:
                sig = test(off)
                if sig > best_sig:
                    best_off, best_sig = off, sig
    return best_off, best_sig


def _factor_pairs(P: int, max_dim: int = 64):
    for r in range(2, min(P, max_dim) + 1):
        if P % r == 0:
            c = P // r
            if 2 <= c <= max_dim:
                yield r, c


def _published_permutations(bits, sig_thr, base_sig) -> list:
    """802.11a/g bit interleaver modes: a structured permutation the
    block/helical/conv sweeps cannot express, tested by the same
    fine-structure gain at stride 1 (sharp alignment). For nbpsc <= 2 it is
    exactly a block interleaver, so a block hypothesis may tie with it."""
    hyps = []
    for ncbps, nbpsc in IEEE80211_MODES:
        perm = ieee80211_permutation(ncbps, nbpsc)
        off, sig_gain = _align_and_test(
            bits, lambda b, p=perm: pn_deinterleave(b, p), ncbps, sig_thr,
            stride=1)
        if sig_gain > max(sig_thr, 2 * base_sig):
            hyps.append(InterleaverHypothesis(
                kind="ieee80211", period=ncbps,
                parameters={"ncbps": ncbps, "nbpsc": nbpsc, "offset": off},
                score=float(sig_gain) * 0.97))
    return hyps


def _scanned(found: list, profile_dict: dict) -> list:
    """Pattern Scan hits, the rank profile on the winner, 'none' fallback."""
    found[0].rank_profile = profile_dict
    return found + [InterleaverHypothesis(kind="none", score=0.1,
                                          parameters={"reason": "fallback"})]


def identify_interleaver(bits: np.ndarray, max_L: int = 512,
                         sig_thr: float = 3.0, rows_factor: int = 2,
                         max_bits: int = 200000,
                         progress=None, llrs: np.ndarray = None,
                         fec_config=None) -> list:
    """Returns ranked list of InterleaverHypothesis (best first).

    Always includes an explicit 'none' hypothesis so downstream stages can
    proceed with the raw stream when no interleaving is detected.  `llrs`
    (aligned with `bits`, > 0 => bit 0) feed the Pattern Scan's code probes;
    `fec_config` (FECConfig, its defaults when None) gives it the RS
    candidates and the convolutional search settings.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    prof = rank_profile(bits, 2, max_L, rows_factor=rows_factor,
                        max_bits=max_bits, progress=progress)
    profile_dict = {"L": prof["L"].tolist(),
                    "deficiency": prof["deficiency"].tolist(),
                    "significance": [round(float(s), 2) for s in prof["significance"]]}
    sig, L = prof["significance"], prof["L"]
    signif_L = L[sig > sig_thr]

    if signif_L.size == 0:
        # the period is gone (a few 0.1% of BER, or RS): look for the code
        found = pattern_scan(bits, llrs=llrs, fec_config=fec_config)
        if found:
            return _scanned(found, profile_dict)
        # a short capture can hide a permutation period from the rank scan
        # (192 bits needs ~80k bits); the published ones are cheap to test
        pub = _published_permutations(bits, sig_thr,
                                      _fine_structure_sig(bits[:20000]))
        if pub:
            pub.sort(key=lambda h: -h.score)
            pub[0].rank_profile = profile_dict
            return pub + [InterleaverHypothesis(
                kind="none", score=0.1, parameters={"reason": "fallback"})]
        return [InterleaverHypothesis(kind="none", score=1.0,
                                      rank_profile=profile_dict,
                                      parameters={"reason": "rank profile flat: no linear structure detected"})]

    P = int(signif_L[0])
    # Dense structure at many L => plain (non-interleaved) linear code.
    # Only count the L range where deficiency is detectable at all: beyond
    # L ~ sqrt(2*n_bits) there are fewer rows than the code dimension and
    # every width reads "not significant", which would dilute the density.
    n_bits_used = min(len(bits), max_bits)
    L_detectable = int(np.sqrt(2 * n_bits_used))
    denom = max(1, int(((L >= P) & (L <= L_detectable)).sum()))
    density = len(signif_L) / denom
    if P <= 8 or density > 0.4:
        return [InterleaverHypothesis(
            kind="none", period=P, score=1.0, rank_profile=profile_dict,
            parameters={"reason": f"dense rank structure from L={P}: stream is "
                        "linearly coded but not interleaved",
                        "code_period_hint": P})]

    # Candidate periods: repetitive payload data can carve spurious weak
    # dips below the true interleaver period, so try the strongest dips
    # first rather than trusting the smallest significant L alone.
    order = np.argsort(-sig[sig > sig_thr])
    by_strength = [int(v) for v in signif_L[order]]
    cand_Ps = []
    for v in by_strength[:3] + [P]:
        if v > 8 and v not in cand_Ps:
            cand_Ps.append(v)

    found = pattern_scan(bits, llrs=llrs, periods=cand_Ps, fec_config=fec_config)
    if found:
        return _scanned(found, profile_dict)

    hyps = []
    base_sig = _fine_structure_sig(bits[:20000])

    for Pc in cand_Ps:
        # --- block hypotheses (with block-phase alignment sweep) ----------
        for r, c in _factor_pairs(Pc):
            off, sig_gain = _align_and_test(
                bits, lambda b, r=r, c=c: block_deinterleave(b, r, c),
                Pc, sig_thr)
            if sig_gain > max(sig_thr, 2 * base_sig):
                hyps.append(InterleaverHypothesis(
                    kind="block", period=Pc,
                    parameters={"rows": r, "cols": c, "offset": off},
                    score=float(sig_gain)))

        # --- helical hypotheses -------------------------------------------
        if not any(h.score > 4 * sig_thr for h in hyps):
            for r, c in list(_factor_pairs(Pc))[:6]:
                for step in range(1, min(r, 6)):
                    off, sig_gain = _align_and_test(
                        bits, lambda b, r=r, c=c, st=step:
                        helical_deinterleave(b, r, c, st), Pc, sig_thr,
                        stride=1)
                    if sig_gain > max(sig_thr, 2 * base_sig):
                        hyps.append(InterleaverHypothesis(
                            kind="helical", period=Pc,
                            parameters={"rows": r, "cols": c, "step": step,
                                        "offset": off},
                            score=float(sig_gain) * 0.98))

        # --- convolutional hypotheses ---------------------------------------
        if not any(h.score > 4 * sig_thr for h in hyps):
            for B in range(2, 17):
                if Pc % B:
                    continue
                for M in (1, 2, 4, 8, max(1, Pc // B)):
                    off, sig_gain = _align_and_test(
                        bits, lambda b, B=B, M=M: conv_deinterleave(b, B, M),
                        Pc, sig_thr, max_offsets=B * M if B * M < Pc else Pc)
                    if sig_gain > max(sig_thr, 2 * base_sig):
                        hyps.append(InterleaverHypothesis(
                            kind="convolutional", period=Pc,
                            parameters={"branches": B, "delay": M, "offset": off},
                            score=float(sig_gain) * 0.95))
        if any(h.score > 4 * sig_thr for h in hyps):
            break

    if not any(h.score > 4 * sig_thr for h in hyps):
        hyps += _published_permutations(bits, sig_thr, base_sig)

    if not hyps:
        # Period detected but no tested inverse revealed finer structure:
        # report a PN/unknown interleaver honestly.
        # ponytail: period only, no permutation search (P! space); add
        # code-constraint matching over many aligned periods if an off-air
        # target needs the PN permutation recovered
        hyps.append(InterleaverHypothesis(
            kind="pseudo_random", period=P, score=0.3,
            parameters={"note": "period detected via rank dip; the "
                        "permutation is not recovered (not implemented)"}))
    hyps.sort(key=lambda h: -h.score)
    # attach profile to the winner only (keep result size sane)
    hyps[0].rank_profile = profile_dict
    # always include explicit none-hypothesis fallback
    hyps.append(InterleaverHypothesis(kind="none", score=0.1,
                                      parameters={"reason": "fallback"}))
    return hyps
