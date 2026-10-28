"""Blind FEC identification (stage S9) - core IP.

Convolutional: the dual-code test.  A coded stream satisfies its code's
parity checks (for rate 1/2 with generators (g1, g2):
g2(D)v1(D) + g1(D)v2(D) = 0).  The candidates - the Code Registry's
rate-1/2 codes with their lightest checks, the standard codes and an
exhaustive small-K generator sweep - are scored together by the Code
Probe at both stream offsets and both syndrome signs and judged by the
Early Verdict (an SPRT), so "no code" is dropped before any Viterbi
decode.  A syndrome-zero rate of min_rate still admits a candidate as
before; below it a "code" must also stand out from the sweep's own
decoys (structure in uncoded data biases wrong checks too), and its
decode must sit no further from the stream than the BER its checks read
(a check that holds on only part of the stream reads a BER no decode
reaches).  The survivors are decoded (soft Viterbi when LLRs are
aligned) with substream-inversion resolution.

Reed-Solomon: the probe's prefix-XOR scan tests every bit offset and
codeword alignment of every candidate at once; each alignment's
clean-codeword indicators get the same Early Verdict.

LDPC: the Pre-screen (ldpc.py) scores the configured candidates and the
bundled codes at every codeword alignment and passes at most one (code,
alignment) on; the layered min-sum decoder (soft when LLRs are aligned)
then has to converge on half the codewords, unless enough of them were
clean as received, and first settles the alignment of a noisy stream
among its neighbours (a quasi-cyclic code converges one bit off too).

The returned hypotheses carry `syndrome_zero_rate` - the objective
confidence metric that drives the retry loop, for convolutional codes
the fraction of zero syndromes of the best check, (1 + its hard mean)/2 -
and the Early Verdict that produced them.
"""
from __future__ import annotations

import functools
import itertools

import numpy as np

from ..common.models import FECHypothesis
from .conv import ConvCode, STANDARD_CODES, viterbi_decode, conv_encode
from .rs import RSCode, bits_to_symbols
from .ldpc import identify_ldpc, settle_offset
from .probe import (DECOY_Z, MIN_CHECKS, ber_spread, decoy_null,
                    probe_checks, probe_pairs, probe_rs_many, row_best,
                    soft_values, stream_stats, summarize, usable_llrs)
from .registry import basic_check, conv_entries, ldpc_entries, rs_entry

# A code admitted below min_rate must explain the stream: the ML decode of
# a true code is never further from the received bits than the codeword
# that was sent, so its re-encode BER stays within sampling error of the
# BER the checks read.  A check that holds on part of the stream only (a
# rate-1/3 or punctured code seen through a rate-1/2 check, NRZI'd text)
# reads a low BER that no decode reaches (measured: 11 to 85 sigma off).
AGREE_SIGMAS = 4.0     # sampling error allowed between the two BER readings
AGREE_SLACK = 0.005    # plus this: soft-decision decodes, slightly bursty errors
# an LDPC code with no clean codeword as received is kept when its layered
# decode converges on this fraction of the codewords: a wrong code (or
# random data) essentially never converges, and half still admits a stream
# near the code's decoding threshold.  A neighbouring alignment of a
# quasi-cyclic code does converge (onto the rotated codewords), which is
# why settle_offset compares the neighbours first.
LDPC_MIN_CONVERGED = 0.5
# a noisy stream's decode is tried on this many codewords before the rest
# (the Pre-screen's confirmation length): a code the decoder cannot correct
# is dropped at about 1/8 of the cost on a 400 kbit 802.11n stream
LDPC_TRIAL_BLOCKS = 64
LDPC_MAX_BLOCKS = 4096      # decode_stream's default, as before


@functools.lru_cache(maxsize=8)
def _conv_hypotheses(constraint_lengths: tuple, exhaustive_max_k: int) -> tuple:
    """Insertion-ordered (K, g1, g2) candidates and how each is scored:
    the registry's rate-1/2 codes and the standard codes by their own
    checks (taps flattened, rows bounds[i]:bounds[i+1]), then the
    exhaustive sweep by its basic dual check, one probe_pairs pass per K
    (sweeps: (K, even parts, odd parts, pair row per hypothesis)).
    sign_at[i] is the hypothesis's row (from bounds[i]) holding its basic
    check g2(D)v1 + g1(D)v2: its tap counts on v1 and v2 are the
    generators' weights, so it sees a one-substream inversion whenever
    the old sweep did, while a lighter multiple u(D)(g2, g1) with u(1) = 0
    has even counts on both substreams and never does.
    Stream-independent, so built once per sweep setting."""
    hyps, checks, sign_at, seen = [], [], [], set()
    for e in conv_entries():
        if e["rate"] == "1/2" and e["puncture"] is None:
            K, (g1, g2) = e["K"], e["generators"]
            seen.add((K, g1, g2))
            hyps.append((K, g1, g2))
            rows = [c["taps"] for c in e["checks"]]
            basic = basic_check(K, g1, g2)
            at = next((j for j, tp in enumerate(rows)
                       if np.array_equal(tp, basic)), None)
            if at is None:
                rows.append(basic)
                at = len(rows) - 1
            checks.append(rows)
            sign_at.append(at)
    for code in STANDARD_CODES:
        g1, g2 = code.generators
        if (code.K, g1, g2) not in seen:
            seen.add((code.K, g1, g2))
            hyps.append((code.K, g1, g2))
            checks.append([basic_check(code.K, g1, g2)])
            sign_at.append(0)
    sweeps = []
    # exhaustive small-K sweep: both polys must tap first and last position;
    # g2(D)v1 + g1(D)v2 = (even taps from g2) x (odd taps from g1)
    for K in constraint_lengths:
        if K > exhaustive_max_k:
            continue
        top = 1 << (K - 1)
        cands = [top | 1 | (m << 1) for m in range(1 << (K - 2))]
        idx = {g: i for i, g in enumerate(cands)}
        i = np.arange(K)
        even = tuple(2 * i[((g >> i) & 1).astype(bool)] for g in cands)
        odd = tuple(2 * i[((g >> i) & 1).astype(bool)] + 1 for g in cands)
        rows = []
        for a, b in itertools.combinations(cands, 2):
            for g1, g2 in ((a, b), (b, a)):
                if (K, g1, g2) not in seen:
                    seen.add((K, g1, g2))
                    hyps.append((K, g1, g2))
                    rows.append(idx[g2] * len(cands) + idx[g1])
                    sign_at.append(0)
        sweeps.append((K, even, odd, np.array(rows, dtype=np.intp)))
    taps = tuple(tp for c in checks for tp in c)
    bounds = np.cumsum([0] + [len(c) for c in checks]
                       + [1] * sum(len(sw[3]) for sw in sweeps))
    return tuple(hyps), taps, tuple(sweeps), bounds, tuple(sign_at)


def identify_convolutional(bits: np.ndarray, constraint_lengths=(3, 5, 7, 9),
                           exhaustive_max_k: int = 7,
                           max_test_bits: int = 100000,
                           min_rate: float = 0.7, llrs: np.ndarray = None) -> list:
    """Rate-1/2 candidates ranked by Code Probe + Early Verdict at both
    stream offsets and both syndrome signs: registry and standard codes
    with their own checks, the generator sweep in one probe_pairs pass per
    constraint length.

    Kept: a hard syndrome-zero rate >= min_rate over MIN_CHECKS checks
    (the old gate; verdict "code" or undecided), and below it a "code"
    whose z also stands DECOY_Z spreads above the sweep's own decoys of
    its check weight (decoy_null): text, repeated frames and fixed headers
    bias wrong checks of uncoded data, and the decoys measure by how much.
    syndrome_is_one comes from the basic dual check (see
    _conv_hypotheses).  Order: verdict "code" first, then mean check value
    (2 dp), lighter check, smaller K, then insertion order (registry and
    standard codes before the sweep), which keeps a code ahead of its
    catastrophic multiples u(D)(g1, g2)."""
    bits = np.asarray(bits, dtype=np.uint8)
    soft = None
    if llrs is not None and len(llrs) == len(bits):
        soft = np.asarray(llrs)[:max_test_bits]
    bits = bits[:max_test_bits]
    if len(bits) < 2 * MIN_CHECKS:
        return []
    hyps, taps, sweeps, bounds, sign_at = _conv_hypotheses(
        tuple(constraint_lengths), exhaustive_max_k)
    t = soft_values(bits, soft)
    st = stream_stats(t, bits)
    parts = [probe_checks(t, bits, list(taps), period=2, _st=st)]
    for K, even, odd, rows in sweeps:
        pr = probe_pairs(t, bits, list(even), list(odd), span=2 * K, _st=st)
        parts.append({k: v[rows] for k, v in pr.items()})
    res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    # every hypothesis and phase of the sweep is a decoy for the others
    null = decoy_null(res["z"], res["weight"])
    key, phase, sign_i, state = row_best(res)
    ranked = []
    for i, (K, g1, g2) in enumerate(hyps):
        b = bounds[i] + int(np.argmax(key[bounds[i]:bounds[i + 1]]))
        if state[b] == -1:
            continue
        p, s = phase[b], 1 - 2 * int(sign_i[b])
        if (1 + s * res["hard_mean"][b, p]) / 2 < min_rate or \
                res["n"][b, p] < MIN_CHECKS:
            if state[b] != 1 or \
                    null(res["z"][b, p], res["weight"][b, p], s) < DECOY_Z:
                continue
        r = summarize(res, range(bounds[i], bounds[i + 1]))
        hit = {"K": K, "g1": g1, "g2": g2, "offset": r["phase"],
               "syndrome_is_one": bool(res["mean"][bounds[i] + sign_at[i],
                                                   r["phase"]] < 0),
               "syndrome_zero_rate": r["syndrome_zero_rate"],
               "early_verdict": r["early_verdict"], "z": r["z"],
               "check_weight": r["weight"], "n_checks": r["n"]}
        ranked.append(((r["early_verdict"]["verdict"] != "code",
                        -round(r["mean"], 2), r["weight"], K, i), hit))
    ranked.sort(key=lambda kh: kh[0])
    return [h for _, h in ranked]


def identify_rs(bits: np.ndarray, candidates: list,
                quick_codewords: int = 8, confirm_codewords: int = 64) -> list:
    """RS candidates over every bit offset (0..m-1) and symbol alignment in
    one prefix-XOR scan, each alignment judged by the Early Verdict on its
    clean-codeword indicators.  syndrome_zero_rate is the clean fraction
    over up to confirm_codewords codewords at the winning alignment, rate
    the same over the first quick_codewords (the scan needs no quick pass;
    the key stays).  Ties go to the candidate with more distinct roots: a
    stream that satisfies 32 roots also satisfies any 16 of them."""
    bits = np.asarray(bits, dtype=np.uint8)
    if not candidates:
        return []
    entries = [rs_entry(c) for c in candidates]
    # enough codewords for the confirmation at every alignment
    bits = bits[: (confirm_codewords + 2) * max(e["n"] * e["m"] for e in entries)]
    found = [(cand, e, r) for cand, e, r in
             zip(candidates, entries, probe_rs_many(bits, entries))
             if r["early_verdict"]["verdict"] == "code"]
    ranked = []
    for cand, e, r in found:
        # an error-free codeword of the true code satisfies all its roots,
        # so a codeword clean for another candidate (same n, same
        # alignment) but not for this one refutes it: a candidate sharing
        # all roots but one (fcr 0 vs 1) otherwise passes a 1-in-256 window
        # and the Early Verdict, planned against random data, accepts it
        if any(o is not e and o["n"] == e["n"] and
               (ro["bit_offset"], ro["symbol_offset"]) ==
               (r["bit_offset"], r["symbol_offset"]) and
               (ro["clean_sequence"] & ~r["clean_sequence"]).any()
               for _, o, ro in found):
            continue
        seq = r["clean_sequence"][:confirm_codewords]
        rate = float(seq.mean()) if len(seq) else 0.0
        quick = seq[:quick_codewords]
        ranked.append(((-rate, -len(np.unique(e["roots"]))),
                       {**cand, "bit_offset": r["bit_offset"],
                        "symbol_offset": r["symbol_offset"],
                        "rate": float(quick.mean()) if len(quick) else 0.0,
                        "syndrome_zero_rate": rate,
                        "n_codewords_tested": int(len(seq)),
                        "early_verdict": r["early_verdict"]}))
    ranked.sort(key=lambda kh: kh[0])
    return [h for _, h in ranked]


def identify_fec(bits: np.ndarray, config,
                 try_rs: bool = True, try_ldpc: bool = True,
                 try_conv: bool = True, llrs: np.ndarray = None) -> list:
    """Full FEC identification: returns ranked FECHypothesis list, always
    ending with an explicit 'none' hypothesis.  `llrs` (aligned with
    `bits`, > 0 => bit 0) feed the probe and the soft Viterbi decode."""
    bits = np.asarray(bits, dtype=np.uint8)
    soft = usable_llrs(llrs, len(bits))
    hyps = []

    conv_hits = identify_convolutional(
        bits, tuple(config.conv_constraint_lengths),
        config.conv_exhaustive_max_k, config.max_test_bits,
        min_rate=config.min_syndrome_zero_rate, llrs=soft) if try_conv else []
    for hit in conv_hits[:3]:
        code = ConvCode(hit["K"], (hit["g1"], hit["g2"]))
        raw = bits[hit["offset"]:]
        raw = raw[: 2 * (len(raw) // 2)]
        # soft decoding takes only the reliabilities: the signs follow the
        # (possibly inverted) bits, so an inverted substream flips its LLRs
        rel = None if soft is None else \
            np.abs(soft[hit["offset"]:hit["offset"] + len(raw)])
        # resolve the substream inversion pattern: syndrome==0 admits
        # (none, both), syndrome==1 admits (v1 only, v2 only); pick the
        # pattern whose decode re-encodes onto the observed stream
        patterns = ((0, 0), (1, 1)) if not hit.get("syndrome_is_one") \
            else ((1, 0), (0, 1))
        decoded, pre_ber, chosen, n_cmp = None, 1.0, None, 0
        for inv1, inv2 in patterns:
            cand = raw.copy()
            if inv1:
                cand[0::2] ^= 1
            if inv2:
                cand[1::2] ^= 1
            dec = viterbi_decode(cand, code, soft=None if rel is None else
                                 np.where(cand == 0, rel, -rel))
            re_enc = conv_encode(dec, code, terminate=False)
            L = min(len(re_enc), len(cand))
            ber = float((re_enc[:L] != cand[:L]).mean())
            if ber < pre_ber:
                decoded, pre_ber, chosen, n_cmp = dec, ber, (inv1, inv2), L
            if ber < 0.02:
                break
        if decoded is None or pre_ber > 0.2:
            # decode inconsistent with observed stream: reject (protects
            # against equivalent-but-wrong trellis descriptions)
            continue
        if hit["syndrome_zero_rate"] < config.min_syndrome_zero_rate and \
                not _decode_agrees(hit, pre_ber, n_cmp):
            continue
        hyps.append(FECHypothesis(
            family="convolutional", code_rate=0.5,
            parameters={"K": hit["K"], "g1_octal": oct(hit["g1"]),
                        "g2_octal": oct(hit["g2"]), "rate": "1/2",
                        "stream_offset": hit["offset"],
                        "substream_inversion": list(chosen),
                        "soft_decision": soft is not None,
                        "early_verdict": hit["early_verdict"]},
            syndrome_zero_rate=hit["syndrome_zero_rate"],
            decoded_bits=decoded, pre_fec_ber=pre_ber,
            score=hit["syndrome_zero_rate"] * (1.0 - min(1.0, 2 * pre_ber))))

    # The RS alignment sweep is the expensive part of this stage; a
    # confident convolutional hit makes it redundant here (an RS outer code
    # is looked for in the Viterbi output by identify_outer).
    best_conv = conv_hits[0]["syndrome_zero_rate"] if conv_hits else 0.0
    rs_hits = [] if (best_conv > 0.9 or not try_rs) else \
        identify_rs(bits, config.rs_candidates)
    for hit in rs_hits[:2]:
        if hit["syndrome_zero_rate"] < config.min_syndrome_zero_rate * 0.5:
            continue
        rs = RSCode(hit["n"], hit["k"], prim=hit.get("prim", 0x11D),
                    fcr=hit.get("fcr", 1), generator=hit.get("generator", 2))
        decoded = _rs_decode_stream(bits, rs, hit["bit_offset"],
                                    hit["symbol_offset"])
        hyps.append(FECHypothesis(
            family="reed_solomon", code_rate=hit["k"] / hit["n"],
            parameters={"n": hit["n"], "k": hit["k"], "fcr": hit.get("fcr"),
                        "prim": hex(hit.get("prim", 0x11D)),
                        "bit_offset": hit["bit_offset"],
                        "symbol_offset": hit["symbol_offset"],
                        "n_codewords_tested": hit["n_codewords_tested"],
                        "early_verdict": hit["early_verdict"]},
            syndrome_zero_rate=hit["syndrome_zero_rate"],
            decoded_bits=decoded,
            score=hit["syndrome_zero_rate"] * 0.98))

    # LDPC: the Pre-screen picks (code, alignment) among the configured
    # candidates and the bundled codes before any decode (skipped when a
    # confident hit exists or no LDPC candidate is configured, as before;
    # config.ldpc_bundled: None adds the bundled codes to a non-empty list,
    # True also to an empty one, False never).
    # A clean-codeword fraction of half min_rate admits a code as before;
    # below it (a noisy stream has no clean codeword) the layered decode
    # must converge on half the codewords, and the hypothesis reports the
    # decoded zero-syndrome fraction (clean_codeword_rate keeps the other)
    best_so_far = max((h.syndrome_zero_rate or 0) for h in hyps) if hyps else 0.0
    cands = list(getattr(config, "ldpc_candidates", None) or [])
    bundled = getattr(config, "ldpc_bundled", None)
    if bundled or (bundled is None and cands):
        listed = {c.get("name") for c in cands}
        cands += [{"name": e["name"]} for e in ldpc_entries()
                  if e["name"] not in listed]
    if try_ldpc and best_so_far < 0.9 and cands:
        for hit in identify_ldpc(bits, cands, min_rate=None, llrs=soft)[:2]:
            if hit["inverted"]:
                # the complemented stream satisfies the code: the bit layer
                # tries that stream itself (S7's polarity fan-out), and
                # taking it here would report the winning rotation 180
                # degrees off
                continue
            clean = hit["syndrome_zero_rate"] >= config.min_syndrome_zero_rate * 0.5
            dec = _ldpc_decode(hit["code"], bits, hit["offset"], soft, clean)
            if dec is None:
                continue
            offset, decoded, converged, pre_ber = dec
            szr = hit["syndrome_zero_rate"] if clean else converged
            hyps.append(FECHypothesis(
                family="ldpc", code_rate=hit["k"] / hit["n"],
                parameters={"n": hit["n"], "k": hit["k"],
                            "seed": hit.get("seed", 1),
                            "offset": offset,
                            "decode_converged_rate": round(converged, 3),
                            "note": "candidate-set match (H known a priori); "
                                    "arbitrary H reconstruction is out of scope",
                            "name": hit["name"],
                            "clean_codeword_rate": hit["syndrome_zero_rate"],
                            "soft_decision": soft is not None,
                            "early_verdict": hit["early_verdict"]},
                syndrome_zero_rate=szr,
                decoded_bits=decoded, pre_fec_ber=pre_ber,
                score=szr * 0.97))

    hyps.append(FECHypothesis(family="none", score=0.2,
                              parameters={"reason": "no code identified above threshold"
                                          if not hyps else "fallback"}))
    hyps.sort(key=lambda h: -h.score)
    return hyps


def _decode_agrees(hit: dict, pre_ber: float, n_coded: int) -> bool:
    """The decode's re-encode BER is within AGREE_SIGMAS (both readings'
    sampling errors) plus AGREE_SLACK of the BER the hit's checks read."""
    p_hat = hit["early_verdict"]["ber_estimate"]
    if p_hat is None or p_hat >= 0.5:
        return False
    sd = np.hypot(ber_spread(2 * hit["syndrome_zero_rate"] - 1,
                             hit["check_weight"], hit["n_checks"]),
                  np.sqrt(pre_ber * (1 - pre_ber) / max(1, n_coded)))
    return pre_ber <= p_hat + AGREE_SIGMAS * sd + AGREE_SLACK


def _ldpc_decode(code, bits: np.ndarray, offset: int, soft, clean: bool):
    """(offset, info bits, converged rate, pre-FEC BER) of an LDPC hit, or
    None when a noisy one does not decode.  A clean hit decodes as before.
    A noisy one first settles its alignment by decoding the neighbours
    (settle_offset), then must converge on LDPC_MIN_CONVERGED of the first
    LDPC_TRIAL_BLOCKS codewords before the rest is decoded.  The pre-FEC
    BER is the fraction of received bits the decode changed (for a
    converged codeword exactly its re-encode distance)."""
    if not clean:
        offset = settle_offset(code, bits, offset, soft)
    first = LDPC_MAX_BLOCKS if clean else LDPC_TRIAL_BLOCKS
    parts = [code.decode_blocks(bits, offset, first, soft)]
    trial = parts[0][2]
    if not clean and not (len(trial) and trial.mean() >= LDPC_MIN_CONVERGED):
        return None
    if len(trial) == first < LDPC_MAX_BLOCKS:
        parts.append(code.decode_blocks(bits, offset + first * code.n,
                                        LDPC_MAX_BLOCKS - first, soft))
    blocks, words, ok = (np.concatenate(x) for x in zip(*parts))
    if not len(ok):
        return offset, np.zeros(0, dtype=np.uint8), 0.0, None
    return (offset, words[:, code.info_columns()].reshape(-1), float(ok.mean()),
            float((words[:, code.transmitted] != blocks).mean()))


def identify_outer(inner: FECHypothesis, config) -> FECHypothesis:
    """Concatenated codes: look for an RS outer code in the inner decoder's
    output. Returns a 'concatenated' hypothesis, or `inner` unchanged.
    An interleaver between the two codes is not searched for."""
    dec = inner.decoded_bits
    if dec is None or len(dec) < 2 * 255 * 8:
        return inner
    outer = identify_fec(dec, config, try_conv=False, try_ldpc=False)[0]
    if outer.family != "reed_solomon" or \
            (outer.syndrome_zero_rate or 0) < config.min_syndrome_zero_rate:
        return inner
    return FECHypothesis(
        family="concatenated", code_rate=inner.code_rate * outer.code_rate,
        parameters={"inner": {"family": inner.family, **inner.parameters},
                    "outer": {"family": outer.family, **outer.parameters}},
        syndrome_zero_rate=min(inner.syndrome_zero_rate,
                               outer.syndrome_zero_rate),
        decoded_bits=outer.decoded_bits, pre_fec_ber=inner.pre_fec_ber,
        score=inner.score)


def _rs_decode_stream(bits: np.ndarray, rs: RSCode, bit_off: int,
                      sym_off: int) -> np.ndarray:
    from .rs import symbols_to_bits
    symbols = bits_to_symbols(bits[bit_off:], rs.m)[sym_off:]
    n_cw = len(symbols) // rs.n
    out = []
    for i in range(n_cw):
        cw = bytes(symbols[i * rs.n:(i + 1) * rs.n].astype(np.uint8))
        try:
            data, _ = rs.decode(cw)
            out.append(np.frombuffer(data[:rs.k], dtype=np.uint8))
        except Exception:
            continue
    if not out:
        return np.zeros(0, dtype=np.uint8)
    data_syms = np.concatenate(out).astype(np.int64)
    return symbols_to_bits(data_syms, 8)
