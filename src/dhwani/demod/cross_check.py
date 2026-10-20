"""Cross-check Gate (after S8-S10): two independent readings of the channel
bit error rate must agree.

From S6, ber_evm: the EVM through the constellation's Gaussian error model,
the nearest-neighbour approximation point by point: each point's errors go
to its neighbours at its own nearest distance d_i, each costing the label
bits the two differ in,

    ber_evm = 1/(M k) sum_i sum_{j nearest i} bits(i xor j) Q(d_i / (2 sigma)),

sigma the noise on a decision axis.  The EVM is the noise RMS against the
unit-power table: a complex table splits it over two axes, sigma =
EVM / sqrt(2); a real table (BPSK, OOK, ASK) decides on the in-phase axis
alone and takes all of it, sigma = EVM, which is exact for real-valued
input (FM-receiver audio, which S6 slices with these tables) and
conservative for an RF carrier, whose quadrature half of the noise never
reaches a decision (it predicts more errors and lets more claims pass).
BPSK and Gray QPSK both come out as Q(1/EVM), as dhwani.md gives it.  On
the Gray PSK, ASK and square-QAM tables every d_i is d_min and every
nearest neighbour differs in one bit, which is exactly (N_nn / k)
Q(d_min / (2 sigma)), N_nn the mean number of neighbours at d_min.  The
ring-major APSK and cross-QAM labels are not Gray (1.75-2.2 bits per
nearest-neighbour error) and APSK rings set their own nearest distances
(one 128APSK point in sixteen sits at the global d_min, the next distance
is 1.02 d_min): the global form reads them 2-50x low at their own EVM
gates, the per-point one within 2x (simulated through the slicer's
decision-directed EVM).  No table or no EVM (the FSK and GMSK receivers,
analog): None.

From the bit layer, ber_code: the Early Verdict's ber_estimate of the
accepted code (a concatenated pair: its inner code's).  A convolutional
claim reads it from the hard parities of its checks; a Reed-Solomon claim
from its clean-codeword fraction r as 1 - r^(1/(8n)), under 1e-3 for any
r the detector accepts (>= 0.35 by default), so RS never trips the gate.
None when the hypothesis carries no Early Verdict ('none').

A real code sees the channel's errors, so its BER cannot sit far above what
the EVM implies; a false detection (a check biased by structured data, a
code that holds on part of the stream only) reads a BER the signal does not
explain.  When ber_code > BER_RATIO * ber_evm + BER_SLACK the bit-layer
verdict is downgraded: the FEC score is scaled by DOWNGRADE, the winning
search node gets a zero "cross_check" evidence (and the frame score, a
copy of the node's, follows it), the demodulation status is capped at
DEGRADED (in the result's dict and in the DemodulationResult whose status
S11's provenance reads) and a warning names both numbers.  A consistent
check adds nothing: the gate is a veto, not a bonus (a full mark would lift
every passing chain's score, and the search it could rank is over).  Either
number missing leaves everything as it was: consistent is None (no
verdict, never a guess; test it with `is False`).
"""
from __future__ import annotations

import functools
import math
from typing import Optional

import numpy as np
from scipy.special import erfc

from .constellations import CONSTELLATIONS

# ber_code may exceed the Gaussian prediction this many times (dhwani.md):
# near a dense table's DEGRADED gate the decision-directed EVM reads low and
# the nearest-neighbour model undercounts (together up to 2.4x, simulated:
# 256QAM at 7% EVM, 8ASK audio at 13%), and a differential stream's decoded
# bits carry twice the channel BER
BER_RATIO = 4.0
# ...plus this much (dhwani.md): a clean EVM predicts ~0 while the code's
# estimate is a sampling statistic (sd 4e-3 at 1% BER over the 64-check
# minimum, 8e-4 over 2000 checks), so no code BER under 1% is ever called
# inconsistent
BER_SLACK = 0.01
DOWNGRADE = 0.5           # FEC score factor for an inconsistent claim
CAPPED_STATUS = "DEGRADED"
_SAME_DISTANCE = 1e-6     # relative: equal distances of a float table
# a table is real when no point's |imag| reaches this (exp(j pi) leaves
# BPSK's -1 with 1e-16)
_REAL_TABLE = 1e-9
# ponytail: a Gaussian error model of the whole stream.  Errors the mean
# EVM does not see - bursts (fades, interference), phase noise, cycle
# slips, or an uncoded stretch (overhead, fill) under the code's checks -
# lift a true code's BER and can trip the gate (a weight-10 check failing
# at random on ~18% of the stream reads 0.01), so the warning names them
# beside a false detection; a per-block EVM would see them.  A real table
# charges all of the EVM to its decision axis and so, on an RF carrier,
# lets more claims pass than it must; the in-phase share measured on the
# symbols (a real table's imaginary part is pure noise) would be exact
# there too.  The FSK and GMSK receivers carry no EVM, so their bit layer
# is never cross-checked.


def _q(x):
    """Gaussian tail Q(x) = P(N(0, 1) > x)."""
    return 0.5 * erfc(np.asarray(x, dtype=np.float64) / math.sqrt(2.0))


@functools.lru_cache(maxsize=None)
def _nearest(modulation: str) -> tuple:
    """(a_i, flips_i, k) per point of the unit-power table: Q(a_i / EVM) is
    its chance of crossing to each neighbour at its nearest distance d_i
    (a_i = d_i / 2 over the noise per decision axis in EVM units: 1 for a
    real table, 1 / sqrt(2) for a complex one), flips_i the label bits
    those neighbours differ in, summed over them (the table index is the
    bit label)."""
    table, k = CONSTELLATIONS[modulation]
    t = np.asarray(table, dtype=np.complex128)
    t = t / math.sqrt(float(np.mean(np.abs(t) ** 2)))
    d = np.abs(t[:, None] - t[None, :])
    np.fill_diagonal(d, np.inf)
    d_i = d.min(axis=1)
    near = d <= d_i[:, None] * (1.0 + _SAME_DISTANCE)
    labels = np.arange(len(t))
    x = labels[:, None] ^ labels[None, :]
    flips = sum((x >> b) & 1 for b in range(k))
    axes = 1 if np.all(np.abs(t.imag) < _REAL_TABLE) else 2
    return (d_i / 2.0 * math.sqrt(axes),
            (flips * near).sum(axis=1).astype(np.float64), k)


def ber_from_evm(evm_percent, modulation) -> Optional[float]:
    """The channel BER an EVM (percent of the table's RMS, as S6 reports
    it) implies under Gaussian noise; None when the EVM is unknown or the
    modulation has no constellation table (FSK, GMSK, analog)."""
    if evm_percent is None or modulation not in CONSTELLATIONS:
        return None
    evm = float(evm_percent) / 100.0
    if not 0.0 <= evm < math.inf:            # NaN, negative, infinite
        return None
    if evm == 0.0:
        return 0.0
    a_i, flips, k = _nearest(modulation)
    return min(float(np.mean(flips * _q(a_i / evm)) / k), 0.5)


def code_ber(fec) -> Optional[float]:
    """The accepted code's Early Verdict BER estimate (a concatenated pair:
    its inner code's); None when the hypothesis carries none."""
    if fec is None:
        return None
    params = fec.parameters or {}
    if fec.family == "concatenated":
        params = params.get("inner") or {}
    ev = params.get("early_verdict")
    ber = ev.get("ber_estimate") if isinstance(ev, dict) else None
    if ber is None or not math.isfinite(float(ber)):
        return None
    return float(ber)


def cross_check(ber_evm: Optional[float], ber_code: Optional[float]) -> dict:
    """{"ber_evm", "ber_code", "consistent"}: consistent unless the code's
    BER exceeds BER_RATIO times the EVM's plus BER_SLACK; None when either
    number is missing."""
    consistent = None
    if ber_evm is not None and ber_code is not None:
        consistent = bool(ber_code <= BER_RATIO * ber_evm + BER_SLACK)
    return {"ber_evm": ber_evm, "ber_code": ber_code,
            "consistent": consistent}


def _capped(status) -> str:
    """No better than CAPPED_STATUS: DEGRADED and FAILED stay as they are."""
    return status if status in ("DEGRADED", "FAILED") else CAPPED_STATUS


def cross_check_gate(result, demod=None, node=None) -> Optional[dict]:
    """The gate on an AnalysisResult after the bit layer, before S11.

    Stores the check as result.fec.cross_check and, for an inconsistent
    claim, scales result.fec.score by DOWNGRADE, adds a zero "cross_check"
    evidence to `node` (the winning bit-layer search node, whose score
    result.frames carries), caps the demodulation status at DEGRADED in
    result.demodulation and in `demod` (the DemodulationResult S11's
    provenance reads) and appends a warning with both numbers.  The EVM,
    modulation and status come from `demod` when given, else from
    result.demodulation.  Returns the check, or None without an FEC
    hypothesis.  The check and its downgrade belong to the hypothesis: one
    already checked is returned unchanged, so the downgrade applies once
    (gate each result with a hypothesis of its own)."""
    fec = result.fec
    if fec is None:
        return None
    if getattr(fec, "cross_check", None) is not None:
        return fec.cross_check
    dm = result.demodulation or {}
    if demod is not None:
        evm, modulation = demod.evm_percent, demod.modulation
        status = demod.demodulation_status
    else:
        evm, modulation = dm.get("evm_percent"), dm.get("modulation")
        status = dm.get("demodulation_status")
    check = cross_check(ber_from_evm(evm, modulation), code_ber(fec))
    fec.cross_check = check
    if check["consistent"] is not False:
        return check
    fec.score = float(fec.score) * DOWNGRADE
    if node is not None:
        node.add_evidence("cross_check", 0.0)
        if result.frames is not None:
            result.frames.score = node.score
    if demod is not None:
        demod.demodulation_status = _capped(demod.demodulation_status)
    if result.demodulation is not None:
        result.demodulation["demodulation_status"] = _capped(
            result.demodulation.get("demodulation_status"))
    code = fec.family
    if code == "concatenated":
        code = "inner " + str((fec.parameters.get("inner") or {})
                              .get("family", "code"))
    # a CRC pass through this very decode proves the code: what is left is
    # errors the model does not see, not a false detection
    cause = ("the CRC validates this decode, so likely errors"
             if node is not None and node.status == "validated" else
             "a likely false detection, or errors")
    capped = _capped(status)
    result.warnings.append(
        f"Cross-check Gate: the {code} code reads a channel BER of "
        f"{check['ber_code']:.3g}, but the {modulation} EVM of {evm}% "
        f"implies {check['ber_evm']:.3g} (inconsistent above {BER_RATIO:g}x "
        f"that + {BER_SLACK:g}): {cause} the Gaussian EVM model does not "
        "see (an uncoded stretch, bursts, phase noise, cycle slips); FEC "
        f"verdict downgraded (score x{DOWNGRADE:g})"
        + (f", demodulation status {status} capped at {capped}"
           if capped != status else ""))
    return check
