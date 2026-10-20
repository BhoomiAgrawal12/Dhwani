"""Stage S7a: phase/IQ/polarity ambiguity fan-out.

A blind carrier loop locks to *a* constellation orientation.  We generate
every candidate interpretation and let the objective bit-layer tests pick
the winner (this is why the pipeline is a search, not a line)."""
from __future__ import annotations

import numpy as np

from ..common.models import BitStream
from ..demod.constellations import CONSTELLATIONS, slice_symbols


def differential_decode(bits: np.ndarray, llrs: np.ndarray = None) -> tuple:
    """d[0] = b[0], d[i] = b[i] xor b[i-1], plus its LLRs when the base
    stream has them (else None).

    The LLR of an XOR of two bits is the box-plus of theirs: tanh(Ld/2) =
    tanh(L[i]/2) tanh(L[i-1]/2).  Its magnitude is taken in the exact
    form min(|a|,|b|) + log1p(e^-(|a|+|b|)) - log1p(e^-||a|-|b||), which
    stays finite where 2 atanh(.) would round to infinity; the sign comes
    from d itself, so the LLRs always agree with the bits they go with."""
    bits = np.asarray(bits, dtype=np.uint8)
    d = np.concatenate([bits[:1], np.bitwise_xor(bits[1:], bits[:-1])])
    if llrs is None or len(llrs) != len(bits):
        return d, None
    mag = np.abs(np.asarray(llrs, dtype=np.float64))
    a, b = mag[1:], mag[:-1]
    box = np.concatenate([mag[:1], np.minimum(a, b) + np.log1p(np.exp(-(a + b)))
                          - np.log1p(np.exp(-np.abs(a - b)))])
    return d, ((1.0 - 2.0 * d) * box).astype(np.float32)


def enumerate_ambiguities(symbols: np.ndarray, modulation: str,
                          max_streams: int = 32,
                          include_differential: bool = True) -> list:
    """Returns list[BitStream] covering rotations x conjugation
    (x differential decode)."""
    if modulation not in CONSTELLATIONS or symbols is None or len(symbols) == 0:
        return []
    # a blind carrier loop can lock at ANY rotation in the constellation's
    # actual symmetry group (M for M-PSK, 4 for square QAM, 8 for these
    # 128APSK rings) - enumerating fewer leaves unrecoverable locks
    from ..demod.constellations import symmetry_order
    n_rot = symmetry_order(modulation)
    # OQPSK: the receiver can pair each I with either neighbouring symbol's
    # Q (bits intact but misaligned; a 90-degree lock swaps which side), so
    # both re-paired streams are offered too
    pairings = [("as received", symbols)]
    if modulation == "OQPSK":
        pairings += [("Q from next symbol",
                      symbols.real + 1j * np.roll(symbols.imag, -1)),
                     ("Q from previous symbol",
                      symbols.real + 1j * np.roll(symbols.imag, 1))]
    # keep the fan-out bounded: differential variants are dropped first
    # for high-order PSK (and OQPSK's three pairings) where rotations alone
    # exhaust the budget
    # ponytail: no differential OQPSK, raise max_ambiguity_streams if one shows up
    if n_rot * 2 * 2 * len(pairings) > max_streams:
        include_differential = False
    streams = []
    noise_var = max(1e-4, float(np.var(np.abs(symbols)) * 0.5))
    for pairing, syms in pairings:
        for conj in (False, True):
            base = np.conj(syms) if conj else syms
            for r in range(n_rot):
                rot = base * np.exp(2j * np.pi * r / n_rot)
                hard, llrs, _ = slice_symbols(rot, modulation, noise_var)
                hyp = {"rotation": f"{r}/{n_rot} turn", "iq_swap": conj,
                       "differential": False}
                if len(pairings) > 1:
                    hyp["stagger_pairing"] = pairing
                streams.append(BitStream(bits=hard, llrs=llrs, hypothesis=hyp))
                if include_differential:
                    d_bits, d_llrs = differential_decode(hard, llrs)
                    streams.append(BitStream(
                        bits=d_bits, llrs=d_llrs,
                        hypothesis={**hyp, "differential": True}))
                if len(streams) >= max_streams:
                    return streams
    return streams
