"""Reed-Solomon over GF(2^m): fast numpy syndrome computation for blind
identification, with encode/decode delegated to `reedsolo` (mature,
pure-Python, MIT-ish licensed) when available.

For identification we only need syndromes: S_j = sum_i c_i * alpha^(i*j+fcr*i)
... precisely S_j = C(alpha^(fcr+j)) for j = 0..2t-1 evaluated over the
received codeword polynomial.  A candidate (n, k, fcr, generator) is
plausible when the fraction of codewords with all-zero syndromes is far
above the ~2^-m*2t chance level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import reedsolo as _reedsolo
    HAVE_REEDSOLO = True
except ImportError:          # pragma: no cover
    HAVE_REEDSOLO = False


@dataclass
class RSCode:
    n: int = 255
    k: int = 223
    prim: int = 0x11D        # field generator polynomial
    fcr: int = 1             # first consecutive root exponent
    generator: int = 2       # alpha

    def __post_init__(self):
        # the field is the one the polynomial generates: shortened codes
        # (DVB's RS(204,188) is a shortened RS(255,239)) keep the full
        # code's GF(2^8), whatever n
        self.m = self.prim.bit_length() - 1
        self.t2 = self.n - self.k          # number of parity symbols = 2t
        self._build_tables()

    def _build_tables(self):
        size = 1 << self.m
        exp = np.zeros(2 * size, dtype=np.int64)
        log = np.zeros(size, dtype=np.int64)
        x = 1
        for i in range(size - 1):
            exp[i] = x
            log[x] = i
            x <<= 1
            if x & size:
                x ^= self.prim
        exp[size - 1: 2 * size - 2] = exp[: size - 1]
        self.exp, self.log = exp, log
        self.field_order = size - 1

    def gf_pow_alpha(self, e: int) -> int:
        return int(self.exp[e % self.field_order])

    def syndromes(self, codewords: np.ndarray) -> np.ndarray:
        """Vectorised syndromes for many codewords.

        codewords: (n_cw, n) uint8 symbol matrix, index 0 = highest-degree
        coefficient (transmission order).  Returns (n_cw, 2t) int array.
        """
        n_cw, n = codewords.shape
        assert n == self.n
        # polynomial: c(x) = sum_i cw[:, i] * x^(n-1-i)
        degrees = np.arange(n - 1, -1, -1)
        out = np.zeros((n_cw, self.t2), dtype=np.int64)
        cw = codewords.astype(np.int64)
        nz = cw != 0
        logs = np.where(nz, self.log[cw], 0)
        glog = int(self.log[self.generator % (self.field_order + 1)]) if self.generator != 2 else 1
        # root_j = generator^(fcr + j); evaluate c(root_j)
        for j in range(self.t2):
            root_exp = (glog * (self.fcr + j)) % self.field_order
            # term exponent: log(c_i) + degree_i * root_exp
            te = (logs + degrees[None, :] * root_exp) % self.field_order
            vals = np.where(nz, self.exp[te], 0)
            out[:, j] = np.bitwise_xor.reduce(vals, axis=1)
        return out

    def syndrome_zero_rate(self, bits: np.ndarray, offset: int = 0) -> float:
        """Fraction of consecutive n-symbol blocks with all-zero syndromes."""
        symbols = bits_to_symbols(bits[offset:], self.m)
        n_cw = len(symbols) // self.n
        if n_cw == 0:
            return 0.0
        cw = symbols[: n_cw * self.n].reshape(n_cw, self.n)
        syn = self.syndromes(cw)
        return float((syn == 0).all(axis=1).mean())

    # --- encode/decode via reedsolo -------------------------------------
    def _rs_codec(self):
        if not HAVE_REEDSOLO:
            raise RuntimeError("reedsolo not installed")
        return _reedsolo.RSCodec(self.t2, nsize=self.n, fcr=self.fcr,
                                 prim=self.prim, generator=self.generator)

    def encode(self, data: bytes) -> bytes:
        return bytes(self._rs_codec().encode(bytearray(data)))

    def decode(self, codeword: bytes):
        """Returns (data, n_errors_corrected) or raises on failure."""
        codec = self._rs_codec()
        decoded, _, errata = codec.decode(bytearray(codeword))
        return bytes(decoded), len(errata)


def bits_to_symbols(bits: np.ndarray, m: int) -> np.ndarray:
    """MSB-first grouping of bits into m-bit symbols."""
    n_sym = len(bits) // m
    b = np.asarray(bits[: n_sym * m], dtype=np.int64).reshape(n_sym, m)
    weights = 1 << np.arange(m - 1, -1, -1)
    return (b * weights).sum(axis=1)


def symbols_to_bits(symbols: np.ndarray, m: int) -> np.ndarray:
    s = np.asarray(symbols, dtype=np.int64)
    out = np.zeros((len(s), m), dtype=np.uint8)
    for j in range(m):
        out[:, j] = (s >> (m - 1 - j)) & 1
    return out.ravel()
