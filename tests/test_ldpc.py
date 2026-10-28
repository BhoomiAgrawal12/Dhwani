"""S9 LDPC: bundled matrices, Layered Decoder and Pre-screen.

The bundled codes (802.11n n=648 rate 1/2 and 3/4, CCSDS AR4JA k=1024 rate
1/2 and 4/5) are read from .qc / .alist files at import; the layered offset
min-sum decoder replaces the hard-decision bit flipping; the Pre-screen
picks (code, alignment) before any decode and must stay silent on uncoded
data, structured or not.  The old API (from_parity_check, decode,
decode_stream, identify_ldpc's candidates and min_rate, an empty
ldpc_candidates turning LDPC off) keeps working as before."""
import dataclasses
import json
import os
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from dhwani.bits.packing import pack_rows
from dhwani.fec import ldpc, registry
from dhwani.fec.detect import identify_fec
from dhwani.fec.ldpc import (LDPCCode, identify_ldpc, layered_decode,
                             prescreen, read_alist, read_qc, settle_offset,
                             write_alist)
from dhwani.gf2.rank import gf2_rank_packed

CODES = os.path.join(os.path.dirname(ldpc.__file__), "codes")
R12, R34 = "802.11n 648 r1/2", "802.11n 648 r3/4"
AR12, AR45 = "AR4JA k1024 r1/2", "AR4JA k1024 r4/5"


def _code(name):
    return registry.LDPC_REGISTRY[name]["code"]


def _rank(H):
    return gf2_rank_packed(pack_rows(np.ascontiguousarray(H).ravel(), H.shape[1]),
                           H.shape[1])


def _bpsk_awgn(c, ebn0_db, rate, rng):
    """Hard decisions and channel LLRs (> 0 => bit 0) of BPSK over AWGN."""
    sigma2 = 1.0 / (2.0 * rate * 10 ** (ebn0_db / 10))
    y = (1.0 - 2.0 * c) + rng.normal(0.0, np.sqrt(sigma2), c.shape)
    return (y < 0).astype(np.uint8), 2.0 * y / sigma2


def _noisy_stream(code, ebn0_db, n_cw, seed):
    """n_cw codewords after a random number (the unknown alignment) of
    random bits, BPSK over AWGN: (hard bits, LLRs, info, alignment)."""
    rng = np.random.default_rng(seed)
    info = rng.integers(0, 2, n_cw * code.k).astype(np.uint8)
    off = int(rng.integers(0, code.n))
    tx = np.concatenate([rng.integers(0, 2, off).astype(np.uint8), code.encode(info)])
    hard, llr = _bpsk_awgn(tx, ebn0_db, code.k / code.n, rng)
    return hard, llr, info, off


def _nrzi(bits):
    """NRZI as AX.25 sends it: a 0 flips the level, a 1 holds it."""
    return (np.cumsum(1 - np.asarray(bits, np.int64)) % 2).astype(np.uint8)


def _hdlc(payloads, flags=4):
    """HDLC bit stream: flag fill, then each payload LSB first with a 0
    stuffed after five 1s, flags between frames."""
    flag = [0, 1, 1, 1, 1, 1, 1, 0]
    out = flag * flags
    for p in payloads:
        run = 0
        for byte in p:
            for k in range(8):
                b = (byte >> k) & 1
                out.append(b)
                run = run + 1 if b else 0
                if run == 5:
                    out.append(0)
                    run = 0
        out += flag * flags
    return np.array(out, np.uint8)


def _bit_flipping(H, c, max_iters=30):
    """The decoder this commit replaces: flip the bits in the most
    unsatisfied checks until the syndrome is zero."""
    c = c.copy()
    for _ in range(max_iters):
        syn = (H @ c) % 2
        if not syn.any():
            return c, True
        votes = H.T @ syn
        if votes.max() == 0:
            break
        c[votes == votes.max()] ^= 1
    return c, not ((H @ c) % 2).any()


# 1 --------------------------------------------------------------------
def test_readers_and_bundled_matrices(tmp_path):
    with open(os.path.join(CODES, "index.json")) as f:
        index = json.load(f)
    assert [e["name"] for e in index] == [R12, R34, AR12, AR45]
    assert all(e["source"] for e in index)
    shapes = {R12: (324, 648), R34: (162, 648), AR12: (1536, 2560),
              AR45: (384, 1408)}
    ranks = {R12: 324, R34: 162, AR12: 1536, AR45: 384}
    for e in index:
        path = os.path.join(CODES, e["file"])
        H = read_qc(path) if e["file"].endswith(".qc") else read_alist(path)
        assert H.shape == shapes[e["name"]] and H.dtype == np.uint8
        assert _rank(H) == ranks[e["name"]], e["name"]
        assert np.array_equal(H, _code(e["name"]).H)
        # alist round trip, with and without zero-padded lists
        p = str(tmp_path / "h.alist")
        write_alist(p, H)
        assert np.array_equal(read_alist(p), H)
    H = _code(R12).H
    rows = [np.flatnonzero(r) + 1 for r in H]
    cols = [np.flatnonzero(c) + 1 for c in H.T]
    dc, dr = max(map(len, cols)), max(map(len, rows))
    pad = lambda v, d: " ".join(map(str, list(v) + [0] * (d - len(v))))
    lines = [f"{H.shape[1]} {H.shape[0]}", f"{dc} {dr}",
             " ".join(str(len(c)) for c in cols), " ".join(str(len(r)) for r in rows)]
    lines += [pad(c, dc) for c in cols] + [pad(r, dr) for r in rows]
    p = str(tmp_path / "padded.alist")
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    assert np.array_equal(read_alist(p), H)
    # rate 1/2 has girth 6: no two checks share two bits; rate 3/4 has
    # 4-cycles in the published matrix (54 row pairs share two bits)
    overlap = H.astype(np.int64) @ H.T.astype(np.int64)
    np.fill_diagonal(overlap, 0)
    assert overlap.max() == 1
    H34 = _code(R34).H.astype(np.int64)
    o34 = H34 @ H34.T
    np.fill_diagonal(o34, 0)
    assert o34.max() == 2 and (o34 == 2).sum() == 2 * 54


def test_registry_ldpc_entries(config):
    entries = registry.ldpc_entries()
    assert [e["name"] for e in entries] == [R12, R34, AR12, AR45]
    expect = {R12: (648, 324, 648, 0), R34: (648, 486, 648, 0),
              AR12: (2048, 1024, 2560, 512), AR45: (1280, 1024, 1408, 128)}
    rng = np.random.default_rng(4)
    seeded = [registry.ldpc_entry(c) for c in config.fec.ldpc_candidates]
    assert registry.ldpc_entry(config.fec.ldpc_candidates[1]) is seeded[1]
    for i, e in enumerate(entries + seeded):
        code = e["code"]
        assert e["family"] == "ldpc"
        if i < len(entries):
            n, k, n_full, n_punct = expect[e["name"]]
            assert (code.n, code.k, code.n_full, len(code.punctured)) == \
                (n, k, n_full, n_punct) == (e["n"], e["k"], e["n_full"],
                                            len(e["punctured"]))
            # the last M columns are the punctured ones
            assert np.array_equal(e["punctured"], np.arange(n, n_full))
        # sparse edge list = H
        H = np.zeros((code.m, code.n_full), np.uint8)
        H[e["rows"], e["cols"]] = 1
        assert np.array_equal(H, code.H) and len(e["rows"]) == code.H.sum()
        # layers: every row once, no column twice inside a layer
        seen = np.concatenate([lay["rows"] for lay in e["layers"]])
        assert np.array_equal(np.sort(seen), np.arange(code.m))
        for lay in e["layers"]:
            assert len(np.unique(lay["cols"])) == lay["cols"].size
            assert np.array_equal(code.H[lay["rows"][:, None], lay["cols"]],
                                  np.ones(lay["cols"].shape, np.uint8))
        # probe checks: lightest first, all hold on codewords, none touches
        # a punctured bit (they are over the transmitted positions)
        info = rng.integers(0, 2, 4 * code.k).astype(np.uint8)
        cw = code.encode(info).reshape(4, code.n)
        weights = [taps.shape[1] for taps in e["checks"]]
        assert weights == sorted(weights) and e["check_weight"] == weights[0]
        for taps in e["checks"]:
            assert taps.max() < code.n
            assert not (cw[:, taps].sum(axis=2) % 2).any(), e["name"]
    # AR4JA checks cancel the punctured columns: rate 1/2 gives weight 11
    # (a weight-3 row plus the two rows that each touch one of its
    # punctured bits), rate 4/5 weights 33 and 35
    assert registry.LDPC_REGISTRY[AR12]["check_weight"] == 11
    assert [t.shape for t in registry.LDPC_REGISTRY[AR12]["checks"]][0] == (512, 11)
    assert registry.LDPC_REGISTRY[AR45]["check_weight"] == 33
    # and they span the sent code's checks: n - k independent rows
    for name in (AR12, AR45):
        code = _code(name)
        T = code.structure()["T_csr"].toarray().astype(np.uint8)
        assert T.shape == (code.n - code.k, code.n) and _rank(T) == code.n - code.k
    # 802.11n layers are its block rows (Z = 27)
    assert len(registry.LDPC_REGISTRY[R12]["layers"]) == 12
    # the seeded codes have very light checks (an info bit and a parity
    # bit): the Pre-screen tests below check them on structured data
    assert min(e["check_weight"] for e in seeded) == 2


# 2 --------------------------------------------------------------------
def test_bundled_codes_encode_to_zero_syndromes():
    rng = np.random.default_rng(6)
    for name in (R12, R34, AR12, AR45):
        code = _code(name)
        info = rng.integers(0, 2, 3 * code.k).astype(np.uint8)
        tx = code.encode(info)
        assert len(tx) == 3 * code.n
        blocks = tx.reshape(3, code.n)
        # systematic, information first (the parity part of H is invertible)
        assert np.array_equal(blocks[:, :code.k], info.reshape(3, code.k))
        assert not code.syndromes(blocks).any()
        assert code.syndrome_zero_rate(tx) == 1.0
        # noiseless LLRs: the punctured bits enter at 0 and come back
        # filled in, so the full word satisfies every row of H
        full, ok = layered_decode(8.0 * (1.0 - 2.0 * blocks), code)
        assert ok.all() and full.shape == (3, code.n_full)
        assert np.array_equal(full[:, :code.n], blocks)
        assert not ((full.astype(np.int64) @ code.H.T) % 2).any()
        dec, conv = code.decode(blocks[1])
        assert conv and np.array_equal(dec, info[code.k:2 * code.k])
        # a partial last block is zero-padded
        assert len(code.encode(info[:code.k + 5])) == 2 * code.n
    # the seeded construction is unchanged, and the same matrix as a
    # parity-check code encodes to the same codewords
    seeded = LDPCCode(256, 128, seed=1)
    info = rng.integers(0, 2, 4 * 128).astype(np.uint8)
    old = np.concatenate([info.reshape(4, 128),
                          (info.reshape(4, 128) @ seeded.P.T) % 2], axis=1)
    assert np.array_equal(seeded.encode(info), old.ravel())
    pc = LDPCCode.from_parity_check(seeded.H, "seeded copy")
    assert (pc.n, pc.k, pc.m, pc.seed) == (256, 128, 128, "seeded copy")
    assert np.array_equal(pc.encode(info), seeded.encode(info))
    assert pc.n_full == pc.n and len(pc.punctured) == 0


def test_punctured_code_without_singleton_rows():
    """The general elimination: a punctured column touched only by a row
    that also touches another punctured bit (no AR4JA-style single row)."""
    H = LDPCCode(64, 32, seed=3).H.copy()
    H[0] ^= H[1]                  # row 0 now touches parity bits 32 and 33
    code = LDPCCode.from_parity_check(H, "punctured test", punctured=[33, 32])
    assert (code.n, code.k, code.n_full) == (62, 32, 64)
    assert list(code.punctured) == [32, 33]
    T = code.structure()["T_csr"].toarray().astype(np.uint8)
    assert T.shape == (30, 62) and _rank(T) == 30       # n - k checks
    rng = np.random.default_rng(5)
    info = rng.integers(0, 2, 6 * 32).astype(np.uint8)
    tx = code.encode(info).reshape(6, 62)
    assert not code.syndromes(tx).any()
    full, ok = layered_decode(8.0 * (1.0 - 2.0 * tx), code)
    assert ok.all() and not ((full.astype(np.int64) @ H.T) % 2).any()
    with pytest.raises(ValueError):
        code.syndromes(tx[:, :50])
    # a redundant row leaves no systematic encoder of k = n_full - m
    with pytest.raises(ValueError):
        LDPCCode.from_parity_check(np.vstack([H, H[:1]]), "redundant").encode(info)


def test_parity_check_codes_decode_as_before():
    """from_parity_check(H).decode / decode_stream return the first k bits
    of the corrected word for any H, as the bit-flipping decoder did: a
    redundant H (published matrices often have dependent rows) and an H
    whose last m columns are singular decode; only encode() needs an
    information set in the first k columns.  Any length works (the packed
    elimination once failed for n mod 64 in 49..56)."""
    rng = np.random.default_rng(14)
    base = LDPCCode(64, 32, seed=3)
    cw = base.encode(rng.integers(0, 2, 32).astype(np.uint8))
    # a dependent row (the sum of two others); an even-column-weight H
    # (every column in one row of each half: the rows sum to zero)
    gal = np.zeros((48, 96), np.uint8)
    for half in (0, 24):
        gal[half + rng.integers(0, 24, 96), np.arange(96)] = 1
    for H, word in ((np.vstack([base.H, base.H[0] ^ base.H[1]]), cw),
                    (gal, np.zeros(96, np.uint8))):
        pc = LDPCCode.from_parity_check(H, "redundant")
        assert _rank(H) < pc.m and pc.k == pc.n - pc.m
        dec, ok = pc.decode(word)
        assert ok and np.array_equal(dec, word[:pc.k])
        out, rate = pc.decode_stream(np.tile(word, 3))
        assert rate == 1.0 and np.array_equal(out, np.tile(word[:pc.k], 3))
        with pytest.raises(ValueError):
            pc.encode(np.zeros(pc.k, np.uint8))
    bad = cw.copy()
    bad[5] ^= 1
    dec, ok = LDPCCode.from_parity_check(np.vstack([base.H, base.H[0] ^ base.H[1]]),
                                         "redundant").decode(bad)
    assert ok and np.array_equal(dec, cw[:31])
    # parity first (the last m columns singular): the first k bits again
    base2 = LDPCCode(96, 48, seed=4)
    perm = np.concatenate([np.arange(48, 96), np.arange(48)])
    H2 = base2.H[:, perm]
    assert _rank(H2) == 48 and _rank(H2[:, 48:]) < 48
    cw2 = base2.encode(rng.integers(0, 2, 48).astype(np.uint8))[perm]
    pc = LDPCCode.from_parity_check(H2, "parity first")
    dec, ok = pc.decode(cw2)
    assert ok and np.array_equal(dec, cw2[:48])
    with pytest.raises(ValueError):
        pc.encode(np.zeros(48, np.uint8))
    # every length: H = [X | I]
    for n in (52, 56, 120, 1016):
        k = n // 2
        X = (rng.random((n - k, k)) < 4 / (n - k)).astype(np.uint8)
        pc = LDPCCode.from_parity_check(np.concatenate([X, np.eye(n - k, dtype=np.uint8)],
                                                       axis=1), f"n={n}")
        info = rng.integers(0, 2, 2 * k).astype(np.uint8)
        tx = pc.encode(info)
        assert not pc.syndromes(tx.reshape(2, n)).any()
        out, rate = pc.decode_stream(tx)
        assert rate == 1.0 and np.array_equal(out, info), n
    # syndromes of any stack of words, and wrong lengths raise ValueError
    stack = np.tile(cw, (2, 3, 1))
    assert base.syndromes(stack).shape == (2, 3, 32) and not base.syndromes(stack).any()
    assert base.syndromes(cw).shape == (32,)
    with pytest.raises(ValueError):
        base.decode(cw[:50])


# 3 --------------------------------------------------------------------
def test_layered_decoder_on_80211n_half_rate_where_bit_flipping_fails():
    code = _code(R12)
    rng = np.random.default_rng(2)
    info = rng.integers(0, 2, 240 * code.k).astype(np.uint8)
    cw = code.encode(info).reshape(240, code.n)
    hard, llr = _bpsk_awgn(cw, 2.5, 0.5, rng)
    assert 0.08 < (hard != cw).mean() < 0.10      # the channel BER is ~9%
    dec, conv = layered_decode(llr, code)
    assert dec.dtype == np.uint8 and conv.dtype == bool
    fer = (dec != cw).any(axis=1).mean()
    assert fer < 0.1                              # measured ~5e-4 for this code
    assert conv.mean() > 0.9 and not code.syndromes(dec[conv]).any()
    # the old hard-decision bit flipping fails at this Eb/N0
    bf = [_bit_flipping(code.H, h)[0] for h in hard[:60]]
    assert (np.array(bf) != cw[:60]).any(axis=1).mean() > 0.5
    # decode_stream takes the soft values too; hard input gets +-L0
    out, rate = code.decode_stream(hard.ravel(), llrs=llr.ravel())
    assert rate > 0.9
    ok = np.repeat(conv, code.k)
    assert np.array_equal(out[ok], info[ok])
    # 2.0 dB (FER ~1.5%, published 1.3%): still well below 5%
    info = rng.integers(0, 2, 400 * code.k).astype(np.uint8)
    cw = code.encode(info).reshape(400, code.n)
    _, llr = _bpsk_awgn(cw, 2.0, 0.5, rng)
    dec, _ = layered_decode(llr, code)
    assert (dec != cw).any(axis=1).mean() < 0.05


def test_hard_input_decoding_and_api_compatibility():
    """decode(codeword) and decode_stream(bits) keep their signatures and
    return types; hard bits become +-L0 from the BER their checks read."""
    code = _code(R12)
    rng = np.random.default_rng(3)
    info = rng.integers(0, 2, 40 * code.k).astype(np.uint8)
    tx = code.encode(info)
    noisy = tx.copy()
    noisy[rng.random(len(tx)) < 0.02] ^= 1
    out, rate = code.decode_stream(noisy, offset=0)
    assert rate > 0.9 and (out != info).mean() < 0.002
    dec, conv = code.decode(noisy[:code.n])
    assert isinstance(conv, bool) and len(dec) == code.k
    # offset and max_blocks as before
    out2, _ = code.decode_stream(np.concatenate([[1, 0, 1], noisy]), offset=3,
                                 max_blocks=5)
    assert len(out2) == 5 * code.k and (out2 != info[:5 * code.k]).mean() < 0.002
    empty, r0 = code.decode_stream(noisy[:100])
    assert len(empty) == 0 and r0 == 0.0
    # the seeded code decodes through the same decoder
    seeded = LDPCCode(256, 128, seed=1)
    msg = rng.integers(0, 2, 20 * 128).astype(np.uint8)
    c = seeded.encode(msg)
    c[rng.choice(len(c), 15, replace=False)] ^= 1
    out, rate = seeded.decode_stream(c)
    assert rate > 0.6


def test_ar4ja_decodes_with_punctured_bits_at_llr_zero():
    rng = np.random.default_rng(8)
    for name, ebn0, frames in ((AR12, 2.5, 30), (AR45, 4.5, 20)):
        code = _code(name)
        info = rng.integers(0, 2, frames * code.k).astype(np.uint8)
        cw = code.encode(info).reshape(frames, code.n)
        hard, llr = _bpsk_awgn(cw, ebn0, code.k / code.n, rng)
        assert llr.shape[1] == code.n < code.n_full
        dec, conv = layered_decode(llr, code)
        assert dec.shape == (frames, code.n_full)
        assert (dec[:, :code.n] != cw).any(axis=1).mean() < 0.1, name
        assert conv.mean() > 0.9
        out, rate = code.decode_stream(hard.ravel(), llrs=llr.ravel())
        assert rate > 0.9 and (out != info).mean() < 0.01


# 4 --------------------------------------------------------------------
@pytest.mark.parametrize("name, ebn0", [(R34, 3.5), (R12, 2.5), (AR12, 3.0),
                                        (AR45, 4.5)])
def test_prescreen_finds_code_and_alignment(name, ebn0):
    code = _code(name)
    rng = np.random.default_rng(7)
    info = rng.integers(0, 2, max(14, 40000 // code.n) * code.k).astype(np.uint8)
    off = 311
    tx = np.concatenate([rng.integers(0, 2, off).astype(np.uint8),
                         code.encode(info),
                         rng.integers(0, 2, 200).astype(np.uint8)])
    hard, llrs = _bpsk_awgn(tx, ebn0, code.k / code.n, rng)
    for soft in (llrs, None):
        hits = prescreen(hard, llrs=soft)
        assert len(hits) == 1, "hard" if soft is None else "soft"
        h = hits[0]
        assert (h["name"], h["offset"], h["inverted"]) == (name, off, False)
        assert h["early_verdict"]["verdict"] == "code"
        assert h["decoy_z"] >= 7.0 and h["confirm"]["verdict"] == "code"
        assert 0 < h["early_verdict"]["ber_estimate"] < 0.15
        # n_checks counts the check values the scan read, hard or soft
        assert h["n_checks"] == h["codewords"] * sum(
            len(t) for t in ldpc._scan_checks(h["entry"], h["codewords"]))
    # prescreen(llrs), as the plan writes it: the bits are the LLRs' signs
    assert prescreen(llrs)[0]["offset"] == off
    # identify_ldpc keeps HEAD's hit keys and adds the name and verdict; a
    # noisy stream has few clean codewords, so the old gate is off here
    hits = identify_ldpc(hard, [{"name": name}], min_rate=None, llrs=llrs)
    assert hits and {"n", "k", "seed", "offset", "syndrome_zero_rate", "code",
                     "name", "early_verdict"} <= set(hits[0])
    assert (hits[0]["n"], hits[0]["k"], hits[0]["offset"]) == (code.n, code.k, off)
    assert hits[0]["code"] is code
    assert hits[0]["syndrome_zero_rate"] < 0.5


def test_prescreen_settles_the_alignment_of_quasi_cyclic_codes(config):
    """A window one bit off an AR4JA codeword is the codeword with every
    128-bit circulant rotated, but for the circulant boundaries: most
    checks hold there, the decoder converges there onto the rotated
    codewords, and the scan's own best alignment is often the neighbour.
    The Pre-screen settles it by the confirmation's z; identify_fec then
    compares the neighbours' decodes (a rotated codeword also flips the
    boundary bits)."""
    code = _code(AR12)
    e = registry.LDPC_REGISTRY[AR12]
    rng = np.random.default_rng(15)
    info = rng.integers(0, 2, 12 * code.k).astype(np.uint8)
    tx = np.concatenate([rng.integers(0, 2, 700).astype(np.uint8), code.encode(info)])
    # noiseless: the neighbour's decode converges onto wrong data ...
    _, words, ok = code.decode_blocks(tx, 701, 6)
    assert ok.all() and (words[:, :code.k] != info[:6 * code.k].reshape(6, -1)).mean() > 0.3
    # ... yet the Pre-screen, settle_offset and identify_fec find 700
    assert prescreen(tx)[0]["offset"] == 700
    assert settle_offset(code, tx, 699) == settle_offset(code, tx, 701) == 700
    # noisy (Eb/N0 2.5 dB soft): the scan's best is the neighbour 729
    hard, llr, info, off = _noisy_stream(code, 2.5, 19, 1031)
    pr = ldpc._Probe(hard, llr)
    scan = ldpc._scan(pr, e, ldpc.PRESCREEN_CODEWORDS, ldpc.ALPHA, ldpc.BETA)
    assert off == 728 and int(np.argmax(np.where(scan["state"] == 1, scan["z"],
                                                 -np.inf)[0])) == 729
    assert prescreen(hard, llrs=llr)[0]["offset"] == 728
    # Eb/N0 2.0 dB: the Pre-screen itself ends a bit off (seeds 2010, 2017),
    # the neighbours' decodes put it right, and the data comes out
    for seed, true_off in ((2010, 844), (2017, 275), (2000, 1196)):
        hard, llr, info, off = _noisy_stream(code, 2.0, 19, seed)
        assert off == true_off
        hyp = identify_fec(hard, config.fec, try_rs=False, try_conv=False, llrs=llr)[0]
        assert (hyp.family, hyp.parameters["offset"]) == ("ldpc", off), seed
        assert (hyp.decoded_bits[:len(info)] != info).mean() < 0.01, seed
    for seed in (2010, 2017):
        hard, llr, info, off = _noisy_stream(code, 2.0, 19, seed)
        assert prescreen(hard, llrs=llr)[0]["offset"] != off
        assert settle_offset(code, hard, prescreen(hard, llrs=llr)[0]["offset"], llr) == off


def test_prescreen_tolerates_junk_before_the_first_codeword(config):
    """A capture that opens with a preamble or mid-codeword has junk at the
    first whole-codeword alignment: the old sweep's 5-of-8 rate gate
    tolerated three such codewords, and the Early Verdict may start at any
    of the first four."""
    seeded = LDPCCode(256, 128, seed=1)
    rng = np.random.default_rng(16)
    msg = rng.integers(0, 2, 30 * 128).astype(np.uint8)
    body = seeded.encode(msg)
    for pre in (rng.integers(0, 2, 300).astype(np.uint8),
                rng.integers(0, 2, 812).astype(np.uint8),
                np.tile(np.array([0, 1], np.uint8), 256)):
        s = np.concatenate([pre, body])
        hits = identify_ldpc(s, config.fec.ldpc_candidates)
        assert [(h["n"], h["offset"]) for h in hits] == [(256, len(pre) % 256)]
        assert hits[0]["syndrome_zero_rate"] > 0.85
        hyp = identify_fec(s, config.fec)[0]
        assert hyp.family == "ldpc" and hyp.parameters["offset"] == len(pre) % 256
        assert np.array_equal(hyp.decoded_bits[-len(msg):], msg)
    # a noisy bundled code after a codeword and a half of junk
    code = _code(R12)
    info = rng.integers(0, 2, 30 * code.k).astype(np.uint8)
    tx = np.concatenate([rng.integers(0, 2, 1000).astype(np.uint8), code.encode(info)])
    hard, llr = _bpsk_awgn(tx, 3.0, 0.5, rng)
    for soft in (llr, None):
        assert prescreen(hard, llrs=soft)[0]["offset"] == 1000 % 648


def test_identify_fec_decodes_noisy_ldpc_with_soft_values(config):
    """Eb/N0 2.5 dB: ~9% of the bits are wrong and almost no codeword is
    clean, which the old clean-codeword gate never admitted; the Pre-screen
    picks the code and the soft layered decode recovers the data."""
    code = _code(R12)
    rng = np.random.default_rng(12)
    info = rng.integers(0, 2, 20 * code.k).astype(np.uint8)
    off = 100
    tx = np.concatenate([rng.integers(0, 2, off).astype(np.uint8), code.encode(info)])
    hard, llrs = _bpsk_awgn(tx, 2.5, 0.5, rng)
    hyp = identify_fec(hard, config.fec, try_rs=False, try_conv=False, llrs=llrs)[0]
    assert hyp.family == "ldpc"
    p = hyp.parameters
    assert (p["n"], p["k"], p["offset"], p["name"]) == (648, 324, off, R12)
    assert {"seed", "decode_converged_rate", "note"} <= set(p)
    assert p["early_verdict"]["verdict"] == "code" and p["soft_decision"]
    assert p["decode_converged_rate"] > 0.9 and hyp.syndrome_zero_rate > 0.9
    assert p["clean_codeword_rate"] < 0.35      # what the old gate read
    assert (hyp.decoded_bits[:len(info)] != info).mean() < 0.005
    assert 0.05 < hyp.pre_fec_ber < 0.13


def test_inverted_streams_are_found_and_decoded(config):
    """A 180-degree phase ambiguity inverts every bit, and an inverted
    codeword fails exactly the odd-weight checks: the Pre-screen tests both
    polarities (802.11n's lightest checks are all odd, the seeded code's
    mixed) and says so, and the complement decodes.  identify_fec leaves
    an inverted hit to the bit layer, which tries the complemented stream
    itself (S7's polarity fan-out): taking it would report the winning
    rotation 180 degrees off."""
    rng = np.random.default_rng(13)
    cands = config.fec.ldpc_candidates + [{"name": R12}]
    for code, ebn0 in ((_code(R12), 3.0), (LDPCCode(256, 128, seed=1), None)):
        info = rng.integers(0, 2, 16 * code.k).astype(np.uint8)
        off = 57
        tx = np.concatenate([rng.integers(0, 2, off).astype(np.uint8),
                             code.encode(info)])
        if ebn0 is None:
            hard, llrs = tx.copy(), None
            hard[rng.random(len(tx)) < 0.003] ^= 1
        else:
            hard, llrs = _bpsk_awgn(tx, ebn0, code.k / code.n, rng)
        inv, inv_llrs = hard ^ 1, None if llrs is None else -llrs
        h = identify_ldpc(inv, cands, min_rate=None, llrs=inv_llrs)[0]
        assert (h["name"], h["offset"], h["inverted"]) == (code.name, off, True)
        out, rate = code.decode_stream(inv ^ 1, off, llrs=inv_llrs)
        assert rate > 0.9 and (out[:len(info)] != info).mean() < 0.002, code.name
        assert not identify_ldpc(hard, cands, min_rate=None, llrs=llrs)[0]["inverted"]
        fec = identify_fec(inv, config.fec, try_rs=False, try_conv=False, llrs=inv_llrs)
        assert all(f.family != "ldpc" for f in fec), code.name
        hyp = identify_fec(hard, config.fec, try_rs=False, try_conv=False, llrs=llrs)[0]
        assert hyp.family == "ldpc" and "inverted" not in hyp.parameters
        assert (hyp.decoded_bits[:len(info)] != info).mean() < 0.002, code.name


def test_seeded_codes_still_identified(config):
    """The configured seeded candidates go through the same Pre-screen;
    below the old clean-codeword gate (0.5% BER: a quarter of the
    codewords clean) only with min_rate off, as identify_fec calls it."""
    seeded = LDPCCode(256, 128, seed=1)
    rng = np.random.default_rng(9)
    info = rng.integers(0, 2, 30 * 128).astype(np.uint8)
    tx = np.concatenate([rng.integers(0, 2, 77).astype(np.uint8), seeded.encode(info)])
    tx[rng.random(len(tx)) < 0.005] ^= 1
    hits = identify_ldpc(tx, config.fec.ldpc_candidates, min_rate=None)
    assert hits and (hits[0]["n"], hits[0]["k"], hits[0]["seed"],
                     hits[0]["offset"]) == (256, 128, 1, 77)
    assert hits[0]["early_verdict"]["verdict"] == "code"
    assert seeded.syndrome_zero_rate(tx, 77, 8) <= 0.5
    assert identify_ldpc(tx, config.fec.ldpc_candidates) == []
    hyp = identify_fec(tx, config.fec, try_rs=False, try_conv=False)[0]
    assert hyp.family == "ldpc" and hyp.parameters["n"] == 256
    assert (hyp.decoded_bits[:len(info)] != info).mean() < 0.002


def test_identify_ldpc_and_identify_fec_keep_the_old_contract(config):
    """The candidates bound identify_ldpc's result and their own keys come
    back in the hit, one hit per candidate naming the code; min_rate gates
    on the clean fraction as before; an empty (or missing) ldpc_candidates
    still turns LDPC off in identify_fec unless ldpc_bundled=True asks for
    the bundled codes, and ldpc_bundled=False leaves them out."""
    seeded = LDPCCode(256, 128, seed=1)
    rng = np.random.default_rng(17)
    clean = np.concatenate([rng.integers(0, 2, 40).astype(np.uint8),
                            seeded.encode(rng.integers(0, 2, 20 * 128).astype(np.uint8))])
    a = identify_ldpc(clean, [{"n": 256, "k": 128, "seed": 1, "label": "A"}])
    b = identify_ldpc(clean, [{"n": 256, "k": 128, "seed": 1, "label": "B"}])
    c = identify_ldpc(clean, [{"n": 256, "k": 128}])
    assert (a[0]["label"], b[0]["label"], "label" in c[0]) == ("A", "B", False)
    assert (a[0]["offset"], a[0]["syndrome_zero_rate"], c[0]["seed"]) == (40, 1.0, 1)
    # the same code listed twice is scored once and reported twice, as HEAD did
    two = identify_ldpc(clean, [{"n": 256, "k": 128, "label": "A"},
                                {"n": 256, "k": 128, "seed": 1, "label": "B"}])
    assert [(h["label"], h["offset"]) for h in two] == [("A", 40), ("B", 40)]
    # min_rate: 0.3% BER leaves about half of the codewords clean
    noisy = clean.copy()
    noisy[rng.random(len(noisy)) < 0.003] ^= 1
    assert identify_ldpc(noisy, [{"n": 256, "k": 128}], min_rate=0.95) == []
    assert identify_ldpc(noisy, [{"n": 256, "k": 128}], min_rate=None)[0]["offset"] == 40
    # an 802.11n stream: found only when the candidates name the code
    code = _code(R12)
    hard, llr = _bpsk_awgn(code.encode(rng.integers(0, 2, 20 * code.k).astype(np.uint8)),
                           3.0, 0.5, rng)
    assert identify_ldpc(hard, config.fec.ldpc_candidates, min_rate=None, llrs=llr) == []
    assert identify_ldpc(hard, [{"name": R12}], min_rate=None, llrs=llr)[0]["name"] == R12
    assert identify_fec(hard, config.fec, try_rs=False, try_conv=False,
                        llrs=llr)[0].family == "ldpc"
    for cfg in (dataclasses.replace(config.fec, ldpc_candidates=[]),
                dataclasses.replace(config.fec, ldpc_bundled=False),
                SimpleNamespace(min_syndrome_zero_rate=0.7)):
        assert identify_fec(hard, cfg, try_rs=False, try_conv=False,
                            llrs=llr)[0].family == "none"
    assert identify_fec(clean, dataclasses.replace(config.fec, ldpc_candidates=[]),
                        try_rs=False, try_conv=False)[0].family == "none"
    only = dataclasses.replace(config.fec, ldpc_candidates=[], ldpc_bundled=True)
    hyps = identify_fec(hard, only, try_rs=False, try_conv=False, llrs=llr)
    assert hyps[0].family == "ldpc" and hyps[0].parameters["name"] == R12
    assert sum(h.family == "ldpc" for h in hyps) == 1
    # a bundled code the config already lists is not added twice
    named = dataclasses.replace(config.fec, ldpc_candidates=[{"name": R12}])
    assert sum(h.family == "ldpc" for h in identify_fec(
        hard, named, try_rs=False, try_conv=False, llrs=llr)) == 1
    with pytest.raises(ValueError):
        registry.ldpc_entry({"name": "802.11n 648 r1/2 typo"})


def test_factory_encodes_with_a_bundled_code():
    from dhwani.synth.factory import WaveformFactory
    fac = WaveformFactory(seed=3)
    bits = np.random.default_rng(0).integers(0, 2, 1000).astype(np.uint8)
    coded, info = fac._encode_fec(bits, {"family": "ldpc", "name": R12})
    assert len(coded) == 4 * 648
    assert (info["name"], info["n"], info["k"], info["rate"]) == (R12, 648, 324, "324/648")
    assert not _code(R12).syndromes(coded.reshape(-1, 648)).any()
    # the seeded form is unchanged, a "name" there is only a label
    for spec in ({"family": "ldpc", "n": 256, "k": 128, "seed": 1},
                 {"family": "ldpc", "n": 256, "k": 128, "seed": 1, "name": "my LDPC"}):
        coded, info = fac._encode_fec(bits, spec)
        assert np.array_equal(coded, LDPCCode(256, 128, 1).encode(bits))
        assert info == {"n": 256, "k": 128, "seed": 1, "rate": "128/256"}
    _, gt = fac.generate(modulation="BPSK", fec={"family": "ldpc", "name": AR45},
                         n_frames=4, n_pad_noise=0)
    assert (gt.fec["name"], gt.fec["n"], gt.fec["k"]) == (AR45, 1280, 1024)
    assert gt.coded_bits_len % 1280 == 0


def test_ldpc_codes_load_by_path(tmp_path):
    """Any .qc / .alist matrix becomes a candidate by path: {"path", "name",
    "punctured"}; a .qc file keeps its lifting size (block-row layers)."""
    qc = str(tmp_path / "wifi.qc")
    shutil.copy(os.path.join(CODES, "wifi_648_r12.qc"), qc)
    alist = str(tmp_path / "ar45.alist")
    write_alist(alist, _code(AR45).H)
    e = registry.ldpc_entry({"path": qc, "name": "my 802.11n"})
    assert e is registry.ldpc_entry({"path": qc, "name": "my 802.11n"})
    assert np.array_equal(e["code"].H, _code(R12).H) and e["code"].lift == 27
    assert len(e["layers"]) == 12 and e["name"] == "my 802.11n"
    a = registry.ldpc_entry({"path": alist, "punctured": 128})
    assert (a["n"], a["k"], a["name"]) == (1280, 1024, "ar45.alist")
    assert np.array_equal(a["punctured"], _code(AR45).punctured)
    rng = np.random.default_rng(18)
    info = rng.integers(0, 2, 12 * 324).astype(np.uint8)
    tx = np.concatenate([rng.integers(0, 2, 99).astype(np.uint8), e["code"].encode(info)])
    hits = identify_ldpc(tx, [{"path": qc, "name": "my 802.11n", "tag": 1}])
    assert [(h["name"], h["offset"], h["tag"]) for h in hits] == [("my 802.11n", 99, 1)]


def test_prescreen_input_forms_and_degenerate_llrs(config):
    """LLRs that say nothing (all zero or NaN) fall back to the bits, in
    the Pre-screen and in the decoder, instead of failing or decoding to
    the all-zero word."""
    code = _code(R12)
    rng = np.random.default_rng(19)
    info = rng.integers(0, 2, 16 * code.k).astype(np.uint8)
    tx = np.concatenate([rng.integers(0, 2, 33).astype(np.uint8), code.encode(info)])
    noisy = tx.copy()
    noisy[rng.random(len(tx)) < 0.01] ^= 1
    for bad in (np.zeros(len(tx)), np.full(len(tx), np.nan)):
        assert prescreen(noisy, llrs=bad)[0]["offset"] == 33
        out, rate = code.decode_stream(noisy, 33, llrs=bad)
        assert rate > 0.9 and (out != info).mean() < 0.002
        hyp = identify_fec(noisy, config.fec, try_rs=False, try_conv=False, llrs=bad)[0]
        assert hyp.family == "ldpc" and (hyp.decoded_bits != info).mean() < 0.002


def test_prescreen_silent_on_uncoded_streams(config):
    """Random and unbalanced bits, frames whose sync and header recur,
    readable text, a beacon repeating one frame, idle runs, HDLC flag fill
    and NRZI'd HDLC text frames, a convolutional code: no LDPC code, no
    alignment, hard, confident or noisy soft.  The seeded codes (weight-2
    checks) are scored on their own too."""
    from dhwani.fec.conv import ConvCode, conv_encode
    from dhwani.synth.factory import WaveformFactory
    r = np.random.default_rng(11)
    idle = np.concatenate([np.concatenate([np.zeros(4000, np.uint8),
                                           r.integers(0, 2, 4000).astype(np.uint8),
                                           np.ones(3000, np.uint8)])
                           for _ in range(3)])
    flags = np.tile(np.array([0, 1, 1, 1, 1, 1, 1, 0], np.uint8), 5000)
    hdlc = _hdlc([b"CQ CQ CQ de EA4ABC position report %03d: 40.4N 3.7W" % i
                  for i in range(120)], flags=6)
    streams = {
        "random": r.integers(0, 2, 40000).astype(np.uint8),
        "unbalanced": (r.random(40000) < 0.15).astype(np.uint8),
        "frames": WaveformFactory(seed=99).build_frames(480)[0],
        "text": WaveformFactory(seed=5).build_frames(160, payload_mode="text")[0],
        "beacon": np.tile(np.random.default_rng(1).integers(0, 2, 1024)
                          .astype(np.uint8), 24),
        "idle": idle,
        "flag fill": flags,
        "flag fill NRZI": _nrzi(flags),
        "HDLC NRZI": _nrzi(hdlc),
        "conv K7": conv_encode(r.integers(0, 2, 15000).astype(np.uint8),
                               ConvCode(7, (0o171, 0o133)), terminate=False)}
    seeded = [registry.ldpc_entry(c) for c in config.fec.ldpc_candidates]
    cands = config.fec.ldpc_candidates + [{"name": e["name"]} for e in registry.ldpc_entries()]
    sig2 = 1.0 / (2.0 * 10 ** 0.4)
    for label, bits in streams.items():
        # LLR magnitudes of BPSK at Es/N0 4 dB, signs from these bits
        noisy = (1.0 - 2.0 * bits) * np.abs(
            2.0 * (1.0 + r.normal(0.0, np.sqrt(sig2), len(bits))) / sig2)
        assert prescreen(bits) == [], label
        assert prescreen(bits, entries=seeded) == [], label
        assert prescreen(bits, llrs=(1.0 - 2.0 * bits) * 9.0) == [], label
        assert prescreen(bits, llrs=noisy, entries=registry.ldpc_entries() + seeded) == [], label
        assert identify_ldpc(bits, cands, min_rate=None) == [], label
        assert identify_fec(bits, config.fec, try_rs=False,
                            try_conv=False)[0].family == "none", label
