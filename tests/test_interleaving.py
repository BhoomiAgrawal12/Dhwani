import numpy as np

from dhwani.fec.conv import ConvCode, conv_encode
from dhwani.interleaving import (block_interleave, block_deinterleave,
                                     helical_interleave, helical_deinterleave,
                                     conv_interleave, conv_deinterleave,
                                     pn_interleave, pn_deinterleave)
from dhwani.interleaving.detect import identify_interleaver


def _coded(rng, n=8000):
    return conv_encode(rng.integers(0, 2, n).astype(np.uint8),
                       ConvCode(7, (0o171, 0o133)), terminate=False)


def test_roundtrips(rng):
    b = rng.integers(0, 2, 8000).astype(np.uint8)
    assert np.array_equal(block_deinterleave(block_interleave(b, 8, 16), 8, 16),
                          b[:7936])
    assert np.array_equal(
        helical_deinterleave(helical_interleave(b.copy(), 8, 16, 3), 8, 16, 3),
        b[:7936])
    assert np.array_equal(conv_deinterleave(conv_interleave(b, 4, 5), 4, 5), b)
    perm = rng.permutation(64)
    assert np.array_equal(pn_deinterleave(pn_interleave(b, perm), perm),
                          b[:len(b) // 64 * 64])


def test_floats_keep_their_dtype_bits_stay_uint8(rng):
    """LLR arrays go through the same inverses as the bits they belong to;
    integer, bool and list input still comes back as uint8."""
    b = rng.integers(0, 2, 8192).astype(np.uint8)
    f = (1 - 2.0 * b) * rng.uniform(0.5, 3, len(b))
    perm = rng.permutation(64)
    pairs = ((block_interleave, block_deinterleave, (8, 16)),
             (helical_interleave, helical_deinterleave, (8, 16, 3)),
             (conv_interleave, conv_deinterleave, (4, 5)),
             (pn_interleave, pn_deinterleave, (perm,)))
    for fwd, inv, args in pairs:
        for x in (b, f, f.astype(np.float32)):
            keep = x.copy()
            y, z = fwd(x, *args), inv(x, *args)
            assert y.dtype == z.dtype == x.dtype
            assert np.array_equal(x, keep)              # input untouched
        assert np.array_equal(inv(f, *args) < 0, inv(b, *args) == 1)
        assert np.array_equal(fwd(f, *args) < 0, fwd(b, *args) == 1)
        for x in (b.astype(np.int64), b.astype(bool), list(b[:1024])):
            for fn in (fwd, inv):
                y = fn(x, *args)
                assert y.dtype == np.uint8
                assert np.array_equal(y, fn(np.asarray(x, np.uint8), *args))


def test_identify_block(rng):
    inter = block_interleave(_coded(rng), 8, 16)
    top = identify_interleaver(inter, max_L=256)[0]
    assert top.kind == "block"
    assert (top.parameters["rows"], top.parameters["cols"]) == (8, 16)


def test_identify_block_with_offset(rng):
    inter = block_interleave(_coded(rng), 8, 16)[37:]
    top = identify_interleaver(inter, max_L=256)[0]
    assert top.kind == "block"
    assert (top.parameters["rows"], top.parameters["cols"]) == (8, 16)
    de = block_deinterleave(inter[top.parameters["offset"]:], 8, 16)
    from dhwani.fec.detect import identify_convolutional
    hits = identify_convolutional(de)
    assert hits and hits[0]["syndrome_zero_rate"] > 0.99


def test_identify_none_on_plain_code(rng):
    top = identify_interleaver(_coded(rng), max_L=256)[0]
    assert top.kind == "none"


def test_identify_none_on_random(rng):
    bits = rng.integers(0, 2, 60000).astype(np.uint8)
    top = identify_interleaver(bits, max_L=256)[0]
    assert top.kind == "none"


def test_identify_helical_with_offset(rng):
    """3.B: helical (the PS's 'diagonal') interleaver, off a boundary."""
    inter = helical_interleave(_coded(rng), 8, 16, 3)[37:]
    top = identify_interleaver(inter, max_L=256)[0]
    assert top.kind == "helical"
    assert (top.parameters["rows"], top.parameters["cols"],
            top.parameters["step"]) == (8, 16, 3)


def test_identify_convolutional_with_offset(rng):
    inter = conv_interleave(_coded(rng), 4, 8)[37:]
    top = identify_interleaver(inter, max_L=256)[0]
    assert top.kind == "convolutional"
    assert (top.parameters["branches"], top.parameters["delay"]) == (4, 8)


def test_pseudo_random_reports_period_not_permutation(rng):
    """Unknown stays unknown: the period is found, no permutation claimed."""
    perm = np.random.default_rng(0).permutation(128)
    top = identify_interleaver(pn_interleave(_coded(rng), perm), max_L=256)[0]
    assert top.kind == "pseudo_random" and top.period == 128
    assert "permutation" not in top.parameters


def test_ieee80211_permutation_structure():
    """R3: the 802.11a/g bit interleaver (k -> i -> j). For BPSK the second
    step is the identity, so it must equal a 16-column block interleaver;
    for higher orders the second step only rotates within groups of s."""
    from dhwani.interleaving.interleavers import ieee80211_permutation
    for ncbps, nbpsc in ((48, 1), (96, 2), (192, 4), (288, 6)):
        perm = ieee80211_permutation(ncbps, nbpsc)
        assert sorted(perm) == list(range(ncbps))
    b = np.arange(48 * 3) % 2 ^ (np.arange(48 * 3) // 7 % 2)
    assert np.array_equal(pn_interleave(b, ieee80211_permutation(48, 1)),
                          block_interleave(b, 3, 16))
    k = np.arange(192)
    i = (192 // 16) * (k % 16) + k // 16
    j = np.argsort(ieee80211_permutation(192, 4))         # k -> j
    assert np.all(j // 2 == i // 2)                         # s = 2 groups


def test_identify_ieee80211_with_offset(rng):
    from dhwani.interleaving.interleavers import ieee80211_permutation
    inter = pn_interleave(_coded(rng), ieee80211_permutation(192, 4))[37:]
    top = identify_interleaver(inter, max_L=256)[0]
    assert top.kind == "ieee80211"
    assert (top.parameters["ncbps"], top.parameters["nbpsc"]) == (192, 4)
