import numpy as np

from dhwani.fec.conv import ConvCode, conv_encode, viterbi_decode
from dhwani.fec.rs import RSCode
from dhwani.fec.detect import identify_fec


def test_viterbi_clean(rng):
    code = ConvCode(7, (0o171, 0o133))
    msg = rng.integers(0, 2, 2000).astype(np.uint8)
    dec = viterbi_decode(conv_encode(msg, code), code)
    assert np.array_equal(dec[:len(msg)], msg)


def test_viterbi_noisy(rng):
    code = ConvCode(7, (0o171, 0o133))
    msg = rng.integers(0, 2, 4000).astype(np.uint8)
    enc = conv_encode(msg, code)
    enc[rng.random(len(enc)) < 0.05] ^= 1
    ber = (viterbi_decode(enc, code)[:len(msg)] != msg).mean()
    assert ber < 0.01


def test_rs_roundtrip(rng):
    rs = RSCode(255, 223)
    data = bytes(rng.integers(0, 256, 223, dtype=np.uint8))
    cw = bytearray(rs.encode(data))
    cw[5] ^= 0x55
    decoded, _ = rs.decode(bytes(cw))
    assert decoded[:223] == data


def test_rs_shortened_204_188_roundtrip(rng):
    """DVB RS(204,188): a shortened RS(255,239) still lives in GF(2^8)."""
    rs = RSCode(204, 188, fcr=0)
    assert rs.m == 8 and RSCode(255, 223).m == 8
    assert RSCode(15, 11, prim=0x13).m == 4
    data = bytes(rng.integers(0, 256, 188, dtype=np.uint8))
    cw = bytearray(rs.encode(data))
    assert len(cw) == 204
    assert not rs.syndromes(np.frombuffer(bytes(cw), np.uint8)[None, :]).any()
    cw[17] ^= 0xA5
    cw[150] ^= 0x01
    decoded, _ = rs.decode(bytes(cw))
    assert decoded[:188] == data
    # the field comes from the polynomial, also for n < 128
    short = RSCode(120, 104)
    assert short.m == 8
    data = bytes(rng.integers(0, 256, 104, dtype=np.uint8))
    cw = short.encode(data)
    assert not short.syndromes(np.frombuffer(cw, np.uint8)[None, :]).any()
    assert short.decode(cw)[0][:104] == data


def test_identify_conv(config, rng):
    code = ConvCode(7, (0o171, 0o133))
    msg = rng.integers(0, 2, 8000).astype(np.uint8)
    enc = conv_encode(msg, code, terminate=False)
    hyp = identify_fec(enc, config.fec)[0]
    assert hyp.family == "convolutional"
    assert hyp.parameters["K"] == 7
    assert hyp.syndrome_zero_rate > 0.99
    assert np.array_equal(hyp.decoded_bits[:len(msg)], msg)


def test_identify_conv_inverted_substream(config, rng):
    """Rotation ambiguities invert substreams; the constancy test must
    still identify the code and resolve the inversion pattern."""
    code = ConvCode(7, (0o171, 0o133))
    msg = rng.integers(0, 2, 6000).astype(np.uint8)
    enc = conv_encode(msg, code, terminate=False)
    enc[0::2] ^= 1
    hyp = identify_fec(enc, config.fec)[0]
    assert hyp.family == "convolutional"
    assert np.array_equal(hyp.decoded_bits[:len(msg)], msg)


def test_identify_rs(config, rng):
    rs = RSCode(255, 223)
    stream = b"".join(rs.encode(bytes(rng.integers(0, 256, 223,
                                                   dtype=np.uint8)))
                      for _ in range(25))
    bits = np.unpackbits(np.frombuffer(stream, dtype=np.uint8))
    hyp = identify_fec(bits.astype(np.uint8), config.fec)[0]
    assert hyp.family == "reed_solomon"
    assert hyp.parameters["n"] == 255 and hyp.parameters["k"] == 223


def test_identify_none_on_random(config, rng):
    bits = rng.integers(0, 2, 20000).astype(np.uint8)
    assert identify_fec(bits, config.fec)[0].family == "none"
