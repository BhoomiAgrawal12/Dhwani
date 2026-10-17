import numpy as np

from dhwani.framing.ax25 import decode_ax25
from dhwani.framing.crc import crc_compute


def _addr(call, ssid, last):
    field = [ord(c) << 1 for c in call.ljust(6)]
    field.append(0x60 | (ssid << 1) | int(last))
    return bytes(field)


def _frame(info):
    body = (_addr("CQ", 0, False) + _addr("EA4ABC", 1, True)
            + bytes([0x03, 0xF0]) + info)
    crc = crc_compute(body, 16, 0x1021, 0xFFFF, True, True, 0xFFFF)
    return body + crc.to_bytes(2, "little")


def _hdlc_bits(frames):
    flag = [0, 1, 1, 1, 1, 1, 1, 0]
    out = flag * 4
    for fr in frames:
        run = 0
        for byte in fr:
            for k in range(8):                  # LSB first
                b = (byte >> k) & 1
                out.append(b)
                run = run + 1 if b else 0
                if run == 5:
                    out.append(0)
                    run = 0
        out += flag * 2
    return np.array(out, np.uint8)


def _nrzi_encode(bits):
    out, level = [], 0
    for b in bits:
        level = level if b else 1 - level       # 0 flips, 1 holds
        out.append(level)
    return np.array(out, np.uint8)


def _scramble(bits):
    out = []
    for n, b in enumerate(bits):
        y = int(b)
        if n >= 12:
            y ^= out[n - 12]
        if n >= 17:
            y ^= out[n - 17]
        out.append(y)
    return np.array(out, np.uint8)


def test_ax25_g3ruh_roundtrip():
    frames = [_frame(b"hello \x7e\xff\xff satellite"), _frame(b"second frame" * 3)]
    tx = _scramble(_nrzi_encode(_hdlc_bits(frames)))
    for stream in (tx, 1 - tx):                 # either polarity
        found = decode_ax25(stream.astype(np.uint8))
        assert [f["data"] for f in found] == frames
        assert found[0]["callsigns"] == ["CQ-0", "EA4ABC-1"]


def test_engine_frame_path_validates_ax25():
    from dhwani.hypothesis.engine import _frame_and_crc
    from dhwani.common.config import FramingConfig
    frames = [_frame(bytes([i]) * 20) for i in range(6)]
    tx = _scramble(_nrzi_encode(_hdlc_bits(frames)))
    fr, hits, _ = _frame_and_crc(tx, FramingConfig())
    assert hits[0]["name"] == "CRC-16-X25"
    assert hits[0]["pass_fraction"] == 1.0
    assert hits[0]["callsigns"] == ["CQ-0", "EA4ABC-1"]
    assert fr["frame_length"] == len(frames[0]) * 8
