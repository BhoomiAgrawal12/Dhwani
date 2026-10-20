"""Configuration: JSON file + dataclass with sensible defaults.

No magic numbers in stage code - everything tunable lives here.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class CFARConfig:
    threshold_db: float = 8.0          # margin over local noise floor
    train_bins: int = 16
    min_bandwidth_bins: int = 3
    min_duration_frames: int = 2
    min_significance: float = 25.0     # below this, every detection is noise
    morph_close_size: int = 3
    nfft: int = 4096
    max_signals: int = 64
    scan_windows: int = 16             # 2^22-sample windows scanned by S1/S2


@dataclass
class ParamEstConfig:
    obw_fraction: float = 0.99
    symbol_rate_min_norm: float = 1e-4
    symbol_rate_candidates: int = 5


@dataclass
class ModulationConfig:
    classes: list = field(default_factory=lambda: [
        "BPSK", "QPSK", "OQPSK", "8PSK", "16PSK", "32PSK",
        "OOK", "4ASK", "8ASK",
        "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
        "16APSK", "32APSK", "64APSK", "128APSK",
        "2FSK", "4FSK", "GMSK"])
    cvnet_enabled: bool = True
    cvnet_checkpoint: str = ""         # local path; resolved at runtime
    cvnet_variant: str = "real"        # "real" (54% val acc) or "complex"
    device: str = "auto"               # auto | cpu | mps | cuda
    frame_size: int = 1024
    max_frames: int = 32
    fusion_weights: dict = field(default_factory=lambda: {
        "cumulant": 0.5, "cvnet": 0.35})


@dataclass
class DemodConfig:
    rrc_rolloff: float = 0.35
    rrc_span_symbols: int = 10
    carrier_loop_bw: float = 0.02
    max_symbols: int = 200000


@dataclass
class BitLayerConfig:
    max_ambiguity_streams: int = 32
    beam_width: int = 8
    rank_scan_max_L: int = 512
    rank_scan_rows_factor: int = 2     # rows = factor*L + 32
    rank_scan_max_bits: int = 200000
    deficiency_significance: float = 3.0
    # no new beam stream is started after this many seconds; the result
    # says so (a stream that never validates otherwise explores every node)
    time_budget_s: float = 90.0


@dataclass
class FECConfig:
    conv_constraint_lengths: list = field(default_factory=lambda: [3, 5, 7, 9])
    conv_exhaustive_max_k: int = 7
    rs_candidates: list = field(default_factory=lambda: [
        {"n": 255, "k": 223, "prim": 0x11d, "fcr": 112, "generator": 11},
        {"n": 255, "k": 223, "prim": 0x11d, "fcr": 1, "generator": 2},
        {"n": 255, "k": 239, "prim": 0x11d, "fcr": 1, "generator": 2},
        {"n": 255, "k": 239, "prim": 0x11d, "fcr": 0, "generator": 2},
        {"n": 255, "k": 223, "prim": 0x11d, "fcr": 0, "generator": 2},
        {"n": 204, "k": 188, "prim": 0x11d, "fcr": 0, "generator": 2},  # DVB
    ])
    ldpc_candidates: list = field(default_factory=lambda: [
        {"n": 128, "k": 64, "seed": 1},
        {"n": 256, "k": 128, "seed": 1},
        {"n": 512, "k": 256, "seed": 1},
    ])
    min_syndrome_zero_rate: float = 0.7
    max_test_bits: int = 100000


@dataclass
class FramingConfig:
    max_frame_bits: int = 8192
    min_frame_bits: int = 16
    crc_candidates: list = field(default_factory=lambda: [
        {"name": "CRC-8", "width": 8, "poly": 0x07, "init": 0x00, "refin": False, "refout": False, "xorout": 0x00},
        {"name": "CRC-8-MAXIM", "width": 8, "poly": 0x31, "init": 0x00, "refin": True, "refout": True, "xorout": 0x00},
        {"name": "CRC-16-CCITT-FALSE", "width": 16, "poly": 0x1021, "init": 0xFFFF, "refin": False, "refout": False, "xorout": 0x0000},
        {"name": "CRC-16-XMODEM", "width": 16, "poly": 0x1021, "init": 0x0000, "refin": False, "refout": False, "xorout": 0x0000},
        {"name": "CRC-16-ARC", "width": 16, "poly": 0x8005, "init": 0x0000, "refin": True, "refout": True, "xorout": 0x0000},
        {"name": "CRC-16-X25", "width": 16, "poly": 0x1021, "init": 0xFFFF, "refin": True, "refout": True, "xorout": 0xFFFF},
        {"name": "CRC-16-MODBUS", "width": 16, "poly": 0x8005, "init": 0xFFFF, "refin": True, "refout": True, "xorout": 0x0000},
        {"name": "CRC-32", "width": 32, "poly": 0x04C11DB7, "init": 0xFFFFFFFF, "refin": True, "refout": True, "xorout": 0xFFFFFFFF},
        {"name": "CRC-32-BZIP2", "width": 32, "poly": 0x04C11DB7, "init": 0xFFFFFFFF, "refin": False, "refout": False, "xorout": 0xFFFFFFFF},
    ])


@dataclass
class PayloadIntelConfig:
    enabled: bool = True
    max_analysis_bytes: int = 1 << 20
    max_decompression_bytes: int = 1 << 20


@dataclass
class Config:
    cfar: CFARConfig = field(default_factory=CFARConfig)
    params: ParamEstConfig = field(default_factory=ParamEstConfig)
    modulation: ModulationConfig = field(default_factory=ModulationConfig)
    demod: DemodConfig = field(default_factory=DemodConfig)
    bitlayer: BitLayerConfig = field(default_factory=BitLayerConfig)
    fec: FECConfig = field(default_factory=FECConfig)
    framing: FramingConfig = field(default_factory=FramingConfig)
    payload_intelligence: PayloadIntelConfig = field(
        default_factory=PayloadIntelConfig)
    cache_dir: str = ""
    signature_db: str = ""
    log_level: str = "INFO"

    def to_dict(self) -> dict:
        return asdict(self)


_SECTION_TYPES = {
    "cfar": CFARConfig, "params": ParamEstConfig, "modulation": ModulationConfig,
    "demod": DemodConfig, "bitlayer": BitLayerConfig, "fec": FECConfig,
    "framing": FramingConfig, "payload_intelligence": PayloadIntelConfig,
}


def load_config(path: Optional[str] = None) -> Config:
    """Load config from JSON, overlaying defaults. Unknown keys are ignored
    with a warning rather than crashing."""
    cfg = Config()
    if path and os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        log = logging.getLogger(__name__)
        for key, val in data.items():
            if key in _SECTION_TYPES and isinstance(val, dict):
                section = getattr(cfg, key)
                for k, v in val.items():
                    if hasattr(section, k):
                        setattr(section, k, v)
                    else:
                        log.warning("config %s: unknown key %s.%s ignored",
                                    path, key, k)
            elif hasattr(cfg, key) and not isinstance(getattr(cfg, key), tuple(_SECTION_TYPES.values())):
                setattr(cfg, key, val)
            else:
                log.warning("config %s: unknown key %s ignored", path, key)
    if not cfg.cache_dir:
        cfg.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "dhwani")
    if not cfg.signature_db:
        cfg.signature_db = os.path.join(cfg.cache_dir, "signatures.sqlite")
    return cfg
