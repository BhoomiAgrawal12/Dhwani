# How the numbers are made

Every number in the README is rendered by `scripts/render_readme.py` from
a `results/*.json` file that a script wrote. Each file starts with a
`provenance` block: UTC time, command, git commit (with a dirty flag),
CPU, platform, Python / numpy / scipy versions and a data label
(`synthetic` or `off-air`).

## Synthetic data

`dhwani.synth.WaveformFactory` builds frames (sync 0xEB90, counter,
constant byte, 6-byte payload, CRC-16/CCITT-FALSE), applies FEC,
interleaver and scrambler, modulates with RRC pulse shaping (rolloff
0.35), then adds CFO, phase offset and AWGN. Seeds are fixed and recorded
in each result's `settings`.

## `results/benchmark.json` (`scripts/benchmark.py`)

Wall-clock seconds of single calls on one machine: wideband detection
(2M samples), symbol-rate estimation, demodulation of about 100k
symbols, a GF(2) rank scan (L = 2..512, 50k bits), FEC identification,
one end-to-end analysis, and with `--big` loading a 1 GB IQ file to its
first waterfall. CVNet is off. The targets column holds the original
design budgets for comparison; it is not a pass/fail gate.

## `results/validation_report.json` (`scripts/validate_matrix.py`)

Every supported modulation at one or two SNRs (`MATRIX` in the script),
8 samples/symbol, 60 frames, CFO 0.004 and phase 0.3 for digital cases,
CVNet off.
Stress axes at the higher SNR (classification + demodulation only): every
digital modulation again at CFO 0, and the FSK family at 4 samples/symbol;
these rows are labelled "(CFO 0)" / "(4 sps)". Per case:

- blind S5 classification, top-1 and top-2 (QPSK/OQPSK and 2FSK/GMSK
  count as equivalent both ways, OOK->BPSK one way; the confusion matrix
  keeps the raw prediction);
- demodulation with the true modulation: status, EVM and a mid-window
  BER aligned against the transmitted bits;
- at the higher SNR, the full blind chain with the modulation pinned:
  frame found, CRC pass, payload trust, and whether the payload is a
  contiguous substring of what was sent.

Verdict: PASS = top-1 correct, BER < 2% and demodulation GOOD; DEGRADED
= BER < 2% but top-1 missed or demodulation DEGRADED; FAIL otherwise
(`FAIL(gated)` when the S6 gate refused the bits).

## `results/bitlayer_report.json` (`scripts/validate_bitlayer.py`)

QPSK with the modulation pinned, PN9 whitening, 80 frames, at 12 and
20 dB, for 5 interleavers (none, block 8x16, helical 8x16 step 3,
convolutional 4x8, pseudo-random period 128) x 5 FEC families (none,
conv K=7, RS(255,223), LDPC(256,128), RS+conv concatenated).
Identification is correct when kind and parameters match (pseudo-random:
the period). `payload_ber` is the bit error rate of the recovered
payload bytes at the best alignment with the transmitted payloads (1.0
when nothing was recovered); `frames_recovered` is the fraction of
payloads recovered.

## `results/offair_report.json` (`scripts/validate_offair.py`)

Licensed real recordings listed in a manifest with the parameters known
for them; per recording, which expected parameters were reproduced.
Labelled off-air. Nothing is reported until such recordings are added.
Each recording also records its `signal_quality`: S6's EVM, the SNR it
implies (-20 log10 EVM), the S6 status and the modulation S6 ran; the
Proof Chart places it by these.

## `results/ber_report.json` (`scripts/benchmark_ber.py`)

Bit error rate against Eb/N0 (0 to 10 dB, 1 dB steps) through Dhwani's
own receiver, for three chains: uncoded QPSK (hard decisions), the
convolutional K=7 (171,133) rate-1/2 code (terminated, soft-decision
Viterbi) and the 802.11n LDPC code n=648 rate 1/2 (the Layered Decoder).
Every frame is: random information bits, encoded, then a burst of a
random preamble (640 symbols, so S6 has settled before the sync word), a
known 64-bit sync word (0x034776C7272895B0), the coded payload (about
2592 QPSK symbols for every chain: 8 LDPC codewords, the convolutional
code's 2592 information bits, or 5184 uncoded bits) and a 64-symbol
random tail; QPSK with RRC pulse shaping (8 samples/symbol, rolloff 0.35,
the factory's filter), carrier offset 0.004 cycles/sample with a random
phase and a random fractional timing offset, AWGN, and 4000 noise-only
samples each side. The noise is set from the burst's measured energy
per symbol so that Es/N0 = Eb/N0 + 10 log10(2 R) exactly, R the
payload's information bits per coded bit.

The receiver is the stage calls `pipeline.py` makes: S1 condition, S2
detect (segment 0), S3 channelize, S4 parameters, S6 demodulate with S4's
samples per symbol and carrier offset; the modulation is pinned to QPSK
(classification is not measured here). Frame sync uses the known sync
word: it is compared at every bit offset of the eight S7 ambiguity
streams (4 rotations x conjugation), and the best match is accepted with
at most 16 of its 64 bits wrong; that stream's bits and LLRs go to the
decoder. So the curve measures demodulation plus decoding, not blind
identification.

A frame is a lock failure, not bit errors, when S2 detects nothing, S6
says FAILED (its quality gate withholds the bits, as in the pipeline) or
the sync word is not found; the report counts each reason. BER is over
the information bits of locked frames only; a point with no errors
reports BER 0 and the 95% upper bound 3/bits. `p_lock` = locked / frames
with its Wilson 95% interval. `theory` is uncoded Gray QPSK,
0.5 erfc(sqrt(Eb/N0)). Each frame has its own seed (seed, chain, Eb/N0
in tenths of a dB, frame), recorded in `settings` with the frame count;
`seconds_per_frame` is the measured cost per chain.

`scripts/plots.py` draws it as `results/charts/proof.svg`: BER (log)
and probability of lock against Eb/N0, with the off-air recordings in the
lock panel at their EVM SNR (Eb/N0 = Es/N0 for the uncoded recordings),
filled when fully correct, ringed when a CRC-valid frame was decoded.
Recordings without an EVM (no signal, FSK, analog) are not placed.

## What is not measured

PR curves (only S5 confidence and CFAR SNR are scored detectors), GUI
feature visibility beyond `tests/test_gui_smoke.py`, and anything off-air.
Analog cases are demodulated with the true modulation supplied, so they
PASS only when blind classification also named them; otherwise they are
DEGRADED with the reason stated.
