# Algorithms

What each stage computes, in the order the pipeline runs. Module paths are
under `src/dhwani/`.

## S0 ingestion (`ingestion/`)

- **Format sniffing** (`sniffer.py`) for headerless raw IQ: file-size
  divisibility, per-dtype value histograms, endianness plausibility,
  real-vs-complex spectral symmetry. Ranked candidates with a confidence
  and an explanation. Never invents a sample rate.
- **WAV** (`wav.py`): own RIFF chunk walk (the SDRuno / HDSDR /
  SDR-Console `auxi` chunk gives centre frequency and timestamp),
  memory-mapped data, soundfile fallback for 24-bit PCM.
- **Mono WAV** (`wav.real_to_complex`): multiply by exp(-j*pi*n/2) to
  move 0..fs/2 around DC, 63-tap Hamming half-band low-pass
  (`firwin(63, 0.5)`), keep every second sample. The JS engine uses the
  same filter (`realToComplex`, matched to 6e-8). Exception: when the
  lowest 1% of the power lies below fs/100 and fs/400..fs/100 is within
  10 dB of the spectral peak (`wav.is_baseband`, JS `isBaseband`), the
  file is baseband audio, e.g. an FM receiver's output.
  It stays real at the file rate, and S3 does not trim it by power (FM
  quieting makes the hiss louder than the signal).

## S1 conditioning (`conditioning/conditioner.py`)

DC offset measured on 1024-sample blocks at least 6 dB below the
90th-percentile block power (receiver DC is in every sample, a zero-CFO
burst's own mean only inside the burst; whole-file mean with a warning
when no such block exists) and removed when it exceeds 1e-4 of the
magnitude spread; clipping
fraction at 98.5% of peak; dead-air fraction (4096-sample frames below
5% of the median power; reported, not excluded); I/Q gain and phase
imbalance measured from the I/Q cross-correlation. Gram-Schmidt
correction is opt-in because it assumes a circular signal.

## S2 detection (`detection/cfar.py`)

Waterfall (up to 4096 bins); per-bin noise floor = median over time,
median-filtered across frequency and capped at the band's 25th
percentile + 1 dB; threshold at floor + `threshold_db`; morphological
closing/opening; connected components become signal boxes.

## S3-S4 channelisation and parameters

Shift the box centre to DC, polyphase-resample so the occupied bandwidth
is about 1/8 of the new rate, trim leading/trailing noise.
OBW (99% cumulative PSD, ITU-R SM.443), CFO from the x^M spectral line
(or the spectral centroid), SNR by M2M4 (later refined by EVM once S6
locks), symbol rate from a time-smoothed cyclic periodogram with
bandwidth-implied candidates, FSK tone histogram and dwell-rate line,
OFDM cyclic-prefix correlation. For headerless IQ the symbol rate per
recorded sample is matched against standard baud rates at common SDR
sample rates (`sample_rate_candidates`, PROBABLE only).

## S5 modulation (`modulation/`)

Physical pre-checks (FSK tones, OFDM, constant envelope) exclude classes.
The cumulant engine compares rotation-invariant higher-order cumulant
moduli with reference vectors computed exactly from the constellation
tables. Shortlisted candidates are trial-demodulated by the real S6
receiver and a candidate wins only if it locks (EVM against its own
gate, constellation occupancy, residual-bias structure); if none locks,
the result is UNKNOWN. A winning 32/16/8-PSK always has its M/2 sibling
trialled (its points sit on the larger grid) and the lower order wins at
equal EVM. CVNet-RF is advisory; disagreement is surfaced, not averaged
away.

## S6 demodulation (`demod/receiver.py`)

Per-family receivers: coarse CFO removal, RRC matched filter, Gardner
timing recovery, decision-directed PLL (PSK <= 8) or feed-forward V&V
(>= 16), symbol-domain CFO and phase grid for dense QAM, ring-gated V&V
for APSK, discriminator for FSK/GMSK, envelope/coherent/discriminator
for analog. Max-log LLRs. Every loop publishes a lock metric, and the
GOOD / DEGRADED / FAILED gate decides whether bits go on. For symmetry
m >= 8 a symbol-domain carrier estimate in the outer quarter of
+-1/(2m) cycles/symbol caps the gate at DEGRADED (alias false lock).

## S7 ambiguity (`bits/ambiguity.py`)

A blind loop locks to *a* constellation orientation, so every rotation,
I/Q swap and differential interpretation becomes a candidate bit stream
(FSK: normal/inverted x plain/differential).

## S8-S10 bit layer (`hypothesis/engine.py`)

- **GF(2) rank deficiency** (`gf2/rank.py`, after Sicot & Houcke 2009):
  rows of width L are confined to a subspace when L is a multiple of the
  code or interleaver period, so the rank drops; a wrong width is
  essentially full rank. Bit-packed uint64 elimination.
- **Descrambling** (`scrambling/`): each library whitener's phase is
  aligned by maximising the rank deficiency at the detected period (DVB
  over its first 512 phases only); Berlekamp-Massey recovers additive
  LFSRs from constant stretches.
- **Interleavers** (`interleaving/detect.py`): the fundamental
  significant period P; dense structure means not interleaved; otherwise
  block (r x c factorisations of P), helical (step sweep) and
  convolutional (branches x delay) inverses are scored by the
  fine-grained structure they reveal, with a block-phase offset sweep
  (stride 2, stride 1 for helical). The published 802.11a/g bit
  interleaver (N_cbps 48/96/192/288) is tested the same way, also when the
  rank scan is flat (a short capture hides a 192-bit period). No inverse
  helps: `pseudo_random` with its period; other permutations are not
  searched.
- **FEC** (`fec/`): convolutional codes by dual-code syndrome rate over
  K and generator sweeps, then Viterbi with substream-inversion
  resolution; Reed-Solomon by syndrome sweep over bit and symbol
  alignment, decode via reedsolo; LDPC by candidate-set matching and
  bit-flipping decode; RS outer code in the Viterbi output
  (`identify_outer`) for concatenated codes.
- **Framing** (`framing/`): binary autocorrelation and column stability
  give the frame length; the longest low-entropy column run is the sync;
  a bounded CRC hunt over the configured catalogue and positions; byte
  and sync-anchored rotations when the first anchor misses. A frame is
  accepted only on a CRC pass or sync + column stability + enough
  frames.

## S11 payload intelligence (`payload/`)

Byte forensics, conservative text decoding, Base64/hex/URL wrapper
discovery, cross-frame field inference, compression detection with
bounded decompression, encryption assessment (high entropy alone is
never called encryption), protocol fingerprints (JSON, HTTP, IPv4 with
checksum), message reconstruction. The input is graded VALIDATED /
PROBABLE / SPECULATIVE from CRC, FEC and demod provenance, and every
finding is capped accordingly.
