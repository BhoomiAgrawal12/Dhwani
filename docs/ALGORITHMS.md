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
(FSK: normal/inverted x plain/differential). Each carries the S6 LLRs of
its own bits (inverted: negated; differential: box-plus of the pair it
XORs), and the bit layer moves them with the bits through descrambling
and de-interleaving.

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
  searched. First, and on a flat profile too (0.5% BER erases the period,
  RS never shows one), the Pattern Scan (`pattern_scan.py`) looks for the
  code through the interleaver: rate-1/2 registry checks dilated by
  D = 1..72 (a block's rows keep a check's taps R apart), the column count
  and block start from the dilated check folded by R x C against its
  validity template, the period's geometries and the 802.11 modes screened
  at every offset, and an RS prefix-XOR pass over every offset of common
  block sizes (row-aligned bytes). It reports a geometry only with its
  code: an Early Verdict "code" on the de-interleaved stream (from where
  the code starts to hold) with every check of the code under one
  substream-inversion pattern, 7 spreads above the scan's other
  dilations, windows and checks, no tap pairing at a lag where the stream
  repeats itself, the check silent at its other phase (a check that holds
  at both is a recurrence of the data: PRBS or scrambled idle decimated by
  D), no code half as strong on the raw stream (a geometry that is the
  identity there, such as convolutional B=2 M=1, is no interleaver), and
  `identify_convolutional` / `identify_rs` agreeing (the decode re-encoding
  at the BER the checks read). Its work is capped by counts, not time.
  Limits: only rate-1/2 unpunctured codes are reported (rate 1/3 and
  punctured codes veto a claim but have no decoder to confirm them); the
  scan reads hard parities, LLRs only the confirmation (K=7 under 8x16 up
  to ~6% BER at 64k bits); RS only under block geometries with whole bytes
  per row (cols a multiple of 8, GF(2^8)), squares and 2:1 rectangles up
  to 32 x 32 or the period's factor pairs, up to ~5e-4 BER; LDPC under an
  interleaver is not searched.
- **FEC** (`fec/`): the Code Registry (`registry.py`) compiles every
  library code at import into its lightest dual-code checks (exact, from
  the null space of a window's generator matrix) or its RS roots; the
  Code Probe (`probe.py`) scores a check by the mean of prod tanh(L/2)
  over its taps at every alignment and sign (idle runs skipped, many
  checks per matrix product), and the Early Verdict, an SPRT (alpha
  1e-6, beta 1e-3) checked against the whole stream's z, calls it code,
  no code or undecided, with a BER read from the hard parities.
  Convolutional codes: registry, standard and exhaustive K <= 7
  candidates, "no code" dropped before a soft Viterbi decode with
  substream-inversion resolution. A syndrome-zero rate >= 0.7 admits a
  code as before; below it a "code" must stand 7 spreads above the
  sweep's own wrong hypotheses of its check weight (text, repeated frames
  and headers bias wrong checks of uncoded data) and its decode must
  re-encode within sampling error of the BER its checks read; Reed-Solomon
  (DVB RS(204,188) included): every bit offset and codeword alignment
  from one prefix-XOR pass per root, decode via reedsolo; LDPC
  (`ldpc.py`) from a set of known matrices read from `.qc` / `.alist`
  files: bundled 802.11n n = 648 (rate 1/2 and 3/4) and CCSDS AR4JA
  k = 1024 at rate 1/2 (2048 of 2560 bits sent) and rate 4/5 (the
  "n = 1280" AR4JA code: 1280 of 1408 sent), plus the configured seeded
  candidates and any matrix file given by path (`{"path": ...}`). An
  empty `ldpc_candidates` still turns LDPC off; `fec.ldpc_bundled`
  (default `null`: the bundled codes join a non-empty list) set to `true`
  scans the bundled codes even then, `false` never (the listed
  candidates only, as before). The Pre-screen scores every code at every codeword
  alignment and both polarities with its lightest checks (for AR4JA, row
  sums that cancel the punctured bits) over at least 8 codewords, about
  2048 check values per alignment (AR4JA r1/2 reads its 512 lightest
  checks whole, AR4JA r4/5 16 codewords); the Early Verdict may start at
  any of the first four codewords (a preamble before the first whole
  codeword). Only the best "code" that stands 7 spreads above the scan's
  other hypotheses goes on; its alignment is then settled among its
  neighbours, because a window a bit off a quasi-cyclic codeword is the
  codeword with every circulant rotated except at the circulant
  boundaries (one bit off, 97% of AR4JA r1/2's checks still hold and the
  decoder converges onto the rotated codewords): the confirmation over 64
  codewords and every probe check picks the alignment, and for a noisy
  stream the neighbours' decodes do (the rotated codewords also flip
  half of the boundary bits). The decoder is layered offset min-sum
  (offset 0.15, then scale 0.8, at most 30 iterations, punctured bits at
  LLR 0, soft when LLRs are aligned); below the old clean-codeword gate
  it must converge on half of the first 64 codewords. An inverted hit
  (the complement satisfies the code) is left to the bit layer's
  polarity fan-out. Measured through `identify_fec` at an unknown offset
  over 40 kbit (BPSK, 40 seeds; the rest are missed, none reported at a
  wrong alignment): with soft LLRs AR4JA r1/2 37/40 at Eb/N0 2.0 dB,
  AR4JA r4/5 39/40 at 3.5 dB, 802.11n r1/2 40/40 at 2.0 dB; with hard
  bits AR4JA r1/2 39/40 at 3.0 dB; with QPSK S6 LLRs AR4JA r1/2 36/40 at
  2.5 dB and r4/5 40/40 at 4.0 dB. Detection therefore lags the decoder
  by about 0.5 dB on AR4JA (its checks are heavy: weight 11 at rate 1/2,
  33 at 4/5). RS outer code in the Viterbi output (`identify_outer`) for
  concatenated codes.
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
