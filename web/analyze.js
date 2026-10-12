/* Dhwani browser analysis engine.
 *
 * Implements the front half of the pipeline entirely client-side so a
 * .wav or raw .iq recording can be analysed without any upload:
 *   S0 ingestion + format sniffing (WAV/RIFF with auxi chunk, raw IQ
 *      dtype/endianness detection)
 *   S1 conditioning (DC removal, clipping, normalisation)
 *   S2 spectrum, waterfall, CFAR-style signal detection
 *   S3 channelisation (frequency shift + FFT brick-wall decimation)
 *   S4 parameter estimation (OBW, M2M4 SNR, CFO, symbol rate, FSK tones)
 *   S5 modulation classification (physical gates + cumulant engine,
 *      same reference table as the Python implementation)
 *
 * The bit layer (S7-S10: descrambling, blind de-interleaving, blind FEC
 * identification, framing, CRC) needs the full search engine and runs in
 * the desktop CLI/GUI; this page exports a partial report JSON that the
 * desktop tool and the report viewer both understand.
 *
 * Works in the browser and under node (for the test harness).
 */
"use strict";

/* ------------------------------------------------------------------ FFT */
function fftRadix2(re, im, inverse) {
  const n = re.length;
  if (n & (n - 1)) throw new Error("fft size must be a power of two");
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inverse ? 2 : -2) * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cwr = 1, cwi = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * cwr - im[i + k + len / 2] * cwi;
        const vi = re[i + k + len / 2] * cwi + im[i + k + len / 2] * cwr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const nwr = cwr * wr - cwi * wi;
        cwi = cwr * wi + cwi * wr; cwr = nwr;
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }
}

function hann(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n);
  return w;
}

function median(arr) {
  const a = Array.from(arr).sort((x, y) => x - y);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : 0.5 * (a[m - 1] + a[m]);
}

function percentile(arr, p) {
  const a = Array.from(arr).sort((x, y) => x - y);
  const idx = Math.min(a.length - 1, Math.max(0, Math.floor(p / 100 * a.length)));
  return a[idx];
}

function medianFilter(arr, size) {
  const half = size >> 1, out = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    const lo = Math.max(0, i - half), hi = Math.min(arr.length, i + half + 1);
    out[i] = median(arr.slice(lo, hi));
  }
  return out;
}

/* ------------------------------------------------------- S0 ingestion */
const DTYPES = [
  { name: "complex64", bytes: 4, read: (dv, i) => dv.getFloat32(i, true) },
  { name: "float32", bytes: 4, read: (dv, i) => dv.getFloat32(i, true) },
  { name: "int16", bytes: 2, read: (dv, i) => dv.getInt16(i, true) / 32768 },
  { name: "int16_be", bytes: 2, read: (dv, i) => dv.getInt16(i, false) / 32768 },
  { name: "int8", bytes: 1, read: (dv, i) => dv.getInt8(i) / 128 },
  { name: "uint8", bytes: 1, read: (dv, i) => (dv.getUint8(i) - 127.5) / 127.5 },
];

function decodeIQ(buffer, dtypeName, maxSamples) {
  const dt = DTYPES.find((d) => d.name === dtypeName);
  const dv = new DataView(buffer);
  const nVals = Math.min(Math.floor(buffer.byteLength / dt.bytes),
                         2 * (maxSamples || 4 << 20));
  const n = nVals >> 1;
  const I = new Float64Array(n), Q = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    I[k] = dt.read(dv, 2 * k * dt.bytes);
    Q[k] = dt.read(dv, (2 * k + 1) * dt.bytes);
  }
  return { I, Q };
}

function histogramScore(vals, name) {
  let sum = 0, sumAbs = 0, maxAbs = 0, finite = 0;
  const n = vals.length;
  for (let i = 0; i < n; i++) {
    const v = vals[i];
    if (Number.isFinite(v)) {
      finite++; sum += v; const a = Math.abs(v);
      sumAbs += a; if (a > maxAbs) maxAbs = a;
    }
  }
  if (finite / n < 0.999 || maxAbs === 0) return 0.0;
  const mean = sum / n;
  let score = 1.0;
  if (name === "uint8") score *= Math.exp(-Math.abs(mean) / 0.3);
  else score *= Math.exp(-Math.abs(mean) / 0.15);
  if (maxAbs > 1e4) score *= 0.05;
  return score;
}

function spectralStructure(I, Q) {
  const n = Math.min(1 << 15, 1 << Math.floor(Math.log2(I.length || 2)));
  if (n < 1024) return 0;
  const re = new Float64Array(n), im = new Float64Array(n);
  const w = hann(n);
  for (let i = 0; i < n; i++) { re[i] = I[i] * w[i]; im[i] = Q[i] * w[i]; }
  fftRadix2(re, im, false);
  const psd = new Float64Array(n);
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    psd[i] = 10 * Math.log10(re[i] * re[i] + im[i] * im[i] + 1e-20);
    if (psd[i] > max) max = psd[i];
  }
  const crest = max - median(psd);
  return Math.max(0, Math.min(1, (crest - 13) / 25));
}

function sniffRawIQ(buffer) {
  const size = buffer.byteLength;
  const results = [];
  for (const dt of DTYPES) {
    if (size % (2 * dt.bytes)) continue;
    const { I, Q } = decodeIQ(buffer, dt.name, 1 << 17);
    const flat = new Float64Array(I.length * 2);
    flat.set(I); flat.set(Q, I.length);
    const h = histogramScore(flat, dt.name);
    if (h <= 0.02) continue;
    const s = spectralStructure(I, Q);
    results.push({
      dtype: dt.name, histogram: +h.toFixed(3), structure: +s.toFixed(3),
      confidence: +(h * (0.35 + 0.65 * s)).toFixed(3),
      explanation: `${dt.name}: histogram ${h.toFixed(2)}, ` +
                   `spectral structure ${s.toFixed(2)}`,
    });
  }
  results.sort((a, b) => b.confidence - a.confidence);
  const total = results.reduce((s, r) => s + r.confidence, 0) || 1;
  for (const r of results) r.confidence = +(r.confidence / total).toFixed(3);
  return results;
}

function parseWav(buffer) {
  const dv = new DataView(buffer);
  const tag = (o) => String.fromCharCode(dv.getUint8(o), dv.getUint8(o + 1),
                                         dv.getUint8(o + 2), dv.getUint8(o + 3));
  if (tag(0) !== "RIFF" || tag(8) !== "WAVE") return null;
  let pos = 12;
  const meta = { format: "wav" };
  let dataOff = null, dataLen = 0;
  while (pos + 8 <= buffer.byteLength) {
    const id = tag(pos), size = dv.getUint32(pos + 4, true);
    if (id === "fmt ") {
      meta.audioFormat = dv.getUint16(pos + 8, true);
      meta.channels = dv.getUint16(pos + 10, true);
      meta.sampleRate = dv.getUint32(pos + 12, true);
      meta.bits = dv.getUint16(pos + 22, true);
    } else if (id === "auxi" && size >= 36) {
      const year = dv.getUint16(pos + 8, true);
      if (year > 1900 && year < 2100) {
        const cf = dv.getInt32(pos + 40, true);
        if (cf > 0 && cf < 40e9) meta.centerFrequency = cf;
      }
    } else if (id === "data") {
      dataOff = pos + 8; dataLen = Math.min(size, buffer.byteLength - dataOff);
    }
    pos += 8 + size + (size & 1);
  }
  if (dataOff === null || !meta.channels) return null;
  const bytesPer = meta.bits / 8;
  const frames = Math.min(Math.floor(dataLen / (bytesPer * meta.channels)),
                          4 << 20);
  const I = new Float64Array(frames), Q = new Float64Array(frames);
  const readSample = (off) => {
    if (meta.audioFormat === 3 && meta.bits === 32) return dv.getFloat32(off, true);
    if (meta.bits === 16) return dv.getInt16(off, true) / 32768;
    if (meta.bits === 8) return (dv.getUint8(off) - 127.5) / 127.5;
    if (meta.bits === 32) return dv.getInt32(off, true) / 2147483648;
    if (meta.bits === 24) {
      const b0 = dv.getUint8(off), b1 = dv.getUint8(off + 1), b2 = dv.getUint8(off + 2);
      let v = b0 | (b1 << 8) | (b2 << 16);
      if (v & 0x800000) v -= 0x1000000;
      return v / 8388608;
    }
    return 0;
  };
  for (let f = 0; f < frames; f++) {
    const off = dataOff + f * bytesPer * meta.channels;
    I[f] = readSample(off);
    Q[f] = meta.channels >= 2 ? readSample(off + bytesPer) : 0;
  }
  meta.isReal = meta.channels < 2;
  meta.nSamples = frames;
  return { meta, I, Q };
}

/* Real input whose band starts at DC (an FM receiver's audio): the lowest
 * 1% of its power lies below fs/100 and fs/400..fs/100 is within 10 dB of
 * the (9-bin smoothed) peak. Same test as is_baseband in ingestion/wav.py
 * (Welch PSD, 4096-point Hann segments, 50% overlap). */
function isBaseband(x) {
  const len = Math.min(x.length, 1 << 20);
  let n = 4096;
  while (n > len) n >>= 1;
  if (n < 16) return false;
  const w = new Float64Array(n), p = new Float64Array(n / 2 + 1);
  for (let k = 0; k < n; k++) w[k] = 0.5 - 0.5 * Math.cos(2 * Math.PI * k / n);
  for (let s = 0; s + n <= len; s += n / 2) {
    let mean = 0;
    for (let k = 0; k < n; k++) mean += x[s + k];
    mean /= n;
    const re = new Float64Array(n), im = new Float64Array(n);
    for (let k = 0; k < n; k++) re[k] = (x[s + k] - mean) * w[k];
    fftRadix2(re, im, false);
    for (let k = 0; k <= n / 2; k++) p[k] += re[k] * re[k] + im[k] * im[k];
  }
  let total = 0;
  for (let k = 0; k <= n / 2; k++) total += p[k];
  let c = 0, k1 = n / 2;
  for (let k = 0; k <= n / 2; k++) {
    c += p[k];
    if (c >= 0.01 * total) { k1 = k; break; }
  }
  if (k1 / n >= 0.01) return false;
  let peak = 0, lo = 0, nlo = 0;
  for (let k = 0; k <= n / 2; k++) {
    let sm = 0;
    for (let j = k - 4; j <= k + 4; j++) if (j >= 0 && j <= n / 2) sm += p[j];
    sm /= 9;
    peak = Math.max(peak, sm);
    if (k / n >= 0.0025 && k / n <= 0.01) { lo += sm; nlo++; }
  }
  return nlo > 0 && lo / nlo > 0.1 * peak;
}

/* Real (mono) input at fs -> complex baseband at fs/2: mix by -fs/4,
 * 63-tap Hamming half-band low-pass, decimate by 2. Same filter as
 * real_to_complex in ingestion/wav.py (scipy firwin(63, 0.5)). */
function realToComplex(x) {
  const taps = 63, c = (taps - 1) / 2, h = new Float64Array(taps);
  let sum = 0;
  for (let k = 0; k < taps; k++) {
    const t = k - c;
    h[k] = (t === 0 ? 0.5 : Math.sin(Math.PI * t / 2) / (Math.PI * t)) *
      (0.54 - 0.46 * Math.cos(2 * Math.PI * k / (taps - 1)));
    sum += h[k];
  }
  for (let k = 0; k < taps; k++) h[k] /= sum;
  const mr = [1, 0, -1, 0], mi = [0, -1, 0, 1];   /* exp(-j*pi*m/2) */
  const n = Math.ceil(x.length / 2);
  const I = new Float64Array(n), Q = new Float64Array(n);
  for (let o = 0; o < n; o++) {
    let sr = 0, si = 0;
    for (let k = 0; k < taps; k++) {
      const m = 2 * o + c - k;
      if (m < 0 || m >= x.length) continue;
      sr += h[k] * x[m] * mr[m & 3];
      si += h[k] * x[m] * mi[m & 3];
    }
    I[o] = sr; Q[o] = si;
  }
  return { I, Q };
}

function ingest(buffer, fileName, overrides) {
  overrides = overrides || {};
  const report = { warnings: [] };
  let I, Q;
  const wav = fileName.toLowerCase().endsWith(".wav") ? parseWav(buffer) : null;
  if (wav) {
    ({ I, Q } = wav);
    report.format = "wav";
    report.datatype = `${wav.meta.bits}-bit pcm/float`;
    report.sampleRate = overrides.sampleRate || wav.meta.sampleRate;
    report.sampleRateSource = overrides.sampleRate ? "user" : "wav_header";
    report.centerFrequency = overrides.centerFrequency ||
      wav.meta.centerFrequency || null;
    report.channels = wav.meta.channels;
    if (wav.meta.isReal && isBaseband(I)) {
      report.warnings.push("single-channel WAV holds baseband audio (its " +
        "band starts at DC, e.g. an FM receiver's output): analysed as a " +
        "real signal at the file rate; FSK, GFSK and GMSK look alike as audio");
    } else if (wav.meta.isReal) {
      ({ I, Q } = realToComplex(I));
      report.sampleRate /= 2;
      report.centerFrequency = (report.centerFrequency || 0) + report.sampleRate / 2;
      report.warnings.push("single-channel WAV: real signal converted to " +
        "complex baseband (mixed by -fs/4, half-band filtered, decimated by " +
        "2); sample rate is fs/2 and the centre frequency is shifted by fs/4");
    }
  } else {
    report.format = "raw_iq";
    const sniff = sniffRawIQ(buffer);
    report.sniff = sniff;
    const chosen = overrides.datatype && overrides.datatype !== "auto"
      ? overrides.datatype
      : (sniff.length ? sniff[0].dtype : null);
    if (!chosen) throw new Error("could not determine the sample format; " +
      "pick a datatype manually");
    report.datatype = chosen;
    report.datatypeSource = overrides.datatype && overrides.datatype !== "auto"
      ? "user" : "sniffed";
    ({ I, Q } = decodeIQ(buffer, chosen));
    report.sampleRate = overrides.sampleRate || null;
    report.sampleRateSource = overrides.sampleRate ? "user" : "unknown";
    if (!report.sampleRate) {
      report.warnings.push("sample rate unknown (headerless raw IQ): all " +
        "frequencies are in normalised units (cycles/sample); enter the " +
        "sample rate to get absolute values");
    }
    report.centerFrequency = overrides.centerFrequency || null;
  }
  report.nSamples = I.length;
  if (report.sampleRate) report.duration = I.length / report.sampleRate;
  return { report, I, Q };
}

/* ---------------------------------------------------- S1 conditioning */
/* Receiver DC offset is in every sample, noise-only stretches included;
 * a zero-CFO burst's own mean (constant frame fields) is not. Measure it on
 * blocks >= 6 dB below the 90th-percentile block power, else fall back to
 * the whole mean (same as _hardware_dc in conditioning/conditioner.py). */
function hardwareDc(I, Q) {
  const B = 1024, nb = Math.floor(I.length / B);
  if (nb >= 4) {
    const pw = new Float64Array(nb);
    for (let b = 0; b < nb; b++) {
      let s = 0;
      for (let i = b * B; i < (b + 1) * B; i++) s += I[i] * I[i] + Q[i] * Q[i];
      pw[b] = s / B;
    }
    const thr = 0.25 * percentile(pw, 90);
    let mi = 0, mq = 0, cnt = 0;
    for (let b = 0; b < nb; b++) {
      if (pw[b] >= thr) continue;
      for (let i = b * B; i < (b + 1) * B; i++) { mi += I[i]; mq += Q[i]; }
      cnt += B;
    }
    if (cnt >= 2 * B) return [mi / cnt, mq / cnt, true];
  }
  let mi = 0, mq = 0;
  for (let i = 0; i < I.length; i++) { mi += I[i]; mq += Q[i]; }
  return [mi / I.length, mq / I.length, false];
}

function conditionIQ(I, Q) {
  const n = I.length;
  const [mi, mq] = hardwareDc(I, Q);
  let peak = 0;
  for (let i = 0; i < n; i++) {
    I[i] -= mi; Q[i] -= mq;
    const a = Math.max(Math.abs(I[i]), Math.abs(Q[i]));
    if (a > peak) peak = a;
  }
  let clipped = 0, power = 0;
  const lvl = 0.985 * peak;
  for (let i = 0; i < n; i++) {
    if (Math.abs(I[i]) >= lvl || Math.abs(Q[i]) >= lvl) clipped++;
    power += I[i] * I[i] + Q[i] * Q[i];
  }
  const rms = Math.sqrt(power / n) || 1;
  for (let i = 0; i < n; i++) { I[i] /= rms; Q[i] /= rms; }
  const rep = {
    dcOffset: [+mi.toFixed(6), +mq.toFixed(6)],
    clippingFraction: +(clipped / n).toFixed(5),
    normalised: true, warnings: [],
  };
  if (rep.clippingFraction > 1e-3) {
    rep.warnings.push(`${(rep.clippingFraction * 100).toFixed(2)}% of samples ` +
      "at full scale: recording appears clipped");
  }
  return rep;
}

/* -------------------------------------------- S2 waterfall + detection */
function computeWaterfall(I, Q, nfftMax) {
  let nfft = nfftMax || 1024;
  while (nfft > 128 && Math.floor(I.length / nfft) < 16) nfft >>= 1;
  const rowsAll = Math.floor(I.length / nfft);
  const stride = Math.max(1, Math.floor(rowsAll / 256));
  const w = hann(nfft);
  const rows = [], rowStart = [];
  for (let r = 0; r < rowsAll; r += stride) {
    const s = r * nfft;
    const re = new Float64Array(nfft), im = new Float64Array(nfft);
    for (let i = 0; i < nfft; i++) { re[i] = I[s + i] * w[i]; im[i] = Q[s + i] * w[i]; }
    fftRadix2(re, im, false);
    const row = new Float64Array(nfft);
    for (let i = 0; i < nfft; i++) {
      const k = (i + nfft / 2) % nfft;    /* fftshift */
      row[i] = 10 * Math.log10(re[k] * re[k] + im[k] * im[k] + 1e-20);
    }
    rows.push(row);
    rowStart.push(s);
  }
  const freqs = new Float64Array(nfft);
  for (let i = 0; i < nfft; i++) freqs[i] = (i - nfft / 2) / nfft;
  return { rows, rowStart, freqs, nfft, hop: nfft * stride };
}

function detectSignals(wf, thresholdDb) {
  const nRows = wf.rows.length, nfft = wf.nfft;
  if (!nRows) return [];
  const med = new Float64Array(nfft);
  const col = new Float64Array(nRows);
  for (let c = 0; c < nfft; c++) {
    for (let r = 0; r < nRows; r++) col[r] = wf.rows[r][c];
    med[c] = median(col);
  }
  const smooth = medianFilter(med, 33);
  const globalFloor = percentile(smooth, 25);
  const floor = smooth.map((v) => Math.min(v, globalFloor + 1));
  const mask = [];
  for (let r = 0; r < nRows; r++) {
    const m = new Uint8Array(nfft);
    for (let c = 0; c < nfft; c++) m[c] = wf.rows[r][c] > floor[c] + thresholdDb ? 1 : 0;
    mask.push(m);
  }
  /* connected components (4-neighbour BFS) */
  const labels = mask.map(() => new Int32Array(nfft).fill(-1));
  const segs = [];
  for (let r = 0; r < nRows; r++) {
    for (let c = 0; c < nfft; c++) {
      if (!mask[r][c] || labels[r][c] >= 0) continue;
      const id = segs.length;
      let cLo = c, cHi = c, rLo = r, rHi = r, count = 0, powSum = 0;
      const stack = [[r, c]];
      labels[r][c] = id;
      while (stack.length) {
        const [rr, cc] = stack.pop();
        count++;
        powSum += Math.pow(10, wf.rows[rr][cc] / 10);
        if (cc < cLo) cLo = cc; if (cc > cHi) cHi = cc;
        if (rr < rLo) rLo = rr; if (rr > rHi) rHi = rr;
        const nb = [[rr - 1, cc], [rr + 1, cc], [rr, cc - 1], [rr, cc + 1],
                    [rr, cc - 2], [rr, cc + 2]];
        for (const [r2, c2] of nb) {
          if (r2 >= 0 && r2 < nRows && c2 >= 0 && c2 < nfft &&
              mask[r2][c2] && labels[r2][c2] < 0) {
            labels[r2][c2] = id;
            stack.push([r2, c2]);
          }
        }
      }
      segs.push({ cLo, cHi, rLo, rHi, count, powSum });
    }
  }
  const out = [];
  let zMax = -Infinity;
  for (const s of segs) {
    const nF = s.cHi - s.cLo + 1, nT = s.rHi - s.rLo + 1;
    if (nF < 3 || nT < 2) continue;
    let nfLin = 0;
    for (let c = s.cLo; c <= s.cHi; c++) nfLin += Math.pow(10, floor[c] / 10);
    nfLin /= nF;
    const sigLin = s.powSum / s.count;
    const snr = 10 * Math.log10(Math.max(sigLin - nfLin, 1e-20) / nfLin);
    if (snr < 1) continue;
    /* significance as in cfar.py: excess over the noise mean (median/ln2) in std errors */
    zMax = Math.max(zMax, (sigLin * Math.LN2 / nfLin - 1) * Math.sqrt(s.count));
    out.push({
      fLow: wf.freqs[s.cLo], fHigh: wf.freqs[Math.min(s.cHi + 1, wf.nfft - 1)],
      center: 0.5 * (wf.freqs[s.cLo] + wf.freqs[Math.min(s.cHi + 1, wf.nfft - 1)]),
      bandwidth: wf.freqs[Math.min(s.cHi + 1, wf.nfft - 1)] - wf.freqs[s.cLo],
      startSample: wf.rowStart[s.rLo],
      endSample: Math.min(wf.rowStart[Math.min(s.rHi, wf.rowStart.length - 1)] + wf.nfft,
                          wf.rowStart[wf.rowStart.length - 1] + wf.nfft),
      snrDb: +snr.toFixed(1),
      confidence: +Math.min(1, snr / 20).toFixed(2),
    });
  }
  if (zMax < 25) return [];   /* only noise clumps: no signal */
  out.sort((a, b) => b.snrDb - a.snrDb);
  return out.slice(0, 32).map((s, i) => ({ id: i, ...s }));
}

/* ------------------------------------------------- S3 channelisation */
function nextPow2Floor(n) { return 1 << Math.floor(Math.log2(n)); }

function channelize(I, Q, seg) {
  const s0 = seg.startSample, s1 = Math.min(seg.endSample, I.length);
  let n = nextPow2Floor(Math.min(s1 - s0, 1 << 19));
  const re = new Float64Array(n), im = new Float64Array(n);
  const f = seg.center;
  for (let i = 0; i < n; i++) {
    const ph = -2 * Math.PI * f * i;
    const c = Math.cos(ph), s = Math.sin(ph);
    const a = I[s0 + i], b = Q[s0 + i];
    re[i] = a * c - b * s;
    im[i] = a * s + b * c;
  }
  /* FFT brick-wall low-pass at 0.75 x bandwidth, then decimate */
  const bw = Math.max(seg.bandwidth, 4 / n);
  fftRadix2(re, im, false);
  const cut = Math.min(0.45, bw * 0.75);
  for (let i = 0; i < n; i++) {
    const fr = i < n / 2 ? i / n : (i - n) / n;
    if (Math.abs(fr) > cut) { re[i] = 0; im[i] = 0; }
  }
  fftRadix2(re, im, true);
  let decim = 1;
  while (bw * decim * 8 < 0.9 && (n / decim) > 4096) decim <<= 1;
  const m = Math.floor(n / decim);
  const oI = new Float64Array(m), oQ = new Float64Array(m);
  for (let i = 0; i < m; i++) { oI[i] = re[i * decim]; oQ[i] = im[i * decim]; }
  /* burst trim by envelope */
  const win = Math.max(64, m >> 9);
  const env = new Float64Array(m);
  let acc = 0;
  for (let i = 0; i < m; i++) {
    acc += oI[i] * oI[i] + oQ[i] * oQ[i];
    if (i >= win) acc -= oI[i - win] * oI[i - win] + oQ[i - win] * oQ[i - win];
    env[i] = acc / Math.min(i + 1, win);
  }
  const hi = percentile(env.slice(win), 60);
  let a = 0, b = m;
  while (a < m && env[a] < 0.25 * hi) a++;
  while (b > a && env[b - 1] < 0.25 * hi) b--;
  a = Math.max(0, a - 32); b = Math.min(m, b + 32);
  return { I: oI.slice(a, b), Q: oQ.slice(a, b), decim,
           rateRatio: 1 / decim, freqShift: f };
}

/* --------------------------------------------- S4 parameter estimation */
function powerSpectrum(I, Q, nfft) {
  const n = Math.min(nfft || 8192, nextPow2Floor(I.length));
  const segs = Math.max(1, Math.floor(I.length / n));
  const acc = new Float64Array(n);
  const w = hann(n);
  for (let s = 0; s < Math.min(segs, 16); s++) {
    const re = new Float64Array(n), im = new Float64Array(n);
    const off = s * n;
    for (let i = 0; i < n; i++) { re[i] = I[off + i] * w[i]; im[i] = Q[off + i] * w[i]; }
    fftRadix2(re, im, false);
    for (let i = 0; i < n; i++) acc[i] += re[i] * re[i] + im[i] * im[i];
  }
  const psd = new Float64Array(n), freqs = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const k = (i + n / 2) % n;
    psd[i] = acc[k];
    freqs[i] = (i - n / 2) / n;
  }
  return { psd, freqs, n };
}

function occupiedBandwidth(I, Q) {
  const { psd, freqs, n } = powerSpectrum(I, Q);
  const floor = median(psd);
  const clean = psd.map((v) => Math.max(v - floor, 0));
  const total = clean.reduce((a, b) => a + b, 0);
  if (total <= 0) return { obw99: null };
  let cum = 0, lo = freqs[0], hi = freqs[n - 1];
  for (let i = 0; i < n; i++) {
    cum += clean[i];
    if (cum / total >= 0.005) { lo = freqs[i]; break; }
  }
  cum = 0;
  for (let i = n - 1; i >= 0; i--) {
    cum += clean[i];
    if (cum / total >= 0.005) { hi = freqs[i]; break; }
  }
  let centroid = 0;
  for (let i = 0; i < n; i++) centroid += freqs[i] * clean[i];
  return { obw99: +(hi - lo).toFixed(5), centroid: centroid / total };
}

function m2m4Snr(I, Q) {
  const n = I.length;
  let m2 = 0, m4 = 0;
  for (let i = 0; i < n; i++) {
    const p = I[i] * I[i] + Q[i] * Q[i];
    m2 += p; m4 += p * p;
  }
  m2 /= n; m4 /= n;
  const inner = 2 * m2 * m2 - m4;
  if (inner <= 0) return -5;
  const s = Math.sqrt(inner), nn = m2 - s;
  if (nn <= 0) return 40;
  return +(10 * Math.log10(s / nn)).toFixed(1);
}

function coarseCfo(I, Q, order, limiter) {
  const n = Math.min(1 << 16, nextPow2Floor(I.length));
  let re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let a = I[i], b = Q[i];
    if (limiter) {
      const m = Math.hypot(a, b) + 1e-12;
      a /= m; b /= m;
    }
    /* raise to the given power */
    let xr = a, xi = b;
    for (let k = 1; k < order; k++) {
      const nr = xr * a - xi * b;
      xi = xr * b + xi * a; xr = nr;
    }
    re[i] = xr; im[i] = xi;
  }
  const w = hann(n);
  for (let i = 0; i < n; i++) { re[i] *= w[i]; im[i] *= w[i]; }
  fftRadix2(re, im, false);
  let best = 0, bestK = 0;
  const mags = new Float64Array(n);
  for (let i = 1; i < n; i++) {
    mags[i] = Math.hypot(re[i], im[i]);
    if (mags[i] > best) { best = mags[i]; bestK = i; }
  }
  const lineSnr = best / (median(mags) + 1e-12);
  const f = bestK < n / 2 ? bestK / n : (bestK - n) / n;
  return { cfo: f / order, lineSnr };
}

function symbolRate(I, Q, obw99) {
  const n = Math.min(1 << 17, nextPow2Floor(I.length));
  const acc = new Float64Array(n);
  for (const d of [0, 1, 2, 4]) {
    const re = new Float64Array(n), im = new Float64Array(n);
    let mr = 0, mi2 = 0;
    for (let i = 0; i < n - d; i++) {
      if (d === 0) {
        re[i] = I[i] * I[i] + Q[i] * Q[i]; im[i] = 0;
      } else {
        re[i] = I[i + d] * I[i] + Q[i + d] * Q[i];
        im[i] = Q[i + d] * I[i] - I[i + d] * Q[i];
      }
      mr += re[i]; mi2 += im[i];
    }
    mr /= n; mi2 /= n;
    const w = hann(n);
    for (let i = 0; i < n; i++) { re[i] = (re[i] - mr) * w[i]; im[i] = (im[i] - mi2) * w[i]; }
    fftRadix2(re, im, false);
    for (let i = 0; i < n; i++) acc[i] += Math.hypot(re[i], im[i]);
  }
  const minRate = Math.max(1e-4, obw99 ? 0.25 * obw99 : 1e-4);
  const cands = [];
  const bg = medianFilter(acc.slice(0, n >> 1), 51);
  for (let i = 1; i < n / 2; i++) {
    const f = i / n;
    if (f < minRate || f >= 0.5) continue;
    const prom = acc[i] / (bg[i] + 1e-12);
    cands.push({ f, prom });
  }
  cands.sort((a, b) => b.prom - a.prom);
  const kept = [];
  for (const c of cands) {
    if (kept.length >= 5) break;
    if (kept.some((k) => [2, 3, 4].some((m) => Math.abs(c.f - m * k.f) < 2 / n))) continue;
    kept.push(c);
  }
  if (!kept.length) return { rate: null, confidence: 0 };
  return {
    rate: +kept[0].f.toFixed(6),
    confidence: +Math.min(1, Math.max(0, (kept[0].prom - 3) / 20)).toFixed(2),
    candidates: kept.map((k) => ({ rate: +k.f.toFixed(6), prominence: +k.prom.toFixed(1) })),
  };
}

function envelopeCv(I, Q) {
  const n = I.length;
  let s = 0, s2 = 0;
  for (let i = 0; i < n; i++) {
    const a = Math.hypot(I[i], Q[i]);
    s += a; s2 += a * a;
  }
  const mean = s / n;
  return Math.sqrt(Math.max(0, s2 / n - mean * mean)) / (mean + 1e-12);
}

function fskTones(I, Q) {
  const cv = envelopeCv(I, Q);
  if (cv > 0.22) return { isFsk: false, cv: +cv.toFixed(3) };
  const n = Math.min(I.length, 1 << 17);
  const inst = new Float64Array(n - 1);
  let prev = Math.atan2(Q[0], I[0]);
  for (let i = 1; i < n; i++) {
    const ph = Math.atan2(Q[i], I[i]);
    let d = ph - prev;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    inst[i - 1] = d / (2 * Math.PI);
    prev = ph;
  }
  let best = null;
  for (const w of [2, 4, 8, 16, 32]) {
    const sm = new Float64Array(inst.length - w);
    let acc = 0;
    for (let i = 0; i < inst.length; i++) {
      acc += inst[i];
      if (i >= w) {
        acc -= inst[i - w];
        sm[i - w] = acc / w;
      }
    }
    const lo0 = percentile(sm, 1), hi0 = percentile(sm, 99);
    const span = hi0 - lo0, lo = lo0 - 0.15 * span, hi = hi0 + 0.15 * span;
    const bins = 256;
    const hist = new Float64Array(bins);
    for (const v of sm) {
      const b = Math.floor((v - lo) / (hi - lo) * bins);
      if (b >= 0 && b < bins) hist[b]++;
    }
    const hs = medianFilter(hist, 5);
    const hMax = Math.max(...hs);
    const peaks = [];
    for (let i = 1; i < bins - 1; i++) {
      if (hs[i] >= 0.25 * hMax && hs[i] >= hs[i - 1] && hs[i] >= hs[i + 1]) {
        if (!peaks.length || i - peaks[peaks.length - 1] >= 8) peaks.push(i);
        else if (hs[i] > hs[peaks[peaks.length - 1]]) peaks[peaks.length - 1] = i;
      }
    }
    /* rank only windows resolving a valid tone count: windows longer than
     * a symbol merge neighbours into spurious peaks (same rule as fsk_tones
     * in params/estimators.py) */
    if (![2, 4, 8].includes(peaks.length)) continue;
    const width = Math.max(2, Math.floor(bins / (4 * peaks.length)));
    const inPeak = new Uint8Array(bins);
    for (const p of peaks) {
      for (let i = Math.max(0, p - width); i < Math.min(bins, p + width); i++) inPeak[i] = 1;
    }
    let mass = 0, total = 0;
    for (let i = 0; i < bins; i++) { total += hs[i]; if (inPeak[i]) mass += hs[i]; }
    const frac = mass / (total || 1);
    const centers = peaks.map((p) => lo + (p + 0.5) / bins * (hi - lo));
    if (!best || frac > best.frac) best = { nTones: peaks.length, frac, centers, w };
  }
  if (!best || best.frac < 0.5) {
    return { isFsk: false, cv: +cv.toFixed(3),
             note: "constant envelope but no clean tone structure" };
  }
  const sorted = best.centers.slice().sort((a, b) => a - b);
  const diffs = sorted.slice(1).map((v, i) => v - sorted[i]);
  return {
    isFsk: true, cv: +cv.toFixed(3), toneCount: best.nTones,
    toneFreqs: sorted.map((v) => +v.toFixed(5)),
    deviation: diffs.length ? +(diffs.reduce((a, b) => a + b, 0) / diffs.length).toFixed(5) : 0,
    massFraction: +best.frac.toFixed(3),
  };
}

/* -------------------------------------- S5 modulation classification */
/* Reference feature vectors exported from the Python cumulant engine so
 * both implementations agree exactly: [|C20|, |C40|, |C42|, m63]. */
const CUMULANT_REFERENCE = {
  "BPSK": [1.0, 2.0, 2.0, 1.0],
  "QPSK": [0.0, 1.0, 1.0, 1.0],
  "8PSK": [0.0, 0.0, 1.0, 1.0],
  "16QAM": [0.0, 0.68, 0.68, 1.96],
  "64QAM": [0.0, 0.619, 0.619, 2.2258],
};
const CUMULANT_WEIGHTS = [3.0, 2.0, 2.0, 1.0];

function cumulantFeatures(sI, sQ) {
  const n = sI.length;
  let p = 0;
  for (let i = 0; i < n; i++) p += sI[i] * sI[i] + sQ[i] * sQ[i];
  const scale = Math.sqrt(p / n) || 1;
  let m20r = 0, m20i = 0, m21 = 0, m40r = 0, m40i = 0, m42 = 0, m63 = 0;
  for (let i = 0; i < n; i++) {
    const a = sI[i] / scale, b = sQ[i] / scale;
    const p2 = a * a + b * b;
    m21 += p2; m42 += p2 * p2; m63 += p2 * p2 * p2;
    m20r += a * a - b * b; m20i += 2 * a * b;
    /* s^4 */
    const r2 = a * a - b * b, i2 = 2 * a * b;
    m40r += r2 * r2 - i2 * i2; m40i += 2 * r2 * i2;
  }
  m20r /= n; m20i /= n; m21 /= n; m40r /= n; m40i /= n; m42 /= n; m63 /= n;
  const c20 = Math.hypot(m20r, m20i) / m21;
  const c40r = m40r - 3 * (m20r * m20r - m20i * m20i);
  const c40i = m40i - 6 * m20r * m20i;
  const c40 = Math.hypot(c40r, c40i) / (m21 * m21);
  const c42 = Math.abs(m42 - (m20r * m20r + m20i * m20i) - 2 * m21 * m21) / (m21 * m21);
  return [c20, c40, c42, m63 / (m21 * m21 * m21)];
}

function classifyCumulants(sI, sQ, snrDb) {
  const f = cumulantFeatures(sI, sQ);
  let temp = 0.05;
  if (snrDb !== null && snrDb < 15) temp *= 1 + (15 - snrDb) / 5;
  const logits = {};
  let max = -Infinity;
  for (const [name, ref] of Object.entries(CUMULANT_REFERENCE)) {
    let d = 0;
    for (let i = 0; i < 4; i++) {
      const e = (f[i] - ref[i]) * CUMULANT_WEIGHTS[i];
      d += e * e;
    }
    logits[name] = -d / temp;
    if (logits[name] > max) max = logits[name];
  }
  let z = 0;
  const probs = {};
  for (const k in logits) { probs[k] = Math.exp(logits[k] - max); z += probs[k]; }
  for (const k in probs) probs[k] = +(probs[k] / z).toFixed(4);
  return {
    probabilities: Object.fromEntries(
      Object.entries(probs).sort((a, b) => b[1] - a[1])),
    features: f.map((x) => +x.toFixed(4)),
    featureNames: ["|C20|", "|C40|", "|C42|", "m63"],
  };
}

function recoverSymbols(I, Q, rateNorm) {
  /* CFO removal via x^4 line (no limiter so QAM works), then integer
   * decimation at the phase with the strongest 4th-power concentration.
   * A browser-grade preview of the S6 receiver, good enough to feed the
   * cumulant engine and the constellation display. */
  const { cfo } = coarseCfo(I, Q, 4, false);
  const n = I.length;
  const rI = new Float64Array(n), rQ = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const ph = -2 * Math.PI * cfo * i;
    const c = Math.cos(ph), s = Math.sin(ph);
    rI[i] = I[i] * c - Q[i] * s;
    rQ[i] = I[i] * s + Q[i] * c;
  }
  const sps = Math.max(2, Math.round(1 / rateNorm));
  let bestPhase = 0, bestScore = -1, best = null;
  for (let p = 0; p < sps; p++) {
    const m = Math.floor((n - p) / sps);
    const sI = new Float64Array(m), sQ = new Float64Array(m);
    let c4r = 0, c4i = 0, e = 0;
    for (let k = 0; k < m; k++) {
      const a = rI[p + k * sps], b = rQ[p + k * sps];
      sI[k] = a; sQ[k] = b;
      const mag = Math.hypot(a, b) + 1e-12;
      const na = a / mag, nb = b / mag;
      const r2 = na * na - nb * nb, i2 = 2 * na * nb;
      c4r += r2 * r2 - i2 * i2; c4i += 2 * r2 * i2;
      e += a * a + b * b;
    }
    const score = Math.hypot(c4r, c4i) / m + e / m;
    if (score > bestScore) { bestScore = score; bestPhase = p; best = { sI, sQ }; }
  }
  /* rotate by the 4th-power angle so square constellations sit upright */
  let c4r = 0, c4i = 0;
  const m = best.sI.length;
  for (let k = 0; k < Math.min(m, 4096); k++) {
    const a = best.sI[k], b = best.sQ[k];
    const r2 = a * a - b * b, i2 = 2 * a * b;
    c4r += r2 * r2 - i2 * i2; c4i += 2 * r2 * i2;
  }
  const rot = -(Math.atan2(c4i, c4r) - Math.PI) / 4;
  const cr = Math.cos(rot), sr = Math.sin(rot);
  let p2 = 0;
  for (let k = 0; k < m; k++) {
    const a = best.sI[k], b = best.sQ[k];
    best.sI[k] = a * cr - b * sr;
    best.sQ[k] = a * sr + b * cr;
    p2 += best.sI[k] * best.sI[k] + best.sQ[k] * best.sQ[k];
  }
  const rms = Math.sqrt(p2 / m) || 1;
  for (let k = 0; k < m; k++) { best.sI[k] /= rms; best.sQ[k] /= rms; }
  return { sI: best.sI, sQ: best.sQ, sps, phase: bestPhase, cfo };
}

function classifyModulation(I, Q, params) {
  const constraints = [];
  if (params.fsk && params.fsk.isFsk) {
    const label = `${params.fsk.toneCount}FSK`;
    constraints.push(`instantaneous-frequency histogram shows ` +
      `${params.fsk.toneCount} tones (constant envelope, cv ${params.fsk.cv})`);
    return {
      prediction: label,
      confidence: Math.min(1, params.fsk.massFraction),
      alternatives: [[label, Math.min(1, params.fsk.massFraction)]],
      engines: { fsk_histogram: [label, params.fsk.massFraction] },
      constraints,
    };
  }
  if (!params.symbolRate || !params.symbolRate.rate) {
    return { prediction: "UNKNOWN", confidence: 0, alternatives: [],
             engines: {}, constraints: ["no symbol rate found: cannot form " +
             "symbol-spaced samples for the cumulant engine"] };
  }
  const sym = recoverSymbols(I, Q, params.symbolRate.rate);
  const cum = classifyCumulants(sym.sI, sym.sQ, params.snrDb);
  const cv = envelopeCv(I, Q);
  const probs = { ...cum.probabilities };
  if (cv < 0.15) {
    for (const k in probs) if (k.endsWith("QAM")) probs[k] *= 0.05;
    constraints.push(`envelope cv ${cv.toFixed(2)}: QAM excluded ` +
      "(constant-envelope signal)");
  }
  const ranked = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  const [top, topP] = ranked[0];
  return {
    prediction: topP > 0.25 ? top : "UNKNOWN",
    confidence: +(topP > 0.25 ? topP : 0).toFixed(3),
    alternatives: ranked.filter(([, p]) => p > 0.01).slice(0, 5)
      .map(([l, p]) => [l, +p.toFixed(3)]),
    engines: { cumulant: [ranked[0][0], +ranked[0][1].toFixed(3)],
               cumulant_features: Object.fromEntries(
                 cum.featureNames.map((n, i) => [n, cum.features[i]])) },
    constraints,
    symbols: sym,
  };
}

/* ------------------------------------------------------- orchestrator */
const STAGE_TITLES = [
  ["S0", "Ingestion & format detection"],
  ["S1", "Conditioning"],
  ["S2", "Signal detection (CFAR)"],
  ["S3", "Channelisation"],
  ["S4", "Parameter estimation"],
  ["S5", "Modulation classification"],
  ["S6", "Demodulation"],
  ["S7", "Ambiguity fan-out & descrambling"],
  ["S8", "Blind de-interleaving"],
  ["S9", "Blind FEC identification"],
  ["S10", "Framing & CRC"],
  ["S11", "Payload intelligence"],
  ["S12", "Reporting & export"],
];

function makeTrace() {
  const entries = {};
  return {
    mark(stage, status, summary, elapsed) {
      entries[stage] = { status, summary, elapsed_s: elapsed ?? null };
    },
    finalize(stopReason) {
      return STAGE_TITLES.map(([sid, title]) => {
        const e = entries[sid] || { status: "skipped",
          summary: stopReason ||
            "requires the desktop engine: dhwani analyze <file>",
          elapsed_s: null };
        return { stage: sid, title, ...e };
      });
    },
  };
}

function analyzeRecording(buffer, fileName, overrides, progressCb) {
  const prog = progressCb || (() => {});
  const trace = makeTrace();
  const tick = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
  let t0 = tick();
  prog("S0 ingesting and sniffing the format", 0.05);
  const { report, I, Q } = ingest(buffer, fileName, overrides);
  trace.mark("S0", "executed",
    `${report.format} / ${report.datatype || "n/a"}, ` +
    `${report.nSamples.toLocaleString()} samples, sample rate ` +
    (report.sampleRate ? report.sampleRate.toLocaleString() + " Hz" : "unknown"),
    +((tick() - t0) / 1000).toFixed(3));

  t0 = tick();
  prog("S1 conditioning (DC, clipping, normalisation)", 0.2);
  const conditioning = conditionIQ(I, Q);
  trace.mark("S1", "executed",
    `DC removed, clipping ${(conditioning.clippingFraction * 100).toFixed(2)}%, ` +
    "normalised to unit RMS", +((tick() - t0) / 1000).toFixed(3));

  t0 = tick();
  prog("S2 waterfall and CFAR signal detection", 0.35);
  const wf = computeWaterfall(I, Q, 1024);
  const signals = detectSignals(wf, 8.0);
  trace.mark("S2", "executed",
    `${signals.length} signal(s) above threshold` +
    (signals.length ? `; best SNR ${signals[0].snrDb} dB` : ""),
    +((tick() - t0) / 1000).toFixed(3));

  const result = {
    generator: "Dhwani web (S0-S5); run the desktop CLI/GUI for the " +
               "bit layer (S7-S10: descrambling, de-interleaving, FEC, CRC)",
    recording: report,
    conditioning,
    signals,
    plots: {
      waterfall: {
        db: wf.rows.map((r) => {
          const step = Math.max(1, Math.floor(r.length / 512));
          const o = [];
          for (let i = 0; i < r.length; i += step) o.push(+r[i].toFixed(1));
          return o;
        }),
      },
    },
    warnings: [...report.warnings, ...conditioning.warnings],
  };
  {
    const { psd, freqs, n } = powerSpectrum(I, Q, 4096);
    const step = Math.max(1, Math.floor(n / 1024));
    result.plots.psd = { freq_norm: [], psd_db: [] };
    for (let i = 0; i < n; i += step) {
      result.plots.psd.freq_norm.push(+freqs[i].toFixed(5));
      result.plots.psd.psd_db.push(+(10 * Math.log10(psd[i] + 1e-20)).toFixed(2));
    }
  }
  if (!signals.length) {
    result.warnings.push("no signals above the detection threshold; " +
      "analysis stopped after S2");
    result.pipeline_trace = trace.finalize("stopped at S2: nothing detected");
    return result;
  }

  const segIdx = Math.min(overrides.signalIndex || 0, signals.length - 1);
  const seg = signals[segIdx];
  result.selectedSignal = segIdx;
  t0 = tick();
  prog("S3 channelising the selected signal", 0.55);
  const ch = channelize(I, Q, seg);
  trace.mark("S3", "executed",
    `signal ${segIdx}: shifted ${seg.center >= 0 ? "+" : ""}${seg.center.toFixed(4)}, ` +
    `${ch.I.length.toLocaleString()} samples after trim`,
    +((tick() - t0) / 1000).toFixed(3));

  t0 = tick();
  prog("S4 estimating parameters", 0.7);
  const obw = occupiedBandwidth(ch.I, ch.Q);
  const params = {
    obw99: obw.obw99,
    snrDb: m2m4Snr(ch.I, ch.Q),
    symbolRate: symbolRate(ch.I, ch.Q, obw.obw99),
    fsk: fskTones(ch.I, ch.Q),
    envelopeCv: +envelopeCv(ch.I, ch.Q).toFixed(3),
  };
  const cfoPsk = coarseCfo(ch.I, ch.Q, 4, true);
  params.cfo = +cfoPsk.cfo.toFixed(6);
  result.parameters = {
    obw99_norm: params.obw99,
    snr_db: params.snrDb,
    carrier_offset_norm: params.cfo,
    symbol_rate_norm: params.symbolRate.rate,
    symbol_rate_confidence: params.symbolRate.confidence,
    samples_per_symbol: params.symbolRate.rate
      ? +(1 / params.symbolRate.rate).toFixed(2) : null,
    envelope_cv: params.envelopeCv,
    fsk: params.fsk.isFsk ? params.fsk : undefined,
  };
  const rateAtInput = ch.rateRatio;
  if (report.sampleRate && params.symbolRate.rate) {
    result.parameters.symbol_rate_hz =
      +(params.symbolRate.rate * rateAtInput * report.sampleRate).toFixed(1);
  }

  trace.mark("S4", "executed",
    `Rs ${params.symbolRate.rate ?? "unknown"} (norm), ` +
    `SNR ${params.snrDb} dB, OBW ${params.obw99 ?? "-"}`,
    +((tick() - t0) / 1000).toFixed(3));

  t0 = tick();
  prog("S5 classifying the modulation", 0.85);
  const mod = classifyModulation(ch.I, ch.Q, params);
  trace.mark("S5", "executed",
    `${mod.prediction} (confidence ${mod.confidence})`,
    +((tick() - t0) / 1000).toFixed(3));
  result.modulation = {
    prediction: mod.prediction,
    confidence: mod.confidence,
    alternatives: mod.alternatives,
    engine_predictions: mod.engines,
    constraints_applied: mod.constraints,
  };
  if (mod.symbols) {
    const m = Math.min(3000, mod.symbols.sI.length);
    result.plots.constellation = {
      i: Array.from(mod.symbols.sI.slice(0, m), (v) => +v.toFixed(4)),
      q: Array.from(mod.symbols.sQ.slice(0, m), (v) => +v.toFixed(4)),
    };
  }
  result.pipeline_trace = trace.finalize();
  prog("done", 1.0);
  return result;
}

/* node export for the test harness */
if (typeof module !== "undefined") {
  module.exports = { analyzeRecording, sniffRawIQ, parseWav, ingest, realToComplex, isBaseband,
                     computeWaterfall, detectSignals, channelize,
                     symbolRate, fskTones, classifyModulation, m2m4Snr,
                     occupiedBandwidth, envelopeCv };
}
