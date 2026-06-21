import type { WaveComponent } from '../types';

/** In-place Cooley–Tukey radix-2 FFT for real-valued input (zero-padded to power of 2). */
export function fft(real: Float64Array): { real: Float64Array; imag: Float64Array } {
  const n = nextPowerOfTwo(real.length);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  re.set(real);

  bitReversePermute(re, im);

  for (let size = 2; size <= n; size *= 2) {
    const half = size / 2;
    const angle = (-2 * Math.PI) / size;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);

    for (let i = 0; i < n; i += size) {
      let curRe = 1;
      let curIm = 0;

      for (let j = 0; j < half; j++) {
        const evenRe = re[i + j];
        const evenIm = im[i + j];
        const oddRe = re[i + j + half];
        const oddIm = im[i + j + half];

        const tRe = curRe * oddRe - curIm * oddIm;
        const tIm = curRe * oddIm + curIm * oddRe;

        re[i + j] = evenRe + tRe;
        im[i + j] = evenIm + tIm;
        re[i + j + half] = evenRe - tRe;
        im[i + j + half] = evenIm - tIm;

        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }

  return { real: re, imag: im };
}

export function magnitude(re: Float64Array, im: Float64Array, index: number): number {
  return Math.sqrt(re[index] * re[index] + im[index] * im[index]);
}

export function nextPowerOfTwo(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

function bitReversePermute(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
}

/** Sieve of Eratosthenes — returns all primes up to limit. */
export function sievePrimes(limit: number): Uint32Array {
  const isPrime = new Uint8Array(limit + 1);
  isPrime.fill(1);
  isPrime[0] = 0;
  isPrime[1] = 0;

  for (let p = 2; p * p <= limit; p++) {
    if (isPrime[p]) {
      for (let m = p * p; m <= limit; m += p) {
        isPrime[m] = 0;
      }
    }
  }

  let count = 0;
  for (let i = 2; i <= limit; i++) {
    if (isPrime[i]) count++;
  }

  const primes = new Uint32Array(count);
  let idx = 0;
  for (let i = 2; i <= limit; i++) {
    if (isPrime[i]) primes[idx++] = i;
  }
  return primes;
}

export function primeGaps(primes: Uint32Array): Float64Array {
  const gaps = new Float64Array(primes.length - 1);
  for (let i = 1; i < primes.length; i++) {
    gaps[i - 1] = primes[i] - primes[i - 1];
  }
  return gaps;
}

export interface FrequencyPeak {
  index: number;
  frequency: number;
  magnitude: number;
  phase: number;
}

/** Extract top spectral peaks (excluding DC) from FFT output. */
export function extractPeaks(
  re: Float64Array,
  im: Float64Array,
  topN: number,
  minSeparation = 3,
): FrequencyPeak[] {
  const half = re.length >> 1;
  const peaks: FrequencyPeak[] = [];

  for (let i = 1; i < half; i++) {
    const mag = magnitude(re, im, i);
    const prev = magnitude(re, im, i - 1);
    const next = i + 1 < half ? magnitude(re, im, i + 1) : 0;
    if (mag > prev && mag >= next) {
      peaks.push({
        index: i,
        frequency: i / re.length,
        magnitude: mag,
        phase: Math.atan2(im[i], re[i]),
      });
    }
  }

  peaks.sort((a, b) => b.magnitude - a.magnitude);

  const selected: FrequencyPeak[] = [];
  for (const peak of peaks) {
    if (selected.some((s) => Math.abs(s.index - peak.index) < minSeparation)) continue;
    selected.push(peak);
    if (selected.length >= topN) break;
  }

  return selected;
}

export function normalizePeaksToWaves(
  peaks: FrequencyPeak[],
  maxMag: number,
  baseAmplitude: number,
  speedScale: number,
  options?: { phaseOffset?: number; rng?: () => number },
): WaveComponent[] {
  const phaseBase = options?.phaseOffset ?? 0;
  const rnd = options?.rng ?? (() => 0);
  return peaks.map((p, i) => {
    const strength = p.magnitude / maxMag;
    const freqJitter = 0.92 + rnd() * 0.16;
    return {
      frequency: 0.006 + p.frequency * 0.18 * (1 + i * 0.04) * freqJitter,
      amplitude: baseAmplitude * (0.45 + strength * 0.55),
      phase: p.phase + i * 0.7 + phaseBase + rnd() * Math.PI * 2,
      speed: speedScale * (0.5 + strength * 0.5 + p.frequency * 2),
      harmonic: i % 3 === 0 ? 2 : undefined,
      peakStrength: strength,
    };
  });
}

/** Subsample an array for lightweight preview overlays. */
export function subsample(data: Float64Array, targetPoints: number): number[] {
  if (data.length <= targetPoints) return Array.from(data);
  const step = data.length / targetPoints;
  const out: number[] = [];
  for (let i = 0; i < targetPoints; i++) {
    out.push(data[Math.floor(i * step)]!);
  }
  return out;
}

export function buildSpectrumPreview(
  re: Float64Array,
  im: Float64Array,
  targetPoints: number,
): { freq: number; mag: number }[] {
  const half = re.length >> 1;
  const out: { freq: number; mag: number }[] = [];
  const step = Math.max(1, Math.floor(half / targetPoints));
  for (let i = 1; i < half; i += step) {
    out.push({
      freq: i / re.length,
      mag: magnitude(re, im, i),
    });
  }
  return out;
}
