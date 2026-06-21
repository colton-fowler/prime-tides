/** Mulberry32 — deterministic PRNG from a 32-bit session seed. */
export class SeededRng {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  next(): number {
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  nextInt(min: number, max: number): number {
    return Math.floor(this.next() * (max - min + 1)) + min;
  }

  pickPowerOfTwo(minExp: number, maxExp: number): number {
    return 1 << this.nextInt(minExp, maxExp);
  }
}

export function createSessionSeed(): number {
  return crypto.getRandomValues(new Uint32Array(1))[0]!;
}

export interface SessionParams {
  sessionSeed: number;
  primeLimit: number;
  windowSize: number;
  windowStartFraction: number;
  topPeaks: number;
  phaseOffset: number;
  deepRatio: number;
  surfaceRatio: number;
  peakSliceOffset: number;
  sparkStride: number;
  sparkTwinCap: number;
  sparkOtherCap: number;
}

export function deriveSessionParams(seed: number, mobile: boolean): SessionParams {
  const rng = new SeededRng(seed);

  return {
    sessionSeed: seed,
    primeLimit: mobile ? rng.nextInt(90_000, 180_000) : rng.nextInt(300_000, 900_000),
    windowSize: mobile ? rng.pickPowerOfTwo(12, 14) : rng.pickPowerOfTwo(13, 15),
    windowStartFraction: rng.next() * 0.82,
    topPeaks: mobile ? rng.nextInt(6, 10) : rng.nextInt(10, 18),
    phaseOffset: rng.next() * Math.PI * 2,
    deepRatio: 0.26 + rng.next() * 0.2,
    surfaceRatio: 0.34 + rng.next() * 0.24,
    peakSliceOffset: rng.nextInt(0, 6),
    sparkStride: rng.nextInt(1, 3),
    sparkTwinCap: rng.nextInt(14, 30),
    sparkOtherCap: rng.nextInt(28, 52),
  };
}

export function resolveWindowBounds(
  gapCount: number,
  windowSize: number,
  windowStartFraction: number,
): { start: number; size: number } {
  const size = Math.min(windowSize, gapCount);
  const maxStart = Math.max(0, gapCount - size);
  const start = Math.floor(maxStart * windowStartFraction);
  return { start, size };
}
