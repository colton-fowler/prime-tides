import {
  sievePrimes,
  primeGaps,
  fft,
  extractPeaks,
  normalizePeaksToWaves,
  subsample,
  buildSpectrumPreview,
  type FrequencyPeak,
} from '../math/fft';
import {
  deriveSessionParams,
  resolveWindowBounds,
  SeededRng,
} from '../math/sessionSeed';
import type {
  OceanSpectrum,
  PrimeFftWorkerRequest,
  PrimeFftWorkerResponse,
  UsedWavePeak,
  WaveComponent,
} from '../types';

function post(msg: PrimeFftWorkerResponse): void {
  self.postMessage(msg);
}

function buildUsedPeakRecords(
  peaks: FrequencyPeak[],
  waves: WaveComponent[],
  layer: UsedWavePeak['layer'],
  maxMag: number,
): UsedWavePeak[] {
  return peaks.map((p, i) => {
    const w = waves[i]!;
    return {
      frequency: p.frequency,
      magnitude: p.magnitude,
      normalizedAmplitude: p.magnitude / maxMag,
      phase: w.phase,
      speed: w.speed,
      layer,
    };
  });
}

function computeSpectrum(sessionSeed: number, mobile: boolean): OceanSpectrum {
  const params = deriveSessionParams(sessionSeed, mobile);
  const rng = new SeededRng(sessionSeed ^ 0x9e3779b9);

  post({ type: 'progress', stage: 'sieving', percent: 10 });

  const primes = sievePrimes(params.primeLimit);
  post({ type: 'progress', stage: 'gaps', percent: 35 });

  const allGaps = primeGaps(primes);
  const { start: windowStart, size: windowSize } = resolveWindowBounds(
    allGaps.length,
    params.windowSize,
    params.windowStartFraction,
  );
  const windowEnd = windowStart + windowSize;
  const windowGaps = allGaps.subarray(windowStart, windowEnd);

  const mean = windowGaps.reduce((s, g) => s + g, 0) / windowGaps.length;
  const centered = new Float64Array(windowSize);
  for (let i = 0; i < windowSize; i++) {
    centered[i] = windowGaps[i]! - mean;
  }

  post({ type: 'progress', stage: 'fft', percent: 60 });

  const { real, imag } = fft(centered);
  const peaks = extractPeaks(real, imag, params.topPeaks * 2, 4);

  if (peaks.length === 0) {
    throw new Error('No spectral peaks found in prime-gap window');
  }

  const maxMag = peaks[0]!.magnitude;
  post({ type: 'progress', stage: 'waves', percent: 85 });

  const peakPool = peaks.slice(0, Math.min(peaks.length, params.topPeaks * 2));
  const sliceAt = Math.min(params.peakSliceOffset, Math.max(0, peakPool.length - params.topPeaks));
  const activePeaks = peakPool.slice(sliceAt);

  const deepCount = Math.max(2, Math.floor(params.topPeaks * params.deepRatio));
  const surfaceCount = Math.max(3, Math.floor(params.topPeaks * params.surfaceRatio));
  const detailCount = Math.max(2, params.topPeaks - deepCount - surfaceCount);

  const deepPeaks = activePeaks.slice(-deepCount).reverse();
  const surfacePeaks = activePeaks.slice(0, surfaceCount);
  const detailPeaks = activePeaks.slice(surfaceCount, surfaceCount + detailCount);

  const waveOpts = { phaseOffset: params.phaseOffset, rng: () => rng.next() };
  const deepWaves = normalizePeaksToWaves(deepPeaks, maxMag, 32, 0.18, waveOpts);
  const surfaceWaves = normalizePeaksToWaves(surfacePeaks, maxMag, 20, 0.32, waveOpts);
  const detailWaves = normalizePeaksToWaves(detailPeaks, maxMag, 10, 0.8, waveOpts);

  const gapMean = windowGaps.reduce((s, g) => s + g, 0) / windowGaps.length;
  const sparkEvents: OceanSpectrum['metadata']['sparkEvents'] = [];
  let twinPrimeGaps = 0;
  let smallGaps = 0;
  let largeGaps = 0;

  for (let i = 0; i < windowSize; i += params.sparkStride) {
    const g = windowGaps[i]!;
    const position = i / windowSize;
    if (g === 2) {
      twinPrimeGaps++;
      sparkEvents.push({ position, gap: g, kind: 'twin', strength: 1 });
    } else if (g <= 4) {
      smallGaps++;
      sparkEvents.push({ position, gap: g, kind: 'small', strength: 0.55 });
    } else if (g > gapMean * 2.2) {
      largeGaps++;
      sparkEvents.push({
        position,
        gap: g,
        kind: 'large',
        strength: Math.min(1, (g - gapMean) / gapMean),
      });
    }
  }
  sparkEvents.sort((a, b) => b.strength - a.strength);
  const capped = [
    ...sparkEvents.filter((e) => e.kind === 'twin').slice(0, params.sparkTwinCap),
    ...sparkEvents.filter((e) => e.kind !== 'twin').slice(0, params.sparkOtherCap),
  ];

  const primeWindowEnd = windowStart + windowSize - 1;

  const usedPeaks = [
    ...buildUsedPeakRecords(deepPeaks, deepWaves, 'horizon', maxMag),
    ...buildUsedPeakRecords(surfacePeaks, surfaceWaves, 'midground', maxMag),
    ...buildUsedPeakRecords(detailPeaks, detailWaves, 'foreground', maxMag),
  ]
    .sort((a, b) => b.magnitude - a.magnitude)
    .slice(0, 12);

  return {
    deepWaves,
    surfaceWaves,
    detailWaves,
    metadata: {
      primeCount: primes.length,
      gapCount: allGaps.length,
      fftSamples: centered.length,
      dominantFrequencies: surfacePeaks.map((p) => p.frequency),
      peakMagnitudes: surfacePeaks.map((p) => p.magnitude),
      gapPreview: subsample(centered, 180),
      spectrumPreview: buildSpectrumPreview(real, imag, 120),
      sparkEvents: capped,
      sessionSeed,
      primeLimit: params.primeLimit,
      primeWindowStart: windowStart,
      primeWindowEnd,
      dominantPeakCount: params.topPeaks,
      usedPeaks,
      primeEventStats: {
        twinPrimeGaps,
        smallGaps,
        largeGaps,
        sparkEventsGenerated: capped.length,
      },
    },
  };
}

self.onmessage = (event: MessageEvent<PrimeFftWorkerRequest>) => {
  const { type, sessionSeed, mobile } = event.data;
  if (type !== 'compute') return;

  try {
    const spectrum = computeSpectrum(sessionSeed, mobile);
    post({ type: 'progress', stage: 'done', percent: 100 });
    post({ type: 'result', spectrum });
  } catch (err) {
    post({
      type: 'error',
      message: err instanceof Error ? err.message : 'Unknown worker error',
    });
  }
};
