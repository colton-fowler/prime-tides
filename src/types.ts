export interface WaveComponent {
  frequency: number;
  amplitude: number;
  phase: number;
  speed: number;
  harmonic?: number;
  /** Normalized FFT magnitude 0–1 for foam weighting */
  peakStrength: number;
}

export interface SpectrumPreview {
  freq: number;
  mag: number;
}

export interface GapSparkEvent {
  /** Normalized position 0–1 along the prime-gap sequence */
  position: number;
  gap: number;
  kind: 'twin' | 'small' | 'large';
  strength: number;
}

export interface UsedWavePeak {
  /** Normalized FFT bin frequency (0–0.5) */
  frequency: number;
  magnitude: number;
  normalizedAmplitude: number;
  phase: number;
  speed: number;
  layer: 'horizon' | 'midground' | 'foreground';
}

export interface PrimeEventStats {
  twinPrimeGaps: number;
  smallGaps: number;
  largeGaps: number;
  sparkEventsGenerated: number;
}

export interface OceanSpectrum {
  deepWaves: WaveComponent[];
  surfaceWaves: WaveComponent[];
  detailWaves: WaveComponent[];
  metadata: {
    primeCount: number;
    gapCount: number;
    fftSamples: number;
    dominantFrequencies: number[];
    peakMagnitudes: number[];
    gapPreview: number[];
    spectrumPreview: SpectrumPreview[];
    sparkEvents: GapSparkEvent[];
    sessionSeed: number;
    primeLimit: number;
    primeWindowStart: number;
    primeWindowEnd: number;
    dominantPeakCount: number;
    usedPeaks: UsedWavePeak[];
    primeEventStats: PrimeEventStats;
  };
}

export interface Ripple {
  x: number;
  y: number;
  radius: number;
  strength: number;
  birth: number;
}

export interface FoamParticle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  size: number;
  brightness: number;
  kind?: 'foam' | 'spray' | 'spark' | 'streak' | 'twin';
}

export interface MouseState {
  x: number;
  y: number;
  normX: number;
  normY: number;
  active: boolean;
}

export interface RenderQuality {
  mobile: boolean;
  reducedMotion: boolean;
  sampleStep: number;
  particleCap: number;
  layerCount: number;
}

export interface SpectrumSource {
  isFallback: boolean;
  workerStatus: 'idle' | 'computing' | 'ready' | 'error';
}

export type PrimeFftWorkerRequest = {
  type: 'compute';
  sessionSeed: number;
  mobile: boolean;
};

export type PrimeFftWorkerResponse =
  | { type: 'progress'; stage: string; percent: number }
  | { type: 'result'; spectrum: OceanSpectrum }
  | { type: 'error'; message: string };
