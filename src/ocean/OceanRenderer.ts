import type {
  FoamParticle,
  GapSparkEvent,
  MouseState,
  OceanSpectrum,
  RenderQuality,
  Ripple,
  WaveComponent,
} from '../types';
import { deriveSessionParams, SeededRng } from '../math/sessionSeed';

const HORIZON_COUNT = 4;
const MID_COUNT = 10;
const FOREGROUND_COUNT = 5;
const PARALLAX_LERP = 0.04;
const MAX_PARALLAX_X = 10;
const MAX_PARALLAX_Y = 5;
const RIBBON_OVERLAP_PAD = 72;
const MOON_RADIUS_PX = 30;
const HORIZON_NORM_Y = 0.42;
const MOON_REFLECTION_FOLLOW_DRAG = 0.85;
const MOON_REFLECTION_FOLLOW_IDLE = 0.18;

/** Visual gain — tuned for quiet moon haze, not a bright lamp. */
const MOON_GLOW_GAIN = 1.35;
const MOON_WISP_GAIN = 1.45;
const HORIZON_HAZE_GAIN = 1.25;
const HORIZON_GLINT_GAIN = 0.85;
const MOON_REFLECT_GAIN = 1.05;

/** Foreground swell blend: low / mid / high frequency pools */
const SWELL_LOW_WEIGHT = 0.68;
const SWELL_MID_WEIGHT = 0.26;
const SWELL_HIGH_WEIGHT = 0.06;

function attenuateByFrequency(w: WaveComponent, damping: number): WaveComponent {
  const scale = Math.exp(-w.frequency * damping);
  return { ...w, amplitude: w.amplitude * scale };
}

/** Smooth neighboring peaks so one FFT bin cannot dominate a swell. */
function smoothPeakBlend(waves: WaveComponent[]): WaveComponent[] {
  if (waves.length <= 1) return waves.map((w) => ({ ...w }));

  const byMag = [...waves].sort((a, b) => b.peakStrength - a.peakStrength);
  const capped = byMag.map((w) => ({
    ...w,
    amplitude: w.amplitude * (0.55 + w.peakStrength * 0.45),
  }));

  const byFreq = [...capped].sort((a, b) => a.frequency - b.frequency);
  return byFreq.map((w, i) => {
    let ampSum = 0;
    let n = 0;
    for (let j = Math.max(0, i - 1); j <= Math.min(byFreq.length - 1, i + 1); j++) {
      ampSum += byFreq[j]!.amplitude;
      n++;
    }
    const neighborAvg = ampSum / n;
    const blended = w.amplitude * 0.48 + neighborAvg * 0.52;
    const maxAllowed = neighborAvg * 1.28 + w.amplitude * 0.12;
    return { ...w, amplitude: Math.min(blended, maxAllowed) };
  });
}

function prepareSwellWaves(waves: WaveComponent[], damping: number): WaveComponent[] {
  return smoothPeakBlend(waves.map((w) => attenuateByFrequency(w, damping)));
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function smoothstep(t: number): number {
  const c = Math.max(0, Math.min(1, t));
  return c * c * (3 - 2 * c);
}

function smootherstep(t: number): number {
  const c = Math.max(0, Math.min(1, t));
  return c * c * c * (c * (c * 6 - 15) + 10);
}

function hash(n: number): number {
  const s = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return s - Math.floor(s);
}

function smoothNoise(x: number, seed: number): number {
  const ix = Math.floor(x);
  const fx = x - ix;
  const a = hash(ix + seed) * 2 - 1;
  const b = hash(ix + 1 + seed) * 2 - 1;
  return a + (b - a) * smoothstep(fx);
}

function fractalNoise(x: number, time: number, octaves: number, seed: number): number {
  let amp = 1;
  let freq = 1;
  let sum = 0;
  let norm = 0;
  for (let i = 0; i < octaves; i++) {
    sum += smoothNoise(x * freq + time * 0.22, seed + i * 19.3) * amp;
    norm += amp;
    amp *= 0.48;
    freq *= 2.05;
  }
  return sum / norm;
}

/** Deterministic 0–1 hash for slow atmospheric noise. */
function hash1(n: number): number {
  return hash(n);
}

/** Smooth 1D noise in approximately -1..1. */
function smoothNoise1D(x: number, seed = 0): number {
  return smoothNoise(x, seed);
}

/** Fractal 1D noise in approximately -1..1. */
function fbm1D(x: number, seed: number, octaves = 3): number {
  let amp = 1;
  let freq = 1;
  let sum = 0;
  let norm = 0;
  for (let i = 0; i < octaves; i++) {
    sum += smoothNoise1D(x * freq, seed + i * 17.3) * amp;
    norm += amp;
    amp *= 0.5;
    freq *= 2.03;
  }
  return sum / norm;
}

/** Normalize animation time to seconds (guards ms accidentally passed in). */
function animSeconds(time: number): number {
  return time > 1000 ? time / 1000 : time;
}

function moonBreathAt(seconds: number): number {
  return (
    1 +
    0.08 * Math.sin(seconds * 0.28) +
    0.055 * smoothNoise1D(seconds * 0.075 + 31.7, 88)
  );
}

type RibbonTier = 'horizon' | 'mid' | 'foreground';

interface RibbonParams {
  index: number;
  depth: number;
  tier: RibbonTier;
  yBase: number;
  amp: number;
  opacity: number;
  speed: number;
  freq: number;
  phase: number;
  peakStrength: number;
  blur: number;
  fillBottom: number;
  /** Swell-smoothed geometry — default ocean */
  waves: WaveComponent[];
  /** Raw FFT peaks — blended in when reveal is active */
  rawWaves: WaveComponent[];
  /** Detail peaks — foam, shimmer, ripples only */
  detailWaves: WaveComponent[];
}

export interface RenderStats {
  time: number;
  frames: number;
  reveal: boolean;
}

export class OceanRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private spectrum: OceanSpectrum;
  private quality: RenderQuality;
  private width = 0;
  private height = 0;
  private dpr = 1;
  private ripples: Ripple[] = [];
  private foam: FoamParticle[] = [];
  private mouse: MouseState = { x: 0, y: 0, normX: 0, normY: 0, active: false };
  private targetParallaxX = 0;
  private targetParallaxY = 0;
  private currentParallaxX = 0;
  private currentParallaxY = 0;
  private moonNorm = { x: 0.66, y: 0.18 };
  private moonLight = { x: 0.66, y: 0.18 };
  private moonReflection = { x: 0.66, y: 0.18 };
  private moonDragging = false;
  private mathReveal = 0;
  private animTime = 0;
  private frameCount = 0;
  private spawnAccumulator = 0;
  private sparkAccumulator = 0;
  private ribbonsCache: RibbonParams[] = [];

  constructor(canvas: HTMLCanvasElement, spectrum: OceanSpectrum, quality?: RenderQuality) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) throw new Error('Canvas 2D context unavailable');
    this.ctx = ctx;
    this.spectrum = spectrum;
    this.quality = quality ?? detectRenderQuality();
    this.resize();
  }

  dispose(): void {
    this.foam = [];
    this.ripples = [];
  }

  setSpectrum(spectrum: OceanSpectrum): void {
    this.spectrum = spectrum;
  }

  isMoonDragging(): boolean {
    return this.moonDragging;
  }

  getMoonNormalized(): { x: number; y: number } {
    return { ...this.moonNorm };
  }

  setMoonNormalized(next: { x: number; y: number }): void {
    this.moonNorm = this.clampMoonNorm(next.x, next.y);
  }

  private clampMoonNorm(x: number, y: number): { x: number; y: number } {
    const moonRadiusNorm = MOON_RADIUS_PX / Math.max(1, this.height);
    const maxY = HORIZON_NORM_Y - moonRadiusNorm * 0.65;
    return {
      x: Math.max(0.08, Math.min(0.92, x)),
      y: Math.max(0.08, Math.min(maxY, y)),
    };
  }

  private moonRenderNorm(): { x: number; y: number } {
    return this.moonDragging ? this.moonNorm : this.moonLight;
  }

  private moonRenderPx(): { x: number; y: number } {
    const render = this.moonRenderNorm();
    const parallax = this.moonDragging ? 0 : 1;
    return {
      x: render.x * this.width + this.currentParallaxX * parallax,
      y: render.y * this.height + this.currentParallaxY * parallax,
    };
  }

  isMoonHit(pxX: number, pxY: number): boolean {
    const moonX = this.moonNorm.x * this.width;
    const moonY = this.moonNorm.y * this.height;
    const dx = pxX - moonX;
    const dy = pxY - moonY;
    return Math.sqrt(dx * dx + dy * dy) <= MOON_RADIUS_PX + 8;
  }

  beginMoonDrag(pxX: number, pxY: number): boolean {
    if (!this.isMoonHit(pxX, pxY)) return false;
    this.moonDragging = true;
    this.setMoonNormalized({ x: pxX / this.width, y: pxY / this.height });
    this.moonLight.x = this.moonNorm.x;
    this.moonLight.y = this.moonNorm.y;
    return true;
  }

  dragMoonTo(pxX: number, pxY: number): void {
    if (!this.moonDragging) return;
    this.setMoonNormalized({ x: pxX / this.width, y: pxY / this.height });
  }

  endMoonDrag(): void {
    this.moonDragging = false;
  }

  setMouse(mouse: MouseState): void {
    if (this.moonDragging) return;
    this.mouse = mouse;
    const mx = this.mouse.active ? this.mouse.x / this.width - 0.5 : 0;
    const my = this.mouse.active ? this.mouse.y / this.height - 0.5 : 0;
    this.targetParallaxX = Math.max(-MAX_PARALLAX_X, Math.min(MAX_PARALLAX_X, mx * MAX_PARALLAX_X));
    this.targetParallaxY = Math.max(-MAX_PARALLAX_Y, Math.min(MAX_PARALLAX_Y, my * MAX_PARALLAX_Y));
  }

  addRipple(x: number, y: number): void {
    this.ripples.push({ x, y, radius: 0, strength: 0.45, birth: this.animTime });
    if (this.ripples.length > 8) this.ripples.shift();
  }

  resize(): void {
    this.dpr = Math.min(window.devicePixelRatio || 1, this.quality.mobile ? 1.5 : 2);
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.canvas.width = Math.floor(this.width * this.dpr);
    this.canvas.height = Math.floor(this.height * this.dpr);
    this.canvas.style.width = `${this.width}px`;
    this.canvas.style.height = `${this.height}px`;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    // Moon position stored as normalized coordinates; nothing needed here.
  }

  render(elapsed: number, revealMath = false): RenderStats {
    this.frameCount++;
    this.animTime = elapsed;

    const revealTarget = revealMath ? 1 : 0;
    this.mathReveal += (revealTarget - this.mathReveal) * 0.08;

    this.currentParallaxX += (this.targetParallaxX - this.currentParallaxX) * PARALLAX_LERP;
    this.currentParallaxY += (this.targetParallaxY - this.currentParallaxY) * PARALLAX_LERP;
    if (!this.mouse.active) {
      this.targetParallaxX *= 0.97;
      this.targetParallaxY *= 0.97;
    }

    const dt = Math.min(0.05, 1 / 60);
    this.updateMoonLighting();
    this.ribbonsCache = this.buildRibbons();
    this.update(dt);
    this.draw();

    return { time: this.animTime, frames: this.frameCount, reveal: this.mathReveal > 0.05 };
  }

  private wavesForTierRaw(tier: RibbonTier): WaveComponent[] {
    if (tier === 'horizon') return this.spectrum.deepWaves;
    if (tier === 'mid') return this.spectrum.surfaceWaves;
    return [...this.spectrum.surfaceWaves.slice(0, 2), ...this.spectrum.detailWaves];
  }

  private geometryWavesForTier(tier: RibbonTier): WaveComponent[] {
    const low = prepareSwellWaves(this.spectrum.deepWaves, 1.2);
    const mid = prepareSwellWaves(this.spectrum.surfaceWaves, 5.5);
    if (tier === 'horizon') return low;
    if (tier === 'mid') return [...low.slice(0, 2), ...mid];
    return [...low.slice(0, 3), ...mid.slice(0, 4)];
  }

  private buildRibbons(): RibbonParams[] {
    const ribbons: RibbonParams[] = [];
    let index = 0;

    for (let i = 0; i < HORIZON_COUNT; i++) {
      const t = i / Math.max(1, HORIZON_COUNT - 1);
      const depth = t * 0.18;
      const waves = this.geometryWavesForTier('horizon');
      const rawWaves = this.wavesForTierRaw('horizon');
      const w = waves[Math.min(waves.length - 1, i)]!;
      ribbons.push({
        index: index++,
        depth,
        tier: 'horizon',
        yBase: lerp(this.height * 0.42, this.height * 0.48, t * 0.55),
        amp: lerp(5, 18, t) * (0.38 + w.peakStrength * 0.28),
        opacity: lerp(0.04, 0.11, t),
        speed: lerp(0.08, 0.14, t),
        freq: w.frequency * lerp(0.3, 0.5, t),
        phase: w.phase + i * 1.3,
        peakStrength: w.peakStrength * 0.35,
        blur: lerp(12, 20, t),
        fillBottom: 0,
        waves,
        rawWaves,
        detailWaves: [],
      });
    }

    for (let i = 0; i < MID_COUNT; i++) {
      const t = i / Math.max(1, MID_COUNT - 1);
      const depth = 0.18 + t * 0.42;
      const waves = this.geometryWavesForTier('mid');
      const rawWaves = this.wavesForTierRaw('mid');
      const w = waves[Math.min(waves.length - 1, Math.floor(t * waves.length))]!;
      ribbons.push({
        index: index++,
        depth,
        tier: 'mid',
        yBase: lerp(this.height * 0.41, this.height * 0.78, Math.pow(t, 0.82)),
        amp: lerp(15, 56, t) * (0.46 + w.peakStrength * 0.5),
        opacity: lerp(0.1, 0.42, t),
        speed: lerp(0.12, 0.3, t),
        freq: w.frequency * lerp(0.5, 1.05, t),
        phase: w.phase + i * 0.9 + 2.1,
        peakStrength: w.peakStrength * 0.7,
        blur: lerp(8, 3, t),
        fillBottom: 0,
        waves,
        rawWaves,
        detailWaves: this.spectrum.detailWaves,
      });
    }

    const fgSpread = [0, 0.07, 0.17, 0.31, 0.5];
    for (let i = 0; i < FOREGROUND_COUNT; i++) {
      const t = i / Math.max(1, FOREGROUND_COUNT - 1);
      const depth = 0.62 + fgSpread[i]! * 0.36;
      const waves = this.geometryWavesForTier('foreground');
      const rawWaves = this.wavesForTierRaw('foreground');
      const w = waves[Math.min(waves.length - 1, i)]!;
      ribbons.push({
        index: index++,
        depth,
        tier: 'foreground',
        yBase: lerp(this.height * 0.55, this.height * 0.92, fgSpread[i]!),
        amp: lerp(62, 118, t) * (0.56 + w.peakStrength * 0.5),
        opacity: lerp(0.58, 0.96, t),
        speed: lerp(0.26, 0.52, t),
        freq: w.frequency * lerp(0.65, 1.15, t),
        phase: w.phase + i * 1.8 + 4.7,
        peakStrength: w.peakStrength,
        blur: 0,
        fillBottom: 0,
        waves,
        rawWaves,
        detailWaves: this.spectrum.detailWaves,
      });
    }

    for (let i = 0; i < ribbons.length; i++) {
      const ribbon = ribbons[i]!;
      if (i < ribbons.length - 1) {
        const next = ribbons[i + 1]!;
        let nextTopY = next.yBase - next.amp;
        const step = Math.max(4, Math.floor(this.width / 80));
        for (let x = 0; x <= this.width; x += step) {
          const y = next.yBase + this.displacement(x, this.animTime, next);
          nextTopY = Math.min(nextTopY, y);
        }
        ribbon.fillBottom = nextTopY + next.amp * 1.35 + RIBBON_OVERLAP_PAD;
      } else {
        ribbon.fillBottom = this.height + 20;
      }
    }

    return ribbons;
  }

  private updateMoonLighting(): void {
    const tx = this.moonNorm.x;
    const ty = this.moonNorm.y;

    if (this.moonDragging) {
      this.moonLight.x = tx;
      this.moonLight.y = ty;
    } else {
      this.moonLight.x += (tx - this.moonLight.x) * MOON_REFLECTION_FOLLOW_IDLE;
      this.moonLight.y += (ty - this.moonLight.y) * MOON_REFLECTION_FOLLOW_IDLE;
    }

    const reflectionFollow = this.moonDragging
      ? MOON_REFLECTION_FOLLOW_DRAG
      : MOON_REFLECTION_FOLLOW_IDLE;
    this.moonReflection.x += (tx - this.moonReflection.x) * reflectionFollow;
    this.moonReflection.y += (ty - this.moonReflection.y) * reflectionFollow;
  }

  private ribbonFillExtent(ribbon: RibbonParams, points: { x: number; y: number }[]): number {
    let maxY = ribbon.yBase + ribbon.amp * 1.2;
    for (const p of points) maxY = Math.max(maxY, p.y);
    const pad = RIBBON_OVERLAP_PAD;
    const floor =
      ribbon.tier === 'foreground' ? this.height + 8 : Math.max(ribbon.fillBottom, maxY + pad);
    return Math.max(floor, maxY + pad);
  }

  /** Closed wave path with edge bleed to avoid anti-alias seams. */
  private traceRibbonPath(
    points: { x: number; y: number }[],
    fillTo: number,
  ): void {
    const { ctx, width } = this;
    const bleed = 5;
    ctx.beginPath();
    ctx.moveTo(-bleed, fillTo + bleed);
    ctx.lineTo(-bleed, fillTo);
    for (const p of points) ctx.lineTo(p.x, p.y);
    ctx.lineTo(width + bleed, fillTo);
    ctx.lineTo(width + bleed, fillTo + bleed);
    ctx.closePath();
  }

  /** Opaque safety fill so ribbon bodies are watertight regardless of crest transparency. */
  private drawRibbonWatertightUnderpaint(
    ribbon: RibbonParams,
    points: { x: number; y: number }[],
    fillTo: number,
  ): void {
    const { ctx } = this;
    const base =
      ribbon.tier === 'horizon'
        ? '#030c14'
        : ribbon.tier === 'mid'
          ? '#020810'
          : '#010508';
    ctx.save();
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
    ctx.filter = 'none';
    this.traceRibbonPath(points, fillTo);
    ctx.fillStyle = base;
    ctx.fill();
    ctx.restore();
  }

  /** Broad rolling crest — rounds sharp FFT peaks into ocean swell */
  private roundedSwellValue(phase: number): number {
    const primary = Math.sin(phase);
    const subharmonic = Math.sin(phase * 0.5 + 0.35) * 0.26;
    const t = (primary + 1) * 0.5;
    const softened = smootherstep(t) * 2 - 1;
    // Bias toward broad crests: softened top, slower transitions.
    return primary * 0.62 + softened * 0.26 + subharmonic * 0.12;
  }

  private sumWaveContributions(
    sampleX: number,
    time: number,
    ribbon: RibbonParams,
    waves: WaveComponent[],
    style: 'raw' | 'swell',
    poolWeight = 1,
  ): number {
    let sum = 0;
    const freqScale =
      style === 'swell'
        ? ribbon.tier === 'foreground'
          ? 0.76
          : ribbon.tier === 'mid'
            ? 0.88
            : 0.92
        : 1;
    const subHarmonicMix = style === 'raw' ? 0.38 : 0.1;

    for (let wi = 0; wi < waves.length; wi++) {
      const w = waves[wi]!;
      const weight =
        ribbon.tier === 'horizon'
          ? 0.35 + w.peakStrength * 0.25
          : ribbon.tier === 'mid'
            ? 0.45 + w.peakStrength * 0.35
            : style === 'raw' && wi >= 2
              ? 0.2 + w.peakStrength * 0.15
              : 0.5 + w.peakStrength * 0.35;

      const ampScale =
        style === 'raw' && ribbon.tier === 'foreground' && wi >= 2
          ? ribbon.amp * 0.12
          : ribbon.amp * (0.55 / Math.max(1, waves.length * 0.45));

      const phase = sampleX * w.frequency * freqScale + w.phase + time * w.speed * 2.1;
      const value =
        style === 'swell' ? this.roundedSwellValue(phase) : Math.sin(phase);

      sum += value * ampScale * weight * poolWeight;

      if (subHarmonicMix > 0) {
        sum +=
          Math.sin(sampleX * w.frequency * freqScale * 0.39 + w.phase * 1.6 - time * w.speed * 1.3) *
          ampScale *
          weight *
          subHarmonicMix *
          poolWeight;
      }
    }

    return sum;
  }

  private swellDisplacement(sampleX: number, time: number, ribbon: RibbonParams): number {
    const low = prepareSwellWaves(this.spectrum.deepWaves, 1.2);
    const mid = prepareSwellWaves(this.spectrum.surfaceWaves, 5.5);
    const high = prepareSwellWaves(this.spectrum.detailWaves, 18);

    let sum = 0;
    if (ribbon.tier === 'horizon') {
      sum = this.sumWaveContributions(sampleX, time, ribbon, low, 'swell');
    } else if (ribbon.tier === 'mid') {
      sum =
        this.sumWaveContributions(sampleX, time, ribbon, low.slice(0, 2), 'swell', 0.38) +
        this.sumWaveContributions(sampleX, time, ribbon, mid, 'swell', 0.62);
    } else {
      // Dedicated macro swell system: big Atlantic rollers under FFT-driven surface.
      const macroFreq = 0.00115;
      const macroSpeed = 0.12;
      const macro =
        this.roundedSwellValue(sampleX * macroFreq + time * macroSpeed) * (ribbon.amp * 0.85);

      const fftLow = this.sumWaveContributions(sampleX, time, ribbon, low, 'swell', 1);
      const fftMid = this.sumWaveContributions(sampleX, time, ribbon, mid, 'swell', 1);
      const fftHigh = this.sumWaveContributions(sampleX, time, ribbon, high.slice(0, 2), 'swell', 1);

      // 70% macro swell, 25% FFT low/mid, 5% FFT detail.
      const detailWeight = SWELL_HIGH_WEIGHT;
      sum =
        macro * 0.7 +
        (fftLow * SWELL_LOW_WEIGHT + fftMid * SWELL_MID_WEIGHT) * 0.25 +
        fftHigh * detailWeight;
    }

    const octaves = ribbon.tier === 'horizon' ? 2 : ribbon.tier === 'mid' ? 3 : 3;
    const noiseScale =
      ribbon.tier === 'horizon' ? 0.0035 : ribbon.tier === 'mid' ? 0.0048 : 0.0065;
    const noiseAmp =
      ribbon.tier === 'horizon'
        ? ribbon.amp * 0.2
        : ribbon.tier === 'mid'
          ? ribbon.amp * 0.26
          : ribbon.amp * 0.2;
    const noise = fractalNoise(
      sampleX * noiseScale + ribbon.index * 5.3,
      time * ribbon.speed * 0.35,
      octaves,
      ribbon.index * 11.9,
    );

    return sum + noise * noiseAmp;
  }

  private rawDisplacement(sampleX: number, time: number, ribbon: RibbonParams): number {
    let sum = this.sumWaveContributions(sampleX, time, ribbon, ribbon.rawWaves, 'raw');

    const octaves = ribbon.tier === 'horizon' ? 2 : ribbon.tier === 'mid' ? 3 : 5;
    const noiseScale =
      ribbon.tier === 'horizon' ? 0.0035 : ribbon.tier === 'mid' ? 0.0055 : 0.01;
    const noiseAmp =
      ribbon.tier === 'horizon'
        ? ribbon.amp * 0.2
        : ribbon.tier === 'mid'
          ? ribbon.amp * 0.31
          : ribbon.amp * 0.48;
    const noise = fractalNoise(
      sampleX * noiseScale + ribbon.index * 5.3,
      time * ribbon.speed * 0.35,
      octaves,
      ribbon.index * 11.9,
    );

    return sum + noise * noiseAmp;
  }

  /** FFT-driven displacement — swell by default, raw structure when reveal is active */
  private displacement(x: number, time: number, ribbon: RibbonParams): number {
    const sampleX = x + time * ribbon.speed * 140;
    const swell = this.swellDisplacement(sampleX, time, ribbon);
    const raw = this.rawDisplacement(sampleX, time, ribbon);
    const reveal = smoothstep(this.mathReveal);
    return lerp(swell, raw, reveal);
  }

  private ribbonY(x: number, time: number, ribbon: RibbonParams): number {
    return ribbon.yBase + this.displacement(x, time, ribbon);
  }

  private smoothRibbonProfile(
    points: { x: number; y: number }[],
    window: number,
  ): { x: number; y: number }[] {
    const half = Math.floor(window / 2);
    return points.map((p, i) => {
      let sum = 0;
      let n = 0;
      for (let j = -half; j <= half; j++) {
        const idx = i + j;
        if (idx >= 0 && idx < points.length) {
          sum += points[idx]!.y;
          n++;
        }
      }
      return { x: p.x, y: sum / n };
    });
  }

  /** Detail FFT peaks drive shimmer/foam — not swell height */
  private detailShimmerAt(x: number, time: number, detailWaves: WaveComponent[]): number {
    if (!detailWaves.length) return 1;
    let s = 0;
    for (const w of detailWaves) {
      s +=
        Math.sin(x * w.frequency * 1.15 + w.phase + time * w.speed * 2.8) *
        w.peakStrength;
    }
    return 0.62 + 0.38 * (0.5 + 0.5 * Math.sin(s * 1.4));
  }

  private sampleRibbon(ribbon: RibbonParams, time: number): { x: number; y: number }[] {
    const step =
      ribbon.tier === 'horizon'
        ? this.quality.sampleStep * 4
        : ribbon.tier === 'mid'
          ? this.quality.sampleStep * 2
          : this.quality.sampleStep;
    const points: { x: number; y: number }[] = [];
    for (let x = 0; x <= this.width; x += step) {
      points.push({ x, y: this.ribbonY(x, time, ribbon) });
    }

    if (ribbon.tier === 'foreground' || ribbon.tier === 'mid') {
      const window = ribbon.tier === 'foreground' ? 7 : 5;
      const smoothed = this.smoothRibbonProfile(points, window);
      const reveal = smoothstep(this.mathReveal);
      return points.map((p, i) => ({
        x: p.x,
        y: lerp(smoothed[i]!.y, p.y, reveal),
      }));
    }

    return points;
  }

  private combinedSurfaceY(x: number, time: number): number {
    const fg = this.ribbonsCache.filter((r) => r.tier === 'foreground');
    if (!fg.length) return this.height * 0.7;
    let y = 0;
    for (const r of fg) y += this.ribbonY(x, time, r);
    return y / fg.length;
  }

  private update(dt: number): void {
    this.ripples = this.ripples.filter((r) => this.animTime - r.birth < 4);
    if (this.quality.reducedMotion) return;

    this.spawnAccumulator += dt;
    this.sparkAccumulator += dt;

    const time = this.animTime;
    const ribbons = this.ribbonsCache;
    const step = this.quality.mobile ? 18 : 12;

    if (this.spawnAccumulator >= 0.035) {
      this.spawnAccumulator = 0;

      for (const ribbon of ribbons) {
        if (ribbon.tier === 'horizon') continue;

        for (let x = 0; x < this.width; x += step) {
          const y = this.ribbonY(x, time, ribbon);
          const yL = this.ribbonY(x - 14, time, ribbon);
          const yR = this.ribbonY(x + 14, time, ribbon);
          if (y >= yL || y >= yR) continue;

          const crestH = Math.min(yL - y, yR - y);
          const strength = crestH * ribbon.peakStrength * (0.35 + ribbon.depth);

          if (ribbon.tier === 'foreground' && strength > 2.2) {
            const detailBoost = this.detailShimmerAt(x, time, ribbon.detailWaves);
            const foamChance = 0.05 * ribbon.depth * (0.52 + ribbon.peakStrength) * detailBoost;
            if (Math.random() < foamChance) {
              this.spawnFoam(x, y, ribbon.speed, strength, ribbon.peakStrength * detailBoost);
            }
            if (Math.random() < 0.02 * ribbon.depth * ribbon.peakStrength * detailBoost) {
              this.spawnSprayCluster(x, y, ribbon.speed, strength * ribbon.peakStrength);
            }

            // Particle clustering: if we have a strong foam zone, bias nearby spawns.
            if (strength > 3.2 && Math.random() < 0.22 * detailBoost) {
              const clusterCount = 1 + (strength > 4.5 ? 1 : 0);
              for (let c = 0; c < clusterCount; c++) {
                const ox = (hash(x * 0.21 + c * 9.7 + time) - 0.5) * 34;
                const oy = (hash(y * 0.31 + c * 5.3 + time) - 0.5) * 6;
                this.spawnFoam(x + ox, y + oy, ribbon.speed, strength * 0.8, ribbon.peakStrength * detailBoost);
              }
            }
          } else if (ribbon.tier === 'mid' && strength > 2 && Math.random() < 0.006 * ribbon.peakStrength) {
            this.spawnFoam(x, y, ribbon.speed, strength * 0.35, ribbon.peakStrength * 0.5);
          }
        }
      }
    }

    if (this.sparkAccumulator >= 0.12) {
      this.sparkAccumulator = 0;
      this.spawnPrimeSparks(time, ribbons);
    }

    for (let i = this.foam.length - 1; i >= 0; i--) {
      const p = this.foam[i]!;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      if (p.kind === 'spray' || p.kind === 'spark') {
        p.vy += 10 * dt;
        p.vx *= 0.97;
      }
      if (p.kind === 'streak') {
        p.vy *= 0.99;
        p.vx += p.vx * 0.002;
      }
      p.life -= dt;
      if (p.life <= 0) this.foam.splice(i, 1);
    }
  }

  private spawnPrimeSparks(time: number, ribbons: RibbonParams[]): void {
    const events = this.spectrum.metadata.sparkEvents;
    if (!events.length) return;

    const fg = ribbons.filter((r) => r.tier === 'foreground');
    if (!fg.length) return;

    const picks = 2 + Math.floor(Math.random() * 3);
    for (let n = 0; n < picks; n++) {
      const event = events[Math.floor(Math.random() * Math.min(events.length, 40))]!;
      const x = event.position * this.width + (hash(event.position * 999 + time) - 0.5) * 40;
      const ribbon = fg[Math.floor(hash(event.position * 77) * fg.length)]!;
      const y = this.ribbonY(x, time, ribbon);

      if (event.kind === 'twin') {
        this.spawnTwinSparkles(x, y, event, ribbon.peakStrength);
      } else if (event.kind === 'large') {
        this.spawnStreak(x, y, ribbon.speed, event.strength * ribbon.peakStrength);
      } else if (Math.random() < event.strength * ribbon.peakStrength) {
        this.spawnSpark(x, y, ribbon.speed, event.strength * ribbon.peakStrength);
      }
    }
  }

  private spawnFoam(
    x: number,
    y: number,
    speed: number,
    strength: number,
    peakStrength: number,
  ): void {
    if (this.foam.length >= this.quality.particleCap) return;
    const bright = 0.54 + peakStrength * 0.58 + Math.min(strength * 0.08, 0.45);
    this.foam.push({
      x: x + (Math.random() - 0.5) * 16,
      y: y + (Math.random() - 0.5) * 4,
      vx: speed * 140 * 0.22 + Math.random() * 14,
      vy: -0.3 - Math.random() * 1.0,
      life: 0.5 + Math.random() * 1.0,
      maxLife: 1.5,
      size: 1.8 + Math.random() * 2.8 + strength * 0.1,
      brightness: bright,
      kind: 'foam',
    });
  }

  private spawnSprayCluster(x: number, y: number, speed: number, strength: number): void {
    const count = 5 + Math.floor(Math.random() * 5);
    for (let i = 0; i < count; i++) {
      if (this.foam.length >= this.quality.particleCap) return;
      const angle = -Math.PI * 0.4 + Math.random() * Math.PI * 0.55;
      const spd = 30 + Math.random() * 50 + speed * 35;
      this.foam.push({
        x: x + (Math.random() - 0.5) * 10,
        y: y + (Math.random() - 0.5) * 5,
        vx: Math.cos(angle) * spd + speed * 35,
        vy: Math.sin(angle) * spd - 18 - Math.random() * 25,
        life: 0.2 + Math.random() * 0.5,
        maxLife: 0.7,
        size: 0.35 + Math.random() * 0.9,
        brightness: 0.65 + Math.min(strength * 0.05, 0.35),
        kind: 'spray',
      });
    }
  }

  private spawnTwinSparkles(x: number, y: number, event: GapSparkEvent, peakStrength: number): void {
    for (let i = 0; i < 2; i++) {
      if (this.foam.length >= this.quality.particleCap) return;
      const offset = i === 0 ? -5 : 5;
      this.foam.push({
        x: x + offset,
        y: y + (Math.random() - 0.5) * 3,
        vx: 8 + Math.random() * 12,
        vy: -12 - Math.random() * 18,
        life: 0.35 + Math.random() * 0.4,
        maxLife: 0.75,
        size: 0.5 + Math.random() * 0.6,
        brightness: 0.85 + event.strength * peakStrength * 0.15,
        kind: 'twin',
      });
    }
  }

  private spawnSpark(x: number, y: number, speed: number, strength: number): void {
    if (this.foam.length >= this.quality.particleCap) return;
    this.foam.push({
      x,
      y,
      vx: speed * 60 + (Math.random() - 0.5) * 20,
      vy: -20 - Math.random() * 30,
      life: 0.3 + Math.random() * 0.5,
      maxLife: 0.8,
      size: 0.6 + Math.random() * 1.0,
      brightness: 0.75 + strength * 0.25,
      kind: 'spark',
    });
  }

  private spawnStreak(x: number, y: number, speed: number, strength: number): void {
    if (this.foam.length >= this.quality.particleCap) return;
    const len = 18 + strength * 40;
    this.foam.push({
      x,
      y,
      vx: speed * 80 + 30,
      vy: -2 - Math.random() * 4,
      life: 0.5 + Math.random() * 0.6,
      maxLife: 1.1,
      size: len,
      brightness: 0.6 + strength * 0.35,
      kind: 'streak',
    });
  }

  private drawSky(): void {
    const { ctx, width, height } = this;
    const { x: moonX, y: moonY } = this.moonRenderPx();
    const horizonY = height * HORIZON_NORM_Y;

    const sky = ctx.createLinearGradient(0, 0, 0, height * 0.5);
    sky.addColorStop(0, '#000308');
    sky.addColorStop(0.4, '#030c16');
    sky.addColorStop(0.75, '#061422');
    sky.addColorStop(1, '#0a1e32');
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, width, height);

    // Moon glow / halo / wisps drawn after sky, crisp disk at end of drawMoonLighting.
    this.drawMoonLighting(moonX, moonY, horizonY);
  }

  /** Procedural moon lighting — disk crisp; glow breathes like slow smoky fog. */
  private drawMoonLighting(moonX: number, moonY: number, horizonY: number): void {
    const { ctx, width, height } = this;
    const seconds = animSeconds(this.animTime);
    const fogT = seconds;
    const breath = moonBreathAt(seconds);
    const edgeNoise = fbm1D(moonX * 0.0028 + seconds * 0.06, 41, 3);
    const edgeNoise2 = fbm1D(moonY * 0.0032 + seconds * 0.04 + 19.3, 73, 2);

    ctx.save();
    ctx.globalCompositeOperation = 'screen';

    // Faint cyan outer bloom — breathes slowly, wide and soft.
    ctx.filter = 'blur(48px)';
    const outerR = (460 + edgeNoise * 42) * breath;
    const outerA = 0.028 * breath * MOON_GLOW_GAIN;
    const outer = ctx.createRadialGradient(moonX, moonY, MOON_RADIUS_PX * 0.8, moonX, moonY, outerR);
    outer.addColorStop(0, `rgba(90, 145, 175, ${outerA * 0.35})`);
    outer.addColorStop(0.35, `rgba(60, 115, 150, ${outerA})`);
    outer.addColorStop(0.7, `rgba(40, 85, 120, ${outerA * 0.35})`);
    outer.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = outer;
    ctx.fillRect(moonX - outerR, moonY - outerR, outerR * 2, outerR * 2);

    // Wide atmospheric halo — low contrast near disk.
    ctx.filter = 'blur(28px)';
    const atmoR = (340 + edgeNoise2 * 38) * breath;
    const atmoA = 0.042 * breath * MOON_GLOW_GAIN;
    const atmo = ctx.createRadialGradient(moonX, moonY, MOON_RADIUS_PX * 1.1, moonX, moonY, atmoR);
    atmo.addColorStop(0, `rgba(130, 165, 195, ${atmoA * 0.25})`);
    atmo.addColorStop(0.25, `rgba(110, 150, 185, ${atmoA * 0.55})`);
    atmo.addColorStop(0.6, `rgba(70, 115, 155, ${atmoA * 0.28})`);
    atmo.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = atmo;
    ctx.fillRect(moonX - atmoR, moonY - atmoR, atmoR * 2, atmoR * 2);

    // Soft near halo — starts outside disk edge.
    ctx.filter = 'blur(18px)';
    const nearR = (148 + smoothNoise1D(seconds * 0.11 + 3.7, 52) * 18) * breath;
    const nearA = 0.055 * breath * MOON_GLOW_GAIN;
    const near = ctx.createRadialGradient(moonX, moonY, MOON_RADIUS_PX * 1.05, moonX, moonY, nearR);
    near.addColorStop(0, `rgba(160, 190, 215, ${nearA * 0.3})`);
    near.addColorStop(0.3, `rgba(140, 175, 205, ${nearA})`);
    near.addColorStop(0.65, `rgba(90, 135, 170, ${nearA * 0.32})`);
    near.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = near;
    ctx.fillRect(moonX - nearR, moonY - nearR, nearR * 2, nearR * 2);
    ctx.filter = 'none';

    this.drawOrganicMoonHalo(ctx, moonX, moonY, seconds, breath, nearR);
    this.drawMoonFogWisps(ctx, moonX, moonY, seconds, fogT, nearR, breath);

    // Vertical moonlight haze cone toward horizon — very faint.
    const coneW = lerp(110, 260, moonX / width) * (0.92 + edgeNoise * 0.12) * breath;
    const coneA = 0.028 * breath * MOON_GLOW_GAIN;
    const cone = ctx.createLinearGradient(moonX, moonY, moonX, horizonY + 40);
    cone.addColorStop(0, `rgba(150, 180, 205, ${coneA * 0.4})`);
    cone.addColorStop(0.35, `rgba(100, 145, 175, ${coneA * 0.45})`);
    cone.addColorStop(0.75, `rgba(60, 100, 135, ${coneA * 0.18})`);
    cone.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.save();
    ctx.filter = 'blur(18px)';
    ctx.fillStyle = cone;
    ctx.fillRect(moonX - coneW, moonY, coneW * 2, horizonY - moonY + 60);
    ctx.restore();

    // Moonlight shafts — very slow drift.
    ctx.filter = 'blur(28px)';
    for (let i = 0; i < 3; i++) {
      const ang =
        -0.55 +
        i * 0.22 +
        Math.sin(seconds * 0.07 + i) * 0.05 +
        (moonX / width - 0.5) * 0.12 +
        smoothNoise1D(fogT + i * 2.1, 31) * 0.04;
      const len = height * (0.55 + i * 0.06);
      const shaftW = width * 0.22;
      const gx = moonX + Math.cos(ang) * 10;
      const gy = moonY + Math.sin(ang) * 10;
      const shaft = ctx.createLinearGradient(
        gx,
        gy,
        gx + Math.cos(ang) * len,
        gy + Math.sin(ang) * len,
      );
      shaft.addColorStop(0, `rgba(150, 180, 205, ${0.018 * breath * MOON_GLOW_GAIN})`);
      shaft.addColorStop(0.4, `rgba(90, 130, 165, ${0.008 * breath * MOON_GLOW_GAIN})`);
      shaft.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.save();
      ctx.translate(gx, gy);
      ctx.rotate(ang);
      ctx.fillStyle = shaft;
      ctx.fillRect(0, -shaftW * 0.5, len, shaftW);
      ctx.restore();
    }
    ctx.filter = 'none';

    // Ultra-low-opacity cloud bands lit by moon position.
    ctx.filter = 'blur(46px)';
    ctx.globalAlpha = 0.028 * MOON_GLOW_GAIN;
    for (let i = 0; i < 3; i++) {
      const y = height * (0.14 + i * 0.12) + Math.sin(seconds * 0.03 + i * 1.7) * 12;
      const band = ctx.createLinearGradient(0, y - 60, 0, y + 60);
      const lit = 0.5 + 0.5 * Math.cos((moonX / width - 0.5) * Math.PI);
      band.addColorStop(0, 'rgba(0,0,0,0)');
      band.addColorStop(0.5, `rgba(70, 120, 155, ${(0.04 + i * 0.005) * lit * breath})`);
      band.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = band;
      ctx.fillRect(-60, y - 80, width + 120, 160);
    }
    ctx.globalAlpha = 1;
    ctx.filter = 'none';
    ctx.globalCompositeOperation = 'source-over';
    ctx.restore();

    this.drawMoonDisk(ctx, moonX, moonY);
    this.drawMoonAtmosphericVeil(ctx, moonX, moonY, seconds, nearR, breath);
  }

  /** Soft cool disk with stable surface mottling — not a white sticker. */
  private drawMoonDisk(ctx: CanvasRenderingContext2D, moonX: number, moonY: number): void {
    const R = MOON_RADIUS_PX;

    ctx.save();
    ctx.beginPath();
    ctx.arc(moonX, moonY, R, 0, Math.PI * 2);

    const disk = ctx.createRadialGradient(
      moonX - R * 0.2,
      moonY - R * 0.16,
      R * 0.08,
      moonX,
      moonY,
      R * 1.04,
    );
    disk.addColorStop(0, 'rgba(188, 198, 212, 0.68)');
    disk.addColorStop(0.45, 'rgba(162, 174, 192, 0.58)');
    disk.addColorStop(0.82, 'rgba(128, 142, 164, 0.42)');
    disk.addColorStop(1, 'rgba(98, 114, 136, 0.18)');
    ctx.fillStyle = disk;
    ctx.fill();

    ctx.clip();
    this.drawMoonSurfaceMottling(ctx, moonX, moonY, R);
    ctx.restore();

    // Feathered outer rim — no neon shadow.
    ctx.save();
    ctx.beginPath();
    ctx.arc(moonX, moonY, R + 2, 0, Math.PI * 2);
    const rim = ctx.createRadialGradient(moonX, moonY, R * 0.88, moonX, moonY, R + 3);
    rim.addColorStop(0, 'rgba(0,0,0,0)');
    rim.addColorStop(0.7, 'rgba(80, 100, 120, 0.08)');
    rim.addColorStop(1, 'rgba(60, 80, 100, 0.14)');
    ctx.fillStyle = rim;
    ctx.fill();
    ctx.restore();
  }

  /** Deterministic cloudy patches clipped inside the moon disk. */
  private drawMoonSurfaceMottling(
    ctx: CanvasRenderingContext2D,
    moonX: number,
    moonY: number,
    R: number,
  ): void {
    const seed = moonX * 0.013 + moonY * 0.017;

    for (let i = 0; i < 4; i++) {
      const ang = hash1(seed + i * 4.7) * Math.PI * 2;
      const dist = R * (0.12 + hash1(seed + i * 2.1) * 0.42);
      const px = moonX + Math.cos(ang) * dist;
      const py = moonY + Math.sin(ang) * dist * 0.82;
      const patchR = R * (0.26 + hash1(seed + i * 7.3) * 0.24);
      const patchA = 0.035 + hash1(seed + i * 11.2) * 0.03;

      const patch = ctx.createRadialGradient(px, py, 0, px, py, patchR);
      patch.addColorStop(0, `rgba(68, 82, 100, ${patchA})`);
      patch.addColorStop(0.55, `rgba(82, 96, 114, ${patchA * 0.45})`);
      patch.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = patch;
      ctx.fillRect(px - patchR, py - patchR, patchR * 2, patchR * 2);
    }

    for (let i = 0; i < 5; i++) {
      const nx = moonX + (hash1(seed + i * 3.3) - 0.5) * R * 1.1;
      const ny = moonY + (hash1(seed + i * 5.9) - 0.5) * R * 0.9;
      const cloudR = R * (0.35 + hash1(seed + i * 8.1) * 0.3);
      const cloudA = 0.018 + hash1(seed + i * 13.4) * 0.015;
      const cloud = ctx.createRadialGradient(nx, ny, 0, nx, ny, cloudR);
      cloud.addColorStop(0, `rgba(108, 122, 142, ${cloudA})`);
      cloud.addColorStop(0.6, `rgba(88, 102, 122, ${cloudA * 0.35})`);
      cloud.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = cloud;
      ctx.fillRect(nx - cloudR, ny - cloudR, cloudR * 2, cloudR * 2);
    }
  }

  /** Slow fog veil partially veiling disk and halo — moon feels embedded in atmosphere. */
  private drawMoonAtmosphericVeil(
    ctx: CanvasRenderingContext2D,
    moonX: number,
    moonY: number,
    seconds: number,
    haloRadius: number,
    breath: number,
  ): void {
    ctx.save();
    ctx.globalCompositeOperation = 'source-over';

    for (let i = 0; i < 3; i++) {
      const drift = seconds * 0.016 + i * 2.7;
      const ox = moonX + smoothNoise1D(drift, 51 + i) * haloRadius * 0.42;
      const oy = moonY + smoothNoise1D(drift + 6.3, 72 + i) * haloRadius * 0.26;
      const vR = haloRadius * (0.75 + i * 0.12) * breath;
      const veilA = (0.045 + i * 0.012) * (0.9 + (breath - 1) * 0.3);

      ctx.filter = `blur(${22 + i * 6}px)`;
      const veil = ctx.createRadialGradient(ox, oy, vR * 0.08, ox, oy, vR);
      veil.addColorStop(0, `rgba(120, 140, 162, ${veilA})`);
      veil.addColorStop(0.45, `rgba(90, 112, 135, ${veilA * 0.55})`);
      veil.addColorStop(0.8, `rgba(70, 90, 112, ${veilA * 0.2})`);
      veil.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = veil;
      ctx.fillRect(ox - vR, oy - vR, vR * 2, vR * 2);
    }

    // Slow smoky band drifting across part of the disk.
    const bandAng = seconds * 0.011 + 0.6;
    const bx = moonX + Math.cos(bandAng) * MOON_RADIUS_PX * 0.3;
    const by = moonY + Math.sin(bandAng) * MOON_RADIUS_PX * 0.2;
    ctx.filter = 'blur(14px)';
    const band = ctx.createLinearGradient(
      bx - MOON_RADIUS_PX,
      by - MOON_RADIUS_PX * 0.4,
      bx + MOON_RADIUS_PX,
      by + MOON_RADIUS_PX * 0.4,
    );
    band.addColorStop(0, 'rgba(0,0,0,0)');
    band.addColorStop(0.35, 'rgba(100, 118, 138, 0.06)');
    band.addColorStop(0.55, 'rgba(115, 132, 152, 0.09)');
    band.addColorStop(0.75, 'rgba(90, 108, 128, 0.05)');
    band.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = band;
    ctx.beginPath();
    ctx.arc(moonX, moonY, MOON_RADIUS_PX + 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.filter = 'none';
    ctx.globalCompositeOperation = 'source-over';
    ctx.restore();
  }

  /** Organic smoky halo edge — deforms radius only, not center. */
  private drawOrganicMoonHalo(
    ctx: CanvasRenderingContext2D,
    moonX: number,
    moonY: number,
    seconds: number,
    breath: number,
    haloRadius: number,
  ): void {
    const samples = 64;

    ctx.save();
    ctx.globalCompositeOperation = 'screen';
    ctx.filter = 'blur(16px)';

    for (let ring = 0; ring < 3; ring++) {
      const baseRadius = (haloRadius * (0.72 + ring * 0.18)) * breath;
      const alpha = (0.032 - ring * 0.008) * breath * MOON_GLOW_GAIN;

      ctx.beginPath();
      for (let i = 0; i <= samples; i++) {
        const angle = (i / samples) * Math.PI * 2;
        const angleSeed = angle * 3.8 + ring * 7.1;
        const n =
          0.55 * smoothNoise1D(angleSeed + seconds * 0.035, 41) +
          0.35 * smoothNoise1D(angleSeed * 2.1 - seconds * 0.022, 62);
        const r = baseRadius * (1 + n * 0.065);
        const px = moonX + Math.cos(angle) * r;
        const py = moonY + Math.sin(angle) * r;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();

      const g = ctx.createRadialGradient(
        moonX,
        moonY,
        baseRadius * 0.22,
        moonX,
        moonY,
        baseRadius * 1.14,
      );
      g.addColorStop(0, `rgba(130, 165, 195, ${alpha})`);
      g.addColorStop(0.55, `rgba(100, 140, 175, ${alpha * 0.45})`);
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.fill();
    }

    ctx.filter = 'none';
    ctx.globalCompositeOperation = 'source-over';
    ctx.restore();
  }

  /** Slow smoky wisps orbiting the moon glow. */
  private drawMoonFogWisps(
    ctx: CanvasRenderingContext2D,
    moonX: number,
    moonY: number,
    seconds: number,
    fogT: number,
    haloRadius: number,
    breath: number,
  ): void {
    const wispCount = 8;

    ctx.save();
    ctx.globalCompositeOperation = 'screen';
    ctx.filter = 'blur(26px)';

    for (let i = 0; i < wispCount; i++) {
      const seed = i * 13.7 + 4.2;
      const drift = seconds * 0.03 + seed;
      const wx = moonX + Math.cos(drift * 0.7 + seed) * haloRadius * 0.45;
      const wy = moonY + Math.sin(drift * 0.5 + seed) * haloRadius * 0.28;
      const wR = haloRadius * (0.55 + smoothNoise1D(fogT * 0.5 + seed, 71) * 0.35);
      const wispBreath = 0.85 + 0.15 * smoothNoise1D(drift * 0.4 + seed, 19);
      const alpha = 0.032 * MOON_WISP_GAIN * wispBreath * breath;

      const wisp = ctx.createRadialGradient(wx, wy, 0, wx, wy, wR);
      wisp.addColorStop(0, `rgba(170, 190, 210, ${alpha})`);
      wisp.addColorStop(0.45, `rgba(130, 160, 185, ${alpha * 0.4})`);
      wisp.addColorStop(0.75, `rgba(90, 125, 155, ${alpha * 0.16})`);
      wisp.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = wisp;
      ctx.fillRect(wx - wR, wy - wR, wR * 2, wR * 2);
    }

    // Faint vertical smears below the moon.
    for (let i = 0; i < 3; i++) {
      const seed = i * 5.1 + 2.3;
      const sx = moonX + (i - 1) * 22 + smoothNoise1D(fogT + seed, 97) * 14;
      const sy = moonY + MOON_RADIUS_PX + 6;
      const smearH = 80 + smoothNoise1D(fogT * 0.6 + i, 44) * 40;
      const smearW = 32 + i * 6;
      const smearA = 0.012 * MOON_WISP_GAIN * breath;
      const smear = ctx.createLinearGradient(sx, sy, sx, sy + smearH);
      smear.addColorStop(0, `rgba(140, 165, 185, ${smearA})`);
      smear.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = smear;
      ctx.fillRect(sx - smearW / 2, sy, smearW, smearH);
    }

    ctx.filter = 'none';
    ctx.globalCompositeOperation = 'source-over';
    ctx.restore();
  }

  private drawMoonReflection(moonX: number, moonY: number, parallaxY: number): void {
    const { ctx, height } = this;
    const seconds = animSeconds(this.animTime);
    const breath = moonBreathAt(seconds);
    const horizon = height * HORIZON_NORM_Y;
    const pathDepth = (height - horizon) * 0.82;
    const moonT = Math.max(0, Math.min(1, (moonY / height - 0.08) / (0.42 - 0.08)));
    // Lower moon => brighter/wider; higher moon => narrower/fainter.
    const heightBright = lerp(0.95, 0.68, moonT);
    const heightWidth = lerp(1.22, 0.88, moonT);
    const horizonBoost = lerp(1.08, 0.88, moonT) * breath;

    ctx.save();
    ctx.globalCompositeOperation = 'screen';

    // Origin bloom where reflection meets horizon under moon.
    const originR = 150 * heightWidth * breath;
    const originGlow = ctx.createRadialGradient(moonX, horizon, 0, moonX, horizon, originR);
    const originA = 0.09 * heightBright * horizonBoost * MOON_REFLECT_GAIN;
    originGlow.addColorStop(0, `rgba(140, 185, 210, ${originA})`);
    originGlow.addColorStop(0.45, `rgba(90, 145, 175, ${originA * 0.38})`);
    originGlow.addColorStop(0.75, `rgba(60, 110, 145, ${originA * 0.12})`);
    originGlow.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = originGlow;
    ctx.fillRect(moonX - originR * 1.3, horizon - 35, originR * 2.6, 90);

    // Wide blurred glow under the path — soft, water-tinted.
    ctx.filter = 'blur(22px)';
    for (let y = horizon; y < horizon + pathDepth; y += 6) {
      const progress = (y - horizon) / (height - horizon);
      const fade = (1 - progress) * (1 - progress);
      const surfaceY = this.combinedSurfaceY(moonX, seconds);
      const wavePull = (surfaceY - horizon) * 0.16 * (1 - progress);
      const wobble =
        Math.sin(seconds * 1.5 + y * 0.022 + parallaxY * 0.08) * (3 + progress * 10) +
        fractalNoise(y * 0.018, seconds * 0.45, 2, 42) * (6 + progress * 5);
      const xCenter = moonX + wavePull + wobble;
      const glowW = lerp(140, 420, progress) * (0.85 + hash(y * 0.12) * 0.35);
      const glowA = fade * lerp(0.16, 0.05, progress) * horizonBoost * MOON_REFLECT_GAIN;
      ctx.fillStyle = `rgba(90, 145, 175, ${glowA * heightBright})`;
      ctx.fillRect(xCenter - (glowW * heightWidth) / 2, y, glowW * heightWidth, 10);
    }
    ctx.filter = 'none';

    // Thin soft core near the horizon.
    ctx.save();
    ctx.globalCompositeOperation = 'screen';
    ctx.filter = 'blur(3px)';
    for (let y = horizon; y < horizon + pathDepth * 0.45; y += 4) {
      const progress = (y - horizon) / (height - horizon);
      const fade = (1 - progress) * (1 - progress);
      const surfaceY = this.combinedSurfaceY(moonX, seconds);
      const wavePull = (surfaceY - horizon) * 0.16 * (1 - progress);
      const wobble =
        Math.sin(seconds * 1.9 + y * 0.03 + parallaxY * 0.08) * (2 + progress * 8) +
        fractalNoise(y * 0.02, seconds * 0.55, 2, 24) * (4 + progress * 4);
      const xCenter = moonX + wavePull + wobble;
      const coreW = lerp(40, 120, progress);
      const coreA = fade * lerp(0.22, 0.07, progress) * horizonBoost * MOON_REFLECT_GAIN;
      ctx.fillStyle = `rgba(160, 200, 220, ${coreA * heightBright})`;
      ctx.fillRect(xCenter - (coreW * heightWidth) / 2, y, coreW * heightWidth, 2.4);
    }
    ctx.filter = 'none';
    ctx.restore();

    let y = horizon;
    while (y < height) {
      const progress = (y - horizon) / (height - horizon);
      const fade = (1 - progress) * (1 - progress) * (1 - progress * 0.25);
      const surfaceY = this.combinedSurfaceY(moonX, seconds);
      const wavePull = (surfaceY - horizon) * 0.16 * (1 - progress);
      const wobble =
        Math.sin(seconds * 1.6 + y * 0.025 + parallaxY * 0.08) * (4 + progress * 14) +
        fractalNoise(y * 0.02, seconds * 0.5, 2, 42) * (8 + progress * 6);
      const xCenter = moonX + wavePull + wobble;

      const gapBreak = hash(Math.floor(y * 0.17) + Math.floor(seconds * 0.4)) > 0.82;
      if (!gapBreak) {
        const segW =
          lerp(26, 170, progress) *
          (0.55 + 0.45 * Math.sin(seconds * 2.2 + y * 0.04)) *
          (0.65 + hash(y * 0.1) * 0.85);
        const flicker = 0.48 + 0.52 * Math.sin(seconds * 3.8 + y * 0.07);
        const alpha = fade * flicker * lerp(0.55, 0.18, progress) * MOON_REFLECT_GAIN;

        ctx.strokeStyle = `rgba(120, 175, 205, ${alpha * heightBright})`;
        ctx.lineWidth = lerp(3.4, 0.6, progress) * (0.55 + hash(y) * 0.85);
        ctx.beginPath();
        ctx.moveTo(xCenter - (segW * heightWidth) / 2, y);
        ctx.lineTo(xCenter + (segW * heightWidth) / 2, y);
        ctx.stroke();

        if (progress < 0.55 && hash(y + seconds) > 0.42) {
          const bw = 5 + hash(y * 2) * 16;
          ctx.fillStyle = `rgba(160, 200, 218, ${alpha * 0.55})`;
          ctx.fillRect(xCenter - bw / 2, y - 0.5, bw, 2.5);
        }
      }

      y += 2 + hash(y * 0.3) * 4;
    }

    ctx.globalCompositeOperation = 'source-over';
    ctx.restore();
  }

  private drawHorizonHaze(): void {
    const { ctx, width, height } = this;
    const horizon = height * HORIZON_NORM_Y;
    const seconds = animSeconds(this.animTime);
    const fogT = seconds;
    const breath = moonBreathAt(seconds);
    const moonGlowX = this.moonRenderPx().x;

    ctx.save();
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
    ctx.filter = 'none';

    // Soft sky haze above horizon — vertical feather, no flat band.
    const skyMistTop = horizon - 160;
    const skyMist = ctx.createLinearGradient(0, skyMistTop, 0, horizon + 8);
    skyMist.addColorStop(0, 'rgba(0,0,0,0)');
    skyMist.addColorStop(0.25, `rgba(12, 36, 56, ${0.04 * HORIZON_HAZE_GAIN})`);
    skyMist.addColorStop(0.55, `rgba(10, 30, 48, ${0.07 * HORIZON_HAZE_GAIN})`);
    skyMist.addColorStop(0.82, `rgba(8, 24, 40, ${0.05 * HORIZON_HAZE_GAIN})`);
    skyMist.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = skyMist;
    ctx.fillRect(0, skyMistTop - 10, width, horizon - skyMistTop + 30);

    // Uneven mist pockets — radial, not horizontal stripes.
    ctx.filter = 'blur(36px)';
    const pocketCount = Math.max(14, Math.floor(width / 90));
    for (let i = 0; i < pocketCount; i++) {
      const px = (i / pocketCount) * width + smoothNoise1D(i * 2.1 + fogT * 0.3, 55) * 40;
      const n = fbm1D(px * 0.004 + fogT * 0.28, 55, 3);
      const pocketY = horizon - 38 + n * 22 + smoothNoise1D(px * 0.006, 12) * 14;
      const pocketR = 50 + (n + 1) * 28;
      const pocketA = (0.028 + (n + 1) * 0.01) * HORIZON_HAZE_GAIN;
      const pocket = ctx.createRadialGradient(px, pocketY, 0, px, pocketY, pocketR);
      pocket.addColorStop(0, `rgba(16, 44, 66, ${pocketA})`);
      pocket.addColorStop(0.55, `rgba(12, 34, 52, ${pocketA * 0.45})`);
      pocket.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = pocket;
      ctx.fillRect(px - pocketR, pocketY - pocketR, pocketR * 2, pocketR * 2);
    }
    ctx.filter = 'none';

    // Very thin dark feather at horizon line.
    const separator = ctx.createLinearGradient(0, horizon - 10, 0, horizon + 14);
    separator.addColorStop(0, 'rgba(0,0,0,0)');
    separator.addColorStop(0.48, `rgba(2, 8, 14, ${0.14 * HORIZON_HAZE_GAIN})`);
    separator.addColorStop(0.54, `rgba(2, 8, 14, ${0.11 * HORIZON_HAZE_GAIN})`);
    separator.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = separator;
    ctx.fillRect(0, horizon - 14, width, 32);

    // Blue-gray mist on the water — gradual vertical falloff.
    const nearMist = ctx.createLinearGradient(0, horizon - 4, 0, horizon + height * 0.22);
    nearMist.addColorStop(0, 'rgba(0,0,0,0)');
    nearMist.addColorStop(0.12, `rgba(10, 32, 50, ${0.22 * HORIZON_HAZE_GAIN})`);
    nearMist.addColorStop(0.38, `rgba(7, 22, 38, ${0.18 * HORIZON_HAZE_GAIN})`);
    nearMist.addColorStop(0.68, `rgba(4, 14, 26, ${0.1 * HORIZON_HAZE_GAIN})`);
    nearMist.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = nearMist;
    ctx.fillRect(0, horizon - 8, width, height * 0.26);

    // Moon-following wide haze — irregular radial layers, not a spotlight.
    ctx.globalCompositeOperation = 'screen';
    ctx.filter = 'blur(42px)';

    const radialW = width * 0.68 + smoothNoise1D(fogT + 8.2, 66) * 60;
    const radial = ctx.createRadialGradient(
      moonGlowX,
      horizon + 4,
      0,
      moonGlowX,
      horizon + 30,
      radialW,
    );
    const radialCore = 0.055 * breath * HORIZON_HAZE_GAIN;
    radial.addColorStop(0, `rgba(100, 155, 190, ${radialCore})`);
    radial.addColorStop(0.3, `rgba(70, 125, 160, ${radialCore * 0.5})`);
    radial.addColorStop(0.65, `rgba(40, 85, 120, ${radialCore * 0.18})`);
    radial.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = radial;
    ctx.fillRect(moonGlowX - radialW, horizon - 55, radialW * 2, height * 0.3);

    const wideGlowW = width * 0.82;
    const wideGlow = ctx.createLinearGradient(
      moonGlowX - wideGlowW,
      horizon - 20,
      moonGlowX + wideGlowW,
      horizon + 40,
    );
    const glowCore = 0.045 * breath * HORIZON_HAZE_GAIN;
    wideGlow.addColorStop(0, 'rgba(0,0,0,0)');
    wideGlow.addColorStop(0.32, `rgba(60, 110, 150, ${glowCore * 0.28})`);
    wideGlow.addColorStop(0.5, `rgba(90, 145, 180, ${glowCore})`);
    wideGlow.addColorStop(0.68, `rgba(60, 110, 150, ${glowCore * 0.28})`);
    wideGlow.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = wideGlow;
    ctx.fillRect(0, horizon - 45, width, height * 0.22);

    ctx.filter = 'none';
    ctx.globalCompositeOperation = 'source-over';

    // Faint uneven glints along horizon.
    ctx.globalCompositeOperation = 'screen';
    ctx.filter = 'blur(16px)';
    const glintStep = Math.max(32, Math.floor(width / 18));
    for (let x = 0; x < width; x += glintStep) {
      const n = fbm1D(x * 0.006 + fogT * 0.18, 91, 2);
      const moonProx = Math.exp(-Math.pow((x - moonGlowX) / (width * 0.28), 2));
      const glintA =
        (0.01 + (n + 1) * 0.006) * (0.45 + moonProx * 0.75) * breath * HORIZON_GLINT_GAIN;
      const glintY = horizon + n * 14 + smoothNoise1D(x * 0.01 + fogT, 33) * 8;
      const glintR = glintStep * (0.5 + hash1(x * 0.17) * 0.45);
      const glint = ctx.createRadialGradient(x, glintY, 0, x, glintY, glintR);
      glint.addColorStop(0, `rgba(100, 155, 185, ${glintA})`);
      glint.addColorStop(0.6, `rgba(70, 120, 155, ${glintA * 0.35})`);
      glint.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = glint;
      ctx.fillRect(x - glintR, glintY - glintR, glintR * 2, glintR * 2);
    }

    ctx.filter = 'none';
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    ctx.restore();
  }

  private drawOceanBase(): void {
    const { ctx, width, height } = this;
    const top = height * 0.33;
    ctx.save();
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
    ctx.filter = 'none';

    // Safety base: opaque ocean under all ribbons so no cracks show through.
    ctx.fillStyle = '#010508';
    ctx.fillRect(0, top - 4, width, height - top + 8);

    const grad = ctx.createLinearGradient(0, height * 0.38, 0, height);
    grad.addColorStop(0, '#071a28');
    grad.addColorStop(0.14, '#061420');
    grad.addColorStop(0.32, '#04121c');
    grad.addColorStop(0.52, '#030c14');
    grad.addColorStop(0.72, '#020810');
    grad.addColorStop(1, '#010508');
    ctx.fillStyle = grad;
    ctx.fillRect(0, height * 0.38 - 2, width, height * 0.62 + 4);
    ctx.restore();
  }

  private drawDepthMist(): void {
    const { ctx, width, height } = this;
    const top = height * 0.38;
    const bottom = height * 0.78;

    ctx.save();
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
    ctx.filter = 'none';

    const mist = ctx.createLinearGradient(0, top, 0, bottom);
    mist.addColorStop(0, 'rgba(12, 40, 64, 0.62)');
    mist.addColorStop(0.28, 'rgba(7, 26, 44, 0.42)');
    mist.addColorStop(0.6, 'rgba(4, 14, 26, 0.18)');
    mist.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = mist;
    ctx.fillRect(0, top, width, bottom - top);

    // Distance fog: far ocean is bluer, softer, lower contrast.
    const distance = ctx.createLinearGradient(0, top - 40, 0, height * 0.72);
    distance.addColorStop(0, 'rgba(80, 160, 205, 0.07)');
    distance.addColorStop(0.35, 'rgba(35, 95, 135, 0.06)');
    distance.addColorStop(0.7, 'rgba(10, 28, 46, 0.03)');
    distance.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = distance;
    ctx.fillRect(0, top - 60, width, height * 0.5);

    ctx.save();
    ctx.globalAlpha = 0.18;
    for (let i = 0; i < 7; i++) {
      const bandY = top + (bottom - top) * (i / 7) + Math.sin(this.animTime * 0.12 + i * 1.1) * 10;
      const band = ctx.createLinearGradient(0, bandY - 24, 0, bandY + 36);
      band.addColorStop(0, 'rgba(0,0,0,0)');
      band.addColorStop(0.5, 'rgba(12, 38, 58, 0.36)');
      band.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = band;
      ctx.fillRect(0, bandY - 24, width, 60);
    }
    ctx.restore();
    ctx.restore();
  }

  private drawHorizonRibbon(ribbon: RibbonParams, points: { x: number; y: number }[]): void {
    const { ctx } = this;
    const fillTo = this.ribbonFillExtent(ribbon, points);

    this.drawRibbonWatertightUnderpaint(ribbon, points, fillTo);

    ctx.save();
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
    ctx.filter = `blur(${ribbon.blur}px)`;

    this.traceRibbonPath(points, fillTo);

    const grad = ctx.createLinearGradient(0, ribbon.yBase - ribbon.amp, 0, fillTo);
    grad.addColorStop(0, `rgba(5, 24, 42, ${ribbon.opacity * 0.55})`);
    grad.addColorStop(0.12, `rgba(3, 14, 26, ${ribbon.opacity * 0.75})`);
    grad.addColorStop(0.35, 'rgba(2, 10, 18, 0.92)');
    grad.addColorStop(1, 'rgba(1, 5, 10, 1)');
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.restore();
  }

  private drawMidRibbon(ribbon: RibbonParams, points: { x: number; y: number }[]): void {
    const { ctx } = this;
    const fillTo = this.ribbonFillExtent(ribbon, points);

    this.drawRibbonWatertightUnderpaint(ribbon, points, fillTo);

    ctx.save();
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
    if (ribbon.blur > 0) ctx.filter = `blur(${ribbon.blur}px)`;

    this.traceRibbonPath(points, fillTo);

    const darkness = lerp(0.28, 0.62, ribbon.depth);
    const grad = ctx.createLinearGradient(0, ribbon.yBase - ribbon.amp, 0, fillTo);
    grad.addColorStop(0, `rgba(8, 38, 58, ${ribbon.opacity * 0.62})`);
    grad.addColorStop(0.1, `rgba(5, 24, 40, ${ribbon.opacity * 0.82})`);
    grad.addColorStop(0.28, `rgba(3, 14, 26, ${darkness * 0.95})`);
    grad.addColorStop(0.55, `rgba(2, 8, 16, ${darkness})`);
    grad.addColorStop(1, `rgba(1, 5, 12, 1)`);
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.restore();
  }

  private drawForegroundRibbon(ribbon: RibbonParams, points: { x: number; y: number }[]): void {
    const { ctx } = this;
    const fillTo = this.ribbonFillExtent(ribbon, points);

    this.drawRibbonWatertightUnderpaint(ribbon, points, fillTo);

    ctx.save();
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
    ctx.filter = 'none';

    this.traceRibbonPath(points, fillTo);

    const darkness = lerp(0.68, 0.98, ribbon.depth);
    const crestLift = ribbon.depth > 0.82 ? 1.3 : 1.08;
    const grad = ctx.createLinearGradient(0, ribbon.yBase - ribbon.amp * 1.4, 0, fillTo);
    grad.addColorStop(0, `rgba(10, 44, 66, ${ribbon.opacity * 0.5 * crestLift})`);
    grad.addColorStop(0.04, `rgba(6, 28, 46, ${ribbon.opacity * 0.65})`);
    grad.addColorStop(0.1, `rgba(3, 14, 26, ${darkness * 0.98})`);
    grad.addColorStop(0.3, `rgba(2, 8, 16, ${darkness})`);
    grad.addColorStop(1, `rgba(1, 3, 8, 1)`);
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.restore();

    this.drawCrestFoamFragments(points, ribbon);
  }

  private drawCrestFoamFragments(
    points: { x: number; y: number }[],
    ribbon: RibbonParams,
  ): void {
    const { ctx } = this;
    const time = this.animTime;
    let crestStart = -1;

    const flushCrest = (start: number, end: number) => {
      if (end - start < 2) return;
      const segLen = end - start + 1;
      const fragmentCount = Math.max(2, Math.floor(segLen / 5));

      for (let f = 0; f < fragmentCount; f++) {
        const t = f / fragmentCount;
        const idx = start + Math.floor(t * (segLen - 1));
        const p = points[idx];
        if (!p) continue;
        if (hash(idx + ribbon.index + f) < (ribbon.depth > 0.85 ? 0.17 : 0.3)) continue;

        const nearCrest = ribbon.depth > 0.85;
        const shimmer = this.detailShimmerAt(p.x, time, ribbon.detailWaves);
        const flicker = 0.52 + 0.48 * Math.sin(time * 5 + p.x * 0.05);
        const depthBoost = nearCrest ? 1.5 : lerp(0.88, 1.12, ribbon.depth);
        const alpha =
          ribbon.opacity *
          flicker *
          ribbon.peakStrength *
          lerp(0.28, 0.78, ribbon.depth) *
          depthBoost *
          shimmer;
        const fragW = (nearCrest ? 7 : 4) + hash(p.x) * (nearCrest ? 18 : 14);
        const fragH = 1.5 + hash(p.y) * (nearCrest ? 4.5 : 3);

        ctx.save();
        ctx.shadowColor = `rgba(130, 255, 255, ${alpha * (nearCrest ? 1.05 : 0.9)})`;
        ctx.shadowBlur = (nearCrest ? 12 : 6) + ribbon.peakStrength * 9;
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, fragW);
        g.addColorStop(0, `rgba(255, 255, 255, ${Math.min(1, alpha * 1.15)})`);
        g.addColorStop(0.3, `rgba(${nearCrest ? 175 : 150}, 255, 255, ${alpha * 0.75})`);
        g.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.ellipse(p.x, p.y - fragH * 0.3, fragW, fragH, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    };

    for (let i = 1; i < points.length - 1; i++) {
      const p = points[i]!;
      const isCrest = p.y < points[i - 1]!.y && p.y < points[i + 1]!.y;
      if (isCrest && crestStart < 0) crestStart = i - 1;
      else if (!isCrest && crestStart >= 0) {
        flushCrest(crestStart, i);
        crestStart = -1;
      }
    }
    if (crestStart >= 0) flushCrest(crestStart, points.length - 1);
  }

  private drawRibbon(ribbon: RibbonParams): void {
    const { ctx } = this;
    ctx.save();
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
    const points = this.sampleRibbon(ribbon, this.animTime);
    switch (ribbon.tier) {
      case 'horizon':
        this.drawHorizonRibbon(ribbon, points);
        break;
      case 'mid':
        this.drawMidRibbon(ribbon, points);
        break;
      case 'foreground':
        this.drawForegroundRibbon(ribbon, points);
        break;
    }
    ctx.restore();
  }

  private drawFoam(): void {
    const { ctx } = this;
    const time = this.animTime;

    for (const p of this.foam) {
      const t = p.life / p.maxLife;
      const flicker = 0.55 + 0.45 * Math.sin(time * 10 + p.x * 0.1);
      const a = t * p.brightness * flicker;

      if (p.kind === 'streak') {
        ctx.save();
        ctx.strokeStyle = `rgba(90, 248, 255, ${a * 0.78})`;
        ctx.lineWidth = 1.2;
        ctx.shadowColor = `rgba(80, 235, 255, ${a})`;
        ctx.shadowBlur = 10;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x - p.size * 0.7, p.y + 1);
        ctx.stroke();
        ctx.restore();
        continue;
      }

      if (p.kind === 'twin' || p.kind === 'spark' || p.kind === 'spray') {
        ctx.shadowColor = 'rgba(110, 252, 255, 0.95)';
        ctx.shadowBlur = p.kind === 'twin' ? 10 : 6;
        ctx.fillStyle = `rgba(${p.kind === 'twin' ? 200 : 180}, 255, 255, ${a})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
        continue;
      }

      ctx.shadowColor = 'rgba(95, 255, 255, 0.95)';
      ctx.shadowBlur = 13 + p.brightness * 10;
      const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.size * 3.4);
      g.addColorStop(0, `rgba(255, 255, 255, ${a * 1.22})`);
      g.addColorStop(0.16, `rgba(185, 255, 255, ${a * 0.98})`);
      g.addColorStop(0.5, `rgba(70, 205, 240, ${a * 0.22})`);
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size * 3.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }

  private drawMidSheen(): void {
    const { ctx, width, height } = this;
    const time = this.animTime;
    const horizon = height * 0.42;
    const detailWaves = this.spectrum.detailWaves;

    // Subtle "alive" sheen to reduce mid-ocean dead space without reading as UI or math.
    ctx.save();
    ctx.globalCompositeOperation = 'screen';
    ctx.filter = 'blur(12px)';

    for (let i = 0; i < 6; i++) {
      const bandT = (i + 1) / 7;
      const y = lerp(height * 0.5, height * 0.78, Math.pow(bandT, 1.1));
      const wobble = fractalNoise(i * 7.1, time * 0.22, 2, 99) * 18 + Math.sin(time * 0.18 + i) * 10;
      const bandY = Math.max(horizon + 10, y + wobble);
      const shimmer = this.detailShimmerAt(width * (0.35 + i * 0.08), time, detailWaves);
      const alpha = (0.04 + 0.03 * (1 - bandT)) * shimmer;

      const g = ctx.createLinearGradient(0, bandY - 40, 0, bandY + 60);
      g.addColorStop(0, 'rgba(0,0,0,0)');
      g.addColorStop(0.5, `rgba(120, 220, 255, ${alpha})`);
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, bandY - 50, width, 120);
    }

    ctx.filter = 'none';
    ctx.restore();
  }

  private drawMathReveal(): void {
    const a = this.mathReveal;
    if (a < 0.02) return;

    const { ctx, width, height } = this;
    const meta = this.spectrum.metadata;
    const mobile = this.quality.mobile;

    ctx.fillStyle = `rgba(0, 2, 8, ${a * 0.1})`;
    ctx.fillRect(0, 0, width, height);

    const panelW = Math.min(mobile ? width - 32 : 420, width * 0.46);
    const panelX = 18;
    const panelY = 20;
    const panelH = Math.min(height - 100, mobile ? height * 0.72 : 560);
    const pad = 14;
    const innerW = panelW - pad * 2;

    this.drawGlassPanel(panelX, panelY, panelW, panelH, a);

    const font = mobile ? '9px' : '10px';
    const fontSm = mobile ? '8px' : '9px';
    const cyan = `rgba(130, 220, 255, ${0.75 * a})`;
    const cyanDim = `rgba(100, 180, 220, ${0.45 * a})`;
    const cyanFaint = `rgba(80, 160, 200, ${0.32 * a})`;
    const accent = `rgba(160, 240, 255, ${0.85 * a})`;

    let y = panelY + pad + 4;

    ctx.font = `600 ${mobile ? '10px' : '11px'} ui-monospace, "Cascadia Mono", monospace`;
    ctx.fillStyle = accent;
    ctx.fillText('PRIME TIDES', panelX + pad, y);
    ctx.font = `${fontSm} ui-monospace, monospace`;
    ctx.fillStyle = cyanDim;
    ctx.fillText('instrument panel', panelX + pad + 88, y);
    y += 18;

    ctx.strokeStyle = `rgba(70, 180, 230, ${0.2 * a})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(panelX + pad, y);
    ctx.lineTo(panelX + panelW - pad, y);
    ctx.stroke();
    y += 14;

    ctx.font = `${font} ui-monospace, monospace`;
    ctx.fillStyle = cyanDim;
    const fmt = (n: number) => n.toLocaleString('en-US');
    const sessionLines = [
      `seed ${meta.sessionSeed}`,
      `window gaps ${fmt(meta.primeWindowStart)}–${fmt(meta.primeWindowEnd)}`,
      `FFT ${fmt(meta.fftSamples)} samples · ${meta.dominantPeakCount} peaks`,
    ];
    for (const line of sessionLines) {
      ctx.fillText(line, panelX + pad, y);
      y += 13;
    }
    y += 6;

    y = this.drawPanelSection(ctx, panelX, pad, panelW, y, 'Dominant Frequencies', a);
    ctx.font = `${fontSm} ui-monospace, monospace`;
    ctx.fillStyle = cyanFaint;
    const col = {
      n: panelX + pad,
      freq: panelX + pad + 14,
      mag: panelX + pad + 68,
      amp: panelX + pad + 108,
      phase: panelX + pad + 142,
      speed: panelX + pad + 178,
      layer: panelX + pad + 214,
    };
    if (!mobile) {
      ctx.fillText('#', col.n, y);
      ctx.fillText('Freq', col.freq, y);
      ctx.fillText('Mag', col.mag, y);
      ctx.fillText('Amp', col.amp, y);
      ctx.fillText('Phase', col.phase, y);
      ctx.fillText('Speed', col.speed, y);
      ctx.fillText('Layer', col.layer, y);
      y += 12;
    }

    const peaks = meta.usedPeaks.slice(0, mobile ? 8 : 12);
    ctx.font = `${font} ui-monospace, monospace`;
    for (let i = 0; i < peaks.length; i++) {
      const p = peaks[i]!;
      ctx.fillStyle = `rgba(120, 210, 245, ${(0.55 + (1 - i / peaks.length) * 0.3) * a})`;
      if (mobile) {
        const line = `${i + 1}  f=${p.frequency.toFixed(5)}  mag=${p.magnitude.toFixed(3)}  amp=${p.normalizedAmplitude.toFixed(2)}  ${p.layer}`;
        ctx.fillText(line, panelX + pad, y);
      } else {
        ctx.fillText(String(i + 1), col.n, y);
        ctx.fillText(p.frequency.toFixed(5), col.freq, y);
        ctx.fillText(p.magnitude.toFixed(3), col.mag, y);
        ctx.fillText(p.normalizedAmplitude.toFixed(2), col.amp, y);
        ctx.fillText(p.phase.toFixed(2), col.phase, y);
        ctx.fillText(p.speed.toFixed(2), col.speed, y);
        ctx.fillText(p.layer, col.layer, y);
      }
      y += 12;
    }
    y += 8;

    y = this.drawPanelSection(ctx, panelX, pad, panelW, y, 'Prime Event Stats', a);
    ctx.font = `${font} ui-monospace, monospace`;
    ctx.fillStyle = cyan;
    const stats = meta.primeEventStats;
    const statLine1 = `twin prime gaps: ${fmt(stats.twinPrimeGaps)}`;
    const statLine2 = `small gaps ≤ 4: ${fmt(stats.smallGaps)}`;
    const statLine3 = `large gaps: ${fmt(stats.largeGaps)}`;
    const statLine4 = `spark events: ${fmt(stats.sparkEventsGenerated)}`;
    for (const line of [statLine1, statLine2, statLine3, statLine4]) {
      ctx.fillText(line, panelX + pad, y);
      y += 13;
    }
    y += 6;

    y = this.drawPanelSection(ctx, panelX, pad, panelW, y, 'Wave Model', a);
    ctx.font = `${font} ui-monospace, monospace`;
    ctx.fillStyle = accent;
    ctx.fillText('wave(x,t) = Σ aᵢ sin(fᵢx + φᵢ + t·sᵢ)', panelX + pad, y);
    y += 14;
    ctx.font = `${fontSm} ui-monospace, monospace`;
    ctx.fillStyle = cyanDim;
    const expl =
      'Ocean motion is generated from dominant frequencies extracted from the Fourier transform of mean-centered prime gaps.';
    y = this.wrapText(ctx, expl, panelX + pad, y, innerW, 12);
    y += 10;

    if (y + 90 < panelY + panelH) {
      y = this.drawPanelSection(ctx, panelX, pad, panelW, y, 'Signals', a);
      const chartH = 28;
      const { gapPreview, spectrumPreview } = meta;

      if (gapPreview.length > 1) {
        ctx.fillStyle = cyanFaint;
        ctx.font = `${fontSm} ui-monospace, monospace`;
        ctx.fillText('prime gaps', panelX + pad, y);
        y += 10;
        this.drawMiniSparkline(
          gapPreview.map((v, i) => ({ x: i / (gapPreview.length - 1), y: v })),
          panelX + pad,
          y,
          innerW,
          chartH,
          Math.max(...gapPreview.map(Math.abs), 1),
          a * 0.45,
        );
        y += chartH + 10;
      }

      if (spectrumPreview.length > 1 && y + chartH < panelY + panelH) {
        ctx.fillStyle = cyanFaint;
        ctx.fillText('FFT spectrum', panelX + pad, y);
        y += 10;
        const maxM = Math.max(...spectrumPreview.map((s) => s.mag), 1);
        this.drawMiniSparkline(
          spectrumPreview.map((s, i) => ({
            x: i / (spectrumPreview.length - 1),
            y: s.mag / maxM,
          })),
          panelX + pad,
          y,
          innerW,
          chartH,
          1,
          a * 0.5,
        );
      }
    }

    void width;
    void height;
  }

  private drawGlassPanel(x: number, y: number, w: number, h: number, a: number): void {
    const { ctx } = this;
    ctx.save();
    ctx.beginPath();
    const r = 10;
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();

    const grad = ctx.createLinearGradient(x, y, x, y + h);
    grad.addColorStop(0, `rgba(10, 28, 48, ${0.55 * a})`);
    grad.addColorStop(0.5, `rgba(6, 18, 32, ${0.62 * a})`);
    grad.addColorStop(1, `rgba(4, 12, 24, ${0.7 * a})`);
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.strokeStyle = `rgba(90, 210, 255, ${0.22 * a})`;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.strokeStyle = `rgba(160, 240, 255, ${0.08 * a})`;
    ctx.beginPath();
    ctx.moveTo(x + 1, y + 1);
    ctx.lineTo(x + w - 1, y + 1);
    ctx.stroke();
    ctx.restore();
  }

  private drawPanelSection(
    ctx: CanvasRenderingContext2D,
    panelX: number,
    pad: number,
    panelW: number,
    y: number,
    title: string,
    a: number,
  ): number {
    ctx.font = `600 9px ui-monospace, monospace`;
    ctx.fillStyle = `rgba(140, 230, 255, ${0.6 * a})`;
    ctx.fillText(title.toUpperCase(), panelX + pad, y);
    y += 12;
    ctx.strokeStyle = `rgba(60, 160, 210, ${0.15 * a})`;
    ctx.beginPath();
    ctx.moveTo(panelX + pad, y);
    ctx.lineTo(panelX + panelW - pad, y);
    ctx.stroke();
    return y + 10;
  }

  private wrapText(
    ctx: CanvasRenderingContext2D,
    text: string,
    x: number,
    y: number,
    maxWidth: number,
    lineHeight: number,
  ): number {
    const words = text.split(' ');
    let line = '';
    for (const word of words) {
      const test = line ? `${line} ${word}` : word;
      if (ctx.measureText(test).width > maxWidth && line) {
        ctx.fillText(line, x, y);
        y += lineHeight;
        line = word;
      } else {
        line = test;
      }
    }
    if (line) {
      ctx.fillText(line, x, y);
      y += lineHeight;
    }
    return y;
  }

  private drawMiniSparkline(
    points: { x: number; y: number }[],
    x: number,
    y: number,
    w: number,
    h: number,
    yScale: number,
    alpha: number,
  ): void {
    const { ctx } = this;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();

    ctx.fillStyle = `rgba(4, 16, 28, ${alpha * 0.35})`;
    ctx.fillRect(x, y, w, h);

    ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
      const p = points[i]!;
      const px = x + p.x * w;
      const py = y + h / 2 - (p.y / yScale) * (h / 2 - 2);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = `rgba(100, 210, 255, ${alpha})`;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();
  }

  private drawVignette(): void {
    const { ctx, width, height } = this;
    const g = ctx.createRadialGradient(
      width / 2,
      height * 0.4,
      height * 0.06,
      width / 2,
      height * 0.5,
      height * 0.98,
    );
    g.addColorStop(0, 'rgba(0,0,0,0)');
    g.addColorStop(1, 'rgba(0,0,0,0.6)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, width, height);
  }

  private draw(): void {
    const { ctx, width, height } = this;
    ctx.fillStyle = '#010508';
    ctx.fillRect(0, 0, width, height);

    // 1. Sky gradient + moon glow/halo/wisps + crisp moon disk.
    this.drawSky();

    // 2. Opaque ocean base (covers lower sky band).
    this.drawOceanBase();

    // 3. Horizon atmosphere on top of ocean so haze is not hidden.
    this.drawHorizonHaze();

    const ribbons = this.ribbonsCache.length ? this.ribbonsCache : this.buildRibbons();
    const horizon = ribbons.filter((r) => r.tier === 'horizon');
    const mid = ribbons.filter((r) => r.tier === 'mid');
    const fg = ribbons.filter((r) => r.tier === 'foreground');

    for (const ribbon of horizon) this.drawRibbon(ribbon);

    // 4. Moon reflection during water pass — after ocean + horizon ribbons.
    const reflX =
      this.moonReflection.x * width + (this.moonDragging ? 0 : this.currentParallaxX);
    const reflY =
      this.moonReflection.y * height + (this.moonDragging ? 0 : this.currentParallaxY);
    this.drawMoonReflection(reflX, reflY, this.currentParallaxY);

    for (const ribbon of mid) this.drawRibbon(ribbon);
    this.drawDepthMist();
    this.drawMidSheen();
    for (const ribbon of fg) this.drawRibbon(ribbon);

    this.drawFoam();
    this.drawMathReveal();
    this.drawVignette();
  }
}

function buildFallbackSparkEvents(seed: number): GapSparkEvent[] {
  const rng = new SeededRng(seed ^ 0xabc123);
  const events: GapSparkEvent[] = [];
  const count = 40 + rng.nextInt(0, 40);
  for (let i = 0; i < count; i++) {
    const position = rng.next();
    const roll = rng.next();
    if (roll < 0.12) events.push({ position, gap: 2, kind: 'twin', strength: 1 });
    else if (roll < 0.28) events.push({ position, gap: 4, kind: 'small', strength: 0.5 });
    else if (roll > 0.88) events.push({ position, gap: 14, kind: 'large', strength: 0.7 });
  }
  return events;
}

export function createFallbackSpectrum(sessionSeed: number, mobile: boolean): OceanSpectrum {
  const params = deriveSessionParams(sessionSeed, mobile);
  const rng = new SeededRng(sessionSeed);

  const mk = (
    freq: number,
    amp: number,
    phase: number,
    speed: number,
    peakStrength = 0.5,
  ): WaveComponent => ({ frequency: freq, amplitude: amp, phase, speed, peakStrength });

  const freqScale = 0.9 + rng.next() * 0.2;
  const phaseShift = params.phaseOffset;

  const windowStart = Math.floor(8000 + rng.next() * 12000);
  const windowSize = params.windowSize;
  const windowEnd = windowStart + windowSize - 1;

  const deepWaves = [
    mk(0.006 * freqScale, 22, 0.5 + phaseShift, 0.18, 0.6),
    mk(0.009 * freqScale, 18, 1.8 + phaseShift, 0.14, 0.5),
    mk(0.005 * freqScale, 24, 3.2 + phaseShift, 0.12, 0.45),
  ];
  const surfaceWaves = [
    mk(0.016 * freqScale, 16, 0.2 + phaseShift, 0.32, 0.85),
    mk(0.022 * freqScale, 13, 1.4 + phaseShift, 0.28, 0.7),
    mk(0.013 * freqScale, 15, 2.6 + phaseShift, 0.3, 0.65),
    mk(0.028 * freqScale, 10, 4.1 + phaseShift, 0.35, 0.55),
  ];
  const detailWaves = [
    mk(0.035 * freqScale, 8, 0.8 + phaseShift, 0.55, 0.4),
    mk(0.042 * freqScale, 6, 2.1 + phaseShift, 0.65, 0.35),
    mk(0.05 * freqScale, 5, 3.5 + phaseShift, 0.8, 0.3),
  ];

  const fallbackPeaks = (
    waves: WaveComponent[],
    layer: 'horizon' | 'midground' | 'foreground',
    freqBase: number,
  ) =>
    waves.map((w, i) => ({
      frequency: freqBase * (0.8 + i * 0.15),
      magnitude: w.peakStrength * 100,
      normalizedAmplitude: w.peakStrength,
      phase: w.phase,
      speed: w.speed,
      layer,
    }));

  const sparkEvents = buildFallbackSparkEvents(sessionSeed);
  const usedPeaks = [
    ...fallbackPeaks(deepWaves, 'horizon', 0.004),
    ...fallbackPeaks(surfaceWaves, 'midground', 0.012),
    ...fallbackPeaks(detailWaves, 'foreground', 0.028),
  ]
    .sort((a, b) => b.magnitude - a.magnitude)
    .slice(0, 12);

  return {
    deepWaves,
    surfaceWaves,
    detailWaves,
    metadata: {
      primeCount: 0,
      gapCount: 0,
      fftSamples: windowSize,
      dominantFrequencies: [0.016, 0.022, 0.013],
      peakMagnitudes: [],
      gapPreview: Array.from({ length: 120 }, (_, i) => {
        const t = i * 0.15 + sessionSeed * 0.00001;
        return Math.sin(t) * 4 + (i % 7) * 0.3;
      }),
      spectrumPreview: Array.from({ length: 80 }, (_, i) => ({
        freq: i / 80,
        mag: Math.exp(-i * 0.04) * 100 + (i % 11 === 0 ? 40 : 0),
      })),
      sparkEvents,
      sessionSeed,
      primeLimit: params.primeLimit,
      primeWindowStart: windowStart,
      primeWindowEnd: windowEnd,
      dominantPeakCount: params.topPeaks,
      usedPeaks,
      primeEventStats: {
        twinPrimeGaps: sparkEvents.filter((e) => e.kind === 'twin').length,
        smallGaps: sparkEvents.filter((e) => e.kind === 'small').length,
        largeGaps: sparkEvents.filter((e) => e.kind === 'large').length,
        sparkEventsGenerated: sparkEvents.length,
      },
    },
  };
}

export function detectRenderQuality(): RenderQuality {
  const mobile =
    /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent) ||
    window.innerWidth < 768;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const lowMemory =
    'deviceMemory' in navigator &&
    (navigator as Navigator & { deviceMemory?: number }).deviceMemory! < 4;

  return {
    mobile: mobile || lowMemory,
    reducedMotion,
    sampleStep: mobile || lowMemory ? 6 : 3,
    particleCap: mobile || lowMemory ? 80 : 220,
    layerCount: 3,
  };
}
