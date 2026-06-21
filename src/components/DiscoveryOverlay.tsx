import type { OceanSpectrum, SpectrumSource } from '../types';

interface DiscoveryOverlayProps {
  open: boolean;
  onClose: () => void;
  spectrum: OceanSpectrum | null;
  source: SpectrumSource;
}

const PIPELINE = [
  'Prime Numbers',
  'Prime Gaps',
  'Fourier Transform',
  'Ocean Waves',
] as const;

export function DiscoveryOverlay({ open, onClose, spectrum, source }: DiscoveryOverlayProps) {
  if (!open) return null;

  const meta = spectrum?.metadata;
  const freqs = meta?.dominantFrequencies ?? [];
  const formattedFreqs = freqs
    .slice(0, 6)
    .map((f) => f.toFixed(5))
    .join(', ');

  return (
    <div className="discovery-overlay" role="dialog" aria-labelledby="discovery-title">
      <div className="discovery-overlay__backdrop" onClick={onClose} aria-hidden="true" />
      <div className="discovery-overlay__panel">
        <button type="button" className="discovery-overlay__close" onClick={onClose} aria-label="Close">
          ×
        </button>

        <h2 id="discovery-title" className="discovery-overlay__title">
          Beneath the Surface
        </h2>
        <p className="discovery-overlay__lead">
          This ocean is not random. Its motion is shaped by frequencies hidden inside the gaps between
          prime numbers.
        </p>

        <div className="discovery-overlay__pipeline">
          {PIPELINE.map((step, i) => (
            <div key={step} className="discovery-overlay__step">
              <span className="discovery-overlay__step-label">{step}</span>
              {i < PIPELINE.length - 1 && (
                <span className="discovery-overlay__arrow" aria-hidden="true">
                  ↓
                </span>
              )}
            </div>
          ))}
        </div>

        <dl className="discovery-overlay__stats">
          <div>
            <dt>Primes analyzed</dt>
            <dd>{meta?.primeCount?.toLocaleString() ?? '—'}</dd>
          </div>
          <div>
            <dt>FFT samples</dt>
            <dd>{meta?.fftSamples?.toLocaleString() ?? '—'}</dd>
          </div>
          <div>
            <dt>Dominant frequencies</dt>
            <dd className="discovery-overlay__freqs">{formattedFreqs || '—'}</dd>
          </div>
          <div>
            <dt>Worker status</dt>
            <dd className={`discovery-overlay__status discovery-overlay__status--${source.workerStatus}`}>
              {source.workerStatus}
            </dd>
          </div>
          <div>
            <dt>Spectrum source</dt>
            <dd>{source.isFallback ? 'Fallback (synthetic)' : 'Live FFT from prime gaps'}</dd>
          </div>
        </dl>

        <p className="discovery-overlay__hint">
          Press <kbd>P</kbd> to briefly glimpse the mathematics beneath the waves.
        </p>
      </div>
    </div>
  );
}
