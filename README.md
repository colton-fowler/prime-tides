# Prime Tides

A fullscreen animated ocean background for a software engineer's portfolio. The waves are driven by frequency peaks extracted from the Fast Fourier Transform of prime number gaps — mathematics hidden beneath a moonlit seascape.

## Concept

1. Generate primes via the Sieve of Eratosthenes
2. Compute consecutive prime gaps
3. FFT the gap sequence
4. Map dominant spectral peaks to layered sine waves
5. Render a calm, moonlit Atlantic ocean on HTML Canvas

The viewer sees a beautiful ocean first. The mathematical origin reveals itself only on closer inspection.

[<video src="screenshots/showcase.mp4" controls width="900"></video>](https://github.com/colton-fowler/prime-tides/blob/main/screenshots/showcase.mp4)

## Stack

- React 19 + TypeScript
- Vite
- HTML Canvas 2D
- Web Worker for prime/FFT precomputation

## Run

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
npm run preview
```

## Integration

Use `PrimeTidesBackground` as a fixed fullscreen layer behind your portfolio content:

```tsx
import { PrimeTidesBackground } from './components/PrimeTidesBackground';

function Portfolio() {
  return (
    <>
      <PrimeTidesBackground />
      <main>{/* your content */}</main>
    </>
  );
}
```

## Performance

- FFT and prime sieving run once in a Web Worker at startup
- `requestAnimationFrame` drives the render loop
- Mobile devices use reduced prime limits, fewer particles, and coarser sampling
- `prefers-reduced-motion` disables animation and particles

## Mouse Interaction

- Gentle parallax on sky and moon
- Subtle localized wave distortion near cursor
- Soft ripples on click/tap
