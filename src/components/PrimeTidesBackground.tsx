import { useCallback, useEffect, useRef, useState } from 'react';
import {
  OceanRenderer,
  createFallbackSpectrum,
  detectRenderQuality,
} from '../ocean/OceanRenderer';
import { createSessionSeed } from '../math/sessionSeed';
import type { OceanSpectrum, PrimeFftWorkerResponse, SpectrumSource } from '../types';
import PrimeFftWorker from '../workers/primeFft.worker?worker';
import { DiscoveryHint } from './DiscoveryHint';
import { DiscoveryOverlay } from './DiscoveryOverlay';

interface PrimeTidesBackgroundProps {
  onReady?: (spectrum: OceanSpectrum) => void;
}

export function PrimeTidesBackground({ onReady }: PrimeTidesBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<OceanRenderer | null>(null);
  const frameRef = useRef<number | null>(null);
  const startTimeRef = useRef<number | null>(null);
  const revealMathRef = useRef(false);
  const sessionSeedRef = useRef<number | null>(null);
  const mouseRef = useRef({ x: 0, y: 0, normX: 0, normY: 0, active: false });
  const moonDragRef = useRef<{ dragging: boolean; pointerId: number | null }>({
    dragging: false,
    pointerId: null,
  });
  const [spectrum, setSpectrum] = useState<OceanSpectrum | null>(null);
  const [revealMath, setRevealMath] = useState(false);
  const [overlayOpen, setOverlayOpen] = useState(false);
  const [source, setSource] = useState<SpectrumSource>({
    isFallback: false,
    workerStatus: 'computing',
  });

  const toggleReveal = useCallback(() => {
    revealMathRef.current = !revealMathRef.current;
    setRevealMath(revealMathRef.current);
  }, []);

  const applySpectrum = useCallback(
    (next: OceanSpectrum, isFallback: boolean) => {
      rendererRef.current?.setSpectrum(next);
      setSpectrum(next);
      setSource({ isFallback, workerStatus: isFallback ? 'error' : 'ready' });
      onReady?.(next);
    },
    [onReady],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const quality = detectRenderQuality();
    if (sessionSeedRef.current === null) {
      sessionSeedRef.current = createSessionSeed();
    }
    const sessionSeed = sessionSeedRef.current;
    const fallbackSpectrum = createFallbackSpectrum(sessionSeed, quality.mobile);
    const renderer = new OceanRenderer(canvas, fallbackSpectrum, quality);
    rendererRef.current = renderer;
    setSpectrum(fallbackSpectrum);

    const animate = (now: number) => {
      if (startTimeRef.current === null) startTimeRef.current = now;
      const elapsed = (now - startTimeRef.current) / 1000;
      rendererRef.current?.render(elapsed, revealMathRef.current);
      frameRef.current = requestAnimationFrame(animate);
    };

    frameRef.current = requestAnimationFrame(animate);

    return () => {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
      renderer.dispose();
      rendererRef.current = null;
      startTimeRef.current = null;
    };
  }, []);

  useEffect(() => {
    let worker: Worker | null = null;
    let cancelled = false;

    const quality = detectRenderQuality();
    if (sessionSeedRef.current === null) {
      sessionSeedRef.current = createSessionSeed();
    }
    const sessionSeed = sessionSeedRef.current;

    setSource({ isFallback: false, workerStatus: 'computing' });

    try {
      worker = new PrimeFftWorker();

      worker.onmessage = (event: MessageEvent<PrimeFftWorkerResponse>) => {
        if (cancelled) return;
        const msg = event.data;
        if (msg.type === 'result') {
          applySpectrum(msg.spectrum, false);
        } else if (msg.type === 'error') {
          applySpectrum(createFallbackSpectrum(sessionSeed, quality.mobile), true);
        }
      };

      worker.onerror = () => {
        if (!cancelled) applySpectrum(createFallbackSpectrum(sessionSeed, quality.mobile), true);
      };

      worker.postMessage({ type: 'compute', sessionSeed, mobile: quality.mobile });
    } catch {
      applySpectrum(createFallbackSpectrum(sessionSeed, quality.mobile), true);
    }

    return () => {
      cancelled = true;
      worker?.terminate();
    };
  }, [applySpectrum]);

  useEffect(() => {
    const onResize = () => rendererRef.current?.resize();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === 'p') toggleReveal();
      if (e.key === 'Escape') setOverlayOpen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [toggleReveal]);

  const handlePointerMove = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (moonDragRef.current.dragging && moonDragRef.current.pointerId === e.pointerId) {
      rendererRef.current?.dragMoonTo(x, y);
      e.currentTarget.style.cursor = 'grabbing';
      return;
    }

    const overMoon = rendererRef.current?.isMoonHit(x, y) ?? false;
    e.currentTarget.style.cursor = overMoon ? 'grab' : 'default';

    mouseRef.current = { x, y, normX: 0, normY: 0, active: true };
    rendererRef.current?.setMouse(mouseRef.current);
  }, []);

  const handlePointerLeave = useCallback(() => {
    mouseRef.current.active = false;
    rendererRef.current?.setMouse(mouseRef.current);
  }, []);

  const handlePointerDown = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Only start dragging when directly clicking on the moon.
    const began = rendererRef.current?.beginMoonDrag(x, y) ?? false;
    if (began) {
      moonDragRef.current = { dragging: true, pointerId: e.pointerId };
      e.currentTarget.setPointerCapture(e.pointerId);
      e.currentTarget.style.cursor = 'grabbing';
      return;
    }

    rendererRef.current?.addRipple(x, y);
  }, []);

  const handlePointerUp = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    if (moonDragRef.current.dragging && moonDragRef.current.pointerId === e.pointerId) {
      moonDragRef.current = { dragging: false, pointerId: null };
      rendererRef.current?.endMoonDrag();
      try {
        e.currentTarget.releasePointerCapture(e.pointerId);
      } catch {
        // ignore
      }
      e.currentTarget.style.cursor = 'default';
    }
  }, []);

  return (
    <>
      <div className="prime-tides">
        <canvas
          ref={canvasRef}
          className="prime-tides__canvas"
          onPointerMove={handlePointerMove}
          onPointerLeave={handlePointerLeave}
          onPointerDown={handlePointerDown}
          onPointerUp={handlePointerUp}
        />
        <DiscoveryHint onClick={toggleReveal} revealActive={revealMath} />
      </div>

      <DiscoveryOverlay
        open={overlayOpen}
        onClose={() => setOverlayOpen(false)}
        spectrum={spectrum}
        source={source}
      />
    </>
  );
}
