import { useCallback, useEffect, useRef, useState } from 'react';
import { isBlankCanvas } from '../ml/image';
import { rgbaToGray, type GrayImage } from '../ml/preprocess';
import { UndoIcon, TrashIcon } from './Icons';

/** Recognise this long after the last stroke ends. */
export const IDLE_MS = 400;

const WIDTHS = [
  { label: 'Fine', value: 4 },
  { label: 'Medium', value: 7 },
  { label: 'Bold', value: 11 },
] as const;

interface Stroke {
  width: number;
  points: [number, number][];
}

interface Props {
  disabled: boolean;
  /** Called after IDLE_MS of no drawing, or on demand, with the canvas as greyscale. */
  onImage: (image: GrayImage) => void;
  onBlank: () => void;
}

function drawStroke(ctx: CanvasRenderingContext2D, stroke: Stroke) {
  const [first, ...rest] = stroke.points;
  if (!first) return;
  ctx.lineWidth = stroke.width;
  ctx.beginPath();
  let [px, py] = first;
  if (rest.length === 0) {
    ctx.arc(px, py, stroke.width / 2, 0, Math.PI * 2);
    ctx.fill();
    return;
  }
  ctx.moveTo(px, py);
  // Quadratic curves through midpoints smooth out pointer jitter.
  for (const [x, y] of rest) {
    ctx.quadraticCurveTo(px, py, (px + x) / 2, (py + y) / 2);
    [px, py] = [x, y];
  }
  ctx.lineTo(px, py);
  ctx.stroke();
}

export function WritePad({ disabled, onImage, onBlank }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const strokes = useRef<Stroke[]>([]);
  const active = useRef<Stroke | null>(null);
  const timer = useRef<number | undefined>(undefined);
  const [width, setWidth] = useState<number>(WIDTHS[1].value);
  const [count, setCount] = useState(0);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d', { willReadFrequently: true });
    if (!canvas || !ctx) return;
    const dpr = canvas.width / canvas.clientWidth || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.strokeStyle = '#1b1a17';
    ctx.fillStyle = '#1b1a17';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (const s of strokes.current) drawStroke(ctx, s);
    if (active.current) drawStroke(ctx, active.current);
  }, []);

  // Match the backing store to the element size and device pixel ratio.
  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const resize = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const rect = wrap.getBoundingClientRect();
      // Stroke points are kept in CSS pixels, so a resize only needs a redraw.
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      redraw();
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(wrap);
    return () => {
      observer.disconnect();
    };
  }, [redraw]);

  useEffect(
    () => () => {
      window.clearTimeout(timer.current);
    },
    [],
  );

  const capture = useCallback((): GrayImage | null => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d', { willReadFrequently: true });
    if (!canvas || !ctx || canvas.width === 0) return null;
    const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
    if (isBlankCanvas(data)) return null;
    return rgbaToGray(data, canvas.width, canvas.height);
  }, []);

  const submit = useCallback(() => {
    window.clearTimeout(timer.current);
    const image = capture();
    if (image) onImage(image);
    else onBlank();
  }, [capture, onImage, onBlank]);

  const point = (event: React.PointerEvent<HTMLCanvasElement>): [number, number] => {
    const rect = event.currentTarget.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  };

  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (event.button !== 0 && event.pointerType === 'mouse') return;
    window.clearTimeout(timer.current);
    event.currentTarget.setPointerCapture(event.pointerId);
    active.current = { width, points: [point(event)] };
    redraw();
  };

  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const stroke = active.current;
    if (!stroke) return;
    // Coalesced events give smoother strokes; Safari before 18 does not have them.
    const native = event.nativeEvent;
    const events = 'getCoalescedEvents' in native ? native.getCoalescedEvents() : [native];
    const rect = event.currentTarget.getBoundingClientRect();
    for (const e of events) stroke.points.push([e.clientX - rect.left, e.clientY - rect.top]);
    redraw();
  };

  const finishStroke = () => {
    const stroke = active.current;
    if (!stroke) return;
    strokes.current.push(stroke);
    active.current = null;
    setCount(strokes.current.length);
    redraw();
    if (!disabled) timer.current = window.setTimeout(submit, IDLE_MS);
  };

  const undo = () => {
    strokes.current.pop();
    setCount(strokes.current.length);
    redraw();
    window.clearTimeout(timer.current);
    if (strokes.current.length === 0) onBlank();
    else if (!disabled) timer.current = window.setTimeout(submit, IDLE_MS);
  };

  const clear = () => {
    strokes.current = [];
    setCount(0);
    redraw();
    window.clearTimeout(timer.current);
    onBlank();
  };

  return (
    <div className="flex flex-col gap-3">
      <div
        ref={wrapRef}
        className="relative h-52 overflow-hidden rounded-xl border border-line shadow-[inset_0_1px_2px_rgb(0_0_0/4%)] sm:h-60"
      >
        <canvas
          ref={canvasRef}
          className="pad absolute inset-0 h-full w-full"
          role="img"
          aria-label="Writing area. Draw a word or short line with a mouse, finger or pen."
          data-testid="write-pad"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={finishStroke}
          onPointerCancel={finishStroke}
          onKeyDown={(event) => {
            if ((event.metaKey || event.ctrlKey) && event.key === 'z') {
              event.preventDefault();
              undo();
            }
          }}
          tabIndex={0}
        />
        <div
          aria-hidden="true"
          className="pad-guide pointer-events-none absolute inset-x-6 top-[68%] h-px"
        />
        {count === 0 && (
          <p
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 grid place-items-center text-lg text-[#8a857c] select-none"
          >
            Write a word here
          </p>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={undo}
          disabled={count === 0}
          className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-line bg-surface px-2.5 text-sm font-medium hover:bg-surface-muted disabled:opacity-50 sm:px-3"
        >
          <UndoIcon /> <span className="sr-only sm:not-sr-only">Undo</span>
        </button>
        <button
          type="button"
          onClick={clear}
          disabled={count === 0}
          className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-line bg-surface px-2.5 text-sm font-medium hover:bg-surface-muted disabled:opacity-50 sm:px-3"
        >
          <TrashIcon /> <span className="sr-only sm:not-sr-only">Clear</span>
        </button>
        <fieldset className="ml-auto flex items-center gap-2 border-0 p-0">
          <legend className="sr-only">Stroke width</legend>
          <span aria-hidden="true" className="hidden text-sm text-ink-muted sm:inline">
            Pen
          </span>
          <div className="inline-flex rounded-lg border border-line bg-surface p-0.5">
            {WIDTHS.map((w) => (
              <label key={w.value} className="relative">
                <input
                  type="radio"
                  name="stroke-width"
                  value={w.value}
                  checked={width === w.value}
                  onChange={() => {
                    setWidth(w.value);
                  }}
                  className="peer absolute inset-0 cursor-pointer opacity-0"
                />
                <span className="flex h-8 items-center gap-1.5 rounded-md px-2.5 text-sm text-ink-muted peer-checked:bg-accent-soft peer-checked:font-medium peer-checked:text-accent peer-focus-visible:outline-2 peer-focus-visible:outline-[var(--focus)]">
                  <span
                    aria-hidden="true"
                    className="inline-block rounded-full bg-current"
                    style={{ width: w.value * 0.8 + 2, height: w.value * 0.8 + 2 }}
                  />
                  <span className="sr-only sm:not-sr-only">{w.label}</span>
                </span>
              </label>
            ))}
          </div>
        </fieldset>
        <button
          type="button"
          onClick={submit}
          disabled={disabled}
          className="inline-flex h-9 items-center rounded-lg bg-accent px-4 text-sm font-semibold text-accent-ink shadow-sm hover:brightness-110 disabled:opacity-50"
        >
          Recognise
        </button>
      </div>
    </div>
  );
}
