import { useEffect, useRef, useState } from 'react';
import { validateSelection, type Rect } from '../ml/image';
import type { GrayImage } from '../ml/preprocess';
import { cropGray, decodePhoto, grayToImageData, rectFromDrag, UploadError } from '../lib/photo';
import { AlertIcon, UploadIcon } from './Icons';

interface Props {
  disabled: boolean;
  onImage: (image: GrayImage) => void;
  onReset: () => void;
}

export function UploadPanel({ disabled, onImage, onReset }: Props) {
  const [photo, setPhoto] = useState<GrayImage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [selection, setSelection] = useState<Rect | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragStart = useRef<[number, number] | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx || !photo) return;
    canvas.width = photo.width;
    canvas.height = photo.height;
    ctx.putImageData(grayToImageData(photo), 0, 0);
    if (selection) {
      ctx.fillStyle = 'rgb(20 20 19 / 45%)';
      ctx.fillRect(0, 0, photo.width, photo.height);
      ctx.putImageData(
        grayToImageData(cropGray(photo, selection)),
        Math.max(0, Math.floor(selection.x)),
        Math.max(0, Math.floor(selection.y)),
      );
      const line = Math.max(2, photo.width / 400);
      ctx.lineWidth = line;
      ctx.strokeStyle = '#2446a6';
      ctx.strokeRect(selection.x, selection.y, selection.width, selection.height);
    }
  }, [photo, selection]);

  const load = async (file: File | undefined) => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setSelection(null);
    onReset();
    try {
      setPhoto(await decodePhoto(file));
    } catch (e) {
      setPhoto(null);
      setError(e instanceof UploadError ? e.message : 'That image could not be opened.');
    } finally {
      setBusy(false);
    }
  };

  const toPhoto = (event: React.PointerEvent<HTMLCanvasElement>): [number, number] => {
    const rect = event.currentTarget.getBoundingClientRect();
    const scale = (photo?.width ?? 1) / rect.width;
    return [(event.clientX - rect.left) * scale, (event.clientY - rect.top) * scale];
  };

  const finish = (rect: Rect) => {
    if (!photo) return;
    const check = validateSelection(rect);
    if (!check.ok) {
      setError(check.message);
      return;
    }
    setError(null);
    onImage(cropGray(photo, rect));
  };

  return (
    <div className="flex flex-col gap-3">
      {!photo && (
        <label
          className={`flex h-52 cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-4 text-center transition-colors sm:h-60 ${
            dragOver
              ? 'border-accent bg-accent-soft'
              : 'border-line bg-surface hover:bg-surface-muted'
          }`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => {
            setDragOver(false);
          }}
          onDrop={(event) => {
            event.preventDefault();
            setDragOver(false);
            void load(event.dataTransfer.files[0]);
          }}
        >
          <span className="grid h-11 w-11 place-items-center rounded-full bg-accent-soft text-accent">
            <UploadIcon />
          </span>
          <span className="font-medium">
            {busy ? 'Opening photo…' : 'Choose a photo of handwriting'}
          </span>
          <span className="text-sm text-ink-muted">
            JPEG, PNG or WebP up to 15 MB. It stays on this device.
          </span>
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="sr-only"
            data-testid="photo-input"
            onChange={(event) => {
              void load(event.target.files?.[0]);
              event.target.value = '';
            }}
          />
        </label>
      )}
      {photo && (
        <>
          <p className="text-sm text-ink-muted" id="select-hint">
            Drag a box around <strong className="font-semibold text-ink">one line</strong> of
            writing. It is read as soon as you let go.
          </p>
          <div className="photo-frame flex max-h-[26rem] justify-center overflow-hidden rounded-xl border border-line">
            <canvas
              ref={canvasRef}
              className="block h-auto max-h-[26rem] w-auto max-w-full cursor-crosshair touch-none"
              role="img"
              aria-label="Your photo. Drag to select a line."
              aria-describedby="select-hint"
              data-testid="photo-canvas"
              onPointerDown={(event) => {
                if (disabled) return;
                event.currentTarget.setPointerCapture(event.pointerId);
                dragStart.current = toPhoto(event);
                setSelection(null);
              }}
              onPointerMove={(event) => {
                if (!dragStart.current) return;
                const [x, y] = toPhoto(event);
                setSelection(rectFromDrag(dragStart.current[0], dragStart.current[1], x, y));
              }}
              onPointerUp={(event) => {
                if (!dragStart.current) return;
                const [x, y] = toPhoto(event);
                const rect = rectFromDrag(dragStart.current[0], dragStart.current[1], x, y);
                dragStart.current = null;
                setSelection(rect);
                finish(rect);
              }}
            />
          </div>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="text-ink-muted">
              {photo.width} × {photo.height} px
            </span>
            <button
              type="button"
              className="ml-auto inline-flex h-9 items-center rounded-lg border border-line bg-surface px-3 font-medium hover:bg-surface-muted"
              onClick={() => {
                setPhoto(null);
                setSelection(null);
                setError(null);
                onReset();
              }}
            >
              Choose another photo
            </button>
          </div>
        </>
      )}
      {error && (
        <p
          role="alert"
          className="flex items-start gap-2 rounded-lg border border-danger/30 bg-danger-soft px-3 py-2 text-sm text-danger"
        >
          <AlertIcon className="mt-0.5 shrink-0" />
          {error}
        </p>
      )}
    </div>
  );
}
