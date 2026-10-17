import { useEffect, useRef } from 'react';
import type { InputPreview, ReadState } from '../lib/useModel';
import type { DecodeMode } from '../ml/meta';
import type { Backend } from '../ml/protocol';
import { AlertIcon } from './Icons';

interface Props {
  state: ReadState;
  mode: DecodeMode;
  onModeChange: (mode: DecodeMode) => void;
  backend: Backend | null;
  expected: string | null;
  modelReady: boolean;
}

function confidenceClass(c: number): string {
  if (c >= 0.9) return '';
  if (c >= 0.6) return 'char-mid';
  return 'char-low';
}

function ModelInputView({ preview }: { preview: InputPreview }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    canvas.width = preview.width;
    canvas.height = preview.height;
    const rgba = new Uint8ClampedArray(preview.width * preview.height * 4);
    for (let i = 0; i < preview.data.length; i++) {
      const v = preview.data[i] ?? 255;
      rgba.set([v, v, v, 255], i * 4);
    }
    ctx.putImageData(new ImageData(rgba, preview.width, preview.height), 0, 0);
  }, [preview]);
  return (
    <figure className="mt-4">
      <figcaption className="mb-1.5 text-xs text-ink-muted">
        What the model sees: {preview.width} × {preview.height} px, contrast-normalised
      </figcaption>
      <div className="overflow-hidden rounded-lg border border-line bg-white p-1.5">
        <canvas
          ref={ref}
          className="block h-12 max-w-full object-contain object-left"
          aria-hidden="true"
          data-testid="model-input"
        />
      </div>
    </figure>
  );
}

function Reading() {
  return (
    <p className="flex items-center gap-2 text-ink-muted" role="status">
      <span className="flex gap-1" aria-hidden="true">
        <span className="reading-dot h-2 w-2 rounded-full bg-accent" />
        <span className="reading-dot h-2 w-2 rounded-full bg-accent" />
        <span className="reading-dot h-2 w-2 rounded-full bg-accent" />
      </span>
      Reading…
    </p>
  );
}

export function ResultPanel({ state, mode, onModeChange, backend, expected, modelReady }: Props) {
  const dictionary = mode === 'dictionary';
  return (
    <section
      aria-labelledby="result-heading"
      className="flex min-w-0 flex-col gap-4 rounded-2xl border border-line bg-surface p-4 shadow-card sm:p-5"
    >
      <div className="flex items-center justify-between gap-3">
        <h2
          id="result-heading"
          className="text-sm font-semibold tracking-wide text-ink-muted uppercase"
        >
          Result
        </h2>
        <button
          type="button"
          role="switch"
          aria-checked={dictionary}
          onClick={() => {
            onModeChange(dictionary ? 'free' : 'dictionary');
          }}
          className="group inline-flex items-center gap-2 text-sm font-medium"
        >
          <span
            aria-hidden="true"
            className={`relative h-5 w-9 rounded-full transition-colors ${dictionary ? 'bg-accent' : 'bg-[#a8a29a]'}`}
          >
            <span
              className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform ${
                dictionary ? 'translate-x-4' : 'translate-x-0.5'
              }`}
            />
          </span>
          Dictionary words
        </button>
      </div>

      <div className="min-h-24" aria-live="polite" data-testid="result" data-status={state.status}>
        {!modelReady && state.status === 'idle' && (
          <p className="text-ink-muted">The model is loading. You can start writing.</p>
        )}
        {modelReady && state.status === 'idle' && (
          <p className="text-ink-muted">
            Write, upload or pick a sample. The reading appears here.
          </p>
        )}
        {state.status === 'reading' && <Reading />}
        {state.status === 'empty' && <p className="text-ink-muted">{state.message}</p>}
        {state.status === 'error' && (
          <p role="alert" className="flex items-start gap-2 text-danger">
            <AlertIcon className="mt-0.5 shrink-0" />
            {state.message}
          </p>
        )}
        {state.status === 'done' && (
          <>
            {state.result.text.trim() === '' ? (
              <p className="text-ink-muted">
                No characters recognised. Try writing larger and clearer.
              </p>
            ) : (
              <p
                className="font-mono text-3xl leading-snug break-words whitespace-pre-wrap"
                data-testid="result-text"
              >
                <span className="sr-only">{state.result.text}</span>
                {Array.from(state.result.text).map((ch, i) => {
                  const c = state.result.confidences[i] ?? 0;
                  return (
                    <span
                      key={i}
                      aria-hidden="true"
                      title={`${ch === ' ' ? 'space' : ch}: ${Math.round(c * 100)}% confident`}
                      className={`rounded-sm ${confidenceClass(c)}`}
                    >
                      {ch}
                    </span>
                  );
                })}
              </p>
            )}
            <dl className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-muted">
              <div className="flex gap-1">
                <dt>Model</dt>
                <dd className="font-medium text-ink">{Math.round(state.result.inferenceMs)} ms</dd>
              </div>
              <div className="flex gap-1">
                <dt>Decoder</dt>
                <dd className="font-medium text-ink">
                  {state.result.mode === 'dictionary' ? 'dictionary beam search' : 'greedy'}
                </dd>
              </div>
              {backend && (
                <div className="flex gap-1">
                  <dt>Runtime</dt>
                  <dd className="font-medium text-ink">WebAssembly</dd>
                </div>
              )}
            </dl>
            {state.result.fallback && (
              <p className="mt-2 text-sm text-warn">
                No dictionary match, so this is the raw reading.
              </p>
            )}
            {state.squeezed && (
              <p className="mt-2 text-sm text-warn">
                This line was squeezed to fit the model, which lowers accuracy. Try fewer words.
              </p>
            )}
            <ModelInputView preview={state.preview} />
            {expected !== null && (
              <p className="mt-2 text-sm text-ink-muted">
                Written text: <span className="font-medium text-ink">{expected}</span>
              </p>
            )}
          </>
        )}
      </div>

      <div className="mt-auto flex items-center gap-3 border-t border-line pt-3 text-xs text-ink-muted">
        <span>Confidence</span>
        <span className="inline-flex items-center gap-1">
          <span aria-hidden="true" className="h-3 w-3 rounded-sm border border-line" /> high
        </span>
        <span className="inline-flex items-center gap-1">
          <span aria-hidden="true" className="char-mid h-3 w-3 rounded-sm" /> medium
        </span>
        <span className="inline-flex items-center gap-1">
          <span aria-hidden="true" className="char-low h-3 w-3 rounded-sm" /> low
        </span>
      </div>
    </section>
  );
}
