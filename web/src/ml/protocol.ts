import type { DecodeMode } from './meta';
import { INPUT_HEIGHT, MAX_WIDTH } from './preprocess';

/** The only runtime: onnxruntime-web's WebAssembly build. */
export type Backend = 'wasm';

export interface DecoderSettings {
  beamWidth: number;
  lmWeight: number;
  prune: number;
  maxDropPerWord: number;
}

/** The model and word list URLs are fixed inside the worker, so init carries nothing. */
export type WorkerRequest =
  | { type: 'init' }
  | {
      type: 'recognise';
      id: number;
      input: Float32Array;
      width: number;
      mode: DecodeMode;
      decoder: DecoderSettings;
    };

export interface Recognition {
  text: string;
  confidences: number[];
  /** Dictionary mode found no plausible word and returned the greedy reading. */
  fallback: boolean;
  mode: DecodeMode;
  inferenceMs: number;
  decodeMs: number;
}

export type WorkerResponse =
  | { type: 'ready'; backend: Backend }
  | { type: 'init-error'; message: string }
  | { type: 'result'; id: number; result: Recognition }
  | { type: 'error'; id: number; message: string };

/** Upper bound on the beam: the shipped decoder uses 16, and each beam costs time per frame. */
export const MAX_BEAM_WIDTH = 64;

export type ParsedRequest =
  { ok: true; request: WorkerRequest } | { ok: false; id: number | null; message: string };

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

const finiteAtLeast = (v: unknown, min: number): v is number =>
  typeof v === 'number' && Number.isFinite(v) && v >= min;

function parseDecoder(v: unknown): DecoderSettings | null {
  if (!isRecord(v)) return null;
  const { beamWidth, lmWeight, prune, maxDropPerWord } = v;
  if (
    !finiteAtLeast(beamWidth, 1) ||
    !Number.isInteger(beamWidth) ||
    beamWidth > MAX_BEAM_WIDTH ||
    !finiteAtLeast(lmWeight, 0) ||
    !finiteAtLeast(prune, 0) ||
    !finiteAtLeast(maxDropPerWord, 0)
  ) {
    return null;
  }
  return { beamWidth, lmWeight, prune, maxDropPerWord };
}

/**
 * Checks a message received by the worker. Only the two request shapes the
 * app sends are accepted, and the request is rebuilt from the checked fields.
 * Anything else is rejected, keeping the request id when there is a valid one
 * so the caller's promise can be settled with an error.
 */
export function parseRequest(data: unknown): ParsedRequest {
  const invalid = (id: number | null, why: string): ParsedRequest => ({
    ok: false,
    id,
    message: `invalid request: ${why}`,
  });
  if (!isRecord(data)) return invalid(null, 'not an object');
  if (data['type'] === 'init') {
    return Object.keys(data).length === 1
      ? { ok: true, request: { type: 'init' } }
      : invalid(null, 'init takes no fields');
  }
  if (data['type'] !== 'recognise') return invalid(null, 'unknown type');
  const { id, input, width, mode, decoder } = data;
  if (typeof id !== 'number' || !Number.isSafeInteger(id) || id < 1) {
    return invalid(null, 'id must be a positive integer');
  }
  if (typeof width !== 'number' || !Number.isInteger(width) || width < 1 || width > MAX_WIDTH) {
    return invalid(id, `width must be an integer from 1 to ${MAX_WIDTH}`);
  }
  if (!(input instanceof Float32Array) || input.length !== INPUT_HEIGHT * width) {
    return invalid(id, `input must be a Float32Array of ${INPUT_HEIGHT} x width values`);
  }
  if (mode !== 'free' && mode !== 'dictionary') return invalid(id, 'unknown mode');
  const settings = parseDecoder(decoder);
  if (!settings) return invalid(id, 'bad decoder settings');
  return { ok: true, request: { type: 'recognise', id, input, width, mode, decoder: settings } };
}
