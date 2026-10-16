import type { DecodeMode } from './meta';

export type Backend = 'webgpu' | 'wasm';

export interface DecoderSettings {
  beamWidth: number;
  lmWeight: number;
  prune: number;
  maxDropPerWord: number;
}

export type WorkerRequest =
  | { type: 'init'; modelUrl: string; lexiconUrl: string }
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
