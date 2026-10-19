import { useCallback, useEffect, useRef, useState } from 'react';
import { ModelClient } from '../ml/client';
import { NUM_CLASSES } from '../ml/decode';
import { parseMeta, type DecodeMode, type ModelMeta } from '../ml/meta';
import {
  EmptyImageError,
  FEATURE_VERSION,
  preprocess,
  type GrayImage,
  type ModelInput,
} from '../ml/preprocess';
import type { Backend, Recognition } from '../ml/protocol';
import { META_URL } from './links';

export type ModelState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; meta: ModelMeta; backend: Backend };

export type ReadState =
  | { status: 'idle' }
  | { status: 'reading' }
  | { status: 'empty'; message: string }
  | { status: 'error'; message: string }
  | { status: 'done'; result: Recognition; squeezed: boolean; preview: InputPreview };

/** The model input as dark-on-white greyscale, for showing what the network sees. */
export interface InputPreview {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

export function toPreview(input: ModelInput): InputPreview {
  const height = input.data.length / input.width;
  const data = new Uint8ClampedArray(input.data.length);
  for (let i = 0; i < input.data.length; i++) {
    data[i] = Math.round(((1 - (input.data[i] ?? -1)) / 2) * 255);
  }
  return { width: input.width, height, data };
}

async function loadMeta(signal: AbortSignal): Promise<ModelMeta> {
  const response = await fetch(META_URL, { signal });
  if (!response.ok) throw new Error(`model details could not be loaded (HTTP ${response.status})`);
  return parseMeta(await response.json(), {
    featureVersion: FEATURE_VERSION,
    classes: NUM_CLASSES,
  });
}

/** Loads the model in a worker and runs preprocessing + recognition on demand. */
export function useModel() {
  const [model, setModel] = useState<ModelState>({ status: 'loading' });
  const [attempt, setAttempt] = useState(0);
  const clientRef = useRef<ModelClient | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let client: ModelClient | null = null;
    loadMeta(controller.signal)
      .then(async (meta) => {
        client = new ModelClient();
        const backend = await client.ready;
        if (controller.signal.aborted) return;
        clientRef.current = client;
        setModel({ status: 'ready', meta, backend });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        client?.dispose();
        const message = error instanceof Error ? error.message : 'unknown error';
        setModel({ status: 'error', message });
      });
    return () => {
      controller.abort();
      client?.dispose();
      clientRef.current = null;
    };
  }, [attempt]);

  const retry = useCallback(() => {
    setModel({ status: 'loading' });
    setAttempt((n) => n + 1);
  }, []);

  const read = useCallback(
    async (image: GrayImage, mode: DecodeMode, emptyMessage: string): Promise<ReadState> => {
      const client = clientRef.current;
      if (!client || model.status !== 'ready') {
        return { status: 'error', message: 'The model is still loading.' };
      }
      let input;
      try {
        input = preprocess(image);
      } catch (error) {
        if (error instanceof EmptyImageError) return { status: 'empty', message: emptyMessage };
        throw error;
      }
      try {
        const result = await client.recognise(input, mode, model.meta.decoder);
        return { status: 'done', result, squeezed: input.squeezed, preview: toPreview(input) };
      } catch (error) {
        const message = error instanceof Error ? error.message : 'unknown error';
        return { status: 'error', message: `Recognition failed: ${message}` };
      }
    },
    [model],
  );

  return { model, retry, read };
}
