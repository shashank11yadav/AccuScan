/// <reference lib="webworker" />
/**
 * Runs the model off the main thread: onnxruntime-web with the WebGPU execution
 * provider when the browser has a GPU adapter, WebAssembly otherwise (and as
 * the fallback if a WebGPU session cannot be created). Decoding runs here too,
 * so the dictionary trie never blocks the page.
 */
import type { InferenceSession, Tensor } from 'onnxruntime-web';
import { beamSearch, greedyDecode, NUM_CLASSES, type Lexicon } from './decode';
import { loadLexicon } from './lexicon';
import { INPUT_HEIGHT } from './preprocess';
import type { Backend, WorkerRequest, WorkerResponse } from './protocol';

declare const self: DedicatedWorkerGlobalScope;

type Ort = typeof import('onnxruntime-web');
let session: InferenceSession | null = null;
let ort: Ort | null = null;
let lexiconUrl = '';
let lexicon: Promise<Lexicon> | null = null;

const post = (message: WorkerResponse) => {
  self.postMessage(message);
};

async function hasWebGpu(): Promise<boolean> {
  if (!('gpu' in navigator)) return false;
  try {
    return (await navigator.gpu.requestAdapter()) !== null;
  } catch {
    return false;
  }
}

async function fetchModel(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`the model could not be downloaded (HTTP ${response.status})`);
  return new Uint8Array(await response.arrayBuffer());
}

async function createSession(model: Uint8Array): Promise<Backend> {
  if (await hasWebGpu()) {
    try {
      const gpuOrt = await import('onnxruntime-web/webgpu');
      gpuOrt.env.wasm.numThreads = 1;
      session = await gpuOrt.InferenceSession.create(model, {
        executionProviders: ['webgpu', 'wasm'],
      });
      ort = gpuOrt;
      return 'webgpu';
    } catch {
      // Fall through to WebAssembly.
    }
  }
  const wasmOrt = await import('onnxruntime-web/wasm');
  // GitHub Pages cannot send the headers that enable threads (cross-origin isolation).
  wasmOrt.env.wasm.numThreads = 1;
  session = await wasmOrt.InferenceSession.create(model, { executionProviders: ['wasm'] });
  ort = wasmOrt;
  return 'wasm';
}

async function init(modelUrl: string, wordsUrl: string) {
  lexiconUrl = wordsUrl;
  try {
    const backend = await createSession(await fetchModel(modelUrl));
    post({ type: 'ready', backend });
  } catch (error) {
    post({ type: 'init-error', message: error instanceof Error ? error.message : String(error) });
  }
}

async function recognise(request: Extract<WorkerRequest, { type: 'recognise' }>) {
  if (!session || !ort) throw new Error('the model is not loaded');
  const started = performance.now();
  const image = new ort.Tensor('float32', request.input, [1, 1, INPUT_HEIGHT, request.width]);
  const outputs = await session.run({ image });
  const out: Tensor | undefined = outputs['log_probs'];
  if (!out || out.dims[2] !== NUM_CLASSES) throw new Error('unexpected model output');
  const data = (await out.getData()) as Float32Array;
  const logProbs = { data, frames: out.dims[0] ?? 0, classes: NUM_CLASSES };
  const inferred = performance.now();
  let decoded = greedyDecode(logProbs);
  if (request.mode === 'dictionary') {
    lexicon ??= loadLexicon(lexiconUrl);
    decoded = beamSearch(logProbs, await lexicon, request.decoder);
  }
  out.dispose();
  return {
    ...decoded,
    mode: request.mode,
    inferenceMs: inferred - started,
    decodeMs: performance.now() - inferred,
  };
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  if (request.type === 'init') {
    void init(request.modelUrl, request.lexiconUrl);
    return;
  }
  recognise(request).then(
    (result) => {
      post({ type: 'result', id: request.id, result });
    },
    (error: unknown) => {
      if (request.mode === 'dictionary') lexicon = null; // allow a retry after a failed download
      post({
        type: 'error',
        id: request.id,
        message: error instanceof Error ? error.message : String(error),
      });
    },
  );
};
