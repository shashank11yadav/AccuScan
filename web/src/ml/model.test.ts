/**
 * Runs the committed model.onnx through onnxruntime-web (WebAssembly, as in
 * the browser) and checks the I/O contract the app relies on.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-web';
import { describe, expect, it } from 'vitest';
import { NUM_CLASSES, greedyDecode } from './decode';
import { INPUT_HEIGHT } from './preprocess';

const model = readFileSync(
  fileURLToPath(new URL('../../public/models/model.onnx', import.meta.url)),
);

describe('model.onnx', () => {
  it('maps image [1,1,64,W] to log_probs [W/4,1,96]', async () => {
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(new Uint8Array(model), {
      executionProviders: ['wasm'],
    });
    expect(session.inputNames).toEqual(['image']);
    expect(session.outputNames).toEqual(['log_probs']);
    for (const width of [4, 128, 1024]) {
      const image = new ort.Tensor('float32', new Float32Array(INPUT_HEIGHT * width).fill(-1), [
        1,
        1,
        INPUT_HEIGHT,
        width,
      ]);
      const out = (await session.run({ image }))['log_probs']!;
      expect(out.dims).toEqual([width / 4, 1, NUM_CLASSES]);
      const data = out.data as Float32Array;
      const first = Array.from(data.slice(0, NUM_CLASSES)).reduce((s, v) => s + Math.exp(v), 0);
      expect(first).toBeCloseTo(1, 3);
      greedyDecode({ data, frames: width / 4, classes: NUM_CLASSES });
    }
    await session.release();
  }, 30_000);
});
