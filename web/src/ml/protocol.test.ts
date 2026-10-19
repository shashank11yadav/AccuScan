import { describe, expect, it } from 'vitest';
import { INPUT_HEIGHT, MAX_WIDTH } from './preprocess';
import { parseRequest } from './protocol';

const decoder = { beamWidth: 16, lmWeight: 0.3, prune: 0.001, maxDropPerWord: 10 };
const recognise = (overrides: Record<string, unknown> = {}) => ({
  type: 'recognise',
  id: 7,
  input: new Float32Array(INPUT_HEIGHT * 128),
  width: 128,
  mode: 'free',
  decoder,
  ...overrides,
});

describe('parseRequest', () => {
  it('accepts init with no other fields', () => {
    expect(parseRequest({ type: 'init' })).toEqual({ ok: true, request: { type: 'init' } });
  });

  it('rejects init carrying URLs: assets are fixed inside the worker', () => {
    const parsed = parseRequest({ type: 'init', modelUrl: 'https://evil.example/m.onnx' });
    expect(parsed.ok).toBe(false);
  });

  it('accepts a well-formed recognise request', () => {
    const parsed = parseRequest(recognise({ mode: 'dictionary' }));
    expect(parsed.ok).toBe(true);
    if (parsed.ok && parsed.request.type === 'recognise') {
      expect(parsed.request.id).toBe(7);
      expect(parsed.request.width).toBe(128);
      expect(parsed.request.mode).toBe('dictionary');
      expect(parsed.request.decoder).toEqual(decoder);
    }
  });

  it('accepts the widest input the model takes', () => {
    const parsed = parseRequest(
      recognise({ input: new Float32Array(INPUT_HEIGHT * MAX_WIDTH), width: MAX_WIDTH }),
    );
    expect(parsed.ok).toBe(true);
  });

  it.each([null, undefined, 42, 'init', [], {}, { type: 'shutdown' }])(
    'rejects %j without an id',
    (data) => {
      expect(parseRequest(data)).toMatchObject({ ok: false, id: null });
    },
  );

  it.each([
    ['a non-integer id', { id: 1.5 }],
    ['a negative id', { id: -1 }],
    ['a string id', { id: '7' }],
  ])('rejects %s and recovers no id', (_name, overrides) => {
    expect(parseRequest(recognise(overrides))).toMatchObject({ ok: false, id: null });
  });

  it.each([
    ['a plain array input', { input: Array.from({ length: INPUT_HEIGHT * 128 }, () => 0) }],
    ['a Float64Array input', { input: new Float64Array(INPUT_HEIGHT * 128) }],
    ['an input of the wrong length', { input: new Float32Array(INPUT_HEIGHT * 128 - 1) }],
    ['a width over the model limit', { width: MAX_WIDTH + 4 }],
    ['a zero width', { width: 0, input: new Float32Array(0) }],
    ['a fractional width', { width: 127.5 }],
    ['an unknown mode', { mode: 'oracle' }],
    ['a missing decoder', { decoder: undefined }],
    ['a huge beam width', { decoder: { ...decoder, beamWidth: 1e6 } }],
    ['a fractional beam width', { decoder: { ...decoder, beamWidth: 2.5 } }],
    ['a NaN lm weight', { decoder: { ...decoder, lmWeight: Number.NaN } }],
    ['a negative prune', { decoder: { ...decoder, prune: -1 } }],
    ['an infinite max drop', { decoder: { ...decoder, maxDropPerWord: Infinity } }],
  ])('rejects %s but keeps the id so the caller can be told', (_name, overrides) => {
    const parsed = parseRequest(recognise(overrides));
    expect(parsed).toMatchObject({ ok: false, id: 7 });
    if (!parsed.ok) expect(parsed.message).toMatch(/invalid request/);
  });
});
