import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { NUM_CLASSES } from './decode';
import { MetaError, isUntrained, parseMeta } from './meta';
import { FEATURE_VERSION } from './preprocess';

const committed = JSON.parse(
  readFileSync(
    fileURLToPath(new URL('../../public/models/model-meta.json', import.meta.url)),
    'utf8',
  ),
) as Record<string, unknown>;
const expected = { featureVersion: FEATURE_VERSION, classes: NUM_CLASSES };

describe('parseMeta', () => {
  it('accepts the committed model-meta.json', () => {
    const meta = parseMeta(committed, expected);
    expect(meta.input).toEqual({ name: 'image', height: 64, maxWidth: 1024, widthMultiple: 4 });
    expect(meta.output.name).toBe('log_probs');
  });

  it('marks a model without metrics as untrained', () => {
    expect(isUntrained(parseMeta({ ...committed, metrics: null }, expected))).toBe(true);
    const trained = {
      ...committed,
      metrics: { sets: {}, gate: { canvas_cer: 0.1, threshold: 0.15, defaultMode: 'free' } },
    };
    expect(isUntrained(parseMeta(trained, expected))).toBe(false);
  });

  it('rejects a model built for different preprocessing', () => {
    expect(() => parseMeta({ ...committed, featureVersion: 99 }, expected)).toThrow(MetaError);
  });

  it('rejects malformed files', () => {
    expect(() => parseMeta(null, expected)).toThrow(MetaError);
    expect(() => parseMeta({ ...committed, defaultMode: 'loud' }, expected)).toThrow(/defaultMode/);
    expect(() => parseMeta({ ...committed, metrics: 3 }, expected)).toThrow(/metrics/);
  });
});
