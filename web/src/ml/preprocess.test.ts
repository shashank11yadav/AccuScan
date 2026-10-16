import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { applyOrientation } from './image';
import {
  EmptyImageError,
  INPUT_HEIGHT,
  MAX_WIDTH,
  preprocess,
  rgbaToGray,
  type GrayImage,
} from './preprocess';

interface FixtureImage {
  height: number;
  width: number;
  channels: number;
  data: string;
}
interface PreprocessCase {
  name: string;
  input: FixtureImage;
  exifOrientation?: number;
  expected: { shape: number[]; data: number[] } | { error: string };
}

const fixture = JSON.parse(
  readFileSync(
    fileURLToPath(new URL('../../../shared/fixtures/preprocess.json', import.meta.url)),
    'utf8',
  ),
) as { cases: PreprocessCase[] };

function decodeImage(spec: FixtureImage): GrayImage {
  const bytes = new Uint8Array(Buffer.from(spec.data, 'base64'));
  if (spec.channels === 1) return { width: spec.width, height: spec.height, data: bytes };
  return rgbaToGray(bytes, spec.width, spec.height);
}

describe('preprocess parity with the Python pipeline', () => {
  it.each(fixture.cases.map((c) => [c.name, c] as const))('%s', (_name, c) => {
    let img = decodeImage(c.input);
    if (c.exifOrientation !== undefined) img = applyOrientation(img, c.exifOrientation);
    if ('error' in c.expected) {
      expect(() => preprocess(img)).toThrow(EmptyImageError);
      return;
    }
    const out = preprocess(img);
    expect([1, INPUT_HEIGHT, out.width]).toEqual(c.expected.shape);
    let worst = 0;
    c.expected.data.forEach((v, i) => {
      worst = Math.max(worst, Math.abs((out.data[i] ?? NaN) - v));
    });
    expect(worst).toBeLessThan(1e-4);
  });

  it('covers a rotated photo, an RGBA canvas and a squeezed line', () => {
    const names = fixture.cases.map((c) => c.name);
    expect(names).toEqual(
      expect.arrayContaining(['exif-orientation-6', 'rgba-canvas', 'very-wide-squeezed', 'blank']),
    );
  });
});

describe('preprocess behaviour', () => {
  const blank = (w: number, h: number, v = 255): GrayImage => ({
    width: w,
    height: h,
    data: new Uint8Array(w * h).fill(v),
  });

  it('throws EmptyImageError for an empty canvas', () => {
    expect(() => preprocess(blank(300, 100))).toThrow(EmptyImageError);
    expect(() => preprocess(blank(0, 0))).toThrow(EmptyImageError);
  });

  it('flags very wide lines as squeezed and caps the width', () => {
    const img = blank(4000, 40);
    for (let y = 10; y < 30; y++) for (let x = 5; x < 3995; x++) img.data[y * 4000 + x] = 0;
    const out = preprocess(img);
    expect(out.width).toBe(MAX_WIDTH);
    expect(out.squeezed).toBe(true);
  });

  it('keeps short words unsqueezed with a multiple-of-4 width', () => {
    const img = blank(200, 80);
    for (let y = 20; y < 60; y++) for (let x = 20; x < 120; x++) img.data[y * 200 + x] = 0;
    const out = preprocess(img);
    expect(out.squeezed).toBe(false);
    expect(out.width % 4).toBe(0);
    expect(out.data.length).toBe(INPUT_HEIGHT * out.width);
  });
});
