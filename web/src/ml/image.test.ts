import { describe, expect, it } from 'vitest';
import {
  MAX_PHOTO_PIXELS,
  MAX_UPLOAD_BYTES,
  MAX_PHOTO_SIDE,
  applyOrientation,
  fitWithin,
  isBlankCanvas,
  orientedSize,
  readImageSize,
  readJpegInfo,
  sniffImageType,
  validateDimensions,
  validateSelection,
  validateUpload,
} from './image';

const PNG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const WEBP = [0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x50];

/** A minimal JPEG: SOI, APP1 Exif with one orientation entry, SOF0, EOI. */
function jpeg(orientation: number | null, width: number, height: number, littleEndian = false) {
  const bytes: number[] = [0xff, 0xd8];
  if (orientation !== null) {
    const u16 = (v: number) => (littleEndian ? [v & 0xff, v >> 8] : [v >> 8, v & 0xff]);
    const u32 = (v: number) =>
      littleEndian
        ? [v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, v >>> 24]
        : [v >>> 24, (v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff];
    const tiff = [
      ...(littleEndian ? [0x49, 0x49] : [0x4d, 0x4d]),
      ...u16(42),
      ...u32(8),
      ...u16(1),
      ...u16(0x0112),
      ...u16(3),
      ...u32(1),
      ...u16(orientation),
      0,
      0,
      ...u32(0),
    ];
    const payload = [0x45, 0x78, 0x69, 0x66, 0, 0, ...tiff];
    const len = payload.length + 2;
    bytes.push(0xff, 0xe1, len >> 8, len & 0xff, ...payload);
  }
  bytes.push(0xff, 0xc0, 0, 17, 8, height >> 8, height & 0xff, width >> 8, width & 0xff, 3);
  bytes.push(...new Array<number>(9).fill(0));
  bytes.push(0xff, 0xd9);
  return new Uint8Array(bytes);
}

describe('sniffImageType', () => {
  it('recognises JPEG, PNG and WebP by magic bytes', () => {
    expect(sniffImageType(jpeg(null, 4, 4))).toBe('image/jpeg');
    expect(sniffImageType(new Uint8Array(PNG))).toBe('image/png');
    expect(sniffImageType(new Uint8Array(WEBP))).toBe('image/webp');
  });

  it('rejects other content even if it claims to be an image', () => {
    expect(
      sniffImageType(new TextEncoder().encode('<svg xmlns="http://www.w3.org/2000/svg">')),
    ).toBe(null);
    expect(sniffImageType(new Uint8Array([0x47, 0x49, 0x46, 0x38]))).toBe(null); // GIF
    expect(sniffImageType(new Uint8Array([]))).toBe(null);
  });
});

describe('validateUpload', () => {
  const head = new Uint8Array(PNG);
  it('accepts a small PNG', () => {
    expect(validateUpload({ size: 1000, head })).toEqual({ ok: true, type: 'image/png' });
  });
  it('rejects empty files', () => {
    expect(validateUpload({ size: 0, head: new Uint8Array() })).toMatchObject({
      ok: false,
      reason: 'empty',
    });
  });
  it('rejects files over 15 MB', () => {
    expect(MAX_UPLOAD_BYTES).toBe(15 * 1024 * 1024);
    expect(validateUpload({ size: MAX_UPLOAD_BYTES + 1, head })).toMatchObject({
      ok: false,
      reason: 'too-large',
    });
  });
  it('rejects unsupported types with a message', () => {
    const result = validateUpload({ size: 10, head: new TextEncoder().encode('GIF89a') });
    expect(result).toMatchObject({ ok: false, reason: 'unsupported' });
    if (!result.ok) expect(result.message).toMatch(/JPEG, PNG or WebP/);
  });
});

describe('readJpegInfo', () => {
  it('reads the size and a big-endian EXIF orientation', () => {
    expect(readJpegInfo(jpeg(6, 640, 480))).toEqual({ width: 640, height: 480, orientation: 6 });
  });
  it('reads a little-endian EXIF orientation', () => {
    expect(readJpegInfo(jpeg(8, 10, 20, true))).toEqual({ width: 10, height: 20, orientation: 8 });
  });
  it('defaults to orientation 1 without EXIF', () => {
    expect(readJpegInfo(jpeg(null, 3, 2))).toEqual({ width: 3, height: 2, orientation: 1 });
  });
  it('returns null for truncated or non-JPEG data', () => {
    expect(readJpegInfo(new Uint8Array([0xff, 0xd8, 0xff]))).toBeNull();
    expect(readJpegInfo(new Uint8Array(PNG))).toBeNull();
  });
  it('ignores out-of-range orientation values', () => {
    expect(readJpegInfo(jpeg(9, 3, 2))?.orientation).toBe(1);
  });
  it('skips fill bytes before a marker', () => {
    const plain = jpeg(null, 30, 20);
    const padded = new Uint8Array([0xff, 0xd8, 0xff, 0xff, ...plain.slice(2)]);
    expect(readJpegInfo(padded)).toEqual({ width: 30, height: 20, orientation: 1 });
  });
});

const be32 = (v: number) => [v >>> 24, (v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff];
const le16 = (v: number) => [v & 0xff, v >> 8];
const le24 = (v: number) => [v & 0xff, (v >> 8) & 0xff, v >> 16];
const ascii = (s: string) => Array.from(s, (c) => c.charCodeAt(0));

/** PNG signature and IHDR, which is all a header check reads. */
const png = (width: number, height: number) =>
  new Uint8Array([...PNG, ...be32(13), ...ascii('IHDR'), ...be32(width), ...be32(height), 8, 0]);
const riff = (chunk: number[]) =>
  new Uint8Array([...ascii('RIFF'), ...be32(0), ...ascii('WEBP'), ...chunk]);
const webpLossy = (width: number, height: number) =>
  riff([...ascii('VP8 '), ...be32(0), 0, 0, 0, 0x9d, 0x01, 0x2a, ...le16(width), ...le16(height)]);
function webpLossless(width: number, height: number) {
  const bits = (width - 1) | ((height - 1) << 14);
  return riff([
    ...ascii('VP8L'),
    ...be32(0),
    0x2f,
    bits & 0xff,
    (bits >> 8) & 0xff,
    (bits >> 16) & 0xff,
    bits >>> 24,
  ]);
}
const webpExtended = (width: number, height: number) =>
  riff([...ascii('VP8X'), ...be32(0), 0, 0, 0, 0, ...le24(width - 1), ...le24(height - 1)]);

describe('readImageSize', () => {
  it('reads PNG, WebP (lossy, lossless, extended) and JPEG sizes from the header', () => {
    expect(readImageSize(png(640, 480), 'image/png')).toEqual({ width: 640, height: 480 });
    expect(readImageSize(webpLossy(300, 200), 'image/webp')).toEqual({ width: 300, height: 200 });
    expect(readImageSize(webpLossless(16383, 2), 'image/webp')).toEqual({
      width: 16383,
      height: 2,
    });
    expect(readImageSize(webpExtended(16000, 9000), 'image/webp')).toEqual({
      width: 16000,
      height: 9000,
    });
    expect(readImageSize(jpeg(6, 4032, 3024), 'image/jpeg')).toEqual({ width: 4032, height: 3024 });
  });

  it('returns null for truncated or inconsistent headers', () => {
    expect(readImageSize(png(10, 10).slice(0, 20), 'image/png')).toBeNull();
    expect(readImageSize(riff(ascii('ALPH')), 'image/webp')).toBeNull();
    expect(readImageSize(new Uint8Array([0xff, 0xd8, 0xff]), 'image/jpeg')).toBeNull();
    expect(readImageSize(png(10, 10), 'image/jpeg')).toBeNull();
  });
});

describe('validateDimensions', () => {
  it('accepts phone-camera photos', () => {
    expect(validateDimensions({ width: 8160, height: 6120 })).toEqual({ ok: true }); // 50 MP
  });

  it('rejects images that would decode to too many pixels (decompression bombs)', () => {
    // A small PNG can declare 50000 x 50000 pixels: 10 GB once decoded to RGBA.
    const result = validateDimensions(readImageSize(png(50_000, 50_000), 'image/png'));
    expect(result).toMatchObject({ ok: false, reason: 'too-many-pixels' });
    if (!result.ok) expect(result.message).toMatch(/50 megapixels/);
    expect(validateDimensions({ width: MAX_PHOTO_PIXELS + 1, height: 1 }).ok).toBe(false);
  });

  it('rejects images whose size cannot be read', () => {
    expect(validateDimensions(null)).toMatchObject({ ok: false, reason: 'unreadable' });
    expect(validateDimensions({ width: 0, height: 10 })).toMatchObject({
      ok: false,
      reason: 'unreadable',
    });
  });
});

describe('orientation', () => {
  // 2 x 3 image, values 0..5 row-major:
  // 0 1 2
  // 3 4 5
  const img = { width: 3, height: 2, data: new Uint8Array([0, 1, 2, 3, 4, 5]) };
  const cases: [number, number[][]][] = [
    [
      1,
      [
        [0, 1, 2],
        [3, 4, 5],
      ],
    ],
    [
      2,
      [
        [2, 1, 0],
        [5, 4, 3],
      ],
    ],
    [
      3,
      [
        [5, 4, 3],
        [2, 1, 0],
      ],
    ],
    [
      4,
      [
        [3, 4, 5],
        [0, 1, 2],
      ],
    ],
    [
      5,
      [
        [0, 3],
        [1, 4],
        [2, 5],
      ],
    ],
    [
      6,
      [
        [3, 0],
        [4, 1],
        [5, 2],
      ],
    ],
    [
      7,
      [
        [5, 2],
        [4, 1],
        [3, 0],
      ],
    ],
    [
      8,
      [
        [2, 5],
        [1, 4],
        [0, 3],
      ],
    ],
  ];
  it.each(cases)('orientation %i matches the EXIF definition', (orientation, rows) => {
    const out = applyOrientation(img, orientation);
    expect(out.height).toBe(rows.length);
    expect(out.width).toBe(rows[0]!.length);
    expect(Array.from(out.data)).toEqual(rows.flat());
  });
  it('swaps dimensions for 5-8', () => {
    expect(orientedSize(640, 480, 6)).toEqual({ width: 480, height: 640 });
    expect(orientedSize(640, 480, 3)).toEqual({ width: 640, height: 480 });
  });
});

describe('fitWithin', () => {
  it('downscales huge photos to at most 2000 px on the long side', () => {
    expect(MAX_PHOTO_SIDE).toBe(2000);
    expect(fitWithin(8000, 6000)).toEqual({ width: 2000, height: 1500, scale: 0.25 });
    expect(fitWithin(1000, 6000)).toEqual({ width: 333, height: 2000, scale: 2000 / 6000 });
  });
  it('leaves small images alone', () => {
    expect(fitWithin(800, 600)).toEqual({ width: 800, height: 600, scale: 1 });
  });
});

describe('validateSelection', () => {
  it('accepts a line-shaped box', () => {
    expect(validateSelection({ x: 0, y: 0, width: 400, height: 60 })).toEqual({ ok: true });
  });
  it('rejects tiny boxes', () => {
    expect(validateSelection({ x: 0, y: 0, width: 5, height: 40 })).toMatchObject({
      ok: false,
      reason: 'too-small',
    });
  });
  it('rejects a box much wider than one line can be read', () => {
    expect(validateSelection({ x: 0, y: 0, width: 3000, height: 40 })).toMatchObject({
      ok: false,
      reason: 'too-wide',
    });
  });
  it('rejects a box taller than it is wide by a lot (several lines)', () => {
    expect(validateSelection({ x: 0, y: 0, width: 100, height: 400 })).toMatchObject({
      ok: false,
      reason: 'too-tall',
    });
  });
});

describe('isBlankCanvas', () => {
  it('is true for a white RGBA buffer and false once something is drawn', () => {
    const data = new Uint8ClampedArray(4 * 16).fill(255);
    expect(isBlankCanvas(data)).toBe(true);
    data[20] = 0;
    expect(isBlankCanvas(data)).toBe(false);
  });
  it('treats fully transparent pixels as blank', () => {
    expect(isBlankCanvas(new Uint8ClampedArray(4 * 4))).toBe(true);
  });
});
