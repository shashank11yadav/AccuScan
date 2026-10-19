/**
 * Input checks for uploads, photo orientation and selections. Everything here
 * is pure so it can be unit-tested without a browser.
 */

import type { GrayImage } from './preprocess';

export const MAX_UPLOAD_BYTES = 15 * 1024 * 1024;
export const MAX_PHOTO_SIDE = 2000;
/**
 * Decoding happens at full size before any downscaling, so the pixel count is
 * checked from the file header first. 50 MP covers phone cameras' default
 * modes; a few kilobytes of PNG can otherwise declare 50000 x 50000 pixels.
 */
export const MAX_PHOTO_PIXELS = 50_000_000;
export const MIN_SELECTION = { width: 16, height: 8 };
/** Wider than this (width / height) is more than one model input can read clearly. */
export const MAX_SELECTION_ASPECT = 32;
/** Taller than this (height / width) almost certainly covers several lines. */
export const MAX_SELECTION_TALLNESS = 2;

export type ImageType = 'image/jpeg' | 'image/png' | 'image/webp';

export type UploadCheck =
  | { ok: true; type: ImageType }
  | { ok: false; reason: 'empty' | 'too-large' | 'unsupported'; message: string };

const startsWith = (bytes: Uint8Array, prefix: readonly number[], offset = 0): boolean =>
  bytes.length >= offset + prefix.length && prefix.every((b, i) => bytes[offset + i] === b);

/** Identify the image type from its first bytes; the file name and MIME type are not trusted. */
export function sniffImageType(head: Uint8Array): ImageType | null {
  if (startsWith(head, [0xff, 0xd8, 0xff])) return 'image/jpeg';
  if (startsWith(head, [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])) return 'image/png';
  if (startsWith(head, [0x52, 0x49, 0x46, 0x46]) && startsWith(head, [0x57, 0x45, 0x42, 0x50], 8)) {
    return 'image/webp';
  }
  return null;
}

export function validateUpload(file: { size: number; head: Uint8Array }): UploadCheck {
  if (file.size === 0) return { ok: false, reason: 'empty', message: 'That file is empty.' };
  if (file.size > MAX_UPLOAD_BYTES) {
    return {
      ok: false,
      reason: 'too-large',
      message: `That file is ${(file.size / 1024 / 1024).toFixed(1)} MB. The limit is 15 MB.`,
    };
  }
  const type = sniffImageType(file.head);
  if (type === null) {
    return {
      ok: false,
      reason: 'unsupported',
      message: 'That file type is not supported. Use a JPEG, PNG or WebP photo.',
    };
  }
  return { ok: true, type };
}

export interface JpegInfo {
  width: number;
  height: number;
  /** EXIF orientation 1-8 (1 when absent or invalid). */
  orientation: number;
}

function exifOrientation(view: DataView, start: number, end: number): number {
  // "Exif\0\0" then a TIFF header.
  if (end - start < 14 || view.getUint32(start) !== 0x45786966) return 1;
  const tiff = start + 6;
  const order = view.getUint16(tiff);
  if (order !== 0x4949 && order !== 0x4d4d) return 1;
  const little = order === 0x4949;
  const ifd = tiff + view.getUint32(tiff + 4, little);
  if (ifd + 2 > end) return 1;
  const entries = view.getUint16(ifd, little);
  for (let i = 0; i < entries; i++) {
    const entry = ifd + 2 + i * 12;
    if (entry + 12 > end) return 1;
    if (view.getUint16(entry, little) === 0x0112) {
      const value = view.getUint16(entry + 8, little);
      return value >= 1 && value <= 8 ? value : 1;
    }
  }
  return 1;
}

/** Size and EXIF orientation from a JPEG's markers, or null if it cannot be parsed. */
export function readJpegInfo(bytes: Uint8Array): JpegInfo | null {
  if (!startsWith(bytes, [0xff, 0xd8])) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let orientation = 1;
  let offset = 2;
  while (offset + 4 <= bytes.length) {
    if (bytes[offset] !== 0xff) return null;
    if (bytes[offset + 1] === 0xff) {
      offset++; // fill byte
      continue;
    }
    const marker = bytes[offset + 1] ?? 0;
    const length = view.getUint16(offset + 2);
    const segment = offset + 4;
    if (length < 2 || offset + 2 + length > bytes.length) return null;
    if (marker === 0xe1) orientation = exifOrientation(view, segment, offset + 2 + length);
    const isSof = marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker);
    if (isSof) {
      if (segment + 5 > bytes.length) return null;
      return {
        height: view.getUint16(segment + 1),
        width: view.getUint16(segment + 3),
        orientation,
      };
    }
    offset += 2 + length;
  }
  return null;
}

export interface ImageSize {
  width: number;
  height: number;
}

const ascii = (bytes: Uint8Array, offset: number, text: string) =>
  startsWith(
    bytes,
    Array.from(text, (c) => c.charCodeAt(0)),
    offset,
  );

/** Pixel size from the file header without decoding, or null if it cannot be read. */
export function readImageSize(bytes: Uint8Array, type: ImageType): ImageSize | null {
  if (sniffImageType(bytes) !== type) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (type === 'image/jpeg') {
    const info = readJpegInfo(bytes);
    return info && { width: info.width, height: info.height };
  }
  if (type === 'image/png') {
    if (bytes.length < 24 || !ascii(bytes, 12, 'IHDR')) return null;
    return { width: view.getUint32(16), height: view.getUint32(20) };
  }
  const u24 = (at: number) => view.getUint16(at, true) | ((bytes[at + 2] ?? 0) << 16);
  if (ascii(bytes, 12, 'VP8X') && bytes.length >= 30) {
    return { width: u24(24) + 1, height: u24(27) + 1 };
  }
  if (ascii(bytes, 12, 'VP8 ') && bytes.length >= 30 && startsWith(bytes, [0x9d, 0x01, 0x2a], 23)) {
    return { width: view.getUint16(26, true) & 0x3fff, height: view.getUint16(28, true) & 0x3fff };
  }
  if (ascii(bytes, 12, 'VP8L') && bytes.length >= 25 && bytes[20] === 0x2f) {
    const bits = view.getUint32(21, true);
    return { width: (bits & 0x3fff) + 1, height: ((bits >>> 14) & 0x3fff) + 1 };
  }
  return null;
}

export type DimensionCheck =
  { ok: true } | { ok: false; reason: 'unreadable' | 'too-many-pixels'; message: string };

export function validateDimensions(size: ImageSize | null): DimensionCheck {
  if (!size || size.width < 1 || size.height < 1) {
    return {
      ok: false,
      reason: 'unreadable',
      message: 'That image could not be opened. It may be damaged.',
    };
  }
  if (size.width * size.height > MAX_PHOTO_PIXELS) {
    return {
      ok: false,
      reason: 'too-many-pixels',
      message: `That image is ${size.width} × ${size.height} pixels. The limit is 50 megapixels.`,
    };
  }
  return { ok: true };
}

export function orientedSize(width: number, height: number, orientation: number) {
  return orientation >= 5 ? { width: height, height: width } : { width, height };
}

/** Turn stored pixels into display orientation (same table as the pipeline's fixture code). */
export function applyOrientation(img: GrayImage, orientation: number): GrayImage {
  const { width: w, height: h, data } = img;
  if (orientation < 2 || orientation > 8) return img;
  const size = orientedSize(w, h, orientation);
  const out = new Uint8Array(w * h);
  for (let y = 0; y < size.height; y++) {
    for (let x = 0; x < size.width; x++) {
      let sx: number;
      let sy: number;
      switch (orientation) {
        case 2:
          [sx, sy] = [w - 1 - x, y];
          break;
        case 3:
          [sx, sy] = [w - 1 - x, h - 1 - y];
          break;
        case 4:
          [sx, sy] = [x, h - 1 - y];
          break;
        case 5:
          [sx, sy] = [y, x];
          break;
        case 6:
          [sx, sy] = [y, h - 1 - x];
          break;
        case 7:
          [sx, sy] = [w - 1 - y, h - 1 - x];
          break;
        default:
          [sx, sy] = [w - 1 - y, x];
      }
      out[y * size.width + x] = data[sy * w + sx] ?? 255;
    }
  }
  return { ...size, data: out };
}

/** Scale so the long side is at most `max` pixels (never upscale). */
export function fitWithin(width: number, height: number, max = MAX_PHOTO_SIDE) {
  const scale = Math.min(1, max / Math.max(width, height));
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
    scale,
  };
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export type SelectionCheck =
  { ok: true } | { ok: false; reason: 'too-small' | 'too-wide' | 'too-tall'; message: string };

export function validateSelection(rect: Rect): SelectionCheck {
  if (rect.width < MIN_SELECTION.width || rect.height < MIN_SELECTION.height) {
    return {
      ok: false,
      reason: 'too-small',
      message: 'That box is too small. Drag a box around one line of writing.',
    };
  }
  if (rect.width / rect.height > MAX_SELECTION_ASPECT) {
    return {
      ok: false,
      reason: 'too-wide',
      message: 'That line is too long to read in one go. Select a few words at a time.',
    };
  }
  if (rect.height / rect.width > MAX_SELECTION_TALLNESS) {
    return {
      ok: false,
      reason: 'too-tall',
      message: 'That box looks like several lines. Select just one line.',
    };
  }
  return { ok: true };
}

/** True when an RGBA buffer has no visible mark (white or transparent everywhere). */
export function isBlankCanvas(rgba: Uint8ClampedArray | Uint8Array): boolean {
  for (let i = 0; i < rgba.length; i += 4) {
    const alpha = rgba[i + 3] ?? 0;
    if (alpha === 0) continue;
    if ((rgba[i] ?? 255) < 250 || (rgba[i + 1] ?? 255) < 250 || (rgba[i + 2] ?? 255) < 250) {
      return false;
    }
  }
  return true;
}
