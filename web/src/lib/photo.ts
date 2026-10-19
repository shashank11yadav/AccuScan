import {
  applyOrientation,
  fitWithin,
  readImageSize,
  readJpegInfo,
  validateDimensions,
  validateUpload,
  type Rect,
} from '../ml/image';
import { rgbaToGray, type GrayImage } from '../ml/preprocess';

export class UploadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'UploadError';
  }
}

/**
 * Validate, decode, orient and downscale a photo to greyscale (long side <= 2000 px).
 * The type comes from the magic bytes and the pixel count from the header, both
 * checked before the browser decodes anything.
 * Browsers apply EXIF orientation when decoding; if one does not (detected by
 * the decoded size still matching the stored, unrotated size), it is applied here.
 */
export async function decodePhoto(file: File): Promise<GrayImage> {
  const head = new Uint8Array(await file.slice(0, 64).arrayBuffer());
  const check = validateUpload({ size: file.size, head });
  if (!check.ok) throw new UploadError(check.message);
  const bytes = new Uint8Array(await file.arrayBuffer());
  const dimensions = validateDimensions(readImageSize(bytes, check.type));
  if (!dimensions.ok) throw new UploadError(dimensions.message);
  const info = check.type === 'image/jpeg' ? readJpegInfo(bytes) : null;
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch {
    throw new UploadError('That image could not be opened. It may be damaged.');
  }
  const fit = fitWithin(bitmap.width, bitmap.height);
  const canvas = document.createElement('canvas');
  canvas.width = fit.width;
  canvas.height = fit.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new UploadError('This browser cannot process images.');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, fit.width, fit.height);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, fit.width, fit.height);
  const notRotated =
    info !== null &&
    info.orientation >= 5 &&
    info.width !== info.height &&
    bitmap.width === info.width &&
    bitmap.height === info.height;
  bitmap.close();
  const gray = rgbaToGray(
    ctx.getImageData(0, 0, fit.width, fit.height).data,
    fit.width,
    fit.height,
  );
  return notRotated ? applyOrientation(gray, info.orientation) : gray;
}

export function cropGray(img: GrayImage, rect: Rect): GrayImage {
  const x0 = Math.max(0, Math.floor(rect.x));
  const y0 = Math.max(0, Math.floor(rect.y));
  const x1 = Math.min(img.width, Math.ceil(rect.x + rect.width));
  const y1 = Math.min(img.height, Math.ceil(rect.y + rect.height));
  const width = Math.max(0, x1 - x0);
  const height = Math.max(0, y1 - y0);
  const data = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    data.set(img.data.subarray((y0 + y) * img.width + x0, (y0 + y) * img.width + x1), y * width);
  }
  return { width, height, data };
}

export function grayToImageData(img: GrayImage): ImageData {
  const rgba = new Uint8ClampedArray(img.width * img.height * 4);
  for (let i = 0; i < img.width * img.height; i++) {
    const v = img.data[i] ?? 255;
    rgba[i * 4] = v;
    rgba[i * 4 + 1] = v;
    rgba[i * 4 + 2] = v;
    rgba[i * 4 + 3] = 255;
  }
  return new ImageData(rgba, img.width, img.height);
}

/** Normalise a drag from (ax, ay) to (bx, by) into a rectangle. */
export function rectFromDrag(ax: number, ay: number, bx: number, by: number): Rect {
  return {
    x: Math.min(ax, bx),
    y: Math.min(ay, by),
    width: Math.abs(bx - ax),
    height: Math.abs(by - ay),
  };
}
