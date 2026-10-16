import { describe, expect, it } from 'vitest';
import { cropGray, rectFromDrag } from './photo';

describe('cropGray', () => {
  const img = { width: 4, height: 3, data: new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) };
  it('copies the selected rows and columns', () => {
    expect(Array.from(cropGray(img, { x: 1, y: 1, width: 2, height: 2 }).data)).toEqual([
      5, 6, 9, 10,
    ]);
  });
  it('clamps boxes that run off the image', () => {
    const out = cropGray(img, { x: 2.5, y: -3, width: 10, height: 4 });
    expect([out.width, out.height]).toEqual([2, 1]);
    expect(Array.from(out.data)).toEqual([2, 3]);
  });
});

describe('rectFromDrag', () => {
  it('works in any drag direction', () => {
    expect(rectFromDrag(10, 20, 4, 2)).toEqual({ x: 4, y: 2, width: 6, height: 18 });
  });
});
