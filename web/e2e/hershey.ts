/**
 * Writes text on the canvas the way a person would: page.mouse follows the
 * pen strokes of a single-stroke Hershey font, one pen-down per stroke.
 * The glyphs are vendored in fixtures/hershey-glyphs.json (source and licence
 * are recorded in that file).
 */
import type { Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

type Point = [number, number];

interface Glyph {
  advance: number;
  strokes: Point[][];
}

interface HersheyFont {
  name: string;
  xHeight: number;
  glyphs: Record<string, Glyph>;
}

const data = JSON.parse(
  readFileSync(fileURLToPath(new URL('./fixtures/hershey-glyphs.json', import.meta.url)), 'utf8'),
) as { fonts: Record<string, HersheyFont> };

export type FontName = 'futural' | 'scripts';

export function hersheyFont(name: FontName): HersheyFont {
  const font = data.fonts[name];
  if (!font) throw new Error(`no Hershey font ${name} in the fixture`);
  return font;
}

export interface WriteOptions {
  /** Height of a lower-case x on screen, in CSS pixels. */
  xHeightPx?: number;
  /** Baseline as a fraction of the pad height (the pad's guide line is at 0.68). */
  baseline?: number;
  /** Distance the mouse travels per move event, in CSS pixels. */
  stepPx?: number;
  /** Pause between words, in ms; longer than the pad's 400 ms idle reads the line so far. */
  wordPauseMs?: number;
}

/** Converts text into pad-relative polylines in CSS pixels. */
export function layoutText(
  text: string,
  font: HersheyFont,
  pad: { width: number; height: number },
  { xHeightPx = 34, baseline = 0.68 }: WriteOptions = {},
): { word: number; points: Point[] }[] {
  const glyphs = Array.from(text).map((ch) => {
    const glyph = font.glyphs[ch];
    if (!glyph) throw new Error(`the fixture has no glyph for ${JSON.stringify(ch)}`);
    return glyph;
  });
  const total = glyphs.reduce((sum, g) => sum + g.advance, 0);
  // Shrink to fit if the line would not leave a margin on each side.
  const scale = Math.min(xHeightPx / font.xHeight, (pad.width * 0.88) / total);
  let x = (pad.width - total * scale) / 2;
  const y0 = pad.height * baseline;
  const strokes: { word: number; points: Point[] }[] = [];
  let word = 0;
  Array.from(text).forEach((ch, i) => {
    const glyph = glyphs[i];
    if (!glyph) return;
    if (ch === ' ') word++;
    for (const stroke of glyph.strokes) {
      strokes.push({
        word,
        points: stroke.map(([gx, gy]) => [x + gx * scale, y0 - gy * scale] as Point),
      });
    }
    x += glyph.advance * scale;
  });
  return strokes;
}

/** Draws text on the write pad with mouse strokes. */
export async function writeText(
  page: Page,
  text: string,
  fontName: FontName,
  options: WriteOptions = {},
): Promise<void> {
  const pad = page.getByTestId('write-pad');
  await pad.scrollIntoViewIfNeeded();
  const box = await pad.boundingBox();
  if (!box) throw new Error('write pad not visible');
  const stepPx = options.stepPx ?? 5;
  let word = 0;
  for (const stroke of layoutText(text, hersheyFont(fontName), box, options)) {
    if (stroke.word !== word) {
      word = stroke.word;
      if (options.wordPauseMs) await page.waitForTimeout(options.wordPauseMs);
    }
    const [first, ...rest] = stroke.points;
    if (!first) continue;
    let [px, py] = first;
    await page.mouse.move(box.x + px, box.y + py);
    await page.mouse.down();
    for (const [x, y] of rest) {
      const steps = Math.max(1, Math.ceil(Math.hypot(x - px, y - py) / stepPx));
      await page.mouse.move(box.x + x, box.y + y, { steps });
      [px, py] = [x, y];
    }
    await page.mouse.up();
  }
}

/** Character error rate: Levenshtein distance divided by the reference length. */
export function cer(reference: string, hypothesis: string): number {
  const a = Array.from(reference);
  const b = Array.from(hypothesis);
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const row = [i];
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      row.push(Math.min((prev[j] ?? 0) + 1, (row[j - 1] ?? 0) + 1, (prev[j - 1] ?? 0) + cost));
    }
    prev = row;
  }
  return (prev[b.length] ?? 0) / Math.max(1, a.length);
}
