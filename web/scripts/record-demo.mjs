// Records docs/demo.gif: words written on the canvas with the mouse along
// Hershey font strokes, with the reading updating after each pause.
// Usage: npm run build && npm run preview (in another shell), then npm run demo-gif.
// Needs ffmpeg on PATH (or FFMPEG=/path/to/ffmpeg).
import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { writeText } from '../e2e/hershey.ts';

const url = process.env.URL ?? 'http://localhost:5471/AccuScan/';
const ffmpeg = process.env.FFMPEG ?? 'ffmpeg';
const output = fileURLToPath(new URL('../../docs/demo.gif', import.meta.url));
const WIDTH = 900;
const viewport = { width: 1180, height: 700 };

const dir = mkdtempSync(join(tmpdir(), 'accuscan-demo-'));
const browser = await chromium.launch();
const context = await browser.newContext({ viewport, recordVideo: { dir, size: viewport } });
const page = await context.newPage();
const started = Date.now();
await page.goto(url);
await page.getByText('Write, upload or pick a sample.').waitFor({ timeout: 30000 });

// Put the input and result panels at the top of the frame and measure them.
const panels = page.locator('section[aria-label="Input"]').locator('..');
await page.evaluate(() => {
  const el = document.querySelector('section[aria-label="Input"]')?.parentElement;
  if (el) window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - 16);
});
const box = await panels.boundingBox();
if (!box) throw new Error('panels not visible');
const crop = {
  x: Math.max(0, Math.round(box.x - 16)),
  y: Math.max(0, Math.round(box.y - 16)),
  w: Math.min(viewport.width, Math.round(box.width + 32)),
  h: Math.min(viewport.height, Math.round(box.height + 32)),
};
await page.waitForTimeout(400);
const t0 = (Date.now() - started) / 1000;

const done = () => page.waitForSelector('[data-testid=result][data-status=done]');
const clear = () => page.getByRole('button', { name: 'Clear' }).click();

await writeText(page, 'hello', 'scripts', { xHeightPx: 38, stepPx: 7 });
await done();
await page.waitForTimeout(1100);
await clear();
await page.waitForTimeout(300);
// Word by word: each pause is longer than the pad's 400 ms idle, so the
// reading grows as the line is written.
await writeText(page, 'the quick fox', 'futural', { xHeightPx: 30, stepPx: 7, wordPauseMs: 750 });
await done();
await page.waitForTimeout(1500);
const t1 = (Date.now() - started) / 1000;

const video = page.video();
await context.close();
await browser.close();
const webm = await video.path();

const filters =
  `trim=start=${t0.toFixed(2)}:end=${t1.toFixed(2)},setpts=PTS-STARTPTS,` +
  `crop=${crop.w}:${crop.h}:${crop.x}:${crop.y},fps=12,scale=${WIDTH}:-1:flags=lanczos`;
execFileSync(ffmpeg, [
  '-loglevel',
  'error',
  '-y',
  '-i',
  webm,
  '-vf',
  `${filters},split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle`,
  output,
]);
rmSync(dir, { recursive: true, force: true });
console.log(
  `${output}: ${(t1 - t0).toFixed(1)} s, ${(statSync(output).size / 1e6).toFixed(2)} MB, ${WIDTH} px wide`,
);
