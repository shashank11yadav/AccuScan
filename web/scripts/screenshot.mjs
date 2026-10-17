// Screenshots of the production build at desktop and phone widths: the Write
// tab with a word drawn in Hershey strokes, a cropped photo, and a sample.
// Usage: npm run build && npm run preview (in another shell), then npm run screenshot.
import { chromium } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import { writeText } from '../e2e/hershey.ts';

const url = process.env.URL ?? 'http://localhost:5471/AccuScan/';
const out = process.env.OUT ?? 'test-results';
const note = fileURLToPath(new URL('../e2e/fixtures/note.png', import.meta.url));

async function done(page) {
  await page.waitForSelector('[data-testid=result][data-status=done]', { timeout: 15000 });
}

const browser = await chromium.launch();
for (const [name, width, height] of [
  ['desktop', 1280, 900],
  ['mobile', 375, 812],
]) {
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 2 });
  await page.goto(url);
  await page.getByText('Write, upload or pick a sample.').waitFor({ timeout: 30000 });

  await writeText(page, 'hello', 'scripts', { xHeightPx: name === 'mobile' ? 26 : 34 });
  await done(page);
  await page.screenshot({ path: `${out}/write-${name}.png`, fullPage: true });

  await page.getByRole('tab', { name: 'Photo' }).click();
  await page.getByTestId('photo-input').setInputFiles(note);
  const canvas = page.getByTestId('photo-canvas');
  await canvas.scrollIntoViewIfNeeded();
  const box = await canvas.boundingBox();
  await page.mouse.move(box.x + box.width * 0.03, box.y + box.height * 0.08);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.45, { steps: 8 });
  await page.mouse.up();
  await done(page);
  await page.screenshot({ path: `${out}/upload-${name}.png`, fullPage: true });

  await page.getByRole('tab', { name: 'Samples' }).click();
  await page.getByRole('button', { name: /hello world/ }).click();
  await done(page);
  await page.screenshot({ path: `${out}/samples-${name}.png`, fullPage: true });
  await page.close();
}
await browser.close();
console.log(`screenshots written to ${out}/`);
