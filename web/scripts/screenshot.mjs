// Screenshots of the production build at desktop and phone widths.
// Usage: npm run build && npm run preview (in another shell), then npm run screenshot.
import { chromium } from '@playwright/test';

const url = process.env.URL ?? 'http://localhost:5471/AccuScan/';
const out = process.env.OUT ?? 'test-results';
const browser = await chromium.launch();
for (const [name, width, height] of [
  ['desktop', 1280, 900],
  ['mobile', 375, 812],
]) {
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 2 });
  await page.goto(url);
  await page.getByTestId('write-pad').waitFor();
  await page.waitForFunction(() => document.querySelector('[data-testid=preview-notice]') !== null);
  await page.getByTestId('write-pad').scrollIntoViewIfNeeded();
  const box = await page.getByTestId('write-pad').boundingBox();
  if (box) {
    const strokes = [
      [
        [0.1, 0.35],
        [0.1, 0.75],
      ],
      [
        [0.1, 0.55],
        [0.2, 0.5],
        [0.22, 0.75],
      ],
      [
        [0.3, 0.55],
        [0.38, 0.52],
        [0.36, 0.75],
        [0.3, 0.7],
        [0.38, 0.62],
      ],
      [
        [0.46, 0.3],
        [0.46, 0.75],
      ],
      [
        [0.53, 0.3],
        [0.53, 0.75],
      ],
      [
        [0.65, 0.55],
        [0.6, 0.65],
        [0.65, 0.75],
        [0.7, 0.65],
        [0.65, 0.55],
      ],
    ];
    for (const stroke of strokes) {
      const [first, ...rest] = stroke;
      await page.mouse.move(box.x + first[0] * box.width, box.y + first[1] * box.height);
      await page.mouse.down();
      for (const [x, y] of rest) {
        await page.mouse.move(box.x + x * box.width, box.y + y * box.height, { steps: 8 });
      }
      await page.mouse.up();
    }
    await page.waitForSelector('[data-testid=result][data-status=done]', { timeout: 15000 });
  }
  await page.screenshot({ path: `${out}/screenshot-${name}.png`, fullPage: true });
  await page.close();
}
await browser.close();
console.log(`screenshots written to ${out}/`);
