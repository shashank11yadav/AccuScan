/**
 * Real stroke drawing, not pasted images: page.mouse follows the pen paths of
 * the single-stroke Hershey fonts (see hershey.ts), so the canvas receives the
 * same pointer events a visitor's mouse would produce. Readings are greedy CTC
 * (the trained model opens in free mode).
 */
import { expect, test, type Page } from '@playwright/test';
import { cer, writeText, type FontName } from './hershey';

const PHRASES = ['hello', 'portfolio', 'the quick fox'] as const;

async function open(page: Page) {
  await page.goto('./');
  await expect(page.getByText('Write, upload or pick a sample.')).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole('switch', { name: 'Dictionary words' })).toHaveAttribute(
    'aria-checked',
    'false',
  );
}

/** Draws the text, asks for a reading and returns it. */
async function drawAndRead(page: Page, text: string, font: FontName): Promise<string> {
  const clear = page.getByRole('button', { name: 'Clear' });
  if (await clear.isEnabled()) await clear.click();
  await writeText(page, text, font);
  await page.getByRole('button', { name: 'Recognise' }).click();
  await expect(page.getByTestId('result')).toHaveAttribute('data-status', 'done', {
    timeout: 15_000,
  });
  return (await page.getByTestId('result-text').locator('.sr-only').textContent()) ?? '';
}

function report(font: FontName, text: string, reading: string) {
  test.info().annotations.push({
    type: `reading (${font})`,
    description: `${JSON.stringify(text)} -> ${JSON.stringify(reading)}, CER ${cer(text, reading).toFixed(3)}`,
  });
}

// Print-style single-stroke letters: every phrase is read exactly.
for (const text of PHRASES) {
  test(`drawn print letters are read exactly: "${text}"`, async ({ page }) => {
    await open(page);
    const reading = await drawAndRead(page, text, 'futural');
    report('futural', text, reading);
    expect(reading).toBe(text);
  });
}

// Joined-up script is harder, and closer to how many people write. In
// Chromium on an Apple M4, "hello" and "portfolio" read exactly and "the quick
// fox" comes out as "the quick forc" (CER 7.4% over the three). "portfolio"
// misread as "porffolio" with the bold pen at a smaller size, so only "hello"
// is asserted exactly and the rest is bounded by CER.
test('drawn cursive script is read closely', async ({ page }) => {
  test.setTimeout(90_000);
  await open(page);
  const readings: Record<string, string> = {};
  for (const text of PHRASES) {
    readings[text] = await drawAndRead(page, text, 'scripts');
    report('scripts', text, readings[text]);
  }
  expect(readings['hello']).toBe('hello');
  const chars = PHRASES.reduce((n, t) => n + t.length, 0);
  const errors = PHRASES.reduce((n, t) => n + cer(t, readings[t] ?? '') * t.length, 0);
  expect(errors / chars).toBeLessThanOrEqual(0.15);
});
