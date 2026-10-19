import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const fixture = (name: string) => fileURLToPath(new URL(`./fixtures/${name}`, import.meta.url));
const publicFile = (name: string) => fileURLToPath(new URL(`../public/${name}`, import.meta.url));

/** Draws polyline strokes given in pad-relative [0, 1] coordinates with page.mouse. */
async function draw(page: Page, strokes: [number, number][][]) {
  const pad = page.getByTestId('write-pad');
  await pad.scrollIntoViewIfNeeded();
  const box = await pad.boundingBox();
  if (!box) throw new Error('write pad not visible');
  for (const stroke of strokes) {
    const [first, ...rest] = stroke;
    if (!first) continue;
    await page.mouse.move(box.x + first[0] * box.width, box.y + first[1] * box.height);
    await page.mouse.down();
    for (const [x, y] of rest) {
      await page.mouse.move(box.x + x * box.width, box.y + y * box.height, { steps: 6 });
    }
    await page.mouse.up();
  }
}

const WORD: [number, number][][] = [
  [
    [0.1, 0.3],
    [0.1, 0.75],
  ],
  [
    [0.1, 0.55],
    [0.18, 0.48],
    [0.22, 0.75],
  ],
  [
    [0.32, 0.3],
    [0.32, 0.75],
  ],
];

async function modelReady(page: Page) {
  await expect(page.getByText('Write, upload or pick a sample.')).toBeVisible({ timeout: 30_000 });
}

test.beforeEach(({ page }) => {
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  (page as Page & { errors?: string[] }).errors = errors;
});

test('the production build ships a strict CSP and no CSP violations', async ({ page }) => {
  await page.goto('./');
  const csp = await page
    .locator('meta[http-equiv="Content-Security-Policy"]')
    .getAttribute('content');
  expect(csp).toContain("script-src 'self' 'wasm-unsafe-eval'");
  expect(csp).toContain("connect-src 'self'");
  await modelReady(page);
  const errors = (page as Page & { errors?: string[] }).errors ?? [];
  expect(errors.filter((e) => /Content Security Policy|Refused to/.test(e))).toEqual([]);
});

test('the trained model shows its measured error rates and no preview notice', async ({ page }) => {
  const meta = JSON.parse(readFileSync(publicFile('models/model-meta.json'), 'utf8')) as {
    metrics: { sets: Record<string, { greedy: { cer: number } }> } | null;
  };
  expect(meta.metrics).not.toBeNull();
  await page.goto('./');
  await modelReady(page);
  await expect(page.getByTestId('preview-notice')).toHaveCount(0);
  const table = page.getByRole('table', { name: /Held-out error rates/ });
  for (const [label, key] of [
    ['GNHK test, words', 'gnhk_test_words'],
    ['GNHK test, lines', 'gnhk_test_lines'],
    ['Canvas style, held-out fonts', 'canvas_heldout_fonts'],
  ] as const) {
    const cer = meta.metrics?.sets[key]?.greedy.cer ?? NaN;
    await expect(table.getByRole('row', { name: new RegExp(`^${label}`) })).toContainText(
      `${(cer * 100).toFixed(1)}%`,
    );
  }
});

test('drawing strokes produces a reading after a pause', async ({ page }) => {
  await page.goto('./');
  await modelReady(page);
  await draw(page, WORD);
  const result = page.getByTestId('result');
  await expect(result).toHaveAttribute('data-status', 'done', { timeout: 15_000 });
  await expect(page.getByTestId('model-input')).toBeVisible();
  await expect(result).toContainText(/ms/);
  await expect(result).toContainText('WebAssembly');
});

test('undo and clear return the pad to empty', async ({ page }) => {
  await page.goto('./');
  await modelReady(page);
  await draw(page, WORD);
  await page.getByRole('button', { name: 'Undo' }).click();
  await page.getByRole('button', { name: 'Clear' }).click();
  await expect(page.getByTestId('result')).toHaveAttribute('data-status', 'idle');
  await expect(page.getByRole('button', { name: 'Undo' })).toBeDisabled();
});

test('recognising an empty canvas explains what to do', async ({ page }) => {
  await page.goto('./');
  await modelReady(page);
  await page.getByRole('button', { name: 'Recognise' }).click();
  await expect(page.getByTestId('result')).toHaveAttribute('data-status', 'idle');
  await draw(page, [
    [
      [0.5, 0.5],
      [0.5, 0.5],
    ],
  ]);
  await page.getByRole('button', { name: 'Clear' }).click();
  await page.getByRole('button', { name: 'Recognise' }).click();
  await expect(page.getByTestId('result')).toHaveAttribute('data-status', 'idle');
});

test('dictionary mode switches the decoder', async ({ page }) => {
  await page.goto('./');
  await modelReady(page);
  await draw(page, WORD);
  await expect(page.getByTestId('result')).toHaveAttribute('data-status', 'done', {
    timeout: 15_000,
  });
  await page.getByRole('switch', { name: 'Dictionary words' }).click();
  await expect(page.getByRole('switch', { name: 'Dictionary words' })).toHaveAttribute(
    'aria-checked',
    'true',
  );
  await expect(page.getByTestId('result')).toContainText('dictionary beam search', {
    timeout: 15_000,
  });
});

interface Box {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

/** Track, knob and label boxes of the dictionary switch, plus the knob/track contrast. */
function switchGeometry(page: Page) {
  return page.getByRole('switch', { name: 'Dictionary words' }).evaluate((el) => {
    const box = (r: DOMRect) => ({ left: r.left, right: r.right, top: r.top, bottom: r.bottom });
    const track = el.querySelector(':scope > span[aria-hidden="true"]');
    const knob = track?.firstElementChild;
    const label = Array.from(el.childNodes).find((n) => n.textContent === 'Dictionary words');
    if (!track || !knob || !label) throw new Error('unexpected switch markup');
    const range = document.createRange();
    range.selectNodeContents(label);
    const luminance = (css: string) => {
      const [r = 0, g = 0, b = 0] = (css.match(/[\d.]+/g) ?? []).map(Number).map((v) => {
        const c = v / 255;
        return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
      });
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };
    const a = luminance(getComputedStyle(knob).backgroundColor);
    const b = luminance(getComputedStyle(track).backgroundColor);
    return {
      track: box(track.getBoundingClientRect()),
      knob: box(knob.getBoundingClientRect()),
      label: box(range.getBoundingClientRect()),
      contrast: (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05),
    };
  });
}

function expectInside(inner: Box, outer: Box) {
  expect(inner.left).toBeGreaterThanOrEqual(outer.left - 0.5);
  expect(inner.right).toBeLessThanOrEqual(outer.right + 0.5);
  expect(inner.top).toBeGreaterThanOrEqual(outer.top - 0.5);
  expect(inner.bottom).toBeLessThanOrEqual(outer.bottom + 0.5);
}

for (const colorScheme of ['light', 'dark'] as const) {
  for (const width of [320, 1280]) {
    test(`the dictionary switch keeps its knob in the track (${colorScheme}, ${width} px)`, async ({
      page,
    }) => {
      await page.setViewportSize({ width, height: 800 });
      await page.emulateMedia({ colorScheme, reducedMotion: 'reduce' });
      await page.goto('./');
      const toggle = page.getByRole('switch', { name: 'Dictionary words' });
      await expect(toggle).toHaveAttribute('aria-checked', 'false');
      for (const checked of [false, true]) {
        await expect(toggle).toHaveAttribute('aria-checked', String(checked));
        const g = await switchGeometry(page);
        expectInside(g.knob, g.track);
        expect(g.track.right).toBeLessThanOrEqual(g.label.left);
        const knobCentre = (g.knob.left + g.knob.right) / 2;
        const trackCentre = (g.track.left + g.track.right) / 2;
        if (checked) expect(knobCentre).toBeGreaterThan(trackCentre);
        else expect(knobCentre).toBeLessThan(trackCentre);
        expect(g.contrast).toBeGreaterThanOrEqual(3);
        if (!checked) await toggle.click();
      }
    });
  }
}

test('the dictionary switch works from the keyboard and shows focus', async ({ page }) => {
  await page.goto('./');
  const toggle = page.getByRole('switch', { name: 'Dictionary words' });
  await toggle.focus();
  await page.keyboard.press('Space');
  await expect(toggle).toHaveAttribute('aria-checked', 'true');
  await page.keyboard.press('Enter');
  await expect(toggle).toHaveAttribute('aria-checked', 'false');
  const outline = await toggle.evaluate((el) => getComputedStyle(el).outlineStyle);
  expect(outline).not.toBe('none');
});

test('a missing model shows an error with a working Retry', async ({ page }) => {
  let block = true;
  await page.route('**/models/model.onnx', (route) =>
    block ? route.fulfill({ status: 404, body: 'not found' }) : route.continue(),
  );
  await page.goto('./');
  const alert = page.getByRole('alert').filter({ hasText: 'The model could not be loaded' });
  await expect(alert).toContainText('404');
  // The page stays usable: the inputs are still there.
  await expect(page.getByTestId('write-pad')).toBeVisible();
  block = false;
  await alert.getByRole('button', { name: 'Retry' }).click();
  await modelReady(page);
  await expect(alert).toHaveCount(0);
});

test('an unsupported upload is rejected with a message', async ({ page }) => {
  await page.goto('./');
  await page.getByRole('tab', { name: 'Photo' }).click();
  await page.getByTestId('photo-input').setInputFiles(fixture('not-an-image.png'));
  await expect(page.getByRole('alert')).toContainText('JPEG, PNG or WebP');
});

test('a small file that declares a huge image is rejected before decoding', async ({ page }) => {
  // A real 50000 x 50000 PNG of one colour compresses to about a megabyte;
  // the header alone is enough to tell it would need 10 GB once decoded.
  const u32 = (v: number) => [v >>> 24, (v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff];
  const header = [
    ...[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a],
    ...u32(13),
    ...[0x49, 0x48, 0x44, 0x52],
    ...u32(50_000),
    ...u32(50_000),
    8,
    0,
    0,
    0,
    0,
  ];
  await page.goto('./');
  await page.getByRole('tab', { name: 'Photo' }).click();
  await page.getByTestId('photo-input').setInputFiles({
    name: 'bomb.png',
    mimeType: 'image/png',
    buffer: Buffer.from(header),
  });
  await expect(page.getByRole('alert')).toContainText('50000 × 50000 pixels');
  await expect(page.getByTestId('photo-canvas')).toHaveCount(0);
});

test('a photo can be cropped to one line and read', async ({ page }) => {
  await page.goto('./');
  await modelReady(page);
  await page.getByRole('tab', { name: 'Photo' }).click();
  await page.getByTestId('photo-input').setInputFiles(fixture('note.png'));
  const canvas = page.getByTestId('photo-canvas');
  await expect(canvas).toBeVisible();
  await expect(page.getByText('640 × 220 px')).toBeVisible();
  const box = await canvas.boundingBox();
  if (!box) throw new Error('photo not visible');
  await page.mouse.move(box.x + box.width * 0.03, box.y + box.height * 0.08);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.45, { steps: 8 });
  await page.mouse.up();
  await expect(page.getByTestId('result')).toHaveAttribute('data-status', 'done', {
    timeout: 15_000,
  });
});

test('a tiny selection is rejected', async ({ page }) => {
  await page.goto('./');
  await modelReady(page);
  await page.getByRole('tab', { name: 'Photo' }).click();
  await page.getByTestId('photo-input').setInputFiles(fixture('note.png'));
  const box = await page.getByTestId('photo-canvas').boundingBox();
  if (!box) throw new Error('photo not visible');
  await page.mouse.move(box.x + 20, box.y + 20);
  await page.mouse.down();
  await page.mouse.move(box.x + 22, box.y + 21);
  await page.mouse.up();
  await expect(page.getByRole('alert')).toContainText('too small');
});

test('EXIF orientation is applied to uploaded JPEGs', async ({ page }) => {
  await page.goto('./');
  await page.getByRole('tab', { name: 'Photo' }).click();
  // Stored as 120 x 360 with orientation 6; it must display as 360 x 120.
  await page.getByTestId('photo-input').setInputFiles(fixture('rotated-exif6.jpg'));
  await expect(page.getByText('360 × 120 px')).toBeVisible();
});

test('samples are read and show the written text', async ({ page }) => {
  await page.goto('./');
  await modelReady(page);
  await page.getByRole('tab', { name: 'Samples' }).click();
  await page.getByRole('button', { name: /hello world/ }).click();
  await expect(page.getByTestId('result')).toHaveAttribute('data-status', 'done', {
    timeout: 15_000,
  });
  await expect(page.getByTestId('result')).toContainText('Written text: hello world');
});

test('tabs work from the keyboard', async ({ page }) => {
  await page.goto('./');
  await page.getByRole('tab', { name: 'Write' }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('tab', { name: 'Photo' })).toBeFocused();
  await expect(page.getByRole('tab', { name: 'Photo' })).toHaveAttribute('aria-selected', 'true');
});

for (const [name, viewport] of [
  ['desktop', { width: 1280, height: 900 }],
  ['phone', { width: 320, height: 700 }],
] as const) {
  test(`has no detectable accessibility violations (${name})`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await page.goto('./');
    await modelReady(page);
    await draw(page, WORD);
    await expect(page.getByTestId('result')).toHaveAttribute('data-status', 'done', {
      timeout: 15_000,
    });
    const results = await new AxeBuilder({ page }).analyze();
    expect(results.violations).toEqual([]);
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth,
    );
    expect(overflow).toBe(false);
  });
}
