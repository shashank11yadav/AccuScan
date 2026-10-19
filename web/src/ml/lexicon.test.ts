import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { LEXICON_URL } from '../lib/links';
import { loadLexicon } from './lexicon';

const gz = readFileSync(
  fileURLToPath(new URL('../../public/lexicon/words.txt.gz', import.meta.url)),
);
const requested: unknown[] = [];
const fakeFetch = (status: number, body: BodyInit | null) =>
  ((input: unknown) => {
    requested.push(input);
    return Promise.resolve(new Response(body, { status }));
  }) as typeof fetch;

describe('loadLexicon', () => {
  it('loads the bundled ~50k word list', async () => {
    const lexicon = await loadLexicon(fakeFetch(200, new Uint8Array(gz)));
    expect(lexicon.size).toBeGreaterThan(40_000);
    expect(lexicon.logprob('the')).not.toBeNull();
    expect(lexicon.logprob('hello')).not.toBeNull();
  });

  it('also accepts a list the server already decompressed', async () => {
    const lexicon = await loadLexicon(fakeFetch(200, new Uint8Array(gunzipSync(gz))));
    expect(lexicon.size).toBeGreaterThan(40_000);
  });

  it('reports HTTP errors', async () => {
    await expect(loadLexicon(fakeFetch(404, null))).rejects.toThrow(/404/);
  });

  it('only ever requests the bundled list, a same-origin path under the base URL', async () => {
    requested.length = 0;
    await loadLexicon(fakeFetch(200, new Uint8Array(gz)));
    expect(requested).toEqual([LEXICON_URL]);
    expect(LEXICON_URL).toBe(`${import.meta.env.BASE_URL}lexicon/words.txt.gz`);
    expect(new URL(LEXICON_URL, 'https://site.test/').origin).toBe('https://site.test');
  });
});
