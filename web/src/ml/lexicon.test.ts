import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { loadLexicon } from './lexicon';

const gz = readFileSync(
  fileURLToPath(new URL('../../public/lexicon/words.txt.gz', import.meta.url)),
);
const fakeFetch = (status: number, body: BodyInit | null) =>
  (() => Promise.resolve(new Response(body, { status }))) as typeof fetch;

describe('loadLexicon', () => {
  it('loads the bundled ~50k word list', async () => {
    const lexicon = await loadLexicon('words.txt.gz', fakeFetch(200, new Uint8Array(gz)));
    expect(lexicon.size).toBeGreaterThan(40_000);
    expect(lexicon.logprob('the')).not.toBeNull();
    expect(lexicon.logprob('hello')).not.toBeNull();
  });

  it('also accepts a list the server already decompressed', async () => {
    const lexicon = await loadLexicon(
      'words.txt.gz',
      fakeFetch(200, new Uint8Array(gunzipSync(gz))),
    );
    expect(lexicon.size).toBeGreaterThan(40_000);
  });

  it('reports HTTP errors', async () => {
    await expect(loadLexicon('missing', fakeFetch(404, null))).rejects.toThrow(/404/);
  });
});
