import { Lexicon } from './decode';

const isGzip = (bytes: Uint8Array) => bytes[0] === 0x1f && bytes[1] === 0x8b;

/**
 * Load the bundled words.txt.gz ("word<TAB>count" lines) and build the trie.
 * Some servers send .gz files with Content-Encoding: gzip, so the browser has
 * already inflated them; GitHub Pages sends the raw archive. Both are handled
 * by checking the gzip magic bytes.
 */
export async function loadLexicon(url: string, fetchImpl: typeof fetch = fetch): Promise<Lexicon> {
  const response = await fetchImpl(url);
  if (!response.ok) throw new Error(`could not load the word list (HTTP ${response.status})`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  let text: string;
  if (isGzip(bytes)) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    text = await new Response(stream).text();
  } else {
    text = new TextDecoder().decode(bytes);
  }
  const lexicon = Lexicon.parse(text);
  if (lexicon.size === 0) throw new Error('the word list is empty');
  return lexicon;
}
