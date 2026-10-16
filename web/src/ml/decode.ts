/**
 * CTC decoding, a port of pipeline/src/accuscan/decode.py with the same
 * iteration order and tie-breaking. Checked against shared/fixtures/decode.json.
 */

export const BLANK = 0;
/** Index 0 is the CTC blank, then printable ASCII 32-126. */
export const CHARSET: readonly string[] = [
  '<blank>',
  ...Array.from({ length: 95 }, (_, i) => String.fromCharCode(32 + i)),
];
export const NUM_CLASSES = CHARSET.length;

export const DEFAULT_BEAM_WIDTH = 16;
export const DEFAULT_LM_WEIGHT = 0.3;
export const DEFAULT_PRUNE = 1e-3;
export const DEFAULT_MAX_DROP_PER_WORD = 10;

/** Log-probs [T, C], row-major. */
export interface LogProbs {
  data: Float32Array | Float64Array;
  frames: number;
  classes: number;
}

export interface Decoded {
  text: string;
  /** One value per character of `text`, in (0, 1]. */
  confidences: number[];
  /** Dictionary mode only: no plausible dictionary reading, so this is the greedy one. */
  fallback: boolean;
}

const at = (lp: LogProbs, t: number, c: number): number => lp.data[t * lp.classes + c] ?? -Infinity;

export function logAdd(a: number, b: number): number {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const hi = Math.max(a, b);
  return hi + Math.log1p(Math.exp(-Math.abs(a - b)));
}

function labelsToText(labels: readonly number[]): string {
  return labels.map((i) => CHARSET[i] ?? '').join('');
}

export function encode(text: string): number[] {
  return Array.from(text, (ch) => {
    const code = ch.charCodeAt(0);
    if (ch.length !== 1 || code < 32 || code > 126) throw new Error(`not in charset: ${ch}`);
    return code - 31;
  });
}

export function isLetter(ch: string): boolean {
  return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z');
}

export function countWords(text: string): number {
  let words = 0;
  let inside = false;
  for (const ch of text) {
    const letter = isLetter(ch);
    if (letter && !inside) words++;
    inside = letter;
  }
  return words;
}

export function greedyDecode(lp: LogProbs): Decoded {
  const labels: number[] = [];
  const confidences: number[] = [];
  let previous = -1;
  for (let t = 0; t < lp.frames; t++) {
    let best = 0;
    let bestValue = at(lp, t, 0);
    for (let c = 1; c < lp.classes; c++) {
      const v = at(lp, t, c);
      if (v > bestValue) {
        bestValue = v;
        best = c;
      }
    }
    const p = Math.exp(bestValue);
    if (best !== BLANK && best !== previous) {
      labels.push(best);
      confidences.push(p);
    } else if (best !== BLANK && best === previous) {
      confidences[confidences.length - 1] = Math.max(confidences[confidences.length - 1] ?? 0, p);
    }
    previous = best;
  }
  return { text: labelsToText(labels), confidences, fallback: false };
}

/** Per-label max posterior along the Viterbi CTC alignment (zeros if impossible). */
export function alignConfidences(lp: LogProbs, labels: readonly number[]): number[] {
  const n = labels.length;
  if (n === 0) return [];
  const ext: number[] = [BLANK];
  for (const label of labels) ext.push(label, BLANK);
  const S = ext.length;
  const T = lp.frames;
  const score = Array.from({ length: T }, () => new Float64Array(S).fill(-Infinity));
  const back = Array.from({ length: T }, () => new Int32Array(S));
  score[0]![0] = at(lp, 0, ext[0]!);
  if (S > 1) score[0]![1] = at(lp, 0, ext[1]!);
  for (let t = 1; t < T; t++) {
    const prev = score[t - 1]!;
    for (let s = 0; s < S; s++) {
      let best = prev[s]!;
      let arg = s;
      if (s >= 1 && prev[s - 1]! > best) {
        best = prev[s - 1]!;
        arg = s - 1;
      }
      if (s >= 2 && ext[s] !== BLANK && ext[s] !== ext[s - 2] && prev[s - 2]! > best) {
        best = prev[s - 2]!;
        arg = s - 2;
      }
      if (best === -Infinity) continue;
      score[t]![s] = best + at(lp, t, ext[s]!);
      back[t]![s] = arg;
    }
  }
  const last = T - 1;
  let end = S - 1;
  if (score[last]![S - 2]! > score[last]![S - 1]!) end = S - 2;
  if (score[last]![end] === -Infinity) return new Array<number>(n).fill(0);
  const confidences = new Array<number>(n).fill(0);
  let s = end;
  for (let t = last; t >= 0; t--) {
    if (ext[s] !== BLANK) {
      const idx = (s - 1) >> 1;
      confidences[idx] = Math.max(confidences[idx]!, Math.exp(at(lp, t, ext[s]!)));
    }
    if (t > 0) s = back[t]![s]!;
  }
  return confidences;
}

export interface TrieNode {
  children: Map<string, TrieNode>;
  /** Set when the path from the root spells a word. */
  logprob: number | null;
}

export class Lexicon {
  readonly root: TrieNode;
  readonly size: number;

  private constructor(root: TrieNode, size: number) {
    this.root = root;
    this.size = size;
  }

  /** Same construction as the Python Lexicon.from_counts (insertion order matters not). */
  static fromCounts(counts: Iterable<readonly [string, number]>): Lexicon {
    const entries = [...counts];
    let total = 0;
    for (const [, count] of entries) total += count;
    const root: TrieNode = { children: new Map(), logprob: null };
    for (const [word, count] of entries) {
      let node = root;
      for (const ch of word.toLowerCase()) {
        let child = node.children.get(ch);
        if (!child) {
          child = { children: new Map(), logprob: null };
          node.children.set(ch, child);
        }
        node = child;
      }
      node.logprob = Math.log(count / total);
    }
    return new Lexicon(root, entries.length);
  }

  /** Parses "word<TAB>count" lines (the bundled words.txt.gz once decompressed). */
  static parse(text: string): Lexicon {
    const entries: [string, number][] = [];
    for (const line of text.split('\n')) {
      const tab = line.indexOf('\t');
      if (tab <= 0) continue;
      const count = Number(line.slice(tab + 1));
      if (Number.isFinite(count) && count > 0) entries.push([line.slice(0, tab), count]);
    }
    return Lexicon.fromCounts(entries);
  }

  logprob(word: string): number | null {
    let node: TrieNode | undefined = this.root;
    for (const ch of word.toLowerCase()) {
      node = node.children.get(ch);
      if (!node) return null;
    }
    return node.logprob;
  }
}

interface Beam {
  prefix: number[];
  key: string;
  node: TrieNode | null;
  lm: number;
  pb: number;
  pnb: number;
}

const total = (b: Beam): number => logAdd(b.pb, b.pnb);

function advance(
  lexicon: Lexicon | null,
  node: TrieNode | null,
  lm: number,
  ch: string,
  lmWeight: number,
): [TrieNode | null, number] | null {
  if (lexicon === null || node === null) return [null, lm];
  const root = lexicon.root;
  if (isLetter(ch)) {
    const child = node.children.get(ch.toLowerCase());
    return child ? [child, lm] : null;
  }
  if (ch === "'" && node !== root && node.children.has("'")) {
    return [node.children.get("'")!, lm];
  }
  if (node === root) return [root, lm];
  if (node.logprob !== null) return [root, lm + lmWeight * node.logprob];
  return null;
}

function finish(lexicon: Lexicon | null, beam: Beam, lmWeight: number): number | null {
  if (lexicon === null || beam.node === lexicon.root) return total(beam) + beam.lm;
  if (beam.node === null || beam.node.logprob === null) return null;
  return total(beam) + beam.lm + lmWeight * beam.node.logprob;
}

/** Lexicographic order of label arrays, as Python compares tuples. */
function comparePrefix(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    if (a[i] !== b[i]) return a[i]! - b[i]!;
  }
  return a.length - b.length;
}

export interface BeamOptions {
  beamWidth?: number;
  lmWeight?: number;
  prune?: number;
  maxDropPerWord?: number;
}

export function beamSearch(
  lp: LogProbs,
  lexicon: Lexicon | null,
  options: BeamOptions = {},
): Decoded {
  const beamWidth = options.beamWidth ?? DEFAULT_BEAM_WIDTH;
  const lmWeight = options.lmWeight ?? DEFAULT_LM_WEIGHT;
  const logPrune = Math.log(options.prune ?? DEFAULT_PRUNE);
  const maxDrop = options.maxDropPerWord ?? DEFAULT_MAX_DROP_PER_WORD;

  let beams: Beam[] = [
    { prefix: [], key: '', node: lexicon ? lexicon.root : null, lm: 0, pb: 0, pnb: -Infinity },
  ];
  for (let t = 0; t < lp.frames; t++) {
    const candidates: number[] = [];
    for (let c = 1; c < lp.classes; c++) if (at(lp, t, c) >= logPrune) candidates.push(c);
    const next = new Map<string, Beam>();
    const get = (prefix: number[], key: string, node: TrieNode | null, lm: number): Beam => {
      let beam = next.get(key);
      if (!beam) {
        beam = { prefix, key, node, lm, pb: -Infinity, pnb: -Infinity };
        next.set(key, beam);
      }
      return beam;
    };
    const blank = at(lp, t, BLANK);
    for (const beam of beams) {
      const tot = total(beam);
      const same = get(beam.prefix, beam.key, beam.node, beam.lm);
      same.pb = logAdd(same.pb, tot + blank);
      const last = beam.prefix.length > 0 ? beam.prefix[beam.prefix.length - 1]! : -1;
      for (const c of candidates) {
        const p = at(lp, t, c);
        let extend: number;
        if (c === last) {
          same.pnb = logAdd(same.pnb, beam.pnb + p);
          extend = beam.pb + p;
        } else {
          extend = tot + p;
        }
        if (extend === -Infinity) continue;
        const state = advance(lexicon, beam.node, beam.lm, CHARSET[c]!, lmWeight);
        if (state === null) continue;
        const prefix = [...beam.prefix, c];
        const target = get(prefix, `${beam.key},${c}`, state[0], state[1]);
        target.pnb = logAdd(target.pnb, extend);
      }
    }
    beams = [...next.values()]
      .sort((a, b) => {
        const diff = total(b) + b.lm - (total(a) + a.lm);
        return diff !== 0 ? diff : comparePrefix(a.prefix, b.prefix);
      })
      .slice(0, beamWidth);
  }

  let best: Beam | null = null;
  let bestScore = -Infinity;
  for (const beam of beams) {
    const final = finish(lexicon, beam, lmWeight);
    if (final !== null && (best === null || final > bestScore)) {
      best = beam;
      bestScore = final;
    }
  }
  if (lexicon !== null) {
    const greedy = greedyDecode(lp);
    let greedyPath = 0;
    for (let t = 0; t < lp.frames; t++) {
      let m = -Infinity;
      for (let c = 0; c < lp.classes; c++) m = Math.max(m, at(lp, t, c));
      greedyPath += m;
    }
    const allowed = maxDrop * Math.max(1, countWords(greedy.text));
    if (best === null || total(best) < greedyPath - allowed) return { ...greedy, fallback: true };
  }
  if (best === null) return { text: '', confidences: [], fallback: false };
  return {
    text: labelsToText(best.prefix),
    confidences: alignConfidences(lp, best.prefix),
    fallback: false,
  };
}
