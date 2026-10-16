import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  CHARSET,
  Lexicon,
  NUM_CLASSES,
  alignConfidences,
  beamSearch,
  encode,
  greedyDecode,
  type LogProbs,
} from './decode';

interface Expected {
  text: string;
  confidences: number[];
  fallback?: boolean;
}
interface DecodeCase {
  name: string;
  logProbs: number[][];
  greedy: Expected;
  beam: Expected;
  dictionary: Expected;
  alignment: { labels: string; confidences: number[] };
}
const fixture = JSON.parse(
  readFileSync(
    fileURLToPath(new URL('../../../shared/fixtures/decode.json', import.meta.url)),
    'utf8',
  ),
) as {
  charset: string[];
  lexicon: Record<string, number>;
  beamWidth: number;
  lmWeight: number;
  cases: DecodeCase[];
};

function toLogProbs(rows: number[][]): LogProbs {
  const classes = rows[0]?.length ?? NUM_CLASSES;
  return { data: Float64Array.from(rows.flat()), frames: rows.length, classes };
}

/** Per-frame {symbol: prob}, the rest spread evenly; '_' is blank (as in the Python tests). */
function frames(...rows: Record<string, number>[]): LogProbs {
  const out: number[][] = rows.map((row) => {
    const keys = Object.keys(row);
    const rest = Math.max(1e-6, 1 - keys.reduce((s, k) => s + (row[k] ?? 0), 0));
    const r = new Array<number>(NUM_CLASSES).fill(rest / (NUM_CLASSES - keys.length));
    for (const k of keys) r[k === '_' ? 0 : CHARSET.indexOf(k)] = row[k] ?? 0;
    const sum = r.reduce((a, b) => a + b, 0);
    return r.map((v) => Math.log(v / sum));
  });
  return toLogProbs(out);
}
const path = (text: string, p = 0.9) => frames(...Array.from(text, (ch) => ({ [ch]: p })));
const close = (a: number[], b: number[]) => {
  expect(a.length).toBe(b.length);
  a.forEach((v, i) => {
    expect(Math.abs(v - (b[i] ?? NaN))).toBeLessThan(1e-4);
  });
};

describe('decode parity with the Python pipeline', () => {
  it('uses the same charset', () => {
    expect(fixture.charset).toEqual(CHARSET);
  });
  const lexicon = Lexicon.fromCounts(Object.entries(fixture.lexicon));
  it.each(fixture.cases.map((c) => [c.name, c] as const))('%s', (_name, c) => {
    const lp = toLogProbs(c.logProbs);
    const greedy = greedyDecode(lp);
    expect(greedy.text).toBe(c.greedy.text);
    close(greedy.confidences, c.greedy.confidences);
    const beam = beamSearch(lp, null, { beamWidth: fixture.beamWidth });
    expect(beam.text).toBe(c.beam.text);
    close(beam.confidences, c.beam.confidences);
    const dictionary = beamSearch(lp, lexicon, {
      beamWidth: fixture.beamWidth,
      lmWeight: fixture.lmWeight,
    });
    expect(dictionary.text).toBe(c.dictionary.text);
    expect(dictionary.fallback).toBe(c.dictionary.fallback);
    close(dictionary.confidences, c.dictionary.confidences);
    close(alignConfidences(lp, encode(c.alignment.labels)), c.alignment.confidences);
  });
});

describe('greedy decoding', () => {
  it('collapses repeats and drops blanks', () => {
    expect(greedyDecode(path('aa_ab_b')).text).toBe('aabb');
  });
  it('returns nothing for all blanks', () => {
    expect(greedyDecode(path('___'))).toEqual({ text: '', confidences: [], fallback: false });
  });
  it('takes the max probability over a run', () => {
    const out = greedyDecode(frames({ h: 0.6 }, { h: 0.8 }, { _: 0.9 }, { i: 0.7 }));
    expect(out.text).toBe('hi');
    close(out.confidences, [0.8, 0.7]);
  });
});

describe('prefix beam search', () => {
  it('sums over alignments where greedy does not', () => {
    const lp = frames({ a: 0.4, _: 0.6 }, { a: 0.4, _: 0.6 });
    expect(greedyDecode(lp).text).toBe('');
    expect(beamSearch(lp, null, { beamWidth: 8 }).text).toBe('a');
  });
  it('corrects a letter to a dictionary word', () => {
    const lex = Lexicon.fromCounts([
      ['cat', 10],
      ['cot', 5],
    ]);
    const lp = frames({ c: 0.9 }, { s: 0.5, a: 0.4 }, { t: 0.9 });
    expect(greedyDecode(lp).text).toBe('cst');
    expect(beamSearch(lp, lex).text).toBe('cat');
  });
  it('keeps the writer’s capitals and allows digits and punctuation', () => {
    const lex = Lexicon.fromCounts([
      ['cat', 1],
      ['dog', 1],
    ]);
    expect(beamSearch(path('Cat_ 42, dog!'), lex).text).toBe('Cat 42, dog!');
  });
  it('falls back to the greedy reading for an unknown word', () => {
    const lex = Lexicon.fromCounts([['house', 1]]);
    const out = beamSearch(path('hou'), lex);
    expect(out.fallback).toBe(true);
    expect(out.text).toBe('hou');
  });
});

describe('lexicon', () => {
  it('parses the bundled word<TAB>count format', () => {
    const lex = Lexicon.parse('the\t3\ncat\t1\n\nbad line\n');
    expect(lex.size).toBe(2);
    expect(lex.logprob('THE')).toBeCloseTo(Math.log(3 / 4));
    expect(lex.logprob('dog')).toBeNull();
  });
});
