import type { ModelMeta } from '../ml/meta';
import { GNHK_URL, REPO_URL } from '../lib/links';

const STEPS = [
  {
    title: 'Normalise',
    body: 'Contrast is stretched so paper is white and ink is black, then the image is cropped to the ink and scaled to 64 px high.',
  },
  {
    title: 'See',
    body: 'A small convolutional network turns every 4-pixel column into a feature vector.',
  },
  {
    title: 'Read',
    body: 'A two-layer bidirectional LSTM scores each column against 95 printable characters plus a blank.',
  },
  {
    title: 'Decode',
    body: 'CTC decoding collapses the column scores into text. Dictionary mode runs a beam search that keeps every word in a 50,000-word list.',
  },
] as const;

const pct = (v: number | undefined) => (v === undefined ? '–' : `${(v * 100).toFixed(1)}%`);

function Metrics({ meta }: { meta: ModelMeta }) {
  if (meta.metrics === null) {
    return (
      <p className="text-ink-muted">
        No accuracy figures yet: this build runs an untrained preview model. Once the model is
        trained, its held-out character and word error rates appear here, measured by the
        pipeline&apos;s evaluation script.
      </p>
    );
  }
  const rows = [
    ['GNHK test, words', meta.metrics.sets.gnhk_test_words],
    ['GNHK test, lines', meta.metrics.sets.gnhk_test_lines],
    ['Canvas style, held-out fonts', meta.metrics.sets.canvas_heldout_fonts],
  ] as const;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm tabular-nums">
        <caption className="mb-2 text-left font-semibold">
          Held-out error rates (lower is better)
        </caption>
        <thead>
          <tr className="border-b border-line text-left text-ink-muted">
            <th scope="col" className="py-2 pr-3 font-medium">
              Test set
            </th>
            <th scope="col" className="py-2 pr-3 text-right font-medium">
              CER
            </th>
            <th scope="col" className="py-2 pr-3 text-right font-medium">
              WER
            </th>
            <th scope="col" className="py-2 text-right font-medium">
              CER, dictionary
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, m]) =>
            m ? (
              <tr key={label} className="border-b border-line/70">
                <th scope="row" className="py-2 pr-3 text-left font-normal">
                  {label}
                </th>
                <td className="py-2 pr-3 text-right">{pct(m.greedy.cer)}</td>
                <td className="py-2 pr-3 text-right">{pct(m.greedy.wer)}</td>
                <td className="py-2 text-right">{pct(m.dictionary?.cer)}</td>
              </tr>
            ) : null,
          )}
        </tbody>
      </table>
      <p className="mt-3 text-xs leading-relaxed text-ink-muted">
        GNHK is photographed handwriting from many writers. The canvas set is synthetic: lines drawn
        in 10 handwriting fonts kept out of training, not people writing with a mouse.
      </p>
    </div>
  );
}

export function AboutSection({ meta }: { meta: ModelMeta | null }) {
  return (
    <section id="how-it-works" aria-labelledby="how-heading" className="scroll-mt-6">
      <h2 id="how-heading" className="text-2xl font-semibold tracking-tight">
        How it works
      </h2>
      <ol className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {STEPS.map((step, i) => (
          <li key={step.title} className="rounded-xl border border-line bg-surface p-4">
            <span className="font-mono text-xs text-accent">0{i + 1}</span>
            <h3 className="mt-1 font-semibold">{step.title}</h3>
            <p className="mt-1 text-sm leading-relaxed text-ink-muted">{step.body}</p>
          </li>
        ))}
      </ol>

      <div className="mt-6 grid items-start gap-4 lg:grid-cols-[3fr_2fr]">
        <div className="rounded-xl border border-line bg-surface p-5">
          <h3 className="mb-3 font-semibold">Accuracy</h3>
          {meta ? (
            <Metrics meta={meta} />
          ) : (
            <p className="text-ink-muted">Loading model details…</p>
          )}
        </div>
        <div className="rounded-xl border border-line bg-surface p-5 text-sm leading-relaxed">
          <h3 className="mb-3 text-base font-semibold">Model and data</h3>
          <ul className="space-y-2 text-ink-muted">
            <li>
              CRNN, {meta ? `${(meta.parameters / 1e6).toFixed(1)}M parameters, ` : ''}
              {meta ? `${(meta.bytes / 1e6).toFixed(1)} MB` : 'about 5 MB'} after int8 quantisation.
            </li>
            <li>
              Trained on the{' '}
              <a className="text-accent underline underline-offset-2" href={GNHK_URL}>
                GNHK
              </a>{' '}
              dataset (GoodNotes, CC BY 4.0) and synthetic lines in 54 open-licence handwriting
              fonts; 10 more fonts are held out for testing.
            </li>
            <li>One line or word at a time; it does not find text on a page by itself.</li>
            <li>
              Your drawing and photos never leave this device: the model runs in your browser.
            </li>
          </ul>
          <a
            className="mt-4 inline-block font-medium text-accent underline underline-offset-2"
            href={REPO_URL}
          >
            Training pipeline and evaluation on GitHub
          </a>
        </div>
      </div>
    </section>
  );
}
