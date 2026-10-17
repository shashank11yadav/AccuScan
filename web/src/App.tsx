import { useCallback, useEffect, useRef, useState } from 'react';
import { AboutSection } from './components/AboutSection';
import {
  BoltIcon,
  ChipIcon,
  GitHubIcon,
  GridIcon,
  ImageIcon,
  PenIcon,
  ShieldIcon,
} from './components/Icons';
import { ResultPanel } from './components/ResultPanel';
import { SampleGallery } from './components/SampleGallery';
import { UploadPanel } from './components/UploadPanel';
import { WritePad } from './components/WritePad';
import { REPO_URL } from './lib/links';
import { useModel, type ReadState } from './lib/useModel';
import { isUntrained, type DecodeMode } from './ml/meta';
import type { GrayImage } from './ml/preprocess';

type Tab = 'write' | 'upload' | 'samples';
const TABS: { id: Tab; label: string; Icon: typeof PenIcon }[] = [
  { id: 'write', label: 'Write', Icon: PenIcon },
  { id: 'upload', label: 'Photo', Icon: ImageIcon },
  { id: 'samples', label: 'Samples', Icon: GridIcon },
];
const EMPTY_MESSAGES: Record<Tab, string> = {
  write: 'The writing area is empty. Draw a word first.',
  upload: 'No writing found in that box. Try a box around darker text.',
  samples: 'That sample has no ink.',
};

export function App() {
  const { model, retry, read } = useModel();
  const [tab, setTab] = useState<Tab>('write');
  const [mode, setMode] = useState<DecodeMode>('free');
  const [state, setState] = useState<ReadState>({ status: 'idle' });
  const [expected, setExpected] = useState<string | null>(null);
  const last = useRef<{ image: GrayImage; tab: Tab } | null>(null);
  const request = useRef(0);
  const modeChosen = useRef(false);
  const mainRef = useRef<HTMLElement>(null);
  const ready = model.status === 'ready';

  // Open in the mode the evaluation chose (spec: canvas gate), unless the visitor picked one.
  useEffect(() => {
    if (model.status === 'ready' && !modeChosen.current) setMode(model.meta.defaultMode);
  }, [model]);

  const run = useCallback(
    (image: GrayImage, source: Tab, decodeMode: DecodeMode) => {
      last.current = { image, tab: source };
      const id = ++request.current;
      setState({ status: 'reading' });
      void read(image, decodeMode, EMPTY_MESSAGES[source]).then((next) => {
        if (id === request.current) setState(next);
      });
    },
    [read],
  );

  const onMode = (next: DecodeMode) => {
    modeChosen.current = true;
    setMode(next);
    if (last.current && state.status === 'done') run(last.current.image, last.current.tab, next);
  };

  const reset = useCallback(() => {
    request.current++;
    last.current = null;
    setExpected(null);
    setState({ status: 'idle' });
  }, []);

  const switchTab = (next: Tab) => {
    setTab(next);
    reset();
  };

  const onTabKey = (event: React.KeyboardEvent, index: number) => {
    if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
    event.preventDefault();
    const step = event.key === 'ArrowRight' ? 1 : TABS.length - 1;
    const next = TABS[(index + step) % TABS.length];
    if (!next) return;
    switchTab(next.id);
    document.getElementById(`tab-${next.id}`)?.focus();
  };

  const meta = ready ? model.meta : null;

  // On one-column layouts the result sits below the eight samples, out of view.
  const revealResult = () => {
    if (!window.matchMedia('(max-width: 1023px)').matches) return;
    const smooth = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    document
      .getElementById('result-heading')
      ?.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto', block: 'start' });
  };

  return (
    <>
      <a
        href="#main"
        className="skip-link"
        onClick={(event) => {
          event.preventDefault();
          mainRef.current?.focus();
        }}
      >
        Skip to content
      </a>
      <header className="border-b border-line/80 bg-paper/90">
        <div className="mx-auto flex max-w-6xl items-center gap-3 px-4 py-3 sm:px-6">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-accent text-accent-ink">
            <PenIcon width={16} height={16} />
          </span>
          <span className="text-lg font-semibold tracking-tight">AccuScan</span>
          <nav aria-label="Site" className="ml-auto flex items-center gap-1 text-sm">
            <a
              href="#how-it-works"
              className="hidden rounded-md px-3 py-1.5 text-ink-muted hover:bg-surface-muted hover:text-ink sm:inline"
            >
              How it works
            </a>
            <a
              href={REPO_URL}
              className="inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-ink-muted hover:bg-surface-muted hover:text-ink"
            >
              <GitHubIcon /> <span>Source</span>
            </a>
          </nav>
        </div>
      </header>

      <main id="main" ref={mainRef} tabIndex={-1} className="outline-none">
        <div className="mx-auto max-w-6xl px-4 pt-10 pb-16 sm:px-6 sm:pt-14">
          <div className="max-w-3xl">
            <p className="text-sm font-medium text-accent">
              Handwriting recognition in the browser
            </p>
            <h1 className="mt-2 text-4xl font-semibold tracking-tight text-balance sm:text-5xl">
              Write a word. Watch it get read.
            </h1>
            <p className="mt-4 text-lg leading-relaxed text-ink-muted text-pretty">
              Draw with a mouse, finger or pen, or snap a photo of a note. A 4-million-parameter
              network reads one line at a time, entirely on your device.
            </p>
            <ul className="mt-5 flex flex-wrap gap-2 text-sm">
              <li className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-3 py-1">
                <ShieldIcon width={15} height={15} className="text-accent" /> Nothing is uploaded
              </li>
              <li className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-3 py-1">
                <ChipIcon width={15} height={15} className="text-accent" /> Runs in a Web Worker
              </li>
              <li className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-3 py-1">
                <BoltIcon width={15} height={15} className="text-accent" /> Reads as you pause
              </li>
            </ul>
          </div>

          {meta && isUntrained(meta) && (
            <aside
              aria-label="Model status"
              data-testid="preview-notice"
              className="mt-8 rounded-xl border border-warn/30 bg-warn-soft px-4 py-3 text-sm leading-relaxed text-warn"
            >
              <strong className="font-semibold">Untrained preview model.</strong> This build ships
              random weights so every part of the demo can be tried; its readings are meaningless
              until the trained model is published.
            </aside>
          )}

          {model.status === 'error' && (
            <div
              role="alert"
              className="mt-8 flex flex-col gap-3 rounded-xl border border-danger/30 bg-danger-soft px-4 py-4 text-danger sm:flex-row sm:items-center"
            >
              <p className="text-sm">
                <strong className="font-semibold">The model could not be loaded.</strong>{' '}
                {model.message}
              </p>
              <button
                type="button"
                onClick={retry}
                className="inline-flex h-9 shrink-0 items-center rounded-lg bg-danger px-4 text-sm font-semibold text-white sm:ml-auto"
              >
                Retry
              </button>
            </div>
          )}

          <div className="mt-8 grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1fr)_22rem]">
            <section
              aria-label="Input"
              className="min-w-0 rounded-2xl border border-line bg-surface p-3 shadow-card sm:p-5"
            >
              <div
                role="tablist"
                aria-label="Input method"
                className="mb-4 flex rounded-xl bg-surface-muted p-1 sm:inline-flex"
              >
                {TABS.map((t, i) => (
                  <button
                    key={t.id}
                    id={`tab-${t.id}`}
                    type="button"
                    role="tab"
                    aria-selected={tab === t.id}
                    aria-controls={`panel-${t.id}`}
                    tabIndex={tab === t.id ? 0 : -1}
                    onClick={() => {
                      switchTab(t.id);
                    }}
                    onKeyDown={(event) => {
                      onTabKey(event, i);
                    }}
                    className={`inline-flex h-9 flex-1 items-center justify-center gap-1.5 rounded-lg px-2 text-sm font-medium transition-colors sm:flex-none sm:px-3.5 ${
                      tab === t.id
                        ? 'bg-surface text-ink shadow-sm'
                        : 'text-ink-muted hover:text-ink'
                    }`}
                  >
                    <t.Icon width={16} height={16} />
                    {t.label}
                  </button>
                ))}
              </div>
              <div id={`panel-${tab}`} role="tabpanel" aria-labelledby={`tab-${tab}`}>
                {tab === 'write' && (
                  <WritePad
                    disabled={!ready}
                    onImage={(image) => {
                      run(image, 'write', mode);
                    }}
                    onBlank={() => {
                      reset();
                    }}
                  />
                )}
                {tab === 'upload' && (
                  <UploadPanel
                    disabled={!ready}
                    onImage={(image) => {
                      run(image, 'upload', mode);
                    }}
                    onReset={reset}
                  />
                )}
                {tab === 'samples' && (
                  <SampleGallery
                    disabled={!ready}
                    onImage={(image, sample) => {
                      setExpected(sample.text);
                      run(image, 'samples', mode);
                      revealResult();
                    }}
                  />
                )}
              </div>
            </section>
            <ResultPanel
              state={state}
              mode={mode}
              onModeChange={onMode}
              backend={ready ? model.backend : null}
              expected={expected}
              modelReady={ready}
            />
          </div>
          {model.status === 'loading' && (
            <p className="mt-3 text-sm text-ink-muted" role="status">
              Loading the model (about 5 MB, cached after the first visit)…
            </p>
          )}

          <div className="mt-16">
            <AboutSection meta={meta} />
          </div>
        </div>
      </main>

      <footer className="border-t border-line/80">
        <div className="mx-auto flex max-w-6xl flex-col gap-1 px-4 py-6 text-sm text-ink-muted sm:flex-row sm:px-6">
          <p>AccuScan by Shashank Yadav. MIT licence.</p>
          <p className="sm:ml-auto">
            GNHK dataset © GoodNotes, CC BY 4.0. Fonts under the SIL Open Font License.
          </p>
        </div>
      </footer>
    </>
  );
}
