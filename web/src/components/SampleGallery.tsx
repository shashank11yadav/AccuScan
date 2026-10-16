import { useEffect, useState } from 'react';
import { rgbaToGray, type GrayImage } from '../ml/preprocess';
import { BASE, SAMPLES_URL } from '../lib/links';

interface Sample {
  file: string;
  text: string;
  font: string;
}

interface Props {
  disabled: boolean;
  onImage: (image: GrayImage, sample: Sample) => void;
}

const isSample = (v: unknown): v is Sample =>
  typeof v === 'object' &&
  v !== null &&
  typeof (v as Sample).file === 'string' &&
  /^sample-\d+\.png$/.test((v as Sample).file) &&
  typeof (v as Sample).text === 'string' &&
  typeof (v as Sample).font === 'string';

async function toGray(url: string): Promise<GrayImage> {
  const img = new Image();
  img.src = url;
  await img.decode();
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('canvas unavailable');
  ctx.drawImage(img, 0, 0);
  return rgbaToGray(
    ctx.getImageData(0, 0, canvas.width, canvas.height).data,
    canvas.width,
    canvas.height,
  );
}

export function SampleGallery({ disabled, onImage }: Props) {
  const [samples, setSamples] = useState<Sample[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetch(SAMPLES_URL, { signal: controller.signal })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((data: unknown) => {
        const list = (data as { samples?: unknown }).samples;
        setSamples(Array.isArray(list) ? list.filter(isSample) : []);
      })
      .catch(() => {
        if (!controller.signal.aborted) setFailed(true);
      });
    return () => {
      controller.abort();
    };
  }, []);

  if (failed) return <p className="text-sm text-danger">The samples could not be loaded.</p>;
  if (!samples) return <p className="text-sm text-ink-muted">Loading samples…</p>;

  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm text-ink-muted">
        Drawn-style lines in fonts the model never saw in training. Pick one to read it.
      </p>
      <ul className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {samples.map((s) => {
          const url = `${BASE}samples/${s.file}`;
          const active = selected === s.file;
          return (
            <li key={s.file}>
              <button
                type="button"
                disabled={disabled}
                aria-pressed={active}
                onClick={() => {
                  setSelected(s.file);
                  void toGray(url).then((img) => {
                    onImage(img, s);
                  });
                }}
                className={`flex h-24 w-full items-center justify-center overflow-hidden rounded-xl border bg-white p-2 transition-colors disabled:opacity-50 ${
                  active
                    ? 'border-accent ring-2 ring-accent/40'
                    : 'border-line hover:border-accent/60'
                }`}
              >
                <img
                  src={url}
                  alt={`Handwritten sample: ${s.text}`}
                  className="max-h-full max-w-full"
                />
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export type { Sample };
