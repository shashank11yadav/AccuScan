/** Validation of model-meta.json, the pipeline's description of the shipped model. */

export type DecodeMode = 'free' | 'dictionary';

export interface SetMetrics {
  greedy: { cer: number; wer: number; samples: number };
  dictionary?: { cer: number; wer: number; samples: number };
}

export interface ModelMeta {
  version: string;
  stub: boolean;
  featureVersion: number;
  classes: number;
  blank: number;
  input: { name: string; height: number; maxWidth: number; widthMultiple: number };
  output: { name: string; downsample: number };
  parameters: number;
  bytes: number;
  defaultMode: DecodeMode;
  decoder: { beamWidth: number; lmWeight: number; prune: number; maxDropPerWord: number };
  /** null until a trained model has been evaluated: the UI then shows the preview notice. */
  metrics: {
    sets: Partial<
      Record<'gnhk_test_words' | 'gnhk_test_lines' | 'canvas_heldout_fonts', SetMetrics>
    >;
    gate: { canvas_cer: number; threshold: number; defaultMode: DecodeMode };
    latency_cpu?: Record<string, number>;
  } | null;
}

export class MetaError extends Error {
  constructor(message: string) {
    super(`model-meta.json: ${message}`);
    this.name = 'MetaError';
  }
}

const isObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

function num(obj: Record<string, unknown>, key: string): number {
  const v = obj[key];
  if (typeof v !== 'number' || !Number.isFinite(v)) throw new MetaError(`${key} must be a number`);
  return v;
}

function str(obj: Record<string, unknown>, key: string): string {
  const v = obj[key];
  if (typeof v !== 'string') throw new MetaError(`${key} must be a string`);
  return v;
}

/** Checks the fields the app relies on and that the contract matches this build. */
export function parseMeta(
  raw: unknown,
  expected: { featureVersion: number; classes: number },
): ModelMeta {
  if (!isObject(raw)) throw new MetaError('expected an object');
  const input = raw['input'];
  const output = raw['output'];
  const decoder = raw['decoder'];
  if (!isObject(input) || !isObject(output) || !isObject(decoder)) {
    throw new MetaError('input, output and decoder are required');
  }
  const featureVersion = num(raw, 'featureVersion');
  if (featureVersion !== expected.featureVersion) {
    throw new MetaError(
      `preprocessing version ${featureVersion} does not match this app (${expected.featureVersion})`,
    );
  }
  const classes = num(raw, 'classes');
  if (classes !== expected.classes) throw new MetaError(`expected ${expected.classes} classes`);
  const defaultMode = raw['defaultMode'];
  if (defaultMode !== 'free' && defaultMode !== 'dictionary') {
    throw new MetaError('defaultMode must be "free" or "dictionary"');
  }
  const metrics = raw['metrics'];
  if (
    metrics !== null &&
    !(isObject(metrics) && isObject(metrics['sets']) && isObject(metrics['gate']))
  ) {
    throw new MetaError('metrics must be null or contain sets and gate');
  }
  return {
    version: str(raw, 'version'),
    stub: raw['stub'] === true,
    featureVersion,
    classes,
    blank: num(raw, 'blank'),
    input: {
      name: str(input, 'name'),
      height: num(input, 'height'),
      maxWidth: num(input, 'maxWidth'),
      widthMultiple: num(input, 'widthMultiple'),
    },
    output: { name: str(output, 'name'), downsample: num(output, 'downsample') },
    parameters: num(raw, 'parameters'),
    bytes: num(raw, 'bytes'),
    defaultMode,
    decoder: {
      beamWidth: num(decoder, 'beamWidth'),
      lmWeight: num(decoder, 'lmWeight'),
      prune: num(decoder, 'prune'),
      maxDropPerWord: num(decoder, 'maxDropPerWord'),
    },
    metrics: metrics as ModelMeta['metrics'],
  };
}

/** The preview notice is shown whenever there are no measured metrics. */
export function isUntrained(meta: ModelMeta): boolean {
  return meta.metrics === null;
}
