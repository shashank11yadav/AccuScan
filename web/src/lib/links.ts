export const REPO_URL = 'https://github.com/shashank11yadav/AccuScan';
export const PORTFOLIO_URL = 'https://www.shashankyadav.dev/work/accuscan';
export const GNHK_URL = 'https://github.com/GoodNotes/GNHK-dataset';
export const BASE = import.meta.env.BASE_URL;
// Same-origin asset paths, fixed at build time. The worker imports these
// itself rather than taking URLs in messages.
export const MODEL_URL = `${BASE}models/model.onnx`;
export const META_URL = `${BASE}models/model-meta.json`;
export const LEXICON_URL = `${BASE}lexicon/words.txt.gz`;
export const SAMPLES_URL = `${BASE}samples/samples.json`;
