# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](https://github.com/shashank11yadav/AccuScan/security/advisories/new)
rather than a public issue. I will acknowledge a report within a week.

## Scope

The deployed site is a static bundle (HTML, JS, CSS, WebAssembly, an ONNX model
and a word list) on GitHub Pages. It has no server, no accounts, no cookies, no
API keys and no third-party scripts. Drawings and photos are processed in the
browser and never uploaded, so the realistic risks are in the build, the supply
chain and untrusted image input:

- Dependencies are pinned (`web/package-lock.json`, `pipeline/uv.lock`) and
  updated by Dependabot. CI runs `npm audit --omit=dev` and `pip-audit`.
- GitHub Actions are pinned to commit SHAs and run with `contents: read` unless
  a job needs more (Pages deploy, CodeQL upload).
- The pipeline downloads fonts and text from pinned commits or URLs and checks
  a SHA-256 for every file before using it. Archives are read in place and only
  expected member names are opened.
- The frontend checks an upload's size and magic bytes before decoding it,
  downscales large photos before any pixel work, and ships a
  Content-Security-Policy that allows only same-origin scripts, styles, workers
  and fetches (plus `'wasm-unsafe-eval'`, which WebAssembly needs).
