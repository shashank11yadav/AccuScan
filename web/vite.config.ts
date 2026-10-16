import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig, type Plugin } from 'vitest/config';

// GitHub Pages serves the site from /<repo>/.
const BASE = '/AccuScan/';

// GitHub Pages cannot set response headers, so the policy ships as a meta tag,
// added only to production builds (the dev server injects inline scripts for
// hot reload). 'wasm-unsafe-eval' lets onnxruntime compile its WebAssembly;
// nothing else is relaxed. frame-ancestors is ignored in meta tags.
const CSP = [
  "default-src 'self'",
  "script-src 'self' 'wasm-unsafe-eval'",
  "style-src 'self'",
  "img-src 'self' blob:",
  "font-src 'self'",
  "connect-src 'self'",
  "worker-src 'self'",
  "manifest-src 'self'",
  "frame-src 'none'",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'none'",
].join('; ');

function contentSecurityPolicy(): Plugin {
  return {
    name: 'content-security-policy',
    apply: 'build',
    transformIndexHtml: {
      order: 'post',
      handler: () => [
        {
          tag: 'meta',
          attrs: { 'http-equiv': 'Content-Security-Policy', content: CSP },
          injectTo: 'head-prepend',
        },
      ],
    },
  };
}

export default defineConfig({
  base: BASE,
  plugins: [react(), tailwindcss(), contentSecurityPolicy()],
  build: {
    target: 'es2022',
    sourcemap: false,
    // Keep everything as external files; inlined assets would need a looser CSP.
    assetsInlineLimit: 0,
    chunkSizeWarningLimit: 1024,
  },
  worker: { format: 'es' },
  // onnxruntime-web locates its .wasm next to its own module; pre-bundling breaks that in dev.
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
  },
});
