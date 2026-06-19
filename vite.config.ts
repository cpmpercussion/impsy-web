import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// LiteRT.js ships a WASM runtime. We serve its glue from the installed package
// at /wasm/ in dev so loadLiteRt() can fetch it without a CDN (the CSP/offline
// path). `optimizeDeps.exclude` keeps Vite from trying to pre-bundle the wasm.
export default defineConfig({
  plugins: [svelte()],
  optimizeDeps: {
    exclude: ["@litertjs/core"],
  },
  server: {
    // SharedArrayBuffer (LiteRT multi-threaded WASM) needs cross-origin isolation.
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
