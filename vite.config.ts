import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// LiteRT.js ships a WASM runtime. We serve its glue from the installed package
// at /wasm/ in dev so loadLiteRt() can fetch it without a CDN (the CSP/offline
// path). `optimizeDeps.exclude` keeps Vite from trying to pre-bundle the wasm.
export default defineConfig(({ command }) => ({
  // GitHub Pages serves this project site under /impsy-web/ (the repo name),
  // even with the user's custom domain — so the production base is the subpath.
  // Dev stays at root. Runtime asset paths (WASM dir, demo model) use
  // import.meta.env.BASE_URL so they resolve under the subpath too.
  base: command === "build" ? "/impsy-web/" : "/",
  plugins: [svelte()],
  optimizeDeps: {
    exclude: ["@litertjs/core"],
  },
  server: {
    // SharedArrayBuffer (LiteRT multi-threaded WASM) needs cross-origin isolation.
    // GitHub Pages can't send these headers, so production relies on
    // public/coi-serviceworker.js to inject them client-side instead.
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
}));
