/// <reference types="node" />
import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";

// Build identity shown in the app footer (issue #17), so a bug report from the
// live site can be traced to the exact commit. Falls back gracefully outside a
// git checkout.
const pkg = JSON.parse(readFileSync(new URL("./package.json", import.meta.url), "utf8"));
function gitHash(): string {
  try {
    return execSync("git rev-parse --short HEAD", { stdio: ["ignore", "pipe", "ignore"] })
      .toString()
      .trim();
  } catch {
    return process.env.GITHUB_SHA?.slice(0, 7) ?? "unknown";
  }
}

// LiteRT.js ships a WASM runtime. We serve its glue from the installed package
// at /wasm/ in dev so loadLiteRt() can fetch it without a CDN (the CSP/offline
// path). `optimizeDeps.exclude` keeps Vite from trying to pre-bundle the wasm.
export default defineConfig(({ command, isPreview }) => ({
  // GitHub Pages serves this project site under /impsy-web/ (the repo name),
  // even with the user's custom domain — so the production base is the subpath.
  // Dev stays at root. Runtime asset paths (WASM dir, demo model) use
  // import.meta.env.BASE_URL so they resolve under the subpath too.
  // `vite preview` runs with command "serve", so it needs `isPreview` to serve
  // dist/ under the same subpath it was built for.
  base: command === "build" || isPreview ? "/impsy-web/" : "/",
  plugins: [svelte()],
  define: {
    __APP_VERSION__: JSON.stringify(pkg.version),
    __GIT_HASH__: JSON.stringify(gitHash()),
    __BUILD_DATE__: JSON.stringify(new Date().toISOString().slice(0, 10)),
  },
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
