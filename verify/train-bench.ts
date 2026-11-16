// In-browser training benchmark: runs trainer.ts in a worker on each TF.js
// backend and model size, with synthetic data, and reports time per batch.
// Open /verify/train-bench.html on the dev server; options in the query string:
//   ?backends=wasm,webgl,cpu  &sizes=xxs,xs,s,m  &batches=12  &dimension=9
//   &delay=35    (seconds to wait before starting — see main())
//   &profile=1   (per-kernel times, on window.__bench_kernels and under the table)
//   &threads=1,4,default   (WASM thread counts to compare; "default" leaves it to the backend)
// Rows land on window.__bench as they finish; window.__bench_done flips at the end.

import { MODEL_SIZES } from "../src/lib/impsy/mdrnnWeights";
import type { BenchRequest, BenchResult } from "./trainBenchWorker";

interface Row {
  /** Seconds since page load when the run started. */
  atS: number;
  backend: string;
  threads: number | null;
  size: string;
  setupMs: number;
  firstBatchMs: number;
  medianMs: number;
  minMs: number;
  maxMs: number;
  yieldMs: number;
  /** Seconds for one epoch over 100 batches (6400 windows), at the median. */
  secPer100Batches: number;
  finalLoss: number;
  error?: string;
}

declare global {
  interface Window {
    __bench: Row[];
    __bench_done: boolean;
    __bench_env: Record<string, unknown>;
  }
}

const params = new URLSearchParams(location.search);
const list = (key: string, fallback: string) => (params.get(key) ?? fallback).split(",");
const backends = list("backends", "wasm,webgl,cpu") as BenchRequest["backend"][];
const sizes = list("sizes", Object.keys(MODEL_SIZES).join(",")) as BenchRequest["size"][];
const threads = params
  .get("threads")
  ?.split(",")
  .map((t) => (t === "default" ? undefined : Number(t))) ?? [undefined];
const batches = Number(params.get("batches") ?? 12);
const dimension = Number(params.get("dimension") ?? 9);
const profile = params.has("profile");
const kernelReports: string[] = [];

const out = document.getElementById("out")!;
const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};

function runOne(request: BenchRequest): Promise<BenchResult> {
  return new Promise((resolve, reject) => {
    // A fresh worker per run, so no backend state or warm caches carry over.
    const worker = new Worker(new URL("./trainBenchWorker.ts", import.meta.url), {
      type: "module",
    });
    worker.onerror = (e) => {
      worker.terminate();
      reject(new Error(e.message || "worker failed to load"));
    };
    worker.onmessage = (event) => {
      worker.terminate();
      if (event.data.ok) resolve(event.data.result);
      else reject(new Error(event.data.error));
    };
    worker.postMessage(request);
  });
}

function render() {
  const cols: (keyof Row)[] = [
    "atS",
    "backend",
    "threads",
    "size",
    "setupMs",
    "firstBatchMs",
    "medianMs",
    "minMs",
    "maxMs",
    "yieldMs",
    "secPer100Batches",
    "finalLoss",
    "error",
  ];
  const cell = (v: unknown) =>
    typeof v === "number" ? v.toFixed(v < 10 ? 2 : 1) : v == null ? "-" : String(v);
  const lines = [cols, ...window.__bench.map((r) => cols.map((c) => cell(r[c])))];
  const widths = cols.map((_, i) => Math.max(...lines.map((l) => l[i].length)));
  out.textContent =
    JSON.stringify(window.__bench_env) +
    "\n\n" +
    lines.map((l) => l.map((v, i) => v.padEnd(widths[i])).join("  ")).join("\n") +
    kernelReports.join("") +
    (window.__bench_done ? "\n\ndone" : "\n\nrunning…");
}

async function main() {
  // Driven through Claude in Chrome, WASM runs several times slower for about
  // 30 s after a navigation (seemingly while the extension's debugger is
  // attached; Playwright doesn't show it), so let that pass first.
  const delay = Number(params.get("delay") ?? 0);
  if (delay > 0) await new Promise((resolve) => setTimeout(resolve, delay * 1000));
  window.__bench = [];
  window.__bench_done = false;
  window.__bench_env = {
    crossOriginIsolated,
    cores: navigator.hardwareConcurrency,
    dimension,
    batches,
    batchSize: 64,
    userAgent: navigator.userAgent,
  };
  const wasmPath = new URL("/tfjs-wasm/", location.href).href;
  for (const backend of backends) {
    for (const t of backend === "wasm" ? threads : [undefined]) {
      for (const size of sizes) {
        const request: BenchRequest = {
          backend,
          size,
          dimension,
          batches,
          batchSize: 64,
          threads: t,
          wasmPath,
          profile,
        };
        const row: Row = {
          atS: performance.now() / 1000,
          backend,
          threads: t ?? null,
          size,
          setupMs: NaN,
          firstBatchMs: NaN,
          medianMs: NaN,
          minMs: NaN,
          maxMs: NaN,
          yieldMs: NaN,
          secPer100Batches: NaN,
          finalLoss: NaN,
        };
        try {
          const r = await runOne(request);
          const m = median(r.batchMs);
          Object.assign(row, {
            threads: r.threads,
            setupMs: r.setupMs,
            firstBatchMs: r.firstBatchMs,
            medianMs: m,
            minMs: Math.min(...r.batchMs),
            maxMs: Math.max(...r.batchMs),
            yieldMs: r.yieldMs,
            secPer100Batches: m / 10,
            finalLoss: r.finalLoss,
          });
          if (r.kernels) {
            const total = r.kernels.reduce((a, k) => a + k.ms, 0);
            kernelReports.push(
              `\n\n${backend} ${size}: ${total.toFixed(0)} ms in kernels over ${batches} batches\n` +
                r.kernels
                  .slice(0, 14)
                  .map(
                    (k) =>
                      `  ${k.name.padEnd(22)} ×${String(k.count).padEnd(6)} ${k.ms.toFixed(1)} ms`,
                  )
                  .join("\n"),
            );
          }
        } catch (err) {
          row.error = err instanceof Error ? err.message : String(err);
        }
        window.__bench.push(row);
        render();
      }
    }
  }
  window.__bench_done = true;
  render();
}

main();
