// Worker for verify/train-bench.ts: times trainer.ts on one backend / model
// size with synthetic data. Unlike ../src/lib/impsy/trainWorker.ts it takes the
// backend as given (no fallback), so each one can be measured.

import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-wasm";
import "@tensorflow/tfjs-backend-webgl";
import { getThreadsCount, setThreadsCount, setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import type { Performance } from "../src/lib/impsy/dataset";
import { MODEL_SIZES } from "../src/lib/impsy/mdrnnWeights";
import { trainMDRNN } from "../src/lib/impsy/trainer";

export interface BenchRequest {
  backend: "wasm" | "cpu" | "webgl";
  size: keyof typeof MODEL_SIZES;
  /** Model dimension, including time. */
  dimension: number;
  /** Batches to run; the first is reported separately as warm-up. */
  batches: number;
  batchSize: number;
  /** WASM thread count; omit for the backend's default. */
  threads?: number;
  wasmPath: string;
  /** Also report time per TF.js kernel (slows the run; ignore the batch times). */
  profile?: boolean;
}

export interface BenchResult {
  request: BenchRequest;
  backend: string;
  /** WASM threads in use; null on other backends. */
  threads: number | null;
  crossOriginIsolated: boolean;
  /** Backend start-up, ms. */
  setupMs: number;
  /** First batch (kernel / shader warm-up), ms. */
  firstBatchMs: number;
  /** Wall-clock per batch after the first, as the app sees it, ms. */
  batchMs: number[];
  /** Cost of the `setTimeout(0)` yield trainer.ts takes after each batch, ms. */
  yieldMs: number;
  finalLoss: number;
  /** With `profile`: kernels by total time, most expensive first. */
  kernels?: { name: string; count: number; ms: number }[];
}

/** Seeded uniform [0,1), so every backend trains the same weights on the same data. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** One performance giving exactly `examples` 50-step training windows. */
function syntheticPerformance(examples: number, dimension: number): Performance {
  const random = mulberry32(1);
  const rows: number[][] = [];
  const x = Array.from({ length: dimension - 1 }, random);
  for (let i = 0; i < examples + 50; i++) {
    for (let d = 0; d < x.length; d++) {
      x[d] = Math.min(1, Math.max(0, x[d] + (random() - 0.5) * 0.1));
    }
    rows.push([0.01 + random() * 0.3, ...x]);
  }
  return rows;
}

async function measureYield(): Promise<number> {
  const n = 20;
  const start = performance.now();
  for (let i = 0; i < n; i++) await new Promise<void>((resolve) => setTimeout(resolve, 0));
  return (performance.now() - start) / n;
}

self.onmessage = async (event: MessageEvent<BenchRequest>) => {
  const request = event.data;
  try {
    const setupStart = performance.now();
    if (request.backend === "wasm") {
      setWasmPaths(request.wasmPath);
      if (request.threads !== undefined) setThreadsCount(request.threads);
    }
    if (!(await tf.setBackend(request.backend))) {
      throw new Error(`TF.js backend "${request.backend}" failed to start`);
    }
    await tf.ready();
    const setupMs = performance.now() - setupStart;

    const performances = [
      syntheticPerformance(request.batches * request.batchSize, request.dimension),
    ];
    const times: number[] = [];
    let finalLoss = NaN;
    let last = performance.now();
    const train = () =>
      trainMDRNN(performances, {
        config: { dimension: request.dimension, ...MODEL_SIZES[request.size] },
        epochs: 1,
        batchSize: request.batchSize,
        validationSplit: 0,
        random: mulberry32(2),
        onBatchEnd: ({ loss }) => {
          const now = performance.now();
          times.push(now - last);
          last = now;
          finalLoss = loss;
        },
      });
    let kernels: BenchResult["kernels"];
    if (request.profile) {
      const info = await tf.profile(async () => {
        await train();
        return [];
      });
      const byName = new Map<string, { name: string; count: number; ms: number }>();
      for (const k of info.kernels) {
        const entry = byName.get(k.name) ?? { name: k.name, count: 0, ms: 0 };
        entry.count++;
        entry.ms += (await k.kernelTimeMs) as number;
        byName.set(k.name, entry);
      }
      kernels = [...byName.values()].sort((a, b) => b.ms - a.ms);
    } else {
      await train();
    }

    const result: BenchResult = {
      request,
      backend: tf.getBackend(),
      threads: request.backend === "wasm" ? getThreadsCount() : null,
      crossOriginIsolated: self.crossOriginIsolated,
      setupMs,
      firstBatchMs: times[0],
      batchMs: times.slice(1),
      yieldMs: await measureYield(),
      finalLoss,
      kernels,
    };
    self.postMessage({ ok: true, result });
  } catch (err) {
    self.postMessage({ ok: false, error: err instanceof Error ? err.message : String(err) });
  }
};
