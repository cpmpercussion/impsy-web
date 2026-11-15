// Web Worker that runs trainer.ts, so training (hundreds of milliseconds per
// batch) never blocks the main thread's MIDI and 10 ms interaction tick.
// Driven by ../training.svelte.ts.
//
// Uses the TF.js WASM backend (several times faster than the plain-JS one for
// these small matrices), falling back to the CPU backend if WASM can't start.

import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-wasm";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import type { Performance } from "./dataset";
import { trainMDRNN, type EpochInfo, type TrainOptions, type TrainResult } from "./trainer";

/** The options that can cross to the worker (no callbacks). */
export type WorkerTrainOptions = Pick<
  TrainOptions,
  | "config"
  | "epochs"
  | "batchSize"
  | "validationSplit"
  | "earlyStopping"
  | "patience"
  | "sequenceLength"
  | "learningRate"
  | "initialWeights"
>;

export type TrainWorkerRequest =
  | {
      type: "start";
      performances: Performance[];
      options: WorkerTrainOptions;
      /** Directory serving the TF.js `.wasm` files, with a trailing slash. */
      wasmPath: string;
    }
  | { type: "stop" };

export type TrainWorkerResponse =
  | { type: "backend"; name: string }
  | { type: "batch"; epoch: number; batch: number; batches: number; loss: number }
  | { type: "epoch"; info: EpochInfo }
  | { type: "done"; result: TrainResult }
  | { type: "error"; message: string };

const scope = self as unknown as DedicatedWorkerGlobalScope;
const post = (message: TrainWorkerResponse) => scope.postMessage(message);
let abort: AbortController | null = null;

async function chooseBackend(wasmPath: string): Promise<string> {
  try {
    setWasmPaths(wasmPath);
    if (await tf.setBackend("wasm")) return "wasm";
    console.warn("[IMPSY] TF.js WASM backend unavailable, training on the slower CPU backend");
  } catch (err) {
    console.warn(
      "[IMPSY] TF.js WASM backend unavailable, training on the slower CPU backend:",
      err,
    );
  }
  await tf.setBackend("cpu");
  return "cpu";
}

scope.onmessage = async (event: MessageEvent<TrainWorkerRequest>) => {
  const request = event.data;
  if (request.type === "stop") {
    abort?.abort();
    return;
  }
  try {
    post({ type: "backend", name: await chooseBackend(request.wasmPath) });
    abort = new AbortController();
    const result = await trainMDRNN(request.performances, {
      ...request.options,
      signal: abort.signal,
      onBatchEnd: (info) => post({ type: "batch", ...info }),
      onEpochEnd: (info) => post({ type: "epoch", info }),
    });
    post({ type: "done", result });
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
