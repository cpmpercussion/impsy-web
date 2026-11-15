// In-browser training (issue #1): turns IMPSY `.log` recordings into a trained
// `.tflite` without Python — dataset.ts → trainer.ts (in a Web Worker) →
// tfliteWriter.ts — and loads the result as the current model. The browser
// counterpart of `impsy dataset` + `impsy train`.

import { app } from "./appState.svelte";
import { SCALE_FACTOR } from "./impsy/constants";
import {
  SEQUENCE_LENGTH,
  dimensionFromLogName,
  logToRows,
  sliceExamples,
  type Performance,
} from "./impsy/dataset";
import { writeTflite } from "./impsy/tfliteWriter";
import { MODEL_SIZES } from "./impsy/mdrnnWeights";
import type { EpochInfo } from "./impsy/trainer";
import type { TrainWorkerRequest, TrainWorkerResponse } from "./impsy/trainWorker";

const TFJS_WASM_PATH = `${import.meta.env.BASE_URL}tfjs-wasm/`;

export type ModelSize = keyof typeof MODEL_SIZES;
export type TrainingStatus = "idle" | "running" | "done" | "error";

export interface LogSource {
  name: string;
  text: string;
}

/**
 * The model dimension a log was recorded for: from its file name, else from
 * its first `interface` row (values + 1 for dt).
 */
export function logDimension(source: LogSource): number | null {
  const named = dimensionFromLogName(source.name);
  if (named !== null) return named;
  for (const line of source.text.split(/\r?\n/)) {
    const row = line.split(",");
    if (row.length > 2 && row[1].trim() === "interface") return row.length - 1;
  }
  return null;
}

/** Model file name, as `mdrnn_model_name` in ../impsy/impsy/mdrnn.py builds it. */
export function modelFileName(
  dimension: number,
  size: { numLayers: number; hiddenUnits: number; numMixtures: number },
): string {
  return `musicMDRNN-dim${dimension}-layers${size.numLayers}-units${size.hiddenUnits}-mixtures${size.numMixtures}-scale${SCALE_FACTOR}.tflite`;
}

export class Training {
  status = $state<TrainingStatus>("idle");
  /** Outcome or error, for display. */
  message = $state<string | null>(null);
  /** What is being trained on. */
  summary = $state<string | null>(null);
  epochs = $state(0);
  progress = $state({ epoch: 0, batch: 0, batches: 0 });
  history = $state<EpochInfo[]>([]);
  backend = $state<string | null>(null);
  /** File name of the last trained model, once there is one to download. */
  trainedName = $state<string | null>(null);

  private trainedBytes: Uint8Array | null = null;
  private worker: Worker | null = null;

  /**
   * Train a model on log files and load it. Resolves when training has
   * finished, been stopped, or failed (see `status` / `message`).
   */
  async start(sources: LogSource[], options: { size: ModelSize; epochs: number }): Promise<void> {
    if (this.status === "running") return;
    this.message = null;
    this.summary = null;
    this.history = [];
    this.progress = { epoch: 0, batch: 0, batches: 0 };
    this.epochs = options.epochs;

    let dimension: number;
    let performances: Performance[];
    try {
      ({ dimension, performances } = this.prepare(sources));
    } catch (err) {
      this.status = "error";
      this.message = err instanceof Error ? err.message : String(err);
      return;
    }
    const size = MODEL_SIZES[options.size];
    const config = { dimension, ...size };
    const name = modelFileName(dimension, size);
    const events = performances.reduce((n, p) => n + p.length, 0);
    this.summary = `${events} events from ${performances.length} log${performances.length === 1 ? "" : "s"} · dim ${dimension}`;
    this.status = "running";

    const worker = new Worker(new URL("./impsy/trainWorker.ts", import.meta.url), {
      type: "module",
    });
    this.worker = worker;
    try {
      const result = await new Promise<Extract<TrainWorkerResponse, { type: "done" }>["result"]>(
        (resolve, reject) => {
          worker.onerror = (e) => reject(new Error(e.message || "Training worker failed"));
          worker.onmessage = (event: MessageEvent<TrainWorkerResponse>) => {
            const m = event.data;
            if (m.type === "backend") this.backend = m.name;
            else if (m.type === "batch") this.progress = m;
            else if (m.type === "epoch") this.history = [...this.history, m.info];
            else if (m.type === "done") resolve(m.result);
            else reject(new Error(m.message));
          };
          const request: TrainWorkerRequest = {
            type: "start",
            performances,
            options: { config, epochs: options.epochs },
            wasmPath: new URL(TFJS_WASM_PATH, location.href).href,
          };
          worker.postMessage(request);
        },
      );

      if (result.stopped === "nan") {
        throw new Error("Training diverged (the loss became NaN), so no model was produced.");
      }
      if (result.history.length === 0) {
        this.status = "idle";
        this.message = "Stopped before the first epoch finished; no model was produced.";
        return;
      }
      this.trainedBytes = writeTflite(result.weights);
      this.trainedName = name;
      const how = {
        epochs: "Trained",
        "early-stopping": "Stopped early (validation loss stopped improving)",
        aborted: "Stopped",
      }[result.stopped];
      this.message = `${how} after ${result.history.length} epoch${result.history.length === 1 ? "" : "s"}; loaded ${name}.`;
      this.status = "done";
      await app.loadModelFromBytes(this.trainedBytes, name);
    } catch (err) {
      this.status = "error";
      this.message = err instanceof Error ? err.message : String(err);
    } finally {
      worker.terminate();
      this.worker = null;
    }
  }

  /** Stop after the current batch; the model trained so far is kept and loaded. */
  stop(): void {
    const request: TrainWorkerRequest = { type: "stop" };
    this.worker?.postMessage(request);
  }

  /** Download the last trained model as a `.tflite` file. */
  download(): void {
    if (!this.trainedBytes || !this.trainedName) return;
    const blob = new Blob([this.trainedBytes.slice()], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = this.trainedName;
    a.click();
    URL.revokeObjectURL(url);
  }

  /** Parse the logs into performances, checking they agree on a dimension and are long enough. */
  private prepare(sources: LogSource[]): { dimension: number; performances: Performance[] } {
    if (sources.length === 0) throw new Error("Choose one or more .log files to train on.");
    const dimensions = sources.map(logDimension);
    const unknown = sources.filter((_, i) => dimensions[i] === null);
    if (unknown.length > 0) {
      throw new Error(`No interface events found in ${unknown.map((s) => s.name).join(", ")}.`);
    }
    const dimension = dimensions[0]!;
    const others = sources.filter((_, i) => dimensions[i] !== dimension);
    if (others.length > 0) {
      throw new Error(
        `Logs must all have the same dimension: ${sources[0].name} is ${dimension}d but ${others.map((s) => s.name).join(", ")} ${others.length === 1 ? "is" : "are"} not.`,
      );
    }
    const performances = sources
      .map((s) => logToRows(s.text.split(/\r?\n/), dimension))
      .filter((rows) => rows.length > 0);
    if (sliceExamples(performances).length === 0) {
      const longest = Math.max(0, ...performances.map((p) => p.length));
      throw new Error(
        `Not enough data: training needs a log with more than ${SEQUENCE_LENGTH + 2} interface events, and the longest here has ${longest + (longest > 0 ? 1 : 0)}.`,
      );
    }
    return { dimension, performances };
  }
}

export const training = new Training();
