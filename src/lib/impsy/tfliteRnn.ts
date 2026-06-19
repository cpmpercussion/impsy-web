// TFLiteRNN — port of TFLiteRNN.swift, over @litertjs/core.
//
// Wraps a LiteRT CompiledModel for single-step IMPSY MDRNN inference, holding
// the LSTM state between calls. NOT safe to call re-entrantly; drive it from a
// single loop (the InteractionEngine).
//
// Tensor convention (verified against ../impsy/impsy/mdrnn.py TfliteMDRNN):
//   inputs   shape (1,1,dimension)  — scaled by SCALE_FACTOR
//   state_h_N / state_c_N  shape (1, hiddenUnits)
//   MDN output  shape (1, numMixtures*(2*dimension+1))
//   + updated state_h_N / state_c_N outputs

import {
  loadLiteRt,
  loadAndCompile,
  Tensor,
  type CompiledModel,
  type Accelerator,
} from "@litertjs/core";
import { SCALE_FACTOR } from "./constants";
import { sampleMDN } from "./mdnSampler";

export interface ModelConfig {
  dimension: number;
  numLayers: number;
  hiddenUnits: number;
  numMixtures: number;
}

let liteRtReady: Promise<void> | null = null;

/**
 * Load the LiteRT WASM runtime once. `wasmPath` is the directory holding the
 * LiteRT wasm glue (served from the installed package at /litert-wasm/ — see
 * vite.config / main.ts), so it works offline without a CDN.
 */
export function initLiteRt(wasmPath: string): Promise<void> {
  if (!liteRtReady) {
    liteRtReady = loadLiteRt(wasmPath).then(() => undefined);
  }
  return liteRtReady;
}

interface TensorDetail {
  name: string;
  index: number;
  shape: Int32Array;
  dtype: "float32" | "int32";
}

export class TFLiteRNN {
  readonly config: ModelConfig;
  private model: CompiledModel;

  // Input tensor names by role (resolved from the compiled model).
  private inputsName: string;
  private stateInputNames: string[]; // [h0, c0, h1, c1, …]
  // Output tensor names by role.
  private mdnOutputName: string;
  private stateOutputNames: string[]; // [h0, c0, h1, c1, …]

  // LSTM state, flat: [h0, c0, h1, c1, …], each Float32Array(hiddenUnits).
  private lstmStates: Float32Array[];

  private constructor(model: CompiledModel, config: ModelConfig) {
    this.model = model;
    this.config = config;
    this.lstmStates = Array.from(
      { length: config.numLayers * 2 },
      () => new Float32Array(config.hiddenUnits),
    );

    const inputs = model.getInputDetails() as TensorDetail[];
    const outputs = model.getOutputDetails() as TensorDetail[];

    // Inputs: the 3-D tensor is `inputs`; the 2-D tensors are LSTM states,
    // ordered by tensor index as [h0, c0, h1, c1, …] (matches mdrnn.py).
    const inputsDetail = inputs.find((d) => d.shape.length === 3);
    if (!inputsDetail) throw new Error("TFLiteRNN: could not find 'inputs' tensor");
    this.inputsName = inputsDetail.name;
    this.stateInputNames = inputs
      .filter((d) => d.shape.length === 2)
      .sort((a, b) => a.index - b.index)
      .map((d) => d.name);

    // Outputs: MDN output is the 2-D tensor whose width = M*(2D+1); the rest
    // are updated states in index order [h0, c0, h1, c1, …].
    const mdnWidth = config.numMixtures * (2 * config.dimension + 1);
    const mdnDetail =
      outputs.find((d) => d.shape.length === 2 && d.shape[d.shape.length - 1] === mdnWidth) ??
      outputs.reduce((max, d) =>
        d.shape.reduce((a, b) => a * b, 1) > max.shape.reduce((a, b) => a * b, 1) ? d : max,
      );
    this.mdnOutputName = mdnDetail.name;
    this.stateOutputNames = outputs
      .filter((d) => d.name !== mdnDetail.name)
      .sort((a, b) => a.index - b.index)
      .map((d) => d.name);
  }

  /** Build an RNN from model bytes (e.g. a fetched/uploaded .tflite file). */
  static async create(
    modelData: Uint8Array,
    accelerator: Accelerator = "wasm",
  ): Promise<TFLiteRNN> {
    const model = await loadAndCompile(modelData, { accelerator });
    const config = inspectModel(model);
    return new TFLiteRNN(model, config);
  }

  /**
   * Run one forward pass.
   * @param input dense vector length `dimension`; index 0 = dt (seconds),
   *   1…N normalised [0,1]. Scaling by SCALE_FACTOR is applied internally.
   * @returns sampled output vector length `dimension`.
   */
  generate(input: number[], piTemp: number, sigmaTemp: number): number[] {
    const scaled = new Float32Array(this.config.dimension);
    for (let i = 0; i < this.config.dimension; i++) scaled[i] = input[i] * SCALE_FACTOR;

    const feed: Record<string, Tensor> = {
      [this.inputsName]: Tensor.fromTypedArray(scaled, [1, 1, this.config.dimension]),
    };
    this.stateInputNames.forEach((name, i) => {
      feed[name] = Tensor.fromTypedArray(this.lstmStates[i], [1, this.config.hiddenUnits]);
    });

    const out = this.model.run(feed) as Record<string, Tensor>;

    const mdnParams = out[this.mdnOutputName].toTypedArray() as Float32Array;
    // Persist updated states (toTypedArray returns a copy we own).
    this.stateOutputNames.forEach((name, i) => {
      this.lstmStates[i] = out[name].toTypedArray() as Float32Array;
    });

    // Free WASM-backed tensors for this step.
    Object.values(feed).forEach((t) => t.delete());
    Object.values(out).forEach((t) => t.delete());

    return sampleMDN(
      mdnParams,
      this.config.dimension,
      this.config.numMixtures,
      piTemp,
      sigmaTemp,
    );
  }

  /** Zero all LSTM states. */
  resetStates(): void {
    this.lstmStates = this.lstmStates.map(() => new Float32Array(this.config.hiddenUnits));
  }

  dispose(): void {
    this.model.delete();
  }
}

/** Introspect (dimension, numLayers, hiddenUnits, numMixtures) from a model. */
export function inspectModel(model: CompiledModel): ModelConfig {
  const inputs = model.getInputDetails() as TensorDetail[];
  const outputs = model.getOutputDetails() as TensorDetail[];

  const inputsDetail = inputs.find((d) => d.shape.length === 3);
  if (!inputsDetail) throw new Error("inspectModel: missing 'inputs' tensor");
  const dimension = inputsDetail.shape[inputsDetail.shape.length - 1];

  const stateInputs = inputs.filter((d) => d.shape.length === 2);
  if (stateInputs.length === 0 || stateInputs.length % 2 !== 0) {
    throw new Error(`inspectModel: expected paired LSTM states, found ${stateInputs.length}`);
  }
  const numLayers = stateInputs.length / 2;
  const hiddenUnits = stateInputs[0].shape[stateInputs[0].shape.length - 1];

  // MDN output width = M*(2D+1) → solve for M.
  const mdnDetail =
    outputs.find((d) => d.shape.length === 2 && d.shape[1] !== hiddenUnits) ??
    outputs.reduce((max, d) =>
      d.shape.reduce((a, b) => a * b, 1) > max.shape.reduce((a, b) => a * b, 1) ? d : max,
    );
  const width = mdnDetail.shape[mdnDetail.shape.length - 1];
  const numMixtures = Math.round(width / (2 * dimension + 1));

  return { dimension, numLayers, hiddenUnits, numMixtures };
}
