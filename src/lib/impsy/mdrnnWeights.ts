// MDRNN weights — the trained parameters of an IMPSY model, independent of any
// file format, plus a plain-TypeScript forward pass over them.
//
// Layouts are Keras's (`model.get_weights()` on the model built by
// `build_mdrnn_model` in ../impsy/impsy/mdrnn.py, and TF.js `getWeights()`):
// dense kernels are `[inputs, outputs]` row-major, and LSTM kernels pack the
// four gates along the output axis in the order input, forget, cell, output.
// tfliteWriter.ts / tfliteReader.ts convert to and from TFLite's `[outputs,
// inputs]` layout.

import type { ModelConfig } from "./tfliteRnn";

export interface LSTMLayerWeights {
  /** `[inputSize, 4·hiddenUnits]`; inputSize is `dimension` for layer 0, else `hiddenUnits`. */
  kernel: Float32Array;
  /** `[hiddenUnits, 4·hiddenUnits]` */
  recurrentKernel: Float32Array;
  /** `[4·hiddenUnits]` */
  bias: Float32Array;
}

export interface DenseWeights {
  /** `[hiddenUnits, outputs]` */
  kernel: Float32Array;
  /** `[outputs]` */
  bias: Float32Array;
}

export interface MDRNNWeights {
  config: ModelConfig;
  /** One entry per LSTM layer, first layer first. */
  lstm: LSTMLayerWeights[];
  /** The MDN head's three dense layers; `mus`/`sigmas` have `numMixtures·dimension` outputs, `pi` has `numMixtures`. */
  mdn: { mus: DenseWeights; sigmas: DenseWeights; pi: DenseWeights };
}

/** Model sizes by name (`SIZE_TO_PARAMETERS` in ../impsy/impsy/utils.py). */
export const MODEL_SIZES = {
  xxs: { hiddenUnits: 16, numLayers: 2, numMixtures: 5 },
  xs: { hiddenUnits: 32, numLayers: 2, numMixtures: 5 },
  s: { hiddenUnits: 64, numLayers: 2, numMixtures: 5 },
  m: { hiddenUnits: 128, numLayers: 2, numMixtures: 5 },
} as const;

/**
 * The MDN layer's sigma activation is `elu(x) + 1 + epsilon` (keras-mdn-layer's
 * `elu_plus_one_plus_epsilon`, Keras epsilon 1e-7); the converter folds the
 * two constants into one float32.
 */
export const SIGMA_OFFSET = Math.fround(1 + 1e-7);

/**
 * Build weights from a Keras-ordered weight list: per LSTM layer `kernel,
 * recurrent_kernel, bias`, then the MDN head's `mus`, `sigmas`, `pi` as
 * `kernel, bias` each.
 */
export function weightsFromKerasList(config: ModelConfig, arrays: Float32Array[]): MDRNNWeights {
  const expected = config.numLayers * 3 + 6;
  if (arrays.length !== expected) {
    throw new Error(`weightsFromKerasList: expected ${expected} arrays, got ${arrays.length}`);
  }
  const lstm = Array.from({ length: config.numLayers }, (_, l) => ({
    kernel: arrays[l * 3],
    recurrentKernel: arrays[l * 3 + 1],
    bias: arrays[l * 3 + 2],
  }));
  const dense = (i: number): DenseWeights => ({ kernel: arrays[i], bias: arrays[i + 1] });
  const base = config.numLayers * 3;
  const weights = {
    config,
    lstm,
    mdn: { mus: dense(base), sigmas: dense(base + 2), pi: dense(base + 4) },
  };
  validateWeights(weights);
  return weights;
}

/** Throw if any array's length disagrees with `weights.config`. */
export function validateWeights(weights: MDRNNWeights): void {
  const { dimension: D, numLayers, hiddenUnits: H, numMixtures: M } = weights.config;
  for (const [name, v] of Object.entries({ dimension: D, numLayers, hiddenUnits: H, M })) {
    if (!Number.isInteger(v) || v < 1) throw new Error(`MDRNN weights: bad ${name} ${v}`);
  }
  const check = (name: string, a: Float32Array, length: number) => {
    if (!(a instanceof Float32Array) || a.length !== length) {
      throw new Error(`MDRNN weights: ${name} should have ${length} values, has ${a?.length}`);
    }
  };
  if (weights.lstm.length !== numLayers) {
    throw new Error(`MDRNN weights: ${weights.lstm.length} LSTM layers, config says ${numLayers}`);
  }
  weights.lstm.forEach((layer, l) => {
    check(`lstm ${l} kernel`, layer.kernel, (l === 0 ? D : H) * 4 * H);
    check(`lstm ${l} recurrent kernel`, layer.recurrentKernel, H * 4 * H);
    check(`lstm ${l} bias`, layer.bias, 4 * H);
  });
  const head = { mus: M * D, sigmas: M * D, pi: M } as const;
  for (const [name, outputs] of Object.entries(head)) {
    const dense = weights.mdn[name as keyof typeof head];
    check(`mdn ${name} kernel`, dense.kernel, H * outputs);
    check(`mdn ${name} bias`, dense.bias, outputs);
  }
}

/** `out[j] = bias[j] + Σ_i x[i]·kernel[i, j]` for a Keras-layout `[x.length, outputs]` kernel. */
function affine(
  x: ArrayLike<number>,
  kernel: Float32Array,
  outputs: number,
  bias?: Float32Array,
): Float64Array {
  const out = new Float64Array(outputs);
  if (bias) out.set(bias);
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    const row = i * outputs;
    for (let j = 0; j < outputs; j++) out[j] += xi * kernel[row + j];
  }
  return out;
}

const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));

/** Fresh zeroed LSTM states `[h0, c0, h1, c1, …]`. */
export function zeroStates(config: ModelConfig): Float32Array[] {
  return Array.from({ length: config.numLayers * 2 }, () => new Float32Array(config.hiddenUnits));
}

/**
 * One forward step of the MDRNN in plain TypeScript: the same function the
 * `.tflite` graph computes. Used to check the reader and writer against real
 * models; inference in the app goes through LiteRT (tfliteRnn.ts).
 *
 * @param input model input, already multiplied by SCALE_FACTOR, length `dimension`
 * @param states `[h0, c0, h1, c1, …]`
 * @returns the raw MDN output `[mus | sigmas | piLogits]` and the updated states
 */
export function mdrnnStep(
  weights: MDRNNWeights,
  input: ArrayLike<number>,
  states: Float32Array[],
): { mdn: Float32Array; states: Float32Array[] } {
  const { dimension: D, hiddenUnits: H, numMixtures: M } = weights.config;
  let x: ArrayLike<number> = input;
  const next: Float32Array[] = [];
  weights.lstm.forEach((layer, l) => {
    const hPrev = states[l * 2];
    const cPrev = states[l * 2 + 1];
    const z = affine(x, layer.kernel, 4 * H, layer.bias);
    const zr = affine(hPrev, layer.recurrentKernel, 4 * H);
    const h = new Float32Array(H);
    const c = new Float32Array(H);
    for (let j = 0; j < H; j++) {
      const i = sigmoid(z[j] + zr[j]);
      const f = sigmoid(z[H + j] + zr[H + j]);
      const g = Math.tanh(z[2 * H + j] + zr[2 * H + j]);
      const o = sigmoid(z[3 * H + j] + zr[3 * H + j]);
      c[j] = f * cPrev[j] + i * g;
      h[j] = o * Math.tanh(c[j]);
    }
    next.push(h, c);
    x = h;
  });

  const mus = affine(x, weights.mdn.mus.kernel, M * D, weights.mdn.mus.bias);
  const sigmas = affine(x, weights.mdn.sigmas.kernel, M * D, weights.mdn.sigmas.bias);
  const pi = affine(x, weights.mdn.pi.kernel, M, weights.mdn.pi.bias);
  const mdn = new Float32Array(M * (2 * D + 1));
  mdn.set(mus, 0);
  for (let j = 0; j < M * D; j++) {
    const s = sigmas[j];
    mdn[M * D + j] = (s > 0 ? s : Math.expm1(s)) + SIGMA_OFFSET;
  }
  mdn.set(pi, 2 * M * D);
  return { mdn, states: next };
}
