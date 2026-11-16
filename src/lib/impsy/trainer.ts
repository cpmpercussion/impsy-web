// MDRNN training in TensorFlow.js — port of the training path in
// ../impsy/impsy/train.py (`train_mdrnn`) and ../impsy/impsy/mdrnn.py
// (`PredictiveMusicMDRNN.train`), producing MDRNNWeights that tfliteWriter.ts
// turns into a `.tflite`.
//
// The network is written against tfjs-core ops, not tfjs-layers: it computes
// the same function as mdrnnWeights.ts `mdrnnStep` (Keras's LSTM with sigmoid
// recurrent activation, weights in Keras layout), unrolled over the training
// sequence, so trained variables are MDRNNWeights with no conversion. The
// LSTM's gradients are written by hand, for speed — see "The network and its
// gradients" below.
//
// Matches Python: inputs and targets ×SCALE_FACTOR, teacher forcing over
// 50-step windows from zero state, Adam (lr 0.001), batch 64, the last 10% of
// examples held out for validation, early stopping on validation loss
// (patience 10), stop on NaN, and Keras's default initialisers.
//
// The caller picks the TF.js backend by importing one (e.g.
// `@tensorflow/tfjs-backend-webgl`) before calling `trainMDRNN`.

import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-core/dist/register_all_gradients";
import { SCALE_FACTOR } from "./constants";
import { SEQUENCE_LENGTH, sliceExamples, type Example, type Performance } from "./dataset";
import { mdnLoss } from "./mdnLoss";
import {
  SIGMA_OFFSET,
  validateWeights,
  weightsFromKerasList,
  type MDRNNWeights,
} from "./mdrnnWeights";
import type { ModelConfig } from "./tfliteRnn";

export interface EpochInfo {
  /** 1-based. */
  epoch: number;
  /** Mean training loss over the epoch. */
  loss: number;
  /** Validation loss after the epoch; null when nothing is held out. */
  valLoss: number | null;
}

export interface TrainOptions {
  config: ModelConfig;
  /** Maximum number of epochs. Default 100. */
  epochs?: number;
  /** Default 64. */
  batchSize?: number;
  /** Fraction of examples (the last ones) held out for validation. Default 0.1. */
  validationSplit?: number;
  /** Stop when validation loss hasn't improved for `patience` epochs. Default true. */
  earlyStopping?: boolean;
  /** Default 10. */
  patience?: number;
  /** Rows per training window. Default 50. */
  sequenceLength?: number;
  /** Adam learning rate. Default 0.001. */
  learningRate?: number;
  /** Start from these weights (fine-tuning) instead of a fresh initialisation. */
  initialWeights?: MDRNNWeights;
  /** Uniform [0,1) source for initialisation and shuffling. Default `Math.random`. */
  random?: () => number;
  onEpochEnd?: (info: EpochInfo) => void;
  /** Called after each training batch with progress through the epoch. */
  onBatchEnd?: (info: { epoch: number; batch: number; batches: number; loss: number }) => void;
  /** Abort to stop after the current batch; the weights so far are still returned. */
  signal?: AbortSignal;
}

export interface TrainResult {
  /** Weights after the last epoch run — what Python saves as the model. */
  weights: MDRNNWeights;
  /** Weights from the epoch with the lowest validation loss (Python's `-ckpt` file); null without validation. */
  bestWeights: MDRNNWeights | null;
  history: EpochInfo[];
  /** Why training ended. */
  stopped: "epochs" | "early-stopping" | "aborted" | "nan";
  /** Number of training / validation examples used. */
  examples: { train: number; validation: number };
}

// ── Initialisation (Keras defaults) ───────────────────────────────────────────

/** Glorot uniform, for a `[fanIn, fanOut]` kernel. */
function glorotUniform(fanIn: number, fanOut: number, random: () => number): Float32Array {
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  return Float32Array.from({ length: fanIn * fanOut }, () => (random() * 2 - 1) * limit);
}

/**
 * Keras's Orthogonal initialiser for a `[rows, cols]` matrix with rows ≤ cols:
 * orthonormal rows, from Gram–Schmidt on a Gaussian matrix.
 */
function orthogonal(rows: number, cols: number, random: () => number): Float32Array {
  const gaussian = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
  const vectors: Float64Array[] = [];
  for (let r = 0; r < rows; r++) {
    const v = Float64Array.from({ length: cols }, gaussian);
    for (const u of vectors) {
      let dot = 0;
      for (let i = 0; i < cols; i++) dot += v[i] * u[i];
      for (let i = 0; i < cols; i++) v[i] -= dot * u[i];
    }
    const norm = Math.hypot(...v);
    vectors.push(v.map((x) => x / norm));
  }
  const out = new Float32Array(rows * cols);
  vectors.forEach((v, r) => out.set(v, r * cols));
  return out;
}

/** Freshly initialised weights, as Keras builds the model: Glorot kernels, orthogonal recurrent kernels, zero biases with the forget gate's at 1. */
export function initialWeights(
  config: ModelConfig,
  random: () => number = Math.random,
): MDRNNWeights {
  const { dimension: D, numLayers, hiddenUnits: H, numMixtures: M } = config;
  const arrays: Float32Array[] = [];
  for (let l = 0; l < numLayers; l++) {
    const inSize = l === 0 ? D : H;
    const bias = new Float32Array(4 * H);
    bias.fill(1, H, 2 * H); // unit_forget_bias
    arrays.push(glorotUniform(inSize, 4 * H, random), orthogonal(H, 4 * H, random), bias);
  }
  for (const outputs of [M * D, M * D, M]) {
    arrays.push(glorotUniform(H, outputs, random), new Float32Array(outputs));
  }
  return weightsFromKerasList(config, arrays);
}

// ── The network as TF.js variables ────────────────────────────────────────────

export interface Variables {
  lstm: { kernel: tf.Variable; recurrentKernel: tf.Variable; bias: tf.Variable }[];
  mdn: Record<"mus" | "sigmas" | "pi", { kernel: tf.Variable; bias: tf.Variable }>;
}

export function toVariables(weights: MDRNNWeights): Variables {
  const { dimension: D, hiddenUnits: H, numMixtures: M } = weights.config;
  // tf.variable copies its initial value; tidy frees those (variables outlive it).
  return tf.tidy(() => {
    const dense = (name: "mus" | "sigmas" | "pi", outputs: number) => ({
      kernel: tf.variable(tf.tensor2d(weights.mdn[name].kernel, [H, outputs])),
      bias: tf.variable(tf.tensor1d(weights.mdn[name].bias)),
    });
    return {
      lstm: weights.lstm.map((layer, l) => ({
        kernel: tf.variable(tf.tensor2d(layer.kernel, [l === 0 ? D : H, 4 * H])),
        recurrentKernel: tf.variable(tf.tensor2d(layer.recurrentKernel, [H, 4 * H])),
        bias: tf.variable(tf.tensor1d(layer.bias)),
      })),
      mdn: { mus: dense("mus", M * D), sigmas: dense("sigmas", M * D), pi: dense("pi", M) },
    };
  });
}

export function allVariables(v: Variables): tf.Variable[] {
  return [
    ...v.lstm.flatMap((l) => [l.kernel, l.recurrentKernel, l.bias]),
    ...[v.mdn.mus, v.mdn.sigmas, v.mdn.pi].flatMap((d) => [d.kernel, d.bias]),
  ];
}

function toWeights(config: ModelConfig, v: Variables): MDRNNWeights {
  return weightsFromKerasList(
    config,
    allVariables(v).map((variable) => Float32Array.from(variable.dataSync())),
  );
}

// ── The network and its gradients ─────────────────────────────────────────────
//
// How this is written is shaped by what is slow on the TF.js WASM backend
// (measured with verify/train-bench.html?profile=1). The matrices here are
// small, so the arithmetic is cheap and the costs are elsewhere:
//
//  - `tf.slice`, `tf.split` and `tf.unstack` copy the *whole source tensor* out
//    of the WASM heap for each piece they take, and `tf.gather` is a scalar
//    loop. Stepping through a sequence tensor that way costs steps × the
//    sequence, and was most of the time. So sequences are lists of per-step
//    tensors, joined with `tf.concat` (cheap) and never taken apart again,
//    and each gate has its own weights so nothing is split per step.
//  - `tf.matMul` only reaches XNNPACK (SIMD, threads) when neither operand is
//    flagged as transposed; with a flag it is a scalar loop. Autodiff's matMul
//    gradients always pass a flag, so the gradients here transpose explicitly.
//  - Every kernel call costs several microseconds whatever its size, and
//    autodiff through an unrolled LSTM makes about three times as many as the
//    hand-written backward pass below.
//
// trainer.test.ts checks the gradients against autodiff of the plain Keras formulation.

/** Training data for one batch, time-major: step `t`'s rows are the batch at that step. */
export interface Batch {
  /** Scaled inputs: one `(batch, dimension)` tensor per step. */
  xs: tf.Tensor2D[];
  /** Scaled targets, `(steps·batch, dimension)`. */
  y: tf.Tensor2D;
}

/** What one LSTM step leaves behind for the backward pass. */
interface StepCache {
  /** `[x_t, h_{t-1}]`, `(batch, inputSize + H)`. */
  xh: tf.Tensor;
  i: tf.Tensor;
  f: tf.Tensor;
  g: tf.Tensor;
  o: tf.Tensor;
  cPrev: tf.Tensor;
  tanhC: tf.Tensor;
}

/**
 * One Keras LSTM layer (`return_sequences`, sigmoid recurrent activation, zero
 * initial state): `inputs` and the result are one `(batch, n)` tensor per step.
 */
function lstmForward(
  layer: Variables["lstm"][number],
  inputs: tf.Tensor[],
): { outputs: tf.Tensor[]; cache: StepCache[] } {
  const batch = inputs[0].shape[0];
  const H = layer.recurrentKernel.shape[0];
  // Per-gate weights over [x, h], in Keras's gate order i, f, g (the cell candidate), o.
  const [Wi, Wf, Wg, Wo] = tf.split(tf.concat([layer.kernel, layer.recurrentKernel], 0), 4, 1);
  const [bi, bf, bg, bo] = tf.split(layer.bias, 4);
  const gate = (xh: tf.Tensor, W: tf.Tensor, bias: tf.Tensor) =>
    tf.fused.matMul({ a: xh, b: W, bias, activation: "sigmoid" });

  let h: tf.Tensor = tf.zeros([batch, H]);
  let c: tf.Tensor = tf.zeros([batch, H]);
  const outputs: tf.Tensor[] = [];
  const cache: StepCache[] = [];
  for (const x of inputs) {
    const xh = tf.concat([x, h], 1);
    const i = gate(xh, Wi, bi);
    const f = gate(xh, Wf, bf);
    const o = gate(xh, Wo, bo);
    const g = tf.tanh(tf.fused.matMul({ a: xh, b: Wg, bias: bg }));
    const cPrev = c;
    c = tf.add(tf.mul(f, cPrev), tf.mul(i, g));
    const tanhC = tf.tanh(c);
    h = tf.mul(o, tanhC);
    outputs.push(h);
    cache.push({ xh, i, f, g, o, cPrev, tanhC });
  }
  return { outputs, cache };
}

/**
 * Backpropagation through time for `lstmForward`.
 *
 * @param dOutputs the loss gradient with respect to each step's output
 * @param needInputs whether to compute the gradient with respect to the inputs (not for the first layer)
 */
function lstmBackward(
  layer: Variables["lstm"][number],
  cache: StepCache[],
  dOutputs: tf.Tensor[],
  needInputs: boolean,
): { kernel: tf.Tensor; recurrentKernel: tf.Tensor; bias: tf.Tensor; dInputs: tf.Tensor[] } {
  const [inputSize, width] = layer.kernel.shape;
  const H = layer.recurrentKernel.shape[0];
  const kernelT = needInputs ? tf.transpose(layer.kernel) : null;
  const recurrentT = tf.transpose(layer.recurrentKernel);
  const one = tf.scalar(1);
  /** Derivative of a sigmoid, from its output. */
  const sigmoidPrime = (s: tf.Tensor) => tf.mul(s, tf.sub(one, s));
  /** Derivative of a tanh, from its output. */
  const tanhPrime = (t: tf.Tensor) => tf.sub(one, tf.square(t));

  const steps = cache.length;
  const dzs: tf.Tensor[] = new Array(steps);
  const dInputs: tf.Tensor[] = new Array(steps);
  let dhNext: tf.Tensor | null = null;
  let dcNext: tf.Tensor | null = null;
  for (let t = steps - 1; t >= 0; t--) {
    const { i, f, g, o, cPrev, tanhC } = cache[t];
    const dh: tf.Tensor = dhNext ? tf.add(dOutputs[t], dhNext) : dOutputs[t];
    let dc: tf.Tensor = tf.mul(tf.mul(dh, o), tanhPrime(tanhC));
    if (dcNext) dc = tf.add(dc, dcNext);
    // Gradients at the gates' pre-activations, in Keras's column order.
    const dz = tf.concat(
      [
        tf.mul(tf.mul(dc, g), sigmoidPrime(i)),
        tf.mul(tf.mul(dc, cPrev), sigmoidPrime(f)),
        tf.mul(tf.mul(dc, i), tanhPrime(g)),
        tf.mul(tf.mul(dh, tanhC), sigmoidPrime(o)),
      ],
      1,
    );
    dzs[t] = dz;
    if (kernelT) dInputs[t] = tf.matMul(dz, kernelT);
    if (t > 0) {
      dcNext = tf.mul(dc, f);
      dhNext = tf.matMul(dz, recurrentT);
    }
  }
  // The weight gradients sum over steps: one matMul over the whole sequence.
  const dz = tf.concat(dzs, 0);
  const xh = tf.concat(
    cache.map((step) => step.xh),
    0,
  );
  const dW = tf.matMul(tf.transpose(xh), dz);
  return {
    kernel: tf.slice(dW, [0, 0], [inputSize, width]),
    recurrentKernel: tf.slice(dW, [inputSize, 0], [H, width]),
    bias: tf.sum(dz, 0),
    dInputs,
  };
}

/** `tf.matMul` whose gradient transposes explicitly (see the note above). Same values as the stock gradient. */
const matMul = tf.customGrad((...args: (tf.Tensor | tf.GradSaveFunc)[]) => {
  const [a, b, save] = args as [tf.Tensor, tf.Tensor, tf.GradSaveFunc];
  save([a, b]);
  return {
    value: tf.matMul(a, b),
    gradFunc: (dy: tf.Tensor, [a, b]: tf.Tensor[]) => [
      tf.matMul(dy, tf.transpose(b)),
      tf.matMul(tf.transpose(a), dy),
    ],
  };
}) as (a: tf.Tensor, b: tf.Tensor) => tf.Tensor;

/** The six MDN head variables, in the order `mdnHead` takes them. */
const headVariables = (v: Variables): tf.Variable[] =>
  [v.mdn.mus, v.mdn.sigmas, v.mdn.pi].flatMap((d) => [d.kernel, d.bias]);

/** The MDN head: `(rows, H)` → `(rows, M·(2D+1))`, laid out `[mus | sigmas | piLogits]`. */
function mdnHead(top: tf.Tensor, ...head: tf.Tensor[]): tf.Tensor {
  const [musK, musB, sigmasK, sigmasB, piK, piB] = head;
  const dense = (kernel: tf.Tensor, bias: tf.Tensor) => tf.add(matMul(top, kernel), bias);
  const sigmas = tf.add(tf.elu(dense(sigmasK, sigmasB)), SIGMA_OFFSET);
  return tf.concat([dense(musK, musB), sigmas, dense(piK, piB)], 1);
}

/** `(steps·batch, n)` → one `(batch, n)` tensor per step, in two levels so that no slice copies much. */
function splitSteps(all: tf.Tensor, steps: number): tf.Tensor[] {
  const [rows, n] = all.shape;
  const batch = rows / steps;
  const chunk = Math.ceil(Math.sqrt(steps));
  const out: tf.Tensor[] = [];
  for (let from = 0; from < steps; from += chunk) {
    const length = Math.min(chunk, steps - from);
    const part = tf.slice(all, [from * batch, 0], [length * batch, n]);
    for (let t = 0; t < length; t++) out.push(tf.slice(part, [t * batch, 0], [batch, n]));
  }
  return out;
}

/** The mean MDN loss of the network on a batch (the Keras training model, from zero state). */
export function batchLoss(v: Variables, { xs, y }: Batch, config: ModelConfig): tf.Scalar {
  return tf.tidy(() => {
    let sequence: tf.Tensor[] = xs;
    for (const layer of v.lstm) sequence = lstmForward(layer, sequence).outputs;
    const pred = mdnHead(tf.concat(sequence, 0), ...headVariables(v));
    return mdnLoss(y, pred, config.dimension, config.numMixtures);
  });
}

/**
 * `batchLoss` and its gradient with respect to every variable, keyed by
 * variable name (as `Optimizer.applyGradients` takes them).
 */
export function lossAndGradients(
  v: Variables,
  { xs, y }: Batch,
  config: ModelConfig,
): { loss: tf.Scalar; grads: tf.NamedTensorMap } {
  return tf.tidy(() => {
    const caches: StepCache[][] = [];
    let sequence: tf.Tensor[] = xs;
    for (const layer of v.lstm) {
      const { outputs, cache } = lstmForward(layer, sequence);
      caches.push(cache);
      sequence = outputs;
    }
    // Autodiff for the head and the loss, by hand below that.
    const head = headVariables(v);
    const { value, grads: headGrads } = tf.valueAndGrads((top: tf.Tensor, ...h: tf.Tensor[]) =>
      mdnLoss(y, mdnHead(top, ...h), config.dimension, config.numMixtures),
    )([tf.concat(sequence, 0), ...head]);

    const grads: tf.NamedTensorMap = {};
    head.forEach((variable, n) => (grads[variable.name] = headGrads[n + 1]));
    let dOutputs = splitSteps(headGrads[0], xs.length);
    for (let l = v.lstm.length - 1; l >= 0; l--) {
      const layer = v.lstm[l];
      const back = lstmBackward(layer, caches[l], dOutputs, l > 0);
      grads[layer.kernel.name] = back.kernel;
      grads[layer.recurrentKernel.name] = back.recurrentKernel;
      grads[layer.bias.name] = back.bias;
      dOutputs = back.dInputs;
    }
    return { loss: value as tf.Scalar, grads };
  });
}

// ── Training ──────────────────────────────────────────────────────────────────

/**
 * Yield to the event loop between batches, so a worker can receive "stop" (and
 * a page stays responsive). A MessageChannel task rather than `setTimeout(0)`,
 * which browsers clamp to 4 ms once nested — a sixth of a small model's batch.
 */
const breathe = () =>
  new Promise<void>((resolve) => {
    const { port1, port2 } = new MessageChannel();
    port1.onmessage = () => {
      port1.close();
      resolve();
    };
    port2.postMessage(null);
  });

/**
 * Train an MDRNN on performances (rows `[dt, x_1, …, x_n]`, unscaled, as
 * dataset.ts produces them). Resolves with the trained weights; pass them to
 * `writeTflite` for a model file.
 */
export async function trainMDRNN(
  performances: Performance[],
  options: TrainOptions,
): Promise<TrainResult> {
  const {
    config,
    epochs = 100,
    batchSize = 64,
    validationSplit = 0.1,
    earlyStopping = true,
    patience = 10,
    sequenceLength = SEQUENCE_LENGTH,
    learningRate = 0.001,
    random = Math.random,
  } = options;
  const { dimension: D, hiddenUnits: H, numMixtures: M } = config;

  for (const rows of performances) {
    for (const row of rows) {
      if (row.length !== D) {
        throw new Error(`Training data has rows of ${row.length} values; the model needs ${D}`);
      }
    }
  }
  const examples = sliceExamples(performances, sequenceLength);
  if (examples.length === 0) {
    throw new Error(
      `No performances are longer than ${sequenceLength + 1} events, so there's nothing to train on. Record some longer logs.`,
    );
  }
  // Keras's validation_split: the last fraction of the examples, taken before shuffling.
  const splitAt = Math.floor(examples.length * (1 - validationSplit));
  const trainExamples = examples.slice(0, splitAt);
  const valExamples = examples.slice(splitAt);
  if (trainExamples.length === 0) throw new Error("Not enough examples to train on.");

  /** Inputs and targets for a batch of examples, scaled and time-major. */
  const makeBatch = (batch: Example[]): Batch => {
    const B = batch.length;
    const rowsOf = batch.map(({ performance }) => performances[performance]);
    const xs: tf.Tensor2D[] = [];
    const y = new Float32Array(sequenceLength * B * D);
    for (let t = 0; t < sequenceLength; t++) {
      const x = new Float32Array(B * D);
      for (let b = 0; b < B; b++) {
        const at = batch[b].start + t;
        for (let d = 0; d < D; d++) {
          x[b * D + d] = rowsOf[b][at][d] * SCALE_FACTOR;
          y[(t * B + b) * D + d] = rowsOf[b][at + 1][d] * SCALE_FACTOR;
        }
      }
      xs.push(tf.tensor2d(x, [B, D]));
    }
    return { xs, y: tf.tensor2d(y, [sequenceLength * B, D]) };
  };

  if (options.initialWeights) {
    validateWeights(options.initialWeights);
    const a = options.initialWeights.config;
    if (
      a.dimension !== D ||
      a.hiddenUnits !== H ||
      a.numMixtures !== M ||
      a.numLayers !== config.numLayers
    ) {
      throw new Error("initialWeights are for a different model shape than config");
    }
  }
  const variables = toVariables(options.initialWeights ?? initialWeights(config, random));
  const optimizer = tf.train.adam(learningRate, 0.9, 0.999, 1e-7);

  const history: EpochInfo[] = [];
  let bestWeights: MDRNNWeights | null = null;
  let bestValLoss = Infinity;
  let sinceBest = 0;
  let stopped: TrainResult["stopped"] = "epochs";

  try {
    epochLoop: for (let epoch = 1; epoch <= epochs; epoch++) {
      // Fisher–Yates shuffle of the training examples.
      const order = [...trainExamples];
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }

      const batches = Math.ceil(order.length / batchSize);
      let lossSum = 0;
      let seen = 0;
      for (let b = 0; b < batches; b++) {
        if (options.signal?.aborted) {
          stopped = "aborted";
          break epochLoop;
        }
        const batch = order.slice(b * batchSize, (b + 1) * batchSize);
        const data = makeBatch(batch);
        // What `optimizer.minimize` does, with our gradients in place of autodiff's.
        const lossTensor = tf.tidy(() => {
          const { loss, grads } = lossAndGradients(variables, data, config);
          optimizer.applyGradients(
            Object.entries(grads).map(([name, tensor]) => ({ name, tensor })),
          );
          return loss;
        });
        const loss = (await lossTensor.data())[0];
        tf.dispose([data.xs, data.y, lossTensor]);
        if (!Number.isFinite(loss)) {
          stopped = "nan";
          break epochLoop;
        }
        lossSum += loss * batch.length;
        seen += batch.length;
        options.onBatchEnd?.({ epoch, batch: b + 1, batches, loss });
        await breathe();
      }

      let valLoss: number | null = null;
      if (valExamples.length > 0) {
        let valSum = 0;
        for (let i = 0; i < valExamples.length; i += batchSize) {
          const batch = valExamples.slice(i, i + batchSize);
          const data = makeBatch(batch);
          const lossTensor = batchLoss(variables, data, config);
          valSum += (await lossTensor.data())[0] * batch.length;
          tf.dispose([data.xs, data.y, lossTensor]);
          await breathe();
        }
        valLoss = valSum / valExamples.length;
      }

      const info = { epoch, loss: lossSum / seen, valLoss };
      history.push(info);
      options.onEpochEnd?.(info);

      if (valLoss !== null) {
        if (valLoss < bestValLoss) {
          bestValLoss = valLoss;
          bestWeights = toWeights(config, variables);
          sinceBest = 0;
        } else if (++sinceBest >= patience && earlyStopping) {
          stopped = "early-stopping";
          break;
        }
      }
    }
    return {
      weights: toWeights(config, variables),
      bestWeights,
      history,
      stopped,
      examples: { train: trainExamples.length, validation: valExamples.length },
    };
  } finally {
    tf.dispose(allVariables(variables));
    optimizer.dispose();
  }
}
