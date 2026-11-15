// MDRNN training in TensorFlow.js — port of the training path in
// ../impsy/impsy/train.py (`train_mdrnn`) and ../impsy/impsy/mdrnn.py
// (`PredictiveMusicMDRNN.train`), producing MDRNNWeights that tfliteWriter.ts
// turns into a `.tflite`.
//
// The network is written against tfjs-core ops, not tfjs-layers: it is the
// same arithmetic as mdrnnWeights.ts `mdrnnStep` (Keras's LSTM with sigmoid
// recurrent activation, weights in Keras layout), unrolled over the training
// sequence, so trained variables are MDRNNWeights with no conversion.
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

interface Variables {
  lstm: { kernel: tf.Variable; recurrentKernel: tf.Variable; bias: tf.Variable }[];
  mdn: Record<"mus" | "sigmas" | "pi", { kernel: tf.Variable; bias: tf.Variable }>;
}

function toVariables(weights: MDRNNWeights): Variables {
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

function allVariables(v: Variables): tf.Variable[] {
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

/**
 * Run the network over whole sequences from zero state (the Keras training
 * model: LSTMs with `return_sequences`, then the MDN head on every step).
 *
 * @param x scaled inputs, shape `(batch, steps, dimension)`
 * @returns MDN outputs for every step, shape `(batch·steps, M·(2D+1))`
 */
function forward(v: Variables, x: tf.Tensor3D, hiddenUnits: number): tf.Tensor2D {
  const batch = x.shape[0];
  let sequence: tf.Tensor[] = tf.unstack(x, 1);
  for (const layer of v.lstm) {
    let h: tf.Tensor = tf.zeros([batch, hiddenUnits]);
    let c: tf.Tensor = tf.zeros([batch, hiddenUnits]);
    const outputs: tf.Tensor[] = [];
    for (const step of sequence) {
      const z = tf.add(
        tf.add(tf.matMul(step, layer.kernel), tf.matMul(h, layer.recurrentKernel)),
        layer.bias,
      );
      const [zi, zf, zg, zo] = tf.split(z, 4, 1);
      c = tf.add(tf.mul(tf.sigmoid(zf), c), tf.mul(tf.sigmoid(zi), tf.tanh(zg)));
      h = tf.mul(tf.sigmoid(zo), tf.tanh(c));
      outputs.push(h);
    }
    sequence = outputs;
  }
  const top = tf.reshape(tf.stack(sequence, 1), [-1, hiddenUnits]);
  const dense = (d: { kernel: tf.Variable; bias: tf.Variable }) =>
    tf.add(tf.matMul(top, d.kernel), d.bias);
  const sigmas = tf.add(tf.elu(dense(v.mdn.sigmas)), SIGMA_OFFSET);
  return tf.concat([dense(v.mdn.mus), sigmas, dense(v.mdn.pi)], 1) as tf.Tensor2D;
}

// ── Training ──────────────────────────────────────────────────────────────────

/** Yield to the event loop so the page (and MIDI) stay responsive between batches. */
const breathe = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

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

  const window = sequenceLength + 1;
  /** Inputs and targets for a batch of examples, scaled: both `(batch, sequenceLength, D)`. */
  const makeBatch = (batch: Example[]): { x: tf.Tensor3D; y: tf.Tensor3D } => {
    const data = new Float32Array(batch.length * window * D);
    batch.forEach(({ performance, start }, b) => {
      const rows = performances[performance];
      for (let t = 0; t < window; t++) {
        const row = rows[start + t];
        const at = (b * window + t) * D;
        for (let d = 0; d < D; d++) data[at + d] = row[d] * SCALE_FACTOR;
      }
    });
    return tf.tidy(() => {
      const all = tf.tensor3d(data, [batch.length, window, D]);
      return {
        x: tf.slice(all, [0, 0, 0], [-1, sequenceLength, -1]) as tf.Tensor3D,
        y: tf.slice(all, [0, 1, 0], [-1, sequenceLength, -1]) as tf.Tensor3D,
      };
    });
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
  const batchLoss = (x: tf.Tensor3D, y: tf.Tensor3D): tf.Scalar =>
    mdnLoss(y, forward(variables, x, H), D, M);

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
        const { x, y } = makeBatch(batch);
        const lossTensor = optimizer.minimize(() => batchLoss(x, y), true) as tf.Scalar;
        const loss = (await lossTensor.data())[0];
        tf.dispose([x, y, lossTensor]);
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
          const { x, y } = makeBatch(batch);
          const lossTensor = tf.tidy(() => batchLoss(x, y));
          valSum += (await lossTensor.data())[0] * batch.length;
          tf.dispose([x, y, lossTensor]);
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
