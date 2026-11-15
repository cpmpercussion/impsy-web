import { readFileSync } from "node:fs";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-cpu";
import { beforeAll, describe, expect, it } from "vitest";
import { SCALE_FACTOR } from "./constants";
import { sliceExamples, type Performance } from "./dataset";
import { mdnLoss } from "./mdnLoss";
import { mdrnnStep, zeroStates, type MDRNNWeights } from "./mdrnnWeights";
import { readTfliteWeights } from "./tfliteReader";
import { writeTflite } from "./tfliteWriter";
import { initialWeights, trainMDRNN } from "./trainer";

beforeAll(async () => {
  await tf.setBackend("cpu");
});

/** Seeded uniform [0,1) (mulberry32). */
function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A slightly noisy sine wave sampled every ~0.1 s: rows `[dt, x]`. */
function sinePerformance(rows: number, random: () => number): Performance {
  return Array.from({ length: rows }, (_, i) => [
    0.1 + (random() - 0.5) * 0.01,
    0.5 + 0.4 * Math.sin(i / 5) + (random() - 0.5) * 0.02,
  ]);
}

/**
 * The MDN loss in plain TypeScript, over windows of rows run through `mdrnnStep` —
 * independent of TF.js, and the same arithmetic the `.tflite` graph does.
 */
function referenceLoss(weights: MDRNNWeights, windows: number[][][]) {
  const { dimension: D, numMixtures: M } = weights.config;
  let total = 0;
  let count = 0;
  for (const rows of windows) {
    let states = zeroStates(weights.config);
    for (let t = 0; t + 1 < rows.length; t++) {
      const input = rows[t].map((v) => v * SCALE_FACTOR);
      const target = rows[t + 1].map((v) => v * SCALE_FACTOR);
      const out = mdrnnStep(weights, input, states);
      states = out.states;
      const logits = Array.from(out.mdn.subarray(2 * M * D));
      const logNorm = Math.log(logits.reduce((a, l) => a + Math.exp(l), 0));
      let prob = 0;
      for (let k = 0; k < M; k++) {
        let logComponent = -0.5 * D * Math.log(2 * Math.PI);
        for (let d = 0; d < D; d++) {
          const mu = out.mdn[k * D + d];
          const sigma = out.mdn[M * D + k * D + d];
          logComponent -= Math.log(sigma) + 0.5 * ((target[d] - mu) / sigma) ** 2;
        }
        prob += Math.exp(logits[k] - logNorm + logComponent);
      }
      total -= Math.log(prob);
      count++;
    }
  }
  return total / count;
}

describe("mdnLoss", () => {
  it("matches keras-mdn-layer's loss", () => {
    const fixture = JSON.parse(
      readFileSync(new URL("./fixtures/mdnLoss.json", import.meta.url), "utf8"),
    );
    const loss = mdnLoss(
      tf.tensor2d(fixture.y_true),
      tf.tensor2d(fixture.y_pred),
      fixture.dimension,
      fixture.mixtures,
    );
    expect(loss.dataSync()[0]).toBeCloseTo(fixture.loss, 4);
  });
});

describe("initialWeights", () => {
  it("initialises like Keras: orthonormal recurrent kernel, unit forget bias", () => {
    const H = 8;
    const weights = initialWeights(
      { dimension: 3, numLayers: 2, hiddenUnits: H, numMixtures: 5 },
      seeded(1),
    );
    const { recurrentKernel, bias, kernel } = weights.lstm[0];
    for (let a = 0; a < H; a++) {
      for (let b = 0; b < H; b++) {
        let dot = 0;
        for (let i = 0; i < 4 * H; i++)
          dot += recurrentKernel[a * 4 * H + i] * recurrentKernel[b * 4 * H + i];
        expect(dot).toBeCloseTo(a === b ? 1 : 0, 5);
      }
    }
    expect(Array.from(bias)).toEqual([
      ...Array(H).fill(0),
      ...Array(H).fill(1),
      ...Array(2 * H).fill(0),
    ]);
    const limit = Math.sqrt(6 / (3 + 4 * H));
    expect(Math.max(...kernel.map(Math.abs))).toBeLessThanOrEqual(limit);
    expect(Array.from(weights.mdn.pi.bias)).toEqual(Array(5).fill(0));
  });
});

describe("trainMDRNN", () => {
  const config = { dimension: 2, numLayers: 2, hiddenUnits: 16, numMixtures: 5 };
  const seqLen = 20;

  it("learns, and computes the same function as mdrnnStep and the written .tflite", async () => {
    const random = seeded(42);
    const performances = [sinePerformance(120, random), sinePerformance(90, random)];
    const before = tf.memory().numTensors;
    const result = await trainMDRNN(performances, {
      config,
      epochs: 12,
      batchSize: 32,
      sequenceLength: seqLen,
      learningRate: 0.01,
      random,
    });

    // 100 + 70 windows; Keras-style split: the last 10% validate.
    expect(result.examples).toEqual({ train: 153, validation: 17 });
    expect(result.stopped).toBe("epochs");
    expect(result.history).toHaveLength(12);
    const first = result.history[0];
    const last = result.history[11];
    expect(last.loss).toBeLessThan(first.loss - 1);
    expect(last.valLoss!).toBeLessThan(first.valLoss! - 1);
    expect(tf.memory().numTensors).toBe(before); // nothing leaked

    // The last epoch's validation loss was computed by TF.js with the final
    // weights; the plain-TS forward pass must agree on the same examples.
    const all = sliceExamples(performances, seqLen);
    const valWindows = all
      .slice(result.examples.train)
      .map(({ performance, start }) => performances[performance].slice(start, start + seqLen + 1));
    expect(referenceLoss(result.weights, valWindows)).toBeCloseTo(last.valLoss!, 3);

    // …and the weights survive the trip through a .tflite file.
    expect(readTfliteWeights(writeTflite(result.weights))).toEqual(result.weights);
    // bestWeights is the lowest-validation-loss epoch's checkpoint.
    const best = Math.min(...result.history.map((h) => h.valLoss!));
    expect(referenceLoss(result.bestWeights!, valWindows)).toBeCloseTo(best, 3);
  }, 120_000);

  it("stops early when validation loss stops improving", async () => {
    const random = seeded(3);
    const result = await trainMDRNN([sinePerformance(60, random)], {
      config,
      epochs: 50,
      sequenceLength: seqLen,
      learningRate: 0, // nothing can improve after the first epoch
      patience: 2,
      random,
    });
    expect(result.stopped).toBe("early-stopping");
    expect(result.history).toHaveLength(3);
  }, 60_000);

  it("stops on abort and still returns weights", async () => {
    const random = seeded(4);
    const controller = new AbortController();
    const result = await trainMDRNN([sinePerformance(60, random)], {
      config,
      epochs: 50,
      sequenceLength: seqLen,
      random,
      signal: controller.signal,
      onEpochEnd: () => controller.abort(),
    });
    expect(result.stopped).toBe("aborted");
    expect(result.history).toHaveLength(1);
    expect(result.weights.config).toEqual(config);
  }, 60_000);

  it("fine-tunes from existing weights", async () => {
    const random = seeded(5);
    const start = initialWeights(config, random);
    const result = await trainMDRNN([sinePerformance(60, random)], {
      config,
      epochs: 1,
      sequenceLength: seqLen,
      learningRate: 0,
      initialWeights: start,
      random,
    });
    expect(result.weights).toEqual(start);
  }, 60_000);

  it("explains when there is nothing to train on", async () => {
    await expect(trainMDRNN([sinePerformance(30, seeded(6))], { config })).rejects.toThrow(
      /nothing to train on/,
    );
    await expect(
      trainMDRNN([sinePerformance(80, seeded(7))], { config: { ...config, dimension: 4 } }),
    ).rejects.toThrow(/rows of 2 values/);
  });
});
