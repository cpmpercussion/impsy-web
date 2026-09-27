// Mixture Density Network sampling — pure TypeScript port of MDNSampler.swift,
// which ports the sampling in ../impsy/impsy/mdrnn.py.
//
// The TFLite model outputs MDN parameters in this flat layout:
//
//   [ mus: M×D | sigmas: M×D | piLogits: M ]
//
// where M = numMixtures, D = dimension (including the time delta at index 0).
// All model values are scaled by SCALE_FACTOR; we divide by 10 on output.

import { SCALE_FACTOR, MINIMUM_DELTA_TIME } from "./constants";

/** One step's mixture in IMPSY units (÷ SCALE_FACTOR), temperatures applied. */
export interface MixtureParams {
  /** Mixture weights, softmax(piLogits / piTemp). Length M. */
  pi: number[];
  /** Means, M × D. */
  mu: number[][];
  /** Sampling standard deviations, sigma · sqrt(sigmaTemp), M × D. */
  std: number[][];
}

/** Split raw MDN output into pi / mu / std. Null if `params` is too short. */
export function mixtureParams(
  params: Float32Array | number[],
  dimension: number,
  numMixtures: number,
  piTemp: number,
  sigmaTemp: number,
): MixtureParams | null {
  const muCount = numMixtures * dimension;
  if (params.length < muCount * 2 + numMixtures) return null;

  const piLogits: number[] = [];
  for (let i = 0; i < numMixtures; i++) piLogits.push(params[muCount * 2 + i]);
  const sqrtSigmaTemp = Math.sqrt(sigmaTemp);
  const mu: number[][] = [];
  const std: number[][] = [];
  for (let m = 0; m < numMixtures; m++) {
    mu.push([]);
    std.push([]);
    for (let i = 0; i < dimension; i++) {
      mu[m].push(params[m * dimension + i] / SCALE_FACTOR);
      std[m].push((params[muCount + m * dimension + i] * sqrtSigmaTemp) / SCALE_FACTOR);
    }
  }
  return { pi: softmaxWithTemperature(piLogits, piTemp), mu, std };
}

/**
 * Sample a prediction vector of length `dimension` from MDN output params.
 * Index 0 of the result is the time delta (seconds); 1…N are normalised values.
 */
export function sampleMDN(
  params: Float32Array | number[],
  dimension: number,
  numMixtures: number,
  piTemp: number,
  sigmaTemp: number,
): number[] {
  const mix = mixtureParams(params, dimension, numMixtures, piTemp, sigmaTemp);
  if (!mix) return new Array(dimension).fill(0);

  // Select a mixture component, then sample its diagonal Gaussian:
  //   x_i = mu_i + sigma_i * sqrt(sigmaTemp) * N(0,1)
  const m = sampleCategorical(mix.pi);
  const out = mix.mu[m].map((mu, i) => mu + mix.std[m][i] * standardNormal());
  return clampPrediction(out);
}

/** Undo SCALE_FACTOR, enforce minimum dt, clamp values 1…N to [0,1]. */
export function postProcess(raw: number[]): number[] {
  return clampPrediction(raw.map((v) => v / SCALE_FACTOR));
}

/** Enforce minimum dt and clamp values 1…N to [0,1] (IMPSY units). */
function clampPrediction(out: number[]): number[] {
  out[0] = Math.max(out[0], MINIMUM_DELTA_TIME); // dim 0 = dt, must be positive
  for (let i = 1; i < out.length; i++) {
    out[i] = Math.min(Math.max(out[i], 0), 1);
  }
  return out;
}

export function softmaxWithTemperature(logits: number[], temperature: number): number[] {
  const safeTemp = Math.max(temperature, 1e-6);
  const scaled = logits.map((x) => x / safeTemp);
  const maxVal = scaled.reduce((a, b) => Math.max(a, b), -Infinity);
  const exps = scaled.map((x) => Math.exp(x - maxVal));
  const sum = exps.reduce((a, b) => a + b, 0);
  if (!(sum > 0)) {
    return logits.map(() => 1 / logits.length); // uniform fallback
  }
  return exps.map((e) => e / sum);
}

/** The first index whose cumulative probability reaches a uniform draw. */
export function sampleCategorical(probs: number[], rand = Math.random): number {
  const u = rand();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i];
    if (cumulative >= u) return i;
  }
  return probs.length - 1;
}

/** Standard normal N(0,1) via the Box-Muller transform. */
export function standardNormal(rand = Math.random): number {
  const u1 = Math.max(rand(), Number.EPSILON); // avoid log(0)
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
