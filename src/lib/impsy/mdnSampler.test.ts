import { describe, it, expect } from "vitest";
import {
  softmaxWithTemperature,
  sampleCategorical,
  postProcess,
  sampleMDN,
} from "./mdnSampler";
import { SCALE_FACTOR, MINIMUM_DELTA_TIME } from "./constants";

describe("softmaxWithTemperature", () => {
  it("produces a probability distribution that sums to 1", () => {
    const p = softmaxWithTemperature([1, 2, 3], 1);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(p[2]).toBeGreaterThan(p[0]); // larger logit → larger prob
  });

  it("is numerically stable for large logits", () => {
    const p = softmaxWithTemperature([1000, 1001, 1002], 1);
    expect(p.every((x) => Number.isFinite(x))).toBe(true);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
  });

  it("low temperature sharpens toward the argmax", () => {
    const p = softmaxWithTemperature([1, 2, 3], 0.01);
    expect(p[2]).toBeGreaterThan(0.99);
  });
});

describe("sampleCategorical", () => {
  it("returns the bucket the random draw falls into", () => {
    const probs = [0.2, 0.5, 0.3];
    expect(sampleCategorical(probs, () => 0.1)).toBe(0);
    expect(sampleCategorical(probs, () => 0.5)).toBe(1);
    expect(sampleCategorical(probs, () => 0.95)).toBe(2);
  });
});

describe("postProcess", () => {
  it("divides by SCALE_FACTOR and clamps values 1…N to [0,1]", () => {
    const out = postProcess([0.5, 20, -5, 5]); // raw (×10) space
    expect(out[0]).toBeCloseTo(0.5 / SCALE_FACTOR);
    expect(out[1]).toBe(1); // 20/10 = 2 → clamp 1
    expect(out[2]).toBe(0); // -5/10 → clamp 0
    expect(out[3]).toBe(0.5); // 5/10
  });

  it("enforces a minimum dt on dimension 0", () => {
    const out = postProcess([0, 5, 5]);
    expect(out[0]).toBe(MINIMUM_DELTA_TIME);
  });
});

describe("sampleMDN", () => {
  it("returns a vector of the requested dimension", () => {
    const D = 4;
    const M = 5;
    // params length = M*(2D+1) = 5*9 = 45
    const params = new Float32Array(M * (2 * D + 1)).fill(1);
    const out = sampleMDN(params, D, M, 1.0, 0.01);
    expect(out).toHaveLength(D);
    expect(out[0]).toBeGreaterThanOrEqual(MINIMUM_DELTA_TIME);
    for (let i = 1; i < D; i++) {
      expect(out[i]).toBeGreaterThanOrEqual(0);
      expect(out[i]).toBeLessThanOrEqual(1);
    }
  });

  it("returns zeros when params are too short", () => {
    expect(sampleMDN(new Float32Array(3), 4, 5, 1, 0.01)).toEqual([0, 0, 0, 0]);
  });
});
