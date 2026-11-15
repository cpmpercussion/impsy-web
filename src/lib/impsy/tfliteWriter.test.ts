import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { SCALE_FACTOR } from "./constants";
import {
  mdrnnStep,
  weightsFromKerasList,
  zeroStates,
  SIGMA_OFFSET,
  type MDRNNWeights,
} from "./mdrnnWeights";
import { parseTflite, readTfliteWeights } from "./tfliteReader";
import { BuiltinOperator } from "./tfliteSchema";
import type { ModelConfig } from "./tfliteRnn";
import { writeTflite } from "./tfliteWriter";

// The writer has no AUv3/Python counterpart to port numbers from, so it is
// pinned three ways:
//   1. the reader + plain-TS forward pass reproduce the Python reference
//      outputs in spec/vectors/model.json from the converter-made spec model,
//      which proves the weight layout (gate order, kernel transposes);
//   2. write → read round-trips weights bit-for-bit, for the spec model, the
//      bundled demo and random models of other sizes;
//   3. the written file's interface matches the model contract.
// That the written graph *runs* identically in LiteRT is checked in the
// browser by e2e/conformance.spec.ts.

const file = (path: string) =>
  new Uint8Array(readFileSync(new URL(`../../../${path}`, import.meta.url)));

const SPEC_MODEL = "spec/models/conformance-dim3-layers2-units16-mixtures5.tflite";
const DEMO_MODEL = "public/models/musicMDRNN-dim9-layers2-units64-mixtures5-scale10.tflite";

interface ModelVectors {
  tolerance: number;
  cases: {
    steps: ({ input: number[] } | { reset: true })[];
    expected: { steps: ({ mdn_output: number[] } | null)[] };
  }[];
}
const vectors: ModelVectors = JSON.parse(
  readFileSync(new URL("../../../spec/vectors/model.json", import.meta.url), "utf8"),
);

/** Deterministic pseudo-random weights (mulberry32), roughly trained-model sized. */
function randomWeights(config: ModelConfig, seed = 1): MDRNNWeights {
  let a = seed >>> 0;
  const random = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const array = (n: number) => Float32Array.from({ length: n }, () => random() - 0.5);
  const { dimension: D, numLayers, hiddenUnits: H, numMixtures: M } = config;
  const arrays: Float32Array[] = [];
  for (let l = 0; l < numLayers; l++) {
    arrays.push(array((l === 0 ? D : H) * 4 * H), array(H * 4 * H), array(4 * H));
  }
  for (const outputs of [M * D, M * D, M]) arrays.push(array(H * outputs), array(outputs));
  return weightsFromKerasList(config, arrays);
}

describe("readTfliteWeights", () => {
  it("introspects the spec model", () => {
    expect(readTfliteWeights(file(SPEC_MODEL)).config).toEqual({
      dimension: 3,
      numLayers: 2,
      hiddenUnits: 16,
      numMixtures: 5,
    });
  });

  it("introspects the bundled demo model", () => {
    expect(readTfliteWeights(file(DEMO_MODEL)).config).toEqual({
      dimension: 9,
      numLayers: 2,
      hiddenUnits: 64,
      numMixtures: 5,
    });
  });

  it("recovers weights that reproduce the Python reference outputs (model.json)", () => {
    const weights = readTfliteWeights(file(SPEC_MODEL));
    for (const c of vectors.cases) {
      let states = zeroStates(weights.config);
      c.steps.forEach((step, i) => {
        if ("reset" in step) {
          states = zeroStates(weights.config);
          return;
        }
        const out = mdrnnStep(
          weights,
          step.input.map((v) => v * SCALE_FACTOR),
          states,
        );
        states = out.states;
        const expected = (c.expected.steps[i] as { mdn_output: number[] }).mdn_output;
        expect(out.mdn.length).toBe(expected.length);
        // Tighter than the file's cross-CPU tolerance: this is the same float
        // arithmetic, so only rounding order differs.
        expected.forEach((e, j) => expect(Math.abs(out.mdn[j] - e)).toBeLessThan(1e-4));
      });
    }
  });

  it("uses the sigma offset the converter folds into the graph", () => {
    const main = parseTflite(file(SPEC_MODEL)).subgraphs[0];
    const scalars = main.tensors.filter((t) => t.data && t.type === 0 && t.shape.length === 0);
    expect(scalars.map((t) => new Float32Array(t.data!.slice().buffer)[0])).toEqual([SIGMA_OFFSET]);
  });

  it("rejects files that are not TFLite models", () => {
    expect(() => readTfliteWeights(new Uint8Array(64))).toThrow(/Not a TFLite model/);
  });
});

describe("writeTflite", () => {
  it.each([SPEC_MODEL, DEMO_MODEL])("round-trips the weights of %s bit-for-bit", (path) => {
    const weights = readTfliteWeights(file(path));
    expect(readTfliteWeights(writeTflite(weights))).toEqual(weights);
  });

  it.each<ModelConfig>([
    { dimension: 2, numLayers: 1, hiddenUnits: 8, numMixtures: 5 },
    { dimension: 4, numLayers: 3, hiddenUnits: 12, numMixtures: 5 },
    // Layer 0's kernel and recurrent kernel have the same shape when dimension = units.
    { dimension: 6, numLayers: 2, hiddenUnits: 6, numMixtures: 3 },
  ])("round-trips random weights for %o", (config) => {
    const weights = randomWeights(config);
    expect(readTfliteWeights(writeTflite(weights))).toEqual(weights);
  });

  it("writes the model interface contract", () => {
    const config = { dimension: 3, numLayers: 2, hiddenUnits: 16, numMixtures: 5 };
    const bytes = writeTflite(randomWeights(config));
    expect(String.fromCharCode(...bytes.subarray(4, 8))).toBe("TFL3");

    const model = parseTflite(bytes);
    expect(model.version).toBe(3);
    expect(model.subgraphs).toHaveLength(1);
    const main = model.subgraphs[0];
    const describe = (i: number) => ({ name: main.tensors[i].name, shape: main.tensors[i].shape });

    // Inputs in feed order, at ascending tensor indices.
    expect(main.inputs).toEqual([0, 1, 2, 3, 4]);
    expect(main.inputs.map(describe)).toEqual([
      { name: "inputs", shape: [1, 1, 3] },
      { name: "state_h_0", shape: [1, 16] },
      { name: "state_c_0", shape: [1, 16] },
      { name: "state_h_1", shape: [1, 16] },
      { name: "state_c_1", shape: [1, 16] },
    ]);
    // MDN output first, then states [h0, c0, h1, c1] at ascending tensor
    // indices (readers pair state outputs with inputs by index).
    expect(main.outputs.map(describe)).toEqual([
      { name: "Identity", shape: [1, 35] },
      { name: "Identity_1", shape: [1, 16] },
      { name: "Identity_2", shape: [1, 16] },
      { name: "Identity_3", shape: [1, 16] },
      { name: "Identity_4", shape: [1, 16] },
    ]);
    const states = main.outputs.slice(1);
    expect([...states].sort((a, b) => a - b)).toEqual(states);

    // Plain builtin ops only: no control flow, nothing needing the Flex delegate.
    const allowed = new Set<number>(Object.values(BuiltinOperator));
    allowed.delete(BuiltinOperator.WHILE);
    for (const op of main.operators) expect(allowed.has(op.code)).toBe(true);
  });

  it("aligns constant data to 16 bytes", () => {
    const bytes = writeTflite(
      randomWeights({ dimension: 3, numLayers: 1, hiddenUnits: 4, numMixtures: 5 }),
    );
    for (const t of parseTflite(bytes).subgraphs[0].tensors) {
      if (t.data) expect((t.data.byteOffset - bytes.byteOffset) % 16).toBe(0);
    }
  });

  it("rejects weights that do not match their config", () => {
    const weights = randomWeights({ dimension: 3, numLayers: 1, hiddenUnits: 4, numMixtures: 5 });
    weights.lstm[0].bias = new Float32Array(3);
    expect(() => writeTflite(weights)).toThrow(/lstm 0 bias/);
  });
});
