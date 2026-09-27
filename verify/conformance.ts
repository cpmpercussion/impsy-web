// Runs spec/vectors/model.json cases through TFLiteRNN in a real browser
// engine (LiteRT.js needs one). Driven by e2e/conformance.spec.ts, which
// compares the result with the case's `expected`. Dev server only: the model
// is fetched from spec/models/ at the project root.

import { initLiteRt, TFLiteRNN } from "../src/lib/impsy/tfliteRnn";
import { mixtureParams, sampleCategorical } from "../src/lib/impsy/mdnSampler";
import { SCALE_FACTOR } from "../src/lib/impsy/constants";

interface ModelCase {
  model_file: string;
  pi_temp: number;
  sigma_temp: number;
  uniform_draws: number[];
  steps: ({ input: number[] } | { reset: true })[];
}

declare global {
  interface Window {
    __impsyModelCase?: (c: ModelCase) => Promise<unknown>;
  }
}

async function runModelCase(c: ModelCase) {
  const bytes = new Uint8Array(await (await fetch(`/spec/${c.model_file}`)).arrayBuffer());
  const rnn = await TFLiteRNN.create(bytes, "wasm");
  const { dimension, numLayers, hiddenUnits, numMixtures } = rnn.config;
  try {
    const steps = c.steps.map((step) => {
      if ("reset" in step) {
        rnn.resetStates();
        return null;
      }
      const mdn = rnn.step(step.input);
      const mix = mixtureParams(mdn, dimension, numMixtures, c.pi_temp, c.sigma_temp)!;
      return {
        model_input: Array.from(Float32Array.from(step.input, (v) => v * SCALE_FACTOR)),
        mdn_output: Array.from(mdn),
        pi: mix.pi,
        mu: mix.mu,
        std: mix.std,
        mixture_for_draw: c.uniform_draws.map((u) => sampleCategorical(mix.pi, () => u)),
      };
    });
    const names = rnn.tensorNames;
    return {
      introspected: { dimension, units: hiddenUnits, mixtures: numMixtures, layers: numLayers },
      tensors: {
        inputs: names.inputs,
        mdn_output: names.mdnOutput,
        state_outputs: names.stateOutputs,
      },
      steps,
    };
  } finally {
    rnn.dispose();
  }
}

initLiteRt("/litert-wasm/").then(
  () => {
    window.__impsyModelCase = runModelCase;
    document.getElementById("out")!.textContent = "ready";
  },
  (e) => {
    document.getElementById("out")!.textContent = "ERROR: " + String(e?.stack ?? e);
  },
);
