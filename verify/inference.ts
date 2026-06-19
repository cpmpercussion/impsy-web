// Headless verification: load the demo model in a real browser engine and run
// the RNN inference loop. Driven by scripts/verify-inference.mjs (Playwright).
// Exposes the result on window.__impsy_result for the driver to read.

import { initLiteRt, TFLiteRNN } from "../src/lib/impsy/tfliteRnn";

declare global {
  interface Window {
    __impsy_result?: unknown;
  }
}

const out = document.getElementById("out")!;

async function main() {
  await initLiteRt("/litert-wasm/");
  const res = await fetch("/models/musicMDRNN-dim9-layers2-units64-mixtures5-scale10.tflite");
  const bytes = new Uint8Array(await res.arrayBuffer());

  const rnn = await TFLiteRNN.create(bytes, "wasm");
  const { dimension, numLayers, hiddenUnits, numMixtures } = rnn.config;

  // Run a short response chain, feeding each prediction back as the next seed.
  let seed = Array.from({ length: dimension }, (_, i) => (i === 0 ? 0.01 : Math.random()));
  const steps: number[][] = [];
  for (let i = 0; i < 5; i++) {
    seed = rnn.generate(seed, 1.0, 0.01);
    steps.push(seed);
  }

  const allValid = steps.every(
    (s) =>
      s.length === dimension &&
      s[0] > 0 &&
      s.slice(1).every((v) => v >= 0 && v <= 1),
  );

  return {
    ok: allValid,
    config: { dimension, numLayers, hiddenUnits, numMixtures },
    firstStep: steps[0].map((v) => Number(v.toFixed(4))),
  };
}

main()
  .then((r) => {
    window.__impsy_result = r;
    out.textContent = JSON.stringify(r, null, 2);
  })
  .catch((e) => {
    window.__impsy_result = { ok: false, error: String(e?.stack ?? e) };
    out.textContent = "ERROR: " + String(e?.stack ?? e);
  });
