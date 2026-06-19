# TFLite-in-browser feasibility — findings

**Verdict: feasible, and proven end-to-end.** A real IMPSY MDRNN `.tflite` model
loads and runs in the browser via LiteRT.js, with the stateful LSTM loop working.

## What was verified

1. **Model op-set.** Inspected every `.tflite` in `../impsy/models/` with the
   builtin-only (no Flex delegate) interpreter — the browser-equivalent op set.
   - **Builtin-compatible** (run in-browser as-is): dim2-u64, dim7, dim9, dim10,
     dim13, dim17, dim25 (all `layers2`, `units64/128`).
   - **Require Flex / Select-TF ops** (will NOT run in-browser): `dim2-units32`,
     `dim4-units64`, `dim9-...-optimised`. These were exported with a converter
     path that emits `FlexTensorListReserve` (dynamic TensorList). Browser TFLite
     runtimes have no Flex delegate. Fix = re-export through IMPSY's builtin-LSTM
     converter (the working models prove it produces Flex-free output).

2. **Tensor contract.** Confirmed against `../impsy/impsy/mdrnn.py` (`TfliteMDRNN`):
   inputs `inputs (1,1,D)` + `state_h_N` / `state_c_N (1,units)`; outputs MDN
   `(1, M·(2D+1))` + updated states. Input order `[inputs, h0, c0, h1, c1, …]`.

3. **Runtime.** `@litertjs/core` v0.1.1. Its `run(Record<string,Tensor>)` →
   `Record<string,Tensor>` API handles multiple named inputs/outputs — exactly
   what the LSTM state feedback needs. WASM/XNNPACK CPU backend; WebGPU optional
   and unnecessary at these model sizes.

4. **Live inference (real Chrome, headless via Playwright).** Loaded the bundled
   `dim9-units64` model: introspected `{dimension:9, numLayers:2, hiddenUnits:64,
   numMixtures:5}` and ran a 5-step self-feeding loop producing valid vectors
   (dt > 0, values ∈ [0,1]). Reproduce: `npm run dev`, then
   `node scripts/verify-inference.mjs`.

## Constraints / open items

- **Web MIDI** is Chromium + Firefox only; **Safari has no Web MIDI** — accept as
  a target gap or add a fallback (virtual keyboard / WebRTC) later.
- **Cross-origin isolation** (COOP/COEP headers) is required for multi-threaded
  WASM — set in `vite.config.ts` for dev; production hosting must send the same.
- During verification one resource 404'd (a WASM feature-probe variant); the
  runtime fell back successfully. Worth confirming the threaded build loads in
  production hosting rather than silently using the single-threaded fallback.
- **Model export**: ship only Flex-free models, or add a re-conversion step in
  `../impsy` so all exported models are browser-compatible.
