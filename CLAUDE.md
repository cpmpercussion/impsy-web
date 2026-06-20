# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`impsy-web` is a **browser-based port of IMPSY** — it runs IMPSY `.tflite` MDRNN models client-side via **LiteRT.js** (`@litertjs/core`) and drives local MIDI software through the **Web MIDI API**. It is the third IMPSY frontend, after the canonical Python app and the AUv3 plugin.

Stack: **Svelte 5 (runes) + Vite + TypeScript**, with LiteRT.js for inference. See `FEASIBILITY.md` for the (proven) feasibility study and `impsy-web-plan.md` for the original brief.

## Sibling projects (the references)

Assumed checked out as siblings:
- **`../impsy/`** — canonical Python implementation. **Source of truth for behaviour.** When something is ambiguous, match what the Python does (`impsy/mdrnn.py`, `impsy/interaction.py`).
- **`../impsy-auv3/`** — Swift/AUv3 plugin. **The UI/UX and feature target.** Most of `src/lib/impsy/` is a direct port of its `IMPSYExtension/Common/*.swift` — keep the correspondence when changing logic.

## Commands

```bash
npm install
npm run dev      # vite dev server on :5173 (predev copies LiteRT WASM to public/)
npm run build    # svelte-check + vite build
npm run check    # svelte-check type check only
npm test         # vitest (unit tests for the ported logic)
npm run test:watch

# End-to-end inference check in real Chrome (needs `npm run dev` running):
node scripts/verify-inference.mjs
```

`npm test` runs a single file via `npx vitest run src/lib/impsy/midiMapper.test.ts`.

## Architecture

Data flow mirrors AUv3, collapsed onto the browser's single main thread (Web MIDI callbacks and timers share it, so no ring buffers / locks are needed):

```
Web MIDI in  → InteractionEngine.enqueueInput
   → 10ms tick: drain → MIDIMapper.decode → inputVector
   → CALL: RNN.generate (state only) ;  RESPONSE: self-feeding chain
   → MDNSampler → MIDIMapper.encode → WebMIDI.send → Web MIDI out
```

### Module map (`src/lib/`)

| File | Ported from (AUv3) | Responsibility |
|------|--------------------|----------------|
| `impsy/constants.ts` | `IMPSYParameters.swift` | SCALE_FACTOR, min dt, param defaults/ranges |
| `impsy/mdnSampler.ts` | `MDNSampler.swift` | softmax-temp → categorical → Box-Muller; postProcess (÷10, clamp, min dt) |
| `impsy/midiMapping.ts` | `MIDIMapping.swift` | `DimensionMapping` / `MIDIMappingSet`, defaults, AiC preset |
| `impsy/midiMapper.ts` | `MIDIMapper.swift` | MIDI bytes ↔ normalised [0,1]; monophonic note_off; dedup window |
| `impsy/config.ts` | `IMPSYConfig.swift` + `…+TOML.swift` | IMPSY `.toml` parse/serialize (params + mappings); round-trips with Python/AUv3 via `smol-toml`; preserves unknown sections |
| `impsy/tfliteRnn.ts` | `TFLiteRNN.swift` + `ModelInspector.swift` | LiteRT wrapper; introspects config; holds LSTM state across `generate()` |
| `impsy/interactionEngine.ts` | `InteractionEngine.swift` | call/response state machine + self-feeding response loop |
| `midi/webMidi.ts` | `CoreMIDIBridge` | Web MIDI access, device lists, send/receive |
| `appState.svelte.ts` | `IMPSYViewModel.swift` | Svelte-runes orchestration; single shared `app` instance |
| `components/*.svelte` | `IMPSYUI/*` | MIDI connection, model status, parameters, mapping editor |

### Model interface contract (must stay exact)

TFLite models follow the convention in `../impsy/impsy/mdrnn.py` (`TfliteMDRNN`):
- **Inputs**: `inputs (1,1,dimension)` + paired `state_h_N`/`state_c_N (1,hiddenUnits)`, ordered `[inputs, h0, c0, h1, c1, …]`.
- **Outputs**: MDN `(1, numMixtures·(2·dimension+1))` + updated states.
- **MDN layout**: `[mus: M×D | sigmas: M×D | piLogits: M]`. M is always 5.
- **Scaling**: ×`SCALE_FACTOR` (10) into the model, ÷10 out. Dim 0 = time delta (s); 1…N normalised [0,1].
- **Init parity**: LSTM states zeroed; first interaction is a `randomInitialSample` (dt≈0.01 + random [0,1)).

`tfliteRnn.ts` resolves input/output tensors by shape+index (3-D tensor = `inputs`, 2-D = states in index order; MDN output = the 2-D tensor whose width matches `M·(2D+1)`) — robust to the generic tensor names TFLite assigns after conversion.

**Flex ops caveat**: some `../impsy/models/*.tflite` need the Select-TF (Flex) delegate, which browser runtimes lack — they fail to load. Only Flex-free models work (see `FEASIBILITY.md`). The bundled demo (`public/models/musicMDRNN-dim9-...tflite`) is Flex-free.

### Parameters & MIDI conventions (align with AUv3)

Four params + a toggle, defaults from `../impsy/configs/AiC-charles-u6midipro.toml`: **Threshold** 0.1s, **Sigma Temp** 0.01, **Pi Temp** 1.0, **Timescale** 1.0×, **MIDI Thru** on. MIDI ↔ [0,1]: Note On `vel/127`; CC `value/127` (through the mapping's min/max); Pitch Bend `(raw+8192)/16383`. Dimension IDs are 1-based (dim 0 = time, not mappable); input and output mappings are independent.

## Conventions & gotchas

- **WASM serving**: LiteRT's WASM runtime is copied from `node_modules/@litertjs/core/wasm` to `public/litert-wasm/` by `scripts/sync-wasm.mjs` (auto-run via `predev`/`prebuild`). `public/litert-wasm/` is gitignored. Production hosting must serve **COOP/COEP** headers (set for dev in `vite.config.ts`) or multi-threaded WASM falls back to single-threaded.
- **LiteRT Tensor lifecycle**: tensors are WASM-backed and must be `.delete()`d. `tfliteRnn.ts` deletes every input/output tensor each `generate()` — don't drop that or memory leaks.
- **Literal-type trap**: `ParameterDefaults` is `as const`, so fields initialised from it infer literal types (`0.1`, not `number`). Annotate such fields/`Params` explicitly (already done in `interactionEngine.ts` and `appState.svelte.ts`).
- **Parity testing**: `src/lib/impsy/*.test.ts` pin the ported math/MIDI logic. When porting more from AUv3, add a test asserting the same numbers the Swift produces.
