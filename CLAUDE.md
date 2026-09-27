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
npm run format   # Prettier (CI runs format:check — run this before committing)
npm run test:watch

npm run test:e2e # Playwright browser tests (e2e/), fake Web MIDI; starts its own servers
                 # --project=preview (prod build under /impsy-web/) | --project=dev

# Diagnostic inference printout in real Chrome (needs `npm run dev` running):
node scripts/verify-inference.mjs
```

Run a single test file with `npx vitest run src/lib/impsy/midiMapper.test.ts`.

CI (`.github/workflows/ci.yml`) runs `npm run format:check` + `npm test` + `npm run build` + `npm run test:e2e` on PRs and `main`; `deploy.yml` publishes `main` to GitHub Pages after the unit tests, then runs `scripts/smoke-check.mjs` against the live site. Releases are semver tags that must match `package.json` (`release.yml` creates the GitHub Release) — see `RELEASING.md`. The app footer shows `__APP_VERSION__` / `__GIT_HASH__`, injected by `vite.config.ts` `define`.

## Architecture

Data flow mirrors AUv3, collapsed onto the browser's single main thread (Web MIDI callbacks and timers share it, so no ring buffers / locks are needed):

```
Web MIDI in  → InteractionEngine.enqueueInput
   → 10ms tick: drain → MIDIMapper.decode → inputVector
   → CALL: RNN.generate (state only) ;  RESPONSE: self-feeding chain
   → MDNSampler → MIDIMapper.encode → WebMIDI.send → Web MIDI out
```

### Module map (`src/lib/`)

| File                         | Ported from (AUv3)                         | Responsibility                                                                                                              |
| ---------------------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `impsy/constants.ts`         | `IMPSYParameters.swift`                    | SCALE_FACTOR, min dt, param defaults/ranges                                                                                 |
| `impsy/mdnSampler.ts`        | `MDNSampler.swift`                         | softmax-temp → categorical → Box-Muller; postProcess (÷10, clamp, min dt)                                                   |
| `impsy/midiMapping.ts`       | `MIDIMapping.swift`                        | `DimensionMapping` / `MIDIMappingSet`, defaults, AiC preset                                                                 |
| `impsy/midiMapper.ts`        | `MIDIMapper.swift`                         | MIDI bytes ↔ normalised [0,1]; per-dimension note-offs; resend CC/pitch bend only on change                                 |
| `impsy/config.ts`            | `IMPSYConfig.swift` + `…+TOML.swift`       | IMPSY `.toml` parse/serialize (params + mappings); round-trips with Python/AUv3 via `smol-toml`; preserves unknown sections |
| `impsy/tfliteRnn.ts`         | `TFLiteRNN.swift` + `ModelInspector.swift` | LiteRT wrapper; introspects config; holds LSTM state across `generate()`                                                    |
| `impsy/interactionEngine.ts` | `InteractionEngine.swift`                  | call/response state machine + self-feeding response loop                                                                    |
| `midi/webMidi.ts`            | `CoreMIDIBridge`                           | Web MIDI access, device lists, send/receive                                                                                 |
| `midi/midiLog.ts`            | `MIDIEvent.summary`                        | MIDI message descriptions + bounded log for the live console / Last Output card                                             |
| `appState.svelte.ts`         | `IMPSYViewModel.swift`                     | Svelte-runes orchestration; single shared `app` instance                                                                    |
| `components/*.svelte`        | `IMPSYUI/*`                                | MIDI connection, model status, parameters, mapping editor                                                                   |

### Model interface contract (must stay exact)

TFLite models follow the convention in `../impsy/impsy/mdrnn.py` (`TfliteMDRNN`):

- **Inputs**: `inputs (1,1,dimension)` + paired `state_h_N`/`state_c_N (1,hiddenUnits)`, ordered `[inputs, h0, c0, h1, c1, …]`.
- **Outputs**: MDN `(1, numMixtures·(2·dimension+1))` + updated states.
- **MDN layout**: `[mus: M×D | sigmas: M×D | piLogits: M]`. M is always 5.
- **Scaling**: ×`SCALE_FACTOR` (10) into the model, ÷10 out. Dim 0 = time delta (s); 1…N normalised [0,1].
- **Init parity**: LSTM states zeroed; first interaction is a `randomInitialSample` (dt≈0.01 + random [0,1)).

`tfliteRnn.ts` resolves input/output tensors by shape (3-D tensor = `inputs`; MDN output = the 2-D tensor whose width matches `M·(2D+1)`). State inputs are ordered `[h0, c0, h1, c1, …]` **by name** when they're named `state_{h,c}_N` — LiteRT lists signature inputs alphabetically (`state_c_0, state_c_1, state_h_0, …`), so index order would cross h and c — else by index; state outputs pair up in index order. `spec/vectors/model.json` pins this.

**Flex ops caveat**: some `../impsy/models/*.tflite` need the Select-TF (Flex) delegate, which browser runtimes lack — they fail to load. Only Flex-free models work (see `FEASIBILITY.md`). The bundled demo (`public/models/musicMDRNN-dim9-...tflite`) is Flex-free.

### Parameters & MIDI conventions (align with AUv3)

Four params + a toggle, defaults from `../impsy/configs/AiC-charles-u6midipro.toml`: **Threshold** 0.1s, **Sigma Temp** 0.01, **Pi Temp** 1.0, **Timescale** 1.0×, **MIDI Thru** on. Dimension IDs are 1-based (dim 0 = time, not mappable); input and output mappings are independent.

MIDI semantics follow IMPSY Python (`midi_message_to_updates` / `MidiOutputState` in `../impsy/impsy/utils.py`) and are pinned by the conformance vectors:

- **Input**: Note On sets `noteOn` dims on its channel to `note/127` and `noteVelocity` dims to `velocity/127` (velocity-0 = note-off, ignored); CC `(raw−min)/(max−min)` through the mapping's range; Pitch Bend `raw14/16383`. One message updates _every_ matching dimension, and is one interaction (one model input, one log row). dt is the unclamped time since the previous interaction.
- **Output**: values clipped to [0,1], rounded to nearest. Notes tracked per dimension (note-off of its previous note unless another dim on the channel holds it), velocity from the channel's first `noteVelocity` dim, else the fixed `["note_on", ch, vel]`, else 127. CCs/pitch bends only sent when their MIDI value changes; `releaseAllNotes` forgets them.
- **Playback**: `prepareRNNPlayback` — wait `max(dt, 1ms) × timescale`, but feed the model back the _unscaled_ dt.

## Conventions & gotchas

- **WASM serving**: LiteRT's WASM runtime is copied from `node_modules/@litertjs/core/wasm` to `public/litert-wasm/` by `scripts/sync-wasm.mjs` (auto-run via `predev`/`prebuild`). `public/litert-wasm/` is gitignored. Production hosting must serve **COOP/COEP** headers (set for dev in `vite.config.ts`) or multi-threaded WASM falls back to single-threaded.
- **LiteRT Tensor lifecycle**: tensors are WASM-backed and must be `.delete()`d. `tfliteRnn.ts` deletes every input/output tensor each `generate()` — don't drop that or memory leaks.
- **Literal-type trap**: `ParameterDefaults` is `as const`, so fields initialised from it infer literal types (`0.1`, not `number`). Annotate such fields/`Params` explicitly (already done in `interactionEngine.ts` and `appState.svelte.ts`).
- **Conformance vectors**: `spec/` is vendored verbatim from a tagged `cpmpercussion/impsy` release (`SPEC_TAG` / `SPEC_VERSION` in `src/lib/impsy/conformance.test.ts`; see `spec/README.md`). Never edit it (it's in `.prettierignore`); to update, `git archive <tag> spec | tar -x` from `../impsy` and bump the constants. `conformance.test.ts` runs the MIDI/playback/pipeline vectors; `model.json` needs LiteRT in a browser, so it runs via `verify/conformance.html` in `e2e/conformance.spec.ts` (dev project only). Deliberate deviations go in `SKIPPED_CASES` with a reason; WebSocket and dataset vectors are skipped (no such IO here).
- **Parity testing**: `src/lib/impsy/*.test.ts` pin the ported math/MIDI logic. When porting more from AUv3, add a test asserting the same numbers the Swift produces — but where Python and AUv3 disagree, the spec/Python wins.
- **Browser tests**: `e2e/midi.ts` installs a fake Web MIDI input/output (`midi.fire(...)`, `midi.sent()`). Faders show user input for only ~250 ms before settling back to the model value, so read them with `faderAfter()` (fire + read in one step) and check "ignored" messages with `expectIgnored()`. Set `PW_CHROMIUM_PATH` to use a preinstalled Chromium.
