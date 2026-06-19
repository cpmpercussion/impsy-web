# IMPSY Web

A browser-based version of [IMPSY](https://github.com/cpmpercussion/impsy) — it
runs IMPSY MDRNN models (`.tflite`) **client-side in the web browser** and drives
local MIDI software through the **Web MIDI API**. It is the third IMPSY frontend,
alongside the canonical [Python application](https://github.com/cpmpercussion/impsy)
and the [AUv3 plugin](../impsy-auv3) (iOS/macOS).

IMPSY learns and predicts musical gestures with an LSTM + Mixture Density Network.
You play, it listens; you pause, it responds — in your browser, into your DAW.

## Status

Early but **working end-to-end**: a real IMPSY model loads via LiteRT.js, runs the
stateful call-and-response loop, and sends MIDI out (verified live into Ableton via
an IAC bus). See [`FEASIBILITY.md`](./FEASIBILITY.md) for the feasibility study.

**Done**
- LiteRT.js (`@litertjs/core`) inference of IMPSY `.tflite` models, with LSTM state fed back each step
- Faithful ports of the AUv3 core: MDN sampler, MIDI mapper, call/response interaction engine
- Web MIDI input/output with device pickers
- Parameter controls (threshold, sigma/pi temp, timescale, MIDI thru)
- Per-dimension mapping editor (input + output)
- Dimension faders that double as **direct-input levers** (drag → red, injects MIDI) and **model-output indicators** (idle → green, follows the RNN)
- Unit tests for the ported math/MIDI logic; headless-Chrome verification of inference, the fader loop, and MIDI input

**Not yet**
- MIDI Learn (capture a control into a mapping by wiggling it)
- Inference in a Web Worker (currently main thread; fine at these model sizes)
- Activity dashboard strip (CALL/RESPONSE + LEDs + last-event), interaction logging
- Safari support (no Web MIDI there); model export guaranteed Flex-op-free

## Quick start

```bash
npm install
npm run dev            # http://localhost:5173
```

In the browser: **Enable Web MIDI** → **Load demo model** → pick a MIDI **Output**
(e.g. an IAC bus into your DAW). Drag the dimension faders to play into IMPSY; pause
and it responds. To light the faders from an external controller, select its **Input**
device and set the **Input** mappings to match what it sends.

> Browser support: Web MIDI requires Chrome, Edge, or Firefox (not Safari).
> Multi-threaded WASM needs cross-origin isolation (COOP/COEP) — set for dev in
> `vite.config.ts`; production hosting must send the same headers.

## Commands

```bash
npm run dev          # dev server (predev copies the LiteRT WASM into public/)
npm run build        # type-check + production build
npm run check        # svelte-check only
npm test             # vitest unit tests

# Headless-Chrome checks (need `npm run dev` running; pass the port if not 5173):
node scripts/verify-inference.mjs        # load a model + run the RNN
node scripts/verify-faders.mjs           # drag a fader → model responds
node scripts/verify-midi-input.mjs 5173  # mock controller lights the faders
```

## Architecture

Stack: **Svelte 5 + Vite + TypeScript**, **LiteRT.js** for inference. Everything runs
on the browser main thread (Web MIDI callbacks and timers share it).

```
Web MIDI in → InteractionEngine (10 ms tick): decode → inputVector
   → CALL: RNN.generate (advance LSTM state)
   → RESPONSE: self-feeding chain, each prediction scheduling the next by its dt
   → MDNSampler → MIDIMapper.encode → Web MIDI out
```

Most of `src/lib/impsy/` is a direct port of `../impsy-auv3/IMPSYExtension/Common/*.swift`,
which in turn mirrors the canonical Python in `../impsy`. The Python project is the
source of truth for behaviour. See [`CLAUDE.md`](./CLAUDE.md) for the full module map,
the model interface contract, and conventions.

## Related

- [cpmpercussion/impsy](https://github.com/cpmpercussion/impsy) — canonical Python implementation
- IMPSY AUv3 — iOS/macOS plugin (UI/UX reference for this port)
