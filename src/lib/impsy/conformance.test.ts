// IMPSY conformance vectors (spec/README.md), run against the web port.
//
// spec/ is vendored from cpmpercussion/impsy at SPEC_TAG. Each vector file
// records what the Python reference does; these tests feed every case through
// the same TypeScript code the app uses and compare with `expected`. Don't edit
// the vectors: to deviate from a case, skip it by name in SKIPPED_CASES with a
// reason. model.json needs LiteRT.js in a browser, so it runs under Playwright
// (e2e/conformance.spec.ts) instead.

import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { MIDIMapper, type MIDIEvent } from "./midiMapper";
import type { DimensionMapping, MIDIMappingSet } from "./midiMapping";
import { parseMappingEntry } from "./config";
import { UserInput, prepareRNNPlayback } from "./interactionEngine";

export const SPEC_VERSION = "1.0.0";
export const SPEC_TAG = "v1.2.1";

interface VectorFile<C> {
  spec_version: string;
  tolerance: number;
  cases: (C & { name: string; description: string })[];
}

function load<C>(file: string): VectorFile<C> {
  const url = new URL(`../../../spec/vectors/${file}`, import.meta.url);
  return JSON.parse(readFileSync(url, "utf8"));
}

/** Compare numbers within `tol`, everything else exactly, recursively. */
function expectClose(actual: unknown, expected: unknown, tol: number, path = "$"): void {
  if (typeof expected === "number") {
    expect(typeof actual, path).toBe("number");
    expect(
      Math.abs((actual as number) - expected),
      `${path}: ${actual} vs ${expected}`,
    ).toBeLessThanOrEqual(tol);
  } else if (Array.isArray(expected)) {
    expect(Array.isArray(actual), `${path} is an array`).toBe(true);
    expect((actual as unknown[]).length, `${path}.length`).toBe(expected.length);
    expected.forEach((e, i) => expectClose((actual as unknown[])[i], e, tol, `${path}[${i}]`));
  } else if (expected !== null && typeof expected === "object") {
    for (const [k, e] of Object.entries(expected))
      expectClose((actual as Record<string, unknown>)[k], e, tol, `${path}.${k}`);
  } else {
    expect(actual, path).toEqual(expected);
  }
}

/** IMPSY config entries (e.g. `["control_change", 1, 42, 0, 63]`) → mappings. */
function mappings(entries: unknown[]): DimensionMapping[] {
  return entries.map((entry, i) => {
    const m = parseMappingEntry(entry, i + 1);
    if (!m) throw new Error(`unparseable mapping entry ${JSON.stringify(entry)}`);
    return m;
  });
}

const mapperFor = (set: Partial<MIDIMappingSet>) =>
  new MIDIMapper({ inputMappings: [], outputMappings: [], ...set });

const bytesOf = (events: MIDIEvent[]) => events.map((e) => e.bytes);

/**
 * Cases this port deliberately doesn't match, by file then case name, with the
 * reason. Empty means full conformance for that file.
 */
const SKIPPED_CASES: Record<string, Record<string, string>> = {};

/**
 * Vector files that cover IO this port doesn't have. Checked for
 * `spec_version` like the rest, so an update is still noticed.
 */
const SKIPPED_FILES: Record<string, string> = {
  "websocket_input.json": "no WebSocket IO in the browser port (MIDI only)",
  "websocket_output.json": "no WebSocket IO in the browser port (MIDI only)",
  "dataset.json": "training runs in Python; the web port only writes .log files",
};

function forEachCase<C>(file: string, run: (c: C, tol: number) => void): void {
  const doc = load<C>(file);
  describe(file, () => {
    it(`is spec_version ${SPEC_VERSION}`, () => expect(doc.spec_version).toBe(SPEC_VERSION));
    for (const c of doc.cases) {
      const skip = SKIPPED_CASES[file]?.[c.name];
      (skip ? it.skip : it)(skip ? `${c.name} — skipped: ${skip}` : c.name, () =>
        run(c, doc.tolerance),
      );
    }
  });
}

for (const [file, reason] of Object.entries(SKIPPED_FILES)) {
  const doc = load<unknown>(file);
  describe(file, () => {
    it(`is spec_version ${SPEC_VERSION}`, () => expect(doc.spec_version).toBe(SPEC_VERSION));
    it.skip(`${doc.cases.length} cases — skipped: ${reason}`, () => {});
  });
}

// ── midi_input.json ──────────────────────────────────────────────────────────

interface MidiInputCase {
  input_mapping: unknown[];
  messages: number[][];
  expected: ([number, number][] | null)[];
}

forEachCase<MidiInputCase>("midi_input.json", (c, tol) => {
  // Each message starts from a fresh mapper, as in the Python runner.
  const actual = c.messages.map((bytes) =>
    mapperFor({ inputMappings: mappings(c.input_mapping) }).decodeInput(bytes),
  );
  expectClose(actual, c.expected, tol);
});

// ── midi_output.json ─────────────────────────────────────────────────────────

interface MidiOutputCase {
  output_mapping: unknown[];
  steps: ({ values: number[] } | { all_notes_off: true })[];
  expected: number[][][];
}

forEachCase<MidiOutputCase>("midi_output.json", (c) => {
  // One mapper for the case: note and last-sent state carry across steps.
  const mapper = mapperFor({ outputMappings: mappings(c.output_mapping) });
  const actual = c.steps.map((step) =>
    bytesOf("values" in step ? mapper.encodeOutput(step.values) : mapper.releaseAllNotes()),
  );
  expect(actual).toEqual(c.expected);
});

// ── playback.json ────────────────────────────────────────────────────────────

interface PlaybackCase {
  timescale: number;
  model_outputs: number[][];
  expected: { wait: number; output: number[]; next_model_input: number[] }[];
}

forEachCase<PlaybackCase>("playback.json", (c, tol) => {
  const actual = c.model_outputs.map((out) => {
    const { wait, values, nextInput } = prepareRNNPlayback(out, c.timescale);
    return { wait, output: values, next_model_input: nextInput };
  });
  expectClose(actual, c.expected, tol);
});

// ── pipeline.json ────────────────────────────────────────────────────────────

interface PipelineCase {
  input_mapping: unknown[];
  initial_values: number[];
  start_time: number;
  events: { time: number; bytes: number[] }[];
  expected: {
    model_inputs: number[][];
    log: { source: string; values: number[] }[];
  };
}

forEachCase<PipelineCase>("pipeline.json", (c, tol) => {
  // The engine's per-message input path (InteractionEngine.tick): decode, merge
  // into the dense vector, log an `interface` row, queue the model input.
  const mapper = mapperFor({ inputMappings: mappings(c.input_mapping) });
  const input = new UserInput(c.initial_values, c.start_time);
  const modelInputs: number[][] = [];
  const log: { source: string; values: number[] }[] = [];
  for (const event of c.events) {
    const updates = mapper.decodeInput(event.bytes);
    if (!updates) continue;
    modelInputs.push(input.receive(updates, event.time));
    log.push({ source: "interface", values: [...input.values] });
  }
  expectClose({ model_inputs: modelInputs, log }, c.expected, tol);
});
