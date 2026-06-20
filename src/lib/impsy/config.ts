// IMPSY TOML config import/export — port of IMPSYConfig.swift.
//
// Round-trips with IMPSY's TOML config format (see `../impsy/configs/*.toml`
// and `../impsy/impsy/data/default.toml`). Import applies the four runtime
// params + MIDI thru and the input/output mappings; export serialises the same
// back out so a setup can move between the Python app, the AUv3 plugin, and
// this web port.
//
// Deviations from IMPSY's schema (the same ones the AUv3 plugin makes, see #3):
//   - `in_device` / `out_device` collapse to a single synthetic device key on
//     export. On import we read every per-device mapping array in the order
//     `in_device` / `out_device` declares, then concatenate.
//   - `[osc]`, `[serial]`, `[websocket]`, `[webui]`, … are preserved verbatim
//     on round-trip (we keep the parsed document) but otherwise ignored.
//   - `note_on` carries no fixed note number in IMPSY (the dimension value IS
//     the note); we default `number` to 60 like the AUv3 port.

import { parse, stringify } from "smol-toml";
import type { DimensionMapping, MIDIMappingSet, MIDIMessageType } from "./midiMapping";
import { ParameterDefaults } from "./constants";

/** Device key used for the single synthetic device written on export. Web MIDI
 *  device names are environment-specific, so — like the AUv3's "AUv3" — we emit
 *  one placeholder device. The importer flattens devices, so the name is
 *  cosmetic and round-trips losslessly through both other platforms. */
export const SYNTHESIZED_DEVICE_NAME = "WebMIDI";

const TYPE_FROM_IMPSY: Record<string, MIDIMessageType> = {
  note_on: "noteOn",
  control_change: "controlChange",
  pitch_bend: "pitchBend",
};

const TYPE_TO_IMPSY: Record<MIDIMessageType, string> = {
  noteOn: "note_on",
  controlChange: "control_change",
  pitchBend: "pitch_bend",
};

export interface IMPSYConfig {
  // Metadata — preserved on round-trip; not surfaced in the UI today.
  title?: string;
  owner?: string;
  description?: string;

  // The four runtime params + MIDI thru.
  threshold: number;
  sigmaTemp: number;
  piTemp: number;
  timescale: number;
  inputThru: boolean;

  // Model metadata. `modelFile` is a hint only — model loading stays a separate
  // user action; we never auto-resolve the path.
  modelFile?: string;
  modelDimension?: number;
  modelSize?: string;

  // Mappings, flattened across all devices in declared order.
  inputMappings: DimensionMapping[];
  outputMappings: DimensionMapping[];

  // The raw parsed document, kept so unmodelled sections survive a round-trip.
  // Undefined when the config was built fresh in memory (e.g. on export with no
  // prior import).
  raw?: Record<string, unknown>;
}

// ── Parse ───────────────────────────────────────────────────────────────────

export class ConfigParseError extends Error {
  constructor(message: string) {
    super(`Malformed IMPSY config: ${message}`);
    this.name = "ConfigParseError";
  }
}

export function parseConfig(toml: string): IMPSYConfig {
  let table: Record<string, unknown>;
  try {
    table = parse(toml) as Record<string, unknown>;
  } catch (err) {
    throw new ConfigParseError(err instanceof Error ? err.message : String(err));
  }

  const config: IMPSYConfig = {
    threshold: ParameterDefaults.threshold,
    sigmaTemp: ParameterDefaults.sigmaTemp,
    piTemp: ParameterDefaults.piTemp,
    timescale: ParameterDefaults.timescale,
    inputThru: ParameterDefaults.inputThru,
    inputMappings: [],
    outputMappings: [],
    raw: table,
  };

  // ── Top-level metadata
  if (typeof table.title === "string") config.title = table.title;
  if (typeof table.owner === "string") config.owner = table.owner;
  if (typeof table.description === "string") config.description = table.description;

  // ── [interaction]
  const interaction = asTable(table.interaction);
  if (interaction) {
    if (typeof interaction.threshold === "number") config.threshold = interaction.threshold;
    if (typeof interaction.input_thru === "boolean") config.inputThru = interaction.input_thru;
  }

  // ── [model]
  const model = asTable(table.model);
  if (model) {
    if (typeof model.file === "string") config.modelFile = model.file;
    if (typeof model.dimension === "number") config.modelDimension = model.dimension;
    if (typeof model.size === "string") config.modelSize = model.size;
    if (typeof model.sigmatemp === "number") config.sigmaTemp = model.sigmatemp;
    if (typeof model.pitemp === "number") config.piTemp = model.pitemp;
    if (typeof model.timescale === "number") config.timescale = model.timescale;
  }

  // ── [midi]
  const midi = asTable(table.midi);
  if (midi) {
    config.inputMappings = readMappings(midi, "input", "in_device");
    config.outputMappings = readMappings(midi, "output", "out_device");
  }

  return config;
}

/**
 * Walk `midi.input` (or `midi.output`) — a table of `"Device" = [ entry, … ]` —
 * and flatten the per-device arrays in the order declared by `in_device` /
 * `out_device`. Devices present in the table but not listed in the order array
 * are appended afterwards in encounter order, so no mapping is silently dropped.
 * Each entry's flattened position is its 1-based dimension id.
 */
function readMappings(
  midi: Record<string, unknown>,
  deviceKey: string,
  orderKey: string,
): DimensionMapping[] {
  const devices = asTable(midi[deviceKey]);
  if (!devices) return [];

  const declaredOrder: string[] = [];
  const order = midi[orderKey];
  if (Array.isArray(order)) {
    for (const entry of order) if (typeof entry === "string") declaredOrder.push(entry);
  }
  const seen = new Set(declaredOrder);
  for (const key of Object.keys(devices)) {
    if (!seen.has(key)) {
      declaredOrder.push(key);
      seen.add(key);
    }
  }

  const mappings: DimensionMapping[] = [];
  for (const device of declaredOrder) {
    const array = devices[device];
    if (!Array.isArray(array)) continue;
    for (const entry of array) {
      const mapping = parseMappingEntry(entry, mappings.length + 1);
      if (mapping) mappings.push(mapping);
    }
  }
  return mappings;
}

/**
 * Convert one IMPSY entry — e.g. `["control_change", 1, 13, 0, 127]` — into a
 * DimensionMapping. Returns null for unknown/malformed entries rather than
 * throwing, so a single bad row never kills the whole load.
 */
function parseMappingEntry(entry: unknown, dimensionID: number): DimensionMapping | null {
  if (!Array.isArray(entry) || entry.length < 2) return null;
  const typeRaw = entry[0];
  const channel = entry[1];
  if (typeof typeRaw !== "string" || typeof channel !== "number") return null;
  const messageType = TYPE_FROM_IMPSY[typeRaw];
  if (!messageType) return null;

  const base = { id: dimensionID, channel, minValue: 0, maxValue: 127, enabled: true };
  switch (messageType) {
    case "noteOn":
      // IMPSY note_on carries no fixed note number — default to middle C.
      return { ...base, messageType, number: 60 };
    case "controlChange": {
      if (entry.length < 3 || typeof entry[2] !== "number") return null;
      const hasRange = entry.length >= 5;
      const minValue = hasRange && typeof entry[3] === "number" ? entry[3] : 0;
      const maxValue = hasRange && typeof entry[4] === "number" ? entry[4] : 127;
      return { ...base, messageType, number: entry[2], minValue, maxValue };
    }
    case "pitchBend":
      return { ...base, messageType, number: 0 };
  }
}

// ── Serialize ─────────────────────────────────────────────────────────────────

export function serializeConfig(config: IMPSYConfig): string {
  // Start from the raw doc (so unmodelled sections survive), else a fresh table.
  const table: Record<string, unknown> = config.raw ? structuredClone(config.raw) : {};

  // ── Top-level metadata
  if (config.title !== undefined) table.title = config.title;
  if (config.owner !== undefined) table.owner = config.owner;
  if (config.description !== undefined) table.description = config.description;

  // ── [interaction]
  const interaction = asTable(table.interaction) ?? {};
  interaction.threshold = config.threshold;
  interaction.input_thru = config.inputThru;
  if (interaction.mode === undefined) interaction.mode = "callresponse";
  table.interaction = interaction;

  // ── [model]
  const model = asTable(table.model) ?? {};
  if (config.modelFile !== undefined) model.file = config.modelFile;
  if (config.modelDimension !== undefined) model.dimension = config.modelDimension;
  if (config.modelSize !== undefined) model.size = config.modelSize;
  model.sigmatemp = config.sigmaTemp;
  model.pitemp = config.piTemp;
  model.timescale = config.timescale;
  table.model = model;

  // ── [midi] — overwrite the device-mapping arrays with our single synthetic
  // device; any other keys under [midi] survive because we only replace ours.
  const midi = asTable(table.midi) ?? {};
  midi.in_device = [SYNTHESIZED_DEVICE_NAME];
  midi.out_device = [SYNTHESIZED_DEVICE_NAME];
  midi.input = { [SYNTHESIZED_DEVICE_NAME]: config.inputMappings.map(makeMappingEntry) };
  midi.output = { [SYNTHESIZED_DEVICE_NAME]: config.outputMappings.map(makeMappingEntry) };
  table.midi = midi;

  return stringify(table);
}

/**
 * IMPSY's positional entry format:
 *   noteOn        → ["note_on", channel]
 *   controlChange → ["control_change", channel, cc]  (+ min,max when not 0–127)
 *   pitchBend     → ["pitch_bend", channel]
 */
function makeMappingEntry(m: DimensionMapping): (string | number)[] {
  switch (m.messageType) {
    case "noteOn":
      return [TYPE_TO_IMPSY.noteOn, m.channel];
    case "controlChange": {
      const entry: (string | number)[] = [TYPE_TO_IMPSY.controlChange, m.channel, m.number];
      if (m.minValue !== 0 || m.maxValue !== 127) entry.push(m.minValue, m.maxValue);
      return entry;
    }
    case "pitchBend":
      return [TYPE_TO_IMPSY.pitchBend, m.channel];
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Narrow an unknown TOML value to a plain table (object), or null. */
function asTable(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

/** Convenience: build a MIDIMappingSet from a parsed config. */
export function mappingSetFromConfig(config: IMPSYConfig): MIDIMappingSet {
  return { inputMappings: config.inputMappings, outputMappings: config.outputMappings };
}
