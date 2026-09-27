// MIDI mapping model — port of MIDIMapping.swift.
//
// One DimensionMapping links an IMPSY dimension (1-based; dim 0 is always the
// time delta and is never user-mappable) to a MIDI message. Input and output
// mappings are independent.

// `noteVelocity` pairs with the note_on dimensions on its channel: on input it
// takes a note-on's velocity; on output it sets their velocity and sends
// nothing itself (IMPSY `note_velocity`, impsy#98).
export type MIDIMessageType = "noteOn" | "noteVelocity" | "controlChange" | "pitchBend";

export const messageTypeDisplayName: Record<MIDIMessageType, string> = {
  noteOn: "Note On",
  noteVelocity: "Note Velocity",
  controlChange: "CC",
  pitchBend: "Pitch Bend",
};

/**
 * Whether this message type uses the `number` field (CC number). Note On
 * doesn't: the note number *is* the value, on input and output alike.
 */
export function usesNumber(t: MIDIMessageType): boolean {
  return t === "controlChange";
}

export interface DimensionMapping {
  /** 1-based dimension index, kept in lock-step with array position. */
  id: number;
  messageType: MIDIMessageType;
  /** MIDI channel 1–16. */
  channel: number;
  /** CC number (0–127); ignored for the other types. */
  number: number;
  /**
   * Fixed output velocity for noteOn (IMPSY `["note_on", ch, velocity]`),
   * clamped to 1–127. Unset means 127. A noteVelocity dimension on the same
   * channel overrides it. Not consulted on input.
   */
  velocity?: number;
  /** CC range lower bound (0–127); only consulted for controlChange. */
  minValue: number;
  /** CC range upper bound (0–127). */
  maxValue: number;
  /** When false, skip this dimension on both encode and decode. */
  enabled: boolean;
}

export function defaultMapping(index: number): DimensionMapping {
  return {
    id: index,
    messageType: "controlChange",
    channel: 1,
    number: Math.max(0, Math.min(127, 73 + index)), // CC 74, 75, 76…
    minValue: 0,
    maxValue: 127,
    enabled: true,
  };
}

/**
 * Map a 7-bit CC value through a mapping's min/max into normalised [0,1] — the
 * inverse of denormalizeCC. Matches `midi_to_value` in ../impsy/impsy/utils.py:
 * the raw value is clamped to the range (either way round), an empty range
 * gives 0.
 */
export function normalizeCC(m: DimensionMapping, raw: number): number {
  const span = m.maxValue - m.minValue;
  if (span === 0) return 0;
  const low = Math.min(m.minValue, m.maxValue);
  const high = Math.max(m.minValue, m.maxValue);
  const clamped = Math.max(low, Math.min(high, raw));
  return (clamped - m.minValue) / span;
}

/**
 * Map normalised [0,1] through a mapping's min/max to a 7-bit CC value,
 * rounding once to nearest, half up (`value_to_midi` in utils.py, impsy#100).
 */
export function denormalizeCC(m: DimensionMapping, v: number): number {
  const clamped = Math.max(0, Math.min(1, v));
  const scaled = m.minValue + clamped * (m.maxValue - m.minValue);
  return Math.max(0, Math.min(127, Math.round(scaled)));
}

export interface MIDIMappingSet {
  /** One entry per model dimension (excluding dim 0). Index 0 = dimension 1. */
  inputMappings: DimensionMapping[];
  outputMappings: DimensionMapping[];
}

/** Default mappings for a model with `dimension` total dims (including time). */
export function defaultMappingSet(dimension: number): MIDIMappingSet {
  const count = Math.max(1, dimension - 1);
  const make = () => Array.from({ length: count }, (_, i) => defaultMapping(i + 1));
  return { inputMappings: make(), outputMappings: make() };
}

/**
 * Default mapping for the bundled 9-dimension IMPSY model, ported from
 * configs/AiC-charles-u6midipro.toml. Input: 8 knobs CC 13–20 ch1. Output:
 * alternating note_on (ch 1–4) and CC 1–4 (ch 11).
 */
export function aicU6MIDIProDefault(): MIDIMappingSet {
  const input: DimensionMapping[] = Array.from({ length: 8 }, (_, i) => ({
    ...defaultMapping(i + 1),
    messageType: "controlChange",
    channel: 1,
    number: 13 + i,
  }));
  const output: DimensionMapping[] = Array.from({ length: 8 }, (_, i) =>
    i % 2 === 0
      ? { ...defaultMapping(i + 1), messageType: "noteOn", channel: i / 2 + 1, number: 60 }
      : {
          ...defaultMapping(i + 1),
          messageType: "controlChange",
          channel: 11,
          number: (i - 1) / 2 + 1,
        },
  );
  return { inputMappings: input, outputMappings: output };
}

/** Resize mappings to a new model dimension, preserving existing entries. */
export function resizeMappingSet(set: MIDIMappingSet, dimension: number): MIDIMappingSet {
  const count = Math.max(0, dimension - 1);
  const fit = (arr: DimensionMapping[]): DimensionMapping[] => {
    const next = arr.slice(0, count);
    while (next.length < count) next.push(defaultMapping(next.length + 1));
    return next.map((m, i) => ({ ...m, id: i + 1 }));
  };
  return { inputMappings: fit(set.inputMappings), outputMappings: fit(set.outputMappings) };
}
