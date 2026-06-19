// MIDIMapper — port of MIDIMapper.swift.
//
// Translates between raw MIDI bytes and normalised [0,1] dimension values.
// Note-on outputs are made monophonic-per-channel (a note_off for the previous
// note is inserted before each new note_on), matching IMPSY's impsio behaviour.

import {
  type DimensionMapping,
  type MIDIMappingSet,
  normalizeCC,
  denormalizeCC,
} from "./midiMapping";

/** A parsed/encoded MIDI event. `bytes` is the 2- or 3-byte message. */
export interface MIDIEvent {
  bytes: number[];
}

const noteOn = (ch: number, note: number, vel: number): MIDIEvent => ({
  bytes: [0x90 | (ch & 0x0f), note & 0x7f, vel & 0x7f],
});
const noteOff = (ch: number, note: number): MIDIEvent => ({
  bytes: [0x80 | (ch & 0x0f), note & 0x7f, 0],
});

interface Emission {
  rawMIDI: number;
  time: number;
}

export class MIDIMapper {
  mappings: MIDIMappingSet;

  /** Last note emitted per output channel (0–15) for monophonic note_off. */
  private lastNotes = new Map<number, number>();
  /** Per-output-dimension last emission, for the dedup window. */
  private lastEmissions = new Map<number, Emission>();

  constructor(mappings: MIDIMappingSet) {
    this.mappings = mappings;
  }

  // MARK: Decode (MIDI → normalised value)

  /**
   * Returns `[dimIndex0Based, value]` if the message matches an input mapping,
   * else null. Used by the interaction loop's input drain.
   */
  decodeInput(bytes: Uint8Array | number[]): [number, number] | null {
    if (bytes.length < 2) return null;
    const status = bytes[0];
    const messageType = status & 0xf0;
    const channel = (status & 0x0f) + 1;

    for (const m of this.mappings.inputMappings) {
      if (!m.enabled || m.channel !== channel) continue;
      switch (m.messageType) {
        case "noteOn":
          if (messageType !== 0x90 || bytes[1] !== m.number) continue;
          return [m.id - 1, (bytes.length >= 3 ? bytes[2] : 0) / 127.0];
        case "controlChange":
          if (messageType !== 0xb0 || bytes[1] !== m.number) continue;
          return [m.id - 1, normalizeCC(m, bytes.length >= 3 ? bytes[2] : 0)];
        case "pitchBend": {
          if (messageType !== 0xe0) continue;
          const lsb = bytes.length >= 2 ? bytes[1] : 0;
          const msb = bytes.length >= 3 ? bytes[2] : 0;
          return [m.id - 1, ((msb << 7) | lsb) / 16383.0];
        }
      }
    }
    return null;
  }

  // MARK: Encode (normalised values → MIDI)

  /**
   * Encode a model output vector (`values[i]` → dimension i+1) into MIDI events.
   *
   * @param dimensions when set, only emit these 0-based dimension indices
   *   (used by the input-thru echo so one input maps to one output).
   * @param now seconds; when set with a positive window, suppress re-emitting
   *   the same MIDI value for a dimension within the window.
   */
  encodeOutput(
    values: number[],
    opts: {
      dimensions?: Set<number>;
      now?: number;
      noteDedupWindow?: number;
      ccDedupWindow?: number;
    } = {},
  ): MIDIEvent[] {
    const { dimensions, now, noteDedupWindow = 0, ccDedupWindow = 0 } = opts;
    const events: MIDIEvent[] = [];

    this.mappings.outputMappings.forEach((m, i) => {
      if (i >= values.length || !m.enabled) return;
      if (dimensions && !dimensions.has(i)) return;
      const v = Math.max(0, Math.min(1, values[i]));
      const ch = (m.channel - 1) & 0x0f;

      switch (m.messageType) {
        case "noteOn": {
          const note = Math.max(0, Math.min(127, Math.round(v * 127.0)));
          if (now !== undefined && noteDedupWindow > 0) {
            const last = this.lastEmissions.get(i);
            if (last && last.rawMIDI === note && now - last.time < noteDedupWindow) return;
          }
          const prev = this.lastNotes.get(ch);
          if (prev !== undefined) events.push(noteOff(ch, prev));
          events.push(noteOn(ch, note, 64));
          this.lastNotes.set(ch, note);
          if (now !== undefined) this.lastEmissions.set(i, { rawMIDI: note, time: now });
          break;
        }
        case "controlChange": {
          const ccVal = denormalizeCC(m, v);
          if (now !== undefined && ccDedupWindow > 0) {
            const last = this.lastEmissions.get(i);
            if (last && last.rawMIDI === ccVal && now - last.time < ccDedupWindow) return;
          }
          events.push({ bytes: [0xb0 | ch, m.number & 0x7f, ccVal] });
          if (now !== undefined) this.lastEmissions.set(i, { rawMIDI: ccVal, time: now });
          break;
        }
        case "pitchBend": {
          const raw = Math.round(v * 16383.0);
          if (now !== undefined && ccDedupWindow > 0) {
            const last = this.lastEmissions.get(i);
            if (last && last.rawMIDI === raw && now - last.time < ccDedupWindow) return;
          }
          events.push({ bytes: [0xe0 | ch, raw & 0x7f, (raw >> 7) & 0x7f] });
          if (now !== undefined) this.lastEmissions.set(i, { rawMIDI: raw, time: now });
          break;
        }
      }
    });
    return events;
  }

  /** Emit note_off for every channel with an outstanding note, then forget. */
  releaseAllNotes(): MIDIEvent[] {
    const offs = [...this.lastNotes.entries()].map(([ch, note]) => noteOff(ch, note));
    this.lastNotes.clear();
    this.lastEmissions.clear();
    return offs;
  }

  /** 0-based dimension index + value for a single incoming MIDI message. */
  denseUpdate(bytes: Uint8Array | number[]): [number, number] | null {
    return this.decodeInput(bytes);
  }
}

/** Encode a single normalised value through one mapping (UI direct input). */
export function encodeSingle(value: number, m: DimensionMapping): MIDIEvent {
  const v = Math.max(0, Math.min(1, value));
  const ch = (m.channel - 1) & 0x0f;
  switch (m.messageType) {
    case "noteOn":
      return noteOn(ch, m.number, Math.max(0, Math.min(127, Math.round(v * 127.0))));
    case "controlChange":
      return { bytes: [0xb0 | ch, m.number & 0x7f, denormalizeCC(m, v)] };
    case "pitchBend": {
      const raw = Math.round(v * 16383.0);
      return { bytes: [0xe0 | ch, raw & 0x7f, (raw >> 7) & 0x7f] };
    }
  }
}
