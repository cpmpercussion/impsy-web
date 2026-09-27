// MIDIMapper — port of MIDIMapper.swift, with the MIDI semantics of IMPSY
// Python (`midi_message_to_updates` / `MidiOutputState` in
// ../impsy/impsy/utils.py), which the conformance vectors in spec/ pin down.
//
// Translates between raw MIDI bytes and normalised [0,1] dimension values.
// Decoding: one message can update several dimensions. Encoding: notes are
// tracked per dimension (a note-off for a dimension's previous note precedes
// its next note-on, unless another dimension on the channel still holds that
// note); CCs and pitch bends are only sent when their MIDI value changes.

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

/** One input-vector update: `[dimIndex0Based, value]`. */
export type DimensionUpdate = [number, number];

const PITCH_BEND_RANGE = 16383;

const noteOn = (ch: number, note: number, vel: number): MIDIEvent => ({
  bytes: [0x90 | (ch & 0x0f), note & 0x7f, vel & 0x7f],
});
const noteOff = (ch: number, note: number): MIDIEvent => ({
  bytes: [0x80 | (ch & 0x0f), note & 0x7f, 0],
});
const pitchBend = (ch: number, raw: number): MIDIEvent => ({
  bytes: [0xe0 | (ch & 0x0f), raw & 0x7f, (raw >> 7) & 0x7f],
});

const clip01 = (v: number) => Math.max(0, Math.min(1, v));
/** [0,1] → 0–127, round to nearest (half up). */
const toMIDI = (v: number) => Math.round(clip01(v) * 127);
/** [0,1] → 14-bit pitch bend 0–16383 (centre 8192). */
const toPitchBend = (v: number) => Math.round(clip01(v) * PITCH_BEND_RANGE);

export class MIDIMapper {
  mappings: MIDIMappingSet;

  /** Output dimension index → the (0-based channel, note) it is sounding. */
  private sounding = new Map<number, [number, number]>();
  /** Last CC / pitch bend value sent, keyed by channel (and controller). */
  private lastSent = new Map<string, number>();

  constructor(mappings: MIDIMappingSet) {
    this.mappings = mappings;
  }

  // MARK: Decode (MIDI → normalised value)

  /**
   * The `[dimIndex, value]` updates one message makes to the input vector, or
   * null if it changes nothing. A message mapped to several dimensions updates
   * all of them. A note-on sets noteOn dimensions on its channel to note/127
   * and noteVelocity dimensions to velocity/127; velocity-0 note-ons are
   * note-offs and are ignored like them.
   */
  decodeInput(bytes: Uint8Array | number[]): DimensionUpdate[] | null {
    if (bytes.length < 3) return null;
    const messageType = bytes[0] & 0xf0;
    const channel = (bytes[0] & 0x0f) + 1;
    const updates: DimensionUpdate[] = [];

    this.mappings.inputMappings.forEach((m, i) => {
      if (!m.enabled || m.channel !== channel) return;
      switch (messageType) {
        case 0x90:
          if (bytes[2] === 0) return;
          if (m.messageType === "noteOn") updates.push([i, bytes[1] / 127]);
          else if (m.messageType === "noteVelocity") updates.push([i, bytes[2] / 127]);
          return;
        case 0xb0:
          if (m.messageType === "controlChange" && m.number === bytes[1])
            updates.push([i, normalizeCC(m, bytes[2])]);
          return;
        case 0xe0:
          if (m.messageType === "pitchBend")
            updates.push([i, ((bytes[2] << 7) | bytes[1]) / PITCH_BEND_RANGE]);
          return;
      }
    });
    return updates.length > 0 ? updates : null;
  }

  // MARK: Encode (normalised values → MIDI)

  /**
   * Encode a model output vector (`values[i]` → dimension i+1) into MIDI events.
   * Values are clipped to [0,1].
   *
   * @param dimensions when set, only emit these 0-based dimension indices
   *   (used by the input-thru echo so one input maps to one output).
   */
  encodeOutput(values: number[], opts: { dimensions?: Set<number> } = {}): MIDIEvent[] {
    const { dimensions } = opts;
    const mappings = this.mappings.outputMappings;
    const events: MIDIEvent[] = [];

    // The first noteVelocity dimension on each channel sets its notes' velocity.
    const velocityIndex = new Map<number, number>();
    mappings.forEach((m, i) => {
      if (m.enabled && m.messageType === "noteVelocity" && !velocityIndex.has(m.channel))
        velocityIndex.set(m.channel, i);
    });

    mappings.forEach((m, i) => {
      if (i >= values.length || !m.enabled) return;
      if (dimensions && !dimensions.has(i)) return;
      const ch = (m.channel - 1) & 0x0f;

      switch (m.messageType) {
        case "noteOn": {
          const note = toMIDI(values[i]);
          const previous = this.sounding.get(i);
          this.sounding.delete(i);
          if (previous && !this.heldElsewhere(i, previous))
            events.push(noteOff(previous[0], previous[1]));
          const vi = velocityIndex.get(m.channel);
          const velocity =
            vi !== undefined && vi < values.length
              ? Math.max(1, toMIDI(values[vi])) // velocity 0 would be a note-off
              : m.velocity !== undefined
                ? Math.trunc(Math.max(1, Math.min(127, m.velocity)))
                : 127;
          events.push(noteOn(ch, note, velocity));
          this.sounding.set(i, [ch, note]);
          break;
        }
        case "noteVelocity":
          break; // read by the note dimensions on its channel
        case "controlChange": {
          const value = denormalizeCC(m, values[i]);
          if (this.changed(`cc:${ch}:${m.number}`, value))
            events.push({ bytes: [0xb0 | ch, m.number & 0x7f, value] });
          break;
        }
        case "pitchBend": {
          const raw = toPitchBend(values[i]);
          if (this.changed(`pb:${ch}`, raw)) events.push(pitchBend(ch, raw));
          break;
        }
      }
    });
    return events;
  }

  /**
   * Note-offs for every sounding (channel, note), then forget them. Also
   * forgets the last CC / pitch bend values so the next step resends them.
   */
  releaseAllNotes(): MIDIEvent[] {
    const seen = new Set<string>();
    const offs: MIDIEvent[] = [];
    for (const [ch, note] of this.sounding.values()) {
      const key = `${ch}:${note}`;
      if (seen.has(key)) continue;
      seen.add(key);
      offs.push(noteOff(ch, note));
    }
    this.sounding.clear();
    this.lastSent.clear();
    return offs;
  }

  /** Record `value` as sent to `key`; false if it's the same as last time. */
  private changed(key: string, value: number): boolean {
    if (this.lastSent.get(key) === value) return false;
    this.lastSent.set(key, value);
    return true;
  }

  private heldElsewhere(index: number, [ch, note]: [number, number]): boolean {
    for (const [i, held] of this.sounding)
      if (i !== index && held[0] === ch && held[1] === note) return true;
    return false;
  }
}

/**
 * Encode a single normalised value through one input mapping as the MIDI
 * message that would produce it (UI fader → injected input). Note messages
 * carry two values, so `companion` supplies the other one: the velocity for a
 * noteOn, the note for a noteVelocity (both [0,1], default 64/127 and 60/127).
 */
export function encodeSingle(value: number, m: DimensionMapping, companion?: number): MIDIEvent {
  const ch = (m.channel - 1) & 0x0f;
  // Velocity must be non-zero or decodeInput treats the note-on as a note-off.
  const velocity = (v: number) => Math.max(1, toMIDI(v));
  switch (m.messageType) {
    case "noteOn":
      return noteOn(ch, toMIDI(value), companion === undefined ? 64 : velocity(companion));
    case "noteVelocity":
      return noteOn(ch, companion === undefined ? 60 : toMIDI(companion), velocity(value));
    case "controlChange":
      return { bytes: [0xb0 | ch, m.number & 0x7f, denormalizeCC(m, value)] };
    case "pitchBend":
      return pitchBend(ch, toPitchBend(value));
  }
}
