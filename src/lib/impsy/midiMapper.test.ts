import { describe, it, expect } from "vitest";
import { MIDIMapper } from "./midiMapper";
import { defaultMappingSet, type MIDIMappingSet } from "./midiMapping";

function ccSet(): MIDIMappingSet {
  // dim1 → CC74 ch1 (default), dim2 → CC75 ch1
  return defaultMappingSet(3);
}

describe("decodeInput", () => {
  it("decodes a CC into the right 0-based dimension and normalised value", () => {
    const m = new MIDIMapper(ccSet());
    // CC74 (0xB0, 74, 127) on ch1 → dimension 1 → index 0, value 1.0
    expect(m.decodeInput([0xb0, 74, 127])).toEqual([0, 1]);
    // CC75 value 0 → index 1, value 0
    expect(m.decodeInput([0xb0, 75, 0])).toEqual([1, 0]);
  });

  it("ignores messages that match no mapping", () => {
    const m = new MIDIMapper(ccSet());
    expect(m.decodeInput([0xb0, 7, 100])).toBeNull(); // CC7 not mapped
    expect(m.decodeInput([0x90, 60, 100])).toBeNull(); // note, but maps are CC
  });
});

describe("encodeOutput", () => {
  it("emits a CC per output dimension", () => {
    const m = new MIDIMapper(ccSet());
    const events = m.encodeOutput([1.0, 0.0]);
    expect(events).toHaveLength(2);
    expect(events[0].bytes).toEqual([0xb0, 74, 127]);
    expect(events[1].bytes).toEqual([0xb0, 75, 0]);
  });

  it("restricts to requested dimensions for the input-thru echo", () => {
    const m = new MIDIMapper(ccSet());
    const events = m.encodeOutput([0.5, 0.5], { dimensions: new Set([1]) });
    expect(events).toHaveLength(1);
    expect(events[0].bytes[1]).toBe(75); // only dim 2 (CC75)
  });

  it("makes note outputs monophonic per channel (note_off before note_on)", () => {
    const set: MIDIMappingSet = {
      inputMappings: [],
      outputMappings: [
        { id: 1, messageType: "noteOn", channel: 1, number: 60, minValue: 0, maxValue: 127, enabled: true },
      ],
    };
    const m = new MIDIMapper(set);
    const first = m.encodeOutput([0.5]); // note 64
    expect(first).toHaveLength(1);
    expect(first[0].bytes[0]).toBe(0x90);
    const second = m.encodeOutput([1.0]); // note 127, preceded by note_off of 64
    expect(second).toHaveLength(2);
    expect(second[0].bytes[0]).toBe(0x80); // note_off previous
    expect(second[1].bytes[0]).toBe(0x90); // note_on new
  });

  it("suppresses a repeat CC value within the dedup window", () => {
    const m = new MIDIMapper(ccSet());
    expect(m.encodeOutput([0.5, 0], { now: 0, ccDedupWindow: 0.03 })).toHaveLength(2);
    // Same values 10 ms later → both suppressed (within 30 ms window)
    expect(m.encodeOutput([0.5, 0], { now: 0.01, ccDedupWindow: 0.03 })).toHaveLength(0);
    // 40 ms later → window expired
    expect(m.encodeOutput([0.5, 0], { now: 0.05, ccDedupWindow: 0.03 }).length).toBeGreaterThan(0);
  });
});

describe("releaseAllNotes", () => {
  it("emits note_off for outstanding notes and clears state", () => {
    const set: MIDIMappingSet = {
      inputMappings: [],
      outputMappings: [
        { id: 1, messageType: "noteOn", channel: 2, number: 60, minValue: 0, maxValue: 127, enabled: true },
      ],
    };
    const m = new MIDIMapper(set);
    m.encodeOutput([0.5]);
    const offs = m.releaseAllNotes();
    expect(offs).toHaveLength(1);
    expect(offs[0].bytes[0]).toBe(0x81); // note_off ch2
    expect(m.releaseAllNotes()).toHaveLength(0); // cleared
  });
});
