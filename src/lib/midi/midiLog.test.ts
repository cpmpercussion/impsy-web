import { describe, it, expect } from "vitest";
import { MIDILog, describeMIDI, isRealtime, summarizeEmission, summarizeMIDI } from "./midiLog";

describe("summarizeMIDI (AUv3 MIDIEvent.summary parity)", () => {
  it("formats note, CC and bend", () => {
    expect(summarizeMIDI([0x90, 67, 64])).toBe("Note 67 ch1");
    expect(summarizeMIDI([0xba, 11, 80])).toBe("CC11=80 ch11");
    expect(summarizeMIDI([0xe0, 0, 64])).toBe("Bend ch1");
  });
});

describe("summarizeEmission", () => {
  it("skips the monophonic note_off and counts the rest", () => {
    const events = [{ bytes: [0x80, 60, 0] }, { bytes: [0x90, 64, 64] }, { bytes: [0xba, 1, 100] }];
    expect(summarizeEmission(events)).toBe("Note 64 ch1 +1");
  });

  it("shows a dash for an empty (fully deduped) emission", () => {
    expect(summarizeEmission([])).toBe("—");
    expect(summarizeEmission([{ bytes: [0x80, 60, 0] }])).toBe("—");
  });
});

describe("describeMIDI", () => {
  it("describes common channel messages", () => {
    expect(describeMIDI([0x91, 60, 100])).toBe("Note On ch2 60 v100");
    expect(describeMIDI([0x91, 60, 0])).toBe("Note Off ch2 60 (v0)");
    expect(describeMIDI([0x80, 60, 0])).toBe("Note Off ch1 60");
    expect(describeMIDI([0xb0, 74, 127])).toBe("CC ch1 #74=127");
    expect(describeMIDI([0xe0, 0, 64])).toBe("Bend ch1 0");
    expect(describeMIDI([0xe0, 0x7f, 0x7f])).toBe("Bend ch1 8191");
  });

  it("flags system real-time bytes", () => {
    expect(isRealtime([0xf8])).toBe(true);
    expect(isRealtime([0xfe])).toBe(true);
    expect(isRealtime([0x90, 60, 1])).toBe(false);
  });
});

describe("MIDILog", () => {
  it("keeps the newest `capacity` entries, newest first", () => {
    const log = new MIDILog(3);
    for (let i = 0; i < 5; i++) log.push("in", [0xb0, i, 0], i);
    const entries = log.flush();
    expect(entries.map((e) => e.bytes[1])).toEqual([4, 3, 2]);
  });

  it("merges successive flushes and preserves direction", () => {
    const log = new MIDILog(4);
    log.push("in", [0x90, 60, 1], 0);
    log.flush();
    log.push("out", [0x90, 61, 64], 1);
    const entries = log.flush();
    expect(entries.map((e) => e.direction)).toEqual(["out", "in"]);
    expect(log.hasPending).toBe(false);
  });

  it("copies bytes so later mutation of the source doesn't leak in", () => {
    const log = new MIDILog();
    const src = new Uint8Array([0xb0, 1, 2]);
    log.push("in", src, 0);
    src[2] = 99;
    expect(log.flush()[0].bytes).toEqual([0xb0, 1, 2]);
  });
});
