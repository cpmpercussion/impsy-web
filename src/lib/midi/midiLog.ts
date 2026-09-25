// Human-readable MIDI formatting + a small bounded log, for the live MIDI
// console (issue #14) and the dashboard's "Last Output" card (issue #9).

/** System real-time bytes (clock, start/stop, active sensing, reset). */
export function isRealtime(bytes: ArrayLike<number>): boolean {
  return bytes.length > 0 && bytes[0] >= 0xf8;
}

/**
 * Short summary of one message, matching AUv3's `MIDIEvent.summary`
 * ("Note 67 ch1", "CC11=80 ch11", "Bend ch1").
 */
export function summarizeMIDI(bytes: ArrayLike<number>): string {
  const status = bytes[0] ?? 0;
  const ch = (status & 0x0f) + 1;
  switch (status & 0xf0) {
    case 0x90:
      return `Note ${bytes[1] ?? 0} ch${ch}`;
    case 0xb0:
      return `CC${bytes[1] ?? 0}=${bytes[2] ?? 0} ch${ch}`;
    case 0xe0:
      return `Bend ch${ch}`;
    default:
      return `0x${hex(status)} ch${ch}`;
  }
}

/**
 * Summary of one RNN emission for the "Last Output" card: the first sounding
 * event plus a count of the rest (AUv3 parity). Note-offs are the monophonic
 * bookkeeping the mapper inserts before each note-on, so they're skipped.
 */
export function summarizeEmission(events: { bytes: number[] }[]): string {
  const sounding = events.filter((e) => !isNoteOff(e.bytes));
  if (sounding.length === 0) return "—";
  const first = summarizeMIDI(sounding[0].bytes);
  return sounding.length > 1 ? `${first} +${sounding.length - 1}` : first;
}

/** Detailed one-line description for the MIDI console. */
export function describeMIDI(bytes: ArrayLike<number>): string {
  const status = bytes[0] ?? 0;
  const d1 = bytes[1] ?? 0;
  const d2 = bytes[2] ?? 0;
  const ch = `ch${(status & 0x0f) + 1}`;
  switch (status & 0xf0) {
    case 0x80:
      return `Note Off ${ch} ${d1}`;
    case 0x90:
      return d2 === 0 ? `Note Off ${ch} ${d1} (v0)` : `Note On ${ch} ${d1} v${d2}`;
    case 0xa0:
      return `Poly AT ${ch} ${d1}=${d2}`;
    case 0xb0:
      return `CC ${ch} #${d1}=${d2}`;
    case 0xc0:
      return `Program ${ch} ${d1}`;
    case 0xd0:
      return `Chan AT ${ch} ${d1}`;
    case 0xe0:
      return `Bend ${ch} ${((d2 << 7) | d1) - 8192}`;
    default:
      return status === 0xf0 ? `SysEx (${bytes.length} bytes)` : `System 0x${hex(status)}`;
  }
}

export function hexBytes(bytes: ArrayLike<number>, max = 8): string {
  const parts = Array.from(bytes).slice(0, max).map(hex);
  return bytes.length > max ? `${parts.join(" ")} …` : parts.join(" ");
}

function hex(b: number): string {
  return b.toString(16).toUpperCase().padStart(2, "0");
}

function isNoteOff(bytes: ArrayLike<number>): boolean {
  const type = (bytes[0] ?? 0) & 0xf0;
  return type === 0x80 || (type === 0x90 && (bytes[2] ?? 0) === 0);
}

// ── Bounded log ──────────────────────────────────────────────────────────────

export type MIDIDirection = "in" | "out";

export interface MIDILogEntry {
  /** Monotonic id, for keyed rendering. */
  seq: number;
  /** Seconds (performance clock). */
  time: number;
  direction: MIDIDirection;
  /** True for input injected by dragging a dashboard fader. */
  fromUI: boolean;
  bytes: number[];
  text: string;
}

/**
 * Keeps the most recent `capacity` messages, newest first. Pushes are cheap
 * and buffered; `flush()` folds them into the visible list so the UI can
 * batch updates (e.g. once per animation frame) under dense MIDI streams.
 */
export class MIDILog {
  entries: MIDILogEntry[] = [];
  private pending: MIDILogEntry[] = [];
  private seq = 0;

  constructor(readonly capacity = 20) {}

  push(direction: MIDIDirection, bytes: ArrayLike<number>, time: number, fromUI = false): void {
    const copy = Array.from(bytes);
    this.pending.push({
      seq: ++this.seq,
      time,
      direction,
      fromUI,
      bytes: copy,
      text: describeMIDI(copy),
    });
    // Anything beyond capacity would be dropped at flush anyway.
    if (this.pending.length > this.capacity) this.pending.shift();
  }

  get hasPending(): boolean {
    return this.pending.length > 0;
  }

  /** Merge pending pushes into `entries`; returns the new list. */
  flush(): MIDILogEntry[] {
    if (this.pending.length > 0) {
      this.entries = [...this.pending.reverse(), ...this.entries].slice(0, this.capacity);
      this.pending = [];
    }
    return this.entries;
  }

  clear(): void {
    this.entries = [];
    this.pending = [];
  }
}
