// SessionLogger — port of SessionLogger.swift.
//
// Records interaction events to the IMPSY `.log` format used by IMPSY Python
// (../impsy/impsy/interaction.py) and the AUv3 plugin. Each line is:
//
//   YYYY-MM-DDTHH:mm:ss.ffffff,{interface|rnn},v1,v2,...,vN
//
// where v1…vN are the *modeled* dimensions in [0,1] — no dt. dt is reconstructed
// from timestamp diffs by the dataset loader (../impsy/impsy/dataset.py), which
// reads the file with csv.reader, keeps only `interface` rows, and silently
// skips anything that fails to parse — so the commented header below is ignored.
//
// File naming matches Python: `<iso-with-dashes>-{dimension}d-mdrnn.log`, where
// `dimension` is the model's full dimension (includes dt as dim 0).
//
// Unlike the native loggers (which stream to a user-picked folder), this keeps
// rows in memory and hands them to a browser download — cross-browser and good
// enough for v1 (see issue #2). A fresh buffer opens on each model load.

/**
 * `YYYY-MM-DDTHH:mm:ss.ffffff` in local time, matching Python's
 * `datetime.now().isoformat()`. The browser's `Date` only resolves to
 * milliseconds, so the trailing three microsecond digits are always zero —
 * enough for `datetime.fromisoformat` and for the dt diffs the loader derives.
 */
export function isoTimestamp(date: Date = new Date()): string {
  const p = (n: number, len = 2) => String(n).padStart(len, "0");
  const micros = p(date.getMilliseconds() * 1000, 6);
  return (
    `${date.getFullYear()}-${p(date.getMonth() + 1)}-${p(date.getDate())}T` +
    `${p(date.getHours())}:${p(date.getMinutes())}:${p(date.getSeconds())}.${micros}`
  );
}

/** `<iso-with-dashes-to-seconds>-{dimension}d-mdrnn.log`, parity with Python. */
export function makeFileName(dimension: number, date: Date = new Date()): string {
  const p = (n: number) => String(n).padStart(2, "0");
  const stem =
    `${date.getFullYear()}-${p(date.getMonth() + 1)}-${p(date.getDate())}T` +
    `${p(date.getHours())}-${p(date.getMinutes())}-${p(date.getSeconds())}`;
  return `${stem}-${dimension}d-mdrnn.log`;
}

/** Shortest round-trip decimal, like Swift's `String(Float)` / Python `str`. */
export function formatValue(v: number): string {
  return String(v);
}

export type LogSource = "interface" | "rnn";

export class SessionLogger {
  private enabled = false;
  private dimension = 0; // full model dimension (incl. dt as dim 0)
  private modelName = "";
  private fileName: string | null = null;
  private rows: string[] = [];

  /** Number of buffered event rows (excludes the header). */
  get rowCount(): number {
    return this.rows.length;
  }

  get isEnabled(): boolean {
    return this.enabled;
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  /**
   * Begin a new logging session for a freshly-loaded model. Clears the buffer
   * and fixes the download filename to "now" (mirrors the native loggers, which
   * open a fresh file per model load).
   */
  startSession(dimension: number, modelName: string, now: Date = new Date()): void {
    this.dimension = dimension;
    this.modelName = modelName;
    this.fileName = makeFileName(dimension, now);
    this.rows = [];
  }

  endSession(): void {
    this.dimension = 0;
    this.modelName = "";
    this.fileName = null;
    this.rows = [];
  }

  /** Record a user-input event (full input vector, modeled dims only, no dt). */
  logInterface(values: number[], now: Date = new Date()): boolean {
    return this.writeLine("interface", values, now);
  }

  /** Record an RNN-generated event (post-clamp output, modeled dims, no dt). */
  logRNN(values: number[], now: Date = new Date()): boolean {
    return this.writeLine("rnn", values, now);
  }

  private writeLine(source: LogSource, values: number[], now: Date): boolean {
    if (!this.enabled || this.dimension <= 0) return false;
    const valueString = values.map(formatValue).join(",");
    this.rows.push(`${isoTimestamp(now)},${source},${valueString}`);
    return true;
  }

  /** Commented header (skipped by the dataset loader) + buffered rows. */
  buildContent(): string {
    const header = [
      `# IMPSY Web log`,
      `# model=${this.modelName}`,
      `# dimension=${this.dimension}`,
      ``,
    ].join("\n");
    return header + this.rows.join("\n") + (this.rows.length ? "\n" : "");
  }

  get downloadFileName(): string {
    return this.fileName ?? makeFileName(this.dimension || 0);
  }
}
