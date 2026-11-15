// Training datasets — port of ../impsy/impsy/dataset.py (log files → rows) and
// the slicing in ../impsy/impsy/train.py (rows → training examples).
//
// A log is the `.log` format sessionLogger.ts writes:
//   timestamp,source,x_1,…,x_n
// A dataset row is `[dt, x_1, …, x_n]`, dt being the seconds since the previous
// `interface` row. spec/vectors/dataset.json pins the conversion.

/** Training sequence length (`SEQ_LEN` in train.py). */
export const SEQUENCE_LENGTH = 50;

/** One performance: its rows `[dt, x_1, …, x_n]` in order. */
export type Performance = number[][];

const TIMESTAMP =
  /^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:[.,](\d{1,6}))?)?)?(Z|[+-]\d{2}:?\d{2})?$/;

/**
 * Whole microseconds for an ISO 8601 timestamp, as Python's
 * `datetime.fromisoformat` reads the logs: a naive wall-clock time (an offset,
 * if present, is applied). Only differences between results are meaningful;
 * integers keep those exact, as Python's timedelta is. Returns null if
 * unparseable.
 */
export function parseTimestamp(text: string): number | null {
  const m = TIMESTAMP.exec(text.trim());
  if (!m) return null;
  const [, y, mo, d, h = "0", mi = "0", s = "0", frac = "", zone] = m;
  const ms = Date.UTC(+y, +mo - 1, +d, +h, +mi, +s);
  // Reject out-of-range fields (Date.UTC would roll them over).
  const back = new Date(ms);
  if (back.getUTCMonth() !== +mo - 1 || back.getUTCDate() !== +d) return null;
  if (+h > 23 || +mi > 59 || +s > 59) return null;
  let micros = ms * 1000 + Number(frac.padEnd(6, "0"));
  if (zone && zone !== "Z") {
    const sign = zone[0] === "-" ? -1 : 1;
    const digits = zone.slice(1).replace(":", "");
    micros -= sign * (+digits.slice(0, 2) * 3600 + +digits.slice(2) * 60) * 1e6;
  }
  return micros;
}

const FLOAT = /^[+-]?(\d+\.?\d*([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?)$/;

/**
 * Convert the lines of a log file to dataset rows `[dt, x_1, …, x_n]`
 * (`transform_log_to_sequence_example`). Only `interface` rows count; rows
 * with too few values, or an unparseable timestamp or number, are skipped. The
 * first usable row only starts the clock, so it produces no dataset row.
 *
 * @param dimension the model dimension, counting dt (so rows carry `dimension − 1` values)
 */
export function logToRows(lines: string[], dimension: number): Performance {
  const columns = dimension - 1;
  const times: number[] = [];
  const values: number[][] = [];
  for (const line of lines) {
    const row = line.split(",");
    if (row.length < 2 + columns) continue;
    if (row[1].trim() !== "interface") continue;
    const time = parseTimestamp(row[0]);
    if (time === null) continue;
    const data = row.slice(2, 2 + columns).map((v) => v.trim());
    if (!data.every((v) => FLOAT.test(v))) continue;
    times.push(time);
    values.push(data.map(Number));
  }
  const rows: Performance = [];
  for (let i = 1; i < times.length; i++) {
    rows.push([(times[i] - times[i - 1]) / 1e6, ...values[i]]);
  }
  return rows;
}

/** The model dimension a log file was written for, from its `…-{dimension}d-mdrnn.log` name. */
export function dimensionFromLogName(name: string): number | null {
  const m = /-(\d+)d-mdrnn\.log$/.exec(name);
  return m ? Number(m[1]) : null;
}

/** A training example: `sequenceLength + 1` consecutive rows of one performance. */
export interface Example {
  performance: number;
  start: number;
}

/**
 * Slice performances into overlapping training examples, as train.py does:
 * performances no longer than `sequenceLength + 1` rows are dropped, and every
 * window of `sequenceLength + 1` rows (step 1) is an example — its first
 * `sequenceLength` rows are the input and its last `sequenceLength` the target.
 */
export function sliceExamples(
  performances: Performance[],
  sequenceLength = SEQUENCE_LENGTH,
): Example[] {
  const examples: Example[] = [];
  performances.forEach((rows, performance) => {
    if (rows.length <= sequenceLength + 1) return;
    for (let start = 0; start + sequenceLength + 1 <= rows.length; start++) {
      examples.push({ performance, start });
    }
  });
  return examples;
}
