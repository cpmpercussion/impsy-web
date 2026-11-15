import { describe, expect, it } from "vitest";
import { dimensionFromLogName, logToRows, parseTimestamp, sliceExamples } from "./dataset";
import { isoTimestamp, makeFileName } from "./sessionLogger";

// The log → rows conversion itself is pinned by spec/vectors/dataset.json
// (conformance.test.ts); these cover the pieces around it.

describe("parseTimestamp", () => {
  it("reads the timestamps sessionLogger writes", () => {
    const a = new Date(2026, 8, 26, 10, 0, 0, 0);
    const b = new Date(2026, 8, 26, 10, 0, 1, 250);
    expect(parseTimestamp(isoTimestamp(b))! - parseTimestamp(isoTimestamp(a))!).toBe(1_250_000);
  });

  it("keeps microseconds and accepts second-resolution and date-only forms", () => {
    const base = parseTimestamp("2026-09-26T10:00:00")!;
    expect(parseTimestamp("2026-09-26T10:00:00.000250")! - base).toBe(250);
    expect(parseTimestamp("2026-09-26 10:00:00.5")! - base).toBe(500_000);
    expect(base - parseTimestamp("2026-09-26")!).toBe(36_000e6);
  });

  it("is wall-clock time: no daylight-saving jumps", () => {
    // 02:30 doesn't exist on this date in zones that spring forward; Python's
    // naive datetimes don't care, and neither should we.
    expect(parseTimestamp("2026-03-08T03:30:00")! - parseTimestamp("2026-03-08T01:30:00")!).toBe(
      7200e6,
    );
  });

  it("rejects what datetime.fromisoformat rejects", () => {
    for (const bad of ["not-a-timestamp", "", "2026-13-01T00:00:00", "2026-02-30", "10:00:00"]) {
      expect(parseTimestamp(bad)).toBeNull();
    }
  });
});

describe("logToRows", () => {
  it("skips the commented header sessionLogger writes and non-interface rows", () => {
    const rows = logToRows(
      [
        "# impsy-web log, model=demo",
        "2026-09-26T10:00:00.000000,interface,0.5",
        "2026-09-26T10:00:00.100000,rnn,0.9",
        "2026-09-26T10:00:00.250000,interface,0.75",
        "",
      ],
      2,
    );
    expect(rows).toEqual([[0.25, 0.75]]);
  });

  it("does not read empty or non-numeric fields as numbers", () => {
    const lines = [
      "2026-09-26T10:00:00,interface,0.5,0.5",
      "2026-09-26T10:00:01,interface,,0.5",
      "2026-09-26T10:00:02,interface,0x10,0.5",
      "2026-09-26T10:00:03,interface,1e-1, .5",
    ];
    expect(logToRows(lines, 3)).toEqual([[3, 0.1, 0.5]]);
  });
});

describe("dimensionFromLogName", () => {
  it("reads the dimension from sessionLogger file names", () => {
    expect(dimensionFromLogName(makeFileName(9))).toBe(9);
    expect(dimensionFromLogName("notes.txt")).toBeNull();
  });
});

describe("sliceExamples", () => {
  const rows = (n: number) => Array.from({ length: n }, (_, i) => [0.1, i]);

  it("drops performances no longer than sequenceLength + 1, like train.py", () => {
    expect(sliceExamples([rows(51)])).toEqual([]);
    expect(sliceExamples([rows(52)])).toEqual([
      { performance: 0, start: 0 },
      { performance: 0, start: 1 },
    ]);
  });

  it("takes every window, performance by performance", () => {
    const examples = sliceExamples([rows(6), rows(3), rows(5)], 3);
    expect(examples).toEqual([
      { performance: 0, start: 0 },
      { performance: 0, start: 1 },
      { performance: 0, start: 2 },
      { performance: 2, start: 0 },
      { performance: 2, start: 1 },
    ]);
  });
});
