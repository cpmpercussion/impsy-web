import { describe, it, expect } from "vitest";
import {
  SessionLogger,
  isoTimestamp,
  makeFileName,
  formatValue,
} from "./sessionLogger";

// A fixed local-time instant for deterministic timestamp/filename assertions.
const D = (ms = 123) => new Date(2025, 5, 20, 14, 35, 22, ms); // 2025-06-20T14:35:22.ms

describe("isoTimestamp", () => {
  it("matches Python datetime.isoformat (local, microsecond, no tz)", () => {
    expect(isoTimestamp(D(123))).toBe("2025-06-20T14:35:22.123000");
    expect(isoTimestamp(D(0))).toBe("2025-06-20T14:35:22.000000");
  });
});

describe("makeFileName", () => {
  it("matches Python's <iso-with-dashes>-{dim}d-mdrnn.log", () => {
    expect(makeFileName(9, D())).toBe("2025-06-20T14-35-22-9d-mdrnn.log");
  });
});

describe("formatValue", () => {
  it("is the shortest round-trip decimal", () => {
    expect(formatValue(0)).toBe("0");
    expect(formatValue(0.5)).toBe("0.5");
  });
});

describe("SessionLogger", () => {
  it("does not record until enabled and a session is started", () => {
    const log = new SessionLogger();
    expect(log.logInterface([0.1, 0.2])).toBe(false); // not enabled, no session
    log.setEnabled(true);
    expect(log.logInterface([0.1, 0.2])).toBe(false); // enabled but no session
    log.startSession(3, "model.tflite");
    expect(log.logInterface([0.1, 0.2])).toBe(true);
    expect(log.rowCount).toBe(1);
  });

  it("writes interface/rnn rows in the IMPSY format (no dt column)", () => {
    const log = new SessionLogger();
    log.setEnabled(true);
    log.startSession(3, "model.tflite", D());
    log.logInterface([0.5, 0.74], D(0));
    log.logRNN([0.51, 0.73], D(100));

    const lines = log.buildContent().trim().split("\n");
    // Header is commented (skipped by the dataset loader), then the two rows.
    expect(lines.filter((l) => l.startsWith("#")).length).toBeGreaterThan(0);
    expect(lines).toContain("2025-06-20T14:35:22.000000,interface,0.5,0.74");
    expect(lines).toContain("2025-06-20T14:35:22.100000,rnn,0.51,0.73");
  });

  it("clears the buffer on a new session and reflects the model dimension in the filename", () => {
    const log = new SessionLogger();
    log.setEnabled(true);
    log.startSession(9, "a.tflite", D());
    log.logInterface([0, 0], D());
    expect(log.rowCount).toBe(1);
    expect(log.downloadFileName).toBe("2025-06-20T14-35-22-9d-mdrnn.log");

    log.startSession(5, "b.tflite", D());
    expect(log.rowCount).toBe(0); // fresh buffer
    expect(log.downloadFileName).toBe("2025-06-20T14-35-22-5d-mdrnn.log");
  });
});
