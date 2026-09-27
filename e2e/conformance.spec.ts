import { readFile } from "node:fs/promises";
import { test, expect } from "@playwright/test";

// spec/vectors/model.json (IMPSY conformance vectors, see
// src/lib/impsy/conformance.test.ts for the rest). LiteRT.js only runs in a
// browser, so the fixed-weight model is run through TFLiteRNN on
// verify/conformance.html and compared here. That page is served by the dev
// server only (it isn't part of the production build).

const SPEC_VERSION = "1.0.0";

interface ModelVectors {
  spec_version: string;
  tolerance: number;
  cases: { name: string; expected: unknown }[];
}

/** Compare numbers within `tol`, everything else exactly, recursively. */
function mismatches(actual: unknown, expected: unknown, tol: number, path = "$"): string[] {
  if (typeof expected === "number") {
    return typeof actual === "number" && Math.abs(actual - expected) <= tol
      ? []
      : [`${path}: ${JSON.stringify(actual)} vs ${expected}`];
  }
  if (Array.isArray(expected)) {
    if (!Array.isArray(actual) || actual.length !== expected.length)
      return [`${path}: length ${JSON.stringify(actual)?.length} vs ${expected.length}`];
    return expected.flatMap((e, i) => mismatches(actual[i], e, tol, `${path}[${i}]`));
  }
  if (expected !== null && typeof expected === "object") {
    return Object.entries(expected).flatMap(([k, e]) =>
      mismatches((actual as Record<string, unknown> | null)?.[k], e, tol, `${path}.${k}`),
    );
  }
  return actual === expected ? [] : [`${path}: ${JSON.stringify(actual)} vs ${expected}`];
}

test.describe("model.json conformance", () => {
  test("model.json cases match the fixed-weight model", async ({ page }, info) => {
    test.skip(info.project.name !== "dev", "verify pages are dev-server only");
    const doc: ModelVectors = JSON.parse(
      await readFile(new URL("../spec/vectors/model.json", import.meta.url), "utf8"),
    );
    expect(doc.spec_version).toBe(SPEC_VERSION);

    await page.goto("/verify/conformance.html");
    await expect(page.locator("#out")).toHaveText("ready", { timeout: 30_000 });

    for (const c of doc.cases) {
      const actual = await page.evaluate((c) => window.__impsyModelCase!(c as never), c);
      expect(mismatches(actual, c.expected, doc.tolerance), c.name).toEqual([]);
    }
  });
});
