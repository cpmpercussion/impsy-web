import { readFile } from "node:fs/promises";
import { test, expect, openApp } from "./midi";

// In-browser training (issue #1): .log → TF.js training in a worker →
// tfliteWriter → the model loads in LiteRT and plays. Runs against the
// production build too, where the worker and its WASM are bundled under
// /impsy-web/.

/** A 3-dimensional session log: two slow sine waves, one event every 100 ms. */
function sineLog(events: number): string {
  const lines = ["# IMPSY Web log"];
  for (let i = 0; i < events; i++) {
    const ms = i * 100;
    const stamp = `2026-09-26T10:${String(Math.floor(ms / 60000)).padStart(2, "0")}:${String(
      Math.floor(ms / 1000) % 60,
    ).padStart(2, "0")}.${String(ms % 1000).padStart(3, "0")}000`;
    const a = 0.5 + 0.4 * Math.sin(i / 5);
    const b = 0.5 + 0.4 * Math.cos(i / 7);
    lines.push(`${stamp},interface,${a},${b}`);
    if (i % 4 === 0) lines.push(`${stamp},rnn,0.1,0.9`);
  }
  return lines.join("\n") + "\n";
}

// `midi` installs the fake Web MIDI devices openApp() waits for.
test("trains a model from a .log, loads it, and offers the .tflite", async ({
  page,
  midi: _midi,
  pageErrors,
}) => {
  test.setTimeout(180_000);
  // The worker warns if it had to fall back from the WASM backend to the CPU one.
  const warnings: string[] = [];
  page.on("console", (m) => m.type() === "warning" && warnings.push(m.text()));
  await openApp(page);

  await page.locator("#train-files").setInputFiles({
    name: "2026-09-26T10-00-00-3d-mdrnn.log",
    mimeType: "text/plain",
    buffer: Buffer.from(sineLog(120)),
  });
  await page.locator("#train-size").selectOption("xxs");
  await page.locator("#train-epochs").fill("2");
  await page.getByRole("button", { name: "Train model" }).click();

  const name = "musicMDRNN-dim3-layers2-units16-mixtures5-scale10.tflite";
  await expect(page.getByText(`Trained after 2 epochs; loaded ${name}.`)).toBeVisible({
    timeout: 150_000,
  });
  // The trained model is now the running model.
  await expect(page.locator(".hint", { hasText: "dim 3 · 2 layers · 16 units" })).toBeVisible();

  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download .tflite" }).click();
  const file = await download;
  expect(file.suggestedFilename()).toBe(name);
  const bytes = await readFile(await file.path());
  expect(bytes.subarray(4, 8).toString("latin1")).toBe("TFL3");
  expect(pageErrors).toEqual([]);
  expect(warnings.filter((w) => w.includes("WASM backend unavailable"))).toEqual([]);
});

test("explains when the logs are too short to train on", async ({ page, midi: _midi }) => {
  await openApp(page);
  await page.locator("#train-files").setInputFiles({
    name: "short-3d-mdrnn.log",
    mimeType: "text/plain",
    buffer: Buffer.from(sineLog(20)),
  });
  await page.getByRole("button", { name: "Train model" }).click();
  await expect(
    page.getByText(/Not enough data: training needs a log with more than 52/),
  ).toBeVisible();
});
