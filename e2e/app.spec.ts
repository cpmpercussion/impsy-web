import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { test, expect, openApp, setThreshold, faderAfter, expectIgnored } from "./midi";

const configFixture = fileURLToPath(new URL("./fixtures/e2e-config.toml", import.meta.url));

// Demo defaults (AiC preset): input dims 1–8 = CC 13–20 on ch1; output
// alternates Note On ch1–4 (dims 1,3,5,7) and CC 1–4 on ch11 (dims 2,4,6,8).

test("loads the demo model cross-origin isolated, with build info and no errors", async ({
  page,
  pageErrors,
}) => {
  await page.goto("./");
  await expect(page.getByText(/dim 9/)).toBeVisible({ timeout: 45_000 });
  expect(await page.evaluate(() => crossOriginIsolated)).toBe(true);
  await expect(page.locator("footer.app-footer")).toContainText(/v\d+\.\d+\.\d+ · [0-9a-f]{7}|unknown/);
  expect(pageErrors).toEqual([]);
});

test("CC input drives the mapped fader and holds CALL", async ({ page, midi }) => {
  await openApp(page);
  await setThreshold(page, 3);
  await expect.poll(() => faderAfter(page, 1, [0xb0, 13, 100])).toBeCloseTo(100 / 127, 2);
  await expect.poll(() => faderAfter(page, 3, [0xb0, 15, 32])).toBeCloseTo(32 / 127, 2);
  await expect(page.locator("header .badge")).toHaveText("CALL");
  // Unmapped CC (wrong number) and wrong channel are ignored.
  await expectIgnored(page, midi, [0xb0, 99, 0], [0xb1, 13, 0]);
});

test("Note On input decodes pitch, ignoring velocity and note-offs (#12)", async ({ page, midi }) => {
  await openApp(page);
  await setThreshold(page, 3);
  // Map input dim 1 to Note On (ch1) in the mapping editor.
  await page.locator("table.mappings select").first().selectOption("noteOn");
  await expect.poll(() => faderAfter(page, 1, [0x90, 64, 100])).toBeCloseTo(64 / 127, 2);
  // Velocity doesn't matter: the value is the pitch.
  await expect.poll(() => faderAfter(page, 1, [0x90, 100, 5])).toBeCloseTo(100 / 127, 2);
  // Note-offs (explicit, or Note On with velocity 0) are not inputs.
  await expectIgnored(page, midi, [0x90, 20, 0], [0x80, 20, 0]);
});

test("after the threshold, the model responds on the MIDI output", async ({ page, midi }) => {
  await openApp(page);
  await midi.fire([0xb0, 13, 64]);
  await midi.clearSent();
  // Default threshold is 0.1 s: silence hands over to the RNN.
  await expect(page.locator("header .badge")).toHaveText("RESPONSE");
  await expect.poll(async () => (await midi.sent()).length).toBeGreaterThan(4);
  const sent = await midi.sent();
  const notes = sent.filter(([s, , v]) => (s & 0xf0) === 0x90 && v > 0);
  const ccs = sent.filter(([s, n]) => s === 0xba && n >= 1 && n <= 4);
  expect(notes.length).toBeGreaterThan(0);
  expect(ccs.length).toBeGreaterThan(0);
  // Every note-on channel is one of the mapped output channels (ch1–4).
  for (const [s] of notes) expect(s & 0x0f).toBeLessThan(4);

  // Dashboard feedback (#9, #14): Last Output count and OUT rows in the console.
  const events = Number(await page.locator(".last-output .metric .v").first().textContent());
  expect(events).toBeGreaterThan(0);
  await expect(page.locator(".log li.out").first()).toBeVisible();
});

test("MIDI thru echoes input through the output mapping, and can be disabled", async ({
  page,
  midi,
}) => {
  await openApp(page);
  await setThreshold(page, 10); // stay in CALL: no RNN output during the test
  await midi.fire([0xb0, 13, 1]);
  await expect(page.locator("header .badge")).toHaveText("CALL");
  await midi.clearSent();

  // CC13 = 100 → dim 1 = 100/127 → output dim 1 is Note On ch1 → note 100.
  await midi.fire([0xb0, 13, 100]);
  await expect.poll(() => midi.sent()).toContainEqual([0x90, 100, 64]);

  await page.locator("#thru").uncheck();
  await midi.clearSent();
  await midi.fire([0xb0, 13, 50]);
  await page.waitForTimeout(200);
  expect(await midi.sent()).toEqual([]);
});

test("console and activity LED show incoming MIDI (#10, #14)", async ({ page, midi }) => {
  await openApp(page);
  await setThreshold(page, 3);
  await midi.fire([0xb0, 14, 42]);
  await expect(page.locator(".log li.in").first()).toContainText("CC ch1 #14=42");
  // The red (input) LED lights for ~40 ms on activity; catch it while firing.
  await expect
    .poll(async () => {
      await midi.fire([0xb0, 14, 43]);
      return page.locator(".act .led.lit").count();
    })
    .toBeGreaterThan(0);
});

test("imports an IMPSY .toml and exports it back", async ({ page, midi }) => {
  await openApp(page);
  await page.locator('input[type="file"][accept*=".toml"]').setInputFiles(configFixture);
  await expect(page.getByText("Imported")).toBeVisible();

  // Params applied.
  await expect(page.locator("#threshold")).toHaveValue("5");
  await expect(page.locator("#thru")).not.toBeChecked();

  // Input dim 1 is now Note On on ch2: a note there drives fader 1 by pitch.
  await expect.poll(() => faderAfter(page, 1, [0x91, 32, 90])).toBeCloseTo(32 / 127, 2);

  // Export round-trips the imported params and mappings.
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export .toml" }).click();
  const text = await readFile(await (await download).path(), "utf8");
  expect(text).toMatch(/threshold\s*=\s*5/);
  expect(text).toMatch(/input_thru\s*=\s*false/);
  expect(text).toMatch(/"note_on",\s*2/);
  expect(text).toMatch(/"control_change",\s*5,\s*47/);
});

test("dragging a fader injects input like MIDI in (UI rows, thru)", async ({ page, midi }) => {
  await openApp(page);
  await setThreshold(page, 3);
  await midi.clearSent();

  // Drag dim 2 (input CC14 ch1 → output CC1 on ch11) to ~75%.
  const track = page.locator(".track").nth(1);
  const box = (await track.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.75, box.y + box.height / 2, { steps: 8 });

  await expect(page.locator(".fader .val.user").first()).toBeVisible();
  await expect(page.locator(".log li.in").first()).toContainText("UI");
  await expect(page.locator(".log li.in").first()).toContainText("CC ch1 #14=");
  await page.mouse.up();

  // MIDI thru carried the drag to the output mapping (CC1 on ch11).
  const thru = (await midi.sent()).filter(([s, n]) => s === 0xba && n === 1);
  expect(thru.length).toBeGreaterThan(0);
  expect(thru.at(-1)![2]).toBeGreaterThan(80); // ~0.75 × 127 ≈ 95
});
