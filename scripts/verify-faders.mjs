// Drives the full UI in real Chrome: loads the demo model, drags a dimension
// fader (which injects MIDI input into the engine), then pauses and checks the
// model responds and the bars reflect its output. Needs `npm run dev` running.
import { chromium } from "playwright";

const browser = await chromium.launch({ channel: "chrome" });
const page = await browser.newPage();
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(String(e)));

await page.goto("http://localhost:5173/");

// Load the bundled demo model.
await page.getByRole("button", { name: "Load demo model" }).click();
await page.getByText(/dim 9/).waitFor({ timeout: 30000 });

const faders = page.locator(".track");
const count = await faders.count();

// Drag the first fader to ~75% — this injects input into the engine.
const box = await faders.first().boundingBox();
await page.mouse.move(box.x + box.width * 0.2, box.y + box.height / 2);
await page.mouse.down();
await page.mouse.move(box.x + box.width * 0.75, box.y + box.height / 2, { steps: 8 });
await page.mouse.up();

// Watch the CALL/RESPONSE badge flip to RESPONSE after the pause.
let sawResponse = false;
for (let i = 0; i < 40; i++) {
  const badge = (await page.locator(".badge").textContent())?.trim();
  if (badge === "RESPONSE") sawResponse = true;
  await page.waitForTimeout(50);
}

// Read the bar value readouts; after a response some should be non-zero.
const vals = await page.locator(".fader .val").allTextContents();
const nonZero = vals.filter((v) => v !== "0.00").length;

console.log(
  JSON.stringify(
    { faderCount: count, sawResponse, vals, nonZero, errors },
    null,
    2,
  ),
);
await browser.close();
