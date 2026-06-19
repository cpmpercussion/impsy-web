// Verifies external MIDI input lights up the faders. Installs a fake Web MIDI
// device (no real hardware needed), loads the demo model, fires CC messages on
// the device, and checks the matching faders snap red to the received values.
// Needs `npm run dev` running. Usage: node scripts/verify-midi-input.mjs [port]
import { chromium } from "playwright";

const port = process.argv[2] || "5173";
const browser = await chromium.launch({ channel: "chrome" });
const page = await browser.newPage();
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(String(e)));

// Mock navigator.requestMIDIAccess with one input device + a window hook to
// fire messages, installed before any page script runs.
await page.addInitScript(() => {
  const input = {
    id: "mock-in",
    name: "Mock Controller",
    onmidimessage: null,
    addEventListener(t, cb) {
      if (t === "midimessage") this.onmidimessage = cb;
    },
  };
  const access = {
    inputs: new Map([["mock-in", input]]),
    outputs: new Map(),
    onstatechange: null,
  };
  Object.defineProperty(navigator, "requestMIDIAccess", {
    value: async () => access,
    configurable: true,
  });
  // @ts-ignore test hook
  window.__fireMIDI = (bytes) => input.onmidimessage?.({ data: new Uint8Array(bytes) });
});

await page.goto(`http://localhost:${port}/`);

// Enable MIDI (mock grants instantly) and load the demo model.
await page.getByRole("button", { name: "Enable Web MIDI" }).click();
await page.getByRole("button", { name: "Load demo model" }).click();
await page.getByText(/dim 9/).waitFor({ timeout: 30000 });

// The demo's input mappings are CC 13–20 on channel 1 → dimensions 1–8.
// Fire CC13=100 (dim 1 → 0.79) and CC15=32 (dim 3 → 0.25), repeatedly to stay
// in CALL mode, then sample the fader readouts.
const samples = [];
for (let i = 0; i < 4; i++) {
  await page.evaluate(() => {
    window.__fireMIDI([0xb0, 13, 100]); // CC13 = 100/127 ≈ 0.79
    window.__fireMIDI([0xb0, 15, 32]); //  CC15 = 32/127  ≈ 0.25
  });
  await page.waitForTimeout(40);
}

const vals = await page.locator(".fader .val").allTextContents();
// A fader is "user-driven red" when its readout carries the .user class.
const redCount = await page.locator(".fader .val.user").count();

console.log(
  JSON.stringify(
    {
      dim1: vals[0],
      dim3: vals[2],
      redCount,
      expectedDim1: "~0.79",
      expectedDim3: "~0.25",
      errors,
    },
    null,
    2,
  ),
);
await browser.close();
