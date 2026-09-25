// Test fixture: a fake Web MIDI device pair installed before any page script
// runs. One input ("Mock Controller") that tests fire messages into, and one
// output ("Mock Synth") that records everything the app sends.
import { test as base, expect, type Page } from "@playwright/test";

declare global {
  interface Window {
    __fireMIDI: (bytes: number[]) => void;
    __midiSent: number[][];
  }
}

export class MockMIDI {
  constructor(private page: Page) {}

  /** Deliver a message on the mock input. */
  async fire(...messages: number[][]): Promise<void> {
    await this.page.evaluate((msgs) => msgs.forEach((m) => window.__fireMIDI(m)), messages);
  }

  /** Everything sent to the mock output so far. */
  async sent(): Promise<number[][]> {
    return this.page.evaluate(() => window.__midiSent.map((m) => [...m]));
  }

  async clearSent(): Promise<void> {
    await this.page.evaluate(() => (window.__midiSent.length = 0));
  }
}

export interface AppFixtures {
  midi: MockMIDI;
  pageErrors: string[];
}

export const test = base.extend<AppFixtures>({
  pageErrors: async ({ page }, use) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    await use(errors);
  },
  midi: async ({ page }, use) => {
    await page.addInitScript(() => {
      const input = {
        id: "mock-in",
        name: "Mock Controller",
        onmidimessage: null as ((e: { data: Uint8Array }) => void) | null,
      };
      window.__midiSent = [];
      const output = {
        id: "mock-out",
        name: "Mock Synth",
        send: (bytes: number[]) => window.__midiSent.push(Array.from(bytes)),
      };
      const access = {
        inputs: new Map([[input.id, input]]),
        outputs: new Map([[output.id, output]]),
        onstatechange: null,
      };
      Object.defineProperty(navigator, "requestMIDIAccess", {
        value: async () => access,
        configurable: true,
      });
      window.__fireMIDI = (bytes) => input.onmidimessage?.({ data: new Uint8Array(bytes) });
    });
    await use(new MockMIDI(page));
  },
});

export { expect };

/**
 * Open the app, wait for the bundled demo model to auto-load, and grant the
 * mock MIDI devices (first input/output are auto-selected).
 */
export async function openApp(page: Page): Promise<void> {
  await page.goto("./");
  await expect(page.getByText(/dim 9/)).toBeVisible({ timeout: 45_000 });
  await page.getByRole("button", { name: "Enable Web MIDI" }).click();
  await expect(page.getByText("Mock Controller")).toBeVisible();
}

/** Slider/number/checkbox helpers for the Settings pane. */
export async function setThreshold(page: Page, seconds: number): Promise<void> {
  await page.locator("#threshold").fill(String(seconds));
}

/**
 * Fire `message` and read fader `dimension` 50 ms later, in one browser-side
 * step. A fader only shows user input for ~250 ms before settling back to the
 * model's value, so reading it in a separate round-trip can miss the window on
 * a busy machine. Use with `expect.poll`, which re-fires on each attempt.
 */
export async function faderAfter(page: Page, dimension: number, message: number[]): Promise<number> {
  return page.evaluate(
    async ([msg, dim]) => {
      window.__fireMIDI(msg);
      await new Promise((r) => setTimeout(r, 50));
      return Number(document.querySelectorAll(".fader .val")[dim - 1]?.textContent);
    },
    [message, dimension] as const,
  );
}

/**
 * Assert that `messages` are ignored by the input decoder: once the faders
 * have settled back from any earlier input (~250 ms), firing them must not
 * snap any fader into the red user-input state.
 */
export async function expectIgnored(page: Page, midi: MockMIDI, ...messages: number[][]): Promise<void> {
  const userFaders = page.locator(".fader .val.user");
  await expect(userFaders).toHaveCount(0);
  await midi.fire(...messages);
  await page.waitForTimeout(150);
  await expect(userFaders).toHaveCount(0);
}
