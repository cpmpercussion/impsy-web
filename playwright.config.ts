import { defineConfig, devices } from "@playwright/test";

// End-to-end browser tests (issue #18). A fake Web MIDI device (e2e/midi.ts)
// stands in for hardware, so the whole path — MIDI in → engine → model → MIDI
// out — runs headless in CI.
//
// Two targets: the production build via `vite preview` (served under
// /impsy-web/ with coi-serviceworker.js, like GitHub Pages) and the dev server
// (headers from vite.config.ts). Run one with `--project=preview` / `dev`.
//
// Set PW_CHROMIUM_PATH to use a preinstalled Chromium instead of Playwright's
// downloaded one.
const executablePath = process.env.PW_CHROMIUM_PATH || undefined;

export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  use: {
    ...devices["Desktop Chrome"],
    launchOptions: { executablePath },
    trace: "retain-on-failure",
  },
  projects: [
    { name: "preview", use: { baseURL: "http://localhost:4173/impsy-web/" } },
    { name: "dev", use: { baseURL: "http://localhost:5173/" } },
  ],
  webServer: [
    {
      command: "npm run build && npx vite preview --port 4173 --strictPort",
      url: "http://localhost:4173/impsy-web/",
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
    },
    {
      command: "npm run dev -- --port 5173 --strictPort",
      url: "http://localhost:5173/",
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
  ],
});
