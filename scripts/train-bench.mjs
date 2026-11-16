// Runs verify/train-bench.html in Chromium via Playwright and prints the table.
// Needs the dev server running:  node scripts/train-bench.mjs [query] [base-url]
//   node scripts/train-bench.mjs "backends=wasm&sizes=xxs,s&threads=1,4"
import { chromium } from "@playwright/test";

const query = process.argv[2] ?? "";
const base = process.argv[3] ?? "http://localhost:5173";
const browser = await chromium.launch({
  executablePath: process.env.PW_CHROMIUM_PATH || undefined,
  channel: process.env.PW_CHANNEL || undefined,
  headless: process.env.HEADED ? false : true,
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("page error:", e.message));
await page.goto(`${base}/verify/train-bench.html?${query}`);
await page.waitForFunction(() => window.__bench_done, null, { timeout: 30 * 60_000 });
console.log(await page.locator("#out").textContent());
await browser.close();
