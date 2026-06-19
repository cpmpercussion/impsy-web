// Drives verify/inference.html in real Chrome (Playwright) and prints the
// result. Assumes the dev server is running on http://localhost:5173.
import { chromium } from "playwright";

const browser = await chromium.launch({ channel: "chrome" });
const page = await browser.newPage();
page.on("console", (m) => console.log("[page]", m.text()));
await page.goto("http://localhost:5173/verify/inference.html");

const result = await page.waitForFunction(() => window.__impsy_result, null, {
  timeout: 30000,
});
console.log(JSON.stringify(await result.jsonValue(), null, 2));
await browser.close();
