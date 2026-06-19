// Copies the LiteRT.js WASM runtime into public/litert-wasm/ so the app can
// serve it locally (no CDN). Run automatically before dev/build.
import { cp, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "node_modules", "@litertjs", "core", "wasm");
const dest = join(root, "public", "litert-wasm");

await mkdir(dest, { recursive: true });
await cp(src, dest, { recursive: true });
console.log(`Copied LiteRT WASM → ${dest}`);
