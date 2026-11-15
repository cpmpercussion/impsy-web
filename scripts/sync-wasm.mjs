// Copies the WASM runtimes the app serves locally (no CDN) into public/:
//   LiteRT.js (inference)          → public/litert-wasm/
//   TF.js WASM backend (training)  → public/tfjs-wasm/
// Run automatically before dev/build.
import { cp, mkdir, readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

const litertSrc = join(root, "node_modules", "@litertjs", "core", "wasm");
const litertDest = join(root, "public", "litert-wasm");
await mkdir(litertDest, { recursive: true });
await cp(litertSrc, litertDest, { recursive: true });
console.log(`Copied LiteRT WASM → ${litertDest}`);

const tfjsSrc = join(root, "node_modules", "@tensorflow", "tfjs-backend-wasm", "dist");
const tfjsDest = join(root, "public", "tfjs-wasm");
await mkdir(tfjsDest, { recursive: true });
for (const file of await readdir(tfjsSrc)) {
  if (file.endsWith(".wasm")) await cp(join(tfjsSrc, file), join(tfjsDest, file));
}
console.log(`Copied TF.js WASM → ${tfjsDest}`);
