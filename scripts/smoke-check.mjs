// Post-deploy smoke check (issue #20): confirm a deployed site serves the app,
// the cross-origin-isolation service worker, the LiteRT WASM runtime and the
// demo model — and, given a commit hash, that it's the expected build.
//
// Usage: node scripts/smoke-check.mjs <baseUrl> [shortCommitHash]
//   node scripts/smoke-check.mjs https://cpmpercussion.github.io/impsy-web/ a1b2c3d
//   node scripts/smoke-check.mjs http://localhost:4173/impsy-web/   # vite preview
//
// Retries for a while so a just-finished Pages deploy has time to propagate.

const [base, expectedHash] = process.argv.slice(2);
if (!base) {
  console.error("usage: node scripts/smoke-check.mjs <baseUrl> [shortCommitHash]");
  process.exit(2);
}
const baseUrl = base.endsWith("/") ? base : `${base}/`;
const attempts = Number(process.env.SMOKE_ATTEMPTS ?? 6);
const delayMs = Number(process.env.SMOKE_DELAY_MS ?? 15_000);

const ASSETS = [
  { path: "coi-serviceworker.js", type: "javascript" },
  { path: "litert-wasm/litert_wasm_internal.wasm", type: "application/wasm" },
  { path: "litert-wasm/litert_wasm_internal.js", type: "javascript" },
  {
    path: "models/musicMDRNN-dim9-layers2-units64-mixtures5-scale10.tflite",
    minBytes: 100_000,
  },
];

async function get(path) {
  // Cache-bust so a CDN copy of the previous deploy can't satisfy the check.
  const url = new URL(path, baseUrl);
  url.searchParams.set("smoke", String(Date.now()));
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${url.pathname}`);
  return res;
}

async function check() {
  const html = await (await get("")).text();
  if (!html.includes("coi-serviceworker.js"))
    throw new Error("index.html lacks coi-serviceworker.js");
  const script = html.match(/<script[^>]+type="module"[^>]+src="([^"]+)"/)?.[1];
  if (!script) throw new Error("index.html has no module script");
  const js = await (await get(script)).text(); // absolute path, resolved against the origin
  if (expectedHash && !js.includes(expectedHash)) {
    throw new Error(`app bundle is not build ${expectedHash} (stale deploy?)`);
  }
  for (const a of ASSETS) {
    const res = await get(a.path);
    const type = res.headers.get("content-type") ?? "";
    if (a.type && !type.includes(a.type)) throw new Error(`${a.path}: content-type ${type}`);
    const bytes = (await res.arrayBuffer()).byteLength;
    if (a.minBytes && bytes < a.minBytes) throw new Error(`${a.path}: only ${bytes} bytes`);
  }
  return script;
}

for (let i = 1; i <= attempts; i++) {
  try {
    const script = await check();
    console.log(`✓ ${baseUrl} ok (${script}${expectedHash ? `, build ${expectedHash}` : ""})`);
    process.exit(0);
  } catch (err) {
    console.log(`attempt ${i}/${attempts}: ${err.message}`);
    if (i < attempts) await new Promise((r) => setTimeout(r, delayMs));
  }
}
console.error(`✗ smoke check failed for ${baseUrl}`);
process.exit(1);
