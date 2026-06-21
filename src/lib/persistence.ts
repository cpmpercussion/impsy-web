// Settings persistence (issue #4).
//
// The browser analog of the AUv3 `fullState` session dictionary: parameters,
// MIDI mappings, device selection, and the recording toggle are written to
// localStorage and rehydrated on startup so a session reopens as it was left.
//
// Model files themselves can't be persisted (no security-scoped bookmarks in
// the browser), so mappings are tagged with the dimension they were authored
// for and only re-applied to a model of the same dimension.
//
// The payload is versioned: a stored blob with an unknown version (or any
// parse/shape error) is discarded rather than half-applied.

import type { MIDIMappingSet } from "./impsy/midiMapping";
import type { Params } from "./appState.svelte";

const STORAGE_KEY = "impsy-web:state";
const SCHEMA_VERSION = 1;

/** Mappings plus the model dimension they belong to (= inputMappings.length + 1). */
export interface PersistedMappings extends MIDIMappingSet {
  dimension: number;
}

export interface PersistedState {
  version: number;
  params: Params;
  mappings: PersistedMappings | null;
  selectedInputIds: string[];
  selectedOutputIds: string[];
  recording: boolean;
}

/** Best-effort localStorage handle — null in environments without one. */
function storage(): Storage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    // Access can throw under strict privacy settings.
    return null;
  }
}

function isParams(v: unknown): v is Params {
  if (typeof v !== "object" || v === null) return false;
  const p = v as Record<string, unknown>;
  return (
    typeof p.threshold === "number" &&
    typeof p.sigmaTemp === "number" &&
    typeof p.piTemp === "number" &&
    typeof p.timescale === "number" &&
    typeof p.inputThru === "boolean"
  );
}

function isStringArray(v: unknown): v is string[] {
  return Array.isArray(v) && v.every((x) => typeof x === "string");
}

function isMappings(v: unknown): v is PersistedMappings {
  if (typeof v !== "object" || v === null) return false;
  const m = v as Record<string, unknown>;
  return (
    typeof m.dimension === "number" &&
    Array.isArray(m.inputMappings) &&
    Array.isArray(m.outputMappings)
  );
}

/** Read and validate persisted state; null if absent, unreadable, or stale. */
export function loadState(): PersistedState | null {
  const store = storage();
  if (!store) return null;
  const raw = store.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    const data = JSON.parse(raw) as Record<string, unknown>;
    if (data.version !== SCHEMA_VERSION) return null;
    if (!isParams(data.params)) return null;
    return {
      version: SCHEMA_VERSION,
      params: data.params,
      mappings: isMappings(data.mappings) ? data.mappings : null,
      selectedInputIds: isStringArray(data.selectedInputIds) ? data.selectedInputIds : [],
      selectedOutputIds: isStringArray(data.selectedOutputIds) ? data.selectedOutputIds : [],
      recording: typeof data.recording === "boolean" ? data.recording : false,
    };
  } catch {
    return null;
  }
}

/** Write the current state. Silently no-ops if localStorage is unavailable. */
export function saveState(state: Omit<PersistedState, "version">): void {
  const store = storage();
  if (!store) return;
  try {
    store.setItem(STORAGE_KEY, JSON.stringify({ version: SCHEMA_VERSION, ...state }));
  } catch {
    // Quota or privacy errors are non-fatal; the app still works in-session.
  }
}

export { STORAGE_KEY, SCHEMA_VERSION };
