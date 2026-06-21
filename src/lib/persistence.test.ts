import { describe, it, expect, beforeEach } from "vitest";
import { loadState, saveState, STORAGE_KEY } from "./persistence";
import { ParameterDefaults } from "./impsy/constants";
import { defaultMappingSet } from "./impsy/midiMapping";

// A minimal localStorage stand-in for the test environment.
function installStorage(): Storage {
  const map = new Map<string, string>();
  const store: Storage = {
    get length() {
      return map.size;
    },
    clear: () => map.clear(),
    getItem: (k) => map.get(k) ?? null,
    key: (i) => [...map.keys()][i] ?? null,
    removeItem: (k) => void map.delete(k),
    setItem: (k, v) => void map.set(k, v),
  };
  (globalThis as { localStorage: Storage }).localStorage = store;
  return store;
}

const SAMPLE = () => ({
  params: { ...ParameterDefaults, threshold: 0.42, inputThru: false },
  mappings: { dimension: 9, ...defaultMappingSet(9) },
  selectedInputIds: ["in-1"],
  selectedOutputIds: ["out-1", "out-2"],
  recording: true,
});

describe("persistence", () => {
  beforeEach(() => installStorage());

  it("round-trips a full state", () => {
    saveState(SAMPLE());
    const s = loadState();
    expect(s?.params.threshold).toBe(0.42);
    expect(s?.params.inputThru).toBe(false);
    expect(s?.mappings?.dimension).toBe(9);
    expect(s?.selectedInputIds).toEqual(["in-1"]);
    expect(s?.selectedOutputIds).toEqual(["out-1", "out-2"]);
    expect(s?.recording).toBe(true);
  });

  it("returns null when nothing is stored", () => {
    expect(loadState()).toBeNull();
  });

  it("discards a blob with an unknown schema version", () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 999, params: SAMPLE().params }));
    expect(loadState()).toBeNull();
  });

  it("discards malformed JSON and malformed params", () => {
    localStorage.setItem(STORAGE_KEY, "{not json");
    expect(loadState()).toBeNull();
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 1, params: { threshold: "x" } }));
    expect(loadState()).toBeNull();
  });

  it("drops invalid mappings/device fields but keeps valid params", () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: 1,
        params: SAMPLE().params,
        mappings: { dimension: "nope" },
        selectedInputIds: [1, 2],
        recording: "yes",
      }),
    );
    const s = loadState();
    expect(s?.mappings).toBeNull();
    expect(s?.selectedInputIds).toEqual([]);
    expect(s?.recording).toBe(false);
  });
});
