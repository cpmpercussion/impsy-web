import { describe, it, expect } from "vitest";
import {
  parseConfig,
  serializeConfig,
  ConfigParseError,
  SYNTHESIZED_DEVICE_NAME,
} from "./config";

// The bundled AiC config (../impsy/configs/AiC-charles-u6midipro.toml), inlined
// so the test doesn't depend on the sibling repo. Mirrors AUv3 IMPSYConfigTests.
const AIC_TOML = `
title = "RPi U6MIDI Pro: nanoKontrol Studio to notes and CCs"
owner = "Charles Martin"
description = "Uses nanoKontrol panning inputs and outputs to CCs"
log_input = true

[interaction]
mode = "callresponse"
threshold = 0.1
input_thru = true

[model]
dimension = 9
file = "models/musicMDRNN-dim9-layers2-units64-mixtures5-scale10.tflite"
size = "s"
sigmatemp = 0.01
pitemp = 1
timescale = 1

[midi]
in_device = ["U2MIDI Pro"]
out_device = ["U2MIDI Pro"]
input."U2MIDI Pro" = [
  ["control_change", 1, 13],
  ["control_change", 1, 14],
  ["control_change", 1, 15],
  ["control_change", 1, 16],
  ["control_change", 1, 17],
  ["control_change", 1, 18],
  ["control_change", 1, 19],
  ["control_change", 1, 20],
]
output."U2MIDI Pro" = [
  ["note_on", 1],
  ["control_change", 11, 1],
  ["note_on", 2],
  ["control_change", 11, 2],
  ["note_on", 3],
  ["control_change", 11, 3],
  ["note_on", 4],
  ["control_change", 11, 4],
]
`;

// Multi-device + CC range tuples (cf. roland-s-1-xtouch.toml). `in_device`
// order is honoured: S-1 first, X-TOUCH second.
const MULTI_DEVICE_TOML = `
[interaction]
threshold = 0.2
input_thru = false

[model]
dimension = 9

[midi]
in_device = ["S-1", "X-TOUCH"]
out_device = ["S-1", "X-TOUCH"]
feedback_protection = true
input."X-TOUCH" = [
  ["control_change", 11, 1],
  ["control_change", 11, 2],
]
input."S-1" = [
  ["note_on", 3],
  ["pitch_bend", 3],
]
output."S-1" = [
  ["note_on", 3],
]
output."X-TOUCH" = [
  ["control_change", 1, 9, 0, 13],
  ["control_change", 1, 10, 0, 13],
]
`;

describe("parseConfig", () => {
  it("parses params and mappings from the AiC config", () => {
    const c = parseConfig(AIC_TOML);

    expect(c.title).toBe("RPi U6MIDI Pro: nanoKontrol Studio to notes and CCs");
    expect(c.owner).toBe("Charles Martin");
    expect(c.threshold).toBeCloseTo(0.1, 6);
    expect(c.sigmaTemp).toBeCloseTo(0.01, 6);
    expect(c.piTemp).toBeCloseTo(1.0, 6);
    expect(c.timescale).toBeCloseTo(1.0, 6);
    expect(c.inputThru).toBe(true);
    expect(c.modelDimension).toBe(9);

    // 8 input CCs on ch1, numbers 13…20.
    expect(c.inputMappings).toHaveLength(8);
    c.inputMappings.forEach((m, i) => {
      expect(m.messageType).toBe("controlChange");
      expect(m.channel).toBe(1);
      expect(m.number).toBe(13 + i);
      expect(m.minValue).toBe(0);
      expect(m.maxValue).toBe(127);
      expect(m.id).toBe(i + 1);
      expect(m.enabled).toBe(true);
    });

    // 8 outputs alternating note_on (ch 1–4) / CC on ch 11 (numbers 1–4).
    expect(c.outputMappings).toHaveLength(8);
    c.outputMappings.forEach((m, i) => {
      if (i % 2 === 0) {
        expect(m.messageType).toBe("noteOn");
        expect(m.channel).toBe(i / 2 + 1);
        expect(m.number).toBe(60); // note_on default note
      } else {
        expect(m.messageType).toBe("controlChange");
        expect(m.channel).toBe(11);
        expect(m.number).toBe((i - 1) / 2 + 1);
      }
    });
  });

  it("honours in_device order and decodes CC range tuples", () => {
    const c = parseConfig(MULTI_DEVICE_TOML);

    expect(c.inputThru).toBe(false);
    expect(c.threshold).toBeCloseTo(0.2, 6);

    // S-1 listed first → its two entries lead, X-TOUCH's follow.
    expect(c.inputMappings.map((m) => m.messageType)).toEqual([
      "noteOn",
      "pitchBend",
      "controlChange",
      "controlChange",
    ]);
    expect(c.inputMappings[0].channel).toBe(3);

    // X-TOUCH output rings carry min/max = 0,13.
    const rings = c.outputMappings.slice(1);
    expect(rings).toHaveLength(2);
    rings.forEach((m, i) => {
      expect(m.messageType).toBe("controlChange");
      expect(m.channel).toBe(1);
      expect(m.number).toBe(9 + i);
      expect(m.minValue).toBe(0);
      expect(m.maxValue).toBe(13);
    });
  });

  it("falls back to defaults when sections are absent", () => {
    const c = parseConfig('title = "bare"');
    expect(c.threshold).toBeCloseTo(0.1, 6);
    expect(c.inputThru).toBe(true);
    expect(c.inputMappings).toEqual([]);
    expect(c.outputMappings).toEqual([]);
  });

  it("skips malformed mapping rows rather than throwing", () => {
    const c = parseConfig(`
[midi]
in_device = ["X"]
input."X" = [
  ["control_change", 1, 13],
  ["control_change", 1],
  ["bogus_type", 1, 2],
  ["note_on", 2],
]
`);
    // Only the valid CC and note_on survive; their ids re-pack to 1,2.
    expect(c.inputMappings.map((m) => m.messageType)).toEqual(["controlChange", "noteOn"]);
    expect(c.inputMappings.map((m) => m.id)).toEqual([1, 2]);
  });

  it("throws ConfigParseError on invalid TOML", () => {
    expect(() => parseConfig("this is = = not toml")).toThrow(ConfigParseError);
  });
});

describe("serializeConfig", () => {
  it("round-trips params and mappings through parse → serialize → parse", () => {
    const first = parseConfig(AIC_TOML);
    const second = parseConfig(serializeConfig(first));

    expect(second.title).toBe(first.title);
    expect(second.owner).toBe(first.owner);
    expect(second.threshold).toBeCloseTo(first.threshold, 6);
    expect(second.sigmaTemp).toBeCloseTo(first.sigmaTemp, 6);
    expect(second.piTemp).toBeCloseTo(first.piTemp, 6);
    expect(second.timescale).toBeCloseTo(first.timescale, 6);
    expect(second.inputThru).toBe(first.inputThru);
    expect(second.modelDimension).toBe(first.modelDimension);

    // Mappings collapse onto one synthetic device but the substantive fields
    // survive in order.
    expect(second.inputMappings.map((m) => m.messageType)).toEqual(
      first.inputMappings.map((m) => m.messageType),
    );
    expect(second.outputMappings.map((m) => m.channel)).toEqual(
      first.outputMappings.map((m) => m.channel),
    );
    expect(second.outputMappings.map((m) => m.number)).toEqual(
      first.outputMappings.map((m) => m.number),
    );
  });

  it("preserves CC range tuples on round-trip", () => {
    const first = parseConfig(MULTI_DEVICE_TOML);
    const second = parseConfig(serializeConfig(first));
    expect(second.outputMappings.map((m) => m.minValue)).toEqual(
      first.outputMappings.map((m) => m.minValue),
    );
    expect(second.outputMappings.map((m) => m.maxValue)).toEqual(
      first.outputMappings.map((m) => m.maxValue),
    );
  });

  it("collapses devices onto a single synthetic device key", () => {
    const toml = serializeConfig(parseConfig(MULTI_DEVICE_TOML));
    expect(toml).toContain(SYNTHESIZED_DEVICE_NAME);
    expect(toml).not.toContain("X-TOUCH");
    expect(toml).not.toContain("S-1");
  });

  it("preserves unknown sections across a round-trip", () => {
    const toml = `
[interaction]
threshold = 0.2

[midi]
in_device = ["X"]
out_device = ["X"]
input."X" = []
output."X" = []

[osc]
server_ip = "0.0.0.0"
server_port = 6000
`;
    const out = serializeConfig(parseConfig(toml));
    expect(out).toContain("[osc]");
    expect(out).toContain("server_port");
  });

  it("emits valid TOML from a fresh config built in memory", () => {
    const out = serializeConfig({
      title: "exported",
      threshold: 0.25,
      sigmaTemp: 0.05,
      piTemp: 1.2,
      timescale: 1.5,
      inputThru: false,
      inputMappings: [
        { id: 1, messageType: "controlChange", channel: 1, number: 74, minValue: 0, maxValue: 127, enabled: true },
        { id: 2, messageType: "noteOn", channel: 2, number: 60, minValue: 0, maxValue: 127, enabled: true },
      ],
      outputMappings: [
        { id: 1, messageType: "pitchBend", channel: 3, number: 0, minValue: 0, maxValue: 127, enabled: true },
      ],
    });
    const reparsed = parseConfig(out);
    expect(reparsed.title).toBe("exported");
    expect(reparsed.threshold).toBeCloseTo(0.25, 6);
    expect(reparsed.inputThru).toBe(false);
    expect(reparsed.inputMappings.map((m) => m.messageType)).toEqual(["controlChange", "noteOn"]);
    expect(reparsed.outputMappings[0].messageType).toBe("pitchBend");
    expect(reparsed.outputMappings[0].channel).toBe(3);
  });
});
