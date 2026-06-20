// Application state + orchestration (Svelte 5 runes).
//
// The web analog of IMPSYViewModel.swift: bridges Web MIDI + InteractionEngine
// to reactive UI state. Single shared instance, imported by the components.

import { WebMIDI, type MIDIPort } from "./midi/webMidi";
import { InteractionEngine, type CallResponseState } from "./impsy/interactionEngine";
import { TFLiteRNN, initLiteRt, type ModelConfig } from "./impsy/tfliteRnn";
import {
  type MIDIMappingSet,
  defaultMappingSet,
  aicU6MIDIProDefault,
} from "./impsy/midiMapping";
import { encodeSingle } from "./impsy/midiMapper";
import { ParameterDefaults } from "./impsy/constants";

// Base-relative so it resolves under the GitHub Pages subpath (/impsy-web/)
// as well as at dev root (/).
const WASM_PATH = `${import.meta.env.BASE_URL}litert-wasm/`;

export type ModelStatus = "none" | "loading" | "ready" | "error";

export interface Params {
  threshold: number;
  sigmaTemp: number;
  piTemp: number;
  timescale: number;
  inputThru: boolean;
}

export class IMPSYApp {
  private midi = new WebMIDI();
  private engine: InteractionEngine | null = null;

  // ── Reactive UI state ────────────────────────────────────────────────────
  midiSupported = $state(WebMIDI.isSupported());
  midiGranted = $state(false);
  inputs = $state<MIDIPort[]>([]);
  outputs = $state<MIDIPort[]>([]);
  // Multiple devices may be active on each side at once (see WebMIDI). A
  // selected id that's currently absent (unplugged) is kept so it reconnects.
  selectedInputIds = $state<string[]>([]);
  selectedOutputIds = $state<string[]>([]);

  modelName = $state<string | null>(null);
  modelConfig = $state<ModelConfig | null>(null);
  modelStatus = $state<ModelStatus>("none");
  errorMessage = $state<string | null>(null);

  callState = $state<CallResponseState>("CALL");
  mappings = $state<MIDIMappingSet>(defaultMappingSet(9));

  params = $state<Params>({ ...ParameterDefaults });

  // Per-dimension fader state (one entry per user dimension = dimension - 1).
  // `outputValues` is the last value the RNN emitted for each dimension (the
  // green bar position); `inputValues` is the last user value (red). The
  // `*Triggers` counters bump on each event so the faders/LEDs can flash and
  // snap — mirrors IMPSYViewModel's outputValues/inputValues/*DimensionCounts.
  outputValues = $state<number[]>([]);
  inputValues = $state<number[]>([]);
  inputTriggers = $state<number[]>([]);
  outputTriggers = $state<number[]>([]);

  // ── MIDI access ───────────────────────────────────────────────────────────
  async requestMIDI(): Promise<void> {
    try {
      await this.midi.requestAccess(false);
      this.midiGranted = true;
      this.midi.onPortsChanged = () => this.refreshPorts();
      this.refreshPorts();
    } catch (err) {
      this.errorMessage = err instanceof Error ? err.message : String(err);
    }
  }

  private refreshPorts(): void {
    this.inputs = this.midi.inputs;
    this.outputs = this.midi.outputs;
    // On first grant (nothing chosen yet) auto-select the first device on each
    // side, matching the previous single-device behaviour.
    if (this.selectedInputIds.length === 0 && this.inputs[0]) this.selectedInputIds = [this.inputs[0].id];
    if (this.selectedOutputIds.length === 0 && this.outputs[0]) this.selectedOutputIds = [this.outputs[0].id];
    // Rebind: a port may have (dis)appeared, so re-attach handlers/outputs.
    this.syncMidi();
  }

  /** Push the current selection arrays into the WebMIDI layer. */
  private syncMidi(): void {
    this.midi.setInputs(this.selectedInputIds, (data) => this.engine?.enqueueInput(data));
    this.midi.setOutputs(this.selectedOutputIds);
  }

  toggleInput(id: string, on: boolean): void {
    const set = new Set(this.selectedInputIds);
    if (on) set.add(id);
    else set.delete(id);
    this.selectedInputIds = [...set];
    this.syncMidi();
  }

  toggleOutput(id: string, on: boolean): void {
    const set = new Set(this.selectedOutputIds);
    if (on) set.add(id);
    else set.delete(id);
    this.selectedOutputIds = [...set];
    this.syncMidi();
  }

  // ── Model loading ───────────────────────────────────────────────────────
  async loadModelFromFile(file: File): Promise<void> {
    const buf = new Uint8Array(await file.arrayBuffer());
    await this.loadModelBytes(buf, file.name);
  }

  async loadModelFromUrl(url: string): Promise<void> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch model: ${res.status}`);
    const buf = new Uint8Array(await res.arrayBuffer());
    await this.loadModelBytes(buf, url.split("/").pop() ?? url);
  }

  private async loadModelBytes(bytes: Uint8Array, name: string): Promise<void> {
    this.modelStatus = "loading";
    this.errorMessage = null;
    try {
      await initLiteRt(WASM_PATH);
      const rnn = await TFLiteRNN.create(bytes, "wasm");
      const config = rnn.config;

      // Default mapping: the 9-dim AiC preset when it fits, else generic.
      this.mappings =
        config.dimension === 9 ? aicU6MIDIProDefault() : defaultMappingSet(config.dimension);

      // Reset per-dimension fader state to the user dimension count.
      const userDims = Math.max(0, config.dimension - 1);
      this.outputValues = new Array(userDims).fill(0);
      this.inputValues = new Array(userDims).fill(0);
      this.inputTriggers = new Array(userDims).fill(0);
      this.outputTriggers = new Array(userDims).fill(0);

      this.ensureEngine();
      this.engine!.updateMappings(this.mappings);
      this.engine!.loadModel(rnn);

      this.modelName = name;
      this.modelConfig = config;
      this.modelStatus = "ready";
    } catch (err) {
      this.modelStatus = "error";
      this.errorMessage = err instanceof Error ? err.message : String(err);
      console.error("[IMPSY] model load failed:", err);
    }
  }

  clearModel(): void {
    this.engine?.clearModel();
    this.modelName = null;
    this.modelConfig = null;
    this.modelStatus = "none";
    this.outputValues = [];
    this.inputValues = [];
    this.inputTriggers = [];
    this.outputTriggers = [];
  }

  /**
   * Inject a normalised value (0…1) for a user input dimension as if its
   * configured MIDI message had just arrived — so dragging a fader drives the
   * same pipeline as real MIDI in. Mirrors IMPSYViewModel.injectInput.
   */
  injectInput(dimensionIndex: number, value: number): void {
    const mapping = this.mappings.inputMappings[dimensionIndex];
    if (!mapping || !this.engine) return;
    this.engine.enqueueInput(encodeSingle(value, mapping).bytes);
  }

  resetStates(): void {
    this.engine?.resetLSTMStates();
  }

  // ── Parameters & mappings ──────────────────────────────────────────────────
  setParam<K extends keyof Params>(key: K, value: Params[K]): void {
    this.params = { ...this.params, [key]: value };
    if (this.engine) (this.engine as unknown as Record<string, unknown>)[key] = value;
  }

  setMappings(mappings: MIDIMappingSet): void {
    this.mappings = mappings;
    this.engine?.updateMappings(mappings);
  }

  private ensureEngine(): void {
    if (this.engine) return;
    const engine = new InteractionEngine(this.mappings, (bytes) => this.midi.send(bytes));
    engine.threshold = this.params.threshold;
    engine.sigmaTemp = this.params.sigmaTemp;
    engine.piTemp = this.params.piTemp;
    engine.timescale = this.params.timescale;
    engine.inputThru = this.params.inputThru;
    engine.onStateChanged = (s) => (this.callState = s);
    engine.onUserInputReceived = (dim, value) => {
      if (dim >= 0 && dim < this.inputValues.length) {
        this.inputValues[dim] = value;
        this.inputTriggers[dim]++;
      }
    };
    engine.onEventGenerated = (_dt, _events, values) => {
      for (let i = 0; i < values.length && i < this.outputValues.length; i++) {
        this.outputValues[i] = values[i];
        this.outputTriggers[i]++;
      }
    };
    engine.start();
    this.engine = engine;
  }
}

export const app = new IMPSYApp();
