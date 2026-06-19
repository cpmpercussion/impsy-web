// InteractionEngine — port of InteractionEngine.swift.
//
// Coordinates call-and-response between MIDI input and the TFLite RNN. On the
// web everything runs on the main thread (Web MIDI callbacks + timers share it),
// so the AUv3 ring buffers collapse to a plain queue and the inference queue
// becomes the main event loop. The state-machine logic is otherwise faithful.
//
//   CALL     — user is playing; the RNN listens (advances LSTM state).
//   RESPONSE — user paused past `threshold`; the RNN feeds itself, each
//              prediction scheduling the next after its own predicted dt.

import { MIDIMapper, type MIDIEvent } from "./midiMapper";
import type { MIDIMappingSet } from "./midiMapping";
import type { TFLiteRNN } from "./tfliteRnn";
import {
  ParameterDefaults,
  MINIMUM_DELTA_TIME,
  RESPONSE_LOOP_MIN_DT,
} from "./constants";

export type CallResponseState = "CALL" | "RESPONSE";

const nowSeconds = () => performance.now() / 1000;

/** dt ≈ 0.0075–0.0125 s, remaining dims random [0,1) — mirrors random_sample(). */
export function randomInitialSample(dimension: number): number[] {
  if (dimension <= 0) return [];
  const s = Array.from({ length: dimension }, () => Math.random());
  s[0] = 0.01 + (Math.random() - 0.5) * 0.005;
  return s;
}

export class InteractionEngine {
  private mapper: MIDIMapper;
  private rnn: TFLiteRNN | null = null;
  private sendMIDI: (bytes: number[]) => void;

  private inputQueue: (Uint8Array | number[])[] = [];
  private timer: ReturnType<typeof setInterval> | null = null;
  private responseTimer: ReturnType<typeof setTimeout> | null = null;

  // Dense input vector (0-based: index 0 → dimension 1), length dimension-1.
  private inputVector: number[] = [];
  // Last full user interaction fed to the RNN: [dt, v_1 … v_N].
  private lastUserInteraction: number[] = [];

  private _state: CallResponseState = "CALL";
  private lastUserInputTime = 0;
  // Token for the active response chain; bumped to cancel an in-flight chain.
  private responseGeneration = 0;

  // Parameters (seconds for threshold; multipliers/temps otherwise).
  threshold: number = ParameterDefaults.threshold;
  sigmaTemp: number = ParameterDefaults.sigmaTemp;
  piTemp: number = ParameterDefaults.piTemp;
  timescale: number = ParameterDefaults.timescale;
  inputThru: boolean = ParameterDefaults.inputThru;

  // UI hooks (all fire on the main thread).
  onStateChanged: ((s: CallResponseState) => void) | null = null;
  onEventGenerated: ((dt: number, events: MIDIEvent[], values: number[]) => void) | null = null;
  onUserInputReceived: ((dimIndex: number, value: number) => void) | null = null;

  constructor(mappings: MIDIMappingSet, sendMIDI: (bytes: number[]) => void) {
    this.mapper = new MIDIMapper(mappings);
    this.sendMIDI = sendMIDI;
  }

  get state(): CallResponseState {
    return this._state;
  }

  // MARK: Lifecycle

  start(): void {
    if (this.timer !== null) return;
    this.lastUserInputTime = nowSeconds();
    this.timer = setInterval(() => this.tick(), 10);
  }

  stop(): void {
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = null;
    this.cancelResponseChain();
  }

  // MARK: Input (called from the Web MIDI message handler)

  enqueueInput(bytes: Uint8Array | number[]): void {
    this.inputQueue.push(bytes);
  }

  // MARK: Model loading

  loadModel(rnn: TFLiteRNN): void {
    this.flushAllNoteOffs();
    this.rnn = rnn;
    // Seed the first interaction with a random sample (parity with IMPSY Python).
    const initial = randomInitialSample(rnn.config.dimension);
    this.inputVector = initial.slice(1);
    this.lastUserInteraction = initial;
    this.responseGeneration++;
    this.setState("CALL");
    this.lastUserInputTime = nowSeconds();
  }

  clearModel(): void {
    this.flushAllNoteOffs();
    this.rnn?.dispose();
    this.rnn = null;
    this.inputVector = [];
    this.cancelResponseChain();
  }

  resetLSTMStates(): void {
    this.flushAllNoteOffs();
    this.rnn?.resetStates();
    if (this.rnn && this.lastUserInteraction.length > 0) {
      const initial = randomInitialSample(this.lastUserInteraction.length);
      this.rnn.generate(initial, this.piTemp, this.sigmaTemp);
      this.inputVector = initial.slice(1);
      this.lastUserInteraction = initial;
    }
    this.responseGeneration++;
    if (this._state === "RESPONSE") {
      this.generateAndScheduleResponse(this.lastUserInteraction, this.responseGeneration);
    }
  }

  updateMappings(mappings: MIDIMappingSet): void {
    this.flushAllNoteOffs();
    this.mapper.mappings = mappings;
  }

  // MARK: Loop

  /** 10 ms tick: drain input, let the RNN listen, watch the call⇄response edge. */
  private tick(): void {
    const packets = this.inputQueue;
    this.inputQueue = [];

    let gotUserInput = false;
    const touched: number[] = [];
    for (const bytes of packets) {
      const update = this.mapper.denseUpdate(bytes);
      if (!update) continue;
      const [index, value] = update;
      if (index >= 0 && index < this.inputVector.length) this.inputVector[index] = value;
      touched.push(index);
      gotUserInput = true;
    }

    const now = nowSeconds();

    if (gotUserInput) {
      for (const dim of new Set(touched)) {
        const value = dim < this.inputVector.length ? this.inputVector[dim] : 0;
        this.onUserInputReceived?.(dim, value);
      }
      const dt = Math.max(now - this.lastUserInputTime, MINIMUM_DELTA_TIME);
      this.lastUserInputTime = now;
      this.lastUserInteraction = [dt, ...this.inputVector];

      // In call mode the RNN consumes input only to advance LSTM state; the
      // generated output is discarded so it has context once it takes over.
      if (this._state === "CALL" && this.rnn) {
        this.rnn.generate(this.lastUserInteraction, this.piTemp, this.sigmaTemp);
      }

      // MIDI thru: echo just the changed dimensions through their output maps.
      if (this.inputThru) {
        const events = this.mapper.encodeOutput(this.inputVector, {
          dimensions: new Set(touched),
        });
        for (const e of events) this.sendMIDI(e.bytes);
      }
    }

    // Call ⇄ response transition.
    const timeSinceInput = now - this.lastUserInputTime;
    const newState: CallResponseState = timeSinceInput > this.threshold ? "RESPONSE" : "CALL";
    if (newState === this._state) return;
    this.setState(newState);

    if (newState === "RESPONSE") {
      this.responseGeneration++;
      this.generateAndScheduleResponse(this.lastUserInteraction, this.responseGeneration);
    } else {
      this.responseGeneration++;
      this.flushAllNoteOffs();
    }
  }

  /** One link in the self-feeding response loop. */
  private generateAndScheduleResponse(seed: number[], generation: number): void {
    if (this._state !== "RESPONSE" || generation !== this.responseGeneration || !this.rnn) return;

    let output: number[];
    try {
      output = this.rnn.generate(seed, this.piTemp, this.sigmaTemp);
    } catch (err) {
      console.error("[IMPSY] inference error:", err);
      return;
    }

    const rawDt = Math.max(output[0], RESPONSE_LOOP_MIN_DT);
    const dt = rawDt * this.timescale;
    const values = output.slice(1);
    const nextSeed = [dt, ...values];

    this.responseTimer = setTimeout(() => {
      if (this._state !== "RESPONSE" || generation !== this.responseGeneration) return;
      const now = nowSeconds();
      const events = this.mapper.encodeOutput(values, {
        now,
        noteDedupWindow: 0.03,
        ccDedupWindow: 0.03,
      });
      for (const e of events) this.sendMIDI(e.bytes);
      this.onEventGenerated?.(dt, events, values);
      this.generateAndScheduleResponse(nextSeed, generation);
    }, dt * 1000);
  }

  private cancelResponseChain(): void {
    this.responseGeneration++;
    if (this.responseTimer !== null) clearTimeout(this.responseTimer);
    this.responseTimer = null;
  }

  private flushAllNoteOffs(): void {
    for (const e of this.mapper.releaseAllNotes()) this.sendMIDI(e.bytes);
  }

  private setState(s: CallResponseState): void {
    this._state = s;
    this.onStateChanged?.(s);
  }
}
