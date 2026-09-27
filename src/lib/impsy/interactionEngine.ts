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

import { MIDIMapper, type MIDIEvent, type DimensionUpdate } from "./midiMapper";
import type { MIDIMappingSet } from "./midiMapping";
import type { TFLiteRNN } from "./tfliteRnn";
import { ParameterDefaults, RESPONSE_LOOP_MIN_DT } from "./constants";

export type CallResponseState = "CALL" | "RESPONSE";

const nowSeconds = () => performance.now() / 1000;

/** dt ≈ 0.0075–0.0125 s, remaining dims random [0,1) — mirrors random_sample(). */
export function randomInitialSample(dimension: number): number[] {
  if (dimension <= 0) return [];
  const s = Array.from({ length: dimension }, () => Math.random());
  s[0] = 0.01 + (Math.random() - 0.5) * 0.005;
  return s;
}

/**
 * The dense user-input vector and the time of the last interaction — the
 * sparse→dense step of `construct_input_list` in ../impsy/impsy/interaction.py.
 * Each message is one interaction, however many dimensions it sets.
 */
export class UserInput {
  /** Current x_1…x_n (index 0 → dimension 1). */
  values: number[];
  /** Seconds (any clock) of the last interaction, or the start time. */
  lastTime: number;

  constructor(initialValues: number[], startTime: number) {
    this.values = [...initialValues];
    this.lastTime = startTime;
  }

  /**
   * Apply one message's updates at `time`; returns the model input
   * `[dt, x_1…x_n]`, with dt the seconds since the previous interaction.
   */
  receive(updates: DimensionUpdate[], time: number): number[] {
    // Copy: the previous vector may still be referenced as a model input.
    const values = [...this.values];
    for (const [index, value] of updates) {
      if (index >= 0 && index < values.length) values[index] = value;
    }
    const dt = time - this.lastTime;
    this.lastTime = time;
    this.values = values;
    return [dt, ...values];
  }
}

/**
 * Turn a model output `[dt, x_1…x_n]` into the seconds to wait before playing
 * it, the values to play, and the next model input — `prepare_rnn_playback` in
 * interaction.py. dt is floored at 1 ms and the values clipped to [0,1].
 * Timescale only changes playback speed: the model is fed back its own
 * (floored) dt, not the scaled one (impsy#103).
 */
export function prepareRNNPlayback(
  output: number[],
  timescale: number,
): { wait: number; values: number[]; nextInput: number[] } {
  const dt = Math.max(output[0], RESPONSE_LOOP_MIN_DT);
  const values = output.slice(1).map((v) => Math.min(Math.max(v, 0), 1));
  return { wait: dt * timescale, values, nextInput: [dt, ...values] };
}

export class InteractionEngine {
  private mapper: MIDIMapper;
  private rnn: TFLiteRNN | null = null;
  private sendMIDI: (bytes: number[]) => void;

  private inputQueue: { bytes: Uint8Array | number[]; time: number }[] = [];
  private timer: ReturnType<typeof setInterval> | null = null;
  private responseTimer: ReturnType<typeof setTimeout> | null = null;

  // Dense input vector (0-based: index 0 → dimension 1) + last interaction time.
  private input = new UserInput([], 0);
  // Last full user interaction fed to the RNN: [dt, v_1 … v_N].
  private lastUserInteraction: number[] = [];

  private _state: CallResponseState = "CALL";
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
  // Fires with a copy of the full input vector after each mapped MIDI message,
  // so a session logger can record one `interface` row per interaction — mirrors
  // logInterface in InteractionEngine.swift / construct_input_list in Python.
  onInterfaceEvent: ((values: number[]) => void) | null = null;

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
    this.input.lastTime = nowSeconds();
    this.timer = setInterval(() => this.tick(), 10);
  }

  stop(): void {
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = null;
    this.cancelResponseChain();
  }

  // MARK: Input (called from the Web MIDI message handler)

  /** Queue a message; `time` (seconds, performance clock) is when it arrived. */
  enqueueInput(bytes: Uint8Array | number[], time: number = nowSeconds()): void {
    this.inputQueue.push({ bytes, time });
  }

  // MARK: Model loading

  loadModel(rnn: TFLiteRNN): void {
    this.flushAllNoteOffs();
    this.rnn = rnn;
    // Seed the first interaction with a random sample (parity with IMPSY Python).
    const initial = randomInitialSample(rnn.config.dimension);
    this.input = new UserInput(initial.slice(1), nowSeconds());
    this.lastUserInteraction = initial;
    this.responseGeneration++;
    this.setState("CALL");
  }

  clearModel(): void {
    this.flushAllNoteOffs();
    this.rnn?.dispose();
    this.rnn = null;
    this.input = new UserInput([], nowSeconds());
    this.cancelResponseChain();
  }

  resetLSTMStates(): void {
    this.flushAllNoteOffs();
    this.rnn?.resetStates();
    if (this.rnn && this.lastUserInteraction.length > 0) {
      const initial = randomInitialSample(this.lastUserInteraction.length);
      this.rnn.generate(initial, this.piTemp, this.sigmaTemp);
      this.input.values = initial.slice(1);
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

    // Each mapped message is one interaction (Python parity): one log row, one
    // model input, and in CALL mode one RNN step.
    for (const { bytes, time } of packets) {
      const updates = this.mapper.decodeInput(bytes);
      if (!updates) continue;
      this.lastUserInteraction = this.input.receive(updates, time);
      this.onInterfaceEvent?.([...this.input.values]);

      const touched = new Set(updates.map(([index]) => index));
      for (const dim of touched) this.onUserInputReceived?.(dim, this.input.values[dim] ?? 0);

      // In call mode the RNN consumes input only to advance LSTM state; the
      // generated output is discarded so it has context once it takes over.
      if (this._state === "CALL" && this.rnn) {
        this.rnn.generate(this.lastUserInteraction, this.piTemp, this.sigmaTemp);
      }

      // MIDI thru: echo just the changed dimensions through their output maps.
      if (this.inputThru) {
        const events = this.mapper.encodeOutput(this.input.values, { dimensions: touched });
        for (const e of events) this.sendMIDI(e.bytes);
      }
    }

    const now = nowSeconds();

    // Call ⇄ response transition.
    const timeSinceInput = now - this.input.lastTime;
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

    const { wait, values, nextInput } = prepareRNNPlayback(output, this.timescale);

    this.responseTimer = setTimeout(() => {
      if (this._state !== "RESPONSE" || generation !== this.responseGeneration) return;
      const events = this.mapper.encodeOutput(values);
      for (const e of events) this.sendMIDI(e.bytes);
      this.onEventGenerated?.(wait, events, values);
      this.generateAndScheduleResponse(nextInput, generation);
    }, wait * 1000);
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
