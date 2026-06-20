// Web MIDI access wrapper.
//
// This is the web analog of CoreMIDIBridge / MIDIEndpointStore in impsy-auv3.
// Web MIDI is main-thread only (not available in Workers), so the interaction
// engine's MIDI I/O hooks are wired here on the main thread.

export interface MIDIPort {
  id: string;
  name: string;
}

type MessageHandler = (data: Uint8Array) => void;

export class WebMIDI {
  private access: MIDIAccess | null = null;
  // Multiple inputs and outputs can be active at once: incoming messages from
  // every selected input are merged into the single handler, and `send` fans
  // out to every selected output. This lets IMPSY drive e.g. a hardware
  // interface and an IAC bus (to a DAW) simultaneously, no external routing.
  private selectedInputIds = new Set<string>();
  private selectedOutputIds = new Set<string>();
  private messageHandler: MessageHandler | null = null;
  private boundInputs = new Map<string, MIDIInput>();

  /** Called whenever the device list changes (connect/disconnect). */
  onPortsChanged: (() => void) | null = null;

  static isSupported(): boolean {
    return typeof navigator !== "undefined" && "requestMIDIAccess" in navigator;
  }

  /** Prompt for Web MIDI access. Throws if denied or unsupported. */
  async requestAccess(sysex = false): Promise<void> {
    if (!WebMIDI.isSupported()) {
      throw new Error("Web MIDI is not supported in this browser (try Chrome, Edge, or Firefox).");
    }
    this.access = await navigator.requestMIDIAccess({ sysex });
    this.access.onstatechange = () => this.onPortsChanged?.();
  }

  get inputs(): MIDIPort[] {
    if (!this.access) return [];
    return [...this.access.inputs.values()].map((p) => ({ id: p.id, name: p.name ?? p.id }));
  }

  get outputs(): MIDIPort[] {
    if (!this.access) return [];
    return [...this.access.outputs.values()].map((p) => ({ id: p.id, name: p.name ?? p.id }));
  }

  /**
   * Bind the incoming-message handler to every selected input. Inputs not
   * currently present (unplugged) are kept in the selection so they rebind
   * automatically when they reappear — call this again after a port change.
   */
  setInputs(ids: string[], handler: MessageHandler): void {
    this.messageHandler = handler;
    this.selectedInputIds = new Set(ids);
    // Detach all previously bound inputs, then (re)bind the current selection.
    for (const input of this.boundInputs.values()) input.onmidimessage = null;
    this.boundInputs.clear();
    if (!this.access) return;
    for (const id of this.selectedInputIds) {
      const input = this.access.inputs.get(id);
      if (!input) continue;
      input.onmidimessage = (e: MIDIMessageEvent) => {
        if (e.data) this.messageHandler?.(new Uint8Array(e.data));
      };
      this.boundInputs.set(id, input);
    }
  }

  setOutputs(ids: string[]): void {
    this.selectedOutputIds = new Set(ids);
  }

  /** Send a raw MIDI message to every selected output. No-op if none selected. */
  send(bytes: number[] | Uint8Array): void {
    if (!this.access || this.selectedOutputIds.size === 0) return;
    for (const id of this.selectedOutputIds) {
      this.access.outputs.get(id)?.send(bytes as number[]);
    }
  }

  get inputIds(): string[] {
    return [...this.selectedInputIds];
  }
  get outputIds(): string[] {
    return [...this.selectedOutputIds];
  }
}
