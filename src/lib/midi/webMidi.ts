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
  private selectedInputId: string | null = null;
  private selectedOutputId: string | null = null;
  private messageHandler: MessageHandler | null = null;
  private boundInput: MIDIInput | null = null;

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

  /** Subscribe to incoming MIDI messages from the chosen input. */
  selectInput(id: string | null, handler: MessageHandler): void {
    this.messageHandler = handler;
    if (this.boundInput) this.boundInput.onmidimessage = null;
    this.boundInput = null;
    this.selectedInputId = id;
    if (!this.access || !id) return;
    const input = this.access.inputs.get(id);
    if (!input) return;
    input.onmidimessage = (e: MIDIMessageEvent) => {
      if (e.data) this.messageHandler?.(new Uint8Array(e.data));
    };
    this.boundInput = input;
  }

  selectOutput(id: string | null): void {
    this.selectedOutputId = id;
  }

  /** Send a raw MIDI message to the chosen output. No-op if none selected. */
  send(bytes: number[] | Uint8Array): void {
    if (!this.access || !this.selectedOutputId) return;
    this.access.outputs.get(this.selectedOutputId)?.send(bytes as number[]);
  }

  get inputId(): string | null {
    return this.selectedInputId;
  }
  get outputId(): string | null {
    return this.selectedOutputId;
  }
}
