<script lang="ts">
  import { app } from "../lib/appState.svelte";
  import type { MIDIPort } from "../lib/midi/webMidi";

  // Selected ids whose device isn't currently present (unplugged). Shown greyed
  // out and still checked so the user knows it'll reconnect when it returns.
  function offline(selected: string[], present: MIDIPort[]): string[] {
    const ids = new Set(present.map((p) => p.id));
    return selected.filter((id) => !ids.has(id));
  }
</script>

<section class="panel">
  <h2>MIDI Connection</h2>

  {#if !app.midiSupported}
    <p class="error">
      Web MIDI is not available in this browser. Use Chrome, Edge, or Firefox.
    </p>
  {:else if !app.midiGranted}
    <button class="primary" onclick={() => app.requestMIDI()}>
      Enable Web MIDI
    </button>
    <p class="hint">Grants access to your local MIDI input and output devices.</p>
  {:else}
    <div class="device-group">
      <span class="group-label">Inputs</span>
      <div class="device-list">
        {#each app.inputs as port (port.id)}
          <label class="device">
            <input
              type="checkbox"
              checked={app.selectedInputIds.includes(port.id)}
              onchange={(e) => app.toggleInput(port.id, e.currentTarget.checked)}
            />
            <span>{port.name}</span>
          </label>
        {/each}
        {#each offline(app.selectedInputIds, app.inputs) as id (id)}
          <label class="device offline">
            <input type="checkbox" checked onchange={() => app.toggleInput(id, false)} />
            <span>Offline device</span>
          </label>
        {/each}
        {#if app.inputs.length === 0 && offline(app.selectedInputIds, app.inputs).length === 0}
          <p class="hint">No input devices found.</p>
        {/if}
      </div>
    </div>

    <div class="device-group">
      <span class="group-label">Outputs</span>
      <div class="device-list">
        {#each app.outputs as port (port.id)}
          <label class="device">
            <input
              type="checkbox"
              checked={app.selectedOutputIds.includes(port.id)}
              onchange={(e) => app.toggleOutput(port.id, e.currentTarget.checked)}
            />
            <span>{port.name}</span>
          </label>
        {/each}
        {#each offline(app.selectedOutputIds, app.outputs) as id (id)}
          <label class="device offline">
            <input type="checkbox" checked onchange={() => app.toggleOutput(id, false)} />
            <span>Offline device</span>
          </label>
        {/each}
        {#if app.outputs.length === 0 && offline(app.selectedOutputIds, app.outputs).length === 0}
          <p class="hint">No output devices found.</p>
        {/if}
      </div>
    </div>

    <p class="hint">Check any number of devices — inputs merge, output is sent to all.</p>
  {/if}
</section>

<style>
  .device-group {
    margin-bottom: 0.85rem;
  }
  .group-label {
    display: block;
    color: var(--muted);
    font-size: 0.85rem;
    margin-bottom: 0.4rem;
  }
  .device-list {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }
  .device {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    cursor: pointer;
  }
  .device input {
    accent-color: var(--accent);
  }
  .device.offline {
    color: var(--muted);
    font-style: italic;
  }
</style>
