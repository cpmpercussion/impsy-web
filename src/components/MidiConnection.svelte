<script lang="ts">
  import { app } from "../lib/appState.svelte";
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
    <div class="row">
      <label for="midi-in">Input</label>
      <select
        id="midi-in"
        value={app.selectedInputId ?? ""}
        onchange={(e) => app.selectInput((e.currentTarget as HTMLSelectElement).value || null)}
      >
        <option value="">— none —</option>
        {#each app.inputs as port (port.id)}
          <option value={port.id}>{port.name}</option>
        {/each}
      </select>
    </div>
    <div class="row">
      <label for="midi-out">Output</label>
      <select
        id="midi-out"
        value={app.selectedOutputId ?? ""}
        onchange={(e) => app.selectOutput((e.currentTarget as HTMLSelectElement).value || null)}
      >
        <option value="">— none —</option>
        {#each app.outputs as port (port.id)}
          <option value={port.id}>{port.name}</option>
        {/each}
      </select>
    </div>
  {/if}
</section>
