<script lang="ts">
  import { app } from "../lib/appState.svelte";
  import {
    type DimensionMapping,
    type MIDIMessageType,
    messageTypeDisplayName,
    usesNumber,
  } from "../lib/impsy/midiMapping";

  let tab = $state<"input" | "output">("input");

  const types: MIDIMessageType[] = ["noteOn", "controlChange", "pitchBend"];

  function current(): DimensionMapping[] {
    return tab === "input" ? app.mappings.inputMappings : app.mappings.outputMappings;
  }

  function update(index: number, patch: Partial<DimensionMapping>) {
    const list = current().map((m, i) => (i === index ? { ...m, ...patch } : m));
    app.setMappings(
      tab === "input"
        ? { ...app.mappings, inputMappings: list }
        : { ...app.mappings, outputMappings: list },
    );
  }
</script>

<section class="panel">
  <h2>MIDI Mapping</h2>

  <div class="tabs">
    <button class:active={tab === "input"} onclick={() => (tab = "input")}>Input</button>
    <button class:active={tab === "output"} onclick={() => (tab = "output")}>Output</button>
  </div>

  {#if current().length === 0}
    <p class="hint">Load a model to configure its dimension mappings.</p>
  {:else}
    <table class="mappings">
      <thead>
        <tr>
          <th>Dim</th>
          <th>Type</th>
          <th>Ch</th>
          <th>Number</th>
          <th>On</th>
        </tr>
      </thead>
      <tbody>
        {#each current() as m, i (m.id)}
          <tr>
            <td>{m.id}</td>
            <td>
              <select
                value={m.messageType}
                onchange={(e) =>
                  update(i, { messageType: (e.currentTarget as HTMLSelectElement).value as MIDIMessageType })}
              >
                {#each types as t}
                  <option value={t}>{messageTypeDisplayName[t]}</option>
                {/each}
              </select>
            </td>
            <td>
              <input
                type="number"
                min="1"
                max="16"
                value={m.channel}
                onchange={(e) => update(i, { channel: Number((e.currentTarget as HTMLInputElement).value) })}
              />
            </td>
            <td>
              {#if usesNumber(m.messageType)}
                <input
                  type="number"
                  min="0"
                  max="127"
                  value={m.number}
                  onchange={(e) => update(i, { number: Number((e.currentTarget as HTMLInputElement).value) })}
                />
              {:else}
                <span class="hint">—</span>
              {/if}
            </td>
            <td>
              <input
                type="checkbox"
                checked={m.enabled}
                onchange={(e) => update(i, { enabled: (e.currentTarget as HTMLInputElement).checked })}
              />
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</section>
