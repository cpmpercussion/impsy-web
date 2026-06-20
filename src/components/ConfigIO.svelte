<script lang="ts">
  import { app } from "../lib/appState.svelte";

  let fileInput: HTMLInputElement;
  let imported = $state<string | null>(null);

  async function onFile(e: Event) {
    const input = e.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (file) {
      await app.importConfigFromFile(file);
      if (!app.errorMessage) imported = file.name;
    }
    // Reset so re-selecting the same file fires `change` again.
    input.value = "";
  }
</script>

<section class="panel">
  <h2>Configuration</h2>

  <p class="hint">
    Import or export an IMPSY <code>.toml</code> config — moves parameters and
    MIDI mappings between this app, the Python platform, and the AUv3 plugin.
  </p>

  <div class="row">
    <button class="primary" onclick={() => fileInput.click()}>Import .toml…</button>
    <button onclick={() => app.downloadConfig()}>Export .toml</button>
    <input
      bind:this={fileInput}
      type="file"
      accept=".toml,text/plain"
      style="display:none"
      onchange={onFile}
    />
  </div>

  {#if imported && !app.errorMessage}
    <p class="hint">Imported <strong>{imported}</strong>.</p>
  {/if}
</section>
