<script lang="ts">
  import { app } from "../lib/appState.svelte";

  let fileInput: HTMLInputElement;

  function onFile(e: Event) {
    const file = (e.currentTarget as HTMLInputElement).files?.[0];
    if (file) app.loadModelFromFile(file);
  }
</script>

<section class="panel">
  <h2>Model</h2>

  <div class="row">
    <span class="status-dot {app.modelStatus}"></span>
    <span style="flex:1">
      {#if app.modelStatus === "none"}
        No model loaded
      {:else if app.modelStatus === "loading"}
        Loading…
      {:else if app.modelStatus === "ready" && app.modelConfig}
        {app.modelName}
        <span class="hint">
          · dim {app.modelConfig.dimension} · {app.modelConfig.numLayers} layers ·
          {app.modelConfig.hiddenUnits} units
        </span>
      {:else}
        Error
      {/if}
    </span>
  </div>

  <div class="row">
    <button class="primary" onclick={() => fileInput.click()}>Load .tflite…</button>
    <button onclick={() => app.loadModelFromUrl("/models/musicMDRNN-dim9-layers2-units64-mixtures5-scale10.tflite")}>
      Load demo model
    </button>
    {#if app.modelStatus === "ready"}
      <button onclick={() => app.resetStates()}>Reset</button>
      <button class="danger" onclick={() => app.clearModel()}>Clear</button>
    {/if}
    <input
      bind:this={fileInput}
      type="file"
      accept=".tflite"
      style="display:none"
      onchange={onFile}
    />
  </div>

  {#if app.errorMessage}
    <p class="error">{app.errorMessage}</p>
  {/if}
</section>
