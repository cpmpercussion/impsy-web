<script lang="ts">
  import { app } from "../lib/appState.svelte";
  import DimensionFader from "./DimensionFader.svelte";
</script>

<section class="panel">
  <h2>Direct Input · Model Output</h2>

  {#if app.outputValues.length === 0}
    <p class="hint">Load a model to drive its dimensions and watch the model play back.</p>
  {:else}
    <div class="faders">
      {#each app.outputValues as _v, i (i)}
        <DimensionFader
          dimension={i + 1}
          modelValue={app.outputValues[i]}
          inputValue={app.inputValues[i] ?? 0}
          inputTrigger={app.inputTriggers[i] ?? 0}
          outputTrigger={app.outputTriggers[i] ?? 0}
          onDrag={(value) => app.injectInput(i, value)}
        />
      {/each}
    </div>
    <p class="hint" style="margin-top:0.7rem">
      Drag a bar to play that dimension into IMPSY (red). When you pause, the model
      responds and the bars follow its output (green).
    </p>
  {/if}
</section>

<style>
  .faders {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
</style>
