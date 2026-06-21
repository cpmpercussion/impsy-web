<script lang="ts">
  import { app } from "../lib/appState.svelte";
</script>

<section class="panel">
  <h2>Session recording</h2>

  <p class="hint">
    Record every interface and RNN event to an IMPSY <code>.log</code> file. Drop
    the download into a Python workspace's <code>logs/</code> folder and
    <code>impsy dataset</code> turns it into training data.
  </p>

  <div class="row">
    <label for="record">Record session</label>
    <input
      id="record"
      type="checkbox"
      checked={app.recording}
      disabled={app.modelStatus !== "ready"}
      onchange={(e) => app.setRecording((e.currentTarget as HTMLInputElement).checked)}
    />
    <span style="flex:1"></span>
    <button onclick={() => app.downloadLog()} disabled={app.recordedEvents === 0}>
      Download .log
    </button>
  </div>

  <p class="hint">
    {#if app.modelStatus !== "ready"}
      Load a model to start recording.
    {:else}
      {app.recordedEvents} event{app.recordedEvents === 1 ? "" : "s"} recorded.
    {/if}
  </p>
</section>
