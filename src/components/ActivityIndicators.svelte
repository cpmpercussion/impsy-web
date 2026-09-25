<script lang="ts">
  // Aggregate MIDI activity (issue #10): red = any input (MIDI in or fader
  // drag), green = any RNN emission. Mirrors the AUv3 dashboard's ACT LEDs.
  import { app } from "../lib/appState.svelte";
  import ActivityLed from "./ActivityLed.svelte";
</script>

<span class="act" role="img" aria-label="MIDI activity">
  <ActivityLed trigger={app.inputActivity} color="var(--danger)" label="Input activity" />
  <ActivityLed trigger={app.outputActivity} color="var(--call)" label="Output activity" />
  <span class="act-label">ACT</span>
</span>

<style>
  .act {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
  }
  .act-label {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.65rem;
    color: var(--muted);
  }
</style>
