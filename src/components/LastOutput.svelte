<script lang="ts">
  // "Last Output" card (issue #9) — port of the AUv3 dashboard card: emission
  // count, Δt and a short MIDI summary of the most recent RNN output.
  import { app } from "../lib/appState.svelte";
</script>

<div class="last-output">
  <div class="label">Last output</div>
  <div class="metrics">
    <div class="metric">
      <span class="k">Events</span>
      <span class="v">{app.lastOutput.count}</span>
    </div>
    <div class="metric">
      <span class="k">Δt</span>
      <span class="v">{app.lastOutput.dt.toFixed(3)} s</span>
    </div>
    <div class="metric wide">
      <span class="k">MIDI</span>
      <span class="v" title={app.lastOutput.summary}>{app.lastOutput.summary}</span>
    </div>
  </div>
</div>

<style>
  .last-output {
    margin-top: 0.8rem;
  }
  .label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.3rem;
  }
  .metrics {
    display: grid;
    grid-template-columns: 1fr 1fr 2fr;
    gap: 0.6rem;
    padding: 0.55rem 0.7rem;
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 8px;
  }
  .metric {
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  .k {
    font-size: 0.68rem;
    color: var(--muted);
  }
  /* Fixed-width digits + ellipsis so updates never shift the layout. */
  .v {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.85rem;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
