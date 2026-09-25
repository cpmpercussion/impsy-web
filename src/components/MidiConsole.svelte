<script lang="ts">
  // Live MIDI console (issue #14): the last 20 messages in and out, newest
  // first. Input (red) is MIDI from selected devices or a fader drag (marked
  // "UI"); output (green) is everything sent — RNN responses, MIDI thru and
  // note-offs. System real-time (clock / active sensing) is filtered out.
  // Hover a row for its raw bytes.
  import { app } from "../lib/appState.svelte";
  import { hexBytes } from "../lib/midi/midiLog";
</script>

<section class="panel">
  <div class="head">
    <h2>MIDI console</h2>
    <span style="flex:1"></span>
    <button
      onclick={() => app.setConsolePaused(!app.consolePaused)}
      aria-label={app.consolePaused ? "Resume MIDI console" : "Pause MIDI console"}
    >
      {app.consolePaused ? "Resume" : "Pause"}
    </button>
    <button
      onclick={() => app.clearConsole()}
      disabled={app.midiLog.length === 0}
      aria-label="Clear MIDI console">Clear</button
    >
  </div>

  {#if app.midiLog.length === 0}
    <p class="hint">
      {app.consolePaused
        ? "Paused."
        : "No MIDI yet — play a controller, drag a fader, or let the model respond."}
    </p>
  {:else}
    <ol class="log" aria-live="off">
      {#each app.midiLog as e (e.seq)}
        <li class={e.direction} title={hexBytes(e.bytes)}>
          <span class="t">{e.time.toFixed(3)}</span>
          <span class="dir">{e.direction === "in" ? (e.fromUI ? "UI" : "IN") : "OUT"}</span>
          <span class="msg">{e.text}</span>
        </li>
      {/each}
    </ol>
  {/if}
</section>

<style>
  .head {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.5rem;
  }
  .head h2 {
    margin: 0;
  }
  .head button {
    padding: 0.25rem 0.6rem;
    font-size: 0.75rem;
  }
  .log {
    list-style: none;
    margin: 0;
    padding: 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.72rem;
    font-variant-numeric: tabular-nums;
  }
  .log li {
    display: grid;
    grid-template-columns: 3.8rem 2rem 1fr;
    gap: 0.5rem;
    padding: 0.12rem 0.3rem;
    border-left: 2px solid transparent;
    white-space: nowrap;
  }
  .log li.in {
    border-left-color: var(--danger);
  }
  .log li.out {
    border-left-color: var(--call);
  }
  .t {
    color: var(--muted);
  }
  .in .dir {
    color: #f2a3a3;
  }
  .out .dir {
    color: #8fd1a5;
  }
  .msg {
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
