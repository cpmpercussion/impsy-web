<script lang="ts">
  import { app } from "./lib/appState.svelte";
  import MidiConnection from "./components/MidiConnection.svelte";
  import ModelStatus from "./components/ModelStatus.svelte";
  import DimensionFaders from "./components/DimensionFaders.svelte";
  import ParameterControls from "./components/ParameterControls.svelte";
  import MappingEditor from "./components/MappingEditor.svelte";
  import ConfigIO from "./components/ConfigIO.svelte";
  import SessionRecording from "./components/SessionRecording.svelte";
  import ActivityIndicators from "./components/ActivityIndicators.svelte";
  import LastOutput from "./components/LastOutput.svelte";
  import MidiConsole from "./components/MidiConsole.svelte";

  type Pane = "dashboard" | "settings" | "mapping";
  const panes: { id: Pane; label: string }[] = [
    { id: "dashboard", label: "Dashboard" },
    { id: "settings", label: "Settings" },
    { id: "mapping", label: "Mapping" },
  ];

  // Which pane is shown on narrow screens (tab mode). On wide screens all three
  // panes are laid out side-by-side and this only governs the active tab style.
  let active = $state<Pane>("dashboard");

  // Build identity (issue #17): lets a bug report name the exact deploy.
  const buildLabel = `v${__APP_VERSION__} · ${__GIT_HASH__}`;
  const buildDate = __BUILD_DATE__;
  const commitUrl = `https://github.com/cpmpercussion/impsy-web/commit/${__GIT_HASH__}`;
</script>

<div class="app">
  <header class="app-header">
    <div>
      <h1>IMPSY Web</h1>
      <div class="subtitle">Interactive music prediction in the browser</div>
    </div>
    <div class="header-status">
      <ActivityIndicators />
      <span class="badge {app.callState.toLowerCase()}">{app.callState}</span>
    </div>
  </header>

  <!-- Tab switcher: visible only when the panes collapse to one column. -->
  <nav class="pane-tabs">
    {#each panes as p (p.id)}
      <button class:active={active === p.id} onclick={() => (active = p.id)}>{p.label}</button>
    {/each}
  </nav>

  <div class="panes">
    <div class="pane" class:active={active === "dashboard"}>
      <section class="panel">
        <h2>Dashboard</h2>
        <div class="row">
          <span class="badge {app.callState.toLowerCase()}">{app.callState}</span>
          <span style="flex:1"></span>
          {#if app.modelStatus === "ready"}
            <button onclick={() => app.resetStates()}>Reset LSTM</button>
          {/if}
        </div>
        <LastOutput />
      </section>
      <DimensionFaders />
      <MidiConsole />
    </div>

    <div class="pane" class:active={active === "settings"}>
      <ModelStatus />
      <MidiConnection />
      <ParameterControls />
      <ConfigIO />
      <SessionRecording />
    </div>

    <div class="pane" class:active={active === "mapping"}>
      <MappingEditor />
    </div>
  </div>

  <footer class="app-footer">
    IMPSY Web
    <a href={commitUrl} target="_blank" rel="noopener" title="Built {buildDate}">{buildLabel}</a>
  </footer>
</div>
