<script lang="ts">
  import { app } from "./lib/appState.svelte";
  import MidiConnection from "./components/MidiConnection.svelte";
  import ModelStatus from "./components/ModelStatus.svelte";
  import DimensionFaders from "./components/DimensionFaders.svelte";
  import ParameterControls from "./components/ParameterControls.svelte";
  import MappingEditor from "./components/MappingEditor.svelte";

  type Pane = "dashboard" | "settings" | "mapping";
  const panes: { id: Pane; label: string }[] = [
    { id: "dashboard", label: "Dashboard" },
    { id: "settings", label: "Settings" },
    { id: "mapping", label: "Mapping" },
  ];

  // Which pane is shown on narrow screens (tab mode). On wide screens all three
  // panes are laid out side-by-side and this only governs the active tab style.
  let active = $state<Pane>("dashboard");
</script>

<div class="app">
  <header class="app-header">
    <div>
      <h1>IMPSY Web</h1>
      <div class="subtitle">Interactive music prediction in the browser</div>
    </div>
    <span class="badge {app.callState.toLowerCase()}">{app.callState}</span>
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
      </section>
      <DimensionFaders />
    </div>

    <div class="pane" class:active={active === "settings"}>
      <ModelStatus />
      <MidiConnection />
      <ParameterControls />
    </div>

    <div class="pane" class:active={active === "mapping"}>
      <MappingEditor />
    </div>
  </div>
</div>
