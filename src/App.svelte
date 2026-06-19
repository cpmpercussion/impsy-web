<script lang="ts">
  import { app } from "./lib/appState.svelte";
  import MidiConnection from "./components/MidiConnection.svelte";
  import ModelStatus from "./components/ModelStatus.svelte";
  import DimensionFaders from "./components/DimensionFaders.svelte";
  import ParameterControls from "./components/ParameterControls.svelte";
  import MappingEditor from "./components/MappingEditor.svelte";
</script>

<div class="app">
  <header class="app-header">
    <div>
      <h1>IMPSY Web</h1>
      <div class="subtitle">Interactive music prediction in the browser</div>
    </div>
    <span class="badge {app.callState.toLowerCase()}">{app.callState}</span>
  </header>

  <MidiConnection />
  <ModelStatus />
  <DimensionFaders />
  <ParameterControls />
  <MappingEditor />
</div>
