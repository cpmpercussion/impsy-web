<script lang="ts">
  import { app } from "../lib/appState.svelte";
  import { ParameterRanges } from "../lib/impsy/constants";

  const sliders = [
    { key: "threshold", label: "Threshold", unit: "s", ...ParameterRanges.threshold, step: 0.1 },
    { key: "sigmaTemp", label: "Sigma Temp", unit: "", ...ParameterRanges.sigmaTemp, step: 0.001 },
    { key: "piTemp", label: "Pi Temp", unit: "", ...ParameterRanges.piTemp, step: 0.01 },
    { key: "timescale", label: "Timescale", unit: "×", ...ParameterRanges.timescale, step: 0.05 },
  ] as const;
</script>

<section class="panel">
  <h2>Parameters</h2>

  {#each sliders as s (s.key)}
    <div class="row">
      <label for={s.key}>{s.label}</label>
      <input
        id={s.key}
        type="range"
        min={s.min}
        max={s.max}
        step={s.step}
        value={app.params[s.key]}
        oninput={(e) => app.setParam(s.key, Number((e.currentTarget as HTMLInputElement).value))}
      />
      <span class="value">{app.params[s.key].toFixed(3)}{s.unit}</span>
    </div>
  {/each}

  <div class="row">
    <label for="thru">MIDI Thru</label>
    <input
      id="thru"
      type="checkbox"
      checked={app.params.inputThru}
      onchange={(e) => app.setParam("inputThru", (e.currentTarget as HTMLInputElement).checked)}
    />
    <span class="hint" style="flex:1">Echo mapped input straight to output.</span>
  </div>
</section>
