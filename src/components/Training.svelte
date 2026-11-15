<script lang="ts">
  import { app } from "../lib/appState.svelte";
  import { training, type LogSource, type ModelSize } from "../lib/training.svelte";
  import { MODEL_SIZES } from "../lib/impsy/mdrnnWeights";

  let fileInput: HTMLInputElement;
  let files = $state<File[]>([]);
  let includeSession = $state(true);
  let size = $state<ModelSize>("s");
  let epochs = $state(100);

  const sizes = Object.keys(MODEL_SIZES) as ModelSize[];
  const useSession = $derived(includeSession && app.recordedEvents > 0);
  const last = $derived(training.history.at(-1));

  function onFiles(e: Event) {
    files = Array.from((e.currentTarget as HTMLInputElement).files ?? []);
  }

  async function train() {
    const sources: LogSource[] = await Promise.all(
      files.map(async (f) => ({ name: f.name, text: await f.text() })),
    );
    const session = useSession ? app.sessionLog() : null;
    if (session) sources.push(session);
    await training.start(sources, { size, epochs });
  }
</script>

<section class="panel">
  <h2>Train a model</h2>

  <p class="hint">
    Train a new model on recorded sessions, here in the browser. The result is a
    <code>.tflite</code> file that also works in IMPSY's Python and AUv3 apps.
  </p>

  <div class="row">
    <button onclick={() => fileInput.click()} disabled={training.status === "running"}>
      Choose .log files…
    </button>
    <span class="hint" style="flex:1">
      {files.length === 0
        ? "None chosen"
        : `${files.length} file${files.length === 1 ? "" : "s"} chosen`}
    </span>
    <input
      bind:this={fileInput}
      id="train-files"
      type="file"
      accept=".log"
      multiple
      style="display:none"
      onchange={onFiles}
    />
  </div>

  {#if app.recordedEvents > 0}
    <div class="row">
      <label for="train-session">Include this session ({app.recordedEvents} events)</label>
      <input id="train-session" type="checkbox" bind:checked={includeSession} />
    </div>
  {/if}

  <div class="row">
    <label for="train-size">Size</label>
    <select id="train-size" bind:value={size} disabled={training.status === "running"}>
      {#each sizes as s (s)}
        <option value={s}>{s} · {MODEL_SIZES[s].hiddenUnits} units</option>
      {/each}
    </select>
    <label for="train-epochs">Max epochs</label>
    <input
      id="train-epochs"
      type="number"
      min="1"
      max="1000"
      bind:value={epochs}
      disabled={training.status === "running"}
      style="width:5em"
    />
  </div>

  <div class="row">
    {#if training.status === "running"}
      <button class="danger" onclick={() => training.stop()}>Stop and keep model</button>
    {:else}
      <button
        class="primary"
        onclick={train}
        disabled={(files.length === 0 && !useSession) || !(epochs >= 1)}
      >
        Train model
      </button>
    {/if}
    {#if training.trainedName && training.status !== "running"}
      <button onclick={() => training.download()}>Download .tflite</button>
    {/if}
  </div>

  {#if training.status === "running"}
    <progress
      max={training.epochs}
      value={Math.max(0, training.progress.epoch - 1) +
        (training.progress.batches ? training.progress.batch / training.progress.batches : 0)}
      style="width:100%"
    ></progress>
    <p class="hint">
      {training.summary}<br />
      {#if training.progress.epoch === 0}
        Starting…
      {:else}
        Epoch {training.progress.epoch} of up to {training.epochs} · batch
        {training.progress.batch}/{training.progress.batches}
      {/if}
      {#if last}
        · loss {last.loss.toFixed(3)}{last.valLoss === null
          ? ""
          : ` · validation ${last.valLoss.toFixed(3)}`}
      {/if}
    </p>
  {/if}

  {#if training.message}
    <p class={training.status === "error" ? "error" : "hint"}>{training.message}</p>
  {/if}
</section>
