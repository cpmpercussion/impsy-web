<script lang="ts">
  // One slim horizontal bar for a single dimension. Doubles as input lever and
  // output indicator — port of DimensionFader in impsy-auv3/DashboardView.swift.
  //
  // Idle, the bar follows `modelValue` (the RNN's last output for this dim) and
  // is tinted green. It turns red and detaches from the model under either user
  // action — dragging the bar (injects MIDI input via onDrag) or live MIDI in
  // (a bump in `inputTrigger` snaps it to `inputValue`). It settles back to the
  // model value ~250 ms after the last interaction.

  interface Props {
    dimension: number;
    modelValue: number;
    inputValue: number;
    inputTrigger: number;
    outputTrigger: number;
    onDrag: (value: number) => void;
  }
  let { dimension, modelValue, inputValue, inputTrigger, outputTrigger, onDrag }: Props =
    $props();

  let localValue = $state(0);
  let dragActive = $state(false);
  let inputActive = $state(false);
  let track: HTMLDivElement;

  let dragTimer: ReturnType<typeof setTimeout> | null = null;
  let inputTimer: ReturnType<typeof setTimeout> | null = null;
  let lastInputTrigger = 0;

  const userActive = $derived(dragActive || inputActive);

  // Follow the model while the user isn't driving this dimension.
  $effect(() => {
    const mv = modelValue;
    if (!dragActive && !inputActive) localValue = mv;
  });

  // Live MIDI in: snap to the received value and flash red (unless dragging).
  $effect(() => {
    const t = inputTrigger;
    if (t !== lastInputTrigger) {
      lastInputTrigger = t;
      if (!dragActive) {
        localValue = inputValue;
        inputActive = true;
        scheduleInputEnd();
      }
    }
  });

  $effect(() => () => {
    if (dragTimer) clearTimeout(dragTimer);
    if (inputTimer) clearTimeout(inputTimer);
  });

  function scheduleDragEnd() {
    if (dragTimer) clearTimeout(dragTimer);
    dragTimer = setTimeout(() => {
      dragActive = false;
      if (!inputActive) localValue = modelValue;
    }, 250);
  }

  function scheduleInputEnd() {
    if (inputTimer) clearTimeout(inputTimer);
    inputTimer = setTimeout(() => {
      inputActive = false;
      if (!dragActive) localValue = modelValue;
    }, 250);
  }

  function valueFromEvent(e: PointerEvent): number {
    const rect = track.getBoundingClientRect();
    return Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  }

  function onPointerDown(e: PointerEvent) {
    track.setPointerCapture(e.pointerId);
    dragActive = true;
    const v = valueFromEvent(e);
    localValue = v;
    onDrag(v);
  }

  function onPointerMove(e: PointerEvent) {
    if (!dragActive) return;
    const v = valueFromEvent(e);
    localValue = v;
    onDrag(v);
  }

  function onPointerUp(e: PointerEvent) {
    if (!dragActive) return;
    track.releasePointerCapture(e.pointerId);
    scheduleDragEnd();
  }

  // Brief flash overlay when the model emits for this dimension.
  let outFlash = $state(false);
  let lastOutputTrigger = 0;
  let outTimer: ReturnType<typeof setTimeout> | null = null;
  $effect(() => {
    const t = outputTrigger;
    if (t !== lastOutputTrigger) {
      lastOutputTrigger = t;
      outFlash = true;
      if (outTimer) clearTimeout(outTimer);
      outTimer = setTimeout(() => (outFlash = false), 120);
    }
  });
</script>

<div class="fader">
  <span class="dim" class:flash={outFlash} class:user={userActive}>{dimension}</span>
  <div
    class="track"
    class:user={userActive}
    bind:this={track}
    onpointerdown={onPointerDown}
    onpointermove={onPointerMove}
    onpointerup={onPointerUp}
    role="slider"
    tabindex="0"
    aria-label="Dimension {dimension}"
    aria-valuemin="0"
    aria-valuemax="1"
    aria-valuenow={localValue}
  >
    <div class="fill" style="width: {localValue * 100}%"></div>
  </div>
  <span class="val" class:user={userActive}>{localValue.toFixed(2)}</span>
</div>

<style>
  .fader {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .dim {
    width: 1.6rem;
    flex: none;
    text-align: center;
    font-size: 0.72rem;
    font-variant-numeric: tabular-nums;
    color: var(--muted);
    border-radius: 4px;
    padding: 0.05rem 0;
    transition: background 0.12s, color 0.12s;
  }
  .dim.flash {
    background: color-mix(in srgb, var(--call) 55%, transparent);
    color: #eaffef;
  }
  .dim.user {
    color: #ff9b98;
  }
  .track {
    position: relative;
    flex: 1;
    height: 14px;
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 7px;
    cursor: ew-resize;
    overflow: hidden;
    touch-action: none;
  }
  .fill {
    position: absolute;
    inset: 0 auto 0 0;
    background: var(--call);
    border-radius: 7px 0 0 7px;
    transition: width 0.05s linear, background 0.12s;
  }
  .track.user .fill {
    background: var(--danger);
    transition: background 0.12s;
  }
  .val {
    width: 2.4rem;
    flex: none;
    text-align: right;
    font-size: 0.75rem;
    font-variant-numeric: tabular-nums;
    color: #8fe6ad;
  }
  .val.user {
    color: #ff9b98;
  }
</style>
