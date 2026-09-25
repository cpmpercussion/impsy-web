<script lang="ts">
  // Flashes once each time `trigger` changes — the value itself is irrelevant,
  // only its change. Port of ActivityLED in impsy-auv3/DashboardView.swift
  // (40 ms hold, then fade out).

  interface Props {
    trigger: number;
    color: string;
    label: string;
  }
  let { trigger, color, label }: Props = $props();

  let lit = $state(false);
  let timer: ReturnType<typeof setTimeout> | null = null;
  let last = 0;

  $effect(() => {
    const t = trigger;
    if (t === last) return;
    last = t;
    lit = true;
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => (lit = false), 40);
  });

  $effect(() => () => {
    if (timer) clearTimeout(timer);
  });
</script>

<span class="led" class:lit style="--led: {color}" title={label} aria-hidden="true"></span>

<style>
  .led {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: color-mix(in srgb, var(--led) 20%, transparent);
    border: 1px solid color-mix(in srgb, var(--led) 45%, transparent);
    transition:
      background 250ms ease-out,
      box-shadow 250ms ease-out;
  }
  .led.lit {
    background: var(--led);
    box-shadow: 0 0 6px var(--led);
    transition: none;
  }
</style>
