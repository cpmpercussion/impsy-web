// IMPSY model + interaction constants.
//
// Ported from ../impsy-auv3/IMPSYExtension/Common/IMPSYParameters.swift, which
// itself mirrors ../impsy/impsy/mdrnn.py and interaction.py. Keep these in sync
// with the Python reference — it is the source of truth for IMPSY behaviour.

/** All values are multiplied by this before feeding the model, divided after. */
export const SCALE_FACTOR = 10.0;

/** Minimum time delta (seconds) — prevents divide-by-zero / runaway speed. */
export const MINIMUM_DELTA_TIME = 0.000454;

/**
 * Scheduling floor for the response-loop dt (seconds). Mirrors
 * `dt = max(dt, 0.001)` in ../impsy/impsy/interaction.py (playback_rnn_loop).
 * Applied before the timescale multiply so the model sees the floored value.
 */
export const RESPONSE_LOOP_MIN_DT = 0.001;

/** Runtime parameter defaults — from configs/AiC-charles-u6midipro.toml in IMPSY. */
export const ParameterDefaults = {
  threshold: 0.1, // seconds of user silence before the RNN responds
  sigmaTemp: 0.01, // Gaussian sampling temperature
  piTemp: 1.0, // mixture-selection temperature
  timescale: 1.0, // output dt multiplier
  inputThru: true, // echo mapped user input straight back out
} as const;

/** Allowed ranges for the runtime parameters (UI slider bounds). */
export const ParameterRanges = {
  threshold: { min: 0.1, max: 10.0 },
  sigmaTemp: { min: 0.001, max: 2.0 },
  piTemp: { min: 0.1, max: 5.0 },
  timescale: { min: 0.1, max: 4.0 },
} as const;
