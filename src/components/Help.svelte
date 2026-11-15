<script lang="ts">
  let { onclose }: { onclose: () => void } = $props();
</script>

<section class="panel help">
  <div class="row">
    <h2 style="flex:1; margin:0">Help</h2>
    <button onclick={onclose}>Back to the app</button>
  </div>

  <h3>How it works</h3>
  <p>
    IMPSY Web listens to what you play and, when you pause, carries on in your style. While you play
    it is in <strong>CALL</strong>; after <em>Threshold</em> seconds of silence it switches to
    <strong>RESPONSE</strong> and the model plays until you play again.
  </p>
  <ol>
    <li>Use Chrome, Edge or Firefox. Safari does not support Web MIDI.</li>
    <li>
      In <em>Settings → MIDI Connection</em>, press <strong>Enable Web MIDI</strong> and allow MIDI access
      when the browser asks.
    </li>
    <li>
      Tick your controller under <strong>Inputs</strong> and wherever the sound should come from
      under <strong>Outputs</strong>.
    </li>
    <li>
      Play. The <em>Mapping</em> pane sets which MIDI message drives each model dimension, and which message
      each dimension sends.
    </li>
  </ol>

  <h3>Connecting to Ableton Live</h3>
  <p>
    Live and the browser talk through a <strong>virtual MIDI port</strong>: IMPSY Web sends to it,
    and Live receives from it as if it were a keyboard. Your controller goes straight into IMPSY
    Web, not into Live.
  </p>

  <h4>1. Create a virtual MIDI port</h4>
  <ul>
    <li>
      <strong>macOS:</strong> open <em>Audio MIDI Setup</em>, choose
      <em>Window → Show MIDI Studio</em>, double-click <strong>IAC Driver</strong> and tick
      <strong>Device is online</strong>. One bus is enough to start; rename it (for example “IMPSY”)
      so it is easy to spot.
    </li>
    <li>
      <strong>Windows:</strong> there is no built-in virtual port. Install
      <a href="https://www.tobias-erichsen.de/software/loopmidi.html" target="_blank" rel="noopener"
        >loopMIDI</a
      >, add a port named “IMPSY” and leave loopMIDI running.
    </li>
  </ul>
  <p class="hint">
    If the new port does not show up here, reload this page (or press Enable Web MIDI again).
  </p>

  <h4>2. Point IMPSY Web at it</h4>
  <ul>
    <li><strong>Inputs:</strong> tick your controller.</li>
    <li>
      <strong>Outputs:</strong> tick the virtual port (“IAC Driver IMPSY” or “IMPSY”). Do not tick the
      same port as an input.
    </li>
  </ul>

  <h4>3. Receive it in Live</h4>
  <ol>
    <li>
      Open Live’s <em>Settings</em> (<em>Preferences</em> before Live 12) and go to
      <strong>Link, Tempo &amp; MIDI</strong>.
    </li>
    <li>
      Under <strong>MIDI Ports</strong>, find the virtual port’s <strong>Input</strong> row and turn
      on <strong>Track</strong>. Turn on <strong>Remote</strong> as well if you want IMPSY’s CC messages
      to control Live parameters.
    </li>
    <li>
      Leave <strong>Track</strong> and <strong>Remote</strong> off on the port’s
      <strong>Output</strong> row, so Live never sends back into it.
    </li>
    <li>
      Create a MIDI track with an instrument. Set <strong>MIDI From</strong> to the virtual port and
      <strong>Monitor</strong>
      to <strong>In</strong> (or arm the track).
    </li>
    <li>
      Play your controller: you should hear the instrument, and the model answer when you pause.
    </li>
  </ol>

  <h4>Channels, notes and CCs</h4>
  <ul>
    <li>
      Each output dimension sends on the channel set in <em>Mapping → Output</em>. To give
      dimensions their own instruments, make one MIDI track per channel and pick that channel beside
      <strong>MIDI From</strong>; otherwise leave it on <strong>All Channels</strong>.
    </li>
    <li>
      To map a CC dimension to a Live parameter, enter Live’s MIDI Map Mode, click the parameter,
      then drag that dimension’s fader on the <em>Dashboard</em>: with <em>MIDI Thru</em> on, it sends
      just that dimension’s message.
    </li>
    <li>
      With <em>MIDI Thru</em> on, your own playing reaches Live through IMPSY Web. If you also enable
      your controller as a Track input in Live you will hear every note twice, so turn one of the two
      off.
    </li>
  </ul>

  <h4>Sending from Live into IMPSY Web</h4>
  <p>
    To drive the model from a Live clip instead of a controller, add a <em>second</em> virtual port.
    In Live turn on <strong>Track</strong> for that port’s <strong>Output</strong>, set a MIDI
    track’s <strong>MIDI To</strong> to it, and tick it under <strong>Inputs</strong> here. Keep the two
    ports separate: one port used in both directions makes a feedback loop.
  </p>

  <h3>Troubleshooting</h3>
  <ul>
    <li>
      <strong>Nothing arrives in Live:</strong> watch the OUT light and the MIDI console on the
      <em>Dashboard</em>. If IMPSY Web is sending, check Track is on for the port’s Input in Live
      and that the track’s Monitor is In.
    </li>
    <li>
      <strong>The model never responds:</strong> the IN light should flash when you play. If it does
      not, check the controller is ticked and that its messages match
      <em>Mapping → Input</em> (type, channel, number).
    </li>
    <li>
      <strong>Timing stutters or playback stops:</strong> browsers slow timers down in hidden tabs. Keep
      IMPSY Web in its own window that stays at least partly visible beside Live.
    </li>
    <li>
      <strong>Controller missing on Windows:</strong> some MIDI drivers allow only one program at a time.
      Close the device in Live (turn off its Track and Remote switches) so the browser can open it.
    </li>
    <li>
      <strong>Runaway notes:</strong> a port is looping. Make sure no port is ticked as both input and
      output here, and that Live is not sending to the port it receives from.
    </li>
  </ul>
</section>

<style>
  .help {
    max-width: 46rem;
    margin: 0 auto;
    line-height: 1.55;
    font-size: 0.92rem;
  }
  .help h3 {
    font-size: 1.05rem;
    margin: 1.6rem 0 0.5rem;
  }
  .help h4 {
    font-size: 0.92rem;
    margin: 1.1rem 0 0.35rem;
  }
  .help p {
    margin: 0.5rem 0;
  }
  .help ol,
  .help ul {
    margin: 0.4rem 0;
    padding-left: 1.3rem;
  }
  .help li {
    margin: 0.3rem 0;
  }
  .help a {
    color: var(--accent);
  }
</style>
