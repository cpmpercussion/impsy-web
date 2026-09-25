# Releasing impsy-web

`main` deploys to GitHub Pages on every push (`.github/workflows/deploy.yml`),
after the unit tests pass. Releases are **tags on `main`** that mark a coherent
set of features. They give the project a changelog and a known-good build to
point people at. The live site always shows the version and commit it was built
from (footer), so bug reports can name the exact build.

## Versioning

[Semver](https://semver.org), staying on `0.x` until the feature set matches
the AUv3 plugin:

- **minor** (`0.2.0`): new features, or a change to the `.toml` / `.log`
  formats the app reads or writes.
- **patch** (`0.1.1`): fixes only.

## Cutting a release

1. Make sure `main` is green (CI workflow) and the deploy has finished.
2. Run the [manual QA checklist](#manual-qa-checklist) against the live site.
3. Bump the version on a branch and merge it:
   ```bash
   npm version 0.2.0 --no-git-tag-version   # updates package.json + lockfile
   ```
4. After the bump is merged, tag the merge commit on `main` and push the tag:
   ```bash
   git checkout main && git pull
   git tag v0.2.0
   git push origin v0.2.0
   ```
   `.github/workflows/release.yml` checks that the tag matches `package.json`
   and creates a GitHub Release with auto-generated notes (from merged PRs).
5. Edit the release notes: add a one-line summary and the **IMPSY
   compatibility** line (below).

## IMPSY compatibility

The Python [IMPSY](https://github.com/cpmpercussion/impsy) owns the `.toml`
config and `.log` formats, and model conventions. Each release notes which
IMPSY version it was checked against, e.g.:

> Compatible with IMPSY 1.1.0 configs, logs and `.tflite` models (Flex-free).

Format changes (such as velocity capture, #13) land in IMPSY first, then here.

## Manual QA checklist

Automation (unit tests, type check, build) can't cover real hardware or
interop. Before tagging, check on the **live site**:

- [ ] Footer shows the expected version and commit.
- [ ] Demo model auto-loads; no errors in the browser console.
- [ ] **Chrome on macOS** with a real MIDI controller and synth/DAW:
  - [ ] CC and Note On input move the right faders (red); the IN LED and
        console respond.
  - [ ] Pausing hands over to RESPONSE and the model plays the synth; the OUT
        LED, Last Output card and console update.
  - [ ] Call → response handover sounds right by ear at default parameters.
  - [ ] MIDI Thru on/off behaves as expected.
- [ ] **One other platform** (Windows or Linux; Chrome, Edge or Firefox)
      loads the model and sends/receives MIDI.
- [ ] **Config interop**: export a `.toml`, load it in Python IMPSY (and
      back); an IMPSY `configs/*.toml` imports correctly here.
- [ ] **Log interop**: record a session, download the `.log`, and
      `impsy dataset` processes it.
- [ ] Phone-width layout is usable (tabs, faders, console).
