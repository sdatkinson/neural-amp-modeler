# NAM: Neural Amp Modeler (Parametric fork)

[![Build](https://github.com/phillipmself/neural-amp-modeler-parametric/actions/workflows/python-package.yml/badge.svg)](https://github.com/phillipmself/neural-amp-modeler-parametric/actions/workflows/python-package.yml)

A fork of [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler) that models a rig
**across its knob range** instead of at one fixed setting. You capture the same amp or pedal at a
few dozen knob positions and train a single model on all of them, ending up with Gain and tone
controls that move the sound the way the real ones do — including at settings you never captured.

What the fork adds:

- **`nam-capture`** — a desktop app that plans the capture set, drives the reamp, measures latency
  per capture, and writes ready-to-train configs. This is the entry point; everything else is
  reachable from it.
- **Two knob-conditioned architectures** — `ConcatWaveNet` (knob values appended to the audio at
  every timestep) and `HyperWaveNet` (a small network generates the WaveNet's weights from the knob
  values). Both train from the same captures.
- **Active-learning capture selection**, which proposes what to capture next. Experimental.

Trained models run in [NamParametricPlugin](https://github.com/phillipmself/NamParametricPlugin)
(VST3 + Standalone), which reads the knobs out of the `.nam` and builds sliders for them.

Upstream NAM's own workflow is untouched and still works exactly as before.

## Install

```bash
conda env create -f environments/environment_gpu.yml   # macOS / CPU: environment_cpu_apple.yml
```

```bash
conda activate nam && pip install -e ".[capture-app]"
```

The conda environment installs the package itself; the second line adds PySide6 for the capture app.

## Capture

```bash
nam-capture
```

**Project.** *New Project…*, pick an empty folder, and point it at the DI WAV you'll play through
the rig. A separate validation input is optional but worth it — held-out settings judged on a
held-out signal. Everything lives in that folder and is saved after every capture, so quitting
mid-session costs nothing.

**Input files.** No reamp signal handy? Take `inputTrunc.wav` and `validation.wav` from
[this repo's releases](https://github.com/phillipmself/neural-amp-modeler-parametric/releases/tag/training-inputs-v1).
They're truncated cuts of NAM's standard `input.wav` v3.0.0 — 38 s and 7 s against the original's
3:10 — and models trained on them still come out well. The trim is what makes a parametric session
practical: the input gets played through your rig once per knob setting, so cutting each pass
fivefold takes a 90-capture run from roughly five hours of reamping down to under one.

For standard single-setting NAM captures, keep using the full-length v3.0.0 file. You only reamp
once there, so there's nothing to save, and the stock trainer keys off blip landmarks that a
truncated cut doesn't preserve.

**Knobs.** One row per control you want the model to learn:

| Field | What it means |
| --- | --- |
| **min / max** | The numbers printed at the knob's physical extremes. `0`–`10` for most, but use whatever the faceplate actually says. |
| **step** | The finest increment you can honestly dial by hand (default `0.5`). Every planned setting is snapped to this grid, so you're never asked for `6.37`. |
| **avoid zero** | Keeps the planner off zero for this knob. Use it wherever zero mutes the rig or collapses it into something degenerate — usually Gain, or a Volume that goes silent. A capture of near-silence teaches the model nothing and destabilizes training. |
| **is gain** | Marks the drive control (at most one). The corner captures then sweep every tone-stack extreme at *both* ends of the gain range, since a tone stack behaves differently clean than saturated. |

**Plan.** Two things go into a plan, and you want both:

- **LHS points** — a Latin hypercube over the knob space. Space-filling, so coverage stays even as
  the knob count grows, where a grid would need kⁿ captures. The app recommends `15 × knobs`
  training points and `3 × knobs` validation points.
- **Corners** — the knob-range extremes, which bound the model's behavior. LHS covers each knob's
  own range well but almost never lands on a corner of the *joint* space, so it can't stand in for
  these. At four or more knobs the corners are a 2ⁿ⁻¹ fractional factorial rather than every vertex,
  which keeps the knobs mutually orthogonal without the count exploding.

*Generate plan* builds the list. *Add corner captures to current plan* appends corners later without
regenerating the LHS points or discarding captured progress.

**Audio I/O.** Pick your interface and set the output and input channels for the amp route. Then
enable the loopback and give it a second output/input pair **on that same interface**, patched
together with a short cable. This is strongly recommended.

Latency is measured per capture from timing blips. Without a loopback those blips return through the
amp, so the measurement drifts as gain and distortion smear their attack. The loopback sends a clean
copy down a parallel path and the delay is taken from that instead, holding steady across the whole
gain range. It only works if both routes travel the same device chain: anything the capture route
crosses that the loopback doesn't — an ADAT link above all — is invisible to it and can jump by
several samples whenever that link re-locks. Run *Route test* before you start.

**Capture.** *Capture next* walks the plan in order. Dial the knobs as shown, hit the button, and it
plays the input through your rig, records the return, measures the delay, checks for clipping and
dropouts, and writes the WAV.

## Train

The Capture tab has *Export ConcatWaveNet Configs* and *Export HyperWaveNet Configs*. Either writes
a matched `model_*.json` and `learning_*.json` into the project folder, next to the `data.json`
that's been kept current all along. Then:

```bash
nam-full-parametric data.json model_hyper.json learning_hyper.json outputs
```

Training writes into a timestamped subfolder of `outputs/`. The file you want is
**`model_parametric.nam`** — load it in
[NamParametricPlugin](https://github.com/phillipmself/NamParametricPlugin) and your knobs appear as
sliders.

A HyperWaveNet run also drops a second file, `model.nam` — a plain fixed-setting WaveNet baked at
your knobs' default positions, which loads in the stock NAM plugin like any ordinary capture. Handy
for A/B'ing your parametric model against a conventional one, but it is not the parametric model
itself. ConcatWaveNet runs produce only `model_parametric.nam`.

### Which architecture?

There's no clear winner yet. Both train from the same captures, so train both and listen — more
people comparing them on real rigs is how a default eventually gets picked.

The one firm difference is runtime cost. A HyperWaveNet generates its weights whenever the knob
values change; once they settle, the audio path is a stock 8-channel WaveNet costing no more per
buffer than an ordinary NAM capture. A ConcatWaveNet instead carries the knob values as extra
channels through every layer, and has to be wider to hold its own — `8 + 2 × (knobs − 1)` channels,
so a 5-knob model runs 16 wide, on every buffer, forever.

## Active learning (experimental)

The **Active Learning** tab is hidden by default; launch the capture app with
`nam-capture --active-learning` to show it. It runs a PANAMA-style
([arXiv 2509.26564v1](https://arxiv.org/html/2509.26564v1)) loop: it trains a throwaway ConcatLSTM
ensemble on the captures you have so far, finds the knob settings where the ensemble members
disagree most, and proposes those as your next round of captures. The idea is that disagreement
marks the regions the model understands least, so capturing there buys the most per reamp.

**In practice it has not reliably beaten a plain LHS + corners plan, so treat it as an experiment
rather than the default.** Three things work against it:

- **It goes corner-hunting first.** Disagreement is highest where the model is least constrained,
  and that's the edges of the knob space, so early rounds spend most of their proposals out on the
  rails. The corner set in the plan above front-loads exactly that coverage — cheaply, and without
  burning training rounds to rediscover it.
- **The edges aren't where people play.** Gain gets used across its whole sweep, but EQ knobs rarely
  sit pinned at 0 or 10. Captures spent nailing down extreme corners are captures not spent on the
  interior, so the model ends up weaker where you actually set the controls, and validation grades it
  on territory nobody visits.
- **The loop is a chore.** Every round wants a fast CUDA GPU; without one you're renting a box and
  shuttling the project folder back and forth, reamping in between. LHS + corners is one plan and one
  capture session.

Reach for it after you've worked through a full LHS + corners plan, if you want to push further and
have reamp time to spend.

[docs/active_learning_usage.md](docs/active_learning_usage.md) has the full workflow, including
running rounds on a remote GPU box, with example configs in
[nam_full_configs/active_learning/](nam_full_configs/active_learning/).

## Standard (non-parametric) NAM

The original workflow is untouched: `nam-full` still trains a single fixed-setting model and exports
a standard `.nam`. Those play in the upstream
[NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin) — the plugin for
anything non-parametric. It can't load a `model_parametric.nam`, which is what
NamParametricPlugin is for.

For more information about the upstream NAM ecosystem, check out
https://www.neuralampmodeler.com/. Its documentation — which covers standard NAM, not the parametric
additions here — is at https://neural-amp-modeler.readthedocs.io, and builds locally with
`make html` (Windows: `make.bat html`) from `docs/`.
