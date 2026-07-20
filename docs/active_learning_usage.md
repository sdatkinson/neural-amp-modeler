# Active-learning capture selection (PANAMA-style)

> **Attribution.** This workflow is adapted from **PANAMA** (Parametric Active-learning for Neural Amp
> Modeling Assistance), a NAM fork published as arXiv
> [2509.26564v1](https://arxiv.org/html/2509.26564v1). The round-based query-by-committee loop, the
> disagreement (output-variance) acquisition objective, the `sigmoid(latent)` control-vector
> optimization, and the threshold-grouping clustering are all due to the PANAMA authors. We
> re-implement the *approach* here for a different model, single-GPU, and our `data.json` format.

## What this is for

When building a **parametric** capture set (one knob-conditioned amp model trained on many knob
settings) the hard question is *which settings to capture next*. Guessing wastes reamping time.
Active learning answers it: train a small **ensemble** of cheap models on the captures collected so
far, then find the control settings where the ensemble members **disagree most** (query by
committee) and propose those as the next captures.

The ensemble member is a **ConcatLSTM** — a parametric LSTM that tiles the encoded control vector
across time and concatenates it to the audio. It is a **disposable acquisition proxy**: its only job
is to produce a cheap, differentiable disagreement signal over the control space. The model you
actually ship is still the **HyperWaveNet**; the LSTM ensemble is thrown away each round.

Design details and rationale live in
[`docs/panama_active_learning_lstm_plan.md`](panama_active_learning_lstm_plan.md).

## Capture app integration

The capture GUI (`python -m nam.capture.gui.main`) plans a starter set of 10 train / 5
validation captures via a Latin hypercube over the knobs, then hands rounds off to the
**Active Learning** tab. "Start round" runs `python -m nam.capture.al_runner` against the
project folder as a background process; each round regenerates the AL configs from
`capture_project.json` fresh (`ny` 22050, physical batch auto-sized to use 75% of each
GPU's available VRAM, and gradient accumulation chosen to approximate PANAMA's effective
batch of 512). Each member records the resolved plan in `batch_plan.json`; knob `step` and
`avoid_zero` metadata are honored in the proposals. Proposals from a
finished round import back into the project as pending train captures automatically (and
the tab offers to import any left over when you reopen a project).

To train remotely: use "Export remote runner files" to write the `active_learning/`
configs and `run_active_learning.sh`, copy the whole project folder to the training
machine, run `./run_active_learning.sh`, then copy the `active_learning/` folder back and
reopen the project in the capture app -- it is the single writer of
`capture_project.json`, so a remote round never touches it directly.

## Example configs

Ready-to-edit examples (Gain/Tone continuous 0–10, Boost switch Off/On) live in
[`nam_full_configs/active_learning/`](../nam_full_configs/active_learning/):

- `model.json` — `net.name = "ConcatLSTM"`, with architecture/loss mirroring PANAMA's LSTM
  (3 layers, hidden size 18, full BPTT, and MSE plus seven-resolution log-mel loss). Its
  `params` block **must match** the params of the parametric production model you are capturing for.
- `learning.json` — PANAMA's 50 epochs and learning rate 0.008. At runtime, each worker
  chooses a memory-fitting physical batch and `accumulate_grad_batches` so the effective
  optimizer batch remains close to 512 across 16–24 GB GPUs. The `val_dataloader.batch_size`
  is independently memory-bounded: validation does no backprop, and ESR is
  a ratio, so validation should run in as few batches as possible (ideally one) or `val_loss` — and the
  best-checkpoint pick — drifts with the batch count. The accelerator is rewritten at runtime to
  `cuda → mps → cpu`, so it is device-agnostic. Unlike PANAMA (which skips in-loop validation), our
  `train_ensemble` keeps validation on to pick the best checkpoint.
- `data.json` — a 10-setting starter set produced by the script below, used as the round-0 seed.

## Workflow

### Step 1 — Starter set (round 0 seed)

Generate the first ~10 settings with a Latin-hypercube over the continuous knobs and balanced switch
assignment:

```bash
python scripts/make_starter_settings.py \
  --model-config nam_full_configs/active_learning/model.json \
  --n 10 --n-validation 2 \
  --output data.json
```

Useful flags: `--input-wav` (the reamp input, default `input.wav`), `--full-grid` (LHS-continuous ×
every switch combination), `--seed`, `--y-path-prefix`, `--no-rounding` (skip capture-grid
quantization), the `--start-seconds`/`--stop-seconds`/`--ny` window controls, and their
`--validation-*` variants. Continuous values are snapped to the realizable knob grid (default 0.5)
so the recorded setting equals the setting a human can actually dial.

The capture-app active-learning runner uses PANAMA's 22050-sample training windows. The
standalone starter script retains configurable `--ny` and `--validation-ny` values for
experimentation.

The script prints a capture checklist. **Reamp `input.wav` at each listed setting**, save each output
to its `y_path` wav, and you have a trainable round-0 `data.json`.

### Step 2 — One active-learning round

One CLI invocation == one round:

```bash
python scripts/active_learn.py \
  --round-idx 0 --output-dir al_runs \
  --data-config data.json \
  --model-config nam_full_configs/active_learning/model.json \
  --learning-config nam_full_configs/active_learning/learning.json \
  --g-opt-input-wav input.wav
```

Each round:

1. Trains a 4-member ConcatLSTM ensemble on one device (different seed per member). By default the
   members train **serially**, except on a multi-GPU CUDA box where each member gets its own GPU.
   Pass `--max-workers N` to train `N` members concurrently — see [Parallel training](#parallel-training).
2. Runs the disagreement g-optimizer: for every switch combination it Adam-**ascends** a latent `z`
   (mapped to in-range continuous knob values) to maximize member-output variance.
3. Clusters the candidates per switch combination, quantizes survivors to the capture grid, dedupes,
   and takes the global top `--max-per-round`.
4. Writes, in `--output-dir`:
   - `proposed_captures_round_{i}.json` — the proposed settings in **user units** (0–10, enum names),
     with suggested `y_path` filenames, plus a printed checklist.
   - `aggregated_data_config_{i}.json` — the previous `data.json` with the proposals appended to
     `train` (placeholder `y_path`s), `common` and `validation` preserved.

Then it **stops**. You reamp `input.wav` at the proposed settings, fill in the `y_path`s in
`aggregated_data_config_{i}.json`, and run the next round:

```bash
python scripts/active_learn.py \
  --round-idx 1 --output-dir al_runs \
  --model-config nam_full_configs/active_learning/model.json \
  --learning-config nam_full_configs/active_learning/learning.json
```

For `--round-idx i > 0` the driver defaults `--data-config` to
`<output-dir>/aggregated_data_config_{i-1}.json`, so you only pass `--data-config` explicitly for
round 0.

Other useful flags: `--ensemble-size` (default 4), `--max-workers` (parallel training, below),
`--num-restarts`, `--num-steps`, `--g-opt-ny`/`--g-opt-batch-size`, `--use-mel` (PANAMA's
multi-resolution mel-variance term), `--seed`, `--ckpts` (reuse member checkpoints instead of
retraining), `--no-plot`.

### Parallel training

Ensemble members are independent (each has its own seed and its own trainer, they never interact), so
they can train concurrently. `--max-workers` controls this:

- **Unset (default).** Serial on a single device; on a **multi-GPU CUDA** box, one member per GPU
  (`min(ensemble_size, gpu_count)` workers). The default never over-subscribes a single device, so it
  can't OOM by fanning out.
- **`--max-workers N`.** Train `N` members at once (capped at `--ensemble-size`), round-robined across
  whatever GPUs exist. On a **single GPU** this over-subscribes that one card — an explicit opt-in,
  because peak memory scales with the number of concurrent members.

Workers run as separate `spawn`ed processes; results are keyed by member index, so the per-member
seeds and checkpoint order are identical to the serial path (reproducible either way). Dataloader
`num_workers` is forced to 0 under parallel training (nested workers under spawned processes are
unsafe), and the per-member progress bars are suppressed.

**Single-GPU runs should normally remain serial.** Automatic sizing gives each worker a physical
batch based on the free VRAM it sees at startup. Explicitly starting several workers on one GPU can
therefore overcommit memory if they probe concurrently; use a fixed conservative `--batch-size` when
you deliberately opt into single-GPU concurrency. Multi-GPU runs remain one member per GPU by
default, and each member sizes itself from its assigned device:

```bash
python scripts/active_learn.py \
  --round-idx 0 --output-dir al_runs \
  --data-config data.json \
  --model-config nam_full_configs/active_learning/model.json \
  --learning-config nam_full_configs/active_learning/learning.json \
  --g-opt-input-wav input.wav \
  --max-workers 4
```

Lower `--max-workers` or set a smaller explicit batch if an over-subscribed run hits OOM. Without
NVIDIA MPS (e.g. on Colab), same-GPU members time-slice rather than running truly simultaneously.

### Step 3 — Train the production model

Once you have grown the capture set, train the shipped **HyperWaveNet** on the aggregated `data.json`
exactly as for any parametric model (see `nam_full_configs/parametric/`). The LSTM ensemble was only
ever an acquisition tool and is not part of the final model.
