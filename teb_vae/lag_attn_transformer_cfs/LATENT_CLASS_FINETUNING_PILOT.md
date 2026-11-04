# Latent-class fine-tuning pilot: outcome separation in the CFS Transformer latents

This document describes the protocol and the code of `latent_pilot/`. The pilot takes one
pretrained `SeqVaeLagAttnTrfCfs` checkpoint, one classification fold and one seed. It fits a linear
classifier on frozen latent summaries, then fine-tunes only the posterior-mean output layers together
with the same kind of classifier. It supervises one final-hour summary per recording, preserves the
pretrained representation across the last three hours, and evaluates separation and trajectories on
recordings that neither fitting nor selection saw.

The question the code is built to answer:

> Can a small supervised change to the pretrained latent mean improve held-out separation of
> healthy versus adverse-outcome recordings near delivery, while it keeps the variation over the
> preceding three hours?

The pilot tests outcome-associated representation learning. It does not identify the time of injury,
an instantaneous fetal-health label or a physiological progression score.

## 1. What the pilot reads from the model

| Component | What the pilot uses |
|---|---|
| Inputs | Causal FHR scattering/phase features and a separate UP scattering/phase stream. The pilot keeps the checkpoint's preprocessing, channel selection, normalisation, warm-up and alignment. |
| Target branch | The causal conv-Transformer encoding of FHR history gives the target-only latent distribution, `mu_prior`, `logvar_prior`. |
| Source branch | Lag attention reads UP history. Its per-head summaries and the FHR state enter the posterior head, which returns `mu_post`, `logvar_post`. The posterior does not read the future forecast labels. |
| Latent means | $\mu^q = \mu^p + \Delta\mu$, both $(B, T, d_z)$. The pilot analyses the means, not the sampled `z_post`. |
| Forecast | The shared decoder forecasts future FHR coefficients from each latent branch. The pretraining objective does not optimise clinical-class separation. |

Code anchors: `nets/model.py` (`SeqVaeLagAttnTrfCfs`), `teb_vae/lag_attn_cfs/nets/causal_inputs.py`
(`CausalWarmupInputs.forward`), `teb_vae/lag_attn_transformer_rws/nets/encoders.py`,
`teb_vae/lag_attn/nets/heads.py` (`PosteriorHead`, `delta_mu_head`) and
`teb_vae/lag_attn_rws/nets/losses.py`.

The pilot never rebuilds a model from today's defaults. It loads the checkpoint's recorded
`model_class`, `model_kwargs` and `resolved_config.yaml`, with the matching statistics and shards,
strictly. Architecture, channel contract, trim, forecast clock, horizon, latent width and anchor
stride come from the checkpoint; the pilot config has no key for any of them.

## 2. Labels

The dataset builder (`hdf5_dataset/new_pipeline/create_new_pipeline.py`, `SUBGROUP_META`) defines:

| Stored class | Binary outcome in the pilot | Kept for reporting |
|---|---|---|
| `1`: healthy | `0` | blood-gas / no-blood-gas and CS / no-CS subgroup |
| `2`: acidosis | `1` | acidosis as a separate descriptive subgroup |
| `3`: HIE | `1` | HIE as a separate descriptive subgroup |

The codes are categories, not an ordinal severity scale. The HDF5 `target` is a class code
multiplied by a validity weight, so a zero can mean invalid data. The pilot takes the recording label
from consistent positive class codes on valid samples (`clinical_class_code` in
`teb_vae/lag_attn/eval/labels.py`, plus a recording-level consistency check).

A recording's outcome is not a label for every earlier anchor. Thus:

- The outcome loss applies to a late recording bag, not to each 4-second anchor.
- Anchors one to three hours before delivery get preservation supervision only.
- Delivery time selects and describes the windows. It is not an input to the classifier or the
  encoder, and neither are CS status, blood-gas status or outcome metadata.

## 3. Data protocol and the time axis

### 3.1 Split and cohort

The pilot uses one fold (`fold_1`) and its existing train/validation/test assignment, and records
whether the dataset was built in `holdout` or `augmented` mode. It never splits segments or anchors
randomly. It asserts disjoint GUID sets across the splits and groups recordings by patient when a
patient map is supplied.

Exposure of the test GUIDs to pretraining and checkpoint selection is recorded from the optional
GUID lists. Unknown provenance stays unknown: it is not a statement that the populations are
disjoint, and the report prints it as a limitation.

The main comparison is all four healthy subgroups against both acidosis and HIE subgroups. The
report adds a healthy-BG-only sensitivity table and CS / no-CS strata where counts permit. For every
split it reports unique recordings, segments, eligible anchors, class and subgroup counts,
missingness and recording end-time distributions. A split that lacks either binary class cannot
give a discrimination estimate, and the preflight says so.

### 3.2 Anchor timestamps

The loader trims the signals but keeps `epoch` as the untrimmed segment start in seconds relative to
delivery. For trimmed anchor index $a$ at trim $m$ minutes:

$$t_{isa} = \mathrm{epoch}_{is} + 60m + 4a, \qquad r_{isa} = -t_{isa}/3600,$$

where $r$ is hours before delivery. For example, $\mathrm{epoch} = -3600$, $m = 1$, $a = 150$ gives
$-2940$ s, which is $49$ minutes before delivery. The step length comes from the stored decimation
contract.

The pilot keeps anchors with $0 < r \le 3$. It keeps complete source segments and their history
when they start earlier than three hours: the coarse loader bound is the window start minus one
stored segment span, and an exact anchor filter follows. It does not crop the input at a window
boundary.

Collection uses the checkpoint's warm-up, source availability, `anchor_valid`, coverage and
scored-anchor masks, at dense deterministic anchors (`anchor_phase=0`, `anchor_stride=1`). So the
latent analysis inherits the forecast-availability exclusions. Every scored forecast endpoint must be
pre-delivery, including the per-channel `target_forecast_shift` of a non-stored clock. The stored UP
timeline is canonical; the pilot applies no downstream timing correction.

The pilot deduplicates `(guid, epoch, anchor)`. Where overlapping segments give the same absolute
time within a GUID, it keeps one: the one with more valid history, then a fixed segment-order
tie-break. Gaps between segment readouts stay gaps in every trajectory.

### 3.3 Late bags and trajectories

The supervised bag uses $0 < r \le 1$ hour. A recording is eligible when it has at least
`eligibility.min_late_segments` ($2$) contributing segments in that hour and a valid anchor within
`eligibility.final_anchor_within_minutes` ($30$) of the end. The same rule applies to both classes,
and the exclusions are reported.

For segment $s$ the pilot mean-pools the valid late anchors,

$$e_{is} = \frac{1}{|A^{\rm late}_{is}|} \sum_{a \in A^{\rm late}_{is}} \mu^q_{isa},$$

and pools the segment vectors into one recording vector with a recency weight,

$$\omega_{is} = 2^{-\widetilde r_{is}/h}, \qquad
v_i = \frac{\sum_s \omega_{is} e_{is}}{\sum_s \omega_{is}},$$

where $\widetilde r_{is}$ is the median time-before-delivery of the segment's included anchors and
$h$ is `bag.halflife_hours` ($0.5$). Segment averaging stops dense anchor counts from dominating, and
each recording gives one supervised loss term.

The analysis window is cut into bins of `windows.bin_hours` ($0.5$): $(2.5, 3]$, $(2, 2.5]$, …,
$(0, 0.5]$ hours before delivery at the default three hours. Within each bin the pilot averages
anchors within segments, then segments within recordings.

## 4. The model change

### 4.1 Trainable set

The pilot initialises from the pretrained checkpoint and trains only:

1. `model.posterior_head.delta_mu_head` (all its per-head linear modules when the posterior is
   head-structured);
2. a new linear binary classifier `Linear(d_z, 1)` on the pooled posterior mean.

Everything else is frozen: encoders, adapters, prior head, attention, posterior fusion and norms,
variance heads and decoder. The optimiser receives an explicit allowlist of parameters. The run logs
the trainable count for the loaded checkpoint. The backbone stays in `eval()` mode, which disables
dropout but not autograd.

With this freeze set, `mu_prior`, both latent log-variances and the attention weights are identical
before and after in a deterministic evaluation. `mu_post`, the source-conditioned KL and the
full-branch forecasts can change.

### 4.2 Objective

The pilot computes a per-coordinate mean $m_d$ and standard deviation $s_d$ from the pretrained
training latent anchors, with equal weight per recording, and freezes them. It floors the scales at
$\max(10^{-3}, 0.1\,\operatorname{median}\{s_d : s_d > 0\})$ and stops with a collapse report if no
dimension varies.

With $S(x) = (x - m)/s$ and $v_i(\theta)$ the pooled vector above:

$$\ell_i = w^\top S(v_i(\theta)) + b, \qquad
\mathcal L_{\rm cls} = \frac12 \sum_{c \in \{0,1\}} \frac{1}{N_c} \sum_{i : y_i = c}
\operatorname{BCEWithLogits}(\ell_i, y_i).$$

Training samples recordings balanced by binary class. Validation and test keep natural prevalence.

The pretrained model is a fixed teacher over the valid anchors of the preservation window:

$$\mathcal L_{\rm keep} = \operatorname{mean}_i \operatorname{mean}_{s \in i}
\operatorname{mean}_{a \in A^{3h}_{is}} \frac{1}{d_z}
\left\| \frac{\mu^q_{\theta,isa} - \mu^q_{0,isa}}{s} \right\|_2^2, \qquad
\mathcal L = \lambda_{\rm cls} \mathcal L_{\rm cls} + \lambda_{\rm keep} \mathcal L_{\rm keep},$$

with $\lambda_{\rm cls}$ = `optim.classification_weight` and $\lambda_{\rm keep}$ =
`optim.preservation_weight`. The teacher has no gradients. The original forecast loss is not part of
the objective; forecast preservation is a validation gate instead (§5). The pilot does not use the
pretraining trainer, so it does not restart the KL warm-up or the learning-rate warm-up.

## 5. Run and selection protocol

| Item | Setting |
|---|---|
| Fold / seed | `fold` / `seed` (`fold_1` / `42`) |
| Supervised / preservation / analysis windows | `windows.supervised_hours` ($1$) / `windows.preservation_hours` ($3$) / `windows.analysis_hours` (`null` = the preservation window) |
| Baseline | the same bag representation and linear classifier, all model weights frozen |
| Fine-tuning | mean-output layers and the linear classifier only |
| Optimiser | AdamW; `optim.mean_head_lr` $10^{-4}$, `optim.classifier_lr` $10^{-3}$, `optim.weight_decay` $10^{-4}$ |
| Batch | `optim.recordings_per_class` distinct recordings per binary class, all late segments per bag |
| Duration | `optim.max_epochs`, early stop after `optim.patience` epochs without a better eligible validation AUROC |
| Numeric policy | float32, gradient-norm clip `optim.grad_clip` ($1.0$), fixed seeds |
| Selection | highest recording-level validation AUROC among checkpoints that pass the preservation gate; tie: lower validation BCE, then earlier epoch |

The built-in defaults in `latent_pilot/config.py` are `max_epochs: 10` and `patience: 3` for the
fine-tune, and `max_steps: 2000` and `patience: 20` for the baseline. The committed
`configs/pilot.yaml` sets `max_epochs: 200`, `patience: 50`, and `baseline.max_steps: 200`,
`baseline.patience: 50`.

The baseline classifier is fitted first on cached vectors, with the same bags and validation rule.
Fine-tuning initialises its classifier from the baseline and the mean-output layers from the
checkpoint. The frozen model is candidate epoch zero, so selection can return it.

**Preservation gate.** On a fixed validation subset (`gates.subset_recordings`, drawn once), the
pilot measures the full-branch forecast MSE in the checkpoint's normalised coefficient space. A
candidate passes when the MSE rises by at most `gates.forecast_mse_max_increase` ($0.10$) over the
baseline, the means and losses are finite, and the posterior-delta saturation rises by at most
`gates.saturation_max_increase_pp` ($5$ percentage points). The gate covers healthy-BG and
adverse-outcome examples and checks healthy-only drift too.

For the baseline and the selected model the pilot also reports the matched-policy Monte Carlo
full-branch NLL (`mc.draws`, $8$ common random draws; `mc.large_draws` adds a convergence check) and
the source-conditioned KL diagnostics.

The test split stays closed until the selection lock exists.

## 6. Analysis

### 6.1 Primary table: one observation per held-out recording

On the same eligible test GUIDs and the same final-hour bags, the pilot compares the frozen and the
adapted model:

- AUROC and average precision, with the adverse-outcome prevalence;
- balanced accuracy at the threshold selected on validation;
- paired changes in those metrics, with 95% intervals from `bootstrap.resamples` ($1000$)
  outcome-stratified GUID resamples (patients when a patient map exists), the same resamples for
  both models;
- full-branch forecast MSE / NLL, standardised latent movement, latent covariance, effective rank
  and saturation before and after;
- a nearest-training-centroid classification in the fixed standardised $d_z$ space, as a geometry
  check that is independent of the linear head.

Acidosis and HIE keep separate colours and separate descriptive contrasts against the same healthy
controls. Small strata show counts.

### 6.2 Figures

| Figure | Content |
|---|---|
| `figure1_latent_space` | One two-component PCA fitted on the pretrained and adapted training recording-bin vectors after the fixed scaling, with no class labels. Validation and test use this saved map and identical axes. Final-hour bags coloured healthy / acidosis / HIE, with a deterministic subset of before/after arrows; explained variance reported. |
| `figure1b_latent_space_by_coverage` | The same map coloured by time, coverage, end time and CS/BG strata. |
| `figure2_supervised_axis` | Test distributions of $w^\top S(v) + b$ before and after, labelled as a supervised axis, with recording-level AUROC / AP and counts. |
| `figure3_trajectories` | Each model's frozen final-hour classifier applied to each recording-bin summary over the analysis window: group means, GUID-bootstrap bands, counts per bin, the supervised window marked, a fixed random sample of individual trajectories. Signed hours put delivery at the right. Gaps stay gaps. |
| `figure4_roc_pr` | Held-out ROC and precision-recall of the final-hour bags, both models, AUROC / AP in the legend, each model's validation-selected operating point marked. |
| `figure5_confusion` | Confusion counts at each model's threshold on one colour scale, with sensitivity, specificity, precision, NPV, $F_1$, accuracy and balanced accuracy. `precision`, `npv` and `accuracy` carry no interval, because the paired bootstrap resamples `evaluate.METRIC_NAMES` only. |
| `figure6_metrics_vs_time` | AUROC, AP, $F_1$ and balanced accuracy per trajectory bin, with bands, the shaded supervised window and adverse/total counts per bin. |
| `figure7_roc_by_time_bin` | One ROC panel per bin that holds both groups, nearest delivery first; a star marks a bin inside the supervised window. A bin with no adverse recording has no panel. |
| `figure8_confusion_counts_vs_time` | The four confusion cells per bin, stacked, one row per model. |

Figures 4 to 8 use the frozen final-hour head and the validation-selected threshold, never a per-bin
refit or threshold. `stage_report` computes them from `per_recording_test.parquet`,
`per_recording_bin.parquet` and the thresholds in `results.json`, so a finished run gains them by
regenerating its report. It writes `classification_metrics.csv`, `per_bin_metrics.parquet` and
`roc_curves_by_bin.parquet` beside the figures.

The pilot does not refit PCA per model or per bin, and does not draw t-SNE or UMAP.

### 6.3 Analysis window

`windows.analysis_hours` separates what the objective optimises from what the run measures. At
`null` it equals `preservation_hours`. When it is larger, extraction, bins, per-bin discrimination
and figures reach that far back, while the classification loss still supervises the final
`supervised_hours` and `train.build_plans` narrows the teacher term to `preservation_hours`. It
cannot be smaller than `preservation_hours`, because the teacher term needs extracted anchors. The
extraction also runs once per epoch on validation, so a longer window costs time per epoch.

### 6.4 Temporal comparison and controls

For the selected classifier, the pilot computes each recording's mean score in an early window
(`windows.early_window_hours`, $(2, 3]$ h) and a late window ($(0, 1]$ h) and takes
$\Delta_i = \mathrm{late}_i - \mathrm{early}_i$ for recordings observed in both. It compares the
$\Delta_i$ distributions of the adverse and healthy groups with a GUID bootstrap, with acidosis and
HIE shown separately, and repeats the calculation for the pretrained baseline.

Controls beyond the frozen baseline:

1. **Shuffled training labels.** One rerun of the adaptation with a fixed GUID-level permutation of
   the training outcomes; selection uses validation labels permuted independently; evaluation uses
   the true test labels once.
2. **Time, coverage and subgroup checks.** Time distributions, valid coverage, end-of-recording
   times and CS/BG strata by class, and the healthy-BG restriction.
3. **FHR-only readout** (`prior_probe: true`). A frozen linear probe on pooled `mu_prior` under the
   same split and protocol.

## 7. Package layout

```text
teb_vae/lag_attn_transformer_cfs/latent_pilot/
  __init__.py                  # package notes and the imported dependencies
  .gitignore                   # runs/, cache/, tests/fixtures/generated/
  run.py                       # RUN_ARGS, main(), CLI, STAGE_HANDLERS, guarded __main__
  config.py                    # DEFAULTS, STAGES, settings resolution, run state, selection lock
  data.py                      # manifest, labels, anchor times, masks, recording bags
  model.py                     # strict loading, trainable-head adapter, invariants
  extract.py                   # keyed latent extraction and the training-only scaler
  train.py                     # baseline, adaptation, shuffled control, prior probe, export
  evaluate.py                  # preservation gate, held-out metrics, bootstrap, confusion rates
  analyze.py                   # bin summaries, geometry, PCA projection, per-bin classification
  report.py                    # figures and report.md
  configs/pilot.yaml           # production settings
  configs/smoke.yaml           # artificial fixture settings
  tests/logic/                 # synthetic logic tests, no model and no data
  tests/                       # model, integration, figure, runner and smoke tests
  tests/fixtures/generate.py   # writes the smoke fixtures into tests/fixtures/generated/
```

Pilot config keys live in their own `latent_pilot:` namespace. The pilot keeps the original
checkpoint stamps and `model_kwargs`, stores the classifier, scaler and metadata separately from the
base model, and exports a base-model-compatible checkpoint with the adapted mean-head weights. It
never overwrites the pretrained checkpoint.

### 7.1 Launch and settings contract

- `run.py` runs from the IDE Run button with `RUN_ARGS` at the bottom of the file, as a module, and
  as `main(**RUN_ARGS)`. Direct execution adds the repository root to `sys.path` before the package
  imports.
- Importing a pilot module launches no work, reads no data and parses no arguments.
- Repository-relative paths resolve against the repository root; absolute paths pass through.
- One resolver serves every launch mode. Precedence: built-in defaults, then YAML, then the
  `RUN_ARGS` overrides, then explicit CLI values. Lists replace rather than concatenate. Parser
  defaults count as unspecified. `main()` does not mutate `RUN_ARGS`.
- CLI flags: `--config`, `--stage`, `--device`, `--run-dir`, `--resume` / `--no-resume`, and
  repeatable `--set KEY=VALUE` (the value is parsed as YAML).

### 7.2 Stages

Stages: `tests`, `smoke`, `preflight`, `extract`, `baseline`, `finetune`, `control`, `evaluate`,
`report`, `all`. `all` runs

```text
tests -> smoke -> preflight -> extract -> baseline -> finetune -> control -> evaluate -> report
```

`tests` runs pytest with the current interpreter from the repository root, without the smoke
scenario. `smoke` runs the pipeline on artificial fixtures in its own run subtree. Both stop the
sequence on failure. Each runtime stage reads the artifacts of the earlier stages and refuses when a
prerequisite has not completed. `extract` collects train and validation only; `evaluate` collects
test latents after the models, thresholds, controls and analysis settings are fixed, and fits nothing
on test data.

## 8. Tests

| Check | What the tests establish |
|---|---|
| Provenance and split | no GUID / patient overlap; label consistency; documented statistics population |
| Timestamp example | the $\mathrm{epoch} = -3600$, one-minute trim, anchor-150 example gives 49 minutes; boundary-crossing segments kept; post-delivery endpoints excluded |
| Mask and identity | no warm-up, padded or invalid anchors; no duplicate times; before and after use the same keys |
| Bag reduction | segment-then-recording weighting on unequal segment lengths; duplicated anchors do not multiply a recording's weight |
| Gradient reach | an update changes the permitted mean-output weights and `mu_post`; frozen weights, `mu_prior`, log-variances and attention stay equal |
| Determinism and reload | repeated `eval()` extraction agrees; a saved and reloaded pilot reproduces means and logits |
| Fit isolation | scaler, PCA and centroids use training data only; early stopping and threshold use validation only |
| Preservation | the gate logic and the collapse check |

`latent_pilot/tests/logic/` needs no model, no checkpoint, no HDF5 file and no GPU. The other test
files build the real net and need the repository environment. `test_fixtures.py` and `test_smoke.py`
also need the generated fixtures and skip with the command that writes them.

```bash
python -m pytest teb_vae/lag_attn_transformer_cfs/latent_pilot/tests/logic -q
python -m pytest teb_vae/lag_attn_transformer_cfs/latent_pilot/tests -q
```

## 9. Running the pilot

### 9.1 Requirements

| Requirement | Detail |
|---|---|
| Repository | The whole checkout. The pilot imports `teb_vae.lag_attn*`, `hdf5_dataset`, `train`, `utils` and `scripts`. |
| Python environment | The packages listed in `latent_pilot/__init__.py` (`torch`, `lightning`, `numpy`, `pandas`, `pyarrow`, `h5py`, `pyyaml`, `loguru`, `matplotlib`, and `scipy` / `scikit-learn` through the imported pipeline). `pytest` is needed for the `tests` stage only. |
| Working directory | Any. Paths resolve against the repository root, and direct execution of `run.py` changes to it. |
| Device | One GPU or none. `device: null` chooses `cuda:0` when available, else CPU. |
| Data | The pretrained checkpoint with the `resolved_config.yaml` beside it, the statistics file it trained under, and the fold's train / validation / test shard lists. |

### 9.2 Paths

Set the five entries under `latent_pilot.paths` in `configs/pilot.yaml`: `checkpoint`,
`statistics`, `train_shards`, `val_shards`, `test_shards`. Optionally set `provenance`
(`pretraining_guids`, `selection_guids`, `patient_map`, `statistics_population`,
`dataset_build_mode`). The `overrides` block of `RUN_ARGS` takes the same nested keys and wins over
the YAML. Both are recorded in the run's protocol.

### 9.3 Steps

1. Open `latent_pilot/run.py` and edit `RUN_ARGS` at the bottom of the file.
2. Set `RUN_ARGS["stage"]` to `"tests"` and press Run.
3. Generate the smoke fixtures once with
   `python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.tests.fixtures.generate`.
4. Set the stage to `"smoke"` and press Run.
5. Set the stage to `"preflight"` and press Run. Resolve every reported issue.
6. Set the stage to `"all"`, or run the production stages one at a time in the same run directory.

The fixture generator writes into `latent_pilot/tests/fixtures/generated/` and runs one one-epoch
fit to make a checkpoint. A smoke run uses invented identities, times and labels.

CLI equivalents:

```bash
python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.run --stage tests
python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.run --stage smoke
python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.run --stage preflight
python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.run \
    --config teb_vae/lag_attn_transformer_cfs/latent_pilot/configs/pilot.yaml --stage all
python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.run --stage evaluate \
    --run-dir <run_root>/fold_1/seed_42/<run_id> --resume
python teb_vae/lag_attn_transformer_cfs/latent_pilot/run.py --stage preflight
python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.run --set optim.max_epochs=3
```

### 9.4 What each stage writes

Under `<run_root>/<fold>/seed_<seed>/<run_id>/`. `run_root` defaults to `latent_pilot/runs`, may be
relative to the repository root or absolute, and is refused when it contains a run input (the
checkpoint, the statistics file or a shard).

| Stage | Artifacts |
|---|---|
| any | `protocol.yaml` (settings, run arguments, git revision, tolerances), `stage_state.json` (completed, failed, `selection_locked`) |
| `preflight` | `segments.parquet`, `recordings.parquet`, `split_manifest.csv`, `coverage.csv`, `preflight.json` (checkpoint identity and digest, statistics record, exposure, grouping, exclusions) |
| `extract` | `train_pretrained_*` and `val_pretrained_*` (`latents.npz`, `latent_index.parquet`, `latents.json`), `latent_scaler.json`, and the manifest and coverage rewritten with eligibility |
| `baseline` | `baseline_classifier.pt`, `baseline_fit.json` |
| `finetune` | `gate_subset.json`, `gate_reference_preservation.json`, `pilot_checkpoint.pt`, `adapted_model/adapted_model.ckpt` (base-model-compatible, classifier keys excluded, the resolved config copied beside it), `train_adapted_*`, `val_adapted_*` |
| `control` | `control_classifier.pt`, `control_fit.json`, and `prior_probe_*` when the probe is on |
| `evaluate` | `selection_lock.json`, `test_pretrained_*`, `test_adapted_*`, `pretrained_preservation.json`, `adapted_preservation.json`, `projection.json`, `metrics.csv`, `per_recording_test.parquet`, `per_recording_bin.parquet`, `trajectory_bands.parquet`, `figure_bags.npz`, `results.json` |
| `report` | the nine figures of §6.2 under `figures/` (in `figure_format`, `pdf` by default), `classification_metrics.csv`, `per_bin_metrics.parquet`, `roc_curves_by_bin.parquet` and `report.md` |

`report.md` prints `not measured` for every value the run did not establish.

### 9.5 Resume, re-report and refusals

- A new run leaves `run_dir` unset and gets a fresh directory.
- To continue a run, name it and set `resume: true`. Completed stages are skipped. Resume is
  stage-level: an interrupted fit restarts from its beginning.
- To re-report, name the directory without `resume`. Only `tests`, `smoke`, `preflight` and
  `report` run this way. `report` needs neither the shards nor the checkpoint.
- `settings_differences` treats a key that is absent from the stored protocol and carries the
  schema default as a schema addition, and logs it. A changed value still refuses.
- The runner refuses: a resume under changed settings, a re-run of a fitting stage in a finished
  directory, a stage whose prerequisite never ran, and a read of the test split before the
  selection lock exists.

### 9.6 When a stage fails

`stage_state.json` records the failed stage and its reason, and the sequence stops there. Keep the
failing stage name, the traceback, the console log, `protocol.yaml`, `stage_state.json`, and, where
they exist, `preflight.json` and `coverage.csv`. A changed protocol setting needs a new run
directory.
