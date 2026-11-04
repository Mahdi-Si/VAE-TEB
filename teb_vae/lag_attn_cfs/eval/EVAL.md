# Evaluating the lag-attention model

This guide explains how to evaluate `teb_vae.lag_attn_cfs`, what each run writes, and what each analysis computes. [FIGURE_GUIDE.md](FIGURE_GUIDE.md) explains each emitted figure.

This evaluation package is a fork of `teb_vae/lag_attn_rws/eval`, adapted for a different prediction target. The model forecasts wavelet-modulus and phase-harmonic coefficients produced by a strictly one-sided filter bank. One forecast block contains $H\times C_{\mathrm{keep}}$ coefficients. The checkpoint sets the horizon $H$ and the retained channel count $C_{\mathrm{keep}}$. This target changes the model calls, masks, baselines, units and analyses. The [divergence register](#the-divergence-register) lists each difference from the source package and follows `divergences.json`.

The same pipeline evaluates both cfs cells. `teb_vae/lag_attn_transformer_cfs/eval` binds this runner, collection pass and analysis registry through its own `ModelBinding` (see [The model binding](#the-model-binding)). Everything below applies to either checkpoint. The two cells differ only through the checkpoint's own geometry and forecast density, which every run records.

## Contents

- [Companion documents](#companion-documents)
- [Reading order](#reading-order)
- [Configuration-dependent geometry](#configuration-dependent-geometry)
- [Essential terms](#essential-terms)
- [What a run is](#what-a-run-is)
- [Output layout](#output-layout)
- [The four layers](#the-four-layers)
- [Configuration reference](#configuration-reference)
- [The analyses](#the-analyses)
- [Interpretation limits](#interpretation-limits)
- [The divergence register](#the-divergence-register)
- [Operations](#operations)

## Companion documents

- [FIGURE_GUIDE.md](FIGURE_GUIDE.md) describes every figure the run writes: what each panel shows and how to read it.
- [ATTRIBUTION.md](ATTRIBUTION.md) describes the Captum attribution pass: design, baselines, methods and interpretation limits. The [`attribution`](#attribution) section below covers only what this evaluator saves.

## Reading order

1. Run the checkpoint evaluation with the command below.
2. Inspect `preflight.json` and `summary.json` for status and configuration.
3. Read `forecast` to compare the model with simple predictors.
4. Read `coupling` to compare target-only and source-conditioned forecasts.
5. Read `perm_control`, `source_null`, `time_shift`, `latent` and `calibration` before you interpret a predictive gap or a KL value.
6. Read `mc_error` before you quote `pred_gap_mc_nats` at the run's $K$.
7. Use `time_to_delivery`, `second_stage` and `cross_subgroup` for recording-level group comparisons.
8. Run the offline verifier. A successful process exit does not mean that the acceptance checks passed.

## Configuration-dependent geometry

The checkpoint and the configured dataset set the channel counts, alignment, horizon, anchor stride, lag window, scored horizon and anchor count. The two cfs cells ship different values for several of them. This document therefore states geometry symbolically:

| Symbol | Meaning |
| --- | --- |
| $H$ | The horizon. |
| $F$ | The anchor floor (`warmup_period`). |
| $T_{\mathrm{valid}}$ | The valid length. |
| $L=\texttt{max\_lag}+1$ | The lag count. |
| $c_y$, $c_u$ | The declared target and source widths. |
| $C_{\mathrm{keep}}$ | The retained target width. |

For one run, read these values from `resolved_config.yaml`, `preflight.json` (anchor geometry, warm-up budget, lag-support margin, group delays), `summary.json` (`run_context`, `likelihood_structure`) and `collection.json` (`target_keep_index`, `block_width`). Do not combine values from runs of different configurations.

The shipped `configs/default.yaml` of this cell sets `horizon: 10`, `c_y: 80`, `c_u: 46` (integer phase-harmonic operator), `causal_align_reference: null` (unaligned), `causal_target_forecast_clock: stored`, `anchor_stride: 5`, `max_lag: 90`, `lag_kv_source: conv_stem` and `alibi_slope_scale: 0.0`.

## Essential terms

| Term | Meaning |
| --- | --- |
| Recording or GUID | One recording, identified by `guid`. It can contribute many segments. |
| Segment | A fixed-length part of a recording given to the model. |
| Anchor | The time within a segment from which a forecast starts. |
| Horizon | The future steps predicted from an anchor. Each stored step spans $4$ seconds. |
| Coefficient or channel | A numerical feature produced by the signal transform. A channel follows one such feature through time. |
| Forecast block | All retained target channels over the forecast horizon at one anchor. |
| Base and full | The forecasts from target history alone, and from target history plus source history. |
| Latent state | A compact, uncertain representation of the history, described by a distribution with a mean and variance. |
| Prior and posterior | The target-only and source-conditioned latent distributions. |
| NLL | Negative log-likelihood: a predictive score for which lower is better. A negative log-density can be negative. |
| Predictive gap | The base score minus the full score. A positive value means that the source improved that score. |
| KL divergence | A measure of the difference between the two latent distributions. It measures a change in belief, not a predictive improvement. |
| Monte Carlo or MC | An estimate from several random latent draws. The predictive score averages likelihoods before it takes their negative logarithm. |
| Bootstrap interval | An uncertainty interval formed by repeated resampling of recordings. |
| Cohort | A group of recordings, such as a clinical class or subgroup. |
| Warm-up | The initial interval a causal filter needs before its output no longer depends on assumed pre-recording history. |
| Lag | A source position, in stored steps before an anchor. |
| Preflight or guard | A check before evaluation that rejects incompatible data, settings or checkpoints. |

## What a run is

One command reads one checkpoint and writes one run directory:

```bash
python -m teb_vae.lag_attn_cfs.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt
```

The runner starts from the checkpoint's `resolved_config.yaml`. It deep-merges the committed `eval/configs/eval_overrides.yaml` over that file. A deep merge replaces the specified settings and keeps the other training settings. The overrides select the causal holdout shards, set the evaluation batch size, set `load_fields` (the clinical fields, `guid`, `epoch` and `second_stage_onset` included) and set `eval_config`. The summary records the original and the replacement value of every overridden key.

The overrides do not replace the statistics file. The run inherits its own file, so the network sees inputs and targets on its training scale. Preflight rejects configurations that contradict the checkpoint.

### Dense anchor geometry

Evaluation decodes every valid anchor. Training uses spaced anchors (`anchor_stride`) to reduce gradient correlation and memory use. Evaluation calls `model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1)`. This matches `SeqVaeLagAttnCfsTask.resolve_anchor_geometry('test', batch)` and the validation and test geometry.

The number of decoded anchors is the checkpoint's `anchor_ceiling - warmup_period` (`metrics.expected_anchors_per_sample`). On the `stored` forecast clock this is $T_{\mathrm{valid}}-F$. A `physical` forecast clock removes more trailing anchors, according to its largest label advance. The run records the evaluation geometry and the training stride in `run_context` and in `results.anchor_geometry`.

### The collection pass

The shared collection pass scores four branches at every valid anchor with $K$ Monte Carlo draws and common random numbers: `base`, `full`, `shuffled` and `base_shuffled_mu`. It also computes `kld_source_null`: it passes a zeroed source through the source gate, adapter and encoder. This fifth arm returns only latent parameters $(\mu^{q,\mathrm{null}},\ell^{q,\mathrm{null}})$. It adds one source encode, no decode and no `randn_like` call, so it does not advance the latent sampling stream.

`metrics.evaluate` scores every batch. Some batches cannot be paired by the permutation controls: a single sample, or one recording that holds more than half of the batch. Such a batch is still scored in full. Only its `CONTROL_COLUMNS` are `NaN`: the `mc_`/`mean_` block scores of `shuffled` and `base_shuffled_mu`, and `source_conditioned_kl_shuffled_raw`. The per-recording aggregation averages each control over the samples that carry it. `results.controls` and `n_batches_skipped_too_small` count both kinds of unpairable batch.

The pass saves `per_sample.csv`, with one row per segment and its clinical labels, and `per_anchor.parquet`, keyed by `(guid, epoch, anchor_index)`. It also saves vector sidecars and aggregate readouts.

### Offline reruns

The analyses that read saved tables can run again without a checkpoint, model or GPU:

```bash
python -m teb_vae.lag_attn_cfs.eval.run --output-dir <a finished run> --only coupling
```

`--max-batches` limits evaluation to an initial batch prefix and is for smoke runs only. `eval_config.max_samples` selects a seeded, stratified sample across the split. The two caps are not interchangeable: a prefix of the unshuffled split can contain only one subgroup and class.

### The verifier

The same offline module checks a finished run and generates the arm tables:

```bash
python -m teb_vae.lag_attn_cfs.eval.verify <run>/eval_results/summary.json
python -m teb_vae.lag_attn_cfs.eval.verify --runs <dir-of-runs> --out <tables.md>
```

- The gate form also takes `--json-out <path>` to write the machine-readable report beside the console table.
- The gate exits non-zero when any criterion is FAIL. An INCONCLUSIVE criterion does not block it, but the report lists it.
- The table form writes to `RESULTS_arms.md` when `--out` is not given (`verify.DEFAULT_ARMS_OUT`).
- `verify.py` has its own `RUN_ARGS` (`summary`, `json_out`, `runs`, `out`) and runs from the IDE Run button. Set exactly one of `summary` or `runs`.

`verify` reads only the files a run left behind: no model, no shard, no `torch`. The layering test walks its imports with `torch` on the forbidden list. The gate reads `pred_gap_mc_nats`, the Monte Carlo marginalised score.

The arm tables key every row by the swept value read from each run's own `resolved_config.yaml`, never by directory name. The swept keys are `anchor_stride`, `warmup_period`, `horizon` and `horizon_depth`. Each row also reads the run's training `metrics_history.csv` for the epoch count, the final `val/kld_active_frac` and the collapse verdict. The collapse verdict comes from `teb_vae/lag_attn_rws/collapse.py::is_collapsed`, which is imported because it is stdlib-only arithmetic. The table form also writes a cross-cell table against the transformer cfs cell.

`check_run.py` (in the model package) checks training while a run is in progress, from `train_results/metrics_history.csv`. `eval/verify.py` checks a finished checkpoint, from the held-out evaluation in `summary.json`. Both work without a checkpoint, dataset shard or PyTorch, but they read different evidence.

### The forecast density every score uses

A block score is the joint negative log-density that the checkpoint's own objective defines. Two terms of the objective are part of the likelihood, not weights on it. `metrics.forecast_likelihood_terms` passes both, detached, to every density readout: the branch block scores, the Monte Carlo marginal and its error, the per-channel and per-horizon splits, the trivial baselines, calibration, the oracle probe, `occlusion`, `time_shift` and the attribution readouts.

- **The per-channel scored horizon.** A $(H, C_{\mathrm{keep}})$ cell mask $m_{\tau,c}=\mathbb 1[\tau<H_c]$, with $H_c\le H$. Training resolves $H_c$ from `target_phase_fast_cutoff_hz` / `target_phase_fast_horizon` and the shards' phase-leg frequencies (`teb_vae/lag_attn_cfs/scored_horizon.py`). A block then sums $\sum_{\tau,c} m_{\tau,c}$ cells, not $H\cdot C_{\mathrm{keep}}$. Point-error readouts use the cell mask alone.
- **The AR(1) residual** (`forecast_ar_residual`). Each cell is scored on its innovation $e_\tau = r_\tau-\phi_c\,r_{\tau-1}$, with $r=x-\mu$, $r_{-1}=0$ and the learned $\phi_c=\tanh(a_c)$. The recursion restarts after a masked step. PIT, coverage and the standardised residual use $e$.

A cell can enable either term, both or neither. With neither, every score is the factorised all-cells score. `summary.json` and `collection.json` record the density as `likelihood_structure`:

- `forecast_ar_residual`;
- the per-block mean $\phi$ (`ar_coef_mean_st`, `ar_coef_mean_ph`);
- `scored_cells` and `block_cells` ($H\cdot C_{\mathrm{keep}}$);
- the per-channel vectors `scored_horizon_per_channel` ($H_c$) and `ar_coef_per_channel` ($\phi_c$, `null` without the AR term).

The console prints the same record as one line. `reconcile_with_checkpoint` re-resolves `target_scored_horizon` from the configured rule and compares it with the checkpoint's vector. A config that disagrees with the checkpoint's density is refused. The objective weights (`target_channel_weight`, `horizon_weight` and the horizon half-life) are not applied, because they would turn a log-density into a weighted score.

### Launching a run

You can start `run.py` from the command line or from the IDE Run button. Every flag has a matching key in the module-level `RUN_ARGS` dict near the bottom of `run.py`. Resolution is per key: a command-line value wins, then the `RUN_ARGS` entry, then the default. `summary.json` records the source of each value (`cli`, `config`, `default`) under `arguments`. A `RUN_ARGS` key that is not a flag raises at startup.

To run from the IDE:

1. Open `run.py` and find `RUN_ARGS`.
2. Set exactly one of `checkpoint` or `output_dir`.
3. Press Run. The working directory moves to the repository root automatically.

| Flag (`RUN_ARGS` key) | Meaning |
|---|---|
| `--checkpoint` (`checkpoint`) | The `.ckpt` to evaluate. Required unless `--output-dir` names a finished run, whose tables are then read again. |
| `--output-dir` (`output_dir`) | The run directory. Default: a timestamped directory under `out_dir_base/<tag>-eval`. |
| `--overrides` (`overrides`) | The override delta merged over the checkpoint's resolved config. Default: the binding's committed `eval/configs/eval_overrides.yaml`. |
| `--device` (`device`) | Torch device. Default: `cuda:0` when available, else CPU. |
| `--num-samples` (`num_samples`) | Monte Carlo draws $K$. Overrides `eval_config.num_mc_samples`. The collection provenance records the draw count actually used. |
| `--max-batches` (`max_batches`) | Stop after this many batches; for smoke runs only. The collection provenance records it, so a smoke run's tables are never reused as a full run's. |
| `--only`, `--skip` (`only`, `skip`) | Comma-separated analysis names from the merged registry. An unknown name raises before the checkpoint loads. `band_partition` is not selectable. |

Put run settings (seed, caps, thresholds, the draw count) in the override file's `eval_config`, not in `RUN_ARGS`. The run directory keeps a copy of `eval_config`, but a value injected from Python appears in no artifact.

## Output layout

Everything goes to `<run>/eval_results/`:

| Artifact | Contents |
|---|---|
| `summary.json` | The whole run: readouts, verdicts, headline, sanity, cohort, coverage, run context, causality disclosure, step records, artifact manifest. |
| `steps.json` | The per-step heartbeat. The runner writes it again as each analysis finishes, so a killed run shows how far it got. |
| `preflight.json` | Every guard's verdict, the causality disclosure, the resolved warm-up budget, the per-block group-delay summary and the computed lag-support margin. A model-free rerun reuses it. |
| `loader_probe.json` | The population record: per-shard, per-class and per-label counts that the sanity checks read. |
| `resolved_config.yaml` | The merged configuration the run used. An offline rerun reads this file. |
| `eval.log` | The run's log, including any refusal. |
| `per_sample.csv`, `per_sample_vectors.npz` | One row per segment: every scalar readout plus labels and provenance; the vector readouts in row order. |
| `per_anchor.parquet` | Per-anchor scores, KL, argmax lag, coverage and `seconds_since_contraction`, keyed on `(guid, epoch, anchor)` with the forward's own `anchor_index`. Every row also carries its recording's `subgroup` and `clinical_class`. |
| `per_anchor_vectors.npz` | The per-anchor lag maps (the pooled KL attribution and the head-averaged attention at every contributing anchor), row for row with `per_anchor.parquet`, in `float16`. Only `lag_high_kl` reads it. |
| `retained_arrays.npz` | The opt-in retention for the capped sample set. `waveforms` holds the gathered target, the mean-decoded forecast pair (`mu_base`/`logvar_base` at the prior mean, `mu_full`/`logvar_full` at the posterior mean), the raw `up_raw` and `fhr_raw` traces where loaded, `weight`, and `anchor_index`. `attention` holds the per-step attention weights. |
| `collection.json` | The collection record: readouts, provenance (checkpoint hash, seed, the $K$ actually used, `max_batches`, `eval_config` digest), denominators, retention plan, the cost of the pass, `likelihood_structure`, and `target_keep_index`. |
| `band_partition.json`, `band_channel_map.csv`, `band_channel_map_kept.csv` | The input channel map on the declared axis and on the kept axis. |
| `<analysis>/…` | One subdirectory per analysis, with its CSVs and figures. |
| `attribution/…` | The Captum attribution tables, arrays and figures. [ATTRIBUTION.md](ATTRIBUTION.md#6-output-files-and-figures) lists them. |
| `recording_traces/…` | `recording_traces.csv` (the manifest), `segment_summary.csv` (one row per segment), `anchor_trace.parquet` (one row per decoded anchor), and per recording, under its class directory, the full vectors (`<guid>_<subgroup>_full.npz`) and the figure. |

### Summary blocks

Four summary blocks organise a run:

- **`headline`** holds the scalar values and verdict statuses that the acceptance gate and the arm tables read. Those tools ignore values that are not registered.
- **`sanity`** holds the consistency checks: KL identities, agreement between tables, lag identities and population counts. The check names include `kl_identity`, `per_anchor_recombines`, `argmax_lag`, `per_file_counts`, `classes_present`, `target_not_truncated` and `headline_finite`. These checks do not change the process exit code.
- **`verdicts`** holds the acceptance criteria of `metrics.VERDICT_REGISTRY`, in registry order, each with an explicit status. The ten criteria are `predictive_improvement`, `source_margin_positive`, `source_specificity`, `prior_carries_target_state`, `latent_not_collapsed`, `prior_variance_not_pinned`, `decoder_variance_not_pinned`, `calibration_near_nominal`, `coupling_exceeds_availability_clock` and `anchor_geometry_intact`. The `time_shift` verdict `coupling_is_time_specific` is not in the registry. The run reports it at `results.time_shift.verdict`, in the same `{name, status, reason, values}` shape.
- **`run_context`** records the parameter count, checkpoint epoch, decoded anchor geometry, training stride, anchor coverage and observed objective magnitude. The forecast density is beside it, in the top-level `likelihood_structure`.

### Three forecast-gap definitions

The headline keeps three forecast-gap definitions separate:

- `pred_gap_mc_nats` is the gate's Monte Carlo score. It takes the negative log of the average likelihood over $K$ draws, then subtracts full from base.
- `pred_gap_mean_nats` decodes both latent means with the decoder's predicted variance and draws nothing.
- `pred_gap_train_path_nats` reproduces the training scoring path. Under `base_decode: mean`, base uses the prior mean and full uses a sampled latent.

Four percentage columns give proportional versions of these comparisons. The saved `pred_gap_convention` explains their differences, possible sign disagreements, block size and budget-local percentages.

### Monte Carlo error of the gate column

`pred_gap_mc_nats` is a log-mean-exp over $K$ draws, which is a nonlinear and biased estimator. Each segment's `mc_pred_gap` therefore carries a delete-one jackknife error over draws (`metrics.mc_gap_error`). Let $\hat g_{(i)}$ be the gap recomputed without draw $i$. The draw is removed from both branches at once, because common random numbers pair them:

$$\mathrm{SE}_{\mathrm{MC}}=\sqrt{\tfrac{K-1}{K}\sum_{i=1}^{K}\bigl(\hat g_{(i)}-\bar g_{(\cdot)}\bigr)^2}.$$

`per_sample.csv` carries this error as `mc_pred_gap_se`. Beside it, `mc_pred_gap_half_k` is the same gap from the first $\lfloor K/2\rfloor$ draws. Both columns exist only at $K\ge 2$.

`results.mc_error` propagates them through the recording chain, $\frac{1}{R^2}\sum_r\frac{1}{n_r^2}\sum_{s\in r}\mathrm{SE}_s^2$. It reports `pred_gap_mc_se_nats` and `pred_gap_mc_half_k_nats` beside `pred_gap_mc_nats`, and the median and maximum of the per-segment error. Both headline scalars reach every arm table through `binding.COLLECTION_HEADLINE_BLOCKS`.

This error is the error that the $K$ draws leave in the number. It is not the between-recording uncertainty that `coupling` bootstraps. Compare the two: when the Monte Carlo error is a large share of the bootstrap interval, $K$ is too small. The half-$K$ distance shows the direction of the estimator's bias.

## The four layers

The evaluator has four layers. `tests/test_eval_self_contained.py` checks the imports by walking the Python syntax tree, including aliased, lazy and relative imports. This keeps offline analysis and verification independent of model execution.

| Layer | Modules | May import |
|---|---|---|
| 0, pure | `config_schema`, `verify`, `events`, `frames`, `lag_axis`, `lag_shape`, `lag_hist`, `cohort`, `dataset_rows`, `figures_seam`, `report_seam`, `launch`, the `_reuse` seam | No Lightning, no `model.*`, no `task`/`trainer`/`plotting`. `verify` also imports no `torch` and no `binding`. |
| 0, shared helpers of the model-reading analyses | `traces`, `traces_html`, `attributions`, `attribution_pass` | The layer-0 rule. They receive a task, model or loader as an argument and may bind layer-1 helpers (`metrics`, and `preflight` for `traces`). They never import a model class, `task` or `trainer`. |
| 1, model-touching | `binding`, `metrics`, `collect`, `preflight`, `probe`, `oracle` | `task`/`trainer` only through the named `EXEMPTIONS` table. |
| 2, I/O and presentation | `analyses/*` | Layers 0–1. Never another analysis, never Lightning, never the model. |
| 3, orchestration | `run` | Every layer below it. |

Two more import rules apply:

- Production modules cannot import `teb_vae.lag_attn_rws.eval`. Only named comparison tests may do so.
- `binding` and `probe` are in layer 1 because they identify or rebuild a model. The CFS forward contract is checked against the rebuilt model, including its required anchor phase when the stride is above $1$.

The shared evaluation package (`teb_vae/lag_attn/eval`) is reachable only through its model-free modules, which an allow-list names (`ALLOWED_SIBLING_EVAL_MODULES`). `model/*` is forbidden everywhere. The test asserts that the `EXEMPTIONS` table is minimal: a module that lists a name it no longer imports fails the test.

### The model binding

`binding` defines the model-specific facts:

- the classes loaded from a checkpoint;
- the constructor keys to reconcile (`GEOMETRY_KEYS`);
- the encoder causality disclosure and the override file;
- the model-specific analyses (`extra_analyses`) and headline scalars;
- the analyses to exclude;
- optionally, the model's own collection pass (`collect`).

`run.main` uses `CFS_BINDING` by default. The transformer sibling, `teb_vae/lag_attn_transformer_cfs/eval`, uses this runner, collection pass and analyses through its own binding for `SeqVaeLagAttnTrfCfs`. It reuses this module's `EXTRA_ANALYSES` and `HEADLINE_SCALARS`, so both cells run the same registry.

`EXTRA_ANALYSES` holds the analyses registered on the binding, in run order: `samples`, `recording_traces`, `warmup`, `source_null`, `time_shift`, `occlusion`, `lag_clocks`, `lag_kld_scaled`, `lag_high_kl`, `spectral_skill`, `attribution`. `attribution` is last because it reads the tables of `occlusion` and `spectral_skill`. The runner merges the extras after the shared analyses and before the trailing `cross_subgroup`. `HEADLINE_SCALARS` appends their headline entries to the shared registry. A headline path may also start from a result block that the collection pass writes; `COLLECTION_HEADLINE_BLOCKS` names those (`mc_error`).

`GEOMETRY_KEYS` holds `sequence_length`, `d_model`, `d_z`, `horizon`, `raw_per_step`, `warmup_period`, `c_y`, `c_u`, `use_up_st`, `max_lag`, `num_heads`, `d_head`, `horizon_attention_blocks`, `causal_norm`, `anchor_stride`, `lag_floor`, `prior_availability_input`, `lag_kv_source` and `persistence_residual`. The transformer cell's tuple also carries `forecast_ar_residual` and its encoder keys.

`preflight.reconcile` compares `model_config.VAE_model[key]` with `model_kwargs[key]` and skips any key absent from either. A key must therefore be a config key and a constructor parameter to be checked. Some keys fail that rule:

- `causal_warmup_budget_steps` is a config key but not a constructor parameter.
- `target_keep_index`, `target_warmup_steps`, `source_keep_index` and `source_warmup_steps` are constructor parameters but not config keys.
- `horizon_weight_halflife_steps` is not in the set because no evaluated readout applies it.

These keys get their own guard. `check_warmup_budget_matches_checkpoint` re-resolves the budget against the configured shards and compares the result with the checkpoint's stamped tuples. `reconcile_with_checkpoint` handles `target_scored_horizon` the same way: it re-resolves the vector from the configured rule and compares it.

## Configuration reference

Set evaluation choices in the override file's `eval_config` block, so the run saves the resolved values. The schema rejects unknown keys, a boolean where an integer is required, and caps of zero. The error messages name the valid keys. The "Committed" column gives the value in `eval/configs/eval_overrides.yaml`; the schema default is the same unless stated.

| Key | Committed | Meaning |
|---|---|---|
| `seed` | `42` | Seeds `random`/`numpy`/`torch` and derives the loader-shuffle, derangement, stratified-cap and Monte Carlo generators by fixed offsets. Two runs of one checkpoint at one seed give byte-identical `results`. |
| `num_mc_samples` | `8` | Monte Carlo draws $K$ per anchor for the marginalised score, under common random numbers across branches. $K = 1$ is one draw of the same estimator, not the training-path score: under `base_decode: mean` the training path decodes base at the prior mean, which the estimator never does. `results.mc_error` reports the estimator's error at every $K\ge 2$. `--num-samples` overrides the key for one run. |
| `max_samples` | `null` | A seeded, stratified global sample cap. `null` evaluates the whole split. |
| `caps` | see below | Limits for retained arrays and the capped analyses. |
| `prior_shuffle_min_nats` | `1.0` | The margin that the prior-shuffle degradation must clear. The verdict reports the measured number beside it. |
| `min_active_dims` | `2` | The active latent dimensions below which the latent counts as collapsed. |
| `event_lag_window_s` | `null` | Seconds after a detected contraction onset within which an anchor counts as event-conditioned. See below. |
| `bootstrap_resamples` | `2000` | Resamples behind every bootstrap interval, drawn over recordings. Anchors are never resampled, because neighbouring forecast windows share $H-1$ of their $H$ steps. |
| `clock_margin_min_nats` | `null` | The margin, in nats per anchor, that the lower end of the $\Delta_{\mathrm{clock}}$ interval must clear for `coupling_exceeds_availability_clock` to PASS. At `null` the verdict is INCONCLUSIVE and the run still emits the measurement. A set value must be finite and at least $10^{-3}$. |
| `occlusion_bands` | `anchor: [0, 14]`, `near: [15, 44]`, `mid: [45, 67]`, `far: [68, 90]` | The lag bands that `occlusion` removes and that `lag_kld_scaled`, `lag_high_kl` and `attribution` read. See below. The schema default is `{}`. |
| `figure_format` | `null` | The image format of every figure, as a matplotlib filetype (`pdf`, `svg`, `png`, `eps`, …). Config load validates it against the installed matplotlib. `null` keeps the `pdf` default, which `figure_manifest.json` and FIGURE_GUIDE.md record. |
| `max_hours_before_delivery` | `null` | How far before delivery a segment may be recorded and still be evaluated, in hours. `null` evaluates everything. Minimum one $0.5$ h bin. |

**`caps`.** The cap names are `waveforms`, `attention`, `pages`, `pages_per_class`, `traces_per_class`, `oracle`, `occlusion`, `time_shift`, `attribution_segments`, `attribution_segments_per_recording`, `attribution_cohort_segments` and `attribution_strata_segments`. The committed values are `waveforms: 64`, `attention: 64`, `pages: 24`, `pages_per_class: 10`, `traces_per_class: 10`, `attribution_segments: 144`, `attribution_segments_per_recording: 4`, `attribution_cohort_segments: 150`, `attribution_strata_segments: 72`, `occlusion: 512` and `time_shift: 512`. Omitted array-retention caps retain no samples. Omitted `oracle` and `occlusion` caps allow every segment. Omitted `time_shift`, trace and attribution caps use their analysis defaults. Cap names are not validated: a cap that nothing reads is inert and the run reports it. Cap values are validated, and zero is refused. `collection.json`'s retention block records the bytes that the retention caps actually hold.

**`event_lag_window_s`.** At `null`, `config_schema.resolve_event_lag_window_s` derives the window from the checkpoint's `event_weight_reach_steps` $R$:

$$w = \Delta (R + 1) - \delta / 2,$$

with $\Delta = 4$ s the step length and $\delta = \Delta/\texttt{raw\_per\_step}$ one raw sample. For example, $R=55$ and `raw_per_step: 16` give $w=223.875$ s. With this $w$, the event and control anchors are exactly the anchors that the training contraction weighting scored at $1$ and at the floor. When the reach is `null` and `event_weight_floor` is set, $R$ is `max_lag`. A run with no reach resolves to $120$ s (`FALLBACK_EVENT_LAG_WINDOW_S`). A set value is kept, with a warning when it splits anchors differently from the reach. `events` and `lag_high_kl` read the same window.

**`occlusion_bands`.** Each entry is `{name: [lo, hi]}`, inclusive, in stored lag steps back from the anchor. Config load refuses, by name, an empty band (`lo > hi`) and a band past the model's `max_lag`. The reserved entry `partition_width: <w>` takes an integer instead of a pair. `config_schema.partition_lag_window` expands it over the lag window $[0, L-1]$ into contiguous bands named `lags_<lo>_<hi>`; the last band absorbs the remainder. A width that covers the whole window is refused. The derived bands come before any declared bands, and a declared name that collides with a derived one is refused. `{}` makes `occlusion` and `lag_kld_scaled` record skips.

**`figure_format`.** Every figure is saved at `EVAL_SAVE_DPI` ($200$ dpi, `teb_vae/lag_attn/eval/figures.py`). This sets the pixel grid of a raster format and of any rasterised artist in a vector format. Heatmaps drawn with `interpolation='none'` embed at their own cell resolution in PDF or SVG. A run that changes the format writes filenames that `figure_manifest.json` does not list.

**`max_hours_before_delivery`.** The bound applies to the population, not to an axis. The runner applies it to the delivery clock before it bins anything. Every clock therefore uses the same segments, and the second-stage clock bins that population again on its own signed axis. The bound changes cohort sizes, window counts and every trajectory, so a bounded run is not comparable with an unbounded one.

**Values that are not keys.** The significance level, the trajectory bin width, any lag-band selection beyond `occlusion_bands`, and the anchor stride are not `eval_config` keys. Validation refuses each of them like any other unknown key.

### Objective keys this pass reads

The objective is not an `eval_config` surface. A checkpointed pass rebuilds the task from the checkpoint's own `hyper_parameters`: `beta_schedule`, `kld_beta`, `beta_prior`, `lambda_full`, `lambda_base`, `likelihood` and `free_bits`. It refuses a checkpoint that carries none of them. On the offline path there is no checkpoint, so the pass reads the same keys from the dumped `model_config.VAE_model` block. Preflight has already reconciled that block against the checkpoint on every checkpointed run.

The preflight guard records the objective weights but does not compare them with the config. $\beta$ and its ramp weight the training total and enter no evaluated readout.

### Cohort order and colour

Every table and figure that resolves a quantity by cohort obeys two conventions. Neither is a setting.

**The order is clinical and runs worst first**, on both axes:

| Axis | Order, left to right |
|---|---|
| `clinical_class` | HIE, acidosis, healthy |
| `subgroup` | `hie_cs`, `hie_no_cs`, `acidosis_cs`, `acidosis_no_cs`, `healthy_bg_cs`, `healthy_bg_no_cs`, `healthy_no_bg_cs`, `healthy_no_bg_no_cs` |

`labels.ordered_groups` decides the order. `cohort.ordered_groups` binds it, so this package and the sibling use one order. Both orders come from `labels.CLASS_NAMES` and `labels.CANONICAL_SUBGROUPS` in reverse. `_reuse.py` binds those tables; they are not forked. A cohort that the order does not know sorts after every known cohort and is never dropped.

**The colour is the severity**: green for healthy, amber for acidosis, red for HIE, and each subgroup a shade of its own class. The mapping is a table, so a cohort keeps its colour in every figure. The palette is this package's own, not `utils.style.CLASS_COLORS_DEFAULT`. An evaluation figure of a cohort can therefore use a different colour from a training-callback figure of the same cohort; use the legend to match them.

**The order also orients every significance test.** `stats.pairwise_comparisons` names each pair in the order it receives the cohorts, and every caller passes them through `cohort.ordered_groups`. A comparison therefore always runs from more severe to less severe: HIE vs acidosis, HIE vs healthy, acidosis vs healthy. Cliff's delta is signed against that naming, so a positive $\delta$ means that the more severe cohort's values are higher.

## The analyses

This section has one subsection per registered analysis, named for its module. `band_partition` always runs and is not selectable; `--only` and `--skip` choose among the rest. The run order is the shared registry (`forecast` through `sufficiency`), then the binding's `EXTRA_ANALYSES`, then `cross_subgroup`. The subsections follow that order, and `--help` prints the merged list. `run.py`'s `RUN_ARGS` comment table describes each analysis in a line. `cross_subgroup` must be last, because it reads the per-recording CSVs that the other analyses write.

### band_partition

This step maps each model input channel to a band, a kind and a centre frequency in Hz. It reads the shards' own `sel_*` provenance and causal attributes. It writes one row per channel across the $c_y$-channel target stream (`fhr_st` then `fhr_ph`) and the $c_u$-channel source stream (`up_st` then `up_ph`). The rows use the block widths that the shards store, in the order the model receives them. The map describes the model's inputs.

Three columns belong to this cell:

- `causal_warmup_steps` is the leading delay that encloses 95% of the one-sided kernel's energy.
- `causal_delay_s` is the composed group delay.
- `kept` marks the channels that the warm-up budget retained.

The shards store `sel_*` provenance on the two phase blocks only. A scattering channel's centre frequency is therefore recoverable only where a selected phase pair names its filter. A channel without one is recorded as `unknown` and is never put into a neighbouring band. A shard with no `sel_*` attributes is a recorded skip.

The step also writes `band_channel_map_kept.csv`: the same map restricted to the $C_{\mathrm{keep}}$ kept channels that the decoder emits, indexed again on that axis. `spectral_skill` joins through this file, so `--only spectral_skill` works on a finished directory with no checkpoint.

The step does not read `causal_leg_alignment` or the per-block novelty record (`causal_novelty_curve`, or `causal_novelty_frac` on legacy shards). `causal_leg_alignment` is a property of the file, and resolution already refuses a run whose config disagrees with its shards. The training path reports the novelty split per epoch as `pred_gap_novel_lo` / `_mid` / `_hi`.

**The `kept` column covers the target stream only.** Every source row is written `kept = 1`. When channel alignment drops source channels, those rows still read `kept = 1`. The authoritative list of dropped source channels is `source_dropped_index` in the run's preflight record.

### forecast

This analysis compares predictive performance with three simple baselines and shows performance over the horizon. A block score sums the scored cells of the $H\times C_{\mathrm{keep}}$ block ($\sum_{\tau,c}m_{\tau,c}$ under a per-channel scored horizon). Its magnitude therefore depends on the forecast dimensions and the scored mask. Compare predictors on the same block, mask and anchors. Use the run's `block_width` and `likelihood_structure.scored_cells`.

The three baselines are built in feature space on the decimated grid, on the gathered kept channels. All three are causal. Each reads the anchor's persistence step $p_{a,c}=t_a+\min(s_c,0)$, where $s_c$ is the model's `target_forecast_shift` (zero without a forecast clock):

- **persistence** fills anchor $t_a$'s window with the coefficient at the last observed step at or before $p_{a,c}$, per channel. "Observed" means valid by `weight`, because the coefficients carry no sentinel value.
- **climatology** is exactly $0$ per channel, which is the z-scored population mean. The statistics exclude the warm-up region.
- **segment mean** is the segment's running mean per channel up to the anchor, $\bar y_{a,c}=\operatorname{mean}\{y_{u,c}: W'_c\le u\le p_{a,c},\ w_u\text{ valid}\}$, over observed, warm steps only. A channel with no warm step behind the anchor falls back to its persistence value.

All three baselines are scored at a fixed `BASELINE_LOGVAR = 0.0`, recorded beside the score. They are scored under the model's own density, with the same cell mask and the same $\phi_c$.

The MSE-space skill $1 - \mathrm{MSE}_m/\mathrm{MSE}_b$ has a natural zero. The NLL-space column beside it is a difference in nats, `advantage_nats_per_anchor`, not one minus a ratio. A log score has no natural zero, so a ratio of two log scores is not bounded above by one.

**Every error column is in the loader's $z$ units, labelled `normalised`.** There is no conversion out of these units. A scattering or phase-harmonic coefficient has no clinical unit. Inverting the per-channel statistics would put the $C_{\mathrm{keep}}$ channels on scales that span orders of magnitude.

The horizon curve covers the checkpoint's $H$ forecast steps and uses the single-draw scoring path; its label states this. Monte Carlo mixture scoring does not commute with a sum over horizon steps, so marginalised step scores would not add to the marginalised headline.

### coupling

This analysis measures how source history changes prediction, with one value per recording. It keeps three scores separate: the Monte Carlo marginalised headline, the mean-decoded gap and the training-path score. The outputs include the fraction of recordings with positive gaps, a paired Wilcoxon test over recordings, bootstrap intervals and quantiles.

The marginalised score averages likelihood over latent draws, so it includes latent uncertainty and depends on $K$. The mean-decoded score decodes each branch at its latent mean. The two can disagree in sign. Under `base_decode: mean`, the mean-decoded base score equals the training-path `nll_base_block`.

The distribution figure compares all three estimators per recording and reports how often their signs agree. The gate uses `pred_gap_mc_nats`.

The positive fraction reports its denominator, because `np.nan > 0` is `False` and unscored segments would otherwise count against the source. The KL is reported beside the gap as a description. It is inflated when the prior variance is on its clamp, and it does not measure forecast improvement.

The percentage scores describe proportional change:

- `pred_gap_rmse_pct` and `pred_gap_mse_pct` measure the reduction in point-forecast error; $100\%$ means zero remaining error.
- `pred_gap_mc_likelihood_pct` is $100(\exp(\Delta/n)-1)$, with $n$ the scored cells per anchor. It is a geometric per-coefficient likelihood-density improvement. $n$ comes from `likelihood_structure.scored_cells`, with a fallback to the recorded `block_width` on an older collection (`coupling.coefficients_per_anchor`).
- `pred_gap_mean_likelihood_pct` is the same expression on the mean-decoded gap.

The values are computed per recording, then averaged and bootstrapped over recordings. `frames.skill_against` requires a strictly positive error denominator and returns `NaN` otherwise.

The likelihood percentage needs a `gaussian_nll` likelihood and the block size from the run's own record. It is also budget-local: its denominator counts $C_{\mathrm{keep}}$ channels (and $H_c$ steps), which the warm-up budget and the scored-horizon rule decide. Two arms at two budgets therefore divide by different numbers. The emitted record states this.

### perm_control

This control tests whether prediction benefits from the matching recording's source. The verdict compares predictive scores: $D_{\mathrm{full}}<D_{\mathrm{base}}<D_{\mathrm{shuffled}}$. A positive source margin, $D_{\mathrm{shuffled}}-D_{\mathrm{full}}$, means that a mismatched source worsened prediction.

Assess specificity in prediction space, not in KL space. A mismatched source can move the posterior farther from the prior, and so give a larger KL, while it makes prediction worse.

The permutation is GUID-aware: it pairs different recordings. A batch without a valid cross-recording pairing is scored for every other readout, and only its control columns are `NaN`. `results.controls` counts such batches and samples. The controls therefore average over fewer samples than the rest of the run, and the excluded samples are not random, because long recordings can fill a batch alone.

Three paired controls are scored per recording. Under one sign convention, a positive value means that the control is worse than its reference branch:

- `shuffle_penalty` is $D_{\mathrm{shuffled}} - D_{\mathrm{base}}$.
- `prior_shuffle_penalty` is the same branch under a shuffled prior mean.
- `source_margin` is $D_{\mathrm{shuffled}} - D_{\mathrm{full}}$. It changes only the source, so it does not inherit the base forecast's behaviour.

This control cannot detect the availability clock. A permutation deranges rows, and the source availability pattern is a function of $t$ that every row of a batch shares. `source_null` measures the clock.

### latent

This analysis measures how strongly each latent dimension changes when the source is added, and whether the prior variance is near its bounds. It reports the per-dimension KL spectrum, the active-dimension count, the largest dimension's share, `mean_logvar_prior`, `logvar_prior_floor_frac` and `mean_logvar_post`.

KL includes $(\mu^q-\mu^p)^2/\sigma_p^2$, so a very small prior variance can inflate KL. The `prior_variance_not_pinned` verdict checks whether the prior variance is within `LOGVAR_FLOOR_MARGIN_FRAC` ($5\%$) of the clamp range from its lower bound. The sigmoid bound is approached but never reached, so exact equality would not work.

`prior_rate` measures the same condition as a distance. It is the objective's own $R_p = \sum_d \tfrac12(e^{\ell^p} - 1 - \ell^p)$, per recording and in nats per anchor, on the KL support. It is zero only when $\sigma_p = 1$, and it is continuous where the floor fraction is a step.

### lag_kl

This analysis distributes each attention head's KL over its lag weights. The attribution is $\widetilde K_{t,\ell}=\sum_m K_t^{(m)}\alpha_{t,\ell}^{(m)}$, and its sum over lags is $K_t$. Every run checks this identity at its worst anchor and records the result under `sanity`. The profile describes the model's allocation over stored lags; it does not establish a physical delay.

The analysis gives three profiles:

- **Raw** divides every lag bin by the same anchor count, so the bins decompose the headline KL.
- **Support-corrected** divides each bin by the number of anchors where that lag was causally valid.
- **Untruncated** uses only anchors where the whole lag window was available. This changes which observations contribute; support correction changes only the denominator.

Preflight computes and records the lag-support margin in `preflight.json`:

$$\texttt{lag\_support\_margin\_steps}=F-(L-1)-F_u,$$

with $F$ the anchor floor, $L-1$ the furthest searched lag and $F_u$ the `lag_floor`. At a non-negative margin every lag is causally valid at every scored anchor, and the two corrections have no effect. The analysis reads the recorded margin, counts the contributing anchors per lag, compares the three profiles and records whether the computed and observed readings agree. A negative margin is allowed; the corrections then change the profile.

Peak width, mass above threshold and secondary peaks are reported beside the argmax. A profile is `degenerate` when its peak-to-median ratio is below `1.1` or its exact-zero fraction is above `0.9` (`lag_shape.DEGENERATE_PEAK_TO_MEDIAN`, `DEGENERATE_ZERO_FRACTION`). `entmax15` produces exact zeros, so an argmax on a flat profile can be meaningless.

The lag axis is stored-coefficient time, and the group-delay caveat is on every artifact this analysis writes (see [Interpretation limits](#interpretation-limits)). The stratified table cuts the profile by class, subgroup and time-before-delivery window, one axis at a time, without a test. `lag_clocks` tests the profile against both clinical clocks.

### attention

This analysis shows how each attention head distributes weight over source lags. Each head controls its own latent group, so the analysis keeps the heads separate. Several heads that focus on different lags can give a broad average profile.

The entropy ceiling is $\operatorname{mean}_t \log \min(t+1, L)$ over the scored anchors. At a non-negative lag-support margin it equals $\log L$. The analysis compares three readings of this property and records their agreement: preflight's margin, the geometry record's truncated-anchor count, and the accumulated ceiling against $\log L$. Both entropies are emitted with distinct names. The ceiling is a per-sample column over the sample's own scored anchors.

The entropy is taken per anchor and then averaged; it is never the entropy of the averaged profile. `kld_per_t_per_head` sums over heads to `kld_per_t` exactly, and the sanity block checks this identity.

### calibration

Calibration checks whether the decoder's predicted uncertainty matches the observed errors. Four diagnostics are calculated over target coefficients, with counts named `n_coefficients` and gains named `gain_per_coefficient`:

- **PIT** is the probability integral transform: the observation's cumulative probability under its forecast distribution. A calibrated continuous forecast gives a uniform PIT distribution.
- **Central coverage** is the fraction of observations inside the predicted intervals. The Gaussian reference levels are $\operatorname{erf}(k/\sqrt{2})=0.6827,\ 0.9545,\ 0.9973$ for $k=1,2,3$.
- **CRPS** is the continuous ranked probability score, in normalised $z$ units.
- **NLL gain** is the improvement over a constant-variance maximum-likelihood reference fitted to the same residuals.

Read `logvar_full_floor_frac` and `logvar_full_ceil_frac` with `mean_logvar_full`, because an average can hide values at both limits. The analysis recommends a per-coefficient `model_config.VAE_model.logvar_clamp` revision when a bound constrains predictions, and reports no change otherwise. An `mse` checkpoint records a skip, because its log-variance head was not fitted.

### residual

This analysis compares the forecasts and the latent states that produced them. `mu_base` and `mu_full` come from the same decoder applied to two latent states. The model has no separate residual tensor or `delta_mu_src` head.

The analysis reports the per-anchor forecast-difference RMS in $z$ units and two different latent-side quantities: `delta_mu_rms`, per element, and `mu_post_prior_gap_rms`, per step with the L2 over $d_z$ taken first.

RMS values accumulate squared differences and take one square root at the end. An average of per-sample RMS values is smaller in general, so both versions are reported. The two branches share one log-variance head, applied to their different latent inputs.

### distributions

This descriptive analysis shows the shape of metric distributions across segments, grouped by cohort. Error quantities stay in the loader's normalised units, and each panel names its metric.

The analysis has no test, no interval, no $p$-value and no headline entry. A per-segment $p$-value is anticonservative because of the anchor overlap; `cross_subgroup` tests cohort differences.

Each panel draws two levels on the same axes. The filled density has one value per segment. The strip above it (median, inter-quartile range, range) has one value per recording. The figures use densities, one bin grid per panel, a nested subgroup figure, and a faint fill under a full-opacity outline drawn in two passes.

The analysis declares no `grouped_frames`. The runner's fan-out draws violins with one value per recording, and a per-segment frame would break that.

### trajectory

This analysis follows predictive gain and KL through time, within a segment and across a recording. It reads the per-anchor table.

Nothing below the anchor floor $F$ is decoded, so the within-segment profile starts at $F$, a fraction $F/T$ into the segment. At a non-negative lag-support margin there is no truncated region inside the profile. Anchors past `anchor_ceiling` are never scored.

Across a delivery, the segments are placed on the absolute time axis (`traces.absolute_seconds`):

$$t_{\mathrm{abs}}=\mathrm{epoch}+T_{\mathrm{trim}}+4t,$$

where `epoch` is the stored segment's start in seconds relative to delivery (negative before it), $T_{\mathrm{trim}}$ = `traces.LOADER_TRIM_S` is the stretch the loader cuts from the front of every stored segment, and $t$ is the loaded step. Preflight refuses any other `trim_minutes`, so $T_{\mathrm{trim}}$ is fixed. Overlapping timesteps are averaged, and `n_contributing` records the count. A gap produces a break (`gap_before_s`), not an interpolation.

### time_to_delivery

This analysis shows whether predictive gain and KL change as delivery approaches, and whether those changes differ by clinical class. It uses half-hour windows of hours before delivery, $h=-\mathrm{epoch}/3600$ (`cohort`), so a segment is placed by its stored start. The per-anchor axes of `trajectory` and `recording_traces` add $T_{\mathrm{trim}}+4t$. Segments are first averaged within each recording and window, so a recording with many segments does not outweigh one with few.

`pred_gap` is tracked beside the KL, because `pred_gap` is in the decoder's own units and the prior-variance inflation does not affect it. Significance is tested per window, with Holm across windows as one family and pairwise tests on the survivors. The `pooled` row is flagged `confounded_by_time`, and nothing reads it. `TRAJECTORY_BIN_HOURS` ($0.5$) is a module constant, not an `eval_config` key.

`lag_clocks` uses the same grid and classes for the lag structure.

The analysis writes four figures, two per readout, because `pred_gap` and the unfloored KL share a unit but not a scale:

- `time_to_delivery_trajectory_<readout>.pdf` is the median line per class with its inter-quartile ribbon.
- `time_to_delivery_windows_<readout>.pdf` has a violin per (window, class) cell over one value per recording, the Holm-adjusted $p$ of each window beneath it, and Cliff's delta for every class pair that survived.

### second_stage

This analysis aligns recordings to the onset of the second stage of labour. It uses the same predictive-gap and KL readouts as `time_to_delivery`, with a different clinical landmark.

The second-stage axis uses signed hours from onset. The stored value is `second_stage_onset = domain_start - t_SSO`, so negative means before onset and positive means after onset. Divide it by $3600$ without negating it. Hours before delivery use a different convention, $-\mathrm{epoch}/3600$. The second-stage figures use a natural left-to-right axis and mark zero.

**Eligibility.** A recording with no recorded onset cannot be placed on this axis; it is excluded and counted. Two further conditions are counted and reported in `second_stage_eligibility.csv`, but they exclude nothing:

- an implied onset at delivery, which a pipeline writes when it substitutes zero for a missing time;
- an implied onset that moves across a recording's own segments by more than 1 s (the float32 tolerance), which indicates a broken write.

**The Holm family belongs to this clock.** The correction runs across this clock's windows and is not joint with `time_to_delivery`. The two clocks are different alignments of an overlapping population.

**The analysis is `capped`.** It scores a subset of the evaluated cohort, so it declares `plan.capped = True` with its reason. The coverage block therefore leaves it out of the population comparison. The grid is `TRAJECTORY_BIN_HOURS`. The analysis records a named skip for an empty table, a table without a `second_stage_onset` column, a cohort with no onset, a cohort whose readouts are all non-finite, and a single-class split.

### events

This analysis compares predictive gain and KL shortly after detected contractions with count-matched control anchors from the same recordings. An event anchor is within `event_lag_window_s` of a contraction. The collection pass detects contractions from the raw uterine-pressure context and stores `seconds_since_contraction` in the per-anchor table, so the analysis uses the whole split. It records a skip unless there are at least `MIN_EVENT_ANCHORS` ($200$) event anchors across at least `MIN_EVENT_RECORDINGS` ($4$) recordings.

The detector masks gaps by `weight`, never by value, in two steps:

1. It interpolates across invalid samples before smoothing, so a gap gives the peak finder no edge.
2. It drops any event whose span touches a gap.

The contraction onset is a level crossing of the peak's own prominence, not a gradient walk-back. An event whose onset walk rises through less than `ONSET_MIN_RISE_FRAC` ($0.5$) of its prominence before the walk bound is dropped (`events.py`).

This cell keeps one of the raw pipeline's three readouts. Deceleration forecast skill and the contraction-triggered response score a clinical heart-rate trace in beats per minute, and this model forecasts coefficients with no clinical unit. `REMOVED_READOUTS` puts both names and their reasons in `summary.json`.

### sufficiency

This analysis estimates how much the latent representation limits prediction. It reports $\Delta_{\mathrm{suff}}=D_{\mathrm{base}}-D_{\mathrm{oracle}}$. The oracle is an evaluation-only decoder with the same capacity as the model's decoder, but it reads `target_state` directly instead of the latent $z$. It is fitted on half of the evaluation recordings and scored on the other half. It uses the same anchored feature builders, $H\cdot C_{\mathrm{keep}}$ outputs per anchor, and the model's own forecast density.

Two gaps are reported:

- `delta_suff_nats` uses the Monte Carlo marginalised base score.
- `delta_suff_mean_nats` uses the mean-decoded base, $D_{\mathrm{base}}^{\mathrm{mean}}-D_{\mathrm{oracle}}$. The oracle is deterministic, so this is the like-for-like comparison.

**The gap is an estimate, not a bound.** The emitted JSON states both bias directions. Conditioning on `target_state` omits the encoder's information loss, which biases the gap down. Fitting the probe on the evaluation population, while $D_{\mathrm{base}}$ comes from a model trained on a disjoint cohort, biases it up.

The probe's convergence flag is a precondition. Convergence is decided from the held-out curve, and a curve that never improved is not converged. The split is at GUID level, and disjointness is asserted at runtime. The fit budget is in passes over the fit half.

### samples

This analysis selects segments for visual inspection. The full page has fifteen rows and uses the task's `forecast_rows`, `forecast_extra_rows` and `input_stream_panels` helpers, so the layout matches the training diagnostics.

The analysis writes three selections:

- `stratified/` holds a seeded, shard-stratified draw over the whole split. The quota follows shard size, so a cap at or above the shard count reaches every shard.
- `by_class/` holds a class-balanced draw: the same number of segments from every clinical class.
- One directory per headline metric and tail holds the segments at the extremes of that metric. These pages are selected on the quantity they display, so they illustrate and do not measure frequency.

Every selected segment is drawn twice from one forward pass. The full page has fifteen rows. The `_compact` page keeps five rows: the raw context, the target block as the encoder receives it, the latent state, $K_t$, and the lag attention on a logarithmic colour scale.

`sample_pages.csv` has one row per file, with a `variant` column for the page type.

`eval_config.caps.pages` sets the stratified count, and `eval_config.caps.pages_per_class` sets the balanced draw's per-class count. The extremes take up to ten per tail, fewer when a metric has too few scored segments. The two tails of one metric are disjoint. Pages are named `sample<index>_<guid>_<subgroup>_epoch<epoch>[_compact]`, with `na` for a missing subgroup or epoch. The `<index>` is the position in the evaluation dataset, not in `per_sample.csv`, because the collection pass runs under a seeded shuffle. A `guid`/`epoch` round trip reconciles the two before rendering.

### recording_traces

This analysis follows selected recordings through all their stored segments. A seeded, class-balanced draw selects up to `eval_config.caps.traces_per_class` recordings per clinical class, each with at least two segments. Every segment is read again in `epoch` order, and the model's outputs are saved at every decoded anchor. The traces are examples for inspection, not a sample for population tests.

The outputs have three forms:

- **Anchor table.** `anchor_trace.parquet` stores the recomputed `kld_per_t`, coverage, argmax lag, attention entropy, latent scalars (`mu_prior_norm`, `delta_mu_norm`, mean log-variances, `prior_rate`, `n_active_dims`, `kld_top_dim_share`) and `kl_lag_*`/`attn_lag_*` shape statistics. Forecast-gap estimators, block scores and `seconds_since_contraction` are joined from `per_anchor.parquet` on `(guid, epoch, anchor)`.
- **Full vectors.** `<class>/<guid>_<subgroup>_full.npz` stores the prior and posterior means and log-variances, per-coordinate and per-head divergence, lag KL attribution and head-averaged attention. It also stores the axes that locate each anchor and segment, and the segment means of the vectors.
- **Segment table and manifest.** `segment_summary.csv` averages scalars over scored anchors, recomputes shape statistics on the segment's mean profile, and records latent dispersion, changes from the previous segment, epoch gaps and breaks. `recording_traces.csv` lists each selected recording, its class, subgroup, dataset and collection segment counts, and its output files.

Shape statistics are recomputed on the mean profile, as in `lag_clocks`.

**Only scored anchors enter a mean, and a segment with no scored anchor is `NaN`.** The full table keeps every decoded anchor with `contributing` beside it. The stored latent is the mean, never a sample, because $z$ carries the reparameterisation noise.

**The trace covers the recording as the dataset holds it.** A segment that a stratified `max_samples` cap left uncollected is traced and carries `NaN` in every joined column. `n_segments_collected` in the manifest counts the collected segments. When `max_hours_before_delivery` is set, only the segments within it are traced, counted for eligibility or drawn from; the plan records the bound. The absolute axis is $t_{\mathrm{abs}}=\mathrm{epoch}+T_{\mathrm{trim}}+4t$ from `traces.absolute_seconds`, the same function that `trajectory` uses. The raw FHR and UA rows use the same trim.

**The analysis checks the new forward against the collected one.** The divergence is recomputed and also joined. `kl_agreement_max_abs` is the worst per-anchor disagreement over every joined anchor; above `kl_agreement_tolerance` the analysis logs a warning. Every batch is also checked against the rows it was built from, because the collection pass runs under a seeded shuffle.

The analysis needs `context.task` and `context.loader` and records a skip without them. A recording that fails is recorded by GUID with its error, and the others still trace. The figures are `recording_traces_summary.pdf` and one `<class>/<guid>_<subgroup>_trace.pdf` per recording. Each recording also gets `<class>/<guid>_<subgroup>_trace.html`, an interactive plotly page with the raw FHR and UA above it (`dashboard_file` in the manifest).

### warmup

This analysis measures the effect of channel warm-up on the evaluated population and forecasts. Channels become valid at different times, so the warm-up budget decides which channels and anchors are usable. Results are aggregated per recording with intervals.

**The gap by warm-up tertile.** `pred_gap_warm_lo`, `_mid` and `_hi` split the $C_{\mathrm{keep}}$ kept channels into three tertiles by their rebased warm-up $W'$ and restrict the gap to each. The three recompose to `pred_gap` over the same denominator, and `frames.recomposition_check` asserts this. The tolerance scales with the block score, not with the gap, because the gap is a difference of two much larger block scores.

**The source-lag warmth fractions.** `source_lag_warmth_frac_st` and `_ph` are the attention mass on lags at which each stored source block is warm. The source blocks have long warm-ups, so much of the searched lag window is still in warm-up. The emitted record states this beside the number.

Two geometry checks can fail:

- `target_warm_frac` must be exactly $1.0$.
- `anchors_per_sample` must equal the checkpoint's `anchor_ceiling - warmup_period` under dense evaluation. The expected count comes from the run's horizon and forecast clock.

A mismatch means that the scored population does not match the declared geometry, or that the checkpoint predates the budget-and-floor checks.

The analysis also draws the warm-up staircase and the budget trade-off curve from the model package's own `warmup_budget.py` and `causal_warmup.py`.

### source_null

This analysis compares KL under the matched source with KL under a zeroed source. It separates the response to varying source values from the response that remains with only the availability pattern and the model's zero-input behaviour.

The source availability pattern $m^u_{t,c}$ is a function of $t$, identical in every row of a batch. It enters $q(z \mid Y, U)$ but not $p(z \mid Y)$. The availability clock alone can therefore push the posterior off the prior.

The analysis reports, per recording and bootstrapped over recordings,

$$\Delta_{\mathrm{clock}} = \texttt{source\_conditioned\_kl\_raw} - \texttt{kld\_source\_null},$$

the part of the coupling readout attributable to source variation. The null arm runs the source gate, adapter and encoder again from a zeroed source stream. It uses a zeroed stream, not a permutation, because the adapter and the encoder are nonlinear.

**Threshold.** `coupling_exceeds_availability_clock` compares the lower end of the $\Delta_{\mathrm{clock}}$ interval with `clock_margin_min_nats`. The committed override and the schema both leave it `null`, so the verdict is INCONCLUSIVE and the measurement is still emitted. `coupling_minus_clock_nats` is a headline scalar regardless of the threshold, so every arm table carries it.

**The null arm encodes through whichever module `lag_kv_source` selected.** It therefore probes the tensor that the lag attention reads.

The emitted record states two limits:

- Zeroing a source is not the same as having only the availability pattern. The encoder's response to a flat trajectory is part of $\mathrm{kld\_source\_null}$, so $\Delta_{\mathrm{clock}}$ is slightly weaker than "the clock alone".
- `kld_source_null` need not go to zero under `prior_availability_input`. The posterior is a bounded residual on the prior, so the mean half of $\mathrm{KL}(q^\varnothing \Vert p)$ depends on the delta head alone. Read the difference and its interval, not the ratio.

**The difference by lag.** The null arm produces its own attention over the lags: the query is the prior mean, and the keys are the null encode's. The head-structured attribution for the null arm therefore needs no extra forward. The matched attribution minus the null attribution, bin by bin, is the clock-excess profile, and

$$\sum_\ell \Delta_\ell \;=\; \texttt{coupling\_minus\_clock},$$

the scalar that `clock_margin_min_nats` gates. The sanity check `null_lag_map_sums_to_kl` measures this identity on every run, separately from the matched one. This is the only lag profile in a run with the availability staircase removed.

`source_null_lag_profile.csv` has one row per lag: both arms, their signed difference, its rectified part, the band of the lag, and whether the delta mask kept it. The `lag` block adds the run-level selection: the clock-excess argmax and its peak share, the degeneracy verdict, the rectified fraction, the per-band shares and the mask. Four of these reach the headline.

The profile is signed, so only the signed sum is the gated scalar. The rectified total is an upper bound on it, larger by the negative mass; `rectified_frac` is that difference. The delta mask is withheld when the clock-excess profile is degenerate, because `entmax15` can give a flat profile a confident argmax.

### time_shift

This control asks whether the coupling reads the source at the right moment, or only a recording-level property of the patient's source. `perm_control` gives a segment another recording's source, which removes the recording and the moment together. `source_null` zeroes the source. This analysis keeps the recording and changes the moment.

**Pairing.** Every scored segment (within `max_hours_before_delivery` when set) is paired with the nearest segment of the same recording that shares no stored sample with it: $|\mathrm{epoch}-\mathrm{epoch}'|\ge T_{\mathrm{seg}}$. Here $T_{\mathrm{seg}}=4T+2T_{\mathrm{trim}}$ is the stored segment length (`time_shift.stored_segment_seconds`). Ties go to the earlier segment. A segment with no such partner is counted and excluded.

**Scoring.** The posterior is computed again on the partner's source values, through the source gate and the run's own K/V path. The target side (encoder state, prior, anchors, persistence input) stays the segment's own. Three readings follow on the same anchors and supports:

- $K^{\mathrm{aligned}}$ and $K^{\mathrm{shifted}}$ are the source-conditioned KL (nats per step) under the segment's own source and the partner's source.
- $g^{\mathrm{aligned}}$ and $g^{\mathrm{shifted}}$ are the mean-decoded forecast gaps under each, scored under the model's own density. The base branch is the same in both, so $\Delta g=g^{\mathrm{aligned}}-g^{\mathrm{shifted}}$ is the forecast cost of the misalignment in nats per anchor.
- The collected `kld_source_null` is joined, so the figure shows the clock floor beside the two arms.

Both arms are mean-decoded. $\Delta K=K^{\mathrm{aligned}}-K^{\mathrm{shifted}}$ and $\Delta g$ are reduced per segment, then per recording, then bootstrapped over recordings with `bootstrap_resamples`.

**The verdict.** `coupling_is_time_specific` is PASS when the lower end of the $\Delta K$ interval is above zero, and FAIL when it is not. It is INCONCLUSIVE when too few recordings have a partner for an interval. $\Delta g$ is reported beside it and does not decide it. The verdict is at `results.time_shift.verdict`. It is not in `metrics.VERDICT_REGISTRY`, so `verify` does not read it.

**Outputs**, under `time_shift/`: `time_shift_per_segment.csv` (identity, partner epoch, separation, both arms, both differences), `time_shift_per_recording.csv`, `time_shift_summary.csv` (each arm and difference with its interval and unit) and `time_shift_control.pdf`. The record adds a `pairing` census (segments scored, locatable, with a partner, recordings with a partner, the minimum separation) and a `cost` block. The runner's cohort fan-out receives the per-recording differences.

`caps.time_shift` limits how many paired segments are scored again. They are drawn with a seeded generator at its own seed offset. When the cap is absent, the analysis uses its own default. The analysis needs `context.task` and `context.loader`, so an offline rerun records a skip. A population in which no recording has two non-overlapping segments also records a skip. The analysis is `capped`.

### occlusion

This analysis tests source use by removing values in selected lag bands and measuring the change in prediction. `lag_kl` and `attention` describe attention-weighted quantities; occlusion measures the fitted model's response to an input intervention.

For each band in `occlusion_bands`, the stored source coefficients in $[t_a - \ell_{\mathrm{hi}},\ t_a - \ell_{\mathrm{lo}}]$ are set to zero, which is the channel mean. The stream is encoded again through the run's own K/V path, and the block NLL is scored again. The reported quantity is the per-horizon-step change against a reference forward with nothing removed.

Five design rules apply:

- **The availability pattern is unchanged.** The intervention changes the source values, not its arrival clock. The invariance is measured on every occluded encode and reported.
- **The band is occluded after the channel gate.** The gate shifts each channel onto the common clock, so a band of gated steps is one lag range for every kept channel. Before the gate, the band would land at $\ell + d_c$ for channel $c$.
- **One scored anchor per segment.** A seeded generator draws it from the anchors the forward marked valid. The same anchor serves every band and the reference. The source pathway has memory, so a second anchor in the same forward would mix the effects of two bands.
- **Common random numbers.** The reference and every band are scored under the same reseeded latent noise. Repeated collection is bit-identical.
- **Reseeded per batch.** The anchor draw and the latent noise are seeded from the analysis seed plus a per-batch stride times the batch position. Each batch draws its own anchors and noise, and the whole draw is reproducible from the run's seed and the loader order.

**The live fraction has its own column.** A band that reaches into the warm-up region, where the availability mechanism already zeroes the source, has less source to remove. Four headline scalars therefore reach every arm table: the winning band's name, its delta, its peak horizon step and its live fraction.

**The analysis has no verdict.**

The outputs are `occlusion_per_recording.csv`, `occlusion_per_horizon.csv`, `occlusion_summary.csv` and the per-horizon figure. `occlusion_bands` names the bands in lag units; see [Configuration reference](#configuration-reference) for validation. `caps.occlusion` limits how many segments are encoded again, because the analysis scores one anchor per segment. Without the key, every segment is used.

**Set `caps.occlusion` from a measurement.** The cap trades wall-clock time against the per-band standard error. The `cost` block holds the rates, and every row of `bands` holds `n_segments`, `n_recordings`, `delta_total_se` and `delta_total_ci_lo`/`delta_total_ci_hi`.

1. Run once on the real checkpoint with `caps: {occlusion: 64}`.
2. Read `results.occlusion.cost.hours_per_1000_samples`.
3. Read `results.occlusion.bands[*].delta_total_se`.
4. Compute `cap_time = target_hours * 1000 / hours_per_1000_samples`.
5. Compute `cap_precision = 64 * (se_observed / se_wanted)^2`, because the standard error falls as $1/\sqrt{n}$.
6. Take the smaller of the two caps.
7. Set the result in the committed override delta, not in `RUN_ARGS`.

Use `cost.seconds_per_arm_per_segment` to extrapolate when the band count changes. The work is one encode and one single-anchor decode per arm, and an arm is the reference plus one per band. The interval is taken over recordings, but the cap limits segments.

**The deltas are also placed on the two clinical clocks.** `occlusion_clocks.csv` holds, per band and per window, the mean per-recording delta with its quartiles. The clinical coordinates come from a join onto the collected per-sample table on `(guid, epoch)`. Both sides read `epoch` through `metrics.batch_field` and the same `float64` cast, so the equality is exact. `clocks.n_unjoined` counts rows that fail to join. This page is descriptive only: no Kruskal–Wallis, no Holm correction. One anchor per segment gives few segments per half-hour window, and a larger `caps.occlusion` fills those cells.

The analysis needs `context.task` and `context.loader` and records a skip without them, because no table can serve an intervention on the model's input.

### lag_clocks

This analysis follows the lag profile through labour on both clinical clocks. It asks whether the location and shape of the profile change over time or differ by class. Read it with `time_to_delivery` and `second_stage`, which measure the amount of predictive gain and KL in the same windows.

The analysis computes fourteen statistics of each segment's own profile on the compensated axis, with $p_\ell = w_\ell / \sum_k w_k$, in four families:

| Family | Statistics | Column |
| --- | --- | --- |
| moments | centre of mass $\bar\tau = \sum_\ell p_\ell \tau_\ell$, spread $\sigma_\tau = \sqrt{\sum_\ell p_\ell (\tau_\ell - \bar\tau)^2}$, skewness | `lag_centroid_*_s`, `lag_spread_*_s`, `lag_skewness_*` |
| quantiles | median lag, inter-quartile range | `lag_median_*_s`, `lag_iqr_*_s` |
| concentration | entropy $H = -\sum_\ell p_\ell \log p_\ell$, effective support $\Delta e^{H}$, near and far mass share | `lag_entropy_*_nats`, `lag_effective_support_*_s`, `lag_near_mass_*`, `lag_far_mass_*` |
| peak | peak lag, its width at half height, its mass, the degeneracy flag and the zero fraction | `lag_peak_*_s`, `lag_peak_width_*_s`, `lag_peak_mass_*`, `lag_peak_degenerate_*`, `lag_zero_fraction_*` |

Each family answers a different question. For example, a bimodal profile has an unremarkable centroid and spread, and only the concentration family shows that it is not one lump. Every statistic is computed twice, over the untruncated KL attribution and over the support-corrected attention, which gives 28 columns. The attribution is $K_t$ times the attention, so it inherits the prior-variance inflation that the attention does not have.

The arithmetic is in `eval/lag_shape.py`, in one vectorised pass per profile.

**The peak is reported with its degeneracy flag.** `lag_peak_*_s` is in the same row as `lag_peak_degenerate_*`. A segment is degenerate when its peak-to-median ratio is below `1.1`, or when more than `90%` of its finite bins are exactly zero. The per-recording mean of that flag is the share of a window's segments whose peak names a bin rather than a lag. `lag_kl/lag_kl_stratified_peaks.csv` holds the pooled positional reading; this is the per-segment one.

Both clocks use the `TRAJECTORY_BIN_HOURS` grid. The unit is one value per recording inside a window. The second clock scores only recordings that carry an onset, by the same eligibility rule as `second_stage`. The analysis is `capped` for that reason.

**There are four Holm families, none joint**: two clocks times two tested readouts. Only the two centroids are tested. The other twelve statistics are drawn and tabled without a $p$-value. To test another statistic, set its `tested` flag in `STATISTICS`.

The analysis writes six figures, three per clock:

- `lag_<clock>.pdf` shows the share of the attribution by lag and window, with one panel per class on a shared colour scale, and the two tested centroid trajectories.
- `lag_<clock>_windows.pdf` shows the violins, the Holm-adjusted $p$ per window, and Cliff's delta for every class pair that survived.
- `lag_<clock>_features.pdf` shows the untested statistics, one panel each, solid for the attribution and dashed for the attention.

### lag_kld_scaled

This analysis reads the lag structure on selected lags and keeps its magnitude. `lag_clocks` resolves the profile over all $L$ lags with statistics of $p_\ell$ alone, which divide out the scale and average the heads. This analysis runs beside `lag_clocks` on the same $0.5$ h grid and does not change its columns.

It uses four profile sources:

- **The geometry-fixed bands** from `occlusion_bands`. Nothing about them is estimated from the KL, so a band statistic does not select on its own outcome. They are the same partition that `occlusion` removes, so the observational and interventional pages share one band set.
- **The soft weight** $\omega_\ell = \Delta^+_\ell / \max_k \Delta^+_k$, from the pooled clock-excess profile of `source_null`. It is computed once at run level and applied to every segment, window and class. It is withheld when that profile is degenerate.
- **The full support**, with `total_nats` and `peak_nats` only. The twelve scale-free statistics on the full support are `lag_clocks` columns.
- **The heads**: each head's own $K^{(m)}\alpha^{(m)}_\ell$, which sums over $m$ to the pooled attribution exactly.

**`near_mass` and `far_mass` are absent from every banded source.** Both are fractions of the axis's own span from its start (`lag_shape.NEAR_SPAN_FRACTION` $=1/6$ and `FAR_SPAN_FRACTION` $=2/3$ of $\tau_{L-1}-\tau_0$). On a band they would mean the first and last stretch of the band, which is a different quantity.

**Nothing here is tested.** The analysis adds no Holm family and writes no significance or pairwise table. The record states this in `no_inference_note`. To test a feature, set its flag.

The emission is long-form: `source` and `statistic` are row keys. `num_heads` can therefore vary by run, and a band can be added without a wider table.

The outputs are `lag_kld_scaled_per_recording.csv`, `lag_kld_scaled_trajectory.csv`, `lag_kld_scaled_selection.csv` and one figure per clock. The selection table records which lags were kept and with what weight.

`occlusion_bands` is read by both `occlusion` and this analysis. With `{}`, this analysis records a named skip.

### lag_high_kl

This analysis examines lag profiles at anchors selected for large KL. A large KL shows a strong latent change; the usefulness readouts below test whether those changes improve prediction.

One set of KL thresholds is computed from all scored anchors across classes within the run's time scope. The same thresholds apply to every segment, window and class. Four anchor bands are defined in `ANCHOR_BANDS`:

| Band | Selection |
| --- | --- |
| `high` | The upper $30\%$ of pooled KL (quantiles $0.7$ to $1.0$). |
| `rest` | Its complement (quantiles $0.0$ to $0.7$). |
| `top` | The upper $10\%$ (quantiles $0.9$ to $1.0$). |
| `gain` | The upper $30\%$ of the pooled forecast gain. |

For each segment and band, the analysis reports the selected-anchor fraction and the KL-attribution and head-averaged-attention profiles of those anchors. The `lag_shape.py` functions compute centroid, spread, median, IQR, entropy, effective support, near/far mass, guarded peak statistics and nats-scale totals. Results are shown by class on both clinical clocks in half-hour windows.

The analysis needs `per_anchor_vectors.npz`. Segment-averaged profiles cannot recover the selected-anchor results. An older run directory without the sidecar records a skip; run the collection again to create it.

**Two primary readouts are tested on each clock:** `high_lag_centroid_kl_s` and `high_anchor_frac`. Each uses a Kruskal–Wallis class comparison within windows, Holm correction across that clock's windows, and pairwise Mann–Whitney tests with Cliff's delta for surviving comparisons. This gives four Holm families.

**Further readings from the same selection:**

- **Hot lags.** The lags whose pooled attribution, over every anchor, is in the upper $30\%$ across the $L$ lags (`HOT_LAG_QUANTILE`). The set is run-level and recorded lag by lag in `lag_high_kl_selection.csv`. The per-segment share of attribution on it is placed on both clocks. The set is chosen from the same attribution that it summarises, and every artifact states this. The selection is pooled over every class, so no class chooses its own lags.
- **KL position by KL magnitude.** The per-anchor `argmax_lag` against the KL decile of the same anchor (`N_KL_QUANTILE_BINS` $=10$), pooled and per class (`lag_high_kl_argmax_by_quantile.csv`).
- **Contraction enrichment.** Whether high-KL anchors are more common within `event_lag_window_s` of a detected contraction than outside it, per recording (`lag_high_kl_contraction.csv`) and by class. A recording's difference is reported only with at least `MIN_ENRICHMENT_ANCHORS` ($5$) anchors in each arm; below that, the row has `reportable = False`.

**Histograms.** The histogram outputs compare the full selected-anchor lag distributions of the `high` and `top` bands, by class and window, on both clocks. Every recording has equal weight:

1. Average the selected-anchor lag maps within a recording.
2. Normalise that recording's profile to sum to one.
3. Average these distributions within each class and window.

`lag_high_kl_profile.csv` normalises after the class/window average instead. The histogram describes a typical recording's relative distribution; the profile table describes the location of pooled coupling mass. Both attention (`attn`) and KL-weighted (`kl`) versions are saved.

The histogram tables are:

- `lag_high_kl_histogram.csv` has one row per (clock, band, source, class, window, lag): the cell's mean density and its inter-quartile range over recordings. Each cell sums to one across the lags.
- `lag_high_kl_histogram_features.csv` holds the `lag_shape.py` statistics of each recording's own histogram. `total_nats` and `peak_nats` are omitted, because the first is $1$ on a normalised histogram.
- `lag_high_kl_histogram_distance.csv` holds two distances between cells: the Jensen–Shannon distance (base $2$, bounded by $1$) and the $1$-Wasserstein distance in seconds on the compensated axis. A signed centroid difference, oriented worst class first, is beside them. There are two comparisons: every class pair within a window, and every window against its own class pooled over the whole clock. A cell below three recordings gets its counts and a `NaN` distance.
- `lag_high_kl_histogram_significance.csv` and `lag_high_kl_histogram_pairwise.csv` hold the tested shape features. Three features of the `high` band's `kl` histogram are tested per window on both clocks: `hist_median_s`, `hist_iqr_s` and `hist_entropy_nats` (`HISTOGRAM_TESTED_STATISTICS`). This gives six Holm families, one per (clock, feature), none joint with the four above.
- `lag_high_kl_histogram_drift.csv` and `lag_high_kl_histogram_drift_summary.csv` fit the same three features within each recording along the clock. The slope is the least-squares slope of the feature against forward labour time, over every recording scored in at least `MIN_DRIFT_WINDOWS` ($3$) windows; the last-minus-first difference is beside it. A positive slope means that the feature rises as delivery approaches, on either clock. Per clock, two more Holm families across the three features test each (feature, class) slope against zero by Wilcoxon signed-rank, and each feature across classes by Kruskal–Wallis, with pairwise Cliff's delta on the survivors.
- `lag_high_kl_subgroup_histogram.csv` holds the `high` band's lag distribution over the whole evaluated population, with no clock and no window, by clinical class and by subgroup. It is descriptive, without a test, and is drawn on `lag_high_kl_subgroup_histogram.pdf`.

The cells, both distances, every feature on the `attn` source and the `top` band, and every other untested feature have no test, and the record states this. No peak-lag histogram is written: on a flat profile, NumPy's first-maximum rule puts the argmax at lag $0$.

**Usefulness through prediction.** The analysis uses the per-anchor gain $D_{\mathrm{base}}-D_{\mathrm{full}}$. It prefers `mean_pred_gap`, then `mc_pred_gap`, then `pred_gap`, and records the chosen column under `usefulness.gain_column`. Four comparisons follow:

- **Mean gain per band** (`high_pred_gap_nats`, `rest_pred_gap_nats`, …) per segment, per recording and on both clocks. The high band against the rest band is tested once, paired within recording, by a Wilcoxon signed-rank test over recordings with a bootstrap interval on the mean difference. This is a family of one.
- **The `gain` band** has its own lag profile and statistics, and its overlap with the high band against the $30\%$ that independence would give (`share_of_high_in_gain`, Jaccard).
- **Gain by KL decile** (`lag_high_kl_gain_by_kl_quantile.csv`, per recording then by class) and **by argmax lag** (`lag_high_kl_gain_by_argmax.csv`, pooled and per band).
- **A gain-weighted attention profile**, $\sum_t \max(g_t, 0)\,\alpha_{t\ell} / \sum_t \max(g_t, 0)$, beside the KL-weighted one in the selection table.

**Observational against interventional.** When `occlusion` has run in the directory, `lag_high_kl_occlusion_consistency.csv` joins, per recording and per band of `occlusion_bands`, the share of the recording's KL attribution inside the band (all anchors, and the high band's) with the forecast cost of occluding that band. It records a descriptive Spearman's $\rho$ per band. Without the occlusion file, `--only lag_high_kl` records a skip for this table.

**Eight headline scalars reach every arm table**: the pooled high threshold in nats, the high band's centroid and total nats as means over recordings, the hot-lag count, the hot-lag share, the high band's mean forecast gain, its paired difference against the rest band, and the high–gain overlap share. `lag_high_kl_recordings.csv`, one row per recording, is the file that `cross_subgroup` reads `high_anchor_frac` from.

**The analysis is `capped`**, because its second-stage half scores only recordings that carry an onset. It writes per-recording and trajectory tables for both clocks, the per-window restricted profiles, the seven histogram tables, the significance and pairwise tables, and twelve figures. The figures are a run-level selection page, a run-level usefulness page and, per clock, a profile-and-trajectory page, a tested page, a lag-distribution page, a tested page for the histogram shape features and a within-recording drift page. The axis is stored-coefficient time, and every output carries the caveat.

### spectral_skill

This analysis groups forecast performance by the frequency band of the target's analysing filter. A first-order scattering coefficient, $|x\star\psi_\lambda|$, is the magnitude of a filtered signal. Its filter already identifies a frequency range, so performance can be grouped by channel without a new spectral estimate.

The analysis reports how well the model forecasts the envelope in each clinical band, per recording and bootstrapped over recordings. It reports both likelihood space and error space. The band gaps recompose to `pred_gap` under the same guard as the warm-up tertiles.

**This is band-resolved skill, not coherence.** A stored scattering coefficient is a modulus: the filter's phase was discarded before the value was stored. Phase agreement, group delay and the three-way split of the residual spectrum therefore have no analogue here. The readout cannot tell whether a forecast is mistimed or mis-scaled. The name `spectral_skill` differs from the raw pipeline's `coherence` for this reason.

The band is the band of the analysing filter, not a bin of the forecast's own spectrum. A phase-harmonic channel has a pair of frequencies; it is banded by `band_partition`'s `freq_hz_primary` convention, which the emitted record states.

The join must use the retained target axis. `band_partition` describes declared channels, but the forecast-gap vectors contain only retained channels. The analysis reads `band_channel_map_kept.csv` from disk and joins through it. It does not access `model.target_gate`, so it works offline and keeps the import rules.

The record holds five channel counts: `declared_total` ($c_y$), `dropped_declared`, `kept_total` ($C_{\mathrm{keep}}$), `known_kept` and `unknown_kept`. Report known-band coverage relative to the retained target channels.

### attribution

This analysis uses integrated gradients to estimate how input coefficients contribute to selected model outputs at selected anchors. It attributes a scalar output over the three stored input streams $(y^{st},y^{ph},u)$ at their declared widths. [ATTRIBUTION.md](ATTRIBUTION.md) gives the full design: readouts, baselines, selection, files, methods and limits. This section summarises what the evaluator runs and saves.

- **Readouts.** A wrapper turns the dense forward into one scalar per row at one anchor per row. The main readouts are `kld`, `pred_gap`, `nll_full` and `mse_full`. The pass also attributes `nll_horizon` at the first and last horizon step, `lag_band` on every `occlusion_bands` band, and `kld_dim` at the anchor's largest per-coordinate divergence. Block scores are mean-decoded, so no reparameterisation noise enters an attributed number.
- **Baselines.** `source_null` zeroes the source values and holds the target streams fixed. `all_zero` zeroes every stream. The availability indicators are identical along the path, so they receive no input attribution.
- **Integration entry.** Integration starts at $x_0=b+10^{-3}(x-b)$ (`BASELINE_ENTRY_FRACTION`), with `IG_STEPS` $=64$. Every row records $f(b)$, $f(x_0)$ and $f(x)$. The entry jump $f(x_0)-f(b)$ is reported separately and assigned to no input step.
- **Methods.** Integrated gradients is the primary method. Layer integrated gradients on the per-head fusion modules gives a per-head split under `source_null`. Feature ablation grouped by lag band applies `occlusion`'s intervention at the attributed anchors. The block's `methods` record lists the evaluated methods that are not used, with reasons.
- **Checks.** Future-step attribution must be zero (`checks.after_anchor_max_abs`). Attribution to source steps excluded by warm-up must be zero (`checks.gated_off_max_abs`). Source attribution for the target-only base score must be zero (`checks.target_only`). The IG sum must reproduce $f(x)-f(x_0)$ within `checks.completeness_tolerance` ($10^{-2}$); `checks.n_rows_over_tolerance` counts exceptions. The conv-LSTM model needs `causal_norm: true` for the future-step check.
- **Selection.** `attribution_pass.informative_anchors` attributes each selected segment's high-$K_t$, clean anchors. Recordings are drawn class-balanced and seeded, with `caps.attribution_segments_per_recording` segments per recording, up to `caps.attribution_segments` segments. Without the cap, the analysis uses `DEFAULT_SEGMENTS` ($24$).
- **Examples.** Up to `EXAMPLES_PER_CLASS` ($3$) example anchors per class get full map pages.
- **Traces.** One recording per class, the most complete one in the window, is attributed for `kld` and `pred_gap` at the same anchors of every segment.
- **Cohort and strata passes.** A larger class-balanced cohort pass (`caps.attribution_cohort_segments`) adds Grad-CAM and class tests. A late-window strata pass (`caps.attribution_strata_segments`) compares high- and low-$K_t$ anchors and contraction anchors.

The null decomposition for `kld` under `source_null` splits the readout into the exact-null value (`kld_source_null` at that anchor), the integrated source-content part, and the entry jump. Under `all_zero`, the attribution splits between target and source streams. `attribution_null.csv` holds the per-class means.

Frequency-band sums use the declared input-channel map, `band_channel_map.csv`, because input attributions have the declared width. Where available, the lag-band table joins `occlusion_summary.csv` and the target-band table joins `spectral_skill_bands.csv`; missing files are recorded as absent dependencies.

The analysis is `capped`. It needs `context.task` and `context.loader` and records a skip without them.

An attribution is a sensitivity of a fitted computation along one path from one baseline, not a causal claim about physiology. The lag axis is stored-coefficient time. [ATTRIBUTION.md](ATTRIBUTION.md#11-interpretation-limits) lists the full limits.

### cross_subgroup

This analysis tests whether recording-level metrics differ across clinical classes or subgroups. With many groups and metrics, some means look separated by chance, so the analysis uses three layers in order:

1. A Kruskal–Wallis omnibus test per metric.
2. Holm correction across metrics as one family.
3. Pairwise Mann–Whitney tests with Cliff's delta (Romano magnitudes) on the surviving metrics only.

The omnibus gate protects the pairs only at three cohorts, so the pairs of each surviving metric are Holm-corrected again as their own family. The pairwise table holds `p_holm`, with the raw Mann–Whitney `p_value` beside it. Every pair is named more severe first. Cohorts with too few finite recordings are excluded and recorded. Every test uses one value per recording, and a test asserts that no source names a `per_sample` file.

The analysis reads finished per-recording CSVs from disk through the `METRIC_SOURCES` table:

| Source analysis | File | Metrics |
|---|---|---|
| `coupling` | `coupling_per_recording.csv` | `mc_pred_gap`, `mean_pred_gap`, `source_conditioned_kl_raw` |
| `forecast` | `forecast_scores.csv` | `sq_error_full`, `nll_full_block` |
| `latent` | `latent_per_recording.csv` | `logvar_prior_floor_frac` |
| `attention` | `attention_per_recording.csv` | `attention_entropy_nats` |
| `residual` | `residual_per_recording.csv` | `forecast_difference_sq` |
| `calibration` | `calibration_per_recording.csv` | `mean_logvar_full` |
| `warmup` | `warmup_per_recording.csv` | `pred_gap_warm_lo`, `pred_gap_warm_mid`, `pred_gap_warm_hi` |
| `source_null` | `source_null_per_recording.csv` | `coupling_minus_clock` |
| `spectral_skill` | `spectral_skill_per_recording.csv` | `pred_gap_variability` (one band, so the family does not grow by five) |
| `lag_high_kl` | `lag_high_kl_recordings.csv` | `high_anchor_frac` |

The last four rows belong to this cell. Every added metric widens the Holm family for every other metric. A missing source is recorded, not raised, so `--only cross_subgroup` works on a finished directory with no checkpoint. The analysis skips itself below two testable groups.

## Interpretation limits

These limits state what each measurement supports and which comparisons need extra checks.

**The forecast is a true forecast; the lag readout needs a caveat.** The stored coefficients come from a strictly one-sided filter bank. A coefficient at step $t$ is a function of $\{x(s) : s \le t\}$ alone, so a forecast of step $t + 1 + \tau$ from history up to $t$ is a genuine forecast. The lag map, however, is an attribution over stored-coefficient time. It is not corrected for the composed one-sided group delay, which can be of the order of the whole lag search $4(L-1)$ s. `preflight.json` records the per-block range as `causality.group_delay_seconds`. The coupling readout is named `source_conditioned_kl_raw` for this reason. Every run carries the disclosure sentence in `preflight.json` and `summary.json`, and every lag-resolved artifact and figure carries the caveat.

**A lag position is a coefficient-time attribution, not a physiological delay.** The compensated lag $\tau = 4(\ell + \delta)$ corrects only the model's own input delay $\delta$, read from `model.source_delay_steps`. The per-channel composed group delay is not corrected. That correction is per channel pair, but the lag map is per head over a pooled source state.

When `causal_align_reference` is set, both streams are on their own reference clocks, and the remaining offset between them is a single constant. The console block prints it beside the delay. The shipped default leaves `causal_align_reference` at `null`.

**The `argmax_lag` sanity check treats both window edges as censoring edges** (`report_seam.check_argmax_lag`):

- A degenerate profile is FAIL at either edge or in the middle, because its argmax names a bin, not a lag. The check decides this first, from `lag_shape`'s degeneracy flag, the peak width and the mass above half the peak.
- A profile peaked at the far edge is FAIL, because the window is too short to express the lag.
- A profile peaked at the near edge is INCONCLUSIVE, because any delay below the shortest attainable compensated lag lands there. The message states the lag identity, so a reader can check the arithmetic.
- A profile peaked strictly inside the attainable range is PASS.

The near edge is `min(attainable)` from the per-lag anchor counts, so a window whose lowest bins carry no anchor moves the floor up from zero.

**Read per-head profiles, not only the pooled argmax.** The console prints the per-head profiles under the pooled row: each head's argmax, peak width, mass above half peak, near and far mass, attention entropy and KL share.

**The run's arm is printed in three readings.** The console block and `summary.run_arm` hold the configured `causal_align_reference` and `causal_align_reference_source`, the built `lag_kv_source`, and the resolved target reference delay, source reference delay and inter-stream offset in seconds. A config that names one arm while the checkpoint carries another shows up as a disagreement between these readings.

**Read specificity in prediction space.** See `perm_control`: a mismatched source can give a larger KL than the true source.

**Read `source_null` before you treat a coupling readout as a source finding.** The availability clock is invisible to the permutation control. With `clock_margin_min_nats: null` (the committed setting), `coupling_exceeds_availability_clock` is INCONCLUSIVE; read the number, not the status.

**Only the unfloored KL is a rate.** `source_conditioned_kl_train` applies free bits per dimension per step before the sum, so it is at least the raw value. With `free_bits: 0.0` the two coincide. A test asserts that no headline path resolves to the train value.

**Read `prior_variance_not_pinned` before you quote the KL.** A prior variance on its clamp inflates every coupling number while every decoder-side diagnostic looks normal.

**A small source-lag warmth is expected.** The stored source blocks warm up late; `preflight.json`'s warm-up budget and `band_partition`'s `causal_warmup_steps` hold the per-channel values. Much of the searched lag window is therefore still in warm-up.

**The likelihood percentage is budget-local.** `pred_gap_mc_likelihood_pct` divides by the scored cells per anchor, and $C_{\mathrm{keep}}$ depends on the warm-up budget. Two arms at two budgets have non-comparable percentages and non-comparable nats, and their checkpoints cannot load into each other.

**No `nll_*_sample` column is emitted.** The objective's per-coefficient companion is a fixed $1/(H\cdot C_{\mathrm{keep}})$ rescale of a block score, whatever the mask dropped. The per-coefficient error `sq_error_*` divides by the coefficients actually scored.

**No percentage is `pred_gap` divided by a block score.** $D_{\mathrm{base}}$ is a negative log-density with no natural zero, and it can be negative. The emitted percentages are in error space and likelihood space, which have a natural zero.

**Anchors are not independent, and every statistic is per recording.** Consecutive anchors' forecast windows share $H - 1$ of their $H$ steps, and one GUID contributes many segments. The aggregation chain is: per anchor, then a support-weighted mean within a segment, then an unweighted mean over a GUID's segments, then across GUIDs. A segment with zero scored anchors is excluded and counted, never averaged in as `0.0`.

**A per-segment histogram is a description.** `distributions` has no test; `cross_subgroup` tests cohort differences on per-recording values.

**Class contrasts can be out of distribution.** When the checkpoint trains on healthy-with-background recordings only, ACIDOSIS, HIE, `healthy_no_bg_cs` and `healthy_no_bg_no_cs` are unseen. The summary computes `training_cohort_disjoint` from both resolved dataset lists. It reports `null` where a list is absent and suppresses the out-of-distribution sentence when the two overlap.

**An eval score is not comparable with a `test_*` metric logged during training**, and scores are not comparable across target axes. Against `lag_attn_fs`, the blocks differ in channel count and meaning. Against `lag_attn_transformer_cfs`, a block-level score is comparable only when the two runs score the same block over the same anchors under the same density. Check the horizon, anchor geometry and kept channels (`preflight.json`, `run_context.target_axis`) and `likelihood_structure` (scored cells, $H_c$ vector, AR term). The cross-cell table in `verify --runs` puts the two cells side by side.

**The sufficiency gap is an estimate, not a bound.** See `sufficiency`.

**The frequency readout has no timing component.** `spectral_skill` cannot detect a forecast that arrives a step late. A $\tau$-slice on this grid gives one sample per decoded anchor at $0.25$ Hz per channel (Nyquist $0.125$ Hz), over band-limited envelopes, and the phase was discarded before storage. The raw pipeline's coherence construction therefore does not apply.

**A cohort's colour here can differ from its colour on a training figure.** See [Cohort order and colour](#cohort-order-and-colour).

**The two clinical clocks are corrected separately.** `time_to_delivery` and `second_stage` each run a Holm step-down across their own windows. The two are different alignments of an overlapping population. Within one clock, the two readouts are not jointly corrected either.

**`second_stage` scores a subset of the evaluated cohort.** Its `n_samples` is not comparable with any other analysis. `second_stage_eligibility.csv` accounts for the difference, recording by recording.

## The divergence register

Every module of `teb_vae/lag_attn_rws/eval` has exactly one entry in the committed `divergences.json` beside this file. Each entry is classified:

- `equivalent`: the module must stay behaviour-equivalent, and at least one named assertion exercises it;
- `divergent`: the module deliberately differs, with the reason recorded;
- `absent`: the module is not ported.

The table below summarises the entries. `divergences.json` is the authoritative record.

| Module | State | Difference or exercising tests |
| --- | --- | --- |
| `__init__.py` | divergent | States that this package is a fork of the raw pipeline and lists the anti-drift measures. |
| `_reuse.py` | equivalent | `test_eval_reuse.py::test_the_bound_names_are_exactly_the_siblings`, `test_eval_reuse.py::test_every_bound_name_resolves_to_the_same_object_both_packages_see` |
| `analyses/__init__.py` | equivalent | `test_eval_sibling_agreement.py::test_the_analysis_protocol_is_the_same_contract_in_both_packages` |
| `binding.py` | divergent | `ModelBinding` gains a `collect` field for a model's own collection pass. The module also holds `CFS_BINDING`, `GEOMETRY_KEYS`, `EXTRA_ANALYSES`, `HEADLINE_SCALARS` and `COLLECTION_HEADLINE_BLOCKS`. It names a model class, so it is layer 1 here and layer 0 in the sibling. |
| `cohort.py` | equivalent | `test_eval_sibling_agreement.py::test_the_cohort_order_is_one_ordering_in_both_packages`, `::test_the_time_axis_bins_identically_in_both_packages`, `::test_the_second_clock_bins_identically_in_both_packages`, `::test_the_two_cohort_modules_are_one_file_with_one_import_line` |
| `collect.py` | divergent | Keys the per-anchor table on the forward's own `anchor_index`. Retains $(A, H, C_{\mathrm{keep}})$ forecast tensors (target, mean-decoded base and full means and log-variances) beside the raw traces, `weight` and `anchor_index`. Records `likelihood_structure`, the `max_batches` cap, `target_keep_index` and the pass cost. Accumulates no cross-spectral sums. `second_stage_onset` joins the identity columns after `time_from_labor_onset`. |
| `config_schema.py` | divergent | Adds `clock_margin_min_nats` and `occlusion_bands` (with `partition_width`). `event_lag_window_s` is nullable and resolves from the training contraction reach. `figure_format` and `max_hours_before_delivery` are shared with the sibling. |
| `events.py` | divergent | Keeps only the contraction detector and its gap handling. The deceleration detector, the block-mode horizon-step helpers and the greedy event matcher are removed. |
| `figures_seam.py` | divergent | Binds `lag_axis.COEFFICIENT_LAG_AXIS_LABEL` and the group-delay caveat instead of the sibling's compensated-lag label. Adds `caveat_note` and `windowed_comparison_figure` with its panels (`binned_violin_panel`, `significance_strip`). |
| `frames.py` | divergent | Adds `recomposition_check`, the guard that a channel-axis split of `pred_gap` sums back to the gap. The rest is the sibling's arithmetic. |
| `lag_axis.py` | divergent | Adds `COEFFICIENT_LAG_AXIS_LABEL`, `GROUP_DELAY_CAVEAT`, `PREFLIGHT_FILENAME` and `read_lag_support`. The compensated-seconds arithmetic and per-lag readers are the sibling's. |
| `launch.py` | equivalent | `test_eval_sibling_agreement.py::test_the_launch_merge_resolves_identically_in_both_packages` |
| `metrics.py` | divergent | Feature-domain readouts: a five-argument dense forward, the model's anchored target and mask builders, feature-space trivial predictors, the source-null branch and the warm-up tertiles. `BPM_UNIT`, `to_bpm`, `sigma_to_bpm` and `fhr_normalization` are removed. |
| `oracle.py` | divergent | Scores its probe through the anchored feature builders and emits $H\cdot C_{\mathrm{keep}}$ outputs per anchor. |
| `preflight.py` | divergent | The causal guard set (see the [guard recovery table](#guard-recovery-table)). Replaces `NOT_CAUSAL_STATEMENT`, widens the shared causality record, and carries `GUARD_RECOVERY`. `second_stage_onset` is in `REQUIRED_EVAL_LOAD_FIELDS`. |
| `probe.py` | divergent | Keeps the sibling's population pass and adds a forward-contract pass behind `--checkpoint`, which loads a checkpoint and refuses any geometry but the dense one. `second_stage_onset` is a required batch field, and the population pass reports its `{n_values, n_nan}` coverage. |
| `report_seam.py` | divergent | The headline registry drops the coherence entries and reports the calibration gain per coefficient. `HEADLINE_VERDICTS` adds `coupling_exceeds_availability_clock` and `anchor_geometry_intact`. The sanity block drops the cross-spectral checks. `PRED_GAP_CONVENTION` states the $H \cdot C_{\mathrm{keep}}$ block and the budget-local percentage. |
| `run.py` | divergent | Registers more analyses. Keeps `samples`, `recording_traces` and `attribution` on the binding. Defaults to `CFS_BINDING`, resolves the collection pass through the binding, records the dense anchor geometry and training stride in `run_context`, and registers no coherence step. |
| `spectra.py` | absent | Not ported. `spectral_skill` answers the frequency-resolved question on the channels' own frequency axis. |
| `verify.py` | divergent | Gates ten verdicts against the sibling's eight; `coupling_exceeds_availability_clock` is INCONCLUSIVE while its threshold is unset. Its arm axes are `anchor_stride`, `warmup_period`, `horizon` and `horizon_depth`. The horizon section carries a refusal instead of a reading rule. It renders a cross-cell table against the transformer cfs cell. It drops the kept-channel column, and it reports the collapse verdict as UNKNOWN unless both per-epoch series are present. |
| `analyses/attention.py` | divergent | Measures the attainable entropy ceiling against preflight's `lag_support_margin_steps` (three readings compared). Truncation accounting is keyed on `anchor_floor`. Relabels the lag axis and prints the group-delay caveat. |
| `analyses/band_partition.py` | divergent | Reads the four stored block widths from the shards. Adds `causal_warmup_steps`, `causal_delay_s` and `kept`, and a second map on the kept axis. |
| `analyses/calibration.py` | divergent | Computes PIT, coverage, CRPS and the MLE gain over coefficients (`n_coefficients`, `gain_per_coefficient`). CRPS stays in $z$ units. The `logvar_clamp` recommendation is per coefficient. |
| `analyses/coherence.py` | absent | Not ported, because a stored scattering coefficient is a modulus. `spectral_skill` replaces the part that applies. |
| `analyses/coupling.py` | divergent | `pred_gap_mc_likelihood_pct` divides by `likelihood_structure.scored_cells` and is budget-local. |
| `analyses/cross_subgroup.py` | divergent | `METRIC_SOURCES` adds the warm-up tertiles, the source-null difference, the band-resolved skill and `high_anchor_frac`. |
| `analyses/distributions.py` | divergent | Eight metrics, in normalised $z$ units with no conversion. |
| `analyses/events.py` | divergent | Keeps contraction detection, `seconds_since_contraction` and contraction-conditioned coupling. Removes deceleration forecast skill and the contraction-triggered response. |
| `analyses/forecast.py` | divergent | Feature-space baselines on the decimated grid; horizon curve over the checkpoint's $H$; $z$-unit error columns only. The overlay draws three kept channels against lead time and indexes the retained anchor axis by position. |
| `analyses/lag_kl.py` | divergent | Keeps the three profiles and measures truncation against preflight's `lag_support_margin_steps`. Relabels the axis and carries the group-delay caveat. |
| `analyses/latent.py` | equivalent | `test_eval_sibling_agreement.py::test_the_latent_spectrum_is_laid_out_the_same_way_in_both_packages`, `::test_the_latent_diagnostics_are_the_same_thirteen_reductions_in_both_packages`, `::test_the_only_gloss_that_differs_is_the_one_naming_this_target_domains_unit` |
| `analyses/perm_control.py` | equivalent | `test_eval_sibling_agreement.py::test_the_specificity_outcome_is_named_the_same_way_in_both_packages`, `::test_the_three_paired_controls_keep_the_same_sign_convention_in_both_packages`, `::test_the_branch_summary_and_the_kl_description_agree_in_both_packages` |
| `analyses/residual.py` | divergent | Reports the forecast-difference RMS in $z$ units. |
| `analyses/samples.py` | divergent | Draws the fifteen-row page through the task's page seams, plus a five-row compact page with a log-scale attention panel. Adds the class-balanced `by_class/` draw and one manifest row per file with a `variant` column. |
| `analyses/second_stage.py` | equivalent | `test_eval_second_stage.py::test_the_analysis_writes_its_five_tables_and_both_figures`, `test_eval_sibling_agreement.py::test_the_second_stage_clock_reaches_the_same_verdicts`, `::test_the_two_second_stage_modules_are_one_file_with_two_import_lines` |
| `analyses/sufficiency.py` | equivalent | `test_eval_sibling_agreement.py::test_the_sufficiency_scores_and_both_gaps_are_defined_identically_in_both_packages`, `::test_the_oracle_score_join_and_its_summary_rows_agree` |
| `analyses/time_to_delivery.py` | equivalent | `test_eval_sibling_agreement.py::test_the_time_before_delivery_grid_and_its_readouts_are_the_same_in_both_packages`, `::test_the_three_layers_of_inference_reach_the_same_verdicts`, `::test_the_two_time_to_delivery_modules_are_one_file_with_two_import_lines` |
| `analyses/trajectory.py` | divergent | The within-segment profile starts at the anchor floor $F$, because nothing below it is decoded. |

Two modules outside the fork are imported, not copied. Both are outside the sibling's `eval/` package, so the layering rule does not apply to them: `teb_vae/lag_attn_rws/collapse.py::is_collapsed` (stdlib-only, so `verify.py` stays free of `torch`) and `teb_vae/lag_attn_rws/trainer.py::RESOLVED_CONFIG_FILENAME`.

## Operations

### Exit codes

The exit code is non-zero if and only if a step raised. Three conditions do not change it:

- a failed sanity check, which logs at ERROR and leaves the code at 0;
- a coverage warning (two analyses report different populations);
- an inert-cap warning (a cap that no analysis read).

Run the offline verifier to assess acceptance. It checks the sanity record even when the evaluation process exited successfully.

### What a refused run leaves behind

Preflight rejects invalid input with `EvalPreconditionUnmet` before the wrapper that allows individual analysis failures. A refused run leaves `resolved_config.yaml` and `eval.log` with the reason, and no `summary.json`. Read the refusal message and use the guard recovery table below.

### Re-running one analysis

```bash
python -m teb_vae.lag_attn_cfs.eval.run --output-dir <a finished run> --only lag_kl
```

This command reads saved tables without a model or GPU. The analyses that draw the model's forward or intervene on its input (`samples`, `recording_traces`, `attribution`, `sufficiency`, `occlusion` and `time_shift`) need `--checkpoint` and record a skip without it.

A rerun does not delete results, but it does not add to them either. The runner renames the existing `summary.json` and `steps.json` to backups before it writes new ones, and it logs the backup paths. The new summary describes only the analyses just run. Read the backup for the other results.

An offline rerun reuses the existing `preflight.json`; without a checkpoint it cannot regenerate the causality disclosure. The table provenance records the checkpoint hash, seed, the Monte Carlo draw count actually used, the `--max-batches` cap, the row count and the `eval_config` digest. An offline pass adopts the recorded draw count and batch cap, because it cannot collect again. A mismatch raises `TablesProvenanceMismatch`: the saved tables belong to a different run or configuration.

### Guard recovery table

Each row lists a preflight guard, the reason it rejects a run, and the recovery action. The guard's error message gives the details.

| Guard | Cause | Recovery |
|---|---|---|
| `check_repointed` | A shard path or `stat_path` still carries the `REPOINT_ME` placeholder. This guard runs first, so the message names the real cause. | Edit `dataset_config.vae_test_datasets` in `eval/configs/eval_overrides.yaml` to name the causal holdout split. `stat_path` is inherited from the checkpoint's resolved config and must name the statistics file the run trained with. |
| `check_test_shards_exist` | A configured evaluation shard is not on disk, or none is configured. | Point `dataset_config.vae_test_datasets` at an existing build. Rebuild in `holdout` mode if the `test/` directory is absent. The pipeline's default `augmented` mode writes per-fold splits, which are not a substitute. |
| `check_stat_path` | `dataset_config.stat_path` is unset or names a missing file. | Make the statistics file the run trained with reachable at the path its resolved config names. The evaluation never normalises with statistics from the holdout split. A statistics file from the two-sided build fails later, at the loader's pairing check. |
| `check_trim_minutes` | The loader's trim is not the one the stored warm-up vectors were rebased at. | Set `dataset_config.dataloader_config.dataset_kwargs.trim_minutes: 1.0`. |
| `check_causal_transform` | A configured shard is the two-sided dataset variant, or declares no `transform`. The two variants share every field name and dtype; only the root attribute and the stored widths differ. | Point `dataset_config.vae_test_datasets` at the causal build (`hdf5_dataset/new_pipeline`, causal variant, `holdout` mode). |
| `check_load_fields` | `load_fields` omits a field that a readout or the anchor tiling's phase needs. The loader skips a field a shard does not carry without a warning. | Add the named field to `dataset_config.dataloader_config.dataset_kwargs.load_fields`. The committed `eval/configs/eval_overrides.yaml` lists the full set. |
| `check_target_normalized` | `fhr_st` or `fhr_ph` is missing from `load_fields` or from `normalize_fields`. The target is their concatenation. | Add both to `dataset_config.dataloader_config.normalize_fields` and to `dataset_config.dataloader_config.dataset_kwargs.load_fields`. |
| `check_no_reach_budget` | `model_config.VAE_model.causal_reach_budget_s` is set on one-sided features. It prunes channels by the forward reach of a two-sided Morlet bank. | Set `model_config.VAE_model.causal_reach_budget_s: null`. |
| `check_declared_widths` | The model's `c_y` / `c_u` disagree with the configured shards' stored widths. The check uses the model, because the evaluation rebuilds from the checkpoint. | Point `dataset_config.vae_test_datasets` at the shards this checkpoint was trained on. Do not change `model_config.VAE_model.c_y` / `c_u`; the checkpoint overrules them. |
| `check_warmup_budget_matches_checkpoint` | The warm-up budget re-resolved against the configured shards does not give the checkpoint's stamped channel tuples. | Set `model_config.VAE_model.causal_warmup_budget_steps` to the training value, or point `dataset_config.vae_test_datasets` at that run's dataset. |
| `reconcile_with_checkpoint` | A declared geometry or objective key contradicts the checkpoint's `model_kwargs` / `hyper_parameters`. The checkpoint always wins. | Evaluate the checkpoint against its own `model_checkpoints/resolved_config.yaml`. |
| `verify_weights_loaded` | Every witness tensor is still at its construction constant, so no checkpoint weights reached the model. `load_checkpoint_strict` returns `None` instead of raising. | Pass `--checkpoint` a trained `.ckpt` whose `model_class` matches this package's model. This is a weight-space check, not a behavioural one. |

### Dependencies

The evaluator uses the project's `torch`, `numpy`, `pandas`, `matplotlib`, `h5py`, `pyyaml` and `loguru` dependencies. It imports `scipy` where needed, `captum` (pinned `captum==0.9.0`) for the attribution pass, and `plotly` for the interactive trace pages. The pinned `pyarrow` dependency writes the per-anchor table. The `verify.py` acceptance check uses the standard library; the arm tables also need `pyyaml`.

### Tests

Run from the repository root. The `slow` marker is registered in `teb_vae/lag_attn_cfs/tests/conftest.py`.

```bash
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_cfs/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_cfs/tests -q -m slow
```
