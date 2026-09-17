# Evaluating the lag-attention model

This guide explains how to evaluate `teb_vae.lag_attn_cfs`, where to find the results, and how to interpret each analysis. Start with the quick reading path below, then use the analysis sections as a reference. [FIGURE_GUIDE.md](FIGURE_GUIDE.md) explains each emitted figure.

Documentation checks require an exact heading for every registered analysis, coverage of every resolved `eval_config` key and preflight guard, and an entry for every figure in `figure_manifest.json`. The generated divergence register records how this package differs from its source implementation.

This evaluation package was adapted from `teb_vae/lag_attn_rws/eval` for a different prediction target. It forecasts wavelet-modulus and phase-harmonic coefficients produced by a strictly one-sided filter bank. One forecast block contains $H\times C_{\mathrm{keep}}$ coefficients, where the checkpoint determines the horizon $H$ and retained channel count $C_{\mathrm{keep}}$. This affects model calls, masks, baselines, units, and analyses. The divergence register below is generated from `divergences.json` to keep those differences traceable.

## Quick reading path

1. Run the checkpoint evaluation using the command below, then inspect `preflight.json` and `summary.json` for status and configuration.
2. Read `forecast` to compare the model with simple predictors. Read `coupling` to compare target-only and source-conditioned forecasts.
3. Read `perm_control`, `source_null`, `latent`, and `calibration` before interpreting a positive predictive gap or a large KL value.
4. Use `time_to_delivery`, `second_stage`, and `cross_subgroup` for recording-level group comparisons. Detailed lag and attribution analyses help investigate the model's behaviour.
5. Run the offline verifier. A successful process exit alone does not establish that the model passed its acceptance checks.

## Configuration-dependent examples

The checkpoint and configured dataset determine channel counts, alignment, horizon, and anchor count. Historical examples in the detailed reference use $102$ declared target channels, $98$ retained target channels, $51$ declared source channels, and horizons of $15$ or $30$ steps. The current `configs/default.yaml` declares $80$ target channels, $46$ source channels, and $H=10$, using the integer phase-harmonic operator without input-channel alignment. Do not combine values from these different configurations. For a specific run, read `resolved_config.yaml`, `preflight.json`, `target_keep_index`, and `block_width`.

## Essential terms

| Term | Meaning |
| --- | --- |
| Recording or GUID | One recording, identified by `guid`. It can contribute many segments. |
| Segment | A fixed-length part of a recording given to the model. |
| Anchor | The time within a segment from which a forecast starts. |
| Horizon | The future steps predicted from an anchor. Each stored step spans $4$ seconds. |
| Coefficient or channel | A numerical feature produced by the signal transform. A channel follows one such feature through time. |
| Forecast block | All retained target channels over the forecast horizon at one anchor. |
| Base and full | Forecasts using target history alone and target history plus source history, respectively. |
| Latent state | A compact, uncertain representation of the history, described by a distribution with a mean and variance. |
| Prior and posterior | The target-only and source-conditioned latent distributions. |
| NLL | Negative log-likelihood: a predictive score for which lower is better. A negative log-density can be negative. |
| Predictive gap | Base score minus full score. A positive value means the source improved that score. |
| KL divergence | A measure of the difference between the two latent distributions. It measures a change in belief, not predictive improvement by itself. |
| Monte Carlo or MC | An estimate using several random latent draws. Here, the predictive score averages likelihoods before taking their negative logarithm. |
| Bootstrap interval | An uncertainty interval formed by repeatedly resampling recordings. |
| Cohort | A group of recordings, such as a clinical class or subgroup. |
| Warm-up | The initial interval a causal filter needs before its output no longer depends on assumed pre-recording history. |
| Lag | A source position measured in stored steps before an anchor. |
| Preflight or guard | A check performed before evaluation to reject incompatible data, settings, or checkpoints. |

## What a run is

One command reads one checkpoint and writes one reviewable directory:

```bash
python -m teb_vae.lag_attn_cfs.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt
```

The runner starts from the checkpoint's `resolved_config.yaml` and deep-merges the committed `eval/configs/eval_overrides.yaml` file over it. A deep merge replaces specified settings while preserving the other training settings. The overrides select the causal holdout shards and their statistics, load the five clinical fields, retain `guid` and `epoch`, and set `eval_config`. The summary records both the original and replacement value of every overridden key. Preflight rejects configurations that contradict the checkpoint.

Evaluation decodes every valid anchor. Training uses spaced anchors, controlled by `anchor_stride`, to reduce gradient correlation and memory use. Evaluation instead calls `model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1)`, matching `SeqVaeLagAttnCfsTask.resolve_anchor_geometry('test', batch)` and the validation/test geometry.

The number of decoded anchors is the checkpoint's `anchor_ceiling - warmup_period`. On the stored forecast clock this is $T_{\mathrm{valid}}-F$: for example, $156$ with $H=10$ and $F=134$, or $136$ with $H=30$ at the same floor. A `physical` forecast clock removes additional trailing anchors according to its largest label advance. The run records both evaluation geometry and training stride in `run_context`; use those values when comparing outputs.

The shared collection pass scores four branches at every valid anchor using $K$ Monte Carlo draws and common random numbers: `base`, `full`, `shuffled`, and `base_shuffled_mu`. It also computes `kld_source_null` by passing a zeroed source through the source gate, adapter, and encoder. This fifth arm returns only latent parameters $(\mu^{q,\mathrm{null}},\ell^{q,\mathrm{null}})$; it adds one source encode, no decode, and no `randn_like` call, so it does not advance the latent sampling stream.

The pass saves `per_sample.csv`, with one row per segment and its clinical labels, and `per_anchor.parquet`, keyed by `(guid, epoch, anchor_index)`. Vector sidecars and aggregate readouts are saved alongside them. Analyses that use these saved results can be rerun offline:

```bash
python -m teb_vae.lag_attn_cfs.eval.run --output-dir <a finished run> --only coupling
```

This command reruns `coupling` without a checkpoint, model, or GPU. The option `--max-batches` limits evaluation to an initial batch prefix and is intended for smoke runs. In contrast, `eval_config.max_samples` selects a seeded, stratified sample across the split. These caps are not interchangeable: a prefix of the unshuffled eight-shard split can contain only one subgroup and class.

A finished run is checked mechanically, and the arm tables are generated, by the same offline module:

```bash
python -m teb_vae.lag_attn_cfs.eval.verify <run>/eval_results/summary.json
python -m teb_vae.lag_attn_cfs.eval.verify --runs <dir-of-runs> --out RESULTS_arms.md
```

`verify` reads files a run left behind and nothing else — no model, no shard, no `torch` (the layering test walks its imports with `torch` on the forbidden list, so the property is proved rather than promised). The gate names which `pred_gap` column it reads: `pred_gap_mc_nats`, the Monte Carlo marginalised score. The arm tables key every row by the swept value read from each run's own dumped `resolved_config.yaml`, never by directory name, and read each run's training `metrics_history.csv` for the epoch count, the final `val/kld_active_frac` and the collapse verdict (`teb_vae/lag_attn_rws/collapse.py::is_collapsed`, imported rather than forked because it is stdlib-only arithmetic over a per-epoch series both raw packages already share).

`check_run.py` answers whether training behaved as expected **while a run is in flight**, using `train_results/metrics_history.csv`. `eval/verify.py` answers **is this finished checkpoint acceptable**, using the held-out evaluation in `summary.json`, recording-level results, and intervals. Both work without a checkpoint, dataset shard, or PyTorch, but they assess different evidence. Read both when assessing a trained model.

## Output layout

Everything lands in `<run>/eval_results/`:

| Artifact | What it is |
|---|---|
| `summary.json` | The whole run: readouts, verdicts, headline, sanity, cohort, coverage, run context, causality disclosure, step records, artifact manifest. |
| `steps.json` | The per-step heartbeat, rewritten as each analysis finishes — a killed run's record of how far it got. |
| `preflight.json` | Every guard's verdict, the causality disclosure, the resolved warm-up budget, the per-block group-delay summary and the measured lag-support margin; reused (not regenerated) by a model-free re-run. |
| `loader_probe.json` | The population record: per-shard, per-class and per-label counts the sanity checks read back. |
| `resolved_config.yaml` | The merged configuration the run actually used — the file an offline re-run reads. |
| `eval.log` | The run's log, including any refusal. |
| `per_sample.csv`, `per_sample_vectors.npz` | One row per segment: every scalar readout plus labels and provenance; the vector readouts in row order. |
| `per_anchor.parquet` | Per-anchor scores, KL, argmax lag, coverage, `seconds_since_contraction`, keyed on the forward's own `anchor_index`. |
| `per_anchor_vectors.npz` | The per-anchor lag maps — the pooled KL attribution and the head-averaged attention over the lags at every contributing anchor — row for row with `per_anchor.parquet`, in `float16`. Read by `lag_high_kl` alone. |
| `retained_arrays.npz` | The opt-in retention: per-anchor forecast blocks and attention maps for the capped sample set. |
| `collection.json` | The collection record: readouts, provenance sidecar, denominators, retention plan, the measured cost of the pass, and `target_keep_index` — the kept-channel axis the band-resolved readout joins through. |
| `band_partition.json`, `band_channel_map.csv`, `band_channel_map_kept.csv` | The input channel map (the unskippable data-side step), on the declared axis and on the kept one. |
| `<analysis>/…` | One subdirectory per analysis: its CSVs and PDFs. |
| `attribution/…` | The Captum attributions: `attribution_rows.csv` (one row per attributed anchor, readout and baseline), the row-aligned `attribution_vectors.npz`, the per-recording, summary, band, lag-band, layer and null tables, one example anchor per class in `attribution_maps.npz` (the input streams beside every example readout's maps under both baselines, listed in `attribution_examples.csv`) with its page under `maps/`, and under `traces/<class>/` one arrays file and one figure per traced recording. |
| `recording_traces/…` | The traced recordings: `recording_traces.csv` (the manifest), `segment_summary.csv` (one row per segment), `anchor_trace.parquet` (one row per decoded anchor), and per recording under its class directory the full vectors (`<guid>_<subgroup>_full.npz`) and the figure. |

Four summary blocks help you read a run:

- **`headline`** contains the scalar values and verdict statuses used by the acceptance gate and arm tables. Unregistered values are not consumed by those tools.
- **`sanity`** records consistency checks, including KL identities, agreement between tables, lag identities, and population counts. These checks do not change the process exit code.
- **`verdicts`** contains the model's ten named acceptance criteria, in registry order, with explicit statuses rather than a single boolean.
- **`run_context`** records parameter count, checkpoint epoch, decoded anchor geometry, training stride, anchor coverage, and observed objective magnitude.

The headline keeps three forecast-gap definitions separate. `pred_gap_mc_nats` is the gate's Monte Carlo score: take the negative log of the average likelihood over $K$ draws, then subtract full from base. `pred_gap_mean_nats` decodes both latent means using the decoder's predicted variance, with no draw. `pred_gap_train_path_nats` reproduces the training scoring path; under `base_decode: mean`, base uses the prior mean while full uses a sampled latent. Four percentage columns give proportional versions of these comparisons. The saved `pred_gap_convention` explains their differences, possible sign disagreements, block size, and budget-local percentages.

## The four layers

The evaluator has four layers. `tests/test_eval_self_contained.py` checks imports by walking the Python syntax tree, including aliased, lazy, and relative imports. This keeps offline analysis and verification independent of model execution.

| Layer | Modules | May import |
|---|---|---|
| 0 — pure | `config_schema`, `verify`, `events`, `frames`, `lag_axis`, `cohort`, `figures_seam`, `report_seam`, `launch`, the `_reuse` seam | no Lightning, no `model.*`, no `task`/`trainer`/`plotting`; `verify` additionally no `torch` and no `binding` |
| 1 — model-touching | `binding`, `metrics`, `collect`, `preflight`, `probe`, `oracle` | `task`/`trainer` only via the named `EXEMPTIONS` table |
| 2 — I/O and presentation | `analyses/*` | layers 0–1; never another analysis, never Lightning, never the model |
| 3 — orchestration | `run` | everything at the layers below it |

Two additional import rules apply here. Production modules cannot import `teb_vae.lag_attn_rws.eval`; only named comparison tests may do so. This prevents the adapted pipeline from silently using parts of the original implementation. Also, `binding` and `probe` belong to layer 1 because they identify or rebuild a model. The CFS forward contract is checked against that rebuilt model, including its required anchor phase when stride exceeds $1$.

The shared evaluation package (`teb_vae/lag_attn/eval`) is reachable only through its model-free modules, named in an allow-list, and `model/*` is forbidden everywhere. The `EXEMPTIONS` table is asserted **minimal**: a module listing a name it no longer imports is a permission that outlived its use.

### The model binding

`binding` defines model-specific facts: the classes loaded from a checkpoint, constructor keys to reconcile, encoder causality disclosure, override file, and supported analyses and headline scalars. `run.main` uses `CFS_BINDING` by default. The transformer sibling, `teb_vae/lag_attn_transformer_cfs/eval`, also uses this runner, collection pass, and analyses through its own binding for `SeqVaeLagAttnTrfCfs`.

`GEOMETRY_KEYS` is **sixteen** here against the raw cells' fourteen, adding `anchor_stride` and `lag_floor`. It obeys a rule narrower than "the constructor's parameters": `preflight.reconcile` compares `model_config.VAE_model[key]` against `model_kwargs[key]` and silently skips any key absent from either, so a key must be a config key **and** a constructor parameter to be checked at all. `causal_warmup_budget_steps` is therefore deliberately absent, and so are the four tuples it resolves to (`target_keep_index`, `target_warmup_steps`, `source_keep_index`, `source_warmup_steps`): the budget is a config key but not a constructor parameter, and the tuples are constructor parameters but not config keys, so reconciling either here would compare against nothing.

They get their own guard instead — `check_warmup_budget_matches_checkpoint` re-resolves the budget against the *configured* shards and compares the result with the checkpoint's stamped tuples, which is the only comparison that can actually fail.

## Configuration reference

Set evaluation choices in the override file's `eval_config` block so the resolved values are saved with the run. The schema rejects unknown keys, a boolean where an integer is required, and caps of zero. Error messages name the valid keys.

| Key | Meaning |
|---|---|
| `seed` | Seeds `random`/`numpy`/`torch` and derives the loader-shuffle, derangement, stratified-cap and Monte Carlo generators by fixed offsets. Two runs of one checkpoint at one seed compare byte-identical on `results`. |
| `num_mc_samples` | Monte Carlo draws $K$ per anchor for the marginalised score, under common random numbers across branches. $K = 1$ is one draw of the same estimator, not the training-path score: under `base_decode: mean` the training path decodes the base branch at the prior mean, which the estimator never does. Convergence in $K$ is unmeasured (CFS-10). |
| `max_samples` | Seeded **stratified** global sample cap; `null` evaluates the whole split. |
| `caps` | Limits for retained arrays and additional analyses: `waveforms`, `attention`, `pages`, `pages_per_class`, `oracle`, `traces_per_class`, `occlusion`, and `attribution_segments`. Omitted array-retention caps retain no samples. Omitted `oracle` and `occlusion` caps allow every segment; omitted trace and attribution caps use their analysis defaults. The current `waveforms` cap is $64$. Memory use depends on anchor count, horizon, retained channels, and dtype. |
| `prior_shuffle_min_nats` | The provisional margin the prior-shuffle degradation must clear; the verdict always reports the measured number beside it. |
| `min_active_dims` | Active latent dimensions below which the latent counts as collapsed. |
| `event_lag_window_s` | Seconds after a detected contraction within which an anchor counts as event-conditioned. Still read here, because contraction-*conditioned* coupling ports even though the two readouts that scored a clinical trace do not; `lag_high_kl` reads the same window for its contraction enrichment of high-KL anchors, so the two analyses agree about what "near a contraction" means. |
| `bootstrap_resamples` | Resamples behind every bootstrap interval, drawn over recordings — never over anchors, whose neighbouring forecast windows share $H-1$ of their $H$ steps. |
| `clock_margin_min_nats` | The current override sets `clock_margin_min_nats` to $0.15$ nats per anchor. The margin $\Delta_{\mathrm{clock}}$ must clear for `coupling_exceeds_availability_clock` to PASS. At `null`, the verdict is INCONCLUSIVE and the measurement is still emitted. See `source_null` below for the threshold's provenance and limitations. |
| `figure_format` | Image format every figure of the run is written in, as a matplotlib filetype (`pdf`, `svg`, `png`, `eps`, …); validated at config load against the installed matplotlib's own list. `null` — the shipped setting — keeps the `pdf` default, which is what `figure_manifest.json` and `FIGURE_GUIDE.md` record and what the smoke suite compares a real run against; a run that changes it writes filenames those files do not list. |
| `max_hours_before_delivery` | How far before delivery a segment may be recorded and still be evaluated, in hours; `4.0` keeps the last four hours, `null` — the shipped setting — evaluates everything. **The bound is on the population, not on an axis**: it is applied to the delivery clock before anything is binned, so every clock answers for the same segments and the second-stage clock re-bins that population on its own signed axis rather than being cut at a second, differently-defined four hours. It moves cohort sizes, window counts and every trajectory, so a bounded run is not comparable with an unbounded one — which is why it is a key, recorded in the run's dumped config. Minimum one 0.5 h bin. |

Deliberately **not** keys: the significance level and the trajectory bin width (an operator who could widen them could make a difference appear or disappear), any lag-band selection, and — the one this cell adds — **the anchor stride**. An operator who could set the stride could change the population every number in the run is computed over. Their absence is asserted by test, not merely intended.

### Objective keys this pass reads rather than sets

The objective is not an `eval_config` surface. A checkpointed pass rebuilds the task from the checkpoint's **own** `hyper_parameters` — `beta_schedule`, `kld_beta`, `beta_prior`, `lambda_full`, `lambda_base`, `likelihood` and `free_bits` — and refuses a checkpoint that carries none, because scoring it under assumed defaults would report a different objective's numbers. On the offline path, where there is no checkpoint, the same keys are read from the dumped `model_config.VAE_model` block, which preflight has already reconciled against the checkpoint on every checkpointed run.

Weights are recorded and, unlike geometry, deliberately **not** compared against the config by the preflight guard: $\beta$ and its ramp weight the training total and enter no evaluated readout.

### Cohort order and colour

Two presentation conventions every table and figure that resolves a quantity by cohort obeys. Neither is a setting, for the reason the significance level is not one.

**The order is clinical, not alphabetical, and it runs worst first**, on both axes:

| Axis | Order, left to right |
|---|---|
| `clinical_class` | HIE, acidosis, healthy |
| `subgroup` | `hie_cs`, `hie_no_cs`, `acidosis_cs`, `acidosis_no_cs`, `healthy_bg_cs`, `healthy_bg_no_cs`, `healthy_no_bg_cs`, `healthy_no_bg_no_cs` |

`labels.ordered_groups` is the one function that decides it — `cohort.ordered_groups` is a one-line binding of it, so this package and the sibling cannot come to disagree — and both orders are read off `labels.CLASS_NAMES` and `labels.CANONICAL_SUBGROUPS` in reverse rather than restated. Those tables are bound through `_reuse.py` and are deliberately **not** forked, because forking them would fork the definition of a cohort. A cohort the order does not know sorts **after** every one it does, and is never dropped.

**The colour is the severity**: green for healthy, amber for acidosis, red for HIE, each subgroup a shade of its own class. The mapping is a *table* rather than an assignment pass, so a cohort keeps its colour whichever others a figure contains. The palette is this package's own and deliberately not `utils.style.CLASS_COLORS_DEFAULT`, which paints healthy blue and is shared with two other projects — so **an evaluation figure of a cohort is not the same colour as a training-callback figure of that cohort**, and the two are reconciled by legend rather than by hue.

**The order is also the orientation of every significance test.** `stats.pairwise_comparisons` names each pair in the order it receives the cohorts, and every caller hands them over through `cohort.ordered_groups` — so a comparison always runs *more severe to less severe*: HIE vs acidosis, HIE vs healthy, acidosis vs healthy on the class axis, and the same order pair by pair on the subgroup axis. That is the direction the clinical question is asked in: the cohort a readout exists to detect is named first, and what it is read against second.

Cliff's delta is signed against that naming, so **a positive $\delta$ means the more severe cohort's values run higher**, on every pair of every metric, window and clock rather than on the ones whose names happened to sort that way. A run directory written before this convention names the class pairs healthy-first instead, with the opposite sign.

## The analyses

One section per registered analysis, named for its module. `band_partition` always runs and is not selectable; the rest are what `--only` and `--skip` choose between, in this order, and `run.py`'s `RUN_ARGS` comment table is the list an operator reads while choosing. `cross_subgroup` is deliberately last, and that ordering is load-bearing: it reads the per-recording CSVs the analyses above it write.

### band_partition

What each of the model's input channels is, read off the shards' own `sel_*` provenance and causal attributes rather than re-derived: one row per channel across the 102-channel target stream (`fhr_st` 36 + `fhr_ph` 66) and the 51-channel source stream (`up_st` 36 + `up_ph` 15), laid out as the model receives them, each mapped to a band, a kind and a centre frequency in Hz. It describes the model's **inputs** and is the data-side companion to the causality disclosure.

Two columns are this cell's own and every statement in this pipeline rests on them: `causal_warmup_steps`, the leading delay enclosing 95% of the one-sided kernel's energy, and `causal_delay_s`, the composed group delay. A third, `kept`, marks the channels the warm-up budget dropped.

**On this dataset the unbanded path is the common case rather than the exception.** `sel_*` provenance is stored on the two *phase* blocks only, so a scattering channel's centre frequency is recoverable only where some selected phase pair named its filter. Seven of the 36 declared `fhr_st` channels have none: three above the phase selection's upper edge and four below its $0.008$ Hz floor. Those channels are recorded as `unknown` and **never bucketed into a neighbour** — a band whose membership quietly absorbed them would misattribute their skill to a frequency they do not have. A shard carrying no `sel_*` attributes at all is a recorded skip, not a raise.

It also emits `band_channel_map_kept.csv`, the same map restricted and re-indexed onto the **98 kept channels** the decoder actually emits. That second file exists for one reason: it is the join `spectral_skill` goes through, and it is a file on disk rather than a model attribute so that `--only spectral_skill` against a finished directory works with no checkpoint and no GPU.

**The two attributes the aligned shard variant added are deliberately not read here, and this says which and why.** A causal shard now also carries `causal_leg_alignment` at the root and a novelty record per block (`causal_novelty_curve`, the horizon-free table; `causal_novelty_frac` on legacy shards).

* `causal_leg_alignment` names which phase-harmonic operator built the phase blocks. It is a property of the **file**, not of a channel, and it is already refused at resolution time: a run whose `causal_leg_alignment` config key disagrees with what its shards record never reaches an analysis. Repeating it as a channel column would put the same fact in two places, one of which could be stale, and neither `band_partition` nor `spectral_skill` needs it — every column both emit means the same thing under either operator, because the alignment changes what a phase coefficient *is* and not what channel it is.
* The novelty record is genuinely per channel and would fit both maps. It is not added, and the reason is that it would be a second, offline copy of a split the **training** path already reports per epoch as `pred_gap_novel_lo` / `_mid` / `_hi` — computed against the model's own gathered channel axis rather than against a positional join. Add the column here when a band-resolved novelty question is actually asked; until then, the number a reader wants is in `metrics_history` and in the run's own tertile columns.

`spectral_skill` reads no shard attribute at all: it carries `causal_warmup_steps` and `causal_delay_s` forward from the kept map above. Whatever `band_partition` emits, it inherits, so the answer for it is the same answer.

**One column that is now narrower than it reads, recorded here rather than repaired.** The `kept` column is computed for the **target** stream alone; every source row is written `kept = 1`. That was exactly true while the source keep-index was the identity, and it stopped being true when the channel alignment began dropping the four source channels whose composed delay exceeds the reference — those rows still read `kept = 1` while the model does not read them. Repairing it needs `source_keep_index` in the collection record, which is a change to the collection schema rather than to this analysis; until then the run's own preflight record carries `source_dropped_index`, which is the authoritative list.

### forecast

This analysis compares predictive performance with three simple baselines and shows how performance changes over the horizon. A block score sums $H\cdot C_{\mathrm{keep}}$ coefficient scores, so its magnitude depends on the forecast dimensions. Compare predictors on the same block, mask, and anchors. Use the run's `block_width` rather than assuming a fixed size.

The three baselines are rebuilt in feature space on the decimated grid, and each has an exact analogue there:

- **persistence** — anchor $t$'s whole window is filled with the coefficient vector at the last **observed** step at or before $t$, per channel. "Last observed" rather than "last": `weight` is the only trustworthy validity signal here, because the coefficients carry no sentinel of their own, and carrying an invalid step forward would measure the gap.
- **climatology** — exactly $0$ per channel, which is the z-scored population mean. The statistics were accumulated *excluding* the warm-up region, which is what makes zero the channel mean over the region the model reads.
- **segment mean** — the per-channel mean over the segment's own valid steps.

All three are scored at a fixed `BASELINE_LOGVAR = 0.0`, recorded beside the score, so a learned-variance model cannot beat a point predictor on variance modelling alone without that being visible.

The MSE-space skill $1 - \mathrm{MSE}_m/\mathrm{MSE}_b$ is the one with a natural zero. The NLL-space column beside it is a **difference** in nats, `advantage_nats_per_anchor`, not one minus a ratio: a log score has no natural zero, so the ratio of two of them is not bounded above by one and changes sign with the baseline's.

**Every error column is in the loader's $z$ units, labelled `normalised`, and there is no conversion out of them anywhere in this pipeline.** The sibling's `BPM_UNIT`, `to_bpm`, `sigma_to_bpm` and `fhr_normalization` are deleted rather than repointed: a scattering or phase-harmonic coefficient has no clinical unit, and inverting the per-channel statistics would put the 98 scored channels on scales spanning orders of magnitude, which destroys every pooled statistic, every shared colour bar and the tertile split.

The horizon curve covers the checkpoint's $H$ forecast steps and uses the single-draw scoring path. Its label states that convention. Monte Carlo mixture scoring does not commute with summing horizon steps, so separately marginalised step scores would not generally add to the marginalised headline.

### coupling

This analysis measures how source history changes prediction, using one value per recording. It keeps three scores separate: the Monte Carlo marginalised headline, the mean-decoded gap, and the training-path parity score. Outputs include the fraction of recordings with positive gaps, a paired Wilcoxon test over recordings, bootstrap intervals, and quantiles.

The marginalised and mean-decoded scores answer different questions and can disagree in sign. The marginalised score averages likelihood over latent draws, so it includes latent uncertainty and is sensitive to draw count $K$. The mean-decoded score asks how the forecast scores when each branch is decoded at its latent mean. Under `base_decode: mean`, its base score exactly matches the training-path `nll_base_block`.

The distribution figure compares all three estimators per recording and reports how often their signs agree. Other figures identify the estimator they display; check their labels. The gate continues to use the pre-registered `pred_gap_mc_nats` criterion. Changing that criterion requires an explicit decision in `RESULTS.md`.

The positive fraction reports its **denominator**: `np.nan > 0` is `False`, so unscored segments would otherwise count silently as evidence against. The KL travels beside the gap as a **description** rather than as a second answer — it is inflated by an arbitrary factor whenever the prior variance sits on its clamp, and unlike `pred_gap` it says nothing about whether the forecast improved.

Percentage scores describe proportional change. `pred_gap_rmse_pct` and `pred_gap_mse_pct` measure the reduction in point-forecast error; $100\%$ means zero remaining error. `pred_gap_mc_likelihood_pct` is $100(\exp(\Delta/(H\cdot C_{\mathrm{keep}}))-1)$, a geometric likelihood-density improvement scaled by the run's block width. Values are computed per recording, then averaged and bootstrapped over recordings. `frames.skill_against` requires a strictly positive error denominator; otherwise it returns `NaN`, rather than infinity or a misleading zero.

The likelihood percentage has the sibling's two preconditions — a `gaussian_nll` likelihood, and the block size read from the run's own geometry — and one more that is this cell's: **it is budget-local**. It divides by $H \cdot C_{\mathrm{keep}}$, and $C_{\mathrm{keep}}$ is whatever the warm-up budget decided, so two arms of this model at two budgets divide by two different numbers. The emitted record states it; nothing tries to normalise it away.

### perm_control

This control tests whether prediction benefits from the matching recording's source. The verdict compares predictive scores: $D_{\mathrm{full}}<D_{\mathrm{base}}<D_{\mathrm{shuffled}}$. A positive source margin, $D_{\mathrm{shuffled}}-D_{\mathrm{full}}$, means replacing the matched source worsened prediction.

Assess specificity in prediction space. A mismatched source can move the latent posterior farther from the prior and produce a larger KL, even while making prediction worse. The permutation is GUID-aware: it pairs different recordings. Batches without a valid cross-recording pairing are excluded and counted. Interpret paired control results using the collection pass's saved scores and sampling convention.

Three paired controls are scored per recording under one sign convention — positive means the control is worse than the branch it is referenced against: `shuffle_penalty` ($D_{\mathrm{shuffled}} - D_{\mathrm{base}}$), `prior_shuffle_penalty` (the same branch under a shuffled prior mean) and `source_margin` ($D_{\mathrm{shuffled}} - D_{\mathrm{full}}$). **The third is referenced against `full`, and that is why it exists**: the two above it inherit whatever the base forecast is doing, while `source_margin` changes only the source. A positive margin beside a negative predictive gain is a real state rather than a contradiction.

**This control structurally cannot see the availability-clock hazard**, and that is why `source_null` exists beside it rather than instead of it. A permutation deranges *rows*, and the source availability pattern is a deterministic function of $t$ that every row of a batch shares, so no permutation of rows can remove it. What this control answers is specificity, which the source-null arm does not.

### latent

This analysis measures how strongly each latent dimension changes when the source is added and whether the prior variance is near its allowed bounds. It reports the per-dimension KL spectrum, active-dimension count, largest dimension's share, `mean_logvar_prior`, `logvar_prior_floor_frac`, and `mean_logvar_post`.

The variance check matters because KL includes $(\mu^q-\mu^p)^2/\sigma_p^2$. A very small prior variance can inflate KL even when forecast diagnostics look normal. The `prior_variance_not_pinned` verdict checks whether prior variance is near its lower bound, using a margin of $5\%$ of the clamp range. Exact equality would be unsuitable because the sigmoid bound is approached rather than reached.

`prior_rate` is the same pathology as a **distance rather than a fraction**: the objective's own $R_p = \sum_d \tfrac12(e^{\ell^p} - 1 - \ell^p)$, per recording and in nats per anchor, reduced on the KL support like the divergence beside it. Zero means $\sigma_p = 1$ exactly, so it is the only one of these readouts bounded below by its own optimum, and it is continuous where the floor fraction is a step. Read the two together.

### lag_kl

This analysis distributes each attention head's KL over its lag weights. The attribution is $\widetilde K_{t,\ell}=\sum_m K_t^{(m)}\alpha_{t,\ell}^{(m)}$, and summing over lags gives $K_t$. Every run checks this identity at its worst anchor and records the result under `sanity`. The profile describes the model's allocation over stored lags; it does not establish a physical delay.

The analysis provides three profiles:

- **Raw:** Divides every lag bin by the same anchor count, so the bins decompose the headline KL.
- **Support-corrected:** Divides each bin by the number of anchors where that lag was causally valid.
- **Untruncated:** Uses only anchors where the entire lag window was available. This also changes which observations contribute, whereas support correction changes only the denominator.

**At this cell's geometry the last two corrections are inert, and the analysis measures that rather than assuming it.** The anchor floor is $F = 134$ and the furthest searched lag is $L - 1 = 90$, so every lag is causally valid at every scored anchor and the support margin is $134 - 90 - 0 = 44 \ge 0$. That margin is a **number preflight computes and records**, not an assumption: the floor, `max_lag` and `lag_floor` move independently, and a lower-floor arm would silently reintroduce truncation. This analysis reads the recorded margin, measures the per-lag contributing-anchor counts, compares the three profiles and records whether the computed and observed readings agree.

A negative margin is legitimate rather than refused — the corrections then do work again, which is what they are for.

An argmax is not by itself a reading. Peak width, mass above threshold and secondary peaks travel beside it, and `degenerate` is defined mechanically — peak-to-median below 1.1, **or** exact-zero fraction above 0.9 — because `entmax15`'s exact zeros can make an argmax on a flat profile meaningless.

**The axis is stored-coefficient time**, and the group-delay caveat travels on every artifact this analysis writes. See *How the output is misread*.

The stratified table cuts the profile by class, by subgroup and by time-before-delivery window, one axis at a time and without a test. `lag_clocks` is where that question is asked properly: the same profile resolved against **both** clinical clocks, by class, drawn, and tested per window.

### attention

This analysis shows how each attention head distributes weight over source lags. Each head controls its own latent group, so inspecting heads separately preserves information that averaging would hide. For example, several heads focused on different lags can have a broad average profile even though each head is individually focused.

The entropy ceiling is $\operatorname{mean}_t \log \min(t+1, L)$ over the anchors actually scored. **At this floor that equals $\log L$ exactly, and the analysis measures it rather than substituting the constant.** Three readings of the one property are compared and their agreement recorded: preflight's own margin, the geometry record's truncated-anchor count, and the accumulated ceiling against $\log L$. Both entropies are emitted, distinctly named, and the ceiling is a per-sample column over the sample's own scored anchors, so their ratio stays a measurement on an arm where it is not a no-op.

The entropy is taken per anchor and then averaged, never as the entropy of the averaged profile — a mixture's entropy is at least the mean of the entropies mixed, so the second reports a model whose lag focus *shifts* as one that has none. `kld_per_t_per_head` sums over heads to `kld_per_t` exactly, and that identity is the second sanity-block check.

### calibration

Calibration checks whether the decoder's predicted uncertainty matches observed errors. A Gaussian NLL remains a log-density score even when the model is miscalibrated, but a good average score alone does not show that its uncertainty intervals have the advertised coverage.

Four diagnostics are calculated over target coefficients, with counts named `n_coefficients` and gains named `gain_per_coefficient`:

- **PIT:** The probability integral transform, or the observation's cumulative probability under its forecast distribution. A calibrated continuous forecast gives a uniform PIT distribution.
- **Central coverage:** The fraction of observations inside the predicted intervals. The Gaussian reference levels are $\operatorname{erf}(k/\sqrt{2})=0.6827,\ 0.9545,\ 0.9973$ for $k=1,2,3$. The two-standard-deviation level is $0.9545$, rather than $0.95$.
- **CRPS:** The continuous ranked probability score, which compares the predictive distribution with the observed value and remains in normalised $z$ units.
- **NLL gain:** Improvement over a constant-variance maximum-likelihood reference fitted to the same residuals.

Read `logvar_full_floor_frac` and `logvar_full_ceil_frac` alongside `mean_logvar_full`. An average alone can hide values concentrated at both limits. The analysis recommends a per-coefficient `model_config.VAE_model.logvar_clamp` revision when a bound constrains predictions, or reports no change when neither bound does. An `mse` checkpoint records a skip because its log-variance head was not fitted.

### residual

This analysis compares the forecasts and the latent states that produced them. `mu_base` and `mu_full` come from the same decoder applied to two latent states. The model does not emit a separate residual tensor or `delta_mu_src` head.

Reported instead: the per-anchor forecast-difference RMS in $z$ units, and the two latent-side quantities that are **not** the same thing — `delta_mu_rms`, per element, and `mu_post_prior_gap_rms`, per step with the L2 over $d_z$ taken first.

Root-mean-square (RMS) values are calculated by accumulating squared differences and taking one square root at the end. Averaging already-rooted sample values gives a smaller result in general. Both versions are reported so the difference is visible. The two branches also share one log-variance head, applied to their different latent inputs.

### distributions

This descriptive analysis shows the shape of metric distributions across 20-minute segments, grouped by cohort. Similar means can hide different tails or a small number of very large errors. The figures also show recording-level summaries so readers can see how repeated segments affect the distribution. Error quantities remain in the loader's normalised units; each panel names its metric.

**It is descriptive by construction.** No test, no interval, no $p$-value, and nothing registered in the headline block. That is not an omission: a per-segment $p$-value is anticonservative by the anchor overlap, and `cross_subgroup` remains the only analysis that adjudicates a cohort difference.

**Both levels are drawn on the same axes, and that is the content.** The filled density is one value per segment; the median / inter-quartile / range **strip** above it is one value per recording. Their difference *is* the pseudo-replication. Four presentation choices are load-bearing: density rather than counts, one bin grid per panel, a nested subgroup figure, and the overlap encoding — a faint fill under a hairline outline at full opacity, drawn in two passes so every outline sits above every fill.

It declares **no** `grouped_frames`. The runner's fan-out draws violins documented as holding one value per recording; handing it this per-segment frame would produce a per-segment violin that reads as a per-recording one.

### trajectory

This analysis follows predictive gain and KL through time, both within a segment and across a recording. It reads the per-anchor table.

**The within-segment structural caveat is this cell's rather than the sibling's, and it is the opposite shape.** The raw cells show a warm-up droop at the left of the profile, because their anchors begin at the model's own 30-step warm-up and the lag support is truncated for a while after it. Here **nothing below the anchor floor $F = 134$ is decoded at all**, so the profile *starts* there: there is no droop to discount and no truncated region inside the profile, and the last $H$ anchors are still never scored.

A reader expecting the sibling's shape and finding a profile that begins two-fifths of the way into the segment is looking at the geometry rather than at a failure.

Across a delivery the segments are assembled on the absolute time axis $t_{\mathrm{abs}} = \mathrm{epoch} + 4t$, with overlapping timesteps **averaged** rather than drawn twice and `n_contributing` travelling so the averaging is visible rather than inferred. A gap produces a **break** in the data — `gap_before_s` — rather than an interpolation.

### time_to_delivery

This analysis shows whether predictive gain and KL change as delivery approaches, and whether those changes differ by clinical class. It uses half-hour windows of hours before delivery, $-\mathrm{epoch}/3600$. Segments are first averaged within each recording and window, so a recording with many segments does not outweigh one with only a few.

`pred_gap` is tracked beside the KL because the two fail differently: `pred_gap` is in the decoder's own units and is immune to the prior-variance inflation. Significance is tested **per window**, with Holm across windows as one family and pairwise tests on the survivors; the `pooled` row is flagged `confounded_by_time` and consumed by nothing. `TRAJECTORY_BIN_HOURS` is a module constant, not an `eval_config` key, for the reason the significance level is not one.

`lag_clocks` resolves the **lag structure** against this same grid and the same classes, so a window here and a window there are the same duration over the same recordings: this one says how much coupling there is, that one says where in the past it came from.

The analysis emits **four** figures — two pages per readout, because `pred_gap` and the unfloored KL share a unit and not a scale, so a page carrying both draws the smaller as a flat line at the bottom of the larger's range. `time_to_delivery_trajectory_<readout>.pdf` is the median line per class with its inter-quartile ribbon; `time_to_delivery_windows_<readout>.pdf` is what that line is made of — a violin per (window, class) cell over one value per recording, the Holm-adjusted $p$ of each window directly beneath it on the same axis, and Cliff's delta for every class pair that survived. The tests were always run; until that page existed nothing drew them.

### second_stage

This analysis aligns recordings to the onset of the second stage of labour. It uses the same predictive-gap and KL readouts as `time_to_delivery`, but a different clinical landmark. Recordings equally far from delivery may be at different stages of labour.

The second-stage axis uses signed hours from onset. The stored value is `second_stage_onset = domain_start - t_SSO`, so negative means before onset, zero means onset, and positive means after onset. Divide by $3600$ without negating it. Delivery time uses a different convention: hours before delivery are $-\mathrm{epoch}/3600$. The second-stage figures use a natural left-to-right axis and mark zero.

**Eligibility: one rule drops a recording, and two diagnostics drop nothing.** A recording with no recorded onset cannot be placed on this axis and is excluded and counted. The two further ways a stored onset can be wrong are **counted and filtered nowhere**, and both reach `second_stage_eligibility.csv` and the record: an implied onset falling *at delivery*, which is what a pipeline writes when it substitutes zero for a missing time, and an implied onset that *moves* across a recording's own segments by more than 1 s, which can only come from a broken write. Excluding a recording changes the population every number is computed over; a count does not.

**The Holm family is this clock's own.** The correction runs across the windows of this clock and is **not** joint with `time_to_delivery`'s. The two are different alignments of an overlapping population, so a window significant on one and not the other is a statement about alignment, and the family-wise error rate each correction controls is within its own clock — a reader combining a claim from both clocks is making two comparisons.

**It is `capped`, deliberately.** It scores a subset of the evaluated cohort, so it declares `plan.capped = True` with its reason and is excluded from the coverage block's population comparison rather than reported there as a disagreement about who was evaluated. The grid is `TRAJECTORY_BIN_HOURS`, the same 0.5 h windows the delivery clock uses and the same module constant rather than an `eval_config` key. Recorded skips, each naming its cause: an empty table, a table collected before the `second_stage_onset` column existed, a cohort with no onset at all, a cohort whose readouts are all non-finite, and a single-class split.

### events

This analysis compares predictive gain and KL shortly after detected contractions with count-matched control anchors from the same recordings. An event anchor is within `event_lag_window_s` of a contraction. The collection pass detects contractions from the raw uterine-pressure context and stores `seconds_since_contraction` in the per-anchor table, allowing the analysis to use the whole split. It records a skip unless there are at least $200$ event anchors across at least $4$ recordings.

Gaps are masked by `weight`, never by value. Masking is two steps and both are needed: invalid samples are interpolated across *before* smoothing, so a gap contributes no edge for the peak finder to lock onto, and any event whose span touches one is then **dropped**, because its shape partly came from that interpolation. The contraction onset is a **level crossing** of the peak's own prominence rather than a gradient walk-back, which is a deliberate correction to the ported detector: a gradient test stops at the apex, where the smoothed gradient is approximately zero.

**One readout of the raw pipeline's three, and the two that are gone are named in the emitted record rather than merely missing.** Deceleration forecast skill and the contraction-triggered response both score a clinical heart-rate trace in beats per minute; this model forecasts 98 coefficients, and defining a deceleration on a channel axis with no order and no clinical unit is a new scientific construction rather than a port. `REMOVED_READOUTS` carries both names and both reasons into `summary.json`, so a reader who expects three meets the absence rather than inferring it from a missing key.

### sufficiency

This analysis estimates how much prediction is limited by the latent representation. It reports $\Delta_{\mathrm{suff}}=D_{\mathrm{base}}-D_{\mathrm{oracle}}$. The oracle is an evaluation-only decoder with the same capacity as the model's decoder, but it reads `target_state` directly instead of the latent $z$. It is fitted on half the evaluation recordings and scored on the other half, using the same anchored feature builders and $H\cdot C_{\mathrm{keep}}$ outputs per anchor.

**It is an estimate, not a bound**, and both bias directions travel in the emitted JSON rather than only here. Conditioning on `target_state` rather than on the target's own history omits the encoder's information loss and biases the gap **down**; fitting the probe on the evaluation population while $D_{\mathrm{base}}$ comes from a model trained on the disjoint, healthier pretraining cohort biases it **up**. The two oppose, neither is measured, so nothing downstream may treat the number as a bound.

The probe's convergence flag is a precondition, not a decoration. Convergence is arithmetic on the held-out curve, and a curve that never improved is **not** converged. The split is at **GUID** level, disjointness is asserted at runtime, and the fit budget is expressed in passes over the fit half rather than in optimizer steps.

### samples

This analysis selects segments for detailed visual inspection. The full page has fifteen rows and uses the task's existing `forecast_rows`, `forecast_extra_rows`, and `input_stream_panels` helpers, keeping the layout consistent with training diagnostics.

`stratified/` holds a seeded, shard-stratified draw over the whole split, so a cap at or above the shard count reaches every shard. `by_class/` holds a **class-balanced** draw: the same number of segments from every clinical class. The two are not interchangeable and neither replaces the other. The stratified quota follows shard size, so what it renders is what the split mostly *contains* — which on this cohort means healthy takes most of the pages, and two classes cannot be compared across it. The balanced draw is the one that supports a comparison and is, by construction, not representative of anything.

Beside them, one directory per headline metric and tail holds the segments at the extremes of that metric. A page is one segment of one recording: an illustration, never evidence — and the extreme pages are selected *on* the quantity they display, so the panel showing it is guaranteed to look unusual and says nothing about how often it does.

**Every selected segment is drawn twice**, from one forward pass. The full page is the fifteen-row one above; the reduced page beside it, named with a `_compact` tail, keeps five of those rows — the raw context, the target block as the encoder receives it, the latent state, $K_t$, and the lag attention on a logarithmic colour scale. It answers what a recording's latent and attention did, which is a different question from what the model predicted, and eight rows of forecast between them is what makes the full page slow to read for it.

`sample_pages.csv` carries one row per **file**, with a `variant` column naming which of the two it is, so the manifest indexes the whole directory rather than half of it.

`eval_config.caps.pages` overrides the stratified count and `eval_config.caps.pages_per_class` the balanced draw's per-class count; the extremes take ten per tail as an upper bound, lowered wherever a metric has too few scored segments. The two tails of one metric are disjoint by construction. The `<index>` in a filename is the position in the evaluation **dataset**, not in `per_sample.csv` — the collection pass runs under a seeded shuffle — and the two are reconciled by a `guid`/`epoch` round trip checked before anything is rendered.

### recording_traces

This analysis follows selected recordings through all their stored segments. A seeded, class-balanced draw selects up to `eval_config.caps.traces_per_class` recordings per clinical class, each with at least two segments. Every segment is reread in `epoch` order, and the model's outputs are saved at every decoded anchor. The traces show how an individual recording's latent state, divergence, and lag profile evolve. They are examples for inspection, not a sample used for population tests.

The outputs have three complementary forms:

- **Anchor table:** `anchor_trace.parquet` stores recomputed `kld_per_t`, coverage, argmax lag, attention entropy, latent scalars (`mu_prior_norm`, `delta_mu_norm`, mean log-variances, `prior_rate`, `n_active_dims`, and `kld_top_dim_share`), and `kl_lag_*`/`attn_lag_*` shape statistics. Forecast-gap estimators, block scores, and `seconds_since_contraction` are joined from `per_anchor.parquet` using `(guid, epoch, anchor)`.
- **Full vectors:** `<class>/<guid>_<subgroup>_full.npz` stores prior and posterior means and log-variances, per-coordinate and per-head divergence, lag KL attribution, and head-averaged attention. It includes the axes locating each anchor and segment means of the vectors.
- **Segment table and manifest:** `segment_summary.csv` averages scalars over scored anchors, recomputes shape statistics on the segment's mean profile, and records latent dispersion, changes from the previous segment, epoch gaps, and breaks. `recording_traces.csv` lists each selected recording, class, subgroup, dataset and collection segment counts, and its output files.

Recomputing shape statistics on the mean profile matches `lag_clocks`. Averaging statistics computed on individual profiles can give a different result.

**Only scored anchors enter a mean, and a segment that scored none is `NaN`.** The latent exists at every decoded anchor, including the ones the coverage floor rejected; averaging those in produces a state excursion that reads as a physiological event and is a gap. The full table keeps every decoded anchor with `contributing` beside it, so the choice is visible rather than made for the reader. The latent stored is the **mean**, never the sample: $z$ carries the reparameterisation noise, so two passes over one checkpoint would draw two different paths.

**The trace covers the recording as the dataset holds it, not as the pass collected it.** A segment a stratified `max_samples` cap left uncollected is traced and carries `NaN` in every joined column; `n_segments_collected` on the manifest says how many were. `max_hours_before_delivery` **is** applied when set: only the segments recorded within it are traced, counted for eligibility, or drawn from, so a bounded run's traces describe the population its clocks are read over; the plan records the bound and whether it was applied. Unset, the whole span is traced. The absolute axis is $t_{\mathrm{abs}} = \mathrm{epoch} + \Delta t$, the convention `trajectory` uses, so the two cannot disagree about where a recording's points are.

**The re-read forward is checked against the collected one.** The divergence is both recomputed and joined, and `kl_agreement_max_abs` in the record is the worst per-anchor disagreement between the two over every joined anchor; above `kl_agreement_tolerance` it is logged as a warning. Every batch is also checked against the rows it was built from before anything is reduced, for the reason the pages check theirs: the collection pass runs under a seeded shuffle, and a trace of the wrong recording is a plausible picture that nothing downstream would notice.

Like `samples`, `sufficiency` and `occlusion`, this analysis reaches for `context.task` and `context.loader`, and records a skip without them. A recording that fails is recorded by GUID with its error and the rest still trace. The figures are `recording_traces_summary.pdf` and one `<class>/<guid>_<subgroup>_trace.pdf` per recording — see `FIGURE_GUIDE.md`. Each recording also gets `<class>/<guid>_<subgroup>_trace.html`, the same trace as an interactive plotly page with the raw FHR and UA above it (`dashboard_file` in the manifest).

### attribution

This analysis uses gradients to estimate how input coefficients contribute to selected model outputs at a few anchors. It attributes a scalar output over the three stored input streams $(y^{st},y^{ph},u)$ at their declared widths. The outputs include divergence, predictive gain, and lag readouts. See [ATTRIBUTION.md](ATTRIBUTION.md) for the design and method comparisons; the details below describe what this evaluator saves.

**What is attributed.** A thin wrapper turns the dense forward into one scalar per sample at one anchor per sample: the divergence $K_t$; the **mean-decoded** block score of either branch and their gap (both branches decoded at $\mu$, no draw, so no reparameterisation noise enters any attributed number); one latent coordinate; and the model's own lag readout on a band — the head-averaged attention mass on the band here, the proposal norm on it in the lag-residual cell.

A production pass attributes `kld` and `pred_gap` under both baselines, the lag readout on every `occlusion_bands` band and the anchor's largest per-coordinate divergence under the source-null baseline, at `ANCHORS_PER_SEGMENT` anchors spread evenly over each segment's scored anchors.

**Baselines define the attribution comparison.** A standardised zero is the channel mean, and the warm-up gate already zeroes unavailable steps. The `source_null` baseline zeroes source values while holding target streams fixed; it measures input attribution relative to a zero-source response. The availability indicators are identical along the path, so they receive no input attribution. The `all_zero` baseline zeroes every stream and provides a reference for target-input attribution.

**Integration starts just above the exact baseline.** Per-step encoder normalisation can change sharply near an all-zero input. Tiny-model checks found a finite readout jump between path positions $\alpha=0$ and $\alpha=10^{-6}$, preventing convergence when integrating from exact zero. The implementation starts at $x_0=b+10^{-3}(x-b)$. At the documented step count, the completeness residual falls below $10^{-3}$ of the readout. Every row records $f(b)$, $f(x_0)$, and $f(x)$, including the entry jump $f(x_0)-f(b)$. That jump is reported separately and assigned to no input step.

**Which Captum methods run here, and which do not.** Integrated gradients is the primary method; layer integrated gradients on the head-structured posterior's per-head fusion modules gives a complete per-head split under the source-null baseline; feature ablation grouped by lag band of the source relative to the anchor is `occlusion`'s intervention read on this analysis's readouts and anchors. `InputXGradient`, `Saliency`, `GradientShap`, sliding-window `Occlusion`, `LayerConductance` and `NeuronConductance` run and were not shipped; `DeepLift` runs but its rescale rule reaches only the module nonlinearities it hooks, so its completeness residual is of the order of the readout. The block's `methods` record carries each verdict with its reason.

**Four checks validate each attribution row.** Future-step attribution must be zero (`checks.after_anchor_max_abs`), attribution to source steps excluded by warm-up must be zero (`checks.gated_off_max_abs`), and source attribution for the target-only base score must be zero (`checks.target_only`). The integrated-gradient sum must also reproduce $f(x)-f(x_0)$ within `checks.completeness_tolerance`; `checks.n_rows_over_tolerance` counts exceptions. These properties are tested on tiny models and measured on real outputs. The conv-LSTM model requires `causal_norm: true` for the future-step check; transformer models enforce step-wise causality by construction.

**Attribution is summarised per recording.** For each attributed anchor, the analysis sums over channels to make time profiles and over steps to make channel profiles. It aligns source attribution to lag using $q_\ell=p_{t_a-\ell}$, then compares $|q|$ with the model's lag readout using Pearson correlation and normalised Jensen–Shannon distance.

Frequency-band sums use the declared input-channel map, `band_channel_map.csv`, because input attributions have the declared width. Other reductions cover lag bands, heads, band-ablation changes, and the readouts at the input, exact baseline, and integration entry point.

`attribution_rows.csv` has one row per anchor, readout, and baseline. `attribution_vectors.npz` stores aligned profiles, and `attribution_recordings.csv` stores recording means used for grouped figures. The five summary tables are `attribution_summary.csv`, `attribution_bands.csv`, `attribution_lag_bands.csv`, `attribution_layer.csv`, and `attribution_null.csv`.

Where available, the lag-band table joins `occlusion_summary.csv`, and the target-band table joins `spectral_skill_bands.csv`. The occlusion gap comparison flips sign because occlusion reports the forecast cost of removal. Missing input files are recorded as absent dependencies.

**The null decomposition.** For the divergence under the source-null baseline, the readout at the exact null is the availability-clock part (`kld_source_null` at that anchor), the integrated attribution is the source-content part, and the entry jump is what the normalisation does between the two; under the all-zero baseline the attribution splits between the target and the source streams. `attribution_null.csv` carries the per-class means and `attribution_null.pdf` draws them.

**The trace.** One recording per class — the **most complete** one, ranked by how many of the segments the window could hold the dataset actually holds (the last `max_hours_before_delivery` hours when that key is set, the recording's own span otherwise, at the segment stride the geometry implies; ties on the segment count, then the identifier) — is attributed at the same anchors of every one of its segments, for both the divergence and the forecast gap, and drawn on the traces' shared figure: the lag-aligned attribution of each readout as a heatmap over hours before delivery beside the model's own lag readout, the agreement, the totals and the values. `attribution_traces.csv` is its manifest; the figures are `attribution/traces/<class>/<guid>_<subgroup>_attribution_trace.pdf`.

**The example pages.** One anchor per class — the middle chosen anchor of the class's first attributed segment — is attributed for the divergence, the forecast gap, the full-branch block score and the lag readout on every configured band, under both baselines, with the full maps kept: `attribution_maps.pdf` shows the input coefficients beside the attribution of $K_t$ for every class, and `attribution/maps/<class>_<guid>_<subgroup>_anchor<step>_attribution_maps.pdf` gives one page per class with every readout's maps; `attribution_examples.csv` lists them and `attribution_maps.npz` carries the arrays. Five population figures read the same rows from other sides: the per-channel profiles (`attribution_channels.pdf`), the lag-by-channel maps accumulated over attributed anchors (`attribution_lag_channel.pdf`, arrays in `attribution_lag_channel.npz`), the signed offset profiles (`attribution_time_profile.pdf`), the per-row numerical checks (`attribution_checks.pdf`) and the anchors on the delivery clock (`attribution_time_to_delivery.pdf`).

**The selection** is the traces' recording-level class-balanced seeded draw at an eligibility floor of one segment, one segment per drawn recording (its middle one by `epoch`), capped by `eval_config.caps.attribution_segments` — recording level rather than the pages' segment-level draw, because every summary here is over recordings and one segment per recording keeps every recording one unit. Absent, the cap means the analysis's own default rather than every segment: an attribution is tens of forwards and backwards per anchor, and the `cost` block records what the pass took so the cap is set from a measurement.

It is `capped`, reads `context.task` and `context.loader` like `samples`, `sufficiency`, `occlusion` and `recording_traces`, and records a skip without them.

**Interpretation.** An attribution is a sensitivity of a fitted computation along one path from one baseline, not a causal claim about the physiology; the lag axis is stored-coefficient time; a source attribution under the source-null baseline is relative to the availability clock, which it can never contain; the classes are out of distribution; and every summary is over recordings. `ATTRIBUTION.md` carries the full list.

### warmup

This analysis measures the effect of channel warm-up on the evaluated population and forecasts. Different channels become valid at different times, so the warm-up budget determines which channels and anchors can be used. Results are aggregated per recording with intervals.

**The gap by warm-up tertile.** `pred_gap_warm_lo`, `_mid` and `_hi` split the 98 kept channels into three tertiles by their rebased warm-up $W'$ and restrict the gap to each. The three recompose to `pred_gap` over the same denominator, and **the recomposition is asserted rather than described** — it is the only property that makes them a decomposition rather than three unrelated numbers. The tolerance is scaled by the block score rather than by the gap, because the gap is a difference of two block scores of order $10^3$ and a tolerance relative to the difference would tighten without limit as a model improved.

**The source-lag warmth fractions.** `source_lag_warmth_frac_st` and `_ph`: the attention mass landing on lags at which each stored source block is warm. **A small value here is the expected finding, not a fault**, and the emitted record says so: the source blocks' own warm-ups are long, so most of the searched lag window is a region where the source coefficient is still affected by warm-up, and a model that attends there is reading what the data offers rather than misbehaving.

Two geometry checks can fail. `target_warm_frac` must be exactly $1.0$, and `anchors_per_sample` must match the checkpoint's `anchor_ceiling - warmup_period` under dense evaluation. The expected count is computed from the run's horizon and forecast clock, rather than fixed to a historical example. A mismatch means the scored population does not match the declared geometry or the checkpoint predates the budget-and-floor checks.

Beside them, the warm-up staircase and the budget tradeoff curve, drawn from the model package's own `warmup_budget.py` and `causal_warmup.py` rather than re-derived.

### source_null

This analysis compares KL under the matched source with KL under a zeroed source. It helps separate the response to varying source values from the response that remains when only the source availability pattern and the model's zero-input behaviour are present.

The source availability pattern $m^u_{t,c}$ is a deterministic function of $t$, identical in every row of a batch, and it enters $q(z \mid Y, U)$ but not $p(z \mid Y)$ — so the posterior can be pushed off the prior by the availability **clock** alone, with no source information in it at all. The permutation control deranges rows, and no permutation of rows can remove something every row shares.

The analysis reports, per recording and bootstrapped over recordings,

$$\Delta_{\mathrm{clock}} = \texttt{source\_conditioned\_kl\_raw} - \texttt{kld\_source\_null},$$

the part of the coupling readout attributable to source variation. The null arm re-runs the source gate, adapter and encoder from a **zeroed** source stream — not a permutation, because both the adapter and the encoder are nonlinear and a zeroed stream is not a rearrangement of a real one.

**The threshold is set, at `clock_margin_min_nats: 0.15`, and both cfs cells' override files carry the same value and the same provenance comment.** It comes from the diagnosed unaligned run's observed spread of $\Delta_{\mathrm{clock}}$ across recordings — $0.160$, interval $[0.157, 0.164]$ — so `coupling_exceeds_availability_clock` decides rather than returning INCONCLUSIVE, and the acceptance gate is ten criteria rather than nine. Its provenance is the unaligned arm; the gated quantity is right on both arms, which is why one number serves both. It was deliberately left `null` until a run had measured the spread: a threshold guessed before then would have decided a FAIL on exactly the runs that were supposed to set it.

What was never conditional on the threshold is the number — `coupling_minus_clock_nats` is a headline scalar whatever the key says, so the arm tables carried it from the first run, which is what let the threshold be set from data.

**The null arm re-encodes through whichever module `lag_kv_source` selected**, not through a deep source encoder the model may not have built. That is what keeps it a control: it probes the tensor the lag attention actually reads.

Two things the emitted record states because each weakens the claim in the model's favour and nothing else would surface it. **Zeroing floors no source variation**, and the encoder's response to a flat trajectory is not literally the availability pattern's response — so $\Delta_{\mathrm{clock}}$ is a slightly *weaker* statement than "the clock alone". And **`kld_source_null` is not expected to collapse to zero even under `prior_availability_input`**: the posterior is a bounded residual on the prior, so the mean half of $\mathrm{KL}(q^\varnothing \Vert p)$ is a function of the delta head alone and no prior-side clock can appear in it. The informative quantity is therefore the difference and its interval, not the ratio.

**The same difference, resolved by lag.** The null arm already produces its own attention over the lags — the query is the prior's mean, unchanged, but the keys are the null encode's — so the head-structured attribution can be built for it from tensors the matched pass already had, at no extra forward. Subtracting it from the matched attribution bin by bin gives the **clock-excess** profile, and

$$\sum_\ell \Delta_\ell \;=\; \texttt{coupling\_minus\_clock},$$

which is the scalar `clock_margin_min_nats` gates. That identity is what makes this a decomposition of the gated quantity rather than a second lag reading that happens to have a clock subtracted, and it is **measured** on every run: `null_lag_map_sums_to_kl` is a third structural identity in the sanity block, checked separately from the matched one because the null arm's attention is its own.

This is the only lag profile in a run with the availability staircase removed. The staircase is a deterministic function of $t$ and is readable from the source state at *any* lag, so it enters the matched attribution wherever the attention happens to sit; no renormalisation of the matched profile removes it, and only an arm carrying the clock and no source content can.

`source_null_lag_profile.csv` carries one row per lag — both arms, their signed difference, its rectified part, the band each lag falls in and whether the delta mask kept it. The `lag` block adds the run-level selection: the clock-excess argmax and its peak share, the degeneracy verdict, the rectified fraction, the per-band shares, and the mask — or, where the profile is degenerate, no mask and the sentence saying why. Four of those reach the headline.

**Two things a reader must carry.** The profile is **signed**, so only the signed sum is the gated scalar and the rectified total is an *upper bound* on it, larger by exactly the negative mass; `rectified_frac` is that gap. And the delta mask is **withheld** whenever the clock-excess profile is degenerate — `entmax15` assigns lags exactly zero, so a flat profile still has a confident argmax, and a mask cut from one would name a band the run has no evidence for. A withheld mask is a measurement; the geometry-fixed `occlusion_bands` remain the selection that needs no estimate.

### occlusion

This analysis tests source use by removing values in selected lag bands and measuring the change in prediction. Read it alongside `lag_kl` and `attention`: those profiles describe attention-weighted quantities, while occlusion measures the fitted model's response to an input intervention.

For each band named in `occlusion_bands`, the stored source coefficients in $[t_a - \ell_{\mathrm{hi}},\ t_a - \ell_{\mathrm{lo}}]$ are set to **zero** — the channel mean, by the statistics convention every causal shard is built under — the stream is re-encoded through the run's own K/V path, and the block NLL is re-scored. The reported quantity is the per-horizon-step change against a reference forward in which nothing was removed.

**Four properties make the number readable, and each is a decision rather than an economy.**

- **The announcement is untouched.** The intervention moves the source's *values* and not its arrival clock, which is the exact confound the analysis exists to avoid; the invariance is **measured** on every occluded encode and reported, not asserted.
- **The band is occluded after the channel gate.** The gate shifts each channel onto the run's common clock, so a band of gated steps is one lag range for every kept channel at once. The same band applied before the gate would land at $\ell + d_c$ for channel $c$ and re-smear precisely the axis the alignment exists to un-smear.
- **One scored anchor per segment, drawn from a seeded generator over the anchors the forward marked valid, and held fixed across every band and the reference.** The source pathway has memory, so a band occluded relative to anchor $a$ contaminates the state of every anchor after it, and a second anchor scored in the same forward would attribute one anchor's loss to another's band. Holding the anchor fixed is what makes the difference *paired*.
- **Common random numbers.** The reference and every band are scored under the same reseeded latent noise, so the difference is not a draw apart. Repeated collection is bit-identical.

**The live fraction earns its own column.** A band reaching into the warm-up region, where the availability mechanism has already zeroed the source, has less source in it to remove — so a small delta there means "there was nothing there" rather than "the source did not matter". Four headline scalars therefore reach every arm table rather than one: the winning band's name, its delta, its peak horizon step and its live fraction.

**It has no verdict, deliberately.** What a healthy per-band delta is has never been measured, and a threshold guessed before the first production runs would decide a pass or a fail on exactly the run that was going to measure it — the same argument that kept `clock_margin_min_nats` unset until a run set it.

Three outputs: `occlusion_per_recording.csv`, `occlusion_per_horizon.csv` and `occlusion_summary.csv`, plus the per-horizon figure. `occlusion_bands` names the bands as `{name: [lo, hi]}` in **lag** units; an empty band (`lo > hi`) and a band above the model's own `max_lag` are both refused by name at config load, the first because a row of zeros would be reported as a finding and the second because it would name a wider band than it measured. `{}` — the schema default — records the analysis as a skip. `caps.occlusion` bounds how many **segments** are re-encoded rather than how many anchors are retained, because this analysis scores one anchor per segment; removing the key means every segment.

**Choosing `caps.occlusion` from a measurement rather than a guess.** The cap trades wall-clock against the per-band standard error, and both sides are properties of the machine and the checkpoint rather than of this pipeline — so the pass measures its own. Two blocks carry it: `cost` holds the rates, and every row of `bands` now holds `n_segments`, `n_recordings`, `delta_total_se` and `delta_total_ci_lo`/`delta_total_ci_hi`.

1. Run once against the real checkpoint with `caps: {occlusion: 64}`.
2. Read `results.occlusion.cost.hours_per_1000_samples` and `results.occlusion.bands[*].delta_total_se`.
3. Take whichever of the two binds:
   * **time** — `cap_time = target_hours * 1000 / hours_per_1000_samples`;
   * **precision** — `cap_precision = 64 * (se_observed / se_wanted)^2`, because the standard error falls as $1/\sqrt{n}$, so halving it costs four times the segments.
4. Set the result in the committed override delta, **never** in `RUN_ARGS`: a value injected from Python appears in no artifact and cannot be recovered from the run afterwards.

`cost.seconds_per_arm_per_segment` is the rate to extrapolate with whenever the **band count** changes, because the work is one encode and one single-anchor decode per *arm* and an arm is the reference plus one per band; `hours_per_1000_samples` alone would make a band count look like a property of the dataset. The two counts beside it differ in unit and both are needed: the interval is taken over **recordings**, for the reason `bootstrap_resamples` states for every other interval here, while the cap bounds **segments**, because segments are what the loop consumes.

**The deltas are also placed on the two clinical clocks.** `occlusion_clocks.csv` carries, per band and per window, the mean per-recording delta with its quartiles — the interventional answer to "did the informative past move", against `lag_kld_scaled`'s observational one on the same partition and the same grid.

The clinical coordinates come from a **join** onto the collected per-sample table on `(guid, epoch)`, not from a second read off the batch. `guid` alone does not identify a segment — a recording contributes many, which is why the collection pass keys its per-anchor table on `(guid, epoch, anchor)` — and joining picks up the class, the subgroup and the second-stage offset at once while guaranteeing this analysis cannot disagree with any other about which class a segment belongs to. Both sides read `epoch` through `metrics.batch_field` and the same `float64` cast, so the equality is exact; `clocks.n_unjoined` is the tripwire and is zero on a healthy run.

**That page is descriptive only** — no Kruskal-Wallis, no Holm correction, no new family. One anchor per segment and a cap in segments means a half-hour window holds tens at best, and most (class, window) cells fall below the minimum group size a test needs; a $p$-value there would be a correction over cells that mostly could not be tested. Raising `caps.occlusion` is what makes those cells readable, which is the second reason the cap procedure above matters.

Like `samples` and `sufficiency`, this analysis reaches for `context.task` and `context.loader`, and records a skip without them. That is structural rather than a convenience: an intervention on the model's *input* cannot be served by any table, because the tables record a forward the source was fully present in.

### lag_clocks

This analysis follows the lag profile through labour on both clinical clocks. It asks whether the location and shape of the profile change over time or differ by class. Read it beside `time_to_delivery` and `second_stage`, which measure the amount of predictive gain and KL in the same windows.

**Fourteen attributes** of each segment's own profile, on the compensated axis, with $p_\ell = w_\ell / \sum_k w_k$, in four families:

| family | statistics | column |
| --- | --- | --- |
| moments | centre of mass $\bar\tau = \sum_\ell p_\ell \tau_\ell$, spread $\sigma_\tau = \sqrt{\sum_\ell p_\ell (\tau_\ell - \bar\tau)^2}$, skewness | `lag_centroid_*_s`, `lag_spread_*_s`, `lag_skewness_*` |
| quantiles | median lag, inter-quartile range | `lag_median_*_s`, `lag_iqr_*_s` |
| concentration | entropy $H = -\sum_\ell p_\ell \log p_\ell$, effective support $\Delta e^{H}$, near and far mass share | `lag_entropy_*_nats`, `lag_effective_support_*_s`, `lag_near_mass_*`, `lag_far_mass_*` |
| peak | peak lag, its width at half height, its mass, the degeneracy flag and the zero fraction behind it | `lag_peak_*_s`, `lag_peak_width_*_s`, `lag_peak_mass_*`, `lag_peak_degenerate_*`, `lag_zero_fraction_*` |

The families exist because each answers what the others cannot: a bimodal profile has an unremarkable centroid and an unremarkable spread, and only the concentration family says it is not one lump; these profiles are skewed, and one distant bin moves $\bar\tau$ far more than it moves the median, so where the two disagree the disagreement *is* the skew. Every one is computed twice, over the untruncated KL attribution and over the support-corrected attention — 28 columns — for the reason both clocks carry two coupling readouts: the attribution is $K_t$ times the attention and inherits the prior-variance inflation the attention is immune to, so a shift visible in one and absent from the other is a finding about which is being read.

The arithmetic is `eval/lag_shape.py`'s, in one vectorised pass per profile.

**The peak is reported, and it is reported with its guard.** `entmax15` assigns lags exactly zero, so a flat or nearly empty profile still has a perfectly confident argmax, and a position quoted without the mechanical criterion that says whether the profile has a shape at all is not a reading. That criterion used to live in `lag_kl` — which an analysis may not import — and this analysis therefore reported no peak.

It now lives one layer down in `eval/lag_shape.py`, so `lag_peak_*_s` travels beside `lag_peak_degenerate_*` in the same row of the same table and on the same page: a segment is degenerate when its peak-to-median ratio is below `1.1` or more than `90%` of its finite bins are exactly zero, and the per-recording mean of that flag is the share of a window's segments whose peak names a bin rather than a lag. `lag_kl/lag_kl_stratified_peaks.csv` remains where the *pooled* positional reading lives; this is the per-segment one.

Both clocks are cut on the same `TRAJECTORY_BIN_HOURS` grid the coupling clocks use, the unit is one value per **recording** inside a window as well as across it, and the second clock scores the recordings that carry an onset only — the same eligibility rule `second_stage` applies, whose per-recording table this analysis carries counts from rather than rewriting. It declares itself `capped` for that reason.

**Four Holm families, and none of them joint**: two clocks times two tested readouts, each correction controlling the family-wise error rate within its own clock and its own readout. A reader quoting a window from two of them is making two comparisons and the `method` string of each says so. **Only the two centroids are tested.** The other twelve statistics are drawn and tabled but carry no $p$-value, which is what keeps each family at two rather than at fourteen and the tested page at five rows rather than twenty-nine; a trajectory on the features page that looks separated is a hypothesis, not a claim.

Promoting one is a single `tested` flag in `STATISTICS`.

It emits **six** figures, three per clock: `lag_<clock>.pdf`, the share of the attribution by lag and window with one panel per class on a shared colour scale and the two tested centroid trajectories beneath it; `lag_<clock>_windows.pdf`, the violins, the Holm-adjusted $p$ per window and Cliff's delta for every class pair that survived; and `lag_<clock>_features.pdf`, the untested statistics one panel each, solid for the attribution and dashed for the attention. The third is its own page rather than more rows on the first because half of what it draws is not in seconds, and a panel sharing a figure with a quantity in different units is a panel that will be read against it.

### lag_kld_scaled

**The same lag structure, read on the lags that carry the coupling and with its magnitude kept.** `lag_clocks` resolves the profile against both clocks over **all** $L$ lags and through statistics that are functions of $p_\ell = w_\ell / \sum_k w_k$ alone. On this family both are limitations rather than conventions: two thirds of the attribution is an availability clock readable at every lag, the scale that would distinguish "the informative past moved" from "there is less of it" is divided out, and the heads are averaged into one profile that one latent group dominates.

This analysis is those three answers, on the same $0.5$ h grid, **beside** `lag_clocks` rather than replacing it — that analysis's columns are untouched.

**Four families of source, and the first two are the selection.**

- **The geometry-fixed bands**, from `occlusion_bands`. Nothing about them is estimated from the KL, so a statistic on a band is free of the circularity that makes a top-$K$-by-KL selection test its own selector. They are also the *same* partition `occlusion` removes source from, so a band names one lag range across the run and the observational and interventional pages are read against each other by filtering rather than by aligning two four-way splits by eye.
- **The soft weight**, $\omega_\ell = \Delta^+_\ell / \max_k \Delta^+_k$, from the pooled clock-excess profile `source_null` reports. Computed **once at run level** and applied identically to every segment, window and class — a per-segment weight would let each segment choose its own lag axis, and a comparison across segments would then compare different axes. Withheld entirely when that profile is degenerate.
- **The full support**, carrying `total_nats` and `peak_nats` only. The twelve scale-free statistics on the full support are `lag_clocks`' own columns and are not restated here.
- **The heads**, each head's own $K^{(m)}\alpha^{(m)}_\ell$, which sums over $m$ to the pooled attribution exactly.

**`near_mass` and `far_mass` are absent from every banded source, and the absence is a measurement.** Both are measured from the axis's own start, so on a band they would silently mean "within `NEAR_SECONDS` of *the band's* start", and `far_mass` would be identically zero on any band narrower than `FAR_SECONDS` — three of the four shipped ones. Four columns of structural zeros presented as measurements is worse than four absent columns.

**Nothing here is tested.** Every feature ships untested, so this analysis adds **no** Holm family to the four `lag_clocks` carries, and it writes no significance or pairwise table at all. At the clock-exceeding coupling this family has measured — $0.160$ nats over $91$ lags on the diagnosed run — per-segment restricted centroids are very likely noise, and correcting eight new families over noise is how a family-wise correction stops being believed. The record says so in `no_inference_note` rather than leaving a reader to infer that a $p$-value was withheld; promoting a feature is one flag.

**The emission is long-form**: `source` and `statistic` are row keys, not columns. That is what lets `num_heads` be a run property and a band be added without widening a table.

Three outputs — `lag_kld_scaled_per_recording.csv`, `lag_kld_scaled_trajectory.csv` and `lag_kld_scaled_selection.csv` — plus one figure per clock. The selection table is the run's durable record of which lags were kept and with what weight: a selection reconstructed later from a re-run is not the selection the numbers beside it were chosen with.

`occlusion_bands` is therefore read by **two** analyses. Emptying it to skip the interventional pass also removes this analysis's selection, and this one records a named skip rather than silently emitting its unrestricted half.

### lag_high_kl

This analysis examines lag profiles at anchors selected for large KL. Averaging over all anchors can hide differences between high- and low-KL periods. The selection identifies strong latent changes; whether those changes improve prediction is tested separately in the usefulness analysis below.

A single set of KL thresholds is computed from all scored anchors across classes within the run's time scope. The same thresholds are applied to every segment, window, and class. Three fixed anchor groups are defined: `high`, the upper $30\%$; `rest`, its complement; and `top`, the upper $10\%$. These quantiles are module constants.

For each segment and group, the analysis reports the selected-anchor fraction and profiles of KL attribution and head-averaged attention restricted to those anchors. The shared `lag_shape.py` functions compute centroid, spread, median, IQR, entropy, effective support, near/far mass, guarded peak statistics, and nats-scale totals. Results are shown by class on both clinical clocks using half-hour windows.

This analysis requires `per_anchor_vectors.npz`, which stores pooled KL attribution and head-averaged attention at every contributing anchor in `float16`, aligned row for row with `per_anchor.parquet`. Segment-averaged profiles cannot recover the selected-anchor results, and an argmax alone cannot describe profile shape. Older result directories without the sidecar record a skip; run collection again to create it.

**Two primary readouts are tested on each clock:** `high_lag_centroid_kl_s` and `high_anchor_frac`. Each uses a Kruskal–Wallis class comparison within windows, Holm correction across that clock's windows, and pairwise Mann–Whitney tests with Cliff's delta for surviving comparisons. This gives four separate Holm families.

The analysis also tests three histogram-shape features per clock, described below, and runs one paired usefulness test. Other outputs are descriptive, including trajectories for `rest`, `top`, and `gain`; attention-profile statistics; hot-lag shares; decile, argmax, and occlusion-join tables; and contraction enrichment.

**Three further readings come from the same selection.**

- **Hot lags.** The lags whose *pooled* attribution — over every anchor of the population — sits in the upper $30\%$ across the $91$ lags. A run-level set, recorded lag by lag in `lag_high_kl_selection.csv`, and the per-segment share of attribution landing on it is placed on both clocks beside the band readouts. **This is the top-$K$-by-KL selection `lag_kld_scaled` declines for its own bands, taken here deliberately and with the circularity stated on every artifact**: the set is chosen from the same attribution it then summarises, so a share on it describes the run's own selection and is not an independent test of it. The selection is pooled over every class, which is what keeps a *class contrast* on it honest — no class chose its own lags. The geometry-fixed bands and the occlusion readout remain the selections that need no estimate.
- **Where the KL sits on the lag axis, by KL magnitude.** The per-anchor `argmax_lag` against the KL decile of the same anchor, pooled and per class (`lag_high_kl_argmax_by_quantile.csv`). A flat picture across deciles says the argmax is a property of the geometry rather than of the coupling; a picture that moves says which lags the coupling actually lives at.
- **Contraction enrichment.** Whether high-KL anchors are more common within `event_lag_window_s` of a detected contraction than outside it, per recording (`lag_high_kl_contraction.csv`) and summarised by class — the coupling-magnitude counterpart of `events`, on the same per-anchor contraction age and with no extra pass. A recording's difference is reported only with at least five anchors in **each** arm; below that a share is a coin toss and the row says `reportable = False`.

The histogram outputs compare the full selected-anchor lag distributions, rather than only their centroids or shares. They cover the `high` and `top` groups by class and time window on both clinical clocks.

Histogram aggregation gives every recording equal weight. First average selected-anchor lag maps within a recording, then normalise that recording's profile to sum to one, then average these distributions within each class and window.

This differs from `lag_high_kl_profile.csv`, which normalises after averaging at the class/window level. The histogram describes a typical recording's relative distribution; the profile table describes the location of pooled coupling mass. Both attention (`attn`) and KL-weighted (`kl`) versions are saved. Attention weights each selected timestep equally, while KL weights it by the size of the latent change.

Seven tables and six figures:

- `lag_high_kl_histogram.csv` — one row per (clock, band, source, class, window, lag): the cell's mean density and its inter-quartile range over recordings. Each cell sums to one across the lags.
- `lag_high_kl_histogram_features.csv` — the same shape vocabulary of `lag_shape.py`, taken of each **recording's own histogram** rather than of each segment's profile with the scalars then averaged; the mean of a centroid is not the centroid of the mean, and this is the one that describes the object the figure draws. `total_nats` and `peak_nats` are omitted: the first is identically $1$ on a normalised histogram and the second is a share the column suffix would spell in nats.
- `lag_high_kl_histogram_distance.csv` — how far apart two cells are, two ways, because neither subsumes the other. **Jensen–Shannon distance** (base $2$, bounded by $1$) is blind to the axis and reads as overlap; **$1$-Wasserstein in seconds** on the compensated axis reads as "the distribution moved this far" and keeps measuring once two supports separate, where Jensen–Shannon has already saturated. A signed centroid difference travels beside them, oriented worst class first, because both distances are non-negative and neither says which way. Two comparisons: every class pair within a window, and every window against its own class pooled over the whole clock — the pooled cell rather than the first window, because "first" means opposite things on the two clocks and a reference defined by window order would silently differ between them.

- `lag_high_kl_histogram_significance.csv` and `lag_high_kl_histogram_pairwise.csv` — the tested shape features, in the shape of `lag_high_kl_significance.csv` and `lag_high_kl_pairwise.csv` with the band and source as columns. **Three features of the `high` band's `kl` histogram are tested, per window on both clocks**: the median lag `hist_median_s`, the inter-quartile lag range `hist_iqr_s` and the entropy `hist_entropy_nats` — one from each of the three scale-free families `lag_shape.py` names (where, how wide, how concentrated), so each way two distributions can differ is asked once. The centroid is deliberately not among them, because `high_lag_centroid_kl_s` already tests the position of the same selection and a second family on it would ask one question twice. Six Holm families, one per (clock, feature), each across that clock's windows, none joint with the four above.
- `lag_high_kl_histogram_drift.csv` and `lag_high_kl_histogram_drift_summary.csv` — the same three features fitted **within each recording** along the clock: the least-squares slope of the feature against forward labour time (the delivery clock's centres negated, so a positive slope means the feature rises as delivery approaches on either clock) over every recording scored in at least three windows, beside its last-minus-first difference. The per-window tests compare different recordings in every window, so a moving class median can be a changing population; the slope is the reading that cannot be. Per clock, two further Holm families across the three features: each (feature, class) slope against zero by Wilcoxon signed-rank, and each feature's Kruskal–Wallis across classes, with pairwise Cliff's delta on the survivors.
- `lag_high_kl_subgroup_histogram.csv` — the `high` band's lag distribution pooled over the **whole** evaluated population, no clock and no window, by clinical class **and by subgroup**: each recording's selected-anchor profile averaged over every one of its segments, normalised once, then averaged over the recordings of the cohort. The one table that asks the eight-cohort question of the lag structure; descriptive, no test. Drawn on `lag_high_kl_subgroup_histogram.pdf`, nested by class exactly as the `distributions` pages are.

**The cells themselves, both distances, every feature on the `attn` source and the `top` band, and every untested feature ship untested** and the record says so. A distance between two *estimated* distributions is positive almost surely even when the two populations coincide, so a value there describes two cells rather than showing that they differ; the recording counts travel on every row, and a cell below the shared minimum of three recordings is emitted with its counts and a `NaN` distance rather than the zero that would read as agreement.

**No peak-lag histogram ships**: the per-anchor argmax is already resolved by KL decile above, and on a flat profile NumPy's first-maximum rule pins it at lag $0$, so a histogram of it would be a picture of that rule.

**Usefulness is assessed through prediction.** A large $K_t$ means that the latent changed, but does not establish a better forecast. The analysis uses per-anchor gain $D_{\mathrm{base}}-D_{\mathrm{full}}$, preferring `mean_pred_gap`, then `mc_pred_gap`, then `pred_gap` if the earlier columns are unavailable. The selected column is recorded under `usefulness.gain_column`. Four comparisons follow:

- **Per band, the mean gain of its anchors** — `high_pred_gap_nats`, `rest_pred_gap_nats`, … — per segment, per recording and on both clocks. The high band's against the rest band's is tested **once, paired within recording** by a Wilcoxon signed-rank test over recordings with a bootstrap interval on the mean difference; it is its own family of one and is not corrected with the four clock families. Positive means the anchors carrying the coupling are the anchors where the source bought forecast; zero or negative means the KL is not where the usefulness is.
- **A fourth band, `gain`** — the anchors in the upper $30\%$ of the pooled forecast gain, selected on usefulness rather than on KL — with its own lag profile and statistics, and its **overlap with the high band** against the $30\%$ that independence would give (`share_of_high_in_gain`, Jaccard). Two selections naming the same anchors is the finding; two that do not is the other one.
- **The gain resolved by KL decile** (`lag_high_kl_gain_by_kl_quantile.csv`, per recording then by class) **and by the anchor's argmax lag** (`lag_high_kl_gain_by_argmax.csv`, pooled and per band): whether more coupling buys more forecast, and whether the lag the attribution names is a lag the forecast profits from.
- **A gain-weighted attention profile**, $\sum_t \max(g_t, 0)\,\alpha_{t\ell} / \sum_t \max(g_t, 0)$ — where the source looks *when it helps* — beside the KL-weighted one on the selection table.

**Observational against interventional, on one partition.** When the `occlusion` analysis has run in the directory, `lag_high_kl_occlusion_consistency.csv` joins, per recording and per geometry band of `occlusion_bands`, the share of the recording's KL attribution inside the band (all anchors, and the high band's) with the forecast cost of occluding that band, and records a descriptive Spearman's $\rho$ per band. Positive means the lags the attribution names are the lags the forecast used; near zero means the two readings disagree about where the source mattered. The dependency is on the file, so `--only lag_high_kl` against a directory whose interventional pass never ran records a skip.

**Eight headline scalars reach every arm table**: the pooled high threshold in nats, the high band's centroid and total nats as means over recordings, the hot-lag count and the hot-lag share; and the usefulness three — the high band's mean forecast gain, its paired difference against the rest band's, and the high–gain overlap share. The threshold comes first because every other number is conditional on it — two arms with different thresholds selected different anchors. `lag_high_kl_recordings.csv`, one row per recording over the whole population, is the source `cross_subgroup` reads `high_anchor_frac` from.

**It is `capped`**, for the reason `lag_clocks` is: the second-stage half scores the recordings that carry an onset only, by the shared eligibility rule. Both clocks' per-recording and trajectory tables, the per-window restricted profiles, the seven histogram tables, the significance and pairwise tables, and twelve figures — a run-level selection page, a run-level usefulness page and, per clock, a profile-and-trajectory page, a tested page, a lag-distribution page, a tested page for the histogram's shape features and a within-recording drift page — are the outputs. The axis is stored-coefficient time and every one of them carries the caveat.

### spectral_skill

This analysis groups forecast performance by the frequency band of the target's analysing filter. A first-order scattering coefficient, $|x\star\psi_\lambda|$, is the magnitude of a filtered signal. Its filter already identifies a frequency range, allowing performance to be grouped by channel without estimating a new spectrum.

What is reported is how well the model forecasts the envelope in each clinical band, per recording and bootstrapped over recordings, in both the likelihood space the objective is stated in and the error space that has a natural zero. The band gaps recompose to `pred_gap` under the same guard the warm-up tertiles use.

**It is band-resolved skill, not coherence, and the difference is not a technicality.** A stored scattering coefficient is a *modulus*: the analysing filter's phase was discarded before the value was written. So the three things the raw pipeline's `coherence` exists to separate — phase agreement, group delay, and the exact three-way split of the residual spectrum into irreducible, timing and amplitude terms — have **no analogue here at any window length**. What this readout says is how well the forecast reproduces each band; what it cannot say is whether a forecast is mistimed rather than mis-scaled.

It is named `spectral_skill` and not `coherence` so that a reader who knows the raw pipeline cannot carry the wrong contract across.

Two further limits belong beside it. The band is the band of the **analysing filter**, not a bin of the forecast's own spectrum. And a phase-harmonic channel has a *pair* of frequencies; it is banded by `band_partition`'s own `freq_hz_primary` convention, which the emitted record states rather than assumes.

The frequency-band join must use the retained target axis. `band_partition` describes declared channels, while forecast-gap vectors contain only retained channels. Joining by position can assign a score to the wrong band when dropped channels occur within the axis. The analysis therefore reads `band_channel_map_kept.csv` from disk and joins through that persisted map. It does not access `model.target_gate`, preserving offline operation and the analysis import rules.

Read all five channel counts in the record: `declared_total`, `dropped_declared`, `kept_total`, `known_kept`, and `unknown_kept`. In the legacy $102$-channel example they are $102$, $4$, $98$, $95$, and $3$, respectively. Report known-band coverage relative to the retained target channels actually scored. Counts depend on the checkpoint and dataset.

### cross_subgroup

This analysis tests whether recording-level metrics differ across clinical classes or subgroups. Visual differences alone are insufficient: with many groups and metrics, some means will look separated by chance.

Three layers, in order, and the order is the point: a Kruskal omnibus per metric, Holm **across metrics as one family**, and pairwise Mann–Whitney with Cliff's delta (Romano magnitudes) on the survivors only. Every pair is named more severe first — HIE vs acidosis, HIE vs healthy, acidosis vs healthy — so a positive $\delta$ means the more severe cohort's values run higher. Every test consumes one value per **recording**, and a test asserts that no source names a `per_sample` file.

It reads finished per-recording CSVs off disk through a `METRIC_SOURCES` table — which here gains this cell's own sources, the warm-up tertiles, the source-null difference and the band-resolved skill — so a missing source is **recorded** rather than raised, which is what keeps `--only cross_subgroup` working against a finished directory with no checkpoint. It self-skips below two testable groups.

## How the output is misread

Use the following limits when drawing conclusions. They explain what each measurement supports, which comparisons require extra checks, and why similar-looking metrics can answer different questions.

**The forecast claim is exact, and the lag claim is not.** The stored coefficients come from a strictly one-sided bank, so a coefficient at step $t$ is a function of $\{x(s) : s \le t\}$ alone and forecasting step $t + 1 + \tau$ from history up to $t$ is a genuine forecast. That is what separates this cell from the four two-sided ones and it needs no hedging. What still needs hedging is the *lag* readout: the coupling number is named `source_conditioned_kl_raw` and the disclosure refuses the name it is not, because the lag map is an attribution over **stored-coefficient time**, uncorrected for a composed one-sided group delay reaching 791 s — the same order as the 364 s lag search itself.

Every run carries that sentence verbatim in `preflight.json` and `summary.json`, every lag-resolved artifact and figure carries the caveat, and `tests/test_eval_naming.py` scans the whole artifact tree — plus this file and the figure guide — for the name the readout refuses.

**A lag position is a coefficient-time attribution, not a physiological delay.** The compensated lag $\tau = 4(\ell + \delta)$ corrects only the model's own input delay $\delta$, read from `model.source_delay_steps` and nowhere else. The per-channel composed group delay is *not* corrected for, and cannot be from this readout: the correction is per channel **pair** while the lag map is per head over a pooled source state, so the mapping would itself be an unvalidated construction. Both `DESIGN.md` records keep that limitation open.

What the dual alignment reference *does* remove is the inter-stream part of that bias: with both streams on their own clocks the residual between them is a single known constant, printed on the console block beside the delay.

**`argmax_lag` at the smallest attainable lag is a censoring reading, not an inertness reading.** The lag window has two censoring edges and not one. A profile pinned at the **far** edge means the model would report a lag the window is too short to express, and that is a FAIL; a profile pinned at the **near** edge means it would report a lag *shorter* than the window's own arithmetic can express, which at this geometry is where any delay below roughly $30$ s lands — and that is INCONCLUSIVE, with the physical-lag identity stated in the message so a reader can check the arithmetic rather than trust the verdict.

The near edge is `min(attainable)` read from the per-lag anchor counts, symmetric with the far edge, so a window whose lowest bins carry no anchor lifts the floor off zero rather than being read as inertness.

**Whether the machinery is alive is judged from the shape vocabulary, not from the argmax.** A degenerate profile — one whose peak is not distinguishable from its bulk — FAILs at either edge or in the middle, because its argmax names a bin rather than a lag; that is decided first, from `lag_shape`'s degeneracy flag, the peak's width and the mass above half the peak, with the per-head entropies beside them. An ideal model at this geometry, peaking strictly inside the attainable range, PASSes.

**The pooled argmax is not the surface an arm comparison is read on**: the per-head profiles are, and they are printed under the pooled row with each head's argmax, peak width, mass above half peak, near and far mass, attention entropy and KL share.

**The run's arm is printed beside the delay, and in three readings rather than one.** The console block and `summary.run_arm` carry the *configured* `causal_align_reference` label and source reference, the *built* `lag_kv_source`, and the *resolved* target clock, source clock and inter-stream offset in seconds. Three readings because a config naming one arm while the checkpoint carries another is exactly the failure this line exists to catch, and merging them would hide it.

**Specificity is read in prediction space, not in KL space.** See `perm_control`: $K_{\mathrm{shuffled}} > K_{\mathrm{true}}$ is what a healthy model does, so a KL-space criterion would fail exactly the models it should pass.

**A coupling readout is not yet a source finding: read `source_null` first.** The availability clock is a hazard the permutation control structurally cannot see, and the difference between the two readouts is the part of the coupling attributable to source *variation*. Until `coupling_exceeds_availability_clock` has a threshold it reports INCONCLUSIVE — which means the measurement is there and the criterion is not, and the number rather than the status is what to read.

**Only the unfloored KL may be read as a rate.** `source_conditioned_kl_train` has free bits applied per dimension per step before summing, so it exceeds the raw value by construction and hides a collapsed source pathway. The shipped `free_bits: 0.0` makes the two coincide today, which is exactly why the distinction lives in code: no headline path may resolve to it, asserted by test.

**Only an unpinned prior variance makes that rate meaningful.** A prior variance on its clamp inflates every coupling number while every decoder-side diagnostic stays healthy. Read `prior_variance_not_pinned` before quoting the KL.

**A small source-lag warmth is the expected finding rather than a fault.** The stored source blocks warm up late — `up_ph` not before step 56 on the committed fixture — so much of the searched lag window is a region in which the source coefficient is still affected by warm-up. `warmup`'s emitted record says so beside the number, and a reader who treats a low fraction as a defect is reading the dataset's geometry as the model's behaviour.

**The percentage is budget-local, and the nats are one step removed from it.** `pred_gap_mc_likelihood_pct` divides by $H \cdot C_{\mathrm{keep}}$, and $C_{\mathrm{keep}}$ is what the warm-up budget decides — so two arms of this model at two budgets have non-comparable percentages as well as non-comparable nats, and mutually unloadable checkpoints besides. Nothing tries to normalise it away.

**A `nll_*_sample` key is a fixed /2940 rescale of a block score**, not a mean over the coefficients that were actually scored, so on any anchor with masked forecast steps it under-reports. It ships beside that statement or not at all.

**The percentage is never `pred_gap` divided by a block score.** $D_{\mathrm{base}}$ is a negative log *density* summed over 2940 coefficients: it has no natural zero, it is legitimately negative for a sharp forecast, and the ratio therefore changes sign with its own denominator. The percentages this pipeline emits live in the two spaces that have a natural zero — error space and likelihood space.

**Anchors are not independent, and every statistic is per recording.** Consecutive anchors' forecast windows overlap in $H - 1$ of their $H$ horizon steps and one GUID contributes many segments, so per-segment $p$-values are anticonservative by that factor. The chain is: per anchor → support-weighted mean within a segment → unweighted mean over a GUID's segments → across GUIDs. A segment scoring zero anchors is excluded and **counted**, never averaged in as `0.0`.

**A per-segment histogram is a description, not a cohort difference.** `distributions` computes no test on purpose; a separation visible there is a reason to look at `cross_subgroup`, which answers the question on per-recording values.

**Every class contrast is out-of-distribution, and the scope is wider than it looks.** The checkpoint trains on healthy-**with-background** only, so ACIDOSIS and HIE are unseen — and so are the `healthy_no_bg_cs` and `healthy_no_bg_no_cs` subgroups. The summary computes `training_cohort_disjoint` from both resolved dataset lists rather than asserting it, reports `null` rather than `false` where a list is absent, and suppresses the out-of-distribution sentence when the two overlap.

**An eval score is not comparable with a `test_*` metric logged during training**, and a loss level here is **not comparable** across the target axis either. Against `lag_attn_fs` the blocks differ (2940 against 2340 coefficients), though the horizons no longer differ; against `lag_attn_transformer_cfs` it *is* comparable, because both cells sum the same 2940 coefficients over the same anchor count under the same objective. The cross-cell table carries only the second comparison for that reason.

**The sufficiency gap is an estimate, not a bound.** Both bias directions, above.

**The frequency statement has no timing half.** `spectral_skill` says how well each band is reproduced and cannot say whether a forecast arrives a step late; a forecast that is right in every band but mistimed reads here as a forecast that is right. Reviving the raw pipeline's construction is not the fix and is recorded so nobody tries: a $\tau$-slice on this grid gives 136 samples at 0.25 Hz per channel, Nyquist $0.125$ Hz, over band-limited envelopes rather than one trace — and the phase the estimator needs was discarded before the coefficient was stored.

**A cohort's colour here is not its colour on a training figure.** See *Cohort order and colour*.

**There are two clinical clocks and two independent families, and they are corrected within a clock rather than across.** `time_to_delivery` resolves the coupling against delivery and `second_stage` resolves it against the onset of the second stage; each runs its own Holm step-down across its own windows. That is deliberate: the two are different alignments of an *overlapping* population, so a window significant on one clock and not the other is a statement about alignment rather than a contradiction — and a reader who quotes a claim from both clocks has made two comparisons and is corrected for neither against the other.

Within one clock the two readouts are not jointly corrected either, because they are two readings of the same recordings rather than two hypotheses.

**`second_stage` scores a subset of the evaluated cohort, which is why it is `capped`.** A recording the labour-onset table has no second stage for cannot be placed on that axis at all, so it is excluded and counted; the analysis reports the eligible segment count with `plan.capped = True` and its reason, and the coverage block therefore leaves it out of the population comparison instead of reporting it as two analyses disagreeing about who was evaluated. Its `n_samples` is **not** comparable with any other analysis's, and `second_stage_eligibility.csv` is where the difference is accounted for, recording by recording.

**A stored onset that is wrong is counted and never dropped.** Two of them are measurable and both are reported rather than filtered: an implied onset falling *at delivery*, which is what a pipeline writes when it substitutes zero for a missing time, and an implied onset that *moves* across a recording's own segments by more than the 1 s float32 tolerance, which can only come from a broken write. Excluding those recordings would change the population every number on that clock is computed over while the numbers themselves went on looking ordinary; a count does not.

If a later reading shows they distort the trajectory, the eligibility rule is one predicate away — and the count is what would say so.

## The divergence register

The fork's third anti-drift measure, and the one that makes the other two auditable. Every module of `teb_vae/lag_attn_rws/eval` has exactly one entry in the committed `divergences.json` beside this file, classified `equivalent` (must stay behaviour-equivalent, and at least one named assertion in `tests/test_eval_sibling_agreement.py` exercises it), `divergent` (deliberately differs, with the reason recorded) or `absent` (not ported at all, and the file is asserted **not** to exist here). A module that is neither classified nor absent fails a test — which is the case a prose register cannot catch, because nobody notices a paragraph that was never written.

**The list below is rendered from that file rather than kept by hand**, and `tests/test_eval_docs.py` asserts the two agree verbatim.

- **`__init__.py`** (*divergent*) States that this package is a fork of the raw pipeline, why the four-field ModelBinding did not reach a target-domain change, and the four anti-drift measures that travel with the fork. The sibling's docstring describes a pipeline that is nobody's fork.
- **`_reuse.py`** (*equivalent*) Behaviour-equivalent to the sibling module, exercised by `test_eval_reuse.py::test_the_bound_names_are_exactly_the_siblings`, `test_eval_reuse.py::test_every_bound_name_resolves_to_the_same_object_both_packages_see`.
- **`analyses/__init__.py`** (*equivalent*) Behaviour-equivalent to the sibling module, exercised by `test_eval_sibling_agreement.py::test_the_analysis_protocol_is_the_same_contract_in_both_packages`.
- **`binding.py`** (*divergent*) The ModelBinding dataclass gains a `collect` field naming a model's own collection pass (the shared pass is written against the lag-attention forward, and a model without one supplies the pass that writes the same tables from its own), and this copy also carries the concrete CFS_BINDING and its GEOMETRY_KEYS -- sixteen against the sibling's fourteen, adding anchor_stride and lag_floor -- so it names a model class and sits at layer 1 here against layer 0 there. The sibling keeps its instance in run.py; this cell's binding is reconciled against preflight long before either package has a runner, which is what makes a key that names nothing cheap to find.
- **`cohort.py`** (*equivalent*) Behaviour-equivalent to the sibling module, exercised by `test_eval_sibling_agreement.py::test_the_cohort_order_is_one_ordering_in_both_packages`, `test_eval_sibling_agreement.py::test_the_time_axis_bins_identically_in_both_packages`, `test_eval_sibling_agreement.py::test_the_second_clock_bins_identically_in_both_packages`, `test_eval_sibling_agreement.py::test_the_two_cohort_modules_are_one_file_with_one_import_line`.
- **`collect.py`** (*divergent*) Keys the per-anchor table on the forward's own anchor_index rather than on a row's position in the decoded set, because this cell gathers A_max anchors out of T_valid and a position is not the decimated step it scores. Retains four (136, 30, 98) forecast tensors per sample rather than blocks of a 4 Hz trace, records target_keep_index in collection.json for the kept-axis join, records what the pass cost and at what rate, and accumulates no cross-spectral sums because coherence is not ported. Added since: second_stage_onset joins the identity columns immediately after time_from_labor_onset, so the second clinical clock reaches per_sample.csv as an identity rather than being reported in collection.json as a scored quantity. The sibling carries the same addition.
- **`config_schema.py`** (*divergent*) Adds exactly one eval_config key, the nullable clock_margin_min_nats, which the availability-clock verdict is decided against. Everything else -- the merge, the provenance walk, the forced single-process loader and the closed valid-key set -- is the sibling's.
- **`events.py`** (*divergent*) Reduced to the contraction detector and the gap machinery it needs. The deceleration detector scores a clinical heart-rate trace in beats per minute and this cell forecasts wavelet coefficients, so it and everything reachable only from it -- the block-mode horizon-step helpers and the greedy event matcher -- are removed with their reason recorded.
- **`figures_seam.py`** (*divergent*) The palette, the style refinement and every bound panel are the sibling's. What differs is the lag-axis label: this seam binds lag_axis.COEFFICIENT_LAG_AXIS_LABEL and the group-delay caveat beside it rather than lag_report.COMPENSATED_LAG_AXIS_LABEL, because the axis here is stored-coefficient time. Binding the sibling's label would put the wrong claim on every lag figure while changing no arithmetic. Added since: caveat_note, which prints the group-delay caveat under a lag-resolved figure. One decision about wording and placement rather than one per analysis, because a figure is the artifact most likely to be shown without the directory it came from. Added since: windowed_comparison_figure, the five-row page both clocks draw -- violins per (window, cohort) cell, the Holm-adjusted significance of each window directly beneath them on the same axis, and the effect sizes that survived -- plus the two shared panels it composes (binned_violin_panel, significance_strip). It lives in the seam rather than in an analysis because two analyses draw it and an analysis may not import another. The sibling carries the same builder; what differs between the two files is still only the lag-axis label.
- **`frames.py`** (*divergent*) The aggregation chain, the skill formula, the summary statistics and the positive fraction are the sibling's arithmetic unchanged, and the sibling-agreement assertions named below keep them so. What diverges is one addition this target domain needs and the raw one has no use for: recomposition_check, the guard that a channel-axis split of pred_gap -- by warm-up tertile or by frequency band -- sums back to the gap it decomposes. Its tolerance is scaled by the block score rather than by the gap, because the gap is a difference of two block scores of order 1e3 and a tolerance relative to the difference would tighten without limit as a model improved. The raw cell's target is 16 raw channels with no such split, so the function would have no caller there.
- **`lag_axis.py`** (*divergent*) The compensated-seconds arithmetic and both per-lag readers are the sibling's, unchanged. Added: COEFFICIENT_LAG_AXIS_LABEL and GROUP_DELAY_CAVEAT, because the coefficients are produced by a one-sided bank whose composed group delay reaches 791 s -- the same order as the 364 s lag search -- so this axis is stored-coefficient time and a caption that did not say so would read as a physiological latency. Added since: PREFLIGHT_FILENAME and read_lag_support, so the two per-lag analyses read the measured support margin off the run's own preflight record instead of assuming the shipped geometry -- the floor, max_lag and lag_floor move independently and an arm can reintroduce truncation.
- **`launch.py`** (*equivalent*) Behaviour-equivalent to the sibling module, exercised by `test_eval_sibling_agreement.py::test_the_launch_merge_resolves_identically_in_both_packages`.
- **`metrics.py`** (*divergent*) The readouts move to the feature domain: a five-argument dense forward, the model's own anchored target and mask builders, feature-space trivial predictors, the source-null branch, the warm-up tertiles, and no conversion out of the loader's z units at all -- BPM_UNIT, to_bpm, sigma_to_bpm and fhr_normalization are removed rather than repointed.
- **`oracle.py`** (*divergent*) Scores its probe through the anchored feature builders and emits H*C_keep per anchor rather than a raw block through build_future_target.
- **`preflight.py`** (*divergent*) The causal guard set: transform == 'causal' on every configured shard, the causal widths, causal_reach_budget_s refused outright, fhr_st/fhr_ph normalised in place of the raw 'fhr', guid added to the required load fields, the warm-up budget re-resolved against the evaluation shards and compared with the checkpoint's stamped tuples, and the lag-support margin measured and recorded rather than assumed. NOT_CAUSAL_STATEMENT is replaced -- the sibling's sentence says the inputs read their own future, which is false here and would be a false disclosure rather than a conservative one -- and the shared half of the causality record is wider, because the warm-up budget, the anchor geometry and the lag support belong to the target domain both cfs cells share rather than to either encoder. Also carries GUARD_RECOVERY, a machine-checked table the sibling keeps by hand in EVAL.md. Added since: second_stage_onset joins REQUIRED_EVAL_LOAD_FIELDS and is named in the refusal, which is the config half of the same guard the probe applies to a batch. No new guard function, so GUARD_RECOVERY is unchanged. The sibling carries the same addition.
- **`probe.py`** (*divergent*) Two halves where the sibling has one. The population pass -- one loader iteration into loader_probe.json, and its four refusals -- is the sibling's, unchanged in behaviour, because it is what run.py and every population sanity check read. Added beside it is a forward-contract pass behind --checkpoint: this cell's forward takes five positional arguments and raises without a phase above stride 1, returns two keys the family's does not, and produces (B, A_max, H, C_keep) forecast tensors rather than (B, T_valid, H, R), so the readout module was written against a contract that was measured rather than read. It therefore loads a checkpoint, which makes this module layer 1 here and layer 0 in the sibling, and it refuses any geometry but the dense one so a contract measured at the training tiling cannot be reported. Added since: second_stage_onset is a required batch field in both halves, and the population pass reports its {n_values, n_nan} coverage beside time_from_labor_onset -- the loader skips a field it was asked for and the shard does not carry, silently, so without the requirement a missing field would present as a cohort with no second stage. The sibling carries the same addition.
- **`report_seam.py`** (*divergent*) The mechanism is the sibling's, bound object for bound object. Three content differences: the headline registry drops the three coherence entries and reports the calibration gain per coefficient rather than per element of a 4 Hz trace; HEADLINE_VERDICTS carries ten rather than eight, adding coupling_exceeds_availability_clock and anchor_geometry_intact; and the sanity block drops the two cross-spectral checks, which describe an estimator this package does not have. PRED_GAP_CONVENTION states the 2940-coefficient block and that the likelihood percentage is budget-local.
- **`run.py`** (*divergent*) Registers more analyses than the sibling, keeps the three that draw the model's own forward (samples, recording_traces, attribution) on the binding rather than on the shared registry so another architecture can register its own under the same names, defaults to CFS_BINDING, resolves the collection pass through the binding, records the dense anchor geometry and the training stride in run_context, and registers no coherence step.
- **`spectra.py`** (*absent*) Not ported at all. It estimates cross-spectra from a 4 Hz raw residual; here a tau-slice gives 136 samples at 0.25 Hz per channel over band-limited envelopes, and the analysing filter's phase was discarded before the coefficient was stored. The frequency-resolved question is answered instead by spectral_skill, on the frequency axis the channels already carry.
- **`verify.py`** (*divergent*) Gates the ten-verdict registry rather than eight, adding coupling_exceeds_availability_clock -- which ships INCONCLUSIVE because its threshold ships unset -- and anchor_geometry_intact. Its arm axes are this cell's four sweep arms (anchor_stride, warmup_period, horizon, horizon_depth) rather than the sibling's five, the horizon section carries a refusal rather than a reading rule because a block score is per anchor over H*C_keep coefficients, and it renders a cross-cell table against the transformer cfs cell where the sibling renders none. Two smaller divergences: the kept-channel column is dropped, because the warm-up budget is fixed across all four arms and the column would be constant -- the anchor count and the warm fraction are what these arms move -- and the collapse verdict is reported UNKNOWN unless both per-epoch series are present, where the sibling answers with clause 1 alone and renders the result as 'no'.
- **`analyses/attention.py`** (*divergent*) The attainable entropy ceiling is MEASURED against preflight's own lag_support_margin_steps rather than assumed: three readings of one property -- the recorded margin, the geometry record's truncated-anchor count and the accumulated ceiling against log L -- are compared and their agreement recorded. The truncation accounting is keyed on anchor_floor rather than on a warm-up prefix, because nothing below the floor is decoded at all here; the lag axis is relabelled stored-coefficient time and the group-delay caveat is printed under both figures.
- **`analyses/band_partition.py`** (*divergent*) Widths 36/66/36/15, two new per-channel columns read off the shard attributes (causal_warmup_steps and causal_delay_s), a kept column marking the four channels the budget dropped, and a second channel map on the 98-wide kept axis that spectral_skill joins through.
- **`analyses/calibration.py`** (*divergent*) PIT, coverage, CRPS and the homoscedastic-MLE gain are computed over coefficients rather than over elements of a 4 Hz trace, and the keys say so -- n_coefficients and gain_per_coefficient rather than the sibling's per-raw-sample names, which over this denominator would be silently non-comparable. The CRPS stays in z units with no conversion out of them, and the logvar_clamp recommendation is per coefficient, which is the axis the objective's own block score reduces over.
- **`analyses/coherence.py`** (*absent*) Not ported at all. A stored scattering coefficient is a modulus, so phase agreement, group delay and the residual's three-way split into irreducible, timing and amplitude terms have no analogue here at any window length. spectral_skill replaces the half that does exist and is named differently on purpose, so a reader who knows the raw pipeline cannot carry the wrong contract across.
- **`analyses/coupling.py`** (*divergent*) Arithmetic unchanged; pred_gap_mc_likelihood_pct divides by H*C_keep = 2940 and is therefore budget-local, which the emitted record states.
- **`analyses/cross_subgroup.py`** (*divergent*) METRIC_SOURCES gains the cfs-only per-recording CSVs -- the warm-up tertiles, the source-null difference and the band-resolved skill.
- **`analyses/distributions.py`** (*divergent*) Eight metrics with every conversion out of z units removed; the unit is the loader's z units, labelled normalised.
- **`analyses/events.py`** (*divergent*) One readout of three. Contraction detection and seconds_since_contraction port unchanged and contraction-conditioned coupling with them; deceleration forecast skill and the contraction-triggered response are removed, because both score a clinical heart-rate trace in beats per minute.
- **`analyses/forecast.py`** (*divergent*) The three trivial baselines are rebuilt in feature space on the decimated grid and the horizon curve runs over the checkpoint's own H steps. Every column in a clinical unit is removed rather than repointed -- a wavelet modulus has no clinical unit and inverting the per-channel statistics would put the C_keep scored channels on scales spanning orders of magnitude, which destroys the pooled mean squared error and the skill ratio -- so the error table reports the z-unit columns alone. The forecast overlay draws three kept channels against lead time rather than one trace, because what is forecast is an H x C_keep block, and it indexes the retained anchor axis by position rather than by decimated step: this model gathers its anchors, so the anchor floor F is not a valid index into an axis of length anchor_ceiling - F.
- **`analyses/lag_kl.py`** (*divergent*) All three profiles are kept -- raw, support-corrected and untruncated -- and the analysis MEASURES the truncation rather than asserting it inert: it reads preflight's own lag_support_margin_steps, measures the per-lag contributing-anchor counts, compares the three profiles, and records whether the computed and observed readings agree. The axis is relabelled stored-coefficient time and the group-delay caveat travels on every artifact that states a lag position and under the figure.
- **`analyses/latent.py`** (*equivalent*) Behaviour-equivalent to the sibling module, exercised by `test_eval_sibling_agreement.py::test_the_latent_spectrum_is_laid_out_the_same_way_in_both_packages`, `test_eval_sibling_agreement.py::test_the_latent_diagnostics_are_the_same_thirteen_reductions_in_both_packages`, `test_eval_sibling_agreement.py::test_the_only_gloss_that_differs_is_the_one_naming_this_target_domains_unit`.
- **`analyses/perm_control.py`** (*equivalent*) Behaviour-equivalent to the sibling module, exercised by `test_eval_sibling_agreement.py::test_the_specificity_outcome_is_named_the_same_way_in_both_packages`, `test_eval_sibling_agreement.py::test_the_three_paired_controls_keep_the_same_sign_convention_in_both_packages`, `test_eval_sibling_agreement.py::test_the_branch_summary_and_the_kl_description_agree_in_both_packages`.
- **`analyses/residual.py`** (*divergent*) The forecast-difference RMS is reported in z units rather than in a clinical unit; the two latent quantities and the Jensen pair are unchanged.
- **`analyses/samples.py`** (*divergent*) Draws this cell's fifteen-row diagnostic page through the task's own page seams rather than the raw cells' nine-row one, and draws every selected segment twice: the full page and a reduced five-row one beside it, off a single forward, whose lag attention carries a logarithmic colour scale. Renders a third selection the sibling has none of -- a class-balanced by_class/ draw of caps.pages_per_class segments from every clinical class, beside the shard-proportional stratified/ one -- and records one manifest row per file with a variant column rather than one per segment.
- **`analyses/second_stage.py`** (*equivalent*) Behaviour-equivalent to the sibling module, exercised by `test_eval_second_stage.py::test_the_analysis_writes_its_five_tables_and_both_figures`, `test_eval_sibling_agreement.py::test_the_second_stage_clock_reaches_the_same_verdicts`, `test_eval_sibling_agreement.py::test_the_two_second_stage_modules_are_one_file_with_two_import_lines`.
- **`analyses/sufficiency.py`** (*equivalent*) Behaviour-equivalent to the sibling module, exercised by `test_eval_sibling_agreement.py::test_the_sufficiency_scores_and_both_gaps_are_defined_identically_in_both_packages`, `test_eval_sibling_agreement.py::test_the_oracle_score_join_and_its_summary_rows_agree`.
- **`analyses/time_to_delivery.py`** (*equivalent*) Behaviour-equivalent to the sibling module, exercised by `test_eval_sibling_agreement.py::test_the_time_before_delivery_grid_and_its_readouts_are_the_same_in_both_packages`, `test_eval_sibling_agreement.py::test_the_three_layers_of_inference_reach_the_same_verdicts`, `test_eval_sibling_agreement.py::test_the_two_time_to_delivery_modules_are_one_file_with_two_import_lines`.
- **`analyses/trajectory.py`** (*divergent*) The within-segment structural caveat changes: nothing below the anchor floor F = 134 exists at all, so the profile starts there rather than showing a warm-up droop.

Two modules outside the fork are deliberately **imported** rather than copied, and both sit outside the sibling's `eval/` package so the layering rule reaches neither: `teb_vae/lag_attn_rws/collapse.py::is_collapsed`, which is stdlib-only and keeps `verify.py` free of `torch`, and `teb_vae/lag_attn_rws/trainer.py::RESOLVED_CONFIG_FILENAME`.

## Operations

### Exit codes

The exit code is non-zero **if and only if a step raised.** Three things deliberately do *not* move it:

- a failed **sanity** check — the self-consistency block warns, logs at ERROR and leaves the code at 0, because a run whose every step succeeded can still be one nobody should quote a number from;
- a **coverage** warning (two analyses reporting different populations);
- an **inert-cap** warning (a cap no analysis read).

Run the offline verifier to assess acceptance. It checks the sanity record even when the evaluation process itself exited successfully.

### What a refused run leaves behind

Preflight rejects invalid input with `EvalPreconditionUnmet` before the wrapper that allows individual analysis failures. A refused run leaves `resolved_config.yaml` and `eval.log` with the reason, and **no `summary.json`**. Read the refusal message and use the guard recovery table below.

### Re-running one analysis

```bash
python -m teb_vae.lag_attn_cfs.eval.run --output-dir <a finished run> --only lag_kl
```

This command reads saved tables without building a model or using a GPU. A rerun is **non-destructive but not additive**: existing `summary.json` and `steps.json` files are renamed to backups before new ones are written, and their backup paths are logged. The new summary describes only the analyses just run. **Read the backup** for results the rerun did not produce.

An offline rerun reuses the existing `preflight.json`; without a checkpoint it cannot regenerate the causality disclosure. Table provenance records include checkpoint hash, seed, row count, and `eval_config` digest. A mismatch raises `TablesProvenanceMismatch`, indicating that the saved tables belong to a different run or configuration.

### Guard recovery table

The table below lists each preflight guard, the reason it rejects a run, and the recovery action. The guard's error message gives the specific details. Tests keep this table aligned with `preflight.GUARD_RECOVERY` and the guard functions.

| Guard | Cause | Recovery |
|---|---|---|
| `check_repointed` | A shard path or `stat_path` still carries the `REPOINT_ME` placeholder. Checked **first**, so the message names the real cause rather than a missing file someone would then go looking for. | Edit `dataset_config.vae_test_datasets` and `dataset_config.stat_path` in `eval/configs/eval_overrides.yaml` to name the causal holdout split and its statistics file. |
| `check_test_shards_exist` | A configured evaluation shard is not on disk, or none is configured. | Point `dataset_config.vae_test_datasets` at an existing build, and rebuild in `holdout` mode if the `test/` directory itself is absent — the pipeline's default `augmented` mode writes per-fold splits instead, and a per-fold split is not a substitute. |
| `check_stat_path` | `dataset_config.stat_path` is unset or names a file that is not there. | Regenerate it with `hdf5_dataset/calculate_dataset_stats.py` from the configured **causal** shards at `trim_minutes: 1.0`. A statistics file belonging to the two-sided build fails later and loudly, at the loader's pairing check, not here. |
| `check_trim_minutes` | The loader's trim is not the one the stored warm-up vectors were rebased at. | Set `dataset_config.dataloader_config.dataset_kwargs.trim_minutes: 1.0`. |
| `check_causal_transform` | A configured shard is the two-sided dataset variant, or declares no `transform` at all. The two variants share every field name and dtype, so only the root attribute and the stored widths tell them apart — which is why this is a refusal rather than a warning. | Repoint `dataset_config.vae_test_datasets` at the causal build (`hdf5_dataset/new_pipeline`, causal variant, `holdout` mode). |
| `check_load_fields` | `load_fields` omits a field a readout is asked in, or a key the anchor tiling's phase is derived from. The loader **skips** a field a shard does not carry, silently, so this otherwise presents as "no classes found" rather than as a data problem. | Add the named field to `dataset_config.dataloader_config.dataset_kwargs.load_fields`; the committed `eval/configs/eval_overrides.yaml` lists the full set, `guid` and `epoch` included. |
| `check_target_normalized` | `fhr_st` or `fhr_ph` is missing from `load_fields` or from `normalize_fields`. The target is their concatenation, so a config carrying one of them is a target with a hole in it. | Add both to `dataset_config.dataloader_config.normalize_fields` and to `dataset_config.dataloader_config.dataset_kwargs.load_fields`. |
| `check_no_reach_budget` | `model_config.VAE_model.causal_reach_budget_s` is set on one-sided features. It prunes channels by the forward reach of a *two-sided* Morlet, measured on a bank that did not produce these coefficients. | Set `model_config.VAE_model.causal_reach_budget_s: null`. |
| `check_declared_widths` | The **model's** `c_y` / `c_u` disagree with the configured shards' stored widths. Compared against the model rather than against the config, because the evaluation rebuilds from the checkpoint. | Point `dataset_config.vae_test_datasets` at the shards this checkpoint was trained on; do **not** change `model_config.VAE_model.c_y` / `c_u`, which the checkpoint overrules. |
| `check_warmup_budget_matches_checkpoint` | The warm-up budget re-resolved against the configured shards does not produce the checkpoint's own stamped channel tuples. Two arms at two budgets have mutually unloadable checkpoints and the class stamp cannot separate them. | Set `model_config.VAE_model.causal_warmup_budget_steps` to the value the training run used, or repoint `dataset_config.vae_test_datasets` at that run's dataset. |
| `reconcile_with_checkpoint` | A declared geometry or objective key contradicts the checkpoint's own `model_kwargs` / `hyper_parameters`. The checkpoint always wins, so the config's values would be reported beside numbers they did not produce. | Evaluate the checkpoint against its own `model_checkpoints/resolved_config.yaml`, which the training run writes beside it. |
| `verify_weights_loaded` | Every witness tensor is still at its construction constant, so no checkpoint weights reached the model. `load_checkpoint_strict` returns `None` rather than raising, so an unchecked load would report randomly initialised weights as a measurement. | Pass `--checkpoint` a trained `.ckpt` whose `model_class` matches this package's model. This is a weight-space check, not a behavioural one: a genuinely trained model whose *source pathway* collapsed still passes here, because that finding must be reported rather than refused. |

### Dependencies

The evaluator uses the project's existing `torch`, `numpy`, `pandas`, `matplotlib`, `h5py`, `pyyaml`, and `loguru` dependencies. It imports `scipy` where needed. The pinned `pyarrow` dependency writes the per-anchor table. The basic `verify.py` acceptance check uses the standard library; generating arm tables also requires `pyyaml`.

### The gate

From the repository root:

```bash
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_cfs/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_cfs/tests -q -m slow
```
