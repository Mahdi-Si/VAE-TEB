# The evaluation contract

What a run of `teb_vae.lag_attn_cfs.eval` is, what it leaves behind, what each analysis means, and
how the output is misread. `FIGURE_GUIDE.md` beside this file documents every emitted PDF; this
document is everything that is not a figure. Both are bound to the code by test: every registered
analysis has a heading here, every resolved `eval_config` key is mentioned here, every way preflight
refuses a run has a recovery row here, and every figure in the committed `figure_manifest.json` has
a guide entry.

**This package is a fork of `teb_vae/lag_attn_rws/eval`, edited for a different target domain.** The
raw pipeline forecasts a 4 Hz heart-rate trace; this one forecasts $15 \times 98$ wavelet-modulus and
phase-harmonic coefficients produced by a strictly one-sided filter bank. That difference reaches
into the forward call, the target and its masks, the baselines, the units and most of the analyses,
which is why the pipeline was copied rather than parameterised through a four-field binding — and
the fork carries named measures against the drift a copy invites. *The divergence register* below is
one of them, and it is rendered from `divergences.json` rather than kept by hand.

## What a run is

One command reads one checkpoint and writes one reviewable directory:

```bash
python -m teb_vae.lag_attn_cfs.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt
```

The configuration is the checkpoint's own `resolved_config.yaml` — found beside it, never a second
config file — with the committed `eval/configs/eval_overrides.yaml` delta deep-merged over it. The
delta repoints the shards at the **causal** holdout split and the statistics accumulated from those
same shards, adds the five clinical `load_fields`, keeps `guid` and `epoch`, and carries the
`eval_config` block; both the original and the merged value of every overridden key are recorded in
the summary. Preflight then refuses the run outright when the merged result contradicts the
checkpoint.

**The forward is called densely, once per run directory, and that is not the geometry the model was
trained at.** Training tiles the anchor set — `anchor_stride: 15` on the shipped configuration — for
gradient decorrelation and activation memory, neither of which applies where there is no backward
pass. The evaluation calls
`model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1)`, which is exactly what
`SeqVaeLagAttnCfsTask.resolve_anchor_geometry('test', batch)` returns and therefore what `val` and
`test` already use, and it decodes **every valid anchor** of every segment — the checkpoint's own
`anchor_ceiling - warmup_period`: 136 on the stored forecast clock, 51 under the shipped
`physical` one, whose 85-step largest advance removes the trailing anchors. The training stride is
recorded in `run_context` beside the decoded geometry, because a table that did not say which
geometry it was produced at cannot be read against the training CSV.

The expensive part happens once. A single shared collection pass decodes **four scored branches**
over every anchor at $K$ Monte Carlo draws under common random numbers — `base`, `full`, `shuffled`
and `base_shuffled_mu` — plus **a fifth arm that is never decoded**: `kld_source_null` re-runs the
source gate, adapter and encoder from a zeroed source stream and returns only
$(\mu^{q,\mathrm{null}}, \ell^{q,\mathrm{null}})$. It costs one source encode and no decode, and
draws no `randn_like`, so it does not move the reparameterisation stream. The pass writes two durable
tables — `per_sample.csv` (one row per segment, with the clinical labels attached) and
`per_anchor.parquet` (keyed `(guid, epoch, anchor_index)`) — plus a vector sidecar and the aggregated
readouts. Every analysis then reads those files, which is why

```bash
python -m teb_vae.lag_attn_cfs.eval.run --output-dir <a finished run> --only coupling
```

re-runs an analysis offline with no checkpoint, no model and no GPU. `--max-batches` is a smoke-run
batch cap (a prefix by nature); `eval_config.max_samples` is the seeded *stratified* cap, and the two
are not interchangeable — a prefix over the unshuffled eight-shard split draws one subgroup and one
class.

A finished run is checked mechanically, and the arm tables are generated, by the same offline module:

```bash
python -m teb_vae.lag_attn_cfs.eval.verify <run>/eval_results/summary.json
python -m teb_vae.lag_attn_cfs.eval.verify --runs <dir-of-runs> --out RESULTS_arms.md
```

`verify` reads files a run left behind and nothing else — no model, no shard, no `torch` (the
layering test walks its imports with `torch` on the forbidden list, so the property is proved rather
than promised). The gate names which `pred_gap` column it reads: `pred_gap_mc_nats`, the Monte Carlo
marginalised score. The arm tables key every row by the swept value read from each run's own dumped
`resolved_config.yaml`, never by directory name, and read each run's training `metrics_history.csv`
for the epoch count, the final `val/kld_active_frac` and the collapse verdict
(`teb_vae/lag_attn_rws/collapse.py::is_collapsed`, imported rather than forked because it is
stdlib-only arithmetic over a per-epoch series both raw packages already share).

**Two green checks answer two different questions, and neither substitutes for the other.**
`check_run.py` reads a run's `train_results/metrics_history.csv` and needs no checkpoint, no shard
and no `torch`; it answers *did the fit behave*, and it answers it **while a run is in flight**,
which is what it was built for. `eval/verify.py` reads `summary.json` and needs the same nothing; it
answers *is this finished checkpoint acceptable*, on a held-out population, per recording, with
intervals. The first is in-sample, per-epoch and has no denominator; the second is neither.

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

Three summary blocks matter more than the rest. The **headline** is a flat registry of scalars and
verdict statuses; a number not registered there is invisible to the acceptance gate and the arm
tables, which read it and nothing else. It carries two `pred_gap` columns under names that say which
is which — `pred_gap_mc_nats` (the headline, the log of the average likelihood over $K$ draws) and
`pred_gap_train_path_nats` (the single-draw objective-parity column) — and three percentage columns
restating the same finding proportionally. `pred_gap_convention` says in the artifact itself which is
which, what a block is here, and which of the three is budget-local. The **sanity** block is the
run's three-valued self-consistency record (the KL identity, the cross-table recombination, the lag
identities, the population checks); it deliberately does *not* move the exit code. The **verdicts**
are the model's own acceptance criteria, in registry order, never a bare boolean — **ten** of them
here against the raw pipeline's eight. The **run context** block beside them records the parameter
count, the checkpoint's training epoch, the decoded anchor geometry, the training stride, the
anchor-coverage distribution and the observed objective magnitude.

## The four layers

The package is layered, and an AST walk (`tests/test_eval_self_contained.py`) enforces the import
rules, resolving aliased, lazy and relative forms alike:

| Layer | Modules | May import |
|---|---|---|
| 0 — pure | `config_schema`, `verify`, `events`, `frames`, `lag_axis`, `cohort`, `figures_seam`, `report_seam`, `launch`, the `_reuse` seam | no Lightning, no `model.*`, no `task`/`trainer`/`plotting`; `verify` additionally no `torch` and no `binding` |
| 1 — model-touching | `binding`, `metrics`, `collect`, `preflight`, `probe`, `oracle` | `task`/`trainer` only via the named `EXEMPTIONS` table |
| 2 — I/O and presentation | `analyses/*` | layers 0–1; never another analysis, never Lightning, never the model |
| 3 — orchestration | `run` | everything at the layers below it |

Two rules are this package's rather than the sibling's. **`teb_vae.lag_attn_rws.eval` is forbidden
outright** — the pipeline this one was forked from is not reachable sideways from any module here,
because a half-fork is worse than either whole, and the only exemption is the handful of *test* files
that exist to compare the two. And `binding` and `probe` sit at layer 1 rather than layer 0: the
first names a model class, and the second loads a checkpoint, because this cell's forward takes five
arguments and refuses a missing anchor phase above stride $1$, so its contract is *measured* against
a rebuilt model rather than read.

The shared evaluation package (`teb_vae/lag_attn/eval`) is reachable only through its model-free
modules, named in an allow-list, and `model/*` is forbidden everywhere. The `EXEMPTIONS` table is
asserted **minimal**: a module listing a name it no longer imports is a permission that outlived its
use.

### The model binding

`binding` carries the handful of facts the pipeline cannot derive — the classes to rebuild from a
checkpoint, the constructor keys reconciled against it, the encoder's own half of the causality
disclosure, the package's committed override delta, and the analyses and headline scalars this model
alone can have. `CFS_BINDING` is the default of `run.main`; the second consumer is
`teb_vae/lag_attn_transformer_cfs/eval`, which evaluates `SeqVaeLagAttnTrfCfs` — the same model with
both history encoders replaced — through this runner, this collection pass and all of these
analyses, adding none of its own.

`GEOMETRY_KEYS` is **sixteen** here against the raw cells' fourteen, adding `anchor_stride` and
`lag_floor`. It obeys a rule narrower than "the constructor's parameters": `preflight.reconcile`
compares `model_config.VAE_model[key]` against `model_kwargs[key]` and silently skips any key absent
from either, so a key must be a config key **and** a constructor parameter to be checked at all.
`causal_warmup_budget_steps` is therefore deliberately absent, and so are the four tuples it resolves
to (`target_keep_index`, `target_warmup_steps`, `source_keep_index`, `source_warmup_steps`): the
budget is a config key but not a constructor parameter, and the tuples are constructor parameters but
not config keys, so reconciling either here would compare against nothing. They get their own guard
instead — `check_warmup_budget_matches_checkpoint` re-resolves the budget against the *configured*
shards and compares the result with the checkpoint's stamped tuples, which is the only comparison
that can actually fail.

## Configuration reference

Everything that shapes a run lives in the `eval_config` block of the override delta, because that
block is dumped into the run directory and is the durable record. Every key is validated against a
closed set — an unknown key raises and names the valid keys, a `bool` where an `int` is expected
raises (`True` would silently cap at 1), and a cap of `0` raises.

| Key | Meaning |
|---|---|
| `seed` | Seeds `random`/`numpy`/`torch` and derives the loader-shuffle, derangement, stratified-cap and Monte Carlo generators by fixed offsets. Two runs of one checkpoint at one seed compare byte-identical on `results`. |
| `num_mc_samples` | Monte Carlo draws $K$ per anchor for the marginalised score, under common random numbers across branches. $K = 1$ reduces it exactly to the training-path score. |
| `max_samples` | Seeded **stratified** global sample cap; `null` evaluates the whole split. |
| `caps` | Per-quantity retention caps (`waveforms`, `attention`, `pages`, `pages_per_class`, `oracle`). Retention is opt-in: a quantity absent from `caps` is retained for no samples — except `oracle`, where absence means every segment, because a probe fitted on nothing is not a cheaper measurement but no measurement. The first cap ships **halved** against the raw cells, at 64: a retained forecast set here is four $(136, 30, 98)$ fp32 tensors, about 6.1 MiB per segment against their 2.0 MiB. |
| `prior_shuffle_min_nats` | The provisional margin the prior-shuffle degradation must clear; the verdict always reports the measured number beside it. |
| `min_active_dims` | Active latent dimensions below which the latent counts as collapsed. |
| `event_lag_window_s` | Seconds after a detected contraction within which an anchor counts as event-conditioned. Still read here, because contraction-*conditioned* coupling ports even though the two readouts that scored a clinical trace do not; `lag_high_kl` reads the same window for its contraction enrichment of high-KL anchors, so the two analyses agree about what "near a contraction" means. |
| `bootstrap_resamples` | Resamples behind every bootstrap interval, drawn over recordings — never over anchors, whose forecast windows overlap 14/15. |
| `clock_margin_min_nats` | **This cell's own, and it ships unset.** The margin $\Delta_{\mathrm{clock}}$ must clear for `coupling_exceeds_availability_clock` to PASS. Nullable exactly as `max_samples` is; at `null` the verdict is INCONCLUSIVE and the measurement is emitted anyway. See *`source_null`* below for why a guessed threshold would be worse than no threshold. |
| `figure_format` | Image format every figure of the run is written in, as a matplotlib filetype (`pdf`, `svg`, `png`, `eps`, …); validated at config load against the installed matplotlib's own list. `null` — the shipped setting — keeps the `pdf` default, which is what `figure_manifest.json` and `FIGURE_GUIDE.md` record and what the smoke suite compares a real run against; a run that changes it writes filenames those files do not list. |
| `max_hours_before_delivery` | How far before delivery a segment may be recorded and still be evaluated, in hours; `4.0` keeps the last four hours, `null` — the shipped setting — evaluates everything. **The bound is on the population, not on an axis**: it is applied to the delivery clock before anything is binned, so every clock answers for the same segments and the second-stage clock re-bins that population on its own signed axis rather than being cut at a second, differently-defined four hours. It moves cohort sizes, window counts and every trajectory, so a bounded run is not comparable with an unbounded one — which is why it is a key, recorded in the run's dumped config. Minimum one 0.5 h bin. |

Deliberately **not** keys: the significance level and the trajectory bin width (an operator who could
widen them could make a difference appear or disappear), any lag-band selection, and — the one this
cell adds — **the anchor stride**. An operator who could set the stride could change the population
every number in the run is computed over. Their absence is asserted by test, not merely intended.

### Objective keys this pass reads rather than sets

The objective is not an `eval_config` surface. A checkpointed pass rebuilds the task from the
checkpoint's **own** `hyper_parameters` — `beta_schedule`, `kld_beta`, `beta_prior`, `lambda_full`,
`lambda_base`, `likelihood` and `free_bits` — and refuses a checkpoint that carries none, because
scoring it under assumed defaults would report a different objective's numbers. On the offline path,
where there is no checkpoint, the same keys are read from the dumped `model_config.VAE_model` block,
which preflight has already reconciled against the checkpoint on every checkpointed run.

Weights are recorded and, unlike geometry, deliberately **not** compared against the config by the
preflight guard: $\beta$ and its ramp weight the training total and enter no evaluated readout.

### Cohort order and colour

Two presentation conventions every table and figure that resolves a quantity by cohort obeys.
Neither is a setting, for the reason the significance level is not one.

**The order is clinical, not alphabetical, and it runs worst first**, on both axes:

| Axis | Order, left to right |
|---|---|
| `clinical_class` | HIE, acidosis, healthy |
| `subgroup` | `hie_cs`, `hie_no_cs`, `acidosis_cs`, `acidosis_no_cs`, `healthy_bg_cs`, `healthy_bg_no_cs`, `healthy_no_bg_cs`, `healthy_no_bg_no_cs` |

`labels.ordered_groups` is the one function that decides it — `cohort.ordered_groups` is a
one-line binding of it, so this package and the sibling cannot come to disagree — and both orders
are read off `labels.CLASS_NAMES` and `labels.CANONICAL_SUBGROUPS` in reverse rather than
restated. Those tables are bound through `_reuse.py` and are deliberately **not** forked, because
forking them would fork the definition of a cohort. A cohort the order does not know sorts
**after** every one it does, and is never dropped.

**The colour is the severity**: green for healthy, amber for acidosis, red for HIE, each subgroup a
shade of its own class. The mapping is a *table* rather than an assignment pass, so a cohort keeps
its colour whichever others a figure contains. The palette is this package's own and deliberately not
`utils.style.CLASS_COLORS_DEFAULT`, which paints healthy blue and is shared with two other projects —
so **an evaluation figure of a cohort is not the same colour as a training-callback figure of that
cohort**, and the two are reconciled by legend rather than by hue.

**The order is also the orientation of every significance test.** `stats.pairwise_comparisons`
names each pair in the order it receives the cohorts, and every caller hands them over through
`cohort.ordered_groups` — so a comparison always runs *more severe to less severe*: HIE vs
acidosis, HIE vs healthy, acidosis vs healthy on the class axis, and the same order pair by pair on
the subgroup axis. That is the direction the clinical question is asked in: the cohort a readout
exists to detect is named first, and what it is read against second. Cliff's delta is signed
against that naming, so **a positive $\delta$ means the more severe cohort's values run higher**, on
every pair of every metric, window and clock rather than on the ones whose names happened to sort
that way. A run directory written before this convention names the class pairs healthy-first
instead, with the opposite sign.

## The analyses

One section per registered analysis, named for its module. `band_partition` always runs and is not
selectable; the rest are what `--only` and `--skip` choose between, in this order, and `run.py`'s
`RUN_ARGS` comment table is the list an operator reads while choosing. `cross_subgroup` is
deliberately last, and that ordering is load-bearing: it reads the per-recording CSVs the analyses
above it write.

### band_partition

What each of the model's input channels is, read off the shards' own `sel_*` provenance and causal
attributes rather than re-derived: one row per channel across the 102-channel target stream
(`fhr_st` 36 + `fhr_ph` 66) and the 51-channel source stream (`up_st` 36 + `up_ph` 15), laid out as
the model receives them, each mapped to a band, a kind and a centre frequency in Hz. It describes the
model's **inputs** and is the data-side companion to the causality disclosure.

Two columns are this cell's own and every statement in this pipeline rests on them:
`causal_warmup_steps`, the leading delay enclosing 95% of the one-sided kernel's energy, and
`causal_delay_s`, the composed group delay. A third, `kept`, marks the channels the warm-up budget
dropped.

**On this dataset the unbanded path is the common case rather than the exception.** `sel_*`
provenance is stored on the two *phase* blocks only, so a scattering channel's centre frequency is
recoverable only where some selected phase pair named its filter. Seven of the 36 declared `fhr_st`
channels have none: three above the phase selection's upper edge and four below its $0.008$ Hz floor.
Those channels are recorded as `unknown` and **never bucketed into a neighbour** — a band whose
membership quietly absorbed them would misattribute their skill to a frequency they do not have. A
shard carrying no `sel_*` attributes at all is a recorded skip, not a raise.

It also emits `band_channel_map_kept.csv`, the same map restricted and re-indexed onto the **98 kept
channels** the decoder actually emits. That second file exists for one reason: it is the join
`spectral_skill` goes through, and it is a file on disk rather than a model attribute so that
`--only spectral_skill` against a finished directory works with no checkpoint and no GPU.

**The two attributes the aligned shard variant added are deliberately not read here, and this says
which and why.** A causal shard now also carries `causal_leg_alignment` at the root and
`causal_novelty_frac` per block.

* `causal_leg_alignment` names which phase-harmonic operator built the phase blocks. It is a
  property of the **file**, not of a channel, and it is already refused at resolution time: a run
  whose `causal_leg_alignment` config key disagrees with what its shards record never reaches an
  analysis. Repeating it as a channel column would put the same fact in two places, one of which
  could be stale, and neither `band_partition` nor `spectral_skill` needs it — every column both
  emit means the same thing under either operator, because the alignment changes what a phase
  coefficient *is* and not what channel it is.
* `causal_novelty_frac` is genuinely per channel and would fit both maps. It is not added, and the
  reason is that it would be a second, offline copy of a split the **training** path already reports
  per epoch as `pred_gap_novel_lo` / `_mid` / `_hi` — computed against the model's own gathered
  channel axis rather than against a positional join. Add the column here when a band-resolved
  novelty question is actually asked; until then, the number a reader wants is in `metrics_history`
  and in the run's own tertile columns.

`spectral_skill` reads no shard attribute at all: it carries `causal_warmup_steps` and
`causal_delay_s` forward from the kept map above. Whatever `band_partition` emits, it inherits, so
the answer for it is the same answer.

**One column that is now narrower than it reads, recorded here rather than repaired.** The `kept`
column is computed for the **target** stream alone; every source row is written `kept = 1`. That was
exactly true while the source keep-index was the identity, and it stopped being true when the channel
alignment began dropping the four source channels whose composed delay exceeds the reference — those
rows still read `kept = 1` while the model does not read them. Repairing it needs `source_keep_index`
in the collection record, which is a change to the collection schema rather than to this analysis;
until then the run's own preflight record carries `source_dropped_index`, which is the authoritative
list.

### forecast

Is the forecast any good, against predictors that know nothing, and where in the horizon. A block
score alone cannot answer that: it is a negative log density summed over
$H \cdot C_{\mathrm{keep}} = 2940$ coefficients, so it is large under every predictor and its scale
is set by the block size rather than by the model. Two things make it readable — skill against
**three trivial baselines** scored through the model's own masked scorer with the identical mask at
the identical anchors — and a third says where in the forecast window the answer holds.

The three baselines are rebuilt in feature space on the decimated grid, and each has an exact
analogue there:

- **persistence** — anchor $t$'s whole window is filled with the coefficient vector at the last
  **observed** step at or before $t$, per channel. "Last observed" rather than "last": `weight` is
  the only trustworthy validity signal here, because the coefficients carry no sentinel of their own,
  and carrying an invalid step forward would measure the gap.
- **climatology** — exactly $0$ per channel, which is the z-scored population mean. The statistics
  were accumulated *excluding* the warm-up region, which is what makes zero the channel mean over the
  region the model reads.
- **segment mean** — the per-channel mean over the segment's own valid steps.

All three are scored at a fixed `BASELINE_LOGVAR = 0.0`, recorded beside the score, so a
learned-variance model cannot beat a point predictor on variance modelling alone without that being
visible.

The MSE-space skill $1 - \mathrm{MSE}_m/\mathrm{MSE}_b$ is the one with a natural zero. The NLL-space
column beside it is a **difference** in nats, `advantage_nats_per_anchor`, not one minus a ratio: a
log score has no natural zero, so the ratio of two of them is not bounded above by one and changes
sign with the baseline's.

**Every error column is in the loader's $z$ units, labelled `normalised`, and there is no conversion
out of them anywhere in this pipeline.** The sibling's `BPM_UNIT`, `to_bpm`, `sigma_to_bpm` and
`fhr_normalization` are deleted rather than repointed: a scattering or phase-harmonic coefficient has
no clinical unit, and inverting the per-channel statistics would put the 98 scored channels on scales
spanning orders of magnitude, which destroys every pooled statistic, every shared colour bar and the
tertile split.

The horizon curve runs over the 15 forecast steps and is computed on the **single-draw** path, and
says so: the marginalisation does not commute with the sum over $\tau$, so a marginalised curve would
not sum back to the marginalised headline.

### coupling

What the source added, per recording, with the uncertainty on it. `pred_gap` in both estimators — the
Monte Carlo marginalised headline and the single-draw training-path parity column, never merged —
with the fraction of recordings where the gap is positive, a paired Wilcoxon over the per-GUID
vector, bootstrap intervals over recordings, and quantiles rather than only means.

The positive fraction reports its **denominator**: `np.nan > 0` is `False`, so unscored segments
would otherwise count silently as evidence against. The KL travels beside the gap as a
**description** rather than as a second answer — it is inflated by an arbitrary factor whenever the
prior variance sits on its clamp, and unlike `pred_gap` it says nothing about whether the forecast
improved.

**The same finding as a percentage**, because nats state no proportion. Three columns, each computed
per recording and then averaged, and each bootstrapped over recordings: `pred_gap_rmse_pct` and
`pred_gap_mse_pct` in error space, where a forecast equal to the truth scores 100%; and
`pred_gap_mc_likelihood_pct` $= 100(e^{\Delta/2940} - 1)$, the extra probability density the
source-conditioned forecast puts on each observed coefficient. The arithmetic is
`frames.skill_against`, the same guarded function `forecast` scores its baselines with: the
denominator is tested **strictly positive** and fails to `NaN`, never to `inf` (which the headline's
finiteness check would refuse) and never to `0.0` (which reads as "no improvement").

The likelihood percentage has the sibling's two preconditions — a `gaussian_nll` likelihood, and the
block size read from the run's own geometry — and one more that is this cell's: **it is
budget-local**. It divides by $H \cdot C_{\mathrm{keep}}$, and $C_{\mathrm{keep}}$ is whatever the
warm-up budget decided, so two arms of this model at two budgets divide by two different numbers. The
emitted record states it; nothing tries to normalise it away.

### perm_control

Does the model use *this* recording's source, or react to any source at all? The verdict is three
losses and nothing else: $D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}}$.

The KL is deliberately not a parameter, and that is the content of the criterion rather than a
simplification of it. A stranger's source is out of distribution for a posterior trained only on
matched pairs, so it routinely moves the posterior **more** — a healthy model has
$K_{\mathrm{shuffled}} > K_{\mathrm{true}}$, and a criterion that read the KL would fail exactly the
models it should pass. The derangement is **GUID-aware**, a batch with no cross-recording pairing
available is excluded and **counted**, and the shuffled branch is scored on its own rather than
differenced sample by sample, because the permutation draws a fresh $\epsilon$.

Three paired controls are scored per recording under one sign convention — positive means the control
is worse than the branch it is referenced against: `shuffle_penalty`
($D_{\mathrm{shuffled}} - D_{\mathrm{base}}$), `prior_shuffle_penalty` (the same branch under a
shuffled prior mean) and `source_margin` ($D_{\mathrm{shuffled}} - D_{\mathrm{full}}$). **The third
is referenced against `full`, and that is why it exists**: the two above it inherit whatever the base
forecast is doing, while `source_margin` changes only the source. A positive margin beside a negative
predictive gain is a real state rather than a contradiction.

**This control structurally cannot see the availability-clock hazard**, and that is why
`source_null` exists beside it rather than instead of it. A permutation deranges *rows*, and the
source availability pattern is a deterministic function of $t$ that every row of a batch shares, so
no permutation of rows can remove it. What this control answers is specificity, which the source-null
arm does not.

### latent

How much of the latent carries source information, and whether its variance is fitted or bound. The
per-dimension KL spectrum, the active-dimension count and the top dimension's share — and the
detectors the evaluation would otherwise never read, though the model computes them and the trainer
logs them every epoch: `mean_logvar_prior`, `logvar_prior_floor_frac`, `mean_logvar_post`.

That second half is the point. The KL carries $(\mu^q - \mu^p)^2 / \sigma_p^2$, so a **prior**
variance pinned on its lower clamp multiplies every coupling readout by an arbitrary factor while
every decoder-side diagnostic stays perfectly healthy. `prior_variance_not_pinned` is the FAIL-able
verdict that catches it, judged at the model's own margin — 5% of the clamp range. The bound is a
sigmoid, so an exact-equality test would read zero forever.

`prior_rate` is the same pathology as a **distance rather than a fraction**: the objective's own
$R_p = \sum_d \tfrac12(e^{\ell^p} - 1 - \ell^p)$, per recording and in nats per anchor, reduced on the
KL support like the divergence beside it. Zero means $\sigma_p = 1$ exactly, so it is the only one of
these readouts bounded below by its own optimum, and it is continuous where the floor fraction is a
step. Read the two together.

### lag_kl

Where in the past the source informed the future. The per-lag KL attribution
$\widetilde K_{t,\ell} = \sum_m K^{(m)}_t \alpha^{(m)}_{t,\ell}$, whose sum over lags is exactly
$K_t$ — an identity re-measured on **this run's worst anchor** every pass and registered in the
sanity block, not inherited from a model test.

Three profiles, not one. The **raw** attribution divides every bin by the same anchor total and is
therefore a decomposition of the headline KL. The **support-corrected** one divides each bin by the
anchors at which that lag was causally valid. The **untruncated** one is recomputed on the anchors at
which every lag exists, because the support correction fixes each bin's denominator and cannot fix
its numerator.

**At this cell's geometry the last two corrections are inert, and the analysis measures that rather
than assuming it.** The anchor floor is $F = 134$ and the furthest searched lag is $L - 1 = 90$, so
every lag is causally valid at every scored anchor and the support margin is
$134 - 90 - 0 = 44 \ge 0$. That margin is a **number preflight computes and records**, not an
assumption: the floor, `max_lag` and `lag_floor` move independently, and a lower-floor arm would
silently reintroduce truncation. This analysis reads the recorded margin, measures the per-lag
contributing-anchor counts, compares the three profiles and records whether the computed and observed
readings agree. A negative margin is legitimate rather than refused — the corrections then do work
again, which is what they are for.

An argmax is not by itself a reading. Peak width, mass above threshold and secondary peaks travel
beside it, and `degenerate` is defined mechanically — peak-to-median below 1.1, **or** exact-zero
fraction above 0.9 — because `entmax15`'s exact zeros can make an argmax on a flat profile
meaningless.

**The axis is stored-coefficient time**, and the group-delay caveat travels on every artifact this
analysis writes. See *How the output is misread*.

The stratified table cuts the profile by class, by subgroup and by time-before-delivery window, one
axis at a time and without a test. `lag_clocks` is where that question is asked properly: the same
profile resolved against **both** clinical clocks, by class, drawn, and tested per window.

### attention

The attention itself: per head, against the entropy it can actually reach. The posterior is
head-structured — latent group $m$ is written by attention head $m$ alone, which is what makes the
per-head KL an additive decomposition rather than an arbitrary slice — so averaging the four heads
before profiling discards exactly what the architecture exists to expose: four heads at four delays
and one head attending everywhere produce the same head-averaged curve.

The entropy ceiling is $\operatorname{mean}_t \log \min(t+1, L)$ over the anchors actually scored.
**At this floor that equals $\log L$ exactly, and the analysis measures it rather than substituting
the constant.** Three readings of the one property are compared and their agreement recorded:
preflight's own margin, the geometry record's truncated-anchor count, and the accumulated ceiling
against $\log L$. Both entropies are emitted, distinctly named, and the ceiling is a per-sample column
over the sample's own scored anchors, so their ratio stays a measurement on an arm where it is not
a no-op.

The entropy is taken per anchor and then averaged, never as the entropy of the averaged profile — a
mixture's entropy is at least the mean of the entropies mixed, so the second reports a model whose
lag focus *shifts* as one that has none. `kld_per_t_per_head` sums over heads to `kld_per_t` exactly,
and that identity is the second sanity-block check.

### calibration

Is the decoder's learned variance the spread of its own errors? Under `gaussian_nll` the block score
is a negative log density only if it is, and nothing else in this pipeline checks it — a model can
drive its NLL down by shrinking $\sigma$ wherever it happens to be right and paying for it elsewhere,
and every score in every other analysis would improve.

Four readings, over **coefficients** rather than over 4 Hz elements of a trace, and the key names say
so — `n_coefficients` and `gain_per_coefficient`, because over this denominator the sibling's names
would be silently non-comparable. The PIT; central coverage at the exact erf nominals
$\operatorname{erf}(k/\sqrt 2) = 0.6827,\ 0.9545,\ 0.9973$ (the two-sigma figure is **not** 0.95 —
that is $\pm 1.96\sigma$, and the half-point difference reads as a real miscalibration); CRPS, which
stays in $z$ units with no conversion out of them; and the NLL gain over the homoscedastic MLE fitted
to the very residuals being scored.

`logvar_full_floor_frac` and `logvar_full_ceil_frac` ship beside `mean_logvar_full`, because a single
mean is equally consistent with a spread distribution and with half the mass pinned on each clamp.
This is the one analysis whose output directly changes a config value: it states the recommended
`model_config.VAE_model.logvar_clamp` revision **per coefficient**, which is the axis the objective's
own block score reduces over, and says **no change** when neither end binds. An `mse` checkpoint
records a skip and accumulates nothing, because the log-variance head is never fitted there.

### residual

How far apart the two forecasts are, and how far the source moved the belief behind them. There is no
residual *tensor* in this model: `mu_base` and `mu_full` are two passes of **one shared decoder** on
two latents, with no `delta_mu_src` and no base-plus-residual head.

Reported instead: the per-anchor forecast-difference RMS in $z$ units, and the two latent-side
quantities that are **not** the same thing — `delta_mu_rms`, per element, and
`mu_post_prior_gap_rms`, per step with the L2 over $d_z$ taken first.

RMS metrics accumulate unrooted and root once. Averaging finished per-sample RMS values is biased low
by Jensen, in the direction that flatters the model; the analysis reports the average-of-roots beside
the rooted-once value, so the bias is a measured number rather than an assumed one. One caveat
weakens in the model's favour and is stated: both branches share one log-variance head applied to
different $z$, rather than reading two separate variance heads.

### distributions

What each metric's distribution over 20-minute **segments** looks like, cohort by cohort. Every other
analysis reduces to one value per recording before reporting anything, and what that hides is the
*shape*: three cohorts with the same mean forecast error can be a uniform shift, a heavier tail, or a
handful of segments the model fails on completely, and those are three different findings. Eight
metrics, all in the loader's $z$ units.

**It is descriptive by construction.** No test, no interval, no $p$-value, and nothing registered in
the headline block. That is not an omission: a per-segment $p$-value is anticonservative by the anchor
overlap, and `cross_subgroup` remains the only analysis that adjudicates a cohort difference.

**Both levels are drawn on the same axes, and that is the content.** The filled density is one value
per segment; the median / inter-quartile / range **strip** above it is one value per recording. Their
difference *is* the pseudo-replication. Four presentation choices are load-bearing: density rather
than counts, one bin grid per panel, a nested subgroup figure, and the overlap encoding — a faint fill
under a hairline outline at full opacity, drawn in two passes so every outline sits above every fill.

It declares **no** `grouped_frames`. The runner's fan-out draws violins documented as holding one
value per recording; handing it this per-segment frame would produce a per-segment violin that reads
as a per-recording one.

### trajectory

The two coupling readouts against time — inside a segment, and across a whole delivery. The
per-anchor table's first general consumer.

**The within-segment structural caveat is this cell's rather than the sibling's, and it is the
opposite shape.** The raw cells show a warm-up droop at the left of the profile, because their
anchors begin at the model's own 30-step warm-up and the lag support is truncated for a while after
it. Here **nothing below the anchor floor $F = 134$ is decoded at all**, so the profile *starts*
there: there is no droop to discount and no truncated region inside the profile, and the last $H$
anchors are still never scored. A reader expecting the sibling's shape and finding a profile that
begins two-fifths of the way into the segment is looking at the geometry rather than at a failure.

Across a delivery the segments are assembled on the absolute time axis
$t_{\mathrm{abs}} = \mathrm{epoch} + 4t$, with overlapping timesteps **averaged** rather than drawn
twice and `n_contributing` travelling so the averaging is visible rather than inferred. A gap
produces a **break** in the data — `gap_before_s` — rather than an interpolation.

### time_to_delivery

Does the coupling change as delivery approaches, and differently by class? Both readouts binned on a
0.5 h grid of `epoch / 3600`, class-stratified, on per-GUID values — per-GUID *inside* a window as
well as across the split, so a recording contributing eleven segments to a window cannot outvote one
contributing two.

`pred_gap` is tracked beside the KL because the two fail differently: `pred_gap` is in the decoder's
own units and is immune to the prior-variance inflation. Significance is tested **per window**, with
Holm across windows as one family and pairwise tests on the survivors; the `pooled` row is flagged
`confounded_by_time` and consumed by nothing. `TRAJECTORY_BIN_HOURS` is a module constant, not an
`eval_config` key, for the reason the significance level is not one.

`lag_clocks` resolves the **lag structure** against this same grid and the same classes, so a window here and a window there are the same duration over the same recordings: this one says how much coupling there is, that one says where in the past it came from.

The analysis emits **four** figures — two pages per readout, because `pred_gap` and the unfloored KL share a unit and not a scale, so a page carrying both draws the smaller as a flat line at the bottom of the larger's range. `time_to_delivery_trajectory_<readout>.pdf` is the median line per class with its inter-quartile ribbon; `time_to_delivery_windows_<readout>.pdf` is what that line is made of — a violin per (window, class) cell over one value per recording, the Holm-adjusted $p$ of each window directly beneath it on the same axis, and Cliff's delta for every class pair that survived. The tests were always run; until that page existed nothing drew them.

### second_stage

The **second clinical clock**. The same two readouts, resolved against signed hours from the onset of
the second stage of labour rather than against delivery — because delivery is the end of a process
whose clinically meaningful landmark is inside it, and two recordings four hours before delivery can
be at completely different points of labour.

**The axis is signed and is not negated.** The shard stores `second_stage_onset = domain_start -
t_SSO`, already negative before onset and positive after, so unlike `epoch` — which is stored as time
*before* delivery — it reaches the axis unchanged. Both figures are therefore drawn in the **natural**
orientation with a line at zero rather than inverted the way the delivery clock's are, and the axis
label names the sign convention outright: a reader who took a negative value for "after" would read
the whole trajectory backwards and nothing on the page would contradict them.

**Eligibility: one rule drops a recording, and two diagnostics drop nothing.** A recording with no
recorded onset cannot be placed on this axis and is excluded and counted. The two further ways a
stored onset can be wrong are **counted and filtered nowhere**, and both reach
`second_stage_eligibility.csv` and the record: an implied onset falling *at delivery*, which is what a
pipeline writes when it substitutes zero for a missing time, and an implied onset that *moves* across
a recording's own segments by more than 1 s, which can only come from a broken write. Excluding a
recording changes the population every number is computed over; a count does not.

**The Holm family is this clock's own.** The correction runs across the windows of this clock and is
**not** joint with `time_to_delivery`'s. The two are different alignments of an overlapping
population, so a window significant on one and not the other is a statement about alignment, and the
family-wise error rate each correction controls is within its own clock — a reader combining a claim
from both clocks is making two comparisons.

**It is `capped`, deliberately.** It scores a subset of the evaluated cohort, so it declares
`plan.capped = True` with its reason and is excluded from the coverage block's population comparison
rather than reported there as a disagreement about who was evaluated. The grid is
`TRAJECTORY_BIN_HOURS`, the same 0.5 h windows the delivery clock uses and the same module constant
rather than an `eval_config` key. Recorded skips, each naming its cause: an empty table, a table
collected before the `second_stage_onset` column existed, a cohort with no onset at all, a cohort
whose readouts are all non-finite, and a single-class split.

### events

Contraction-conditioned coupling: does the source matter more when the uterus is contracting? Both
readouts restricted to anchors within `event_lag_window_s` of a detected contraction, against
count-matched control anchors drawn from the same recordings. The contraction timing is computed in
the collection pass and lands on the per-anchor table as `seconds_since_contraction` — it has to be,
since the model reads the source as scattering and phase channels and a contraction exists nowhere in
the tables unless the one pass holding the raw uterine-pressure trace puts it there — so this runs
over every anchor of the split rather than only over retained samples. Guards: at least 200 event
anchors over at least 4 recordings, else a recorded skip.

Gaps are masked by `weight`, never by value. Masking is two steps and both are needed: invalid
samples are interpolated across *before* smoothing, so a gap contributes no edge for the peak finder
to lock onto, and any event whose span touches one is then **dropped**, because its shape partly came
from that interpolation. The contraction onset is a **level crossing** of the peak's own prominence
rather than a gradient walk-back, which is a deliberate correction to the ported detector: a gradient
test stops at the apex, where the smoothed gradient is approximately zero.

**One readout of the raw pipeline's three, and the two that are gone are named in the emitted record
rather than merely missing.** Deceleration forecast skill and the contraction-triggered response both
score a clinical heart-rate trace in beats per minute; this model forecasts 98 coefficients, and
defining a deceleration on a channel axis with no order and no clinical unit is a new scientific
construction rather than a port. `REMOVED_READOUTS` carries both names and both reasons into
`summary.json`, so a reader who expects three meets the absence rather than inferring it from a
missing key.

### sufficiency

What the latent bottleneck costs the forecast:
$\Delta_{\mathrm{suff}} = D_{\mathrm{base}} - D_{\mathrm{oracle}}$, where $D_{\mathrm{oracle}}$ comes
from an evaluation-only decoder of the same capacity reading `target_state` instead of $z$, fitted on
half the evaluation recordings and scored on the other half. The probe emits
$H \cdot C_{\mathrm{keep}}$ per anchor and is scored through the same anchored feature builders as
the model.

**It is an estimate, not a bound**, and both bias directions travel in the emitted JSON rather than
only here. Conditioning on `target_state` rather than on the target's own history omits the encoder's
information loss and biases the gap **down**; fitting the probe on the evaluation population while
$D_{\mathrm{base}}$ comes from a model trained on the disjoint, healthier pretraining cohort biases
it **up**. The two oppose, neither is measured, so nothing downstream may treat the number as a
bound.

The probe's convergence flag is a precondition, not a decoration. Convergence is arithmetic on the
held-out curve, and a curve that never improved is **not** converged. The split is at **GUID** level,
disjointness is asserted at runtime, and the fit budget is expressed in passes over the fit half
rather than in optimizer steps.

### samples

Per-recording diagnostic pages, and the triage that picks which recordings to look at. The page is
this cell's **fifteen-row** page, drawn through the task's own seams (`forecast_rows`,
`forecast_extra_rows`, `input_stream_panels`) rather than a second builder that could disagree with
the one the fit was watched through.

`stratified/` holds a seeded, shard-stratified draw over the whole split, so a cap at or above the
shard count reaches every shard. `by_class/` holds a **class-balanced** draw: the same number of
segments from every clinical class. The two are not interchangeable and neither replaces the
other. The stratified quota follows shard size, so what it renders is what the split mostly
*contains* — which on this cohort means healthy takes most of the pages, and two classes cannot be
compared across it. The balanced draw is the one that supports a comparison and is, by
construction, not representative of anything. Beside them, one directory per headline metric and
tail holds the segments at the extremes of that metric. A page is one segment of one recording: an
illustration, never evidence — and the extreme pages are selected *on* the quantity they display,
so the panel showing it is guaranteed to look unusual and says nothing about how often it does.

**Every selected segment is drawn twice**, from one forward pass. The full page is the fifteen-row
one above; the reduced page beside it, named with a `_compact` tail, keeps five of those rows — the
raw context, the target block as the encoder receives it, the latent state, $K_t$, and the lag
attention on a logarithmic colour scale. It answers what a recording's latent and attention did,
which is a different question from what the model predicted, and eight rows of forecast between
them is what makes the full page slow to read for it. `sample_pages.csv` carries one row per
**file**, with a `variant` column naming which of the two it is, so the manifest indexes the whole
directory rather than half of it.

`eval_config.caps.pages` overrides the stratified count and `eval_config.caps.pages_per_class` the
balanced draw's per-class count; the extremes take ten per tail as an upper bound, lowered wherever
a metric has too few scored segments. The two tails of one metric are disjoint by construction. The
`<index>` in a filename is the position in the evaluation **dataset**, not in `per_sample.csv` — the
collection pass runs under a seeded shuffle — and the two are reconciled by a `guid`/`epoch` round
trip checked before anything is rendered.

### warmup

What the causal front end's warm-up cost this run, per recording and with intervals. This cell's own,
because no other cell in the grid has a channel axis whose members become honest at different times.

**The gap by warm-up tertile.** `pred_gap_warm_lo`, `_mid` and `_hi` split the 98 kept channels into
three tertiles by their rebased warm-up $W'$ and restrict the gap to each. The three recompose to
`pred_gap` over the same denominator, and **the recomposition is asserted rather than described** —
it is the only property that makes them a decomposition rather than three unrelated numbers. The
tolerance is scaled by the block score rather than by the gap, because the gap is a difference of two
block scores of order $10^3$ and a tolerance relative to the difference would tighten without limit
as a model improved.

**The source-lag warmth fractions.** `source_lag_warmth_frac_st` and `_ph`: the attention mass landing
on lags at which each stored source block is warm. **A small value here is the expected finding, not
a fault**, and the emitted record says so: the source blocks' own warm-ups are long, so most of the
searched lag window is a region where the source coefficient is not yet honest, and a model that
attends there is reading what the data offers rather than misbehaving.

**Two geometry guards, and they are the FAIL-able part.** `target_warm_frac` must read exactly $1.0$
and `anchors_per_sample` exactly the checkpoint's own `anchor_ceiling - warmup_period` at the dense
set ($136$ on the stored forecast clock, $51$ under the shipped `physical` one), both computed from
the checkpoint's own geometry rather than from a constant, so a legitimate arm states its own
expectation. A value off
either means the checkpoint predates the constructor's budget-and-floor pairing refusal, or the
anchor geometry is not the one the configuration states — and then every number in the run was
computed over a different population, which is why this fails rather than warns.

Beside them, the warm-up staircase and the budget tradeoff curve, drawn from the model package's own
`warmup_budget.py` and `causal_warmup.py` rather than re-derived.

### source_null

How much of the coupling readout is source *variation*, and how much is a clock. The single most
valuable thing this pipeline adds, because it is the one hazard no other cell has and no existing
control can see.

The source availability pattern $m^u_{t,c}$ is a deterministic function of $t$, identical in every row
of a batch, and it enters $q(z \mid Y, U)$ but not $p(z \mid Y)$ — so the posterior can be pushed off
the prior by the availability **clock** alone, with no source information in it at all. The
permutation control deranges rows, and no permutation of rows can remove something every row shares.

The analysis reports, per recording and bootstrapped over recordings,

$$\Delta_{\mathrm{clock}} = \texttt{source\_conditioned\_kl\_raw} - \texttt{kld\_source\_null},$$

the part of the coupling readout attributable to source variation. The null arm re-runs the source
gate, adapter and encoder from a **zeroed** source stream — not a permutation, because both the
adapter and the encoder are nonlinear and a zeroed stream is not a rearrangement of a real one.

**The threshold is set, at `clock_margin_min_nats: 0.15`, and both cfs cells' override files carry
the same value and the same provenance comment.** It comes from the diagnosed unaligned run's
observed spread of $\Delta_{\mathrm{clock}}$ across recordings — $0.160$, interval
$[0.157, 0.164]$ — so `coupling_exceeds_availability_clock` decides rather than returning
INCONCLUSIVE, and the acceptance gate is ten criteria rather than nine. Its provenance is the
unaligned arm; the gated quantity is right on both arms, which is why one number serves both. It was
deliberately left `null` until a run had measured the spread: a threshold guessed before then would
have decided a FAIL on exactly the runs that were supposed to set it. What was never conditional on
the threshold is the number — `coupling_minus_clock_nats` is a headline scalar whatever the key says,
so the arm tables carried it from the first run, which is what let the threshold be set from data.

**The null arm re-encodes through whichever module `lag_kv_source` selected**, not through a deep
source encoder the model may not have built. That is what keeps it a control: it probes the tensor
the lag attention actually reads.

Two things the emitted record states because each weakens the claim in the model's favour and nothing
else would surface it. **Zeroing floors no source variation**, and the encoder's response to a flat
trajectory is not literally the availability pattern's response — so $\Delta_{\mathrm{clock}}$ is a
slightly *weaker* statement than "the clock alone". And **`kld_source_null` is not expected to
collapse to zero even under `prior_availability_input`**: the posterior is a bounded residual on the
prior, so the mean half of $\mathrm{KL}(q^\varnothing \Vert p)$ is a function of the delta head alone
and no prior-side clock can appear in it. The informative quantity is therefore the difference and
its interval, not the ratio.

**The same difference, resolved by lag.** The null arm already produces its own attention over the
lags — the query is the prior's mean, unchanged, but the keys are the null encode's — so the
head-structured attribution can be built for it from tensors the matched pass already had, at no
extra forward. Subtracting it from the matched attribution bin by bin gives the **clock-excess**
profile, and

$$\sum_\ell \Delta_\ell \;=\; \texttt{coupling\_minus\_clock},$$

which is the scalar `clock_margin_min_nats` gates. That identity is what makes this a decomposition
of the gated quantity rather than a second lag reading that happens to have a clock subtracted, and
it is **measured** on every run: `null_lag_map_sums_to_kl` is a third structural identity in the
sanity block, checked separately from the matched one because the null arm's attention is its own.

This is the only lag profile in a run with the availability staircase removed. The staircase is a
deterministic function of $t$ and is readable from the source state at *any* lag, so it enters the
matched attribution wherever the attention happens to sit; no renormalisation of the matched
profile removes it, and only an arm carrying the clock and no source content can.

`source_null_lag_profile.csv` carries one row per lag — both arms, their signed difference, its
rectified part, the band each lag falls in and whether the delta mask kept it. The `lag` block adds
the run-level selection: the clock-excess argmax and its peak share, the degeneracy verdict, the
rectified fraction, the per-band shares, and the mask — or, where the profile is degenerate, no
mask and the sentence saying why. Four of those reach the headline.

**Two things a reader must carry.** The profile is **signed**, so only the signed sum is the gated
scalar and the rectified total is an *upper bound* on it, larger by exactly the negative mass;
`rectified_frac` is that gap. And the delta mask is **withheld** whenever the clock-excess profile
is degenerate — `entmax15` assigns lags exactly zero, so a flat profile still has a confident
argmax, and a mask cut from one would name a band the run has no evidence for. A withheld mask is
a measurement; the geometry-fixed `occlusion_bands` remain the selection that needs no estimate.

### occlusion

**When did the source matter?** — asked by intervening rather than by reading attention weights. The
observational readouts (`lag_kl`, `attention`) report where a *distribution over lags* puts its mass,
which the window's near edge can pin for reasons that are about the geometry (see `argmax_lag` under
*How the output is misread*). This one removes source content at a chosen lag band and measures what
the forecast loses.

For each band named in `occlusion_bands`, the stored source coefficients in
$[t_a - \ell_{\mathrm{hi}},\ t_a - \ell_{\mathrm{lo}}]$ are set to **zero** — the channel mean, by
the statistics convention every causal shard is built under — the stream is re-encoded through the
run's own K/V path, and the block NLL is re-scored. The reported quantity is the per-horizon-step
change against a reference forward in which nothing was removed.

**Four properties make the number readable, and each is a decision rather than an economy.**

- **The announcement is untouched.** The intervention moves the source's *values* and not its
  arrival clock, which is the exact confound the analysis exists to avoid; the invariance is
  **measured** on every occluded encode and reported, not asserted.
- **The band is occluded after the channel gate.** The gate shifts each channel onto the run's
  common clock, so a band of gated steps is one lag range for every kept channel at once. The same
  band applied before the gate would land at $\ell + d_c$ for channel $c$ and re-smear precisely the
  axis the alignment exists to un-smear.
- **One scored anchor per segment, drawn from a seeded generator over the anchors the forward marked
  valid, and held fixed across every band and the reference.** The source pathway has memory, so a
  band occluded relative to anchor $a$ contaminates the state of every anchor after it, and a second
  anchor scored in the same forward would attribute one anchor's loss to another's band. Holding the
  anchor fixed is what makes the difference *paired*.
- **Common random numbers.** The reference and every band are scored under the same reseeded latent
  noise, so the difference is not a draw apart. Repeated collection is bit-identical.

**The live fraction earns its own column.** A band reaching into the warm-up region, where the
availability mechanism has already zeroed the source, has less source in it to remove — so a small
delta there means "there was nothing there" rather than "the source did not matter". Four headline
scalars therefore reach every arm table rather than one: the winning band's name, its delta, its peak
horizon step and its live fraction.

**It has no verdict, deliberately.** What a healthy per-band delta is has never been measured, and a
threshold guessed before the first production runs would decide a pass or a fail on exactly the run
that was going to measure it — the same argument that kept `clock_margin_min_nats` unset until a run
set it.

Three outputs: `occlusion_per_recording.csv`, `occlusion_per_horizon.csv` and
`occlusion_summary.csv`, plus the per-horizon figure. `occlusion_bands` names the bands as
`{name: [lo, hi]}` in **lag** units; an empty band (`lo > hi`) and a band above the model's own
`max_lag` are both refused by name at config load, the first because a row of zeros would be
reported as a finding and the second because it would name a wider band than it measured. `{}` — the
schema default — records the analysis as a skip. `caps.occlusion` bounds how many **segments** are
re-encoded rather than how many anchors are retained, because this analysis scores one anchor per
segment; removing the key means every segment.

**Choosing `caps.occlusion` from a measurement rather than a guess.** The cap trades wall-clock
against the per-band standard error, and both sides are properties of the machine and the
checkpoint rather than of this pipeline — so the pass measures its own. Two blocks carry it: `cost`
holds the rates, and every row of `bands` now holds `n_segments`, `n_recordings`, `delta_total_se`
and `delta_total_ci_lo`/`delta_total_ci_hi`.

1. Run once against the real checkpoint with `caps: {occlusion: 64}`.
2. Read `results.occlusion.cost.hours_per_1000_samples` and
   `results.occlusion.bands[*].delta_total_se`.
3. Take whichever of the two binds:
   * **time** — `cap_time = target_hours * 1000 / hours_per_1000_samples`;
   * **precision** — `cap_precision = 64 * (se_observed / se_wanted)^2`, because the standard error
     falls as $1/\sqrt{n}$, so halving it costs four times the segments.
4. Set the result in the committed override delta, **never** in `RUN_ARGS`: a value injected from
   Python appears in no artifact and cannot be recovered from the run afterwards.

`cost.seconds_per_arm_per_segment` is the rate to extrapolate with whenever the **band count**
changes, because the work is one encode and one single-anchor decode per *arm* and an arm is the
reference plus one per band; `hours_per_1000_samples` alone would make a band count look like a
property of the dataset. The two counts beside it differ in unit and both are needed: the interval
is taken over **recordings**, for the reason `bootstrap_resamples` states for every other interval
here, while the cap bounds **segments**, because segments are what the loop consumes.

**The deltas are also placed on the two clinical clocks.** `occlusion_clocks.csv` carries, per
band and per window, the mean per-recording delta with its quartiles — the interventional answer to
"did the informative past move", against `lag_kld_scaled`'s observational one on the same partition
and the same grid.

The clinical coordinates come from a **join** onto the collected per-sample table on
`(guid, epoch)`, not from a second read off the batch. `guid` alone does not identify a segment — a
recording contributes many, which is why the collection pass keys its per-anchor table on
`(guid, epoch, anchor)` — and joining picks up the class, the subgroup and the second-stage offset
at once while guaranteeing this analysis cannot disagree with any other about which class a segment
belongs to. Both sides read `epoch` through `metrics.batch_field` and the same `float64` cast, so
the equality is exact; `clocks.n_unjoined` is the tripwire and is zero on a healthy run.

**That page is descriptive only** — no Kruskal-Wallis, no Holm correction, no new family. One
anchor per segment and a cap in segments means a half-hour window holds tens at best, and most
(class, window) cells fall below the minimum group size a test needs; a $p$-value there would be a
correction over cells that mostly could not be tested. Raising `caps.occlusion` is what makes those
cells readable, which is the second reason the cap procedure above matters.

Like `samples` and `sufficiency`, this analysis reaches for `context.task` and `context.loader`, and
records a skip without them. That is structural rather than a convenience: an intervention on the
model's *input* cannot be served by any table, because the tables record a forward the source was
fully present in.

### lag_clocks

Does the lag structure itself move through labour? `lag_kl` says where in the past the source
informed the future, pooled over a whole recording; `time_to_delivery` and `second_stage` say how
*much* coupling there is at each point of two clinical clocks. This says **where** the informative
past sits at each point of both of them, and whether that differs by class.

**Fourteen attributes** of each segment's own profile, on the compensated axis, with
$p_\ell = w_\ell / \sum_k w_k$, in four families:

| family | statistics | column |
| --- | --- | --- |
| moments | centre of mass $\bar\tau = \sum_\ell p_\ell \tau_\ell$, spread $\sigma_\tau = \sqrt{\sum_\ell p_\ell (\tau_\ell - \bar\tau)^2}$, skewness | `lag_centroid_*_s`, `lag_spread_*_s`, `lag_skewness_*` |
| quantiles | median lag, inter-quartile range | `lag_median_*_s`, `lag_iqr_*_s` |
| concentration | entropy $H = -\sum_\ell p_\ell \log p_\ell$, effective support $\Delta e^{H}$, near and far mass share | `lag_entropy_*_nats`, `lag_effective_support_*_s`, `lag_near_mass_*`, `lag_far_mass_*` |
| peak | peak lag, its width at half height, its mass, the degeneracy flag and the zero fraction behind it | `lag_peak_*_s`, `lag_peak_width_*_s`, `lag_peak_mass_*`, `lag_peak_degenerate_*`, `lag_zero_fraction_*` |

The families exist because each answers what the others cannot: a bimodal profile has an
unremarkable centroid and an unremarkable spread, and only the concentration family says it is not
one lump; these profiles are skewed, and one distant bin moves $\bar\tau$ far more than it moves the
median, so where the two disagree the disagreement *is* the skew. Every one is computed twice, over
the untruncated KL attribution and over the support-corrected attention — 28 columns — for the
reason both clocks carry two coupling readouts: the attribution is $K_t$ times the attention and
inherits the prior-variance inflation the attention is immune to, so a shift visible in one and
absent from the other is a finding about which is being read. The arithmetic is
`eval/lag_shape.py`'s, in one vectorised pass per profile.

**The peak is reported, and it is reported with its guard.** `entmax15` assigns lags exactly zero,
so a flat or nearly empty profile still has a perfectly confident argmax, and a position quoted
without the mechanical criterion that says whether the profile has a shape at all is not a reading.
That criterion used to live in `lag_kl` — which an analysis may not import — and this analysis
therefore reported no peak. It now lives one layer down in `eval/lag_shape.py`, so `lag_peak_*_s`
travels beside `lag_peak_degenerate_*` in the same row of the same table and on the same page: a
segment is degenerate when its peak-to-median ratio is below `1.1` or more than `90%` of its finite
bins are exactly zero, and the per-recording mean of that flag is the share of a window's segments
whose peak names a bin rather than a lag. `lag_kl/lag_kl_stratified_peaks.csv` remains where the
*pooled* positional reading lives; this is the per-segment one.

Both clocks are cut on the same `TRAJECTORY_BIN_HOURS` grid the coupling clocks use, the unit is one
value per **recording** inside a window as well as across it, and the second clock scores the
recordings that carry an onset only — the same eligibility rule `second_stage` applies, whose
per-recording table this analysis carries counts from rather than rewriting. It declares itself
`capped` for that reason.

**Four Holm families, and none of them joint**: two clocks times two tested readouts, each
correction controlling the family-wise error rate within its own clock and its own readout. A reader
quoting a window from two of them is making two comparisons and the `method` string of each says so.
**Only the two centroids are tested.** The other twelve statistics are drawn and tabled but carry no
$p$-value, which is what keeps each family at two rather than at fourteen and the tested page at
five rows rather than twenty-nine; a trajectory on the features page that looks separated is a
hypothesis, not a claim. Promoting one is a single `tested` flag in `STATISTICS`.

It emits **six** figures, three per clock: `lag_<clock>.pdf`, the share of the attribution by lag and
window with one panel per class on a shared colour scale and the two tested centroid trajectories
beneath it; `lag_<clock>_windows.pdf`, the violins, the Holm-adjusted $p$ per window and Cliff's
delta for every class pair that survived; and `lag_<clock>_features.pdf`, the untested statistics
one panel each, solid for the attribution and dashed for the attention. The third is its own page
rather than more rows on the first because half of what it draws is not in seconds, and a panel
sharing a figure with a quantity in different units is a panel that will be read against it.

### lag_kld_scaled

**The same lag structure, read on the lags that carry the coupling and with its magnitude kept.**
`lag_clocks` resolves the profile against both clocks over **all** $L$ lags and through statistics
that are functions of $p_\ell = w_\ell / \sum_k w_k$ alone. On this family both are limitations
rather than conventions: two thirds of the attribution is an availability clock readable at every
lag, the scale that would distinguish "the informative past moved" from "there is less of it" is
divided out, and the heads are averaged into one profile that one latent group dominates. This
analysis is those three answers, on the same $0.5$ h grid, **beside** `lag_clocks` rather than
replacing it — that analysis's columns are untouched.

**Four families of source, and the first two are the selection.**

- **The geometry-fixed bands**, from `occlusion_bands`. Nothing about them is estimated from the
  KL, so a statistic on a band is free of the circularity that makes a top-$K$-by-KL selection test
  its own selector. They are also the *same* partition `occlusion` removes source from, so a band
  names one lag range across the run and the observational and interventional pages are read
  against each other by filtering rather than by aligning two four-way splits by eye.
- **The soft weight**, $\omega_\ell = \Delta^+_\ell / \max_k \Delta^+_k$, from the pooled
  clock-excess profile `source_null` reports. Computed **once at run level** and applied
  identically to every segment, window and class — a per-segment weight would let each segment
  choose its own lag axis, and a comparison across segments would then compare different axes.
  Withheld entirely when that profile is degenerate.
- **The full support**, carrying `total_nats` and `peak_nats` only. The twelve scale-free
  statistics on the full support are `lag_clocks`' own columns and are not restated here.
- **The heads**, each head's own $K^{(m)}\alpha^{(m)}_\ell$, which sums over $m$ to the pooled
  attribution exactly.

**`near_mass` and `far_mass` are absent from every banded source, and the absence is a
measurement.** Both are measured from the axis's own start, so on a band they would silently mean
"within `NEAR_SECONDS` of *the band's* start", and `far_mass` would be identically zero on any band
narrower than `FAR_SECONDS` — three of the four shipped ones. Four columns of structural zeros
presented as measurements is worse than four absent columns.

**Nothing here is tested.** Every feature ships untested, so this analysis adds **no** Holm family
to the four `lag_clocks` carries, and it writes no significance or pairwise table at all. At the
clock-exceeding coupling this family has measured — $0.160$ nats over $91$ lags on the diagnosed
run — per-segment restricted centroids are very likely noise, and correcting eight new families
over noise is how a family-wise correction stops being believed. The record says so in
`no_inference_note` rather than leaving a reader to infer that a $p$-value was withheld; promoting
a feature is one flag.

**The emission is long-form**: `source` and `statistic` are row keys, not columns. That is what
lets `num_heads` be a run property and a band be added without widening a table.

Three outputs — `lag_kld_scaled_per_recording.csv`, `lag_kld_scaled_trajectory.csv` and
`lag_kld_scaled_selection.csv` — plus one figure per clock. The selection table is the run's
durable record of which lags were kept and with what weight: a selection reconstructed later from
a re-run is not the selection the numbers beside it were chosen with.

`occlusion_bands` is therefore read by **two** analyses. Emptying it to skip the interventional
pass also removes this analysis's selection, and this one records a named skip rather than
silently emitting its unrestricted half.

### lag_high_kl

**The lag structure of the anchors that carry the coupling, selected by their own $K_t$.** Every
other lag readout averages a segment's anchors together before it reads a lag position, and on this
family that average is dominated by anchors whose KL is the availability clock and little else: the
diagnosed runs put $67$–$92\%$ of the coupling readout in a term that survives zeroing the source,
spread over every anchor. The anchors at which the source actually informed the future are the
minority with a large $K_t$, and their lag profile is what the pooled one dilutes.

The selection is a **quantile band of the pooled per-anchor KL** — pooled over every scored anchor
of every clinical class within the run's horizon, so one run has one threshold in nats and every
segment, window and class is cut by the same number. Three bands ship, as module constants rather
than `eval_config` keys for the reason the significance level is not one: `high`, the upper $30\%$
($q \in [0.7, 1]$); `rest`, its complement, so the two recompose to every anchor and a contrast is
against everything unselected; and `top`, the upper $10\%$. Per segment and per band: the **share
of the segment's anchors** in the band, and the **lag profile restricted to those anchors** — the
mean per-anchor KL attribution $\widetilde K_{t,\ell} = \sum_m K^{(m)}_t \alpha^{(m)}_{t,\ell}$ over
the selected anchors, and the head-averaged attention over the same anchors — reduced through the
shared shape vocabulary of `lag_shape.py` (centroid, spread, median, IQR, entropy, effective
support, near and far mass, peak with its guard, and the two nats-scale totals). Both are resolved
against **both clinical clocks** on the same $0.5$ h grid, by class.

**It reads a third sidecar the collection pass writes for it.** `per_anchor_vectors.npz` carries the
pooled KL attribution and the head-averaged attention at every contributing anchor, row for row with
`per_anchor.parquet`, in `float16`. No per-sample profile can stand in for it — a per-sample profile
*is* the average over anchors this analysis exists to undo — and the per-anchor `argmax_lag` column
cannot either once a profile is flat. A directory collected before the sidecar existed records a
named skip; re-collect to produce it.

**Exactly two readouts are tested, on two clocks: four Holm families, none joint.** The high band's
KL centroid `high_lag_centroid_kl_s` and the high-anchor share `high_anchor_frac`, per window with
Kruskal–Wallis across classes, Holm across that clock's windows, and pairwise Mann–Whitney with
Cliff's delta on the survivors — the same three layers and the same severity orientation every clock
analysis uses — plus the one run-level paired usefulness test below. Everything else ships
**untested** and the record says so: the `rest`, `top` and `gain` bands' clock trajectories, every
attention-profile statistic, the hot-lag shares, the decile, argmax and occlusion-join tables and
the contraction enrichment.

**Three further readings come from the same selection.**

- **Hot lags.** The lags whose *pooled* attribution — over every anchor of the population — sits in
  the upper $30\%$ across the $91$ lags. A run-level set, recorded lag by lag in
  `lag_high_kl_selection.csv`, and the per-segment share of attribution landing on it is placed on
  both clocks beside the band readouts. **This is the top-$K$-by-KL selection `lag_kld_scaled`
  declines for its own bands, taken here deliberately and with the circularity stated on every
  artifact**: the set is chosen from the same attribution it then summarises, so a share on it
  describes the run's own selection and is not an independent test of it. The selection is pooled
  over every class, which is what keeps a *class contrast* on it honest — no class chose its own
  lags. The geometry-fixed bands and the occlusion readout remain the selections that need no
  estimate.
- **Where the KL sits on the lag axis, by KL magnitude.** The per-anchor `argmax_lag` against the KL
  decile of the same anchor, pooled and per class (`lag_high_kl_argmax_by_quantile.csv`). A flat
  picture across deciles says the argmax is a property of the geometry rather than of the coupling;
  a picture that moves says which lags the coupling actually lives at.
- **Contraction enrichment.** Whether high-KL anchors are more common within `event_lag_window_s` of
  a detected contraction than outside it, per recording (`lag_high_kl_contraction.csv`) and
  summarised by class — the coupling-magnitude counterpart of `events`, on the same per-anchor
  contraction age and with no extra pass. A recording's difference is reported only with at least
  five anchors in **each** arm; below that a share is a coin toss and the row says `reportable =
  False`.

**Whether any of it is *useful* is asked directly, in forecast space, and it is the question the
analysis exists to settle.** A large $K_t$ says the source moved the belief; it does not say the
forecast got better. The per-anchor table carries the Monte Carlo forecast gain of the same anchor,
`mc_pred_gap` $= D_{\mathrm{base}} - D_{\mathrm{full}}$ in nats (the single-draw `pred_gap` on a
pass without it), and the selection is scored by it four ways:

- **Per band, the mean gain of its anchors** — `high_pred_gap_nats`, `rest_pred_gap_nats`, … — per
  segment, per recording and on both clocks. The high band's against the rest band's is tested
  **once, paired within recording** by a Wilcoxon signed-rank test over recordings with a bootstrap
  interval on the mean difference; it is its own family of one and is not corrected with the four
  clock families. Positive means the anchors carrying the coupling are the anchors where the source
  bought forecast; zero or negative means the KL is not where the usefulness is.
- **A fourth band, `gain`** — the anchors in the upper $30\%$ of the pooled forecast gain, selected
  on usefulness rather than on KL — with its own lag profile and statistics, and its **overlap with
  the high band** against the $30\%$ that independence would give (`share_of_high_in_gain`,
  Jaccard). Two selections naming the same anchors is the finding; two that do not is the other one.
- **The gain resolved by KL decile** (`lag_high_kl_gain_by_kl_quantile.csv`, per recording then by
  class) **and by the anchor's argmax lag** (`lag_high_kl_gain_by_argmax.csv`, pooled and per band):
  whether more coupling buys more forecast, and whether the lag the attribution names is a lag the
  forecast profits from.
- **A gain-weighted attention profile**, $\sum_t \max(g_t, 0)\,\alpha_{t\ell} / \sum_t \max(g_t,
  0)$ — where the source looks *when it helps* — beside the KL-weighted one on the selection table.

**Observational against interventional, on one partition.** When the `occlusion` analysis has run in
the directory, `lag_high_kl_occlusion_consistency.csv` joins, per recording and per geometry band of
`occlusion_bands`, the share of the recording's KL attribution inside the band (all anchors, and the
high band's) with the forecast cost of occluding that band, and records a descriptive Spearman's
$\rho$ per band. Positive means the lags the attribution names are the lags the forecast used; near
zero means the two readings disagree about where the source mattered. The dependency is on the file,
so `--only lag_high_kl` against a directory whose interventional pass never ran records a skip.

**Eight headline scalars reach every arm table**: the pooled high threshold in nats, the high band's
centroid and total nats as means over recordings, the hot-lag count and the hot-lag share; and the
usefulness three — the high band's mean forecast gain, its paired difference against the rest band's,
and the high–gain overlap share. The threshold comes first because every other number is conditional
on it — two arms with different thresholds selected different anchors. `lag_high_kl_recordings.csv`,
one row per recording over the whole population, is the source `cross_subgroup` reads
`high_anchor_frac` from.

**It is `capped`**, for the reason `lag_clocks` is: the second-stage half scores the recordings that
carry an onset only, by the shared eligibility rule. Both clocks' per-recording and trajectory
tables, the per-window restricted profiles, the significance and pairwise tables, and six figures
— a run-level selection page, a run-level usefulness page and, per clock, a profile-and-trajectory
page and a tested page — are the outputs. The axis is stored-coefficient time and every one of them carries the caveat.

### spectral_skill

The forecast gap resolved by the frequency band of the target coefficient. **The channel axis of this
target domain is a frequency axis, and this is the analysis that uses it**: a first-order scattering
coefficient is $|x \star \psi_\lambda|$, the envelope of the signal filtered at one centre frequency,
so the frequency resolution the raw cells build with a Welch periodogram is already present here, per
channel, for free.

What is reported is how well the model forecasts the envelope in each clinical band, per recording
and bootstrapped over recordings, in both the likelihood space the objective is stated in and the
error space that has a natural zero. The band gaps recompose to `pred_gap` under the same guard the
warm-up tertiles use.

**It is band-resolved skill, not coherence, and the difference is not a technicality.** A stored
scattering coefficient is a *modulus*: the analysing filter's phase was discarded before the value
was written. So the three things the raw pipeline's `coherence` exists to separate — phase agreement,
group delay, and the exact three-way split of the residual spectrum into irreducible, timing and
amplitude terms — have **no analogue here at any window length**. What this readout says is how well
the forecast reproduces each band; what it cannot say is whether a forecast is mistimed rather than
mis-scaled. It is named `spectral_skill` and not `coherence` so that a reader who knows the raw
pipeline cannot carry the wrong contract across.

Two further limits belong beside it. The band is the band of the **analysing filter**, not a bin of
the forecast's own spectrum. And a phase-harmonic channel has a *pair* of frequencies; it is banded by
`band_partition`'s own `freq_hz_primary` convention, which the emitted record states rather than
assumes.

**One join is the whole correctness risk of this analysis.** The band map is over the **102 declared**
channels; the gap vector is over the **98 kept** ones. Joining them positionally would shift band
membership across the axis — and on the shipped dataset the four dropped channels happen to be the
trailing four, so a positional join would look right here while being wrong on any dataset whose
survivors are not a prefix. A join that is accidentally correct is worse than one that is wrong,
because no test would catch it. The join therefore goes through the **persisted kept-axis map**
`band_channel_map_kept.csv`, read off disk, never through `model.target_gate` — `analyses/*` is layer
2 and may not touch the model at all, and the context's task is `None` on exactly the path this has to
work on.

The record emits **five counts rather than one ratio**: `declared_total` 102, `dropped_declared` 4
(all `unknown`-band), `kept_total` 98, `known_kept` 95, `unknown_kept` 3. The declared and scored
numerators coincide at 95 by the arithmetic accident $102 - 7 = 98 - 3$, and quoting "95 of 102"
would imply the analysis scored channels the decoder never emitted.

### cross_subgroup

Do the cohorts actually differ, or does the by-subgroup table only look as though they do? Eight
cohorts each with a mean always produce a highest and a lowest; with eight metrics that is sixty-four
numbers, and some will look separated whether or not anything is there.

Three layers, in order, and the order is the point: a Kruskal omnibus per metric, Holm **across
metrics as one family**, and pairwise Mann–Whitney with Cliff's delta (Romano magnitudes) on the
survivors only. Every pair is named more severe first — HIE vs acidosis, HIE vs healthy,
acidosis vs healthy — so a positive $\delta$ means the more severe cohort's values run higher.
Every test consumes one value per **recording**, and a test asserts that no source
names a `per_sample` file.

It reads finished per-recording CSVs off disk through a `METRIC_SOURCES` table — which here gains
this cell's own sources, the warm-up tertiles, the source-null difference and the band-resolved skill
— so a missing source is **recorded** rather than raised, which is what keeps
`--only cross_subgroup` working against a finished directory with no checkpoint. It self-skips below
two testable groups.

## How the output is misread

These are the readings the numbers invite and do not support.

**The forecast claim is exact, and the lag claim is not.** The stored coefficients come from a
strictly one-sided bank, so a coefficient at step $t$ is a function of $\{x(s) : s \le t\}$ alone and
forecasting step $t + 1 + \tau$ from history up to $t$ is a genuine forecast. That is what separates
this cell from the four two-sided ones and it needs no hedging. What still needs hedging is the *lag*
readout: the coupling number is named `source_conditioned_kl_raw` and the disclosure refuses the name
it is not, because the lag map is an attribution over **stored-coefficient time**, uncorrected for a
composed one-sided group delay reaching 791 s — the same order as the 364 s lag search itself. Every
run carries that sentence verbatim in `preflight.json` and `summary.json`, every lag-resolved artifact
and figure carries the caveat, and `tests/test_eval_naming.py` scans the whole artifact tree — plus
this file and the figure guide — for the name the readout refuses.

**A lag position is a coefficient-time attribution, not a physiological delay.** The compensated lag
$\tau = 4(\ell + \delta)$ corrects only the model's own input delay $\delta$, read from
`model.source_delay_steps` and nowhere else. The per-channel composed group delay is *not* corrected
for, and cannot be from this readout: the correction is per channel **pair** while the lag map is per
head over a pooled source state, so the mapping would itself be an unvalidated construction. Both
`DESIGN.md` records keep that limitation open. What the dual alignment reference *does* remove is the
inter-stream part of that bias: with both streams on their own clocks the residual between them is a
single known constant, printed on the console block beside the delay.

**`argmax_lag` at the smallest attainable lag is a censoring reading, not an inertness reading.** The
lag window has two censoring edges and not one. A profile pinned at the **far** edge means the model
would report a lag the window is too short to express, and that is a FAIL; a profile pinned at the
**near** edge means it would report a lag *shorter* than the window's own arithmetic can express,
which at this geometry is where any delay below roughly $30$ s lands — and that is INCONCLUSIVE, with
the physical-lag identity stated in the message so a reader can check the arithmetic rather than
trust the verdict. The near edge is `min(attainable)` read from the per-lag anchor counts, symmetric
with the far edge, so a window whose lowest bins carry no anchor lifts the floor off zero rather than
being read as inertness.

**Whether the machinery is alive is judged from the shape vocabulary, not from the argmax.** A
degenerate profile — one whose peak is not distinguishable from its bulk — FAILs at either edge or in
the middle, because its argmax names a bin rather than a lag; that is decided first, from
`lag_shape`'s degeneracy flag, the peak's width and the mass above half the peak, with the per-head
entropies beside them. An ideal model at this geometry, peaking strictly inside the attainable range,
PASSes. **The pooled argmax is not the surface an arm comparison is read on**: the per-head profiles
are, and they are printed under the pooled row with each head's argmax, peak width, mass above half
peak, near and far mass, attention entropy and KL share.

**The run's arm is printed beside the delay, and in three readings rather than one.** The console
block and `summary.run_arm` carry the *configured* `causal_align_reference` label and source
reference, the *built* `lag_kv_source`, and the *resolved* target clock, source clock and
inter-stream offset in seconds. Three readings because a config naming one arm while the checkpoint
carries another is exactly the failure this line exists to catch, and merging them would hide it.

**Specificity is read in prediction space, not in KL space.** See `perm_control`:
$K_{\mathrm{shuffled}} > K_{\mathrm{true}}$ is what a healthy model does, so a KL-space criterion
would fail exactly the models it should pass.

**A coupling readout is not yet a source finding: read `source_null` first.** The availability clock
is a hazard the permutation control structurally cannot see, and the difference between the two
readouts is the part of the coupling attributable to source *variation*. Until
`coupling_exceeds_availability_clock` has a threshold it reports INCONCLUSIVE — which means the
measurement is there and the criterion is not, and the number rather than the status is what to read.

**Only the unfloored KL may be read as a rate.** `source_conditioned_kl_train` has free bits applied
per dimension per step before summing, so it exceeds the raw value by construction and hides a
collapsed source pathway. The shipped `free_bits: 0.0` makes the two coincide today, which is exactly
why the distinction lives in code: no headline path may resolve to it, asserted by test.

**Only an unpinned prior variance makes that rate meaningful.** A prior variance on its clamp inflates
every coupling number while every decoder-side diagnostic stays healthy. Read
`prior_variance_not_pinned` before quoting the KL.

**A small source-lag warmth is the expected finding rather than a fault.** The stored source blocks
warm up late — `up_ph` not before step 56 on the committed fixture — so much of the searched lag
window is a region in which the source coefficient is not yet honest. `warmup`'s emitted record says
so beside the number, and a reader who treats a low fraction as a defect is reading the dataset's
geometry as the model's behaviour.

**The percentage is budget-local, and the nats are one step removed from it.**
`pred_gap_mc_likelihood_pct` divides by $H \cdot C_{\mathrm{keep}}$, and $C_{\mathrm{keep}}$ is what
the warm-up budget decides — so two arms of this model at two budgets have non-comparable
percentages as well as non-comparable nats, and mutually unloadable checkpoints besides. Nothing tries
to normalise it away.

**A `nll_*_sample` key is a fixed /2940 rescale of a block score**, not a mean over the coefficients
that were actually scored, so on any anchor with masked forecast steps it under-reports. It ships
beside that statement or not at all.

**The percentage is never `pred_gap` divided by a block score.** $D_{\mathrm{base}}$ is a negative log
*density* summed over 2940 coefficients: it has no natural zero, it is legitimately negative for a
sharp forecast, and the ratio therefore changes sign with its own denominator. The percentages this
pipeline emits live in the two spaces that have a natural zero — error space and likelihood space.

**Anchors are not independent, and every statistic is per recording.** Consecutive anchors' forecast
windows overlap in 14 of their 15 horizon steps and one GUID contributes many segments, so per-segment
$p$-values are anticonservative by that factor. The chain is: per anchor → support-weighted mean
within a segment → unweighted mean over a GUID's segments → across GUIDs. A segment scoring zero
anchors is excluded and **counted**, never averaged in as `0.0`.

**A per-segment histogram is a description, not a cohort difference.** `distributions` computes no
test on purpose; a separation visible there is a reason to look at `cross_subgroup`, which answers the
question on per-recording values.

**Every class contrast is out-of-distribution, and the scope is wider than it looks.** The checkpoint
trains on healthy-**with-background** only, so ACIDOSIS and HIE are unseen — and so are the
`healthy_no_bg_cs` and `healthy_no_bg_no_cs` subgroups. The summary computes
`training_cohort_disjoint` from both resolved dataset lists rather than asserting it, reports `null`
rather than `false` where a list is absent, and suppresses the out-of-distribution sentence when the
two overlap.

**An eval score is not comparable with a `test_*` metric logged during training**, and a loss level
here is **not comparable** across the target axis either. Against `lag_attn_fs` the blocks differ
(2940 against 2340 coefficients), though the horizons no longer differ; against
`lag_attn_transformer_cfs` it *is* comparable, because both cells sum the same 2940 coefficients over
the same anchor count under the same objective. The cross-cell table carries only the second
comparison for that reason.

**The sufficiency gap is an estimate, not a bound.** Both bias directions, above.

**The frequency statement has no timing half.** `spectral_skill` says how well each band is
reproduced and cannot say whether a forecast arrives a step late; a forecast that is right in every
band but mistimed reads here as a forecast that is right. Reviving the raw pipeline's construction is
not the fix and is recorded so nobody tries: a $\tau$-slice on this grid gives 136 samples at 0.25 Hz
per channel, Nyquist $0.125$ Hz, over band-limited envelopes rather than one trace — and the phase
the estimator needs was discarded before the coefficient was stored.

**A cohort's colour here is not its colour on a training figure.** See *Cohort order and colour*.

**There are two clinical clocks and two independent families, and they are corrected within a clock
rather than across.** `time_to_delivery` resolves the coupling against delivery and `second_stage`
resolves it against the onset of the second stage; each runs its own Holm step-down across its own
windows. That is deliberate: the two are different alignments of an *overlapping* population, so a
window significant on one clock and not the other is a statement about alignment rather than a
contradiction — and a reader who quotes a claim from both clocks has made two comparisons and is
corrected for neither against the other. Within one clock the two readouts are not jointly corrected
either, because they are two readings of the same recordings rather than two hypotheses.

**`second_stage` scores a subset of the evaluated cohort, which is why it is `capped`.** A recording
the labour-onset table has no second stage for cannot be placed on that axis at all, so it is
excluded and counted; the analysis reports the eligible segment count with `plan.capped = True` and
its reason, and the coverage block therefore leaves it out of the population comparison instead of
reporting it as two analyses disagreeing about who was evaluated. Its `n_samples` is **not**
comparable with any other analysis's, and `second_stage_eligibility.csv` is where the difference is
accounted for, recording by recording.

**A stored onset that is wrong is counted and never dropped.** Two of them are measurable and both
are reported rather than filtered: an implied onset falling *at delivery*, which is what a pipeline
writes when it substitutes zero for a missing time, and an implied onset that *moves* across a
recording's own segments by more than the 1 s float32 tolerance, which can only come from a broken
write. Excluding those recordings would change the population every number on that clock is computed
over while the numbers themselves went on looking ordinary; a count does not. If a later reading
shows they distort the trajectory, the eligibility rule is one predicate away — and the count is what
would say so.

## The divergence register

The fork's third anti-drift measure, and the one that makes the other two auditable. Every module of
`teb_vae/lag_attn_rws/eval` has exactly one entry in the committed `divergences.json` beside this
file, classified `equivalent` (must stay behaviour-equivalent, and at least one named assertion in
`tests/test_eval_sibling_agreement.py` exercises it), `divergent` (deliberately differs, with the
reason recorded) or `absent` (not ported at all, and the file is asserted **not** to exist here). A
module that is neither classified nor absent fails a test — which is the case a prose register cannot
catch, because nobody notices a paragraph that was never written.

**The list below is rendered from that file rather than kept by hand**, and
`tests/test_eval_docs.py` asserts the two agree verbatim.

- **`__init__.py`** (*divergent*) States that this package is a fork of the raw pipeline, why the four-field ModelBinding did not reach a target-domain change, and the four anti-drift measures that travel with the fork. The sibling's docstring describes a pipeline that is nobody's fork.
- **`_reuse.py`** (*equivalent*) Behaviour-equivalent to the sibling module, exercised by `test_eval_reuse.py::test_the_bound_names_are_exactly_the_siblings`, `test_eval_reuse.py::test_every_bound_name_resolves_to_the_same_object_both_packages_see`.
- **`analyses/__init__.py`** (*equivalent*) Behaviour-equivalent to the sibling module, exercised by `test_eval_sibling_agreement.py::test_the_analysis_protocol_is_the_same_contract_in_both_packages`.
- **`binding.py`** (*divergent*) The ModelBinding dataclass is unchanged, but this copy also carries the concrete CFS_BINDING and its GEOMETRY_KEYS -- sixteen against the sibling's fourteen, adding anchor_stride and lag_floor -- so it names a model class and sits at layer 1 here against layer 0 there. The sibling keeps its instance in run.py; this cell's binding is reconciled against preflight long before either package has a runner, which is what makes a key that names nothing cheap to find.
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
- **`run.py`** (*divergent*) Registers twenty-one analyses rather than seventeen, defaults to CFS_BINDING, records the dense anchor geometry and the training stride in run_context, and registers no coherence step.
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
- **`analyses/forecast.py`** (*divergent*) The three trivial baselines are rebuilt in feature space on the decimated grid and the horizon curve runs over 30 steps. Every column in a clinical unit is removed rather than repointed -- a wavelet modulus has no clinical unit and inverting the per-channel statistics would put the 98 scored channels on scales spanning orders of magnitude, which destroys the pooled mean squared error and the skill ratio -- so the error table reports the z-unit columns alone. The forecast overlay draws three kept channels against lead time rather than one trace, because what is forecast is an H x C_keep block, and it indexes the retained anchor axis by position rather than by decimated step: this model gathers its anchors, so the floor of 134 is not a valid index into a 136-long axis.
- **`analyses/lag_kl.py`** (*divergent*) All three profiles are kept -- raw, support-corrected and untruncated -- and the analysis MEASURES the truncation rather than asserting it inert: it reads preflight's own lag_support_margin_steps, measures the per-lag contributing-anchor counts, compares the three profiles, and records whether the computed and observed readings agree. The axis is relabelled stored-coefficient time and the group-delay caveat travels on every artifact that states a lag position and under the figure.
- **`analyses/latent.py`** (*equivalent*) Behaviour-equivalent to the sibling module, exercised by `test_eval_sibling_agreement.py::test_the_latent_spectrum_is_laid_out_the_same_way_in_both_packages`, `test_eval_sibling_agreement.py::test_the_latent_diagnostics_are_the_same_thirteen_reductions_in_both_packages`, `test_eval_sibling_agreement.py::test_the_only_gloss_that_differs_is_the_one_naming_this_target_domains_unit`.
- **`analyses/perm_control.py`** (*equivalent*) Behaviour-equivalent to the sibling module, exercised by `test_eval_sibling_agreement.py::test_the_specificity_outcome_is_named_the_same_way_in_both_packages`, `test_eval_sibling_agreement.py::test_the_three_paired_controls_keep_the_same_sign_convention_in_both_packages`, `test_eval_sibling_agreement.py::test_the_branch_summary_and_the_kl_description_agree_in_both_packages`.
- **`analyses/residual.py`** (*divergent*) The forecast-difference RMS is reported in z units rather than in a clinical unit; the two latent quantities and the Jensen pair are unchanged.
- **`analyses/samples.py`** (*divergent*) Draws this cell's fifteen-row diagnostic page through the task's own page seams rather than the raw cells' nine-row one, and draws every selected segment twice: the full page and a reduced five-row one beside it, off a single forward, whose lag attention carries a logarithmic colour scale. Renders a third selection the sibling has none of -- a class-balanced by_class/ draw of caps.pages_per_class segments from every clinical class, beside the shard-proportional stratified/ one -- and records one manifest row per file with a variant column rather than one per segment.
- **`analyses/second_stage.py`** (*equivalent*) Behaviour-equivalent to the sibling module, exercised by `test_eval_second_stage.py::test_the_analysis_writes_its_five_tables_and_both_figures`, `test_eval_sibling_agreement.py::test_the_second_stage_clock_reaches_the_same_verdicts`, `test_eval_sibling_agreement.py::test_the_two_second_stage_modules_are_one_file_with_two_import_lines`.
- **`analyses/sufficiency.py`** (*equivalent*) Behaviour-equivalent to the sibling module, exercised by `test_eval_sibling_agreement.py::test_the_sufficiency_scores_and_both_gaps_are_defined_identically_in_both_packages`, `test_eval_sibling_agreement.py::test_the_oracle_score_join_and_its_summary_rows_agree`.
- **`analyses/time_to_delivery.py`** (*equivalent*) Behaviour-equivalent to the sibling module, exercised by `test_eval_sibling_agreement.py::test_the_time_before_delivery_grid_and_its_readouts_are_the_same_in_both_packages`, `test_eval_sibling_agreement.py::test_the_three_layers_of_inference_reach_the_same_verdicts`, `test_eval_sibling_agreement.py::test_the_two_time_to_delivery_modules_are_one_file_with_two_import_lines`.
- **`analyses/trajectory.py`** (*divergent*) The within-segment structural caveat changes: nothing below the anchor floor F = 134 exists at all, so the profile starts there rather than showing a warm-up droop.

Two modules outside the fork are deliberately **imported** rather than copied, and both sit outside
the sibling's `eval/` package so the layering rule reaches neither:
`teb_vae/lag_attn_rws/collapse.py::is_collapsed`, which is stdlib-only and keeps `verify.py` free of
`torch`, and `teb_vae/lag_attn_rws/trainer.py::RESOLVED_CONFIG_FILENAME`.

## Operations

### Exit codes

The exit code is non-zero **if and only if a step raised.** Three things deliberately do *not* move
it:

- a failed **sanity** check — the self-consistency block warns, logs at ERROR and leaves the code at
  0, because a run whose every step succeeded can still be one nobody should quote a number from;
- a **coverage** warning (two analyses reporting different populations);
- an **inert-cap** warning (a cap no analysis read).

That asymmetry is exactly why the offline acceptance gate exists separately: `verify` reads the
sanity block and refuses on it.

### What a refused run leaves behind

Preflight runs **outside** the fail-soft step wrapper and raises `EvalPreconditionUnmet` — a distinct
type, so a reader seeing it in a traceback is looking at a refusal with an actionable message rather
than at a crash inside an analysis. A refused run leaves `resolved_config.yaml` and `eval.log`
carrying the refusal, and **no `summary.json`**: a rejected input must not produce a file that reads
like a result.

### Re-running one analysis

```bash
python -m teb_vae.lag_attn_cfs.eval.run --output-dir <a finished run> --only lag_kl
```

reads the tables, builds no model and touches no GPU. It is **non-destructive but not additive**. The
prior `summary.json` and `steps.json` are renamed aside before the new pass writes — byte identical,
with the backup path logged — but the new summary describes **only what this pass ran**: its headline
is mostly `null`, its manifest lists a handful of files, and its exit code is 0. So **read the backup,
not the new summary, for anything the re-run did not itself produce.**

`preflight.json` is deliberately *not* preserved: a pass with no checkpoint cannot regenerate the
causality disclosure and reads that file back instead. The tables carry a provenance sidecar naming
the checkpoint hash, the seed, the row count and the `eval_config` digest; a mismatch raises
`TablesProvenanceMismatch`, because the tables are intact and readable and simply belong to another
run.

### Guard recovery table

One row per way preflight refuses a run, keyed by the guard function that raises. Each refusal's own
message names the fix; this is the index, and it is not hand-kept — `preflight.GUARD_RECOVERY` is a
module-level mapping, `tests/test_eval_preflight.py` walks the module's AST and asserts that every
raise site has a row, and `tests/test_eval_docs.py` asserts that every row appears here.

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

No new ones. `torch`, `numpy`, `pandas`, `matplotlib`, `h5py`, `pyyaml` and `loguru` are already in
use; `scipy` is imported lazily at each call site; `pyarrow` is pinned in `requirements.txt` and
carries the per-anchor table, so there is one format and no fallback branch. `verify.py` needs none of
them except `pyyaml`, and that only for the arm tables — the acceptance gate itself is a stdlib parse.

### The gate

From the repository root:

```bash
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_cfs/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_cfs/tests -q -m slow
```
