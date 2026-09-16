# The evaluation contract

What a run of `teb_vae.lag_attn_transformer_rws.eval` is, what it leaves behind, what each
analysis means, and how the output is misread. `FIGURE_GUIDE.md` beside this file documents every
emitted PDF; this document is everything that is not a figure. Both are bound to the code by test:
every registered analysis has a heading here, every resolved `eval_config` key is mentioned here,
every way a run can be refused has a recovery row here, and every figure in the committed
`figure_manifest.json` has a guide entry.

**This document stands alone, and the cost of that is stated rather than hidden.** Almost the whole
pipeline is imported from `teb_vae/lag_attn_rws/eval`, whose own `EVAL.md` describes the same
seventeen shared analyses. Two documents now say the same thing about them and are kept equal by
review. What is mechanically bound is the *code*: an analysis added or removed, a config key
renamed, a refusal added, a figure emitted or dropped fails a test in both packages. A prose
divergence does not, so the sections that genuinely differ are enumerated here rather than left for
a reader to find — the launch commands, the run layout, the four layers table, `encoder_attention`,
the arm and cross-model tables, the encoder half of the causality disclosure, and the non-goals.

## What a run is

One command reads one checkpoint and writes one reviewable directory:

```bash
python -m teb_vae.lag_attn_transformer_rws.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt
```

**Or with no command line at all.** Every entry point here — `run`, `probe`, `verify` — ships a
`RUN_ARGS` dict immediately above its `__main__` guard, keyed by the same names the flags carry.
Fill in `checkpoint` (or `output_dir`, for an offline re-run against a finished directory) and press
Run in the IDE. The two routes are both first-class and resolve **per key**, so `--checkpoint
other.ckpt` overrides that one value and leaves the rest of the dict standing; `summary.json`
records which source supplied each argument. `RUN_ARGS` carries only what the command line already
accepts — anything that shapes what the run *measures* belongs in the override delta below, which is
dumped into the run directory and is therefore the durable record.

The configuration is the checkpoint's own `resolved_config.yaml` — found beside it, never a second
config file — with this package's committed `eval/configs/eval_overrides.yaml` delta deep-merged
over it. That delta is the sibling's, key for key and value for value, and a test asserts it: the
two architectures exist to be compared, so a holdout split or a Monte Carlo draw count that differed
between them would make every side-by-side number a comparison of two protocols rather than of two
models. It repoints the shards at the shared k-fold holdout split, adds the five clinical
`load_fields`, and carries the `eval_config` block; both the original and the merged value of every
overridden key are recorded in the summary. Preflight then refuses the run outright when the merged
result contradicts the checkpoint.

The expensive part happens once. A single shared collection pass decodes four latent branches over
every anchor at $K$ Monte Carlo draws and writes two durable tables — `per_sample.csv` (one row per
segment, with the clinical labels attached) and `per_anchor.parquet` (keyed `(guid, epoch, anchor)`)
— plus a vector sidecar and the aggregated readouts. Every analysis then reads those files, which is
why

```bash
python -m teb_vae.lag_attn_transformer_rws.eval.run --output-dir <a finished run> --only coupling
```

re-runs an analysis offline with no checkpoint, no model and no GPU.

**One thing an offline re-run cannot do: report a criterion the tables predate.** The verdict block
travels with the collection record, so a re-run reports the verdicts the *collecting* pass decided —
correct until the registry moves. When it has, the reuse path refuses with `StaleCachedVerdicts`, naming
the criteria that appeared or disappeared. It is a refusal rather than a repair because only some
criteria are decidable from what a collection record keeps: the predictive pair is, the calibration
census is not. The fix is to re-collect — a new `--output-dir`, or the collection deleted from this one,
with `--checkpoint` so the pass has a model. Every *analysis* number, `source_margin` included, is
recomputed from the per-sample tables and needs no re-collection.

`--max-batches` is a smoke-run
batch cap (a prefix by nature); `eval_config.max_samples` is the seeded *stratified* cap, and the two
are not interchangeable — a prefix over the unshuffled eight-shard split draws one subgroup and one
class.

One analysis is an exception to "the pass happens once", and it is this model's own:
`encoder_attention` runs a second, bounded pass off the task and the loader, because the quantity it
needs is a probability tensor the model never materialises. It is opt-in and costs nothing when it
is not asked for. See its section below.

Before paying for a run, a config can be checked against the shards it names:

```bash
python -m teb_vae.lag_attn_transformer_rws.eval.probe --config <run>/model_checkpoints/resolved_config.yaml
```

A finished run is checked mechanically, and the arm and cross-model tables are generated, by the same
offline module:

```bash
python -m teb_vae.lag_attn_transformer_rws.eval.verify <run>/eval_results/summary.json
python -m teb_vae.lag_attn_transformer_rws.eval.verify --runs <dir-of-runs> --out RESULTS_arms.md
```

`verify` reads files a run left behind and nothing else — no model, no shard, no `torch` (the layering
test walks its imports with `torch` on the forbidden list, on this module *and* on the sibling's gate
it delegates to, so the property is proved rather than promised). The gate itself is the sibling's in
full: the criteria are the run's exit code, the weight-space load check, the named `pred_gap` column
(`pred_gap_mc_nats`, the Monte Carlo marginalised score), the model's acceptance verdicts and the
sanity block — every one a property of the shared objective rather than of either encoder, and a
second copy here would be a second set of thresholds for two models that exist to be compared under
one.

### Selecting analyses

`--only` and `--skip` both take a comma-separated list of the names below. An unknown name raises at
startup, before the checkpoint is loaded, so a misspelling costs a parse rather than a first pass over
the shards. `band_partition` is valid for neither: it always runs and is not selectable. They run in
this order, and `cross_subgroup` being last is load-bearing — it reads the per-recording CSVs the
analyses above it write.

| Name | What it answers |
|---|---|
| `forecast` | Is the forecast any good. Skill against persistence, climatology and the segment's own mean, in nats and in bpm, resolved by horizon step. |
| `coupling` | What the source added. `pred_gap` per recording in both estimators, with a paired Wilcoxon, bootstrap intervals, the positive fraction and three percentage forms. |
| `perm_control` | Is it *this* recording's source. The GUID-aware shuffle control, whose verdict is three losses and deliberately not the KL. |
| `latent` | The per-dimension KL spectrum and active dimensions — plus the prior-variance-pinned detector that catches an inflated coupling number. |
| `lag_kl` | Where in the past the source informed the future. The per-lag KL attribution in its raw, support-corrected and untruncated forms. |
| `attention` | The **lag** cross-attention per head, and its entropy against the ceiling truncated lag support actually allows rather than against $\log L$. |
| `calibration` | Is the decoder's learned variance the spread of its own errors. PIT, coverage, CRPS. An `mse` checkpoint records a skip. |
| `residual` | How far apart the two forecasts are, in bpm, and the two latent-drift quantities behind them. |
| `coherence` | The forecast in the frequency domain, resolved by lead time: coherence, spectral gain, phase, and an exact split of the residual spectrum. |
| `distributions` | The shape of each metric over 20-minute segments, by cohort. Histograms at both levels; descriptive only, and deliberately tests nothing. |
| `trajectory` | The readouts against time — within one segment, and assembled across a whole delivery on the absolute time axis. |
| `time_to_delivery` | The readouts binned on a 0.5 h grid of time before delivery, class-stratified, with Holm across windows. |
| `second_stage` | The same two readouts on the second clinical clock — signed hours from second-stage onset — over the recordings that have one, with its own Holm family. |
| `events` | What the raw target unlocks. Deceleration forecast skill, the contraction-triggered response, and contraction-conditioned coupling. |
| `sufficiency` | What the latent bottleneck costs, against an evaluation-only oracle decoder. The one analysis whose cost is a training loop rather than a forward. |
| `samples` | Per-recording diagnostic PDF pages — a stratified draw, plus the extremes of each headline metric. Needs a checkpoint; skips without one. |
| `cross_subgroup` | Do the cohorts actually differ. Kruskal, Holm, then Mann-Whitney, over the per-recording CSVs the analyses above it wrote, so it runs last. |
| `encoder_attention` | **This model's own.** What the two encoders' self-attention attends to: per-head entropy against its attainable ceiling, attention mass by temporal distance, and the measured source reach against the lag range. |

`encoder_attention` is appended after the shared registry rather than inserted beside `attention`,
where a reader would want it: reordering the shared registry for one model's addition would change
the sibling's run order too. Read the two together anyway — they profile two different mechanisms and
the confusion between them is the one this pipeline most has to prevent.

## Output layout

Everything lands in `<run>/eval_results/`. Without `--output-dir` the run directory is a timestamped
directory under `out_dir_base/<tag>-eval`, where the tag is `general_config.tag` or, absent that,
this model's binding tag `lag_attn_trf_rws` — so a transformer run and a comparison-model run never
land in one another's tree by default.

| Artifact | What it is |
|---|---|
| `summary.json` | The whole run: readouts, verdicts, headline, sanity, cohort, coverage, run context, causality disclosure, step records, artifact manifest. |
| `steps.json` | The per-step heartbeat, rewritten as each analysis finishes — a killed run's record of how far it got. |
| `preflight.json` | Every guard's verdict and the causality disclosure; reused (not regenerated) by a model-free re-run. |
| `loader_probe.json` | The population record: per-shard, per-class and per-label counts the sanity checks read back. |
| `resolved_config.yaml` | The merged configuration the run actually used — the file an offline re-run reads. It carries every constructor keyword and **not** the class they build. |
| `eval.log` | The run's log, including any refusal. |
| `per_sample.csv`, `per_sample_vectors.npz` | One row per segment: every scalar readout plus labels and provenance; the vector readouts in row order. |
| `per_anchor.parquet` | Per-anchor scores, KL, argmax lag, coverage, `seconds_since_contraction`. |
| `coherence_spectra.npz` | The cross-spectral maps pooled over the split and over each clinical class, resolved by frequency and lead time. Fixed size, independent of the split's length; **not** row-aligned with `per_sample.csv`, which is why it is its own sidecar. |
| `collection.json` | The collection record: readouts, provenance sidecar, denominators, retention plan, accumulators. |
| `band_partition.json`, `band_channel_map.csv` | The input channel map (the unskippable data-side step). |
| `<analysis>/…` | One subdirectory per selectable analysis: its CSVs and PDFs. Seventeen of them — `band_partition` writes the two files above instead. |

Three summary blocks matter more than the rest. The **headline** is a flat registry of scalars and
verdict statuses; a number not registered there is invisible to the acceptance gate and the arm
tables, which read it and nothing else. It carries two `pred_gap` columns under names that say which
is which — `pred_gap_mc_nats` (the headline, the log of the average likelihood over $K$ draws) and
`pred_gap_train_path_nats` (the single-draw objective-parity column) — and three percentage columns
restating the same finding proportionally, `pred_gap_rmse_pct`, `pred_gap_mse_pct` and
`pred_gap_mc_likelihood_pct`. `pred_gap_convention` says in the artifact itself which is which and
which spaces they are measured in. **The six `encoder_attention_*` scalars are appended to that block
by this model's binding**, not merged into the shared registry: every entry there is asserted to
resolve on a comparison-model run, so a transformer-only entry would read as a number that model
failed to produce rather than as one it cannot have.

The **sanity** block is the run's three-valued self-consistency record (the KL identity, the
cross-table recombination, the lag identities, the coherence Parseval gate, the population checks); it
deliberately does *not* move the exit code. The **verdicts** are the model's own acceptance criteria,
in registry order, never a bare boolean. The **run context** block beside them records the parameter
count, the checkpoint's training epoch, the anchor-coverage distribution, the observed objective
magnitude and `model_class` — the facts the arm tables, the cross-model table and the first-run
checklist consume. `model_class` is copied out of the checkpoint blob's own stamp, because that stamp
appears in no other artifact a finished run keeps; it is `null` on a pass that built no model, exactly
as `train_epoch` beside it is.

## The four layers

The package is layered, and an AST walk (`tests/test_eval_self_contained.py`) enforces the import
rules, resolving aliased, lazy and relative forms alike. The rules are the sibling's with one
deliberate inversion, recorded in the last row:

| Layer | Modules | May import |
|---|---|---|
| 0 — pure | `verify` | no Lightning, no `model.*`, no `task`/`trainer`; additionally no `torch` |
| 1 — model-touching | `binding`, `encoder_attention` | `binding` reaches this package's `task` through the named `EXEMPTIONS` table; nothing else may |
| 2 — I/O and presentation | `analyses/encoder_attention` | layers 0–1; never another analysis, never Lightning |
| 3 — orchestration | `run`, `probe` | everything permitted above |
| all — the sibling | `teb_vae.lag_attn_rws.eval.*` and `teb_vae.lag_attn_rws.nets.*` | **permitted everywhere**, which is the design |

That last row is why this package is nine modules against the sibling's thirty-four. The sibling's own
walk allows *its* neighbour only through a narrow allow-list of model-free modules; here the whole
evaluation package is the dependency the design rests on, and a rule forbidding it would be a rule
against the architecture. What is *not* relaxed: `model/*` is forbidden at every layer, no analysis
imports another, and this package's own `task` and `trainer` are forbidden by default — the task is
reachable only from `binding`, which is what keeps the coupling to this architecture in one file, and
the trainer is reachable from nothing at all, because an evaluation that reached the training driver
would be building an experiment rather than reading one.

The `EXEMPTIONS` table is asserted **minimal**: a module listing a name it no longer imports is a
permission that outlived its use, and the next reach for that name would go unreported.

## Configuration reference

Everything that shapes a run lives in the `eval_config` block of the override delta, because that
block is dumped into the run directory and is the durable record. Every key is validated against a
closed set — an unknown key raises and names the valid keys, a `bool` where an `int` is expected
raises (`True` would silently cap at 1), and a cap of `0` raises.

| Key | Meaning |
|---|---|
| `seed` | Seeds `random`/`numpy`/`torch` and derives the loader-shuffle, derangement, Monte Carlo and stratified-draw generators by fixed offsets. Two runs of one checkpoint at one seed compare byte-identical on `results`. |
| `num_mc_samples` | Monte Carlo draws $K$ per anchor for the marginalised score, under common random numbers across branches. $K = 1$ reduces it exactly to the training-path score. |
| `max_samples` | Seeded **stratified** global sample cap; `null` evaluates the whole split. |
| `caps` | Per-quantity retention caps (`waveforms`, `attention`, `pages`, `oracle`, and this model's `encoder_attention`). Retention is opt-in: a quantity absent from `caps` is retained for no samples — except `oracle`, where absence means every segment, because a probe fitted on nothing is not a cheaper measurement but no measurement. |
| `prior_shuffle_min_nats` | The provisional margin the prior-shuffle degradation must clear; the verdict always reports the measured number beside it. |
| `min_active_dims` | Active latent dimensions below which the latent counts as collapsed. |
| `event_lag_window_s` | Seconds after a detected contraction within which an anchor counts as event-conditioned. |
| `bootstrap_resamples` | Resamples behind every bootstrap interval, drawn over recordings — never over anchors, whose windows overlap 29/30. |
| `figure_format` | Image format every figure of the run is written in, as a matplotlib filetype (`pdf`, `svg`, `png`, `eps`, …); validated at config load against the installed matplotlib's own list. `null` — the shipped setting — keeps the `pdf` default, which is what `figure_manifest.json` and `FIGURE_GUIDE.md` record and what the smoke suite compares a real run against; a run that changes it writes filenames those files do not list. |
| `max_hours_before_delivery` | How far before delivery a segment may be recorded and still be evaluated, in hours; `4.0` keeps the last four hours, `null` — the shipped setting — evaluates everything. **The bound is on the population, not on an axis**: it is applied to the delivery clock before anything is binned, so every clock answers for the same segments and the second-stage clock re-bins that population on its own signed axis rather than being cut at a second, differently-defined four hours. It moves cohort sizes, window counts and every trajectory, so a bounded run is not comparable with an unbounded one — which is why it is a key, recorded in the run's dumped config. Minimum one 0.5 h bin. |

**The committed delta sets the caps, so a stock run emits the complete artifact set.**
`waveforms: 128`, `attention: 64` and `pages: 24` are the sibling's values exactly — they gate
analyses both models run, so they must match or the two runs cover different sample counts —
and `encoder_attention: 256` is this model's own. `oracle` stays **absent deliberately**: it is the
one cap whose absence means *every* segment, so naming a number would reduce what the sufficiency
probe is fitted on.

`caps.encoder_attention` is the one cap that gates a whole **analysis** rather than a figure. Absent,
`encoder_attention` records a skip naming the key and costs nothing — which is what an offline
re-run and any copy of this file that drops the key will get. It may differ from the sibling's delta
where the other caps may not, because it gates an analysis the sibling does not have: it reads no
shared table and moves no number the cross-model table compares. The equality test exempts exactly
the caps named by this model's own extra analyses, derived from the binding rather than written out.
Cap names are not schema-validated (only their values are), so adding one needs no schema change.

Every value is a bound on **retention**, never on what is measured: each analysis's readouts are
computed over the whole split either way. Lower `waveforms` first on a memory-tight box — at roughly
2.4 MB per retained sample it dominates, and it bounds the event analysis's runtime too.

Deliberately **not** keys: the significance level and the trajectory bin width (an operator who could
widen them could make a difference appear or disappear), and any lag-band selection (the band ablation
is a non-goal, so a band key would be inert by construction). Their absence is asserted by test, not
merely intended.

### Objective keys this pass reads rather than sets

The objective is not an `eval_config` surface, and it is the sibling's: `compute_loss` lives in
`teb_vae/lag_attn_rws/nets/losses.py` and both models call it. A checkpointed pass rebuilds the task
from the checkpoint's **own** `hyper_parameters` — `beta_schedule`, `kld_beta`, `beta_prior`,
`lambda_full`, `lambda_base`, `likelihood` and `free_bits` — and refuses a checkpoint carrying none,
because scoring it under assumed defaults would report a different objective's numbers. On the
offline path the same keys come from the dumped `model_config.VAE_model` block, which preflight has
already reconciled against the checkpoint on every checkpointed run.

`beta_prior` weights the prior's scale rate $R_p$, the objective's fourth term, and **this
architecture is where the pathology it answers was measured**: the $1018$-epoch baseline finished
with `logvar_prior_floor_frac` at $0.992$, reached inside one epoch. A checkpoint trained at a
non-zero weight is scored under that same weight rather than a default, and the value appears in
`run_context.observed_loss_scale` beside the term it weights. Two readouts follow and exist whatever
the weight was — including on the pre-anchor baseline, which is exactly where they diagnose: the
per-recording `prior_rate` column and the `prior_rate_nats` headline scalar the
`sweep_beta_prior_*` arm table reads. Neither is conditional on `beta_prior` being non-zero.

Weights are recorded and, unlike geometry, deliberately **not** compared against the config by the
preflight guard: $\beta$ and its ramp weight the training total and enter no evaluated readout.

### Cohort order and colour

Two presentation conventions every table and figure that resolves a quantity by cohort obeys. Neither
is a setting, for the reason the significance level is not one: an operator who could reorder or
recolour the cohorts could make a difference look like a trend.

**The order is clinical, not alphabetical, and it runs worst first**, on both axes:

| Axis | Order, left to right |
|---|---|
| `clinical_class` | HIE, acidosis, healthy |
| `subgroup` | `hie_cs`, `hie_no_cs`, `acidosis_cs`, `acidosis_no_cs`, `healthy_bg_cs`, `healthy_bg_no_cs`, `healthy_no_bg_cs`, `healthy_no_bg_no_cs` |

`labels.ordered_groups` is the one function that decides it — `cohort.ordered_groups` is a one-line
binding of it — and both orders are read off tables that already exist rather than restated:
`labels.CLASS_NAMES`, keyed by the dataset's own class codes $1, 2, 3$ and read in reverse, and
`labels.CANONICAL_SUBGROUPS`, likewise. So a subgroup added to the dataset appears in the right
place with no edit here. The default everywhere the function is *not* called is alphabetical, and
alphabetical is wrong in a way that looks fine: it puts `acidosis` left of `hie` on every class
figure, and on the subgroup axis it interleaves the three classes so that neither the severity
ordering nor the background/caesarean structure is visible. A cohort the order does not know — a
non-canonical shard stem — sorts **after** every one it does, and is never dropped.

It reaches the CSVs as well as the figures, and that pairing is the point: the grouped `*_by_*`
tables, the stratified lag profiles and peak rows, the conditioned-coupling rows, the per-window
significance records, `encoder_attention`'s own class-cut tables and the summary's population counts
are all written in it, so a table can be read against the figure beside it row for row.

**The colour is the severity**: green for healthy, amber for acidosis, red for HIE, with each subgroup
a shade of its own class, light to dark in the order above. The mapping is a *table* rather than an
assignment pass, so a cohort keeps its colour whichever others a figure contains.

The palette is the shared evaluation's, `figures_seam.CLINICAL_CLASS_COLORS`, and deliberately not
`utils.style.CLASS_COLORS_DEFAULT`, which paints healthy blue and is shared with two other projects —
repainting it there would restyle them to satisfy this convention. The cost is stated rather than
hidden: **an evaluation figure of a cohort is not the same colour as a training-callback figure of
that cohort**, and the two are reconciled by legend rather than by hue.

The convention reaches `cross_subgroup`'s effect heatmap too, whose x axis is cohort *pairs*. The
column order is clinical and so is the naming inside a column: which cohort of a pair is `left`
comes from the shared `pairwise_comparisons`, which names a pair in the order it receives the
cohorts and receives them in the canonical one, worst first — so a column reads more severe
against less severe and a positive Cliff's delta means the more severe cohort's values run higher.
Reorienting a column by eye still flips its sign against the number in
`cross_subgroup_pairwise.csv`.

## The analyses

One section per registered analysis, named for its module. `band_partition` always runs and is not
selectable; the rest are what `--only` and `--skip` choose between, in the order of the table above.

### band_partition

What each of the model's input channels is, read off the shards' own `sel_*` provenance rather than
re-derived: one row per input channel across the 109-channel target stream and the 58-channel source
stream, laid out as the model receives them, each mapped to a band, a kind and a centre frequency in
Hz. It describes the model's **inputs** and is the data-side companion to the causality disclosure.
The feature bank is the comparison model's, unchanged, which is what makes the two architectures'
numbers comparable at all.

The fourteen scattering channels with no recoverable centre frequency are recorded as such rather than
omitted, per stream. A shard carrying no `sel_*` attributes is a recorded skip, not a raise.

### forecast

Is the forecast any good, in units a clinician reads, and where in the horizon. A block score alone
cannot answer that: it is a negative log density summed over $H \cdot R = 480$ raw samples, so it is
large under every predictor and its scale is set by the block size rather than by the model. Three
things make it readable — skill against **three trivial baselines** (persistence, climatology, and the
segment's own mean) scored through the model's own masked scorer with the identical mask; the error in
**bpm**; and the score resolved by **horizon step**.

The MSE-space skill $1 - \mathrm{MSE}_m/\mathrm{MSE}_b$ is the one with a natural zero. The NLL-space
column beside it is a **difference** in nats, `advantage_nats_per_anchor`, not one minus a ratio: a log
score has no natural zero, so the ratio of two of them is not bounded above by one and changes sign
with the baseline's. The baseline $\sigma$ is fixed at 1 in $z$-space and recorded, because a skill
score against a point predictor is entirely determined by the $\sigma$ handed to it — and a
learned-$\sigma$ model would otherwise beat a fixed-$\sigma$ baseline partly on variance modelling
alone.

Persistence carries the last **observed** sample forward rather than the last one: a gap is stored as
0 bpm, roughly $-11\sigma$ after z-scoring, and carrying that would measure the gap. The horizon curve
is computed on the **single-draw** path and says so: the marginalisation does not commute with the sum
over $\tau$, so a marginalised curve would not sum back to the marginalised headline.

### coupling

What the source added, per recording, with the uncertainty on it. `pred_gap` in both estimators — the
Monte Carlo marginalised headline and the single-draw training-path parity column, never merged — with
the fraction of recordings where the gap is positive, a paired Wilcoxon over the per-GUID vector,
bootstrap intervals over recordings, and quantiles rather than only means.

The positive fraction reports its **denominator**: `np.nan > 0` is `False`, so unscored segments would
otherwise count silently as evidence against. The KL travels beside the gap as a **description** rather
than as a second answer — it is inflated by an arbitrary factor whenever the prior variance sits on its
clamp, and unlike `pred_gap` it says nothing about whether the forecast improved. That distinction
carries extra weight for this architecture: the whole design claim behind the replacement is a
*stronger target prior*, and a stronger prior lowers the source-conditioned KL without the coupling
having weakened.

**The same finding as a percentage**, because nats state no proportion: whether 3 nats over a
480-sample block is a large improvement is not readable off the number, and two checkpoints whose block
scores differ in scale cannot be compared on it at all. Three columns, in the two spaces where a ratio
has a natural zero, each computed per recording and then averaged and each bootstrapped over recordings
like everything else here:

| Column | What it is |
|---|---|
| `pred_gap_rmse_pct` | $100(1 - \mathrm{RMSE}_{\mathrm{full}}/\mathrm{RMSE}_{\mathrm{base}})$ — the percentage of the point-forecast error the source removed. **Scale-free**: the same number in $z$ units and in bpm, so no normalisation has to be inverted to read it. |
| `pred_gap_mse_pct` | the same ratio unrooted — `forecast`'s own `mse_skill` convention, applied source-versus-no-source rather than model-versus-baseline. |
| `pred_gap_mc_likelihood_pct` | $100(e^{\Delta/(H\cdot R)} - 1)$ — the percentage form of the headline nats: the extra probability density the source-conditioned forecast puts on each observed raw sample. |

The arithmetic is `frames.skill_against`, the same function `forecast` scores its baselines with — the
guard is the content: the denominator is tested **strictly positive** and fails to `NaN`, never to
`inf` (which the headline's finiteness check would refuse) and never to `0.0` (which reads as "no
improvement"). The RMSE percentage is taken as the root of $1 -$ the MSE one rather than by dividing a
second time, so the two share that one guard and cannot disagree about the sign.

The likelihood percentage has **two preconditions, and failing either omits it entirely** — no column,
no row, no headline key — with the reason recorded under `coupling.pred_gap_percent.likelihood_space`
in the package's usual `skipped`/`reason` shape. The first is the likelihood: exponentiating a block
score means something only where the score is a log density, so an `mse` checkpoint gets the two
error-space percentages and not the likelihood one, and an *unknown* likelihood is a skip too rather
than a pass. The second is the block size, read from the run's own geometry rather than assumed. It is
the **fixed** $H \cdot R$, not each anchor's scored-sample count, so the number under-reports wherever
forecast steps are masked — the same caveat every `/480` figure here carries, and it makes the value a
floor rather than an estimate.

### perm_control

Does the model use *this* recording's source, or react to any source at all? The verdict is three
losses and nothing else: $D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}}$.

The KL is deliberately not a parameter, and that is the content of the criterion rather than a
simplification of it. A stranger's source is out of distribution for a posterior trained only on
matched pairs, so it routinely moves the posterior **more** — a healthy model has
$K_{\mathrm{shuffled}} > K_{\mathrm{true}}$, and a criterion that read the KL would fail exactly the
models it should pass. `shuffled_exceeds_true` is therefore recorded as a description and consumed by
nothing, and `influential_not_specific` is a real finding about a checkpoint rather than a pipeline
failure.

The derangement is **GUID-aware**: Sattolo's algorithm guarantees only $\pi(i) \ne i$, and the test
split is eight per-subgroup shards read in order, so an unshuffled batch would pair a recording with
its own neighbouring segment. The pass runs under a seeded shuffle, a batch with no cross-recording
pairing available is excluded **and counted**, and both the `same_recording_pairing_rate` and the
excluded count reach the summary even at zero.

The shuffled branch is scored **on its own**, never differenced sample by sample against the matched
branch, because the permutation draws a fresh $\epsilon$; and only the seven keys the control actually
recomputes may be read from its shallow-copied output.

Three paired controls are scored per recording, all under the same sign convention — positive means the
control is worse than the branch it is referenced against:

| Row | Quantity | Referenced against |
|---|---|---|
| `shuffle_penalty` | $D_{\mathrm{shuffled}} - D_{\mathrm{base}}$ | no source at all |
| `prior_shuffle_penalty` | $D_{\mathrm{base}}(\text{shuffled } \mu^p) - D_{\mathrm{base}}$ | its own prior latent |
| `source_margin` | $D_{\mathrm{shuffled}} - D_{\mathrm{full}}$ | this recording's own source |

**The third one is referenced against `full`, and that is why it exists.** The two above it are both
referenced against the target-only branch, so both inherit whatever the base forecast is doing — and a
model whose latent geometry charges more for the source than the source delivers fails every
base-referenced comparison while still reading *this* recording's source rather than any source.
`source_margin` changes only the source: prior, decoder and latent geometry are identical between its two
branches. So a **positive margin beside a negative predictive gain is a real state, not a contradiction**
— no forecast improvement, and the source pathway is still recording-specific. That is not a hypothetical
here: it is what this architecture's first production run measured, at $+1.06$ nats of margin against a
$-7.37$ nat gain, and the readout exists because that run could not say so.

All three carry the same describe, bootstrap, positive-fraction and Wilcoxon columns, in `summary.json`
under `perm_control.penalties`. They are **not** in `perm_control_summary.csv`, which carries the branch
table only. `source_margin`'s mean is additionally emitted as the keyed scalar
`perm_control.source_margin_nats` and promoted to the headline as `source_margin_nats`: the headline is
assembled by walking key paths and `penalties` is a list, which is why the two penalties beside it have
never appeared there.

### latent

How much of the latent carries source information, and whether its variance is fitted or bound. The
per-dimension KL spectrum, the active-dimension count and the top dimension's share — and the detectors
the evaluation would otherwise never read, though the model computes them and the trainer logs them
every epoch: `mean_logvar_prior`, `logvar_prior_floor_frac`, `mean_logvar_post`.

That second half is the point. The KL carries $(\mu^q - \mu^p)^2 / \sigma_p^2$, so a **prior** variance
pinned on its lower clamp multiplies every coupling readout by an arbitrary factor while every
decoder-side diagnostic stays perfectly healthy. `prior_variance_not_pinned` is the FAIL-able verdict
that catches it, judged at the model's own margin — 5% of the clamp range, which on the shipped
$[-5, 3]$ is 0.4 nats. The bound is a sigmoid, so an exact-equality test would read zero forever.

`prior_rate` is the same pathology as a **distance rather than a fraction**: the objective's own
$R_p = \sum_d \tfrac12(e^{\ell^p} - 1 - \ell^p)$, per recording and in nats per anchor, reduced on the
KL support like the divergence beside it and reported as the `prior_rate_nats` headline scalar. Zero
means $\sigma_p = 1$ exactly, so it is the only readout here bounded below by its own optimum, and it
is continuous where the floor fraction is a step — it rises from the first epoch, while
`logvar_prior_floor_frac` stays at $0$ until mass reaches the margin. Read them together: a large
`prior_rate` at a zero floor fraction is a prior drifting off unit scale with room left; a large one
at a floor fraction near $1$ is the collapse the verdict already failed on, which is what this
architecture's pre-anchor baseline reports.

`mu_prior_sat_frac` and `delta_mu_sat_frac` are flat means over every element, warm-up prefix and
untrained tail included, so only those two are recomputed `_masked` beside the model's own `_raw`
values; the two may legitimately disagree. The log-variance fractions are already masked over
`elem_mask` and `kl_support`.

### lag_kl

Where in the past the source informed the future. The per-lag KL attribution
$\widetilde K_{t,\ell} = \sum_m K^{(m)}_t \alpha^{(m)}_{t,\ell}$, whose sum over lags is exactly $K_t$ —
an identity re-measured on **this run's worst anchor** every pass and registered in the sanity block,
not inherited from a model test. A maximum rather than a mean, because the one mechanism that breaks it
(dropout on the attention probabilities) has a per-anchor error that is zero-mean by construction.

Three profiles, not one, and they answer different questions. The **raw** attribution divides every bin
by the same anchor total and is therefore a decomposition of the headline KL. The **support-corrected**
one divides each bin by the anchors at which that lag was causally valid — lag $\ell$ is valid only at
anchors $t \ge \ell$, so over the trained range lags 0–30 receive 240 contributing anchors while lag 90
receives 180, a 25% under-weight that biases the argmax short. The **untruncated** one is recomputed on
the anchors at which every lag exists, because the support correction fixes each bin's denominator and
cannot fix its numerator: attention rows are renormalised per anchor, so a truncated anchor pushes mass
onto the short lags and no per-lag count knows it happened.

An argmax is not by itself a reading. Peak width, mass above threshold and secondary peaks travel beside
it, and `degenerate` is defined mechanically — peak-to-median below 1.1, **or** exact-zero fraction above
0.9 — because `entmax15`'s exact zeros can make an argmax on a flat profile meaningless. Both profiles
are also cut by clinical class and by time window on the shared grid.

### attention

The **lag cross-attention** itself: per head, against the entropy it can actually reach. The posterior is
head-structured — latent group $m$ is written by attention head $m$ alone, which is what makes the
per-head KL an additive decomposition rather than an arbitrary slice — so averaging the four heads before
profiling discards exactly what the architecture exists to expose: four heads at four delays and one head
attending everywhere produce the same head-averaged curve.

The entropy ceiling is $\operatorname{mean}_t \log \min(t+1, L)$, **not** $\log L$: at the shipped
geometry exactly 60 of the 240 trained anchors have structurally truncated lag support. Both entropies
are emitted, distinctly named, and the ceiling is a per-sample column over the sample's own scored
anchors so their ratio is a measurement rather than an approximation. The entropy is taken per anchor and
then averaged, never as the entropy of the averaged profile — a mixture's entropy is at least the mean of
the entropies mixed, so the second reports a model whose lag focus *shifts* as one that has none.

`kld_per_t_per_head` sums over heads to `kld_per_t` exactly, and that identity is the second sanity-block
check.

This analysis and `encoder_attention` are two different mechanisms and the confusion between them is the
one this package most has to prevent. Here $M = 4$ heads run over $L = 91$ **lags** and decide which past
*source neighbourhood* informs the latent. There $H_e = 4$ heads run over the within-stream **time axis**
and decide how each history state is built. Neither is a rescaling of the other.

### calibration

Is the decoder's learned variance the spread of its own errors? Under `gaussian_nll` the block score is a
negative log density only if it is, and nothing else in this pipeline checks it — a model can drive its
NLL down by shrinking $\sigma$ wherever it happens to be right and paying for it elsewhere, and every
score in every other analysis would improve.

Four readings over the raw samples themselves: the PIT, central coverage at the exact erf nominals, CRPS,
and the NLL gain over the homoscedastic MLE fitted to the very residuals being scored. The nominals are
$\operatorname{erf}(k/\sqrt 2) = 0.6827,\ 0.9545,\ 0.9973$; the two-sigma figure is **not** 0.95 — that is
$\pm 1.96\sigma$ — and the half-point difference reads as a real miscalibration.

`logvar_full_floor_frac` and `logvar_full_ceil_frac` ship beside `mean_logvar_full`, because a single mean
is equally consistent with a spread distribution and with half the mass pinned on each clamp. This is the
one analysis whose output directly changes a config value: it states the recommended
`model_config.VAE_model.logvar_clamp` revision, and says **no change** when neither end binds — a
recommendation emitted unconditionally is one that gets applied unconditionally. An `mse` checkpoint
records a skip and accumulates nothing, because the log-variance head is never fitted there.

### residual

How far apart the two forecasts are, and how far the source moved the belief behind them. `mu_base` and
`mu_full` are two passes of **one shared decoder** on two latents, so the residual is not a tensor the
model computes and has to be measured between the two outputs.

Reported: the per-anchor forecast-difference RMS in bpm — via the **sigma** inversion, which scales by
`std` with no offset, because inverting a standard deviation affinely is a silent, plausible-looking
error — and the two latent-side quantities that are **not** the same thing: `delta_mu_rms`, per element,
and `mu_post_prior_gap_rms`, per step with the L2 over $d_z$ taken first.

RMS metrics accumulate unrooted and root once. Averaging finished per-sample RMS values is biased low by
Jensen, in the direction that flatters the model; the analysis reports the average-of-roots beside the
rooted-once value, so the bias is a measured number rather than an assumed one. One caveat weakens in the
model's favour and is stated: both branches share one log-variance head applied to different $z$, rather
than reading two separate variance heads.

### coherence

Which frequencies the forecast reproduces, and for how far ahead. Every other analysis scores it in the
time domain, and that hides a distinction with clinical content: a forecast that holds the baseline while
flattening beat-to-beat variability and one that tracks variability while drifting on baseline score
identically on mean squared error.

**The construction is what makes it possible.** A single forecast block is $H \cdot R = 480$ samples; a
Welch window that fits inside it puts the whole $[0, 0.04)$ Hz deceleration span in the DC bin the detrend
has already removed. Fixing a horizon step $\tau$ and concatenating over consecutive anchors instead yields
a **contiguous, gap-free, non-overlapping** 4 Hz series: incrementing $t$ by one advances the block start
by exactly $R$, and each anchor contributes exactly $R$ consecutive samples, so slice $\tau$ *is* the raw
trace over $[R(w+1+\tau),\,R(T_{\mathrm{valid}}+1+\tau))$ — every sample once. That identity is asserted
against the model's own target builder rather than re-derived.

Three things follow. The window is `nperseg = 512` (128 s), so $\Delta f = 7.8125$ mHz and four bins sit
below $0.03$ Hz. **Lead time becomes an axis rather than a trade**: slice $\tau$ holds lead times
$[4\tau + 0.25,\ 4\tau + 4]$ s and the thirty of them tile $0$–$120$ s, each at full frequency resolution.
And because `nperseg` and the hop are integer multiples of $R$, a window spans a whole number of anchors,
so its validity is an exact `all()` over the forecast mask.

**A window touching a gap is dropped whole; nothing is interpolated.** A spectral estimate cannot take the
interpolation trade `events` takes, because the interpolant is a deterministic ramp whose own spectrum —
concentrated low, absent high — would be attributed to the model in exactly the bands this is read for. The
drop is proved rather than promised: poisoning every invalid anchor with $\pm 10^9$ leaves every accumulator
bit-identical. One consequence is that **this analysis's population may be smaller than `forecast`'s**,
because whole-window validity is stricter than per-step validity. That is a real population difference, not
a fault.

Four readouts per band and lead time, and they fail independently. **Coherence** $\gamma^2$ is how much of
the truth's variation the forecast reproduces in phase. **Spectral gain** $g = \sqrt{S_{yy}/S_{xx}}$ is
whether it has the truth's amplitude — $g < 1$ is the over-smoothing every mean-square-trained forecaster is
prone to, and it is invisible to the coherence. **Phase and a group delay** say whether it arrives at the
right moment. And the **exact three-way split of the normalised residual spectrum**

$$\frac{S_{ee}}{S_{xx}} \;=\; \underbrace{(1-\gamma^2)}_{\text{irreducible}} \;+\; \underbrace{2g\gamma(1-\cos\phi)}_{\text{timing}} \;+\; \underbrace{(g-\gamma)^2}_{\text{amplitude}}$$

is what turns a coherence into an actionable statement: the terms vanish exactly when nothing is
unpredictable, when the phase is right, and when the amplitude equals the mean-square-optimal
$g = \gamma$. The algebraically equivalent $\gamma^2\sin^2\phi + (g-\gamma\cos\phi)^2$ is **not** used,
because it charges a purely mistimed forecast for amplitude error — the exact confusion this analysis exists
to remove. Beside them travel $\Delta\gamma^2$, the frequency-resolved `pred_gap`; the UP–FHR coherence of
the truth and of both branches; and a token-seam check.

**The bands are the fetal-HRV table**, under an `hrv_band` column that cannot collide with
`band_channel_map.csv`'s `band`: `vlf` $[0, 0.03)$, `lf` $[0.03, 0.15)$, `mf` $[0.15, 0.50)$, `hf`
$[0.50, 1.00)$, `noise` $[1.00, 2.00]$ Hz, holding $4/16/44/64/129$ bins. They are deliberately **not**
`band_partition`'s `CLINICAL_BANDS`, whose crosswalk is `slow_baseline`+`deceleration` $\approx$ `vlf`,
`variability` $\approx$ `lf`+the lower half of `mf`, `beat_to_beat` $\approx$ the rest — that table's
$0.25$ Hz edge is exactly the decoder's token-seam frequency, and a band boundary is the worst place for an
artifact to sit, and it has no LF/MF split, which is the distinction a spectral statement about a forecast
most needs. The bands partition every bin from DC to Nyquist exactly once, asserted at import. `vlf`
includes the DC bin on purpose: the per-window mean is removed but the Hann taper leaves a residue.

**Sums in, ratios out.** The collection pass stores unnormalised cross-spectral sums per segment and this
analysis is the only place a ratio is formed — so within a recording the statistics are **summed**, a named
departure from `frames.per_recording_means`, which every other analysis uses. Averaging per-segment
coherences would be wrong rather than merely different: coherence is exactly $1.0$ on a single window for
any two signals whatever, and is biased upward by $(1-\gamma^2)/n_d$ at $n_d$ windows, of which one segment
holds at most $14$. Both estimators ship side by side (`..._segment_mean`), so the size of that bias is
measured on every run. Across recordings the chain resumes as usual: one value per recording, unweighted,
bootstrapped over recordings.

**The phase is never averaged.** It is an angle; the mean of $-3.1$ and $+3.1$ radians is $0$, half a turn
from both. Where a pooled phase is reported it is the argument of the *summed* cross-spectrum, and a delay
comes from a magnitude-weighted grid search rather than a phase unwrap — which needs the phase to advance by
less than $\pi$ per bin and fails silently past that. The delay is identifiable only modulo
$1/\Delta f = 128$ s, so the search runs over $\pm 64$ s and a wider request is refused rather than served an
arbitrary tie.

Two checks reach the sanity block. `coherence_parseval` is an **exact** identity gated at the same
`IDENTITY_RTOL` as the other two: the band-summed residual spectrum against a time-domain residual
accumulated independently over the identical kept windows. It reports `NaN` — and so lands INCONCLUSIVE —
when nothing was actually compared. `coherence_detrended_share` is the loose one, and it is **not** a
comparison against the block scores: what it bounds is the detrended windowed residual power against the raw
one, the share of the forecast's error the spectrum can see at all. The complement is the **level** error,
which the per-window mean removal makes invisible here.

### distributions

What each metric's distribution over 20-minute **segments** looks like, cohort by cohort. Every other
analysis reduces to one value per recording before reporting anything, and what that hides is the *shape*:
three cohorts with the same mean forecast error can be a uniform shift, a heavier tail, or a handful of
segments the model fails on completely, and those are three different findings. Eight metrics, two per
question the pipeline asks — the two branches' per-segment RMSE in bpm and the block score; `mc_pred_gap`
and the unfloored KL; `delta_mu_rms`, the decoder's mean log-variance and the lag-attention entropy.

**It is descriptive by construction.** No test, no interval, no $p$-value, and nothing registered in the
headline block. That is not an omission: a per-segment $p$-value is anticonservative by the ~30× anchor
overlap, and `cross_subgroup` remains the only analysis that adjudicates a cohort difference. A visible
separation here is a reason to look.

**Both levels are drawn on the same axes, and that is the content.** The filled density is one value per
segment; the median / inter-quartile / range **strip** above it is one value per recording. Their difference
*is* the pseudo-replication — a strip far narrower than the density beneath it says most of the visible
spread is within-recording variation. The two levels take two different forms on purpose: a forty-bin
density over a cohort's six recordings is a row of spikes that estimates nothing and takes the panel's
y-limit with it.

Four presentation choices are load-bearing. **Density rather than counts**, because the healthy cohort
contributes an order of magnitude more segments than HIE and a count axis would report the cohort sizes
rather than the metric. **One bin grid per panel**, computed from the pooled values across the cohorts drawn
in it — two histograms on two grids are not a comparison. The **subgroup figure is nested rather than flat**:
one column per clinical class, that class's subgroups overlaid inside it. And the **overlap encoding**: each
cohort is a faint fill under a hairline outline at full opacity, drawn in two passes so every outline sits
above every fill, because one pass per cohort would leave the first cohort's outline veiled by every fill
after it and the first legend entry would be the hardest curve to trace.

The error metrics are **rooted per segment and converted to bpm**, which is legitimate here and would not be
elsewhere: `residual`'s rule is that *averaging* finished roots is Jensen-biased low, and the object drawn
here is the distribution rather than its mean. `per_segment_root_note` says so in the record.

It declares **no** `grouped_frames`. The runner's fan-out draws violins documented as holding one value per
recording; handing it this per-segment frame would produce a per-segment violin that reads as a
per-recording one, which is the exact confusion this analysis exists to make visible.

### trajectory

The two coupling readouts against time — inside a segment, and across a whole delivery. The per-anchor
table's first general consumer.

Within a segment the shape is **structural before it is physiological**: the warm-up prefix carries no loss
term, the lag support is truncated until $t \ge L - 1$, and the last $H$ anchors are never scored, so a
profile that rises or falls at either end is the geometry rather than the model. Across a delivery the
segments are assembled on the absolute time axis $t_{\mathrm{abs}} = \mathrm{epoch} + 4t$, with overlapping
timesteps **averaged** rather than drawn twice and `n_contributing` travelling so the averaging is visible
rather than inferred. A gap produces a **break** in the data — `gap_before_s` — rather than an
interpolation, so an analysis reading the table sees it too and not only the figure.

### time_to_delivery

Does the coupling change as delivery approaches, and differently by class? Both readouts binned on a 0.5 h
grid of `epoch / 3600`, class-stratified, on per-GUID values — per-GUID *inside* a window as well as across
the split, so a recording contributing eleven segments to a window cannot outvote one contributing two.

`pred_gap` is tracked beside the KL because the two fail differently: `pred_gap` is in the decoder's own
units and is immune to the prior-variance inflation. Significance is tested **per window**, with Holm across
windows as one family and pairwise tests on the survivors; the `pooled` row is flagged `confounded_by_time`
and consumed by nothing.

`TRAJECTORY_BIN_HOURS` is a module constant, not an `eval_config` key, for the reason the significance level
is not one — and it is defined one layer down so the lag structure is cut on the same grid.

The analysis emits **four** figures — two pages per readout, because `pred_gap` and the unfloored KL share a unit and not a scale, so a page carrying both draws the smaller as a flat line at the bottom of the larger's range. `time_to_delivery_trajectory_<readout>.pdf` is the median line per class with its inter-quartile ribbon; `time_to_delivery_windows_<readout>.pdf` is what that line is made of — a violin per (window, class) cell over one value per recording, the Holm-adjusted $p$ of each window directly beneath it on the same axis, and Cliff's delta for every class pair that survived. The tests were always run; until that page existed nothing drew them.

### second_stage

The **second clinical clock**: the same two readouts resolved against signed hours from the onset of the
second stage of labour rather than against delivery, because delivery is the end of a process whose
clinically meaningful landmark is inside it, and two recordings four hours before delivery can be at
completely different points of labour.

The axis is **signed and is not negated**: the shard stores `second_stage_onset = domain_start - t_SSO`,
already negative before onset and positive after, unlike `epoch`, which is stored as time *before* delivery.
Both figures are therefore drawn in the natural orientation with a line at zero rather than inverted, and the
axis label names the sign convention outright.

**One rule drops a recording** — it has no recorded onset — and it is counted. The two further ways a stored
onset can be wrong are **counted and filtered nowhere**, both reaching `second_stage_eligibility.csv`: an
implied onset falling *at delivery*, which is what a pipeline writes when it substitutes zero for a missing
time, and an implied onset that *moves* across a recording's own segments. The analysis therefore scores a
subset of the cohort, declares `plan.capped = True` with its reason, and is excluded from the coverage
block's population comparison rather than reported there as a disagreement.

**The Holm family is this clock's own** and is not corrected jointly with `time_to_delivery`'s: the two are
different alignments of an overlapping population, so a window significant on one and not the other is a
statement about alignment, and a reader quoting both clocks is making two comparisons. The grid is the same
0.5 h `TRAJECTORY_BIN_HOURS` the delivery clock uses.

### events

The analysis the raw target exists for: the forecast scored as a *waveform*, against the two events a
clinician reads a trace for.

**Deceleration skill.** The detector runs on the true raw FHR and on each branch's forecast, both in bpm, and
rates are computed **per event** under a de-duplication rule that is exact rather than approximate: fixing
the horizon step. For a given $\tau$ exactly one anchor places a given absolute raw sample there, so a
per-$\tau$ rate counts each physiological event once by construction — a de-duplication pass applied
afterwards would need a clustering tolerance, and that tolerance would be the answer. The usable interior is
240 of the 480 samples, because the ported detector drops any event within 30 s of either block end, and the
measured pseudo-replication factor is 14 rather than the horizon's own 30.

**Contraction-triggered response**, against a count-matched per-recording random-trigger null passed through
the **identical** min-over-window operator. The statistic is a minimum over a window and is negative on any
data at all, so the null is what measures that selection bias rather than assuming it away.

**Conditioned coupling**: both readouts restricted to anchors within `event_lag_window_s` of a detected
contraction, against count-matched control anchors drawn from the same recordings. The contraction timing is
computed in the collection pass and lands on the per-anchor table as `seconds_since_contraction` — it has to
be, since the model reads the source as scattering and phase channels and a contraction exists nowhere in
the tables unless the one pass holding the raw UP trace puts it there — so this runs over every anchor of the
split rather than only over retained samples. Guards: at least 200 event anchors over at least 4 recordings,
else a recorded skip.

Gaps are masked by `weight`, never by value — 0 bpm is roughly $-11\sigma$ after z-scoring, not a detectable
sentinel. Masking is two steps and both are needed: invalid samples are interpolated across *before*
smoothing, so a gap contributes no edge for the peak finder to lock onto, and any event whose span touches
one is then **dropped**, because its shape partly came from that interpolation.

The contraction onset is a **level crossing** of the peak's own prominence rather than a gradient walk-back,
which is a deliberate correction to the ported detector and is named as such in the module: a gradient test
stops at the apex, where the smoothed gradient is approximately zero, and a two-stage gradient walk stops
mid-flank.

### sufficiency

What the latent bottleneck costs the forecast:
$\Delta_{\mathrm{suff}} = D_{\mathrm{base}} - D_{\mathrm{oracle}}$, where $D_{\mathrm{oracle}}$ comes from an
evaluation-only decoder of the same capacity reading `target_state` instead of $z$, fitted on half the
evaluation recordings and scored on the other half. Until this exists, `pred_gap` is a gap between two models
rather than an information rate.

**It is an estimate, not a bound**, and both bias directions travel in the emitted JSON rather than only here.
Conditioning on `target_state` rather than on the raw target history omits the encoder's own information loss
and biases the gap **down**; fitting the probe on the evaluation population while $D_{\mathrm{base}}$ comes
from a model trained on the disjoint, healthier pretraining cohort biases it **up** by a domain shift the
probe does not suffer. The two oppose, neither is measured, so nothing downstream may treat the number as a
bound.

For this architecture the first bias direction is worth reading twice: `target_state` here is the output of a
conv-Transformer stack with full causal prefix access, not of a recurrent cell, so the encoder's own
information loss is a different quantity than it is for the comparison model — and the sufficiency gaps of
the two architectures are therefore comparable only with that stated.

The probe's convergence flag is a precondition, not a decoration: an unfinished probe understates the gap.
Convergence is arithmetic on the held-out curve — the final quarter contributed at most a tenth of everything
the score ever gained — and a curve that never improved is **not** converged, because a probe that did not
move has failed to start rather than finished. The capacity check refits at double width and flags
`capacity_bound` if that improves the held-out score by more than the stated margin. $D_{\mathrm{oracle}}$ is
measured once over the whole held-out half at the final state, never at the curve's best point, which would be
selecting the step by the score it is about to report.

The split is at **GUID** level, disjointness is asserted at runtime rather than only tested, and the fit budget
is expressed in passes over the fit half rather than in optimizer steps.

### samples

Per-recording diagnostic pages, and the triage that picks which recordings to look at. The page is the same
seven-row diagnostic the training callback writes every validation epoch, from the same builder rather than a
second one that could disagree with it.

`stratified/` holds a seeded, shard-stratified draw over the whole split, so a cap at or above the shard count
reaches every shard. Beside it, one directory per headline metric and tail holds the segments at the extremes
of that metric. A page is one segment of one recording: an illustration, never evidence — and the extreme
pages are selected *on* the quantity they display, so the panel showing it is guaranteed to look unusual and
says nothing about how often it does.

**Ten pages per directory**, so every selection holds the same number and two of them can be read against each
other. `eval_config.caps.pages` overrides the stratified count; the extremes take ten per tail as an **upper
bound**, lowered to half the finite rows wherever a metric has too few scored segments to fill both. The two
tails of one metric are disjoint by construction.

The `<index>` in a filename is the position in the evaluation **dataset**, not in `per_sample.csv` — the
collection pass runs under a seeded shuffle — and the two are reconciled by a `guid`/`epoch` round trip checked
before anything is rendered. One page failing is recorded by index and does not stop the rest.

This is one of exactly two analyses that read the task and the loader off the context rather than the tables;
`encoder_attention` is the other, and for a related reason: what it needs was never in the tables.

### cross_subgroup

Do the cohorts actually differ, or does the by-subgroup table only look as though they do? Eight cohorts each
with a mean always produce a highest and a lowest; with eight metrics that is sixty-four numbers, and some will
look separated whether or not anything is there.

Three layers, in order, and the order is the point: a Kruskal omnibus per metric, Holm **across metrics as one
family**, and pairwise Mann–Whitney with Cliff's delta (Romano magnitudes) on the survivors only. Every test
consumes one value per **recording** — a source naming a `per_sample` file would test segments while reading as
though it tested recordings, and a test asserts none does.

It reads finished per-recording CSVs off disk through a `METRIC_SOURCES` table, so a missing source is
**recorded** rather than raised — which is what keeps `--only cross_subgroup` working against a finished
directory with no checkpoint — and it self-skips below two testable groups.

### encoder_attention

**The one question this architecture makes askable, and the only analysis the comparison model cannot have.**
`attention` above profiles the lag cross-attention. This profiles the **encoder self-attention**: $H_e = 4$
heads over the within-stream time axis, a full causal prefix on the target and a $W_U = 16$-step window on the
source. That mechanism is the whole content of the encoder replacement, and nothing else in the repository
reads it.

**Where the numbers come from.** `CausalSelfAttention.forward` runs through fused
`scaled_dot_product_attention`, which never materialises the probabilities, and the forward contract is pinned
at exactly twenty keys against the comparison model — so the analysis **recomputes** them. A forward hook
captures each block's input and the recompute re-applies that module's own `norm`, `q_proj`, `k_proj`, head
reshape and `rope`, then its own attention mask, then an explicit softmax. Every operand is read off the
module; nothing is rebuilt from config. For a windowed block the mask is the module's *own* `attn_mask` buffer,
sliced — that buffer is what was handed to the fused kernel, so a rebuild could agree today and drift tomorrow;
the full-prefix case builds the lower triangle the kernel's `is_causal` flag means, because there is no buffer
to read. The equivalence is asserted rather than assumed: on a perturbed model in float64, the recomputed
probabilities contracted with $V$ and pushed through `out_proj` equal the module's actual output, with the wrong
mask and the omitted rotary encoding each asserted to fail.

**Where it runs.** Not in the shared collection pass, whose retained quantities are what that pass produces —
threading an encoder-only tensor through it would edit a shared module for one model's benefit and pay for it on
every run of both. Instead this analysis runs its **own bounded pass** off the task and the loader on the
context, over a seeded, shard-stratified draw. It is opt-in: `caps.encoder_attention` caps how many segments it
scores, an absent cap means zero per this pipeline's opt-in rule, and the skip names the key. Nothing is
retained beyond the streamed accumulators plus the full maps for the capped draw the heatmap needs — the bound
matters, because the target's $(B, T, T)$ map is roughly 46 MB per batch at the evaluation batch size.

**Three readouts**, each per block, per stream, and cut by clinical class on the shared cohort grid.

1. **Per-head entropy against its truncation-aware ceiling** $\operatorname{mean}_t \log \min(t+1, c)$, with
   $c = T$ for the target and $c = W_U$ for the source — never $\log T$, for the same reason `attention` refuses
   $\log L$: at anchor $t$ only $\min(t+1, c)$ keys exist at all, so a head attending uniformly over everything
   available to it reads as increasingly *concentrated* the earlier the anchor when measured against a ceiling
   it could not reach. Taken per anchor then averaged, never as the entropy of the averaged profile.
2. **Attention mass by temporal distance** $t - j$, per head. This tests the design claim directly: that the
   target encoder gives the prior content-dependent access to long-range history the recurrent branch could not,
   and that the source encoder stays inside its window. The table stops at each block's own admitted reach — a
   source block admits no key beyond $W_U - 1$ and every bin past it is exactly zero by construction, so writing
   them out would be a table two thirds zeroes inviting a reader to find a shape in them. That the zeroes are
   exact is asserted against the accumulator.
3. **The measured source reach against the lag range.** The structural bound is
   $R_U = \min(R_{\mathrm{conv}} + N_U(W_U - 1),\ T) = 66$ steps $= 264$ s at the shipped geometry, deliberately
   shorter than the $360$ s furthest searched lag, because an encoder whose reach exceeded the lag range would
   already be doing the alignment the lag cross-attention exists to do. The mass-weighted median and 95th
   percentile of $t - j$ compose into $\widehat R = \min(R_{\mathrm{conv}} + \sum_b \widehat d_b,\ T)$, the same
   arithmetic with each block's *measured* hop in place of the largest hop it was allowed — so a stack whose
   every block put its mass at its window edge reproduces the structural bound exactly, which is what makes the
   two comparable. The cap at $T$ is the structural formula's own and is not cosmetic: on the full-prefix target
   encoder the per-block hops routinely sum past the segment. The unbounded arm
   (`source_attention_window: null`) reports its bound as **absent** rather than as $T$, because "no bound" and
   "a bound that happens to equal the sequence length" are different statements.

**It computes no test and produces no verdict.** Like `distributions`, it describes a mechanism rather than
adjudicating a difference, and a separation visible here is a reason to look rather than a finding.

**It does register headline scalars, and it has to.** The arm tables read the headline block and nothing else, so
a measured source reach that stayed in a CSV could never give the `sweep_window_*` family the measured x-axis
that is half its purpose. Six scalars: the two per-stream mean entropy ratios, and the source reach's median and
95th percentile in steps and in seconds. No verdict is registered beside them, because there is no threshold here
anyone has earned the right to set. Every key registers `null` on a skip rather than being omitted, so an arm
table's column exists whether the analysis ran or not.

**The class cut is drawn by the analysis, not by the runner's fan-out.** That fan-out resolves per-*recording
scalars* into violins; two of the three readouts here are fields — per head, per block, per distance — and a
violin cannot carry one. So this analysis writes its own `*_by_clinical_class.pdf` files from its own class-cut
CSVs, with the cohort order and palette read from the same two places every other figure reads them from. The
fan-out still runs, over the per-recording frame the analysis declares, and writes its own files beside them:
the stems are distinct, so nothing collides.

## How the output is misread

These are the readings the numbers invite and do not support.

**The coupling readout is not causal under the shipped configuration.** Under `causal_reach_budget_s: null` the
input features at step $t$ read far into their own future; the reach guard is a 95%-energy quantile rather than
a hard support, measured at roughly 20× suppression at 120 s rather than removal; and no finite budget is
currently trainable. This is a property of the shared **feature bank**, not of either encoder, so replacing the
history encoders changed nothing about it. Every run carries the refusal sentence verbatim in `preflight.json`
and `summary.json`:

```text
The source-conditioned KL is a coupling readout, NOT a transfer entropy and NOT causal. The input features come
from two-sided filters whose forward reach is bounded by a 95%-energy quantile rather than by a hard support, so
a feature at step t carries energy from its own future under every configured budget -- max_channel_reach_s
records how far the longest-reaching channel of this feature bank looks ahead, and under the shipped
causal_reach_budget_s: null no channel is pruned and no delay is applied at all. No number in this run may be
labelled a transfer entropy.
```

`tests/test_eval_naming.py` scans the whole artifact tree — plus this file and the figure guide — for the name
the readout refuses, with that sentence removed first.

**The encoder's causality disclosure is this model's own, and it is not the sibling's under another name.** The
comparison model records `causal_norm` and `n_causalized_norms`, which describe a time-pooling `GroupNorm` on a
history path. There is no such module here and no key that would turn one on: every normaliser on a history path
is per-token, which `tests/test_construct.py` proves by enumerating the survivors. So the record says
`time_pooling_normalisers: 0` with `time_pooling_normalisers_are_structural: true` and the test that proves it,
and reports instead `n_depthwise_init`, the two block counts, the source window, and the structural source
receptive field in steps and seconds beside the lag range with which is larger stated. A shared key that meant
nothing here would read as a setting somebody could change.

**An encoder attention weight is not a lag attribution.** `encoder_attention` measures how a *history state* was
built out of its own stream's past; `lag_kl` and `attention` measure which past *source* neighbourhood informed
the latent. A source encoder that attends 60 s back is not a model that found its coupling at 60 s, and the two
axes are not the same axis — one is $t - j$ within a stream, the other is the lag $\ell$ between the streams.

**A recomputed attention probability is the model's own.** It is not an approximation or a surrogate: every
operand is read off the module that computed the forward pass, and the equivalence through `out_proj` is asserted
in float64 against that module's real output. What the recompute cannot tell you is anything about a *different*
checkpoint, and it is measured over a capped, seeded draw rather than the whole split — the record carries the
draw.

**Specificity is read in prediction space, not in KL space.** See `perm_control`:
$K_{\mathrm{shuffled}} > K_{\mathrm{true}}$ is what a healthy model does, so a KL-space criterion would fail
exactly the models it should pass.

**Only the unfloored KL may be read as a rate.** `source_conditioned_kl_train` has free bits applied per dimension
per step before summing, so it exceeds the raw value by construction and hides a collapsed source pathway. The
shipped `free_bits: 0.0` makes the two coincide today, which is exactly why the distinction lives in code: no
headline path may resolve to it, asserted by test.

**Only an unpinned prior variance makes that rate meaningful.** A prior variance on its clamp inflates every
coupling number while every decoder-side diagnostic stays healthy. Read `prior_variance_not_pinned` before quoting
the KL.

**A smaller KL is not a weaker coupling, and this architecture is the reason the rule is stated here.** The whole
point of replacing the target encoder is a stronger prior, and a stronger prior lowers the source-conditioned KL
without the source having become less informative. Select on KL that comes with source-specific predictive gain;
see the selection rule below, which the cross-model table emits into its own output.

**The lag is the compensated lag** $\tau = 4(\ell + \delta)$, with $\delta$ read from `model.source_delay_steps`
and nowhere else. There is no "original-sensor" twin of this figure: the stored UP/FHR timeline is canonical,
the dataset builder's UP shift is part of the signal, and no reported lag adds it back or subtracts it. `source_delay_is_max_over_channels` travels beside
every reported lag, because per-channel delays have no single representative and the maximum makes every figure an
upper bound.

**A per-segment histogram is a description, not a cohort difference.** `distributions` is the one analysis that
draws the segment-level population directly, and it computes no test on purpose — segments overlap by the factor
below — so a separation visible there is a reason to look at `cross_subgroup`.

**Anchors are not independent, and every statistic is per recording.** Consecutive anchors' forecast windows
overlap in 29 of their 30 horizon steps and one GUID contributes up to ~37 segments, so per-segment $p$-values are
anticonservative by that factor. The chain is: per anchor → support-weighted mean within a segment → unweighted
mean over a GUID's segments → across GUIDs. A segment scoring zero anchors is excluded and **counted**, never
averaged in as `0.0`.

**Every class contrast is out-of-distribution, and the scope is wider than it looks.** The checkpoint trains on
`pre_training_dataset/*`, which is healthy-**with-background** only. So ACIDOSIS and HIE are unseen — and so are
the `healthy_no_bg_cs` and `healthy_no_bg_no_cs` subgroups. The summary computes `training_cohort_disjoint` from
both resolved dataset lists rather than asserting it, reports `null` rather than `false` where a list is absent,
and suppresses the out-of-distribution sentence when the two overlap.

**An eval score is not comparable with a `test_*` metric logged during training.** The populations differ, and
nothing in either number says so. The summary carries that sentence.

**A cohort's colour here is not its colour on a training figure.** See *Cohort order and colour* above.

**Coherence is not skill, and the gain is not a fault.** A forecast that reproduces every wiggle at half amplitude
scores $\gamma^2 = 1$ while carrying a quarter of the truth's variance as error. The converse trap is the same
number read the other way: the mean-square-optimal amplitude given a coherence is $g = \gamma$, not $g = 1$, so a
well-trained forecaster is *supposed* to shrink where it is uncertain.

**A band-integrated coherence is not a mean of per-bin coherences.** The emitted band number is
$\big|\sum_{\mathcal B} S_{xy}\big|^2 / (\sum S_{xx} \sum S_{yy})$ — the squared correlation of the band-passed
signals, and the only form under which the residual decomposition and the Parseval identity hold simultaneously at
band level.

**Nothing below 7.8 mHz is resolved, and nothing here is time-resolved *within* a recording.** Every spectral
estimate pools windows, so a coherence that *varies* through a recording reads as a lower constant one.

**Source coherence is a timing and association statement, not a directed one.** $\gamma^2(u,\cdot)$ is measured
against the **contemporaneous** uterine pressure, which the model never read. The causality refusal above applies
to it exactly as to every other readout here.

**The seam ratio is a property of the decoder, not of the fetus** — and the decoder is shared, so a seam artifact
found here is not evidence about the encoder replacement. The truth's own ratio is the control and must be read
first.

**`coherence`'s population may be smaller than `forecast`'s.** Whole-window validity is stricter than per-step
validity, so the coverage block can legitimately report a disagreement between them. That is the gap rule working,
not a fault.

**The sufficiency gap is an estimate, not a bound.** Both bias directions, above.

**Deceleration rates are per event, not per anchor.** Above.

**Lag ablation is absent, and so is necessity.** Band-restricted sufficiency ranking is blocked model-side: in
`LagCrossAttention` a band mask *replaces* rather than intersects the causal-validity mask, and `lag_band_mask` is
not a forward parameter. Unblocking it is a model-layer change outside this pipeline. Necessity is not measured
anywhere either — a band ranking measures **sufficiency** and reads exactly backwards if taken for a removal
ablation, so this pipeline emits neither rather than emitting the one that is routinely misread.

**A `nll_*_sample` key is a fixed /480 rescale of a block score**, not a mean over unmasked samples, so on any
anchor with masked forecast steps it under-reports. It ships beside that statement or not at all.

**The percentage is never `pred_gap` divided by a block score.** That ratio is the one a reader reconstructs from
the headline — `pred_gap_mc_nats / d_base_mc_nats`, both of which are registered — and it is not a percentage of
anything. $D_{\mathrm{base}}$ is a negative log *density* summed over 480 raw samples: it has no natural zero, it
is legitimately negative for a sharp forecast, and the ratio therefore changes sign with its own denominator and
is unbounded near it. The percentages this pipeline does emit live in the two spaces that have a natural zero.

## Arm tables and the cross-model comparison

`verify --runs <dir>` scans a directory of finished runs and emits one markdown document. Rows are keyed by the
value read from each run's **own** dumped `resolved_config.yaml`, never by directory name, and an `_ABSENT`
sentinel distinguishes a key a config never set from one it set to `null` — which matters more here than for the
comparison model, because `source_attention_window: null` is a real arm rather than a missing value.

**This model's arm tables** cover the twelve shipped `sweep_*.yaml` arms across five axes plus the architecture
row: the four source-window arms (8, 32, 64 and unbounded against the shipped 16), two target-depth and two
source-depth arms, the feed-forward width, the reach budget, and the stem arm — which is a pair of *empty* kernel
and dilation lists rather than a scalar, and so is legible in the architecture row rather than on an axis of its
own. The window table carries the **measured** source reach from the headline beside the configured window, which
is what `encoder_attention` exists to supply; a run whose `caps.encoder_attention` was unset reads `(missing)`
there rather than a zero, because that analysis recorded a skip. The reach-budget table carries the surviving
channel counts per stream. A run whose verdict could not be computed says why; a blank cell is never emitted.

**The cross-model table** is the comparison this package exists to make. It reads finished runs of *both*
architectures, keys each row by the `model_class` the run recorded in `summary.json`'s `run_context`, and puts side
by side: the parameter count, $D_0$, $D_1$, `pred_gap_mc_nats`, the unfloored KL, the lag peak with the run's own
`argmax_lag` verdict beside it, `kld_active_frac`, the collapse verdict and the epoch count. A directory holding
only one model's runs emits the table anyway and says so. A run that recorded no class is keyed `(unrecorded)` and
listed incomplete rather than guessed from its directory name.

The lag peak is never quoted alone. `results.sanity.checks.argmax_lag` is where a run says whether its own peak
means anything — inert at $\ell = 0$, or censored against the largest attainable lag — and an argmax is defined on a
flat profile exactly as it is on a real peak.

**The selection rule travels in the emitted document, not only here.** Do not select on KL magnitude: a stronger
target prior lowers the source-conditioned KL without the coupling having weakened, so a smaller KL is not a worse
model and a larger one is not a better one. Select on KL that comes with source-specific predictive gain, and treat
a competitive `d_base_mc_nats` as a precondition — an arm whose base reconstruction is worse than the comparison
model's best has not earned a reading on `pred_gap` at all. Such a row is **flagged with a footnote, never dropped**:
a suppressed arm reads as an arm that was not run. The baseline row is identified by `model_class` and sorted first,
so which row the flag is against is explicit, and no row is flagged at all when the directory holds no comparison-model
run.

## Non-goals

Named decisions, not gaps.

**Lag-band ablation and necessity.** Blocked model-side, exactly as for the comparison model, and unblocking it is a
model-layer change. See the misreadings section above.

**Raw-signal causality.** Every readout stays token-causal, but the input features are two-sided and
`causal_reach_budget_s` bounds the leak on an energy quantile rather than closing it. The refusal ships verbatim in
every run and is scanned for.

**Encoder KV-cache or streaming evaluation.** No streaming path exists in the model package, and the evaluation does
not create one. Every pass here is a whole-segment forward.

**Positional-encoding arms.** ALiBi, learned absolute, learned relative bias and a no-positional-encoding control have
no code path in the model. The evaluation does not add model code to measure arms that do not exist — what it can
measure about position is what the rotary encoding produced, which is `encoder_attention`'s distance profile.

**Any change to `nets/` in any package.** The model trains against those modules and both architectures share them;
an eval-only forward key would fork the objective the comparison depends on.

**A second implementation of anything the sibling already computes.** Every shared analysis, statistic, identity check
and figure primitive is imported. A fix to `coherence`'s Parseval gate or `lag_kl`'s support correction lands once and
both models get it.

## Operations

### Exit codes

The exit code is non-zero **if and only if a step raised.** Three things deliberately do *not* move it:

- a failed **sanity** check — the self-consistency block warns, logs at ERROR and leaves the code at 0, because a run
  whose every step succeeded can still be one nobody should quote a number from;
- a **coverage** warning (two analyses reporting different populations);
- an **inert-cap** warning (a cap no analysis read). `caps.encoder_attention` is read by this model's analysis and by
  nothing in the shared registry, so a run of the comparison model that set it would warn here — correctly.

That asymmetry is exactly why the offline acceptance gate exists separately: `verify` reads the sanity block and
refuses on it. A pipeline that conflated the two would either pass runs it should not, or fail runs whose only problem
was a skipped analysis.

### What a refused run leaves behind

Preflight runs **outside** the fail-soft step wrapper and raises `EvalPreconditionUnmet` — a distinct type, so a reader
seeing it in a traceback is looking at a refusal with an actionable message rather than at a crash inside an analysis.
A refused run leaves `resolved_config.yaml` and `eval.log` carrying the refusal, and **no `summary.json`**: a rejected
input must not produce a file that reads like a result, and a refusal an operator cannot read afterwards is half a
refusal.

### Re-running one analysis

```bash
python -m teb_vae.lag_attn_transformer_rws.eval.run --output-dir <a finished run> --only lag_kl
```

reads the tables, builds no model and touches no GPU. It is **non-destructive but not additive**. The prior
`summary.json` and `steps.json` are renamed aside to `summary.bak.<stamp>.json` before the new pass writes — byte
identical, with the backup path logged — but the new summary describes **only what this pass ran**: its headline is
mostly `null`, its manifest lists a handful of files instead of forty, and its exit code is 0. So **read the backup, not
the new summary, for anything the re-run did not itself produce.**

`--only encoder_attention` against a finished directory is the one selection that can never produce numbers offline: it
needs a model and a loader, and a pass with no checkpoint has neither. It records a skip saying so and exits 0, rather
than raising.

`preflight.json` is deliberately *not* preserved: a pass with no checkpoint cannot regenerate the causality disclosure
and reads that file back instead, so renaming it would take it from the one pass that needs it. It is stable within a
directory anyway — a different checkpoint's tables are refused before it is ever read.

The tables carry a provenance sidecar naming the checkpoint hash, the seed, the row count and the `eval_config` digest.
A mismatch raises `TablesProvenanceMismatch` — again a distinct type, because the tables are intact and readable and
simply belong to another run.

### Guard recovery table

One row per way preflight refuses a run. Each refusal's own message names the fix; this is the index, and
`tests/test_eval_docs.py` asserts every raise site in the **shared** `preflight.py` has a row — so a refusal added there
is reported in both packages.

| Refusal begins | Cause | Recovery |
|---|---|---|
| `dataset_config still carries the` | A shard or statistics path is still a `REPOINT_ME` placeholder. Checked **first**, so the message names the real cause rather than a missing file someone would then go looking for. | Point `vae_test_datasets` at the shared k-fold holdout split and `stat_path` at statistics regenerated from the same dataset at `trim_minutes: 1.0`. |
| `dataset_config.vae_test_datasets is empty` | No evaluation shard is configured. | Set the eight holdout subgroup shards in the override delta. |
| `dataset_config.vae_test_datasets names shard(s) that do not exist` | A configured shard is absent. When the containing directory is absent too, the message adds the two dataset build modes. | Build the dataset in `holdout` mode: the default `augmented` mode writes per-fold test splits and no shared `test/` directory, and a per-fold split is not a substitute — one pool, no fold loop, no double counting. |
| `dataset_config.dataloader_config.dataset_kwargs.trim_minutes must be` | Not `1.0`. The whole raw-index geometry — the forecast of anchor $t$ starting at raw sample $16(t+1)$ — assumes the trimmed grid; untrimmed it starts at $16(t+16)$, one full minute later, and nothing fails loudly. | Set `trim_minutes: 1.0`, and confirm the statistics file was computed on the same grid (a mismatch there only warns). |
| `dataset_config.dataloader_config.dataset_kwargs.load_fields is missing` | One of the five clinical fields is absent. The loader **skips** a field a shard does not carry, silently, so this presents downstream as "no classes found" or "no trajectory data" rather than as a data problem. | Merge this package's committed override delta, which adds `target`, `epoch`, `cs_label`, `bg_label` and `time_from_labor_onset`. |
| `the config disagrees with the checkpoint it is evaluating` | A geometry or objective key in the config contradicts the checkpoint's own `model_kwargs` / `hyper_parameters`. Nineteen keys are reconciled here against the comparison model's thirteen: `causal_norm` is dropped (the constructor refuses it) and the seven encoder keys are added, every one of which changes what the numbers mean. | Evaluate the checkpoint against its own `resolved_config.yaml`, which the training run writes beside it. ($\beta$ and its ramp are recorded and deliberately not compared.) Note that `source_attention_window: null` is a **value** here, not an absent key — a checkpoint built unbounded refuses a config declaring `16`, and it should. |
| `every witness tensor in this model is still exactly at the value the constructor gave it` | No checkpoint weights reached the model. `load_checkpoint_strict` returns `None` rather than raising, so an unchecked load would report randomly initialised weights as a measurement. | Check the checkpoint path and that its state dict aligns. This is a weight-space check, not a behavioural one: a genuinely trained model whose *source pathway* collapsed still has nonzero weights here and passes, because that finding must be reported rather than refused. |
| reused guard: `stat_path` | The normalisation statistics file is missing or unreadable. | The trainer's own message names the command that regenerates it. |
| reused guard: `raw_target_normalized` | `'fhr'` is not in `normalize_fields`, so the raw target arrives at ~140 bpm while the decoder's learned log-variance models a $z$-scale. | Add `'fhr'` to `normalize_fields`. |
| reused guard: `causal_budget_resolves` | The configured reach budget does not resolve against the shipped filter bank. This is a property of the shared feature bank, so it fails identically for both architectures. | The trainer's message names the surviving channel counts per block. |
| reused guard: `declared_widths` | The **model's** `c_y` / `c_u` disagree with the test shard's stored widths. Compared against the model rather than the config (the evaluation rebuilds from the checkpoint), and against `vae_test_datasets[0]` rather than the training list. | Do **not** "fix" this by reverting `c_y` / `c_u`; the shard and the checkpoint were built from different channel selections. |

### Dependencies

No new ones. `torch`, `numpy`, `pandas`, `matplotlib`, `h5py`, `pyyaml` and `loguru` are already in use; `scipy` is
imported lazily at each call site; `pyarrow` carries the per-anchor table, so there is one format and no fallback branch.
`verify.py` needs none of them except `pyyaml`, and that only for the arm tables — the acceptance gate itself is a stdlib
parse.

### The gate

From the repository root:

```bash
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_rws/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_rws/tests -q -m slow
```

Both are needed, and they cover different things. The fast gate holds the binding, the layering walk, the registry parity
against the sibling, the encoder-attention equivalence and its arithmetic, the verify tables and the documentation
bindings. The slow gate holds the one full pipeline run — which is what keeps `figure_manifest.json` equal to what a run
actually emits, and therefore what makes the fast documentation gate mean anything.

## The first production checkpoint

Everything above was built and proved against a CPU-trained throwaway checkpoint over generated fixture shards, because
no production run of this architecture had finished. The first one is therefore the first time these guards meet real
paths, a real split and a real fit. This is the order to run things in, what each refusal means, which numbers to read
before quoting any of them, and what a healthy first run looks like beside a collapsed one. **Every command below runs
from the repository root**, which is also where the runner puts itself: both entry points `chdir` there and say so in the
log, because a relative shard path resolved against some other working directory surfaces as "no samples match the
specified filters" and names nothing.

### Step one — repoint the shards, and confirm the statistics

Do this before anything else. It is the refusal a first run actually hits, and preflight checks it **first** for exactly
that reason: a placeholder path would otherwise fail later as a missing file, and the operator would go looking for the
file instead of for the placeholder.

The committed delta ships eight `REPOINT_ME` shard paths on purpose — a delta that pointed at one machine's filesystem
would be wrong everywhere else, silently. Copy it and edit the copy:

```bash
cp teb_vae/lag_attn_transformer_rws/eval/configs/eval_overrides.yaml ~/eval_overrides.local.yaml
```

In that copy:

1. Set the eight `dataset_config.vae_test_datasets` paths to the shared k-fold **holdout** split — one HDF5 per canonical
   subgroup, all eight, all three clinical classes. If the `test/` directory is absent, the dataset was built in the
   default `augmented` mode, which writes per-fold test splits instead; rebuild in `holdout` mode. A per-fold split is not
   a substitute: this evaluation wants one pool, no fold loop and no double counting.
2. Confirm the `dataset_config.stat_path` the checkpoint's own `resolved_config.yaml` names still exists **and** was
   regenerated from that same dataset build at `trim_minutes: 1.0`, with
   `hdf5_dataset/calculate_dataset_stats.py`. A missing file is a refusal; a *stale* one only warns, and it is
   load-bearing rather than cosmetic here — the raw FHR is the reconstruction target, so the decoder's learned
   log-variance models a $z$-scale and unnormalised input arrives at $\sim 140$ bpm.
3. Nothing, unless the box is memory-tight. The `caps` block already ships set, so a stock run emits the complete
   artifact set — including this model's own `encoder_attention` and the window arm table's measured-reach column.
   If memory is short, lower `waveforms` first (~2.4 MB per retained sample, ~310 MB at the shipped 128); it bounds
   the event analysis's runtime as well. Do not set `oracle`: absence there means *every* segment, so naming a number
   would reduce what the sufficiency probe is fitted on.

Change nothing else. Every other key in that file is the comparison model's, key for key — including the three shared
caps — and a value that differed would make every side-by-side number a comparison of two protocols rather than of two
architectures. The single exception is `caps.encoder_attention`, which gates an analysis the comparison model does not
have; the equality test exempts exactly that and nothing else.

### Step two — probe the split before paying for a pass

```bash
python -m teb_vae.lag_attn_transformer_rws.eval.probe \
    --config <run>/model_checkpoints/resolved_config.yaml \
    --overrides ~/eval_overrides.local.yaml
```

No model, no checkpoint, no GPU: it opens the loader and reports what the split yields. Read the per-shard, per-class and
per-label counts. Eight shards with recordings in each, all three clinical classes present, and a `target` field that is
not truncated to one value are the three things the run's own sanity block will re-check afterwards — meeting them here
costs a loader pass instead of a full one.

### Step three — a bounded smoke run

```bash
python -m teb_vae.lag_attn_transformer_rws.eval.run \
    --checkpoint <run>/model_checkpoints/<name>.ckpt \
    --overrides ~/eval_overrides.local.yaml \
    --max-batches 2
```

Every preflight guard runs at full strength on a two-batch pass, and so does every analysis, for minutes rather than
hours. **Read the refusals and the step records; do not read the numbers.** `--max-batches` is a prefix over the
unshuffled eight-shard split, so it draws one subgroup and one class — which is the whole reason `eval_config.max_samples`
exists separately and is drawn stratified.

### Step four — the full pass

```bash
python -m teb_vae.lag_attn_transformer_rws.eval.run \
    --checkpoint <run>/model_checkpoints/<name>.ckpt \
    --overrides ~/eval_overrides.local.yaml
```

### Step five — gate it before reading it

```bash
python -m teb_vae.lag_attn_transformer_rws.eval.verify <run>/eval_results/summary.json
```

The runner's exit code says only whether a step raised. This is the code that says whether the run is quotable, and the
two are deliberately different — see the exit-code section above.

### If it refuses

Each refusal's own message names the fix, and the guard recovery table above is the index. Two entries matter more than
the rest on a first run:

- **`the config disagrees with the checkpoint it is evaluating`** — evaluate a checkpoint against the
  `resolved_config.yaml` its own training run wrote beside it, never against a config file that currently says something.
  Nineteen keys are reconciled, and `source_attention_window: null` is a **value** among them: a checkpoint built
  unbounded refuses a config declaring `16`, and it should.
- **`declared_widths`** — the shard's stored channel widths disagree with the *model's* $c_y$ / $c_u$. Do **not** "fix"
  this by editing the widths. The shard and the checkpoint were built from different channel selections, and the
  reconciliation is what caught it.

A refused run leaves `resolved_config.yaml` and `eval.log` carrying the refusal and no `summary.json`, so there is never a
file that reads like a result for an input that was rejected.

### Which numbers to read, and in which order

The order is not a preference. Each step decides whether the next one means anything.

1. **The sanity block, before quoting anything at all.** `results.sanity.n_failed` and `results.sanity.failed`. It does
   not move the exit code by design, so a run every step of which succeeded can still be one nobody should quote. A failed
   `kl_identity` or `per_anchor_recombines` says the tables do not describe the same pass.
2. **`weights_loaded`**, at `preflight.checks.weights_loaded` and re-read by the gate. A weight-space check: it says the
   checkpoint reached the model, not that the model is any good — a genuinely trained model whose *source pathway*
   collapsed passes it, which is the point.
3. **`prior_variance_not_pinned`, and `logvar_prior_floor_frac` beside it, before quoting any KL.** If the prior's
   log-variance is sitting on its clamp, the source-conditioned KL is a number divided by a bound rather than a rate — and
   nothing else in the headline shows it. The decoder's variances stay healthy while this fails.
4. **`d_base_mc_nats` before `pred_gap_mc_nats`.** The gap is a difference *of* the base reconstruction; a base branch that
   reconstructs badly changes what the difference means. Read the gap with its Wilcoxon, its bootstrap interval over
   recordings and its positive fraction, from `coupling`, rather than as one number.
5. **The lag peak with `results.sanity.checks.argmax_lag` beside it, never alone.** An argmax is defined on a flat profile
   exactly as it is on a real peak; that check is where the run says whether its own peak means anything.
6. **`encoder_attention`'s measured source reach last**, against the configured window and against the lag range. It
   describes a mechanism and adjudicates nothing, so a separation visible there is a reason to look rather than a finding.

### What a healthy first run looks like, and what a collapsed one looks like

**Healthy.** `verify` exits 0 with no `FAIL` and no surprising `INCONCLUSIVE`; `sanity.n_failed` is 0. `d_base_mc_nats` is
finite and in the same range as the comparison model's. `pred_gap_mc_nats` is positive, its bootstrap interval excludes
zero and its positive fraction over recordings is well above a half. `kl_active_dims` clears `min_active_dims` and
`kl_top_dimension_share` is not near 1. `logvar_prior_floor_frac` is small. The prior-shuffle degradation clears
`prior_shuffle_min_nats` and `perm_control`'s three losses degrade under the GUID-aware shuffle. The lag peak is neither
at $\ell = 0$ nor censored against the largest attainable lag, and `argmax_lag` passes.

**Collapsed.** `source_conditioned_kl_raw_nats` near zero, `kl_active_dims` at zero or one, `pred_gap_mc_nats` at
approximately zero with an interval straddling it, and `latent_not_collapsed` reporting `fail`. The training side says the
same thing independently and earlier: a completed run is collapsed when `val/source_conditioned_kl_raw` is below $0.02$
nats per anchor at every one of its last five epochs, or when its final `val/kld_active_frac` is below $2 / d_z$. A
collapsed run is a finding to report, not a run to refuse — but nothing downstream of the bottleneck is quotable from it.

**The impostor to know about.** A *large* KL beside `logvar_prior_floor_frac` near 1 is not a strong coupling. It is a
prior pinned at its clamp, and the KL above it is measuring the clamp. This is why step 3 above comes before step 4.

### Before comparing arms

State the rule before running the command, because the table invites exactly the ranking it forbids: **do not select on KL
magnitude.** A stronger target prior lowers the source-conditioned KL *without the coupling having weakened* — the prior
simply predicts more of what the source was carrying — so a smaller KL is not a worse model and a larger one is not a
better one. Select on KL that comes with **source-specific predictive gain**, and treat a competitive `d_base_mc_nats` as
a precondition: an arm whose base reconstruction is worse than the comparison model's has not earned a reading on
`pred_gap` at all. Such a row is flagged with a footnote rather than dropped, because a suppressed arm reads as an arm that
was never run.

Then, over a directory holding finished runs of the arms and — for the cross-model table to have two models in it — of the
comparison model as well:

```bash
python -m teb_vae.lag_attn_transformer_rws.eval.verify --runs <dir-of-runs> --out RESULTS_arms.md
```

The rule travels in that document, so it is read where the table is read rather than only here.

**One ordering trap, because it is silent.** An offline `--only` re-run rewrites `summary.json` to describe only what that
pass ran, and a pass that built no model records `model_class: null` — so a directory whose runs were last touched by a
re-run keys every cross-model row `(unrecorded)` and lists it incomplete. Generate the tables before re-running a single
analysis, or point `--runs` at directories whose `summary.json` is the full pass's. The backup beside it
(`summary.bak.<stamp>.json`) is the full pass's record either way.

The comparison model's suite must stay green too, because this package's changes to the shared pipeline live there:

```bash
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_rws/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_rws/tests -q -m slow
```
