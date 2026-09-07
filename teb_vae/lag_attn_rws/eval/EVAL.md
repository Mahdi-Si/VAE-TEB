# The evaluation contract

What a run of `teb_vae.lag_attn_rws.eval` is, what it leaves behind, what each analysis means,
and how the output is misread. `FIGURE_GUIDE.md` beside this file documents every emitted PDF;
this document is everything that is not a figure. Both are bound to the code by test: every
registered analysis has a heading here, every resolved `eval_config` key is mentioned here, and
every figure in the committed `figure_manifest.json` has a guide entry.

## What a run is

One command reads one checkpoint and writes one reviewable directory:

```bash
python -m teb_vae.lag_attn_rws.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt
```

The configuration is the checkpoint's own `resolved_config.yaml` — found beside it, never a
second config file — with the committed `eval/configs/eval_overrides.yaml` delta deep-merged
over it. The delta repoints the shards at the shared k-fold holdout split, adds the five
clinical `load_fields`, and carries the `eval_config` block; both the original and the merged
value of every overridden key are recorded in the summary. Preflight then refuses the run
outright when the merged result contradicts the checkpoint.

The expensive part happens once. A single shared collection pass decodes four latent branches
over every anchor at $K$ Monte Carlo draws and writes two durable tables — `per_sample.csv`
(one row per segment, with the clinical labels attached) and `per_anchor.parquet` (keyed
`(guid, epoch, anchor)`) — plus a vector sidecar and the aggregated readouts. Every analysis
then reads those files, which is why

```bash
python -m teb_vae.lag_attn_rws.eval.run --output-dir <a finished run> --only coupling
```

re-runs an analysis offline with no checkpoint, no model and no GPU. `--max-batches` is a
smoke-run batch cap (a prefix by nature); `eval_config.max_samples` is the seeded *stratified*
cap, and the two are not interchangeable — a prefix over the unshuffled eight-shard split draws
one subgroup and one class.

**One thing an offline re-run cannot do: report a criterion the tables predate.** The verdict
block travels with the collection record, so a re-run reports the verdicts the *collecting* pass
decided — which is correct until the registry moves. When it has, the reuse path refuses with
`StaleCachedVerdicts`, naming the criteria that appeared or disappeared. It is a refusal rather
than a repair because only some criteria are decidable from what a collection record keeps: the
predictive pair is, the calibration census is not, and a repair that worked for the cheap
criteria and failed for the rest would be a worse failure than a clear stop. The fix is to
re-collect — a new `--output-dir`, or the collection deleted from this one, with `--checkpoint`
so the pass has a model. Every *analysis* number, `source_margin` included, is recomputed from
the per-sample tables and needs no re-collection.

A finished run is checked mechanically, and the calibration study's tables are generated, by
the same offline module:

```bash
python -m teb_vae.lag_attn_rws.eval.verify <run>/eval_results/summary.json
python -m teb_vae.lag_attn_rws.eval.verify --runs <dir-of-runs> --out RESULTS_arms.md
```

`verify` reads files a run left behind and nothing else — no model, no shard, no `torch` (the
layering test walks its imports with `torch` on the forbidden list, so the property is proved
rather than promised). The gate names which `pred_gap` column it reads: `pred_gap_mc_nats`,
the Monte Carlo marginalised score. The arm tables key every row by the swept value read from
each run's own dumped `resolved_config.yaml`, never by directory name, and read each run's
training `metrics_history.csv` for the epoch count, the final `val/kld_active_frac` and the
collapse verdict (`collapse.py::is_collapsed`, over the tail of the per-epoch series).

## Output layout

Everything lands in `<run>/eval_results/`:

| Artifact | What it is |
|---|---|
| `summary.json` | The whole run: readouts, verdicts, headline, sanity, cohort, coverage, run context, causality disclosure, step records, artifact manifest. |
| `steps.json` | The per-step heartbeat, rewritten as each analysis finishes — a killed run's record of how far it got. |
| `preflight.json` | Every guard's verdict and the causality disclosure; reused (not regenerated) by a model-free re-run. |
| `loader_probe.json` | The population record: per-shard, per-class and per-label counts the sanity checks read back. |
| `resolved_config.yaml` | The merged configuration the run actually used — the file an offline re-run reads. |
| `eval.log` | The run's log, including any refusal. |
| `per_sample.csv`, `per_sample_vectors.npz` | One row per segment: every scalar readout plus labels and provenance; the vector readouts in row order. |
| `per_anchor.parquet` | Per-anchor scores, KL, argmax lag, coverage, `seconds_since_contraction`. |
| `coherence_spectra.npz` | The cross-spectral maps pooled over the split and over each clinical class, resolved by frequency and lead time. Fixed size, independent of the split's length; **not** row-aligned with `per_sample.csv`, which is why it is its own sidecar. |
| `collection.json` | The collection record: readouts, provenance sidecar, denominators, retention plan, accumulators. |
| `band_partition.json`, `band_channel_map.csv` | The input channel map (the unskippable data-side step). |
| `<analysis>/…` | One subdirectory per analysis: its CSVs and PDFs. |

Three summary blocks matter more than the rest. The **headline** is a flat registry of scalars
and verdict statuses; a number not registered there is invisible to the acceptance gate and the
arm tables, which read it and nothing else. It carries two `pred_gap` columns under names that
say which is which — `pred_gap_mc_nats` (the headline, the log of the average likelihood over
$K$ draws) and `pred_gap_train_path_nats` (the single-draw objective-parity column) — and three
percentage columns restating the same finding proportionally, `pred_gap_rmse_pct`,
`pred_gap_mse_pct` and `pred_gap_mc_likelihood_pct`. `pred_gap_convention` says in the artifact
itself which is which and which spaces they are measured in. The
**sanity** block is the run's three-valued self-consistency record (the KL identity, the
cross-table recombination, the lag identities, the population checks); it deliberately does
*not* move the exit code. The **verdicts** are the model's own acceptance criteria, in registry
order, never a bare boolean. The **run context** block beside them records the parameter count,
the checkpoint's training epoch, the anchor-coverage distribution and the observed objective
magnitude — the facts the arm tables and the first-run checklist consume.

## The four layers

The package is layered, and an AST walk (`tests/test_eval_self_contained.py`) enforces the
import rules, resolving aliased, lazy and relative forms alike:

| Layer | Modules | May import |
|---|---|---|
| 0 — pure | `binding`, `config_schema`, `verify`, `events`, `frames`, `lag_axis`, `cohort`, the `_reuse` seam | no Lightning, no `model.*`, no `task`/`trainer`/`plotting`; `verify` additionally no `torch` |
| 1 — model-touching | `metrics`, `collect`, `preflight`, `oracle` | `task`/`trainer` only via the named `EXEMPTIONS` table |
| 2 — I/O and presentation | `analyses/*` | layers 0–1; never another analysis, never Lightning |
| 3 — orchestration | `run` | everything, with `task`/`trainer` named in `EXEMPTIONS` |

The sibling evaluation package (`teb_vae/lag_attn/eval`) is reachable only through its
model-free modules, named in an allow-list; `model/*` is forbidden everywhere. Anything two
analyses share moves one layer down — that rule is why `frames`, `lag_axis`, `cohort` and
`events` exist.

The `EXEMPTIONS` table is asserted **minimal**: a module listing a name it no longer imports is
a permission that outlived its use, and the next reach for that name would go unreported. The
exemptions themselves are deliberate rather than reluctant — `metrics` assembles the model's
inputs through the task's own builders precisely so an evaluation cannot feed the model a
differently assembled stream than training did, and re-implementing those builders to win a
layering rule would reintroduce the exact drift they exist to prevent.

### The model binding

`binding` is the newest layer-0 module and it is the reason `run.main` takes a parameter this
document did not previously mention. **The pipeline no longer names one model; it takes one.**
A `ModelBinding` is a frozen record of the handful of facts the pipeline cannot derive — the
classes to rebuild from a checkpoint, the constructor keys reconciled against it, the encoder's
own half of the causality disclosure, the package's committed override delta, and any analyses
and headline scalars that model alone can have. The module itself imports nothing but
`dataclasses`, `pathlib` and `typing`, which is what keeps it in layer 0; the concrete instances
name model classes and therefore live beside the code that constructs one.

**Nothing about an `lag_attn_rws` run changed.** `RWS_BINDING` lives in `run.py`, `main` takes it
as a keyword default, `GEOMETRY_KEYS` is still the constant in `preflight.py` and is what the
binding is set to, and its `extra_analyses` and `headline_scalars` are empty and pinned empty by
test — so the registry every run selects from, and the headline block every run emits, are the
ones they were before the seam existed. That the refactor moved no number is proved rather than
asserted: a `slow`-marked gate re-runs the pipeline against the tiny fixture and compares every
emitted artifact against a manifest of digests captured before any binding code was written.

The second consumer is `teb_vae/lag_attn_transformer_rws/eval`, which evaluates
`SeqVaeLagAttnTrfRws` — the same model with both history encoders replaced — through this
pipeline's runner, this pipeline's collection pass and all seventeen of these analyses, adding one
of its own that profiles the encoders it replaced them with. Its own `EVAL.md` is the contract for
that package; what matters here is why the seam is worth a parameter. The two architectures exist to be compared, and a second copy of this pipeline is how
two things that must stay comparable stop being comparable: the first fix to an analysis lands on
one side, and the two summaries quietly stop meaning the same thing.

## Configuration reference

Everything that shapes a run lives in the `eval_config` block of the override delta, because
that block is dumped into the run directory and is the durable record. Every key is validated
against a closed set — an unknown key raises and names the valid keys, a `bool` where an `int`
is expected raises (`True` would silently cap at 1), and a cap of `0` raises.

| Key | Meaning |
|---|---|
| `seed` | Seeds `random`/`numpy`/`torch` and derives the loader-shuffle, derangement and Monte Carlo generators by fixed offsets. Two runs of one checkpoint at one seed compare byte-identical on `results`. |
| `num_mc_samples` | Monte Carlo draws $K$ per anchor for the marginalised score, under common random numbers across branches. $K = 1$ is one draw of the same estimator, not the training-path score: under `base_decode: mean` the training path decodes the base branch at the prior mean, which the estimator never does. |
| `max_samples` | Seeded **stratified** global sample cap; `null` evaluates the whole split. |
| `caps` | Per-quantity retention caps (`waveforms`, `attention`, `pages`, `oracle`). Retention is opt-in: a quantity absent from `caps` is retained for no samples — except `oracle`, where absence means every segment, because a probe fitted on nothing is not a cheaper measurement but no measurement. |
| `prior_shuffle_min_nats` | The provisional margin the prior-shuffle degradation must clear; the verdict always reports the measured number beside it. |
| `min_active_dims` | Active latent dimensions below which the latent counts as collapsed. |
| `event_lag_window_s` | Seconds after a detected contraction within which an anchor counts as event-conditioned. |
| `bootstrap_resamples` | Resamples behind every bootstrap interval, drawn over recordings — never over anchors, whose windows overlap 29/30. |
| `figure_format` | Image format every figure of the run is written in, as a matplotlib filetype (`pdf`, `svg`, `png`, `eps`, …); validated at config load against the installed matplotlib's own list. `null` — the shipped setting — keeps the `pdf` default, which is what `figure_manifest.json` and `FIGURE_GUIDE.md` record and what the smoke suite compares a real run against; a run that changes it writes filenames those files do not list. |
| `max_hours_before_delivery` | How far before delivery a segment may be recorded and still be evaluated, in hours; `4.0` keeps the last four hours, `null` — the shipped setting — evaluates everything. **The bound is on the population, not on an axis**: it is applied to the delivery clock before anything is binned, so every clock answers for the same segments and the second-stage clock re-bins that population on its own signed axis rather than being cut at a second, differently-defined four hours. It moves cohort sizes, window counts and every trajectory, so a bounded run is not comparable with an unbounded one — which is why it is a key, recorded in the run's dumped config. Minimum one 0.5 h bin. |

Deliberately **not** keys: the significance level and the trajectory bin width (an operator who
could widen them could make a difference appear or disappear), and any lag-band selection (the
band ablation is a non-goal, so a band key would be inert by construction). Their absence is
asserted by test, not merely intended.

### Objective keys this pass reads rather than sets

The objective is not an `eval_config` surface. A checkpointed pass rebuilds the task from the
checkpoint's **own** `hyper_parameters` — `beta_schedule`, `kld_beta`, `beta_prior`, `lambda_full`,
`lambda_base`, `likelihood` and `free_bits` — and refuses a checkpoint that carries none, because
scoring it under assumed defaults would report a different objective's numbers. On the offline path,
where there is no checkpoint, the same keys are read from the dumped `model_config.VAE_model` block,
which preflight has already reconciled against the checkpoint on every checkpointed run.

`beta_prior` is the newest of them and the one an operator is most likely to assume defaults for. It
weights the prior's scale rate $R_p$, the fourth term of the training objective; a checkpoint
trained at a non-zero weight must be scored under that same weight, so it is read back rather than
defaulted, and it appears in `run_context.observed_loss_scale` beside the term it weights. Two
readouts follow from it and exist whatever the weight was — including on runs that predate the
anchor, which is exactly where they diagnose: the per-recording `prior_rate` column, and the
`prior_rate_nats` headline scalar the arm tables read. Neither is conditional on `beta_prior` being
non-zero; a column that appeared only for anchored runs would be missing from the runs it exists to
explain.

Weights are recorded and, unlike geometry, deliberately **not** compared against the config by the
preflight guard: $\beta$ and its ramp weight the training total and enter no evaluated readout.

### Cohort order and colour

Two presentation conventions every table and figure that resolves a quantity by cohort obeys.
Neither is a setting, for the reason the significance level is not one: an operator who could
reorder or recolour the cohorts could make a difference look like a trend.

**The order is clinical, not alphabetical, and it runs worst first**, on both axes:

| Axis | Order, left to right |
|---|---|
| `clinical_class` | HIE, acidosis, healthy |
| `subgroup` | `hie_cs`, `hie_no_cs`, `acidosis_cs`, `acidosis_no_cs`, `healthy_bg_cs`, `healthy_bg_no_cs`, `healthy_no_bg_cs`, `healthy_no_bg_no_cs` |

`labels.ordered_groups` is the one function that decides it — `cohort.ordered_groups` is a
one-line binding of it, so this package and the sibling cannot come to disagree — and both orders
are read off tables that already exist rather than restated: `labels.CLASS_NAMES`, keyed by the
dataset's own class codes $1, 2, 3$ and read in reverse, and `labels.CANONICAL_SUBGROUPS`, likewise.
So a subgroup added to the dataset appears in the right place with no edit here. The default
everywhere the function is *not* called is alphabetical, and alphabetical is wrong in a way that
looks fine: it puts `acidosis` left of `hie` on every class figure, and on the subgroup axis it
interleaves the three classes (`acidosis_cs`, `acidosis_no_cs`, `healthy_bg_cs`, …) so that neither
the severity ordering nor the background/caesarean structure is visible. A cohort the order does
not know — a non-canonical shard stem — sorts **after** every one it does, and is never dropped.

It reaches the CSVs as well as the figures, and that pairing is the point: the grouped `*_by_*`
tables, the stratified lag profiles and peak rows, the conditioned-coupling rows, the per-window
significance records and the summary's own population counts are all written in it, so a table can
be read against the figure beside it row for row.

**The colour is the severity**: green for healthy, amber for acidosis, red for HIE, with each
subgroup a shade of its own class, light to dark in the order above. The mapping is a *table*
rather than an assignment pass, so a cohort keeps its colour whichever others a figure contains.

The palette is **this package's**, `figures_seam.CLINICAL_CLASS_COLORS`, and deliberately not
`utils.style.CLASS_COLORS_DEFAULT`, which paints healthy blue and is shared with the `lag_attn`
sibling and with `model/transformer_experiment` — repainting it there would restyle two other
projects to satisfy this one's convention. The cost is stated rather than hidden: **an evaluation
figure of a cohort is not the same colour as a training-callback figure of that cohort**, and the
two are reconciled by legend rather than by hue.

The convention reaches `cross_subgroup`'s effect heatmap too, whose x axis is cohort *pairs*. The
column order is clinical and so is the naming inside a column: which cohort of a pair is `left`
comes from the shared `pairwise_comparisons`, which names a pair in the order it receives the
cohorts and receives them in the canonical one, worst first — so a column reads more severe
against less severe and a positive Cliff's delta means the more severe cohort's values run higher.
Reorienting a column by eye still flips its sign against the number in
`cross_subgroup_pairwise.csv`.

## The analyses

One section per registered analysis, named for its module. `band_partition` always runs and is
not selectable; the rest are what `--only` and `--skip` choose between, in this order.
`cross_subgroup` is deliberately last, and that ordering is load-bearing: it reads the
per-recording CSVs the analyses above it write.

### band_partition

What each of the model's input channels is, read off the shards' own `sel_*` provenance rather
than re-derived: one row per input channel across the 109-channel target stream and the
58-channel source stream, laid out as the model receives them, each mapped to a band, a kind and
a centre frequency in Hz. It describes the model's **inputs** and is the data-side companion to
the causality disclosure.

The fourteen scattering channels with no recoverable centre frequency are recorded as such rather
than omitted, per stream. A shard carrying no `sel_*` attributes is a recorded skip, not a raise.

### forecast

Is the forecast any good, in units a clinician reads, and where in the horizon. A block score
alone cannot answer that: it is a negative log density summed over $H \cdot R = 480$ raw samples,
so it is large under every predictor and its scale is set by the block size rather than by the
model. Three things make it readable — skill against **three trivial baselines** (persistence,
climatology, and the segment's own mean) scored through the model's own masked scorer with the
identical mask; the error in **bpm**; and the score resolved by **horizon step**.

The MSE-space skill $1 - \mathrm{MSE}_m/\mathrm{MSE}_b$ is the one with a natural zero. The
NLL-space column beside it is a **difference** in nats, `advantage_nats_per_anchor`, not one
minus a ratio: a log score has no natural zero, so the ratio of two of them is not bounded above
by one and changes sign with the baseline's. The baseline $\sigma$ is fixed at 1 in $z$-space and
recorded, because a skill score against a point predictor is entirely determined by the $\sigma$
handed to it — and a learned-$\sigma$ model would otherwise beat a fixed-$\sigma$ baseline partly
on variance modelling alone.

Persistence carries the last **observed** sample forward rather than the last one: a gap is
stored as 0 bpm, roughly $-11\sigma$ after z-scoring, and carrying that would measure the gap.
The horizon curve is computed on the **single-draw** path and says so: the marginalisation does
not commute with the sum over $\tau$, so a marginalised curve would not sum back to the
marginalised headline.

### coupling

What the source added, per recording, with the uncertainty on it. `pred_gap` in both estimators —
the Monte Carlo marginalised headline and the single-draw training-path parity column, never
merged — with the fraction of recordings where the gap is positive, a paired Wilcoxon over the
per-GUID vector, bootstrap intervals over recordings, and quantiles rather than only means.

The positive fraction reports its **denominator**: `np.nan > 0` is `False`, so unscored segments
would otherwise count silently as evidence against. The KL travels beside the gap as a
**description** rather than as a second answer — it is inflated by an arbitrary factor whenever
the prior variance sits on its clamp, and unlike `pred_gap` it says nothing about whether the
forecast improved.

**The same finding as a percentage**, because nats state no proportion: whether 3 nats over a
480-sample block is a large improvement is not readable off the number, and two checkpoints whose
block scores differ in scale cannot be compared on it at all. Three columns, in the two spaces
where a ratio has a natural zero, each computed per recording and then averaged and each
bootstrapped over recordings like everything else here:

| Column | What it is |
|---|---|
| `pred_gap_rmse_pct` | $100(1 - \mathrm{RMSE}_{\mathrm{full}}/\mathrm{RMSE}_{\mathrm{base}})$ — the percentage of the point-forecast error the source removed. **Scale-free**: the same number in $z$ units and in bpm, so no normalisation has to be inverted to read it. |
| `pred_gap_mse_pct` | the same ratio unrooted — `forecast`'s own `mse_skill` convention, applied source-versus-no-source rather than model-versus-baseline. |
| `pred_gap_mc_likelihood_pct` | $100(e^{\Delta/(H\cdot R)} - 1)$ — the percentage form of the headline nats: the extra probability density the source-conditioned forecast puts on each observed raw sample. |

The arithmetic is `frames.skill_against`, the same function `forecast` scores its baselines with —
moved one layer down rather than copied, because the guard is the content: the denominator is
tested **strictly positive** and fails to `NaN`, never to `inf` (which the headline's finiteness
check would refuse) and never to `0.0` (which reads as "no improvement"). The RMSE percentage is
taken as the root of $1 -$ the MSE one rather than by dividing a second time, so the two share that
one guard and cannot disagree about the sign.

The likelihood percentage has **two preconditions, and failing either omits it entirely** — no
column, no row, no headline key — with the reason recorded under
`coupling.pred_gap_percent.likelihood_space` in the package's usual `skipped`/`reason` shape.

The first is the likelihood. Exponentiating a block score means something only where the score is
a log density; under `mse` it is a sum of squared errors, which is why `marginalise_block_scores`
averages there rather than taking a `logsumexp`. So an `mse` checkpoint gets the two error-space
percentages and **not** the likelihood one — the same conditional shape `calibration` already has,
and for a related reason. An *unknown* likelihood is a skip too rather than a pass: the record has
carried that key since before this readout existed, so its absence means the tables are wrong
rather than old.

The second is the block size, read from the run's own geometry rather than assumed. It is the
**fixed** $H \cdot R$, not each anchor's scored-sample count, so the number under-reports wherever
forecast steps are masked — the same caveat every `/480` figure here carries, and it makes the
value a floor rather than an estimate.

### perm_control

Does the model use *this* recording's source, or react to any source at all? The verdict is three
losses and nothing else: $D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}}$.

The KL is deliberately not a parameter, and that is the content of the criterion rather than a
simplification of it. A stranger's source is out of distribution for a posterior trained only on
matched pairs, so it routinely moves the posterior **more** — a healthy model has
$K_{\mathrm{shuffled}} > K_{\mathrm{true}}$, and a criterion that read the KL would fail exactly
the models it should pass. `shuffled_exceeds_true` is therefore recorded as a description and
consumed by nothing, and `influential_not_specific` is a real finding about a checkpoint rather
than a pipeline failure.

The derangement is **GUID-aware**: Sattolo's algorithm guarantees only $\pi(i) \ne i$, and the
test split is eight per-subgroup shards read in order, so an unshuffled batch would pair a
recording with its own neighbouring segment. The pass runs under a seeded shuffle, a batch with
no cross-recording pairing available is excluded **and counted**, and both the
`same_recording_pairing_rate` and the excluded count reach the summary even at zero.

The shuffled branch is scored **on its own**, never differenced sample by sample against the
matched branch, because the permutation draws a fresh $\epsilon$; and only the seven keys the
control actually recomputes may be read from its shallow-copied output.

Three paired controls are scored per recording, all under the same sign convention — positive
means the control is worse than the branch it is referenced against:

| Row | Quantity | Referenced against |
|---|---|---|
| `shuffle_penalty` | $D_{\mathrm{shuffled}} - D_{\mathrm{base}}$ | no source at all |
| `prior_shuffle_penalty` | $D_{\mathrm{base}}(\text{shuffled } \mu^p) - D_{\mathrm{base}}$ | its own prior latent |
| `source_margin` | $D_{\mathrm{shuffled}} - D_{\mathrm{full}}$ | this recording's own source |

**The third one is referenced against `full`, and that is why it exists.** The two above it are
both referenced against the target-only branch, so both inherit whatever the base forecast is
doing — and a model whose latent geometry charges more for the source than the source delivers
fails every base-referenced comparison while still reading *this* recording's source rather than
any source. `source_margin` changes only the source: prior, decoder and latent geometry are
identical between its two branches. So a **positive margin beside a negative predictive gain is a
real state, not a contradiction** — no forecast improvement, and the source pathway is still
recording-specific — and it is the state the readout was added to make sayable.

All three carry the same describe, bootstrap, positive-fraction and Wilcoxon columns, in
`summary.json` under `perm_control.penalties`. They are **not** in `perm_control_summary.csv`,
which carries the branch table only. `source_margin`'s mean is additionally emitted as the keyed
scalar `perm_control.source_margin_nats` and promoted to the headline as `source_margin_nats`:
the headline is assembled by walking key paths and `penalties` is a list, which is why the two
penalties beside it have never appeared there.

### latent

How much of the latent carries source information, and whether its variance is fitted or bound.
The per-dimension KL spectrum, the active-dimension count and the top dimension's share — and the
detectors the evaluation would otherwise never read, though the model computes them and the
trainer logs them every epoch: `mean_logvar_prior`, `logvar_prior_floor_frac`,
`mean_logvar_post`.

That second half is the point. The KL carries $(\mu^q - \mu^p)^2 / \sigma_p^2$, so a **prior**
variance pinned on its lower clamp multiplies every coupling readout by an arbitrary factor while
every decoder-side diagnostic stays perfectly healthy. `prior_variance_not_pinned` is the
FAIL-able verdict that catches it, judged at the model's own margin — 5% of the clamp range,
which on the shipped $[-5, 3]$ is 0.4 nats. The bound is a sigmoid, so an exact-equality test
would read zero forever.

`prior_rate` is the same pathology as a **distance rather than a fraction**: the objective's own
$R_p = \sum_d \tfrac12(e^{\ell^p} - 1 - \ell^p)$, per recording and in nats per anchor, reduced on
the KL support like the divergence beside it and reported as the `prior_rate_nats` headline scalar.
Zero means $\sigma_p = 1$ exactly, so it is the only one of these readouts bounded below by its own
optimum, and it is continuous where the floor fraction is a step — it rises from the first epoch,
while `logvar_prior_floor_frac` stays at $0$ until mass actually reaches the margin. Read the two
together: a large `prior_rate` at a zero floor fraction is a prior drifting off unit scale with room
left, and a large one at a floor fraction near $1$ is the collapse the verdict already failed on.
The expression is recomputed here rather than imported, because the training-side function reduces a
whole batch to one scalar and this table needs it per sample; a test pins the two equal on the same
inputs.

The masked/unmasked framing here is the opposite of the sibling's: the log-variance fractions are
already masked over `elem_mask` and `kl_support`, and it is `mu_prior_sat_frac` and
`delta_mu_sat_frac` that are flat means over every element, warm-up prefix and untrained tail
included. Only those two are recomputed `_masked`, beside the model's own `_raw` values, and the
two may legitimately disagree.

### lag_kl

Where in the past the source informed the future. The per-lag KL attribution
$\widetilde K_{t,\ell} = \sum_m K^{(m)}_t \alpha^{(m)}_{t,\ell}$, whose sum over lags is exactly
$K_t$ — an identity re-measured on **this run's worst anchor** every pass and registered in the
sanity block, not inherited from a model test. A maximum rather than a mean, because the one
mechanism that breaks it (dropout on the attention probabilities) has a per-anchor error that is
zero-mean by construction.

Three profiles, not one, and they answer different questions. The **raw** attribution divides
every bin by the same anchor total and is therefore a decomposition of the headline KL. The
**support-corrected** one divides each bin by the anchors at which that lag was causally valid —
lag $\ell$ is valid only at anchors $t \ge \ell$, so over the trained range lags 0–30 receive 240
contributing anchors while lag 90 receives 180, a 25% under-weight that biases the argmax short.
The **untruncated** one is recomputed on the anchors at which every lag exists, because the
support correction fixes each bin's denominator and cannot fix its numerator: attention rows are
renormalised per anchor, so a truncated anchor pushes mass onto the short lags and no per-lag
count knows it happened.

An argmax is not by itself a reading. Peak width, mass above threshold and secondary peaks travel
beside it, and `degenerate` is defined mechanically — peak-to-median below 1.1, **or** exact-zero
fraction above 0.9 — because `entmax15`'s exact zeros can make an argmax on a flat profile
meaningless. Both profiles are also cut by clinical class and by time window on the shared grid.

### attention

The attention itself: per head, against the entropy it can actually reach. The posterior is
head-structured — latent group $m$ is written by attention head $m$ alone, which is what makes
the per-head KL an additive decomposition rather than an arbitrary slice — so averaging the four
heads before profiling discards exactly what the architecture exists to expose: four heads at
four delays and one head attending everywhere produce the same head-averaged curve.

The entropy ceiling is $\operatorname{mean}_t \log \min(t+1, L)$, **not** $\log L$: at the
shipped geometry exactly 60 of the 240 trained anchors have structurally truncated lag support.
Both entropies are emitted, distinctly named, and the ceiling is a per-sample column over the
sample's own scored anchors so their ratio is a measurement rather than an approximation. The
entropy is taken per anchor and then averaged, never as the entropy of the averaged profile — a
mixture's entropy is at least the mean of the entropies mixed, so the second reports a model
whose lag focus *shifts* as one that has none.

`kld_per_t_per_head` sums over heads to `kld_per_t` exactly, and that identity is the second
sanity-block check.

### calibration

Is the decoder's learned variance the spread of its own errors? Under `gaussian_nll` the block
score is a negative log density only if it is, and nothing else in this pipeline checks it — a
model can drive its NLL down by shrinking $\sigma$ wherever it happens to be right and paying for
it elsewhere, and every score in every other analysis would improve.

Four readings over the raw samples themselves: the PIT, central coverage at the exact erf
nominals, CRPS, and the NLL gain over the homoscedastic MLE fitted to the very residuals being
scored. The nominals are $\operatorname{erf}(k/\sqrt 2) = 0.6827,\ 0.9545,\ 0.9973$; the
two-sigma figure is **not** 0.95 — that is $\pm 1.96\sigma$ — and the half-point difference reads
as a real miscalibration.

`logvar_full_floor_frac` and `logvar_full_ceil_frac` ship beside `mean_logvar_full`, because a
single mean is equally consistent with a spread distribution and with half the mass pinned on
each clamp. This is the one analysis whose output directly changes a config value: it states the
recommended `model_config.VAE_model.logvar_clamp` revision, and says **no change** when neither
end binds — a recommendation emitted unconditionally is one that gets applied unconditionally. An
`mse` checkpoint records a skip and accumulates nothing, because the log-variance head is never
fitted there.

### residual

How far apart the two forecasts are, and how far the source moved the belief behind them. The
sibling's `residual` analysis has no direct analogue: that model has a `delta_mu_src` and a
base-plus-residual decoder, so its residual *is* a tensor. Here `mu_base` and `mu_full` are two
passes of **one shared decoder** on two latents.

Reported instead: the per-anchor forecast-difference RMS in bpm — via the **sigma** inversion,
which scales by `std` with no offset, because inverting a standard deviation affinely is a
silent, plausible-looking error — and the two latent-side quantities that are **not** the same
thing: `delta_mu_rms`, per element, and `mu_post_prior_gap_rms`, per step with the L2 over $d_z$
taken first.

RMS metrics accumulate unrooted and root once. Averaging finished per-sample RMS values is biased
low by Jensen, in the direction that flatters the model; the analysis reports the average-of-roots
beside the rooted-once value, so the bias is a measured number rather than an assumed one. One
caveat weakens in the model's favour and is stated: both branches share one log-variance head
applied to different $z$, rather than reading two separate variance heads.

### coherence

Which frequencies the forecast reproduces, and for how far ahead. Every other analysis scores it in
the time domain, and that hides a distinction with clinical content: a forecast that holds the
baseline while flattening beat-to-beat variability and one that tracks variability while drifting on
baseline score identically on mean squared error.

**The construction is what makes it possible, and it is the reason this was deferred until now.**
A single forecast block is $H \cdot R = 480$ samples; a Welch window that fits inside it puts the
whole $[0, 0.04)$ Hz deceleration span in the DC bin the detrend has already removed. Fixing a
horizon step $\tau$ and concatenating over consecutive anchors instead yields a **contiguous,
gap-free, non-overlapping** 4 Hz series: incrementing $t$ by one advances the block start by exactly
$R$, and each anchor contributes exactly $R$ consecutive samples, so slice $\tau$ *is* the raw trace
over $[R(w+1+\tau),\,R(T_{\mathrm{valid}}+1+\tau))$ — $3840$ samples, $960$ s, every sample once.
That identity is asserted against the model's own target builder rather than re-derived.

Three things follow. The window is `nperseg = 512` (128 s), so $\Delta f = 7.8125$ mHz and four bins
sit below $0.03$ Hz where the deferred design had one. **Lead time becomes an axis rather than a
trade**: slice $\tau$ holds lead times $[4\tau + 0.25,\ 4\tau + 4]$ s and the thirty of them tile
$0$–$120$ s, each at full frequency resolution — there is no STFT-inside-a-block compromise
anywhere. And because `nperseg` and the hop are integer multiples of $R$, a window spans a whole
number of anchors, so its validity is an exact `all()` over the forecast mask.

**A window touching a gap is dropped whole; nothing is interpolated.** `events` fills gaps before
smoothing because a peak finder needs a continuous trace; a spectral estimate cannot take that
trade, because the interpolant is a deterministic ramp whose own spectrum — concentrated low, absent
high — would be attributed to the model in exactly the bands this is read for. The drop is proved
rather than promised: poisoning every invalid anchor with $\pm 10^9$ leaves every accumulator
bit-identical. The cost is reported three ways, and one consequence is that **this analysis's
population may be smaller than `forecast`'s**, because whole-window validity is stricter than
per-step validity. That is a real population difference, not a fault.

Four readouts per band and lead time, and they fail independently. **Coherence** $\gamma^2$ is how
much of the truth's variation the forecast reproduces in phase. **Spectral gain**
$g = \sqrt{S_{yy}/S_{xx}}$ is whether it has the truth's amplitude — $g < 1$ is the over-smoothing
every mean-square-trained forecaster is prone to, and it is invisible to the coherence. **Phase and
a group delay** say whether it arrives at the right moment. And the **exact three-way split of the
normalised residual spectrum**

$$\frac{S_{ee}}{S_{xx}} \;=\; \underbrace{(1-\gamma^2)}_{\text{irreducible}} \;+\; \underbrace{2g\gamma(1-\cos\phi)}_{\text{timing}} \;+\; \underbrace{(g-\gamma)^2}_{\text{amplitude}}$$

is what turns a coherence into an actionable statement: the terms vanish exactly when nothing is
unpredictable, when the phase is right, and when the amplitude equals the mean-square-optimal
$g = \gamma$. The algebraically equivalent $\gamma^2\sin^2\phi + (g-\gamma\cos\phi)^2$ is **not**
used, because it charges a purely mistimed forecast for amplitude error — the exact confusion this
analysis exists to remove. Beside them travel $\Delta\gamma^2$, the frequency-resolved `pred_gap`;
the UP–FHR coherence of the truth and of both branches; and a token-seam check.

**The bands are the fetal-HRV table**, under an `hrv_band` column that cannot collide with
`band_channel_map.csv`'s `band`: `vlf` $[0, 0.03)$, `lf` $[0.03, 0.15)$, `mf` $[0.15, 0.50)$,
`hf` $[0.50, 1.00)$, `noise` $[1.00, 2.00]$ Hz, holding $4/16/44/64/129$ bins. They are deliberately
**not** `band_partition`'s `CLINICAL_BANDS`, whose crosswalk is `slow_baseline`+`deceleration`
$\approx$ `vlf`, `variability` $\approx$ `lf`+the lower half of `mf`, `beat_to_beat` $\approx$ the
rest — two reasons, both specific here: that table's $0.25$ Hz edge is exactly the decoder's
token-seam frequency, and a band boundary is the worst place for an artifact to sit; and it has no
LF/MF split, which is the distinction a spectral statement about a forecast most needs. The bands
partition every bin from DC to Nyquist exactly once, asserted at import, because that is what makes
each band sum a term of an exact identity. `vlf` includes the DC bin on purpose: the per-window mean
is removed but the Hann taper leaves a residue, so bin $0$ carries signal rather than the level.

**Sums in, ratios out.** The collection pass stores unnormalised cross-spectral sums per segment and
this analysis is the only place a ratio is formed — so within a recording the statistics are
**summed**, a named departure from `frames.per_recording_means`, which every other analysis uses.
Averaging per-segment coherences would be wrong rather than merely different: coherence is exactly
$1.0$ on a single window for any two signals whatever, and is biased upward by $(1-\gamma^2)/n_d$ at
$n_d$ windows, of which one segment holds at most $14$. Both estimators ship side by side
(`..._segment_mean`), so the size of that bias is measured on every run. Across recordings the chain
resumes as usual: one value per recording, unweighted, bootstrapped over recordings.

**The phase is never averaged.** It is an angle; the mean of $-3.1$ and $+3.1$ radians is $0$, half
a turn from both. Where a pooled phase is reported it is the argument of the *summed* cross-spectrum,
and a delay comes from a magnitude-weighted grid search rather than a phase unwrap — which needs the
phase to advance by less than $\pi$ per bin and fails silently past that. The delay is identifiable
only modulo $1/\Delta f = 128$ s, so the search runs over $\pm 64$ s and a wider request is refused
rather than served an arbitrary tie.

Two checks reach the sanity block, and they do different jobs. `coherence_parseval` is an **exact**
identity gated at the same `IDENTITY_RTOL` as the other two: the band-summed residual spectrum
against a time-domain residual accumulated independently over the identical kept windows. It reports
`NaN` — and so lands INCONCLUSIVE — when nothing was actually compared, because a worst-case seeded
at zero would let the one exact gate on the estimator certify itself on precisely the runs where it
measured nothing.

`coherence_detrended_share` is the loose one, and it is **not** a comparison against the block
scores: whole-window dropping, the 50% overlap and the $w^2$ weighting mean no tolerance against
`sq_error_full` would be a measurement rather than a fudge. What it bounds is the detrended windowed
residual power against the raw one — the share of the forecast's error the spectrum can see at all.
The complement is the **level** error, which the per-window mean removal makes invisible here: a
share near 1 means the band numbers describe essentially all of the error, and a share near 0 means
almost all of it is a constant offset and no coherence, gain or band figure is describing the
forecast's actual failure.

### distributions

What each metric's distribution over 20-minute **segments** looks like, cohort by cohort. Every
other analysis reduces to one value per recording before reporting anything, and what that hides is
the *shape*: three cohorts with the same mean forecast error can be a uniform shift, a heavier
tail, or a handful of segments the model fails on completely, and those are three different
findings. Eight metrics, two per question the pipeline asks — the two branches' per-segment RMSE
in bpm and the block score; `mc_pred_gap` and the unfloored KL; `delta_mu_rms`, the decoder's mean
log-variance and the attention entropy.

**It is descriptive by construction.** No test, no interval, no $p$-value, and nothing registered
in the headline block. That is not an omission: a per-segment $p$-value is anticonservative by the
~30× anchor overlap, and `cross_subgroup` remains the only analysis that adjudicates a cohort
difference. A visible separation here is a reason to look.

**Both levels are drawn on the same axes, and that is the content.** The filled density is one
value per segment; the median / inter-quartile / range **strip** above it is one value per
recording. Their difference *is* the pseudo-replication — a strip far narrower than the density
beneath it says most of the visible spread is within-recording variation, and the density is
showing roughly thirty views of the same delivery. The two levels take two different forms on
purpose: a forty-bin density over a cohort's six recordings is a row of spikes that estimates
nothing and takes the panel's y-limit with it.

Four presentation choices are load-bearing. **Density rather than counts**, because the healthy
cohort contributes an order of magnitude more segments than HIE and a count axis would report the
cohort sizes rather than the metric; each curve's $n$ travels in the legend at both levels instead.
**One bin grid per panel**, computed from the pooled values across the cohorts drawn in it — two
histograms on two grids are not a comparison, and the difference between them can be the binning.
And the **subgroup figure is nested rather than flat**: one column per clinical class, that class's
subgroups overlaid inside it, so a cell holds at most four curves and they are four tints of one
hue.

The fourth is the **overlap encoding**, and it is the one the whole figure rests on, because the
cohorts are drawn on top of one another and where they differ is the subject. Each cohort is a
faint fill under a hairline outline at full opacity: a solid fill hides what is behind it and a
heavy one blends with its neighbours into a colour no legend explains, while a line survives a
stack of three. The translucency is carried by the *face colour* rather than by the artist's
`alpha`, which would fade the border along with the fill; and the fills and outlines are drawn in
two passes, so every outline sits above every fill. That last part is what keeps the figure honest
about draw order: one pass per cohort would leave the first cohort's outline veiled by the fill of
each cohort after it, and the first entry in the legend would be the hardest curve to trace — an
artifact of the drawing reading as a property of the data. The outline is the cohort's own colour
rather than a darkened one, so the amber of `acidosis` stays amber and the severity hue can still
be read without the legend.

The error metrics are **rooted per segment and converted to bpm**, which is legitimate here and
would not be elsewhere: `residual`'s rule is that *averaging* finished roots is Jensen-biased low,
and the object drawn here is the distribution rather than its mean. The recording-level series
still roots after the per-recording mean, as the rest of the pipeline does, so the two levels
differ by that bias as well as by the aggregation — `per_segment_root_note` says so in the record.

It declares **no** `grouped_frames`. The runner's fan-out draws violins documented as holding one
value per recording; handing it this per-segment frame would produce a per-segment violin that
reads as a per-recording one, which is the exact confusion this analysis exists to make visible.

### trajectory

The two coupling readouts against time — inside a segment, and across a whole delivery. The
per-anchor table's first general consumer.

Within a segment the shape is **structural before it is physiological**: the warm-up prefix
carries no loss term, the lag support is truncated until $t \ge L - 1$, and the last $H$ anchors
are never scored, so a profile that rises or falls at either end is the geometry rather than the
model. Across a delivery the segments are assembled on the absolute time axis
$t_{\mathrm{abs}} = \mathrm{epoch} + 4t$, with overlapping timesteps **averaged** rather than
drawn twice and `n_contributing` travelling so the averaging is visible rather than inferred. A
gap produces a **break** in the data — `gap_before_s` — rather than an interpolation, so an
analysis reading the table sees it too and not only the figure.

### time_to_delivery

Does the coupling change as delivery approaches, and differently by class? Both readouts binned
on a 0.5 h grid of `epoch / 3600`, class-stratified, on per-GUID values — per-GUID *inside* a
window as well as across the split, so a recording contributing eleven segments to a window
cannot outvote one contributing two.

The sibling tracks the KL alone; `pred_gap` is tracked beside it because the two fail
differently: `pred_gap` is in the decoder's own units and is immune to the prior-variance
inflation. Significance is tested **per window**, with Holm across windows as one family and
pairwise tests on the survivors; the `pooled` row is flagged `confounded_by_time` and consumed by
nothing.

`TRAJECTORY_BIN_HOURS` is a module constant, not an `eval_config` key, for the reason the
significance level is not one — and it is defined one layer down so the lag structure is cut on
the same grid.

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

The analysis the raw target exists for: the forecast scored as a *waveform*, against the two
events a clinician reads a trace for.

**Deceleration skill.** The detector runs on the true raw FHR and on each branch's forecast, both
in bpm, and rates are computed **per event** under a de-duplication rule that is exact rather
than approximate: fixing the horizon step. For a given $\tau$ exactly one anchor places a given
absolute raw sample there, so a per-$\tau$ rate counts each physiological event once by
construction — a de-duplication pass applied afterwards would need a clustering tolerance, and
that tolerance would be the answer. The usable interior is 240 of the 480 samples, because the
ported detector drops any event within 30 s of either block end, and the measured
pseudo-replication factor is 14 rather than the horizon's own 30.

**Contraction-triggered response**, against a count-matched per-recording random-trigger null
passed through the **identical** min-over-window operator. The statistic is a minimum over a
window and is negative on any data at all, so the null is what measures that selection bias
rather than assuming it away.

**Conditioned coupling**: both readouts restricted to anchors within `event_lag_window_s` of a
detected contraction, against count-matched control anchors drawn from the same recordings. The
contraction timing is computed in the collection pass and lands on the per-anchor table as
`seconds_since_contraction` — it has to be, since the model reads the source as scattering and
phase channels and a contraction exists nowhere in the tables unless the one pass holding the raw
UP trace puts it there — so this runs over every anchor of the split rather than only over
retained samples. Guards: at least 200 event anchors over at least 4 recordings, else a recorded
skip.

Gaps are masked by `weight`, never by value — 0 bpm is roughly $-11\sigma$ after z-scoring, not a
detectable sentinel. Masking is two steps and both are needed: invalid samples are interpolated
across *before* smoothing, so a gap contributes no edge for the peak finder to lock onto, and any
event whose span touches one is then **dropped**, because its shape partly came from that
interpolation.

The contraction onset is a **level crossing** of the peak's own prominence rather than a gradient
walk-back, which is a deliberate correction to the ported detector and is named as such in the
module: a gradient test stops at the apex, where the smoothed gradient is approximately zero, and
a two-stage gradient walk stops mid-flank.

### sufficiency

What the latent bottleneck costs the forecast:
$\Delta_{\mathrm{suff}} = D_{\mathrm{base}} - D_{\mathrm{oracle}}$, where $D_{\mathrm{oracle}}$
comes from an evaluation-only decoder of the same capacity reading `target_state` instead of $z$,
fitted on half the evaluation recordings and scored on the other half. Until this exists,
`pred_gap` is a gap between two models rather than an information rate.

**It is an estimate, not a bound**, and both bias directions travel in the emitted JSON rather
than only here. Conditioning on `target_state` rather than on the raw target history omits the
encoder's own information loss and biases the gap **down**; fitting the probe on the evaluation
population while $D_{\mathrm{base}}$ comes from a model trained on the disjoint, healthier
pretraining cohort biases it **up** by a domain shift the probe does not suffer. The two oppose,
neither is measured, so nothing downstream may treat the number as a bound.

The probe's convergence flag is a precondition, not a decoration: an unfinished probe understates
the gap. Convergence is arithmetic on the held-out curve — the final quarter contributed at most
a tenth of everything the score ever gained — and a curve that never improved is **not**
converged, because a probe that did not move has failed to start rather than finished. The
capacity check refits at double width and flags `capacity_bound` if that improves the held-out
score by more than the stated margin. $D_{\mathrm{oracle}}$ is measured once over the whole
held-out half at the final state, never at the curve's best point, which would be selecting the
step by the score it is about to report.

The split is at **GUID** level, disjointness is asserted at runtime rather than only tested, and
the fit budget is expressed in passes over the fit half rather than in optimizer steps — a step
count is not portable across populations, and what makes a fixed budget honest is the convergence
flag rather than a knob an operator could turn until the gap looked right.

### samples

Per-recording diagnostic pages, and the triage that picks which recordings to look at. The page
is the same seven-row diagnostic the training callback writes every validation epoch, from the
same builder rather than a second one that could disagree with it.

`stratified/` holds a seeded, shard-stratified draw over the whole split, so a cap at or above
the shard count reaches every shard. Beside it, one directory per headline metric and tail holds
the segments at the extremes of that metric. A page is one segment of one recording: an
illustration, never evidence — and the extreme pages are selected *on* the quantity they display,
so the panel showing it is guaranteed to look unusual and says nothing about how often it does.

**Ten pages per directory**, so every selection holds the same number and two of them can be read
against each other. `eval_config.caps.pages` overrides the stratified count; the extremes take ten
per tail as an **upper bound**, lowered to half the finite rows wherever a metric has too few
scored segments to fill both. The two tails of one metric are disjoint by construction: a segment
appearing under both `<metric>_low/` and `<metric>_high/` reads as simultaneously the best and the
worst case, and the segments that would double up are the ones nearest the median — extreme in
neither direction.

The `<index>` in a filename is the position in the evaluation **dataset**, not in
`per_sample.csv` — the collection pass runs under a seeded shuffle — and the two are reconciled
by a `guid`/`epoch` round trip checked before anything is rendered. One page failing is recorded
by index and does not stop the rest.

This is one of exactly two analyses that read the task and the loader off the context rather than
the tables, and the reason is structural: a page is the whole forward output of one segment, and
the extreme pages are chosen by sorting a table that did not exist while the pass ran.

### cross_subgroup

Do the cohorts actually differ, or does the by-subgroup table only look as though they do? Eight
cohorts each with a mean always produce a highest and a lowest; with eight metrics that is
sixty-four numbers, and some will look separated whether or not anything is there.

Three layers, in order, and the order is the point: a Kruskal omnibus per metric, Holm **across
metrics as one family**, and pairwise Mann–Whitney with Cliff's delta (Romano magnitudes) on the
survivors only. Every test consumes one value per **recording** — a source naming a `per_sample`
file would test segments while reading as though it tested recordings, and a test asserts none
does.

It reads finished per-recording CSVs off disk through a `METRIC_SOURCES` table, so a missing
source is **recorded** rather than raised — which is what keeps `--only cross_subgroup` working
against a finished directory with no checkpoint — and it self-skips below two testable groups.

## How the output is misread

These are the readings the numbers invite and do not support.

**The coupling readout is not causal under the shipped configuration.** Under
`causal_reach_budget_s: null` the input features at step $t$ read far into their own future; the
reach guard is a 95%-energy quantile rather than a hard support, measured at roughly 20×
suppression at 120 s rather than removal; and no finite budget is currently trainable. Every run
carries the refusal sentence verbatim in `preflight.json` and `summary.json`, and
`tests/test_eval_naming.py` scans the whole artifact tree — plus this file and the figure guide —
for the name the readout refuses.

**Specificity is read in prediction space, not in KL space.** See `perm_control`:
$K_{\mathrm{shuffled}} > K_{\mathrm{true}}$ is what a healthy model does, so a KL-space criterion
would fail exactly the models it should pass.

**Only the unfloored KL may be read as a rate.** `source_conditioned_kl_train` has free bits
applied per dimension per step before summing, so it exceeds the raw value by construction and
hides a collapsed source pathway. The shipped `free_bits: 0.0` makes the two coincide today,
which is exactly why the distinction lives in code: no headline path may resolve to it, asserted
by test.

**Only an unpinned prior variance makes that rate meaningful.** A prior variance on its clamp
inflates every coupling number while every decoder-side diagnostic stays healthy. Read
`prior_variance_not_pinned` before quoting the KL.

**The lag is the compensated lag** $\tau = 4(\ell + \delta)$, with $\delta$ read from
`model.source_delay_steps` and nowhere else. There is no "original-sensor" twin of this figure: the
stored UP/FHR timeline is canonical, the dataset builder's UP shift is part of the signal, and no
reported lag adds it back or subtracts it. The historical misreadings came from a different place — two consumers once probed
model internals under different names and disagreed by up to 30 steps, two minutes, with nothing
raising. `tests/test_lag_consistency.py` now pins all four consumers by identity as well as by
value. `source_delay_is_max_over_channels` travels beside every reported lag, because per-channel
delays have no single representative and the maximum makes every figure an upper bound.

**A per-segment histogram is a description, not a cohort difference.** `distributions` is the one
analysis that draws the segment-level population directly, and it exists because the shape is
invisible after the per-recording reduction. It computes no test on purpose — segments overlap by
the factor below — so a separation visible there is a reason to look at `cross_subgroup`, which
answers the question on per-recording values. Its own figures draw the per-recording distribution
over the per-segment one so the gap between the two is on the page rather than in this paragraph.

**Anchors are not independent, and every statistic is per recording.** Consecutive anchors'
forecast windows overlap in 29 of their 30 horizon steps and one GUID contributes up to ~37
segments, so per-segment $p$-values are anticonservative by that factor. The chain is: per anchor
→ support-weighted mean within a segment → unweighted mean over a GUID's segments → across GUIDs.
A segment scoring zero anchors is excluded and **counted**, never averaged in as `0.0` — the
per-sample mean divides by a denominator clamped to 1, so an empty numerator reads as exactly
zero, and averaging that in would pull a summed-480-sample block score of hundreds of nats toward
zero and shrink `pred_gap` with no other symptom.

**Every class contrast is out-of-distribution, and the scope is wider than it looks.** The
checkpoint trains on `pre_training_dataset/*`, which is healthy-**with-background** only. So
ACIDOSIS and HIE are unseen — and so are the `healthy_no_bg_cs` and `healthy_no_bg_no_cs`
subgroups. The summary computes `training_cohort_disjoint` from both resolved dataset lists
rather than asserting it (case-normalised absolute paths, since a train list written with forward
slashes and a test list with backslashes name the same file), reports `null` rather than `false`
where a list is absent, and suppresses the out-of-distribution sentence when the two overlap.

**An eval score is not comparable with a `test_*` metric logged during training.** The
populations differ, and nothing in either number says so. The summary carries that sentence.

**A cohort's colour here is not its colour on a training figure.** This package owns its clinical
palette — green, amber, red by severity — while the training callback draws from the shared
`utils.style` mapping, in which healthy is blue. Two figures of the same cohort from the two
sources are reconciled by legend, not by hue. See *Cohort order and colour* above, which also
states why the cohort ordering is clinical rather than alphabetical everywhere.

**Coherence is not skill, and the gain is not a fault.** A forecast that reproduces every wiggle at
half amplitude scores $\gamma^2 = 1$ while carrying a quarter of the truth's variance as error, so a
coherence quoted without the gain beside it says less than it appears to. The converse trap is the
same number read the other way: the mean-square-optimal amplitude given a coherence is
$g = \gamma$, not $g = 1$, so a well-trained forecaster is *supposed* to shrink where it is
uncertain. `coherence`'s `gain` measures distance from the truth's variance and its `amplitude` term
measures distance from that optimum; a model can be near-zero on one and far from the other, and
neither reading alone is the finding.

**A band-integrated coherence is not a mean of per-bin coherences.** The emitted band number is
$\big|\sum_{\mathcal B} S_{xy}\big|^2 / (\sum S_{xx} \sum S_{yy})$ — the squared correlation of the
band-passed signals, and the only form under which the residual decomposition and the Parseval
identity hold simultaneously at band level. The mean of the per-bin ratios is a different quantity,
systematically larger, and reconciles with nothing; the per-bin curve is emitted in
`coherence_spectrum.csv`, where it is what it says it is.

**Nothing below 7.8 mHz is resolved, and nothing here is time-resolved *within* a recording.** A
$0.003$ Hz VLF floor is unreachable at any window length a 20-minute segment supports, so `vlf` is
four bins and its lower edge is decorative. And every estimate pools windows, so a coherence that
*varies* through a recording reads as a lower constant one — this is a stationary estimate, not a
spectrogram.

**Source coherence is a timing and association statement, not a directed one.** $\gamma^2(u,\cdot)$
is measured against the **contemporaneous** uterine pressure — the pressure during the window being
forecast, which the model never read, since it conditions on the source only up to each anchor. At
long lead times that makes it evidence of anticipation; at short ones the two are hard to separate.
The causality refusal above applies to it exactly as to every other readout here, and a
`preservation` above $1$ means the forecast is *more* stereotyped than the record rather than better
than it.

**The seam ratio is a property of the decoder, not of the fetus.** The truth's own ratio is the
control and must be read first: a value above $1$ on a branch means nothing until the truth's is
subtracted, because the heart rate has its own content at $0.25$ Hz. A genuine excess lands inside
the `mf` band and contaminates every `mf` number in the analysis, which is why the check ships
beside them rather than as a footnote — and why it is a description rather than a verdict, since the
remedy is a model change.

**`coherence`'s population may be smaller than `forecast`'s.** Whole-window validity is stricter
than per-step validity, so the coverage block can legitimately report a disagreement between them.
That is the gap rule working, not a fault.

**The sufficiency gap is an estimate, not a bound.** Both bias directions, above.

**Deceleration rates are per event, not per anchor.** Above.

**Lag ablation is absent, and so is necessity.** Band-restricted sufficiency ranking is blocked
model-side: in `LagCrossAttention` a band mask *replaces* rather than intersects the
causal-validity mask, and `lag_band_mask` is not a forward parameter. Unblocking it is a
model-layer change outside this pipeline. Necessity is not measured anywhere either — the
sibling's band ranking measures **sufficiency** and reads exactly backwards if taken for a
removal ablation, so this pipeline emits neither rather than emitting the one that is routinely
misread.

**A `nll_*_sample` key is a fixed /480 rescale of a block score**, not a mean over unmasked
samples, so on any anchor with masked forecast steps it under-reports. It ships beside that
statement or not at all.

**The percentage is never `pred_gap` divided by a block score.** That ratio is the one a reader
reconstructs from the headline — `pred_gap_mc_nats / d_base_mc_nats`, both of which are registered
— and it is not a percentage of anything. $D_{\mathrm{base}}$ is a negative log *density* summed
over 480 raw samples: it has no natural zero, it is legitimately negative for a sharp forecast, and
the ratio therefore changes sign with its own denominator and is unbounded near it. This is the
same rule `forecast` states for `advantage_nats_per_anchor`, which is a **difference** rather than
$1 -$ a ratio for exactly this reason. The percentages this pipeline does emit live in the two
spaces that have a natural zero — error space (`pred_gap_rmse_pct`, `pred_gap_mse_pct`) and
likelihood space (`pred_gap_mc_likelihood_pct`) — and the likelihood one is the correct
proportional reading of the nats headline: a log-score difference exponentiates to a likelihood
*ratio*, it does not divide into one.

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

## Operations

### Exit codes

The exit code is non-zero **if and only if a step raised.** Three things deliberately do *not*
move it:

- a failed **sanity** check — the self-consistency block warns, logs at ERROR and leaves the code
  at 0, because a run whose every step succeeded can still be one nobody should quote a number
  from;
- a **coverage** warning (two analyses reporting different populations);
- an **inert-cap** warning (a cap no analysis read).

That asymmetry is exactly why the offline acceptance gate exists separately: `verify` reads the
sanity block and refuses on it. A pipeline that conflated the two would either pass runs it
should not, or fail runs whose only problem was a skipped analysis.

### What a refused run leaves behind

Preflight runs **outside** the fail-soft step wrapper and raises `EvalPreconditionUnmet` — a
distinct type, so a reader seeing it in a traceback is looking at a refusal with an actionable
message rather than at a crash inside an analysis. A refused run leaves `resolved_config.yaml`
and `eval.log` carrying the refusal, and **no `summary.json`**: a rejected input must not produce
a file that reads like a result, and a refusal an operator cannot read afterwards is half a
refusal.

### Re-running one analysis

```bash
python -m teb_vae.lag_attn_rws.eval.run --output-dir <a finished run> --only lag_kl
```

reads the tables, builds no model and touches no GPU. It is **non-destructive but not additive**.
The prior `summary.json` and `steps.json` are renamed aside to `summary.bak.<stamp>.json` before
the new pass writes — byte identical, with the backup path logged — but the new summary describes
**only what this pass ran**: its headline is mostly `null`, its manifest lists a handful of files
instead of forty, and its exit code is 0. So **read the backup, not the new summary, for anything
the re-run did not itself produce.**

`preflight.json` is deliberately *not* preserved: a pass with no checkpoint cannot regenerate the
causality disclosure and reads that file back instead, so renaming it would take it from the one
pass that needs it. It is stable within a directory anyway — a different checkpoint's tables are
refused before it is ever read.

The tables carry a provenance sidecar naming the checkpoint hash, the seed, the row count and the
`eval_config` digest. A mismatch raises `TablesProvenanceMismatch` — again a distinct type,
because the tables are intact and readable and simply belong to another run; re-collecting on top
would leave two runs' rows under one summary.

### Guard recovery table

One row per way preflight refuses a run. Each refusal's own message names the fix; this is the
index, and `tests/test_eval_docs.py` asserts every raise site in `preflight.py` has a row.

| Refusal begins | Cause | Recovery |
|---|---|---|
| `dataset_config still carries the` | A shard or statistics path is still a `REPOINT_ME` placeholder. Checked **first**, so the message names the real cause rather than a missing file someone would then go looking for. | Point `vae_test_datasets` at the shared k-fold holdout split and `stat_path` at statistics regenerated from the same dataset at `trim_minutes: 1.0`. |
| `dataset_config.vae_test_datasets is empty` | No evaluation shard is configured. | Set the eight holdout subgroup shards in the override delta. |
| `dataset_config.vae_test_datasets names shard(s) that do not exist` | A configured shard is absent. When the containing directory is absent too, the message adds the two dataset build modes. | Build the dataset in `holdout` mode: the default `augmented` mode writes per-fold test splits and no shared `test/` directory, and a per-fold split is not a substitute — one pool, no fold loop, no double counting. |
| `dataset_config.dataloader_config.dataset_kwargs.trim_minutes must be` | Not `1.0`. The whole raw-index geometry — the forecast of anchor $t$ starting at raw sample $16(t+1)$ — assumes the trimmed grid; untrimmed it starts at $16(t+16)$, one full minute later, and nothing fails loudly. | Set `trim_minutes: 1.0`, and confirm the statistics file was computed on the same grid (a mismatch there only warns). |
| `dataset_config.dataloader_config.dataset_kwargs.load_fields is missing` | One of the five clinical fields is absent. The loader **skips** a field a shard does not carry, silently, so this presents downstream as "no classes found" or "no trajectory data" rather than as a data problem. | Merge the committed override delta, which adds `target`, `epoch`, `cs_label`, `bg_label` and `time_from_labor_onset`. |
| `the config disagrees with the checkpoint it is evaluating` | A geometry or objective key in the config contradicts the checkpoint's own `model_kwargs` / `hyper_parameters`. The checkpoint always wins, so the config's values would be reported beside numbers they did not produce. | Evaluate the checkpoint against its own `resolved_config.yaml`, which the training run writes beside it. ($\beta$ and its ramp are recorded and deliberately not compared: they weight the training total and enter no evaluated readout.) |
| `every witness tensor in this model is still exactly at the value the constructor gave it` | No checkpoint weights reached the model. `load_checkpoint_strict` returns `None` rather than raising, so an unchecked load would report randomly initialised weights as a measurement. | Check the checkpoint path and that its state dict aligns. This is a weight-space check, not a behavioural one: a genuinely trained model whose *source pathway* collapsed still has nonzero weights here and passes, because that finding must be reported rather than refused. |
| reused guard: `stat_path` | The normalisation statistics file is missing or unreadable. | The trainer's own message names the command that regenerates it. |
| reused guard: `raw_target_normalized` | `'fhr'` is not in `normalize_fields`, so the raw target arrives at ~140 bpm while the decoder's learned log-variance models a $z$-scale. | Add `'fhr'` to `normalize_fields`. |
| reused guard: `causal_budget_resolves` | The configured reach budget does not resolve against the shipped filter bank. | The trainer's message names the surviving channel counts per block. |
| reused guard: `declared_widths` | The **model's** `c_y` / `c_u` disagree with the test shard's stored widths. Compared against the model rather than the config (the evaluation rebuilds from the checkpoint), and against `vae_test_datasets[0]` rather than the training list — the reused guard reads the training key and returns silently when it is absent, which on an eval run is both the wrong population and a silent no-op. | Do **not** "fix" this by reverting `c_y` / `c_u`; the shard and the checkpoint were built from different channel selections. |

### Dependencies

No new ones. `torch`, `numpy`, `pandas`, `matplotlib`, `h5py`, `pyyaml` and `loguru` are already
in use; `scipy` is imported lazily at each call site; `pyarrow` is pinned in `requirements.txt`
and carries the per-anchor table, so there is one format and no fallback branch. `verify.py`
needs none of them except `pyyaml`, and that only for the arm tables — the acceptance gate itself
is a stdlib parse.

### The gate

From the repository root:

```bash
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_rws/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_rws/tests -q -m slow
```
