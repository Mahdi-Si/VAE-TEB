# Causal-feature conv-Transformer forecaster — results

Status: criteria registered for the revised architecture, runs not yet made.
Last updated: 2026-08-27

Every table below is a form with its columns named and its cells empty. It is written **before** the
headline run rather than after it, so the criteria a run is judged against cannot be chosen once the
numbers are in view. Fill the cells; do not add a criterion, and do not soften one.

**This form is re-issued for the revised architecture and describes no trained run.** Six mechanisms
arrived after the previous form's runs were read — the lag attention's local keys and values, the
prior's clock, the decoder's target-only persistence residual, the horizon-weighted reconstruction,
the flat lag-bias seed and the source stream's own alignment clock — and each is a key whose
off-state reproduces the previous architecture bitwise. `lag_attn_cfs/DESIGN.md` §17 is the
inventory, shared by both cfs cells; both ship the same six values, which is what keeps the encoder
edge below readable. The one section carrying measurements is *The identifiability record*, and those
are measurements on a synthetic fixture whose answer is known, not on any production run.

---

## What this study measures, and the rules for reading it

This is the sixth cell of the encoder-by-target grid, and the one that closes it:

```
                          conv-LSTM encoders          conv-Transformer encoders
  raw FHR target          lag_attn_rws                lag_attn_transformer_rws
  two-sided feature       lag_attn_fs                 lag_attn_transformer_fs
  causal feature          lag_attn_cfs                lag_attn_transformer_cfs   <- this
```

It sits at the corner of two edges and is read along both. Against `lag_attn_cfs` the configs differ
in the encoder block alone — seven keys added, five removed, plus `lr_warmup_steps` and a re-derived
gradient clip — so a difference in results is attributable to the **encoder**. Against
`lag_attn_transformer_fs` they differ in the target domain alone, so a difference there is
attributable to the **transform**. Both edges are pinned leaf-for-leaf by
`tests/test_config_load.py`, outside a declared allow-list.

**Five rules govern every number on this page.**

1. **Every number on this page is in-sample and carries no uncertainty.** It is a scalar read off a
   run's own `train_results/metrics_history.csv`: no confidence interval, no held-out population,
   and no held-out claim available from a config whose two splits are one shard. A difference
   between two runs of this model is evidence about those two runs. There **is** now an evaluation
   package — `teb_vae/lag_attn_transformer_cfs/eval`, four files that supply a binding, an override
   delta, a runner and a gate, and delegate every readout to `teb_vae/lag_attn_cfs/eval` — and it is
   where a claim with an uncertainty on it comes from. That delegation is load-bearing for the
   encoder edge below: two architectures are only comparable if they are measured by one
   implementation. *The evaluation's second reading of these criteria* below maps each criterion
   here onto the verdict that re-asks it there.
2. **The nats are comparable only within this family, at this budget, under this objective.** The
   reconstruction is summed over `H * C_keep = 30 * 98 = 2940` coefficients.
   `lag_attn_transformer_fs` sums `30 * 78 = 2340` and the raw cells sum `H * R` raw samples, so no
   cross-target comparison of a loss level is meaningful; and because `C_keep` is what the warm-up
   budget decides, two runs of *this* model at two budgets are not comparable to each other either.
   The one model whose loss level is comparable to this one's is `lag_attn_cfs` — and it stays
   comparable only because both cells ship the same horizon half-life and the same persistence
   state, since the objective now carries **two** weights, per channel and per horizon step. The
   evaluation applies neither, so a training-path number is not the evaluation's number for the same
   quantity.
3. **A negative `pred_gap` is not a failure of anything.** `lag_attn_rws/PRIOR_SCALE_AND_PREDICTIVE_GAP.md`
   section 5 establishes that the held-out predictive gain is negative because the source pathway
   does not generalise — a failure in the source encoder, the lag attention and the posterior
   fusion. Replacing the encoders changes which modules those are but not the finding, and this cell
   is expected to reproduce it. Its sign is a criterion nowhere on this page.
4. **A lag-resolved figure is an attribution over stored-coefficient time, now with a known constant
   offset.** One-sidedness and zero latency are different properties and only the first is bought
   here: beyond its warm-up a causal channel still lags by its composed group delay, 13.3 to 791.0 s
   depending on the channel. The alignment re-indexes every kept target channel onto 402.1604 s and
   every kept source channel onto 288.2672 s as `causal_delay_s` reports them, so what remains
   between the streams is one constant of −113.8932 s reported, −99.65 s realised — which is what
   puts a 20–60 s physiological delay at lags 8.5–47.5 of a [0, 90] window instead of below its near
   edge. What is still not divided out is the target term, because this cell's target is itself a
   stored coefficient on its own clock. The forecast claim survives that untouched, but no attention
   peak on this page may be read as a physiological delay.
5. **The pooled lag argmax is predicted at the near edge on every arm and is not the comparison.**
   The window's lowest attainable lag is a *censoring* edge exactly as its highest is, and the
   evaluation's sanity check now reports a pin there as censoring rather than as inertness. What the
   arms are compared on is named in the causal parent's record under *The comparison protocol*, and
   it applies unchanged here: the shape vocabulary, the per-head profiles and entropies, the
   occlusion analysis's per-horizon-step deltas per band, `coupling_minus_clock_nats` against its
   0.15-nat margin, and the warm/novelty gap splits beside the pooled `pred_gap`. The
   pre-registered directions are carried over unchanged: an alignment arm should not improve
   `pred_gap` and should improve profile concentration if a real lag exists.

**One architectural claim this cell can make that the conv-LSTM causal cell cannot.** Step-wise
causality of the *history states* holds unconditionally: there is no time-pooling normaliser to
causalise, and `causal_norm` is not a constructor keyword of this model at all, so no configuration
of it exists in which the claim fails.

---

## Pre-registered acceptance criteria

Two tiers, and the distinction is the point. **Tier 1** asks whether the machinery did what it was
built to do; a failure there voids the run. **Tier 2** is the science, and this is the first
causal-feature model at this architecture — there is no prior against which a threshold on it could
have been calibrated, and the decoded anchor count per step fell by roughly 15x against the two-sided
cells, which changes the optimisation regime.
Tier 2 numbers are **reported and interpreted, not passed or failed**;
a fixed threshold on any of them would be a guess dressed as a gate.

### Tier 1 — must hold, or the run is void

| # | Criterion | Where it is read | Result | Value |
|---|---|---|---|---|
| 1 | `target_warm_frac` is exactly `1.0` on every logged row of both stages | `metrics_history.csv` | | |
| 2 | `anchors_per_sample` is in `[10, 11]` on training rows and exactly `51` on validation rows | `metrics_history.csv` | | |
| 3 | The loss is finite on every step and the spike breaker never latches | `train/total_loss`, `train/spike_skipped` | | |
| 4 | `pred_gap_warm_lo + _mid + _hi`, `pred_gap_novel_lo + _mid + _hi` and `pred_gap_st + pred_gap_ph` agree with each other on every logged row, to `1e-6` relative | `metrics_history.csv` | | |
| 5 | Two evaluations of the final checkpoint produce an identical metric row set | two run directories | | |
| 6 | The resolved configuration states every architecture switch and both alignment references, and the run's identity keys name the arm | `resolved_config.yaml` | | |

Criterion 1 is a **stamped provenance column**, not a runtime measurement: it is resolved at
construction and the constructor already refuses a violating budget-and-floor pairing, so a value
other than `1.0` means the checkpoint was built by code that predates that refusal.

Criterion 4 compares the three splits **against each other** rather than against
`pred_gap`, deliberately. `pred_gap` is `nll_base_block - nll_full_block`, a difference of two
order-`1e3` sums over 2940 coefficients, so it loses several decimal digits to cancellation *before*
any split is formed. All three are computed under **both** objective weights, because each is a
partial sum of the `pred_gap` printed beside it.

Criterion 6 is new, and it is the one criterion here that is about provenance rather than geometry.
The driver builds a run's model kwargs by sweeping the constructor's signature and **silently drops**
any key the class does not re-list, so an arm can train as the baseline with nothing in the metric
history saying so — which is how a finished evaluation of this cell once came to be read as the
aligned arm when the run had trained unaligned. Presence rather than value is what is asserted: each
of the seven keys has a comparison arm on the other side of it, so no single value is the right one.
The parent's `check_run.py` scores it off a run directory of either cell.

### Tier 2 — reported and interpreted

| # | Quantity | What it separates | Value | Reading |
|---|---|---|---|---|
| 6 | `source_conditioned_kl_raw` | the coupling readout: its trajectory, its final value, and whether it is still rising at the end | | |
| 7 | `kld_active_frac`, `logvar_prior_floor_frac` | whether the latent collapsed and whether the prior scale pinned on its clamp floor | | |
| 8 | `kld_source_null` beside `source_conditioned_kl_raw` | whether the coupling readout is measuring source *content* or the availability *clock* | | |
| 9 | `shuffle_penalty`, `source_lag_warmth_frac_st`, `source_lag_warmth_frac_ph`, and the spread across the two tertile families | whether a stranger's source is worse than this one's, how much attention mass lands on cold source lags, and whether slow or history-heavy channels forecast differently from fast or novel ones | | |
| 10 | The epoch at which `val/total_loss` is minimised, beside the epoch at which `val/nll_full_block` is | whether the composite optimum and the best conditioned forecast are the same epoch, which on the previously diagnosed run they were not (336 against 278) | | |

**Criterion 8 is the single most important number on this page, and its pre-registration has a
caveat that must be read with it.** The source availability pattern is a deterministic function of
the step, identical in every row of the batch, and it enters `q(z | Y, U)` and not `p(z | Y)` — so it
can push the posterior off the prior and inflate the coupling readout with no source information in
it at all. The permutation control deranges rows, and no permutation of rows can remove something
every row shares. The design's intent was that `kld_source_null` collapse to ~0 once the prior is
given the same clock; **it will not**, because the posterior is a bounded residual on the prior, so
the mean half of the divergence at a silent source is a function of the delta head alone and the
prior's input cannot appear in it. The criterion is therefore registered as **reported**: the
informative quantity is `source_conditioned_kl_raw - kld_source_null` and its evaluation twin
`coupling_minus_clock_nats` against the 0.15-nat margin, and a collapse to ~0 would mean the
posterior parameterisation had changed. Recording that here, before the run, is what keeps a large
value from being read afterwards as a defect it is not.

A `source_lag_warmth_frac` near zero is **not** a failure. It sizes the compromise the design makes
on the source: lag attention searches back into a region where much of the source is still inside its
own warm-up, and the design keeps every source channel the warm-up budget would have taken. The
shipped source clock narrows that residual: `up_st` reaches half-warmth at step 59 and `up_ph` at
step 86, against 84 and 117 under one shared clock, and the mean per-channel warm fraction over the
whole (anchor, lag) grid is 0.970 shipped against 0.863 and 0.885.

### The two edges, read after both cells have run

Neither is a Tier 1 criterion — a difference along either edge is a *finding*, not a gate — but both
are the reason this cell exists, so both are recorded here.

| Edge | Compared against | Quantity | This cell | The other | Reading |
|---|---|---|---|---|---|
| encoder | `lag_attn_cfs` | `source_conditioned_kl_raw` | | | |
| encoder | `lag_attn_cfs` | `pred_gap` | | | comparable: same block, same anchor count |
| encoder | `lag_attn_cfs` | `kld_source_null` | | | |
| encoder | `lag_attn_cfs` | the per-head lag profiles and the shape vocabulary | | | comparable, and it is the surface the lag comparison is expressible on |
| encoder | `lag_attn_cfs` | the occlusion delta per band, by horizon step | | | comparable: one implementation, one band set, one anchor-drawing rule |
| transform | `lag_attn_transformer_fs` | `source_conditioned_kl_raw` | | | the coupling readout on inputs that do and do not contain their own future |
| transform | `lag_attn_transformer_fs` | `pred_gap` | | | **not** comparable as a level; read the sign and the trajectory only |

**The encoder edge is where the K/V localisation gets its honest test, and that is a fact about the
two stems rather than about the two encoders.** Each cell's local stem reuses its own parent
encoder's convolution schedule, so this cell's reaches 21 steps against a 91-lag window while the
conv-LSTM cell's reaches 387 — longer than the window and longer than the sequence. A lag profile
read off this cell's `conv_stem` arm is therefore a reading about a *local* K/V; the same row on the
other cell is not, and the parameter tables in both records say so at the point where the difference
is priced.

The encoder edge has a second, better source once both cells have been evaluated:
`python -m teb_vae.lag_attn_transformer_cfs.eval.verify --runs <dir> --out RESULTS_arms.md` renders
a cross-cell table putting `SeqVaeLagAttnCfs` beside `SeqVaeLagAttnTrfCfs` on the headline block,
per recording and with intervals. The transform edge deliberately has no such table: the blocks
differ, so it stays a signs-and-orderings reading and the level columns are not rendered for it.

### The evaluation's second reading of these criteria

Every criterion above is now asked a second time, on a held-out population, per recording, with a
bootstrap interval — by the ten verdicts of `teb_vae/lag_attn_cfs/eval`, reached through this
package's binding and gated by `eval/verify.py`. Neither reading supersedes the other: the criteria
above come off a run's own CSV while it is still going, and the verdicts need a finished checkpoint
and a completed evaluation pass over the causal holdout split. The mapping is the causal parent's
and is identical here, because the verdicts are: see `teb_vae/lag_attn_cfs/RESULTS.md`, *The
evaluation's second reading of these criteria*, which maps criteria 1–9 onto
`anchor_geometry_intact`, `warmup`'s recomposition guard, `source_specificity`,
`source_margin_positive`, `latent_not_collapsed`, `prior_variance_not_pinned`,
`decoder_variance_not_pinned` and `coupling_exceeds_availability_clock`, names the criteria
with no verdict and the three verdicts with no criterion, and records why each.

**`clock_margin_min_nats` is now set, at 0.15 nats**, so criterion 8's verdict
(`coupling_exceeds_availability_clock`) decides rather than returning INCONCLUSIVE and the gate is
ten criteria rather than nine. The value comes from the diagnosed unaligned run's observed spread of
$\Delta_{\mathrm{clock}}$ across recordings — 0.160, interval [0.157, 0.164] — and its provenance is
stated beside it in both cells' override files. The key belongs to the causal parent's override delta
and is set there once, for both cells: a margin set independently per cell would gate the two
architectures this table exists to compare against two different bars. Its provenance is the unaligned
arm; the gated quantity is right on both arms.

**The interventional readout has no verdict on purpose.** The occlusion analysis reports, per lag
band, how many nats the forecast loses when the source's *values* are removed there with the
availability announcement held fixed. What a healthy value is has never been measured, so four
scalars reach every arm table instead of a threshold: the winning band's name, its delta, its peak
horizon step, and its live fraction — the last because a band lying inside the warm-up scores near
zero for a reason that is about the geometry rather than about the source.

---

## The identifiability record

The one section on this page carrying measurements, and they are not from a production run. They are
from the committed planted-delay fixture, whose FHR modulation is a deterministic function of the UP
envelope a planted `delta = 45` stored steps (180 s) earlier, pushed through the real causal bank —
so the informative lags are `[15, 44]` of a `[0, 90]` window and nowhere else. The instrument was
validated by direct cross-correlation on the written coefficients before any model saw it: 12 coupled
channels peak within 4 steps of 45, the strongest at lag 45 with `r = +1.000`, against 22 flat control
channels. `lag_attn_cfs/RESULTS.md` carries the full table for both cells; what belongs here is this
cell's column and the one thing it establishes that the other cannot.

| | this cell | `lag_attn_cfs` |
|---|---|---|
| K/V receptive field under `conv_stem` | **21 steps** | 387 steps |
| Pooled support-corrected argmax, every arm | 0 | 0 |
| Band mass, previous architecture -> revised default at the flat seed | 0.192 -> 0.234 | 0.208 -> 0.383 |
| Peak width at the flat seed / the decaying seed | 18 bins / 3 bins | 38 bins / 3 bins |
| Per-head argmax, revised default at the flat seed | 0, 66, 20, 0 | 33, 0, 90, 0 |
| Per-head argmax under `lag_kv_source: adapter` | 0, 0, 0, 31 (share 0.486) | 0, 0, 0, 41 (share 0.538) |
| `kld_source_null` as a fraction of the raw KL, previous architecture | 91.9% | 81.6% |
| KL at initialisation, every arm | exactly 0 | exactly 0 |

**What this cell establishes that the other cannot: the local K/V arm has had one honest test at this
geometry, and it did not move the pooled statistic.** At 21 steps against a 91-lag window the stem is
genuinely local, so the data-processing argument the localisation was built against really is
relieved here — and the pooled argmax stays at the censoring edge anyway, while band mass falls
against the same-tree `encoder` run (0.113 against 0.154). That is why rule 5 above is written as it
is, and why the comparison protocol reads the per-head profiles and the occlusion deltas instead.

**What both cells establish jointly**: the lag-bias initialisation was doing a large share of the
pinning (at the decaying seed every head on both parents peaks at 0 and the pooled peak is 3 bins
wide), and the sharpest K/V arm is the only one on which a head reads the plant on either parent. And
the interventional readout does find the plant where the observational one does not: on the conv-LSTM
cell's own fixture checkpoint, removing the source's values at lags [15, 44] costs the forecast
**+14.94 nats** while every other band costs nothing distinguishable from zero.

---

## The eleven readouts this target domain adds

Ten are emitted on both stages and one on the evaluation stages alone. All eleven are the conv-LSTM
causal cell's own code, reached by import, which is what makes the encoder edge readable at all.

| Metric | Stages | What it separates |
|---|---|---|
| `target_warm_frac` | train, val | the budget-and-floor pairing; a constant, resolved at construction, exactly `1.0` |
| `anchors_per_sample` | train, val | the tiling actually firing; `[10, 11]` in train, `51` in val |
| `source_lag_warmth_frac_st` | train, val | attention mass on lags where the first stored source block is warm |
| `source_lag_warmth_frac_ph` | train, val | the same for the second, which is the block with the problem |
| `pred_gap_warm_lo` | train, val | the forecast gap over the slowest third of the kept target channels |
| `pred_gap_warm_mid` | train, val | the middle third |
| `pred_gap_warm_hi` | train, val | the fastest third |
| `pred_gap_novel_lo` | train, val | the forecast gap over the third of the kept target channels the anchor has most nearly already seen |
| `pred_gap_novel_mid` | train, val | the middle third |
| `pred_gap_novel_hi` | train, val | the third that is most genuinely new at the horizon |
| `kld_source_null` | val | the KL floor the availability clock induces with no source content |

The novelty tertiles are the causal cell's own split, inherited here with the mixin rather than
restated: they partition the same 98 kept channels by the envelope-mass share after the anchor,
looked up in the shard's horizon-free `causal_novelty_curve` at the run's horizon and per-channel
advance (legacy shards: `causal_novelty_frac`), so a good
score over the low third is the model inverting its own delayed history and over the high third it
is a forecast. Not the warm-up split renamed -- the slowest kept channel is warm across the whole
window and still only `0.026` new.

---

## Launch lines

```bash
# Dev box: the local smoke, one device, the committed causal fixture
python -m teb_vae.lag_attn_transformer_cfs.trainer \
    --config teb_vae/lag_attn_transformer_cfs/configs/tiny.yaml

# Dev box: the local validation, one device, a causal HIE shard (see the config's header --
# the shard does not exist yet and cannot be substituted with a two-sided one)
python -m teb_vae.lag_attn_transformer_cfs.trainer \
    --config teb_vae/lag_attn_transformer_cfs/configs/smoke_hie.yaml

# Production box: the baseline, seven ranks. The rank count must equal len(cuda_devices).
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_transformer_cfs.trainer \
    --config teb_vae/lag_attn_transformer_cfs/configs/default.yaml

# Dev box: can this architecture recover a delay it is known to be looking at? This is the cell
# whose local K/V stem is genuinely local, so it is the cell that argument gets tested on.
python teb_vae/lag_attn_cfs/lag_recovery_check.py \
    --config teb_vae/lag_attn_transformer_cfs/configs/planted.yaml

# Production box: the named arms. One axis each; each writes its own run name.
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_transformer_cfs.trainer \
    --config teb_vae/lag_attn_transformer_cfs/configs/sweep_anchor_stride_1.yaml
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_transformer_cfs.trainer \
    --config teb_vae/lag_attn_transformer_cfs/configs/sweep_lag_bias_decay.yaml
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_transformer_cfs.trainer \
    --config teb_vae/lag_attn_transformer_cfs/configs/sweep_align_target_max.yaml
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_transformer_cfs.trainer \
    --config teb_vae/lag_attn_transformer_cfs/configs/sweep_legacy_dualref_physclock.yaml
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_transformer_cfs.trainer \
    --config teb_vae/lag_attn_transformer_cfs/configs/sweep_lag_kv_adapter.yaml
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_transformer_cfs.trainer \
    --config teb_vae/lag_attn_transformer_cfs/configs/sweep_source_dropout_02.yaml
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_transformer_cfs.trainer \
    --config teb_vae/lag_attn_transformer_cfs/configs/sweep_source_dropout_03.yaml

# While a run is in flight: score it against the tier-1 criteria above, off its own CSV and its
# own resolved configuration. The causal parent's checker reads a run directory of either cell.
python -m teb_vae.lag_attn_cfs.check_run --run-dir <run>

# After it finishes: the held-out evaluation, one reviewable directory per checkpoint.
python -m teb_vae.lag_attn_transformer_cfs.eval.run \
    --checkpoint <run>/model_checkpoints/<name>.ckpt

# The offline gate, and the arm and cross-cell tables. Neither imports torch.
python -m teb_vae.lag_attn_transformer_cfs.eval.verify <run>/eval_results/summary.json
python -m teb_vae.lag_attn_transformer_cfs.eval.verify --runs <dir-of-runs> --out RESULTS_arms.md
```

The training lines and the identifiability check run today. The evaluation lines need one
precondition this repository does not yet contain: `eval/configs/eval_overrides.yaml` points
`vae_test_datasets` and `stat_path` at `REPOINT_ME` placeholders, and preflight refuses the run by
name until they are edited to a causal holdout build and statistics regenerated from those same
shards at `trim_minutes: 1.0`.

**Each production run writes two checkpoints under distinct stems**, one selected on
`val/total_loss` and one on `val/nll_full_block`, and early stopping is on at patience 50 validation
epochs. Which of the two a given evaluation reads is a decision the operator records, because the
two are different epochs and neither the arm tables nor the encoder edge may mix them.

**The arm inventory is closed**: the six above plus the default. Four are cfs-family axes and exist
in both cfs cells, because every reading in this record is per parent; the two source-dropout arms
exist here alone, which is the cell where that seam — the one that regularises the source map
without touching the target pathway — is named. Each file is the default plus its one named leaf plus
the two identity keys, and the unaligned arm's second leaf is forced rather than a second delta,
because the resolver refuses a source reference against an unaligned target by name.
`planted.yaml` is not an arm: it is the identifiability check's own geometry, with its alignment
pinned so a default flip cannot move the instrument.

---

## Where the numbers come from

| Source | What it provides | Caveat |
|---|---|---|
| `train_results/metrics_history.csv` | every scalar on this page | validation columns are the epoch mean over validation steps; training columns are the epoch mean over training steps |
| `train/grad_norm` | the pre-clip gradient norm | sampled one optimizer step per epoch, so read it as a distribution over epochs rather than per row |
| `lr` | the step-granular ramp | logged at train-epoch start, so its first cell is always empty |
| the per-epoch diagnostic page | the warm-up staircase, the anchor overlay, the forecast rows | drawn at the dense anchor set and at phase 0, which is not the geometry a training step used |
| the run-level warm-up budget figure | the channels the budget dropped beside the ones it kept | a constant of the shard, not of the run |

---

## Before launching: what reverts, and when to stop

### What reverts, and how

This package and its conv-LSTM sibling arrived together and edited files outside both. The revert is
a list of **files**, not a count of commits, so that undoing it is a checkout rather than an
archaeology exercise.

| File | What was added | Inert without these packages? |
|---|---|---|
| `teb_vae/lag_attn_rws/nets/raw_masks.py` | an optional anchor set on the three mask functions | yes — the default reproduces the dense range bitwise |
| `teb_vae/lag_attn_rws/nets/losses.py` | an optional anchor set on `compute_loss` | yes |
| `teb_vae/lag_attn_fs/nets/feature_target.py` | an optional anchor set on `_build_forecast_target` | yes |
| `teb_vae/lag_attn_rws/nets/controls.py` | the anchor argument on the permutation control, and the source-null arm | yes |
| `teb_vae/lag_attn_rws/task.py` | the `anchors=` keyword at the shared call site, and the `_added_metrics` hook | yes — the hook returns `{}` |
| `teb_vae/lag_attn/nets/encoders.py` | the availability adapter masks its own warm-up region | yes — a gated model's positions there are already exactly zero |
| `teb_vae/lag_attn_rws/plotting.py` | the diagnostic callback resolves three page seams off the task | yes — an absent seam resolves to the shipped builder |
| `teb_vae/lag_attn_rws/sample_page.py` | the forecast-row seam | yes |
| `teb_vae/lag_attn_rws/input_budget.py` | the input-panel seam | yes |
| `hdf5_dataset/hdf5_dataset.py` | `read_causal_warmup`, a public numpy-only boundary reader | yes — nothing else calls it |
| `scripts/make_tiny_shard.py` | the causal variant of the committed fixture | yes |
| `teb_vae/lag_attn/tests/fixtures/tiny_shard_causal.hdf5` | the committed causal fixture | yes |
| `teb_vae/lag_attn/tests/fixtures/tiny_stats_causal.hdf5` | its statistics | yes |
| `teb_vae/lag_attn_cfs/nets/causal_inputs.py` | the encoder-agnostic half both cells compose | it is this package's dependency, so reverting this package alone leaves it |
| seven copies of `tests/test_nets_are_framework_free.py` | this package in each `_PACKAGES` tuple | yes |

Every "inert" claim in the last column is a tested one: `scripts/print_objective_metrics.py` prints
every objective metric of every shipped forecaster in about a minute, and its output is unchanged by
each of the edits above.

Reverting **this** package alone is deleting `teb_vae/lag_attn_transformer_cfs/` and removing its
name from the seven `_PACKAGES` tuples. Nothing outside it imports it.

### Go/no-go while a run is in flight

| Signal | Threshold | Action |
|---|---|---|
| `train/total_loss` non-finite | any row | stop; the breaker's non-finite guard should have caught it first |
| `train/spike_skipped` sustained above zero | | stop; the margin is mis-tuned for this objective |
| `target_warm_frac` other than `1.0` | any row | stop; the geometry broke, and every nat on the page is over the wrong block |
| `anchors_per_sample` outside its band | any row | stop; the tiling is not the one the configuration states |
| `lr` flat at its base value from step 0 | | investigate; the step-granular ramp did not engage |
| `train/grad_clip_frac` near 1 | | investigate; the clip is binding on ordinary steps rather than on blow-ups |

### When a stop fires

Record the epoch, the row that fired it, and the run directory. A stopped run is a result about the
configuration, not a run to be quietly relaunched at other settings.

---

## Parameter budget — measured

| Model | Configuration | Total | Decoder width | Block |
|---|---|---:|---:|---:|
| `lag_attn_transformer_cfs` | shipped: local K/V, prior clock, persistence residual, weighted horizon, flat bias, both clocks | 4,284,556 | 98 | 2940 |
| `lag_attn_transformer_cfs` | shipped but one shared clock | 4,286,604 | 98 | 2940 |
| `lag_attn_transformer_cfs` | shipped but unaligned | 4,287,372 | 98 | 2940 |
| `lag_attn_transformer_cfs` | shipped but `lag_kv_source: encoder` | 5,072,524 | 98 | 2940 |
| `lag_attn_transformer_cfs` | shipped but `lag_kv_source: adapter` | 4,183,564 | 98 | 2940 |
| `lag_attn_transformer_cfs` | every switch off, one shared clock — bitwise the previous architecture | 5,054,992 | 98 | 2940 |
| `lag_attn_transformer_cfs` | every switch off, unaligned | 5,055,760 | 98 | 2940 |
| `lag_attn_transformer_cfs` | every switch off, ungated | 5,039,256 | 102 | 3060 |
| `lag_attn_cfs` | shipped | 4,655,987 | 98 | 2940 |
| `lag_attn_cfs` | every switch off, one shared clock | 5,146,334 | 98 | 2940 |
| `lag_attn_transformer_fs` | reach budget 120 s | 5,034,984 | 78 | 2340 |

**The five architecture switches cost −768,388 here**, against −488,299 on the conv-LSTM cell, and it
factorises exactly: the deep source encoder is not built (−888,960), the local K/V stem is (+100,992),
the prior's clock projection and its own norm are (+16,640 = `128 * 128` plus `2 * 128`), and the
persistence weight is (+2,940 = `30 * 98`). The horizon weight is a non-persistent buffer and the
flat bias seed reuses an existing parameter, so both cost zero. **The whole of the difference between
the two cells' figures is the two stems**, and that is a schedule rather than a design difference:
each reuses its own parent encoder's convolution schedule, which is `(5, 9)` at dilations `(1, 2)`
here and reaches 21 steps, against `(3, 5, 11, 15, 15)` at `(1, 2, 4, 8, 16)` there, which reaches
387.

**The second alignment clock costs −2,048**, eight source channels off two 128-wide linears
(`-8 * 128 * 2`), with the target stream contributing nothing. **The alignment itself costs −768**:
the source adapter loses four channels from two 128-wide linears (`-4 * 128 * 2`) because those
channels are slower than the reference, and both adapters gain a start-of-record vector
(`+2 * 128`). Ungated means the whole guard, the mask and both clocks together.

**The alignment-delay factor moved the shift magnitudes and moved no total.** The resolver scales the
difference $\tau_{\mathrm{ref}} - \tau_c$ by the impulse-response centroid factor
$\kappa = 1 - 1/(2\gamma) = 0.875$ before quantising it, so the target's $d_c$ spans $0$–$85$ steps
rather than $0$–$97$, $\min_c (W'_c + d_c)$ falls from $91$ to $80$ on that stream, and the worst
quantisation residual is $1.9865$ s against the $2$ s half-step bound. The keep-index is
scale-invariant under that factor — $\tau_c \le \tau_{\mathrm{ref}}$ either way — so the surviving
channel counts do not move with it, the anchor floor does not move, and every total in this table is
the one it was. `DESIGN.md` §12 carries the derivation and the measurement behind $\kappa$.

Two decompositions, and both are checkable rather than narrative, and **both are read on the
off-state row** — because `lag_attn_transformer_fs` is two-sided and never takes the new keys, so at
the shipped configuration the difference between the two models is dominated by mechanisms one of
them does not have. The **encoder-axis** delta against `lag_attn_cfs` is the two history stacks and
nothing else. The **target-axis** delta against `lag_attn_transformer_fs` is the decoder's two output
heads at `2 * 257 = 514` parameters per channel plus the input adapters' width change. A delta that
does not decompose is recorded as not decomposing rather than rounded away.

Both decompose, measured parameter by parameter on constructed models:

- **Encoder axis, −91,342 on the off-state row** (this cell against `lag_attn_cfs`, guarded against
  guarded). Identical to the same delta in the two-sided feature pair and in the raw pair at their
  own budgets, and identical again on the ungated arm — which is what a difference living entirely in
  the two history encoders must look like.
- **Encoder axis, −371,431 at the shipped configuration**, identical across the shipped, one-clock,
  unaligned and ungated rows, so it is still the two history stacks alone. It moved because *both*
  stacks moved: the target encoders differ by +331,929 as they always did, and the two local stems by
  −703,360 where the two deep source encoders differed by −423,271. Quoting one of these two numbers
  where the other belongs is the easiest mistake this table now admits, so each says which row it is
  read on.
- **Target axis, +20,008** against `lag_attn_transformer_fs` on the off-state row, in two terms: the
  decoder's output head `514 * (98 - 78) = +10,280`; and the two input adapters `+9,728`, being
  `128 * (98 - 78)` and `128 * (47 - 29)` on the input linear *and* the availability projection. The
  same two terms give the same total on the conv-LSTM edge, because every module outside the encoders
  is shared. Two further terms are exactly zero and are computed rather than deleted: the horizon
  embedding, worth `-3,840` while this cell forecast one minute against the two-sided cell's two; and
  the two start embeddings, worth `-256` until the alignment's shift made this cell build them as
  well.
- **The guard costs parameters here and saves them on the two-sided sibling**, +13,568 at the shipped
  configuration against −9,662, and both are right: the budget drops 4 target channels of 102 and the
  alignment 12 source channels of 51, so the two availability projections (`128 * 98` plus
  `128 * 39`) and the two start embeddings dominate the −2,056 off the decoder head, the −512 and
  −1,536 off the two input linears and the −120 off the persistence weight; the reach budget drops
  31 of 109 so the narrowing does. On the off-state row the same identity is the six-term one at a
  47-channel source and sums to +15,736.

`DESIGN.md` §13 carries the same table; `tests/test_docs.py` measures every total in both documents
by constructing the models rather than comparing against literals, so a change to a shared component
re-costs the tables instead of failing an unrelated assertion.

---

## The loss-scale constants — measured

The encoder edge changes neither the block (2940 coefficients) nor the anchor count (~4.6 per step),
so the two constants stated in nats of the summed block must equal the conv-LSTM causal cell's. Both
were **re-measured** rather than inherited, because a faster-fitting encoder could in principle
produce larger excursions.

| Constant | Value | Moved on the encoder edge? | Statistic it was set from | Measured |
|---|---:|---|---|---|
| `gradient_clip_val` | 14000.0 | yes | smallest round value above the pre-clip gradient norm's q99, below its maximum | |
| `additive_margin` | 9.0e+3 | no | above the worst excursion above the breaker's own EMA in the noisiest regime the fixture produces, held equal to the conv-LSTM cell's and clearing the larger of the two measured | |
| `ema_floor` | 1.0e+9 | no | above any loss the objective can reach, which switches the relative test off | |
| `horizon_embed_std` | 0.8 | no | the post-initialisation correlation between two horizon tokens | |

**The horizon weighting does not move any of the four, and that is measured rather than assumed.**
The weight is renormalised to `sum(w) = H`, so it redistributes the block's magnitude rather than
rescaling it: on a symmetric batch at the shipped geometry the weighted block sum is **4543.25**
against the uniform **4544.01**, a ratio of 0.99983. Without the renormalisation the same weight
would have shrunk the block by **1.807x** against an unmoved KL, which is exactly where
`gradient_clip_val`, `additive_margin` and $\beta$'s standing against the reconstruction would have
gone out of date with nothing saying so. At the shipped half-life of 15.0 steps the resolved weights
run 1.8063 at the first horizon step to 0.4729 at the last, a 3.82x spread that sums to 30.

**The persistence residual does not move them either, by construction**: it enters the mean head of
both decoder invocations identically, so it changes the level of `nll_full_block` and `nll_base_block`
together and leaves `pred_gap` exactly where it was.

Re-derive the clip from the headline run's own `train/grad_norm` column once it exists, and
`additive_margin` from that run's `main_loss` column; the values shipped were measured on four
in-sample windows, which is a thinner tail than a production run's.

## Distributed smoke, memory and throughput

| Quantity | Definition | Dev box | Production box |
|---|---|---|---|
| Peak memory per rank | `torch.cuda.max_memory_allocated` at the end of epoch 1 | | |
| Anchor-axis tensors | `4 x (B, A_max, H, C_keep)` plus the target | | |
| Steps per second | optimizer steps, epoch 2 onward | | |
| Wall clock per epoch | | | |

The anchor tiling reduces the five anchor-axis tensors by roughly 14x at the shipped stride: at
`B = 128` each is about 8 MB rather than 114 MB. The batch size stays at 128, matching both
comparison models, so neither edge is confounded by a different gradient-noise scale.

**Escalation order if a rank runs out of memory:** lower `batch_size`, then raise
`accumulate_grad_batches` to hold the effective batch, then — and only with the confound recorded —
consider `attention_grad_checkpoint`, which additionally makes `compile` unavailable.

---

## Headline baseline

| Metric | Train | Val | Note |
|---|---|---|---|
| `total_loss` | | | |
| `main_loss` | | | |
| `nll_full_block` | | | nats per anchor over 2940 coefficients |
| `nll_base_block` | | | |
| `pred_gap` | | | sign is not a criterion |
| `source_conditioned_kl_raw` | | | |
| `anchor_coverage_frac` | | | |

---

## Bottleneck health

A headline number can look healthy while the bottleneck is not, and each row below is a different way
for that to happen.

| Metric | Value | What a bad value means |
|---|---|---|
| `source_conditioned_kl_raw` | | at zero the source pathway carries nothing |
| `kld_active_frac` | | near zero the latent collapsed |
| `mu_post_prior_gap_rms` | | at zero the posterior never leaves the prior |
| `logvar_prior_floor_frac` | | near one the prior scale pinned on its clamp floor |
| `mu_prior_sat_frac` | | near one the tanh bound on the prior mean is binding |
| `delta_mu_sat_frac` | | near one the bound on the posterior delta is binding |
| `logvar_full_floor_frac` | | near one the decoder's observation variance pinned low |
| `logvar_full_ceil_frac` | | near one it pinned high |

---

## Forecasting or reconstructing?

A stored coefficient is an average over a window, so a share of the short-horizon target is already
determined by signal the model has legitimately observed. On this target domain that share is smaller
than on the two-sided one — a one-sided kernel's support lies entirely behind its own step — which is
the point of the whole family, and these four columns are how it is read.

| Metric | Value | Reading |
|---|---|---|
| `pred_gap_tau_first` | | the first horizon step |
| `pred_gap_tau_last` | | the last, which no observed history reaches |
| `pred_gap_st` | | the first stored block |
| `pred_gap_ph` | | the second |

---

## The warm-up and the tiling

The two geometry guards, and the two readouts that size the source compromise.

| Metric | Expected | Measured | Reading |
|---|---|---|---|
| `target_warm_frac` | exactly `1.0` | | a stamped constant; any other value voids the run |
| `anchors_per_sample` (train) | `[10, 11]` | | the tiling firing at the configured stride |
| `anchors_per_sample` (val) | `51` | | the dense evaluation set |
| `source_lag_warmth_frac_st` | — | | a small value is expected, not a failure |
| `source_lag_warmth_frac_ph` | — | | the block with the problem |
| `pred_gap_warm_lo` / `_mid` / `_hi` | sums to `pred_gap_st + pred_gap_ph` | | whether slow channels forecast differently from fast ones |
| `pred_gap_novel_lo` / `_mid` / `_hi` | sums to the same total | | how much of the block score is a forecast rather than an inversion of history |
| `kld_source_null` | — | | close to `source_conditioned_kl_raw` means the readout is a clock |

**Read the two `source_lag_warmth_frac_*` columns as new numbers, not as a continuation.**
They were built from $W'_c$ alone while the availability mask and the anchor floor both used
$W'_c + d_c$; since $d_c \ge 0$ the bias was one-sided and could only report the source as
*warmer* than it is, and on the shipped aligned configuration it pinned
`source_lag_warmth_frac_st` at exactly $1.0000$ for any attention distribution at all — a column
that could not vary. Both readouts now use $W'_c + d_c$. Any value carried over from a run made
before that fix is not comparable to one made after it.

---

## What the production runs still owe

- Every empty cell above.
- The three per-run figures, attached, with any disagreement against the metrics recorded.
- Both edges of the square, read only after both causal cells have run on the same shards.
- The two deferred decisions: whether the three tertile columns earn their place, and whether the
  beta pair carried across from the two-sided sibling holds at this block and anchor count. Both are
  judgements to be taken **with the number that drove them recorded**, not threshold gates.

---

## Amendment (2026-09): the forecast-target clock

The shipped default scored the forecast target on the **physical** clock
(`causal_target_forecast_clock: physical`, `anchor_stride: 5`) from this amendment until
2026-09-05, when the default returned to the stored clock at `anchor_stride: 13` on the corrected
representation (see the last amendment); the physical-clock configuration is now
`sweep_legacy_dualref_physclock.yaml`. Three consequences for reading
this document:

- Rows recorded before this amendment were produced on the stored clock. They are comparable to
  the stored-clock arm and to nothing else here: the physical clock changes the question, so its
  nats are not the stored clock's nats.
- The pre-registered alignment reading — alignment should **not** improve `pred_gap` — holds only
  within stored-clock arms. Between clock arms, movement in `pred_gap` is expected and is the
  comparison, not a confound.
- Under the physical clock the lag axis is re-registered to the contraction-recurrence window
  ($[374.9, 618.9]$ s at the retained source clock and `max_lag: 90`): the 20–60 s proximate band
  is structurally censored by strict futurity, so no lag-profile reading on a physical-clock run
  may be stated against it. See the `DESIGN.md` amendment for the arithmetic.


## Amendment (2026-09-05): the corrected representation is the shipped default

`default.yaml` now carries what the CFS review recommended and the task list CFS-09 built as a
baseline: the integer phase operator (`causal_phase_operator: integer_harmonic_v1`, so the phase
blocks are $44$ `fhr_ph` and $10$ `up_ph` and the declared widths are $c_y = 80$, $c_u = 46$), **no**
input-channel alignment (`causal_align_reference: null`, `causal_align_reference_source: null`:
every channel is read at its own availability time), the **stored** forecast clock (labels are the
next $H = 30$ stored coefficients, exact availability time, ceiling $T_{\mathrm{valid}} = 270$, dense
span $136$) and `anchor_stride: 13`, which tiles that span into the same $A_{\max} = 11$ tiles the
physical-clock geometry had at stride $5$. At the shipped budget the kept widths are $76/80$ target
($32$ `fhr_st` + $44$ `fhr_ph`) and $46/46$ source, so the block is $30 \times 76 = 2280$
coefficients; no nat from this default is comparable to a $2940$-coefficient row above. The scattering
coefficients themselves are unchanged by the operator.

The reasons are mathematical rather than empirical: the fractional $2^{3/2}$ phase family is
discontinuous at the principal-angle branch; the $0.875$ input-alignment convention is not an exact
content clock and withholds up to $340$ s of the fastest channels' trajectories from the encoder;
and the `physical` clock is an approximate delay compensation that costs $85$ trailing anchors. The
configuration this file shipped before is preserved verbatim as `sweep_legacy_dualref_physclock.yaml`
(legacy shards, `REPOINT_ME_causal`); `sweep_align_unaligned.yaml` and `sweep_target_clock_stored.yaml`
were deleted because the default now is them, and `tiny.yaml` reads the committed integer-operator
fixture. `sweep_align_target_max.yaml` now *adds* the single-reference alignment and
`sweep_target_clock_input.yaml` carries that alignment with it, since the `input` clock copies an
input shift. The guard bands above that read `anchors_per_sample` $\in [10, 11]$ train / $51$ val
are the legacy arm's; the promoted default's are $[10, 11]$ train / $136$ val. Shards for the default
are built with `create_new_pipeline.py` under `phase_operator=integer_harmonic_v1`
(`REPOINT_ME_causal_int`); they also carry the horizon-free `causal_novelty_curve`.

## Amendment (2026-09-05, later the same day): the horizon moves to 10 steps, the tiling to stride 5

`default.yaml` now forecasts $H = 10$ stored steps ($40$ s) over $C_{\mathrm{keep}} = 76$ channels,
so every nat a run of it reports is summed over a **$760$-coefficient block** — comparable to no
row of the parameter table above (all $2940$), to no $2280$-coefficient run of the promoted default
as it stood earlier today, and to neither two-sided cell, both of which still forecast $30$ steps;
it is comparable to the conv-LSTM causal cell, which carries the same block since the same day.
The `anchors_per_sample` guard bands become $[31, 32]$ train / $156$ val (`anchor_stride: 5`, dense span $[134, 290)$). The two loss-scale constants were **scaled, not measured**:
`gradient_clip_val` $3500.0$ and `additive_margin` $2.2 \times 10^{3}$, both from the recorded
$H = 30$ figures by the block ratio $760/2940$; the "Measured" column above does not describe them,
and both are to be re-derived from the headline run's own `train/grad_norm` and `main_loss` columns.
`horizon_weight_halflife_steps` is $5.0$ (the $H/2$ rule; weights $1.7260 \to 0.4957$).

**The encoder edge holds.** `teb_vae/lag_attn_cfs/configs/default.yaml` carries the same horizon, anchor
stride, horizon half-life and additive margin (mirrored the same day), so both causal cells optimise
the same criterion over the same $760$-coefficient block and $156$ anchors and the level columns of
the cross-cell table remain readable across them; `tests/test_config_load.py` pins the four leaves.
The target edge now differs in the horizon as well as the block. `DESIGN.md`'s amendment of the same
date carries the full derivation.
