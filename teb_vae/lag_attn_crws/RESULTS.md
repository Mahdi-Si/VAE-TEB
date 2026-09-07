# Causal-input raw-target forecaster — results

Status: criteria registered for the revised architecture, runs not yet made.
Last updated: 2026-08-27

Every table below is a form with its columns named and its cells empty. It is written **before** the
headline run rather than after it, so the criteria a run is judged against cannot be chosen once the
numbers are in view. Fill the cells; do not add a criterion, and do not soften one.

**This form is re-issued for the revised architecture and describes no trained run.** Four
mechanisms arrived — the lag attention's local keys and values, the prior's clock, the
horizon-weighted reconstruction and the flat lag-bias seed — and each is a key whose off-state
reproduces the previous architecture bitwise; `DESIGN.md` §17 is the inventory. Two mechanisms the
feature-target cells took are **declined on this row**, the decoder's persistence residual and a
second alignment reference, and `DESIGN.md` §14 records why each. The training controls are also on
for the first time: early stopping at patience 50, and a second checkpoint criterion on
`val/nll_full_block` beside the composite one.

---

## What this study measures, and the rules for reading it

This is the seventh cell of the encoder-by-target grid, and the first of the one row in which
neither side of the objective contains its own future:

```
                          conv-LSTM encoders          conv-Transformer encoders
  raw FHR target          lag_attn_rws                lag_attn_transformer_rws
  two-sided feature       lag_attn_fs                 lag_attn_transformer_fs
  causal feature          lag_attn_cfs                lag_attn_transformer_cfs
  causal in / raw out     lag_attn_crws  <- this      lag_attn_transformer_crws
```

It is read against `lag_attn_rws`, the direct control: same raw target, same objective, same
horizon family, differing only in whether the inputs contain the answer. The two configs are pinned
leaf-for-leaf by `tests/test_config_load.py` outside a declared allow-list of twenty-one exemptions, so a
difference between the two runs is attributable to the **input representation** — with the block and
the anchor count as the two stated confounds rule 2 names.

**Five rules govern every number on this page.**

1. **There is no evaluation package, so every number on this page is in-sample and carries no
   uncertainty.** It is a scalar read off a run's own `train_results/metrics_history.csv`: no
   confidence interval, no held-out population, and no held-out claim available from a config whose
   two splits are one shard. A difference between two runs of this model is evidence about those two
   runs. Stated once, here, so no row below is read as though it had a confidence interval; a
   `ModelBinding` against `teb_vae/lag_attn_rws/eval` is the follow-up that would change this, and
   `DESIGN.md` §14 records what it would cost.
2. **The nats are comparable only within this row, at this horizon, under this objective.** The
   reconstruction is summed over `H * R = 30 * 16 = 480` raw samples per anchor, which is the same
   block `lag_attn_rws` sums — but the decoded anchors per training step are about 4.5 here against
   a dense 240, and this row's objective weights the horizon axis while that cell's does not, so no
   cross-cell comparison of a loss *level* is meaningful, the direct control included; what carries
   across to `lag_attn_rws` is a sign, a trajectory, the bottleneck-health columns and the ordering
   of arms. The feature cells sum `H * C_keep` coefficients and are further away again. The one
   model whose loss level is comparable to this one's is `lag_attn_transformer_crws`, which ships
   the identical geometry and the identical half-life. `sweep_horizon_15.yaml` restores the
   raw-signal sibling's earlier `15 * 16 = 240` block; it restores neither the anchor count nor the
   uniform horizon axis.
3. **A negative `pred_gap` is not a failure of anything.** `lag_attn_rws/PRIOR_SCALE_AND_PREDICTIVE_GAP.md`
   section 5 establishes that the held-out predictive gain is negative because the source pathway
   does not generalise — a failure in the source encoder, the lag attention and the posterior
   fusion, none of which an input-representation change touches. This cell is expected to reproduce
   it. Its sign is a criterion nowhere on this page.
4. **A lag-resolved figure is an attribution over stored-coefficient time on the input side —
   unless the source stream has been aligned, and then it is a physical lead time.** The raw target
   carries no group delay, so the anchor is exact and the forecast claim needs no correction; a
   causal input coefficient beyond its warm-up still lags by its own composed group delay, 13.3 to
   791.0 s depending on the channel. Unaligned, that bias is indexed by a channel *pair*, no single
   number labels the axis, and no attention peak on this page may be read as a physiological delay.

   Under `causal_align_reference` every source channel is shifted onto one reference, the bias
   collapses to a single known constant, and the correction is one-sided — there is no target-side
   term to subtract, exactly rather than approximately. A peak at lag `l`, horizon element `h`, is
   then a lead time of `4*(l + 1 + h) + kappa*tau_ref` seconds on the canonical stored timeline (no
   dataset-shift term: the builder's UP shift is part of the signal). The arithmetic is
   `teb_vae/lag_attn/nets/lag_report.py::physical_lag_seconds`, and the reference is read from the
   run's own resolved-config dump rather than from the model, whose `source_delay_steps` is the
   largest *stored-step* shift (6 here) and is a different quantity. **This is the only row of the
   grid where the claim is available**: the causal-feature cells have a nonzero target reference, so
   the same expression there is a lag between two coefficient epochs and not between two signals.

   **Two numbers about that constant, both of which must be carried into any reading.** First, this
   row aligns onto **42.21 s** and not onto the feature cells' `target_max` (402.1604 s), because
   with a raw target the reference does not cancel out of the identity: at 402.1604 s the smallest
   expressible lead is 355.9 s and the 20-120 s contraction-to-deceleration band is off the axis at
   every lag index. At 42.21 s the reachable lead is [40.9, 516.9] s, so 79 of the 100 s band is
   inside the lag axis, at `l + h` in [0, 19] — the next index lands at 120.9 s, just outside. `DESIGN.md` section 3 carries the derivation and what it costs — the whole
   `up_ph` block leaves the source stream, 51 channels becoming 17. Second, the constant to
   substitute is the **effective** reference `kappa*tau_ref = 36.93 s` and not the 42.2066 s the run
   logs as `source_reference_delay_s`: the shift cancels each channel's realised centroid delay,
   `kappa = 0.875` of the envelope mean the shards store. Nothing in the tree applies `kappa` to the
   logged number, so substituting it verbatim overstates every lead by 5.3 s. `DESIGN.md` section 14
   registers that as a limitation.

5. **What a lag readout can express is now partly a configuration, and both keys must be read with
   it.** `lag_kv_source` decides which source representation the attention's keys *and* values come
   from: under the previous `encoder` arm the lag-0 key and value were a function of the whole causal
   prefix, so by the data-processing inequality lag 0 already contained whatever any later lag
   carried and a profile pinned there was reporting a representation rather than an absence of
   delay. The shipped `conv_stem` removes the recurrence — but on **this** parent the stem reuses the
   encoder's own convolution schedule and reaches **387 steps**, longer than the 91-lag window and
   longer than the sequence, so on this cell the arm bounds recurrence and not memory. Any statement
   about what localising the K/V does to a lag profile has to be read on `lag_attn_transformer_crws`,
   whose stem reaches 21 steps. And `alibi_slope_scale` decides what the learnable lag bias is
   *seeded* with: the shipped `0.0` seeds it flat, so a profile reports what training put there, and
   the `1.0` seed of the previous default is a monotone decay towards lag 0 which was measured on a
   planted-delay fixture to be doing a large share of the pinning by itself. Neither key changes what
   the physical-lag identity of rule 4 means; both change what a profile is capable of saying.

**What this cell can claim that no other cell can.** Its forecast claim and its lag claim are
simultaneously exact on the target side: the target is the signal itself, with no warm-up, no
group delay and no channel selection, and the inputs carry no future. That is the whole value of the
run, and it is a property of the *pairing* rather than of any number below. It is also the reason
this row alone had to move its alignment reference: the same $\tau^y \equiv 0$ that makes the lag
claim exact is what stops the reference cancelling, so reaching the coupling band at all is a
configuration decision here where it is automatic everywhere else (rule 4). It is likewise why this
row takes **no second alignment reference**: with $\tau^y \equiv 0$ the single key is already
source-only, and a second one would be a second way to say what one key says.

---

## Pre-registered acceptance criteria

Two tiers, and the distinction is the point. **Tier 1** asks whether the machinery did what it was
built to do; a failure there voids the run. **Tier 2** is the science, and this is the first
causal-input raw-target model in the tree — there is no prior against which a threshold on it could
have been calibrated, and against the direct control the block halved and the decoded anchor count
per step fell by roughly 24x, which changes the optimisation regime. Tier 2 numbers are **reported
and interpreted, not passed or failed**; a fixed threshold on any of them would be a guess dressed as
a gate.

### Tier 1 — must hold, or the run is void

| # | Criterion | Where it is read | Result | Value |
|---|---|---|---|---|
| 1 | `anchors_per_sample` is in `[4, 5]` on training rows and exactly `136` on validation rows | `metrics_history.csv` | | |
| 2 | The loss is finite on every step and the spike breaker never latches | `train/total_loss`, `train/spike_skipped` | | |
| 3 | Two evaluations of the final checkpoint produce an identical metric row set | two run directories | | |
| 4 | The resolved configuration states every architecture switch and the alignment reference, and the run's identity keys name the arm | `resolved_config.yaml` | | |

Four criteria where the causal-feature cells register six, and the two absences are deliberate.
That cell's stamped provenance column is the share of scored *target* coefficients past their own
warm-up, and its recomposition criterion checks that channel-axis splits of the gap agree; a raw
target has no warm-up and no channel axis to split, so neither quantity exists here and neither is
tracked. Criterion 1 is the one geometry **guard** this cell carries: it must sit at its
geometry-derived value, and a row outside that band means the tiling is not the one the
configuration states — not that the model learned something.

Criterion 4 is about provenance rather than geometry, and it is the one this row must read **by
hand** because it ships no run checker. The driver builds a run's model kwargs by sweeping the
constructor's signature and **silently drops** any key the class does not re-list, so an arm can
train as the baseline with nothing in the metric history saying so. Five leaves must be present in
`model_checkpoints/resolved_config.yaml` — `lag_kv_source`, `prior_availability_input`,
`horizon_weight_halflife_steps`, `alibi_slope_scale` and `causal_align_reference` — and the run's
`run_name` and `variant` tag must name the arm. Presence rather than value: each key has a
comparison arm on the other side of it, so no single value is the right one, and what is not
acceptable is a run whose own artifacts cannot say which side it was on. The two keys this row
declines, `persistence_residual` and `causal_align_reference_source`, must be **absent**: both
constructors refuse them by name, so a config carrying one fails at the key rather than training a
model no forward reaches.

### Tier 2 — reported and interpreted

| # | Quantity | What it separates | Value | Reading |
|---|---|---|---|---|
| 4 | `source_conditioned_kl_raw` | the coupling readout: its trajectory, its final value, and whether it is still rising at the end | | |
| 5 | `kld_active_frac`, `logvar_prior_floor_frac` | whether the latent collapsed and whether the prior scale pinned on its clamp floor — and whether the beta pair carried across from the raw-signal sibling holds at this block and anchor count | | |
| 6 | `kld_source_null` beside `source_conditioned_kl_raw` | whether the coupling readout is measuring source *content* or the availability *clock* | | |
| 7 | `shuffle_penalty`, `source_lag_warmth_frac_st`, `source_lag_warmth_frac_ph` | whether a stranger's source is worse than this one's. The two warmth columns are **constants at the shipped reference** and carry no information there: `_st` is 1.0 because every reachable lag is warm, `_ph` is 1.0 over zero channels because the block was dropped whole | | |
| 8 | `pred_gap`, read beside `lag_attn_rws`'s | the headline: the same quantity over the same raw target, from inputs that do not contain the answer — sign and trajectory only, since the anchor count and the horizon weighting differ | | |
| 9 | `pred_gap_tau_first` beside `pred_gap_tau_last` | whether the horizon weighting did what it was introduced to do: the near steps improve and the far steps do not degrade by more than the weight removed from them | | |
| 10 | The epoch at which `val/total_loss` is minimised, beside the epoch at which `val/nll_full_block` is | whether the composite optimum and the best conditioned forecast are the same epoch, which the two checkpoint criteria exist because they need not be | | |

**Criterion 6 is the single most important number on this page, and its pre-registration has a
caveat that must be read with it.** The source availability pattern is a deterministic function of
the step, identical in every row of the batch, and it enters `q(z | Y, U)` and not `p(z | Y)` — so it
can push the posterior off the prior and inflate the coupling readout with no source information in
it at all. The permutation control deranges rows, and no permutation of rows can remove something
every row shares. If `kld_source_null` and `source_conditioned_kl_raw` are close, the coupling
readout is measuring a clock.

**`prior_availability_input` gives the prior the same clock, and it will not drive the ratio to
zero.** The posterior is a bounded residual on the prior, so the mean half of the divergence at a
silent source is a function of the delta head alone and the prior's input cannot appear in it;
conditioning the prior reaches only the variance half. The informative quantity is therefore the
**difference** rather than the ratio, and a collapse to ~0 would mean the posterior parameterisation
had changed. On this row the mechanism has less to do than on the feature-target cells for a stated
reason: every kept source channel arrives by step 6, so there is very little arrival transient for
the encode of silence to carry past the anchor floor. Recording that here, before the run, is what
keeps a large `kld_source_null` from being read afterwards as a defect it is not.

**Read the two warmth columns against the arm, not against a threshold.** In the causal-feature
cells a `source_lag_warmth_frac` near zero is not a failure: it sizes the compromise those cells
make, keeping every source channel rather than gating them and measuring the residual instead of
resolving it. This cell's 42.21 s reference resolves it instead, by amputation. The kept source
channels are honest by step 6, the deepest step any lag reaches from the anchor floor is 44, and so
`source_lag_warmth_frac_st` is exactly 1.0 — a true statement about the geometry rather than a
measurement of the attention. `source_lag_warmth_frac_ph` is 1.0 over **zero channels**, because the
whole `up_ph` block sits above the reference and was dropped: an empty block reads as warm at every
step by deliberate design, since a zero there would look like a measurement rather than an absence.
Both columns regain information on the unaligned arm, which is the only place the compromise they
were built for still exists — and it is where the input-warmth policy still matters, since a floor
that cleared the *unaligned* source would sit at 277 and cost about 143 of the 136 anchors.

### The input-representation edge, read against `lag_attn_rws`

Not a Tier 1 criterion — a difference along the edge is a *finding*, not a gate — but it is the
reason this cell exists, so it is recorded here. Every row is a sign-and-trajectory reading, never a
level: the block is 480 raw samples against 480 and the decoded anchor count is about 10.1 against
240, and both are stated confounds.

| Quantity | This cell | `lag_attn_rws` | Reading |
|---|---|---|---|
| `source_conditioned_kl_raw` | | | the coupling readout on inputs that do and do not contain their own future |
| `pred_gap` | | | **not** comparable as a level; read the sign and the trajectory only |
| `kld_source_null` | | | the raw-signal cell has no such column; the availability clock is a property of the causal inputs |
| `kld_active_frac`, `logvar_prior_floor_frac` | | | comparable: both are fractions |

`sweep_horizon_15.yaml` is the arm that restores the raw-signal sibling's block and horizon
together; run it before reading a level across this edge, and note that even then the anchor count
is not restored.

---

## The four readouts this cell adds

Three are emitted on both stages and one on the evaluation stages alone. All three per-stage
columns are partial sums or fractions of quantities the objective already computes, so they add no
second definition of anything; `kld_source_null` is the exception, and it costs one source encode per
validation step. **Eight columns the causal-feature cell carries are absent here on purpose.** Four
are the feature target's own gap splits — `pred_gap_tau_first` and `pred_gap_tau_last` by horizon
step, `pred_gap_st` and `pred_gap_ph` by stored target block — which the raw-signal family never
tracked; four are the causal-feature cell's — `pred_gap_warm_lo`, `pred_gap_warm_mid` and
`pred_gap_warm_hi` partition kept target channels by warm-up, and `target_warm_frac` is the
target-side warm fraction. A raw target has no blocks, no kept channels and no warm-up, so each would
be either undefined or a vacuous constant that reads as a measurement.

| Metric | Stages | What it separates |
|---|---|---|
| `anchors_per_sample` | train, val | the tiling actually firing; `[4, 5]` in train, `136` in val — a guard, not a result |
| `source_lag_warmth_frac_st` | train, val | attention mass on lags where the first stored source block is warm; 1.0 at the shipped reference |
| `source_lag_warmth_frac_ph` | train, val | the same for the second block, which the shipped reference drops whole — 1.0 over zero channels |
| `kld_source_null` | val | the KL floor the availability clock induces with no source content |

`LagAttnCrwsTrainer.TRACKED_METRICS` is 77 entries: the raw-signal driver's 70, the three above on
both stages, and `kld_source_null` on validation alone.

---

## Launch lines

```bash
# Dev box: the local smoke, one device, the committed causal fixture
python -m teb_vae.lag_attn_crws.trainer --config teb_vae/lag_attn_crws/configs/tiny.yaml

# Dev box: the instrumented run the two loss-scale constants were measured on -- shipped widths,
# gaussian_nll, the committed fixture, the clip parked at 1e9
python -m teb_vae.lag_attn_crws.trainer --config teb_vae/lag_attn_crws/configs/smoke_causal.yaml

# Production box: the baseline, seven ranks. The rank count must equal len(cuda_devices).
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_crws.trainer --config teb_vae/lag_attn_crws/configs/default.yaml

# The two arms: dense training anchors, and the raw-signal sibling's horizon and block.
python -m teb_vae.lag_attn_crws.trainer --config teb_vae/lag_attn_crws/configs/sweep_anchor_stride_1.yaml
python -m teb_vae.lag_attn_crws.trainer --config teb_vae/lag_attn_crws/configs/sweep_horizon_15.yaml
```

The two dev-box lines run today. The production and arm lines need one precondition this
repository does not yet contain: `default.yaml` points `vae_train_datasets`, `vae_test_datasets`
and `stat_path` at `REPOINT_ME` placeholders, deliberately non-existent rather than pointing at
two-sided data, so a run fails on a missing file instead of on a `transform` refusal someone might
"fix" by dropping the budget. There is **no evaluation line and no run-checker line**: this cell
ships neither, and every number below comes off the training CSV by hand.

**Each production run writes two checkpoints under distinct stems**, one selected on
`val/total_loss` and one on `val/nll_full_block`, and early stopping is on at patience 50 validation
epochs. Which of the two any later reading uses is a decision recorded beside the number, because the
two are different epochs. Neither arm above moves an architecture switch: the two switches with a
named comparison arm on this row — `lag_kv_source` and `alibi_slope_scale` — are contrasted by an
explicit one-key override rather than by a shipped file, because no run of this row is in scope yet
and a config with nothing to launch it is a file that goes stale unread.

---

## Where the numbers come from

| Source | What it provides | Caveat |
|---|---|---|
| `train_results/metrics_history.csv` | every scalar on this page | validation columns are the epoch mean over validation steps; training columns are the epoch mean over training steps |
| `train/grad_norm` | the pre-clip gradient norm | sampled one optimizer step per epoch, so read it as a distribution over epochs rather than per row |
| the per-epoch diagnostic page | the raw forecast tiled off `anchor_index`, the anchor overlay, the two input rows with their warm-up staircases | drawn at the dense anchor set and at phase 0, which is not the geometry a training step used |
| the run-level warm-up budget figure (`causal_warmup_budget`) | the channels the budget dropped beside the ones it kept | a constant of the shard, not of the run |

**No run checker.** The causal-feature cell ships `check_run.py` because its headline runs are
blocked on the production box and its criteria must be scored by code; no production run is in scope
here, so a checker would have nothing to read. `anchors_per_sample` is therefore read by hand from
the CSV against its band until one exists — `tests/test_train_smoke.py` asserts it on the fixture
fit, which is the one run this package has made.

---

## Before launching: what reverts, and when to stop

### What reverts, and how

This package's arrival edited **seven** existing files, all of them one kind of file, and nothing
else. Every shared-tree seam the anchor tiling needs — the optional anchor set on the three mask
functions and on the shared objective, the anchor argument on the permutation control, the
source-null arm, the availability adapter's own mask, the page seams and the causal fixture — was
already in the tree when this package arrived, having been landed and proven inert by the
causal-feature cell; `teb_vae/lag_attn_cfs/RESULTS.md` carries that revert record and it is not
restated here. The revert is a list of **files**, not a count of commits, so that undoing it is a
checkout rather than an archaeology exercise.

| File | What was added | Inert without this package? |
|---|---|---|
| `teb_vae/lag_attn_rws/tests/test_nets_are_framework_free.py` | `lag_attn_crws` and `lag_attn_transformer_crws` in `_PACKAGES` | yes — the tuple builds forbidden dotted-prefix strings and imports nothing |
| `teb_vae/lag_attn_transformer_rws/tests/test_nets_are_framework_free.py` | the same two strings | yes |
| `teb_vae/lag_attn_transformer_e2e/tests/test_nets_are_framework_free.py` | the same two strings | yes |
| `teb_vae/lag_attn_fs/tests/test_nets_are_framework_free.py` | the same two strings | yes |
| `teb_vae/lag_attn_transformer_fs/tests/test_nets_are_framework_free.py` | the same two strings | yes |
| `teb_vae/lag_attn_cfs/tests/test_nets_are_framework_free.py` | the same two strings | yes |
| `teb_vae/lag_attn_transformer_cfs/tests/test_nets_are_framework_free.py` | the same two strings | yes |

Every "inert" claim in the last column is a tested one: each file's own cross-product assertion walks
ten package names rather than eight, so both new names are asserted present rather than merely added,
and the seven files pass with unchanged counts before and after the registration. No shipped source
module moved: `scripts/print_objective_metrics.py` prints every objective metric of the four shipped
forecasters in about a minute, and its output is byte-identical to the fingerprint recorded before
this package existed. Every other member this package needed from a sibling is **bound by reference**
rather than edited into place — `DESIGN.md` §6 is the record — so `git status --porcelain` over
`teb_vae/lag_attn_rws`, `teb_vae/lag_attn_cfs` and `teb_vae/lag_attn_fs` shows the three files above
that live there and nothing else of this package's making.

### Go/no-go while a run is in flight

| Signal | Threshold | Action |
|---|---|---|
| `train/total_loss` non-finite | any row | stop; the breaker's non-finite guard should have caught it first |
| `train/spike_skipped` sustained above zero | | stop; the margin is mis-tuned for this objective |
| `anchors_per_sample` outside its band | any row | stop; the tiling is not the one the configuration states |
| `train/grad_clip_frac` near 1 | | investigate; the clip is binding on ordinary steps rather than on blow-ups |
| `logvar_prior_floor_frac` climbing past 0.5 | | investigate; the prior scale is collapsing onto its clamp floor and the beta pair did not transfer |

### When a stop fires

Record the epoch, the row that fired it, and the run directory. A stopped run is a result about the
configuration, not a run to be quietly relaunched at other settings.

---

## Parameter budget — measured

**Two rows per configuration, and both are the record.** The **shipped** rows carry the four
architecture switches at their revised defaults; the **off-state** rows carry every one at its inert
value, which is bitwise the pair that shipped before this revision — and it is the row on which the
input-representation comparison against `lag_attn_rws` is still readable, because that cell never
takes the new keys.

| Model | Configuration | Total | Decoder width | Block |
|---|---|---:|---:|---:|
| `lag_attn_crws` | **shipped**: local K/V, prior clock, weighted horizon, flat bias; budget 134, aligned to 42.2066 s | 4,589,907 | 16 | 480 |
| `lag_attn_crws` | shipped but unaligned | 4,613,715 | 16 | 480 |
| `lag_attn_crws` | shipped but ungated | 4,595,155 | 16 | 480 |
| `lag_attn_crws` | shipped but `lag_kv_source: encoder` | 5,097,786 | 16 | 480 |
| `lag_attn_crws` | every switch off, aligned — bitwise the previous architecture | 5,081,146 | 16 | 480 |
| `lag_attn_crws` | every switch off, unaligned | 5,104,954 | 16 | 480 |
| `lag_attn_crws` | every switch off, ungated | 5,086,394 | 16 | 480 |
| `lag_attn_transformer_crws` | shipped | 4,218,476 | 16 | 480 |
| `lag_attn_transformer_crws` | every switch off, aligned | 4,989,804 | 16 | 480 |
| `lag_attn_transformer_crws` | every switch off, unaligned | 5,013,612 | 16 | 480 |
| `lag_attn_rws` | reach budget 120 s | 5,094,458 | 16 | 480 |
| `lag_attn_rws` | ungated | 5,088,186 | 16 | 480 |

**The four architecture switches cost −491,239 here** and −771,328 on the conv-Transformer cell, and
each term is a module rather than a rounding: the deep source encoder is not built (−1,312,231 here,
−888,960 there), the local K/V stem is (+804,352 here, +100,992 there), and the prior's clock
projection and its own norm are (+16,640 = `128 * 128` plus `2 * 128`). The horizon weight is a
non-persistent buffer and the flat bias seed reuses an existing parameter, so both cost zero. There
is no persistence term on this row, because this row does not take that key. **The whole of the
difference between the two cells' figures is the two stems**, and that is a schedule rather than a
design choice: each reuses its own parent encoder's convolution schedule, which reaches 387 steps
here and 21 there.

**Every number below is unchanged by those four switches**, which is the check that they and the
alignment are independent: the alignment narrows adapters and the switches replace a source module,
and neither touches the other's tensors.

The unaligned rows are the named comparison arm, one key away at `causal_align_reference: null`, and
are bitwise the models that shipped before the common clock existed. **At this row's reference the
unaligned arm is the *larger* model**, which is the opposite of the feature cells' ordering: the
alignment costs **−23,808** here against **−768** there, and the whole of the difference is the
reference. Aligning onto 42.2066 s rather than 402.1604 s drops 60 target-stream channels and 34
source ones instead of four source ones, so the source adapter loses `34 * 128 * 2` and the target
adapter `60 * 128 * 2` across each one's input linear and availability projection, against which both
adapters gain a start-of-record vector (`+2 * 128`) because the shifted warm-up no longer starts at
zero on either stream. Ungated means the whole guard, the warm-up mask and the shift together — a
shift vector is positional over the survivors, so a stream with no keep-index has no width for one to
be positional against.

Decompose the input-representation delta against `lag_attn_rws` rather than reporting it whole, and
read it on the **off-state** row: that cell never takes the new keys, so at the shipped configuration
the difference between the two is dominated by mechanisms one of them does not have. There it
decomposes, measured parameter name by parameter name, into **−13,312** in exactly **one surviving
term**, and the decoder head is deliberately not one of them:

| Term | Value | What it is |
|---|---:|---|
| the horizon embedding | 0 | `nn.Parameter(torch.zeros(horizon, decoder_hidden))`; both cells forecast 30 steps |
| the two input adapters | −13,312 | `128 * (38 - 78)` and `128 * (17 - 29)` on the input linear *and* the availability projection of each stream |
| the two start embeddings | 0 | the reach guard builds both, and under the alignment so does this cell |
| the decoder's output head | 0 | `raw_per_step` in both cells |

**That axis changed sign with the reference**: it was +9,728 while this cell carried 98 target and 47
source channels against the reach guard's 78 and 29. A delta that does not decompose into exactly
that one term is recorded as not decomposing rather than rounded away. It is the same −13,312 in the
conv-Transformer pair, because every module outside the encoders is shared; and the encoder edge
against `lag_attn_transformer_crws` is **91,342** on the off-state row, the same value every other
pair of the grid shows.

**At the shipped configuration that edge is 371,431**, identical across the shipped, unaligned and
ungated rows, so it is still the two history stacks alone. It moved because *both* stacks moved: the
target encoders differ by −331,929 as they always did, and the two local stems by +703,360 where the
two deep source encoders differed by +423,271. Quoting one of these two numbers where the other
belongs is the easiest mistake this table now admits, so each says which row it is read on.

**The guard costs −5,248 here against +6,272 on the raw-signal sibling**, and both are right. The
sign is decided by how much the guard narrows. Here the budget and the alignment together drop 64 of
102 target-stream channels and 34 of 51 source ones, so the −8,192 and −4,352 off the two input
linears outweigh the two availability projections coming into existence (`128 * 38` plus `128 * 17`)
and the two start embeddings (`2 * 128`), and the guarded model is *smaller* than the ungated one. On
`lag_attn_rws` the reach budget drops only 31 of 109 and 29 of 58, so the projections dominate
instead. It was +17,792 at the feature cells' reference, which is the same expression at 98 and 47.
Every parameter the guard moves here is under an adapter — nothing in this target domain widens a
head.

`DESIGN.md` §13 carries the same table; `tests/test_docs.py` measures every total in both documents
by constructing the models rather than comparing against literals, so a change to a shared component
re-costs the tables instead of failing an unrelated assertion. `tests/conftest.py` resolves this
row's own $42.21$ s reference, so the totals above are the ones that suite measures and pins.

## The loss-scale constants — measured

Two of the four were re-derived at this objective's own scale rather than carried across, because
both are stated in nats of the summed block and this cell changes both the block (240 against 480)
and the anchor count (~4.57 against ~240) — in opposite directions, so neither transfers by
arithmetic and neither was scaled. The instrumented run is `configs/smoke_causal.yaml`: 600
optimizer steps at the shipped widths over the committed causal fixture, the clip parked at 1e9.
Neither regime below skipped a batch or bound the clip.

**The horizon weighting does not move any of the four, and that is measured rather than assumed.**
The weight is renormalised to `sum(w) = H`, so it redistributes the block's magnitude rather than
rescaling it: on a symmetric batch the weighted block sum is **4543.25** against the uniform
**4544.01**, a ratio of 0.99983. Without the renormalisation the same weight would have shrunk the
block by **1.807x** against an unmoved KL, which is exactly where both re-derived constants and
$\beta$'s standing against the reconstruction would have gone out of date with nothing saying so. At
the shipped half-life of 15.0 steps the resolved weights run 1.8063 at the first horizon step to
0.4729 at the last, a 3.82x spread that sums to 30.

| Constant | Value | Was | Statistic it was set from | Measured |
|---|---:|---:|---|---|
| `gradient_clip_val` | 12000.0 | 5000.0 | smallest round value above the pre-clip gradient norm's q99, whole-shard batch | q50 3347, q90 6952, q95 8499, **q99 11806**, max 19370 |
| `additive_margin` | 2.5e+3 | 1.0e+3 | 1.46x the worst excursion of `main_loss` above the breaker's own EMA at its `ema_decay`, after the 100-batch priming window, in the noisiest regime the fixture can produce (batch 1) — the *lower* bracket, because at this block it has crossed the reconstruction-only upper one | batch 1: q50 −207, q90 157, q95 276, q99 488, **max 1713**; whole shard: q99 1.0, max 10.4 |
| `ema_floor` | 1.0e+9 | 1.0e+9 | above any loss the objective can reach, which switches the relative test off | — |
| `horizon_embed_std` | 0.8 | 0.8 | the post-initialisation correlation between two horizon tokens | 0.445915 at H = 15 against 0.447476 at H = 30 |

The last two did **not** move from the raw-signal sibling's values, and that is a measurement rather
than an inheritance: `ema_floor` is a switch rather than a scale, and the horizon-token correlation is
a per-pair quantity that does not depend on how many tokens there are.

Two things about the margin are worth reading twice. Its bracket is (248, 759): strictly above the
worst measured excursion, so ordinary batches are not skipped, and strictly below the ~7.6e+2
magnitude the objective can reach — `2 * 240 * 1.58` from the per-sample NLL's lower bound at the
shipped clamp floor — so the additive test can still fire rather than being decoration, which is the
check the sibling's 1.0e+3 would *fail* at this block size. And the batch-1 excursion needed its own
harness: `metrics_history.csv` is per epoch, and at four optimizer steps per epoch the per-step
statistic the breaker actually compares is not recoverable from it. The batch-1 gradient norms (q50
6481, q99 33580, max 54448) are recorded and deliberately not used; that regime is 128x smaller than the
shipped batch. `main_loss` stayed inside [+148, +648] with the whole shard in one batch and [+34,
+863] at batch 1, so the run never reached the negative-loss regime the sign-agnostic block exists for.

**Every number in this section is provisional**: four in-sample windows are a thinner tail than a
production run's, so the distribution describes a memorised window rather than the production
objective. Re-derive both from the headline run's own `train/grad_norm` column once it exists.

**No distributed-smoke table.** The family's records carry one — peak memory per rank, steps per
second, wall clock per epoch — and this one deliberately does not: no production run is in scope, and
a heading with no reachable number is worse than no heading. What is known without a run: the anchor
tiling reduces the five anchor-axis tensors by 13.8x against a dense decode of `[F, T_valid)` and by
25.9x against `[0, T_valid)`, both measured; the raw block's last axis is `R = 16` rather than
`C_keep = 98`, so these tensors are cheaper than the causal-feature cell's at equal anchor counts;
and the batch size stays at the raw-signal sibling's 128 deliberately, so the comparison is not
confounded by a different gradient-noise scale.

---

## Headline baseline

| Metric | Train | Val | Note |
|---|---|---|---|
| `total_loss` | | | a mixed-unit criterion: the two shape terms are L1 and Huber on z-scored samples |
| `main_loss` | | | the pure-nats criterion |
| `nll_full_block` | | | nats per anchor over 480 raw samples |
| `nll_base_block` | | | |
| `nll_full_sample` | | | nats per raw sample |
| `nll_base_sample` | | | |
| `pred_gap` | | | sign is not a criterion |
| `source_conditioned_kl_raw` | | | |
| `anchor_coverage_frac` | | | |
| epoch of the `val/total_loss` checkpoint | | | |
| epoch of the `val/nll_full_block` checkpoint | | | |
| epoch early stopping fired | | | against the configured budget, at patience 50 |

Every block-scored row above is under the horizon weighting, and there is no unweighted second
reading of the same quantity anywhere in this package — this row ships no evaluation pipeline, so
the training CSV is the only reading there is. A number here is therefore not comparable to the
raw-signal sibling's same-named column, which is uniform on the horizon axis.

---

## Bottleneck health

A headline number can look healthy while the bottleneck is not, and each row below is a different way
for that to happen.

| Metric | Value | What a bad value means |
|---|---|---|
| `source_conditioned_kl_raw` | | at zero the source pathway carries nothing |
| `kld_active_frac` | | near zero the latent collapsed |
| `mu_post_prior_gap_rms` | | at zero the posterior never leaves the prior — averaged over the decoded anchor set, the same support as the KL beside it |
| `logvar_prior_floor_frac` | | near one the prior scale pinned on its clamp floor |
| `mu_prior_sat_frac` | | near one the tanh bound on the prior mean is binding |
| `delta_mu_sat_frac` | | near one the bound on the posterior delta is binding |
| `logvar_full_floor_frac` | | near one the decoder's observation variance pinned low |
| `logvar_full_ceil_frac` | | near one it pinned high |

---

## Forecasting or reconstructing?

The question the feature cells fill this table with is whether a share of the short-horizon target is
already determined by signal the model has legitimately observed, and they answer it with the gap
resolved per horizon step and per stored block. **Neither readout exists on a raw target**: a raw
sample is not an average over any window, so nothing in it is a smear of observed input, and the
block's last axis has no stored blocks to split — those four columns are feature-domain and are not
tracked here. What remains of the question is the raw trace's own memory at the horizon: on the
committed fixture it autocorrelates at 0.205 at 60 s, which is the trivial-predictor floor a forecast
has to beat, and the horizon arm is the one lever that separates a near-horizon gain from a
far-horizon one.

| Metric | Value | Reading |
|---|---|---|
| `pred_gap` at `horizon: 15` (shipped) | | the one-minute block |
| `pred_gap` under `sweep_horizon_15.yaml` | | the two-minute block, which the raw-signal sibling scores; not a level comparison against this row's other cells |
| `nll_full_sample` against `nll_base_sample` | | per raw sample rather than per block, so the horizon length is divided out |
| `aux_multiscale`, `aux_derivative` | | the envelope and slope terms, in L1 and Huber units — whether the mean is over-smoothed |

---

## The warm-up and the tiling

The one geometry guard, and the readouts that used to size the source compromise.

| Metric | Expected | Measured | Reading |
|---|---|---|---|
| `anchors_per_sample` (train) | `[4, 5]` | | the tiling firing at the configured stride; 5 tiles at phase 0-15, 4 otherwise |
| `anchors_per_sample` (val) | `136` | | the dense evaluation set, `[134, 270)` |
| `source_lag_warmth_frac_st` | `1.0` | | a constant here, not a measurement: every reachable lag is warm |
| `source_lag_warmth_frac_ph` | `1.0` | | over **zero** channels — an absence, since `up_ph` was dropped whole |
| `kld_source_null` | — | | close to `source_conditioned_kl_raw` means the readout is a clock |

The input-warmth policy behind the floor is over the **target-stream** input channels alone: every
kept one is warm by the first forecast step, and every shifted one is warm at the anchor. At this
cell's 42.21 s reference the survivors are the *fast* channels, so that policy asks for a floor of
only **6** steps against the shipped 134 — **nothing binds the floor any more**, and it is retained
as an anchor-cost policy rather than lowered, because lowering it is a training-cost decision that
belongs with a run. What it costs is stated rather than hidden: 136 anchors against the 240 a floor
at the model's own 30-step warm-up would give.

The source is never gated by the warm-up budget — all 51 channels survive it — and the alignment
then drops the 34 whose composed delay exceeds the reference, including the whole 15-channel `up_ph`
block. That is why the two warmth columns above are constants: the compromise they were written to
size does not exist on this arm. It does on the unaligned one, where the source keeps channels
waiting up to 278 steps by design. `DESIGN.md` sections 3 and 8 are the record.

## What the production runs still owe

- Every empty cell above.
- The per-epoch diagnostic page and the run-level budget figure, attached, with any disagreement
  against the metrics recorded.
- The two deferred decisions: whether the beta pair carried across from the raw-signal sibling holds
  at this block and anchor count, and whether the floor is worth its anchors — the second answerable
  only by running the `F = 30` geometry `DESIGN.md` §14 records as its first `lean-limit:` line, which
  no shipped arm does. Both are judgements to be taken **with the number that drove them recorded**,
  not threshold gates.
- The re-derivation of both loss-scale constants from a production `train/grad_norm` column.
