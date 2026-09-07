# Causal-input raw-target conv-Transformer forecaster — results

Status: criteria registered for the revised architecture, runs not yet made.
Last updated: 2026-08-27

Every table below is a form with its columns named and its cells empty. It is written **before** the
headline run rather than after it, so the criteria a run is judged against cannot be chosen once the
numbers are in view. Fill the cells; do not add a criterion, and do not soften one.

**This form is re-issued for the revised architecture and describes no trained run.** Four
mechanisms arrived — the lag attention's local keys and values, the prior's clock, the
horizon-weighted reconstruction and the flat lag-bias seed — and each is a key whose off-state
reproduces the previous architecture bitwise; `teb_vae/lag_attn_crws/DESIGN.md` §17 is the
inventory, shared by both cells of this row. Two mechanisms the feature-target cells took are
**declined on this row**, the decoder's persistence residual and a second alignment reference, and
both constructors refuse them by name. The training controls are also on for the first time: early
stopping at patience 50, and a second checkpoint criterion on `val/nll_full_block` beside the
composite one. Both cells of the row ship the same four values, which is what keeps the encoder edge
below readable.

---

## What this study measures, and the rules for reading it

This is the eighth cell of the encoder-by-target grid, and the conv-Transformer half of the one row in
which neither side of the objective contains its own future:

```
                          conv-LSTM encoders          conv-Transformer encoders
  raw FHR target          lag_attn_rws                lag_attn_transformer_rws
  two-sided feature       lag_attn_fs                 lag_attn_transformer_fs
  causal feature          lag_attn_cfs                lag_attn_transformer_cfs
  causal in / raw out     lag_attn_crws               lag_attn_transformer_crws   <- this
```

It sits at the corner of two edges and is read along both. Against `lag_attn_crws` the configs differ
in the encoder block alone — seven keys added, five removed, plus `lr_warmup_steps` and a re-measured
gradient clip — so a difference in results is attributable to the **encoder**. Against
`lag_attn_transformer_rws` they differ in the input representation alone, so a difference there is
attributable to the **inputs**. The encoder edge is pinned leaf-for-leaf by
`tests/test_config_load.py` outside a declared allow-list of nineteen exemptions.

**Five rules govern every number on this page.**

1. **There is no evaluation package, so every number on this page is in-sample and carries no
   uncertainty.** It is a scalar read off a run's own `train_results/metrics_history.csv`: no
   confidence interval, no held-out population, and no held-out claim available from a config whose
   two splits are one shard. A difference between two runs of this model is evidence about those two
   runs. Stated once, here, so no row below is read as though it had a confidence interval; a
   `ModelBinding` against `teb_vae/lag_attn_rws/eval` is the follow-up that would change this for
   both cells of the row, and `teb_vae/lag_attn_crws/DESIGN.md` §14 records what it would cost.
2. **The nats are comparable only within this row, at this horizon, under this objective.** The
   reconstruction is summed over `H * R = 30 * 16 = 480` raw samples per anchor, which is the same
   block `lag_attn_transformer_rws` sums — but the decoded anchors per training step are about 4.5
   here against a dense 240, and this row's objective weights the horizon axis while that cell's does
   not, so no cross-cell comparison of a loss *level* is meaningful across the input-representation
   edge; what carries across is a sign, a trajectory, the bottleneck-health columns and the ordering
   of arms. The feature cells sum `H * C_keep` coefficients and are further away again. The one model
   whose loss level is comparable to this one's is `lag_attn_crws`, which ships the identical
   geometry and the identical half-life — the earlier `15 * 16 = 240` block is the arm that cell's
   `sweep_horizon_15.yaml` restores, and it restores neither the anchor count nor the uniform horizon
   axis.
3. **A negative `pred_gap` is not a failure of anything.** `lag_attn_rws/PRIOR_SCALE_AND_PREDICTIVE_GAP.md`
   section 5 establishes that the held-out predictive gain is negative because the source pathway
   does not generalise — a failure in the source encoder, the lag attention and the posterior
   fusion. Replacing the encoders changes which modules those are but not the finding, and this cell
   is expected to reproduce it. Its sign is a criterion nowhere on this page.
4. **A lag-resolved figure is an attribution over stored-coefficient time on the input side —
   unless the source stream has been aligned, and then it is a physical lead time.** The raw target
   carries no group delay, so the anchor is exact and the forecast claim needs no correction; a
   causal input coefficient beyond its warm-up still lags by its own composed group delay, 13.3 to
   791.0 s depending on the channel. Unaligned, that bias is indexed by a channel *pair*, no single
   number labels the axis, and no attention peak on this page may be read as a physiological delay.

   Under `causal_align_reference` every source channel is shifted onto one reference
   `source_reference_delay_s`, the bias collapses to a single known constant, and the correction is
   one-sided — there is no target-side term to subtract, exactly rather than approximately. A peak
   at lag `l`, horizon element `h`, is then a lead time of
   `4*(l + 1 + h) + 0.875 * source_reference_delay_s - 20` seconds, the `-20` being the acquisition
   shift preprocessing already removed. The arithmetic is
   `teb_vae/lag_attn/nets/lag_report.py::physical_lag_seconds`, and the reference is read from the
   run's own resolved-config dump rather than from the model, whose `source_delay_steps` is the
   largest *stored-step* shift and is a different quantity. **This is the only row of the grid where
   the claim is available**: the causal-feature cells have a nonzero target reference, so the same
   expression there is a lag between two coefficient epochs and not between two signals.

   **The 0.875 is load-bearing and is the one factor a reader will drop.** A shift is applied at the
   energy centroid of the impulse response, not at its envelope mean, so the delay every aligned
   channel actually lands on is 0.875 of the reference the dump records. Substituting the recorded
   42.21 s straight into the expression overstates every lead on this page by 5.3 s.

   **And the reference this row ships is 42.21 s, not the grid's `target_max`, because otherwise the
   expression has no useful values at all.** With a raw target nothing cancels the source reference
   out, so the lead is *minimised* at `l = h = 0` and grows with the lag: at `target_max` the
   smallest expressible lead is 335.9 s, the whole 20-120 s contraction-to-deceleration band is
   unreachable at every lag index, and raising `max_lag` only makes it worse. At 42.21 s the
   reachable lead runs 20.9 to 496.9 s, so 99 of the 100 s band is inside the lag axis, at
   `l + h` in [0, 24]. What it costs is in the warm-up section below and in `DESIGN.md` section 3:
   38 of 102 target-stream channels survive and 17 of 51 source ones, and the 17 are all `up_st` —
   the whole `up_ph` block goes, so contraction morphology leaves the source stream entirely.

5. **What a lag readout can express is now partly a configuration, and this is the cell where that
   configuration is honest.** `lag_kv_source` decides which source representation the attention's
   keys *and* values come from: under the previous `encoder` arm the lag-0 key and value were a
   function of the whole causal prefix, so by the data-processing inequality lag 0 already contained
   whatever any later lag carried and a profile pinned there was reporting a representation rather
   than an absence of delay. The shipped `conv_stem` removes the deep stage, and on **this**
   architecture the stem reaches **21 steps** against a 91-lag window — genuinely local, where the
   conv-LSTM cell of this row inherits a schedule reaching 387 and bounds recurrence rather than
   memory. So a lag profile read off this cell's shipped arm is a reading about a local K/V and that
   cell's is not, which is a second thing the encoder edge below carries. `alibi_slope_scale`
   decides what the learnable lag bias is *seeded* with: the shipped `0.0` seeds it flat, so a
   profile reports what training put there, and the `1.0` seed of the previous default is a monotone
   decay towards lag 0 which was measured on a planted-delay fixture to be doing a large share of the
   pinning by itself. Neither key changes what the physical-lag identity of rule 4 means; both change
   what a profile is capable of saying.

**One architectural claim this cell can make that the conv-LSTM cell of this row cannot.** Step-wise
causality of the *history states* holds unconditionally: there is no time-pooling normaliser to
causalise, and `causal_norm` is not a constructor keyword of this model at all, so no configuration
of it exists in which the claim fails. Together with the causal transform on the inputs and the raw
target, that makes this the one cell of the grid whose token causality, input causality and target
exactness are all unconditional.

---

## Pre-registered acceptance criteria

Two tiers, and the distinction is the point. **Tier 1** asks whether the machinery did what it was
built to do; a failure there voids the run. **Tier 2** is the science, and this is the first
causal-input raw-target model at this architecture — there is no prior against which a threshold on
it could have been calibrated, and against the raw-signal cell on these encoders the block halved and
the decoded anchor count per step fell by roughly 24x, which changes the optimisation regime. Tier 2
numbers are **reported and interpreted, not passed or failed**; a fixed threshold on any of them
would be a guess dressed as a gate.

### Tier 1 — must hold, or the run is void

| # | Criterion | Where it is read | Result | Value |
|---|---|---|---|---|
| 1 | `anchors_per_sample` is in `[4, 5]` on training rows and exactly `136` on validation rows | `metrics_history.csv` | | |
| 2 | The loss is finite on every step and the spike breaker never latches | `train/total_loss`, `train/spike_skipped` | | |
| 3 | Two evaluations of the final checkpoint produce an identical metric row set | two run directories | | |
| 4 | The step-granular learning-rate ramp was live: `lr` on the first logged rows sits below its configured value and reaches it by `lr_warmup_steps` | `metrics_history.csv` | | |
| 5 | The resolved configuration states every architecture switch and the alignment reference, and the run's identity keys name the arm | `resolved_config.yaml` | | |

Three criteria are the conv-LSTM cell of this row's, and the two absences are deliberate there and
here alike: the causal-feature cells' stamped provenance column and their recomposition criterion
are both readings of a target that has a warm-up and a channel axis, and a raw target has neither.
Criterion 4 is this cell's own, because this package sets `lr_warmup_steps` and a pre-normalised
attention stack needs the ramp in exactly its first few hundred updates; `tests/test_train_smoke.py`
asserts it on the fixture fit.

Criterion 5 is about provenance rather than geometry, and it is read **by hand** because this row
ships no run checker. The driver builds a run's model kwargs by sweeping the constructor's signature
and **silently drops** any key the class does not re-list, so an arm can train as the baseline with
nothing in the metric history saying so. Five leaves must be present in
`model_checkpoints/resolved_config.yaml` — `lag_kv_source`, `prior_availability_input`,
`horizon_weight_halflife_steps`, `alibi_slope_scale` and `causal_align_reference` — and the run's
`run_name` and `variant` tag must name the arm. Presence rather than value: each key has a comparison
arm on the other side of it. The two keys this row declines must be **absent**, and the constructors
refuse them by name so a config carrying one fails at the key.

### Tier 2 — reported and interpreted

| # | Quantity | What it separates | Value | Reading |
|---|---|---|---|---|
| 5 | `source_conditioned_kl_raw` | the coupling readout: its trajectory, its final value, and whether it is still rising at the end | | |
| 6 | `kld_active_frac`, `logvar_prior_floor_frac` | whether the latent collapsed and whether the prior scale pinned on its clamp floor | | |
| 7 | `kld_source_null` beside `source_conditioned_kl_raw` | whether the coupling readout is measuring source *content* or the availability *clock* | | |
| 8 | `shuffle_penalty`, `source_lag_warmth_frac_st`, `source_lag_warmth_frac_ph` | whether a stranger's source is worse than this one's, and how much attention mass lands on cold source lags, per stored source block | | |
| 9 | `pred_gap`, read beside `lag_attn_crws`'s and `lag_attn_transformer_rws`'s | the headline, along both edges: as a level across the encoder edge, as a sign and trajectory across the input-representation one | | |
| 10 | `pred_gap_tau_first` beside `pred_gap_tau_last` | whether the horizon weighting did what it was introduced to do: the near steps improve and the far steps do not degrade by more than the weight removed from them | | |
| 11 | The epoch at which `val/total_loss` is minimised, beside the epoch at which `val/nll_full_block` is | whether the composite optimum and the best conditioned forecast are the same epoch, which the two checkpoint criteria exist because they need not be | | |

**Criterion 7 carries a caveat that must be read with it.** `prior_availability_input` gives the
prior the same availability clock the posterior already had, so that the clock cancels in the
divergence rather than being subtracted from it afterwards — but it will not drive the ratio to
zero. The posterior is a bounded residual on the prior, so the mean half of the divergence at a
silent source is a function of the delta head alone and the prior's input cannot appear in it;
conditioning the prior reaches only the variance half. The informative quantity is therefore the
**difference** rather than the ratio, and a collapse to ~0 would mean the posterior parameterisation
had changed. On this row the mechanism has less to do than on the feature-target cells, because every
kept source channel arrives by step 6 and there is correspondingly little arrival transient for the
encode of silence to carry past the anchor floor.

Criterion 7 is the single most important number on this page. The source availability pattern is a
deterministic function of the step, identical in every row of the batch, and it enters `q(z | Y, U)`
and not `p(z | Y)` — so it can push the posterior off the prior and inflate the coupling readout with
no source information in it at all. The permutation control deranges rows, and no permutation of rows
can remove something every row shares.

A `source_lag_warmth_frac` near zero is **not** a failure. It sizes the compromise the design makes
on the source: lag attention searches back into a region where much of the source is still inside its
own warm-up, and the design keeps every source channel rather than gating them.

**At the shipped reference both warmth columns are pinned at 1.0 and criterion 8 loses two of its
three quantities.** `source_lag_warmth_frac_ph` is 1.0 over *zero* channels: the 42.21 s reference
keeps none of `up_ph`, and an empty block is reported warm at every step by deliberate design,
because a zero there would read as a measurement rather than as an absence. `source_lag_warmth_frac_st`
is 1.0 for an unrelated reason: the surviving `up_st` channels are honest from step 6, the anchor
floor is 134 and `max_lag` is 90, so the coldest source step any decoded anchor can read is 44 and
every reachable lag is warm. Neither is a result. Record them, record the kept width per block beside
them, and read `shuffle_penalty` as the whole of criterion 8 until a lower floor arm exists.

### The two edges, read after both cells have run

Neither is a Tier 1 criterion — a difference along either edge is a *finding*, not a gate — but both
are the reason this cell exists, so both are recorded here.

| Edge | Compared against | Quantity | This cell | The other | Reading |
|---|---|---|---|---|---|
| encoder | `lag_attn_crws` | `source_conditioned_kl_raw` | | | |
| encoder | `lag_attn_crws` | `pred_gap` | | | comparable: same block, same anchor count |
| encoder | `lag_attn_crws` | `kld_source_null` | | | |
| encoder | `lag_attn_crws` | `nll_base_block`, `nll_full_block` | | | comparable as levels |
| encoder | `lag_attn_crws` | the lag profile and its shape — peak width, mass above half peak, degeneracy | | | **asymmetric**: this cell's local K/V reaches 21 steps and that cell's 387, so a difference here is about the stems and not only about the encoders |
| inputs | `lag_attn_transformer_rws` | `source_conditioned_kl_raw` | | | the coupling readout on inputs that do and do not contain their own future |
| inputs | `lag_attn_transformer_rws` | `pred_gap` | | | **not** comparable as a level; read the sign and the trajectory only |
| inputs | `lag_attn_transformer_rws` | `kld_active_frac`, `logvar_prior_floor_frac` | | | comparable: both are fractions |

The input-representation edge has no horizon arm on this encoder, deliberately: `lag_attn_transformer_rws`
*is* the `H = 30` raw-target model on these encoders, and the conv-LSTM cell's `sweep_horizon_15.yaml`
already carries the level comparison one package over. Neither edge has a rendered table with
intervals, because there is no evaluation package for either cell; both are filled by hand from two
runs' CSVs.

---

## The four readouts this row adds

Three are emitted on both stages and one on the evaluation stages alone, all of them the conv-LSTM
cell of this row's and reached by import. **Eight columns the causal-feature cell carries are absent
here on purpose** — the feature target's four gap splits by horizon step and by stored block, the
three warm-up tertiles and the target-side warm fraction — because a raw target has no blocks, no
kept channels and no warm-up; `teb_vae/lag_attn_crws/RESULTS.md` names them.

| Metric | Stages | What it separates |
|---|---|---|
| `anchors_per_sample` | train, val | the tiling actually firing; `[4, 5]` in train, `136` in val — a guard, not a result |
| `source_lag_warmth_frac_st` | train, val | attention mass on lags where the first stored source block is warm; pinned at 1.0 at the shipped geometry |
| `source_lag_warmth_frac_ph` | train, val | the same for the second, which the 42.21 s reference empties — 1.0 over zero channels |
| `kld_source_null` | val | the KL floor the availability clock induces with no source content |

`LagAttnTrfCrwsTrainer.TRACKED_METRICS` is 77 entries and is the conv-LSTM cell's tuple by identity.

---

## Launch lines

```bash
# Dev box: the local smoke, one device, the committed causal fixture
python -m teb_vae.lag_attn_transformer_crws.trainer \
    --config teb_vae/lag_attn_transformer_crws/configs/tiny.yaml

# Dev box: the instrumented run the clip was re-measured on -- shipped widths, gaussian_nll, the
# committed fixture, the clip parked at 1e9, the LR ramp shortened to 100 steps
python -m teb_vae.lag_attn_transformer_crws.trainer \
    --config teb_vae/lag_attn_transformer_crws/configs/smoke_causal.yaml

# Production box: the baseline, seven ranks. The rank count must equal len(cuda_devices).
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_transformer_crws.trainer \
    --config teb_vae/lag_attn_transformer_crws/configs/default.yaml

# The one arm: dense training anchors.
python -m teb_vae.lag_attn_transformer_crws.trainer \
    --config teb_vae/lag_attn_transformer_crws/configs/sweep_anchor_stride_1.yaml
```

The two dev-box lines run today. The production and arm lines need one precondition this repository
does not yet contain: `default.yaml` points `vae_train_datasets`, `vae_test_datasets` and
`stat_path` at `REPOINT_ME` placeholders, deliberately non-existent rather than pointing at two-sided
data, so a run fails on a missing file instead of on a `transform` refusal someone might "fix" by
dropping the budget. There is **no evaluation line and no run-checker line**: neither cell of this
row ships either, and every number below comes off the training CSV by hand.

---

## Where the numbers come from

| Source | What it provides | Caveat |
|---|---|---|
| `train_results/metrics_history.csv` | every scalar on this page | validation columns are the epoch mean over validation steps; training columns are the epoch mean over training steps |
| `train/grad_norm` | the pre-clip gradient norm | sampled one optimizer step per epoch, so read it as a distribution over epochs rather than per row |
| the per-epoch diagnostic page | the raw forecast tiled off `anchor_index`, the anchor overlay, the two input rows with their warm-up staircases — the conv-LSTM cell's page, reached through two levels of inheritance | drawn at the dense anchor set and at phase 0, which is not the geometry a training step used |
| the run-level warm-up budget figure (`causal_warmup_budget`) | the channels the budget dropped beside the ones it kept | a constant of the shard, not of the run |

**No run checker.** No production run is in scope for either cell of this row, so a checker would
have nothing to read; `anchors_per_sample` is read by hand from the CSV against its band until one
exists, and `tests/test_train_smoke.py` asserts it on the fixture fit.

---

## Before launching: what reverts, and when to stop

### What reverts, and how

This package's arrival, together with the conv-LSTM cell of this row's, edited **seven** existing
files, all of them one kind of file, and nothing else — the two packages were registered in the same
edit, so the revert record is shared and is stated in full here as well as in
`teb_vae/lag_attn_crws/RESULTS.md`. Every shared-tree seam the anchor tiling needs was already in the
tree, landed and proven inert by the causal-feature cell; `teb_vae/lag_attn_cfs/RESULTS.md` carries
that revert record and it is not restated here. The revert is a list of **files**, not a count of
commits, so that undoing it is a checkout rather than an archaeology exercise.

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
ten package names rather than eight, and the seven files pass with unchanged counts before and after
the registration. No shipped source module moved: `scripts/print_objective_metrics.py` prints every
objective metric of the four shipped forecasters in about a minute, and its output is byte-identical
to the fingerprint recorded before either cell of this row existed. Every other member either cell
needed from a sibling is **bound by reference** or reached by inheritance rather than edited into
place — `teb_vae/lag_attn_crws/DESIGN.md` §6 is the record — so `git status --porcelain` over the
sibling packages shows the files above and nothing else of this row's making.

### Go/no-go while a run is in flight

| Signal | Threshold | Action |
|---|---|---|
| `train/total_loss` non-finite | any row | stop; the breaker's non-finite guard should have caught it first |
| `train/spike_skipped` sustained above zero | | stop; the margin is mis-tuned for this objective |
| `anchors_per_sample` outside its band | any row | stop; the tiling is not the one the configuration states |
| `train/grad_clip_frac` near 1 | | investigate; the clip is binding on ordinary steps rather than on blow-ups |
| `lr` flat at its configured value from the first row | | investigate; the step-granular ramp was not applied and the attention stack was trained through its first updates unramped |
| `logvar_prior_floor_frac` climbing past 0.5 | | investigate; the prior scale is collapsing onto its clamp floor |

### When a stop fires

Record the epoch, the row that fired it, and the run directory. A stopped run is a result about the
configuration, not a run to be quietly relaunched at other settings.

---

## Parameter budget — measured

**Two rows per configuration, and both are the record.** The **shipped** rows carry the four
architecture switches at their revised defaults; the **off-state** rows carry every one at its inert
value, which is bitwise the pair that shipped before this revision — and it is the row on which the
input-representation comparison against `lag_attn_transformer_rws` is still readable, because that
cell never takes the new keys.

| Model | Configuration | Total | Decoder width | Block |
|---|---|---:|---:|---:|
| `lag_attn_transformer_crws` | **shipped**: local K/V, prior clock, weighted horizon, flat bias; budget 134, reference 42.21 s | 4,218,476 | 16 | 480 |
| `lag_attn_transformer_crws` | shipped but unaligned | 4,242,284 | 16 | 480 |
| `lag_attn_transformer_crws` | shipped but ungated | 4,223,724 | 16 | 480 |
| `lag_attn_transformer_crws` | shipped but `lag_kv_source: encoder` | 5,006,444 | 16 | 480 |
| `lag_attn_transformer_crws` | every switch off, reference 42.21 s | 4,989,804 | 16 | 480 |
| `lag_attn_transformer_crws` | every switch off, `target_max` reference 402.1604 s | 5,012,844 | 16 | 480 |
| `lag_attn_transformer_crws` | every switch off, unaligned | 5,013,612 | 16 | 480 |
| `lag_attn_transformer_crws` | every switch off, ungated | 4,995,052 | 16 | 480 |
| `lag_attn_crws` | shipped, reference 42.21 s | 4,589,907 | 16 | 480 |
| `lag_attn_crws` | every switch off, reference 42.21 s | 5,081,146 | 16 | 480 |
| `lag_attn_crws` | every switch off, `target_max` reference 402.1604 s | 5,104,186 | 16 | 480 |
| `lag_attn_transformer_rws` | reach budget 120 s | 5,003,116 | 16 | 480 |
| `lag_attn_transformer_rws` | ungated | 4,996,844 | 16 | 480 |

**The four architecture switches cost −771,328 here** and −491,239 on the conv-LSTM cell of this row,
and the whole of the difference is the two stems: the deep source encoder is not built (−888,960
here, −1,312,231 there), the local K/V stem is (+100,992 here, +804,352 there), and the prior's
clock projection and its own norm are (+16,640 = `128 * 128` plus `2 * 128`). The horizon weight is
a non-persistent buffer and the flat bias seed reuses an existing parameter, so both cost zero, and
there is no persistence term on this row because this row does not take that key. **Each stem
reuses its own parent encoder's convolution schedule**, so the arms differ in what is *removed*
rather than in two chosen front ends: `(5, 9)` at dilations `(1, 2)` reaching 21 steps here, against
`(3, 5, 11, 15, 15)` at `(1, 2, 4, 8, 16)` reaching 387 there. **Every number below is unchanged by
those four switches**, which is the check that they and the alignment are independent.

Three `causal_align_reference` arms, each reachable at one key. The `target_max` row is the
reference every *feature* cell of
the grid still uses and this row cannot, for the reason in rule 4 above; it is kept here because it
is what the family's shared test fixture resolves and what the grid-wide comparisons were first
written against. The unaligned row is `causal_align_reference: null`. Ungated means the whole guard,
the mask and the shift together.

**Moving the reference from `target_max` down to 42.21 s costs −23,040**, identically in both cells
of this row: the target adapter loses 60 channels and the source adapter 30, each from an input
linear and an availability projection of width 128, at `128 * 60 * 2 + 128 * 30 * 2`.

**The alignment costs −23,808 at the shipped reference**, and −768 at `target_max` — the latter is
the number the other three causal cells share, because there the reference is the target's own
maximum and only four source channels sit above it. At 42.21 s the target loses 60 of 98 and the
source 34 of 51, against which both adapters gaining a start-of-record vector (`+2 * 128`) is a
rounding error.

**The encoder edge is −91,342 on the off-state rows**, guarded against guarded — the same value at
every reference, at every guard, and in the raw-signal, two-sided and causal-feature pairs, which is
what a difference living entirely in the two *deep* history encoders must look like. **At the shipped
configuration it is −371,431**, identical across the shipped, unaligned and ungated rows, so it is
still the two history stacks alone: the target encoders differ by +331,929 as they always did, and
the two local stems by −703,360 where the two deep source encoders differed by −423,271. Quoting one
of these two numbers where the other belongs is the easiest mistake this table now admits, so each
says which row it is read on.

**The input-representation edge is −13,312** against `lag_attn_transformer_rws`, read on the
**off-state** row because that cell never takes the new keys — it is the same number the conv-LSTM
pair shows. It **changed sign** with the reference: at `target_max` it is +9,728, because there the
causal streams are the wider pair. It decomposes, measured parameter name by parameter name, into
exactly **one surviving term** with the decoder head deliberately not among them:

| Term | Value | What it is |
| --- | ---: | --- |
| the horizon embedding | 0 | `nn.Parameter(torch.zeros(horizon, decoder_hidden))`; both cells forecast 30 steps |
| the two input adapters | −13,312 | `128 * (38 - 78)` and `128 * (17 - 29)` on the input linear *and* the availability projection of each stream |
| the two start embeddings | 0 | the reach guard builds both and, under the alignment, so does this cell |
| the decoder's output head | 0 | `raw_per_step` in both cells |

A delta that does not decompose into exactly that term is recorded as not decomposing rather than
rounded away. **The guard costs −5,248 here against +6,272 on `lag_attn_transformer_rws`**, and both
are right, and the sign is the whole of the difference: the reference drops 64 of 102 target-stream
channels and 34 of 51 source ones, so the narrowing of the two input linears outruns everything the
guard adds; the reach budget drops only 31 of 109 and 29 of 58, so there the two availability
projections and the two start embeddings still dominate. At `target_max` this cell sat on the
raw-signal side of that line too, at +17,792. Every parameter the guard adds here is under an
adapter.

`DESIGN.md` section 13 carries the same table; `tests/test_docs.py` measures every total in both
documents by constructing the models rather than comparing against literals.

---

## The loss-scale constants — measured

The encoder edge changes neither the block (480 raw samples) nor the anchor count (~4.57), so the
constants stated in nats of the summed block should not move across it and the gradient statistic
might — and only a run could say which. The instrumented run is `configs/smoke_causal.yaml`: 600
optimizer steps at the shipped widths over the committed causal fixture, the clip parked at 1e9 and
`lr_warmup_steps` shortened to 100 so the ramp does not outlast the run. Neither regime below skipped
a batch or bound the clip.

**The horizon weighting does not move any of the four, and that is measured rather than assumed.**
The weight is renormalised to `sum(w) = H`, so it redistributes the block's magnitude rather than
rescaling it: on a symmetric batch the weighted block sum is **4543.25** against the uniform
**4544.01**, a ratio of 0.99983. Without the renormalisation the same weight would have shrunk the
block by **1.807x** against an unmoved KL, which is exactly where the clip, the margin and $\beta$'s
standing against the reconstruction would have gone out of date with nothing saying so. At the
shipped half-life of 15.0 steps the resolved weights run 1.8063 at the first horizon step to 0.4729
at the last, a 3.82x spread that sums to 30. Both cells of this row ship the same half-life, which is
what lets the encoder edge above compare levels at all.

| Constant | Value | Conv-LSTM cell | Moved on the encoder edge? | Statistic it was set from | Measured |
|---|---:|---:|---|---|---|
| `gradient_clip_val` | 11100.0 | 12000.0 | **yes** | above the pre-clip gradient norm's q99 and below its maximum, whole-shard batch, rounded to 100 | q50 3690, q90 7073, q95 8213, **q99 11078**, **max 13180** |
| `additive_margin` | 2.5e+3 | 2.5e+3 | no — re-measured | 1.56x this cell's worst excursion, and set from the conv-LSTM cell's larger one so the two share a bar across an edge that changes neither the block nor the anchor count | batch 1: q50 −213, q90 89, q95 236, q99 871, **max 1598**; whole shard: q99 −4.9, max 5.6 |
| `ema_floor` | 1.0e+9 | 1.0e+9 | no | above any loss the objective can reach, which switches the relative test off | — |
| `horizon_embed_std` | 0.8 | 0.8 | no | the post-initialisation correlation between two horizon tokens | a per-pair quantity independent of the token count |

**The clip is rounded to 100 rather than to 500 or 1000, and the reason is falsifiable rather than
aesthetic.** The family's rule is the smallest round value above q99; at q99 = 11078 and max = 13180
both 1500 and 2000 sit *above the observed maximum*, so a guard set at either would not have bound on
a single step of the run it came from, while the conv-LSTM cell's 1000 sits *below* this encoder's
q99 and would rescale more than one step in a hundred. 1100 keeps "above q99, below max".

**The margin's bracket is (283, 759)**: strictly above the worst measured excursion, so ordinary
batches are not skipped, and strictly below the ~7.6e+2 magnitude the objective can reach, so the
additive test can still fire. 5.0e+2 sits at 1.8x the worst excursion against the conv-LSTM cell's
2.0x, so the equality across the edge is a measurement with a stated distance from its floor rather
than an inheritance. The batch-1 gradient norms (q50 5330, q99 34094, max 51514) are recorded and
deliberately not used. `main_loss` stayed inside [+173, +646] with the whole shard in one batch and
[+63, +873] at batch 1.

**Every number in this section is provisional**: four in-sample windows are a thinner tail than a
production run's. Re-derive both from the headline run's own `train/grad_norm` column once it exists.

**No distributed-smoke table.** The family's records carry one and this one deliberately does not:
no production run is in scope, and a heading with no reachable number is worse than no heading. What
is known without a run is the conv-LSTM cell's: the anchor tiling reduces the five anchor-axis tensors
by 13.8x against a dense decode of `[F, T_valid)`, and the batch size stays at 128 so the comparison
across both edges is not confounded by a different gradient-noise scale.

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
resolved per horizon step and per stored block. **Neither readout exists on a raw target** — a raw
sample is not an average over any window and has no stored blocks — so those columns are not tracked
here. What remains of the question is the raw trace's own memory at the horizon (0.205 at 60 s on the
committed fixture, the trivial-predictor floor) and the horizon comparison, which on this encoder is
not an arm of this package but the raw-signal cell one package over.

| Metric | Value | Reading |
|---|---|---|
| `pred_gap` at `horizon: 15` (shipped) | | the one-minute block |
| `pred_gap` on `lag_attn_transformer_rws` at `horizon: 30` | | the two-minute block on the same encoders; a sign-and-trajectory reading, not a level |
| `nll_full_sample` against `nll_base_sample` | | per raw sample rather than per block, so the horizon length is divided out |
| `aux_multiscale`, `aux_derivative` | | the envelope and slope terms, in L1 and Huber units — whether the mean is over-smoothed |

---

## The warm-up and the tiling

The one geometry guard, and the readouts that size the source compromise.

| Metric | Expected | Measured | Reading |
|---|---|---|---|
| `anchors_per_sample` (train) | `[4, 5]` | | the tiling firing at the configured stride; 5 tiles at phase 0-15, 4 otherwise |
| `anchors_per_sample` (val) | `136` | | the dense evaluation set, `[134, 270)` |
| `source_lag_warmth_frac_st` | 1.0 | | saturated: the survivors are honest from step 6 and the coldest reachable source step is 44 |
| `source_lag_warmth_frac_ph` | 1.0 | | saturated over an empty block: the reference keeps none of `up_ph` |
| `kld_source_null` | — | | close to `source_conditioned_kl_raw` means the readout is a clock |

The input-warmth policy behind the floor is over the **target-stream** input channels alone: every
kept one is warm by the first forecast step and every shifted one is warm at the anchor. At the
shipped 42.21 s reference the survivors are the *fast* channels, so `B = 1`, the shifts span 0 to 6,
and the policy requires only `F >= 6`. `F` ships at 134 regardless — more than twenty times the
requirement — so the floor has stopped being a constraint and is now purely an anchor-cost policy:
the dense set is 136 anchors over [134, 270), against the 264 a floor at the requirement would give
and the 240 the model's own 30-step warm-up would give.

The source is still never gated *by the warm-up budget*, but the alignment now takes 34 of its 51
channels for a different reason — a channel slower than the reference would need a negative shift,
which would read its own future — and what survives waits at most 6 steps rather than 278. That is
why both warmth columns are saturated at this geometry; see the Tier 2 note above.

---

## What the production runs still owe

- Every empty cell above, along both edges.
- The per-epoch diagnostic page and the run-level budget figure, attached, with any disagreement
  against the metrics recorded.
- The two deferred decisions the conv-LSTM cell of this row also owes: whether the beta pair carried
  across from the raw-signal family holds at this block and anchor count, and whether the floor is
  worth its anchors — the second answerable only by running the `F = 30` geometry the design records
  as its first `lean-limit:` line, which no shipped arm does. Both are judgements to be taken **with
  the number that drove them recorded**, not threshold gates.
- Whether the alignment helps any model outcome at all. Nothing measured so far says it does: the
  reference move is justified by an arithmetic argument about what the lag axis can express, not by
  a result. The comparison it asks for is the shipped arm against `causal_align_reference: null`,
  read on `source_conditioned_kl_raw` and on the lag profile's concentration rather than on
  `pred_gap` — the pre-registered reading is that a common clock should **not** improve the forecast
  gap, because it is bought with recency.
- Whether `warmup_period` should follow the reference down. At 42.21 s the input-warmth policy
  requires only `F >= 6` and the shipped 134 clears it twenty-fold, so the floor is now pure anchor
  cost; lowering it toward the requirement would roughly double the anchors per sample and change
  the optimisation regime. Deliberately not decided here.
- The re-derivation of the clip from a production `train/grad_norm` column, and a re-reading of
  whether the margin's equality across the encoder edge survives a production tail.
