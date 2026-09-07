# Causal-feature-domain forecaster — results

Status: criteria registered for the revised architecture, runs not yet made.
Last updated: 2026-08-27

Every table below is a form with its columns named and its cells empty. It is written **before** the
headline run rather than after it, so the criteria a run is judged against cannot be chosen once the
numbers are in view. Fill the cells; do not add a criterion, and do not soften one.

**This form is re-issued for the revised architecture and describes no trained run.** The
architecture changed after the previous form's runs were read: the lag attention's keys and values
now come from a local source representation, the prior is conditioned on the source pathway's encode
of silence, the decoder mean carries a target-only persistence residual, the reconstruction weights
the horizon axis, the lag bias ships seeded flat, and the source stream is aligned onto its own
faster clock. `DESIGN.md` §17 is the inventory and every one of the six has an off-state that
reproduces the previous architecture bitwise. The one section below that carries measurements is
*The identifiability record*, and those are measurements on a synthetic fixture whose answer is
known — not on any production run.

---

## What this study measures, and the rules for reading it

This is the fifth cell of the encoder-by-target grid and the first whose inputs do not contain their
own future:

```
                          conv-LSTM encoders          conv-Transformer encoders
  raw FHR target          lag_attn_rws                lag_attn_transformer_rws
  two-sided feature       lag_attn_fs                 lag_attn_transformer_fs
  causal feature          lag_attn_cfs   <- this      lag_attn_transformer_cfs
```

**Five rules govern every number on this page.**

1. **Every number on this page is in-sample and carries no uncertainty.** It is a scalar read off a
   run's own `train_results/metrics_history.csv`: no confidence interval, no held-out population,
   and no held-out claim available from a config whose two splits are one shard. A difference
   between two runs of this model is evidence about those two runs. There **is** now an evaluation
   package — `teb_vae/lag_attn_cfs/eval`, one checkpoint in and one reviewable directory out, with
   per-recording bootstrap intervals and ten pre-registered verdicts — and it is where a claim with
   an uncertainty on it comes from. *The evaluation's second reading of these criteria* below maps
   each criterion here onto the verdict that re-asks it there; nothing on this page is superseded,
   because the two answer different questions and at different times.
2. **The nats are comparable only within this family, at this budget, under this objective.** The
   reconstruction is summed over `H * C_keep = 30 * 98 = 2940` coefficients. `lag_attn_fs` sums
   `30 * 78 = 2340` and the raw cells sum `H * R` raw samples, so no cross-target comparison of a
   loss level is meaningful; and because `C_keep` is what the warm-up budget decides, two runs of
   *this* model at two budgets are not comparable to each other either. The objective now carries
   **two** weights — per channel and per horizon step — so a training-path number is not the
   evaluation's number for the same quantity, and the evaluation applies neither.
3. **A negative `pred_gap` is not a failure of anything.** `lag_attn_rws/PRIOR_SCALE_AND_PREDICTIVE_GAP.md`
   section 5 establishes that the held-out predictive gain is negative because the source pathway
   does not generalise — a failure in the source encoder, the lag attention and the posterior
   fusion, none of which a target-domain change touches. This family is expected to reproduce it,
   probably more strongly, because causal features are strictly weaker than two-sided ones. Its sign
   is a criterion nowhere on this page.
4. **A lag-resolved figure is an attribution over stored-coefficient time, now with a known constant
   offset.** One-sidedness and zero latency are different properties and only the first is bought
   here: beyond its warm-up a causal channel still lags by its composed group delay, 13.3 to 791.0 s
   depending on the channel. The alignment re-indexes every kept target channel onto 402.1604 s and
   every kept source channel onto 288.2672 s as `causal_delay_s` reports them, so what remains
   between the streams is one *constant* bias of −113.8932 s rather than a channel-pair-dependent
   range — but this cell's target is itself a stored coefficient on its own clock, so the target
   term cannot be divided out here and the attribution stays in coefficient time. The gather carries
   the centroid factor `ALIGNMENT_DELAY_FACTOR`, $\kappa = 1 - 1/(2\gamma) = 0.875$, because the
   delay a channel's own spectrum *realises* is a fixed fraction of the envelope mean the attribute
   stores; the common realised delays are $\kappa \cdot 402.1604 = 351.9$ s and
   $\kappa \cdot 288.2672 = 252.2$ s, and the realised inter-stream offset is −99.65 s. DESIGN.md §3
   carries the derivation. The forecast claim survives either way — a coefficient at `t` is a
   function of the past, so predicting `t + 1 + tau` from history up to `t` is a genuine forecast
   whatever the internal latency — but no attention peak on this page may be read as a physiological
   delay.
5. **The pooled lag argmax is predicted at the near edge on every arm and is not the comparison.**
   The searched window's lowest attainable lag is a *censoring* edge exactly as its highest is: at
   this geometry a 20–60 s physiological delay lands at lags 4.9–43.9 (at the realised offset
   $\kappa(\tau^u_{\mathrm{ref}} - \tau^y_{\mathrm{ref}}) = -99.66$ s; the 8.5–47.5 quoted earlier
   was priced at the unscaled −113.89 s), and anything faster is
   outside what the window can express. So a pooled argmax of zero is not evidence about an arm, and
   the acceptance gate does not read it as such. What the arms are compared on is named in *The
   comparison protocol* below, and it was fixed before any revised run existed.

**What this cell can claim that no other cell can.** Its coupling readout is measured on inputs that
do not contain their own future. That is the whole value of the run, and it is a property of the
*inputs* rather than of any number below.

### The comparison protocol, fixed before any number exists

Registered here so that no arm comparison can be defined once the numbers are in view. Across the
alignment arms, the K/V arms and the two lag-bias seeds, the quantities that carry the comparison
are:

| Read on | Why it and not the pooled argmax |
|---|---|
| the shape vocabulary — peak width, mass above half peak, degeneracy | the pooled peak's *location* is pinned by the censoring edge; its *shape* is not |
| the per-head profiles and per-head entropies | the pooled profile is an average over heads, and on this instrument single heads move while the pool does not |
| the occlusion analysis's per-horizon-step deltas per band | an interventional readout the window's near edge cannot pin, resolved on the one axis a physiological peak can sit on |
| `coupling_minus_clock_nats` against its 0.15-nat margin, with the bootstrap interval | the coupling readout net of the availability clock, which is the quantity the whole design exists to make readable |
| `pred_gap_warm_lo/_mid/_hi` and `pred_gap_novel_lo/_mid/_hi` beside the pooled `pred_gap` | a gain in `_novel_hi` is a forecast improving; a gain in `_novel_lo` is history inversion, and the pooled number cannot tell them apart |

And the pre-registered directions, carried over unchanged: an alignment arm **should not** improve
`pred_gap`, and **should** improve profile concentration if a real lag exists. A result that
improves `pred_gap` and leaves the profile flat is evidence about something other than alignment.

---

## Pre-registered acceptance criteria

Two tiers, and the distinction is the point. **Tier 1** asks whether the machinery did what it was
built to do; a failure there voids the run. **Tier 2** is the science, and this is the first
causal-feature model in the tree — there is no prior against which a threshold on it could have been
calibrated, and the decoded anchor count per step fell by roughly 15x, which changes the optimisation
regime. Tier 2 numbers are **reported and interpreted, not passed or failed**; a fixed threshold on
any of them would be a guess dressed as a gate.

`check_run.py` scores tier 1 by code, off a run's own CSV and resolved configuration, while the run
is still in flight; its verdict numbering is this table's.

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
other than `1.0` means the checkpoint was built by code that predates that refusal. It is in Tier 1
because it is what makes the pairing readable off a run months later.

Criterion 4 compares the three splits **against each other** rather than against `pred_gap`,
deliberately. `pred_gap` is `nll_base_block - nll_full_block`, a difference of two order-`1e3` sums
over 2940 coefficients, so it loses several decimal digits to cancellation *before* any split is
formed. The splits difference elementwise and agree to float32 noise; none of them agrees with
`pred_gap` to `1e-6`. All three are computed under **both** objective weights, because each is a
partial sum of the `pred_gap` printed beside it and the objective applies both to it.

Criterion 6 is new, and it is the one criterion here that is about provenance rather than geometry.
The driver builds a run's model kwargs by sweeping the constructor's signature and **silently drops**
any key the class does not re-list, so an arm can train as the baseline with nothing in the metric
history saying so — which is how a finished evaluation once came to be read as the aligned arm when
the run had trained unaligned. Presence rather than value is what is asserted: each of the seven keys
has a comparison arm on the other side of it, so no single value is the right one, and what is not
acceptable is a run whose own artifacts cannot say which side it was on.

### Tier 2 — reported and interpreted

| # | Quantity | What it separates | Value | Reading |
|---|---|---|---|---|
| 7 | `source_conditioned_kl_raw` | the coupling readout: its trajectory, its final value, and whether it is still rising at the end | | |
| 8 | `kld_active_frac`, `logvar_prior_floor_frac` | whether the latent collapsed and whether the prior scale pinned on its clamp floor | | |
| 9 | `kld_source_null` beside `source_conditioned_kl_raw` | whether the coupling readout is measuring source *content* or the availability *clock* | | |
| 10 | `shuffle_penalty`, `source_lag_warmth_frac_st`, `source_lag_warmth_frac_ph`, and the spread across the two tertile families | whether a stranger's source is worse than this one's, how much attention mass lands on cold source lags, and whether slow or history-heavy channels forecast differently from fast or novel ones | | |
| 11 | The epoch at which `val/total_loss` is minimised, beside the epoch at which `val/nll_full_block` is | whether the composite optimum and the best conditioned forecast are the same epoch, which on the previously diagnosed run they were not (336 against 278) | | |

**Criterion 9 is the single most important number on this page, and its pre-registration has a
caveat that must be read with it.** The source availability pattern is a deterministic function of
the step, identical in every row of the batch, and it enters `q(z | Y, U)` and not `p(z | Y)` — so it
can push the posterior off the prior and inflate the coupling readout with no source information in
it at all. The permutation control deranges rows, and no permutation of rows can remove something
every row shares. The design's intent was that `kld_source_null` collapse to ~0 once the prior is
given the same clock.

**It will not, and the reason is structural rather than a tuning failure.** The posterior is a
bounded residual on the prior, $\mu^q = \mu^p + \Delta_\mu$, so the *mean* half of
$\mathrm{KL}(q^\varnothing \Vert p)$ is a function of the delta head alone and the prior's input
cannot appear in it. Conditioning the prior reaches only the variance half. So criterion 9 is
registered as **reported**, with the prediction stated as: `kld_source_null` is expected to stay a
large fraction of the raw KL, the informative quantity is
`source_conditioned_kl_raw - kld_source_null` and its evaluation twin `coupling_minus_clock_nats`
against the 0.15-nat margin, and a collapse to ~0 would mean the posterior parameterisation had
changed. DESIGN.md §14 carries the open item and its trigger. Recording the prediction here, before
the run, is what keeps a large value from being read afterwards as a defect it is not.

A `source_lag_warmth_frac` near zero is **not** a failure. It sizes the compromise the design makes
on the source: lag attention searches back into a region where much of the source is still inside its
own warm-up, and the design keeps every source channel the warm-up budget would have taken, so the
residual is measured instead of resolved. The shipped source clock **narrows** that residual —
`up_st` reaches half-warmth at step 59 and `up_ph` at step 86, against 84 and 117 under one shared
clock, and at the lowest anchor 134 the searched source steps span 44 to 134, of which 76 of the 91
lags are `up_st`-warm and 49 are `up_ph`-warm against 51 and 18 — which is why these columns are read
beside the alignment arm rather than against a threshold. The mean per-channel warm fraction over the
whole (anchor, lag) grid is 0.970 shipped, 0.863 at one shared clock and 0.885 unaligned.

### The evaluation's second reading of these criteria

Every criterion above is now asked a second time, on a held-out population, per recording, with a
bootstrap interval — by `teb_vae/lag_attn_cfs/eval`, whose ten verdicts land in
`eval_results/summary.json` and are gated by `eval/verify.py`. The two readings are **not**
redundant and neither supersedes the other: the criteria above are read off a run's own CSV, in
sample, while the run is still going; the verdicts below need a finished checkpoint and a completed
evaluation pass over the causal holdout split. A criterion with no verdict beside it is one the
evaluation deliberately does not re-ask, and the reason is given rather than left as a gap.

| # | Criterion above | Evaluation verdict | Note |
|---|---|---|---|
| 1 | `target_warm_frac == 1.0` | `anchor_geometry_intact` | The same stamped column, re-read from the checkpoint's own geometry rather than from a logged row. FAIL-able there as here. |
| 2 | `anchors_per_sample` in `[10, 11]` / `51` | `anchor_geometry_intact` | The evaluation always decodes **densely**, so it re-asks the validation half only: exactly `51`. The training band is a fact about the tiling and has no held-out analogue. |
| 3 | finite loss, breaker never latched | — | None. This is a property of the optimisation trajectory, which a finished checkpoint no longer carries; `check_run.py` remains the only reading of it. |
| 4 | the three splits recompose | `warmup`'s recomposition guard | Re-asserted on the held-out population as the tertile split's own identity, under a tolerance scaled by the block score rather than by the gap — and **unweighted** there, since the evaluation applies neither objective weight. |
| 5 | two evaluations produce an identical row set | the reproducibility contract | Not a verdict: two runs of one checkpoint at one seed compare byte-identical on `results`, which the evaluation suite asserts directly rather than reporting. |
| 6 | the resolved configuration states the arm | — | None, but not a gap: the evaluation prints the configured arm label, both resolved clocks, the inter-stream offset and the K/V source on its own console block and carries the same three readings — configured, built, resolved — in `summary.json`, so a config naming one arm while the checkpoint carries another is visible rather than merged away. |
| 7 | `source_conditioned_kl_raw` | `source_specificity`, `source_margin_positive` | Reported per recording with an interval, and the trajectory question is answered by `time_to_delivery` rather than by an epoch axis. |
| 8 | `kld_active_frac`, `logvar_prior_floor_frac` | `latent_not_collapsed`, `prior_variance_not_pinned`, `decoder_variance_not_pinned` | Three verdicts where this row has two columns: the prior's clamp and the decoder's are separate failures and were one line here. |
| 9 | `kld_source_null` beside `source_conditioned_kl_raw` | `coupling_exceeds_availability_clock` | **Now a deciding verdict.** `clock_margin_min_nats` ships at 0.15, from the diagnosed unaligned run's observed spread (0.160, interval [0.157, 0.164]), so the difference must clear a stated bar rather than returning INCONCLUSIVE. `coupling_minus_clock_nats` and both interval ends are headline scalars from the first run. |
| 10 | `shuffle_penalty`, the warmth fractions, the tertile spread | `source_specificity`, and the `warmup` analysis | The warmth fractions get intervals but no verdict, deliberately: a small value is the expected finding and a threshold on it would gate a design compromise rather than a defect. |
| 11 | the two checkpoint criteria's epochs | — | None. Which epoch a criterion selected is a property of the trajectory; what the evaluation reads is whichever checkpoint it is pointed at, so the two epochs belong on this page and the two evaluations belong beside each other. |
| — | (no criterion here) | `predictive_improvement`, `prior_carries_target_state`, `calibration_near_nominal` | Three verdicts with no row above, because none of the three is answerable from a training CSV: the first needs a held-out population, the second a prior-shuffle control, the third a decoder calibrated against its own residuals. |

**The interventional readout has no verdict on purpose.** The occlusion analysis reports, per lag
band, how many nats the forecast loses when the source's *values* are removed there — with the
availability announcement held fixed, so the intervention moves content and not the clock. What a
healthy value is has never been measured, and a threshold guessed before the first production runs
would decide a pass or a fail on exactly the run that was going to measure it. Four scalars reach
every arm table instead: the winning band's name, its delta, its peak horizon step, and its **live
fraction** — because a band lying inside the warm-up scores near zero for a reason that is about the
geometry rather than about the source.

---

## The identifiability record

The one section on this page carrying measurements, and they are not from a production run. They are
from the committed planted-delay fixture: a synthetic shard in which the FHR modulation is a
deterministic function of the UP envelope a planted `delta = 45` stored steps (180 s) earlier, pushed
through the real causal bank. Informative lags are therefore `[delta - H, delta - 1] = [15, 44]` of a
`[0, 90]` window **and nowhere else**, which is what makes a peak outside that band a failure rather
than an ambiguity.

**The instrument was validated before any model saw it**, by direct cross-correlation on the written
coefficients: 12 coupled channels peak within 4 steps of 45 — the strongest at lag 45 with
`r = +1.000` — against 22 flat control channels at `|r| <= 0.11`. Nothing below is a property of the
fixture.

| | `lag_attn_cfs` | `lag_attn_transformer_cfs` |
|---|---|---|
| Pooled support-corrected argmax, every arm measured | 0 | 0 |
| Band mass, previous architecture | 0.208 | 0.192 |
| Band mass, revised default at the flat bias seed | 0.383 | 0.234 |
| Peak width at the flat seed / at the decaying seed | 38 bins / 3 bins | 18 bins / 3 bins |
| Per-head argmax, previous architecture | 0, 0, 0, 0 | 0, 0, 0, 0 |
| Per-head argmax, revised default at the flat seed | 33, 0, 90, 0 | 0, 66, 20, 0 |
| Band share of the in-band head | 0.424 at lag 33 | 0.324 at lag 20 |
| Per-head argmax under `lag_kv_source: adapter` | 0, 0, 0, 41 | 0, 0, 0, 31 |
| Band share of that head | 0.538 | 0.486 |
| `kld_source_null` as a fraction of the raw KL, previous architecture | 81.6% | 91.9% |
| KL at initialisation, every arm | exactly 0 | exactly 0 |

**Three things this establishes, and one it does not.**

- **The pooled argmax does not move, on any arm, on either parent.** That is the measurement rule 5
  is registered against: on ground truth where the informative lags are known, the pooled statistic
  reports the one lag the plant does not occupy, so it cannot be the quantity an arm comparison is
  read on.
- **The lag-bias initialisation was doing a large share of the pinning.** At the decaying seed every
  head on both parents peaks at lag 0 and the pooled peak is 3 bins wide; at the flat seed a head
  sits *inside the planted band* on each parent, the peak widens to 38 and 18 bins, band mass rises
  by 1.4x and 1.6x, and the per-head entropies rise across the board. That is the initialisation
  being measured as a cause rather than argued to be one, which is why the flat seed ships and the
  decaying one is a named arm.
- **The sharpest K/V arm is the only one on which a head reads the plant on both parents.** Under
  `adapter` — a one-step representation — head 3 peaks at lag 41 and lag 31. Under `conv_stem` this
  cell's stem reaches 387 steps and is therefore not local at all, so its row is not evidence about
  the localisation argument; the conv-Transformer cell's 21-step stem is, and it did not move the
  pooled statistic either.
- **What it does not establish is that the revised default recovers the plant.** It does not, on the
  pooled criterion, on any arm. That is recorded here rather than smoothed over, and it is why rule
  5 and the comparison protocol are written the way they are.

**The interventional readout does find the plant**, on the same checkpoint whose every head peaks at
the censoring edge. Four bands partitioning the window, scored under common random numbers on one
common anchor support:

| band | lags | seconds back | delta total (nats) | peak horizon step | live fraction |
|---|---|---|---|---|---|
| `anchor` | [0, 14] | 0–60 | −1.83 | 19 | 1.00 |
| **`near`** | **[15, 44]** | **60–180** | **+14.94** | **5** | **1.00** |
| `mid` | [45, 67] | 180–272 | −1.12 | 5 | 0.98 |
| `far` | [68, 90] | 272–364 | +0.64 | 25 | 0.81 |

Removing the source's values at exactly the lags the plant occupies costs the forecast 14.9 nats;
removing any other band costs nothing distinguishable from zero at eight segments, where the per-band
standard deviations at the peak horizon step are 1.9–2.3 nats. The two slightly negative bands are a
real state rather than a defect — removing source the model was mildly misusing improves the forecast
— and are inside that noise. Nothing here is a claim about a production run; what it establishes is
that the instrument resolves the axis it was built to resolve, which is why the occlusion deltas are
in the comparison protocol above.

---

## The named arms

The inventory is closed: exactly these, one axis each, plus the two identity keys that put the arm in
the run's own name. A config differing from the default in anything else is not an arm of this study.

| Arm | File | Delta from the default |
|---|---|---|
| Decaying lag bias | `sweep_lag_bias_decay.yaml` | `alibi_slope_scale: 1.0` |
| One shared clock | `sweep_align_target_max.yaml` | `causal_align_reference: target_max` (the default is unaligned since 2026-09-05) |
| Legacy comparator | `sweep_legacy_dualref_physclock.yaml` | the pre-2026-09-05 default: fractional phase operator, dual reference, physical clock, stride 5, legacy shards |
| Sharp K/V | `sweep_lag_kv_adapter.yaml` | `lag_kv_source: adapter` |
| Dense anchors | `sweep_anchor_stride_1.yaml` | `anchor_stride: 1` |
| Short horizon | `sweep_horizon_15.yaml` | `horizon: 15` and the stride that pairs with it |
| Higher floor | `sweep_floor_150.yaml` | `warmup_period: 150` |
| Shallower horizon stack | `sweep_horizon_depth_3.yaml` | `horizon_depth: 3` |

The unaligned arm moves **two** leaves and the second is forced rather than a second delta: the
resolver refuses a source reference against an unaligned target by name, so an arm moving only the
first key would not launch.

`planted.yaml` is not in this table because it is not an arm: it is the identifiability check's own
geometry, tiny channel widths at the production lag window, with its alignment pinned so a default
flip cannot move the instrument.

---

## The eleven readouts this cell adds

Ten are emitted on both stages and one on the evaluation stages alone. Almost all are partial sums
or fractions of quantities the objective already computes, so they add no second definition of
anything; `kld_source_null` is the exception, and it costs one source encode per validation step.

| Metric | Stages | What it separates |
|---|---|---|
| `target_warm_frac` | train, val | the budget-and-floor pairing; a constant, resolved at construction, exactly `1.0` |
| `anchors_per_sample` | train, val | the tiling actually firing; `[10, 11]` in train, `51` in val |
| `source_lag_warmth_frac_st` | train, val | attention mass on lags where the first stored source block is warm, at `W' + d` |
| `source_lag_warmth_frac_ph` | train, val | the same for the second, which is the block with the problem |
| `pred_gap_warm_lo` | train, val | the forecast gap over the slowest third of the kept target channels |
| `pred_gap_warm_mid` | train, val | the middle third |
| `pred_gap_warm_hi` | train, val | the fastest third |
| `pred_gap_novel_lo` | train, val | the forecast gap over the third of the kept target channels the anchor has most nearly already seen |
| `pred_gap_novel_mid` | train, val | the middle third |
| `pred_gap_novel_hi` | train, val | the third that is most genuinely new at the horizon |
| `kld_source_null` | val | the KL floor the availability clock induces with no source content |

The three warm-up tertile columns partition the 98 kept target channels by `W'` and recompose to
`pred_gap` over the same denominator. They are **not** a restatement of `pred_gap_st` /
`pred_gap_ph`: the kept set is 32 channels of the first stored block plus all 66 of the second, and
both blocks span nearly the same rebased range, so the tertiles cut *across* the block boundary.

The three novelty tertile columns are a third partition of the same 98 channels, by the share of
each coefficient's envelope mass lying after the anchor at the last scored element — looked up in the
shard's horizon-free `causal_novelty_curve` at the run's own horizon and per-channel advance (legacy
shards: the fixed-horizon `causal_novelty_frac`). On the shipped kept set at $H = 30$ that runs from
`1.000` to `0.026`, so `pred_gap` mixes two claims: a
good score on the low end is the model inverting its own delayed history, and on the high end it is
a forecast. The split makes which one is being reported readable per channel instead of assumed
uniform. It is not the warm-up split renamed — the slowest kept channel is warm across the whole
window and still `0.026` new.

**No readout was added by the revision, and that is deliberate.** Every mechanism it introduced is
visible in an existing column or in the resolved configuration: the horizon weight moves the
`pred_gap` family it already reweights, the persistence residual moves `nll_full_block` and
`nll_base_block` equally so `pred_gap` is untouched by it, and the K/V source and the prior clock are
constants of the configuration, so a per-step column of either would be the same value in every row.

---

## Launch lines

```bash
# Dev box: the local smoke, one device, the committed causal fixture
python -m teb_vae.lag_attn_cfs.trainer --config teb_vae/lag_attn_cfs/configs/tiny.yaml

# Dev box: the local validation, one device, a causal HIE shard (see the config's header --
# the shard does not exist yet and cannot be substituted with a two-sided one)
python -m teb_vae.lag_attn_cfs.trainer --config teb_vae/lag_attn_cfs/configs/smoke_hie.yaml

# Dev box: can this architecture recover a delay it is known to be looking at? Fits the committed
# planted-delay fixture and reads the lag profile back through the evaluation's own code.
python teb_vae/lag_attn_cfs/lag_recovery_check.py \
    --config teb_vae/lag_attn_cfs/configs/planted.yaml

# Production box: the revised default, seven ranks. The rank count must equal len(cuda_devices).
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_cfs.trainer --config teb_vae/lag_attn_cfs/configs/default.yaml

# Production box: the named arms. One axis each; each writes its own run name.
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_cfs.trainer \
    --config teb_vae/lag_attn_cfs/configs/sweep_lag_bias_decay.yaml
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_cfs.trainer \
    --config teb_vae/lag_attn_cfs/configs/sweep_align_target_max.yaml
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_cfs.trainer \
    --config teb_vae/lag_attn_cfs/configs/sweep_legacy_dualref_physclock.yaml
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_cfs.trainer \
    --config teb_vae/lag_attn_cfs/configs/sweep_lag_kv_adapter.yaml

# While a run is in flight: score it against the tier-1 criteria above, off its own CSV and its
# own resolved configuration.
python -m teb_vae.lag_attn_cfs.check_run --run-dir <run>

# After it finishes: the held-out evaluation, one reviewable directory per checkpoint. This is
# the only launch line here that needs the causal holdout split; see the note below.
python -m teb_vae.lag_attn_cfs.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt

# The offline gate, and the arm tables. Neither imports torch.
python -m teb_vae.lag_attn_cfs.eval.verify <run>/eval_results/summary.json
python -m teb_vae.lag_attn_cfs.eval.verify --runs <dir-of-runs> --out RESULTS_arms.md

# The same two for the conv-Transformer cell, through the same pipeline.
python -m teb_vae.lag_attn_transformer_cfs.eval.run \
    --checkpoint <run>/model_checkpoints/<name>.ckpt
python -m teb_vae.lag_attn_transformer_cfs.eval.verify <run>/eval_results/summary.json
```

**Each production run writes two checkpoints under distinct stems**, one selected on
`val/total_loss` and one on `val/nll_full_block`, and early stopping is on at patience 50 validation
epochs. Which of the two a given evaluation reads is a decision the operator records, because the two
are different epochs and the arm tables must not mix them.

The training lines run today. The evaluation lines need one precondition this repository does not yet
contain: `eval/configs/eval_overrides.yaml` points `vae_test_datasets` and `stat_path` at
`REPOINT_ME` placeholders, and preflight refuses the run by name until they are edited to a causal
holdout build and statistics regenerated from those same shards at `trim_minutes: 1.0`. That refusal
is deliberate — a placeholder that resolved to something would evaluate the wrong dataset silently.

---

## Where the numbers come from

| Source | What it provides | Caveat |
|---|---|---|
| `train_results/metrics_history.csv` | every scalar on this page | validation columns are the epoch mean over validation steps; training columns are the epoch mean over training steps |
| `model_checkpoints/resolved_config.yaml` | the arm, both alignment references and every architecture switch | the merged document, so it states values a launch config inherited rather than set |
| `train/grad_norm` | the pre-clip gradient norm | sampled one optimizer step per epoch, so read it as a distribution over epochs rather than per row |
| the per-epoch diagnostic page | the warm-up staircase, the anchor overlay, the forecast rows | drawn at the dense anchor set and at phase 0, which is not the geometry a training step used |
| the run-level warm-up budget figure | the channels the budget dropped beside the ones it kept | a constant of the shard, not of the run |
| the identifiability check's own directory | the switch header, the lag profile against the planted band, the state-dict manifest | a synthetic fixture at tiny widths; an instrument reading, never a result |

---

## Before launching: what reverts, and when to stop

### What reverts, and how

This package's arrival edited files outside it. The revert is a list of **files**, not a count of
commits, so that undoing it is a checkout rather than an archaeology exercise.

| File | What was added | Inert without this package? |
|---|---|---|
| `teb_vae/lag_attn_rws/nets/raw_masks.py` | an optional anchor set on the three mask functions | yes — the default reproduces the dense range bitwise |
| `teb_vae/lag_attn_rws/nets/losses.py` | an optional anchor set on `compute_loss`, and an optional horizon weight on the four score functions | yes — both default to the unweighted dense computation bitwise |
| `teb_vae/lag_attn_fs/nets/feature_target.py` | an optional anchor set on `_build_forecast_target` | yes |
| `teb_vae/lag_attn_rws/nets/controls.py` | the anchor argument on the permutation control, the source-null arm and the occlusion arm | yes — the two new arms are reached only from this cell's evaluation |
| `teb_vae/lag_attn_rws/nets/heads.py` | the optional clock input on the prior head | yes — `clock=None` is the unmodified original computation |
| `teb_vae/lag_attn_rws/nets/model.py` | the K/V source selection, the prior-clock key, the persistence key and the horizon-weight key | yes — every one defaults to the pre-revision construction |
| `teb_vae/lag_attn/nets/decoders.py` | the optional target-only persistence input on the mean head | yes — off, the parameter is not built |
| `teb_vae/lag_attn/nets/encoders.py` | the availability adapter masks its own warm-up region, and the local convolution stem | yes — a gated model's positions there are already exactly zero, and the stem is built only when selected |
| `teb_vae/lag_attn_rws/task.py` | the `anchors=` keyword at the shared call site, and the `_added_metrics` hook | yes — the hook returns `{}` |
| `teb_vae/lag_attn_rws/plotting.py` | the diagnostic callback resolves three page seams off the task | yes — an absent seam resolves to the shipped builder |
| `teb_vae/lag_attn_rws/sample_page.py` | the forecast-row seam | yes |
| `teb_vae/lag_attn_rws/input_budget.py` | the input-panel seam | yes |
| `teb_vae/lag_attn_rws/trainer.py` | the second checkpoint criterion behind one optional monitor key | yes — absent, no second callback is built |
| `hdf5_dataset/hdf5_dataset.py` | `read_causal_warmup`, a public numpy-only boundary reader | yes — nothing else calls it |
| `scripts/make_tiny_shard.py` | the causal variant of the committed fixture, and the planted-delay variant | yes |
| `teb_vae/lag_attn/tests/fixtures/tiny_shard_causal.hdf5` | the committed causal fixture | yes |
| `teb_vae/lag_attn/tests/fixtures/tiny_stats_causal.hdf5` | its statistics | yes |
| seven copies of `tests/test_nets_are_framework_free.py` | this package in each `_PACKAGES` tuple | yes |

Every "inert" claim in the last column is a tested one: `scripts/print_objective_metrics.py` prints
every objective metric of every shipped forecaster in about a minute, and its output is unchanged by
each of the edits above.

### Go/no-go while a run is in flight

| Signal | Threshold | Action |
|---|---|---|
| `train/total_loss` non-finite | any row | stop; the breaker's non-finite guard should have caught it first |
| `train/spike_skipped` sustained above zero | | stop; the margin is mis-tuned for this objective |
| `target_warm_frac` other than `1.0` | any row | stop; the geometry broke, and every nat on the page is over the wrong block |
| `anchors_per_sample` outside its band | any row | stop; the tiling is not the one the configuration states |
| `check_run.py` criterion 6 FAILs | at launch | stop; the run cannot be attributed to an arm afterwards, and relaunching costs less than the run does |
| `train/grad_clip_frac` near 1 | | investigate; the clip is binding on ordinary steps rather than on blow-ups |

### When a stop fires

Record the epoch, the row that fired it, and the run directory. A stopped run is a result about the
configuration, not a run to be quietly relaunched at other settings.

---

## Parameter budget — measured

| Model | Configuration | Total | Decoder width | Block |
|---|---|---:|---:|---:|
| `lag_attn_cfs` | shipped: local K/V, prior clock, persistence residual, weighted horizon, flat bias, both clocks | 4,655,987 | 98 | 2940 |
| `lag_attn_cfs` | shipped but one shared clock | 4,658,035 | 98 | 2940 |
| `lag_attn_cfs` | shipped but unaligned | 4,658,803 | 98 | 2940 |
| `lag_attn_cfs` | shipped but `lag_kv_source: encoder` | 5,163,866 | 98 | 2940 |
| `lag_attn_cfs` | shipped but `lag_kv_source: adapter` | 3,851,635 | 98 | 2940 |
| `lag_attn_cfs` | every switch off, one shared clock — bitwise the previous architecture | 5,146,334 | 98 | 2940 |
| `lag_attn_cfs` | every switch off, unaligned | 5,147,102 | 98 | 2940 |
| `lag_attn_cfs` | every switch off, ungated | 5,130,598 | 102 | 3060 |
| `lag_attn_fs` | reach budget 120 s | 5,126,326 | 78 | 2340 |

**The five architecture switches cost −488,299 at fixed clocks**, and it factorises exactly: the deep
source encoder is not built (−1,312,231), the local K/V stem is (+804,352), the prior's clock
projection and its own norm are (+16,640 = `128 * 128` plus `2 * 128`), and the persistence weight is
(+2,940 = `30 * 98`). The horizon weight is a non-persistent buffer and the flat bias seed reuses an
existing parameter, so both cost zero.

**The second alignment clock costs −2,048**, eight source channels off two 128-wide linears
(`-8 * 128 * 2`), with the target stream contributing nothing — which is the whole point of the
second key. **The alignment itself costs −768** and factorises exactly: the source adapter loses four
channels from two 128-wide linears (`-4 * 128 * 2`) because those channels are slower than the
reference, and both adapters gain a start-of-record vector (`+2 * 128`) because the shifted warm-up
no longer starts at zero on either stream. Ungated means the whole guard, the warm-up mask and both
clocks together — a shift vector is positional over the survivors, so a stream with no keep-index has
no width for one to be positional against.

**The comparison against `lag_attn_fs` is read on the off-state row**, deliberately: that cell is
two-sided and never takes the new keys, so at the shipped configuration the difference between the
two models is dominated by mechanisms one of them does not have. There the target-axis delta is
**+20,008** in two terms:

| Term | Value | What it is |
|---|---:|---|
| the decoder's output head | +10,280 = `514 * (98 - 78)` | two per-channel output rows plus their biases |
| the two input adapters | +9,728 | `128 * (98 - 78)` and `128 * (47 - 29)` on the input linear *and* the availability projection |

`horizon_depth` is **not** a term — it stays at the sibling's 4 — so a delta that does not decompose
into exactly these two means something else moved.

**The guard costs parameters here and saves them on the two-sided sibling**, +13,568 at the shipped
configuration against −9,662, and both are right: the warm-up budget drops 4 target channels of 102
and the alignment 12 source channels of 51, so the machinery the guard adds — the two availability
projections at `128 * 98` and `128 * 39` plus the two start embeddings at `2 * 128` — dominates the
−2,056 off the decoder head, the −512 and −1,536 off the two input linears and the −120 off the
persistence weight; the reach budget drops 31 of 109, so the narrowing dominates instead. On the
off-state row the same identity is the six-term one at a 47-channel source and sums to +15,736.

`DESIGN.md` §13 carries the same tables; `tests/test_docs.py` measures every total in both documents
by constructing the models rather than comparing against literals, so a change to a shared component
re-costs the tables instead of failing an unrelated assertion.

---

## The loss-scale constants — measured

Both moving constants were re-derived at this objective's own scale rather than carried across,
because both are stated in nats of the summed block and this cell changes both the block (2940
against 2340) and the anchor count (~4.6 against ~240).

| Constant | Value | Statistic it was set from | Measured |
|---|---:|---|---|
| `gradient_clip_val` | 15000.0 | smallest round value above the pre-clip gradient norm's q99, measured at 14181 on the fixture at the shipped batch | |
| `additive_margin` | 9.0e+3 | above the worst excursion above the breaker's own EMA in the noisiest regime the fixture produces (5090 here, 3928 on the conv-Transformer cell), and held equal to that cell's | |
| `ema_floor` | 1.0e+9 | above any loss the objective can reach, which switches the relative test off | |
| `horizon_embed_std` | 0.8 | the post-initialisation correlation between two horizon tokens | |

The last two did **not** move, and that is a measurement rather than an inheritance: `ema_floor` is a
switch rather than a scale, and the horizon-token correlation is a per-pair quantity that does not
depend on how many tokens there are.

**The horizon weighting does not move any of the four, and that is measured rather than assumed.**
The weight is renormalised to `sum(w) = H`, so it redistributes the block's magnitude rather than
rescaling it: on a symmetric batch at the shipped geometry the weighted block sum is **4543.25**
against the uniform **4544.01**, a ratio of 0.99983. Without the renormalisation the same weight
would have shrunk the block by **1.807x** — to 2515.17 — against an unmoved KL, which is exactly
where `gradient_clip_val`, `additive_margin` and $\beta$'s standing against the reconstruction would
have gone out of date with nothing saying so. At the shipped half-life of 15.0 steps the resolved
weights run 1.8063 at the first horizon step to 0.4729 at the last, a 3.82x spread that sums to 30.

**The persistence residual does not move them either, by construction**: it enters the mean head of
both decoder invocations identically, so it changes the *level* of `nll_full_block` and
`nll_base_block` together and leaves `pred_gap` exactly where it was. Re-derive `gradient_clip_val`
from the headline run's own `train/grad_norm` column once it exists, and `additive_margin` from that
run's `main_loss` column rather than from the fixture — the fixture's four in-sample windows are a
thinner tail than a production run's.

---

## Distributed smoke, memory and throughput

| Quantity | Definition | Dev box | Production box |
|---|---|---|---|
| Peak memory per rank | `torch.cuda.max_memory_allocated` at the end of epoch 1 | | |
| Anchor-axis tensors | `4 x (B, A_max, H, C_keep)` plus the target | | |
| Steps per second | optimizer steps, epoch 2 onward | | |
| Wall clock per epoch | | | |

The anchor tiling reduces the five anchor-axis tensors by roughly 14x at the shipped stride: at
`B = 128` each is about 8 MB rather than 114 MB. The batch size stays at the two-sided sibling's 128
deliberately, so the comparison is not confounded by a different gradient-noise scale.

**The revised architecture is cheaper, not dearer**: the local K/V path removes about 508K parameters
per rank on this cell and the additions are a few thousand, while the prior clock costs one extra
batch-1 source encode per forward and the persistence residual one gather and one multiply. If the
peak memory does not fall against the previous architecture's record, something is holding the deep
source encoder that should not exist.

**Escalation order if a rank runs out of memory:** lower `batch_size`, then raise
`accumulate_grad_batches` to hold the effective batch, then — and only with the confound recorded —
consider `attention_grad_checkpoint`.

---

## Headline baseline

| Metric | Train | Val | Note |
|---|---|---|---|
| `total_loss` | | | |
| `main_loss` | | | |
| `nll_full_block` | | | nats per anchor over 2940 coefficients, under both objective weights |
| `nll_base_block` | | | |
| `pred_gap` | | | sign is not a criterion |
| `source_conditioned_kl_raw` | | | |
| `anchor_coverage_frac` | | | |
| epoch of the `val/total_loss` checkpoint | | | |
| epoch of the `val/nll_full_block` checkpoint | | | |
| epoch early stopping fired | | | against the 5000-epoch budget and the 709-epoch record |

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
determined by signal the model has legitimately observed. On this cell that share is smaller than on
the two-sided ones — a one-sided kernel's support lies entirely behind its own step — which is the
point of the whole family, and these four columns are how it is read.

| Metric | Value | Reading |
|---|---|---|
| `pred_gap_tau_first` | | the first horizon step, which the horizon weight now weights most |
| `pred_gap_tau_last` | | the last, which no observed history reaches and the horizon weight weights least |
| `pred_gap_st` | | the first stored block |
| `pred_gap_ph` | | the second |

All four are computed under **both** objective weights, because each is a partial sum of the
`pred_gap` printed beside it. The first two are the pair the horizon weighting was introduced to
move: if it worked, the near steps improve and the far steps do not degrade by more than the weight
removed from them.

---

## The warm-up and the tiling

The two geometry guards, and the two readouts that size the source compromise.

| Metric | Expected | Measured | Reading |
|---|---|---|---|
| `target_warm_frac` | exactly `1.0` | | a stamped constant; any other value voids the run |
| `anchors_per_sample` (train) | `[10, 11]` | | the tiling firing at the configured stride |
| `anchors_per_sample` (val) | `51` | | the dense evaluation set |
| `source_lag_warmth_frac_st` | below 1.0, and above the one-clock arm's | | a small value is expected, not a failure; exactly 1.0000 means a pre-fix `W'`-only pattern |
| `source_lag_warmth_frac_ph` | below 1.0 | | the block with the problem, and the block the source clock costs six channels |
| `pred_gap_warm_lo` / `_mid` / `_hi` | sums to `pred_gap_st + pred_gap_ph` | | whether slow channels forecast differently from fast ones |
| `pred_gap_novel_lo` / `_mid` / `_hi` | sums to the same total | | how much of the block score is a forecast rather than an inversion of history |
| `kld_source_null` | a large fraction of the raw KL | | close to `source_conditioned_kl_raw` means the readout is a clock; the informative quantity is the difference, against the 0.15-nat margin |

---

## What the production runs still owe

- Every empty cell above.
- The three per-run figures, attached, with any disagreement against the metrics recorded.
- The arm comparison, read under *The comparison protocol* and on no other quantity.
- The three deferred decisions: whether the two tertile families earn their place, whether the beta
  pair carried across from the two-sided sibling holds at this block and anchor count, and whether
  the horizon half-life of 15 steps is the right one. All three are judgements to be taken **with
  the number that drove them recorded**, not threshold gates.

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
  ($[340.6, 584.6]$ s at the retained source clock and `max_lag: 90`, at the realised offset
  $\kappa(\tau^u_{\mathrm{ref}} - \tau_{\min}) = 240.6$ s; $[374.9, 618.9]$ s was the unscaled
  figure): the 20–60 s proximate band
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
it is comparable to the conv-Transformer causal cell, which carries the same block since the same
day. The `anchors_per_sample` guard bands become $[31, 32]$ train / $156$ val (`anchor_stride: 5`, dense span $[134, 290)$). The two loss-scale constants were **scaled, not measured**:
`gradient_clip_val` $4000.0$ and `additive_margin` $2.2 \times 10^{3}$, both from the recorded
$H = 30$ figures by the block ratio $760/2940$; the "Measured" column above does not describe them,
and both are to be re-derived from the headline run's own `train/grad_norm` and `main_loss` columns.
`horizon_weight_halflife_steps` is $5.0$ (the $H/2$ rule; weights $1.7260 \to 0.4957$).

**The encoder edge holds.** Both cfs cells carry the same horizon, anchor stride, horizon half-life
and additive margin, so they optimise the same criterion over the same block and anchor count and the
level columns of the cross-cell table remain readable across them; both cells' `test_config_load.py`
pin the four leaves. The target edge now differs in the horizon as well as the block. `DESIGN.md`'s
amendment of the same date carries the full derivation.
