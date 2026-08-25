# Figure guide

One entry per PDF the evaluation emits: what it shows, what its axes are, and how it is misread. Each is written under the filename the run produces, inside the analysis subdirectory named in its heading. `EVAL.md` beside this file is everything that is not a figure.

Four rules apply to every figure here.

**An empty panel is a statement, not a bug.** A panel that found no finite value draws the words `no finite values` rather than an empty frame, and that means the analysis measured nothing rather than that the plotting failed.

**Every unit is the loader's $z$ unit, labelled `normalised`, and there is no clinical unit anywhere.** The target is 98 wavelet-modulus and phase-harmonic coefficients, which have no clinical scale to convert to; inverting the per-channel statistics would put them on scales spanning orders of magnitude and destroy every pooled statistic and shared colour bar. So an error axis here is in standard deviations of a coefficient, and a number on it cannot be compared with a number from the raw-target cells.

**Every lag axis is stored-coefficient time.** The seconds figure shown is the compensated lag $\tau = 4(\ell + \delta)$, which adds back the model's own input delay $\delta$ and nothing else. It is **not** a physiological delay: the stored coefficients come from a one-sided bank whose composed group delay reaches 791 s — the same order as the 364 s lag search itself — and that correction is per channel *pair* while the lag map is per head over a pooled source state. Every lag-resolved figure prints the caveat under it, drawn from one constant so no two figures can quote different numbers.

**The channel alignment narrows that refusal without lifting it here.** Once every input channel is shifted onto one reference, the correction stops being per channel *pair* and becomes a single constant plus the $\pm 16\%$ intra-band dispersion — which is why the two raw-target cells can label a lag axis in physical seconds and this one still cannot. The obstruction that remains is the target: what is forecast here is a *coefficient* block, so $\tau^y_{\mathrm{ref}}$ is nonzero and a shifted target tile is arithmetically impossible at its first horizon element. A lag on these figures is therefore a lag between two coefficient epochs, and the run's own `source_reference_delay_s` is recorded beside it rather than folded into the axis.

**The order is clinical and the colour is the severity**, on every figure that resolves a quantity by cohort. Left to right in ascending severity: `healthy`, `acidosis`, `hie` on the class axis, and `healthy_no_bg_no_cs`, `healthy_no_bg_cs`, `healthy_bg_no_cs`, `healthy_bg_cs`, `acidosis_no_cs`, `acidosis_cs`, `hie_no_cs`, `hie_cs` on the subgroup axis. Green for healthy, amber for acidosis, red for HIE, each subgroup a shade of its class. One function decides both — `cohort.ordered_groups` — and every CSV row order reads it too, so a table can be read against the figure beside it row for row. A cohort the order does not know is drawn **after** the ones it does rather than dropped. Two consequences: this palette is **this evaluation's**, so a figure here and a training-callback figure of the same cohort are different colours and are reconciled by legend rather than by hue; and a colour ordering is not a result — `cross_subgroup` is the analysis that says whether a visible difference survives being asked properly.

---

# Terminology

Every name that appears on an axis, in a legend, in a panel title or in a CSV column beside a figure. Read this once and every figure below is readable without guessing.

## What the model does, in one paragraph

The recording is a fetal heart-rate trace and a uterine-pressure trace, both sampled at 4 Hz, but the model never sees either directly. Both have been passed through a **strictly one-sided** scattering and phase-harmonic transform, so what the model reads and forecasts are 4-second-step **coefficients**: 102 target channels from the heart rate (36 scattering + 66 phase-harmonic) and 51 source channels from the pressure. The warm-up budget drops the four slowest target channels, leaving **98** the decoder emits.

At each anchor $t$ the model forecasts the next **30 steps** — two minutes — of all 98 channels, and it does this **twice**: the **base** forecast, from a belief built out of the target history alone; and the **full** forecast, from a belief that also read the source history. Everything this evaluation reports is some version of *how much better was the second one*.

Two things about the geometry are worth carrying:

- **The forecast cannot start at step 0.** A one-sided filter's output depends on assumed pre-recording history until its warm-up has passed, and the floor is the **larger of two requirements** rather than one. Every *scored target* channel has to be honest at the earliest step a forecast reads, which asks for $B - 1 = 133$ with $B$ the slowest kept channel's wait; and, because the inputs are gathered onto one clock, every *input* channel is honest only from $W'_c + d_c$, which asks for $\max_c(W'_c + d_c) = 134$ on both streams. $F = \max(133, 134) = 134$ is what ships — the input-warmth half binds, at exactly $B$ — and 136 anchors are decoded per segment.
- **Because $F = 134$ and the furthest searched lag is 90, every lag exists at every scored anchor.** The truncation corrections the raw cells need are inert here, and the pipeline *measures* that rather than assuming it.

## How to read a name

| Piece | Means |
|---|---|
| `base` | the target-only branch (prior belief) |
| `full` | the source-conditioned branch (posterior belief) |
| `shuffled` | the negative control: the *same* model fed **another recording's** source |
| `null` | the second control: the same model fed a **zeroed** source stream |
| `mc_` prefix | **Monte Carlo**: averaged over $K$ latent draws instead of one. The headline form. |
| `_block` suffix | summed over the whole $15 \times 98 = 2940$-coefficient forecast block, then averaged over anchors |
| `_raw` suffix (on a KL) | **unfloored** — no free-bits floor applied. The only form readable as a rate. |
| `_sq` suffix | left **unrooted** (a mean square). Its `_rms` partner is the rooted version. |

## The quantities

### Scores — how good was a forecast

A **score** here is always a negative log-likelihood: **lower is better**, and it is not bounded below by zero.

| Name | Plain meaning | Units |
|---|---|---|
| `nll_base_block` | how badly the target-only forecast fit the truth | nats per anchor |
| `nll_full_block` | the same for the source-conditioned forecast | nats per anchor |
| `mc_nll_base_block`, `mc_nll_full_block` | the same two, averaged over $K$ latent draws — the **headline** pair | nats per anchor |
| `mc_nll_shuffled_block` | the score when fed a **stranger's** source; the negative control | nats per anchor |
| `nll_persistence_block` | baseline: hold the last **observed** coefficient vector for the whole window | nats per anchor |
| `nll_climatology_block` | baseline: predict the population mean, which after z-scoring is exactly $0$ | nats per anchor |
| `nll_segment_mean_block` | baseline: predict this segment's own per-channel mean | nats per anchor |
| `nll_oracle_block` | an evaluation-only decoder reading the encoder state **directly**, bypassing the latent bottleneck | nats per anchor |

**"Nats per anchor" is the unit to internalise.** One score is a sum over all 2940 coefficients of one forecast block, then averaged across the anchors that were scored. So it is a *large* number under every predictor including a perfect one, because it is 2940 terms added up. Its absolute size says nothing; only differences and ratios between predictors are readable. Dividing by 2940 gives a per-coefficient figure, but that is a flat rescale rather than a mean over the coefficients actually scored, so it under-reports on any anchor with masked forecast steps.

### Gaps — what the source added

| Name | Plain meaning | Units |
|---|---|---|
| `pred_gap` | $D_{\mathrm{base}} - D_{\mathrm{full}}$ on the **single-draw** path. Positive = the source helped. | nats per anchor |
| `mc_pred_gap` | the same difference on the marginalised scores. **This is the headline coupling number.** | nats per anchor |
| `pred_gap_warm_lo` / `_mid` / `_hi` | the same gap restricted to each warm-up tertile of the 98 kept channels. The three **sum to** `pred_gap`. | nats per anchor |
| `pred_gap_<band>` | the same gap restricted to one frequency band of the target coefficient. These sum to `pred_gap` too. | nats per anchor |
| `delta_suff_nats` | $D_{\mathrm{base}} - D_{\mathrm{oracle}}$: what the latent bottleneck *costs* the forecast | nats per anchor |
| `mse_skill` | $1 - \mathrm{MSE}_{\mathrm{model}}/\mathrm{MSE}_{\mathrm{baseline}}$. $1$ = perfect, $0$ = no better than the baseline, negative = worse. | dimensionless |
| `advantage_nats_per_anchor` | the NLL-space analogue of skill, and a **difference**, not $1 -$ a ratio — a log score has no natural zero | nats per anchor |
| `pred_gap_rmse_pct`, `pred_gap_mse_pct` | the percentage of the target-only branch's point-forecast error the source removed, rooted and unrooted. **Scale-free.** | percent |
| `pred_gap_mc_likelihood_pct` | $100(e^{\texttt{mc\_pred\_gap}/2940} - 1)$: the extra probability density the source-informed forecast puts on each observed coefficient. **Budget-local.** | percent |

`pred_gap` and `mc_pred_gap` are two estimators of *one* quantity, not two findings. Wherever both are drawn, their difference is the price of the Monte Carlo marginalisation.

None of the three percentages is `pred_gap` divided by a block score. $D_{\mathrm{base}}$ is a negative log *density* summed over 2940 coefficients: it has no natural zero and is legitimately negative for a sharp forecast, so that ratio would change sign with its own denominator. The two spaces that do have a natural zero are used instead. The likelihood one exists only under `gaussian_nll`, and it divides by the **fixed** 2940 — which is what the warm-up budget decided, so two arms of this model at two budgets divide by two different numbers.

### The KL family — how much the belief moved

$K_t = \mathrm{KL}(q_t \Vert p_t)$ measures how far reading the source moved the belief at anchor $t$. Zero means the source changed nothing.

| Name | Plain meaning | Units |
|---|---|---|
| `source_conditioned_kl_raw` | $K_t$ averaged over the scored anchors. **The** KL readout. | nats per anchor |
| `source_conditioned_kl_shuffled_raw` | the same under a **stranger's** source — the specificity control | nats per anchor |
| `kld_source_null` | the same under a **zeroed** source — the availability-clock control | nats per anchor |
| `coupling_minus_clock_nats` | their difference: the part of the coupling attributable to source *variation* | nats per anchor |
| `kld_per_t` | $K_t$ *before* averaging: one value per anchor | nats per anchor |
| `kld_per_dim`, `kld_per_head` | $K_t$ split across latent dimensions or attention heads; each sums back to the total | nats per anchor |
| `source_kl_lag_map` | $K_t$ split across the 91 lags; sums back to the total | nats per anchor |
| `active_dims`, `top_dimension_share` | how many latent dimensions carry more than a threshold, and the largest one's share | count, fraction |

Four warnings this pipeline repeats because each has cost a result somewhere:

1. **Only the `_raw` (unfloored) KL is a rate.** A floored variant exceeds it by construction and would hide a collapsed source pathway.
2. **The KL is inflated whenever the prior variance sits on its clamp**, because $K_t$ carries $(\mu^q - \mu^p)^2/\sigma_p^2$. Check `prior_variance_not_pinned` before quoting a KL. `pred_gap` is immune to this, which is why the two travel together.
3. **A bigger KL under a stranger's source is healthy, not broken.** A stranger's source is out of distribution, so it moves the posterior *more*. Specificity is judged on the **scores**, never on the KL.
4. **Part of the KL may be a clock rather than a coupling.** The source availability pattern is a deterministic function of $t$ that every row of a batch shares, so no permutation of rows can remove it. `kld_source_null` is the only readout that can, and the difference is the number to quote.

### Warm-up, bands and geometry

| Name | Plain meaning |
|---|---|
| `causal_warmup_steps` | per channel: the leading delay, in 4-second steps, before that channel's coefficients are honest |
| `causal_delay_s` | per channel: the composed one-sided group delay, in seconds. What makes the lag axis coefficient time. |
| `target_warm_frac` | the fraction of scored anchors at which every kept target channel is warm. Must be exactly $1.0$. |
| `anchors_per_sample` | anchors decoded per segment. Must be exactly $136$ at the dense set. |
| `source_lag_warmth_frac_st` / `_ph` | the fraction of attention mass landing on lags at which that stored source block is warm. **A small value is the expected finding.** |
| band | one of `slow_baseline`, `deceleration`, `variability`, `beat_to_beat`, `unknown` — the band of the **analysing filter** that produced the coefficient, not a bin of the forecast's own spectrum |
| `unknown` | a channel whose centre frequency is not recoverable, because no selected phase-harmonic pair named its filter. Never bucketed into a neighbour. |

## The shipped numbers

| Symbol | Value | What it is |
|---|---|---|
| $T$ | 300 | steps per segment |
| $H$ | 30 | horizon steps = 2 minutes |
| $F$ | 134 | anchor floor: nothing below it is decoded at all; $\max(B - 1, \max_c(W'_c + d_c))$ |
| anchors | 136 | decoded per segment, at the dense set |
| $C_{\mathrm{keep}}$ | 98 | target channels the warm-up budget kept, of 102 declared |
| $H \cdot C_{\mathrm{keep}}$ | 2940 | coefficients in one forecast block |
| $d_z$ | 64 | latent dimensions |
| $L$ | 91 | lag bins = ~6 minutes of source history |
| $M$ | 4 | attention heads |
| lag support margin | 44 | $F - (L-1) - \texttt{lag\_floor}$; $\ge 0$ means no anchor has truncated lag support |

---

## The grouped variants: `<stem>_by_clinical_class.pdf` and `<stem>_by_subgroup.pdf`

**In plain terms.** "Does this number differ between the clinical groups?" Take any per-recording number the pipeline computes, split the recordings by class (or by subgroup), and draw one violin per group.

**The mark inside each violin** is the ordinary five-number summary and is the same on every violin in this pipeline: the heavy bar is the middle half of the cohort's recordings ($Q_1$ to $Q_3$), the hairline runs to the furthest recording within $1.5$ inter-quartile ranges of that bar, and the white dot is the median. A recording past the end of the hairline is not omitted — it is in the violin body, which is drawn between the cohort's own extremes.

These have no entry of their own because they are not one figure. Every analysis that writes a per-recording table declares it, and the **runner** fans one violin figure per cohort axis over whatever was declared, so the set grows with the analyses and each is named after the table it resolves.

**Axes.** One row per metric; the metric's own units on the vertical axis, cohorts across in ascending severity. The CSV beside each figure carries its rows in that same order.

**How they are misread.** Every violin holds one value per **recording**, not per segment, so a cohort with eight recordings is eight points however many segments they contributed — the width of a violin is not evidence. A cohort split producing fewer than two groups emits **no figure at all** and records a skip. And a visible separation here is not a result: `cross_subgroup` is the analysis that says which separations survive being asked properly.

---

## `forecast/baseline_comparison.pdf`

**In plain terms.** *"Is this forecast any good at all?"* — the first question, before any question about the source. The model is put in a line-up with three predictors so stupid that beating them is the minimum bar: hold the last observed coefficient vector, predict the population average, predict this segment's own average. The top panel scores all five; the bottom turns each comparison into a number where **1 is perfect, 0 is "no better than the stupid predictor" and negative is worse than it**.

**What it shows.** Top: the per-recording block score of every predictor — the two model branches and the three trivial baselines — as violins, in nats per anchor, lower better. Bottom: the squared-error skill of each branch against each baseline, with a percentile bootstrap interval over **recordings**.

**Axes.** Top: nats per anchor, one violin per predictor. Bottom: skill, dimensionless, zero marked; whiskers are asymmetric because a percentile interval is not symmetric about its point estimate.

**Terms on this figure.** `base`, `full`, then `persistence`, `climatology`, `segment_mean`; each violin is `nll_<name>_block` across recordings. The bars are `mse_skill` with `mse_skill_lo` / `_hi` as whiskers.

- **persistence** carries forward the last *observed* step, not the last one: `weight` is the only trustworthy validity signal here, since the coefficients carry no sentinel of their own, and carrying an invalid step forward would measure the gap.
- **climatology** is exactly $0$ per channel, which is the z-scored population mean — and it is a meaningful baseline only because the normalisation statistics were accumulated *excluding* the warm-up region.
- The baselines are scored at a fixed `BASELINE_LOGVAR = 0.0`, recorded rather than fitted: a point predictor has no variance of its own, and the whole score would otherwise be decided by whatever $\sigma$ it was handed.

**How it is misread.** The block score is a *sum over 2940 coefficients*, so it is large under every predictor and its scale says nothing about the model. Only the comparison is readable. The skill drawn here is the MSE-space one; the NLL-space column beside it in the CSV is a **difference** in nats, not $1 -$ a ratio.

## `forecast/anchor_profile.pdf`

**In plain terms.** *"Does the model do better early in a segment or late?"* Walk along the segment from left to right and, at each 4-second anchor, average the score over every segment that had an anchor there.

**What it shows.** The two block scores and `pred_gap` against position in the segment, averaged over every segment that scored that anchor.

**Axes.** Anchor index in decimated (4 s) steps from the start of the trimmed segment; nats per anchor.

**How it is misread.** **The profile starts at 134, and that is the geometry rather than a finding** — and it is the opposite shape from the raw-target cells, where the curve begins at the model's own 30-step warm-up and droops through a truncated-lag region. Here nothing below the anchor floor is decoded at all, so there is no droop to discount and no truncated region inside the profile. The last $H$ anchors are still never scored, because their forecast window would run past the end of the segment. A reader expecting the raw cells' shape and finding a curve that begins two-fifths of the way in is looking at the anchor floor.

## `forecast/horizon_skill.pdf`

**In plain terms.** *"How far ahead can it actually see?"* A single block score lumps the whole minute into one number. This unpacks it: how good is the forecast 4 seconds ahead, 30 seconds ahead, a minute ahead. Expect the error to rise with lead time. The gap panel is where the source's contribution lives.

**What it shows.** $D_{\mathrm{base}}(\tau)$ and $D_{\mathrm{full}}(\tau)$, their gap, and each branch's error, against lead time over the 15 horizon steps.

**Axes.** Lead time in **seconds**; nats **per horizon step**, so the 15 values sum back to the block.

**How it is misread.** The curve is computed on the **single-draw** path and says so in its title: the Monte Carlo marginalisation does not commute with the sum over $\tau$, so a marginalised curve would not sum back to the marginalised headline. Horizon step $0$ is $4$ s ahead, not $0$ — the anchor's own step is the past, not the forecast.

## `forecast/forecast_overlay.pdf`

**In plain terms.** *"What does one forecast actually look like?"* Everything else in this pipeline is an average over thousands of anchors. This is a single one-minute forecast drawn against what actually happened, so the shape can be seen rather than a number.

**What it shows.** One anchor's truth and both branch means, for **three kept channels** drawn against lead time. Three channels rather than one line, because what is forecast is an $H \times C_{\mathrm{keep}}$ block and there is no single trace to overlay.

**Axes.** Lead time in seconds within this one block; the coefficient's value in $z$ units.

**How it is misread.** It is **one anchor of one retained recording**, drawn from a seeded stratified draw, not a representative case. Retention is opt-in (`eval_config.caps.waveforms`), so a run that did not ask for it emits no such figure at all; the absence is silence, not failure. And the three channels are a sample of 98 — a channel that tracks well says nothing about the ones not drawn.

## `coupling/pred_gap_distribution.pdf`

**In plain terms.** **This is the headline figure of the whole evaluation.** *"Did reading the uterine pressure make the forecast better, and for how many deliveries?"* Each recording contributes one number: how many nats the source-informed forecast beat the target-only one by. **Zero is the null.**

**What it shows.** Top: the distribution of `mc_pred_gap` over **recordings**, with zero marked and the bootstrap interval on the *mean* shaded. Bottom: the two estimators side by side, each under its own name.

**Axes.** Nats per anchor; the histogram's height is a count of recordings, not of segments.

**How it is misread.** Three ways. The shaded band is the interval on the **mean**, not the range of the data. The two violins are two *estimators of the same quantity* and their difference is the cost of the marginalisation, not a second finding. And the unit is one recording, so a recording that scored no anchors is absent rather than at zero — the $n$ in the title is the count actually available.

A fourth is this cell's own: **a positive gap here is not yet a source finding.** Read `source_null/source_null_difference.pdf` beside it, because part of a coupling readout can be an availability clock that no control on this figure can see.

## `coupling/pred_gap_percent.pdf`

**In plain terms.** *"By what percentage did the source improve the forecast?"* The figure beside this one answers in nats, which tells a reader nothing about proportion. The top two panels are about the **error**; the bottom is about the **likelihood**. **Zero is the null on all three.**

**What it shows.** Top: the distribution of `pred_gap_rmse_pct` over recordings, zero marked, the interval on the mean shaded. Middle: `pred_gap_rmse_pct` and `pred_gap_mse_pct` side by side — the same ratio under a root, so the mean-square figure is the larger wherever both are positive. Bottom: `pred_gap_mc_likelihood_pct`, **empty under an `mse` checkpoint**, where it is undefined rather than zero.

**Axes.** Percent on every panel; the histogram's height is a count of recordings.

**How it is misread.** The bottom panel is **budget-local**: it divides by $H \cdot C_{\mathrm{keep}} = 2940$, and $C_{\mathrm{keep}}$ is whatever the warm-up budget decided, so it cannot be compared across two arms at two budgets — nor, for that matter, against any other cell of the grid. The two error-space panels are scale-free and do not carry that caveat.

## `latent/kl_spectrum.pdf`

**In plain terms.** *"Is the model using its whole memory, or one corner of it?"* The latent is 64 numbers per anchor; this shows how much of the source information each one carries.

**What it shows.** The per-dimension KL, sorted, with the active-dimension count and the top dimension's share annotated.

**Axes.** Latent dimension rank; nats per anchor.

**How it is misread.** A tall first bar is not automatically a fault — one dimension carrying most of a small total is a different finding from one carrying most of a large one, and the total is on the panel for that reason. What *is* a fault is a spectrum read without checking `prior_variance_not_pinned` first: a prior variance on its clamp multiplies every bar by an arbitrary factor while every decoder-side diagnostic stays healthy.

## `lag_kl/lag_kl_profile.pdf`

**In plain terms.** *"How far back in the pressure history did the model find what it used?"* The KL at each anchor is split across the 91 lags the model attends over, so the profile says where in the past the source informed the future.

**What it shows.** The per-lag KL attribution in its raw, support-corrected and untruncated forms, with the peak and its width marked.

**Axes.** Lag in 4-second steps, with the compensated seconds axis beside it; nats per anchor.

**How it is misread.** Two ways, and the caveat printed under the figure states the second. **The three profiles coincide here, and that is a measurement rather than a redundancy**: at this anchor floor every lag exists at every scored anchor, so the support correction and the untruncated recomputation are inert — and an arm that lowered the floor would separate them again, which is why all three are still drawn. And **a peak is a position in stored-coefficient time**, not a physiological delay: the composed group delay is uncorrected and reaches the same order as the search window itself.

## `attention/attention_profile.pdf`

**In plain terms.** *"Where is each attention head looking?"* Four heads, four curves, because the posterior is head-structured — latent group $m$ is written by head $m$ alone — and averaging them before profiling would make four heads at four delays indistinguishable from one head attending everywhere.

**What it shows.** The per-head attention profile over lags, with the head-averaged curve beside them **named as such**, and each head's entropy against the ceiling it can actually reach.

**Axes.** Lag in steps, with compensated seconds beside it; attention probability.

**How it is misread.** The entropy is quoted against the **attainable** ceiling $\operatorname{mean}_t \log \min(t+1, L)$, which at this floor equals $\log L$ exactly — measured, not substituted. Reading a head's entropy against a hand-computed $\log L$ on an arm with a lower floor would report a model attending uniformly over what exists as increasingly concentrated. And the entropy is taken per anchor and then averaged, never as the entropy of the averaged profile: a mixture's entropy is at least the mean of the entropies mixed, so the second reports a model whose lag focus *shifts* as one that has none.

## `attention/lag_heatmap.pdf`

**In plain terms.** *"Did the model look at the same place all the way through the recording, or did it move?"* The profile figure averages over anchors; this one does not.

**What it shows.** Attention mass as a heat map over (anchor, lag) for a retained sample, per head.

**Axes.** Anchor index across, lag down; colour is attention probability on a shared scale.

**How it is misread.** It is **one retained segment**, drawn only where `eval_config.caps.attention` asked for retention — an absent figure is silence, not failure. The colour scale is shared across heads so they can be compared, which means a head with little structure looks flat rather than noisy. The vertical axis is the same coefficient-time lag as everywhere else.

## `calibration/pit_reliability.pdf`

**In plain terms.** *"When the model says it is 68% sure, is it right 68% of the time?"* A score can be driven down by shrinking the predicted spread wherever the forecast happens to be right and paying for it elsewhere, and every other figure in this guide would improve while it happened. This is the only one that checks.

**What it shows.** The probability integral transform of the standardised residuals against uniform, and the empirical central coverage at the exact erf nominals.

**Axes.** Nominal probability against realised; the diagonal is perfect calibration.

**How it is misread.** The nominals are $\operatorname{erf}(k/\sqrt{2}) = 0.6827,\ 0.9545,\ 0.9973$ — the two-sigma figure is **not** 0.95, which is $\pm 1.96\sigma$, and the half-point difference reads as a real miscalibration if the wrong nominal is assumed. The unit here is one **coefficient**, not one element of a 4 Hz trace, which is why the count beside it is `n_coefficients`. An `mse` checkpoint emits no such figure at all: its log-variance head was never fitted.

## `calibration/logvar_distribution.pdf`

**In plain terms.** *"Is the model's own uncertainty estimate pinned against a wall?"* The decoder's log-variance is clamped at both ends, and a single mean is equally consistent with a healthy spread and with half the mass sitting on each clamp.

**What it shows.** The distribution of the decoder's log-variance with both clamp bounds marked, and the floor and ceiling fractions annotated.

**Axes.** Log-variance; density.

**How it is misread.** This is the one figure whose reading changes a config value — the analysis states a recommended `logvar_clamp` revision **per coefficient**, which is the axis the objective's block score reduces over, and says *no change* when neither end binds. A recommendation emitted unconditionally would be applied unconditionally.

## `distributions/class_histograms.pdf`

**In plain terms.** *"Is a cohort's higher error a uniform shift, a heavier tail, or a handful of segments the model fails on completely?"* Three distributions with the same mean are three different findings, and every other figure in this pipeline has already reduced each recording to one number before drawing anything.

**What it shows.** Eight metrics, one panel each, drawn at **two levels on the same axes**: a filled density of one value per **segment**, and a median / inter-quartile / range **strip** above it of one value per **recording**.

**Axes.** The metric's own units in $z$ space; density rather than counts, so a cohort contributing ten times the segments does not simply draw a taller curve.

**How it is misread.** **The difference between the two levels is the content, not a redundancy**: a strip far narrower than the density beneath it says most of the visible spread is within-recording variation, and the density is showing many views of the same delivery. This analysis computes **no test on purpose** — consecutive anchors overlap in 14 of their 15 horizon steps, so a per-segment $p$-value is anticonservative by roughly that factor. A separation visible here is a reason to look at `cross_subgroup`, not a result.

## `distributions/subgroup_histograms.pdf`

The same eight metrics resolved by subgroup, **nested rather than flat**: one column per clinical class with that class's subgroups overlaid inside it, so a cell holds at most four curves and they are four tints of one hue. Each cohort is a faint fill under a hairline outline at full opacity, drawn in two passes so every outline sits above every fill — one pass per cohort would leave the first cohort's outline veiled by every later fill, and the first legend entry would be the hardest curve to trace.

## `trajectory/trajectory_profile.pdf`

**In plain terms.** *"Does the coupling change through a recording?"* Two views: within one 20-minute segment, and across a whole delivery assembled from all of its segments.

**What it shows.** The two coupling readouts against time in segment, and against absolute time $t_{\mathrm{abs}} = \mathrm{epoch} + 4t$ across a delivery, with overlapping steps averaged and `n_contributing` travelling beside them.

**Axes.** Anchor index or hours before delivery; nats per anchor.

**How it is misread.** The within-segment panel **starts at the anchor floor**, for the same reason `forecast/anchor_profile.pdf` does. Across a delivery, a gap is drawn as a **break** rather than interpolated, so a discontinuity is missing data rather than a jump — and the averaging of overlapping steps is visible in `n_contributing` rather than inferred.

## `time_to_delivery/time_to_delivery_trajectory.pdf`

**In plain terms.** *"Does the coupling change as delivery approaches, and differently for the sick babies?"* Both readouts binned on a half-hour grid of time before delivery, one series per clinical class.

**What it shows.** `pred_gap` and the unfloored KL against `epoch / 3600`, class-stratified, on per-GUID values — per-GUID *inside* a window as well as across the split.

**Axes.** Hours before delivery (negative, increasing to the right); nats per anchor.

**How it is misread.** Significance is tested **per window**, with Holm across windows as one family; the `pooled` row is flagged `confounded_by_time` and consumed by nothing, because a pooled difference between classes with different recording lengths is a difference in when they were recorded. The bin width is a module constant rather than a config key, for the same reason the significance level is not one.

**Beside it.** `time_to_delivery_windows.pdf` draws the per-recording distribution behind every point of this figure, the Holm-adjusted significance of each window, and the effect size of every class pair that survived — on this same axis. Read that one before quoting a gap between two lines here.

## `time_to_delivery/time_to_delivery_windows.pdf`

**In plain terms.** *"That trajectory has three lines on it — is the gap between them real, and where?"* The trajectory figure draws one number per class per window; this draws what that number was computed from, and the verdict on it, on the same axis.

**What it shows.** Five panels. For each of the two coupling readouts, a violin per (window, clinical class) cell over one value per **recording**, and directly beneath it $-\log_{10}$ of that window's Holm-adjusted $p$ against the $\alpha$ line. Then one heatmap: Cliff's delta for every class pair that survived Holm, in any window, for either readout. Each cell is annotated with the number of recordings behind it; a cell below `MIN_GROUP_SIZE` = 3 recordings, or one whose values are all equal, is drawn as its own points rather than as a density the smoother invented — and those are the same cells the test excludes.

**Axes.** Hours before delivery on the same $0.5$ h grid as the trajectory, inverted so delivery is at the right — on every panel including the heatmap, whose columns run in the same direction as the panels above it; nats per anchor on the violins; $-\log_{10} p$ on the strips.

**How it is misread.** **A bar that is absent and a grey cross at zero are different statements**: no bar means the window was tested and its $p$ came out at or near 1, while a cross means fewer than two classes had enough recordings there and nothing was tested. **Every heatmap row reads less severe against worse**: `healthy vs acidosis`, `healthy vs hie`, `acidosis vs hie`, in that order down the axis and in the cohort order the violins above are drawn in — the pairwise sweep names each pair in the order it receives the classes. A positive Cliff's delta therefore means the *less severe* class runs higher, on every row; reorienting a pair by eye still flips its sign against the number in `time_to_delivery_pairwise.csv`. The correction is across the windows of this clock as one family, which is what makes "eight windows survived" a claim rather than an artefact of having asked twenty-two times; the two readouts are **not** jointly corrected, because they are two readings of the same recordings rather than two hypotheses.

## `second_stage/second_stage_trajectory.pdf`

**In plain terms.** *"Does the coupling change around the moment the second stage of labour begins?"* The same two readouts as the delivery clock, on the other clinical landmark — the one inside labour rather than at its end. Only the recordings the labour-onset table places a second stage for are on this figure.

**What it shows.** `pred_gap` and the unfloored KL against `second_stage_onset / 3600`, class-stratified, as a median with an inter-quartile ribbon over **recordings** — one value per recording per window, averaged over that recording's own segments in it. Each point is annotated with the number of recordings behind it, and a dotted vertical marks the onset itself.

**Axes.** Signed hours from second-stage onset, **negative before onset and positive after**, on the same $0.5$ h grid the delivery clock uses; **not** inverted, because this coordinate reads naturally left to right. Nats per anchor.

**How it is misread.** **The sign is the opposite convention from the delivery clock's**, and the axis label says so: negative is *before* the onset. **The positive side is short by construction** — the second stage begins a couple of hours before delivery, so the windows after onset hold far fewer recordings than those before it, and the annotated $n$ is what says which is which. The population is a **subset**: recordings with no recorded onset are dropped and counted in `second_stage_eligibility.csv`, so this figure describes fewer recordings than any other in the run.

## `second_stage/second_stage_windows.pdf`

**In plain terms.** *"That second-stage trajectory has three lines on it — is the gap between them real, and where?"* The same page the delivery clock draws, on the other landmark.

**What it shows.** Five panels. For each of the two coupling readouts, a violin per (window, clinical class) cell over one value per **recording**, and directly beneath it $-\log_{10}$ of that window's Holm-adjusted $p$ against the $\alpha$ line. Then one heatmap: Cliff's delta for every class pair that survived Holm, in any window, for either readout. Each cell is annotated with the number of recordings behind it; a cell below `MIN_GROUP_SIZE` = 3 recordings, or one whose values are all equal, is drawn as its own points rather than as a density the smoother invented — and those are the same cells the test excludes.

**Axes.** Signed hours from second-stage onset, negative before and positive after, on the same $0.5$ h grid; not inverted, with the onset marked at zero on every panel that carries the clock. Nats per anchor on the violins; $-\log_{10} p$ on the strips.

**How it is misread.** **This clock's Holm family is its own** and is not corrected jointly with `time_to_delivery`'s: the two are different alignments of an overlapping population, so a window significant on one and not the other is a statement about alignment, and a reader quoting both is making two comparisons. **Every heatmap row reads less severe against worse**: `healthy vs acidosis`, `healthy vs hie`, `acidosis vs hie`, in that order down the axis and in the cohort order the violins above are drawn in — the pairwise sweep names each pair in the order it receives the classes. A positive Cliff's delta therefore means the *less severe* class runs higher, on every row; reorienting a pair by eye still flips its sign against the number in `second_stage_pairwise.csv`. **A bar that is absent and a grey cross at zero are different statements**: no bar means the window was tested and its $p$ came out at or near 1, a cross means fewer than two classes had enough recordings there — which on the positive side of this axis is the common case rather than the exception.

## `lag_clocks/lag_time_to_delivery.pdf`

**In plain terms.** *"Does the informative past get closer as delivery approaches — and is it closer for the cohorts that end badly?"* The coupling clocks say how much the source told the model at each point of labour; this says **where in the past** it told it.

**What it shows.** One heatmap per clinical class: lag down, time before delivery across, colour the **share** of the KL attribution sitting in that lag bin, over one profile per **recording** in each window. Then two panels reducing the same thing to a number — the median centre of mass across recordings with its inter-quartile ribbon, once for the attribution and once for the attention profile, with each window's recording count annotated and the median spread as a dashed line beside it.

**Axes.** Hours before delivery on the same $0.5$ h grid as the coupling clocks, inverted so delivery is at the right; lag in seconds of **stored-coefficient time**, lag $0$ at the bottom; colour is a share in $[0, 1]$.

**How it is misread.** **The class panels share one colour scale, and that is what makes them comparable** — three panels each scaled to its own extremes would paint the same colour for three different shares while every colourbar stayed correct. **It is a share, not a magnitude**: every window is normalised to sum to one, so a band moving down means the attribution moved toward the anchor, not that there is more of it — how much there is is what `time_to_delivery_trajectory.pdf` draws. **A centroid is not a peak**: no argmax is reported anywhere in this analysis, because `entmax15` gives a flat profile a confident one; `lag_kl/lag_kl_stratified_peaks.csv` carries the positional reading with its degeneracy verdict. And the lag axis is stored-coefficient time, so a centroid that moves ninety seconds is a shift over the axis the coefficients are stored on rather than a physiological latency.

## `lag_clocks/lag_time_to_delivery_windows.pdf`

**In plain terms.** *"Those centroid lines separate — is the separation real, and in which windows?"* The same page the coupling clocks draw, on the lag centroid instead of on the coupling magnitude.

**What it shows.** Five panels. For each tested readout — the centroid of the attribution and the centroid of the attention — a violin per (window, class) cell over one value per **recording**, and directly beneath it $-\log_{10}$ of that window's Holm-adjusted $p$ against the $\alpha$ line. Then one heatmap: Cliff's delta for every class pair that survived Holm, in any window, for either readout.

**Axes.** Hours before delivery, inverted; lag in seconds of stored-coefficient time on the violins; $-\log_{10} p$ on the strips.

**How it is misread.** **This clock's two families are its own** and are not corrected jointly with the second-stage clock's, nor with each other: four families across this analysis, and a reader quoting two of them is making two comparisons. **Every heatmap row reads less severe against worse**: `healthy vs acidosis`, `healthy vs hie`, `acidosis vs hie`, so a positive Cliff's delta means the *less severe* class's centroid sits further back in the past. A cell below `MIN_GROUP_SIZE` = 3 recordings is drawn as its own points rather than as a density the smoother invented, and those are the same cells the test excludes.

## `lag_clocks/lag_second_stage.pdf`

**In plain terms.** *"Does the informative past move around the moment the second stage begins?"* The same reading as the delivery clock's page, on the landmark inside labour rather than at its end. Only the recordings the labour-onset table places a second stage for are on it.

**What it shows.** The same panels: a share-of-attribution heatmap per class, then the centroid trajectories with their ribbons and the median spread.

**Axes.** Signed hours from second-stage onset, **negative before onset and positive after**, on the same $0.5$ h grid; **not** inverted, with the onset marked at zero. Lag in seconds of stored-coefficient time, lag $0$ at the bottom.

**How it is misread.** **The sign is the opposite convention from the delivery clock's** — negative is *before* the onset, and the axis label says so. **The positive side is short by construction**, so the windows after onset hold far fewer recordings than those before it and the annotated $n$ is what says which is which. The population is a **subset**: recordings with no recorded onset are dropped, counted in `second_stage/second_stage_eligibility.csv`, and reported in this analysis's own record.

## `lag_clocks/lag_second_stage_windows.pdf`

**In plain terms.** *"Those second-stage centroid lines separate — is it real, and where?"* The tested page on the other landmark.

**What it shows.** Five panels, exactly as the delivery clock's tested page: a violin per (window, class) cell for each tested readout, the Holm-adjusted $p$ beneath it, and Cliff's delta for every surviving class pair.

**Axes.** Signed hours from second-stage onset, not inverted, onset at zero; lag in seconds of stored-coefficient time; $-\log_{10} p$ on the strips.

**How it is misread.** **This clock's Holm family is its own**, per readout, and is not corrected jointly with the delivery clock's: the two are different alignments of an overlapping population, so a window significant on one and not the other is a statement about alignment. **Every heatmap row reads less severe against worse**, so a positive delta means the less severe class's centroid sits further back. A grey cross at zero on a strip means fewer than two classes had enough recordings in that window — which on the positive side of this axis is the common case rather than the exception.

## `cross_subgroup/subgroup_heatmap.pdf`

**In plain terms.** *"Which of the differences the other figures show survive being asked properly?"* Eight cohorts each with a mean always produce a highest and a lowest; some will look separated whether or not anything is there.

**What it shows.** Cliff's delta per (metric, cohort pair), with the Holm-surviving cells marked.

**Axes.** Metrics down, cohort **pairs** across; colour is the effect size, signed.

**How it is misread.** **The column order and the sign convention are both clinical.** Each column is one cohort pair named less severe first — the shared pairwise helper names a pair in the order it receives the cohorts, and this analysis hands them over in the canonical order — so a positive Cliff's delta means the less severe cohort's values run higher, in every column. Reorienting a column by eye still flips its sign against the number in the CSV. Every test here consumes one value per **recording**; a source naming a per-segment file would test segments while reading as though it tested recordings.

## `events/conditioned_coupling.pdf`

**In plain terms.** *"Does the source matter more when the uterus is contracting?"* Both coupling readouts restricted to anchors shortly after a detected contraction, against count-matched control anchors from the same recordings.

**What it shows.** The event and control distributions per readout, and their difference per recording.

**Axes.** Nats per anchor; zero marked on the difference panel.

**How it is misread.** The contraction timing comes from the **raw pressure trace** carried through the collection pass, not from the source coefficients the model reads — a contraction exists nowhere in the tables unless that one pass puts it there — so this runs over every anchor of the split rather than only over retained samples. Gaps are masked by `weight` and never by value; an event whose span touches an interpolated region is **dropped**, because its shape partly came from that interpolation. Below 200 event anchors over 4 recordings the analysis records a skip and draws nothing.

Two readouts the raw-target pipeline draws here are **absent**, and the emitted record names both with their reasons: deceleration forecast skill and the contraction-triggered response both score a clinical heart-rate trace in beats per minute, and defining a deceleration on a channel axis with no order and no clinical unit is a new construction rather than a port.

## `sufficiency/sufficiency.pdf`

**In plain terms.** *"How much is the model losing by squeezing everything through 64 numbers?"* An evaluation-only decoder of the same capacity is fitted to read the encoder state directly, and the gap between the two is what the bottleneck costs.

**What it shows.** $D_{\mathrm{base}}$ against $D_{\mathrm{oracle}}$ per recording, the gap's distribution, and the probe's held-out learning curve.

**Axes.** Nats per anchor; passes over the fit half for the curve.

**How it is misread.** **It is an estimate, not a bound**, and both bias directions travel in the emitted JSON: conditioning on the encoder state rather than on the target's own history omits the encoder's information loss and biases the gap **down**, while fitting the probe on the evaluation population biases it **up**. They oppose, neither is measured, and nothing downstream may treat the number as a bound. The convergence flag is a precondition rather than a decoration: an unfinished probe understates the gap, and a curve that never improved is **not** converged.

## `warmup/warmup_tertiles.pdf`

**In plain terms.** *"Do the channels that take longest to become honest forecast differently from the quick ones?"* The 98 kept target channels are split into three tertiles by how long their warm-up is, and the forecast gap is computed inside each. Beside it: how much of the model's attention lands on parts of the source history that were still warming up.

**What it shows.** Top: `pred_gap_warm_lo`, `_mid` and `_hi` per recording, as violins, with zero marked. Bottom: `source_lag_warmth_frac_st` and `_ph` per recording.

**Axes.** Top: nats per anchor. Bottom: fraction of attention mass — a **separate axis on purpose**, because a fraction and a nats figure on one axis would flatten whichever is smaller into a line at zero.

**How it is misread.** **The three tertiles are a decomposition, not three readouts**: they sum to `pred_gap` over the same denominator, and the run asserts that they do rather than describing it — so their *relative* sizes are the finding and their absolute sizes carry the block's scale. And **a small warmth fraction in the bottom panel is the expected finding, not a fault**: the stored source blocks warm up late, so much of the searched lag window is a region where the source coefficient is not yet honest. The panel's own title says so, and a reader who treats a low value as a defect is reading the dataset's geometry as the model's behaviour.

## `warmup/causal_warmup_budget.pdf`

**In plain terms.** *"Which channels survived, how long did each take to become honest, and where did the floor end up?"* The staircase behind the two numbers every other figure depends on.

**What it shows.** Two panels, one per stream (target above, source below). One horizontal bar per **declared** channel — the dropped ones included, which is the point of drawing it — laid out against a seconds axis whose origin is the anchor's own causal endpoint. A kept channel's bar spans $[-\Delta W'_c, 0]$: it *ends* at the anchor, and how far left it starts is its warm-up. A dropped channel is drawn at $\delta_c = 0$ instead, so its bar runs **forward** through the shaded forecast window it was still warming up for. The budget threshold is a dashed line, and each panel's title carries the kept-of-declared count per block.

**Axes.** Seconds relative to the anchor, negative to the left; one row per channel. The shaded span on the right is the forecast window.

**How it is misread.** **A bar starting before zero is how long that channel spent becoming honest — it is the mirror image of the two-sided cells' reading.** In a two-sided build a channel's boundary is a symmetric smear on both sides of a step; here it is a strictly leading delay, and a long bar means a slow filter rather than a broken one. Two more. **The line is drawn on the source panel too, where it is not a guard**: the source is never gated, so its bars crossing the budget is the design compromise being visible rather than a violation. And bars that run past the right edge are **clipped and counted** in the caption, so a truncated bar is a reported clip rather than a channel that ends there. The figure is a constant of the **shard**, not of the run: two runs over the same dataset draw the identical staircase.

## `warmup/causal_warmup_tradeoff.pdf`

**In plain terms.** *"What did the shipped budget buy, and what would a different one have bought?"* Raising the warm-up threshold keeps more channels and raises the anchor floor, which costs anchors; lowering it does the reverse.

**What it shows.** Three step curves against the candidate budget $B$: target channels kept, anchors admitted, and tiles a training step decodes at phase $0$. The shipped threshold is marked with its three values annotated, and the region where no tile fits at all is shaded behind the curves.

**Axes.** Budget threshold $B$ in decimated steps; a count axis shared by all three curves.

**How it is misread.** Three ways. The curves are **steps, not a smooth trade**: the anchor count is computed from the **survivors' own maximum** warm-up rather than from the threshold, so a threshold of 151 keeps exactly the channels 134 keeps and admits exactly the same anchors — reading it as continuous suggests tuning room that is not there. The shaded region is **not a bad choice, it is not a choice**: no tile fits there at all. And the curve says nothing about **quality** — two budgets produce mutually unloadable checkpoints whose nats are not comparable, so this figure is about feasibility rather than about which budget forecasts better.

## `source_null/source_null_difference.pdf`

**In plain terms.** **This is the figure that says whether the coupling number means anything.** The source arrives with an availability pattern — which channels are honest at which step — that is a deterministic function of time and identical for every recording. It enters the source-conditioned belief and not the target-only one, so it can push the two apart with no source *information* in it at all. Feeding the model a zeroed source leaves that clock intact and removes the content; what is left after subtracting it is the part attributable to the source actually varying.

**What it shows.** Top: the distribution of `coupling_minus_clock_nats` over **recordings**, with zero marked. Bottom: `source_conditioned_kl_raw` and `kld_source_null` side by side under their own names, on one support.

**Axes.** Nats per anchor; the histogram's height is a count of recordings.

**How it is misread.** Four ways, and the first two are the point of the bottom panel.

- **A large difference between two large numbers is not the same as a large coupling.** The violins are drawn so that case is visible rather than inferred from one subtraction.
- **The reference line means "the clock accounts for all of it"**, not "no coupling". Mass at or below zero says the coupling readout is measuring an availability pattern.
- **The verdict is INCONCLUSIVE until a threshold is set**, and that is deliberate: `eval_config.clock_margin_min_nats` ships `null`, because the first production runs are the ones that are supposed to measure the spread it should be set from. The *number* on this figure is the reading, not the status beside it.
- **Zeroing floors no source variation.** The encoder's response to a flat trajectory is not literally the availability pattern's own response, so this difference is a slightly **weaker** statement than "the clock alone" — it errs in the model's favour, and the emitted record says so.

## `spectral_skill/spectral_skill_bands.pdf`

**In plain terms.** *"Which frequencies does the model forecast well?"* The channel axis of this target domain **is** a frequency axis — a scattering coefficient is the envelope of the signal filtered at one centre frequency — so the forecast can be resolved by band without estimating a spectrum at all.

**What it shows.** Top: the forecast gap per recording, one violin per band, with each violin's **channel count in its label**. Bottom: the error-space skill of the source-conditioned branch against the target-only one, per band.

**Axes.** Top: nats per anchor, zero marked. Bottom: $1 - \mathrm{MSE}_{\mathrm{full}}/\mathrm{MSE}_{\mathrm{base}}$, zero marked. Two axes because the two are in different units and one shared axis would flatten whichever is smaller into a line at zero.

**How it is misread.** Four ways.

- **This is band-resolved skill, not coherence.** A stored coefficient is a *modulus*: the analysing filter's phase was discarded before the value was written, so nothing here can say whether a forecast is mistimed rather than mis-scaled. A forecast that is right in every band but arrives a step late reads here as a forecast that is right.
- **The band is the band of the analysing filter**, not a bin of the forecast's own spectrum. Those are two different objects.
- **The channel counts in the labels are load-bearing.** A band carried by three channels and one carried by forty are not comparable as evidence, and the label is where that shows.
- **`unknown` is a band with no frequency, not a leftover.** Three of the 98 scored channels have no recoverable centre frequency because no selected phase-harmonic pair named their filter, and they are reported under their own label rather than bucketed into a neighbour — which would misattribute their skill to a frequency they do not have.

## The per-recording pages: `samples/<selection>/sample<index>_<guid>_epoch<epoch>.pdf`

**In plain terms.** *"Show me one, in full."* Every other figure reduces the split to a distribution; these render individual segments so that a number nobody believes can be looked at.

**What it shows.** This cell's **fifteen-row** diagnostic page, drawn through the task's own page seams — the same layout the training callback draws, rather than a second builder that could disagree with it. The rows are: the raw context, the forecast lanes, six causal extra rows (truth, both branch means, the signed skill difference, the posterior's own $\sigma$, and the per-window score), two gated-input rows, and the five latent and lag rows the layout owns.

**Directories.** `stratified/` is a seeded, shard-stratified draw over the whole split, so a cap at or above the shard count reaches every shard. `by_class/` is a **class-balanced** draw: the same number of segments from every clinical class, `eval_config.caps.pages_per_class` of them. Beside them, one directory per headline metric and tail holds the segments at the extremes of that metric.

**The two draws are not interchangeable, and reading one as the other is the mistake to avoid.** `stratified/` allocates its quota in proportion to shard size, so what it renders is what the split mostly *contains* — on the shipped cohort, mostly healthy. `by_class/` gives `hie` as many pages as `healthy` and is therefore, by construction, not representative of anything: it is what supports a comparison *across* classes, and it says nothing about how common either class is. Counting pages in `by_class/` as evidence of prevalence inverts the one property it was drawn for.

**Every segment here appears twice** — the full page, and the reduced page beside it. See the next section.

**How they are misread.** A page is **one segment of one recording: an illustration, never evidence.** The extreme pages are selected *on* the quantity they display, so the panel showing it is guaranteed to look unusual and says nothing about how often it does. The `<index>` in the filename is the position in the evaluation **dataset**, not in `per_sample.csv` — the collection pass runs under a seeded shuffle — and the two are reconciled by a `guid`/`epoch` round trip checked before anything is rendered. Rendering needs a checkpoint; a model-free re-run records a skip.

## The reduced per-recording pages: `samples/<selection>/sample<index>_<guid>_epoch<epoch>_compact.pdf`

**In plain terms.** *"Show me one, without the forecast."* The same segment as the page above,
drawn from the same forward pass, reduced to the rows that answer what the latent and the attention
did.

**What it shows.** Five of the full page's fifteen rows, in the full page's own order, drawn by the
same code: the raw context, the **target** stream as the encoder receives it (`fhr_st` | `fhr_ph`,
block dividers and warm-up staircase intact), the latent state over its source-derived shift, $K_t$,
and the lag attention. It is the full page with rows removed, not a second page — a reader who knows
one knows the other.

**What it drops, and why that is the point.** The forecast lanes and the six causal extra rows: the
truth, both branch means, the signed skill difference, $\sigma^q$ and the per-window score. Those
eight rows answer *what did the model predict*, which is a different question, and on a $14 \times
48$ in page they sit between the input row and the latent row that are read against each other.

**The lag attention is drawn on a logarithmic colour scale here**, and on a linear one on the full
page. Attention is a softmax over 91 lags, so on a linear scale one dominant lag flattens the rest of
the panel into the bottom colour — acceptable when the panel is one of fifteen, not when it is one of
five. The scale is floored four decades below the panel's own maximum, so a single near-zero cell
cannot stretch the colormap over decades that hold nothing.

**How it is misread.** Three ways.

- **A grey cell is a forbidden lag, not a small one.** The log scale masks the non-positive cells,
  and they are painted light grey rather than left showing the axes background. Every lag below the
  source floor $F_u$ is zeroed by the lag mask: the model was never allowed to attend there. On the
  linear full page those cells are simply the bottom colour and are indistinguishable from genuinely
  low attention.
- **The colour scale is per page.** It is taken from that segment's own attention, so two reduced
  pages cannot be compared by colour. The argmax overlay can be.
- **Everything the full page's caveats say still applies**, including the lag-time one printed at the
  foot of both: a lag axis here is stored-coefficient time, not physical delay.
