# Figure guide

This guide explains what each evaluation figure shows, how to read its axes, and what conclusions it supports. Entries use the filenames written inside each analysis directory. PDF is the default format; another configured format changes the extension. [EVAL.md](EVAL.md) explains the workflow, outputs, settings, and checks.

Start with `forecast/baseline_comparison.pdf`, then `coupling/pred_gap_distribution.pdf` and the calibration figures. Read the source controls before interpreting KL or lag structure. Use the terminology section below whenever a metric name is unfamiliar.

## Conventions used throughout

These conventions apply to all figures unless an entry states otherwise.

**Missing results are labelled.** A panel with no finite observations displays `no finite values`. Treat this as an unavailable measurement and check the analysis status and counts.

**Error values use normalised coefficient units.** The targets are retained wavelet-modulus and phase-harmonic coefficients, standardised by the loader. They have no direct clinical scale. Reversing standardisation would place channels on very different scales and invalidate pooled comparisons. Read error axes as normalised coefficient errors, and compare scores only when target definitions and scoring geometry agree.

**Lag axes show stored-coefficient time.** The displayed compensated lag is $\tau=4(\ell+\delta)$ seconds, where $\ell$ is the lag bin and $\delta$ is the model's input delay. This corrects the input delay only. It does not establish physiological delay: the one-sided feature bank can introduce composed group delays up to $791$ seconds, comparable to or larger than the $364$-second search window. Each lag figure prints the shared qualification.

**Channel alignment reduces timing differences but does not make these lag axes physiological.** Aligning input channels to a common reference replaces channel-pair differences with a reference offset and remaining within-band delay dispersion of about $\pm16\%$. The target is still a coefficient block with a nonzero target reference delay, and advancing the first target horizon element can require an unavailable stored index. These figures therefore compare coefficient epochs. The run records `source_reference_delay_s` separately from the axis.

**Cohort order is worst first.** Clinical classes appear as `hie`, `acidosis`, then `healthy`. Subgroups appear as `hie_cs`, `hie_no_cs`, `acidosis_cs`, `acidosis_no_cs`, `healthy_bg_cs`, `healthy_bg_no_cs`, `healthy_no_bg_cs`, and `healthy_no_bg_no_cs`. Healthy is green, acidosis amber, and HIE red; subgroups use shades of their class colour. `cohort.ordered_groups`, bound to the shared label helper, also orders the corresponding CSV rows. Unrecognised cohorts appear after the known groups.

Colours in training-callback plots may differ; match cohorts by labels and legends. Clinical ordering is a display convention, not evidence of a group difference. Use `cross_subgroup` for statistical comparisons.

**Figures share one publication style.** The style is defined in `teb_vae/lag_attn/eval/figures.py` and accessed through `figures_seam`: double-column default width, a 5.5–7 pt serif type scale with STIX mathematics, hairline frames and data lines, histogram bars with a thin dark outline, open frames except for boxed heatmaps, and unframed legends above full-width curves. Non-cohort series use the Okabe-Ito colour-blind-safe palette. Panel letters (**a**, **b**, and so on) run left to right, then top to bottom. Footnotes use reserved space below the axes.

---

# Terminology

This section explains the model, common filename and column conventions, and the main quantities. A recording contains segments; each segment contains forecast anchors. An anchor is the time from which the model predicts the next $H$ stored steps. Most statistical summaries give each recording one value.

## What the model predicts

The original recordings contain fetal heart-rate and uterine-pressure signals sampled at $4$ Hz. The model receives features computed from them using a strictly one-sided scattering and phase-harmonic transform. These coefficients are stored every $4$ seconds. In the legacy geometry used for examples below, the declared inputs contain $102$ target channels from heart rate ($36$ scattering and $66$ phase-harmonic) and $51$ source channels from pressure. The warm-up budget in the documented configuration retains $98$ target channels for prediction; use the run's retained-channel record for other configurations.

For current and legacy configurations alike, at anchor $t$ the model predicts all $C_{\mathrm{keep}}$ retained target channels over the next $H$ steps, or $4H$ seconds. It makes a **base** forecast using target history and a **full** forecast that also uses source history. Their predictive gap measures whether adding source history improved the score. The checkpoint supplies `horizon`; the current default is $H=10$, while older examples used other horizons.

Two timing constraints determine which anchors can be evaluated:

- **Warm-up sets the first usable anchor.** Target channels must be valid at the earliest scored step, and aligned input channels must have enough history after their shifts. In the documented geometry, the two requirements are $B-1=133$ and $\max_c(W'_c+d_c)=134$, so the anchor floor is $F=134$.
- **The searched lags must fit before the anchor.** With $F=134$ and a furthest lag of $90$, the full lag window exists at each scored anchor. The pipeline measures this property on each run. Anchor count also depends on the horizon and forecast clock; it is not fixed by the floor alone.

## Statistical terms used in the figures

| Term | How to read it |
| --- | --- |
| Median | The middle value after observations are sorted. |
| Quartiles and IQR | $Q_1$ and $Q_3$ bound the middle half of the values. The interquartile range is $Q_3-Q_1$. |
| Density | Relative concentration of observations along an axis. A density's width or height is not a recording count. |
| Paired comparison | A difference calculated within each recording before combining recordings. |
| Centroid | The weighted average location of a lag profile. |
| Entropy | A measure of how spread out normalised weights are. Lower entropy indicates more concentrated weights. |
| Argmax | The position with the largest value. On a flat profile, it can reflect a tie-breaking rule rather than a meaningful peak. |
| Degenerate profile | A profile that fails the stated shape criterion, so its peak position should not be interpreted alone. |
| Holm correction | A procedure that adjusts several tests together to limit false-positive findings within their declared family. |
| Kruskal–Wallis | A rank-based test for differences among groups. A significant result does not identify which pair differs. |
| Mann–Whitney | A rank-based comparison between two independent groups. |
| Wilcoxon signed-rank | A rank-based test of paired differences, such as two scores from the same recordings. |
| Cliff's delta | An effect size describing how often values in one group exceed values in another. Its sign follows the stated group order. |
| Nominal and adjusted intervals | A nominal interval applies to one specified comparison. A family-adjusted interval accounts for a declared set of comparisons. |
| Attribution | A way of assigning a fitted model's output change to inputs or intermediate quantities under a specified method. It does not by itself establish a physiological cause. |

## How to read a name

| Piece | Means |
|---|---|
| `base` | the target-only branch (prior belief) |
| `full` | the source-conditioned branch (posterior belief) |
| `shuffled` | the negative control: the *same* model fed **another recording's** source |
| `null` | the second control: the same model fed a **zeroed** source stream |
| `mc_` prefix | Monte Carlo scoring: average likelihood over $K$ latent draws, then take its negative logarithm. |
| `_block` suffix | summed over the whole $H \times C_{\mathrm{keep}}$-coefficient forecast block (`block_width` in `preflight.json`), then averaged over anchors |
| `_raw` suffix (on a KL) | **unfloored** — no free-bits floor applied. The only form readable as a rate. |
| `_sq` suffix | left **unrooted** (a mean square). Its `_rms` partner is the rooted version. |

## The quantities

### Scores: predictive performance

A **score** here is always a negative log-likelihood: **lower is better**, and it is not bounded below by zero.

| Name | Plain meaning | Units |
|---|---|---|
| `nll_base_block` | how badly the target-only forecast fit the truth | nats per anchor |
| `nll_full_block` | the same for the source-conditioned forecast | nats per anchor |
| `mc_nll_base_block`, `mc_nll_full_block` | Negative log of the likelihood averaged over $K$ latent draws; the headline pair. | nats per anchor |
| `mc_nll_shuffled_block` | the score when fed another recording's source; the negative control | nats per anchor |
| `nll_persistence_block` | baseline: hold the last **observed** coefficient vector for the whole window | nats per anchor |
| `nll_climatology_block` | baseline: predict the population mean, which after z-scoring is exactly $0$ | nats per anchor |
| `nll_segment_mean_block` | baseline: predict this segment's own per-channel mean | nats per anchor |
| `nll_oracle_block` | an evaluation-only decoder reading the encoder state **directly**, bypassing the latent bottleneck | nats per anchor |

**Nats per anchor** means that a score is summed over the valid coefficients of each forecast block, then averaged across scored anchors. Its magnitude depends on block size, predicted variance, and the observations. A negative log-density can be negative, so compare score differences on the same target and support rather than interpreting zero as a universal reference. Dividing by $H\cdot C_{\mathrm{keep}}$ gives a fixed-width rescaling. With masked forecast steps, that divisor is larger than the number of coefficients actually scored.

### Gaps: the predictive contribution of the source

| Name | Plain meaning | Units |
|---|---|---|
| `pred_gap` | $D_{\mathrm{base}}-D_{\mathrm{full}}$ using the training scoring path. Positive means source benefit under that convention. | nats per anchor |
| `mean_pred_gap` | The same difference with both branches decoded at their latent means, without sampling. | nats per anchor |
| `mc_pred_gap` | the same difference on the marginalised scores. **This is the headline coupling number.** | nats per anchor |
| `pred_gap_warm_lo` / `_mid` / `_hi` | the same gap restricted to each warm-up tertile of the 98 kept channels. The three **sum to** `pred_gap`. | nats per anchor |
| `pred_gap_<band>` | the same gap restricted to one frequency band of the target coefficient. These sum to `pred_gap` too. | nats per anchor |
| `delta_suff_nats` | $D_{\mathrm{base}}-D_{\mathrm{oracle}}$: an estimate of the cost of the latent bottleneck, subject to probe and population biases. | nats per anchor |
| `mse_skill` | $1 - \mathrm{MSE}_{\mathrm{model}}/\mathrm{MSE}_{\mathrm{baseline}}$. $1$ = perfect, $0$ = no better than the baseline, negative = worse. | dimensionless |
| `advantage_nats_per_anchor` | the NLL-space analogue of skill, and a **difference**, not $1 -$ a ratio — a log score has no natural zero | nats per anchor |
| `pred_gap_rmse_pct`, `pred_gap_mse_pct` | the percentage of the target-only branch's point-forecast error the source removed, rooted and unrooted. **Scale-free.** | percent |
| `pred_gap_mc_likelihood_pct` | $100(e^{\texttt{mc\_pred\_gap}/(H \cdot C_{\mathrm{keep}})} - 1)$: the extra probability density the source-informed forecast puts on each observed coefficient. **Budget-local.** | percent |

Keep the scoring conventions separate. `mc_pred_gap` integrates over latent draws and is the acceptance headline. `mean_pred_gap` decodes both latent means without sampling. `pred_gap` follows the training scoring path; under `base_decode: mean`, its base and full branches use different decoding conventions. Their values and even their signs can differ. A figure's labels identify which comparison it displays.

Do not divide a predictive gap by a block NLL to obtain a percentage. NLL has no natural zero and may be negative, making that ratio misleading. The error percentages use a positive error denominator. The likelihood percentage is defined only for `gaussian_nll` and uses the fixed block width $H\cdot C_{\mathrm{keep}}$. It is budget-local: changing the horizon or retained channels changes the denominator.

### KL: changes in the latent distribution

$K_t = \mathrm{KL}(q_t \Vert p_t)$ measures how far reading the source moved the belief at anchor $t$. Zero means the source changed nothing.

| Name | Plain meaning | Units |
|---|---|---|
| `source_conditioned_kl_raw` | $K_t$ averaged over the scored anchors. **The** KL readout. | nats per anchor |
| `source_conditioned_kl_shuffled_raw` | the same under another recording's source — the specificity control | nats per anchor |
| `kld_source_null` | the same under a **zeroed** source — the availability-clock control | nats per anchor |
| `coupling_minus_clock_nats` | their difference: the part of the coupling attributable to source *variation* | nats per anchor |
| `kld_per_t` | $K_t$ *before* averaging: one value per anchor | nats per anchor |
| `kld_per_dim`, `kld_per_head` | $K_t$ split across latent dimensions or attention heads; each sums back to the total | nats per anchor |
| `source_kl_lag_map` | $K_t$ split across the 91 lags; sums back to the total | nats per anchor |
| `active_dims`, `top_dimension_share` | how many latent dimensions carry more than a threshold, and the largest one's share | count, fraction |

Check these four points before interpreting KL:

1. **Use unfloored KL.** A free-bits floor imposes a minimum contribution and can hide a weak or collapsed source pathway. The `_raw` columns retain the measured value.
2. **Check prior variance.** KL contains $(\mu^q-\mu^p)^2/\sigma_p^2$, so a prior variance near its lower clamp can inflate it. Read `prior_variance_not_pinned` and the predictive gap alongside KL.
3. **Judge source specificity by predictive scores.** Another recording's source may increase KL while worsening prediction. A larger shuffled KL alone is not a failure or success.
4. **Check the availability clock.** Source availability varies with time in the same way across batch rows. Row permutation cannot remove that shared pattern. Compare matched KL with `kld_source_null` and report their difference with its limitations.

### Warm-up, bands and geometry

| Name | Plain meaning |
|---|---|
| `causal_warmup_steps` | per channel: the leading delay, in 4-second steps, before that channel's coefficients have completed warm-up |
| `causal_delay_s` | per channel: the composed one-sided group delay, in seconds. What makes the lag axis coefficient time. |
| `target_warm_frac` | the fraction of scored anchors at which every kept target channel is warm. Must be exactly $1.0$. |
| `anchors_per_sample` | anchors decoded per segment. Must be exactly the checkpoint's own `anchor_ceiling - warmup_period` at the dense set: $T_{\mathrm{valid}} - F$ on the stored forecast clock, less the clock's largest label advance under a `physical` one (`anchors_per_sample` in `preflight.json` is the expected value). |
| `source_lag_warmth_frac_st` / `_ph` | the fraction of attention mass landing on lags at which that stored source block is warm. **A small value is the expected finding.** |
| band | one of `slow_baseline`, `deceleration`, `variability`, `beat_to_beat`, `unknown` — the band of the **analysing filter** that produced the coefficient, not a bin of the forecast's own spectrum |
| `unknown` | a channel whose centre frequency is not recoverable, because no selected phase-harmonic pair named its filter. Never bucketed into a neighbour. |

## Geometry reference

The table below illustrates the legacy aligned geometry used by many examples in this guide. It is not a list of universal dimensions. The current `configs/default.yaml` uses $80$ declared target channels, $46$ declared source channels, and $H=10$ without input-channel alignment. The retained counts must be resolved from the checkpoint and dataset. Use `preflight.json` and the saved configuration for the run being plotted.

| Symbol | Value | What it is |
|---|---|---|
| $T$ | 300 | steps per segment |
| $H$ | 30 in this legacy example | A $120$-second horizon. The current default is $H=10$; use the checkpoint's `horizon`. |
| $F$ | 134 | anchor floor: nothing below it is decoded at all; $\max(B - 1, \max_c(W'_c + d_c))$ |
| anchors | `anchor_ceiling - warmup_period` | decoded per segment, at the dense set; `preflight.json` records the run's own |
| $C_{\mathrm{keep}}$ | 98 | target channels the warm-up budget kept, of 102 declared |
| $H \cdot C_{\mathrm{keep}}$ | `block_width` | coefficients in one forecast block; `preflight.json` records the run's own |
| $d_z$ | 64 | latent dimensions |
| $L$ | 91 | lag bins = ~6 minutes of source history |
| $M$ | 4 | attention heads |
| lag support margin | 44 | $F - (L-1) - \texttt{lag\_floor}$; $\ge 0$ means no anchor has truncated lag support |

---

## The grouped variants: `<stem>_by_clinical_class.pdf` and `<stem>_by_subgroup.pdf`

**Purpose.** Compare the distribution of a metric across clinical classes or subgroups, using one value per recording.

**Read the violin width as estimated density.** Wider regions contain more of the group's values. The thick internal bar spans the middle half, from the first quartile $Q_1$ to the third quartile $Q_3$. Thin whiskers extend to observations within $1.5$ interquartile ranges of that bar, and the white dot marks the median. Values beyond the whiskers remain represented in the violin body, which extends to the group's observed extremes.

Grouped variants are created automatically from each analysis's declared per-recording tables. Their filenames identify the source table and grouping axis, so they are described here as a family rather than listed individually.

**Axes.** One row per metric; the metric's own units on the vertical axis, cohorts across in descending severity. The CSV beside each figure carries its rows in that same order.

**Interpretation.** Each violin contains one value per recording. Eight recordings remain eight observations regardless of how many segments they contribute. Violin width is density, not sample size. Fewer than two groups produces a recorded skip. Use `cross_subgroup` to test group differences rather than relying on visual separation.

---

## `forecast/baseline_comparison.pdf`

**Purpose.** Compare both model branches with three simple predictors: the last observed coefficient vector, the population mean, and the segment mean. The lower panel reports squared-error skill: $1$ is perfect prediction, $0$ matches the baseline, and a negative value is worse.

**What it shows.** Top: the per-recording block score of every predictor — the two model branches and the three trivial baselines — as violins, in nats per anchor, lower better. Bottom: the squared-error skill of each branch against each baseline, with a percentile bootstrap interval over **recordings**.

**Axes.** Top: nats per anchor, one violin per predictor. Bottom: skill, dimensionless, zero marked; whiskers are asymmetric because a percentile interval is not symmetric about its point estimate.

**Terms on this figure.** `base`, `full`, then `persistence`, `climatology`, `segment_mean`; each violin is `nll_<name>_block` across recordings. The bars are `mse_skill` with `mse_skill_lo` / `_hi` as whiskers.

- **persistence** carries forward the last *observed* step, not the last one: `weight` is the only trustworthy validity signal here, since the coefficients carry no sentinel of their own, and carrying an invalid step forward would measure the gap.
- **climatology** is exactly $0$ per channel, which is the z-scored population mean — and it is a meaningful baseline only because the normalisation statistics were accumulated *excluding* the warm-up region.
- The baselines are scored at a fixed `BASELINE_LOGVAR = 0.0`, recorded rather than fitted: a point predictor has no variance of its own, and the whole score would otherwise be decided by whatever $\sigma$ it was handed.

**Interpretation.** The block score is a *sum over $H \cdot C_{\mathrm{keep}}$ coefficients*, so it is large under every predictor and its scale says nothing about the model. Only the comparison is readable. The skill drawn here is the MSE-space one; the NLL-space column beside it in the CSV is a **difference** in nats, not $1 -$ a ratio.

## `forecast/anchor_profile.pdf`

**Purpose.** Show whether predictive scores change from early to late anchors within a segment. At each stored position, the curve averages over segments scored there.

**What it shows.** The two block scores and `pred_gap` against position in the segment, averaged over every segment that scored that anchor.

**Axes.** Anchor index in decimated (4 s) steps from the start of the trimmed segment; nats per anchor.

**Interpretation.** **The profile starts at 134, and that is the geometry rather than a finding** — and it is the opposite shape from the raw-target cells, where the curve begins at the model's own 30-step warm-up and droops through a truncated-lag region. Here nothing below the anchor floor is decoded at all, so there is no droop to discount and no truncated region inside the profile. The last $H$ anchors are still never scored, because their forecast window would run past the end of the segment. A reader expecting the raw cells' shape and finding a curve that begins two-fifths of the way in is looking at the anchor floor.

## `forecast/horizon_skill.pdf`

**Purpose.** Show how predictive scores and errors change with forecast lead time. The gap panel identifies the future steps where adding source history helps.

**What it shows.** $D_{\mathrm{base}}(\tau)$ and $D_{\mathrm{full}}(\tau)$, their gap, and each branch's error, against lead time over the checkpoint's $H$ horizon steps.

**Axes.** Lead time in **seconds**; nats **per horizon step**, so the 15 values sum back to the block.

**Interpretation.** The curve is computed on the **single-draw** path and says so in its title: the Monte Carlo marginalisation does not commute with the sum over $\tau$, so a marginalised curve would not sum back to the marginalised headline. Horizon step $0$ is $4$ s ahead, not $0$ — the anchor's own step is the past, not the forecast.

## `forecast/forecast_overlay.pdf`

**Purpose.** Compare an individual forecast with its observed target over the checkpoint's horizon. This reveals details that averages can hide.

**What it shows.** One anchor's truth and both branch means, for **three kept channels** drawn against lead time. Three channels rather than one line, because what is forecast is an $H \times C_{\mathrm{keep}}$ block and there is no single trace to overlay.

**Axes.** Lead time in seconds within this one block; the coefficient's value in $z$ units.

**Interpretation.** It is **one anchor of one retained recording**, drawn from a seeded stratified draw, not a representative case. Retention is opt-in (`eval_config.caps.waveforms`), so a run that did not ask for it emits no such figure at all; the absence is silence, not failure. And the three channels are a sample of 98 — a channel that tracks well says nothing about the ones not drawn.

## `coupling/pred_gap_distribution.pdf`

**Purpose.** Show whether adding uterine-pressure history improves prediction and how consistently it helps across recordings. This is the main predictive-gain figure. Zero means no improvement under the displayed scoring convention.

**What it shows.** Three blocks. **Top, one histogram per estimator** — `mc_pred_gap` (Monte Carlo marginalised, the gate's headline), `mean_pred_gap` (both branches decoded at their latent **mean** under the decoder's own variance, no draw) and `pred_gap` (the training path, the parity column) — each over **recordings**, each with zero marked and the bootstrap interval on its *own* mean shaded, and the share of recordings above zero in its title. **Middle:** the three estimators side by side as violins, one colour per estimator kept across the page.

**Bottom:** the mean-decoded estimator against the marginalised one and against the training-path one, one point per recording, with the identity line, the two zero lines, Spearman's $\rho$ and the share of recordings on which the two agree in sign.

**Axes.** Nats per anchor throughout; a histogram's height is a count of recordings, not of segments.

**Interpretation.** Four ways. The shaded band is the interval on the **mean**, not the range of the data. The three histograms are three *estimators of the same quantity*, and a sign that differs between them is a fact about the estimator rather than a contradiction: the marginalised score is a log of an average likelihood over $K$ draws, so a broad prior can out-score a sharper posterior there while its mean forecast is worse, and on a `base_decode: mean` checkpoint that is the ordinary case — which is why the mean-decoded estimator is the one the other figures foreground and why the scatters exist, since two histograms cannot say whether the recordings that flipped are the same recordings.

The unit is one recording, so a recording that scored no anchors is absent rather than at zero — the $n$ in each title is the count actually available. And the gate still reads `pred_gap_mc_nats`; the mean-decoded column is beside it in the headline under its own name, not in its place.

A fourth is this cell's own: **a positive gap here is not yet a source finding.** Read `source_null/source_null_difference.pdf` beside it, because part of a coupling readout can be an availability clock that no control on this figure can see.

## `coupling/pred_gap_percent.pdf`

**Purpose.** Express the predictive comparison as percentage changes in error and likelihood density. Zero means no change. The panels separate these quantities because they use different denominators.

**What it shows.** Top: the distribution of `pred_gap_rmse_pct` over recordings, zero marked, the interval on the mean shaded. Middle: `pred_gap_rmse_pct` and `pred_gap_mse_pct` side by side — the same ratio under a root, so the mean-square figure is the larger wherever both are positive. Bottom: `pred_gap_mc_likelihood_pct` and `pred_gap_mean_likelihood_pct` side by side — the same per-coefficient density ratio on the marginalised and on the mean-decoded gap, in the estimator colours of the figure beside this one — **empty under an `mse` checkpoint**, where they are undefined rather than zero.

**Axes.** Percent on every panel; the histogram's height is a count of recordings.

**Interpretation.** The bottom panel is **budget-local**: it divides by $H \cdot C_{\mathrm{keep}}$, and $C_{\mathrm{keep}}$ is whatever the warm-up budget decided, so it cannot be compared across two arms at two budgets — nor, for that matter, against any other cell of the grid. The two error-space panels are scale-free and do not carry that caveat.

## `latent/kl_spectrum.pdf`

**Purpose.** Show how much each latent dimension changes when source history is added. A concentrated spectrum means a small number of dimensions account for most of the KL; it does not establish forecast usefulness by itself.

**What it shows.** The per-dimension KL, sorted. The active-dimension count, the top dimension's share and the total are in the analysis record beside it rather than annotated on the panel.

**Axes.** Latent dimension rank; nats per anchor.

**Interpretation.** A tall first bar is not automatically a fault — one dimension carrying most of a small total is a different finding from one carrying most of a large one, and the total is in the record for that reason. What *is* a fault is a spectrum read without checking `prior_variance_not_pinned` first: a prior variance on its clamp multiplies every bar by an arbitrary factor while every decoder-side diagnostic stays healthy.

## `lag_kl/lag_kl_profile.pdf`

**Purpose.** Show how the model's attention weights distribute KL across stored source lags. Read this as a coefficient-time attribution, with the lag and variance limitations below.

**What it shows.** The per-lag KL attribution in its raw, support-corrected and untruncated forms, with the peak and its width marked.

**Axes.** Lag in 4-second steps, with the compensated seconds axis beside it; nats per anchor.

**Interpretation.** Two ways, and the caveat printed under the figure states the second. **The three profiles coincide here, and that is a measurement rather than a redundancy**: at this anchor floor every lag exists at every scored anchor, so the support correction and the untruncated recomputation are inert — and an arm that lowered the floor would separate them again, which is why all three are still drawn. And **a peak is a position in stored-coefficient time**, not a physiological delay: the composed group delay is uncorrected and reaches the same order as the search window itself.

## `attention/attention_profile.pdf`

**Purpose.** Show where each attention head places weight over source history. Separate head curves reveal patterns that a head-averaged profile can hide.

**What it shows.** Two panels: the head-averaged attention profile over lags, and the per-head profiles beside it, one curve per head. Each head's entropy against the ceiling it can actually reach is **not on the page**; it is in `attention_per_recording.csv` and the analysis record.

**Axes.** Lag in seconds (stored-coefficient time); attention weight.

**Interpretation.** The entropy is quoted against the **attainable** ceiling $\operatorname{mean}_t \log \min(t+1, L)$, which at this floor equals $\log L$ exactly — measured, not substituted. Reading a head's entropy against a hand-computed $\log L$ on an arm with a lower floor would report a model attending uniformly over what exists as increasingly concentrated. And the entropy is taken per anchor and then averaged, never as the entropy of the averaged profile: a mixture's entropy is at least the mean of the entropies mixed, so the second reports a model whose lag focus *shifts* as one that has none.

## `attention/lag_heatmap.pdf`

**Purpose.** Show how attention over source lags changes across the anchors of one retained segment.

**What it shows.** Head-averaged attention as a heat map over (time in segment, lag) for one retained sample, with the anchor floor marked.

**Axes.** Time in segment (s) across; lag in seconds (stored-coefficient time) down, each lag row spanning its own 4 s bin; colour is attention weight.

**Interpretation.** It is **one retained segment**, drawn only where `eval_config.caps.attention` asked for retention — an absent figure is silence, not failure. It is head-averaged, so a single head's structure can be hidden under three flat ones; the per-head profiles are on `attention_profile.pdf`. The vertical axis is the same coefficient-time lag as everywhere else.

## `calibration/pit_reliability.pdf`

**Purpose.** Check whether forecast uncertainty matches observed errors. For example, an interval with about $68\%$ nominal coverage should contain about $68\%$ of observations over repeated forecasts.

**What it shows.** The probability integral transform of the standardised residuals against uniform, and the empirical central coverage at the exact erf nominals.

**Axes.** Nominal probability against realised; the diagonal is perfect calibration.

**Interpretation.** The nominals are $\operatorname{erf}(k/\sqrt{2}) = 0.6827,\ 0.9545,\ 0.9973$ — the two-sigma figure is **not** 0.95, which is $\pm 1.96\sigma$, and the half-point difference reads as a real miscalibration if the wrong nominal is assumed. The unit here is one **coefficient**, not one element of a 4 Hz trace, which is why the count beside it is `n_coefficients`. An `mse` checkpoint emits no such figure at all: its log-variance head was never fitted.

## `calibration/logvar_distribution.pdf`

**Purpose.** Show whether predicted log-variances concentrate near their allowed lower or upper limits. A mean log-variance alone can hide concentration at both ends.

**What it shows.** The distribution of the decoder's log-variance with both clamp bounds marked. The floor and ceiling fractions are in the analysis record and `summary.json`, not annotated on the panel.

**Axes.** Log-variance; fraction of coefficients.

**Interpretation.** This is the one figure whose reading changes a config value — the analysis states a recommended `logvar_clamp` revision **per coefficient**, which is the axis the objective's block score reduces over, and says *no change* when neither end binds. A recommendation emitted unconditionally would be applied unconditionally.

## `distributions/class_histograms.pdf`

**Purpose.** Compare distribution shapes across clinical classes. Similar means can hide different tails or a small set of large errors. Segment densities and recording summaries show different levels of aggregation.

**What it shows.** Nine metrics, one panel each, drawn at **two levels on the same axes**: a filled density of one value per **segment**, and a median / inter-quartile / range **strip** above it of one value per **recording**. The coupling pair is `mc_pred_gap` and `mean_pred_gap`, adjacent, because the two estimators of the same gap disagree wherever a branch's latent spread matters and a reader should see both distributions before trusting either sign.

**Axes.** The metric's own units in $z$ space; density rather than counts, so a cohort contributing ten times the segments does not simply draw a taller curve.

**Interpretation.** **The difference between the two levels is the content, not a redundancy**: a strip far narrower than the density beneath it says most of the visible spread is within-recording variation, and the density is showing many views of the same delivery. This analysis computes **no test on purpose** — consecutive anchors overlap in 14 of their 15 horizon steps, so a per-segment $p$-value is anticonservative by roughly that factor. A separation visible here is a reason to look at `cross_subgroup`, not a result.

## `distributions/subgroup_histograms.pdf`

The same nine metrics resolved by subgroup, **nested rather than flat**: one column per clinical class with that class's subgroups overlaid inside it, so a cell holds at most four curves and they are four tints of one hue. Each cohort is a faint fill under a hairline outline at full opacity, drawn in two passes so every outline sits above every fill — one pass per cohort would leave the first cohort's outline veiled by every later fill, and the first legend entry would be the hardest curve to trace.

## `trajectory/trajectory_profile_pred_gap.pdf`

**Purpose.** Follow predictive gain within a 20-minute segment and across a recording assembled from its segments.

**What it shows.** `mc_pred_gap` against time in segment, and against absolute time $t_{\mathrm{abs}} = \mathrm{epoch} + 4t$ across a delivery, with overlapping steps averaged and `n_contributing` travelling beside them.

**Axes.** Time in segment (s) on the within-segment panel; hours before delivery on the whole-delivery panel; nats per anchor.

**Interpretation.** The within-segment panel **starts at the anchor floor**, for the same reason `forecast/anchor_profile.pdf` does. Across a delivery the line is **lifted wherever nothing was decoded** — and on this cell that includes the undecoded warm-up prefix of every segment, so the whole-delivery line is a run of short pieces one segment apart by construction, not a recording full of gaps. A **break** in `trajectory_delivery_summary.csv` is only a gap longer than one segment stride, i.e. a missing segment. The averaging of overlapping steps is visible in `n_contributing` rather than inferred.

**Why it is one readout.** The lower panel is a single axis in nats per anchor, and the KL sitting on it beside `pred_gap` is routinely orders of magnitude larger — so a shared page draws the gap as a flat line at the bottom of the KL's range and reports a real movement of a tenth of a nat as nothing. The KL has its own page beside this one.

## `trajectory/trajectory_profile_pred_gap_mean.pdf`

The same two views of `mean_pred_gap` — the gap on the mean-decoded forecasts, no latent draw — on its own page beside the marginalised one. Same axes, same unit, a different estimator; on a `base_decode: mean` checkpoint the two pages routinely disagree in sign, and a reader comparing them is comparing estimators rather than recordings. The caveats of the page above apply unchanged.

## `trajectory/trajectory_profile_kl.pdf`

The same two views of `kld_per_t`, on its own axis. Everything about the reading is the page above's; what differs is the quantity, and its one caveat: the KL is **inflated by an arbitrary factor whenever the prior variance sits on its clamp**, so its level is not comparable across checkpoints and a rise here is a rise only if the clamp state did not change. `pred_gap` carries no such factor, which is why the pair is reported and why neither is read alone.

## `time_to_delivery/time_to_delivery_trajectory_pred_gap.pdf`

**Purpose.** Compare predictive-gain trajectories across clinical classes as delivery approaches, using half-hour windows.

**What it shows.** `mc_pred_gap` against hours before delivery, $-\mathrm{epoch}/3600$, class-stratified, using one value per recording within each window.

**Axes.** Hours before delivery (negative, increasing to the right); nats per anchor.

**Interpretation.** Significance is tested **per window**, with Holm across windows as one family; the `pooled` row is flagged `confounded_by_time` and consumed by nothing, because a pooled difference between classes with different recording lengths is a difference in when they were recorded. The bin width is a module constant rather than a config key, for the same reason the significance level is not one.

**Why it is one readout.** The KL is on the same nominal unit and not on the same scale — it is multiplied by an arbitrary factor whenever the prior variance sits on its clamp — so a page carrying both puts `pred_gap` on a range the KL set. It has its own page beside this one, and the two are compared in `time_to_delivery_trajectory.csv` rather than by eye.

**Beside it.** `time_to_delivery_windows_pred_gap.pdf` draws the per-recording distribution behind every point of this figure, the Holm-adjusted significance of each window, and the effect size of every class pair that survived — on this same axis. Read that one before quoting a gap between two lines here.

## `time_to_delivery/time_to_delivery_trajectory_kl.pdf`

The same figure for `source_conditioned_kl_raw`, on its own axis, with `time_to_delivery_windows_kl.pdf` beside it. Read exactly as the page above, with one addition: the unfloored KL is **inflated by an arbitrary factor whenever the prior variance sits on its clamp**, so a trajectory visible here and absent from the `pred_gap` page is a statement about which of the two is being read rather than about the coupling.

## `time_to_delivery/time_to_delivery_windows_pred_gap.pdf`

**Purpose.** Show the recording-level values behind each class trajectory and the statistical comparisons within each time window.

**What it shows.** Three panels of `mc_pred_gap`. A violin per (window, clinical class) cell over one value per **recording**; directly beneath it, $-\log_{10}$ of that window's Holm-adjusted $p$ against the $\alpha$ line; then a heatmap of Cliff's delta for every class pair that survived Holm, in any window. Each cell is annotated with the number of recordings behind it; a cell below `MIN_GROUP_SIZE` = 3 recordings, or one whose values are all equal, is drawn as its own points rather than as a density the smoother invented — and those are the same cells the test excludes.

**Axes.** Hours before delivery on the same $0.5$ h grid as the trajectory, inverted so delivery is at the right — on every panel including the heatmap, whose columns run in the same direction as the panels above it; nats per anchor on the violins; $-\log_{10} p$ on the strips.

**Interpretation.** **A bar that is absent and a grey cross at zero are different statements**: no bar means the window was tested and its $p$ came out at or near 1, while a cross means fewer than two classes had enough recordings there and nothing was tested. **Every heatmap row reads more severe against less severe**: `hie vs acidosis`, `hie vs healthy`, `acidosis vs healthy`, in that order down the axis and in the cohort order the violins above are drawn in — the pairwise sweep names each pair in the order it receives the classes, and it receives them worst first.

A positive Cliff's delta therefore means the *more severe* class runs higher, on every row; reorienting a pair by eye still flips its sign against the number in `time_to_delivery_pairwise.csv`. The correction is across the windows of this clock as one family, which is what makes "eight windows survived" a claim rather than an artefact of having asked twenty-two times; the two readouts are **not** jointly corrected, because they are two readings of the same recordings rather than two hypotheses.

## `time_to_delivery/time_to_delivery_windows_kl.pdf`

The same three panels for `source_conditioned_kl_raw`. Read exactly as the page above — including the row order and the sign of Cliff's delta — remembering that the KL's *level* is inflated wherever the prior variance is clamped. The Holm family is still this clock's windows: the two readouts are not corrected jointly, so a window significant on one page and not the other is one comparison each rather than one comparison twice.

## `second_stage/second_stage_trajectory_pred_gap.pdf`

**Purpose.** Follow predictive gain around the onset of the second stage of labour. Only recordings with a recorded onset can be placed on this axis.

**What it shows.** `mc_pred_gap` against `second_stage_onset / 3600`, class-stratified, as a median with an inter-quartile ribbon over **recordings** — one value per recording per window, averaged over that recording's own segments in it. Each point is annotated with the number of recordings behind it, and a dotted vertical marks the onset itself.

**Axes.** Signed hours from second-stage onset, **negative before onset and positive after**, on the same $0.5$ h grid the delivery clock uses; **not** inverted, because this coordinate reads naturally left to right. Nats per anchor.

**Interpretation.** **The sign is the opposite convention from the delivery clock's**, and the axis label says so: negative is *before* the onset. **The positive side is short by construction** — the second stage begins a couple of hours before delivery, so the windows after onset hold far fewer recordings than those before it, and the annotated $n$ is what says which is which. The population is a **subset**: recordings with no recorded onset are dropped and counted in `second_stage_eligibility.csv`, so this figure describes fewer recordings than any other in the run.

## `second_stage/second_stage_trajectory_kl.pdf`

The same figure for `source_conditioned_kl_raw` on this clock, on its own axis. Read as the page above, with the KL's clamp caveat: its level is inflated by an arbitrary factor wherever the prior variance sits on its clamp, so the shape is readable and the height is not.

## `second_stage/second_stage_windows_pred_gap.pdf`

**Purpose.** Show recording-level predictive gaps and class comparisons within windows before and after second-stage onset.

**What it shows.** Three panels of `mc_pred_gap`. A violin per (window, clinical class) cell over one value per **recording**; directly beneath it, $-\log_{10}$ of that window's Holm-adjusted $p$ against the $\alpha$ line; then a heatmap of Cliff's delta for every class pair that survived Holm, in any window. Each cell is annotated with the number of recordings behind it; a cell below `MIN_GROUP_SIZE` = 3 recordings, or one whose values are all equal, is drawn as its own points rather than as a density the smoother invented — and those are the same cells the test excludes.

**Axes.** Signed hours from second-stage onset, negative before and positive after, on the same $0.5$ h grid; not inverted, with the onset marked at zero on every panel that carries the clock. Nats per anchor on the violins; $-\log_{10} p$ on the strips.

**Interpretation.** **This clock's Holm family is its own** and is not corrected jointly with `time_to_delivery`'s: the two are different alignments of an overlapping population, so a window significant on one and not the other is a statement about alignment, and a reader quoting both is making two comparisons. **Every heatmap row reads more severe against less severe**: `hie vs acidosis`, `hie vs healthy`, `acidosis vs healthy`, in that order down the axis and in the cohort order the violins above are drawn in — the pairwise sweep names each pair in the order it receives the classes, and it receives them worst first.

A positive Cliff's delta therefore means the *more severe* class runs higher, on every row; reorienting a pair by eye still flips its sign against the number in `second_stage_pairwise.csv`. **A bar that is absent and a grey cross at zero are different statements**: no bar means the window was tested and its $p$ came out at or near 1, a cross means fewer than two classes had enough recordings there — which on the positive side of this axis is the common case rather than the exception.

## `second_stage/second_stage_windows_kl.pdf`

The same three panels for `source_conditioned_kl_raw` on this clock. Read exactly as the page above, including the row order and the sign of Cliff's delta, and with the KL's clamp caveat on the violins' heights.

## `lag_clocks/lag_time_to_delivery.pdf`

**Purpose.** Follow the location of the lag attribution as delivery approaches and compare it across classes. Read this alongside the coupling trajectories, which show the magnitude of predictive gain and KL.

**What it shows.** One heatmap per clinical class: lag down, time before delivery across, colour the **share** of the KL attribution sitting in that lag bin, over one profile per **recording** in each window. Then two panels reducing the same thing to a number — the median centre of mass across recordings with its inter-quartile ribbon, once for the attribution and once for the attention profile, with each window's recording count annotated and the median spread as a dashed line beside it.

**Axes.** Hours before delivery on the same $0.5$ h grid as the coupling clocks, inverted so delivery is at the right; lag in seconds of **stored-coefficient time**, lag $0$ at the bottom; colour is a share in $[0, 1]$.

**Interpretation.** **The class panels share one colour scale, and that is what makes them comparable** — three panels each scaled to its own extremes would paint the same colour for three different shares while every colourbar stayed correct. **It is a share, not a magnitude**: every window is normalised to sum to one, so a band moving down means the attribution moved toward the anchor, not that there is more of it — how much there is is what `time_to_delivery_trajectory_pred_gap.pdf` draws.

**A centroid is not a peak**, and this page draws no peak: the argmax and the eleven other per-segment statistics are on `lag_time_to_delivery_features.pdf` beside the guard that says whether a peak may be read at all. And the lag axis is stored-coefficient time, so a centroid that moves ninety seconds is a shift over the axis the coefficients are stored on rather than a physiological latency.

## `lag_clocks/lag_time_to_delivery_windows.pdf`

**Purpose.** Test class differences in lag centroids within each delivery-time window. A centroid is the profile's weighted average lag.

**What it shows.** Five panels. For each tested readout — the centroid of the attribution and the centroid of the attention — a violin per (window, class) cell over one value per **recording**, and directly beneath it $-\log_{10}$ of that window's Holm-adjusted $p$ against the $\alpha$ line. Then one heatmap: Cliff's delta for every class pair that survived Holm, in any window, for either readout.

**Axes.** Hours before delivery, inverted; lag in seconds of stored-coefficient time on the violins; $-\log_{10} p$ on the strips.

**Interpretation.** **This clock's two families are its own** and are not corrected jointly with the second-stage clock's, nor with each other: four families across this analysis, and a reader quoting two of them is making two comparisons. **Every heatmap row reads more severe against less severe**: `hie vs acidosis`, `hie vs healthy`, `acidosis vs healthy`, so a positive Cliff's delta means the *more severe* class's centroid sits further back in the past. A cell below `MIN_GROUP_SIZE` = 3 recordings is drawn as its own points rather than as a density the smoother invented, and those are the same cells the test excludes.

## `lag_clocks/lag_time_to_delivery_features.pdf`

**Purpose.** Describe lag-profile shape beyond its centroid, including spread, asymmetry, concentration, and peak behaviour. These additional curves are exploratory.

**What it shows.** Nine panels against the same clock, one per statistic, each the **median across recordings** with the attribution's inter-quartile ribbon: skewness, the median lag, the inter-quartile range, the entropy, the effective support, the near and far mass shares, the peak lag, and the share of segments whose peak is degenerate. On every panel the **solid line is the KL attribution and the dashed line is the attention profile**, in the same class colours as every other cohort figure. On the peak panel each point additionally carries that window's degenerate share as a percentage.

**Axes.** Hours before delivery on the same $0.5$ h grid, inverted so delivery is at the right. The $y$ axis differs per panel and is labelled per panel: seconds of stored-coefficient time for the median, the inter-quartile range, the effective support and the peak; nats for the entropy; a dimensionless ratio for the skewness; a share in $[0, 1]$ for the three mass and segment shares.

**Interpretation.** **Nothing on this page is tested.** These statistics are tabled and drawn only, which is what keeps each clock's Holm family at two; a trajectory here that looks separated is a hypothesis, and `lag_clocks_significance.csv` carries the only claims this analysis makes. **The peak panel must be read with the panel beneath it**: `entmax15` assigns lags exactly zero, so a flat or nearly empty profile still has a perfectly confident argmax, and a window whose degenerate share is high has a peak line that means nothing — the annotated percentage on each point is that number for the attribution, and the panel below carries it for both profiles.

**Solid and dashed are two readings of the same recordings, not two cohorts**: the attribution is $K_t$ times the attention and inherits the prior-variance inflation the attention is immune to, so where the two diverge the divergence is the finding. **The near and far shares are measured from the shortest lag the axis carries**, not from zero, so they mean the same thing at any causal input delay — and they do not sum to one, because the middle of the axis belongs to neither.

## `lag_clocks/lag_second_stage.pdf`

**Purpose.** Follow lag-profile location around second-stage onset using the same panels as the delivery-time version.

**What it shows.** The same panels: a share-of-attribution heatmap per class, then the centroid trajectories with their ribbons and the median spread.

**Axes.** Signed hours from second-stage onset, **negative before onset and positive after**, on the same $0.5$ h grid; **not** inverted, with the onset marked at zero. Lag in seconds of stored-coefficient time, lag $0$ at the bottom.

**Interpretation.** **The sign is the opposite convention from the delivery clock's** — negative is *before* the onset, and the axis label says so. **The positive side is short by construction**, so the windows after onset hold far fewer recordings than those before it and the annotated $n$ is what says which is which. The population is a **subset**: recordings with no recorded onset are dropped, counted in `second_stage/second_stage_eligibility.csv`, and reported in this analysis's own record.

## `lag_clocks/lag_second_stage_windows.pdf`

**Purpose.** Test class differences in lag centroids in windows aligned to second-stage onset.

**What it shows.** Five panels, exactly as the delivery clock's tested page: a violin per (window, class) cell for each tested readout, the Holm-adjusted $p$ beneath it, and Cliff's delta for every surviving class pair.

**Axes.** Signed hours from second-stage onset, not inverted, onset at zero; lag in seconds of stored-coefficient time; $-\log_{10} p$ on the strips.

**Interpretation.** **This clock's Holm family is its own**, per readout, and is not corrected jointly with the delivery clock's: the two are different alignments of an overlapping population, so a window significant on one and not the other is a statement about alignment. **Every heatmap row reads more severe against less severe**, so a positive delta means the more severe class's centroid sits further back. A grey cross at zero on a strip means fewer than two classes had enough recordings in that window — which on the positive side of this axis is the common case rather than the exception.

## `lag_clocks/lag_second_stage_features.pdf`

**Purpose.** Describe the additional lag-shape statistics on the second-stage time axis.

**What it shows.** The same nine panels as the delivery clock's features page — skewness, median lag, inter-quartile range, entropy, effective support, near and far mass share, peak lag, degenerate share — solid for the KL attribution and dashed for the attention, median across recordings with the attribution's inter-quartile ribbon.

**Axes.** Signed hours from second-stage onset, **negative before onset and positive after**, **not** inverted, with the onset marked at zero. The $y$ axis is per panel, as on the delivery clock's page.

**Interpretation.** **The sign is the opposite convention from the delivery clock's**, and the population is a **subset**: recordings with no recorded onset are not on this axis at all, and the positive side is short by construction, so a statistic that looks unstable after onset is usually a statistic computed over very few recordings. **Nothing on this page is tested**, and the peak panel must be read with the degenerate share beneath it — both for the reasons the delivery clock's features page states.

## `cross_subgroup/subgroup_heatmap.pdf`

**Purpose.** Show effect sizes for cohort comparisons and identify which pass the stated multiple-testing correction. Visual separation alone is insufficient when many groups and metrics are compared.

**What it shows.** Cliff's delta per (metric, cohort pair), with the Holm-surviving cells marked.

**Axes.** Metrics down, cohort **pairs** across; colour is the effect size, signed.

**Interpretation.** **The column order and the sign convention are both clinical.** Each column is one cohort pair named more severe first — the shared pairwise helper names a pair in the order it receives the cohorts, and this analysis hands them over worst first — so a positive Cliff's delta means the more severe cohort's values run higher, in every column. Reorienting a column by eye still flips its sign against the number in the CSV. Every test here consumes one value per **recording**; a source naming a per-segment file would test segments while reading as though it tested recordings.

## `events/conditioned_coupling.pdf`

**Purpose.** Compare predictive gain and KL shortly after detected contractions with count-matched control anchors from the same recordings.

**What it shows.** The event and control distributions per readout, and their difference per recording.

**Axes.** Nats per anchor; zero marked on the difference panel.

**Interpretation.** The contraction timing comes from the **raw pressure trace** carried through the collection pass, not from the source coefficients the model reads — a contraction exists nowhere in the tables unless that one pass puts it there — so this runs over every anchor of the split rather than only over retained samples. Gaps are masked by `weight` and never by value; an event whose span touches an interpolated region is **dropped**, because its shape partly came from that interpolation. Below 200 event anchors over 4 recordings the analysis records a skip and draws nothing.

Two readouts the raw-target pipeline draws here are **absent**, and the emitted record names both with their reasons: deceleration forecast skill and the contraction-triggered response both score a clinical heart-rate trace in beats per minute, and defining a deceleration on a channel axis with no order and no clinical unit is a new construction rather than a port.

## `sufficiency/sufficiency.pdf`

**Purpose.** Estimate the predictive cost of the latent bottleneck by comparing the base branch with an evaluation-only decoder that reads the encoder state directly. The estimate has probe and population biases, described below.

**What it shows.** $D_{\mathrm{base}}$ against $D_{\mathrm{oracle}}$ per recording, the gap's distribution, and the probe's held-out learning curve.

**Axes.** Nats per anchor; passes over the fit half for the curve.

**Interpretation.** **It is an estimate, not a bound**, and both bias directions travel in the emitted JSON: conditioning on the encoder state rather than on the target's own history omits the encoder's information loss and biases the gap **down**, while fitting the probe on the evaluation population biases it **up**. They oppose, neither is measured, and nothing downstream may treat the number as a bound. The convergence flag is a precondition rather than a decoration: an unfinished probe understates the gap, and a curve that never improved is **not** converged.

## `warmup/warmup_tertiles.pdf`

**Purpose.** Compare predictive gaps for three groups of retained channels with short, medium, and long warm-up periods. The lower panel shows attention on source coefficients that had completed warm-up.

**What it shows.** Top: `pred_gap_warm_lo`, `_mid` and `_hi` per recording, as violins, with zero marked. Bottom: `source_lag_warmth_frac_st` and `_ph` per recording.

**Axes.** Top: nats per anchor. Bottom: fraction of attention mass — a **separate axis on purpose**, because a fraction and a nats figure on one axis would flatten whichever is smaller into a line at zero.

**Interpretation.** **The three tertiles are a decomposition, not three readouts**: they sum to `pred_gap` over the same denominator, and the run asserts that they do rather than describing it — so their *relative* sizes are the finding and their absolute sizes carry the block's scale. And **a small warmth fraction in the bottom panel is the expected finding, not a fault**: the stored source blocks warm up late, so much of the searched lag window is a region where the source coefficient is still affected by warm-up.

The panel's own title says so, and a reader who treats a low value as a defect is reading the dataset's geometry as the model's behaviour.

## `warmup/causal_warmup_budget.pdf`

**Purpose.** Show which channels were retained or dropped, their warm-up requirements, and the resulting anchor floor.

**What it shows.** Two panels, one per stream (target above, source below). One horizontal bar per **declared** channel — the dropped ones included, which is the point of drawing it — laid out against a seconds axis whose origin is the anchor's own causal endpoint. A kept channel's bar spans $[-\Delta W'_c, 0]$: it *ends* at the anchor, and how far left it starts is its warm-up. A dropped channel is drawn at $\delta_c = 0$ instead, so its bar runs **forward** through the shaded forecast window it was still warming up for. The budget threshold is a dashed line, and each panel's title carries the kept-of-declared count per block.

**Axes.** Seconds relative to the anchor, negative to the left; one row per channel. The shaded span on the right is the forecast window.

**Interpretation.** **A bar starting before zero is how long that channel spent becoming honest — it is the mirror image of the two-sided cells' reading.** In a two-sided build a channel's boundary is a symmetric smear on both sides of a step; here it is a strictly leading delay, and a long bar means a slow filter rather than a broken one. Two more. **The line is drawn on the source panel too, where it is not a guard**: the source is never gated, so its bars crossing the budget is the design compromise being visible rather than a violation.

And bars that run past the right edge are **clipped and counted** in the caption, so a truncated bar is a reported clip rather than a channel that ends there. The figure is a constant of the **shard**, not of the run: two runs over the same dataset draw the identical staircase.

## `warmup/causal_warmup_tradeoff.pdf`

**Purpose.** Show the tradeoff between retaining more channels and having fewer usable anchors as the warm-up budget changes.

**What it shows.** Three step curves against the candidate budget $B$: target channels kept, anchors admitted, and tiles a training step decodes at phase $0$. The shipped threshold is marked with its three values annotated, and the region where no tile fits at all is shaded behind the curves.

**Axes.** Budget threshold $B$ in decimated steps; a count axis shared by all three curves.

**Interpretation.** Three ways. The curves are **steps, not a smooth trade**: the anchor count is computed from the **survivors' own maximum** warm-up rather than from the threshold, so a threshold of 151 keeps exactly the channels 134 keeps and admits exactly the same anchors — reading it as continuous suggests tuning room that is not there. The shaded region is **not a bad choice, it is not a choice**: no tile fits there at all. And the curve says nothing about **quality** — two budgets produce mutually unloadable checkpoints whose nats are not comparable, so this figure is about feasibility rather than about which budget forecasts better.

## `source_null/source_null_difference.pdf`

**Purpose.** Compare matched-source KL with zero-source KL. Source availability changes predictably with time, so it can affect the latent even without varying source values. This comparison measures the excess over the zero-source response; read its limitations below.

**What it shows.** Top: the distribution of `coupling_minus_clock_nats` over **recordings**, with zero marked. Bottom: `source_conditioned_kl_raw` and `kld_source_null` side by side under their own names, on one support.

**Axes.** Nats per anchor; the histogram's height is a count of recordings.

**Interpretation.** Four ways, and the first two are the point of the bottom panel.

- **A large difference between two large numbers is not the same as a large coupling.** The violins are drawn so that case is visible rather than inferred from one subtraction.
- **The reference line means "the clock accounts for all of it"**, not "no coupling". Mass at or below zero says the coupling readout is measuring an availability pattern.
- **The verdict decides, at `eval_config.clock_margin_min_nats: 0.15`.** It was left unset until a run had measured the spread it should be set from; the shipped value is that run's, and its provenance is the unaligned arm. The *number* on this figure is still the reading, and the status beside it is now a gate rather than a placeholder.
- **Zeroing floors no source variation.** The encoder's response to a flat trajectory is not literally the availability pattern's own response, so this difference is a slightly **weaker** statement than "the clock alone" — it errs in the model's favour, and the emitted record says so.
- **A large `kld_source_null` is not a defect.** Giving the prior the same clock cannot drive it to zero: the posterior is a bounded residual on the prior, so the mean half of the divergence at a silent source is a function of the delta head alone and no prior-side clock appears in it. The bottom panel's two violins overlapping heavily is the expected picture; the top panel is where the finding is.

## `source_null/source_null_lag_profile.pdf`

**Purpose.** Show where matched-source KL attribution exceeds or falls below the zero-source attribution. Their signed difference sums to the scalar source-null comparison.

**What it shows.** Top: the matched attribution and the null arm, both in nats per anchor, on the compensated lag axis. Bottom: their signed difference with zero drawn, the four `occlusion_bands` shaded and labelled, and the delta mask marked where one was emitted.

**Axes.** Compensated lag seconds across, on the shared stored-coefficient axis; nats per anchor up on both panels. Two panels because a $0.16$-nat excess drawn against a $0.49$-nat total is a flat line.

**Interpretation.** Four ways.

- **The difference is signed, and only the signed sum is the gated scalar.** It sums over lags to `coupling_minus_clock` exactly — that identity is what makes this a decomposition of the quantity `clock_margin_min_nats` gates rather than a second lag reading. Every *share* on this page is taken of the **positive part**, because a share of a signed vector is not a share, so the rectified total is an **upper bound** on the gated scalar and not a partition of it. The gap is `rectified_frac`.
- **A negative bin is a real state.** The null arm re-poses the posterior against a zeroed source, so its attention is its own and can exceed the matched arm's at a lag. That means no clock-exceeding coupling there; it does not mean negative information.
- **The delta mask is withheld when the profile is degenerate, and that is the expected outcome.** `entmax15` assigns lags exactly zero, so a flat or nearly empty profile still has a perfectly confident argmax. A withheld mask is a measurement — the geometry-fixed bands remain the selection that needs no estimate.
- **This is still stored-coefficient time.** A peak's position here is not a physiological latency, for the reason every lag page in this run carries.

## `occlusion/occlusion_horizon_delta.pdf`

**Purpose.** Measure the predictive cost of zeroing source values within each lag band and re-encoding the stream. The horizon curves show which future steps are affected. Read these intervention results alongside attention-based lag profiles.

**What it shows.** One curve per configured band: the change in block NLL against an un-occluded reference, resolved by **horizon step**. Positive is worse without the band, which means the band mattered.

**Axes.** Horizon step $0 \ldots H-1$ across; nats per anchor up, with zero marked.

**Interpretation.** Five ways.

- **A near-zero band is not always an uninformative band.** A band reaching into the warm-up has less source in it to remove; the summary's **live fraction** column is what separates "the source did not matter there" from "there was no source there". Read the two together, never the curve alone.
- **The bands are lag ranges relative to each segment's own scored anchor**, not absolute step ranges. One anchor is drawn per segment and held fixed across every band and the reference, because the source pathway has memory and a second anchor scored in the same forward would attribute one anchor's loss to another's band.
- **A negative curve is a real state, not a defect.** Removing source the model was mildly misusing improves the forecast. Whether a small negative value means anything is a question about the spread across segments, which the per-recording table carries and this figure does not.
- **The availability announcement does not move.** The intervention edits values and leaves the arrival clock exactly where it was, and that invariance is *measured* on every occluded encode rather than assumed — which is what stops this being a second reading of the clock the `source_null` figure already reports.
- **There is no threshold and no verdict here on purpose.** What a healthy per-band delta is has never been measured; a bar guessed before the first production runs would decide a pass or a fail on exactly the run that was going to set it.

## `occlusion/occlusion_clock_delta.pdf`

**Purpose.** Follow each band's occlusion cost through clinical time. A changing cost suggests that the fitted model's use of that source interval changes over the evaluated window.

**What it shows.** One panel per band, one line per clinical class: the mean per-recording delta in that band, window by window, with zero marked.

**Axes.** Hours before delivery across, drawn with delivery at the right; nats per anchor up.

**Interpretation.** Three ways.

- **Nothing on this page is tested.** No Kruskal-Wallis, no Holm correction, no new family. This analysis scores one anchor per segment and is capped in segments, so a half-hour window holds tens at best and most (class, window) cells fall below the minimum group size a test needs. A $p$-value here would be a correction over cells that mostly could not be tested.
- **The line is thin where the cap is binding.** Read `n_recordings` in `occlusion_clocks.csv` before reading a movement; the cap is chosen from the cost block and the per-band standard errors, and `EVAL.md`'s occlusion section carries the arithmetic.
- **This is the interventional half of a two-part question.** The observational half is `lag_kld_scaled`'s band trajectories, on the same partition and the same grid. Where the two disagree, that disagreement is the finding — on the fixture whose informative lags are known, they did.

## `lag_kld_scaled/lag_kld_scaled_time_to_delivery.pdf`

**Purpose.** Show the amount of KL attribution in each fixed lag band as delivery approaches. This preserves magnitude, whereas a normalised lag profile shows only relative shares.

**What it shows.** One panel per `occlusion_bands` band, one line per clinical class: the mean per-recording `total_nats` of the KL attribution restricted to that band, window by window.

**Axes.** Hours before delivery across, drawn with delivery at the right; nats per anchor up. Each panel's title carries its lag range and the seconds it spans.

**Interpretation.** Four ways.

- **Nothing on this page is tested.** Every feature in this analysis ships untested, so it adds no Holm family to the four `lag_clocks` carries and writes no significance table at all. A trajectory quoted from here is a description, not a claim.
- **The bands are geometry-fixed, and that is what makes them readable.** They are not chosen from the KL, so a statistic on one is free of the circularity that makes a top-$K$-by-KL selection test its own selector. They are the same bands the interventional page removes source from.
- **A line moving *between* panels is the informative past moving; a line moving *within* one is that band's coupling changing magnitude.** Neither is visible on a scale-free statistic over the whole window, which is what `lag_clocks` draws — the two pages answer different halves.
- **`near_mass` and `far_mass` are absent from banded sources on purpose.** Both are measured from the axis's own start, so on a band they would re-base onto the band's start and `far_mass` would be identically zero on three of the four. Their absence is a measurement.

## `lag_kld_scaled/lag_kld_scaled_second_stage.pdf`

**Purpose.** Show the same band-resolved KL magnitudes against signed hours from second-stage onset: negative before onset and positive after.

**What it shows and its axes.** As the delivery-clock page above, on the second-stage axis drawn left to right with the onset where it falls.

**Interpretation.** As above, plus one. **The population is a strict subset**: a recording with no recorded onset cannot be placed on this axis at all, so this page and the delivery-clock page answer for different recordings by design. The analysis declares itself capped for that reason, and the eligibility rule is the shared one — the same the `second_stage` analysis applies.

## `lag_high_kl/lag_high_kl_selection.pdf`

**Purpose.** Show how anchors are selected by KL or predictive gain, and compare their pooled lag profiles. The thresholds define which observations contribute to later figures.

**What it shows.** Four panels. **Top:** the pooled per-anchor KL, one step histogram per clinical class on a $\log_{10}$ axis, with the band thresholds drawn as dashed lines and labelled with their quantile and their value in nats — the histogram the bands were cut on. **Second:** the pooled KL attribution per lag for the `high`, `rest` and `top` anchor bands, with the **hot lags** — the lags whose pooled attribution sits in the upper $30\%$ across the lag axis — shaded behind them.

**Third:** a heatmap of where each anchor's attribution peaks, by KL decile (lowest at the bottom) and lag: the share of that decile's anchors whose `argmax_lag` falls at each lag. **Bottom:** one violin per class of the per-recording contraction enrichment — the high-anchor share within `event_lag_window_s` of a contraction minus the share outside it — over recordings with at least five anchors in each arm, zero marked.

**Axes.** Panel 1: $\log_{10}$ nats across, density of anchors up. Panels 2 and 3: stored-coefficient lag in seconds across; nats per anchor up in panel 2, KL decile up in panel 3. Panel 4: class across, share difference up.

**Interpretation.** Five ways.

- **The thresholds are one number per run, pooled over every class.** A class whose histogram sits to the right has more high anchors *under the same cut*; nothing here re-cuts per class, and that is what makes the class shares comparable.
- **The hot-lag set is selected from the attribution it shades.** A pooled profile that is high on the hot lags is a tautology, not a finding; what the set is for is the per-segment share placed on the clocks, and even there it describes the run's own selection. The geometry-fixed bands on the `lag_kld_scaled` page and the `occlusion` page need no such estimate.
- **A decile heatmap that is flat across rows is the geometry talking.** If the top decile peaks at the same lags as the bottom one, the argmax is a property of the window's censoring edges and the K/V receptive field rather than of the coupling; read the `attention` and `source_null` pages before reading a peak position off this one.
- **The enrichment violin is descriptive.** No test is run on it, and a recording enters it only with enough anchors in both arms, so its population is a subset that the record counts.
- **The lag axis is stored-coefficient time**, as the foot of the page says: a lag here is not a physiological delay.

## `lag_high_kl/lag_high_kl_time_to_delivery.pdf`

**Purpose.** Follow the lag profiles and prevalence of high-KL anchors as delivery approaches. This focuses on anchors where adding source history moved the latent most strongly.

**What it shows.** One heatmap per class — lag down, window across, colour the **share** of the high band's KL attribution in that lag bin, every window normalised to one and the three classes on one colour scale — then four trajectory panels, each the median per class with its inter-quartile ribbon and the recording count on every point: the **high-anchor share** of a segment's anchors, with the `top` band's share dashed; the **high band's KL centroid** in seconds, with the `rest` band's centroid dashed beside it; the **hot-lag share** of the attribution, with the attention's dashed; and the **high band's forecast gain** in nats per anchor, with the `rest` band's dashed.

The first two are the tested readouts and their titles say so; the last two are untested per window (the gain's one paired run-level test is on the usefulness page).

**Axes.** Hours before delivery across, drawn with delivery at the right; stored-coefficient lag in seconds up on the heatmaps and the centroid panel; a share of anchors, or of the attribution, up on the other two.

**Interpretation.** Four ways.

- **The heatmap is a share, not a magnitude.** It answers *where* the high anchors' attribution sat, never how much of it there was; the high band's `total_nats` is on the trajectory table and the amount of coupling is what `time_to_delivery` draws.
- **A centroid that separates the classes is a hypothesis until the windows page says otherwise.** The ribbon is a quartile range over recordings, not an interval on the median, and the test lives on `lag_high_kl_time_to_delivery_windows.pdf`.
- **The high-anchor share is measured against a pooled threshold.** A class whose share rises toward delivery has more anchors clearing the *run's* cut, not a cut of its own; and the pooled share over the whole population is $30\%$ by construction, so only its distribution over classes and windows carries information.
- **The `rest` centroid beside the `high` one is the contrast that matters.** If the two move together the lag structure is not specific to the coupling; if only the `high` one moves, the anchors carrying the coupling looked somewhere the others did not.

## `lag_high_kl/lag_high_kl_time_to_delivery_windows.pdf`

**Purpose.** Test class differences in the two selected high-KL readouts within each delivery-time window.

**What it shows.** For each of `high_lag_centroid_kl_s` and `high_anchor_frac`: one violin per (window, class) cell over one value per recording, the Holm-adjusted $p$ of each window directly beneath it on the same axis, and — at the foot — Cliff's delta for every class pair that survived, oriented more severe against less severe so a positive value means the more severe class runs higher.

**Axes.** Hours before delivery across, delivery at the right. The violin rows share one $y$ label because the page carries two readouts in two units: seconds for the centroid, a share of anchors for the fraction; each row's title names which it is.

**Interpretation.** Two ways. **Each (clock, readout) is its own Holm family** — two here, two on the second-stage page, none joint — so a reader quoting a window from two of them has made two comparisons. And **a cell too small to test is still drawn**: classes with fewer than three recordings in a window are excluded from the test and recorded, but their violins stay on the page so a cohort thinning toward the edge of the axis does not simply vanish.

## `lag_high_kl/lag_high_kl_second_stage.pdf`

**Purpose.** Show the high-KL profiles and trajectories against signed hours from second-stage onset.

**What it shows and its axes.** As the delivery-clock page, on the second-stage axis drawn left to right with the onset marked where it falls.

**Interpretation.** As above, plus one. **The population is a strict subset**: a recording with no recorded onset cannot be placed on this axis at all, so this page and the delivery-clock page answer for different recordings by design, under the shared eligibility rule the `second_stage` analysis applies.

## `lag_high_kl/lag_high_kl_second_stage_windows.pdf`

**Purpose.** Test the same two high-KL readouts on the second-stage clock, with separate multiple-testing families for this clock.

**What it shows and its axes.** As `lag_high_kl_time_to_delivery_windows.pdf`, on the signed second-stage axis with the onset marked at zero.

**Interpretation.** As that page, and with the same population caveat as the second-stage profile page above.

## `lag_high_kl/lag_high_kl_time_to_delivery_histogram.pdf`

**Purpose.** Compare full lag distributions for selected high-KL anchors across classes and delivery-time windows. Distribution shape can reveal differences that a centroid alone hides.

**What it shows.** Four blocks, both profile sources side by side in every row — `kl` on the left, `attn` on the right. **Rows 1–6, three per band** (`high` then `top`), pooled over every window of the clock: **the classes overlaid** as outlines only, one step curve each, with the recording-weighted pool of every class dashed in grey behind them and each class's delivery count in the legend; **each class minus that pooled distribution**, in percentage points of share per lag with zero marked — the panel a class contrast is actually read on, because three monotone decays that coincide to within a few percent of their peak are indistinguishable overlaid and their differences from a common reference are not; and **the cumulative distributions**, on which a shift between skewed distributions is a horizontal offset readable in seconds, with each class's median lag dropped to the axis as a dotted line.

**Row 7, the density violins:** the `high` band's cells on the clock, one body per (window, class) dodged by class inside each window. The body is **the cell's own lag distribution** — its half-width at a lag is that lag's share — drawn as a step outline because a share is per lag bin; it is not a kernel density of anything. Inside each body the heavy bar spans the distribution's quartile lags, the white dot is its median lag and the short black tick its centroid, which on these skewed profiles sits well away from the median.

The recording count stands above every body; a cell below three recordings is faint and dashed rather than absent. **Row 8, the ridges:** the same cells with the lag across, one ridge per window, **one column per class** rather than the classes overlaid — three filled decays of nearly the same shape on one baseline blend into a colour no legend explains — with **labour running down the page** on both clocks (farthest from delivery at the top).

Inside every ridge the class's own distribution pooled over the whole clock is outlined in grey and dashed, so a ridge that leaves its outline has moved and a column whose ridges all sit on it is a stationary class; the same window sits at the same height in every column, so a horizontal read compares the classes. The window centre is labelled on the left of the first column, the recording count on the right of each ridge, and the cell's median lag is ticked on the baseline. Every body on a panel is drawn at one scale, so widths and heights compare across windows and classes.

**Bottom two rows:** the distances against the clock, **metric down and comparison across**. The left column is each window against its own class pooled over the whole clock, one line per class in the severity colours — a cohort whose lag structure is stationary sits flat and near zero. The right column is the distance between the classes within each window, one line per class **pair**; a pair belongs to two cohorts and to neither alone, so those lines use a separate blue/purple/black palette that is none of the class green, amber or red, and the pair is read off the legend rather than off the hue.

The upper of the two rows is the $1$-Wasserstein distance **in seconds**, which says how far the distribution moved; the lower is the Jensen–Shannon distance, which says how much overlap is left.

**Axes.** Stored-coefficient lag in seconds across on the pooled panels and the ridges; share of the distribution up on the overlay, percentage points of share on the difference panel, cumulative share on the third, and window centre up on the ridges. Hours before delivery across on the density violins and on all four distance panels, drawn with delivery at the right; lag in seconds up on the violins; seconds up on the Wasserstein row, and Jensen–Shannon (base $2$, bounded by $1$) up on the row below it.

**Interpretation.** Six ways.

- **It is a distribution per recording, then averaged — not a pooled histogram.** Each recording is normalised before the cell mean, so a recording carrying ten times the coupling of its neighbours counts once, not ten times. This is why the shape here can differ from the heatmap on `lag_high_kl_time_to_delivery.pdf`, which normalises after averaging and therefore reports where a cohort's coupling *mass* sits. Neither is wrong; they answer different questions.
- **Nothing on this page carries a $p$-value.** The violins and ridges are the cells themselves and both distances are descriptive; a distance between two estimated distributions is positive almost surely even when the two populations coincide, so a non-zero value is not evidence of a difference — read it beside the recording counts on the row. The tests on these cells' shape features are on `lag_high_kl_time_to_delivery_histogram_features.pdf`, and the within-recording drift on `lag_high_kl_time_to_delivery_histogram_drift.pdf`.
- **A density violin's width is a share, and its inner marks are the distribution's own.** The bar is the quartile *lags* of the cell's distribution, not the spread of the recordings behind it; the spread over recordings is what the features page's violins show. Two cells with the same body can stand on three recordings or thirty, which is why the count is printed above each.
- **Neither distance row answers the other's question, so a column is read down, not alone.** Wasserstein measures displacement along the lag axis and is blind to a distribution that broadened or split *in place*; Jensen–Shannon measures overlap and is blind to displacement once two supports have separated, at which point it sits at its ceiling of $1$ however far apart they are. A cohort whose distribution widened toward delivery without shifting reads as near zero on the upper row and plainly non-zero on the lower one, and a single row would lose that finding.
- **The two columns are not a consistency check, they are the finding.** `attn` counts every selected timestep once; `kl` weights each by how far the source moved the belief there. A shift visible in one column and absent from the other says which readout is carrying it.
- **A missing body or ridge is a window with no scored recording for that class**, not a window where the distribution collapsed; the dodge keeps every class's slot, so a gap stays a gap.
- **The lag axis is stored-coefficient time.** A distance quoted in seconds is a displacement on that axis and not a change in physiological latency; the caveat at the foot of the page applies to every number on it.

## `lag_high_kl/lag_high_kl_second_stage_histogram.pdf`

**Purpose.** Compare selected-anchor lag distributions across classes and windows aligned to second-stage onset.

**What it shows and its axes.** As `lag_high_kl_time_to_delivery_histogram.pdf`, with the density violins and the four bottom distance panels on the signed second-stage axis and the onset marked at zero; the ridgeline still runs earliest-before-onset at the top.

**Interpretation.** As that page, and with the same population caveat as the second-stage profile page above: the recordings scored here are those carrying an onset, a strict subset of the cohort, so a class's distribution on this page and on the delivery page are over different populations.

## `lag_high_kl/lag_high_kl_subgroup_histogram.pdf`

**Purpose.** Compare high-KL lag distributions across classes and their subgroups, pooled over the evaluated population without time windows.

**What it shows.** The `high` band pooled over the **whole** evaluated population: each recording's selected-anchor profile averaged over every one of its segments, normalised once, then averaged over the recordings of the cohort. One column per clinical class; inside each, that class's subgroups as tints of the class colour with the class's own pooled distribution as a black dashed reference, so a subgroup is read against its class rather than against the population. Two rows per profile source (`kl` then `attn`): the distributions, and their cumulative forms with the medians dropped to the axis. The legend carries each cohort's delivery count.

The table beside it, `lag_high_kl_subgroup_histogram.csv`, carries the same curves on both cohort axes.

**Axes.** Stored-coefficient lag in seconds across; share of the distribution up on the first row of each pair, cumulative share on the second.

**Interpretation.** Three ways. **Nothing here is tested**: the curves are descriptive, and a subgroup of six recordings draws as confidently as one of sixty — read the count. **It is population-pooled, so it is not the delivery-clock page's pooled row**: that row pools one clock's binned recordings, this one every segment within the run's horizon, and a recording enters both once. And **the lag axis is stored-coefficient time**, as the foot of the page says.

## `lag_high_kl/lag_high_kl_time_to_delivery_histogram_features.pdf`

**Purpose.** Test class differences in three properties of each recording's lag histogram: median lag, interquartile range, and entropy. The tests run within delivery-time windows.

**What it shows.** For each of `hist_median_s`, `hist_iqr_s` and `hist_entropy_nats`: one violin per (window, class) cell over one value per recording, the Holm-adjusted $p$ of that window's Kruskal–Wallis directly beneath it on the same axis, and — at the foot — Cliff's delta for every class pair that survived, oriented more severe against less severe so a positive value means the more severe class runs higher. Exactly the page `lag_high_kl_time_to_delivery_windows.pdf` is, for three more readouts.

**Axes.** Hours before delivery across, delivery at the right. The violin rows share one $y$ label because the page carries three units: seconds for the median and the range, nats for the entropy; each row's title names which it is.

**Interpretation.** Three ways. **These are three more Holm families per clock**, six across the two clocks, and they are not corrected jointly with the four the analysis already defends; a reader quoting a window from two families has made two comparisons. **The centroid is deliberately not here**: `high_lag_centroid_kl_s` already tests the position of the same selection on the same band and source, and a second family on it would ask one question twice. And **the features are of the `kl` source alone** — the attention source's features are in `lag_high_kl_histogram_features.csv` and drawn on the histogram page, untested.

## `lag_high_kl/lag_high_kl_second_stage_histogram_features.pdf`

**Purpose.** Test the same three histogram features on the second-stage clock, using this clock's own correction families.

**What it shows and its axes.** As `lag_high_kl_time_to_delivery_histogram_features.pdf`, on the signed second-stage axis with the onset marked at zero.

**Interpretation.** As that page, and with the same population caveat as the second-stage profile page above.

## `lag_high_kl/lag_high_kl_time_to_delivery_histogram_drift.pdf`

**Purpose.** Measure how lag-histogram features change within each recording as delivery approaches. This helps distinguish within-recording change from a changing mix of recordings across windows.

**What it shows.** One row per tested feature (`hist_median_s`, `hist_iqr_s`, `hist_entropy_nats`). **Left:** every recording's own trajectory of the feature along the clock, thin in its class colour, with the class median per window heavy over it — so a moving median can be read as many recordings moving together or as a few entering and leaving. **Right:** the least-squares slope of the feature against **forward labour time**, one value per recording scored in at least three windows, as one violin per class with zero marked.

The title carries the tests: the Holm-adjusted Kruskal–Wallis $p$ across classes, each class's Holm-adjusted Wilcoxon signed-rank $p$ against zero, and Cliff's delta for any class pair that survived.

**Axes.** Left: hours before delivery across, delivery at the right; the feature's unit up. Right: class across; the feature's unit **per hour of labour** up, zero marked.

**Interpretation.** Four ways.

- **Positive means the feature rises as delivery approaches, on both clocks.** The delivery clock counts backwards, so its centres are negated before the fit; a slope here and a slope on the second-stage page carry the same sign for the same drift. The raw window centres are on `lag_high_kl_histogram_drift.csv` for a reader who wants the fit's inputs.
- **Two families per clock, each Holm across the three features.** The class-against-zero tests are one family; the across-class tests are another; pairwise runs on survivors only. Neither is joint with the per-window families.
- **A recording scored in fewer than three windows is absent**, not at zero — the count in each class's tick label is the fitted recordings, which is a subset of the class.
- **A slope is a straight line through a trajectory that need not be straight.** A distribution that moves out and back reads as no drift; the left panel is where that would be seen.

## `lag_high_kl/lag_high_kl_second_stage_histogram_drift.pdf`

**Purpose.** Measure within-recording changes on the second-stage clock for recordings with a recorded onset.

**What it shows and its axes.** As `lag_high_kl_time_to_delivery_histogram_drift.pdf`; the left panels are on the signed second-stage axis, not inverted, with the onset marked at zero, and the slopes are already in forward time so the right panels read identically.

**Interpretation.** As that page, and with the same population caveat as the second-stage profile page above. A recording with few windows on either side of the onset spans a short clock here even when it spanned a long one on the delivery clock, so its slope is the noisier of the two.

## `lag_high_kl/lag_high_kl_usefulness.pdf`

**Purpose.** Check whether anchors with large KL also show improved prediction. Large KL means the latent changed; this figure tests whether that change was useful for forecasting.

**What it shows.** Four panels. **Top:** the forecast gain $D_{\mathrm{base}} - D_{\mathrm{full}}$ of an anchor against the KL decile of the same anchor — the median per recording with its inter-quartile ribbon, pooled in black and one line per class, zero marked. The title carries the run-level paired test: the high band's mean gain minus the rest band's within recording, its Wilcoxon $p$ and the share of recordings in which it is positive. **Second:** the mean gain by the lag the anchor's KL attribution peaks at, for every anchor and for the `high`, `rest` and `gain` bands.

**Third:** the pooled lag profiles as shares — the all-anchor KL attribution, the attention weighted by positive forecast gain (where the source looks *when it helps*), and the `gain` and `high` bands' attributions — with the high–gain overlap in the title against the $30\%$ independence would give. **Bottom:** per recording and per `occlusion_bands` band, the share of KL attribution inside the band against the forecast cost of occluding it, one colour per band, zero marked; or the note that the interventional pass did not run in this directory.

**Axes.** Panel 1: KL decile across (0 = lowest), nats per anchor up. Panel 2: stored-coefficient lag in seconds across, nats per anchor up. Panel 3: lag across, share of the profile up. Panel 4: share of the recording's attribution across, occlusion delta in nats up.

**Interpretation.** Five ways.

- **A rising top panel is the finding, a flat one is the other finding.** If the gain does not rise with the KL decile — or the high-minus-rest difference is not positive — the coupling readout is measuring something the forecast does not use, however sharp the lag profile looks. The availability clock is the standing candidate for what it is measuring instead; read `source_null`.
- **The paired test is one test, its own family.** It is not corrected with the four clock families, and it says nothing about *which* lags helped; panels 2 to 4 are where that is read, and they are descriptive.
- **Panel 2 conditions on the argmax, which the near edge can pin.** A lag with many anchors and a small gain is a censoring artefact as readily as a finding; read it beside `n_anchors` in `lag_high_kl_gain_by_argmax.csv` and beside the degeneracy share on the clock pages.
- **The gain-weighted profile is not a second lag detector.** It weights the same attention by the same run's forecast gain, so it can only redistribute mass the model already put somewhere; a profile identical to the KL-weighted one means the useful anchors look where all anchors look.
- **The bottom panel's agreement is a property of the run, not of the method.** On the planted-delay fixture the two readings disagreed; a positive $\rho$ here says the observational and the interventional answers coincide for this checkpoint, and nothing more.

## `spectral_skill/spectral_skill_bands.pdf`

**Purpose.** Compare forecast performance across bands defined by the target coefficients' analysing filters. The bands describe the input transform, not a new spectrum computed from the forecast.

**What it shows.** Top: the forecast gap per recording, one violin per band, each labelled by the band's **frequency range with the period in parentheses** (not its clinical name) and with its **channel count in its label**. Bottom: the error-space skill of the source-conditioned branch against the target-only one, per band.

**Axes.** Top: nats per anchor, zero marked. Bottom: $1 - \mathrm{MSE}_{\mathrm{full}}/\mathrm{MSE}_{\mathrm{base}}$, zero marked. Two axes because the two are in different units and one shared axis would flatten whichever is smaller into a line at zero.

**Interpretation.** Four ways.

- **This is band-resolved skill, not coherence.** A stored coefficient is a *modulus*: the analysing filter's phase was discarded before the value was written, so nothing here can say whether a forecast is mistimed rather than mis-scaled. A forecast that is right in every band but arrives a step late reads here as a forecast that is right.
- **The band is the band of the analysing filter**, not a bin of the forecast's own spectrum. Those are two different objects.
- **The channel counts in the labels are load-bearing.** A band carried by three channels and one carried by forty are not comparable as evidence, and the label is where that shows.
- **`unknown` is a band with no frequency, not a leftover.** Three of the 98 scored channels have no recoverable centre frequency because no selected phase-harmonic pair named their filter, and they are reported under their own label rather than bucketed into a neighbour — which would misattribute their skill to a frequency they do not have.

## `recording_traces/recording_traces_summary.pdf`

**Purpose.** Follow selected recordings through their segments up to delivery, and read each class's typical course against them. These examples support visual exploration rather than population-level testing.

**What it shows.** A coverage row, then one panel per per-segment summary column — the divergence $K_t$, the mean-decoded forecast gap, the source shift of the latent mean $\lVert \mu^q - \mu^p \rVert_2$, the active coordinate count, the lag centroid of the KL attribution and its entropy — all on one shared axis of hours before delivery with delivery on the right. Every recording is a thin line in its **class** colour with one marker per segment, lifted at a break (a gap longer than one segment stride), so a missing hour is a hole rather than a slope. Over the lines, the **class median** per half-hour window is drawn bold, with the inter-quartile band over recordings wherever at least three recordings fall in the window; each recording contributes one value per window (the mean of its segments there), so a recording with many segments weighs the same as one with a single segment. The coverage row counts the recordings each window holds per class, which is what a median rests on. When `max_hours_before_delivery` is set the axis is bounded to it and the footnote states how many segments lie beyond.

**Interpretation.** These are up to `eval_config.caps.traces_per_class` recordings per class, drawn for looking at rather than for testing: a class whose median sits higher is a hypothesis, and `cross_subgroup` and the clock analyses are where such a difference is asked properly, on every recording, with a correction. A median over three recordings is a median of three; read the coverage row before the band. Every value is a mean over the segment's **scored** anchors; a segment that scored none is absent from its line rather than drawn at zero. The lag centroid is in stored-coefficient time.

## The per-recording traces: `recording_traces/<class>/<guid>_<subgroup>_trace.pdf`

**Purpose.** Inspect every available segment of one recording at anchor resolution, on a shared axis of hours before delivery. The filename identifies both the recording and its subgroup.

**What it shows.** Ten rows on one shared axis of hours before delivery, delivery on the right, every model row at the anchor step so a column is the same anchor on every row: the divergence $K_t$; both `pred_gap` estimators, **joined from the collection pass** at those anchors and therefore absent wherever the pass did not collect the segment; the posterior mean $\mu^q$ over the latent coordinates and the source shift $\mu^q - \mu^p$ on a symmetric colour scale, as heatmaps; the per-coordinate divergence; the KL attribution over lags on a **logarithmic** colour scale with the peak lag drawn over it; the latent norms; the mean log-variances of prior and posterior; the lag centre of the attribution (centroid and median, in seconds); and the lag entropies of the attribution and of the attention. The colour axis of every heatmap row sits in its own column, so every row's data axes span the same hours.

On the $K_t$ and forecast-gap rows a black step marks each segment's mean over its scored anchors — the value the summary figure carries for that segment — so the two figures can be read against each other. Segments alternate a faint background on the line rows; a stretch of the recording the dataset holds no segment for (a **break**, a gap longer than one segment stride) is shaded darker on every row; and a line row is lifted at every unscored anchor so nothing is drawn across a gap. Where the recording carries a clinical clock, its onset is ruled across the page: labour onset dashed, second-stage onset dotted, each labelled on the first row. The title carries the GUID, the subgroup, the class, the segment count, the span and the break count.

**Interpretation.** Four ways. *A step at a join is geometry*: segments are separate forward passes with a reset encoder state, so a discontinuity at a marked join is a property of the harness rather than of the fetus. *A smooth path is not evidence of smooth physiology*: there is no latent transition density, and consecutive anchors are smooth because the encoders are. *The lag axis is stored-coefficient time*, and the caveat printed under the figure applies to every lag row. *The colour scales are per recording*: two recordings' heatmaps cannot be compared by colour, only the lines and the summary figure can.

The full arrays behind every panel are in the `_full.npz` beside the figure.

## `attribution/attribution_maps.pdf`

**Purpose.** Show the coefficients the encoders read at one example anchor per clinical class, and which of them the divergence was attributed to. The example is the middle chosen anchor of the first selected recording of each class.

**What it shows.** One row per class, five columns. First the **inputs**: the declared target stream (the scattering block above the phase-harmonic block, divided by a rule) and the declared source stream, standardised coefficients over stored step and declared channel. Then the attribution of $K_t$: to the target stream under the **all-zero** baseline — the only baseline along which the target inputs move, so the only one under which a target map exists — and to the source stream under the **source-null** baseline, the primary comparison. Last, the source attribution re-indexed by offset from the anchor, as a share per lag, against the model's own lag readout at the same anchor normalised the same way, with their correlation and Jensen–Shannon distance in the title. Every map carries the anchor as a vertical rule and each channel's warm-up boundary as a staircase; the row label carries the GUID, the subgroup, the class, the anchor step and its hours before delivery.

**Axes.** Stored step across; declared channel down; standardised coefficient (inputs) or readout units (attribution) in colour. The lag panel is in stored-coefficient seconds.

**Interpretation.** Four ways. *Every cell after the anchor is exactly zero* on a causal model and is drawn, not cropped — a non-zero there is a defect, not a finding. *The map is one anchor of one segment of one recording*, drawn for looking at; the population reductions are the other figures, and the per-class example pages below give the same anchor under every readout. *The colour scale is per panel*, so two panels cannot be compared by colour, only by the totals in `attribution_rows.csv`. *A cold channel's cells are zero because the gate multiplied them by zero*, not because the model found nothing there. The caveat under the figure applies to every lag panel.

## The per-class example pages: `attribution/maps/<class>_<guid>_<subgroup>_anchor<step>_attribution_maps.pdf`

**Purpose.** Put the input coefficients and every readout's attribution of them on one page, for the same anchor: what the model read, and what in it moved the latent, the forecast and the lag readout.

**What it shows.** One page per class example, three columns. The first row is the input: the target and source coefficients as on the overview, and the model's own lag readout at the anchor in its own units. Every further row is one readout — the divergence $K_t$, the forecast gap (base minus full), the full-branch block score, then the lag readout on each configured `occlusion_bands` band — with its target attribution under the all-zero baseline, its source attribution under the source-null baseline, and the lag-aligned source attribution against the model's lag readout, both as shares per lag, with the readout's value at the input and at the exact null printed in the corner. The page title carries the GUID, the subgroup, the class, the anchor step and its hours before delivery.

**Axes.** As the overview: stored step across, declared channel down; the lag panels in stored-coefficient seconds.

**Interpretation.** Read down a column to see whether the same coefficients drive different readouts, and across a row to see where in stored time and in which channels one readout was sensitive. The sign matters and differs by readout: a positive cell raises the readout along the path, which is a larger latent change for $K_t$, a larger advantage of full over base for the gap, and a *worse* forecast for the block score. The all-zero and source-null columns are attributions along different paths from different baselines and do not sum to anything together; the lag-band readout's own attribution is expected to concentrate on its band, and how far it does is what the last rows show. `attribution_maps.npz` carries every map on the pages, keyed by example, readout, baseline and band, beside the inputs.

## `attribution/attribution_lag_profile.pdf`

**Purpose.** Compare source-input attribution over lag offsets with the model's own lag readouts at the attributed anchors.

**What it shows.** One row per main readout, source-null baseline. Left: the mean over recordings of the normalised $|q_\ell|$ beside the mean normalised model profile, pooled, with the recording-mean Pearson correlation and Jensen–Shannon distance in the title. Right: the same by class, solid for the attribution and dashed for the model readout, in the severity colours, with the recording count per class in the legend. A last row puts every readout on one lag axis: left, the normalised source attribution of the divergence, of the forecast gap and of the lag readout on each configured band, pooled over recordings, with the model's lag readout dashed in black; right, the lag-band readouts alone, each against its own band shaded in its colour.

**Axes.** Lag in seconds, stored-coefficient time; share per lag.

**Interpretation.** The two profiles are different objects that happen to share an axis: the model readout is $K_t$ times the attention, an allocation; the attribution is a sensitivity of $K_t$ to the source coefficient at that offset, which can be negative and is drawn by magnitude. Agreement is a finding about the model's self-description, disagreement is not a fault in either. On the comparison row, the readouts are sensitivities of different quantities to the same inputs, so where their profiles coincide the same stored source time moves the latent, the forecast and the lag readout together, and where they part it does not; a band readout whose attribution sits outside its own band is a readout the model computes from beyond the band it names. The axis is stored-coefficient time, uncorrected for the composed group delay. Up to `eval_config.caps.attribution_segments` recordings, one segment each.

## `attribution/attribution_bands.pdf`

**Purpose.** Summarise input attribution by frequency band and source lag band, and compare it with related predictive and intervention analyses.

**What it shows.** Top, one column per readout: the mean over recordings of the source-null attribution summed over each frequency band of the declared channel map, one bar group per stream, with `spectral_skill`'s per-band gap drawn on a twin axis for the target bands where that pass ran. The ticks name each band by its frequency range with the period in parentheses; the `band` column of the table keeps the key.

Bottom: per `occlusion_bands` band of the source relative to the anchor, the integrated-gradient sum over the band, this analysis's own feature-ablation delta of the readout, and the `occlusion` pass's delta where it ran — sign-flipped on the gap readout, because that pass reports the forecast cost of removing a band and the gap moves the other way.

**Axes.** Bands across; readout units up (nats per anchor for both readouts; the occlusion series in nats per anchor).

**Interpretation.** A frequency-band sum over an input stream is over the **declared** channels of that stream, including the ones the budget dropped (their attribution is exactly zero) — the channel counts are in `attribution_bands.csv`. The ablation delta and the occlusion delta are the same intervention at different anchors and on different readouts, so they agree in sign rather than in value. Nothing here is tested.

## `attribution/attribution_layer.pdf`

**Purpose.** Show the attention-head contributions to the attributed output and the inputs associated with the latent coordinate carrying the largest divergence.

**What it shows.** Left: the layer integrated gradients on the head-structured posterior's per-head fusion modules, summed over each head's feature at the anchor — one bar per head per readout, mean over recordings, source-null baseline. Right: for the anchor's largest per-coordinate divergence $K_{t,d}$, the magnitude of its attribution by offset from the anchor, per stream.

**Axes.** Head across (left); lag in stored-coefficient seconds across (right); readout units up.

**Interpretation.** The per-head split sums to the readout difference along the source-null path only; under the all-zero baseline the prior's direct route into the posterior bypasses the fusion, which is why the split is not taken there. A flat (non-head-structured) posterior has no split and the panel is empty. The top coordinate is chosen per anchor, so the right panel pools anchors whose coordinate differs.

## `attribution/attribution_null.pdf`

**Purpose.** Separate the exact zero-source response, integrated source-content attribution, and the entry jump caused by moving away from the zero baseline.

**What it shows.** Left, per class: the mean over recordings of $K_t$ at the input, at the exact null (the availability-clock part), the integrated source-content attribution and the entry jump, for the divergence under the source-null baseline. Right: under the all-zero baseline, the split of the attribution between the target and the source streams, and the entry jump.

**Axes.** Class across, with the recording count; nats per anchor up.

**Interpretation.** The four bars on the left are not a partition: input equals null plus entry jump plus attributed content only up to the completeness residual, which `attribution_summary.csv` reports. The entry jump is a property of the encoders' normalisation at the exactly-zero point, not of any input step. Every class contrast is out of distribution.

## `attribution/attribution_channels.pdf`

**Purpose.** Show which declared input channels each readout responded to, summed over stored time — the channel marginal of the maps.

**What it shows.** One row per main readout; the target stream (all-zero baseline) left, the source stream (source-null baseline) right. Bars are the mean over recordings of the **signed** channel profile, the sum of a channel's attribution over stored steps; the line is the mean of the **unsigned** one, the sum of absolute values, which says how much a channel mattered when its contributions cancelled over time. Bars are coloured by the channel's frequency band where the run wrote `band_channel_map.csv` (the legend names the bands by their frequency range), grey otherwise. On the target stream the scattering block sits left of the dashed rule and the phase-harmonic block right of it.

**Axes.** Declared channel across; attribution in readout units up.

**Interpretation.** A channel with a tall line and a short bar pulled the readout both ways at different steps; a channel whose bar and line coincide pulled one way. Channels the warm-up budget dropped are exactly zero, because the gate multiplied them by zero. The two streams are on different baselines and different vertical scales, so they are compared by shape, not by height. The frequency-band figure sums these same profiles over bands.

## `attribution/attribution_lag_channel.pdf`

**Purpose.** Show which channel at which offset from the anchor each readout responded to — the full map that the lag profile and the channel profile are the marginals of.

**What it shows.** One row per main readout: the target stream re-indexed by offset from the anchor under the all-zero baseline (left) and the source stream by lag under the source-null baseline (right), each the mean over attributed anchors of the **unsigned** attribution at that offset and channel. The scattering/phase-harmonic boundary is ruled on the target map. `attribution_lag_channel.npz` carries the signed means beside the unsigned ones.

**Axes.** Offset from the anchor in stored-coefficient seconds across; declared channel down; mean unsigned attribution in colour, one scale per panel.

**Interpretation.** A peak here says both what the marginals say separately: which coefficient, how far back. A row that is bright across every offset is a channel the model reads throughout its window; a column that is bright across many channels is an offset the model reads on every channel. The mean is over attributed anchors (a few per segment, one segment per recording), not a per-recording mean, so a recording with more scored anchors weighs slightly more. The axis is stored-coefficient time.

## `attribution/attribution_time_profile.pdf`

**Purpose.** Show the sign of the attribution by offset from the anchor, which the magnitude profiles discard.

**What it shows.** One row per main readout; the source stream (source-null baseline) left and the target stream (all-zero baseline) right, by offset from the anchor. The shaded area above the axis is the mean over recordings of the **positive** part of the per-offset attribution, the area below it the mean of the **negative** part, the black line the net mean and the dashed line the unsigned mean.

**Axes.** Offset in stored-coefficient seconds across; attribution in readout units up.

**Interpretation.** A positive value raises the readout along the path: a larger latent change for $K_t$, a larger advantage of full over base for the gap. An offset whose two parts are both large and whose net is near zero is one where channels pull the readout both ways; the unsigned line measures that. A net that changes sign along the axis is a model that reads the recent past and the further past in opposite directions. The axis is stored-coefficient time.

## `attribution/attribution_checks.pdf`

**Purpose.** Put the numerical checks behind every attribution row on one page, so a map is read after its residual and not before.

**What it shows.** Left: the relative completeness residual of every row, one histogram per readout and baseline on a logarithmic axis, with the tolerance the pass counts against as a dotted rule and the count over it in the title. Middle: the entry jump $f(x_0) - f(b)$ against the readout at the input, one point per row, per baseline. Right: the largest attribution to a stored step after the anchor and to a gated-off source step, per readout, on a symmetric-log axis.

**Axes.** Relative residual (log) and rows; readout units on both middle axes; readout units (symmetric log) on the right.

**Interpretation.** A row over tolerance is one whose integrated gradients do not account for the readout change they claim; its map is a picture, not a decomposition. Raise `IG_STEPS` before reading such rows. A large entry jump is a readout that moved sharply between the exact baseline and the entry point, which belongs to the baseline and the normalisation, not to any input step. The right panel is exactly zero on a causal, gated model; anything else is a defect, not a finding.

## `attribution/attribution_time_to_delivery.pdf`

**Purpose.** Place the attributed anchors on the clinical clock.

**What it shows.** One row per main readout under the source-null baseline: the source attribution total (left) and the $|q|$-weighted lag centroid of the source attribution (right) of every attributed anchor against its hours before delivery, one point per anchor in its class colour, with the class median per one-hour window drawn where at least three recordings fall in it. Delivery is on the right.

**Axes.** Hours before delivery across; readout units and stored-coefficient seconds up.

**Interpretation.** A sparse view: a few anchors of one segment per recording. It says whether the attributed recordings sit at comparable stages of labour and whether the source attribution or its lag centre drifts with that stage, and no more; the traces read the clock densely for one recording per class, and the clock analyses read it on every recording with a correction. The axis of the right panels is stored-coefficient time.

## The per-recording attribution traces: `attribution/traces/<class>/<guid>_<subgroup>_attribution_trace.pdf`

**Purpose.** Follow attribution through one recording per class. Selection favours the recording whose stored segments most completely cover the evaluation window, helping readers inspect change over time.

**What it shows.** Seven rows on the traces' shared figure, on one axis of hours before delivery: the magnitude of the source attribution of $K_t$ by offset from the anchor as a heatmap; the same for the forecast gap; the model's own lag readout at the same anchors; the correlation of each readout's lag-aligned attribution with the model's readout; the two readouts' source attribution totals; $K_t$ at the input and at the exact null; and the forecast gap at the input and at the exact null. Each heatmap cell spans the gap to its neighbouring anchors, so the sparse anchors read as blocks rather than hairlines. The filename carries the GUID **and the subgroup**.

**Interpretation.** As the traces are: a step at a segment join is geometry, the colour scales are per recording, and the lag axis is stored-coefficient time. The anchors are `ANCHORS_PER_SEGMENT` per segment rather than every scored anchor, so a feature narrower than the spacing between them is invisible here. The two attribution rows answer different questions of the same inputs — where the source moved the belief, and where it helped the forecast — and an hour where the two part is an hour where the latent change was not the change that paid.

## The per-recording pages: `samples/<selection>/sample<index>_<guid>_epoch<epoch>.pdf`

**Purpose.** Inspect one selected segment in detail, including inputs, forecasts, latent quantities, and lag readouts. Use it to investigate examples behind aggregate scores.

**What it shows.** This cell's **fifteen-row** diagnostic page, drawn through the task's own page seams — the same layout the training callback draws, rather than a second builder that could disagree with it. The rows are: the raw context, the forecast lanes, six causal extra rows (truth, both branch means, the signed skill difference, the posterior's own $\sigma$, and the per-window score), two gated-input rows, and the five latent and lag rows the layout owns.

**Clocks.** Every model row is at the anchor step, so a column is the same anchor on all of them; only the raw row is in physical time. An aligned input channel at step $t$ carries content centred $\kappa\tau_{\mathrm{ref}}$ earlier — $352$ s on the target input, $252$ s on the source — so the input rows sit that far to the right of the raw row, as their x labels state.

To make the alignment checkable, **each input row overlays its own raw signal delayed by that constant**: the raw FHR on the target row, the raw UP on the source row, as a thin black trace on a twin axis. A deceleration and the coefficient columns it produced then coincide on the row itself. The constants come from the resolved budget, which the runner attaches to the loaded task after preflight; an unaligned run has no single constant and draws no overlay.

**Directories.** `stratified/` is a seeded, shard-stratified draw over the whole split, so a cap at or above the shard count reaches every shard. `by_class/` is a **class-balanced** draw: the same number of segments from every clinical class, `eval_config.caps.pages_per_class` of them. Beside them, one directory per extreme metric and tail holds the segments at the extremes of that metric — `mean_pred_gap_low/` and `mean_pred_gap_high/` on the mean-decoded gap (the estimator that scores the mean forecast lanes the page draws), and the same pair for `nll_full_block` and `source_conditioned_kl_raw`.

**Whose page it is.** The title of every page carries the GUID **and the subgroup** the recording came from (`guid … — subgroup acidosis_cs`), because the filename carries only the GUID and a page lifted out of its directory would otherwise not say which cohort it belongs to. `sample_pages.csv` beside the directories carries the same identity per file — `guid`, `epoch`, `clinical_class`, `subgroup` — so a directory can be filtered by cohort without opening a PDF.

**The two draws are not interchangeable, and reading one as the other is the mistake to avoid.** `stratified/` allocates its quota in proportion to shard size, so what it renders is what the split mostly *contains* — on the shipped cohort, mostly healthy. `by_class/` gives `hie` as many pages as `healthy` and is therefore, by construction, not representative of anything: it is what supports a comparison *across* classes, and it says nothing about how common either class is. Counting pages in `by_class/` as evidence of prevalence inverts the one property it was drawn for.

**Every segment here appears twice** — the full page, and the reduced page beside it. See the next section.

**Interpretation.** A page is **one segment of one recording: an illustration, never evidence.** The extreme pages are selected *on* the quantity they display, so the panel showing it is guaranteed to look unusual and says nothing about how often it does. The `<index>` in the filename is the position in the evaluation **dataset**, not in `per_sample.csv` — the collection pass runs under a seeded shuffle — and the two are reconciled by a `guid`/`epoch` round trip checked before anything is rendered. Rendering needs a checkpoint; a model-free re-run records a skip.

## The reduced per-recording pages: `samples/<selection>/sample<index>_<guid>_epoch<epoch>_compact.pdf`

**Purpose.** Inspect the same segment and forward pass in a shorter layout focused on the latent state and attention.

**What it shows.** Five of the full page's fifteen rows, in the full page's own order, drawn by the same code: the raw context, the **target** stream as the encoder receives it (`fhr_st` | `fhr_ph`, block dividers and warm-up staircase intact), the latent state over its source-derived shift, $K_t$, and the lag attention. It is the full page with rows removed, not a second page — a reader who knows one knows the other.

**What it drops, and why that is the point.** The forecast lanes and the six causal extra rows: the truth, both branch means, the signed skill difference, $\sigma^q$ and the per-window score. Those eight rows answer *what did the model predict*, which is a different question, and on a $14 \times 48$ in page they sit between the input row and the latent row that are read against each other.

**The lag attention is drawn on a logarithmic colour scale here**, and on a linear one on the full page. Attention is a softmax over 91 lags, so on a linear scale one dominant lag flattens the rest of the panel into the bottom colour — acceptable when the panel is one of fifteen, not when it is one of five. The scale is floored four decades below the panel's own maximum, so a single near-zero cell cannot stretch the colormap over decades that hold nothing.

**Interpretation.** Three ways.

- **A grey cell is a forbidden lag, not a small one.** The log scale masks the non-positive cells, and they are painted light grey rather than left showing the axes background. Every lag below the source floor $F_u$ is zeroed by the lag mask: the model was never allowed to attend there. On the linear full page those cells are simply the bottom colour and are indistinguishable from genuinely low attention.
- **The colour scale is per page.** It is taken from that segment's own attention, so two reduced pages cannot be compared by colour. The argmax overlay can be.
- **Everything the full page's caveats say still applies**, including the lag-time one printed at the foot of both: a lag axis here is stored-coefficient time, not physical delay.
