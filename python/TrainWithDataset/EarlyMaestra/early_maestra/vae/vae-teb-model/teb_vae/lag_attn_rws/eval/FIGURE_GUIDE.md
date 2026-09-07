# Figure guide

One entry per PDF the evaluation emits: what it shows, what its axes are, and how it is misread. Each is written under the filename the run produces, inside the analysis subdirectory named in its heading.

A rule that applies to every figure here: **an empty panel is a statement, not a bug.** A panel that found no finite value draws the words `no finite values` rather than an empty frame, and that means the analysis measured nothing rather than that the plotting failed.

A second rule applies to every *lag* axis this pipeline will ever draw: the seconds figure shown is the **compensated** lag $\tau = 4(\ell + \delta)$, the residual physiological delay after the model's own input delay $\delta$ is added back. There is no "original-sensor" twin: the stored UP/FHR timeline is canonical, the dataset builder's UP shift is part of the signal, and no lag axis adds it back or subtracts it.

A third applies to every figure that resolves a quantity **by cohort**, and it is the one an operator is most likely to assume rather than check.

**The order is clinical, not alphabetical.** Cohorts run left to right in ascending severity: `healthy`, `acidosis`, `hie` on the class axis, and on the subgroup axis `healthy_no_bg_no_cs`, `healthy_no_bg_cs`, `healthy_bg_no_cs`, `healthy_bg_cs`, `acidosis_no_cs`, `acidosis_cs`, `hie_no_cs`, `hie_cs`. One function decides it — `cohort.ordered_groups` — and every figure and every CSV row order in this pipeline reads it, so a table can be read against the figure beside it row for row. A cohort the order does not know (a non-canonical shard stem) is drawn **after** the ones it does rather than dropped.

**The colour is the severity.** Green for healthy, amber for acidosis, red for HIE, and each subgroup a shade of its own class — light to dark in the order above, so the shading carries the ordering too and a violin's hue says which class it belongs to before its label is read. The mapping is a table rather than an assignment pass, so a cohort keeps its colour whichever other cohorts a figure happens to contain, and two figures over different subsets can be laid side by side.

Two consequences worth stating. This palette is **this evaluation's**, not the repository's: the shared `utils.style` mapping paints `healthy` blue and is used by the `lag_attn` sibling and by `model/transformer_experiment`, so a figure from this pipeline and a training-callback figure of the same cohort are **different colours** and must be read by legend, not by hue. And a colour ordering is not a result: it makes a difference easy to see, and `cross_subgroup` is the analysis that says whether the difference survives being asked properly.

---

# Terminology

Every name that appears on an axis, in a legend, in a panel title or in a CSV column beside a figure. Read this once and every figure below is readable without guessing.

## What the model does, in one paragraph

The recording is a fetal heart-rate trace (**FHR**) and a uterine-pressure trace (**UP**), both sampled at 4 Hz. The model steps through the recording in **4-second anchors**. At each anchor $t$ it forecasts the **next two minutes of raw FHR** — 480 raw samples — and it does this **twice**:

- the **base** forecast, from a belief built out of the FHR history alone;
- the **full** forecast, from a belief that also read the UP history.

Everything this evaluation reports is some version of *"how much better was the second one"*. If UP genuinely tells you something about the FHR two minutes ahead, the full forecast is better and the two beliefs differ. If it does not, they coincide and every number below is zero.

The belief is a 48-dimensional Gaussian, and there are two of them per anchor:

| Name | Symbol | What it is |
|---|---|---|
| **prior** | $p(z_t \mid Y_{\le t})$, mean $\mu^p_t$ | the belief from FHR history alone — feeds the **base** forecast |
| **posterior** | $q(z_t \mid Y_{\le t}, U_{\le t})$, mean $\mu^q_t$ | the belief after also reading UP — feeds the **full** forecast |

One decoder turns either belief into a forecast, so the *only* difference between base and full is the belief handed to it. That is what makes the comparison meaningful rather than a comparison of two models.

The model reads UP through **lag attention**: for each anchor it distributes attention over the last 91 source steps (~6 minutes) and reports where it looked. That is what the lag figures show.

## How to read a name

Names are compositional. Once these five pieces are known, most columns decode themselves.

| Piece | Means |
|---|---|
| `base` | the FHR-only branch (prior belief) |
| `full` | the UP-conditioned branch (posterior belief) |
| `shuffled` | the negative control: the *same* model fed **another recording's** UP |
| `mc_` prefix | **Monte Carlo**: averaged over $K$ latent draws instead of one. The headline form. |
| `_block` suffix | summed over the whole 480-sample forecast block, then averaged over anchors |
| `_raw` suffix (on a KL) | **unfloored** — no free-bits floor applied. The only form readable as a rate. |
| `_sq` suffix | left **unrooted** (a mean square). Its `_rms` partner is the rooted version. |

So `mc_nll_full_block` reads as "Monte Carlo negative log-likelihood of the UP-conditioned forecast, per anchor block", and `pred_gap` versus `mc_pred_gap` are the same subtraction computed on the single-draw and the marginalised score respectively.

## The quantities

### Scores — how good was a forecast

A **score** here is always a negative log-likelihood: **lower is better**, and it is not bounded below by zero.

| Name | Plain meaning | Units |
|---|---|---|
| `nll_base_block` | how badly the FHR-only forecast fit the truth | nats per anchor |
| `nll_full_block` | the same for the UP-conditioned forecast | nats per anchor |
| `mc_nll_base_block`, `mc_nll_full_block` | the same two, averaged over $K$ latent draws — the **headline** pair | nats per anchor |
| `mc_nll_shuffled_block` | the score when fed a **stranger's** UP; the negative control | nats per anchor |
| `nll_persistence_block` | baseline: repeat the last **observed** FHR sample for two minutes | nats per anchor |
| `nll_climatology_block` | baseline: predict the population mean ($z = 0$) | nats per anchor |
| `nll_segment_mean_block` | baseline: predict this segment's own mean | nats per anchor |
| `nll_oracle_block` | an evaluation-only decoder reading the encoder state **directly**, bypassing the 48-dim bottleneck | nats per anchor |

**"nats per anchor" is the unit to internalise.** One score is a sum over all 480 raw samples of one forecast block, then averaged across the anchors that were scored. So it is a *large* number — hundreds — under every predictor including a perfect one, because it is 480 terms added up. Its absolute size says nothing; only differences and ratios between predictors are readable. Divide by 480 for a per-raw-sample figure, but note that this is a flat rescale and **not** a mean over unmasked samples, so it under-reports on any anchor with masked forecast steps.

### Gaps — what the source added

| Name | Plain meaning | Units |
|---|---|---|
| `pred_gap` | $D_{\mathrm{base}} - D_{\mathrm{full}}$ on the **single-draw** path. Positive = UP helped. | nats per anchor |
| `mc_pred_gap` | the same difference on the marginalised scores. **This is the headline coupling number.** | nats per anchor |
| `delta_suff_nats` | $D_{\mathrm{base}} - D_{\mathrm{oracle}}$: what the 48-dim bottleneck *costs* the forecast | nats per anchor |
| `mse_skill` | $1 - \mathrm{MSE}_{\mathrm{model}}/\mathrm{MSE}_{\mathrm{baseline}}$. $1$ = perfect, $0$ = no better than the baseline, negative = worse. | dimensionless |
| `advantage_nats_per_anchor` | the NLL-space analogue of skill, and a **difference**, not $1 -$ a ratio — a log score has no natural zero | nats per anchor |
| `pred_gap_rmse_pct` | $100\left(1 - \mathrm{RMSE}_{\mathrm{full}}/\mathrm{RMSE}_{\mathrm{base}}\right)$: the percentage of the FHR-only forecast's error the UP removed. **Scale-free** — the same number in $z$ units and in bpm. | percent |
| `pred_gap_mse_pct` | the same in mean-square, i.e. `mse_skill` applied source-vs-no-source rather than model-vs-baseline. The larger of the two wherever both are positive. | percent |
| `pred_gap_mc_likelihood_pct` | $100\left(e^{\mathrm{mc\_pred\_gap}/480} - 1\right)$: how much more probability density the UP-informed forecast puts on each observed raw sample. The percentage form of the headline nats. **`gaussian_nll` only** — see below. | percent |

`pred_gap` and `mc_pred_gap` are two estimators of *one* quantity, not two findings. Wherever both are drawn, their difference is the price of the Monte Carlo marginalisation.

The three percentages are that one quantity restated proportionally, and **none of them is `pred_gap` divided by a block score**. $D_{\mathrm{base}}$ is a negative log *density* summed over 480 raw samples: it has no natural zero and is legitimately negative for a sharp forecast, so that ratio would change sign with its own denominator. The two spaces that do have a natural zero are used instead — error space, where a forecast equal to the truth scores $100\%$, and likelihood space, where equal forecasts score $0\%$. Each is computed **per recording and then averaged**, never as a ratio of two averages. The likelihood one divides by the fixed 480 rather than by a per-anchor scored-sample count, so like every `/480` figure in this guide it **under-reports** wherever forecast steps are masked.

**The likelihood percentage exists only under `gaussian_nll`.** Under `mse` a block score is a sum of squared errors rather than a log density, so exponentiating it yields no density ratio — it is not a probability of anything, and even read charitably as a unit-variance Gaussian it would be out by a factor of two. An `mse` run therefore emits no such column, no row and no headline key, and records why in `results.coupling.pred_gap_percent.likelihood_space`. The two error-space percentages are unaffected: a squared error is a squared error under either likelihood.

### Point error — how far off, in heartbeats

Scores mix accuracy and confidence. These are pure accuracy, computed **per scored raw sample** rather than per anchor.

| Name | Plain meaning | Units |
|---|---|---|
| `sq_error_base`, `sq_error_full` | mean squared error of each branch's forecast | $z$-units², unrooted |
| `sq_error_persistence` / `_climatology` / `_segment_mean` | the same for each baseline — the denominators of `mse_skill` | $z$-units² |
| `rmse_full`, `rmse_base` | $\sqrt{\texttt{sq\_error\_*}}$ converted to heartbeats — the readable form of the two above, and the name the `distributions` figures draw | bpm |
| `abs_error_full` | mean absolute error | $z$-units |
| `signed_error_full` | mean *signed* residual: the forecast's **bias**, positive = forecasting too high | $z$-units |
| `forecast_difference_sq` | mean squared distance between the two forecasts themselves, $(\mu^{\mathrm{full}} - \mu^{\mathrm{base}})^2$ | $z$-units² |

`forecast_difference_sq` is not `pred_gap`: two forecasts can differ everywhere and score identically. One measures *movement*, the other measures *improvement*.

**$z$-units versus bpm.** The loader z-scores FHR, so the model works in standard deviations, not heartbeats. Wherever the normalisation statistics are available a figure converts to **bpm** and says so in its axis label; where they are not, the label says `normalised`. Roughly, $0.1$ $z$-unit $\approx 1$ bpm. Levels convert affinely (scale **and** shift); spreads, RMSEs and differences convert by scale only.

### The KL family — how much the belief moved

$K_t = \mathrm{KL}(q_t \Vert p_t)$ measures how far reading UP moved the belief at anchor $t$. Zero means UP changed nothing.

| Name | Plain meaning | Units |
|---|---|---|
| `source_conditioned_kl_raw` | $K_t$ averaged over the scored anchors. **The** KL readout. | nats per anchor |
| `source_conditioned_kl_shuffled_raw` | the same under a **stranger's** UP — the control | nats per anchor |
| `kld_per_t` | $K_t$ *before* averaging: one value per anchor | nats per anchor |
| `kld_per_dim` | $K_t$ split across the 48 latent dimensions; sums back to the total | nats per anchor |
| `kld_per_head` | $K_t$ split across the 4 attention heads; sums back to the total | nats per anchor |
| `source_kl_lag_map` | $K_t$ split across the 91 lags; sums back to the total | nats per anchor |
| `active_dims` | how many latent dimensions carry more than `KLD_ACTIVE_EPS` nats | count |
| `top_dimension_share` | the largest dimension's fraction of the total — near $1$ means one dimension carries everything | fraction |

Three warnings this pipeline repeats because each has cost a result somewhere:

1. **Only the `_raw` (unfloored) KL is a rate.** A floored variant exceeds it by construction and would hide a collapsed source pathway.
2. **The KL is inflated whenever the prior variance sits on its clamp**, because $K_t$ carries $(\mu^q - \mu^p)^2/\sigma_p^2$ and a pinned $\sigma_p^2$ divides by an arbitrarily small number. Check the `prior_variance_not_pinned` verdict before quoting a KL. `pred_gap` is immune to this, which is why the two travel together.
3. **A bigger KL under a stranger's UP is healthy, not broken.** A stranger's source is out of distribution, so it moves the posterior *more*. Specificity is judged on the **scores** ($D_{\mathrm{full}} < D_{\mathrm{shuffled}}$), never on the KL.

### Belief movement, in the latent

| Name | Plain meaning |
|---|---|
| `delta_mu_sq` / `delta_mu_rms` | how far $\mu^q$ sits from $\mu^p$, **per latent coordinate** |
| `mu_post_prior_gap_sq` / `_rms` | the same distance as a whole vector (summed over $d_z$ first) — larger by $\sqrt{48}$ |
| `mu_prior_rms` | the size of the prior belief itself, for scale |

### Attention — where in the past the model looked

| Name | Plain meaning | Units |
|---|---|---|
| **lag** $\ell$ | how far back in UP history, in 4-second steps. $\ell = 0$ is now; $\ell = 90$ is 6 minutes ago. | steps |
| **compensated lag** $\tau = 4(\ell + \delta)$ | the same thing in **seconds**, after adding back the model's own input delay $\delta$. Every seconds axis in this pipeline is this one. | s |
| `attention_profile` | attention weight per lag, averaged over anchors | probability |
| `attention_profile_support_corrected` | the same, each lag divided by the anchors at which that lag **existed** |  probability |
| `attention_profile_untruncated` | recomputed on the anchors where *every* lag exists — the strictest form | probability |
| `attention_profile_per_head` | the four heads separately, unaveraged | probability |
| `attention_entropy_nats` | how *spread out* the attention is. Low = focused on one delay; high = looking everywhere. | nats |
| `attention_entropy_attainable_nats` | the **highest** entropy this anchor could have reached, $\log\min(t{+}1, L)$. The entropy is only readable as a ratio against this. | nats |
| `lag_profile`, `lag_profile_support_corrected`, `lag_profile_untruncated` | the same three forms for the **KL** attributed across lags rather than for the attention | nats per anchor |
| `lag_support` | how many anchors contributed to each lag bin — the denominator behind the correction | count |

**Why three forms of the same profile.** Lag $\ell$ only exists at anchors $t \ge \ell$, so early anchors cannot look far back. Untouched, that shortage biases every argmax toward short lags. The *raw* form is the honest decomposition (it sums to the total); the *support-corrected* form is the fair comparison between lags; the *untruncated* form is the one an argmax claim should rest on.

`degenerate` is a mechanical flag meaning "this profile is too flat for its argmax to mean anything" — peak-to-median below 1.1, or more than 90% of bins exactly zero.

### The decoder's variance, and whether it can be believed

The decoder outputs a mean **and** a variance per raw sample. The variance is what turns a squared error into a likelihood, so if it is wrong every score above is wrong too.

| Name | Plain meaning |
|---|---|
| `mean_logvar_full` | the average log-variance the decoder emitted. $0$ means $\sigma = 1$ $z$-unit. |
| `logvar_full_floor_frac` | fraction of the mass pinned near the **bottom** clamp — an over-confident decoder |
| `logvar_full_ceil_frac` | fraction pinned near the **top** clamp — a decoder that has given up and predicts noise |
| `mean_logvar_prior`, `logvar_prior_floor_frac` | the same two for the **prior belief's** variance — the detectors behind warning 2 above |
| **PIT** | probability integral transform, $u = \Phi((x-\mu)/\sigma)$. If the variance is right, $u$ is **uniform** on $[0,1]$. |
| **coverage** | what fraction of truth actually landed inside $\pm 1/2/3\sigma$. Should be $0.6827 / 0.9545 / 0.9973$. |
| **CRPS** | a single accuracy-and-sharpness score for the whole predictive distribution; lower is better |
| **NLL gain over the homoscedastic MLE** | how much the *learned* variance beat one constant variance fitted to the same residuals. Near zero = the learned variance bought nothing. |

The clamp is `logvar_clamp: [-5, 3]`, and "pinned" means within 5% of the range of an end — $0.4$ nats — because the bound is a sigmoid and exact equality never occurs.

### Geometry and axis words

| Word | Means |
|---|---|
| **segment** | one 20-minute window of a recording: $T = 300$ anchors |
| **recording** / `guid` | one delivery, contributing up to ~37 segments. **Every statistic in this pipeline is per recording**, never per segment. |
| `epoch` (as a column) | the segment's start time in **seconds relative to delivery**, negative before it — not a training epoch |
| **anchor** $t$ | one 4-second step; the point a forecast is made from |
| **horizon step** $\tau$ | how far ahead within one forecast, $0..29$. Step $0$ is $4$ s ahead, not $0$. |
| **lead time** | the same axis in seconds: $4$ s to $120$ s |
| **block** | one anchor's whole forecast: $H \times R = 30 \times 16 = 480$ raw samples |
| **warm-up** | the first 30 anchors (2 min), which carry no loss term at all |
| **untrained tail** | the last 30 anchors, whose forecast runs past the end of the segment |
| **support** | the anchors a quantity was actually defined at — a denominator, not a measurement |
| **contributing anchors** | the anchors that survived masking and were scored; the divisor behind "per anchor" |

Anchors are **not independent**: consecutive anchors' forecast windows overlap in 29 of their 30 steps. That is why every violin, interval and $p$-value in this pipeline is computed over recordings.

### Cohorts

| Word | Means |
|---|---|
| `clinical_class` | `healthy` / `acidosis` / `hie` — the three-way severity axis |
| `subgroup` | the eight-way split: class × background × caesarean |
| `bg` / `no_bg` | with or without background (comorbidity) |
| `cs` / `no_cs` | delivered by caesarean section or not |

**Every class contrast is out of distribution.** The checkpoint trains on healthy-with-background recordings only, so acidosis, HIE *and* both healthy-no-background subgroups are unseen. A difference between cohorts here is a difference in how the model generalises, not held-out clinical discrimination.

## The shipped numbers

| Symbol | Value | What it is |
|---|---|---|
| $T$ | 300 | anchors per segment |
| $H$ | 30 | horizon steps = 2 minutes |
| $R$ | 16 | raw samples per horizon step (4 Hz × 4 s) |
| $H \cdot R$ | 480 | raw samples in one forecast block |
| $d_z$ | 48 | latent dimensions |
| $L$ | 91 | lag bins = ~6 minutes of UP history |
| $M$ | 4 | attention heads |
| warm-up | 30 | unscored leading anchors |
| trained range | $[30, 270)$ | the 240 anchors the loss ever sees |
| clamp | $[-5, 3]$ | log-variance bounds |

---

## The grouped variants: `<stem>_by_clinical_class.pdf` and `<stem>_by_subgroup.pdf`

**In plain terms.** "Does this number differ between the clinical groups?" Take any per-recording number the pipeline computes, split the recordings by class (or by subgroup), and draw one violin per group. A violin is the distribution of that number across the recordings in that group: fat where many recordings sit, thin where few do, with a thin box plot inside it.

**The mark inside each violin** is the ordinary five-number summary, and it is the same on every violin in this pipeline: the heavy black bar is the **middle half** of the cohort's recordings ($Q_1$ to $Q_3$), the hairline through it runs to the furthest recording within $1.5$ inter-quartile ranges of that bar, and the white dot is the **median**. A recording past the end of the hairline is not omitted — it is the tail of the violin body itself, which is drawn between the cohort's own extremes, so nothing is hidden by the whisker rule. Read the bar for where the cohort sits and the body for what shape it has; the two disagree exactly when the distribution is bimodal, which is the case this figure is drawn as a violin to expose.

These have no entry of their own because they are not one figure. Every analysis that writes a per-recording table declares it, and the **runner** fans one violin figure per cohort axis over whatever was declared -- so the set of them grows with the analyses and each is named after the table it resolves. One panel per metric, one violin per cohort, in the clinical order and palette described at the top of this file.

**Axes.** One row per metric; the metric's own units on the vertical axis, cohorts across in ascending severity. The CSV beside each figure carries its rows in that same order, so the two are read together rather than reconciled.

**Terms on these figures.** Which metrics appear depends on which analysis declared the table:

| From | Metrics drawn |
|---|---|
| `coupling` | `mc_pred_gap`, `pred_gap`, `source_conditioned_kl_raw`, `source_conditioned_kl_shuffled_raw`, `pred_gap_rmse_pct` |
| `forecast` | `sq_error_base`, `sq_error_full`, `nll_full_block`, `signed_error_full` |
| `perm_control` | `mc_nll_base_block`, `mc_nll_full_block`, `mc_nll_shuffled_block` |
| `latent` | `source_conditioned_kl_raw`, `logvar_prior_floor_frac`, `mean_logvar_prior` |
| `lag_kl` | `source_conditioned_kl_raw` and the two identity residuals |
| `attention` | `attention_entropy_nats`, `attention_entropy_attainable_nats` |
| `calibration` | `mean_logvar_full`, `logvar_full_floor_frac`, `logvar_full_ceil_frac` |
| `residual` | `forecast_difference_sq`, `delta_mu_sq`, `mu_post_prior_gap_sq` |
| `sufficiency` | `delta_suff_nats`, `nll_oracle_block`, `mc_nll_base_block` |
| `events` | `difference` (event-minus-control, per readout) |

All of them are defined in **Terminology** above.

**How they are misread.** Every violin holds one value per **recording**, not per segment, so a cohort with eight recordings is eight points however many segments they contributed -- the width of a violin is not evidence. A cohort split that produces fewer than two groups emits **no figure at all** and records a skip: on the healthy-only pretraining population that is the ordinary outcome, and the pooled output beside it is untouched. And a visible separation here is not a result: `cross_subgroup` is the analysis that says which separations survive being asked properly, and it tests a deliberately short list rather than everything drawn here.

---

## `forecast/baseline_comparison.pdf`

**In plain terms.** *"Is this forecast any good at all?"* — the first question to ask, before any question about UP. The model is put in a line-up with three predictors so stupid that beating them is the minimum bar: repeat the last observed value, predict the population average, predict this segment's own average. The top panel scores all five; the bottom panel turns each comparison into a single number where **1 is perfect, 0 is "no better than the stupid predictor" and negative is worse than it**.

**What it shows.** Top: the per-recording block score of every predictor -- the two model branches and the three trivial baselines -- as violins, in nats per anchor, lower better. Bottom: the squared-error skill $1 - \mathrm{MSE}_{\mathrm{model}} / \mathrm{MSE}_{\mathrm{baseline}}$ of each branch against each baseline, with a percentile bootstrap interval over **recordings**.

**Axes.** Top: nats per anchor, one violin per predictor. Bottom: skill, dimensionless, zero marked; whiskers are asymmetric because a percentile interval is not symmetric about its point estimate.

**Terms on this figure.**

- **top, one violin each** — `base`, `full` (the two model branches), then `persistence`, `climatology`, `segment_mean` (the three baselines). Each violin is `nll_<name>_block` across recordings.
- **bottom, one bar each** — `<branch> vs <baseline>`, the value being `mse_skill`. The whiskers are `mse_skill_lo` / `mse_skill_hi`, a bootstrap interval over recordings.
- **persistence** carries forward the last *observed* sample, not the last sample: a gap is stored as 0 bpm ($\approx -11\sigma$ once z-scored), and carrying that would measure the gap.
- The baselines are scored with a fixed $\sigma = 1$, stated rather than fitted — a point predictor has no variance of its own, and the whole score would otherwise be decided by whatever $\sigma$ it was handed.

**How it is misread.** The block score is a *sum over 480 raw samples*, so it is a large number under every predictor and its scale says nothing about the model. Only the comparison is readable. And the skill drawn here is the MSE-space one, in which the observation variance cancels; the NLL-space column beside it in `forecast_skill.csv` is a **difference** in nats, not $1 -$ a ratio, because a log score has no natural zero.

## `forecast/anchor_profile.pdf`

**In plain terms.** *"Does the model do better at the start of a segment or at the end?"* Walk along the 20-minute segment from left to right and, at each 4-second anchor, average the score over every segment in the split that had an anchor there. A flat line means position in the segment does not matter. The shaded bands at the two ends are regions the model was **never trained to score** — ignore whatever the curve does inside them.

**What it shows.** The two block scores and `pred_gap` against time in segment, averaged over every segment that scored that anchor.

**Axes.** Anchor index in decimated (4 s) steps from the start of the trimmed segment; nats per anchor.

**Terms on this figure.**

- **curves** — `nll_base_block`, `nll_full_block` and their difference `pred_gap`, each averaged across segments at that anchor index. This is the *single-draw* path.
- **left shaded span** — the **warm-up**, anchors $[0, 30)$: the loss ignores them entirely.
- **right shaded span** — the **untrained tail**, anchors $[270, 300)$: their two-minute forecast would run past the end of the segment, so no anchor there was ever scored.
- **x axis** — anchor index, so $\times 4$ gives seconds into the segment.

**How it is misread.** The two shaded spans are **structural, not findings**: the warm-up prefix $[0, w)$ carries no loss term at all, and the tail $[T - H, T)$ holds anchors whose forecast window runs past the end of the segment, so no anchor there is ever scored. An unshaded version of this figure reads as a model that fails at both ends of every recording.

## `forecast/horizon_skill.pdf`

**In plain terms.** *"How far ahead can it actually see?"* A single block score lumps all two minutes of the forecast into one number. This unpacks it: how good is the forecast 4 seconds ahead, 30 seconds ahead, two minutes ahead. Expect the error to rise with lead time — a forecast that is as good at 120 s as at 4 s is usually predicting a flat line. The gap panel is where UP's contribution lives: if UP helps at all, it should help most at the lead times where the FHR-only branch is losing.

**What it shows.** $D_{\mathrm{base}}(\tau)$ and $D_{\mathrm{full}}(\tau)$, their gap, and each branch's RMSE, against lead time.

**Axes.** Lead time in **seconds** on every panel, spanning the whole forecast window; nats per horizon step, and bpm for the error panel where the loader's statistics are known.

**Terms on this figure.**

- **lead time** — horizon step $\tau$ converted to seconds: $\tau = 0$ is **4 s ahead**, $\tau = 29$ is 120 s ahead. The anchor's own block is the past, not the forecast, so there is no $0$ s point.
- $D_{\mathrm{base}}(\tau)$, $D_{\mathrm{full}}(\tau)$ — the two block scores resolved by horizon step instead of summed over it. Units are nats **per horizon step** here, not per anchor: this is the block score's per-$\tau$ decomposition, so the 30 values sum back to the block.
- **gap** — their difference, the per-$\tau$ version of `pred_gap`.
- **RMSE** — the point error of each branch at that lead time, in **bpm** where the loader's statistics are known.
- Computed on the **single-draw** path, and the title says so.

**How it is misread.** The curve is computed on the **single-draw** path, and says so in its title. The Monte Carlo marginalisation does not commute with the sum over $\tau$ -- by Jensen -- so a marginalised curve would not sum back to the marginalised headline. Horizon step $0$ is $4$ s ahead, not $0$: the anchor's own block is the past, not the forecast.

## `forecast/forecast_overlay.pdf`

**In plain terms.** *"What does one forecast actually look like?"* Everything else in this pipeline is an average over thousands of anchors. This is a single two-minute forecast drawn against what actually happened, so you can see the shape rather than a number: the truth, the FHR-only guess, and the UP-informed guess on the same axes. Use it for intuition; it is one anchor and proves nothing.

**What it shows.** One anchor's truth, target-only forecast and source-conditioned forecast on the raw sample grid.

**Axes.** Lead time in seconds; FHR in bpm where the loader's statistics are known, and in normalised units where they are not -- the label says which.

**Terms on this figure.**

- **truth** — the recorded FHR over the 480 raw samples this anchor forecast.
- **base** — $\mu^{\mathrm{base}}$, the FHR-only forecast.
- **full** — $\mu^{\mathrm{full}}$, the UP-conditioned forecast. Where the two lines separate is literally what UP changed; `forecast_difference_sq` is that separation squared and averaged.
- **x axis** — lead time in seconds within this one block, 4 s to 120 s.

**How it is misread.** It is **one anchor of one retained recording**, drawn from a seeded stratified draw, not a representative case. Waveform retention is opt-in, so a run that did not ask for it emits no such figure at all; the absence is silence, not failure.

## `coupling/pred_gap_distribution.pdf`

**In plain terms.** **This is the headline figure of the whole evaluation.** *"Did reading the uterine pressure make the heart-rate forecast better, and for how many deliveries?"* Each recording contributes one number: how many nats the UP-informed forecast beat the FHR-only one by. The histogram is the distribution of those numbers. **Zero is the null.** A histogram sitting clearly right of zero says UP helped; one straddling zero says it did not. The bottom panel repeats the same answer computed two ways, as a consistency check.

**What it shows.** Top: the distribution of `pred_gap` over **recordings**, with zero marked and the bootstrap interval on the *mean* shaded. Bottom: the two estimators side by side, each under its own name.

**Axes.** Nats per anchor; the histogram's height is a count of recordings, not of segments.

**Terms on this figure.**

- **top histogram** — `mc_pred_gap` per recording: $D_{\mathrm{base}} - D_{\mathrm{full}}$ on the Monte Carlo marginalised scores. Positive = UP helped.
- **dashed vertical at 0** — "no improvement".
- **shaded band** — the 95% bootstrap interval **on the mean**, over recordings. It is not the spread of the data; a wide histogram with a narrow band is entirely ordinary.
- **$n$ in the title** — recordings that actually scored at least one anchor.
- **bottom violins** — the same quantity under both estimators, `mc_pred_gap` and `pred_gap`. Their difference is the cost of the marginalisation, not a second result.

**How it is misread.** Three ways. The shaded band is the interval on the **mean**, not the range of the data -- a wide distribution with a tight band is an ordinary result. The two violins are two *estimators of the same quantity*: the Monte Carlo marginalised score, which is the headline, and the single-draw training path, which is the objective-parity column; their difference is the cost of the marginalisation, not a second finding. And the unit of this figure is one recording, so a recording that scored no anchors is absent rather than at zero -- the $n$ in the title is the count that was actually available.

## `coupling/pred_gap_percent.pdf`

**In plain terms.** *"By what percentage did reading the uterine pressure improve the forecast?"* The figure beside this one answers the same question in nats, which is the model's own unit and tells a reader nothing about proportion -- whether 3 nats is a large improvement or a negligible one is not readable off the number. This one answers it as a percentage, in the two ways a percentage can honestly be taken. The top two panels are about the **error**: how much of the heart-rate prediction error the pressure signal removed. The bottom panel is about the **likelihood**: how much more probable the observed trace became. **Zero is the null on all three**, and a negative value means the source made that recording's forecast worse.

**What it shows.** Top: the distribution of `pred_gap_rmse_pct` over **recordings**, with zero marked and the bootstrap interval on the *mean* shaded. Middle: `pred_gap_rmse_pct` and `pred_gap_mse_pct` side by side -- the same ratio under a root, so the mean-square figure is the larger wherever both are positive. Bottom: `pred_gap_mc_likelihood_pct`, on its own axis, **empty under an `mse` checkpoint** -- where it is undefined rather than zero.

**Axes.** Percent on every panel; the histogram's height is a count of recordings, not of segments.

**Terms on this figure.**

- **top histogram** — `pred_gap_rmse_pct` per recording: the percentage of the FHR-only forecast's root-mean-square error that the UP-informed branch removed. Scale-free, so it is the same number whether the error is measured in $z$ units or in bpm.
- **dashed vertical at 0** — "no improvement".
- **shaded band** — the 95% bootstrap interval **on the mean**, over recordings, exactly as on the nats figure. Not the spread of the data.
- **middle violins** — the two error-space percentages. They are one ratio under a root and so can never disagree about the sign; their separation is arithmetic, not a finding.
- **bottom violin** — `pred_gap_mc_likelihood_pct`: $100(e^{\Delta/480}-1)$, the extra probability density placed on each observed raw sample.

**How it is misread.** Four ways. The two spaces are on **separate axes on purpose** — a per-sample density ratio and an error reduction routinely differ by an order of magnitude, and reading the bottom panel's scale off the middle one will be wrong. The likelihood percentage divides by the **fixed** 480-sample block rather than by each anchor's own scored-sample count, so it under-reports wherever forecast steps are masked; it is a floor, not an estimate — and an **empty bottom panel means the checkpoint was scored under `mse`**, where the quantity is undefined, not that the source added nothing. The middle panel's two violins are **not two results** — `pred_gap_mse_pct` exceeds `pred_gap_rmse_pct` whenever both are positive, by construction and not because the model did better on one. And none of these is `pred_gap` divided by a block score: that ratio is not defined here, for the reason given under *Gaps* above, and a reader reconstructing it from `d_base_mc_nats` will get a number that changes sign with its own denominator.

## `latent/kl_spectrum.pdf`

**In plain terms.** *"How much of the 48-dimensional belief is actually doing anything?"* The model has 48 latent coordinates in which to record what UP told it. This bar chart shows how much each one carries, tallest first. A healthy spectrum has several bars above the line; a spectrum with one tall bar and 47 flat ones means the model compressed everything UP said into a single number, and a spectrum entirely below the line means UP is being ignored. The bars add up to the headline KL, so this is that number taken apart rather than a new one.

**What it shows.** The per-dimension KL $\bar K_d$, sorted descending, with the activity threshold `KLD_ACTIVE_EPS` marked.

**Axes.** Latent dimensions ordered by the KL they carry -- **not** by index -- against nats per anchor.

**Terms on this figure.**

- **bars** — `kld_per_dim`, the KL split across the 48 latent coordinates; they sum to `source_conditioned_kl_raw`.
- **x position** — a **rank**, not a coordinate. "Dimension 0" here is the largest contributor; `latent_spectrum.csv` maps rank back to the real latent index.
- **dashed horizontal** — `KLD_ACTIVE_EPS`, the threshold a dimension must clear to be counted in `active_dims`.
- Read `prior_variance_not_pinned` before quoting the total: an inflated KL makes every bar taller by the same arbitrary factor.

**How it is misread.** The x position is a rank, so "dimension 0" on this figure is the largest contributor and not latent coordinate $0$; the mapping is in `latent_spectrum.csv`. The bars sum to the headline `source_conditioned_kl_raw`, which is what makes the spectrum a decomposition rather than a second quantity -- and that total is only a *rate* while the prior variance is off its clamp, which is what `logvar_distribution.pdf` and the `prior_variance_not_pinned` verdict exist to establish.

## `lag_kl/lag_kl_profile.pdf`

**In plain terms.** *"How far back in the uterine-pressure trace did the useful information sit?"* The model's belief-shift is split across the 91 possible delays and plotted against how many seconds ago that was. A peak at 40 s means the UP activity that mattered for forecasting happened about 40 seconds before the anchor — which is what a contraction-then-deceleration story would predict. Two curves are drawn because early anchors physically cannot look far back: the **raw** curve is the honest split of the total, the **corrected** one is the fair comparison between delays, and the bottom panel is the evidence count that separates them.

**What it shows.** Top: the per-lag KL attribution $\widetilde K_\ell$ in two forms on one axis -- the **raw** attribution, which divides every bin by the same anchor total and therefore sums over lags to the headline `source_conditioned_kl_raw`, and the **support-corrected** one, which divides each bin by the anchors at which that lag was causally valid. Each form's argmax is marked. Bottom: the contributing-anchor count per lag, which is the only thing that differs between them.

**Axes.** Compensated lag $\tau = 4(\ell + \delta)$ seconds against nats per anchor; the bottom panel is anchors per segment.

**Terms on this figure.**

- **raw curve** — `lag_profile`: `source_kl_lag_map` averaged over anchors, every bin divided by the *same* anchor count. Sums over lags to `source_conditioned_kl_raw`.
- **corrected curve** — `lag_profile_support_corrected`: each bin divided by the anchors at which that lag actually existed. Over the trained range lags 0-30 have 240 contributing anchors while lag 90 has 180, a 25% under-weight that biases the raw argmax short.
- **bottom panel** — `lag_support`, the per-lag anchor count. It is the only difference between the two curves above.
- **x axis** — compensated lag in seconds, $\tau = 4(\ell + \delta)$. Not the original-sensor figure; see the rule at the top of this file.
- **argmax markers** — the peak of each curve. Check `degenerate` in `lag_kl_summary.csv` first: a flat profile has an argmax that names a bin, not a delay.

**How it is misread.** The two curves are not two measurements of one quantity: only the raw one is a decomposition of the KL, and only the corrected one is a fair comparison *between* lags. Where their argmaxes disagree, the difference is the short-lag bias -- a finding about the geometry, not about the model. And an argmax is not by itself a reading: `lag_kl_summary.csv` carries the peak width, the mass near the peak, any secondary peaks, and a mechanical `degenerate` flag, and a profile flagged degenerate has an argmax that names a bin rather than a delay.

## `attention/attention_profile.pdf`

**In plain terms.** *"Where did the model look, as opposed to where the information turned out to be?"* `lag_kl_profile.pdf` above shows where the *useful* information was; this shows where the attention mechanism actually pointed. They can disagree — the model can stare at a delay that carries nothing. The bottom panel splits the four attention heads apart, and that split is the point: four heads each locked onto a different delay average out to a flat curve that looks like a model with no lag preference at all.

**What it shows.** Top: the head-averaged attention over lags in three forms -- raw, support-corrected, and restricted to the anchors at which *every* lag exists. Bottom: the same axis, one curve per attention head. The shaded span on both panels is the lag range that only the untruncated anchors could have contributed to.

**Axes.** Compensated lag seconds against attention weight, which is a probability per head and sums to one across the whole lag axis at an untruncated anchor.

**Terms on this figure.**

- **top, three curves** — `attention_profile` (raw), `attention_profile_support_corrected` (each bin divided by the anchors at which that lag existed), and `attention_profile_untruncated` (recomputed only on anchors where **all** 91 lags exist). The third is the one an argmax claim should rest on.
- **bottom, four curves** — `attention_profile_per_head`, one per attention head. Head $m$ writes latent group $m$, which is what makes the split additive rather than an arbitrary slice.
- **y axis** — attention weight, a probability: each head's curve sums to $1$ across the whole lag axis at an untruncated anchor.
- **shaded span** — the lag range only untruncated anchors could have contributed to. It marks where the **geometry**, not the model, thins the evidence.
- Related but not drawn here: `attention_entropy_nats` (how spread the attention is) against `attention_entropy_attainable_nats` (the most it could have been). Their ratio, not the raw entropy, is the readable quantity.

**How it is misread.** Three ways. The head-averaged curve is a mean over four distributions that need not agree, and reading it as "the" attention hides the case the per-head panel exists for -- four heads at four delays average to a flat curve. The support correction fixes each bin's *denominator* and cannot fix its numerator: attention rows are renormalised per anchor, so a truncated anchor pushes mass onto the short lags and no per-lag count knows it happened. The restricted curve is the one an argmax claim should rest on. And the shading is **structural**: it marks where the geometry, not the model, thins the evidence.

## `attention/lag_heatmap.pdf`

**In plain terms.** *"Did the model's chosen delay stay put, or move as the recording went on?"* The profile figure above averages over the whole recording, which hides exactly this. Here time runs left to right and lag runs bottom to top, with brightness showing how much attention that (moment, delay) cell received. A horizontal bright stripe = one fixed delay throughout. A sloping or wandering stripe = a delay that changes with the labour. A bright bottom edge = the model is mostly looking at the immediate past.

**What it shows.** One retained recording's attention as an anchor $\times$ lag field, head-averaged, with the warm-up boundary marked.

**Axes.** Time in segment (seconds) against compensated lag seconds, **lag increasing upward** as on the training callback's lag panels; colour is the attention weight on a sequential scale, because the quantity is non-negative. The dashed vertical line is the warm-up boundary.

**Terms on this figure.**

- **the field** — `attn_weights` for one recording, averaged over the four heads: rows are lags, columns are anchors.
- **colour** — attention weight, sequential (not diverging) because the quantity cannot be negative. Brighter = more attention.
- **x axis** — time into the segment in seconds; **y axis** — compensated lag in seconds.
- **dashed vertical** — the warm-up boundary at anchor 30. Everything left of it was never scored.
- **empty upper-left triangle** — structural, not missing data: lag $\ell$ does not exist at anchor $t < \ell$, so early anchors cannot look far back.

**How it is misread.** It is **one recording**, drawn from the seeded stratified retention draw, not a representative case. Attention retention is opt-in (`eval_config.caps.attention`), so a run that did not ask for it emits no such figure at all; the absence is silence, not failure. The upper-left triangle is empty by construction rather than by measurement: lag $\ell$ does not exist at anchor $t < \ell$.

## `calibration/pit_reliability.pdf`

**In plain terms.** *"When the model says it is 95% sure, is it right 95% of the time?"* The decoder emits a mean **and** an error bar per sample. Every score in this pipeline is a likelihood, so if the error bars are wrong, every score is wrong too — and a model can drive its score down dishonestly by shrinking its error bars where it happens to be right. The test: for every predicted sample, ask what fraction of the predicted distribution lies below the truth. If the error bars are honest, those fractions are spread **uniformly** over $[0, 1]$ — a flat histogram and a straight diagonal. A hump in the middle means the error bars are too wide; piles at both ends mean they are too narrow.

**What it shows.** Top: the probability integral transform $u = \Phi((x - \mu)/\sigma)$ of every scored raw sample, as a density, against the flat line a calibrated observation model produces. Bottom: the reliability curve -- the empirical CDF of that PIT against the diagonal -- with the three central-coverage levels quoted in the panel title.

**Axes.** Top: PIT value in $[0, 1]$ against density, where $1.0$ is the calibrated value whatever the bin count. Bottom: nominal against observed cumulative probability.

**Terms on this figure.**

- **PIT** — $u = \Phi((x - \mu)/\sigma)$ per scored raw sample: the truth's percentile inside its own predicted distribution.
- **flat line at 1.0** (top) — the density a perfectly calibrated model produces, whatever the bin count.
- **diagonal** (bottom) — the reliability target: observed cumulative probability equals nominal.
- **coverage figures in the title** — the fraction of truth that fell inside $\pm 1\sigma$, $\pm 2\sigma$, $\pm 3\sigma$. The targets are $0.6827$, $0.9545$, $0.9973$ — **not** $0.95$, which is $\pm 1.96\sigma$.
- **$\cup$ shape** = variance too **small** (too much truth in the tails). **$\cap$ shape** = variance too large. This is the direction most often inverted on sight.

**How it is misread.** The shape is the diagnosis and the direction is easy to invert: a $\cup$ means the variance is too **small** (too much mass in the tails), a $\cap$ that it is too large. The nominal coverages quoted are $\operatorname{erf}(k/\sqrt{2}) = 0.6827,\ 0.9545,\ 0.9973$; the two-sigma figure is *not* $0.95$, and a reader checking against $0.95$ will find a correctly calibrated model half a point off. This census is pooled over raw samples rather than averaged per recording, so it weights a recording by how much of it was scored -- the per-recording figures are in `calibration_per_recording.csv`.

## `calibration/logvar_distribution.pdf`

**In plain terms.** *"Are the model's error bars free, or are they jammed against a limit?"* The decoder's variance is squeezed into a fixed range $[-5, 3]$ in log units. If a lot of the mass has piled up against either end, the model is not choosing its uncertainty — the clamp is. This histogram shows where the variance actually lives. Mass at the **bottom** end is an over-confident decoder claiming near-certainty; mass at the **top** end is one that has given up and is predicting noise. The dashed lines, not the dotted ones, are what "pinned" means.

**What it shows.** The decoder's log-variance over every scored raw sample, as a fraction of the mass per bin, with both clamp **margins** drawn as dashed lines and the clamp's own asymptotes as dotted ones.

**Axes.** Log-variance over the model's own clamp range; fraction of scored raw samples.

**Terms on this figure.**

- **the histogram** — the decoder's per-raw-sample log-variance. $0$ on this axis means $\sigma = 1$ $z$-unit ($\approx 10$ bpm).
- **dotted verticals** — the clamp itself, $[-5, 3]$. The bound is a sigmoid, so mass never reaches them exactly.
- **dashed verticals** — the **margins**, 5% of the clamp range in from each end ($0.4$ nats). What falls beyond them is what `logvar_full_floor_frac` and `logvar_full_ceil_frac` count.
- Related: `mean_logvar_full` is the average of this distribution — and a single mean is equally consistent with a healthy spread and with half the mass pinned on each end, which is why the two fractions ship beside it.
- The **ceiling** failure is the dangerous one to miss: it reads as a healthily *falling* NLL while `pred_gap` quietly goes to zero.

**How it is misread.** The dashed lines, not the dotted ones, are what "pinned" means. The bound is a sigmoid, so the asymptote is never reached and mass exactly *at* a clamp would be invisible; every fraction this pipeline reports counts what lies within 5% of the clamp range of an end, which on the shipped $[-5, 3]$ is $0.4$ nats. The two ends also fail in opposite directions: mass on the floor is an over-confident decoder whose squared NLL term explodes, mass on the ceiling is one that has given up and is predicting noise -- which reads as a healthy *falling* NLL while `pred_gap` goes to zero.

## `coherence/coherence_lead_time.pdf`

**In plain terms.** *"Which rhythms does the forecast get right, and how fast does it lose them?"* A fetal heart-rate trace is several things at once — a slowly drifting baseline, decelerations over a minute or two, beat-to-beat wobble. A single error number averages all of them together. This map splits them apart: every row is one rhythm (a frequency), every column is one distance into the future, and the colour says how much of that rhythm the forecast reproduces there. A healthy model is bright at the bottom (slow things, which persist) and fades upward and rightward.

**What it shows.** Top panel: the coherence $\gamma^2$ between the truth and the source-conditioned forecast, as a field over frequency and lead time. Bottom panel: the same field for the source-conditioned forecast minus the target-only one — what the UP added, per frequency and per lead time. This is `coupling`'s `pred_gap` resolved into the two axes it hides.

**Axes.** Frequency in Hz on a **log** scale, from $\Delta f = 7.8$ mHz to $2$ Hz; lead time in seconds, $0$–$120$; colour is coherence (top, unsigned) and a signed coherence difference (bottom, symmetric about $0$). The scale is logarithmic because the bands this analysis exists to report are crowded into the bottom of the range — VLF and LF together end at $0.15$ Hz, which is $7\%$ of a linear axis. **The DC bin is not drawn**: a log axis has no room for $f = 0$, and that bin is the Hann taper's residue of the removed per-window mean rather than a frequency. It is not lost — it is part of the `vlf` band sum in every other coherence output.

**Terms on this figure.**

- **coherence** — the fraction of the truth's variation at that frequency the forecast reproduces *in phase*. $1$ is perfect, $0$ is unrelated.
- **lead time** — how far past the anchor the forecast is. Column $\tau$ covers $[4\tau + 0.25,\ 4\tau + 4]$ s, so the thirty columns tile the whole two-minute horizon.

**How it is misread.** **Coherence is not skill.** It is invariant to amplitude: a forecast that reproduces every wiggle at half size scores $\gamma^2 = 1$ here while carrying a quarter of the truth's variance as error. Read this figure with `coherence_spectrum.pdf`'s gain panel, or with `coherence_decomposition.pdf`, which splits the error properly. And nothing below $7.8$ mHz exists on this axis at all — the bottom row is the whole sub-$0.03$ Hz story, not a resolved view of it.

## `coherence/coherence_spectrum.pdf`

**In plain terms.** *"Three different ways the forecast can be wrong at a frequency, one panel each."* It can reproduce too little of the truth's variation (coherence), reproduce it at the wrong size (gain), or reproduce it at the wrong moment (phase). The fourth panel asks whether the three clinical classes differ.

**What it shows.** Coherence, spectral gain in linear units, and cross-spectral phase against frequency — each drawn for both branches at the nearest and furthest lead times — plus a per-clinical-class coherence curve pooled over lead time.

**Axes.** Frequency in Hz, log scale. Gain is a ratio with a reference line at $1$; phase is in radians with a reference at $0$. Dotted verticals mark the decoder's token-seam frequency and its harmonics.

**Terms on this figure.**

- **spectral gain** $g = \sqrt{S_{yy}/S_{xx}}$ — the forecast's amplitude over the truth's at that frequency. **$g < 1$ is over-smoothing**, the characteristic failure of a mean-square-trained forecaster: it hedges by shrinking towards the mean.
- **phase** — negative means the forecast **lags** the truth. Wrapped to $(-\pi, \pi]$, so a steep slope crosses the axis repeatedly; the unwrapped version is `group_delay_s` in `coherence_bands.csv`.
- **dotted verticals** — $0.25$ Hz and harmonics, where the decoder's per-token output head could place an artifact. See `coherence_seam.pdf`.

**How it is misread.** A low gain is **not** automatically a fault. The mean-square-optimal amplitude given a coherence $\gamma$ is $g = \gamma$, not $g = 1$, so a well-trained model is *supposed* to shrink where it is uncertain. `coherence_decomposition.pdf`'s amplitude term measures distance from that optimum; this panel measures distance from the truth's own variance. They answer different questions and a forecast can be fine on one and poor on the other. The bottom panel is **descriptive**: it pools each class's recordings rather than treating them as a sample, so it carries no interval and no test — `cross_subgroup` adjudicates cohort differences.

## `coherence/coherence_bands.pdf`

**In plain terms.** *"The same three questions, one line per clinical frequency band, against how far ahead the forecast is looking."* This is the figure to read for a statement like "the model holds the baseline for two minutes but loses beat-to-beat variability after thirty seconds".

**What it shows.** Coherence, spectral gain, and the coherence the source added — each per band, against lead time.

**Axes.** Lead time in seconds, $0$–$120$; one line per `hrv_band`.

**Terms on this figure.**

- **`vlf`, `lf`, `mf`, `hf`, `noise`** — the fetal-HRV bands: $[0, 0.03)$, $[0.03, 0.15)$, $[0.15, 0.50)$, $[0.50, 1.00)$ and $[1.00, 2.00]$ Hz. LF is the baroreflex band, MF carries fetal movement and maternal breathing, HF fetal breathing. **These are not the `band` names in `band_channel_map.csv`**, which describe the model's scattering *inputs* on a different edge set; the column here is `hrv_band` for exactly that reason, and `EVAL.md` carries the crosswalk.
- **`vlf` includes the DC bin.** The per-window mean is removed, but the taper leaves a residue, and it has to live in some band for the residual identity to close.

**How it is misread.** A band's line is only as trustworthy as its bin count, which travels in `coherence_bands.csv` as `n_bins`: `vlf` holds **four** bins and `noise` holds $129$. And every curve here is a mean over recordings of per-recording values, so the interval in the CSV is over recordings — never over segments, whose windows overlap.

## `coherence/coherence_decomposition.pdf`

**In plain terms.** *"Of the error the forecast makes at each frequency, how much was never predictable, how much is bad timing, and how much is bad sizing?"* This is the figure that turns a coherence number into something actionable, because the three answers imply three different fixes.

**What it shows.** The exact three-way split of the normalised residual spectrum,
$$\frac{S_{ee}}{S_{xx}} = \underbrace{(1-\gamma^2)}_{\text{irreducible}} + \underbrace{2g\gamma(1-\cos\phi)}_{\text{timing}} + \underbrace{(g-\gamma)^2}_{\text{amplitude}},$$
against frequency at the nearest lead time, and against lead time in the LF band.

**Axes.** Frequency in Hz (log) and lead time in seconds; the $y$ axis is a share of the truth's own power at that frequency, so $1.0$ means the forecast is no better there than predicting nothing.

**Terms on this figure.**

- **irreducible** — what no per-frequency rescaling or retiming of *this* forecast could remove.
- **timing** — vanishes exactly when the phase is right, whatever the amplitude.
- **amplitude** — vanishes exactly when $g = \gamma$, the mean-square-optimal amplitude.

**How it is misread.** The three are a **budget**, not three independent diagnostics: they sum to the normalised residual by construction, so a large `irreducible` mechanically leaves less room for the other two and does not mean the timing is good. Read the *shares*, and read the residual total beside them. The split is also not the one a reader may expect — the algebraically equivalent $\gamma^2\sin^2\phi + (g - \gamma\cos\phi)^2$ charges a purely mistimed forecast for amplitude error, which is precisely the confusion this figure exists to remove.

## `coherence/coherence_source.pdf`

**In plain terms.** *"The record shows uterine pressure and heart rate moving together. Does the forecast reproduce that, and does reading UP help it do so?"* An independent line of evidence for the source pathway: not "is the forecast better" but "does the forecast carry the relationship".

**What it shows.** Top: the coherence between UP and each of the truth, the target-only forecast and the source-conditioned forecast, in the LF band, against lead time. Bottom: the fraction of the truth's own UP coherence each branch reproduces, per band.

**Axes.** Lead time in seconds; coherence with uterine pressure. The bottom panel's reference line at $1$ is "reproduces the relationship exactly".

**How it is misread.** **This is not a causal or directed claim, and it is not a coupling measurement.** The UP it compares against is the **contemporaneous** pressure — the pressure during the window being forecast, which the model never read, since it conditions on the source only up to each anchor. So a forecast coherent with it at long lead times has *anticipated* a contraction rather than copied one, and at short lead times the two are hard to separate. The run's causality disclosure applies here exactly as everywhere else. A `preservation` above $1$ is also not "better than the truth": it means the forecast is *more* linearly tied to UP than the record is, which is a smoother, more stereotyped forecast rather than a better one.

## `coherence/coherence_seam.pdf`

**In plain terms.** *"Is the model leaving a mark of its own architecture in the signal?"* The decoder produces each 16-sample chunk of its forecast from one small linear layer, and nothing forces the last sample of one chunk to join smoothly to the first of the next. A regular kink every 16 samples is a $0.25$ Hz tone that exists in the model and cannot exist in a real heart rate.

**What it shows.** Power at $0.25$ Hz and its harmonics, divided by the median power of the neighbouring bins — for the truth (the control) and both forecast branches. Top: per harmonic, pooled over lead time. Bottom: the fundamental against lead time.

**Axes.** Harmonic index and lead time; the ratio, with a reference line at $1$.

**How it is misread.** **The truth's line is the control and must be read first.** A ratio above $1$ on a branch means nothing on its own — the fetal heart rate has its own content at $0.25$ Hz, and the truth's ratio is what that content looks like. Only the *excess over the truth* is an artifact. If there is one, it lands inside the `mf` band, so `mf`'s coherence and gain in every other coherence figure are contaminated and should be quoted with that stated. This is a property of the decoder, not a fault of the run: the remedy is a model change, which is why no verdict fails on it.

## `distributions/class_histograms.pdf`

**In plain terms.** *"What does the spread actually look like?"* Every other figure in this guide reports one number per **delivery**. This one goes back to the raw 20-minute segments and draws their whole distribution, one curve per clinical class — the histogram of segment-level forecast error, of the coupling gap, of the KL, and five more. It answers the question a mean cannot: two cohorts with the same average error can be a uniform shift or a fat tail of segments the model fails on completely, and only the shape tells you which.

Each panel carries **two** things per class. The **filled histogram** is one point per 20-minute segment — what was asked for. The **strip floating above it** is one point per recording, drawn as a median dot, a thick inter-quartile bar and a thin full-range line. If the strip is much narrower than the histogram below it, most of the width you are looking at is the *same baby measured thirty times*, not differences between babies. That comparison is the reason both are drawn.

The two levels take **different forms on purpose**. There are hundreds to thousands of segments per cohort, so a density is a fair picture of them; there are six to forty recordings, and a forty-bin histogram over six values is a row of spikes that estimates nothing, grabs the panel's y-axis and flattens the distribution you came to read. The strip says the one thing the recording level is there to say — how wide the between-recording spread is — and says it robustly at $n = 6$.

**What it shows.** Eight rows, one per metric. Within each row, one filled step-density per clinical class at segment level on a shared bin grid, and one per-recording strip per class above them, in the clinical order and palette described at the top of this file.

**Axes.** The metric in its own unit horizontally — bpm for the two RMSE rows, nats per anchor for the score and coupling rows, $z$-units and log $z$-units for the latent and variance rows. Vertical is **density**, not count.

**Terms on this figure.**

| Row | Drawn quantity |
|---|---|
| `rmse_full` | per-segment RMSE of the source-conditioned forecast, in bpm |
| `rmse_base` | the same for the target-only forecast — the pair shows what the source removed |
| `nll_full_block` | the source-conditioned block score, nats per anchor |
| `mc_pred_gap` | the headline coupling readout; the dotted line at **0** is the null |
| `source_conditioned_kl_raw` | the unfloored KL between the two latents |
| `delta_mu_rms` | per-element RMS of $\mu^q - \mu^p$: how far the source moved the belief |
| `mean_logvar_full` | the decoder's mean log-variance; the dotted line at **0** is $\sigma = 1$ $z$-unit |
| `attention_entropy_nats` | how spread the lag attention was |

- **how the curves overlap** — each cohort is a **faint fill under a hairline outline in the same colour**. Follow the *outline*: it is drawn at full opacity above every cohort's fill, so a curve stays traceable across the whole axis even where two others are stacked on it, and which cohort is legible is not decided by which happened to be drawn last. The fill only says where that cohort's mass sits; where fills overlap the tint is a blend and means nothing on its own.
- **strip** — per recording: dot = median, thick bar = inter-quartile range, thin line = full range. Same colour as its histogram, so it needs no second legend.
- **legend** — `<cohort> (<n> seg / <n> rec)`. Both denominators, because the histogram and the strip have different ones, and the strip's is the one that matters for any claim.
- **density** — each curve integrates to 1 over its own bins, so a cohort ten times larger is not a curve ten times taller. `distribution_summary.csv` carries the counts, means and quartiles for every cohort at both levels.
- **shared bins** — one grid per panel, from the pooled values of the cohorts drawn in it.

**How it is misread.** **This figure tests nothing, deliberately.** Segments overlap 29/30 in their forecast windows, so a $p$-value computed on them would be anticonservative by roughly thirty; a separation visible here is a reason to open `cross_subgroup/subgroup_heatmap.pdf`, which asks the question properly on per-recording values. And the mean of a filled `rmse_*` curve is **not** this pipeline's RMSE: it is a mean of per-segment roots, which by Jensen sits at or below the rooted-once figure the rest of the pipeline reports — the strip is on the pipeline's own arithmetic, the histogram is not, and `per_segment_root_note` in `summary.json` says so.

## `distributions/subgroup_histograms.pdf`

**In plain terms.** The same eight distributions cut the other way — by the eight subgroups rather than the three classes. Eight densities on one axes would be unreadable, so the figure uses the fact that a subgroup is a *subdivision of a class*: each column is one clinical class, and inside that column only that class's own subgroups are drawn, in the four tints of its colour. So you read across for "does the class matter" and within a cell for "does background or caesarean matter inside this class".

**What it shows.** A grid: eight rows (metrics) × three columns (clinical classes). Each cell holds at most four filled segment-level densities and their recording-level strips — the subgroups that appear under that class in this split.

**Axes.** Identical to the by-class figure, per cell.

**Terms on this figure.** The same eight metrics as above. The cells are populated from the data rather than from the subgroup stem's prefix: the clinical class comes from the target tensor and the subgroup from the shard basename, so their pairing is a property of the split being evaluated. `n_segments_with_class` and `n_segments_with_subgroup` in `summary.json` are the two denominators; where they differ, segments carrying a subgroup but no class fall outside this figure's columns and that difference is what counts them.

**How it is misread.** A cell with one curve is not a finding — it means only one subgroup of that class is present in this split. Cells are **not** comparable in height across columns any more than across rows: each density is normalised within itself. And the same caveat as above governs the whole figure: it describes, it does not test.

## `trajectory/trajectory_profile_pred_gap.pdf`

**In plain terms.** *"Does the predictive gain come and go over time?"* Two time scales of **one** readout, stacked. The top panel zooms in on **one 20-minute segment**: is UP more useful early or late within a window? The bottom panel zooms out to **a whole delivery**, gluing that recording's segments end to end on a real clock, so you can see the gain rise and fall over hours. The band around the top line is the spread across recordings, not uncertainty about the average.

**What it shows.** Top: `mc_pred_gap` against **time in segment**, as a median with an inter-quartile ribbon over recordings -- each anchor first averaged within a recording, so a recording contributing thirty-seven segments does not decide the shape. Bottom: one recording's segments assembled end to end on the absolute time axis, with overlapping timesteps averaged rather than drawn twice.

**Axes.** Top: seconds into the trimmed segment; nats per anchor. Bottom: hours before delivery, inverted so delivery is at the right; nats per anchor.

**Terms on this figure.**

- **top panel** — `mc_pred_gap` per anchor (how much UP improved the forecast at each anchor).
- **line** — the **median** over recordings; **ribbon** — the inter-quartile range (25th to 75th percentile) over recordings. Each recording is reduced to one value per anchor first, so a recording contributing 37 segments does not decide the shape.
- **bottom panel** — one recording's segments assembled on `t_abs = epoch + 4t`, hours before delivery, x axis inverted so delivery sits at the right.
- **`n_contributing`** — how many segments were averaged at each absolute timestep where they overlap. **`gap_before_s`** — a break in the data; the line stops rather than interpolating.
- The recording drawn is the **longest** one, not a representative one.

**How it is misread.** The shape of the top panel is **structural before it is physiological**: the warm-up prefix carries no loss term, the lag support is truncated until $t \ge L - 1$, and the last $H$ anchors are never scored -- so a profile that rises or falls at either end is the geometry rather than the model. On the bottom panel a **break is a break**: where a recording has a gap the line stops rather than being interpolated across, and a step at a segment join is an artifact of assembly, which is what `whole_delivery_boundaries.csv` exists to identify.

**Why it is one readout.** The bottom panel is a single axis in nats per anchor, and $K_t$ sitting on it beside `pred_gap` is routinely orders of magnitude larger — so a shared page draws the gap as a flat line along the bottom of the KL's range and reports a real movement as nothing. $K_t$ has its own page beside this one.

## `trajectory/trajectory_profile_kl.pdf`

The same two views of `kld_per_t` ($K_t$, how far UP moved the belief at each anchor), on its own axis. Everything about the reading is the page above's; what differs is the quantity and its one caveat: the unfloored KL is **inflated by an arbitrary factor whenever the prior variance sits on its clamp**, so its level is not comparable across checkpoints and a rise here is a rise only if the clamp state did not change. `pred_gap` carries no such factor, which is why both are reported and neither is read alone.

## `time_to_delivery/time_to_delivery_trajectory_pred_gap.pdf`

**In plain terms.** *"Does the predictive gain strengthen as labour progresses, and does it do so differently for the sick babies?"* The clinically interesting question. Time before delivery runs along the bottom with **delivery at the right**, so you read left to right toward birth. One line per clinical class. Each point is annotated with how many recordings are behind it — and that number is as much the figure as the line is, because at the far left only a few long recordings remain and a median over three recordings will wander for reasons that have nothing to do with the model.

**What it shows.** `mc_pred_gap` against time before delivery, one line per clinical class, as a median with an inter-quartile ribbon over **recordings**. Each point is annotated with the number of recordings behind it.

**Axes.** Time before delivery in hours on a $0.5$ h grid, inverted so delivery is at the right; nats per anchor.

**Terms on this figure.**

- **one line per class** — `hie` (red), `acidosis` (amber), `healthy` (green), in that order in the legend: cohorts are read worst first everywhere in this evaluation, which is also what orients every significance test.
- **line** — median over recordings in that window; **ribbon** — inter-quartile range over recordings. Each recording contributes one value per window, averaged over its own segments in that window, so a recording with eleven segments in a window does not outvote one with two.
- **annotated number at each point** — `n_recordings` in that class-and-window cell.
- **x axis** — `hours_before_delivery` $= -\mathrm{epoch}/3600$, binned on a fixed $0.5$ h grid (`TRAJECTORY_BIN_HOURS`, deliberately not a config key).
- Significance is tested **per window** with Holm across windows; the `pooled` row is flagged `confounded_by_time` and consumed by nothing.

**How it is misread.** The annotated $n$ is the point of the figure as much as the line is: a window's median can move because the cohort changed rather than because the coupling did, and at the tails of the axis the counts fall away sharply. The classes do not cover the axis equally, so comparing two lines at a window where one has three recordings and the other forty is comparing a number against noise -- which is why the significance is tested **per window** and the pooled test beside it is flagged `confounded_by_time` and consumed by nothing.

**Why it is one readout.** The unfloored KL is on the same nominal unit and not on the same scale — it is multiplied by an arbitrary factor whenever the prior variance sits on its clamp — so a page carrying both puts `pred_gap` on a range the KL set. It has its own page beside this one, and the two are compared in `time_to_delivery_trajectory.csv` rather than by eye.

**Beside it.** `time_to_delivery_windows_pred_gap.pdf` draws the per-recording distribution behind every point of this figure, the Holm-adjusted significance of each window, and the effect size of every class pair that survived — on this same axis. Read that one before quoting a gap between two lines here.

## `time_to_delivery/time_to_delivery_trajectory_kl.pdf`

The same figure for `source_conditioned_kl_raw`, on its own axis, with `time_to_delivery_windows_kl.pdf` beside it. Read exactly as the page above, with one addition: the unfloored KL is **inflated by an arbitrary factor whenever the prior variance sits on its clamp**, so a trajectory visible here and absent from the `pred_gap` page is a statement about which of the two is being read rather than about the coupling.

## `time_to_delivery/time_to_delivery_windows_pred_gap.pdf`

**In plain terms.** *"That trajectory has three lines on it — is the gap between them real, and where?"* The trajectory figure draws one number per class per window; this draws what that number was computed from, and the verdict on it, on the same axis. Read it top to bottom: the violins say what the recordings in each window actually looked like, and the strip directly beneath says whether that window's classes differ once every window has been corrected for. The bottom panel answers the separate question of whether a real difference is a *large* one.

**What it shows.** Three panels of `mc_pred_gap`. A violin per (window, clinical class) cell over one value per **recording**; beneath it $-\log_{10}$ of that window's Holm-adjusted $p$ against the $\alpha$ line; then a heatmap of Cliff's delta for every class pair that survived Holm, in any window.

**Axes.** Time before delivery in hours on the same $0.5$ h grid as the trajectory, inverted so delivery is at the right — on every panel including the heatmap, whose columns run in the same direction as the panels above it; nats per anchor on the violins; $-\log_{10} p$ on the strips.

**Terms on this figure.**

- **violin body** — the kernel density of that cell's per-recording values; **heavy bar inside it** — $Q_1$ to $Q_3$; **hairline** — Tukey's adjacent values; **white dot** — the median. The same mark as every other violin in the run.
- **dots instead of a body** — a cell with fewer than `MIN_GROUP_SIZE` = 3 recordings, or one whose values are all equal. Those cells are also the ones the test excludes, so a cell drawn as dots is a cell that carries no verdict either.
- **annotated number above each cell** — how many **recordings** are in it, never segments.
- **bar in the strip** — $-\log_{10}$ of `p_holm` for that window; **dashed line** — $\alpha = 0.05$ after Holm across the windows of this clock; **grey cross at zero** — a window that could not be tested at all.
- **heatmap row** — `<readout>: <left> vs <right>`; **colour** — Cliff's delta, signed, red where the left cohort runs higher.

**How it is misread.** **Every heatmap row reads more severe against less severe, as the axis order does.** The pair naming comes from the shared pairwise helper, which names a pair in the order it receives the cohorts and receives them worst first — so the rows run `hie vs acidosis`, `hie vs healthy`, `acidosis vs healthy`, a positive Cliff's delta means the *more severe* cohort runs higher, and reorienting a row by eye still flips its sign against the number in `time_to_delivery_pairwise.csv`.

**A bar that is absent and a cross at zero are different statements.** No bar with no cross means the window was tested and its $p$ came out at or near 1; a cross means fewer than two classes had enough recordings there, and nothing was tested. The strip is the only place that distinction is visible — both look like empty axis otherwise.

**A window is not an independent experiment.** The correction is across the windows of this clock as one family, which is what makes "eight windows survived" a claim rather than an artefact of having asked twenty-two times. The two readouts' pages are **not** jointly corrected, because they are two readings of the same recordings rather than two hypotheses.

## `time_to_delivery/time_to_delivery_windows_kl.pdf`

The same three panels for `source_conditioned_kl_raw`. Read exactly as the page above — including the row order and the sign of Cliff's delta — remembering that the KL's *level* is inflated wherever the prior variance is clamped. The Holm family is still this clock's windows.

## `second_stage/second_stage_trajectory_pred_gap.pdf`

**In plain terms.** *"Does the predictive gain change around the moment the second stage of labour begins?"* The same readout as the delivery clock, on the other clinical landmark — the one inside labour rather than at its end. Delivery is the end of a process, and two recordings four hours before it can be at completely different points of labour; this figure aligns them on the event instead. Only the recordings the labour-onset table places a second stage for appear on it.

**What it shows.** `mc_pred_gap` against signed hours from second-stage onset, one line per clinical class, as a median with an inter-quartile ribbon over **recordings**. Each point is annotated with the number of recordings behind it, and a dotted vertical marks the onset itself.

**Axes.** Signed hours from second-stage onset on a $0.5$ h grid, **negative before onset and positive after**; **not** inverted, because this coordinate reads naturally left to right. Nats per anchor.

**Terms on this figure.**

- **one line per class** — `hie` (red), `acidosis` (amber), `healthy` (green), worst first as everywhere in this evaluation.
- **line** — median over recordings in that window; **ribbon** — inter-quartile range over recordings. Each recording contributes one value per window, averaged over its own segments in that window, so a recording with eleven segments in a window does not outvote one with two.
- **annotated number at each point** — `n_recordings` in that class-and-window cell.
- **x axis** — `hours_from_second_stage` $= \texttt{second\_stage\_onset}/3600$, taken from the shard **without a sign flip**, unlike the delivery clock's $-\mathrm{epoch}/3600$.
- **dotted vertical at zero** — the onset of the second stage.

**How it is misread.** **The sign is the opposite convention from the delivery clock's**, and the axis label says so: negative is *before* the onset, positive after it. **The positive side is short by construction** — the second stage begins a couple of hours before delivery, so windows after onset hold far fewer recordings than windows before it, and the annotated $n$ is what says which is which. And the population is a **subset**: a recording with no recorded onset is dropped and counted in `second_stage_eligibility.csv`, so this figure describes fewer recordings than any other figure in the run.

## `second_stage/second_stage_trajectory_kl.pdf`

The same figure for `source_conditioned_kl_raw` on this clock, on its own axis. Read as the page above, with the KL's clamp caveat: its level is inflated by an arbitrary factor wherever the prior variance sits on its clamp, so the shape is readable and the height is not.

## `second_stage/second_stage_windows_pred_gap.pdf`

**In plain terms.** *"That second-stage trajectory has three lines on it — is the gap between them real, and where?"* The same page `time_to_delivery/time_to_delivery_windows_pred_gap.pdf` draws, on the other landmark: the violins say what the recordings in each window actually looked like, the strip beneath says whether that window's classes differ once every window of **this clock** has been corrected for, and the bottom panel says whether a real difference is a large one.

**What it shows.** Three panels of `mc_pred_gap`. A violin per (window, clinical class) cell over one value per **recording**; beneath it $-\log_{10}$ of that window's Holm-adjusted $p$ against the $\alpha$ line; then a heatmap of Cliff's delta for every class pair that survived Holm, in any window.

**Axes.** Signed hours from second-stage onset on the same $0.5$ h grid as the trajectory, negative before onset and positive after, **not** inverted and with the onset marked at zero — on every panel including the heatmap, whose columns run in the same direction as the panels above it; nats per anchor on the violins; $-\log_{10} p$ on the strips.

**Terms on this figure.**

- **violin body** — the kernel density of that cell's per-recording values; **heavy bar inside it** — $Q_1$ to $Q_3$; **hairline** — Tukey's adjacent values; **white dot** — the median. The same mark as every other violin in the run.
- **dots instead of a body** — a cell with fewer than `MIN_GROUP_SIZE` = 3 recordings, or one whose values are all equal. Those cells are also the ones the test excludes, so a cell drawn as dots carries no verdict either.
- **annotated number above each cell** — how many **recordings** are in it, never segments.
- **bar in the strip** — $-\log_{10}$ of `p_holm` for that window; **dashed line** — $\alpha = 0.05$ after Holm across the windows of **this** clock; **grey cross at zero** — a window that could not be tested at all.
- **heatmap row** — `<readout>: <left> vs <right>`; **colour** — Cliff's delta, signed, red where the left cohort runs higher.

**How it is misread.** **This clock's Holm family is its own.** The correction runs across the windows of the second-stage clock and is *not* joint with the delivery clock's, because the two are different alignments of an overlapping population — so a window significant on one and not the other is a statement about alignment, and a reader quoting a claim from both clocks is making two comparisons and is told so here.

**Every heatmap row reads more severe against less severe, as the axis order does.** The pair naming comes from the shared pairwise helper, which names a pair in the order it receives the cohorts and receives them worst first — so the rows run `hie vs acidosis`, `hie vs healthy`, `acidosis vs healthy`, and Cliff's delta is signed against that. Reorienting a pair by eye flips its sign against the number in `second_stage_pairwise.csv`.

**A bar that is absent and a cross at zero are different statements.** No bar with no cross means the window was tested and its $p$ came out at or near 1; a cross means fewer than two classes had enough recordings there and nothing was tested — which on the positive side of this axis is the common case rather than the exception, because it is short by construction.

## `second_stage/second_stage_windows_kl.pdf`

The same three panels for `source_conditioned_kl_raw` on this clock. Read exactly as the page above, including the row order and the sign of Cliff's delta, and with the KL's clamp caveat on the violins' heights.

## `cross_subgroup/subgroup_heatmap.pdf`

**In plain terms.** *"The grouped violins look different — is that real, or am I fooling myself?"* Eight cohorts each with a mean always produce a highest and a lowest; across eight metrics that is sixty-four numbers, and some will look separated whatever the truth. This figure applies the discipline. The **top** panel asks, per metric, "is there any difference at all across the cohorts?" — bars past the dashed line survived the multiple-comparison correction. The **bottom** panel asks, only for the survivors, "how *big* is the difference between each pair?" A metric can clear the line and still be negligible below, and at eight cohorts that is the common case.

**What it shows.** Top: $-\log_{10}$ of the Holm-adjusted $p$ of the omnibus test per metric, against the $\alpha$ line. Bottom: Cliff's delta for every pair of every metric that survived Holm.

**Axes.** Top: $-\log_{10} p$, one bar per metric. Bottom: metrics down, cohort pairs across, colour symmetric about zero. The pair columns are ordered by where their two cohorts fall in the clinical order, so a cohort's pairs sit together and the healthy pairs sit left.

**Terms on this figure.**

- **top bars** — $-\log_{10}$ of `p_holm`, the Holm-adjusted $p$-value of a **Kruskal-Wallis** omnibus test per metric. Taller = more significant. Floored so that $p = 0$ does not become an infinite bar.
- **dashed vertical** — $\alpha = 0.05$, Holm-adjusted. Bars to its right survived.
- **Kruskal-Wallis** — a rank-based "do these groups differ at all" test, used because these distributions are skewed and heavy-tailed.
- **Holm** — a step-down correction across the eight metrics **as one family**, so testing eight things does not manufacture a significant one.
- **bottom cells** — **Cliff's delta** for a pair of cohorts: $P(X > Y) - P(X < Y)$, running from $-1$ to $+1$. $0$ = the two distributions overlap completely. Magnitude labels follow Romano: $<0.147$ negligible, $<0.330$ small, $<0.474$ medium, above that large.
- **colour** — diverging, symmetric about zero, because the sign is meaningful. Which cohort is `left` is set alphabetically by the shared pairwise test; read the label for the sign.
- **empty bottom panel** — a **result** (nothing survived Holm), not a missing figure.
- Every test consumes **one value per recording**, so a bar reflects tens of observations, not the thousands of segments behind them.

**How it is misread.** The two panels answer different questions and only the second is about size. A metric can clear the $\alpha$ line and be negligible on the lower panel, which at eight cohorts is the common case rather than the exception. The lower panel is **empty by design** when nothing survived Holm -- that is a result, not a missing figure. Every test behind it consumes one value per **recording**, so a bar's height reflects tens of observations rather than the thousands of segments they were reduced from, and a signed delta's direction is only interpretable next to the `higher_is_better` column in `cross_subgroup_significance.csv`.

And the `<left> vs <right>` naming inside each column runs the same way the column ordering does: which cohort of a pair is `left` comes from the shared pairwise test, which names a pair in the order it receives the cohorts and receives them in the clinical one, and Cliff's delta is signed against that naming. So a column reads `healthy vs acidosis` in the healthy-first position, and a positive delta means the less severe cohort runs higher -- read the label, not the position, for the sign all the same.

## `events/deceleration_skill.pdf`

**In plain terms.** *"When the baby's heart rate actually dipped, did the model see it coming?"* This is the clinically meaningful version of the forecast question: not "was the average error small" but "did the forecast contain the **deceleration** that really happened". A deceleration detector is run on the true trace and on each forecast, and the two event lists are matched. Three outcomes: how many real dips were caught (**hit rate**), how many predicted dips never happened (**false alarms**), and how far off in time the caught ones were. All three are plotted against how far ahead the dip sat, because catching a dip 10 s out is easy and catching one 100 s out is not.

**What it shows.** Whether a forecast block contains the deceleration the true block contains, as a function of how far ahead in the horizon that deceleration sits. Three panels: the hit rate with a percentile bootstrap interval over recordings, the false-alarm rate, and the mean absolute timing error of the events that did match -- each with one line per branch.

**Axes.** Lead time in **seconds**, not horizon steps, running from the first usable step to the last; a fraction on the first two panels and seconds on the third.

**Terms on this figure.**

- **deceleration** — a dip in FHR, found by the ported clinical detector run on the trace in **bpm**.
- **hit rate** — of the true decelerations at this lead time, the fraction the forecast also contained. $1$ = all caught, $0$ = none.
- **false-alarm rate** — of the decelerations the forecast produced, the fraction with no true counterpart.
- **timing error** — for the matched pairs only, mean absolute distance in seconds between forecast and true onset.
- **one line per branch** — `base` and `full`.
- **lead time** — where in the two-minute horizon the event sat. The axis does **not** span the whole horizon: the detector discards anything within 30 s of either end of the block, leaving a 240-sample usable interior out of 480.
- Rates are **per event**, not per anchor — fixing the horizon step is what makes that exact, because exactly one anchor places a given raw sample at a given step.
- A hit rate of **zero** on a checkpoint that has not learned to forecast is the expected reading, not a broken detector. `n_forecast_events` tells the two apart: a model predicting a flat line produces no events to match and no false alarms either.

**How it is misread.** The horizontal axis does **not** span the whole two-minute horizon, and the part it omits is not missing data. The ported detector drops any event within 30 s of either end of what it is given, so a 480-sample block leaves a 240-sample interior and half the horizon is unsearchable by construction; `deceleration_skill.csv` carries the usable interval it was computed over. Every rate is **per event**, not per anchor: consecutive anchors' blocks overlap in 29 of their 30 steps, so an anchor-level rate would count one physiological deceleration once per anchor, and fixing the horizon step is what removes that -- exactly one anchor places a given raw sample at a given step. The `pseudo_replication_factor` in the JSON is that ratio, measured. A hit rate of zero on a checkpoint that has not learned to forecast is the expected reading rather than a detector failure, and the `n_forecast_events` column is what distinguishes the two: a model forecasting a flat line produces no events to match and no false alarms either.

## `events/contraction_triggered.pdf`

**In plain terms.** *"After a contraction, does the model expect the heart rate to dip?"* This is the textbook obstetric pattern the whole model is built around. Find every uterine contraction, line up the two minutes that follow each one, and average. The truth curve shows whether the dip really happens in this population; the two forecast curves show whether the model expected it. The grey band is the crucial part: it is the same average taken at **random** moments, so it shows what a dip of this depth looks like when nothing is going on. **A dip is only a finding if it leaves the band.**

**What it shows.** The truth, both forecast branches and their difference, averaged over the forecast blocks that follow a detected uterine contraction, each drawn over a **count-matched random-trigger null band**. The band is the null draws' mean $\pm 2$ standard deviations, so it brackets the null mean at every point; the shaded vertical span is the response window the dip statistic is taken over.

**Axes.** Seconds after the detected contraction onset; bpm, and bpm of the difference $\mu^q - \mu^p$ on the last panel.

**Terms on this figure.**

- **panels** — `truth`, `base`, `full`, and `difference` (full minus base, in bpm).
- **contraction onset** — a level crossing of the detected peak's own prominence on the stored UP trace, taken as $t = 0$.
- **grey band** — the **count-matched random-trigger null**: the same averaging done at randomly chosen anchors, drawn as mean $\pm 2$ standard deviations. It exists because the reported statistic is a *minimum over a window*, which is negative on any data at all — the band is what measures that selection bias instead of assuming it away.
- **shaded vertical span** — the response window the dip statistic is taken over.
- The truth, base and full curves are each corrected by the **truth's** pre-onset level so they stay comparable. The difference curve is **not** corrected, because a level cancels exactly out of a difference of two forecasts.
- The UP trace has already been advanced by 20 s upstream; adding that back would double-count a correction made once.

**How it is misread.** The null is the figure. The reported statistic is the *deepest point* of an average inside a window, and a minimum over a window is negative on any data at all -- so a dip is only a response if it leaves the band, and the band is what measures that selection bias rather than assuming it away. The contraction times are read from the stored UP trace, which the preprocessing has already advanced by 20 s; adding that back would double-count a correction made once, upstream. The truth, base and full curves are corrected by the **truth's** pre-onset level so they stay comparable; the difference curve is not corrected at all, because the level cancels out of a difference of two forecasts exactly.

## `events/conditioned_coupling.pdf`

**In plain terms.** *"Is UP more useful right after a contraction than at a quiet moment?"* If the model's coupling is real physiology rather than a general correlation, it should be strongest when something is actually happening in the uterus. So: measure the coupling at anchors near a contraction, measure it again at the same number of anchors picked from quiet stretches **of the same recording**, and subtract. The violins are those differences, one value per recording. **Zero is the null**, because the quantity drawn is already a subtraction — a violin above zero says the coupling really does concentrate around contractions.

**What it shows.** Both coupling readouts restricted to anchors within `event_lag_window_s` of a detected contraction, minus the same readouts on count-matched control anchors drawn from the same recordings -- one violin per clinical class, per readout, on per-recording values.

**Axes.** One panel per readout; the difference in nats per anchor, zero marked.

**Terms on this figure.**

- **panels** — one per coupling readout: `mc_pred_gap` and `source_conditioned_kl_raw`.
- **the plotted value** — `difference` = (mean over that recording's **event** anchors) − (mean over its **control** anchors). Already a difference, so zero is the null and not the value.
- **event anchor** — an anchor within `event_lag_window_s` seconds after a detected contraction.
- **control anchor** — drawn from the **same recording**, matched in count, away from contractions. Same-recording matching is what stops this being a comparison between two sets of deliveries.
- **one violin per clinical class**, one point per recording.
- **guards** — below 200 event anchors over 4 recordings the analysis draws nothing and records a skip rather than reporting a difference decided by which recordings happened to contribute.

**How it is misread.** Zero is the null here, not the value: the quantity drawn is already a difference, so a violin sitting above zero says the source pathway carries more near a contraction and one straddling it says nothing was detected. Controls are drawn **within** each recording and matched to that recording's event count, so this is not a comparison between recordings wearing two labels -- but it is still observational: an anchor near a contraction is also an anchor at a particular point in labour. The guards matter as much as the number: below 200 event anchors over four recordings the analysis records a skip and draws nothing rather than reporting a difference dominated by which recordings contributed.

## `sufficiency/sufficiency.pdf`

**In plain terms.** *"How much is the 48-number bottleneck costing us?"* Everything the model knows about the future has to squeeze through a 48-dimensional latent. This asks what would happen if it did not have to: a separate probe decoder is trained to forecast straight from the encoder's full internal state, skipping the bottleneck. The distance between that probe's score and the model's own is the price of the compression. It sits on the same axis as `pred_gap`, so the two are directly comparable: *what UP added* versus *what the bottleneck took away*. The lower panel is not decoration — it says whether the probe finished training, and an unfinished probe understates the cost.

**What it shows.** Two panels. The upper one is three violins over the **held-out** recordings -- $D_{\mathrm{oracle}}$, $D_{\mathrm{base}}$ and $D_{\mathrm{full}}$ -- with both gaps annotated in the title: $\Delta_{\mathrm{suff}} = D_{\mathrm{base}} - D_{\mathrm{oracle}}$, what the latent bottleneck costs the forecast, and `pred_gap` $= D_{\mathrm{base}} - D_{\mathrm{full}}$, the coupling readout, each with its bootstrap interval over recordings. The lower one is the probe's own fit: held-out and fit score against optimizer step, for the comparison probe and for the doubled-width refit the capacity check uses.

**Axes.** Upper: nats per anchor, lower is better. Lower: optimizer step against nats per anchor.

**Terms on this figure.**

- **upper violins** — `nll_oracle_block` (the bottleneck-free probe), `mc_nll_base_block` and `mc_nll_full_block`, each over the **held-out** recordings only.
- **oracle** — an evaluation-only decoder of the same capacity reading `target_state` (the encoder's own history state) instead of $z$. It is trained on half the evaluation recordings and scored on the other half, split at **recording** level.
- **`delta_suff_nats`** $= D_{\mathrm{base}} - D_{\mathrm{oracle}}$: the bottleneck's cost.
- **`pred_gap`** $= D_{\mathrm{base}} - D_{\mathrm{full}}$: UP's contribution. Both are annotated in the panel title with bootstrap intervals over recordings.
- **lower panel** — the probe's own training curve: `held_out_nats` and `fit_nats` against optimizer step, for the probe and for a **doubled-width** refit. If doubling the width still improves the held-out score, the probe was capacity-bound and the flag `capacity_bound` says so.
- **`convergence.converged`** — arithmetic on the held-out curve: the final quarter contributed at most a tenth of everything the score ever gained. A curve still descending at the right-hand edge is a probe that has not finished.
- $D_{\mathrm{base}}$ here is **not** the headline `d_base_mc_nats`: it is restricted to the held-out half, so all three violins share one denominator.

**How it is misread.** Three ways, and the first is the one that matters.

**It is an estimate, not a bound.** Two biases oppose and neither is measured. The probe conditions on `target_state` rather than on the raw target history, so it inherits the encoder's own information loss and the gap is *understated* by whatever the encoder already discarded. But it is fitted on the evaluation population while $D_{\mathrm{base}}$ comes from a model trained on the disjoint, healthier pretraining cohort, so part of the gap is a domain shift the probe does not suffer and it is *overstated* by that. Both sentences travel in the emitted JSON beside the number.

**The lower panel is a precondition, not a decoration.** A held-out curve still descending at its right-hand edge is a probe that has not finished, and an unfinished probe understates the gap -- the run's `convergence.converged` flag says so mechanically, and on a small population it will routinely be `false`. Read the upper panel only once the lower one has flattened.

**Only the held-out recordings are drawn.** $D_{\mathrm{base}}$ here is *not* the run's headline `d_base_mc_nats`, which is computed over the whole split; it is the same quantity restricted to the recordings the probe never saw, so that all three violins and both gaps share one denominator.

## The per-sample pages: `samples/<selection>/sample<index>_<guid>_epoch<epoch>.pdf`

**In plain terms.** *"Show me one baby."* Every other figure is an aggregate. This is one 20-minute segment of one delivery with everything the model did to it stacked on a shared time axis, so the rows can be read against each other: a contraction in row 1 lines up with a bump in row 5 and a bright band in row 6. It is the figure to open when a number looks strange and you want to see what produced it.

These have no fixed filename, so like the grouped variants they are documented as a family. Each page is the same seven-row diagnostic the training callback writes every validation epoch, drawn from the same builder.

**The seven rows, top to bottom.**

| Row | What it is |
|---|---|
| 1 | the raw **FHR** (the thing being forecast) and the raw **UP** (the source) on one time axis |
| 2 | the **forecast**, tiled across the recording: both branches with their $\pm 2\sigma$ bands against the truth |
| 3 | the **prior latent state** $\mu^p_t$ over the source-derived shift $\mu^q_t - \mu^p_t$ |
| 4 | the **per-dimension KL** — which of the 48 coordinates carried UP's information, over time |
| 5 | $K_t$ (`kld_per_t`) — how much UP moved the belief at each anchor |
| 6 | the head-averaged **lag attention** with its argmax marked |
| 7 | `source_kl_lag_map` — the KL attributed across lags |

Both lag panels carry the compensated-seconds axis described at the top of this file.

**The forecast row is tiled, not zoomed -- and the tiling is the figure's, not the model's.** The pass decodes every valid anchor at stride $1$, and every number in `per_sample.csv` is scored over all of them; the row merely selects which of those forecasts to draw. One anchor forecasts $H \cdot R$ raw samples, so the windows drawn are the anchors spaced $H$ apart starting at the first trained one -- consecutive, non-overlapping, with a thin dashed vertical at every edge. Drawn at the model's own stride, adjacent windows would overlap by $29/30$ and the panel would show each instant thirty times. Each window is decoded from one latent and never sees the window before it, which is what those edges are there to say. The span before the first window and whatever tail will not fit another whole one are blank by construction.

**Rows 3-7 show the trained anchors only.** The warm-up prefix is cut from all five and the untrained tail from all but the attention, so no column the objective never scored can set a colour scale. This is a property of the panels alone -- no metric, table or headline in this evaluation is computed from the cut arrays. The axes still span the whole recording -- the rows stay column-aligned with the two above them -- and the empty margins are marked in grey. A reader comparing an older page to a new one should expect the colour ranges to differ for this reason alone.

**Which pages exist.** `stratified/` holds a seeded, shard-stratified draw over the whole split, so a cap at or above the shard count reaches every shard. Beside it, one directory per headline metric and tail -- `pred_gap_low/`, `pred_gap_high/`, and the same for the block score and the KL -- holds the segments at the extremes of that metric. `sample_pages.csv` is the manifest of what was drawn.

**How they are misread.** A page is one segment of one recording, chosen because it was extreme or because a seeded draw picked it; it is an illustration, never evidence. The extreme pages in particular are selected *on* the quantity they display, so the panel showing that quantity is guaranteed to look unusual and says nothing about how often it does. The `<index>` in the filename is the position in the evaluation **dataset**, not in `per_sample.csv` -- the collection pass runs under a seeded shuffle -- and the two are reconciled by the `guid`/`epoch` round trip the analysis checks before it renders anything.
