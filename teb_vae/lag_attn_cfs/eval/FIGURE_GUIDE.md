# Figure guide

This guide explains what each evaluation figure shows, how the code computes it, and how to read its axes and encodings. [EVAL.md](EVAL.md) explains the workflow, outputs, settings and checks. [ATTRIBUTION.md](ATTRIBUTION.md) explains the attribution pass.

Each entry uses the filename that the analysis writes inside its own directory. PDF is the default format, and `eval_config.figure_format` changes the extension. `figure_manifest.json` beside this guide lists the fixed figure names per analysis directory. It also lists the figure families with dynamic names (sample pages, traces, attribution example pages and grouped variants) by pattern. The manifest is maintained by hand. No test or code reads it, so a new figure without a manifest entry is not detected automatically.

**Geometry is symbolic.** $T$ (steps per segment), $H$ (horizon), $F$ (anchor floor), $C_{\mathrm{keep}}$ (retained target channels), $L$ (lag bins, $L-1=$ `max_lag`), $d_z$ (latent dimensions) and $M$ (attention heads) belong to the checkpoint under evaluation. Read them from that run's `preflight.json` (under `causality`: `anchor_geometry`, `lag_support`, `group_delay_seconds`), from `summary.json` and from the checkpoint's `model_kwargs`.

A suggested reading order is `forecast/baseline_comparison.pdf`, then `coupling/pred_gap_distribution.pdf`, then the calibration figures. Read the source controls before you interpret KL or lag structure.

## Contents

- [Conventions used throughout](#conventions-used-throughout)
- [Terminology](#terminology)
- [Grouped variants](#grouped-variants)
- [Forecast figures](#forecast-figures)
- [Coupling figures](#coupling-figures)
- [Latent and lag-KL figures](#latent-and-lag-kl-figures)
- [Attention figures](#attention-figures)
- [Calibration figures](#calibration-figures)
- [Distribution figures](#distribution-figures)
- [Trajectory figures](#trajectory-figures)
- [Time-to-delivery figures](#time-to-delivery-figures)
- [Second-stage figures](#second-stage-figures)
- [Lag-clock figures](#lag-clock-figures)
- [Cross-subgroup figure](#cross-subgroup-figure)
- [Event figures](#event-figures)
- [Sufficiency figure](#sufficiency-figure)
- [Warm-up figures](#warm-up-figures)
- [Source-null figures](#source-null-figures)
- [Time-shift figure](#time-shift-figure)
- [Occlusion figures](#occlusion-figures)
- [Band-resolved lag-KL figures](#band-resolved-lag-kl-figures)
- [High-KL anchor figures](#high-kl-anchor-figures)
- [Spectral-skill figure](#spectral-skill-figure)
- [Recording-trace figures](#recording-trace-figures)
- [Attribution figures](#attribution-figures)
- [Sample pages](#sample-pages)

---

## Conventions used throughout

These conventions apply to all figures unless an entry states otherwise.

**Missing results are labelled.** A panel with no finite observations displays `no finite values`. Treat this as an unavailable measurement, and check the analysis status and counts.

**Error values use normalised coefficient units.** The targets are retained wavelet-modulus and phase-harmonic coefficients, which the loader standardises. They have no direct clinical scale. If you reverse the standardisation, the channels go onto very different scales and pooled comparisons become invalid. Compare scores only when the target definitions and the scoring geometry agree.

**Lag axes show stored-coefficient time.** The compensated lag is

$$\tau_\ell = 4(\ell + \delta)\ \text{s},$$

where $\ell$ is the lag bin and $\delta$ is the causal input delay of the source stream (the model's `source_delay_steps`). `compensated_seconds_axis` in `lag_axis.py` builds this axis, and `COEFFICIENT_LAG_AXIS_LABEL` labels it. In an unaligned configuration, such as `configs/default.yaml`, the source gate shifts no channel and $\delta = 0$. The stored timeline is canonical, and no other term enters the axis.

The axis does not show a physiological delay. The one-sided filter bank gives each channel a group delay, and the axis does not correct it. `causality.group_delay_seconds` in `preflight.json` gives the per-block range for the run. Each lag figure prints `GROUP_DELAY_NOTE`, and the records carry the long form `GROUP_DELAY_CAVEAT` as `axis_caveat`.

**Channel alignment does not make the lag axes physiological.** `causal_align_reference` is `null` in `configs/default.yaml`, so the default run is unaligned. An aligned run shifts its input channels onto a common reference. This replaces the channel-pair delay differences with one reference offset plus a remaining within-band dispersion. The target is still a coefficient block with its own reference delay. An aligned run records `source_reference_delay_s` beside the axis, not inside it.

**Cohort order is worst first.** Clinical classes appear as `hie`, `acidosis`, then `healthy`. Subgroups appear as `hie_cs`, `hie_no_cs`, `acidosis_cs`, `acidosis_no_cs`, `healthy_bg_cs`, `healthy_bg_no_cs`, `healthy_no_bg_cs` and `healthy_no_bg_no_cs`. Healthy is green, acidosis is amber and HIE is red. Subgroups use shades of their class colour. `cohort.ordered_groups` also orders the matching CSV rows. Unrecognised cohorts come after the known groups.

Training-callback plots can use other colours, so match cohorts by labels and legends. The clinical order is a display convention. Use `cross_subgroup` for statistical comparisons.

**A GUID always appears with its subgroup.** Each figure, title, filename or manifest that identifies a recording by GUID also names its subgroup and, where there is space, its clinical class. The per-recording traces, the attribution pages and the sample pages carry the subgroup in the filename and in the title. The forecast overlay, the contraction-detection figure and the lag heatmap carry it in the panel title.

**All figures share one publication style.** `teb_vae/lag_attn/eval/figures.py` defines the style, and `figures_seam` exposes it. The style has these properties:

- A double-column default width and a 6.5–8.5 pt serif type scale with STIX mathematics.
- Panels that share no axis sit two or three to a row. Only a time axis that needs the width gets a full-width panel.
- Open frames in a slate tone lighter than the text, except on boxed heatmaps. The grid is on the value axis only.
- Data lines are about twice the frame weight. Histogram bars are separated by white gaps, not outlined.
- Six saturated hues in a fixed order for non-cohort series. The order was checked for colour-vision separation.
- The two forecast branches have fixed styles on every figure, as on the sample pages: target-only is dashed slate and source-conditioned is solid vermillion.
- Legends have no border but a white backing, and sit in headroom above the data. A significance bar or a latent dimension that passes its threshold is drawn in blue, and one that does not is drawn in light grey.

`render_figure` writes every evaluation figure at `EVAL_SAVE_DPI` $=200$. The training callbacks keep their own resolution. Heatmaps and dense traces are rasterised inside the vector file, while text and frames stay vector. `label_panels` stamps panel letters (**a**, **b**, and so on) into the left-aligned titles, left to right, then top to bottom. Colour bars, hidden axes, insets and twin axes get no letter, and a single-panel figure gets none. Footnotes use reserved space below the axes.

**Forecast densities follow the checkpoint's likelihood structure.** A checkpoint can carry a per-channel scored horizon $H_c \le H$, with the cell mask $m_{\tau,c}=\mathbb 1[\tau<H_c]$. It can also carry an AR(1) residual (`forecast_ar_residual`) with coefficient $\phi_c$ per retained channel. Every score, calibration readout and predictive band uses the same terms. Cells past $H_c$ are not scored. Under the AR term, the density scores the innovation

$$e_\tau = r_\tau - \phi_c\, r_{\tau-1},$$

so the marginal spread of a forecast step is wider than the decoder's $\sigma_\tau$. `likelihood_structure` in `summary.json` and in the collection record states which terms a run used (`forecast_ar_residual`, `scored_horizon_per_channel`, `ar_coef_per_channel`, `scored_cells` of `block_cells`).

---

## Terminology

This section explains the model, the filename and column conventions, and the main quantities. A recording contains segments, and each segment contains forecast anchors. An anchor is the time from which the model predicts the next $H$ stored steps. Most statistical summaries give each recording one value.

### What the model predicts

The recordings contain fetal heart rate (FHR) and uterine pressure (UP) sampled at $4$ Hz. A strictly one-sided scattering and phase-harmonic transform turns them into coefficients, stored every $4$ s. The target stream comes from FHR (a scattering block `fhr_st` and a phase-harmonic block `fhr_ph`). The source stream comes from UP (`up_st` and `up_ph`). The warm-up budget keeps $C_{\mathrm{keep}}$ of the declared target channels for prediction. `causality.anchor_geometry.target_kept_width` in `preflight.json` is $C_{\mathrm{keep}}$.

At anchor $t$ the model predicts all $C_{\mathrm{keep}}$ retained target channels over the next $H$ steps, or $4H$ s. Under a scored-horizon mask, channel $c$ is scored over its first $H_c$ steps only. The model makes a **base** forecast from target history and a **full** forecast that also uses source history. Their predictive gap measures the score change from adding source history. The checkpoint supplies `horizon`, and `preflight.json` echoes it.

Two timing constraints decide which anchors are evaluated:

- **Warm-up sets the first usable anchor.** The anchor floor is $F=\max\bigl(B-1,\ \max_c(W'_c+d_c)\bigr)$, where $B$ is the warm-up budget, $W'_c$ is the leading delay of channel $c$, and $d_c$ is its alignment shift. `preflight.json` records it as `causality.anchor_geometry.anchor_floor`.
- **The searched lags must fit before the anchor.** When $F \ge$ `max_lag` $+$ `lag_floor`, the full lag window exists at each scored anchor. The pipeline measures this on each run (`causality.lag_support.lag_support_margin_steps` and `every_lag_valid_at_every_anchor`).

### Statistical terms used in the figures

| Term | How to read it |
| --- | --- |
| Median | The middle value after the observations are sorted. |
| Quartiles and IQR | $Q_1$ and $Q_3$ bound the middle half of the values. The interquartile range is $Q_3-Q_1$. |
| Density | The relative concentration of observations along an axis. Its width or height is not a recording count. |
| Paired comparison | A difference calculated within each recording before recordings are combined. |
| Centroid | The weighted mean location of a lag profile. |
| Entropy | A measure of how spread out normalised weights are. Lower entropy means more concentrated weights. |
| Argmax | The position of the largest value. On a flat profile, it can reflect a tie-breaking rule. |
| Degenerate profile | A profile that fails the stated shape criterion, so its peak position is not read alone. |
| Holm correction | A procedure that adjusts several tests together to limit false positives within a declared family. |
| Kruskal–Wallis | A rank-based test for differences among groups. A significant result does not identify which pair differs. |
| Mann–Whitney | A rank-based comparison between two independent groups. |
| Wilcoxon signed-rank | A rank-based test of paired differences, such as two scores from the same recordings. |
| Cliff's delta | An effect size that says how often values in one group exceed values in another. Its sign follows the stated group order. |
| Nominal and adjusted intervals | A nominal interval applies to one comparison. A family-adjusted interval accounts for a declared set of comparisons. |
| Attribution | A method that assigns a fitted model's output change to inputs or intermediate quantities. It does not establish a physiological cause. |

### How to read a name

| Piece | Means |
|---|---|
| `base` | The target-only branch (prior belief). |
| `full` | The source-conditioned branch (posterior belief). |
| `shuffled` | The negative control: the same model fed another recording's source. |
| `null` | The second control: the same model fed a zeroed source stream. |
| `mc_` prefix | Monte Carlo scoring: average the likelihood over $K$ latent draws, then take its negative logarithm. |
| `_block` suffix | Summed over the $H \times C_{\mathrm{keep}}$ forecast block (`block_width` in `preflight.json`), then averaged over anchors. |
| `_raw` suffix (on a KL) | Unfloored: no free-bits floor applied. |
| `_sq` suffix | Not rooted (a mean square). Its `_rms` partner is the rooted version. |

### Scores

A score is a negative log-likelihood. Lower is better, and a score is not bounded below by zero.

| Name | Meaning | Units |
|---|---|---|
| `nll_base_block` | The score of the target-only forecast. | nats per anchor |
| `nll_full_block` | The score of the source-conditioned forecast. | nats per anchor |
| `mc_nll_base_block`, `mc_nll_full_block` | The negative log of the likelihood averaged over $K$ latent draws. | nats per anchor |
| `mc_nll_shuffled_block` | The score with another recording's source (the negative control). | nats per anchor |
| `nll_persistence_block` | Baseline: hold the last observed coefficient vector for the whole window. | nats per anchor |
| `nll_climatology_block` | Baseline: predict the population mean, which is $0$ after z-scoring. | nats per anchor |
| `nll_segment_mean_block` | Baseline: predict the segment's own per-channel mean. | nats per anchor |
| `nll_oracle_block` | An evaluation-only decoder that reads the encoder state directly and bypasses the latent bottleneck. | nats per anchor |

**Nats per anchor** means that the score is summed over the valid coefficients of each forecast block, then averaged over scored anchors. Its magnitude depends on the block size, the predicted variance and the observations. A negative log-density can be negative. Compare score differences on the same target and support, and do not treat zero as a universal reference. Division by $H\cdot C_{\mathrm{keep}}$ gives a fixed-width rescaling. With masked forecast steps, that divisor is larger than the number of scored coefficients.

### Gaps

| Name | Meaning | Units |
|---|---|---|
| `pred_gap` | $D_{\mathrm{base}}-D_{\mathrm{full}}$ on the training scoring path. Positive means a source benefit under that convention. | nats per anchor |
| `mean_pred_gap` | The same difference with both branches decoded at their latent means, without sampling. | nats per anchor |
| `mc_pred_gap` | The same difference on the marginalised scores. The acceptance gate reads this estimator (`pred_gap_mc_nats`). | nats per anchor |
| `pred_gap_warm_lo` / `_mid` / `_hi` | The gap restricted to each warm-up tertile of the $C_{\mathrm{keep}}$ kept channels. The three sum to `pred_gap`. | nats per anchor |
| `pred_gap_<band>` | The gap restricted to one frequency band of the target coefficients. These also sum to `pred_gap`. | nats per anchor |
| `delta_suff_nats` | $D_{\mathrm{base}}-D_{\mathrm{oracle}}$: an estimate of the cost of the latent bottleneck, with probe and population biases. | nats per anchor |
| `mse_skill` | $1 - \mathrm{MSE}_{\mathrm{model}}/\mathrm{MSE}_{\mathrm{baseline}}$. $1$ is perfect, $0$ matches the baseline, and a negative value is worse. | dimensionless |
| `advantage_nats_per_anchor` | The NLL-space analogue of skill. It is a difference, not $1 -$ a ratio, because a log score has no natural zero. | nats per anchor |
| `pred_gap_rmse_pct`, `pred_gap_mse_pct` | The percentage of the target-only point-forecast error that the source removed, rooted and not rooted. Scale-free. | percent |
| `pred_gap_mc_likelihood_pct` | $100\bigl(e^{\texttt{mc\_pred\_gap}/(H \cdot C_{\mathrm{keep}})} - 1\bigr)$: the extra density that the full forecast puts on each observed coefficient. Budget-local. | percent |

Keep the scoring conventions separate. `mc_pred_gap` integrates over latent draws. `mean_pred_gap` decodes both latent means without sampling. `pred_gap` follows the training scoring path; under `base_decode: mean` (the value in `configs/default.yaml`), its base and full branches use different decoding conventions. Their values and their signs can differ. Each figure's labels identify which estimator it shows.

Do not divide a predictive gap by a block NLL to get a percentage. NLL has no natural zero and can be negative. The error percentages use a positive error denominator. The likelihood percentage is defined only for `gaussian_nll` and uses the fixed block width $H\cdot C_{\mathrm{keep}}$. It is budget-local: a change of horizon or of retained channels changes the denominator.

### KL

$K_t = \mathrm{KL}(q_t \Vert p_t)$ measures how far source history moved the belief at anchor $t$. Zero means that the source changed nothing.

| Name | Meaning | Units |
|---|---|---|
| `source_conditioned_kl_raw` | $K_t$ averaged over the scored anchors. The KL readout. | nats per anchor |
| `source_conditioned_kl_shuffled_raw` | The same with another recording's source (the specificity control). | nats per anchor |
| `kld_source_null` | The same with a zeroed source (the availability-clock control). | nats per anchor |
| `coupling_minus_clock_nats` | Their difference: the part of the coupling that comes from source variation. | nats per anchor |
| `kld_per_t` | $K_t$ before averaging: one value per anchor. | nats per anchor |
| `kld_per_dim`, `kld_per_head` | $K_t$ split across latent dimensions or attention heads. Each sums back to the total. | nats per anchor |
| `source_kl_lag_map` | $K_t$ split across the $L$ lags. It sums back to the total. | nats per anchor |
| `active_dims`, `top_dimension_share` | The number of latent dimensions above a threshold, and the share of the largest one. | count, fraction |

Do these four checks before you interpret KL:

1. Use the unfloored KL. A free-bits floor sets a minimum contribution and can hide a weak source pathway. The `_raw` columns keep the measured value.
2. Check the prior variance. KL contains $(\mu^q-\mu^p)^2/\sigma_p^2$, so a prior variance near its lower clamp can inflate it. Read `prior_variance_not_pinned` and the predictive gap beside KL.
3. Judge source specificity by predictive scores. Another recording's source can increase KL and also make prediction worse.
4. Check the availability clock. Source availability changes with time in the same way across batch rows, and row permutation cannot remove that pattern. Compare the matched KL with `kld_source_null`.

### Warm-up, bands and geometry

| Name | Meaning |
|---|---|
| `causal_warmup_steps` | Per channel: the leading delay, in 4 s steps, before the channel's coefficients complete warm-up. |
| `causal_delay_s` | Per channel: the composed one-sided group delay, in seconds. This delay makes the lag axis coefficient time. |
| `target_warm_frac` | The fraction of scored anchors at which every kept target channel is warm. The check requires exactly $1.0$. |
| `anchors_per_sample` | Anchors decoded per segment: `anchor_ceiling - anchor_floor` at the dense set. `causality.anchor_geometry.anchors_per_sample` in `preflight.json` is the expected value. |
| `source_lag_warmth_frac_st` / `_ph` | The fraction of attention mass on lags at which that stored source block is warm. |
| band | One of `slow_baseline`, `deceleration`, `variability`, `beat_to_beat`, `unknown`. It is the band of the analysing filter that produced the coefficient, not a bin of the forecast's own spectrum. |
| `unknown` | A channel whose centre frequency is not recoverable, because no selected phase-harmonic pair names its filter. It is never put into a neighbouring band. |

### Geometry reference

Each symbol below is a property of the checkpoint and dataset of the plotted run. The value column says where the run records it. The `preflight.json` paths are under `causality`.

| Symbol | Value | What it is |
|---|---|---|
| $T$ | the checkpoint's `model_kwargs`; `anchor_geometry.t_valid` is the valid span $T_{\mathrm{valid}}$ | steps per loaded segment |
| $H$ | `anchor_geometry.horizon` | forecast horizon in stored steps, $4H$ s |
| $H_c$ | `summary.json` `likelihood_structure.scored_horizon_per_channel` | scored horizon steps for kept channel $c$; $H$ without a scored-horizon mask |
| $F$ | `anchor_geometry.anchor_floor` | anchor floor, $\max(B - 1, \max_c(W'_c + d_c))$; nothing below it is decoded |
| anchors | `anchor_geometry.anchors_per_sample` | `anchor_ceiling - anchor_floor`, decoded per segment at the dense set |
| $C_{\mathrm{keep}}$ | `anchor_geometry.target_kept_width` | target channels that the warm-up budget kept |
| $H \cdot C_{\mathrm{keep}}$ | `anchor_geometry.block_width` | coefficients in one forecast block |
| $d_z$ | the checkpoint's `model_kwargs` | latent dimensions |
| $L$ | `lag_support.n_lags` | lag bins, `max_lag` $+1$; $4L$ s of source history |
| $M$ | the checkpoint's `model_kwargs` | attention heads |
| lag support margin | `lag_support.lag_support_margin_steps` | $F - (L-1) - \texttt{lag\_floor}$; $\ge 0$ means no anchor has truncated lag support |

---

## Grouped variants

### `<stem>_by_clinical_class.pdf` and `<stem>_by_subgroup.pdf`

**Purpose.** Compare the distribution of a metric across clinical classes or subgroups, with one value per recording.

**What it shows.** The analyses create these variants automatically from their declared per-recording tables. The filename names the source table and the grouping axis. Each violin's width is the estimated density. The thick internal bar spans $Q_1$ to $Q_3$. The thin whiskers extend to observations within $1.5$ interquartile ranges of the bar, and the white dot marks the median. The violin body extends to the group's observed extremes.

**Axes.** One panel per metric, titled with the metric, two panels to a row (one per row when there are more than four cohorts). Cohorts run across in descending severity. The CSV beside each figure has its rows in the same order.

**How to read it.** Each violin contains one value per recording, so eight recordings are eight observations, independent of their segment counts. Fewer than two groups produce a recorded skip. Use `cross_subgroup` to test group differences.

---

## Forecast figures

The `forecast` analysis (`analyses/forecast.py`) writes these figures into `forecast/`.

### `forecast/baseline_comparison.pdf`

**Purpose.** Compare both model branches with three simple predictors: the last observed coefficient vector, the population mean and the segment mean.

**What it shows.** The left panel shows the per-recording block score of each predictor as violins: `base`, `full`, `persistence`, `climatology` and `segment_mean`. Each violin is `nll_<name>_block` across recordings. The right panel shows `mse_skill` as one horizontal bar per (branch, baseline) pair, blue where the skill is positive and vermillion where it is negative. The whiskers are `mse_skill_lo` and `mse_skill_hi`, a percentile bootstrap interval over recordings.

- **persistence** carries forward the last observed step. `weight` is the validity signal, because the coefficients carry no sentinel value.
- **climatology** is $0$ per channel, the z-scored population mean. The normalisation statistics exclude the warm-up region.
- The baselines are scored at a fixed `BASELINE_LOGVAR` $= 0.0$ (`metrics.py`), because a point predictor has no variance of its own.

**Axes.** Top: nats per anchor, one violin per predictor. Bottom: skill across, dimensionless, with zero marked; one bar per pair down. The whiskers are asymmetric, because a percentile interval is not symmetric about its point estimate.

**How to read it.** The block score is a sum over the scored cells of the $H \cdot C_{\mathrm{keep}}$ block, so its absolute scale depends on the block size. Read the comparison between predictors. The baselines use the same density terms as the branches (the scored-horizon mask and, when on, the AR(1) residual). The figure draws the MSE-space skill. The NLL-space column in the CSV is a difference in nats.

### `forecast/anchor_profile.pdf`

**Purpose.** Show how predictive scores change from early to late anchors within a segment.

**What it shows.** Panel a shows both block scores against the anchor position. Panel b shows `pred_gap` against the same axis. Each position averages over every segment that scored an anchor there.

**Axes.** Anchor index in decimated (4 s) steps from the start of the trimmed segment; nats per anchor.

**How to read it.** The profile starts at the anchor floor $F$, because nothing below $F$ is decoded. When the lag support margin is non-negative, no position inside the profile has truncated lag support. The last $H$ anchors are not scored, because their forecast window runs past the end of the segment.

### `forecast/horizon_skill.pdf`

**Purpose.** Show how predictive scores and errors change with forecast lead time.

**What it shows.** Four panels against lead time, in a $2 \times 2$ grid:

1. Panel a shows $D_{\mathrm{base}}(\tau)$ and $D_{\mathrm{full}}(\tau)$. When the record carries the per-step cell counts, the curve is in nats per scored coefficient; otherwise it is in nats per horizon step.
2. Panel b shows the gap in nats per horizon step.
3. Panel c shows each branch's RMSE in normalised units.
4. Panel d shows the scored channels per anchor at each step.

**Axes.** Lead time in seconds, from $0$ to $4H$; the units above on the vertical axes.

**How to read it.** The curves use the single-draw path (`score_path` in the record). The Monte Carlo marginalisation does not commute with the sum over $\tau$, so a marginalised curve would not sum back to the marginalised headline. Horizon step $0$ is $4$ s ahead. Under a scored-horizon mask, step $\tau$ contains only the channels with $H_c > \tau$; panel d shows that count, and the per-coefficient normalisation of panel a removes it from the score. Under the AR(1) residual, each step scores the innovation given the previous step.

### `forecast/forecast_overlay.pdf`

**Purpose.** Compare individual forecasts with their observed targets, beside the raw signals they came from.

**What it shows.** The figure has one column per retained sample, at most `OVERLAY_SAMPLES` $= 3$. The samples come from a seeded draw over the retained set, class by class in cohort order. Each column uses the sample's median scored anchor.

- The top two rows show raw FHR (bpm) and raw UP (mmHg) on the fixed CTG scales. Green shading marks the contractions that the event detector finds on UP.
- Below them is one row per drawn channel, at most `OVERLAY_CHANNELS` $= 5$. With the kept-axis channel map (`band_channel_map_kept.csv`), `overlay_channels` picks the middle scattering channel of each clinical band in ascending frequency, plus one phase-harmonic channel. That channel is the one with the fewest scored horizon steps when $H_c$ is known. Without the map, it picks evenly spaced kept positions.
- Each channel row draws the target history as a dotted black line at $x<0$ (one horizon of history), the truth over the forecast window as a solid black line, the base (dashed slate) and full (solid vermillion) means, and each branch's predictive $\pm1\sigma$ band.
- Under the AR(1) residual, the band uses the marginal variance $v_\tau=\sigma^2_\tau+\phi_c^2 v_{\tau-1}$. Without it, the band uses the decoder's per-cell $\sigma$.
- Grey shading marks cells past the channel's scored horizon $H_c$, and the row title states "scored $H_c$ of $H$ steps". $H_c$ and $\phi_c$ come from the collection record's `likelihood_structure`.

Each column title carries the GUID, the subgroup and the class. `forecast_overlay_samples.csv` carries the same identity, the anchor step, the hours before delivery and the drawn channels.

**Axes.** Seconds from the anchor's causal endpoint, shared by every row. $x<0$ is the history that the model read, and $x>0$ is the $4H$ s forecast window; zero is ruled. Raw rows are in bpm and mmHg, and channel rows are in the loader's $z$ units. Each row shares its $y$ range across the columns. A footnote keys the band.

**How to read it.** Each column is one anchor of one retained recording. The figure needs `eval_config.caps.waveforms`; without that cap, the analysis writes no overlay. The drawn channels are a few of the $C_{\mathrm{keep}}$ channels. A mean that departs from the truth inside the grey region does not change any score, because those cells are not scored. An `mse` checkpoint gets no band, because its log-variance head is not trained.

---

## Coupling figures

The `coupling` analysis (`analyses/coupling.py`) writes these figures into `coupling/`.

### `coupling/pred_gap_distribution.pdf`

**Purpose.** Show whether source history improves prediction and how consistently it does so across recordings. Zero means no improvement under the displayed estimator.

**What it shows.** Three blocks:

1. **Top row: one histogram per estimator, side by side.** The estimators are `mc_pred_gap` (Monte Carlo marginalised, which the gate reads), `mean_pred_gap` (both branches decoded at their latent means) and `pred_gap` (the training path). Each histogram is over recordings, with zero marked and the bootstrap interval on its own mean shaded.
2. **Bottom left: the three estimators as violins**, one colour per estimator (`ESTIMATOR_COLORS`), kept across the page.
3. **Bottom middle and right: two scatters, one point per recording.** They show the mean-decoded estimator against the marginalised one and against the training-path one. Each scatter has the identity line and the two zero lines. The title gives Spearman's $\rho$.

The summary rows carry the share of recordings above zero (`positive_fraction`), the sign agreement of each pair (`same_sign_share`) and `n_recordings_scored`.

**Axes.** Nats per anchor throughout. A histogram's height is a count of recordings.

**How to read it.** The shaded band is the interval on the mean, not the range of the data. The three histograms are three estimators of one quantity. Their signs can differ, because the marginalised score is the log of an average likelihood over $K$ draws: a broad prior can score better there while its mean forecast is worse. The scatters show whether the same recordings change sign. A recording that scored no anchors is absent, not at zero. A positive gap on this figure does not separate source content from the availability clock; read `source_null/source_null_difference.pdf` beside it.

### `coupling/pred_gap_percent.pdf`

**Purpose.** Express the predictive comparison as percentage changes in error and in likelihood density.

**What it shows.** Three panels side by side. The left panel shows the distribution of `pred_gap_rmse_pct` over recordings, with zero marked and the interval on the mean shaded. The middle panel ("Error space") shows `pred_gap_rmse_pct` and `pred_gap_mse_pct` side by side. They are the same ratio with and without a root, so the mean-square value is larger where both are positive. The right panel ("Likelihood space") shows `pred_gap_mc_likelihood_pct` and `pred_gap_mean_likelihood_pct` in the estimator colours of `pred_gap_distribution.pdf`. Under an `mse` checkpoint the right panel is empty, because these values are undefined.

**Axes.** Percent on every panel. The histogram's height is a count of recordings.

**How to read it.** The right panel divides by $H \cdot C_{\mathrm{keep}}$, and the warm-up budget sets $C_{\mathrm{keep}}$. Two runs at two budgets therefore have different denominators. The two error-space panels are scale-free.

---

## Latent and lag-KL figures

### `latent/kl_spectrum.pdf`

**Purpose.** Show how much each latent dimension changes when source history is added.

**What it shows.** The per-dimension KL, sorted. Dimensions above the activity threshold are blue and the rest are light grey. The analysis record (`analyses/latent.py`) carries the active-dimension count, the top dimension's share and the total.

**Axes.** Latent dimension (sorted) across; nats per anchor up.

**How to read it.** A concentrated spectrum means that a few dimensions carry most of the KL. It does not show forecast usefulness. Read the total in the record beside the spectrum. Check `prior_variance_not_pinned` first: a prior variance at its clamp multiplies every bar by an arbitrary factor.

### `lag_kl/lag_kl_profile.pdf`

**Purpose.** Show how the attention weights distribute KL across stored source lags.

**What it shows.** The left panel draws the per-lag KL attribution in three forms (`analyses/lag_kl.py`):

- *raw* sums to the KL;
- *support-corrected* divides each lag by its contributing-anchor count;
- *untruncated* uses only the anchors where every lag exists.

Dashed vertical lines mark the raw argmax and the support-corrected argmax. The right panel shows the contributing-anchor count per lag, which is the denominator of the correction. A footnote carries `GROUP_DELAY_NOTE`, and the records carry the full `axis_caveat`.

**Axes.** Compensated lag in seconds (stored-coefficient time) across. Nats per anchor up on the left panel; anchors per segment up on the right panel.

**How to read it.** With a non-negative lag support margin (`causality.lag_support` in `preflight.json`), every lag exists at every scored anchor, so the support correction and the untruncated form do not change the profile and the three curves coincide. A run with a lower floor or a wider `max_lag` separates them. A peak is a position in stored-coefficient time, not a physiological delay.

---

## Attention figures

The `attention` analysis (`analyses/attention.py`) writes these figures into `attention/`.

### `attention/attention_profile.pdf`

**Purpose.** Show where the attention places weight over source history, averaged over heads and per head.

**What it shows.** Panel a shows the head-averaged attention profile over lags in its raw, support-corrected and untruncated forms. Panel b shows one curve per head. Both panels shade any truncated-support region. Each head's entropy and its ceiling are in `attention_per_recording.csv` and the analysis record, not on the page.

**Axes.** Lag in seconds (stored-coefficient time) across; attention weight up.

**How to read it.** The analysis quotes entropy against the attainable ceiling $\operatorname{mean}_t \log \min(t+1, L)$. This ceiling equals $\log L$ whenever the lag support margin is non-negative. The analysis takes the entropy per anchor and then averages it. It never takes the entropy of the averaged profile, because the entropy of a mixture is at least the mean entropy of its parts; a model whose lag focus moves between anchors would then read as unfocused.

### `attention/lag_heatmap.pdf`

**Purpose.** Show how attention over source lags changes across the anchors of one retained segment.

**What it shows.** Head-averaged attention as a heatmap over time in segment and lag, for one retained sample, with the anchor floor marked. The title names the GUID, the subgroup and the class. `per_sample.csv` carries the segment's place before delivery.

**Axes.** Time in segment (s) across; lag in seconds (stored-coefficient time) down, each row spanning its own 4 s bin. Colour is attention weight.

**How to read it.** The heatmap shows one retained segment. It needs `eval_config.caps.attention`; without that cap, the analysis writes no heatmap. The map is head-averaged, so the structure of one head can be hidden; `attention_profile.pdf` shows the per-head profiles.

---

## Calibration figures

The `calibration` analysis (`analyses/calibration.py`) writes these figures into `calibration/`.

### `calibration/pit_reliability.pdf`

**Purpose.** Check whether forecast uncertainty matches the observed errors. For example, an interval with about $68\%$ nominal coverage should contain about $68\%$ of observations.

**What it shows.** Panel a shows the histogram of the probability integral transform (PIT) of the standardised residuals, against the uniform density. Panel b shows the empirical central coverage at the exact erf nominals, against the diagonal. The coverage table and the record carry the observed and nominal values.

**Axes.** Panel a: PIT value across, density up. Panel b: nominal probability across, observed coverage up; the diagonal is perfect calibration.

**How to read it.** The nominals are $\operatorname{erf}(k/\sqrt{2}) = 0.6827,\ 0.9545,\ 0.9973$ for $k = 1, 2, 3$. The two-sigma value is not $0.95$, which belongs to $\pm 1.96\sigma$. The unit is one scored coefficient, so the count is `n_coefficients`. Cells past a channel's $H_c$ are left out. Under the AR(1) residual, the standardised residual is the innovation $e_\tau$ over the decoder's $\sigma_\tau$, which is the quantity the density scores. An `mse` checkpoint gets no figure, because its log-variance head is not fitted.

### `calibration/logvar_distribution.pdf`

**Purpose.** Show whether predicted log-variances concentrate near their lower or upper clamp.

**What it shows.** The distribution of the decoder's log-variance, with both `logvar_clamp` bounds marked. The analysis record and `summary.json` carry the floor and ceiling fractions.

**Axes.** Log-variance across; fraction of coefficients up.

**How to read it.** A mean log-variance alone can hide concentration at both ends. The analysis states a recommended `logvar_clamp` revision per coefficient, which is the axis over which the block score reduces. It states *no change* when neither end binds.

---

## Distribution figures

The `distributions` analysis (`analyses/distributions.py`) writes these figures into `distributions/`.

### `distributions/class_histograms.pdf`

**Purpose.** Compare distribution shapes across clinical classes at two levels of aggregation.

**What it shows.** Nine metrics, one panel each, at two levels on the same axes. A filled density shows one value per segment. A strip above it shows the median, interquartile range and range of one value per recording. The coupling pair `mc_pred_gap` and `mean_pred_gap` sits in adjacent panels, so you can compare the two estimators.

**Axes.** The $x$ label is the metric's unit in $z$ space, and the panel title names the metric. The legend (drawn once, in the first row) names the cohorts without counts. The summary table carries the segment and recording counts. The vertical axis is density, so a cohort with ten times more segments does not draw a taller curve.

**How to read it.** A strip that is much narrower than the density below it means that most of the spread is within-recording variation. The analysis computes no test. At the dense evaluation geometry, consecutive anchors share $H-1$ of their $H$ horizon steps, so a per-segment $p$-value would be anticonservative. Use `cross_subgroup` for tests.

### `distributions/subgroup_histograms.pdf`

**What it shows.** The same nine metrics by subgroup, nested in one column per clinical class. Each cell holds at most four curves, in tints of one hue. Each cohort is a faint fill under a full-opacity hairline outline. The code draws all fills first and all outlines second, so each outline stays visible above every fill.

---

## Trajectory figures

The `trajectory` analysis (`analyses/trajectory.py`) writes these figures into `trajectory/`.

### `trajectory/trajectory_profile_pred_gap.pdf`

**Purpose.** Follow the predictive gain within a segment and across a recording assembled from its segments.

**What it shows.** `mc_pred_gap` against time in segment, and against absolute time across a delivery:

$$t_{\mathrm{abs}} = \mathrm{epoch} + T_{\mathrm{trim}} + 4t.$$

Here $\mathrm{epoch}$ is the stored segment start (seconds, negative before delivery), $T_{\mathrm{trim}}$ is `LOADER_TRIM_S` (the fixed trim that the loader cuts from the front of each stored segment; preflight refuses any other `trim_minutes`), and $t$ is the loaded anchor step. Overlapping steps are averaged, and `n_contributing` travels beside them. `traces.absolute_seconds` applies the same formula in the traces, the forecast overlay and the attribution pages.

**Axes.** Time in segment (s) on the within-segment panel; hours before delivery on the whole-delivery panel; nats per anchor. The whole-delivery panel's title names the recording (GUID, subgroup, class).

**How to read it.** The within-segment panel starts at the anchor floor, as `forecast/anchor_profile.pdf` does. Across a delivery, the line is lifted wherever nothing was decoded. This includes the undecoded warm-up prefix of every segment, so the whole-delivery line is a series of short pieces one segment apart. In `trajectory_delivery_summary.csv`, a **break** is a gap longer than one segment stride, which means a missing segment.

The KL has its own page, because KL and `pred_gap` can differ in scale by a large factor. On a shared axis, the gap would draw as a flat line.

### `trajectory/trajectory_profile_pred_gap_mean.pdf`

**What it shows.** The same two views of `mean_pred_gap`, the gap on the mean-decoded forecasts without a latent draw. The axes, the unit and the reading notes are those of the page above. The two pages show two estimators of one quantity, and their signs can differ.

### `trajectory/trajectory_profile_kl.pdf`

**What it shows.** The same two views of `kld_per_t`, on its own axis. The reading is that of the `pred_gap` page. A prior variance at its clamp inflates the KL by an arbitrary factor, so its level is not comparable across checkpoints. `pred_gap` carries no such factor.

---

## Time-to-delivery figures

The `time_to_delivery` analysis (`analyses/time_to_delivery.py`) writes these figures into `time_to_delivery/`. Windows are $0.5$ h wide (`BIN_WIDTH_HOURS`). The bin width and the significance level are module constants, not config keys.

### `time_to_delivery/time_to_delivery_trajectory_pred_gap.pdf`

**Purpose.** Compare predictive-gain trajectories across clinical classes as delivery approaches.

**What it shows.** `mc_pred_gap` against hours before delivery, $-\mathrm{epoch}/3600$, by class, with one value per recording in each window. The legend names the classes. `n_recordings_total` in the trajectory table gives the recordings per class.

**Axes.** Hours before delivery, with delivery at the right; nats per anchor.

**How to read it.** The analysis tests significance per window, with Holm across windows as one family. The `pooled` row carries the flag `confounded_by_time`, and nothing consumes it: a pooled difference between classes with different recording lengths mixes in when they were recorded. The KL has its own page, because the clamp factor puts it on a different scale. `time_to_delivery_trajectory.csv` holds both readouts. `time_to_delivery_windows_pred_gap.pdf` shows the per-recording distribution and the tests behind each point.

### `time_to_delivery/time_to_delivery_trajectory_kl.pdf`

**What it shows.** The same figure for `source_conditioned_kl_raw`, on its own axis, with `time_to_delivery_windows_kl.pdf` beside it. The unfloored KL is inflated by an arbitrary factor wherever the prior variance sits at its clamp.

### `time_to_delivery/time_to_delivery_windows_pred_gap.pdf`

**Purpose.** Show the recording-level values behind each class trajectory, and the statistical comparisons within each window.

**What it shows.** Three panels of `mc_pred_gap`:

1. A violin per (window, class) cell over one value per recording.
2. Below it, $-\log_{10}$ of the window's Holm-adjusted $p$, against the $\alpha$ line.
3. A heatmap of Cliff's delta for every class pair that survives Holm in any window.

Each cell carries its recording count. A cell below `MIN_GROUP_SIZE` $= 3$ recordings, or a cell whose values are all equal, is drawn as points instead of a density. The test excludes the same cells.

**Axes.** Hours before delivery on the $0.5$ h grid, with delivery at the right on every panel, including the heatmap. Nats per anchor on the violins; $-\log_{10} p$ on the strips.

**How to read it.** A missing bar means that the window was tested and its $p$ is at or near $1$. A grey cross at zero means that fewer than two classes had enough recordings, so nothing was tested. Each heatmap row reads more severe against less severe: `hie vs acidosis`, `hie vs healthy`, `acidosis vs healthy`, in that order. A positive Cliff's delta therefore means that the more severe class runs higher. `time_to_delivery_pairwise.csv` carries the same signs. The Holm family is the windows of this clock. The two readouts are not corrected jointly, because they are two readings of the same recordings.

### `time_to_delivery/time_to_delivery_windows_kl.pdf`

**What it shows.** The same three panels for `source_conditioned_kl_raw`, with the same row order and sign convention. The KL level is inflated wherever the prior variance is clamped. The Holm family is again this clock's windows.

---

## Second-stage figures

The `second_stage` analysis (`analyses/second_stage.py`) writes these figures into `second_stage/`. The axis is signed hours from second-stage onset (`second_stage_onset / 3600`): negative before onset and positive after. It uses the same $0.5$ h grid as the delivery clock and is not inverted. Recordings without a recorded onset are dropped and counted in `second_stage_eligibility.csv`.

### `second_stage/second_stage_trajectory_pred_gap.pdf`

**Purpose.** Follow predictive gain around the onset of the second stage of labour.

**What it shows.** `mc_pred_gap` against the signed onset clock, by class, as a median with an interquartile ribbon over recordings. Each recording gives one value per window, the mean over its segments in that window. Each point carries its recording count, and a dotted vertical line marks the onset. `n_recordings_total` in the trajectory table gives the recordings per class.

**Axes.** Signed hours from second-stage onset; nats per anchor.

**How to read it.** The sign convention is the opposite of the delivery clock: negative is before the onset. The second stage begins a short time before delivery, so the windows after onset hold fewer recordings; the annotated $n$ shows how many. The population is the subset of recordings with a recorded onset.

### `second_stage/second_stage_trajectory_kl.pdf`

**What it shows.** The same figure for `source_conditioned_kl_raw` on its own axis. The clamp factor inflates the KL level, so read the shape and not the height.

### `second_stage/second_stage_windows_pred_gap.pdf`

**Purpose.** Show recording-level predictive gaps and class comparisons within windows before and after second-stage onset.

**What it shows.** The same three panels as `time_to_delivery_windows_pred_gap.pdf`, on the signed onset clock: violins per (window, class) cell, the Holm-adjusted $p$ strip, and the Cliff's delta heatmap. The `MIN_GROUP_SIZE` rule is the same.

**Axes.** Signed hours from second-stage onset, not inverted, with the onset marked at zero on each panel that carries the clock. Nats per anchor on the violins; $-\log_{10} p$ on the strips.

**How to read it.** This clock has its own Holm family. It is not corrected jointly with `time_to_delivery`, because the two clocks are different alignments of an overlapping population. The heatmap rows and the sign of Cliff's delta follow the delivery-clock page; `second_stage_pairwise.csv` carries the same signs. A grey cross at zero means that fewer than two classes had enough recordings in the window.

### `second_stage/second_stage_windows_kl.pdf`

**What it shows.** The same three panels for `source_conditioned_kl_raw` on this clock, with the same row order and sign convention, and with the clamp caveat on the violin heights.

---

## Lag-clock figures

The `lag_clocks` analysis (`analyses/lag_clocks.py`) writes these figures into `lag_clocks/`. It computes per-segment statistics of two lag profiles: the KL attribution (`kl`) and the attention (`attn`). The two tested readouts are the centroids of the two profiles. All other statistics are tabled and drawn without tests.

### `lag_clocks/lag_time_to_delivery.pdf`

**Purpose.** Follow the location of the lag attribution as delivery approaches, and compare it across classes.

**What it shows.** One heatmap per clinical class: lag down, time before delivery across, and colour as the share of the KL attribution in each lag bin, over one profile per recording in each window. Two trajectory panels follow. They show the median centroid across recordings with its interquartile ribbon, once for the attribution and once for the attention. Each window carries its recording count, and the median spread is a dashed line. Each heatmap's title gives its class and recording count. The trajectory legend appears once, on the first trajectory panel.

**Axes.** Hours before delivery on the $0.5$ h grid, with delivery at the right. Lag in seconds of stored-coefficient time, with lag $0$ at the bottom. Colour is a share in $[0, 1]$.

**How to read it.** All class heatmaps share one colour scale, so one colour means one share in every panel. Each window is normalised to sum to one, so the heatmap shows where the attribution sits, not how much there is; `time_to_delivery_trajectory_pred_gap.pdf` shows the amount. The page draws centroids and no peak; `lag_time_to_delivery_features.pdf` draws the argmax beside the degeneracy share.

### `lag_clocks/lag_time_to_delivery_windows.pdf`

**Purpose.** Test class differences in lag centroids within each delivery-time window.

**What it shows.** Five panels. For each tested readout (the attribution centroid and the attention centroid), a violin per (window, class) cell over one value per recording, with the window's Holm-adjusted $-\log_{10} p$ below it against the $\alpha$ line. A final heatmap shows Cliff's delta for every class pair that survives Holm in any window, for either readout.

**Axes.** Hours before delivery, with delivery at the right; lag in seconds of stored-coefficient time on the violins; $-\log_{10} p$ on the strips.

**How to read it.** Each (clock, readout) pair is its own Holm family, which gives four families across the analysis. They are not corrected jointly. The heatmap rows read more severe against less severe, so a positive Cliff's delta means that the more severe class's centroid sits further back in the past. A cell below `MIN_GROUP_SIZE` $= 3$ recordings is drawn as points, and the test excludes it.

### `lag_clocks/lag_time_to_delivery_features.pdf`

**Purpose.** Describe lag-profile shape beyond the centroid.

**What it shows.** Eight panels against the clock, one per drawn statistic (`DRAWN_STATISTICS`): skewness, median lag, interquartile range, entropy, near-mass share, far-mass share, peak lag, and the share of segments whose peak is degenerate. Each panel shows the median across recordings, with the attribution's interquartile ribbon. The solid line is the KL attribution and the dashed line is the attention, in the class colours. On the peak panel, each point also carries the window's degenerate share as a percentage (`PANEL_ANNOTATIONS`).

**Axes.** Hours before delivery on the $0.5$ h grid, with delivery at the right. Each panel labels its own $y$ axis: seconds of stored-coefficient time for the median, the interquartile range and the peak; nats for the entropy; a dimensionless value for the skewness; a share in $[0, 1]$ for the mass and segment shares.

**How to read it.** Nothing on this page is tested, which keeps each clock's Holm family at two. `lag_clocks_significance.csv` carries the only tests of the analysis. Read the peak panel together with the degenerate-share panel: `entmax15` gives some lags exactly zero weight, so a flat or nearly empty profile still has an argmax. The attribution is $K_t$ times the attention, so it inherits the prior-variance inflation and the attention does not. The near and far shares are measured from the shortest lag on the axis, so their meaning does not depend on $\delta$. They do not sum to one, because the middle of the axis belongs to neither.

### `lag_clocks/lag_second_stage.pdf`

**What it shows.** The panels of `lag_time_to_delivery.pdf` on the second-stage clock: a share-of-attribution heatmap per class, then the centroid trajectories with their ribbons and the median spread.

**Axes.** Signed hours from second-stage onset, negative before onset, not inverted, with the onset marked at zero. Lag in seconds of stored-coefficient time, with lag $0$ at the bottom.

**How to read it.** The sign convention is the opposite of the delivery clock. The windows after onset hold fewer recordings, and the annotated $n$ shows how many. The population is the subset with a recorded onset, counted in `second_stage/second_stage_eligibility.csv` and in this analysis's record.

### `lag_clocks/lag_second_stage_windows.pdf`

**What it shows.** The five panels of `lag_time_to_delivery_windows.pdf` on the signed second-stage clock, with the onset at zero.

**How to read it.** This clock's Holm families are its own, per readout. The heatmap rows read more severe against less severe. A grey cross at zero on a strip means that fewer than two classes had enough recordings in the window.

### `lag_clocks/lag_second_stage_features.pdf`

**What it shows.** The eight panels of `lag_time_to_delivery_features.pdf` on the signed second-stage clock, not inverted, with the onset at zero.

**How to read it.** The reading notes of the delivery-clock features page apply. The population is the subset with a recorded onset, and the windows after onset hold fewer recordings.

---

## Cross-subgroup figure

### `cross_subgroup/subgroup_heatmap.pdf`

**Purpose.** Show effect sizes for cohort comparisons, and mark which comparisons pass the multiple-testing correction.

**What it shows.** Two panels (`analyses/cross_subgroup.py`):

- **Top:** the omnibus test per metric, a Kruskal–Wallis across cohorts over one value per recording. Each bar is $-\log_{10}$ of the Holm-adjusted $p$ (Holm across the metrics as one family), against the dashed $\alpha$ line. Each metric label carries its recording count. A metric that cannot be tested keeps its row, labelled `not testable`.
- **Bottom:** Cliff's delta for every cohort pair of the metrics that survive the correction. The pairs come from Mann–Whitney tests whose $p$-values are Holm-corrected again within each metric (`p_holm` beside `p_value` in `cross_subgroup_pairwise.csv`). When no metric survives, the panel is empty and says so.

**Axes.** Top: $-\log_{10}$ Holm-adjusted $p$ across, metrics down in omnibus order. Bottom: surviving metrics down in the same order, cohort pairs across; colour is the signed effect size on a symmetric scale.

**How to read it.** A bar past the line is not an effect size. With many cohorts and recordings, a metric can pass the omnibus test while every pairwise delta is small, so read the bottom panel before you quote a difference. Each column names the more severe cohort first, so a positive Cliff's delta means that the more severe cohort runs higher. Every test uses one value per recording.

---

## Event figures

The `events` analysis (`analyses/events.py`) writes these figures into `events/`.

### `events/conditioned_coupling.pdf`

**Purpose.** Compare predictive gain and KL shortly after detected contractions with count-matched control anchors from the same recordings.

**What it shows.** One panel per readout: `pred_gap_mc_nats` (the Monte Carlo gap per anchor) and `source_conditioned_kl_raw` (the per-anchor $K_t$). Each panel has one violin per clinical class, in cohort order, of the per-recording difference:

$$\Delta = \overline{x}_{\mathrm{event}} - \overline{x}_{\mathrm{control}}.$$

Event anchors are anchors within `event_lag_window_s` after a detected contraction onset. Control anchors are count-matched anchors from the same recording. Each violin label carries its recording count, and zero is marked as "no effect". `conditioned_coupling.csv` and `conditioned_coupling_per_recording.csv` carry the event and control means.

**Axes.** Clinical class across; event minus control in nats up, with zero marked.

**How to read it.** The controls are matched within each recording, count for count. An anchor with no preceding contraction counts as a control. The contraction timing comes from the raw UP trace that the collection pass carries, so the analysis runs over every anchor of the split. `weight` masks the gaps. An event whose span touches an interpolated region is dropped. Below `MIN_EVENT_ANCHORS` $= 200$ event anchors or `MIN_EVENT_RECORDINGS` $= 4$ recordings, the analysis records a skip and draws nothing.

The record names two readouts that this cell does not compute, with the reasons: deceleration forecast skill and the contraction-triggered response. Both score a clinical FHR trace in bpm, and the target here is a coefficient block.

### `events/contraction_detection.pdf`

**Purpose.** Show the detected contractions on the traces they were found in, so you can check the detector by eye.

**What it shows.** One panel per drawn segment: up to `DETECTION_SAMPLES_PER_CLASS` $= 2$ retained segments per clinical class, in cohort order. Each panel shows the raw UP trace in mmHg on the fixed CTG scale (the loader's $z$ units when its statistics are unknown). The detector's onset (up-triangle), peak (circle) and end (down-triangle) are vermillion. Green shading marks the `event_lag_window_s` seconds after each onset. Grey shading marks the stretch before the first decoded anchor. The panel title carries the GUID and its subgroup. One footnote keys the markers and the shading. `contraction_detections.csv` lists every detection on every retained segment with its identity (GUID, class, subgroup, epoch), its onset, peak and end in seconds, and the peak pressure with its unit.

**Axes.** Seconds into the loaded (trimmed) segment across; UP pressure up.

**How to read it.**

- The figure re-runs the detector that wrote `seconds_since_contraction` on the per-anchor table, on the same trace and validity. It checks the detector on a few retained segments, not the population.
- The detector interpolates invalid samples before smoothing and removes any event whose span touches one. A visible rise without markers over lost signal is such a removed event. The onset follows the rising-edge walk-back (`onset_convention` in the record).
- The figure needs retained raw UP (`eval_config.caps.waveforms`). Without it, the analysis record states why the figure is absent.

---

## Sufficiency figure

### `sufficiency/sufficiency.pdf`

**Purpose.** Estimate the predictive cost of the latent bottleneck. The analysis compares the base branch with an evaluation-only decoder that reads the encoder state directly.

**What it shows.** Two panels (`analyses/sufficiency.py`):

- **Left ("Block score"):** per-recording violins of `nll_oracle_block`, `mc_nll_base_block`, `mean_nll_base_block` and `mc_nll_full_block`. $\Delta_{\mathrm{suff}}$ is read between the oracle and base violins, and `pred_gap` between the base and full violins.
- **Right ("Oracle probe fit"):** the probe's held-out and fit curves against the optimizer step, one pair per probe width multiplier. The hue names the width, the held-out curve is solid and the fit curve is dashed.

`sufficiency_summary.csv` carries the two gaps with their intervals.

**Axes.** Top: nats per anchor. Bottom: optimizer step across; nats per anchor up.

**How to read it.** The gap is an estimate, not a bound, and the JSON record carries both bias directions. The probe conditions on the encoder state instead of the target's own history, which omits the encoder's information loss and biases the gap down. The probe is fitted on the evaluation population, which biases the gap up. The two biases oppose, and neither is measured. The convergence flag is a precondition: an unfinished probe understates the gap, and a held-out curve that never improves is not converged.

---

## Warm-up figures

The `warmup` analysis (`analyses/warmup.py`) writes these figures into `warmup/`. It draws the two budget figures through `teb_vae/lag_attn_cfs/warmup_budget.py`.

### `warmup/warmup_tertiles.pdf`

**Purpose.** Compare predictive gaps for three groups of retained channels with short, medium and long warm-up, and show how much attention falls on warm source coefficients.

**What it shows.** Left ("Gap by warm-up tertile"): `pred_gap_warm_lo`, `_mid` and `_hi` per recording as violins, with zero marked. Right ("Warm-lag attention"): `source_lag_warmth_frac_st` and `_ph` per recording. The right panel appears only when the table carries the warmth fractions.

**Axes.** Top: nats per anchor. Bottom: fraction of attention mass. The two quantities have separate axes, because one shared axis would flatten the smaller one.

**How to read it.** The three tertiles are a decomposition: they sum to `pred_gap` over the same denominator, and the analysis checks that identity (`recomposition`). Their relative sizes carry the information, and their absolute sizes carry the block's scale. The stored source blocks have long warm-up delays, so part of the searched lag window can fall on source coefficients that are still in warm-up. A low warmth fraction therefore reflects that geometry.

### `warmup/causal_warmup_budget.pdf`

**Purpose.** Show which channels the warm-up budget kept or dropped, their warm-up requirements, and the resulting anchor floor.

**What it shows.** Two panels, target stream above and source stream below. Each declared channel, including the dropped ones, gets one horizontal bar against a seconds axis whose origin is the anchor's causal endpoint. A kept channel's bar spans $[-\Delta W'_c, 0]$ with $\Delta = 4$ s: it ends at the anchor, and its left end is its warm-up. A dropped channel is drawn at $\delta_c = 0$, so its bar runs forward through the shaded forecast window. A dashed line marks the budget $-\Delta B$. Each panel's title names its stream, and its legend counts the kept and dropped channels. The record's `summary` carries the per-block counts and the warm-up range.

**Axes.** Seconds relative to the anchor, negative to the left; one row per channel. The shaded span at the right is the forecast window of $H$ steps.

**How to read it.** A one-sided filter gives each channel a strictly leading delay, so a long bar means a slow filter. The source panel also shows the budget line, but the source is not gated, so source bars that cross the line do not violate anything. Bars that run past the right edge are clipped at the frame. The figure depends only on the shard and the budget, so two runs over the same dataset and budget draw the same figure.

### `warmup/causal_warmup_tradeoff.pdf`

**Purpose.** Show the trade between more retained channels and fewer usable anchors as the warm-up budget changes.

**What it shows.** Three step curves against the candidate budget $B$: target channels kept, anchors admitted, and tiles per sample at phase $0$. A dashed vertical line marks the configured budget, with its three values annotated. Grey shading marks the budgets at which no tile fits.

**Axes.** Warm-up budget $B$ in steps across, with $\Delta B$ in seconds on a top axis; one count axis for all three curves.

**How to read it.** The curves are steps. The anchor count comes from the maximum warm-up of the kept channels, so all thresholds between the same two consecutive channel warm-ups keep the same channels and admit the same anchors. The grey region is infeasible. The figure shows feasibility, not forecast quality: checkpoints at two budgets cannot load each other's geometry, and their nats are not comparable.

---

## Source-null figures

The `source_null` analysis (`analyses/source_null.py`) writes these figures into `source_null/`.

### `source_null/source_null_difference.pdf`

**Purpose.** Compare the matched-source KL with the zero-source KL. Source availability changes predictably with time, so it can move the latent without any source variation.

**What it shows.** Left ("Coupling minus clock", with $n$ in the title): the distribution of `coupling_minus_clock_nats` over recordings, with zero marked. Right ("Matched and source-null arms"): `source_conditioned_kl_raw` (`matched`) and `kld_source_null` (`source-null`) side by side on one support.

**Axes.** Nats per anchor. The histogram's height is a count of recordings.

**How to read it.**

- A large difference between two large numbers is not the same as a large coupling. The right panel shows both terms.
- The reference line (legend `no excess`) marks the point where the zero-source response accounts for the whole matched KL.
- The verdict `coupling_exceeds_availability_clock` is INCONCLUSIVE while `eval_config.clock_margin_min_nats` is `null`, which is the value in `configs/eval_overrides.yaml`. With a margin set, the verdict is stated on the lower end of the bootstrap interval.
- A zeroed source is a flat trajectory, and the encoder's response to it is not exactly the availability pattern's own response. The difference is therefore a slightly weaker statement than "the clock alone", and the record says so.
- The posterior is a bounded residual on the prior. At a silent source, the mean term of the KL depends on the delta head alone, so `kld_source_null` can be large.

### `source_null/source_null_lag_profile.pdf`

**Purpose.** Show at which lags the matched-source KL attribution exceeds or falls below the zero-source attribution.

**What it shows.** Top ("Lag profile"): the matched attribution and the null arm, both in nats per anchor. `matched` is $\mathrm{KL}(q(z|Y,U)\,\|\,p(z|Y))$, and `source-null` is the same divergence with a zeroed source. Bottom: their signed difference with zero drawn, the four `occlusion_bands` shaded and labelled, and the delta mask where one is emitted. The bottom panel can carry two notes: `degenerate, no mask` (no mask emitted) and `edge-censored peak` (the peak is on the longest searched lag, so the true peak can lie beyond the window).

**Axes.** Compensated lag in seconds (stored-coefficient time) across; nats per anchor up on both panels.

**How to read it.**

- The signed difference sums over lags to `coupling_minus_clock`, the quantity that `clock_margin_min_nats` gates. Every share on the page is a share of the positive part, so the rectified total is an upper bound on the gated scalar. `rectified_frac` gives the ratio.
- A negative bin is a valid state. The null arm has its own attention, which can exceed the matched arm's at a lag. A negative bin means no clock-exceeding coupling at that lag.
- The analysis withholds the delta mask when the profile is degenerate, because `entmax15` gives a flat or nearly empty profile a confident argmax. The geometry-fixed bands need no such estimate.
- The axis is stored-coefficient time.

---

## Time-shift figure

### `time_shift/time_shift_control.pdf`

**Purpose.** Separate a coupling that reads the current uterine activity from a coupling that reads a recording-level source state. The analysis (`analyses/time_shift.py`) poses each segment's posterior again on the source of the nearest segment of the same recording that shares no stored sample with it. Everything on the target side (encoder state, prior, anchors) stays the segment's own. Both arms are mean-decoded.

**What it shows.** Three panels:

- **a ("KL by source"):** one line per recording, in its class colour, joining its mean source-conditioned KL under three sources: the zeroed source (`kld_source_null`, from the collection pass), the own source time-shifted to the partner segment, and the own source aligned.
- **b:** $\Delta K = K^{\mathrm{aligned}} - K^{\mathrm{shifted}}$, one grey point per recording, with the mean over recordings and its $95\%$ bootstrap interval in blue, and zero ruled.
- **c:** the same for $\Delta g = g^{\mathrm{aligned}} - g^{\mathrm{shifted}}$, the mean-decoded forecast gap under the two sources. The base branch is the same in both arms and cancels.

**Axes.** a: the three source arms across; KL in nats per step up. b: $\Delta K$ up, in the same unit. c: $\Delta g$ in nats per anchor up. Panels b and c have no horizontal scale; the points are jittered for visibility.

**The verdict.** The analysis writes `coupling_is_time_specific` to its record and the headline, not to the page. The rule is:

1. PASS when the lower end of the bootstrap interval of $\Delta K$ is above zero.
2. FAIL when it is not.
3. INCONCLUSIVE when too few recordings have a non-overlapping partner for an interval.

$\Delta g$ travels beside the verdict and does not decide it. The verdict is reported beside the pre-registered acceptance verdicts, not among them.

**How to read it.**

- A FAIL means that the same patient's source from another stretch of labour moves the posterior as much as the aligned source. In panel a, the two arms then sit close to each other.
- The partner is wrong in time by at least one stored segment. A segment whose recording has no non-overlapping partner is counted in the pairing census and excluded, so the recordings here are a subset. `eval_config.caps.time_shift` caps the arm with a seeded draw.
- $\Delta K$ and $\Delta g$ can disagree. `lag_high_kl_usefulness.pdf` asks the same question at the anchor level.
- The class colours in panel a are labels. `time_shift_per_recording.csv` feeds the grouped variants for class comparisons.

---

## Occlusion figures

The `occlusion` analysis (`analyses/occlusion.py`) zeroes the source values within each lag band in `eval_config.occlusion_bands`, re-encodes the stream, and scores one anchor per segment. It writes these figures into `occlusion/`. `eval_config.caps.occlusion` caps the segments.

### `occlusion/occlusion_horizon_delta.pdf`

**Purpose.** Measure the predictive cost of removing each lag band, resolved by horizon step.

**What it shows.** One curve per configured band ("Forecast cost of band removal"): the change in NLL against an un-occluded reference, per horizon step. Positive means a worse forecast without the band. The legend gives each band's name and its live fraction. The band's lag range is its `occlusion_bands` entry.

**Axes.** Horizon step $0 \ldots H-1$ across; nats per anchor per step up, with zero marked.

**How to read it.**

- Read each curve with the band's live fraction. A band that reaches into the warm-up has less source to remove, so a near-zero curve there can mean that little source was present.
- The bands are lag ranges relative to each segment's scored anchor. The analysis draws one anchor per segment and holds it fixed across every band and the reference, because the source pathway has memory.
- A negative curve means that removing the band improved the forecast. The per-recording table carries the spread across segments.
- The intervention edits values and leaves the availability clock unchanged. The analysis measures that invariance on every occluded encode.
- The analysis applies no threshold and writes no verdict.

### `occlusion/occlusion_clock_delta.pdf`

**Purpose.** Follow each band's occlusion cost through clinical time.

**What it shows.** One panel per band, titled with the band and its lag range in compensated seconds, with one line per clinical class. Each line is the mean per-recording delta in that band, window by window, with zero marked.

**Axes.** Hours before delivery across, with delivery at the right; nats per anchor up.

**How to read it.** Nothing on this page is tested. The analysis scores one anchor per segment and is capped in segments, so many (class, window) cells are below the minimum group size. Read `n_recordings` in `occlusion_clocks.csv` before you read a movement. [EVAL.md](EVAL.md) gives the arithmetic behind the cap. The observational counterpart is `lag_kld_scaled`, on the same bands and the same grid.

---

## Band-resolved lag-KL figures

The `lag_kld_scaled` analysis (`analyses/lag_kld_scaled.py`) writes these figures into `lag_kld_scaled/`.

### `lag_kld_scaled/lag_kld_scaled_time_to_delivery.pdf`

**Purpose.** Show the amount of KL attribution in each fixed lag band as delivery approaches. A normalised lag profile shows only relative shares; this figure keeps the magnitude.

**What it shows.** One panel per `occlusion_bands` band, with one line per clinical class. Each line is the mean per-recording `total_nats` of the KL attribution restricted to the band, window by window.

**Axes.** Hours before delivery across, with delivery at the right; nats per anchor up. Each panel title gives the band and the seconds it spans.

**How to read it.**

- Nothing on this page is tested. The analysis adds no Holm family and writes no significance table.
- The bands are fixed by geometry and not chosen from the KL, so a statistic on a band does not test its own selection. They are the bands that the occlusion analysis removes.
- A line that moves between panels shows the informative past moving. A line that moves within one panel shows that band's coupling changing magnitude.
- `near_mass` and `far_mass` are absent on bands, because both are measured from the axis start; on a band they would re-base onto the band start.

### `lag_kld_scaled/lag_kld_scaled_second_stage.pdf`

**What it shows.** The panels of the delivery-clock page against signed hours from second-stage onset, drawn left to right with the onset marked.

**How to read it.** The notes above apply. The population is the subset of recordings with a recorded onset, under the eligibility rule of the `second_stage` analysis.

---

## High-KL anchor figures

The `lag_high_kl` analysis (`analyses/lag_high_kl.py`) selects anchors by KL or by forecast gain and writes these figures into `lag_high_kl/`. The anchor bands are `high`, `rest`, `top` and `gain` (`ANCHOR_BANDS`). The pooled thresholds are in `lag_high_kl_thresholds.csv`.

### `lag_high_kl/lag_high_kl_selection.pdf`

**Purpose.** Show how the analysis selects anchors, and compare the pooled lag profiles of the selected bands.

**What it shows.** Four panels:

1. The pooled per-anchor KL, one step histogram per clinical class on a $\log_{10}$ axis. Dashed lines mark the band thresholds, labelled with the band and its value in nats.
2. The pooled KL attribution per lag for the `high`, `rest` and `top` bands. Shading marks the hot lags: the lags whose pooled attribution is in the upper $30\%$ across the lag axis (`HOT_LAG_QUANTILE` $= 0.7$).
3. A heatmap of where each anchor's attribution peaks, by KL decile (`N_KL_QUANTILE_BINS` $= 10$, lowest at the bottom) and lag. Each cell is the share of that decile's anchors whose `argmax_lag` falls at that lag.
4. One violin per class of the per-recording contraction enrichment: the high-anchor share within `event_lag_window_s` of a contraction minus the share outside it. Only recordings with at least `MIN_ENRICHMENT_ANCHORS` $= 5$ anchors in each arm enter, and zero is marked.

**Axes.** Panel 1: $\log_{10}$ nats across, density of anchors up. Panels 2 and 3: stored-coefficient lag in seconds across; nats per anchor up in panel 2, KL decile up in panel 3. Panel 4: class across, share difference up.

**How to read it.**

- The thresholds are one number per run, pooled over all classes. A class whose histogram sits to the right has more high anchors under the same cut.
- The hot-lag set comes from the attribution that it shades, so a pooled profile that is high on the hot lags is true by construction. The set is used for the per-segment share on the clock pages.
- If all deciles peak at the same lags, the argmax follows the window's edges and the K/V receptive field. Read the `attention` and `source_null` pages before you read a peak position here.
- The enrichment violin is descriptive. No test runs on it, and its population is a subset that the record counts.
- The lag axis is stored-coefficient time.

### `lag_high_kl/lag_high_kl_time_to_delivery.pdf`

**Purpose.** Follow the lag profiles and the prevalence of high-KL anchors as delivery approaches.

**What it shows.** One heatmap per class: lag down, window across, and colour as the share of the `high` band's KL attribution in each lag bin. Each window is normalised to one, and the classes share one colour scale. Four trajectory panels follow, each the median per class with its interquartile ribbon and the recording count on each point:

1. The high-anchor share of a segment's anchors, with the `top` band's share dashed.
2. The `high` band's KL centroid in seconds, with the `rest` band's centroid dashed.
3. The hot-lag share of the attribution, with the attention's share dashed.
4. The `high` band's forecast gain in nats per anchor, with the `rest` band's gain dashed.

Panels 1 and 2 are the tested readouts. Panels 3 and 4 are not tested per window; the one paired run-level test of the gain is on the usefulness page. The class legend appears on the first trajectory panel only.

**Axes.** Hours before delivery across, with delivery at the right. Stored-coefficient lag in seconds up on the heatmaps and the centroid panel; a share or nats per anchor up on the other panels.

**How to read it.**

- The heatmap shows where the high anchors' attribution sits, not how much there is. The `high` band's `total_nats` is in the trajectory table.
- The ribbon is an interquartile range over recordings, not an interval on the median. The test is on `lag_high_kl_time_to_delivery_windows.pdf`.
- The high-anchor share uses the pooled threshold. Over the whole population it is $30\%$ by construction, so only its distribution over classes and windows carries information.
- The `rest` centroid is the contrast for the `high` centroid. If both move together, the lag structure is not specific to the high-KL anchors.

### `lag_high_kl/lag_high_kl_time_to_delivery_windows.pdf`

**Purpose.** Test class differences in the two tested high-KL readouts within each delivery-time window.

**What it shows.** For `high_lag_centroid_kl_s` and `high_anchor_frac`: one violin per (window, class) cell over one value per recording, the window's Holm-adjusted $p$ below it, and at the foot Cliff's delta for every class pair that survives. The pairs read more severe against less severe, so a positive value means that the more severe class runs higher.

**Axes.** Hours before delivery across, with delivery at the right. The violin rows share one $y$ label because the page carries two units (seconds for the centroid, a share for the fraction); each row's title names its readout.

**How to read it.** Each (clock, readout) pair is its own Holm family: two here and two on the second-stage page, none joint. Classes with fewer than three recordings in a window are excluded from the test and recorded, but their violins stay on the page.

### `lag_high_kl/lag_high_kl_second_stage.pdf`

**What it shows.** The panels of `lag_high_kl_time_to_delivery.pdf` on the signed second-stage clock, drawn left to right with the onset marked.

**How to read it.** The population is the subset of recordings with a recorded onset, under the eligibility rule of the `second_stage` analysis.

### `lag_high_kl/lag_high_kl_second_stage_windows.pdf`

**What it shows.** The panels of `lag_high_kl_time_to_delivery_windows.pdf` on the signed second-stage clock, with the onset at zero. This clock has its own Holm families, and the population is the onset subset.

### `lag_high_kl/lag_high_kl_time_to_delivery_histogram.pdf`

**Purpose.** Compare full lag distributions of the selected high-KL anchors across classes and delivery-time windows.

**What it shows.** Both profile sources appear side by side in each row: `kl` on the left and `attn` on the right (`HISTOGRAM_BANDS` $=$ `high`, `top`).

- **Rows 1–6, three per band, pooled over every window of the clock.**
  - The classes overlaid as step outlines, with the recording-weighted pool of all classes dashed in grey. The legend appears on the first panel.
  - Each class minus the pooled distribution, in percentage points of share per lag, with zero marked.
  - The cumulative distributions, with each class's median lag dropped to the axis as a dotted line.
- **Row 7, the density violins.** The `high` band's cells on the clock, one body per (window, class), dodged by class within each window. The body is the cell's own lag distribution: its half-width at a lag is that lag's share, drawn as a step outline. It is not a kernel density. Inside each body, a heavy bar spans the quartile lags, a white dot marks the median lag and a short black tick marks the centroid. The recording count stands above each body. A cell below three recordings is faint and dashed.
- **Row 8, the ridges.** The same cells with lag across, one ridge per window and one column per class. Labour runs down the page on both clocks, with the window farthest from delivery at the top. Inside each ridge, the class's own distribution pooled over the clock is a grey dashed outline. The window centre is labelled at the left of the first column, the recording count at the right of each ridge, and the median lag is ticked on the baseline. All bodies on a panel share one scale.
- **Bottom two rows: distances against the clock.** The left column shows each window against its own class pooled over the clock, one line per class. The right column shows the distance between classes within each window, one line per class pair, on the `high` band's `attn` profile. The pair lines use a blue, purple and black palette that is different from the class colours. The upper row is the $1$-Wasserstein distance in seconds. The lower row is the Jensen–Shannon distance (base $2$, bounded by $1$).

**Axes.** Stored-coefficient lag in seconds across on the pooled panels and the ridges. Share up on the overlay, percentage points on the difference panel, cumulative share on the third panel, and window centre up on the ridges. Hours before delivery across on the violins and the distance panels, with delivery at the right. Lag in seconds up on the violins; seconds up on the Wasserstein row; Jensen–Shannon distance up on the row below.

**How to read it.**

- The code normalises each recording's distribution before the cell mean, so each recording counts once. The heatmap on `lag_high_kl_time_to_delivery.pdf` normalises after averaging and therefore shows where the coupling mass sits.
- Nothing on this page carries a $p$-value. A distance between two estimated distributions is positive almost surely, even when the populations coincide, so read it beside the recording counts. The tests on the shape features are on `lag_high_kl_time_to_delivery_histogram_features.pdf`, and the within-recording drift is on `lag_high_kl_time_to_delivery_histogram_drift.pdf`.
- A violin's inner bar shows the quartile lags of the cell's distribution, not the spread across recordings.
- Wasserstein measures displacement along the lag axis and does not detect a distribution that broadens in place. Jensen–Shannon measures overlap and saturates at $1$ once two supports separate. Read the two rows together.
- `attn` counts each selected timestep once, and `kl` weights each by $K_t$. A shift in one column only tells you which readout carries it.
- A missing body or ridge is a window with no scored recording for that class.
- The lag axis is stored-coefficient time.

### `lag_high_kl/lag_high_kl_second_stage_histogram.pdf`

**What it shows.** The panels of `lag_high_kl_time_to_delivery_histogram.pdf`, with the density violins and the four distance panels on the signed second-stage clock and the onset at zero. The ridgeline still puts the earliest window before onset at the top. The population is the onset subset.

### `lag_high_kl/lag_high_kl_subgroup_histogram.pdf`

**Purpose.** Compare high-KL lag distributions across classes and their subgroups, pooled over the evaluated population without time windows.

**What it shows.** The `high` band pooled over the whole evaluated population. Each recording's selected-anchor profile is averaged over all its segments and normalised once, then averaged over the recordings of the cohort. There is one column per clinical class. Inside each column, the subgroups are tints of the class colour, and the class's pooled distribution is a black dashed reference. Each profile source (`kl` then `attn`) has two rows: the distributions, and their cumulative forms with the medians dropped to the axis. `lag_high_kl_subgroup_histogram.csv` carries the curves and the delivery counts on both cohort axes.

**Axes.** Stored-coefficient lag in seconds across; share up on the first row of each pair, cumulative share on the second.

**How to read it.** Nothing here is tested, and a small subgroup draws as confidently as a large one, so read the count. This pool covers every segment in the run; the pooled rows of the clock page cover one clock's binned recordings.

### `lag_high_kl/lag_high_kl_time_to_delivery_histogram_features.pdf`

**Purpose.** Test class differences in three properties of each recording's lag histogram within delivery-time windows.

**What it shows.** For `hist_median_s`, `hist_iqr_s` and `hist_entropy_nats`: one violin per (window, class) cell over one value per recording, the Holm-adjusted Kruskal–Wallis $p$ of the window below it, and at the foot Cliff's delta for every class pair that survives (more severe against less severe). The layout is that of `lag_high_kl_time_to_delivery_windows.pdf`.

**Axes.** Hours before delivery across, with delivery at the right. The violin rows share one $y$ label; each row's title names its readout and unit (seconds for the median and the range, nats for the entropy).

**How to read it.** These are three more Holm families per clock, and they are not corrected jointly with the other families. The centroid is not here, because `high_lag_centroid_kl_s` already tests the position of the same selection. The tested features use the `kl` source only. `lag_high_kl_histogram_features.csv` carries the attention-source features, which are drawn on the histogram page without tests.

### `lag_high_kl/lag_high_kl_second_stage_histogram_features.pdf`

**What it shows.** The panels of `lag_high_kl_time_to_delivery_histogram_features.pdf` on the signed second-stage clock, with the onset at zero. This clock has its own families, and the population is the onset subset.

### `lag_high_kl/lag_high_kl_time_to_delivery_histogram_drift.pdf`

**Purpose.** Measure how lag-histogram features change within each recording as delivery approaches.

**What it shows.** One row per tested feature (`hist_median_s`, `hist_iqr_s`, `hist_entropy_nats`).

- **Left:** each recording's own trajectory of the feature along the clock, thin in its class colour, with the class median per window drawn heavy.
- **Right:** the least-squares slope of the feature against forward labour time, one value per recording scored in at least `MIN_DRIFT_WINDOWS` $= 3$ windows, as one violin per class with zero marked.

The title carries the Holm-adjusted Kruskal–Wallis $p$ across classes. `lag_high_kl_histogram_drift_summary.csv`, `lag_high_kl_histogram_pairwise.csv` and the record carry each class's Holm-adjusted Wilcoxon signed-rank $p$ against zero and Cliff's delta for any surviving pair.

**Axes.** Left: hours before delivery across, with delivery at the right; the feature's unit up. Right: class across; the feature's unit per hour of labour up, with zero marked.

**How to read it.**

- A positive slope means that the feature rises as delivery approaches, on both clocks. The code negates the delivery clock's centres before the fit. `lag_high_kl_histogram_drift.csv` carries the raw window centres.
- Each clock has two families, each with Holm across the three features: the class-against-zero tests and the across-class tests. Pairwise tests run on survivors only.
- A recording scored in fewer than three windows is absent. The tick label of each class counts the fitted recordings.
- A slope is a straight line through a trajectory that need not be straight. The left panel shows the shape.

### `lag_high_kl/lag_high_kl_second_stage_histogram_drift.pdf`

**What it shows.** The panels of `lag_high_kl_time_to_delivery_histogram_drift.pdf`. The left panels use the signed second-stage clock, not inverted, with the onset at zero. The slopes are already in forward time, so the right panels read the same way. The population is the onset subset.

### `lag_high_kl/lag_high_kl_usefulness.pdf`

**Purpose.** Check whether anchors with large KL also show improved prediction.

**What it shows.** Four panels:

1. The forecast gain $D_{\mathrm{base}} - D_{\mathrm{full}}$ of an anchor against its KL decile: the median per recording with its interquartile ribbon, pooled in black and one line per class, with zero marked. The record's `usefulness` block carries the run-level paired test: the `high` band's mean gain minus the `rest` band's within each recording, its Wilcoxon $p$, and the share of recordings in which it is positive.
2. The mean gain by the lag at which the anchor's KL attribution peaks, for all anchors and for the `high`, `rest` and `gain` bands.
3. The pooled lag profiles as shares: the all-anchor KL attribution, the attention weighted by positive forecast gain, and the `gain` and `high` bands' attributions. The `usefulness` block carries the high–gain overlap against the $30\%$ that independence gives.
4. Per recording and per `occlusion_bands` band, the share of KL attribution inside the band against the forecast cost of occluding it, one colour per band, with zero marked. Without the occlusion pass in the directory, the panel carries a note instead.

**Axes.** Panel 1: KL decile across (0 is lowest), nats per anchor up. Panel 2: stored-coefficient lag in seconds across, nats per anchor up. Panel 3: lag across, share up. Panel 4: share of the recording's attribution across, occlusion delta in nats up.

**How to read it.**

- If the gain rises with the KL decile, high-KL anchors also forecast better. If it stays flat, the KL readout measures something that the forecast does not use. Read `source_null` beside it.
- The paired test is one test in its own family. Panels 2 to 4 are descriptive.
- Panel 2 conditions on the argmax, which the near edge of the window can pin. Read it beside `n_anchors` in `lag_high_kl_gain_by_argmax.csv`.
- The gain-weighted profile reweights the same attention, so it can only redistribute mass that the model already placed.
- Panel 4 compares the observational and the interventional readings for this checkpoint only.

---

## Spectral-skill figure

### `spectral_skill/spectral_skill_bands.pdf`

**Purpose.** Compare forecast performance across bands defined by the analysing filters of the target coefficients.

**What it shows.** Top ("Forecast gap by band"): the forecast gap per recording, one violin per band. Each label gives the band's frequency range with the period in parentheses, and its channel count. Bottom ("Error-space skill by band"): the skill of the full branch against the base branch, per band.

**Axes.** Top: nats per anchor, with zero marked. Bottom: $1 - \mathrm{MSE}_{\mathrm{full}}/\mathrm{MSE}_{\mathrm{base}}$, with zero marked. The two quantities have different units, so they have separate axes.

**How to read it.**

- This is band-resolved skill, not coherence. A stored scattering coefficient is a modulus, so the figure cannot separate a mistimed forecast from a mis-scaled one.
- The band is the band of the analysing filter, not a bin of the forecast's own spectrum.
- The channel counts in the labels show how many channels carry each band.
- `unknown` holds the channels without a recoverable centre frequency. They keep their own label.

---

## Recording-trace figures

The `recording_traces` analysis (`analyses/recording_traces.py`) follows up to `eval_config.caps.traces_per_class` recordings per class through all their segments. It writes these figures into `recording_traces/`.

### `recording_traces/recording_traces_summary.pdf`

**Purpose.** Follow selected recordings through their segments up to delivery, beside each class's median course.

**What it shows.** A coverage row comes first. Then there is one panel per per-segment summary column: the divergence $K_t$, the mean-decoded forecast gap, the source shift of the latent mean $\lVert \mu^q - \mu^p \rVert_2$, the active coordinate count, the lag centroid of the KL attribution and its entropy. All panels share one axis of hours before delivery, with delivery on the right.

- Each recording is a thin line in its class colour, with one marker per segment. The line is lifted at a break (a gap longer than one segment stride).
- The class median per half-hour window is bold, with the interquartile band over recordings wherever at least three recordings fall in the window. Each recording gives one value per window (the mean of its segments there).
- The coverage row counts the recordings per class in each window.

When `max_hours_before_delivery` is set, the axis stops there, and `segment_summary.csv` keeps the segments beyond it.

**How to read it.** The figure is for inspection, not for testing; `cross_subgroup` and the clock analyses test class differences. Read the coverage row before the band. Each value is a mean over the segment's scored anchors, and a segment that scored none is absent. The lag centroid is in stored-coefficient time.

### The per-recording traces: `recording_traces/<class>/<guid>_<subgroup>_trace.pdf`

**Purpose.** Inspect every available segment of one recording at anchor resolution, on a shared axis of hours before delivery.

**What it shows.** The page starts with the raw signals: raw FHR (bpm) and raw UP (mmHg), one row each on the fixed CTG scales, painted over the whole recording in `epoch` order at the raw rate. A signal that the loader did not serve is left out. Ten model rows follow, each at the anchor step, so a column is the same anchor on every row:

1. The divergence $K_t$.
2. Both `pred_gap` estimators, joined from the collection pass at those anchors and absent where the pass did not collect the segment.
3. The posterior mean $\mu^q$ over the latent coordinates, as a heatmap.
4. The source shift $\mu^q - \mu^p$ on a symmetric colour scale, as a heatmap.
5. The per-coordinate divergence.
6. The KL attribution over lags on a logarithmic colour scale, with the peak lag drawn over it.
7. The latent norms.
8. The mean log-variances of prior and posterior.
9. The lag centre of the attribution (centroid and median, in seconds).
10. The lag entropies of the attribution and of the attention.

Each heatmap's colour axis sits in its own column, so all rows span the same hours. On the $K_t$ and forecast-gap rows, a black step marks each segment's mean over its scored anchors, which is the value that the summary figure carries. Segments alternate a faint background on the line rows. A break is shaded darker on every row. A line row is lifted at every unscored anchor. Where the recording has a clinical clock, the page rules the onsets across it: labour onset dashed, second-stage onset dotted, each labelled on the first row. The title carries the GUID, the subgroup and the class. `recording_traces.csv` carries the segment count and the span. The `_full.npz` beside the figure holds the arrays behind every panel.

**How to read it.**

- A step at a segment join comes from the harness: each segment is a separate forward pass with a reset encoder state.
- A smooth path comes from the smooth encoders. There is no latent transition density.
- The lag rows are in stored-coefficient time, and the note under the figure applies to each of them (long form `GROUP_DELAY_CAVEAT`).
- `traces.shared_panel_scales` sets one colour-scale top per heatmap row and one $y$ range per line row over all traced recordings, and the raw rows use the fixed CTG scales. Two recordings' pages therefore compare by colour and height. A quiet recording draws pale.

### The interactive page: `recording_traces/<class>/<guid>_<subgroup>_trace.html`

**What it shows.** The same trace as a plotly page (plotly.js from its CDN), written by `traces_html.py`. It shows the raw FHR and UP on top, then every model row, on one shared zoomable time axis relative to delivery. Each series is painted onto a uniform grid in `epoch` order. Where a dataset stores overlapping segments, the later segment overwrites the earlier one, and a break is an empty stretch. The lag map and, on the lag-slot cell, the bounded mean update use a logarithmic colour scale (a signed log for the update), ticked in the panel's own units.

---

## Attribution figures

The attribution pass (`attribution_pass.py`, `attributions.py`, `class_contrast.py`, `attribution_strata.py`, `gradcam.py`) writes these figures into `attribution/`. [ATTRIBUTION.md](ATTRIBUTION.md) explains the methods, baselines and checks. Unless an entry states otherwise:

- The main pass attributes up to `eval_config.caps.attribution_segments` segments, `attribution_segments_per_recording` per recording, and up to `ANCHORS_PER_SEGMENT` $= 4$ anchors per segment from the top $30\%$ of that segment's $K_t$ (`HIGH_KL_QUANTILE` $= 0.7$).
- The class-comparison pass (`attribution_classes.pdf`, `attribution_class_maps.pdf`, `gradcam_classes.pdf`) uses `eval_config.caps.attribution_cohort_segments` segments, one per recording.
- The strata pass (`late_behaviour.pdf` and the `strata_*` figures) uses `eval_config.caps.attribution_strata_segments` late segments, one per recording, from segments that start at most `LATE_WINDOW_HOURS` $= 2$ h before delivery.
- The model was trained on the healthy-only pretraining split, so a class contrast describes the fitted model's response to inputs outside its training distribution. It is not a physiological effect.

### `attribution/attribution_maps.pdf`

**Purpose.** Show the coefficients that the encoders read at one example anchor per clinical class, and which coefficients the divergence is attributed to. The example is the highest-KL clean anchor of each class. Without a per-anchor table, it is the middle chosen anchor of the first selected recording of each class.

**What it shows.** One column per class example and seven rows. All rows except the last share the segment's stored-time axis $[0, 4T]$ s, the same in every column:

1. Raw FHR (bpm) on the fixed CTG scale.
2. Raw UP (mmHg) on the fixed CTG scale.
3. The declared target stream as input: the scattering block above the phase-harmonic block, divided by a rule.
4. The declared source stream as input.
5. The attribution of $K_t$ to the target stream under the **all-zero** baseline. Only along this baseline do the target inputs move, so only this baseline gives a target map.
6. The attribution of $K_t$ to the source stream under the **source-null** baseline, the primary comparison.
7. The source attribution re-indexed by offset from the anchor, as a share per lag on a symmetric-log axis, against the model's own lag readout at the same anchor normalised the same way.

The input rows show standardised coefficients on a sequential scale with robust (1st–99th percentile) limits, taken over the read cells of all examples, one scale per stream. Cells that the model never read are blanked in light grey: each channel's cold steps, which the input gate multiplies by zero. The attribution rows use a symmetric-log colour scale (linear within `LOG_DECADES` decades of the largest magnitude, logarithmic beyond), shared by all columns. The cold cells and every step after the anchor are blanked, because both are exactly zero by construction. `attribution_rows.csv` carries the correlation and the Jensen–Shannon distance of row 7 (`lag_corr`, `lag_js`).

Panels are titled in the first column only, and the lag legend is in its first panel. Each map carries the anchor as a vertical rule and the scored forecast block as a shaded window. The column heading carries the GUID, the subgroup and the class. `attribution_examples.csv` carries the anchor and its epoch. One colour bar per row sits at the right of the last column.

**Axes.** Stored time in seconds across; raw rows in bpm and mmHg; declared channel down on the map rows; standardised coefficient (inputs) or readout units on a symmetric-log scale (attribution) in colour. The lag panel is in stored-coefficient seconds.

**How to read it.**

- On a causal model every cell after the anchor is exactly zero. `attribution_checks.pdf` shows any non-zero value there as a defect.
- The map shows one anchor of one segment of one recording. The per-class example pages give the same anchor under every readout.
- The colour scales are shared (`attributions.example_norms`): one linear scale per input stream, and one symmetric-log scale per readout, baseline and stream, over all class examples. A colour means the same value in every column and on every per-class example page. Rows have their own scales, so use the totals in `attribution_rows.csv` to compare rows.
- A cold channel's cells are zero because the gate multiplied them by zero.
- The note under the figure is the short form of `ATTRIBUTION_CAVEAT`: an attribution is a sensitivity, not a causal effect or a latency. `lag_qualification` in `summary.json` qualifies the model's own lag readout.

### The per-class example pages: `attribution/maps/<class>_<guid>_<subgroup>_anchor<step>_attribution_maps.pdf`

**Purpose.** Put the input coefficients and every readout's attribution of them on one page, for one anchor.

**What it shows.** One page per example, up to `EXAMPLES_PER_CLASS` $= 3$ per class: the highest-KL clean anchors of the class, each from a different recording, ranked by $K_t$. The page is a stack of full-width rows on one shared stored-time axis, with a colour-axis column on the right.

- The first two rows are raw FHR and raw UP on the fixed CTG scales. The next two are the input streams, with cold cells blanked.
- Each readout variant takes two rows: its target attribution under the all-zero baseline above its source attribution under the source-null baseline, each on a symmetric-log scale. `value_input` and `value_baseline` in `attribution_rows.csv` give the readout at the input and at the exact null.
- The variants, in order: the divergence $K_t$, the forecast gap (base minus full), the full-branch block score, the full-branch squared error, the squared-error gap, the full-branch score at the first and the last horizon step, and the lag readout on each configured `occlusion_bands` band.
- Three closing rows are off the time axis. The first shows the latent at the anchor: prior mean, source shift $\mu^q - \mu^p$ and per-coordinate divergence $K_{t,d}$, with the top coordinate marked. The second shows the activation split: the layer integrated gradients of every variant on the cell's own layer, under the source-null baseline. The third overlays every variant's lag-aligned source attribution on one lag axis against the model's lag readout.

The page title carries the GUID, the subgroup and the class. `attribution_examples.csv` carries the anchor and its epoch. The filename carries the class, the GUID and the subgroup. Every page shares one set of colour scales with the other example pages and with `attribution_maps.pdf`.

**Axes.** Stored time in seconds across on every map row; raw rows in bpm and mmHg; declared channel down on the map rows. Latent coordinate, head or lag, and stored-coefficient seconds on the three closing rows.

**How to read it.** Read down a column to see whether the same coefficients drive different readouts. Read across a row to see where in stored time and in which channels one readout is sensitive. The sign depends on the readout: a positive cell raises the readout along the path, which is a larger latent change for $K_t$, a larger advantage of full over base for the gap, and a worse forecast for the block score. The all-zero and source-null rows use different baselines and paths and do not add together. The two horizon-step rows compare the near and far ends of the block. The squared-error rows show which inputs moved the mean forecast. `attribution_maps.npz` carries every map, keyed by example, readout, baseline and band (`h<step>` for a horizon step), with the inputs, the latent (`example_latent_*`) and the activation split (`map_layer`).

### `attribution/attribution_blocks.pdf`

**Purpose.** Show, per readout, how the attribution splits over the four input blocks: target scattering, target phase harmonics, source scattering and source phase harmonics.

**What it shows.** Left: for each readout variant (divergence, forecast gap, full-branch block score, full-branch squared error, and the score at the first and last horizon step), the mean over recordings of the unsigned attribution share of each block under the all-zero baseline, as a stacked bar that sums to one. Right: the signed block sums of the same rows on a symmetric-log axis. `attribution_blocks.csv` carries both under both baselines.

**Axes.** Left: share of the unsigned attribution across. Right: signed attribution in readout units, symmetric log. Readout variants down.

**How to read it.** A share is a split of sensitivity along one path from one baseline, not a variance decomposition. A block that carries the score but not the squared error moved the forecast variance and not its mean. Under the all-zero baseline, the source share includes the source contribution when it enters together with the target. That is not the source-null comparison of the lag figures.

### `attribution/attribution_horizon.pdf`

**Purpose.** Show how the near and the far ends of the forecast block read the inputs.

**What it shows.** The per-step score rows (`nll_horizon`, tagged `h<step>` in the `band` column).

- Left: per named horizon step, the mean over recordings of the unsigned attribution total of the target stream (all-zero baseline) and of the source stream (source-null baseline), with the step's score at the input as a line.
- Middle: the source attribution magnitude by lag per step, source-null baseline.
- Right: the target attribution by offset from the anchor per step, all-zero baseline.

All vertical axes are symmetric-log.

**Axes.** Horizon step across (left); offset from the anchor in stored-coefficient seconds across (middle, right); nats up.

**How to read it.** The steps are the first and last of the block, so the figure shows the two ends and not the shape between them. A lag profile that differs between the steps means that the two ends read different source history. The axis is stored-coefficient time.

### `attribution/attribution_lag_profile.pdf`

**Purpose.** Compare the source-input attribution over lag offsets with the model's own lag readout at the attributed anchors.

**What it shows.** One row per main readout (divergence, forecast gap, full-branch block score, full-branch squared error), under the source-null baseline.

- Left: the mean over recordings of the normalised $|q_\ell|$ beside the mean normalised model profile, pooled. `attribution_rows.csv` carries the recording-mean Pearson correlation and Jensen–Shannon distance (`lag_corr`, `lag_js`).
- Right: the same by class, solid for the attribution and dashed for the model readout, in the severity colours. `attribution_recordings.csv` lists the recordings per class.
- A last row puts every readout on one lag axis. Left: the normalised source attribution of the divergence, of the forecast gap and of the lag readout on each configured band, pooled over recordings, with the model's lag readout dashed in black. Right: the lag-band readouts alone, each with its own band shaded in its colour.

**Axes.** Lag in seconds, stored-coefficient time; share per lag, symmetric log.

**How to read it.** The two profiles are different objects on one axis. The model readout is $K_t$ times the attention, an allocation. The attribution is the sensitivity of $K_t$ to the source coefficient at that offset; it can be negative and is drawn by magnitude. On the comparison row, coinciding profiles mean that the same stored source time moves the latent, the forecast and the lag readout. A band readout whose attribution sits outside its own band is computed from source history beyond that band. The axis is not corrected for the composed group delay.

### `attribution/attribution_bands.pdf`

**Purpose.** Summarise the input attribution by frequency band and by source lag band, beside the related predictive and intervention analyses.

**What it shows.** Top, one column per readout: the mean over recordings of the attribution summed over each frequency band of the declared channel map, one bar group per stream. The target uses the all-zero path and the source uses the source-null path. Where the `spectral_skill` pass ran, its per-band gap appears on a twin axis for the target bands. The ticks name each band by its frequency range with the period in parentheses, and the `band` column keeps the key.

Bottom: per `occlusion_bands` band of the source, three bars: the integrated-gradient sum over the band, this pass's own feature-ablation delta of the readout, and the `occlusion` pass's delta where it ran. The occlusion delta is sign-flipped on the gap readout, because that pass reports the forecast cost of removing a band.

**Axes.** Bands across; readout units up, symmetric log (nats per anchor for the divergence, the forecast gap and the full-branch score; squared standardised units for the full-branch squared error; nats per anchor for the occlusion series).

**How to read it.** A frequency-band sum covers the declared channels of the stream, including the dropped ones, whose attribution is exactly zero. `attribution_bands.csv` carries the channel counts. The ablation delta and the occlusion delta are the same intervention at different anchors and on different readouts. Nothing here is tested.

### `attribution/attribution_layer.pdf`

**Purpose.** Show the attention-head contributions to the attributed output, and the inputs linked to the latent coordinate with the largest divergence.

**What it shows.** Left: the layer integrated gradients on the per-head fusion modules of the head-structured posterior, summed over each head's feature at the anchor. There is one bar per head per readout, as a mean over recordings, under the source-null baseline. Right: for the anchor's largest per-coordinate divergence $K_{t,d}$, the magnitude of its attribution by offset from the anchor, per stream.

**Axes.** Head across (left); lag in stored-coefficient seconds across (right); readout units up.

**How to read it.** The per-head split sums to the readout difference along the source-null path only. Under the all-zero baseline, the prior's direct route into the posterior bypasses the fusion, so the split is not taken there. A posterior without head structure has no split, and the panel is empty. The top coordinate is chosen per anchor, so the right panel pools anchors whose top coordinate differs.

### `attribution/attribution_null.pdf`

**Purpose.** Separate the exact zero-source response, the integrated source-content attribution, and the entry jump at the zero baseline.

**What it shows.** Left, per class: the mean over recordings of $K_t$ at the input, $K_t$ at the exact null (the availability-clock part), the integrated source-content attribution and the entry jump, under the source-null baseline. Right: under the all-zero baseline, the split of the attribution between the target and source streams, and the entry jump.

**Axes.** Class across, with the recording count; nats per anchor up.

**How to read it.** The four left bars are not a partition. The input equals the null plus the entry jump plus the attributed content only up to the completeness residual, which `attribution_summary.csv` reports. The entry jump is a property of the encoders' normalisation at the exactly-zero point, not of any input step.

### `attribution/attribution_channels.pdf`

**Purpose.** Show which declared input channels each readout responds to, summed over stored time.

**What it shows.** One row per main readout. The target stream (all-zero baseline) is on the left, and the source stream (source-null baseline) is on the right. The bars are the mean over recordings of the signed channel profile, the sum of a channel's attribution over stored steps. The line is the mean of the unsigned profile, the sum of absolute values. Where the run wrote `band_channel_map.csv`, the bars are coloured by frequency band, and the legend names each band by its frequency range; otherwise the bars are grey. On the target stream, the scattering block sits left of a dashed rule and the phase-harmonic block sits right of it.

**Axes.** Declared channel across; attribution in readout units up, symmetric log.

**How to read it.** A tall line over a short bar means that the channel pulled the readout both ways at different steps. A bar and line that coincide mean that the channel pulled one way. Dropped channels are exactly zero. The two streams use different baselines and scales, so compare them by shape. `attribution_bands.pdf` sums these profiles over bands.

### `attribution/attribution_lag_channel.pdf`

**Purpose.** Show which channel at which offset from the anchor each readout responds to. The lag profile and the channel profile are the marginals of this map.

**What it shows.** One row per main readout. Left: the target stream by offset from the anchor under the all-zero baseline. Right: the source stream by lag under the source-null baseline. Each cell is the mean over attributed anchors of the unsigned attribution at that offset and channel. A rule marks the scattering/phase-harmonic boundary on the target map. `attribution_lag_channel.npz` carries the signed means beside the unsigned ones.

**Axes.** Offset from the anchor in stored-coefficient seconds across; declared channel down; mean unsigned attribution in colour on a logarithmic scale, one scale per panel.

**How to read it.** A bright row is a channel that the model reads across its whole window. A bright column is an offset that the model reads on many channels. The mean is over attributed anchors, not per recording, so a recording with more scored anchors weighs slightly more. The axis is stored-coefficient time.

### `attribution/attribution_time_profile.pdf`

**Purpose.** Show the sign of the attribution by offset from the anchor.

**What it shows.** One row per main readout. The source stream (source-null baseline) is on the left and the target stream (all-zero baseline) is on the right, by offset from the anchor. The shaded area above the axis is the mean over recordings of the positive part of the per-offset attribution. The area below is the mean of the negative part. The black line is the net mean, and the dashed line is the unsigned mean.

**Axes.** Offset in stored-coefficient seconds across; attribution in readout units up, symmetric log.

**How to read it.** A positive value raises the readout along the path. An offset with two large parts and a net near zero is an offset where channels pull the readout both ways; the unsigned line measures that. A net that changes sign along the axis means that recent and older history push the readout in opposite directions.

### `attribution/attribution_checks.pdf`

**Purpose.** Show the numerical checks behind every attribution row on one page.

**What it shows.** Left: the relative completeness residual of every row, one histogram per readout and baseline on a logarithmic axis, with the tolerance as a dotted rule. `n_rows_over_tolerance` in `summary.json` gives the count above it. Each strip names its baseline on its tick label, and a black tick marks the median. Middle: the entry jump $f(x_0) - f(b)$ against the readout at the input, one point per row, per baseline. Right: the largest attribution to a stored step after the anchor and to a gated-off source step, per readout, on a symmetric-log axis.

**Axes.** Relative residual (log) and rows (left); readout units on both middle axes; readout units, symmetric log (right).

**How to read it.** A row above the tolerance has integrated gradients that do not account for its readout change, so its map is not a decomposition. Raise `IG_STEPS` (default $64$) before you read such rows. A large entry jump belongs to the baseline and the normalisation. The right panel is exactly zero on a causal, gated model; any other value is a defect.

### `attribution/attribution_time_to_delivery.pdf`

**Purpose.** Place the attributed anchors on the clinical clock.

**What it shows.** One row per main readout under the source-null baseline. Left: the source attribution total of every attributed anchor against its hours before delivery. Right: the $|q|$-weighted lag centroid of the source attribution. Each point is one anchor in its class colour. The class median per one-hour window is drawn where at least three recordings fall in it. Delivery is on the right.

**Axes.** Hours before delivery across; readout units (left) and stored-coefficient seconds (right) up.

**How to read it.** This is a sparse view: a few anchors per attributed segment. It shows whether the attributed recordings sit at comparable stages of labour and whether the attribution or its lag centre moves with that stage. The traces read the clock densely for a few recordings per class, and the clock analyses read it on every recording with a correction.

### `attribution/attribution_classes.pdf`

**Purpose.** Compare the clinical classes on their source attribution over many recordings.

**What it shows.** One row per cohort readout (`kld`, `pred_gap`, `COHORT_READOUTS`), all under the source-null baseline, from the cohort pass:

1. The class-mean signed source attribution by lag, with a $95\%$ bootstrap band over recordings. The legend gives each class's recording count and the median $K_t$ of its attributed anchors.
2. The class-mean unsigned lag share: each anchor's $|a_\ell|$ divided by its sum over lags.
3. The difference of those shares for every class pair, more severe minus less severe (HIE − acidosis, HIE − healthy, acidosis − healthy), with its own bootstrap band and the zero line. The pairs are black, blue and purple, which are not class colours. A rug under the curves has one row per pair and marks each lag where that pair's band excludes zero.
4. One violin per class of the per-recording lag centroid, titled with the Holm-adjusted Kruskal–Wallis $p$.

**Axes.** Stored-coefficient lag across; readout units, share per lag, or seconds up.

**How to read it.** Read panels 3 and 4 first. The cohort anchors are each segment's top $30\%$ of $K_t$, so the classes are attributed at different coupling levels. The IG of `kld` sums to approximately $K_t - K_{\mathrm{null}}$, so a class difference in signed totals is mostly a class difference in $K_t$. The legend medians and the `level:kld` row of `attribution_class_stats.csv` show that level. A share does not scale with $K_t$. The rug is pointwise at $95\%$ and not adjusted over lags, so about $5\%$ of lags carry a mark by chance; read runs of adjacent marks. Each recording's anchors are averaged first, so a long recording counts once. `strata_matched.pdf` compares the classes at one matched $K_t$ level. `attribution_class_stats.csv` and `attribution_class_pairwise.csv` carry the tests.

### `attribution/attribution_class_maps.pdf`

**Purpose.** Show on which source channel and lag each class's attribution sits, and where the classes differ.

**What it shows.** One row per cohort readout. The class columns show the mean unsigned source attribution by lag and declared source channel, on one shared log scale per row. The pair columns show the difference of those class means, left minus right, on one linear symmetric scale per row, clipped at the 99th percentile of the differences.

**How to read it.** A red cell in `hie − healthy` is a channel and lag where HIE recordings carry more attribution magnitude than healthy ones. These maps are means without a per-cell test, so treat an isolated cell as noise and a coherent band of channels or lags as a pattern to check in the tables. The channel axis is the declared input axis, and dropped channels stay at zero.

### `attribution/gradcam_classes.pdf`

**Purpose.** Compare where in time each class's readouts look, through Grad-CAM, at the cost of one forward and one backward pass per readout.

**What it shows.** One row per main readout (`kld`, `pred_gap`, `nll_full`, `mse_full`; `GRADCAM_READOUTS`). Each of three views has a class-mean panel and a class-difference panel, with $95\%$ bootstrap bands over recordings:

1. The target stream at the input of the target encoder's last attention block, on a minutes axis over the history after the anchor floor $F$. An anchor at $t_a$ reaches back $t_a - F$ steps, so the far end of the axis averages only the later anchors. The view is summed into bins of `TARGET_BIN_STEPS` $= 8$ steps ($32$ s), and a partial last bin is dropped.
2. The source K/V stream on the lag axis.
3. The lag-attention weights at the anchor, scored as gradient-weighted attention, on the same lag axis.

The difference panels carry the same per-pair rug as `attribution_classes.pdf`.

**Axes.** Offset before the anchor (minutes, bin centres) or stored-coefficient lag across; Grad-CAM share per $32$ s bin or per lag up. Each anchor's map sums to one.

**How to read it.** Grad-CAM keeps only the evidence that raises the readout: more divergence for `kld`, more advantage of the full branch for `pred_gap`, and a worse score for `nll_full` and `mse_full`. Each map is a share, so the panels compare where the model looked, not how strongly. `attribution_class_stats.csv` tests the per-recording totals and centroids. The target view uses the input of the last attention block, because the per-step prior and posterior heads read the encoder output only at the anchor. Grad-CAM has no baseline and no completeness property, so read it beside the integrated gradients.

### `attribution/late_behaviour.pdf`

**Purpose.** Show what the network does over every scored anchor of the last `LATE_WINDOW_HOURS` $= 2$ h before delivery, by class, without gradients.

**What it shows.** Six panels, each a class mean over recordings with a $95\%$ bootstrap band:

- **(a)** $K_t$ against minutes before delivery, in `DELIVERY_BIN_MINUTES` $= 10$ min bins, with delivery on the right.
- **(b)** The forecast gain $D_{\mathrm{base}} - D_{\mathrm{full}}$ on the same axis (`mean_pred_gap`, else `mc_pred_gap`, else `pred_gap`).
- **(c)** $K_t$ against seconds since the last detected contraction onset, in `ONSET_BIN_SECONDS` $= 15$ s bins up to `ONSET_MAX_SECONDS` $= 300$ s. The grey span is the contraction window $w$ (`event_lag_window_s`).
- **(d)** The forecast gain on the same axis.
- **(e)** The head-averaged attention by lag at high-$K_t$ anchors minus low-$K_t$ anchors, as a paired difference per recording.
- **(f)** The same for the $K_t$ lag map, with each anchor's map first divided by its sum.

In (e) and (f), the rug marks lags where a class band excludes zero.

**How to read it.** Panels (c) and (d) show whether the source moves the belief, and whether that helps the forecast, during a contraction compared with between contractions. Panels (e) and (f) show where on the lag axis the attention and the divergence move when $K_t$ is high. The tiers are per segment: $q \ge 0.7$ against $q \le 0.3$ of that segment's $K_t$. Anchors without an onset before them in the segment are absent from (c) and (d). Minutes before delivery is the segment start plus the anchor step, so it inherits the segment's trim offset.

### `attribution/strata_kl_ig.pdf` and `attribution/strata_ua_ig.pdf`

**Purpose.** Show how the integrated-gradient attribution changes between two kinds of anchor in the same late segments, per class.

**What it shows.** The `kl` file contrasts high-$K_t$ anchors (A, black) with low-$K_t$ anchors (B, blue). The `ua` file contrasts anchors at most $w$ seconds after a contraction onset (A) with anchors between contractions (B). There is one row per readout and stream, on the lag axis:

- the source attribution of `kld` under the source-null baseline;
- the target (FHR history) attribution of `kld` under the all-zero baseline;
- the same two for `pred_gap`.

The target rows use the all-zero baseline, because the source-null baseline holds the target streams at the input, so its target attribution is exactly zero. One column per class shows A and B with $95\%$ bootstrap bands over recordings. Its title gives the median $K_t$ of the class's A and B anchors. The last column shows, per class, the mean over recordings of the per-recording difference A $-$ B, with its band and a rug where the band excludes zero.

**How to read it.** Read the last column first. The difference is paired, so recording-level offsets cancel. A positive band at a lag means that the model draws more attribution from that lag at A anchors. The `kl` strata are defined by $K_t$, so a larger source attribution of `kld` at high-$K_t$ anchors is partly built in; the scale-free metrics in `strata_paired.pdf` test a change in where the model reads. The source and target rows use different baselines, so compare each row across strata and classes, not the two rows with each other.

### `attribution/strata_kl_gradcam.pdf` and `attribution/strata_ua_gradcam.pdf`

**Purpose.** Show the same two contrasts through Grad-CAM, in all three views.

**What it shows.** One row per readout (`kld`, `pred_gap`) and view (target encoder, source K/V, lag attention). The layout is that of the IG strata figures. The target view is binned as in `gradcam_classes.pdf`, on a minutes axis; the other two views are on the lag axis.

**How to read it.** Each map is a share, so a paired difference compares where the readout looked, not how strongly. Read each figure beside the IG figure for the same contrast. Grad-CAM keeps only the evidence that raises the readout and has no baseline.

### `attribution/strata_paired.pdf`

**Purpose.** Test the stratum contrasts per recording on scale-free and scalar metrics, per class.

**What it shows.** Two rows: high against low $K_t$, then in against between contractions. Six metrics (`PAIRED_METRICS`):

1. the anchor's $K_t$;
2. the forecast gain;
3. the source share of the unsigned IG of $K_t$ under the all-zero baseline, $\sum|a_{\mathrm{src}}| / (\sum|a_{\mathrm{src}}| + \sum|a_{\mathrm{tgt}}|)$;
4. the IG lag centroid of $K_t$;
5. the Grad-CAM attention centroid of $K_t$;
6. the IG lag centroid of the gap.

Metrics 3 to 6 are scale-free, so the KL tiers cannot produce them by construction. Per class, each thin line joins one recording's A value (left) to its B value (right). The bold markers are the class means with $95\%$ bootstrap intervals. The tick label under each class gives the Wilcoxon signed-rank $p$, starred when it passes Holm. The panel title gives the Kruskal–Wallis $p$ of the paired differences across classes.

**How to read it.** A starred class means that the metric moves between the two strata within that class. A small Kruskal–Wallis $p$ means that the size of the move differs between classes. The first column is a manipulation check for the `kl` row: it separates by construction. The Holm families are one contrast, one class, and one method and readout. `strata_stats.csv` carries every test with its family.

### `attribution/strata_matched.pdf`

**Purpose.** Compare the classes at one coupling level, so that a class difference in attribution is not a class difference in $K_t$.

**What it shows.** Only the `kl_matched` anchors: the anchors whose $K_t$ falls inside one band of the pooled late-window $K_t$ distribution, the same nats for every class. The pooled distribution is cut into `MATCHED_BANDS` $= 5$ equal-count bands. The chosen band is the one whose worst-covered class holds the most recordings; a tie goes to the narrowest band. The title gives the band.

- Five profile rows, each with a class-mean panel and a class-difference panel with the pointwise rug: the unsigned IG lag share of `kld` and of `pred_gap` under the source-null baseline, and the Grad-CAM share of `kld` in the three views.
- Two rows of per-recording violins, each titled with the Holm-adjusted Kruskal–Wallis $p$: $K_t$ at the anchor, the IG lag centroid of $K_t$, the source share of the IG of $K_t$ under the all-zero baseline, and the Grad-CAM attention centroid.

**How to read it.** Read the $K_t$ violin first. It checks the match, and it should not differ between classes. If it does, the band is too wide for the cohort, and the other panels still carry a level difference. A large $p$ there does not prove equal levels. Also read the class difference of `anchor:kld_per_t` and its bootstrap interval in `strata_matched_pairwise.csv`, against the band width `strata.rule.matched.width_nats`. The $K_t$ check is its own Holm family. When the match holds, a class difference in the other panels is a difference in how the network reads the UP history at equal coupling. The rug is pointwise at $95\%$, so read runs of adjacent marks. The legend gives each class's recording count. `strata_matched_stats.csv` and `strata_matched_pairwise.csv` carry every test, with the cohort's rule (Kruskal–Wallis, Holm within one method and readout, then Mann–Whitney $U$ and Cliff's $\delta$ per pair).

### `attribution/strata_onset_locked.pdf`

**Purpose.** Show whether the attribution locks onto the contraction onset, wherever that onset falls in the lag window.

**What it shows.** Every attributed late anchor whose most recent contraction onset lies inside the lag window, from any stratum and counted once. The figure re-indexes on the onset each anchor's unsigned IG lag share of `kld` and of `pred_gap` (source-null baseline), and its Grad-CAM attention and source K/V shares of `kld`. With $a$ the anchor's seconds since the onset, the source step at lag $\ell$ sits

$$x = a - 4\ell$$

seconds after the onset. Both $a$ and $\ell$ are on the stored timeline. The two Grad-CAM views are on the lag attention's own axis, which reads the source $\delta$ steps further back, so they are aligned at $x = a - 4(\ell + \delta)$. In an unaligned configuration, $\delta = 0$. One column per class shows the class mean over recordings with a $95\%$ bootstrap band. Grey dashes show the onset-shuffle null: the same mean after the onset ages are shuffled among the class's anchors.

**Axes.** Seconds after the most recent onset across, with $x = 0$ at the onset and $x < 0$ before it; share up.

**How to read it.** A peak at or just after $x = 0$ that the grey null does not show means that the attribution follows the onset from anchor to anchor. A curve that follows the null means that the profile shape comes from the lag geometry. Only the most recent onset is used, so an earlier contraction inside the window adds a smaller, blurred peak at negative $x$. Compare the classes by the height of the peak above the null. The axis is coefficient time, not a physiological latency.

### The per-recording attribution traces: `attribution/traces/<class>/<guid>_<subgroup>_attribution_trace.pdf`

**Purpose.** Follow the attribution through one recording per class. The selection prefers the recording whose stored segments cover the evaluation window most completely.

**What it shows.** The traces' shared figure, on one axis of hours before delivery. The raw FHR and raw UP rows come first, as on the recording traces. Seven attribution rows follow:

1. The magnitude of the source attribution of $K_t$ by offset from the anchor, as a heatmap.
2. The same for the forecast gap.
3. The model's own lag readout at the same anchors.
4. The correlation of each readout's lag-aligned attribution with the model's readout.
5. The two readouts' source attribution totals.
6. $K_t$ at the input and at the exact null.
7. The forecast gap at the input and at the exact null.

Each heatmap cell spans the gap to its neighbouring anchors. The filename carries the GUID and the subgroup.

**How to read it.** The reading notes of the recording traces apply: a step at a segment join comes from the harness, the scales are shared across the traced recordings (`traces.shared_panel_scales`), and the lag axis is stored-coefficient time. The pass uses `ANCHORS_PER_SEGMENT` anchors per segment, so a feature narrower than their spacing is not visible. Rows 1 and 2 show where the source moved the belief and where it helped the forecast; an hour where they differ is an hour where the latent change did not improve the forecast.

---

## Sample pages

The `samples` analysis (`analyses/samples.py`) draws the per-segment diagnostic pages into `samples/<selection>/`. It draws each page through the task's own page seams, with the layout of the training callback (`teb_vae/lag_attn_cfs/sample_page.py`, built on `teb_vae/lag_attn_rws/sample_page.py`). Rendering needs a checkpoint; a model-free re-run records a skip.

### The per-recording pages: `samples/<selection>/sample<index>_<guid>_<subgroup>_epoch<epoch>.pdf`

**Purpose.** Inspect one selected segment in detail: inputs, forecasts, latent quantities and lag readouts.

**What it shows.** A fifteen-row page:

- The raw context: raw FHR in bpm and raw UP in mmHg on one time axis, with one $y$ axis each.
- The forecast lanes: a few kept channels as offset lanes, with the truth against the base and full means, their $\pm2\sigma$ bands, and each channel's scored horizon $H_c$ marked on its lane.
- Six causal extra rows (`CAUSAL_EXTRA_ROWS`): the truth, both branch means, the signed skill difference, the posterior $\sigma$, and the per-window score.
- Two gated-input rows: the target stream (`fhr_st` | `fhr_ph`) and the source stream.
- Five latent and lag rows: the latent state over its source shift, the per-dimension KL, $K_t$, the lag attention with its argmax, and the KL by lag.

The page title is one line: the dataset index, the GUID, the subgroup, the class and the epoch. `sample_pages.csv` carries the hours before delivery.

**Row meanings.** The panel titles name only the quantity:

- *base* is the target-only latent $z^p$, and *full* is the source-conditioned latent $z^q$.
- The forecast lanes are the worst, middle and best kept channels by $2\sigma$ calibration, named by declared channel and coverage on the $y$ ticks. They are stitched from consecutive windows, and the first anchor wins.
- The dotted grid is the sparser anchor set that a training step tiles.
- The three field rows share one fixed normalised colour scale. Grey cells are steps that no drawn window covers.
- *Source skill* is $|Y^{+}-\mu^p| - |Y^{+}-\mu^q|$ on the scored cells, red where source conditioning helped.
- The per-window score is the objective's own likelihood, shaded green where the source helps.

`sample_pages.csv` carries the loss readouts (`nll_base_block`, `nll_full_block`, `pred_gap`).

**Scales.** Colour scales are shared across the run, and $y$ axes are fitted per page.

- `fitted_limits` in `teb_vae/lag_attn_rws/sample_page.py` fits each line row to its segment: the range of the finite values, with `FIT_PAD_FRACTION` $= 5\,\%$ head room on each side. $K_t$ runs from $0$ to its padded maximum over the trained anchors.
- The raw FHR and UP rows, and the raw overlay on an input row, are fitted inside the CTG ranges `RAW_TRACE_LIMITS`: FHR $50$–$210$ bpm and UP $0$–$100$ mmHg. Values outside them, such as the $0$ bpm gap sentinel, are left out of the fit. The range is never narrower than `MIN_TRACE_SPAN`: $30$ bpm and $20$ mmHg.
- The forecast lanes and the per-window score autoscale to the segment.
- The latent, shift, KL, lag-map, attention, error, skill and $\sigma$ heatmaps use the colour limits that `shared_row_limits` computes once per run from the collection's tables. Each limit is a high percentile of its quantity over the whole split. The analysis records them under `plan.row_limits`. One colour is therefore one value on every page of the run.
- The lag attention uses a logarithmic colour scale on both the full and the reduced page (see the next section).
- `render_figure(..., crop=False)` writes every page uncropped at the builder's fixed size, so two pages of a run overlay row for row.

**Clocks.** Every model row is at the anchor step, so a column is the same anchor on all model rows. Only the raw row is in physical time. On an aligned run, an input channel at step $t$ carries content centred $\kappa\tau_{\mathrm{ref}}$ earlier, where $\tau_{\mathrm{ref}}$ is the stream's alignment reference (`source_reference_delay_s` for the source). The input rows then sit that far to the right of the raw row, as their $x$ labels state. An aligned run's input rows overlay their own raw signal delayed by $\kappa\tau_{\mathrm{ref}}$, as a thin black trace on a twin axis: the raw FHR on the target row and the raw UP on the source row. The constant comes from the resolved budget, which the runner attaches to the task after preflight. An unaligned run, such as the `configs/default.yaml` configuration, has no single constant and draws no overlay. Its input rows' $x$ label says that each channel's content sits at its own delay.

**Directories.**

- `stratified/` holds a seeded, shard-stratified draw over the whole split, capped by `eval_config.caps.pages`. Its quota is proportional to shard size, so it shows what the split contains.
- `by_class/` holds a class-balanced draw: `eval_config.caps.pages_per_class` segments per clinical class. It supports comparisons across classes and does not show prevalence.
- One directory per extreme metric and tail holds the segments at the extremes of that metric (`EXTREME_METRICS`, `EXTREME_PAGES_PER_TAIL` $= 10$): `mean_pred_gap_low/` and `mean_pred_gap_high/`, and the same pair for `nll_full_block` and `source_conditioned_kl_raw`. The gap pages use the mean-decoded estimator, which scores the mean forecast lanes drawn on the page.

**Filename.** The filename is `sample<index>_<guid>_<subgroup>_epoch<epoch>` with the run's format extension. `<index>` is the four-digit dataset index, and `<epoch>` is the segment's `epoch` in whole seconds. The subgroup is sanitised by the GUID's rule (letters, digits, `_` and `-` only) and is `na` for a row without one. The epoch is `na` for a segment without one. `sample_pages.csv` carries the same identity per file: selection, variant, both file names of the segment, `dataset_index`, `guid`, `subgroup`, `clinical_class`, `epoch`, `hours_before_delivery` and the page's headline scalars.

**How to read it.** A page shows one segment of one recording, as an illustration. The extreme pages are selected on the quantity that they display, so that panel always looks unusual. `<index>` is the position in the evaluation dataset, not in `per_sample.csv`, because the collection pass runs under a seeded shuffle. A `guid`/`epoch` round trip reconciles the two before rendering. Every selected segment is drawn twice: as the full page and as the reduced page beside it.

### The reduced per-recording pages: `samples/<selection>/sample<index>_<guid>_<subgroup>_epoch<epoch>_compact.pdf`

**Purpose.** Inspect the same segment and forward pass in a shorter layout, without the per-channel detail.

**What it shows.** Eight of the fifteen rows (`COMPACT_PAGE_ROWS` in `teb_vae/lag_attn_cfs/sample_page.py`), in the order of the full page and from the same forward pass:

1. The raw FHR and UP.
2. The forecast lanes: truth against both branch means, with their bands and the scored-horizon mask.
3. The target input stream (`fhr_st` | `fhr_ph`), with block dividers and the warm-up staircase.
4. The source input stream.
5. The latent state over its source-derived shift.
6. $K_t$.
7. The lag attention.
8. The KL by lag.

The page drops the five per-channel field rows (the truth, both branch means, the signed skill difference and $\sigma^q$), the per-window score and the per-dimension KL. It uses the same run-wide row limits, so a reduced page and a full page compare by colour.

**The logarithmic attention scale.** The attention is an `entmax15` over the $L$ lags. Most of its mass is spread thin, and some lags are exactly zero. On a linear scale, one dominant lag would flatten the rest of the panel into the bottom colour. The top of the log scale is the run-wide shared limit. The bottom is the run-wide low percentile of the positive weights, never more than `_LOG_ATTENTION_FLOOR` $= 10^{-4}$ times the top below it.

**How to read it.**

- A grey cell is a zero or masked lag. The log scale masks non-positive cells and paints them light grey. The lag mask zeroes every lag below the source floor $F_u$, and `entmax15` sets other lags to exactly zero.
- The colour scale comes from the whole split's attention, so all pages of the run compare by colour. A page can saturate at the top where its segment exceeds the run's high percentile. The argmax overlay is not affected.
- The reading notes of the full page apply, including the lag note at the foot of both pages: a lag axis here is stored-coefficient time, not a physical delay.
