# Figure guide for the lag-slot transformer

Use this guide to read the figures produced by evaluation and multi-run acceptance. Each entry explains the panels, axes, and main interpretation limits. [EVAL.md](EVAL.md) explains how to run the evaluation and defines terms such as anchor, arm, proposal, and margin.

Start with `headline_arms` for overall predictive performance, then `pred_gap_recordings` for variation between recordings. Read `band_suppression` before `lag_profile`. Use the calibration plots to assess uncertainty and the acceptance plots to compare separately trained models.

## Conventions used throughout

- **Check where the data came from.** Scoring and acceptance figures use saved summary statistics and tables. They can be redrawn without a checkpoint, dataset shard, or PyTorch. Detailed recording traces and input attributions are exceptions: their separate stages rerun the model to collect information that the summary does not contain.
- **Read the units on each panel.** Different units get separate panels, with no second vertical axis. Predictive scores are in nats per anchor, and lower is better. A predictive gap is base minus full; an intervention margin is the intervened score minus full. Positive values mean source benefit and intervention harm, respectively.
- **Use paired intervals for comparisons.** Margins are calculated within each recording before bootstrapping recordings. Use the margin's interval to assess the comparison; overlap between the two arms' separate intervals does not answer the same question.
- **Use labels to identify arms.** Dot-plot colours indicate families: blue for matched branches, grey for reference identities, green for lag bands, and amber for source controls. Curves use fixed band colours in declaration order: amber, green, purple, and vermilion. The text label or legend names the exact arm.
- **Read missing-data messages.** An unavailable result is labelled with a reason. Target-only models have no band or lag profile, and normalised fusion has no additive latent profile. An absent measurement should not be read as zero.
- **Lag axes show stored-coefficient time.** Lag $\ell$ identifies a source coefficient stored $\ell$ steps before the anchor. The seconds axis converts stored steps to seconds. Upstream causal feature extraction combines raw history within each coefficient, so these axes do not measure physiological delay. Every lag figure carries this qualification.

The plots use the shared style in `teb_vae/lag_attn/eval/figures.py`: double-column width, 7 pt serif text, open frames, legends above the data, and the Okabe-Ito colour-blind-safe palette. Panel letters (**a**, **b**, and so on) run left to right, then top to bottom. Qualifications appear as 6 pt footnotes in reserved space below the axes.

## Scoring figures

These figures are saved in `eval_results/figures/`. Filenames use the configured figure format.

### `headline_arms`

**Question:** How well does each condition predict, and how much does each intervention change prediction?

**Left panel:** One dot per arm shows its equal-recording predictive score in nats per anchor, with a recording-level bootstrap interval. Lower is better. Arms are grouped as matched branches, band suppressions, the silence identity, and source controls. Each row also prints the point estimate.

**Right panel:** The first row shows the predictive gap, base minus full. Other rows show intervention minus full, each with the paired interval of its per-recording differences. A positive intervention margin means the model predicts worse after that intervention. Three rows check exact identities: `suppress:none` is zero, while `suppress:all` and `silence` equal the predictive gap.

**Interpretation:** An internal gap can grow because joint training weakens the base branch. Use the acceptance figures to compare both branches with a separately trained, frozen target-only reference. Assess margins using the right panel's paired intervals.

### `pred_gap_recordings`

**Question:** Is the average improvement shared across recordings, and do enough latent draws contribute to the score?

**Left panel:** A histogram of the per-recording predictive gap, with zero and the median marked. Positive gaps mean the source improved prediction. An interval on the overall mean can exclude zero even when many recordings have negative gaps; this panel reveals that variation.

**Right panel:** A histogram of the full branch's effective draw count, $1/\sum_k\alpha_k^2$, with the requested count $K$ marked. Here, $\alpha_k$ is a draw's normalised likelihood weight. Values near one indicate that very few draws dominate the score. Values near $K$ mean the draws received similar weights, but do not prove that increasing $K$ would leave the estimate unchanged.

### `band_suppression`

**Question:** Does removing a broad interval of source history worsen prediction?

**Left panel:** Each declared band's suppression margin and paired interval. The joint `all` removal appears last in grey. Positive values mean removal worsened prediction.

**Right panel:** Exposure for each band: the number of scored anchor-lag pairs with at least one available source channel. A band with no available input is labelled `not measured`.

**Interpretation:** Band margins need not add to the full predictive gap or to a joint-removal margin, because a nonlinear limiter acts after the proposals are summed. They describe the fitted model's response to removal. Contributions can be rearranged across lags while preserving their total and all full-model predictions, yet changing individual removal margins.

### `lag_profile`

**Question:** Which stored lags are available, how do their proposals affect the latent state, and what happens when each lag is removed?

Four panels share the lag axis. Declared bands are shaded and named above the first panel; the upper axis converts stored lag steps to seconds.

1. **Exposure:** The fraction of scored anchors with any available channel at that lag, and the mean fraction of declared source channels available there. Low exposure limits the evidence for the panels below.
2. **Latent profile:** The proposal norm $\lVert r_{t,\ell}\rVert_2$, the change in bounded mean update after removal $\lVert a_t-a_t^{\setminus\ell}\rVert_2$, and the scale proposal norm where supported. Values are averaged over anchors where the lag was available. A large proposal may have little effect when the limiter is saturated.
3. **Divergence drop:** The signed change $K_t-K_t^{\setminus\ell}$. It can be negative: removing a proposal that cancelled another proposal may increase divergence.
4. **Predictive margin:** The score change from removing the lag alone, with its paired recording-level interval. This uses the same draws as the other arms but only the segments allowed by the profile cap. A reason is shown if the cap is unset, the fusion has no per-lag additive updates, or the model has no source pathway.

**Interpretation:** The latent curves do not allocate a fixed total over lags. Read individual peaks alongside broad-band and joint removals; a peak alone is weak evidence when removing the surrounding window has no effect. The predictive profile is based on a capped subset, whose recording count appears in the panel title.

### `horizon_resolved`

**Question:** At which future forecast steps does the source help?

Three panels share the forecast-step axis. The top panel shows the predictive gap and interval. The middle shows each band's suppression margin with paired intervals and a fixed band colour. The bottom shows the source-control margins by step.

Each step is scored as its own likelihood mixture under the shared draws:

$$
D_\tau^{(K)}=-\operatorname{logsumexp}_k(-D_\tau^{(k)})+\log K.
$$

These step scores do not generally sum to the joint block score, because $\log\mathbb E_Z\prod_\tau p_\tau\ne\sum_\tau\log\mathbb E_Zp_\tau$. Read the curves to see where improvement occurs across the forecast window. For example, a band may help the first predicted step more than the last.

### `block_resolved`

**Question:** Which target feature block benefits from the source?

One dot plot is shown for each stored target block: scattering coefficients and phase-harmonic coefficients, in kept-channel order. Each panel lists the gap and intervention margins with paired intervals, summed over that block's channels and the horizon. The title gives the number of retained channels.

**Interpretation:** Block size affects the score's scale, so compare channel counts before comparing magnitudes. As with the horizon plot, separately mixed subset scores do not generally sum to the joint block score.

### `calibration`

**Question:** Does predicted uncertainty match how often observations fall inside forecast intervals?

**Left panel:** Observed coverage versus nominal central coverage for both branches. A calibrated forecast follows the diagonal. Points below it indicate overconfidence; points above it indicate underconfidence. The three nominal levels help distinguish errors near the centre of the distribution from errors in its tails.

**Right panel:** The mean and variance of the probability integral transform, or PIT, for each branch. For a calibrated continuous predictive distribution, PIT values are uniform, with mean $1/2$ and variance $1/12$.

**Interpretation:** Compare both branches to assess whether using the source changes calibration. The full branch alone cannot show whether a calibration problem was already present in the base branch.

## Recording traces

These figures are saved in `eval_results/recording_traces/`. Their stage rereads model outputs for every segment of each selected recording. The file layout, tables, and figures follow the shared family format, so recordings can be examined across architectures.

### `recording_traces_summary`

**Question:** How do selected recordings change as delivery approaches?

A coverage row, then one panel per segment-level summary quantity: divergence, single-draw forecast gap, source-induced latent mean shift, active coordinate count, proposal-norm lag centroid, and cancellation ratio. All rows share one horizontal axis of hours before delivery, with delivery on the right. Each recording is a thin line in its class colour with one marker per segment, lifted at a gap; over the lines the class median per half-hour window is drawn bold, with the inter-quartile band over recordings where at least three recordings fall in the window. The coverage row counts the recordings each window holds per class. When `max_hours_before_delivery` is set the axis is bounded to it and the footnote states how many segments lie beyond.

Up to `eval_config.caps.traces_per_class` recordings are shown per class. These traces support visual exploration. A median over three recordings is a median of three; read the coverage row before the band. A difference between the class medians needs a separate statistical analysis before it can support a class-level claim.

### Per-recording trace: `<class>/<guid>_<subgroup>_trace`

**Question:** What happens within one recording at individual forecast anchors?

All segments share one axis of hours before delivery, delivery on the right. Rows show divergence, the forward pass's single-draw forecast gap, full-branch latent means, bounded mean update $a_t$, per-coordinate divergence, proposal norms over lags, latent norms, mean log-variances, the proposal-norm lag centroid, and mean-update cancellation ratio. The mean and update heatmaps use symmetric colour scales; every heatmap's colour axis sits in its own column so all rows span the same hours. The largest-proposal lag is drawn over the lag heatmap. On the divergence and forecast-gap rows a black step marks each segment's mean over its scored anchors, the value the summary figure carries. Segments alternate a faint background on the line rows, a gap the dataset holds no segment for is shaded darker on every row, line plots break at unscored anchors, and a clinical clock the recording carries is ruled across the page at its onset (labour onset dashed, second stage dotted).

**Interpretation:** The lag heatmap shows proposal magnitude before summation and limiting. It is neither a lag probability distribution nor an allocation of divergence. A jump at a segment boundary can reflect the reset encoder state because each segment is a separate forward pass. Colour scales are set per recording. The lag row is absent for normalised fusion and target-only models, as recorded by `lag_family_present`.

## Input-attribution figures

These outputs are saved in `eval_results/attribution/`. The stage reruns and differentiates the model to estimate how inputs affect selected outputs. It produces `attribution_maps`, `attribution_lag_profile`, `attribution_bands`, `attribution_layer`, `attribution_null`, `attribution_channels`, `attribution_lag_channel`, `attribution_time_profile`, `attribution_checks`, `attribution_time_to_delivery`, one `maps/<class>_<guid>_<subgroup>_anchor<step>_attribution_maps` page per class example (the input coefficients beside every readout's attribution maps at one anchor), and one `traces/<class>/<guid>_<subgroup>_attribution_trace` per traced recording (the divergence and the forecast gap attributed at the same anchors).

The panel layouts are shared with the lag-attentive models. See [their figure guide](../../lag_attn_cfs/eval/FIGURE_GUIDE.md#attributionattribution_mapspdf) for the individual panels. Two differences apply here: each lag row shows a proposal norm, with the same limitations as the recording traces; and the layer figure's left panel splits attribution over the proposal head's lag outputs. That split follows the sum and limiter and is drawn against compensated lag, rather than attention head. It does not allocate divergence among lags.

## Acceptance figures

These plots are drawn from the assembled multi-run acceptance record into the directory supplied with `--figures`.

### `acceptance_comparisons`

**Question:** Which separately trained model variant predicts better in each declared comparison?

Each row shows the paired difference in `nll_full` between the left and right variants. Scores are averaged across training seeds within each recording, then bootstrapped once over recordings. Negative values favour the left variant. Blue indicates the declared draw count and minimum seed count were satisfied for both variants; grey indicates they were not. The record contains the detailed status.

### `acceptance_arms`

**Question:** Does an internal gain also hold against the frozen target-only reference?

Three dot plots share the model-variant axis. They show each variant's internal gap, its full branch minus the frozen reference, and its base branch minus that reference. Negative reference differences favour the evaluated variant. A confidently positive base-minus-reference difference can fail the gate: the jointly trained base has fallen behind the reference, which weakens the interpretation of its internal gap.

### `acceptance_bands`

**Question:** Does a selected lag band still have evidence after accounting for the band search?

Each panel represents a model variant that searched lag bands. A band's coloured interval is the nominal interval. The wider grey interval behind it is adjusted to cover all searched bands together. The selected peak band appears in the title.

**Interpretation:** Use the family-adjusted interval for a claim about the selected peak. The same recordings were used to choose the peak and estimate its margin, so selecting the most favourable nominal interval would overstate the evidence.
