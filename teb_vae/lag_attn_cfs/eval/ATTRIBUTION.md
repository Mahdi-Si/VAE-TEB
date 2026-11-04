# Understanding input attribution

This guide explains how the `attribution` analysis relates a model output to the input coefficients that influenced it. It covers the lag-attention forecasters, including the conv-LSTM and transformer variants, and the lag-slot transformer in `lag_slot_transformer_cfs`. This guide calls the lag-slot transformer the **lag-residual model**, because it combines per-lag proposals into a bounded latent update.

Sections 1–4 explain the method and its checks. Sections 5–7 explain the saved files, the selection and the cost. Sections 8–9 give implementation details. Section 10 lists what to check on a trained model, and section 11 lists the interpretation limits. Section 12 explains the class comparison over many recordings and the Grad-CAM views it adds. Section 13 compares high- and low-$K_t$ anchors, and anchors inside and between contractions, in the last two hours before delivery.

[EVAL.md](EVAL.md) explains the overall evaluation workflow, and [FIGURE_GUIDE.md](FIGURE_GUIDE.md) explains the individual plots. The lag-residual model runs the same attribution implementation as its own registered `attribution` analysis under the shared runner; see [its evaluation guide](../../lag_slot_transformer_cfs/eval/EVAL.md).

## Contents

- [1. What attribution tells you](#1-what-attribution-tells-you)
- [2. Which outputs are attributed](#2-which-outputs-are-attributed)
- [3. Baselines and the integration path](#3-baselines-and-the-integration-path)
- [4. Checks to read before interpreting a map](#4-checks-to-read-before-interpreting-a-map)
- [5. How maps become profiles and summaries](#5-how-maps-become-profiles-and-summaries)
- [6. Output files and figures](#6-output-files-and-figures)
- [7. Selection, settings, and cost](#7-selection-settings-and-cost)
- [8. Implementation: one output per anchor](#8-implementation-one-output-per-anchor)
- [9. Captum methods](#9-captum-methods)
- [10. What to check on a trained checkpoint](#10-what-to-check-on-a-trained-checkpoint)
- [11. Interpretation limits](#11-interpretation-limits)
- [12. Class comparison and Grad-CAM](#12-class-comparison-and-grad-cam)
- [13. Late-window strata: high against low $K_t$, and contractions](#13-late-window-strata-high-against-low-k_t-and-contractions)

## 1. What attribution tells you

Attribution asks which parts of the input contributed to one output of the fitted model. For example, at one forecast anchor, which uterine-pressure coefficients contributed to a change in the predictive gap? The result is a map with one value for each stored time step and input channel.

An **anchor** is the time from which a forecast starts. A **coefficient** is a numerical feature produced by the upstream signal transform. A **channel** follows one such feature through time. A **readout** is the single model quantity that the attribution explains, such as a score or a KL divergence.

| Term | Meaning in this guide |
| --- | --- |
| Target streams | The two feature streams being forecast: scattering features $y^{st}$ and phase-harmonic features $y^{ph}$. |
| Source stream | The additional input $u$ used by the source-conditioned branch. |
| Base and full | The target-only and source-conditioned forecast branches. |
| Latent state | The model's compact, uncertain representation of the input history. |
| Prior and posterior | The target-only and source-conditioned latent distributions. |
| KL divergence, $K_t$ | How much the posterior differs from the prior at anchor $t$. A large value does not mean a better forecast. |
| Predictive gap | Base score minus full score. A positive value means that the full branch scored better. |
| Baseline | A reference input that defines the change the attribution explains. |
| Gradient | How sensitive the chosen output is to a small change in an input value. |
| Integrated gradients, or IG | Attribution obtained by accumulating gradients along a path from a reference input to the observed input. |
| Completeness | The requirement that the summed attributions reproduce the output change along that path, within numerical tolerance. |
| Lag band | An inclusive range of stored steps before the anchor. |

### How this differs from the other analyses

The lag profiles of the other analyses describe the model's behaviour with the source present. In the attentive models, these include attention weights and the attention-weighted KL attribution:

$$
\widetilde K_{t,\ell}=\sum_m K_t^{(m)}\alpha_{t,\ell}^{(m)}.
$$

Here, $m$ identifies an attention head and $\ell$ identifies a lag. In the lag-residual model, a profile can instead show the proposal norm $\lVert r^\mu_{t,\ell}\rVert_2$: the magnitude that a lag proposes before summation and limiting. A proposal norm is not a probability distribution.

The `occlusion` analysis removes source values from a lag band and measures the resulting forecast change. Input attribution is finer: it assigns an output change to individual coefficients across time, channels and streams. Read the two together.

The availability pattern also matters. It shows which source channels are usable at each step, and it depends on time, not on the source values. The attentive model can respond to that pattern even with a zeroed source. Section 3 explains how the baselines handle this response.

The implementation uses Captum, pinned as `captum==0.9.0` in `requirements.txt`, for path integration, interpolation batching and completeness calculations.

## 2. Which outputs are attributed

The wrapper exposes the readouts below. The symbols $\mu^p$ and $\mu^q$ are the prior and posterior means; $\ell^p$ and $\ell^q$ are their log-variances. The index $d$ identifies one latent coordinate.

| Readout | Quantity explained |
| --- | --- |
| `kld` | The divergence $K_t$ at the chosen anchor, from `kld_per_t` or `kld_per_anchor`. |
| `kld_dim` | The divergence contribution $K_{t,d}$ of one latent coordinate. |
| `mu_post_dim`, `mu_prior_dim` | One posterior or prior mean coordinate, $\mu^q_{t,d}$ or $\mu^p_{t,d}$. |
| `nll_full`, `nll_base` | The masked forecast-block score after decoding that branch at its latent mean, under the model's own forecast density. |
| `pred_gap` | The mean-decoded difference $D_{\mathrm{base}}-D_{\mathrm{full}}$, which the collection pass reports as `mean_pred_gap`. |
| `lag_band` | For attentive models, the attention mass averaged over heads within a band. For the lag-residual model, the sum of proposal norms within the band. |
| `nll_horizon` | The full-branch block score at one horizon step $\tau$, summed over the channels still scored at that step, so the per-step scores sum to `nll_full`. It is attributed at the first and the last step (once, when the block has one step). A row's `band` column holds the step as `h<step>`. |
| `mse_full`, `mse_gap` | The forecast fidelity: the masked squared error of the mean-decoded full forecast, and its base-minus-full gap. They are scored under `'mse'` whatever the objective's likelihood, so the learned variance cannot trade against them. |

### Block scores use the model's own forecast density

Every block-score readout uses the two density terms that `metrics.forecast_likelihood_terms(model)` returns, detached, from the model's `forecast_likelihood_kwargs()`:

- the per-channel scored-cell mask $m_{\tau,c}=\mathbb 1[\tau<H_c]$;
- the AR(1) coefficient $\phi_c$ of the forecast residual along the horizon.

With $\phi_c$, each cell is scored on its innovation $e_\tau=r_\tau-\phi_c\,r_{\tau-1}$, with $r=x-\mu$ and $r_{-1}=0$. The recursion restarts after a masked step. The collection pass scores `mean_*` under the same density, so an attributed `nll_full` equals that column at the anchor. A model that builds neither term scores every cell factorised.

Two consequences follow. `nll_horizon` at the last step $\tau=H-1$ sums only the channels with $H_c=H$; at the first step it sums every scored channel. The fidelity readouts `mse_full` and `mse_gap` keep the cell mask but drop $\phi_c$ (`ar_coef = None`), because the error of the mean forecast has no innovation.

### The attribution uses latent means

Forecast-score readouts decode the latent mean, not a sampled latent state. The other readouts also use deterministic model outputs. The forward runs under `attributions.attributed_forward` (section 8), which replaces the reparameterisation by $(\mu^p,\mu^q)$, so no $\epsilon$ is drawn. Repeated calls are therefore comparable under the same execution conditions.

The Monte Carlo predictive score `mc_pred_gap` is not attributed, because that would need a convention for fixing or integrating over random draws. The mean-decoded gap is the comparison that the coupling figures display and that the `lag_high_kl` usefulness analysis prefers.

### What a standard pass computes

- The main readouts (`MAIN_READOUTS`: `kld`, `pred_gap`, `nll_full`, `mse_full`) under both baselines.
- `nll_horizon` at the first and the last horizon step under both baselines.
- `lag_band` for each configured `occlusion_bands` band under `source_null`.
- `kld_dim` for the coordinate with the largest divergence at each selected anchor, under `source_null`.
- Layer attribution for every main readout under `source_null`, split by attention head or lag slot.
- Source-band ablation for every main readout under `source_null`.
- Target-only readouts on one segment per run, to verify that source attribution is zero.
- Up to `EXAMPLES_PER_CLASS` $=3$ example anchors per class: the highest-KL clean anchors of the class (section 7). Each example is attributed for `kld`, `pred_gap`, `nll_full`, `mse_full`, `mse_gap`, `nll_horizon` at both steps and `lag_band` on every configured band, under both baselines. The pass keeps the full maps, the input streams, the raw FHR and UP of the segment, the latent at the anchor and every readout's layer split for the map pages. The example anchor is one of the main rows, so only the variants that no main call takes are integrated again.
- The per-recording trace readouts (`TRACE_READOUTS`: `kld`, `pred_gap`) at a few anchors of every segment of one recording per class, under `source_null`.
- A separate cohort pass for the class comparison (section 12): a larger class-balanced draw, with `kld` and `pred_gap` under `source_null` and Grad-CAM of every main readout.

The lag bands are the validated `eval_config.occlusion_bands`. Besides explicit inclusive `[lo, hi]` pairs, that block can carry the reserved entry `partition_width`. `config_schema.partition_lag_window` expands it against the model's own lag window into contiguous bands named `lags_<lo>_<hi>`, the last band absorbing the remainder, listed before the explicit bands. The runner validates the block before any analysis runs and passes the validated copy to this pass, so the pass sees only pairs.

### Read the sign before the magnitude

A positive signed attribution increases the selected readout along the integration path. Its meaning depends on the readout:

- an increase in `pred_gap` is a better score for full relative to base;
- an increase in an NLL is a worse predictive score;
- an increase in `kld` is a larger latent change.

Plots of absolute attribution, such as $|q_\ell|$, show magnitude and discard the sign. They show where the model was sensitive, but not whether the contribution increased or decreased the readout.

## 3. Baselines and the integration path

Attribution explains a change relative to a baseline. You must therefore know both the observed input and its reference.

### Two zero-based references

The loader standardises each channel, so zero is its population mean under those statistics. Zero does not mean that the signal had no physical activity. The model also gates unavailable steps to zero, so replacing their stored values with zero does not change what the model reads.

| Baseline | What changes along the path | What the attribution describes |
| --- | --- | --- |
| `source_null` | Source values move from zero to their observed values; both target streams stay fixed. | The response to source content relative to the model's zero-source response. Target attribution is exactly zero, because the target inputs do not move. |
| `all_zero` | All three streams move from zero to their observed values. | The output change as target and source inputs are introduced together. The split between streams depends on this path. |

Under `source_null`, the availability indicators are identical at both ends of the path. They are not varying input coefficients, so they receive no input attribution. The exact zero-source readout is saved separately. It includes the model's availability-related response and its behaviour at zero input, so it is not a pure measurement of the clock.

### Why integration starts slightly above zero

Normalisation can make the model very sensitive near an all-zero input. The conv-LSTM uses per-step `CausalGroupNorm`, and the transformer variants use token-wise `RMSNorm`. As the variance approaches zero, the normalisation scale can become very large, and so can the gradients.

The `attributions` module docstring states this degeneracy for every encoder in the family. It applies to every target encoder and to the attentive models' source paths. The lag-residual model's default `pointwise` source stem has no parameters and no temporal normalisation, so its source-null path does not pass through a normaliser at the stem; its all-zero path still passes through the target encoder's `RMSNorm`. The pass applies the same entry point to every path.

To avoid integration through the most sensitive region, the shared implementation starts slightly toward the observed input:

$$
x_0=b+\alpha_0(x-b),\qquad \alpha_0=10^{-3}.
$$

Here, $b$ is the exact baseline, $x$ is the observed input and $x_0$ is the integration entry point. The constant is `BASELINE_ENTRY_FRACTION`. Integrated gradients explain the change from $x_0$ to $x$, not the whole change from $b$ to $x$.

For an input element $i$, the attribution is:

$$
\mathrm{IG}_i=(x_i-x_{0,i})\int_0^1\frac{\partial f\bigl(x_0+s(x-x_0)\bigr)}{\partial x_i}\,ds.
$$

Captum approximates this integral numerically with `IG_STEPS` $=64$ steps. More steps can improve numerical accuracy. The saved per-row residual shows whether the approximation was adequate for that row.

### The entry jump is saved separately

Each row records `value_baseline` $=f(b)$, `value_entry` $=f(x_0)$ and `value_input` $=f(x)$. The **entry jump** is:

$$
\mathrm{entry\_jump}=f(x_0)-f(b).
$$

Together, these values account for the input readout approximately as:

$$
f(x)\approx f(b)+\mathrm{entry\_jump}+\sum_i\mathrm{IG}_i.
$$

The entry jump belongs to no input element of the map. A large jump means that the baseline is in a region where the model changes sharply. It describes the baseline comparison and the model normalisation; do not distribute it over time as a source attribution.

## 4. Checks to read before interpreting a map

The tests in [the attentive model's attribution suite](../tests/test_eval_attribution.py) and [the lag-residual model's suite](../../lag_slot_transformer_cfs/tests/test_eval_attribution.py) exercise four properties on small models. The evaluation also records diagnostics in the attribution block's `checks`.

### 4.1 No attribution after the anchor

Input-gradient attribution to stored steps after the forecast anchor must be exactly zero. This shows that the attributed output does not use future input values.

For the conv-LSTM, this requires `causal_norm: true`, because ordinary group normalisation pools over time and can expose future information. The transformer variants use step-wise causal computation.

This is a gradient check. A separate value-level intervention that ablates only future positions must change the readout by no more than floating-point rerun noise. Do not confuse that future-only intervention with the broader `rest` ablation group of section 5.

### 4.2 No attribution to unavailable source inputs

A source coefficient that the warm-up gate excludes must have zero attribution. Multiplication by zero also makes its gradient zero.

For an aligned channel, the encoder reads stored step $t-d_c$ and masks it while $t<W'_c+d_c$. In original stored coordinates, the channel therefore becomes usable at $W'_c$. The helper `warm_from_step` computes this threshold for each declared channel. Dropped channels never become usable.

### 4.3 No source attribution for target-only outputs

The prior mean and the base block score (`TARGET_ONLY_READOUTS`) must have exactly zero source attribution. The attentive model's prior clock comes from a detached zero-source encoding. The lag-residual model's metadata clock is a function of position. Neither gives these target-only readouts a gradient path to observed source values.

### 4.4 Attributions reproduce the integrated output change

The sum of integrated gradients must reproduce $f(x)-f(x_0)$. The relative completeness residual divides the absolute discrepancy by the larger of $|f(x)-f(x_0)|$ and $|f(x)|$. This avoids dividing a small numerical error by an almost-zero output change.

The block's `checks` record, over the main readouts' rows, the median residual, the maximum residual and the number of rows above `COMPLETENESS_TOLERANCE` $=10^{-2}$ (`n_rows_over_tolerance`). `attribution_summary.csv` holds the maximum and median per readout, baseline and band. Inspect these values before you treat a map as an accurate decomposition.

The conv-LSTM fixture tests allow a median below $10^{-2}$ and a maximum below $5\times10^{-2}$. The lag-residual fixture tests require a maximum below $10^{-3}$. These are test tolerances on tiny models, not guarantees for a trained checkpoint.

Together, the checks show whether the maps follow the model's causal input support and account for the specified output difference. They do not establish a physiological cause.

## 5. How maps become profiles and summaries

Each attributed row corresponds to an anchor, readout and baseline, with a band or coordinate where relevant. The two target-stream maps are concatenated along the target-channel axis. The arrays have shapes $(N,T,c_y)$ for target attribution and $(N,T,c_u)$ for source attribution, where $N$ is the number of attributed rows.

Reductions are first calculated per row, then combined within each recording, then summarised across recordings. The main selection uses `caps.attribution_segments_per_recording` segments per recording and several anchors from each.

| Reduction | Calculation |
| --- | --- |
| Time profile | Sum over channels for a value at each stored step, with shape $(N,T)$. |
| Channel profile | Sum over time for a value per input channel, with shape $(N,C)$. |
| Lag-aligned source profile | Index the source time profile again relative to the anchor: $q_\ell=p^u_{t_a-\ell}$. |
| Unsigned channel profile | Sum absolute values over time, $\sum_t \lvert a_{t,c} \rvert$, so a channel whose contributions cancel over time still shows. |
| Lag-by-channel maps | Index the full map by offset from the anchor, $(L, C)$, and accumulate the signed and unsigned means over attributed anchors, for the main readouts under both baselines. |
| Frequency-band sums | Sum input-channel attribution with the declared-channel band map. |
| Lag-band sums | Sum the lag-aligned profile within each configured inclusive lag band, clipped to the available lag window. |
| Layer split | Attribute through the attention-head fusion modules or through the lag-slot proposal outputs. |
| Band ablation | Zero a group of source inputs and report the change in the readout. |
| Null decomposition | Report the exact baseline readout, the entry jump and the integrated attribution separately. |

### Lag alignment and agreement

For $\ell=0,\ldots,L-1$, the lag profile reads the source attribution at $t_a-\ell$. Positions before the recording are `NaN`. The compensated seconds axis comes from `lag_axis.compensated_seconds_axis(L, delay_steps)`. It uses the collection's `lag.delay_steps` for attentive models and zero for the lag-residual model.

Every lag in this analysis is the stored-step offset $t-s$ from the anchor; the plan records this as `lag_definition`. This equals the attention's own lag only while the source gate shifts no channel. On an aligned arm, the gate shifts channel $c$ by $d_c$ stored steps, and the attention's lag $\ell$ reads stored step $t-\ell-d_c$. The lag profile, the lag-by-channel maps and the band ablation are then $d_c$ steps away from the attention's lags and from the occlusion bands, which apply to the gated stream. The pass records the largest shift as `source_channel_shift_steps_max` and logs a warning when it is nonzero. It does not correct the offset. With the shipped `causal_align_reference: null`, no channel is shifted.

The profile is compared with the model's own lag readout at the same anchor: `source_kl_lag_map[t_a]` for attentive models, or the proposal norms masked by `lag_valid` for the lag-residual model.

Two statistics describe agreement on the shared valid lags:

- `lag_corr` is the Pearson correlation between $|q_\ell|$ and the model profile.
- `lag_js` is their Jensen–Shannon distance after normalisation to distributions, in base $2$, with a range of $[0,1]$. A smaller distance means more similar distributions.

Empty or unsuitable profiles can give `NaN`; check counts and validity before you compare these values.

### Frequency bands use the input-channel map

Frequency-band sums join through `band_channel_map.csv`, keyed by `stream`, `channel` and `band`. Attribution has the model's declared input width, so it must use the declared-channel map. Decoder-side skill analyses use the retained target-channel map instead.

Dropped input channels stay on the declared axis with zero attribution, because their gates exclude them. When the map is missing, the lag-residual stage creates it through `band_partition.emit_partition`, with the same shards and resolved budget. It records a skip if channel provenance is unavailable.

### Ablation has its own sign convention

Source-band ablation reports:

$$
\Delta_{\mathrm{ablation}}=f(x^{\setminus\mathrm{band}})-f(x).
$$

A positive value means that the readout increased after the band was removed. This is the opposite subtraction order from an attribution of present input, so compare meanings before you compare signs. An increase in NLL is worse prediction; an increase in `pred_gap` is a larger advantage for full over base.

`FeatureAblation` removes each band across all source channels relative to the row's anchor. It also removes a `rest` group that contains the source positions outside the configured bands. That group can contain future positions and earlier history before the lag window. Future positions must have no effect, but earlier history can matter when the source encoder's receptive field extends beyond the searched lags. A nonzero `ablation_rest` therefore does not by itself show future leakage.

### Read the null decomposition as a baseline comparison

Under `source_null`, $f(b)$ is the model's exact zero-source divergence at the anchor. The IG sum is the integrated source-content response, and the entry jump accounts for the excluded start of the path. Under `all_zero`, the IG sum is split across the target and source streams. Both decompositions depend on the chosen baseline and path.

## 6. Output files and figures

All files go under `attribution/` in the evaluation results directory. NLL, predictive-gap and divergence attributions are in nats per anchor. The fidelity readouts are in squared standardised coefficient units (`squared z`). Mean-coordinate readouts are in latent-coordinate units. The attentive `lag_band` readout is an attention share in $[0,1]$; the lag-residual version is a norm in latent units (`READOUT_UNITS`). Signed attributions need not lie within the range of the readout. The per-row, summary, frequency-band and lag-band tables hold this as a `unit` column, which is also the unit of every attribution column beside it.

Every lag axis is stored-coefficient time, not a physiological delay axis.

### Tables and arrays

| File | Contents |
| --- | --- |
| `attribution_rows.csv` | One row per attributed anchor, readout, baseline and band where applicable: identity, readout values, attribution totals, numerical checks, agreement and intervention results. |
| `attribution_vectors.npz` | Time, lag, channel and layer profiles, row for row with `attribution_rows.csv`, with each row's identity restated as `row_*` arrays. |
| `attribution_maps.npz` | The example anchors: the input streams the encoders read (`input_target`, `input_source`), the live steps, the model's lag readout, the raw FHR and UP of the segment (`example_raw_fhr`, `example_raw_up`, with units), the latent at the anchor, and every example readout's full maps under both baselines (`map_target`, `map_source`, one entry per example, readout, baseline and band, keyed by `map_example`, `map_readout`, `map_baseline`, `map_band`), with the readout at the input and at the exact baseline and the layer split. |
| `attribution_examples.csv` | The manifest of the example pages: identity, class, subgroup, epoch, anchor and figure path. |
| `attribution_lag_channel.npz` | The population lag-by-channel maps: for each main readout, baseline and stream, the mean over attributed anchors of the signed (`<readout>__<baseline>__<stream>__mean`) and unsigned (`__mean_abs`) attribution indexed by offset from the anchor, $(L, C)$, with the anchor count. |
| `attribution_recordings.csv` | One row per recording: main readout totals and lag agreement under `source_null`, cohort labels and `n_segments`. Used for grouped figures. |
| `attribution_summary.csv` | One row per readout, baseline and band: counts, recording-mean values, totals and structural-check summaries. |
| `attribution_bands.csv` | One row per main readout, stream and frequency band: `baseline`, `attribution_mean`, `n_recordings`, and the target-band `spectral_skill_pred_gap_nats` where available. Each stream is read on the path along which it moves: the target under `all_zero`, the source under `source_null`. |
| `attribution_lag_bands.csv` | One row per main readout and lag band under `source_null`: bounds, IG sum, ablation change, recording count and an occlusion comparison where available. |
| `attribution_layer.csv` | One row per readout, layer unit, and class or `pooled`: recording count and mean attribution. |
| `attribution_null.csv` | One row per readout, baseline, and class or `pooled`: recording means of the input, baseline and entry values, the entry jump, the attributed sum and the stream totals. |
| `attribution_blocks.csv` | One row per readout, band, baseline and input block (`target_scattering`, `target_phase`, `source_scattering`, `source_phase`), for every readout except `kld_dim` and `lag_band`: recording-mean signed and unsigned sums and the unsigned share. |
| `attribution_traces.csv` | The manifest of traced recordings: identity, class, subgroup, segment and anchor counts, span, coverage and array/figure paths. |
| `attribution_trace_anchors.csv` | Every traced anchor of every traced recording: identity, time axis, and each trace readout's scalars and lag statistics. |
| `traces/<class>/<guid>_<subgroup>_attribution_trace.npz` | The shared trace arrays for one recording, including `attribution_lag_map` and `model_lag_map`. |
| `attribution_cohort_rows.csv`, `attribution_cohort_vectors.npz` | The cohort pass's integrated-gradient rows and row-aligned vectors, in the layout of `attribution_rows.csv` and `attribution_vectors.npz`. |
| `gradcam_rows.csv`, `gradcam_vectors.npz` | One row per cohort anchor and Grad-CAM readout: the readout value and, per view, the unnormalised `<view>_total` and the `<view>_centroid_s`. The arrays hold the normalised profiles `target` $(N,T)$, `source` $(N,L)$ and `attention` $(N,L)$. |
| `attribution_class_recordings.csv` | One row per cohort recording: its class and every tested metric (section 12). |
| `attribution_class_stats.csv` | One row per metric: per-class `n`, mean and bootstrap interval, the Kruskal–Wallis statistic, `p_value`, `p_holm` within the family, and `significant`. |
| `attribution_class_pairwise.csv` | One row per metric and class pair: the two means, `mean_diff` with its bootstrap interval, the Mann–Whitney `p_value`, `p_holm_pairs`, `cliffs_delta`, `magnitude` and the metric's `omnibus_p_holm`. |
| `attribution_class_lag_channel.npz` | Per class, the mean signed and unsigned lag-by-channel maps of the cohort readouts, keyed `<readout>__<baseline>__<stream>__<class>__mean[_abs]`. |

### Per-anchor column reference

| Column group | Names |
| --- | --- |
| Recording and anchor | `guid`, `epoch`, `clinical_class`, `subgroup`, `anchor` (stored step), `column` (anchor-axis position). |
| Attributed quantity | `readout`, `baseline`, `band`, `unit`, `coordinate` ($-1$ when no coordinate applies), `kld_top_coordinate`. |
| Readout values | `value_input`, `value_baseline`, `value_entry`, `entry_jump`. |
| Attribution totals | `attributed` (IG sum), `target_total`, `source_total`, `target_abs_total`, `source_abs_total`. |
| Numerical checks | `ig_delta`, `completeness_rel`, `after_anchor_max_abs`, `gated_off_max_abs`. |
| Lag agreement | `lag_corr`, `lag_js`. |
| Band, block and layer summaries | `lagband_<band>`, `band_<stream>_<band>`, `block_<block>`, `block_abs_<block>`, `layer_total` (main readouts under `source_null`). |
| Ablation results | `ablation_<band>`, `ablation_rest` (main readouts under `source_null`). |

`attribution_vectors.npz` contains:

- `time_profile_target` and `time_profile_source`, shape $(N,T)$;
- `lag_profile`, `target_lag_profile` and `model_profile`, shape $(N,L)$;
- the signed `channel_profile_target` and unsigned `channel_abs_profile_target`, shape $(N,c_y)$;
- the signed `channel_profile_source` and unsigned `channel_abs_profile_source`, shape $(N,c_u)$;
- `lag_seconds`, shape $(L,)$;
- `layer_per_unit`, shape $(N,M)$ for $M$ attention heads or $(N,L)$ for lag slots, with `NaN` rows where no split was taken;
- the identity arrays `row_guid`, `row_subgroup`, `row_clinical_class`, `row_epoch`, `row_anchor`, `row_readout`, `row_baseline` and `row_band`, as plain strings and numbers.

`attribution_maps.npz` identifies each example with `example_guid`, `example_subgroup`, `example_clinical_class`, `example_epoch` and `example_anchor`. Its `map_target` and `map_source` arrays have shapes $(n,T,c_y)$ and $(n,T,c_u)$. Here $n$ counts map entries (one per example, readout, baseline and band), not examples; `map_example` indexes each entry back to its example.

In `attribution_recordings.csv`, the metric columns are `<readout>_source_total`, `<readout>_target_total`, `<readout>_lag_corr` and `<readout>_lag_js` for every main readout. The summary table includes `unit`, `completeness_rel_max`, `completeness_rel_median`, `after_anchor_max_abs` and `gated_off_max_abs` beside counts and means.

The lag-band table records `lag_lo`, `lag_hi`, `ig_attribution_mean`, `ablation_delta_mean`, `occlusion_delta_total_nats`, `n_recordings` and `unit`. The occlusion comparison reads the occlusion summary's `delta_total_recording_mean_nats`, the same recording-mean aggregation as the columns beside it. It falls back to the pooled `delta_total_nats` only on a summary without the recording mean. That delta is the change of the full-branch block score, so the join uses:

- sign $+1$ for `nll_full`;
- the flipped sign for `pred_gap`, because the base branch reads no source;
- no value for `kld` and `mse_full`.

The join is a population comparison, not a row match. Occlusion draws its own random anchor per segment over every evaluated segment, seeded per batch, scores it under a paired latent draw, and zeroes the gated source stream. The attribution ablation zeroes the stored source at the attributed anchors. The lag-residual model's pass joins no occlusion summary. The joined columns exist only when the corresponding analysis files are available.

### Figures

| Figure | Contents |
| --- | --- |
| `attribution_maps.pdf` | One column per class example. Each time row uses the segment's stored-time axis, with the anchor ruled and its forecast block shaded. From top to bottom: the raw FHR and UP on the fixed CTG scales, the target and source input coefficients, the target attribution of $K_t$ under `all_zero`, the source attribution of $K_t$ under `source_null`, and a lag panel that compares the source attribution with the model's lag readout. Each map row shares one colour scale across the columns. |
| `maps/<class>_<guid>_<subgroup>_anchor<step>_attribution_maps.pdf` | One page per class example on one stored-time axis, laid out as the samples pages. First the raw FHR and UP, then the inputs with cold cells blanked. Then, for every example readout (`kld`, `pred_gap`, `nll_full`, `mse_full`, `mse_gap`, `nll_horizon` at both steps, and `lag_band` per band), the target map (`all_zero`) above the source map (`source_null`) on symmetric-log colour scales. Then the latent at the anchor, the per-head or per-lag split of every readout, and every readout's lag-aligned source attribution on one axis. |
| `attribution_lag_profile.pdf` | The normalised absolute lag attribution against the normalised model profile, averaged over recordings, pooled and by class, with agreement in the titles. A last row overlays every readout on one lag axis, and the band readouts against their own bands. |
| `attribution_bands.pdf` | Frequency-band attribution by stream, with target-band spectral skill on a second vertical axis where available. Below: lag-band IG sums, ablation changes and occlusion comparisons. Check the axes and sign conventions separately. |
| `attribution_layer.pdf` | Per-head or per-lag attribution on the left; input profiles for the anchor's highest-divergence coordinate on the right. |
| `attribution_null.pdf` | Input divergence, exact null response, integrated content attribution and entry jump by class on the left; the `all_zero` target/source split on the right. |
| `attribution_channels.pdf` | Per main readout, the signed and unsigned attribution per declared channel of each stream, coloured by frequency band where the channel map exists. |
| `attribution_lag_channel.pdf` | Per main readout, the mean unsigned attribution by offset from the anchor and declared channel: the target stream under `all_zero`, the source stream under `source_null`. |
| `attribution_time_profile.pdf` | Per main readout and stream, the mean positive and negative parts of the attribution by offset from the anchor, with the net and unsigned means. |
| `attribution_checks.pdf` | One horizontal strip per attributed variant (readout, band and baseline), one point per row: blue on the `source_null` path, vermilion on the `all_zero` path, with the median as a bar. Left: the relative completeness residual on a log axis against the tolerance. Middle: the entry jump as a share of the readout's whole move along the path, $\lvert f(x_0)-f(b)\rvert/\lvert f(x)-f(b)\rvert$. Right: the two structural checks per readout. |
| `attribution_time_to_delivery.pdf` | The source attribution total and the lag centroid of every attributed anchor against hours before delivery, by class. |
| `attribution_blocks.pdf` | Per readout, the unsigned share and the signed sum of the attribution in each of the four input blocks. |
| `attribution_horizon.pdf` | The per-step score at the first and the last horizon step: stream totals, and the source and target attribution by offset per step. |
| `attribution_classes.pdf` | Per cohort readout: the class-mean signed source attribution by lag with each class's median $K_t$, the class-mean unsigned lag share, every class pair's difference of shares with its band, and the per-recording lag centroid by class with the Holm-adjusted Kruskal–Wallis $p$. |
| `attribution_class_maps.pdf` | Per cohort readout: the class-mean unsigned source attribution by lag and channel, and the difference map of every class pair. |
| `gradcam_classes.pdf` | Per main readout and Grad-CAM view: the class-mean profile and every class pair's difference, with bootstrap bands. |
| `traces/<class>/<guid>_<subgroup>_attribution_trace.pdf` | The raw FHR and UP of every segment on the top rows, then the attribution lag maps of the divergence and of the forecast gap through one recording beside the model lag map, with each readout's agreement, totals and values on the hours-before-delivery axis. Every panel shares its colour scale or $y$ range across the traced recordings. |

[FIGURE_GUIDE.md](FIGURE_GUIDE.md) gives panel-by-panel interpretation. Every figure except the checks page prints `ATTRIBUTION_NOTE`, the one-line form of `ATTRIBUTION_CAVEAT`. The long caveat, the group-delay caveat and the model-specific lag qualification are in the record (`caveat`, `lag_qualification`). Every map blanks the cells that the model never read: a channel's cold steps, and the steps after the anchor on an attribution map. Attributions use a symmetric-log colour scale that spans `LOG_DECADES` below the scale's largest magnitude. Line and bar panels of attributions use symmetric-log axes for the same reason.

**Shared scales.** The map overview and every example page use one set of colour scales built by `attributions.example_norms` over all class examples; `SHARED_SCALE_NOTE` states this. The set has:

- one linear scale per input stream (1st to 99th percentile of the read cells of all examples);
- one symmetric-log scale per readout, baseline and stream, at plus or minus its largest magnitude over all examples;
- fixed CTG scales for the raw FHR and UP.

A colour therefore means the same value on every page. The trace pages do the same through `traces.shared_panel_scales`: each heatmap and line panel is scaled once over every traced recording.

### The summary block

The block is `results.attribution` in `summary.json` for both model types, because both run it as a registered analysis under the shared runner.

It contains `n_samples` (segments), `composition`, `plan`, `selection`, `trace_selection`, `cost`, `checks`, `class_contrast` (section 12), `summary`, `lag_bands`, `blocks`, `joined`, `methods`, `lag_qualification`, `caveat`, `traces`, `examples`, `failures` and `files`.

The `plan` records the cap, seed, anchors per segment, integration steps, entry fraction, baselines, main readouts, horizon steps, example readouts, lag readout and lag bands, `lag_definition`, `source_channel_shift_steps_max`, layer, `delay_steps`, the anchor geometry, `attributed_forward` (one anchor per row, decoded at the latent means, no $\epsilon$ drawn), and the trace settings. `joined` records which supporting files were found. Attentive evaluators also include `grouped_frames`; the lag-residual block adds its channel-map record and an ablated-input check.

The lag-residual tests check that this block avoids names that its acceptance gate reserves for unsupported attention distributions or per-lag KL allocations.

## 7. Selection, settings, and cost

### Main attribution selection

The pass attributes anchors where the source moved the belief and the forecast window is clean. The rule (`attribution_pass.informative_anchors`) uses the collection pass's `per_anchor.parquet`:

1. In each segment, take the `HIGH_KL_QUANTILE` $=0.7$ quantile of that segment's own $K_t$ as its threshold. Every segment therefore offers its most coupled anchors, including segments of a subgroup with weak coupling everywhere.
2. Keep the anchors at or above their segment's threshold whose forecast `coverage` reaches `CLEAN_COVERAGE` $=0.95$. These are the candidates.
3. Draw recordings class-balanced and seeded, round-robin over each class's subgroups, so every subgroup of a class is drawn before any subgroup is drawn twice. Each class draws $\lceil \texttt{cap} / (n_{\mathrm{classes}} \cdot k) \rceil$ recordings, with $k$ = `caps.attribution_segments_per_recording` (main pass; the cohort pass uses $k = 1$).
4. From each drawn recording, take $k$ segments that hold a candidate, spread evenly over the recording's timeline by `epoch`. With $k = 1$ this is the middle segment.
5. In each segment, keep up to `ANCHORS_PER_SEGMENT` $=4$ candidates, highest $K_t$ first, at least one horizon $H$ apart. A candidate whose mean validity over the searched lag window $[t_a-L+1,t_a]$ is below `CLEAN_COVERAGE` is used only when too few clean ones remain; `plan.anchor_selection.n_unclean_anchors` counts these.
6. The `EXAMPLES_PER_CLASS` $=3$ recordings of each class with the largest top $K_t$ give the example pages, each at the highest-KL anchor of its best segment. The overview `attribution_maps.pdf` shows the first of each class.

The population figures and tables therefore describe the average behaviour at each segment's most coupled, clean anchors. They do not describe a typical anchor, or anchors above one cohort-wide level of $K_t$. `plan.anchor_selection` records the median per-segment threshold, the pooled level for comparison, the candidate count and the recordings with candidates. Rows are averaged per segment and then per recording, so a recording stays one unit.

Without a per-anchor table (an offline rerun, or the lag-residual cell, whose pass does not supply one), the pass falls back to the spread rule: the middle segment of each recording, `ANCHORS_PER_SEGMENT` anchors spread evenly over its scored anchors, and one example per class. `plan.anchor_selection.rule` names the rule that ran.

| Setting or constant | Value and meaning |
| --- | --- |
| `eval_config.caps.attribution_segments` | The segment cap of the main pass. The committed overrides set $144$. Without the key, the analysis uses `DEFAULT_SEGMENTS`, not every segment. |
| `eval_config.caps.attribution_segments_per_recording` | Segments per drawn recording, $k$. The committed overrides set $4$. Without the key, `DEFAULT_SEGMENTS_PER_RECORDING` $=1$. |
| `eval_config.caps.attribution_cohort_segments` | The segment cap of the cohort pass (section 12). The committed overrides set $150$; `DEFAULT_COHORT_SEGMENTS` is also $150$. |
| `eval_config.caps.attribution_strata_segments` | The segment cap of the strata pass (section 13). The committed overrides set $72$; `DEFAULT_STRATA_SEGMENTS` is also $72$. |
| `DEFAULT_SEGMENTS` | $24$: the fallback main-selection cap. |
| `ANCHORS_PER_SEGMENT` | $4$: anchors per segment. |
| `IG_STEPS` | $64$: integration steps. |
| `IG_INTERNAL_BATCH_SIZE` | $64$: Captum's internal batch of interpolated inputs, counted in rows times steps per forward, and raised to at least the call's row count. |
| `BASELINE_ENTRY_FRACTION` | $10^{-3}$: how far to move from the exact baseline before integration. |
| `COMPLETENESS_TOLERANCE` | $10^{-2}$: the relative residual threshold that the pass counts. |
| `TRACE_RECORDINGS_PER_CLASS` | $1$: detailed attribution traces per clinical class. |

The caps are evaluation settings. The other names are implementation constants, not `eval_config` keys. Use the saved `plan` to identify the values of a completed run.

### Detailed trace selection

The trace selection is separate from the main draw. For each class, it chooses the recording whose stored segments most completely cover the window. When `max_hours_before_delivery` is set, the window is that interval before delivery; otherwise it is the recording's own span.

`attribution_pass.recording_completeness` compares the number of stored segments inside the window with the number expected from the segment stride. Ties go to the larger segment count and then to the recording identifier. A recording with fewer than two segments inside the window is not eligible. `trace_selection` records the selected recording's coverage.

This rule selects for fewer gaps, not for strong uterine activity, large KL or large predictive gain. The trace attributes every readout in `TRACE_READOUTS` (`kld` and `pred_gap`) under `source_null`, at the same selected anchors across the recording's segments, with no layer split or ablation.

### How to estimate runtime

Let $n$ be the number of attributed rows in a batch ($k$ anchors per segment times its segments). Each IG call processes the $n$ rows through the integration steps plus three plain forwards (input, exact baseline, entry point). The main work per batch is:

- eight IG calls for the four main readouts under both baselines;
- two IG calls per horizon step for `nll_horizon`, under both baselines;
- one `source_null` IG call per lag band, and one for the top coordinate;
- four layer attributions, one per main readout under `source_null`;
- four grouped band ablations, one per main readout, each costing the reference plus one intervention per band and `rest`.

Under `source_null`, the IG calls and the ablations give Captum the source stream alone and carry the target streams as fixed arguments. Their backward pass therefore never enters the target encoder.

The example anchors add little, because their maps are the main calls' own rows at that anchor. Only the variants that no main call takes are integrated again, as single-row calls: `mse_gap` under both baselines, `lag_band` under `all_zero` per band, and the layer split of every variant that is not a main readout (`mse_gap`, `nll_horizon` at each step, `lag_band` per band). The trace attributes two readouts at each of its anchors.

The attributed forward decodes one anchor per row, not the whole dense anchor axis (section 8). Doubling the step count roughly doubles the integration part of the work. Runtime also depends on model size, device, batching and trace length.

Read the completed pass's `cost` values: `elapsed_s`, `n_rows`, `n_forward_equivalents`, `seconds_per_row`, `hours_per_1000_samples` and, on CUDA, `peak_allocated_bytes`. They time the drawn segments only; the traces are timed separately as `trace_elapsed_s` over `trace_n_anchors` anchors. Choose a segment cap from those rates, and inspect completeness before you reduce the integration step count.

## 8. Implementation: one output per anchor

`attributions.AnchorReadout` is an `nn.Module` wrapper around the real model. It accepts the three input streams, two extra tensors for scoring, and two per-row integer tensors:

- The extra tensors are the declared-width target features and the decimated validity `weight`.
- `columns` gives each row's position on the decoded anchor axis.
- `coordinates` gives the latent coordinate for a coordinate-specific readout.

The wrapper first converts each row's column into a stored step through the model's own `_build_anchor_index` at the dense evaluation geometry, with no forward. It then calls the model inside `attributions.attributed_forward(model, anchors)`; the lag-residual call also sets `return_proposals=True`. It selects each row's anchor and returns a vector of shape $(B,)$, which gives Captum one scalar per row.

### The attributed forward decodes one anchor at the latent means

`attributed_forward` is a context manager. For the duration of the call, it replaces two of the model's seams on the instance, and it restores them on exit in all cases:

- **The anchor builder.** `_build_anchor_index` returns each row's own anchor as a one-anchor set. The decoder is most of a dense forward's cost, and every readout reads one anchor.
- **The reparameterisation.** `_reparameterize_shared` (attentive models) or `reparameterize_shared` (lag-residual model) returns $(\mu^p,\mu^q)$ instead of a sample (`REPARAMETERISATION_SEAMS`). The forward's two forecasts are therefore the mean-decoded ones that the collection pass scores as `mean_*`, through the same decoder call and persistence input. No $\epsilon$ is drawn.

This is equivalent to the dense forward for every readout here, because everything read at an anchor depends on that anchor alone. The latents are dense over $T$ or computed per anchor, and the decoder decodes each anchor independently. The tests assert the equality. The forward decodes $1$ anchor instead of the whole axis, and an attributed readout consumes no random number. Tensors on the anchor axis, including `anchor_valid`, the decoded forecasts and the lag-residual proposals, have a single column, so the readouts take `[:, 0]`.

### Several anchors can share one call

`expand_rows` repeats each segment for its chosen anchors. For $B$ segments and $k$ anchors, the expanded batch has $Bk$ rows. Captum expands the per-row anchor arguments with its interpolated inputs. Tests compare this batched calculation with one-anchor-at-a-time attribution at an absolute tolerance of $10^{-5}$.

### Targets and masks follow Captum's expanded batch

Block-score readouts build the target block and mask again on every wrapper call, from the extra tensors and each row's anchor. They use `_build_forecast_target`, the anchored `forecast_mask` over `model.scored_weight(weight)`, the model's `coverage_floor`, and the density terms of `forecast_likelihood_terms(model)` (section 2), which matches collection scoring. They score the forward's own mean-decoded forecasts `mu_full`/`mu_base` at the one decoded anchor. A target block built once outside the wrapper could become misaligned when Captum expands the batch.

### Unused inputs return zero attribution

Some readouts do not depend on an input; for example, a target-only score does not depend on source values. The wrapper connects every input and both posterior parameters to the computation graph with a zero coefficient. The derivative is then exactly zero, and autograd does not reject an unused tensor.

The same mechanism supports layer outputs that affect a parameter the readout does not use, such as scale proposals for a mean-decoded gap.

### The binding handles different time axes

Attentive models expose dense latent tensors over stored time, with shape $(B,T,\ldots)$, so the wrapper gathers the anchor's stored step. The lag-residual model indexes its latent tensors by decoded anchor, and under the attributed forward that axis holds only the row's anchor, so the wrapper takes column $0$. `CellBinding` (`ATTENTION_CELL`, `SLOT_CELL`) identifies the layout; baselines, reductions and figures are shared.

## 9. Captum methods

The block's `methods` record (`attributions.METHOD_RECORD`) saves each method's status and reason in the output.

| Method | Status | Reason |
| --- | --- | --- |
| `IntegratedGradients` | Used: primary method. | The path enters the baseline at the entry fraction, and the completeness residual is measured per row. |
| `LayerIntegratedGradients` | Used. | Gives a per-head split through the attentive fusion modules, or a per-lag split through the residual proposal head, under `source_null`. |
| `FeatureAblation` | Used. | Removes source values in groups defined by lag band relative to the anchor, which is `occlusion`'s intervention on this analysis's readouts and anchors. |
| Grad-CAM | Used in the cohort and strata passes, lag-attentive cells. | Computed in `gradcam.py` (section 12.2). |
| `InputXGradient` | Evaluated; not used. | A single-point gradient with no completeness. |
| `Saliency` | Evaluated; not used. | The absolute gradient: unsigned, with no completeness. |
| `GradientShap` | Evaluated; not used. | Integrated gradients averaged over a random baseline distribution. Its per-sample completeness residual is of the order of the readout by construction, and the random baseline makes two runs disagree. |
| Sliding-window `Occlusion` | Evaluated; not used. | A window that crosses the anchor assigns its effect to steps after the anchor, so its per-step output fails the causality check by construction. |
| `DeepLift` | Evaluated; not usable. | Its rescale rule reaches only the module nonlinearities it hooks (ReLU, Tanh, Sigmoid). The GELU and SiLU functionals, the smooth bounds and `entmax15` pass through as plain gradients. |
| `LayerConductance` | Evaluated; not used. | Its layer sums did not equal the readout difference on the tested layers, where layer integrated gradients is exact. |
| `NeuronConductance` | Evaluated; not used. | The conductance of one latent coordinate is the input attribution of that coordinate's readout, which the readout registry already gives. The heads return tuples that the selector cannot index. |

### Where layer attribution is applied

In attentive models, layer attribution uses the per-head fusion modules of the head-structured posterior. Under `source_null`, the target state is fixed and the source-driven change passes through those modules, which gives a complete per-head split.

In the lag-residual model, layer attribution uses the proposal head's output across both the mean and scale proposal channels. It attributes through the following sum and limiter. Mean proposals alone would omit the scale contribution to divergence.

The layer outputs must stay connected to the readout's graph. Two other layer choices do not meet this condition:

- The posterior head's inputs include a raw prior log-variance that is unused under `posterior_logvar_mode: independent`.
- The lag-attention module has a discarded first output, including the unused `W_o` projection.

This implementation also needs the layer to be called once per forward. The residual model normally calls its proposal head repeatedly in chunks. Attribution runs it unchunked and restores the original setting in a `finally` block.

## 10. What to check on a trained checkpoint

Read `results.attribution` for attentive models, or `attribution` for the lag-residual model, together with the saved tables.

1. Read `checks.completeness_rel_median`, `checks.completeness_rel_max` and `n_rows_over_tolerance`.
2. Look at `attribution_checks.pdf`, which draws every row's residual against the tolerance beside the entry jumps.
3. Decide whether the integration step count and entry fraction are adequate for the trained weights. Check the `all_zero` `pred_gap` rows in particular.
4. In `attribution_null.csv`, compare `value_baseline_mean`, `value_input_mean` and `entry_jump_mean` by class.
5. Inspect the `all_zero` target/source split separately.
6. In `attribution_summary.csv`, read `lag_corr_mean` and `lag_js_mean` for `kld` and `pred_gap`.
7. In `attribution_lag_bands.csv`, compare IG sums, ablation changes and occlusion changes with their sign conventions.
8. In `attribution_bands.csv`, inspect the source-band attribution to `pred_gap`. Target-band `spectral_skill` is a predictive measurement, not a substitute for attribution.
9. In `attribution_layer.csv`, inspect the per-head or per-lag split and any opposing contributions.
10. Record the `cost` values (including `trace_elapsed_s`), the selected recording counts, the cap and the trace coverage before you choose a larger run.
11. Confirm that `plan.source_channel_shift_steps_max` is zero. If it is not, the stored-step lags are offset from the attention's lags (section 5), so do not read the lag figures against the occlusion bands.

## 11. Interpretation limits

- **Attribution describes a fitted computation.** A large value means that the selected output responded to an input along the chosen path. It does not show that uterine activity caused a heart-rate change. Models with the same predictions can have different attributions.
- **Lag positions are stored-coefficient time.** Each coefficient already combines raw history through a one-sided transform. Its group delay can be of the order of the lag search window, so a peak is not a physiological delay estimate.
- **Source-null attribution is relative to the zero-source response.** The availability indicators are fixed along the path and receive no input attribution. Read the baseline value and entry jump beside the IG sum.
- **The entry jump is separate from the map.** It accounts for the excluded part of the path and belongs to no input step or channel.
- **The all-zero split depends on the path.** It describes target and source streams that change together from that reference. It is not a unique division of the target and source contributions.
- **Lag readouts differ by architecture.** Attentive models use attention mass; the lag-residual model uses proposal norms before summation and limiting. A proposal norm is neither a lag distribution nor a KL allocation. Zero-sum changes to proposals can keep the predictions and change the lag-wise attribution.
- **Class comparisons depend on the training population.** When the checkpoint trained on healthy recordings only, the other clinical groups are out of distribution. Check the run's cohort provenance before you generalise a class contrast.
- **The main-pass summaries are descriptive and use recordings as their unit.** The main draw has a few segments per recording and a few anchors per segment, and it has no group hypothesis test. Sections 12 and 13 add the tests.
- **Frequency-band attribution is not frequency-band skill.** It sums over declared input channels, including dropped channels at zero. `spectral_skill` measures predictive performance on retained target channels.
- **Example maps show individual anchors.** Use the recording-level summaries to see whether a pattern is common in the evaluated selection.
- **Completeness is relative to the entry point.** A row above tolerance does not accurately account for its integrated output difference. Inspect the per-row residuals and summary counts before you interpret it.

## 12. Class comparison and Grad-CAM

The example pages show a few anchors per class. A few anchors cannot separate a class pattern from a recording pattern, so the class comparison has its own, larger draw. `class_contrast.py` reads that draw, and `gradcam.py` supplies a second, cheaper attribution method.

### 12.1 The cohort pass

`attribution_pass.run_cohort` draws segments with the same rule as the main pass, with one segment per recording: class-balanced, spread over subgroups and seeded. Its cap is `caps.attribution_cohort_segments` (`DEFAULT_COHORT_SEGMENTS` $=150$). Each segment gets only what the comparison reads:

- integrated gradients of `kld` and `pred_gap` (`COHORT_READOUTS`) under `source_null`, at `ANCHORS_PER_SEGMENT` anchors;
- Grad-CAM of the four main readouts at the same anchors.

The cohort pass uses the same high-KL clean anchors as the main pass (section 7). It has no layer split, ablation, lag-band readout, horizon readout, `all_zero` path or example page. The block's `class_contrast.cost` records the measured rate of each run.

### 12.2 Grad-CAM

For a layer with time-major activations $A\in\mathbb R^{T\times D}$ and a readout $f$ at anchor $t_a$, Grad-CAM gives each feature channel one weight and keeps the positive part of the weighted sum:

$$
w_d=\frac{1}{|\mathcal R|}\sum_{t\in\mathcal R}\frac{\partial f}{\partial A_{t,d}},\qquad
\mathrm{CAM}_t=\mathrm{ReLU}\Bigl(\sum_d w_d\,A_{t,d}\Bigr).
$$

$\mathcal R$ is the set of steps with a nonzero gradient. On these causal models, that set never passes the anchor. Each map is indexed by offset $\ell=t_a-t$ and divided by its sum, so a class mean shows where the readout looked. The unnormalised sum is kept as `<view>_total`.

| View | Layer | Why this layer |
| --- | --- | --- |
| `target` | The input of `target_encoder.attention_blocks[-1]`; the encoder input on the conv-LSTM cell | The prior and posterior heads are per-step, so the readout at $t_a$ reads the encoder's output only at $t_a$, and a map there is a spike at lag $0$. The last attention block's input is the deepest target layer whose earlier steps the readout still reads. |
| `source` | The forward's `source_state`, the K/V stream of the lag attention | The lag attention reads this stream at every lag. |
| `attention` | The output of `lag_attn.attn_dropout`, flipped into lag order | $\mathrm{CAM}_\ell=\frac1M\sum_m\mathrm{ReLU}(\alpha^{(m)}_{t_a,\ell}\,\partial f/\partial\alpha^{(m)}_{t_a,\ell})$, which is gradient-weighted attention. The forward's `attn_weights` is a flipped copy of these weights that no readout depends on, so it carries no gradient. |

The ReLU keeps only evidence that raises $f$. A peak therefore means more divergence for `kld`, more advantage of the full branch for `pred_gap`, and a worse score for `nll_full` and `mse_full`. Grad-CAM needs one forward and one backward per readout. It has no baseline and no completeness property, so read it beside the integrated gradients. Captum's `LayerGradCam` is not used, because it pools gradients over every axis after the second and so assumes channel-first maps.

The tests in `lag_attn_transformer_cfs/tests/test_eval_class_attribution.py` check three exact facts on the tiny transformer:

- every view sums to one;
- a target-only readout (`nll_base`) gives zero source and attention evidence;
- for a `lag_band` readout, the attention view equals the forward's own head-mean attention, restricted to the band and renormalised. This pins the hook and the lag order.

### 12.3 How the classes are compared

Every comparison uses the same four steps:

1. Average the anchors of each recording, so that one recording is one unit.
2. Per class, take the mean over recordings, with a $95\%$ percentile bootstrap band over recordings (`BOOTSTRAP_RESAMPLES` $=1000$).
3. Per class pair, take the difference of means, more severe minus less severe, with a band that resamples each class independently.
4. Test each per-recording scalar across classes with Kruskal–Wallis, Holm-adjusted within one family. Then test every pair with Mann–Whitney $U$ and Cliff's $\delta$, Holm-adjusted over the metric's pairs.

A family is one method and one readout, for example `ig:kld` or `gradcam:pred_gap`. The tested metrics are:

| Family | Metrics per recording |
| --- | --- |
| `level:kld` | `level_kld_at_anchor`: the $K_t$ of the recording's attributed anchors. |
| `ig:<readout>` | `source_total` (signed), `source_abs_total`, `lag_centroid_s` (the centroid of the unsigned lag profile), `lagband_<band>` for every configured band, and `lagshare_<band>`, the band's share of the unsigned lag profile. |
| `gradcam:<readout>` | `<view>_centroid_s` and `<view>_total` for each of the three views. |

**The anchors are not at one coupling level.** The cohort takes each segment's own top $30\%$ of $K_t$ (section 7), so a healthy high-KL anchor can be below an HIE low-KL anchor. The integrated gradients of `kld` sum to approximately $K_t - K_{\mathrm{null}}$, so a class difference in `source_total`, `source_abs_total` or `lagband_<band>` is largely a class difference in $K_t$. The `level:kld` row shows how far apart the levels are. The scale-free metrics (`lag_centroid_s`, `lagshare_<band>` and the Grad-CAM centroids) do not scale with $K_t$, and `attribution_classes.pdf` draws the unsigned lag share for this reason. Section 13 adds a comparison at one matched $K_t$ level.

A positive Cliff's $\delta$ means that the more severe class is higher. The statistics come from the shared `teb_vae.lag_attn.eval.stats` module, so they mean the same as in every other class table of the pipeline.

### 12.4 Reading the comparison

- **Read the difference panels first.** A band that excludes zero marks offsets where two classes differ. A band that includes zero means that the cohort cannot separate them there.
- **The rug is pointwise.** Each mark is a $95\%$ band at one lag, with no adjustment over lags, so about $5\%$ of lags carry a mark by chance. Read runs of adjacent marks, and confirm them in the tables.
- **Compare shares and centroids before totals.** Totals carry the class's $K_t$ level; see the note under the metric table.
- **Bands and tests are over recordings.** The cohort draws one segment per recording, so the sample size is the recording count in each legend.
- **The classes can be out of distribution.** When the checkpoint trained on healthy recordings only, a class difference describes how the fitted model responds to ACIDOSIS and HIE input, not a physiological mechanism.
- **The difference maps have no per-cell test.** Treat an isolated cell as noise, and check a coherent band of channels or lags in the tables.
- **Grad-CAM shares compare location, not size.** Compare sizes with `<view>_total` in the stats table.

## 13. Late-window strata: high against low $K_t$, and contractions

`attribution_strata.py` asks two paired questions in the last `LATE_WINDOW_HOURS` $=2$ hours before delivery, and it compares the classes at one matched coupling level. The cohort pass of section 12 cannot do either, because it attributes only each segment's high-KL anchors, from the middle segment of each recording.

### 13.1 The strata

Each anchor of a late segment can fall in up to three strata:

| Stratum | Rule |
| --- | --- |
| `kl_high` | $K_t$ at or above the $0.7$ quantile of its own segment's $K_t$ (`KL_HIGH_QUANTILE`). |
| `kl_low` | $K_t$ at or below the $0.3$ quantile of its own segment's $K_t$ (`KL_LOW_QUANTILE`). |
| `ua_contraction` | $0 \le a < w$, with $a$ = `seconds_since_contraction` and $w$ = `event_lag_window_s`, derived from the checkpoint's training contraction reach ([EVAL.md](EVAL.md#configuration-reference)). |
| `ua_quiet` | $a \ge w$, or no onset behind the anchor, in a segment that holds at least one `ua_contraction` anchor. |
| `kl_matched` | $K_t$ inside one band of the pooled late-window $K_t$ distribution: the same nats for every class. |

**Tiers within a recording, one level between classes.** The two KL tiers are cut per segment. That suits the paired contrast, because both strata come from the same recording. It does not suit a class comparison, because one tier can mean different nats in different classes. The `kl_matched` stratum is the class comparison:

1. Pool the clean late anchors of every class.
2. Cut them into `MATCHED_BANDS` $=5$ equal-count bands of $K_t$.
3. For each band, count each class's recordings with an anchor in it.
4. Choose the band whose smallest class count is largest.
5. Break a tie by the narrowest band in nats, because the top band of a skewed $K_t$ distribution can span a wide range.

The summary holds the band edges and the counts under `strata.rule.matched`.

The contraction age comes from the collection pass, which runs the contraction detector of `events.py` on the raw UP trace. The `ua_quiet` rule needs a contraction in the same segment, so a flat or missing UP trace does not count as "between contractions". Every anchor must also reach the clean forecast coverage of section 7.

### 13.2 Selection and what is attributed

1. Keep the segments that start at most $2$ h before delivery and that hold both KL strata.
2. Of each recording's kept segments, keep only those that hold the most strata.
3. Draw recordings class-balanced and seeded, `SEGMENTS_PER_RECORDING` $=1$ late segment each, up to `caps.attribution_strata_segments` (`DEFAULT_STRATA_SEGMENTS` $=72$).
4. Per stratum, choose up to `ANCHORS_PER_SEGMENT` anchors per segment, one horizon apart, in a seeded random order inside the stratum.
5. Attribute each anchor's `kld` and `pred_gap` with integrated gradients under both baselines.
6. Attribute the same readouts with Grad-CAM in the three views of section 12.2.

The source lag profile is read under `source_null`. The target lag profile and the source share $\sum|a_{\mathrm{src}}| / (\sum|a_{\mathrm{src}}| + \sum|a_{\mathrm{tgt}}|)$ are read under `all_zero`, because `source_null` holds the target streams at the input and its target attribution is exactly zero. Each row also carries the collection pass's $K_t$, forecast gain and contraction age at the same anchor (`anchor_kld_per_t`, `anchor_gain`, `anchor_contraction_age_s`).

Each segment is attributed once per stratum (five strata), with two baselines. The summary records the cost under `attribution.strata`.

### 13.3 How the strata are compared

One recording is one unit. The anchors of each recording are averaged per stratum. A contrast (`kl`: `kl_high` $-$ `kl_low`; `ua`: `ua_contraction` $-$ `ua_quiet`) is the paired difference over the recordings that hold both strata.

- **Within a class:** the mean paired difference with a $95\%$ percentile bootstrap interval, and a Wilcoxon signed-rank test. Holm adjusts within one contrast, one class, and one method and readout.
- **Across classes:** Kruskal–Wallis on the paired differences, Holm-adjusted within one contrast and one method and readout. A small $p$ means that the effect of the stratum differs between classes.
- **At matched coupling:** the `kl_matched` stratum goes through the cohort's class tests (section 12.3): Kruskal–Wallis across classes, Holm within one method and readout, then Mann–Whitney $U$ and Cliff's $\delta$ per pair. Its `anchor:kld_per_t` row checks the match and must not differ. Each `anchor:` metric is its own Holm family, so no readout moves the check's $p$. A large $p$ does not prove equal levels, so `strata_matched_pairwise.csv` gives the class difference of $K_t$ with its bootstrap interval, to read against `width_nats`. The summary records the band, its width, the drawn recordings per class in it and the check's $p$ under `strata.rule.matched`. It warns when the band collapses to one value.

**Some results are built in by the tiers.** In the `kl` contrast, `anchor:kld_per_t` and the IG totals of `kld` separate the tiers by construction. `strata_stats.csv` flags them `by_construction`, and the summary's `significant` list leaves them out.

`strata_stats.csv` holds both kinds of paired row (`test` = `wilcoxon` or `kruskal-wallis`; `class` = `across_classes` for the second). `strata_matched_stats.csv` and `strata_matched_pairwise.csv` hold the matched tests. `strata_recordings.csv` holds the per-recording, per-stratum means that every test reads.

### 13.4 The forward-only summary

`late_behaviour.pdf` uses no gradient and every scored anchor of the late window. It plots $K_t$ and the forecast gain against minutes before delivery and against seconds since contraction onset. It also plots the paired high-minus-low difference of the attention and of the normalised $K_t$ lag map. It reads the per-anchor table and its lag-map sidecar, so it is cheap, and it covers every recording, not only the drawn ones.

### 13.5 Files and figures

| File | Contents |
| --- | --- |
| `strata_rows.csv`, `strata_vectors.npz` | IG rows with `stratum` and the joined anchor values; the source and target lag profiles. |
| `strata_gradcam_rows.csv`, `strata_gradcam_vectors.npz` | Grad-CAM rows with `stratum`; the three view profiles. |
| `strata_recordings.csv` | One row per recording and stratum. |
| `strata_stats.csv` | The paired and across-class tests. |
| `strata_matched_stats.csv`, `strata_matched_pairwise.csv` | The class tests at matched $K_t$. |
| `late_behaviour.pdf` | Section 13.4. |
| `strata_kl_ig.pdf`, `strata_ua_ig.pdf` | IG profiles per class and stratum, and the paired difference by class. |
| `strata_kl_gradcam.pdf`, `strata_ua_gradcam.pdf` | The same for Grad-CAM. |
| `strata_paired.pdf` | Per-recording A-to-B lines on six scalar metrics, with the tests. |
| `strata_matched.pdf` | The classes at one pooled $K_t$ band: IG and Grad-CAM shares, and the scalar tests. |
| `strata_onset_locked.pdf` | Attribution shares indexed on the most recent contraction onset, per class, against an onset-shuffle null. |

**Onset-locked attribution.** A contraction can be anywhere in the lag window, so a mean lag profile over anchors blurs it. `strata_onset_locked.pdf` indexes each anchor's profile on its own most recent onset. With $a$ = `anchor_contraction_age_s` and lag $\ell$, the source step is $x = a - 4\ell$ seconds after the onset. The age runs from the anchor's causal endpoint, and the lag counts stored steps back from the anchor, so both are on the stored timeline. The null shuffles the ages among one class's anchors, which keeps the profile shapes and breaks only their link to the onset. A peak above the null at $x \approx 0$ is attribution locked to the contraction.

[FIGURE_GUIDE.md](FIGURE_GUIDE.md) explains how to read each figure. The summary block holds the rule, the selection, the plan, the cost and the significant results under `attribution.strata`.

### 13.6 Reading the strata

- **The KL contrast is partly circular.** The strata are cut on $K_t$, so any readout that grows with $K_t$ separates them. Read the shares and centroids, which are scale-free, before totals. The $K_t$ column of `strata_paired.pdf` is a manipulation check.
- **Ask class questions of the matched stratum.** The tier figures print each class's median $K_t$ per tier in their column titles, but a class difference read at a tier still carries the level gap. Read `strata_matched.pdf`, and check its $K_t$ violin first: if $K_t$ differs there, the match failed for this cohort.
- **The classes can be out of distribution.** As in section 12.4, a class difference describes the fitted model's response, not a mechanism.
