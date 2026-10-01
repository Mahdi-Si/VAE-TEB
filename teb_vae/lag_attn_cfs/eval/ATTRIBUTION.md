# Understanding input attribution

This guide explains how the `attribution` analysis relates a model output to the input coefficients that influenced it. It covers the lag-attention forecasters, including the conv-LSTM and transformer variants, and the lag-slot transformer in `lag_slot_transformer_cfs`. The latter is also called the **lag-residual model** because it combines per-lag proposals into a bounded latent update.

Start with sections 1–4 to understand the method and read its results. Sections 5–7 explain the saved files, selection, and cost. Sections 8–10 provide implementation details and recorded fixture findings. Section 11 lists what to check on a trained model.

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
- [9. Captum methods and their limitations](#9-captum-methods-and-their-limitations)
- [10. Recorded fixture findings](#10-recorded-fixture-findings)
- [11. What to record from a production run](#11-what-to-record-from-a-production-run)
- [12. Interpretation limits](#12-interpretation-limits)

## 1. What attribution tells you

Attribution asks which parts of the input contributed to a particular output of the fitted model. For example, at one forecast anchor, which uterine-pressure coefficients contributed to a change in the predictive gap? The result is a map with one value for each stored time step and input channel.

An **anchor** is the time from which a forecast starts. A **coefficient** is a numerical feature produced by the upstream signal transform. A **channel** follows one such feature through time. A **readout** is the single model quantity being explained, such as a score or KL divergence.

| Term | Meaning in this guide |
| --- | --- |
| Target streams | The two feature streams being forecast: scattering features $y^{st}$ and phase-harmonic features $y^{ph}$. |
| Source stream | The additional input $u$ used by the source-conditioned branch. |
| Base and full | The target-only and source-conditioned forecast branches. |
| Latent state | The model's compact, uncertain representation of the input history. |
| Prior and posterior | The target-only and source-conditioned latent distributions, respectively. |
| KL divergence, $K_t$ | How much the posterior differs from the prior at anchor $t$. A large value does not by itself mean a better forecast. |
| Predictive gap | Base score minus full score. A positive value means the full branch scored better. |
| Baseline | A reference input used to define the change that attribution explains. |
| Gradient | How sensitive the chosen output is to a small change in an input value. |
| Integrated gradients, or IG | Attribution obtained by accumulating gradients along a path from a reference input to the observed input. |
| Completeness | The requirement that the summed attributions reproduce the output change along that path, within numerical tolerance. |
| Lag band | An inclusive range of stored steps before the anchor. |

### How this differs from the other analyses

The existing lag profiles describe the model's behaviour with the source present. In the attentive models, these include attention weights and attention-weighted KL attribution:

$$
\widetilde K_{t,\ell}=\sum_m K_t^{(m)}\alpha_{t,\ell}^{(m)}.
$$

Here, $m$ identifies an attention head and $\ell$ identifies a lag. In the lag-residual model, a profile may instead show the proposal norm $\lVert r^\mu_{t,\ell}\rVert_2$, the magnitude proposed by that lag before summation and limiting. A proposal norm is not a probability distribution.

The `occlusion` analysis removes source values from a lag band and measures the resulting forecast change. Input attribution provides a finer view: it assigns an output change to individual coefficients across time, channels, and streams. These analyses describe different aspects of the fitted computation and should be read together.

The availability pattern also matters. It indicates which source channels are usable at each step and is determined by time rather than by the source values. The attentive model can respond to that pattern even with a zeroed source. The baseline comparison in section 3 explains how attribution handles this response.

The implementation uses Captum, pinned as `captum==0.9.0` in `requirements.txt`, for path integration, interpolation batching, and completeness calculations.

## 2. Which outputs are attributed

The wrapper exposes the following readouts. The symbols $\mu^p$ and $\mu^q$ are prior and posterior means; $\ell^p$ and $\ell^q$ are their log-variances. The coordinate index $d$ identifies one latent dimension.

| Readout | Quantity explained |
| --- | --- |
| `kld` | Divergence $K_t$ at the chosen anchor, taken from `kld_per_t` or `kld_per_anchor`. |
| `kld_dim` | The divergence contribution $K_{t,d}$ of one latent coordinate. |
| `mu_post_dim`, `mu_prior_dim` | One posterior or prior mean coordinate, $\mu^q_{t,d}$ or $\mu^p_{t,d}$. |
| `nll_full`, `nll_base` | The masked forecast-block score after decoding that branch at its latent mean, under the model's own forecast density (see below). |
| `pred_gap` | The mean-decoded difference $D_{\mathrm{base}}-D_{\mathrm{full}}$, reported as `mean_pred_gap` by the collection pass. |
| `lag_band` | Attention mass averaged over heads within a band for attentive models, or the sum of proposal norms within that band for the lag-residual model. |
| `nll_horizon` | The full-branch block score at one horizon step $\tau$: summed over the channels still scored at that step only, so the per-step scores sum to `nll_full`. Attributed at the first and the last step (once, when the block has one step); a row's `band` column carries the step as `h<step>`. |
| `mse_full`, `mse_gap` | The forecast **fidelity**: the masked squared error of the mean-decoded full forecast, and its base-minus-full gap. Scored under `'mse'` whatever the objective's likelihood, so the learned variance cannot trade against it. |

### Block scores use the model's own forecast density

Every block-score readout is scored with the two density terms `metrics.forecast_likelihood_terms(model)` returns, detached, from the model's `forecast_likelihood_kwargs()`: the per-channel scored-cell mask $m_{\tau,c}=\mathbb 1[\tau<H_c]$ and the AR(1) coefficient $\phi_c$ of the forecast residual along the horizon. With $\phi_c$, each cell is scored on its innovation $e_\tau=r_\tau-\phi_c\,r_{\tau-1}$ with $r=x-\mu$, $r_{-1}=0$, and the recursion restarting after a masked step. This is the same density the collection pass scores `mean_*` under, so an attributed `nll_full` is that column's number at the anchor. A model that builds neither term scores every cell factorised, exactly as before.

Two consequences follow. `nll_horizon` at the last step $\tau=H-1$ sums only the channels with $H_c=H$; at the first step it sums every scored channel. The fidelity readouts `mse_full` and `mse_gap` keep the cell mask but drop $\phi_c$ (`ar_coef = None`), because the error of the mean forecast has no innovation.

### The attribution uses latent means

Forecast-score readouts decode the latent mean, rather than a sampled latent state. The other readouts also use deterministic model outputs. The forward runs under `attributions.attributed_forward` (section 8), which replaces the reparameterisation by $(\mu^p,\mu^q)$, so no $\epsilon$ is drawn at all. This removes latent sampling noise from the attributed quantity and makes repeated calls comparable under the same execution conditions.

The Monte Carlo predictive score `mc_pred_gap` is not attributed here. Attributing it would require an additional convention for fixing or integrating over random draws. The mean-decoded gap matches the comparison displayed by the relevant coupling figures and preferred by the `lag_high_kl` usefulness analysis.

### What a standard pass computes

- The main readouts `kld`, `pred_gap`, `nll_full` and `mse_full` under both baselines.
- `nll_horizon` at the first and the last horizon step under both baselines.
- `lag_band` for each configured `occlusion_bands` band under `source_null`.
- `kld_dim` for the coordinate with the largest divergence at each selected anchor, under `source_null`.
- Layer attribution for every main readout under `source_null`, split by attention head or lag slot.
- Source-band ablation for every main readout under `source_null`.
- Target-only readouts on one segment per run to verify that source attribution is zero.
- One **example anchor per class**, attributed for `kld`, `pred_gap`, `nll_full`, `mse_full`, `mse_gap`, `nll_horizon` at both steps and `lag_band` on every configured band under both baselines, with the full maps, the input streams, the raw FHR and UP of its segment, the latent at the anchor and every readout's layer split kept for the map pages. The example anchor is one of the main rows, so its maps are the main calls' own rows at that anchor, kept as they are made; only the variants no main call takes are integrated again for it.
- The per-recording **trace** readouts, `kld` and `pred_gap`, at a few anchors of every segment of one recording per class under `source_null`.

The lag bands are the validated `eval_config.occlusion_bands`. Besides explicit inclusive `[lo, hi]` pairs, that block may carry the reserved entry `partition_width`, an integer width that `config_schema.partition_lag_window` expands against the model's own lag window into contiguous bands named `lags_<lo>_<hi>`, the last absorbing the remainder, listed before the explicit bands. The runner validates the block before any analysis runs and hands the validated copy to this pass, so width-form bands arrive already expanded and the pass sees only pairs.

### Read the sign before interpreting magnitude

A positive signed attribution contributes to increasing the selected readout along the integration path. Its practical meaning depends on the readout. Increasing `pred_gap` means a better score for full relative to base; increasing an NLL means a worse predictive score; increasing `kld` means a larger latent change.

Plots of absolute attribution, such as $|q_\ell|$, show magnitude and discard the sign. They identify where the model was sensitive, but cannot tell you whether that contribution increased or decreased the readout.

## 3. Baselines and the integration path

Attribution explains a change relative to a baseline. It is therefore essential to know both the observed input and the reference it is compared with.

### Two zero-based references

The loader standardises each channel, so zero represents its population mean under those statistics. Zero does not mean that an observed signal had no physical activity. The model also gates unavailable steps to zero, so replacing their stored values with zero does not change what the model reads.

| Baseline | What changes along the path | What the attribution describes |
| --- | --- | --- |
| `source_null` | Source values move from zero toward their observed values; both target streams stay fixed. | The response to source content relative to the model's zero-source response. Target attribution is exactly zero because target inputs do not move. |
| `all_zero` | All three streams move from zero toward their observed values. | The output change as target and source inputs are introduced together. The split between streams depends on this path. |

Under `source_null`, the availability indicators remain identical at both ends of the path. They are not varying input coefficients, so they receive no input attribution. The exact zero-source readout is saved separately. It includes the model's availability-related response and its behaviour at zero input; it should not be interpreted as a purified measurement of the clock alone.

### Why integration starts slightly above zero

Normalisation can make the model extremely sensitive near an all-zero input. The conv-LSTM uses per-step `CausalGroupNorm`, and the transformer variants use token-wise `RMSNorm`. Near zero variance, the normalisation scale can become very large. The recorded tiny-model experiments found very large gradients and sharp readout changes close to zero; increasing the integration step count did not reliably fix integration from the exact baseline.

The code's module docstring states the degeneracy for every encoder in the family. That holds for every target encoder and for the attentive models' source paths, but the recorded lag-residual source-null path is an exception: its default `pointwise` source stem has no parameters and no temporal normalisation, and the fixture response there was smooth. Its all-zero path still showed strong sensitivity through the target encoder. The source-null measurement covers the default stem only; comparator arms that route the source through a convolution stem or a normalised key-value side were not measured. The pass applies the same entry point to every path regardless. Section 10 records the measurements for each model.

To avoid integrating through the most sensitive region, the shared implementation starts slightly toward the observed input:

$$
x_0=b+\alpha_0(x-b),\qquad \alpha_0=10^{-3}.
$$

Here, $b$ is the exact baseline, $x$ is the observed input, and $x_0$ is the integration entry point. The constant is `BASELINE_ENTRY_FRACTION`. Integrated gradients explain the change from $x_0$ to $x$, rather than the entire change from $b$ to $x$.

For an input element $i$, the corresponding attribution has the form:

$$
\mathrm{IG}_i=(x_i-x_{0,i})\int_0^1\frac{\partial f\bigl(x_0+s(x-x_0)\bigr)}{\partial x_i}\,ds.
$$

Captum approximates this integral numerically using `IG_STEPS`, currently $64$. More integration steps can improve numerical accuracy, but the saved residual is what determines whether the approximation was adequate for a particular row.

### The entry jump is saved separately

Each row records `value_baseline` $=f(b)$, `value_entry` $=f(x_0)$, and `value_input` $=f(x)$. The **entry jump** is:

$$
\mathrm{entry\_jump}=f(x_0)-f(b).
$$

Together, these values account for the input readout approximately as:

$$
f(x)\approx f(b)+\mathrm{entry\_jump}+\sum_i\mathrm{IG}_i.
$$

The entry jump belongs to no individual input element in the attribution map. A large jump means that the chosen baseline lies in a region where the model changes sharply. It is information about the baseline comparison and model normalisation, rather than a source attribution that should be distributed over time.

## 4. Checks to read before interpreting a map

The tests in [the attentive model's attribution suite](../tests/test_eval_attribution.py) and [the lag-residual model's suite](../../lag_slot_transformer_cfs/tests/test_eval_attribution.py) exercise four properties on small models. The evaluation also records diagnostics in the attribution block's `checks`.

### 4.1 No attribution after the anchor

Input-gradient attribution to stored steps after the forecast anchor must be exactly zero. This checks that the attributed output does not use future input values.

For the conv-LSTM, this requires `causal_norm: true`: ordinary group normalisation pools over time and can expose future information. The recorded tiny-model comparison without causal normalisation produced $0.26$–$0.30$ nats of attribution after the anchor. The transformer variants use step-wise causal computation.

This is a gradient check. A separate value-level intervention that ablates only future positions should change the readout by no more than floating-point rerun noise. Do not confuse that future-only intervention with the broader `rest` ablation group described in section 5.

### 4.2 No attribution to unavailable source inputs

A source coefficient excluded by the warm-up gate must have zero attribution. Multiplication by zero also makes its gradient zero.

For an aligned channel, the encoder reads stored step $t-d_c$ and masks it while $t<W'_c+d_c$. In original stored coordinates, that channel therefore becomes usable at $W'_c$. The helper `warm_from_step` computes this threshold for each declared channel. Dropped channels never become usable.

### 4.3 No source attribution for target-only outputs

The prior mean and the base block score must have exactly zero source attribution. The attentive model's prior clock is derived from a detached zero-source encoding; the lag-residual model's metadata clock is a function of position. Neither gives these target-only readouts a gradient path to observed source values.

### 4.4 Attributions reproduce the integrated output change

The sum of integrated gradients should reproduce $f(x)-f(x_0)$. The relative completeness residual scales the absolute discrepancy by the larger of $|f(x)-f(x_0)|$ and $|f(x)|$. This avoids dividing a small numerical error by an almost-zero output change.

The block's `checks` record the median residual, maximum residual, and number of rows above `COMPLETENESS_TOLERANCE`, currently $10^{-2}$, over the main readouts' rows; `attribution_summary.csv` carries the maximum and median per readout, baseline and band. Inspect these values before treating a map as an accurate decomposition.

**Known issue:** at `IG_STEPS` $=64$, the `pred_gap` rows under `all_zero` did not converge in the 2026-09-26 audit's tiny real run. Their residuals are reported per row and counted in `n_rows_over_tolerance`; the step count has not been changed to fix it. Read those maps as unconverged, and prefer the `source_null` `pred_gap` rows for the source attribution of the gap.

The conv-LSTM fixture tests allow a median below $10^{-2}$ and a maximum below $5\times10^{-2}$. The lag-residual fixture tests require a maximum below $10^{-3}$. These are test tolerances, not evidence that every trained checkpoint will achieve the same accuracy.

Together, the checks establish whether the maps follow the model's causal input support and account for the specified output difference. They do not establish a physiological cause.

## 5. How maps become profiles and summaries

Each attributed row corresponds to an anchor, readout, and baseline, with a band or coordinate where relevant. The two target-stream maps are combined along the target-channel axis. The resulting arrays have shapes $(N,T,c_y)$ for target attribution and $(N,T,c_u)$ for source attribution, where $N$ is the number of attributed rows.

Reductions are first calculated per row, then combined within each recording, then summarised across recordings. The main selection uses one segment per recording and several anchors from that segment.

| Reduction | Calculation and interpretation |
| --- | --- |
| Time profile | Sum over channels to obtain a value at each stored step, with shape $(N,T)$. |
| Channel profile | Sum over time to obtain a value for each input channel, with shape $(N,C)$. |
| Lag-aligned source profile | Reindex the source time profile relative to the anchor: $q_\ell=p^u_{t_a-\ell}$. |
| Unsigned channel profile | Sum absolute values over time, $\sum_t \lvert a_{t,c} \rvert$, so a channel whose contributions cancel over time still registers. |
| Lag-by-channel maps | Reindex the full map by offset from the anchor, $(L, C)$, and accumulate the signed and unsigned means over attributed anchors for the main readouts under both baselines. |
| Frequency-band sums | Sum input-channel attribution using the declared-channel band map. |
| Lag-band sums | Sum the lag-aligned profile within each configured inclusive lag band, clipped to the available lag window. |
| Layer split | Attribute through attention-head fusion modules or through lag-slot proposal outputs. |
| Band ablation | Zero a group of source inputs and report the resulting change in the readout. |
| Null decomposition | Report the exact baseline readout, entry jump, and integrated attribution separately. |

### Lag alignment and agreement

For $\ell=0,\ldots,L-1$, the lag profile reads the source attribution at $t_a-\ell$. Positions before the recording are `NaN`. The compensated seconds axis comes from `lag_axis.compensated_seconds_axis(L, delay_steps)`, using the collection's `lag.delay_steps` for attentive models and zero for the lag-residual model.

Every lag in this analysis is the **stored-step offset** $t-s$ from the anchor; the plan records this as `lag_definition`. That is the attention's own lag only while the source gate shifts no channel. On an aligned arm, whose gate shifts channel $c$ by $d_c$ stored steps, the attention's lag $\ell$ reads stored step $t-\ell-d_c$, so the lag profile, the lag-by-channel maps and the band ablation sit $d_c$ steps away from the attention's lags and from the occlusion bands, which are applied to the gated stream. The pass records the largest shift as `source_channel_shift_steps_max` and logs a warning when it is nonzero, but it does not correct the offset.

The profile is compared with the model's own lag readout at the same anchor: `source_kl_lag_map[t_a]` for attentive models, or proposal norms masked by `lag_valid` for the lag-residual model.

Two statistics describe agreement on their shared valid lags. `lag_corr` is Pearson correlation between $|q_\ell|$ and the model profile. `lag_js` is their Jensen–Shannon distance after normalisation to distributions, using base $2$ and a range of $[0,1]$. Smaller distance means more similar distributions. Empty or unsuitable profiles can produce `NaN`; check counts and validity before comparing these values.

### Frequency bands use the input-channel map

Frequency-band sums join through `band_channel_map.csv`, keyed by `stream`, `channel`, and `band`. Attribution has the model's **declared input width**, so it must use the declared-channel map. Decoder-side skill analyses instead use the retained target-channel map.

Dropped input channels remain on the declared axis with zero attribution because their gates exclude them. The lag-residual stage creates a missing map through `band_partition.emit_partition`, using the same shards and resolved budget. It records a skip if channel provenance is unavailable.

### Ablation has its own sign convention

Source-band ablation reports:

$$
\Delta_{\mathrm{ablation}}=f(x^{\setminus\mathrm{band}})-f(x).
$$

A positive value means the readout increased after the band was removed. This is the opposite subtraction order from an attribution describing the contribution of present input, so compare meanings before comparing signs. In particular, an increase in NLL is worse prediction, while an increase in `pred_gap` is a larger advantage for full over base.

`FeatureAblation` removes each band across all source channels relative to that row's anchor. It also removes a `rest` group containing source positions outside the configured bands. That group may contain both future positions and earlier history before the lag window. Future positions should have no effect, but earlier history can matter when the source encoder's receptive field extends beyond the searched lags. A nonzero `ablation_rest` therefore does not by itself establish future leakage.

### Read the null decomposition as a baseline comparison

Under `source_null`, $f(b)$ is the model's exact zero-source divergence at the anchor. The IG sum describes the integrated source-content response, and the entry jump accounts for the excluded start of the path. Under `all_zero`, the IG sum is split across the target and source streams. Both decompositions depend on the chosen baseline and path.

## 6. Output files and figures

All files are written under `attribution/` in the evaluation results directory. NLL, predictive-gap, and divergence attributions use nats per anchor. The fidelity readouts use squared standardised coefficient units (`squared z`). Mean-coordinate readouts use latent-coordinate units. The attentive `lag_band` readout is an attention share in $[0,1]$; the lag-residual version is a norm in latent units. Signed attributions need not lie within the range of the readout itself. The per-row, summary, frequency-band and lag-band tables carry this as a `unit` column, which is also the unit of every attribution column beside it.

Every lag axis represents stored-coefficient time. It is not a physiological delay axis.

### Tables and arrays

| File | Contents |
| --- | --- |
| `attribution_rows.csv` | One row per attributed anchor, readout, baseline, and band where applicable. Includes identity, readout values, attribution totals, numerical checks, agreement, and intervention results. |
| `attribution_vectors.npz` | Time, lag, channel, and layer profiles aligned row for row with `attribution_rows.csv`, with each row's identity restated as `row_*` arrays. |
| `attribution_maps.npz` | One example anchor per class: the input streams the encoders read (`input_target`, `input_source`), the live steps, the model's lag readout, the raw FHR and UP of its segment (`example_raw_fhr`, `example_raw_up`, with their units), the latent at the anchor, and every example readout's full maps under both baselines (`map_target`, `map_source`, one entry per example, readout, baseline and band, keyed by `map_example`, `map_readout`, `map_baseline`, `map_band`), with the readout at the input and at the exact baseline and the layer split. |
| `attribution_examples.csv` | Manifest of the example pages: identity, class, subgroup, epoch, anchor and figure path. |
| `attribution_lag_channel.npz` | The population lag-by-channel maps: for each main readout, baseline and stream, the mean over attributed anchors of the signed (`<readout>__<baseline>__<stream>__mean`) and unsigned (`__mean_abs`) attribution re-indexed by offset from the anchor, $(L, C)$, with the anchor count. |
| `attribution_recordings.csv` | One row per recording: main readout totals and lag agreement under `source_null`, cohort labels, and `n_segments`. Used for grouped figures. |
| `attribution_summary.csv` | One row per readout, baseline, and band: counts, recording-mean values, totals, and structural-check summaries. |
| `attribution_bands.csv` | One row per main readout, stream, and frequency band: `baseline`, `attribution_mean`, `n_recordings`, and target-band `spectral_skill_pred_gap_nats` where available. Each stream is read on the path along which it moves: the target under `all_zero`, the source under `source_null`. |
| `attribution_lag_bands.csv` | One row per main readout and lag band under `source_null`: bounds, IG sum, ablation change, recording count, and an occlusion comparison where available. |
| `attribution_layer.csv` | One row per readout, layer unit, and class or `pooled`: recording count and mean attribution. |
| `attribution_null.csv` | One row per readout, baseline, and class or `pooled`: recording means of input, baseline and entry values, entry jump, attributed sum, and stream totals. |
| `attribution_blocks.csv` | One row per readout, band, baseline and input block (target scattering, target phase, source scattering, source phase), for every readout except `kld_dim` and `lag_band`: recording-mean signed and unsigned sums and the unsigned share. |
| `attribution_traces.csv` | Manifest of traced recordings: identity, class, subgroup, segment and anchor counts, span, coverage, and array/figure paths. |
| `attribution_trace_anchors.csv` | Every traced anchor of every traced recording in one table: identity, time axis, and each trace readout's scalars and lag statistics, as the trace figures draw them. |
| `traces/<class>/<guid>_<subgroup>_attribution_trace.npz` | Shared trace arrays for one recording, including `attribution_lag_map` and `model_lag_map`. |

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

`attribution_vectors.npz` contains `time_profile_target` and `time_profile_source` with shape $(N,T)$; `lag_profile`, `target_lag_profile`, and `model_profile` with shape $(N,L)$; the signed `channel_profile_target` and unsigned `channel_abs_profile_target` with shape $(N,c_y)$; the signed `channel_profile_source` and unsigned `channel_abs_profile_source` with shape $(N,c_u)$; and `lag_seconds` with shape $(L,)$. `layer_per_unit` has shape $(N,M)$ for $M$ attention heads or $(N,L)$ for lag slots, with `NaN` rows where no split was taken. The `row_guid`, `row_subgroup`, `row_clinical_class`, `row_epoch`, `row_anchor`, `row_readout`, `row_baseline`, and `row_band` arrays restate each row's identity as plain strings and numbers, so the file reads without the table beside it.

`attribution_maps.npz` identifies each example with `example_guid`, `example_subgroup`, `example_clinical_class`, `example_epoch`, and `example_anchor`. Its `map_target` and `map_source` arrays have shapes $(n,T,c_y)$ and $(n,T,c_u)$, where $n$ counts **map entries**, one per example, readout, baseline, and band, not examples; `map_example` indexes each entry back to its example.

In `attribution_recordings.csv`, the metric columns follow `<readout>_source_total`, `<readout>_target_total`, `<readout>_lag_corr`, and `<readout>_lag_js` for every main readout. The summary table includes `unit`, `completeness_rel_max`, `completeness_rel_median`, `after_anchor_max_abs`, and `gated_off_max_abs` alongside counts and means.

The lag-band table records `lag_lo`, `lag_hi`, `ig_attribution_mean`, `ablation_delta_mean`, `occlusion_delta_total_nats`, `n_recordings`, and `unit`. The occlusion comparison reads the occlusion summary's `delta_total_recording_mean_nats`, the same recording-mean aggregation as the columns beside it, and falls back to its pooled `delta_total_nats` only on a summary without the recording mean. That delta is the change of the full-branch block score, so it is joined with sign $+1$ for `nll_full`, sign-flipped for `pred_gap` (the base branch reads no source), and absent for `kld` and `mse_full`. The join is a population comparison, not a row match: occlusion draws its own uniformly random anchor per segment over every evaluated segment, seeded per batch, scores it under a paired latent draw rather than at the means, and zeroes the gated source stream, while the attribution ablation zeroes the stored source at the attributed anchors. The lag-residual model's pass joins no occlusion summary. These joined columns exist only when the corresponding analysis files are available.

### Figures

| Figure | How to read it |
| --- | --- |
| `attribution_maps.pdf` | One column per class example, every time row on that segment's stored-time axis with the anchor ruled and its forecast block shaded. Top to bottom: the raw FHR and UP on the fixed CTG scales, the target and source input coefficients, the target attribution of $K_t$ under `all_zero`, the source attribution of $K_t$ under `source_null`, and a lag panel comparing the source attribution with the model's lag readout. Each map row shares one colour scale across the columns. |
| `maps/<class>_<guid>_<subgroup>_anchor<step>_attribution_maps.pdf` | One page per class example on one shared stored-time axis, laid out as the samples pages: the raw FHR and UP of the segment first, then the inputs with cold cells blanked, then for every example readout — `kld`, `pred_gap`, `nll_full`, `mse_full`, `mse_gap`, `nll_horizon` at both steps and `lag_band` per configured band — the target map (`all_zero`) above the source map (`source_null`) on symmetric-log colour scales, then the latent at the anchor, the per-head or per-lag activation split of every readout, and every readout's lag-aligned source attribution on one axis. All example pages share their colour scales (see below). |
| `attribution_lag_profile.pdf` | Normalised absolute lag attribution versus the normalised model profile, averaged over recordings, pooled and by class, with titles reporting agreement; a last row overlays every readout — divergence, forecast gap and each lag-band readout — on one lag axis, and the band readouts against their own bands. |
| `attribution_bands.pdf` | Frequency-band attribution by stream above, with target-band spectral skill on a second vertical axis where available. Lag-band IG sums, ablation changes, and occlusion comparisons appear below. Check the axes and sign conventions separately. |
| `attribution_layer.pdf` | Per-head or per-lag attribution on the left; input profiles for the anchor's highest-divergence coordinate on the right. |
| `attribution_null.pdf` | Input divergence, exact null response, integrated content attribution, and entry jump by class on the left; the `all_zero` target/source split on the right. |
| `attribution_channels.pdf` | Per main readout, the signed and unsigned attribution per declared channel of each stream, coloured by frequency band where the channel map exists. |
| `attribution_lag_channel.pdf` | Per main readout, the mean unsigned attribution by offset from the anchor and declared channel: the target stream under `all_zero`, the source stream under `source_null`. |
| `attribution_time_profile.pdf` | Per main readout and stream, the mean positive and negative parts of the attribution by offset from the anchor, with the net and unsigned means. |
| `attribution_checks.pdf` | One horizontal strip per attributed variant (readout, band and baseline), one point per row, blue on the `source_null` path and vermilion on the `all_zero` one, with the median as a bar. Left: the relative completeness residual on a log axis against the tolerance. Middle: the entry jump as a share of the readout's whole move along the path, $\lvert f(x_0)-f(b)\rvert/\lvert f(x)-f(b)\rvert$. Right: the two structural checks per readout, printed beside their bars. |
| `attribution_time_to_delivery.pdf` | The source attribution total and the lag centroid of every attributed anchor against hours before delivery, by class. |
| `attribution_blocks.pdf` | Per readout, the unsigned share and the signed sum of the attribution in each of the four input blocks. |
| `attribution_horizon.pdf` | The per-step score at the first and the last horizon step: stream totals, and the source and target attribution by offset per step. |
| `traces/<class>/<guid>_<subgroup>_attribution_trace.pdf` | The raw FHR and UP of every segment on the top rows, then the attribution lag maps of the divergence and of the forecast gap through one recording beside the model lag map, with each readout's agreement, totals and values on the hours-before-delivery axis. Every panel shares its colour scale or $y$ range across the traced recordings. |

The [figure guide](FIGURE_GUIDE.md#attributionattribution_mapspdf) provides panel-by-panel interpretation. Every figure but the checks page prints `ATTRIBUTION_NOTE`, the one-line form of `ATTRIBUTION_CAVEAT`; the long caveat, the group-delay caveat and the model-specific lag qualification travel in the record (`caveat`, `lag_qualification`). Every map blanks the cells the model never read (a channel's cold steps, and the steps after the anchor on an attribution map) and draws attributions on a symmetric-log colour scale spanning `LOG_DECADES` below the scale's largest magnitude; line and bar panels of attributions are on symmetric-log axes for the same reason.

**Shared scales.** The map overview and every example page draw on one set of colour scales built by `attributions.example_norms` over all class examples, and state it here rather than under the figure (`SHARED_SCALE_NOTE`): one linear scale per input stream (1st to 99th percentile of the read cells of all examples), one symmetric-log scale per readout, baseline, and stream at plus or minus its largest magnitude over all examples, and fixed CTG scales for the raw FHR and UP. A colour therefore means the same value on every page, and a quiet class looks quiet. The trace pages do the same through `traces.shared_panel_scales`: each heatmap and line panel is scaled once over every traced recording.

### The summary block

The block is `results.attribution` in `summary.json` for both model types, since both run it as a registered analysis under the shared runner.

It contains `n_samples` (segments), `composition`, `plan`, `selection`, `trace_selection`, `cost`, `checks`, `summary`, `lag_bands`, `blocks`, `joined`, `methods`, `lag_qualification`, `caveat`, `traces`, `examples`, `failures`, and `files`. The `plan` records the cap, seed, anchors per segment, integration steps, entry fraction, baselines, main readouts, horizon steps, example readouts, lag readout and lag bands, `lag_definition`, `source_channel_shift_steps_max`, layer, `delay_steps`, the anchor geometry the anchors are chosen on, `attributed_forward` (one anchor per row, decoded at the latent means, no $\epsilon$ drawn), and the trace settings. `joined` records which supporting files were found. Attentive evaluators also include `grouped_frames`; the lag-residual block adds its channel-map record and an ablated-input check.

The lag-residual tests check that this block avoids names its acceptance gate reserves for unsupported attention distributions or per-lag KL allocations.

## 7. Selection, settings, and cost

### Main attribution selection

The main pass uses a seeded selection balanced across clinical classes, subject to available recordings and the total cap. It selects one segment per recording: the middle segment in `epoch` order. A recording needs at least one segment to be eligible. Using one segment per recording prevents recordings with many segments from dominating the summaries.

The pass attributes up to `ANCHORS_PER_SEGMENT` anchors spread across each selected segment's scored anchors. The current value is $4$.

| Setting or constant | Current value and meaning |
| --- | --- |
| `eval_config.caps.attribution_segments` | Segment cap, currently $24$ in the committed overrides. Omitting it uses the analysis default, rather than every segment. |
| `DEFAULT_SEGMENTS` | $24$: fallback for the main selection cap. |
| `ANCHORS_PER_SEGMENT` | $4$: anchors spread over each segment's scored support. |
| `IG_STEPS` | $64$: integration steps. |
| `IG_INTERNAL_BATCH_SIZE` | $64$: Captum's internal batch of interpolated inputs, counted in rows times steps per forward, and raised to at least the call's row count. |
| `BASELINE_ENTRY_FRACTION` | $10^{-3}$: how far to move from the exact baseline before integration. |
| `COMPLETENESS_TOLERANCE` | $10^{-2}$: relative residual threshold counted by the pass. |
| `TRACE_RECORDINGS_PER_CLASS` | $1$: detailed attribution trace per clinical class. |

The cap is an evaluation setting. The other names are implementation constants, not additional `eval_config` keys. Use the saved `plan` to identify the values used by a completed run.

### Detailed trace selection

The trace selection is separate from the main random draw. For each class, it chooses the recording whose stored segments most completely cover the relevant window. If `max_hours_before_delivery` is set, that window is the specified interval before delivery; otherwise it uses the recording's own span.

`attribution_pass.recording_completeness` compares the number of stored segments inside the window with the number expected from the segment stride. Ties are broken by segment count and then recording identifier. A recording with fewer than two segments inside the window is ineligible. `trace_selection` records the selected recording's coverage.

This rule favours fewer gaps; it does not select for strong uterine activity, large KL, or large predictive gain. The trace attributes every readout in `TRACE_READOUTS`, `kld` and `pred_gap`, under `source_null` at the same selected anchors across the recording's segments, with no layer split or ablation.

### How to estimate runtime

With $n$ attributed rows in a batch ($k$ anchors per segment times its segments), each IG call processes the $n$ rows through the integration steps plus three plain forwards (input, exact baseline, entry point). The main work per batch is:

- eight IG calls for the four main readouts under both baselines;
- two IG calls per horizon step for `nll_horizon`, under both baselines;
- one `source_null` IG call per lag band, and one for the top coordinate;
- four layer attributions, one per main readout under `source_null`;
- four grouped band ablations, one per main readout, each costing the reference plus one intervention per band and `rest`.

Under `source_null` the IG calls and the ablations hand Captum the source stream alone and carry the target streams as fixed arguments, so their backward never enters the target encoder.

The example anchors add little, once per class, because their maps are the main calls' own rows at that anchor. Only the variants no main call takes are integrated again as single-row calls: `mse_gap` under both baselines and `lag_band` under `all_zero` per band, plus the layer split of every variant that is not a main readout (`mse_gap`, `nll_horizon` at each step, and `lag_band` per band). The trace attributes two readouts at each of its anchors.

The attributed forward decodes one anchor per row rather than the whole dense anchor axis (section 8), which made each call about $20\times$ faster on the 2026-09-26 audit measurement. The older CPU fixture benchmark of about $0.37$ seconds per row at $32$ integration steps predates that change and no longer describes the pass. Doubling the step count still roughly doubles the integration part of the work; runtime on a trained checkpoint depends on model size, device, batching, and trace length.

Use the completed pass's `cost` values: `elapsed_s`, `n_rows`, `n_forward_equivalents`, `seconds_per_row`, `hours_per_1000_samples`, and, on CUDA, `peak_allocated_bytes`. They time the drawn segments only; the traces are timed apart as `trace_elapsed_s` over `trace_n_anchors` anchors. Choose a practical segment cap from those measured rates and inspect completeness before reducing the integration step count.

## 8. Implementation: one output per anchor

`attributions.AnchorReadout` is an `nn.Module` wrapper around the real model. It accepts the three input streams, two extra tensors used for scoring, and two per-row integer tensors:

- The extra tensors are the declared-width target features and decimated validity `weight`.
- `columns` identifies each row's position on the decoded anchor axis.
- `coordinates` identifies the latent coordinate for a coordinate-specific readout.

The wrapper first converts each row's column into a stored step through the model's own `_build_anchor_index` at the dense evaluation geometry, with no forward. It then calls the model inside `attributions.attributed_forward(model, anchors)`; the lag-residual call also sets `return_proposals=True`. It selects each row's anchor and returns a vector of shape $(B,)$, giving Captum one scalar per row.

### The attributed forward decodes one anchor at the latent means

`attributed_forward` is a context manager that shadows two of the model's seams on the instance for the duration of the call, and restores them on exit whatever happens:

- **The anchor builder** `_build_anchor_index` returns each row's own anchor as a one-anchor set. The decoder is most of a dense forward's cost, and every readout reads one anchor, so decoding the whole anchor axis would spend almost every integration step on forecasts nothing reads.
- **The reparameterisation** (`_reparameterize_shared` on the attentive models, `reparameterize_shared` on the lag-residual model) returns $(\mu^p,\mu^q)$ instead of a sample. The forward's own two forecasts are therefore the mean-decoded ones the collection pass scores as `mean_*`, through the same decoder call and persistence input, and no $\epsilon$ is drawn.

This is equivalent to the dense forward for every readout here, because everything read at an anchor depends on that anchor alone: the latents are dense over $T$ or computed per anchor, and the decoder decodes each anchor independently. The tests assert the equality. What changes is cost and determinism: the forward decodes $1$ anchor rather than the whole axis, which made each call about $20\times$ faster, and an attributed readout consumes no random number. Tensors on the anchor axis, including `anchor_valid`, the decoded forecasts, and the lag-residual proposals, now have a single column, so the readouts take `[:, 0]`.

### Several anchors can share one call

`expand_rows` repeats each segment for its chosen anchors. For $B$ segments and $k$ anchors, the expanded batch has $Bk$ rows. Captum expands the per-row anchor arguments along with its interpolated inputs. Tests compare this batched calculation with one-anchor-at-a-time attribution at an absolute tolerance of $10^{-5}$.

### Targets and masks follow Captum's expanded batch

Block-score readouts rebuild the target block and mask on every wrapper call using the extra tensors and each row's anchor. They use `_build_forecast_target`, the anchored `forecast_mask` over `model.scored_weight(weight)`, the model's `coverage_floor`, and the density terms of `forecast_likelihood_terms(model)` (section 2), matching collection scoring. They score the forward's own mean-decoded forecasts `mu_full`/`mu_base` at the one decoded anchor. A target block built only once outside the wrapper could become misaligned when Captum expands the batch.

### Unused inputs return zero attribution

Some readouts intentionally do not depend on an input, such as a target-only score's lack of dependence on source values. The wrapper connects every input and both posterior parameters to the computation graph with a zero coefficient. This makes the corresponding derivative exactly zero instead of causing autograd to reject an unused tensor.

The same mechanism supports layer outputs that affect a parameter the chosen readout does not use, such as scale proposals for a mean-decoded gap.

### The binding handles different time axes

Attentive models expose dense latent tensors over stored time, with shape $(B,T,\ldots)$, so the wrapper gathers the anchor's stored step. The lag-residual model indexes its latent tensors by decoded anchor, and under the attributed forward that axis holds only the row's anchor, so the wrapper takes column $0$. `CellBinding` identifies the layout; baselines, reductions, and figures remain shared.

## 9. Captum methods and their limitations

The following decisions come from experiments on the small models used by the test suites: the conv-LSTM `make_task` and stub batch, the transformer `tiny_warmup_kwargs`, and the lag-residual `build_tiny_model` and `tiny_streams`. These are implementation-specific findings, not general rankings of attribution methods.

| Method | Status here | Reason |
| --- | --- | --- |
| `IntegratedGradients` | Primary method. | Integrating from the entry point gives much smaller completeness residuals on the recorded fixtures and supports the causal-support checks. |
| `LayerIntegratedGradients` | Used. | Produces a per-head split through attentive fusion modules, or a per-lag split through the residual proposal head, under the source-null comparison. |
| `FeatureAblation` | Used. | Removes source values in groups defined by lag band and anchor, then measures the readout change. |
| `InputXGradient`, `Saliency` | Evaluated; not included. | Pass the structural support checks but use single-point gradients and do not provide the complete path decomposition used here. |
| `GradientShap` | Evaluated; not included. | The sampled baseline/path calculation showed large per-sample residuals in these experiments and requires managing additional randomness. |
| Sliding-window `Occlusion` | Evaluated; not included. | A window crossing the anchor assigns its group effect to future positions as well as past ones. Anchor-relative grouped ablation avoids that attribution layout. |
| `DeepLift` | Unsuitable for these forwards as tested. | Its rescale hooks miss functional nonlinearities and other operations in these models; measured completeness residuals were large. |
| `LayerConductance` | Evaluated; not included. | Tested layer sums did not reproduce the readout change where layer integrated gradients did. |
| `NeuronConductance` | Evaluated; not included. | Coordinate readouts already expose the desired input attribution, and the tested selector does not handle the heads' tuple outputs. |

The `methods` record saves each decision and its reason in the output block.

### Where layer attribution is applied

In attentive models, layer attribution uses the head-structured posterior's per-head fusion modules. Under `source_null`, the target state is fixed and the source-driven change passes through those modules, allowing a complete per-head split.

In the lag-residual model, it uses the proposal head's output across both mean and scale proposal channels. This attributes through the subsequent sum and limiter. Including only mean proposals would omit the scale contribution to divergence.

Other layer choices failed in the recorded tests. Attributing the posterior head's inputs encounters an unused raw prior log-variance with `posterior_logvar_mode: independent`. Attributing the lag-attention module encounters its discarded first output, including the unused `W_o` projection. The relevant layer outputs must remain connected to the readout's graph.

The layer must also be called once per forward for this implementation. The residual model can normally call its proposal head repeatedly in chunks; attribution temporarily runs it unchunked and restores the original setting in a `finally` block.

## 10. Recorded fixture findings

These results were recorded on untrained tiny models and synthetic Gaussian-noise inputs. They describe numerical behaviour of the attribution machinery, not clinical effects or trained-model performance. The values below are retained from the existing design record; this documentation revision does not represent a new experiment.

### 10.1 Behaviour close to the zero baseline

The table follows the source-null path $b+\alpha(x-b)$ for the divergence readout at one anchor. The notation $f(\alpha)$ means the readout evaluated at that path position. Readout values are in nats; the last column is the directional derivative with respect to $\alpha$.

| Model | $f(0)$ | $f(10^{-6})$ | $f(10^{-3})$ | $f(1)$ | Derivative at $0$ |
| --- | --- | --- | --- | --- | --- |
| conv-LSTM, `causal_norm` | $2.752$ | $2.915$ | $2.865$ | $2.380$ | $1.3\times10^{22}$ |
| transformer, `entmax15` | $7.136$ | $4.539$ | $3.570$ | Not recorded. | $2.5\times10^{14}$ |
| transformer, softmax | $5.702$ | $4.258$ | $3.773$ | Not recorded. | $5.1\times10^{13}$ |
| lag-residual | $4.387$ | $4.387$ | $4.386$ | $5.036$ | $-0.96$ |

The attentive source-null paths show sharp changes near zero. The lag-residual source-null path, with the default `pointwise` source stem, is smooth in this example, but its all-zero path has a large target-side derivative, about $-1.1\times10^{18}$, associated with the target encoder's `RMSNorm`. This row is the evidence behind the exception in section 3; the blanket statement in the `attributions` module docstring is true of the all-zero paths and of the attentive source paths, not of this one.

### 10.2 Integration accuracy

Starting at the exact baseline produced relative completeness residuals of $0.23$–$0.43$ for the conv-LSTM source-null case, $0.9$–$1.0$ for transformer source-null cases, and $1.5$–$15$ for all-zero cases across the models. Increasing the step count from $32$ to $512$ did not consistently improve these results.

Starting at $\alpha_0=10^{-3}$ gave the following recorded residuals:

| Model | $64$ steps | $256$ steps |
| --- | --- | --- |
| conv-LSTM | $7\times10^{-4}$ | $4\times10^{-6}$ |
| transformer, `entmax15` | $1.3\times10^{-3}$ | $1.1\times10^{-5}$ |
| transformer, softmax | $3\times10^{-4}$ | $1.5\times10^{-6}$ |
| lag-residual | About $9\times10^{-7}$ | About $9\times10^{-7}$ |

The `pred_gap` and `lag_band` readouts showed the same qualitative improvement when the entry point moved away from exact zero. The recorded transformer `entmax15` residual at $64$ steps is slightly above $10^{-3}$, so the fixtures do not support a universal claim that every residual is below that value. These are single-readout fixture measurements: the audit's tiny real run later found the `all_zero` `pred_gap` rows unconverged at $64$ steps (section 4.4).

On the attentive transformer example, the exact-null divergence was about $7.1$ nats and the entry-point value about $3.6$ nats. This large entry jump shows why it must be reported separately. The residual model's source-null jump was of order $10^{-3}$ of the readout. A trained model's own `entry_jump_mean` is needed to assess whether either pattern persists.

### 10.3 Structural support and batching

Causality, warm-up gating, and target-only purity were exact for the tested readouts and baselines on the causal model configurations. Removing `causal_norm` from the conv-LSTM produced the $0.26$–$0.30$ nats of future attribution noted earlier. Batched-anchor attribution agreed with one-at-a-time attribution to $8\times10^{-7}$.

### 10.4 Layer and band results

On the transformer `entmax15` fixture, at one anchor (fixture value: stored step $13$), the fixture's heads contributed $[0.017,0.022,-0.188,-0.191]$ nats to the divergence change, summing to $-0.340$. Target-state and prior contributions were zero under `source_null`.

On the residual fixture, the per-lag split across both proposal channels reproduced the readout difference to $10^{-3}$. The mean channel alone did not, because divergence also depends on the scale update.

The transformer band-ablation example reduced divergence by $1.08$, $0.04$, and $0.45$ nats when zeroing the anchor, near, and far source bands, respectively. Removing `rest` reduced it by $0.009$ nats, consistent with earlier history retained by the convolution stem. The recorded target-stream ablation control changed the readout by no more than $2\times10^{-6}$.

### 10.5 End-to-end stub run

The stub loader supplied fixture values of three segments, four anchors per segment, and four tiny lag bands. The conv-LSTM pass produced $108$ rows, with zero causality, gating, and target-only check values. That row count was recorded before the fidelity, per-horizon, and extra main readouts were added; a current stub run produces more rows. The residual pass also satisfied those checks, with completeness residuals below $5\times10^{-5}$.

These fixture results validate the small test setup. They do not replace checks on a real checkpoint.

## 11. What to record from a production run

**Production findings have not yet been filled into this design record.** The checklist below specifies what to collect after running a trained checkpoint; it is not a report of measured production results.

Use `results.attribution` for attentive models or `attribution` for the lag-residual model, together with the saved tables.

1. **Numerical accuracy:** Record `checks.completeness_rel_median`, `checks.completeness_rel_max`, and `n_rows_over_tolerance`, and look at `attribution_checks.pdf`, which draws every row's residual against the tolerance beside the entry jumps. Decide whether the integration step count and entry fraction are adequate for the trained weights. Check in particular the `pred_gap` rows under `all_zero`, which did not converge at $64$ steps in the audit's tiny run; this remains an open item.
2. **Baseline behaviour:** In `attribution_null.csv`, compare `value_baseline_mean`, `value_input_mean`, and `entry_jump_mean` by class. Inspect the `all_zero` target/source split separately.
3. **Lag agreement:** Read `lag_corr_mean` and `lag_js_mean` for `kld` and `pred_gap` in `attribution_summary.csv`. These compare input sensitivity with the model's own lag profile.
4. **Band comparisons:** Compare IG sums, ablation changes, and available occlusion changes in `attribution_lag_bands.csv`, accounting for their sign conventions.
5. **Frequency bands:** Inspect source-band attribution to `pred_gap` in `attribution_bands.csv`. Target-band `spectral_skill` is a related predictive measurement, not a substitute for attribution.
6. **Layer contributions:** Inspect `attribution_layer.csv` to see the per-head or per-lag split and any opposing contributions.
7. **Runtime and coverage:** Record measured `cost` (including `trace_elapsed_s`), the selected recording counts, cap, and trace coverage before choosing a larger run. No post-speedup benchmark on a trained checkpoint has been recorded yet.
8. **Lag definition:** Confirm `plan.source_channel_shift_steps_max` is zero. If it is not, the stored-step lags are offset from the attention's lags (section 5) and the lag figures should not be read against the occlusion bands.

## 12. Interpretation limits

- **Attribution describes a fitted computation.** A large value means that the selected output responded to an input along the chosen path. It does not establish that uterine activity caused a heart-rate change. Models with the same predictions can have different attributions.
- **Lag positions are stored-coefficient time.** Each coefficient already combines raw history through a one-sided transform. Its group delay can be comparable to the lag search window, so a peak is not a physiological delay estimate.
- **Source-null attribution is relative to the zero-source response.** The availability indicators are fixed along the path and receive no input attribution. Read the baseline value and entry jump beside the IG sum.
- **The entry jump is separate from the map.** It accounts for the excluded part of the path and is assigned to no input step or channel.
- **The all-zero split depends on the path.** It describes what happens when target and source streams change together from that reference. It is not a unique division of everything the target and source contribute to the model.
- **Lag readouts differ by architecture.** Attentive models use attention mass; the lag-residual model uses proposal norms before summation and limiting. A proposal norm is neither a lag distribution nor a KL allocation. Zero-sum changes to proposals can preserve predictions while changing lag-wise attribution.
- **Class comparisons depend on the training population.** For the documented healthy-pretraining setup, unseen clinical groups are out of distribution. Check the run's cohort provenance before generalising a class contrast.
- **Summaries are descriptive and use recordings as their unit.** The main draw contains one segment per recording and only a few anchors per segment. Read recording counts and selection limits; no group hypothesis tests are performed by this attribution analysis.
- **Frequency-band attribution is not frequency-band skill.** It sums over declared input channels, including dropped channels at zero. `spectral_skill` measures predictive performance on retained target channels.
- **Example maps show individual anchors.** A striking map is one selected example. Use the recording-level summaries to understand whether its pattern is common in the evaluated selection.
- **Completeness is relative to the entry point.** A row above tolerance does not accurately account for its claimed integrated output difference. Inspect the per-row residuals and summary counts before interpreting its attribution.
