# Attributing the forecaster's readouts to its inputs

What the `attribution` analysis of the two causal-feature forecasters measures, how it is built,
which Captum methods it rests on and which it rejected, what its four structural checks prove,
what every table and figure carries, what it costs, and the ways its output invites a reading it
does not support. `EVAL.md` beside this file is the contract the analysis is bound to by test;
this is the design record and the findings. The lag-residual cell runs the same pass as a
post-pass stage and its `EVAL.md` points here.

## Table of contents

0. What is being asked, and what the pipeline already answered
1. The wrapper: one scalar per anchor
2. The readouts, and which branch is attributed
3. The two baselines, and why the path enters them from the side
4. Which Captum methods run on these forwards, and which do not
5. The four structural checks, and what they prove
6. The reductions
7. Every table and figure
8. The cost
9. Fixture findings
10. Production-run findings
11. How this output will be misread

---

## 0. What is being asked, and what the pipeline already answered

Every lag readout the pipeline emits is one of two kinds. The **observational** ones — the
attention over lags, the KL attribution $\widetilde K_{t,\ell} = \sum_m K^{(m)}_t
\alpha^{(m)}_{t,\ell}$ built from it, the proposal norm $\lVert r^\mu_{t,\ell} \rVert_2$ in the
lag-residual cell — say where a fitted distribution puts its mass, on a forward in which the
source was fully present. The **interventional** one, `occlusion`, zeroes a band of source lags
relative to one anchor per segment and reports what the forecast lost. Neither says which
**coefficients** — which stored steps, which channels, which frequency bands, of which stream —
the model's per-anchor quantities actually responded to. The availability-clock hazard
`source_null` names makes that question sharper than usual here: the source availability pattern
is a deterministic function of the step index and enters the posterior alone, so the divergence
carries a term no input coefficient produced.

Gradient attribution answers the question at the resolution the model's inputs have. A scalar
readout $f(y^{st}, y^{ph}, u)$ at one anchor is attributed back over every element of the three
declared-width input streams on the stored grid, and the maps are reduced onto the axes the rest
of the pipeline reads: stored time, offset from the anchor (the lag axis), channel, frequency
band, lag band, and — through a layer attribution — head or lag slot. The tools are Captum's,
installed for this purpose (`captum==0.9.0`; the path integration, the batching over
interpolations and the completeness deltas are not a few lines to own locally).

## 1. The wrapper: one scalar per anchor

`attributions.AnchorReadout` is an `nn.Module` whose forward takes the three input streams and
two extra tensors — the declared-width target stream the block scores are taken against and the
decimated validity `weight` — plus two per-row `long` tensors: `columns`, each row's anchor as a
position on the anchor axis, and `coordinates`, each row's latent coordinate for the
per-coordinate readouts. It calls the real model at the dense evaluation geometry
(`anchor_phase=0, anchor_stride=1`, plus `return_proposals=True` in the lag-residual cell),
selects each row's own anchor, and returns the readout as a $(B,)$ vector.

Three properties of the wrapper are load-bearing.

* **The anchor is per row, so several anchors of one segment are attributed in one call.** The
  pass repeats every segment once per chosen anchor (`expand_rows`) and hands Captum a per-row
  column tensor as an additional forward argument, which Captum expands along its interpolation
  axis exactly as it expands the inputs. A batch of $B$ segments at $k$ anchors is one call of
  $Bk$ rows; the tests assert the batched attribution equals the one-at-a-time one to $10^{-5}$.
* **The block-score readouts rebuild the target block and the mask on every call**, from the two
  extra tensors and the row's own anchor, through the model's `_build_forecast_target` and the
  anchored `forecast_mask` at the model's own `coverage_floor` — the same two functions the
  collection pass scores with. A target built once outside would no longer match a batch Captum
  has expanded.
* **Every input and both posterior parameters are tied into the graph with a zero coefficient.**
  A readout that does not depend on the source — the prior mean, the base block score — would
  otherwise make autograd raise about an unused tensor; with the tie its source attribution is an
  exact zero, which is the structural fact the tests assert. The same tie lets a layer attribution
  on a module whose output reaches only the parameter a readout does not read (the scale
  proposals under the mean-decoded gap) still find that output on the graph.

The two cells differ in how a row's anchor is read: the lag-attentive cells gather a dense
$(B, T, \cdot)$ tensor at the anchor's stored step, the lag-residual cell selects a column of a
tensor already on the anchor axis. `CellBinding` names which, and the wrapper, the baselines, the
reductions and the figures are otherwise one implementation.

## 2. The readouts, and which branch is attributed

| readout | what it is | branch |
|---|---|---|
| `kld` | $K_t$ at the anchor: `kld_per_t` gathered, or `kld_per_anchor` selected | means |
| `kld_dim` | $K_{t,d}$ at the row's coordinate | means |
| `mu_post_dim`, `mu_prior_dim` | $\mu^q_{t,d}$, $\mu^p_{t,d}$ | means |
| `nll_full`, `nll_base` | the masked block score of the branch **decoded at its mean**, at the anchor | means |
| `pred_gap` | `nll_base` − `nll_full`: the mean-decoded gap the collection pass reports as `mean_pred_gap` | means |
| `lag_band` | the model's own lag readout on an inclusive lag band: the head-averaged attention mass on it (attentive cells), the proposal norm summed over it (residual cell) | — |

**Means, never samples.** Every readout is a function of $(\mu^p, \ell^p, \mu^q, \ell^q)$ and of
the decoder applied to $\mu$, so no reparameterisation draw enters any attributed number and two
runs of one checkpoint agree bitwise. The Monte Carlo score `mc_pred_gap` is deliberately not a
readout: attributing it would need a frozen $\epsilon$ and would attribute the draw as well as the
input, and no figure could separate the two. The mean-decoded gap is the estimator the coupling
figures foreground and the one `lag_high_kl` scores usefulness by, so the attribution is of the
number those pages show.

A production pass attributes `kld` and `pred_gap` under both baselines; the lag readout on every
`occlusion_bands` band, and the anchor's largest $K_{t,d}$ coordinate, under the source-null
baseline alone; the layer split of `kld` and `pred_gap`; and the band ablation of both. The
target-only readouts are attributed on one segment per run as the purity check.

## 3. The two baselines, and why the path enters them from the side

A baseline is a modelling decision. Under the loader's z-scoring an exact zero is the channel mean
over the region the model reads — the climatology baseline `forecast` already scores against, and
the fill `occlusion` and `source_null` already use — and the input warm-up gate multiplies every
not-yet-warm step by exactly zero, so a zero baseline changes nothing on the steps the model never
read and the gate's own zeros are the baseline's zeros.

* **`source_null`** zeroes the source stream and holds both target streams fixed. It is the exact
  null arm `source_null` measures, so a source attribution under it is to source **content**: the
  availability announcement $W_m(m_t - \mathbf 1)$ is a constant of $t$, identical on both ends of
  the path, and is attributable to no input at all. The target streams do not move along the
  path, so their attribution is exactly zero, which the tests assert.
* **`all_zero`** zeroes every stream. It is the reference on which the target streams' own
  attribution is read, and it is what splits a readout between the target and the source.

**The exactly-zero stream is a degenerate point of every encoder in the family, and the path
cannot start there.** The conv-LSTM cell's `CausalGroupNorm` normalises each step over the
channels of a group; the transformer cells' `RMSNorm` normalises each token over its channels. At
an all-zero input the adapter emits a constant vector at every step past the warm-up, and
downstream a group or a token with zero variance sits on the $1/\sqrt{\epsilon}$ singularity. The
measurement, on the three tiny models along the straight path $b + \alpha(x - b)$ for the
divergence readout, is in §9: the directional derivative at $\alpha = 0$ is of order $10^{13}$ to
$10^{22}$, the readout moves by a finite amount between $\alpha = 0$ and $\alpha = 10^{-6}$, and
integrated gradients from the exact zero do not converge at $32$, $128$ or $512$ steps. From
$\alpha_0 = 10^{-3}$ the singularity is behind the start and the integral converges to a relative
residual below $10^{-3}$ at $64$ steps on every fixture.

The design therefore starts every path at $x_0 = b + \alpha_0 (x - b)$ with
$\alpha_0 =$ `BASELINE_ENTRY_FRACTION` $= 10^{-3}$, and every row records the readout at the exact
baseline, at the entry point and at the input. The **entry jump** $f(x_0) - f(b)$ is a reported
scalar rather than a hidden one, and it belongs to no input step: it is the normalisation snapping
out of its degenerate state. Completeness is measured against $f(x) - f(x_0)$, which is what the
integral is of. For the lag-residual cell's source-null baseline the source pathway is a pointwise
encoder into a small multilayer perceptron, the path is smooth from $\alpha = 0$, and the jump is
of order $10^{-3}$ of the readout; for the attentive cells it is not small on the tiny fixtures
(§9), which is a finding about where `kld_source_null` is evaluated rather than a numerical
nuisance.

## 4. Which Captum methods run on these forwards, and which do not

Every method below was run on the tiny models the test suites build (`make_task` and the stub
batch of the conv-LSTM cell, the transformer cell's `tiny_warmup_kwargs`, the lag-residual cell's
`build_tiny_model` and `tiny_streams`), with the wrapper of §1 and the entry point of §3.

| method | verdict | why |
|---|---|---|
| `IntegratedGradients` | **shipped, primary** | completeness holds to $10^{-3}$ at 64 steps from the entry point on every fixture; causality, gating and target-only purity exact |
| `LayerIntegratedGradients` | **shipped** | on the per-head fusion modules of the head-structured posterior (attentive cells): a complete per-head split under the source-null baseline, where the target state is fixed and everything that moves flows through the fusion; on the proposal head's output (residual cell): a complete per-lag split of the readout through the summation and the limiter, over both proposal channels. Taken on the posterior head's *inputs* it fails on a `posterior_logvar_mode: independent` head, whose raw prior log-variance is an unused input; taken on the lag attention module it fails because its first output is discarded by the forward |
| `FeatureAblation` | **shipped** | model-agnostic; grouped by lag band of the source relative to each row's anchor, it is `occlusion`'s intervention read on this analysis's readouts and anchors, reported in `occlusion`'s sign |
| `InputXGradient`, `Saliency` | evaluated, not shipped | run and pass every structural check; single-point gradients with no completeness, adding nothing the integrated form does not |
| `GradientShap` | evaluated, not shipped | integrated gradients averaged over a random baseline distribution; its per-sample completeness residual is of the order of the readout by construction, and a random baseline makes two runs disagree |
| sliding-window `Occlusion` | evaluated, not shipped | a window straddling the anchor assigns its effect to steps after the anchor, so the per-step output fails the causality check by construction; the grouped ablation is the same intervention on the right groups |
| `DeepLift` | evaluated, **not usable** | runs, but its rescale rule reaches only the module nonlinearities it hooks; the GELU and SiLU functionals, the smooth bounds, the tanh limiter and `entmax15` pass through as plain gradients, and its completeness residual is of the order of the readout |
| `LayerConductance` | evaluated, not shipped | its sums differ from the readout difference on every layer tested where the layer integrated gradient is exact |
| `NeuronConductance` | evaluated, not shipped | the conductance of one latent coordinate is the input attribution of that coordinate's readout, which the readout registry already provides; and the heads return tuples its selector cannot index |

The verdicts travel in every block under `methods`, with the reasons, so a reader of a summary
sees what was rejected and not only what was kept.

Two properties of these forwards decide which layer a split can be taken on. A Captum layer
attribution replaces the layer's output with an interpolated tensor and differentiates the readout
against it, so **every output of the layer must reach the readout** (the lag attention's discarded
`W_o` projection does not) and **the layer must be called once per forward** (the lag-residual
cell's chunked proposal pass calls the head several times; the pass runs it unchunked for the
attribution and restores the setting in a `finally`). The per-head fusion modules and the proposal
head satisfy both.

## 5. The four structural checks, and what they prove

Each is asserted exactly on the tiny models by `tests/test_eval_attribution.py` in both cells and
measured on every row of a real run into the block's `checks`.

1. **Causality.** Attribution to any stored step after the anchor is exactly zero, for every
   readout, under both baselines, on both streams. On the conv-LSTM cell this holds only with
   `causal_norm: true` — the shipped setting, and the one its own causality tests build with —
   because the plain group norm pools over time; on the tiny model without it the attribution
   after the anchor reaches $0.26$–$0.30$ nats, which is the leak that switch exists to close. The
   transformer cells are step-wise causal by construction. It is a *gradient* statement, so it is
   exact; the value-level check (`FeatureAblation` of every step after the anchor, the `rest`
   group) is zero to the float noise of a kernel re-run.
2. **The gate.** Attribution to a source step a channel had not warmed up at is exactly zero: the
   adapter multiplies those positions by exactly zero, so the gradient through them is exactly
   zero. In stored coordinates a gathered-and-shifted channel is live from $W'_c$ (the encoder
   reads stored step $t - d_c$ and masks while $t < W'_c + d_c$), which `warm_from_step` computes
   per declared channel; the dropped channels are live from nowhere.
3. **Target-only purity.** The source attribution of $\mu^p$ and of the base block score is
   exactly zero. The prior clock (attentive cells) is the encode of a *zero* source, detached; the
   metadata clock (residual cell) is a sinusoid of position; neither carries a gradient to the
   source input.
4. **Completeness.** The integrated-gradient sum reproduces $f(x) - f(x_0)$. The residual is
   measured per row against the larger of $|f(x) - f(x_0)|$ and $|f(x)|$ — so a source that barely
   moves a readout does not report a residual of order one over a difference of order nothing —
   and the block reports its median, its maximum and the count of rows above
   `COMPLETENESS_TOLERANCE` $= 10^{-2}$. The tests pin a median below $10^{-2}$ and a maximum below
   $5 \times 10^{-2}$ on the conv-LSTM tiny model (its per-step group norm makes its path the
   roughest of the family's) and a maximum below $10^{-3}$ on the residual cell's.

What they prove: that an attribution reaching before the anchor is a statement about the
history, that a non-zero attribution on a source step is a statement about a coefficient the
model actually read, that the source-null attribution of a readout is entirely about the source's
content, and that the maps sum to the number they decompose. What they do not prove is anything
about the physiology; see §11.

## 6. The reductions

All from the $(N, T, c_y)$ and $(N, T, c_u)$ maps of $N$ rows, all per row, and every summary
statistic then per recording (a row's segment is its recording's one segment in the main draw)
and then over recordings.

* **Time profile** per stream: the sum over channels, $(N, T)$.
* **Channel profile** per stream: the sum over steps, $(N, C)$.
* **Lag-aligned source profile** $q_\ell = p^{u}_{t_a - \ell}$ for $\ell = 0, \ldots, L - 1$,
  `NaN` where $t_a - \ell$ falls before the record. Drawn on the compensated seconds axis
  `lag_axis.compensated_seconds_axis(L, delay_steps)` — the model's own `source_delay_steps` in
  the attentive cells, zero in the residual cell — beside the model's own lag readout at the same
  anchor: the KL attribution `source_kl_lag_map[t_a]` or the proposal norm masked by `lag_valid`.
  Two **agreement** statistics per row over the lags both carry: the Pearson correlation of
  $|q_\ell|$ against the readout, and the Jensen–Shannon distance (base 2, in $[0, 1]$, through
  `lag_hist.jensen_shannon`) between the two normalised to distributions.
* **Frequency-band sums** per stream through the **declared-axis** channel map
  `band_channel_map.csv`, with `stream`, `channel` (the index within the stream, which is the
  index the model's input tensors use) and `band`. The attributions live on the model's input
  axis, which is the declared width, so the join is through the declared map and not the
  kept-axis one the decoder-side per-channel readouts use; a dropped channel's attribution is
  exactly zero (its gate multiplies it by zero) and its band sum carries it. In the lag-residual
  cell the stage builds the map through `band_partition.emit_partition` when the directory holds
  none, from the same shards and resolved budget, and records a skip by name when the shards carry
  no channel provenance.
* **Lag-band sums** of the lag-aligned profile over each `occlusion_bands` band, clipped to the
  window.
* **The layer split**: per head (attentive cells) or per lag (residual cell), as §4 states.
* **Band ablation**: `FeatureAblation` grouped by lag band relative to each row's anchor over
  every source channel at once, reported as $f(x^{\setminus \mathrm{band}}) - f(x)$ — `occlusion`'s
  sign, positive meaning the readout rose without the band — beside a `rest` group of every
  source position outside the bands. `rest` holds the steps after the anchor, whose ablation is
  zero to float noise, **and** the steps before the earliest band's reach, whose ablation is not
  zero on an arm whose source pathway has memory beyond the searched window (the conv stem's
  receptive field, the deep encoder's unbounded one): that is what the source state at lag $\ell$
  remembers from before $t_a - \ell$, and it is a measured number rather than an assumption that
  the window is the whole of what the model reads.
* **The null decomposition** for the divergence: under `source_null`, $f(b)$ is the null arm's
  own divergence at the anchor (the availability-clock part), the integrated attribution is the
  source-content part and the entry jump is what the normalisation does between the two; under
  `all_zero`, the attribution splits between the target and the source streams.

## 7. Every table and figure

All under `attribution/` in the results directory. Units are the readout's: nats per anchor for
`kld`, `kld_dim`, `nll_*` and `pred_gap`; a share in $[0, 1]$ for the attentive cells' `lag_band`,
a norm in latent units for the residual cell's. Every lag axis is stored-coefficient seconds.

| file | one row per | columns |
|---|---|---|
| `attribution_rows.csv` | attributed (anchor, readout, baseline, band) | `guid`, `epoch`, `clinical_class`, `subgroup`, `anchor` (stored step), `column`, `readout`, `baseline`, `band`, `coordinate` ($-1$ where none), `value_input`, `value_baseline`, `value_entry`, `entry_jump`, `attributed` (the IG sum), `ig_delta`, `completeness_rel`, `target_total`, `source_total`, `target_abs_total`, `source_abs_total`, `after_anchor_max_abs`, `gated_off_max_abs`, `lag_corr`, `lag_js`, `kld_top_coordinate`, `lagband_<band>`, `band_<stream>_<band>`, `layer_total`, `layer_off_axis_total`, `ablation_<band>`, `ablation_rest` |
| `attribution_vectors.npz` | the same rows, row-aligned | `time_profile_target` $(N, T)$, `time_profile_source` $(N, T)$, `lag_profile` $(N, L)$, `target_lag_profile` $(N, L)$, `model_profile` $(N, L)$, `channel_profile_target` $(N, c_y)$, `channel_profile_source` $(N, c_u)$, `layer_per_unit` $(N, M$ or $L)$ (`NaN` rows where no split was taken), `lag_seconds` $(L,)$ |
| `attribution_maps.npz` | the first `kld`/`source_null` row per class | `guid`, `subgroup`, `clinical_class`, `anchor`, `target` $(n, T, c_y)$, `source` $(n, T, c_u)$ |
| `attribution_recordings.csv` | recording | `<readout>_source_total`, `<readout>_target_total`, `<readout>_lag_corr`, `<readout>_lag_js` for the two main readouts under `source_null`, the cohort columns, `n_segments`; the frame the runner fans by class and subgroup |
| `attribution_summary.csv` | (readout, baseline, band) | counts, recording-mean values and totals, `completeness_rel_max`, `completeness_rel_median`, `after_anchor_max_abs`, `gated_off_max_abs` |
| `attribution_bands.csv` | (readout, stream, frequency band) | `attribution_mean`, `n_recordings`, `spectral_skill_pred_gap_nats` (target bands, where that pass ran) |
| `attribution_lag_bands.csv` | (readout, lag band) | `lag_lo`, `lag_hi`, `ig_attribution_mean`, `ablation_delta_mean`, `occlusion_delta_total_nats` (sign-flipped on `pred_gap`, absent on `kld`), `n_recordings` |
| `attribution_layer.csv` | (readout, unit, class or `pooled`) | `n_recordings`, `mean` |
| `attribution_null.csv` | (readout, baseline, class or `pooled`) | recording means of the values, the entry jump, the attributed sum and both totals |
| `attribution_traces.csv` | traced recording | `guid`, class, subgroup, `n_segments`, `n_anchors`, `span_hours`, `arrays_file`, `figure_file` |
| `traces/<class>/<guid>_<subgroup>_attribution_trace.npz` | the traced recording | the traces' shared arrays with `attribution_lag_map` and `model_lag_map` |

| figure | what it draws |
|---|---|
| `attribution_maps.pdf` | one row per class: both streams' maps of $K_t$ at one anchor (stored step × declared channel, anchor marked, warm-up staircase drawn) and the lag-aligned source attribution against the model's lag readout |
| `attribution_lag_profile.pdf` | per main readout: the recording-mean normalised $\lvert q_\ell \rvert$ against the normalised model readout, pooled and by class, with the agreement in the title |
| `attribution_bands.pdf` | top: attribution by frequency band per stream with the band-resolved skill gap on a twin axis; bottom: per lag band, the IG sum, the ablation delta and the occlusion delta |
| `attribution_layer.pdf` | left: the per-head or per-lag split per readout; right: how the anchor's top $K_{t,d}$ coordinate is fed, by offset, per stream |
| `attribution_null.pdf` | left: $K_t$ at the input, at the null, the attributed content and the entry jump by class; right: the target/source split under `all_zero` |
| `traces/<class>/<guid>_<subgroup>_attribution_trace.pdf` | the traces' shared figure: $\lvert q_\ell \rvert$ of $K_t$ over hours before delivery, the model's lag readout, the agreement, the totals, the values |

`FIGURE_GUIDE.md` carries the reading rules of each. Every figure prints
`ATTRIBUTION_CAVEAT`, the lag-resolved ones the group-delay caveat and the cell's lag
qualification as well.

**The block** in `summary.json` (`results.attribution` here, `attribution` in the lag-residual
cell) carries `n_samples` (segments), `composition`, `plan` (cap, seed, anchors per segment, step
count, entry fraction, baselines, readouts, lag bands, the layer, the geometry), `selection`,
`trace_selection`, `cost`, `checks`, `summary`, `lag_bands`, `joined` (which of the three files
were found), `methods`, `lag_qualification`, `caveat`, `traces`, `failures` and `files`; the
attentive cells add `grouped_frames`. None of its keys is one the lag-residual cell's gate refuses,
and its test asserts that on the block it writes.

## 8. The cost

Per segment, at $k$ anchors: $2 \times 2$ integrated-gradient calls (two readouts, two baselines)
plus one per lag band plus one for the top coordinate, each of $k$ rows at `IG_STEPS` interpolated
forwards-and-backwards, plus two layer attributions of the same size, plus two band ablations of
$1 + n_{\mathrm{bands}}$ forwards per row. On the tiny conv-LSTM fixture on CPU, three segments at
four anchors each — 108 rows — took $40$ s at 32 steps, about $0.37$ s per row; the shipped 64
steps roughly doubles the per-row figure. The block's `cost` records `elapsed_s`, `n_rows`,
`n_forward_equivalents`, `seconds_per_row` and `hours_per_1000_samples` for the pass that ran, and
`caps.attribution_segments` is set from that measurement rather than guessed: it ships at $24$,
which on a GPU is minutes rather than the collection pass's hours, and absent it means the
analysis's own default rather than every segment. The trace adds one recording per class at the
same anchors of every segment, `kld` under `source_null` only.

## 9. Fixture findings

Measured on the tiny models the suites build, so evidence about the **machinery** and about
nothing clinical: the models are untrained, the batches are Gaussian noise at the causal widths
(the conv-LSTM and transformer cells) or at the integer-operator widths (the residual cell), and
every number below is about what the forwards do to a gradient rather than about any recording.

**The exactly-zero baseline is singular, on all three cells.** Along the straight path from the
source-null baseline for the divergence readout at one anchor (values in nats):

| cell | $f(0)$ | $f(10^{-6})$ | $f(10^{-3})$ | $f(1)$ | directional derivative at $0$ |
|---|---|---|---|---|---|
| conv-LSTM, `causal_norm` | 2.752 | 2.915 | 2.865 | 2.380 | $1.3 \times 10^{22}$ |
| transformer, `entmax15` | 7.136 | 4.539 | 3.570 | — | $2.5 \times 10^{14}$ |
| transformer, softmax | 5.702 | 4.258 | 3.773 | — | $5.1 \times 10^{13}$ |
| lag-residual | 4.387 | 4.387 | 4.386 | 5.036 | $-0.96$ |

The residual cell's source-null path is smooth (its source pathway is a pointwise encoder into a
small multilayer perceptron); its all-zero path is singular on the target side ($-1.1 \times
10^{18}$), for the transformer target encoder's `RMSNorm`. Integrated gradients from the exact
zero: relative completeness residual $0.23$–$0.43$ (conv-LSTM, source-null), $0.9$–$1.0$
(transformer, source-null), $1.5$–$15$ (all-zero, every cell), unchanged or worse from 32 to 512
steps. From the entry point $\alpha_0 = 10^{-3}$: $7 \times 10^{-4}$ at 64 steps and $4 \times
10^{-6}$ at 256 (conv-LSTM), $1.3 \times 10^{-3}$ and $1.1 \times 10^{-5}$ (transformer, `entmax15`),
$3 \times 10^{-4}$ and $1.5 \times 10^{-6}$ (transformer, softmax), $9 \times 10^{-7}$ either way
(residual). The `pred_gap` and `lag_band` readouts behave the same way.

**What the entry jump says about the null arm.** On the attentive tiny models the divergence at
the exact null is far from its value an infinitesimal source away — $7.1$ against $3.6$ nats on
the transformer cell — so `kld_source_null` is evaluated at a point the encoders' normalisation
makes discontinuous in the source. Whether a trained checkpoint sits as close to that
discontinuity is the first thing a production run's `entry_jump_mean` answers (§10); on the
residual cell the jump is of order $10^{-3}$ of the readout.

**Causality, the gate and purity are exact**, on every readout, both baselines and both streams,
on all three cells — and on the conv-LSTM cell only with `causal_norm`, without which the tiny
model leaks $0.26$–$0.30$ nats of attribution past the anchor. The batched-anchor call agrees with
the one-at-a-time call to $8 \times 10^{-7}$.

**The per-head split is complete.** On the transformer tiny model (`entmax15`) at anchor step 13
the four heads' attributions of $K_t$ were $[0.017, 0.022, -0.188, -0.191]$, summing to the
readout difference $-0.340$ exactly, with the target-state and prior parts zero under the
source-null baseline. On the residual cell the per-lag split over both proposal channels sums to
the readout difference to $10^{-3}$; over the mean channel alone it does not, because the
divergence carries the scale update.

**The band ablation is `occlusion`'s intervention.** On the transformer tiny model the divergence
fell by $1.08$, $0.04$ and $0.45$ nats when the source was zeroed on the anchor, near and far
bands, and by $0.009$ on the `rest` group — the memory the conv stem carries from before the
searched window — while zeroing the target streams changed nothing to $2 \times 10^{-6}$.

**End to end on the stub loader** (three segments, four anchors each, the four tiny bands): 108
rows in the conv-LSTM cell, every structural check exactly zero, the target-only check exactly
zero; and the same in the residual cell with every completeness residual below $5 \times 10^{-5}$.

## 10. Production-run findings

*To be filled from a real checkpoint. Nothing below is measured yet.*

For each cell and arm, after one run with the shipped cap, record from `results.attribution`:

* `checks.completeness_rel_median` and `_max`, `n_rows_over_tolerance` — whether 64 steps and the
  entry fraction are enough on the trained weights, and whether the step count should move;
* `attribution_null.csv`: `value_baseline_mean` against `value_input_mean` (how much of $K_t$ the
  clock is, per class), `entry_jump_mean` (how close the trained null arm sits to the
  normalisation's discontinuity), and the `all_zero` target/source split;
* `attribution_summary.csv`: `lag_corr_mean` and `lag_js_mean` for `kld` and `pred_gap` — whether
  the model's own lag readout points where its sensitivity is;
* `attribution_lag_bands.csv`: the IG sum, the ablation delta and the occlusion delta per band,
  and whether they agree in sign;
* `attribution_bands.csv`: which frequency bands of the source carried `pred_gap`, against the
  `spectral_skill` gap of the target bands;
* `attribution_layer.csv`: which head (or lag) the source reached the latent through;
* `cost`: the measured rate, and the cap it supports.

## 11. How this output will be misread

* **An attribution is not a causal claim about the physiology.** It is the sensitivity of a
  fitted computation along one straight path from one baseline, under one trained
  parameterisation. A large source attribution says the model's readout responded to that
  coefficient; it does not say the uterus caused the heart-rate change, and two models with the
  same forecasts can attribute differently. The lag-residual cell's own qualification applies with
  full force: a zero-sum reallocation of proposals leaves every prediction identical while moving
  every per-lag attribution.
* **The lag axis is stored-coefficient time.** $q_\ell$ is the attribution to the source
  coefficient stored $\ell$ steps before the anchor, on a grid of one-sided filter outputs whose
  composed group delay reaches the order of the lag search itself; it is not a physiological
  delay. Every lag figure prints the caveat.
* **The source-null attribution is relative to the availability clock, which it can never
  contain.** The announcement is a constant of the step index and is on both ends of the path.
  $K_t$ at the null is the clock's part, and it is reported beside the content, not inside it.
* **The entry jump is not an attribution.** It is what the normalisation does between the exact
  zero and a stream infinitesimally off it, and it belongs to no input. A large jump on a trained
  model says the null arm is evaluated at a discontinuity, which is a statement about the control,
  not about the source.
* **The all-zero baseline's target attribution is not "what the target contributed to $K_t$".**
  The divergence at an all-zero input is a property of the biases and the clock; the attribution
  says how the readout moved as both streams were switched on together, and the split between
  them depends on the path.
* **The lag readout attributed under a band name is what the cell's own readout is.** An
  attention mass in the attentive cells; a proposal norm — an update magnitude before the sum and
  the limiter, neither a distribution nor an allocation — in the residual cell.
* **The classes are out of distribution.** The checkpoints train on healthy recordings; every
  class contrast in a by-class panel compares a fitted response on data the model never saw.
* **Every summary is over recordings**, one segment per recording in the main draw and a few
  anchors per segment. A class mean is a mean over a handful of recordings up to the cap; the
  recording counts are on every table and figure, and nothing here is tested.
* **A frequency-band sum over an input stream is over its declared channels**, dropped ones
  included at exactly zero, and it is not the skill of that band: `spectral_skill` is, and is drawn
  beside it for the target bands only.
* **The maps are one anchor of one segment of one recording**, drawn for looking at. The
  reductions are the evidence, and they are descriptive.
* **Completeness is against the entry point**, and a row above the tolerance is a row whose
  attribution does not sum to what it claims to decompose; the block counts them and the tables
  carry the residual per row.
