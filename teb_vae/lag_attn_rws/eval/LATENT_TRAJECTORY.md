# The latent trajectory through labour

This file is the specification of a planned analysis, `latent_trajectory`. The analysis summarises,
projects, draws and tests the path of a recording's latent state from the start of monitoring to
delivery, and it splits that path at the second-stage onset. The code does not implement it yet.

All values below use the geometry of `configs/default.yaml`: $T = 300$ anchors, warm-up $w = 30$,
horizon $H = 30$, $T_{\mathrm{valid}} = T - H = 270$, $L = 91$ lags (`max_lag: 90`), $\Delta = 4$ s
per anchor, $d_z = 64$, and $M = 4$ attention heads. Each head writes one latent group of
$g = d_z / M = 16$ coordinates.

---

## Table of contents

0. [Scope, and what the code already does](#0-scope-and-what-the-code-already-does)
1. [What a trajectory is in this model](#1-what-a-trajectory-is-in-this-model)
2. [Structural geometry of the path](#2-structural-geometry-of-the-path)
3. [Summarising a segment: the window scheme](#3-summarising-a-segment-the-window-scheme)
4. [From $d_z$ dimensions to a picture](#4-from-d_z-dimensions-to-a-picture)
5. [Projection-free readouts](#5-projection-free-readouts)
6. [First stage and second stage](#6-first-stage-and-second-stage)
7. [Comparing two cohorts' trajectories](#7-comparing-two-cohorts-trajectories)
8. [The figures](#8-the-figures)
9. [Where this lands in the pipeline](#9-where-this-lands-in-the-pipeline)
10. [Order of work](#10-order-of-work)

---

## 0. Scope, and what the code already does

| Readout | Status |
|---|---|
| $K_t$ and `mc_pred_gap` against absolute time, per recording | Implemented in `analyses/trajectory.py`. |
| $K_t$ and `pred_gap` binned by time before delivery, by class, with tests | Implemented in `analyses/time_to_delivery.py`, on a $0.5$ h grid (`cohort.TRAJECTORY_BIN_HOURS`) with Holm across windows. |
| $K_t$ and `pred_gap` binned by time from second-stage onset | Implemented in `analyses/second_stage.py`, on the same grid. |
| The latent path of $\mu^q_t$, drawn in 2-D and 3-D | Not implemented. No artifact carries the input. |
| The latent path split at the second-stage onset | Not implemented. |

No file that an evaluation run writes carries a latent mean vector:

* `per_sample.csv` carries norms only: `mu_prior_rms`, `delta_mu_rms`, `mu_post_prior_gap_sq`,
  and the saturation fractions `mu_prior_sat_frac_raw`/`_masked` and
  `delta_mu_sat_frac_raw`/`_masked`.
* `per_anchor.parquet` has one row per contributing anchor, keyed by `(guid, epoch, anchor)`. It
  carries `coverage`, `kld_per_t`, `nll_base_block`, `nll_full_block`, `pred_gap`, `argmax_lag`,
  the `mc_*` scores with `mc_pred_gap`, and `seconds_since_contraction`.
* `per_sample_vectors.npz` carries the members of `metrics.VECTOR_READOUTS` (per-dimension and
  per-head KL, lag and attention profiles, per-head attention entropy) and the cross-spectral sums
  under `collect.SPECTRAL_VECTOR_PREFIX`.
* `retained_arrays.npz` carries the `collect.RETAINED_QUANTITIES` groups `waveforms` and
  `attention`. Neither group holds a latent tensor.

The analysis therefore needs one addition to the collection pass (§9.2). All other parts read the
tables offline.

### The latent quantities

The model returns `mu_prior` ($\mu^p_t$) and `mu_post` ($\mu^q_t$) in its forward dict.

| Quantity | Meaning |
|---|---|
| $\mu^p_t$ | the target-only predictive state of the fetal heart rate at $t$ |
| $\mu^q_t - \mu^p_t$ | the additional predictive information that the source contributes |
| $\mu^q_t$ | the complete state after the source is incorporated |

The decoder takes exactly one tensor, the latent sample. A path through $\mathbb{R}^{d_z}$ is
therefore a path through the model's forecast state.

Use the means, not the samples $z^p_t, z^q_t$. The samples carry reparameterisation noise, so two
runs of one checkpoint would draw two different paths.

Store $\mu^p$ and $\mu^q$, and do not store $\Delta\mu$ separately. The posterior mean is a bounded
residual on the prior mean in the same coordinates, so $\Delta\mu = \mu^q - \mu^p$ follows exactly.

---

## 1. What a trajectory is in this model

### 1.1 There is no latent transition density

The model is not a latent state-space model. It has no $p(z_t \mid z_{t-1})$ and no filtering
recursion, and the objective factorises over anchors. Consecutive latents $z_t$ and $z_{t+1}$ are
correlated because the encoder states $H^y_t$ and $H^y_{t+1}$ are correlated, through the dilated
convolution receptive field and the LSTM state.

The trajectory is thus a sequence of independent readouts of a smoothly varying input. Its
smoothness comes from the encoders. The readable quantities are position, displacement and rate,
and how they change at a clinical landmark.

### 1.2 The encoder resets at every segment boundary

Each 20-minute segment is a separate forward pass. The LSTM state starts from zero and the
convolution left padding is zeros; the $w = 30$ warm-up anchors exist for this reason. The latent
at the start of segment $k$ does not continue from the end of segment $k-1$.

A whole-delivery path is therefore a chain of short arcs. Every join carries a re-encoding
discontinuity that comes from the harness. §5.2 uses the joins to measure that discontinuity.

### 1.3 The coordinates are not orthogonal, ordered or scaled

The prior mean is tanh-bounded into $[-\texttt{mu\_scale}, \texttt{mu\_scale}]^{d_z}$ and
$\Delta\mu$ into $[-\texttt{delta\_mu\_scale}, \texttt{delta\_mu\_scale}]^{d_z}$ ($5$ and $3$ in
`default.yaml`). Nothing orthogonalises the coordinates or orders them by importance, so a
projection is necessary (§4).

The posterior is head-structured: latent group $m$ (coordinates $gm \dots gm + g - 1$) is written
only by attention head $m$. A per-group view of the path is therefore an architectural
decomposition (§4.3).

Read the saturation fractions before any distance. A state on the tanh bound has its motion
clipped. For KL-based readouts, read the `latent` analysis verdict `prior_variance_not_pinned`
first.

---

## 2. Structural geometry of the path

Six structural facts produce features in the drawn path:

| # | Fact | Effect on the path |
|---|---|---|
| 1 | Anchors $t < w = 30$ carry no loss term | the first $120$ s of every segment is an untrained transient |
| 2 | Anchors $t \ge T - H = 270$ are not decoded | the last $120$ s of every segment has no point |
| 3 | Lag support is truncated until $t \ge L - 1 = 90$ | $\Delta\mu$ and $K_t$ have truncated support over the first $6$ min of every segment |
| 4 | Segment stride is $1200$ s and the trained span is $960$ s | a $244$ s hole at every segment join |
| 5 | The loader does not trim-correct `epoch` | `trajectory` places anchor $0$ at `epoch`, $60$ s before its trimmed-grid position |
| 6 | The coverage floor drops whole anchors | a signal-loss stretch removes points without removing time |

### 2.1 The trained anchor set, and the join arithmetic

The trained set is $\mathcal{A} = [30, 270)$: $240$ anchors, $960$ s of every $1200$ s segment.
`trajectory.whole_delivery` uses $t_{\mathrm{abs}} = \mathrm{epoch} + 4t$, so a segment that starts
at $e$ contributes $[e + 120,\ e + 1076]$.

The dataset builder (`hdf5_dataset/new_pipeline/create_new_pipeline.py`) writes stored segments of
`SIGNAL_LENGTH = 5280` raw samples ($22$ min) at `STEP_SIZE = 4800` samples ($1200$ s). The
loader's symmetric $1$-minute trim (`trim_minutes: 1.0`) removes the $2$-minute stored overlap. The
trimmed grids of consecutive segments therefore abut, but their trained spans do not:

$$
\underbrace{e + 1076}_{\text{last trained anchor of segment } k}
\quad\longrightarrow\quad
\underbrace{e + 1200 + 120 = e + 1320}_{\text{first trained anchor of segment } k+1},
\qquad
\text{gap} = 244\ \mathrm{s}.
$$

Three consequences follow:

* A trained-anchor delivery covers at most $960 / 1200 = 80\%$ of its span by construction.
* Trimmed segments do not overlap, so the overlap averaging in `trajectory.whole_delivery` is
  normally not exercised and `n_contributing` is normally $1$.
* `trajectory.BREAK_TOLERANCE_S` is $1.5\Delta = 6$ s, so every segment join counts as a break.
  The latent trajectory must classify a join as **structural** when `gap_before_s` is within one
  anchor step of $244$ s, and as a **real break** otherwise. Draw a structural join as a lighter
  connector, and do not draw a real break.

### 2.2 The trim offset of `epoch`

`epoch` is the start of the stored $22$-minute segment. The loader trims one minute from each end of
the signal arrays and passes `epoch` through unchanged. Anchor $0$ of the trimmed grid is therefore
at $\mathrm{epoch} + 60$ s, while `trajectory.whole_delivery` places it at $\mathrm{epoch}$.

The offset is constant, so it changes no shape, distance, rate or cohort contrast. It moves the
hours-before-delivery axis and the second-stage boundary of §6 by $60$ s. If the code corrects it,
correct it in the shared assembly (§9.1), so that `trajectory` and `latent_trajectory` agree.

### 2.3 The population statement

The training split is `pre_training_dataset/*`, which holds the healthy with-background subgroups
only (`cohort.PRETRAINING_SUBGROUPS`). On the evaluation split, acidosis, HIE and both healthy
no-background subgroups are therefore out of distribution. `cohort.build_cohort_block` computes
this statement. The analysis must repeat it in its own record.

---

## 3. Summarising a segment: the window scheme

### 3.1 Four windows with boundaries at 30 / 90 / 150 / 210 / 270

Split $\mathcal{A} = [30, 270)$ into four windows of $60$ anchors:

| Window $j$ | Anchors | Duration | Lag support |
|---|---|---|---|
| $0$ | $[30, 90)$ | $240$ s | truncated: every anchor has $t < L - 1$ |
| $1$ | $[90, 150)$ | $240$ s | full |
| $2$ | $[150, 210)$ | $240$ s | full |
| $3$ | $[210, 270)$ | $240$ s | full |

The boundary at $90$ is $L - 1$, the anchor where the attention window stops being truncated. Window
$0$ is therefore exactly the truncated-support region. Report every window, and make window $0$
excludable by one filter. Keep window $0$ in the drawn path, and exclude it from every
$\Delta\mu$-derived or KL-derived number.

The window centres are at $t_{\mathrm{abs}} = e + 238 + 240j$. The spacing is therefore:

$$
\underbrace{240\ \mathrm{s}}_{\text{within a segment}}
\qquad\text{versus}\qquad
\underbrace{(e + 1438) - (e + 958) = 480\ \mathrm{s}}_{\text{across a join}} .
$$

At $k = 4$ windows the join step is exactly twice the within-segment step. §5.2 uses this ratio, so
$k$ is fixed at $4$.

### 3.2 What is averaged, and what is `NaN`

For window $j$ of each segment, average over the contributing anchors only. The contributing
indicator $c_t$ is the one that the reconstruction and the KL support use:

$$
\bar\mu^{q}_{j} = \frac{\sum_{t \in W_j} c_t\, \mu^q_t}{\sum_{t \in W_j} c_t},
\qquad
\bar\mu^{p}_{j} = \frac{\sum_{t \in W_j} c_t\, \mu^p_t}{\sum_{t \in W_j} c_t},
\qquad
n_j = \sum_{t \in W_j} c_t .
$$

Three rules apply:

* The model produces $\mu^q_t$ for every anchor, also for anchors that the coverage floor rejects
  and for anchors inside a signal-loss gap. The window mean must use the contributing anchors only.
* Every window point carries its count $n_j$.
* $n_j = 0$ gives `NaN` for that window, never $0.0$. A zeroed latent is the origin of the space.

Each window also carries the within-window dispersion

$$
s_j = \sqrt{\frac{\sum_{t \in W_j} c_t \lVert \mu^q_t - \bar\mu^q_j \rVert^2}{\sum_{t \in W_j} c_t}} .
$$

The analysis reports $s_j$ beside the distance between consecutive window means.

### 3.3 Storage cost

Per segment, at $k = 4$, $d_z = 64$, two latents and `float64`, the window means take
$4 \times 64 \times 2 \times 8\ \mathrm{B} = 4{,}096\ \mathrm{B}$, plus $4 \times 2$ scalars for
$n_j$ and $s_j$. This is small enough to store for every segment of the split, with no `caps`
entry and no retention plan.

The full per-anchor latent is $240 \times 64 \times 2 \times 4\ \mathrm{B} = 120$ KiB per segment in
`float32`. The specification does not store it:

```text
lean-limit: the latent path is summarised at 4-minute windows; retain the full per-anchor
mu under a `caps.latents` entry when an analysis needs sub-window structure -- e.g. a
contraction-triggered latent response, which needs anchor resolution around each onset.
```

---

## 4. From $d_z$ dimensions to a picture

### 4.1 Requirements for the projection

1. **Out-of-sample.** A second checkpoint, arm or cohort must project into the same picture.
2. **Deterministic.** Two runs of one checkpoint at one seed must give identical `results`.
3. **Metric.** The readouts of §5 are distances and rates.
4. **No new dependency.** `scikit-learn` is not a dependency of `lag_attn_rws`.
5. **Recording-fair.** Each recording has equal weight in the basis.

### 4.2 Recording-weighted PCA, by SVD, with a recorded basis

Fit once on the whole split, on the per-(segment, window) posterior means $m_i \in
\mathbb{R}^{d_z}$. Use the weights $w_i = 1 / n_{g(i)}$, where $n_g$ is the number of points that
recording $g$ contributes, normalised so that $\sum_i w_i = 1$:

$$
\bar m = \sum_i w_i m_i,
\qquad
A_{i\cdot} = \sqrt{w_i}\,(m_i - \bar m),
\qquad
A = U S V^{\!\top}.
$$

The components are the first three columns of $V$, with explained-variance ratios

$$
\mathrm{EVR}_j = \frac{s_j^2}{\sum_{j'} s_{j'}^2}.
$$

The fit uses `numpy.linalg.svd` with these rules:

* **Weight by $1/n_g$.** This is the rule of `frames.per_recording_means`, applied to the basis.
* **Fit on per-window points**, not on per-recording means. Recording centroids give a cohort axis,
  not a trajectory axis.
* **Use the covariance, not the correlation.** All coordinates share one unit and one bound.
  Standardisation would give a near-constant dimension the same weight as a live one.
* **Pin the sign.** For each component $v_j$, let $d^\star = \arg\max_d |v_{j,d}|$ (lowest $d$ on
  a tie), and negate $v_j$ when $v_{j,d^\star} < 0$.
* **Record the basis.** Write $\bar m$ ($d_z$ floats), $V_{1:3}$ ($d_z \times 3$) and the three
  $\mathrm{EVR}$ values to the analysis CSV. A second run projects into this basis by reading the
  file.
* **Report $\mathrm{EVR}_{1:3}$ next to every PC figure.**

Fit on $\mu^q$. Project $\mu^p$ into the same basis, so that $\Delta\mu$ is an arrow from the prior
point to the posterior point on the same axes.

### 4.3 The per-head-group view

The posterior has $M$ disjoint groups of $g$ coordinates, and attention head $m$ alone writes group
$m$. Project each group's coordinates onto its own first two components, and draw $M$ small panels.
The per-head attention profile is a separate readout (`analyses/attention.py`).

The scalar companion is the per-group KL $K^{(m)}_t$. The model returns it as `kld_per_t_per_head`,
and `per_sample_vectors.npz` carries its per-sample mean as `kld_per_head`. The per-anchor table does
not carry it. Adding it costs $M$ `float32` columns per contributing anchor.

### 4.4 A cross-fitted supervised axis

A supervised "labour axis" is the latent direction that best predicts hours before delivery. An axis
fitted and applied on the same points trends with time by construction. The axis must therefore use
the construction of `sufficiency`: split at GUID level into two halves, fit ridge on half A and
project half B, then fit on B and project A. Every plotted coordinate is then an out-of-fold
prediction. Assert the GUID disjointness at runtime.

Ship the axis with its held-out $R^2$. Below a stated threshold, the analysis records a skip and does
not emit the panel. Build PCA first, and build the supervised axis in a later step.

### 4.5 Methods out of scope

| Method | Reason |
|---|---|
| t-SNE, UMAP | No out-of-sample map (requirement 1), stochastic (2), not metric (3), UMAP is not installed (4). Their tuning knobs are kept out of `eval_config`. |
| Diffusion maps, PHATE | They assume a manifold traversed at varying density. Same out-of-sample and dependency limits. |
| An autoencoder of the latent | A second learned model with its own training run and checkpoint. |
| Raw coordinate triples ($\mu_0, \mu_1, \mu_2$) | The coordinates are not ordered by importance. |
| Per-recording PCA | Each recording gets its own basis, so no two paths share axes. |

---

## 5. Projection-free readouts

Each readout is a per-recording (or per-recording-per-stage) scalar. This is the unit that the
aggregation chain and `cross_subgroup` consume. All readouts are rotation-invariant, so none
depends on §4.

Let $x_1, \dots, x_N$ be a recording's window means in time order, at times $\tau_1 < \dots <
\tau_N$.

### 5.1 Displacement from the healthy centroid

$$
D_i = \lVert x_i - c_{\mathrm{healthy}} \rVert_2,
\qquad
c_{\mathrm{healthy}} = \operatorname{mean}_{g \in \text{healthy}} \big(\operatorname{mean}_i x_i^{(g)}\big).
$$

The outer mean is over healthy recordings, not segments. Use leave-one-out when the scored
recording is itself healthy. An inactive dimension at a constant offset adds a constant to every
$D_i$; the record states this.

```text
lean-limit: Euclidean distance in the raw latent box; replace with Mahalanobis against the
healthy cohort's ridge-regularised covariance when the healthy cohort carries more than
2 * d_z recordings, below which the d_z x d_z covariance is not estimable.
```

### 5.2 Step rate, and the join noise floor

$$
v_i = \frac{\lVert x_{i+1} - x_i \rVert_2}{\tau_{i+1} - \tau_i}\ \ \big[\text{latent units per second}\big].
$$

The step is $240$ s within a segment and $480$ s across a join (§3.1), so the rate is normalised by
$\Delta\tau$.

Consecutive windows within a segment come from one encoder pass. Across a join they come from two.
The ratio

$$
\boxed{\ \rho = \frac{\operatorname{median}\ v_i\ \text{across joins}}{\operatorname{median}\ v_i\ \text{within segments}}\ }
$$

therefore isolates the re-encoding contribution of §1.2. The analysis reports $\rho$ beside every
rate number. Only join steps whose `gap_before_s` is structural (§2.1) enter the numerator.

### 5.3 Straightness

$$
\Sigma = \frac{\lVert x_N - x_1 \rVert_2}{\sum_{i=1}^{N-1} \lVert x_{i+1} - x_i \rVert_2} \in (0, 1].
$$

$\Sigma$ is undefined across a real break. Compute it per unbroken run, and report the longest run
with its span.

### 5.4 Within-segment dispersion

This is the $s_j$ of §3.2, averaged per recording.

### 5.5 The KL and the gap on the same axis

`kld_per_t` and `mc_pred_gap` are per-anchor columns of `per_anchor.parquet`, so their window means
are a `groupby`. Carry both, as `trajectory.READOUTS` does. Draw them as the third channel of the
trajectory figure (point size or marker fill). Draw only the unfloored KL
(`source_conditioned_kl_raw` and its per-anchor `kld_per_t`), never `source_conditioned_kl_train`.

### 5.6 The per-recording table

| Column | Meaning |
|---|---|
| `n_points`, `n_segments`, `span_hours` | the population behind the row |
| `mean_step_rate`, `median_step_rate` | §5.2, within-segment steps only |
| `join_within_step_rate_ratio` | $\rho$, §5.2 |
| `straightness`, `straightness_span_hours` | §5.3, on the longest unbroken run |
| `mean_within_window_dispersion` | §5.4 |
| `mean_distance_from_healthy_centroid`, `final_distance_from_healthy_centroid` | §5.1 |
| `mean_kld_per_t`, `mean_mc_pred_gap` | §5.5 |
| the same columns again, with suffix `_stage1` / `_stage2` | §6 |

`cross_subgroup` reads this table through a new entry in `cross_subgroup.METRIC_SOURCES`.

---

## 6. First stage and second stage

### 6.1 The field

`second_stage_onset` is a per-segment scalar that the new dataset pipeline
(`hdf5_dataset/new_pipeline/create_new_pipeline.py`) writes from `second_stage_onset_hours` in the
labour-onset CSV:

$$
\texttt{second\_stage\_onset} = \mathrm{epoch} - 3600 \cdot \texttt{second\_stage\_onset\_hours}.
$$

The value is in seconds: negative before the onset, zero at it, positive after it. It is `NaN` where
the labour-onset table has no row for the recording.

The field is in `eval/configs/eval_overrides.yaml`'s `load_fields`, and
`preflight.REQUIRED_EVAL_LOAD_FIELDS` lists it, so preflight refuses a config without it.
`probe.py` reports how many values are `NaN`. `collect.IDENTITY_COLUMNS` carries it into
`per_sample.csv`, and `cohort.add_second_stage_bins` converts it to hours. The test fixture
(`tests/conftest.py`) writes it.

### 6.2 The stage boundary

The onset in absolute coordinates is constant across a recording:

$$
\theta_g = \mathrm{epoch} - \texttt{second\_stage\_onset}
\qquad\text{(seconds relative to delivery, negative)} .
$$

Resolve it per GUID with the one-value-or-`None` rule of `frames.per_recording_labels`. If a
recording's segments disagree, carry no onset.

An anchor is in the second stage when its own absolute time is at or past the onset:

$$
\boxed{\ \text{anchor } t \text{ of a segment is second-stage} \iff \texttt{second\_stage\_onset} + \Delta t \ \ge\ 0 .\ }
$$

Assign the stage per anchor, not per segment, because a $20$-minute segment can straddle the onset.
A summary window takes a stage only when all its contributing anchors agree. Otherwise the window is
`crossing`, and it is excluded from both stages.

### 6.3 Counts reported with every stage number

* `n_recordings_with_onset` and `n_recordings_missing_onset`, with the missing fraction. Reuse
  `cohort.second_stage_readout` and `cohort.second_stage_eligibility`, which `second_stage` already
  uses.
* Per recording, `n_points_stage1` and `n_points_stage2`.
* The number of recordings that clear the minimum for a paired test.

A recording with fewer than `stats.MIN_GROUP_SIZE` ($3$) points in either stage is excluded and
counted.

### 6.4 The paired stage test

Compute every §5 readout twice per recording, once over its first-stage points and once over its
second-stage points. This gives a paired vector over recordings. Test it with `stats.wilcoxon_paired`
(from `_reuse`). Apply Holm across the readouts as one family, as `cross_subgroup` does across
metrics. Do not emit an unpaired pooled comparison.

The second stage is also the part of the recording nearest delivery, so a stage difference and a
time-before-delivery trend are confounded. `time_to_delivery` and `second_stage` tag their pooled
rows `confounded_by_time` for this reason. As a partial control, also compare each recording's second
stage against the equally long stretch of first stage just before the onset. Emit both comparisons.

---

## 7. Comparing two cohorts' trajectories

The worked example is `hie_no_cs` against `healthy_no_bg_no_cs`. The same method applies to a class
pair (`hie` against `healthy`).

### 7.1 The pair

Both cohorts are `no_cs`, so the pair holds the caesarean axis fixed.

The pair does not hold the blood-gas axis fixed. The dataset builder's `SUBGROUP_META` resolves
`bg` only inside the healthy class: healthy has four subgroups ($\text{bg} \times \text{cs}$), and
acidosis and HIE have two each, with `bg_label` true. The pretraining split is `healthy_bg_cs` and
`healthy_bg_no_cs` only (`cohort.PRETRAINING_SUBGROUPS`). Emit both pairs and label them:

| Pair | bg axis | Pretraining cohort contains |
|---|---|---|
| `hie_no_cs` vs `healthy_no_bg_no_cs` | unmatched | neither side |
| `hie_no_cs` vs `healthy_bg_no_cs` | matched | the healthy side only |

Every statistic runs over recordings. `cohort.cohort_counts` reports segments and recordings per
cohort. At small recording counts, use an exact or permutation $p$-value, and report Cliff's delta
with its Romano magnitude beside every $p$-value.

### 7.2 Three levels of comparison

| Level | Question | Machinery |
|---|---|---|
| (a) Scalar | Do the cohorts' trajectory properties differ? | `cross_subgroup`, unchanged |
| (b) Curve | When do they differ, against time before delivery? | the `time_to_delivery` method, reused |
| (c) Geometric | Do they occupy different regions? | one new permutation test (§7.4) |

**Level (a).** Declare the §5 per-recording table in `cross_subgroup.METRIC_SOURCES` as one more
`MetricSource`. The existing protocol then runs: a Kruskal-Wallis omnibus per metric over all
subgroups, Holm across metrics as one family, and pairwise Mann-Whitney with Cliff's delta on the
survivors only. `pairwise_comparisons` names the pair in the cohort order it receives, least severe
first. For this contrast `left = healthy_no_bg_no_cs` and `right = hie_no_cs`, so a positive
Cliff's delta means that the healthy values are higher. A cohort with fewer than
`stats.MIN_GROUP_SIZE` finite recordings is excluded and recorded.

**Level (b).** Bin on the shared $0.5$ h grid (`cohort.add_time_bins`), reduce to one value per
recording inside each window (`cohort.per_recording_in_bins`), test per window with Holm across
windows, and run pairwise tests on the survivors. The binned columns are the §5 scalars and the PC
coordinates. Use `cohort.TRAJECTORY_BIN_HOURS`; do not add a second binning.

### 7.3 Alignment

| Axis | Landmark at zero | Cost |
|---|---|---|
| Time before delivery (primary) | delivery | left-censoring, §7.3.1 |
| Time from labour onset | onset | `time_from_labor_onset` is `NaN` for part of the cohort |
| Time from second-stage onset | second-stage onset | only recordings with a recorded onset |

Every recording has a delivery time, and the pipeline already bins on that axis, so delivery
alignment is primary. Emit the two onset alignments as secondary views. Fractional time on $[0, 1]$
is out of scope, because it removes recording duration and absorbs the rate into the time axis.

#### 7.3.1 Left-censoring

On the delivery-anchored axis, a bin at $-10$ h contains only recordings monitored for at least ten
hours. The cohort composition can therefore change from bin to bin. Three requirements follow:

* Draw $n_{\text{recordings}}$ per cohort per bin on the figure, not only in the CSV. Do not draw or
  test a bin below `MIN_GROUP_SIZE` for either cohort.
* Report the two cohorts' span distributions (`span_hours`, §5.6).
* Emit a duration-matched view. Restrict both cohorts to the last $X$ hours, where $X$ is the largest
  window in which both clear `MIN_GROUP_SIZE`, and repeat the contrast there.

### 7.4 The geometric comparison

#### Position against time

Per time bin $h$, form each cohort's mean position from one value per recording:

$$
\bar x_A(h) = \frac{1}{n_A(h)} \sum_{g \in A} x_g(h),
\qquad
d(h) = \big\lVert \bar x_A(h) - \bar x_B(h) \big\rVert_2 ,
$$

where $x_g(h)$ is recording $g$'s mean window point in that bin. Compute $d(h)$ in the full $d_z$
dimensions, not in PCA coordinates, because the PCA basis is fitted on the pooled data. Draw in PCA;
test in $d_z$ dimensions.

Standardise by the pooled within-cohort spread:

$$
\hat d(h) = \frac{d(h)}{\sigma_{\mathrm{pooled}}(h)},
\qquad
\sigma^2_{\mathrm{pooled}}(h) = \frac{\sum_{c \in \{A,B\}} \sum_{g \in c} \lVert x_g(h) - \bar x_c(h)\rVert^2}{n_A(h) + n_B(h) - 2} .
$$

The null is a permutation of the cohort labels over recordings, `eval_config.bootstrap_resamples`
times (default $2000$). Recompute $\hat d(h)$ each time and take the bin-wise null band. Permute
recordings, not points, so that each recording's own trajectory stays intact. Apply Holm across bins
as one family, and report the earliest bin whose $\hat d(h)$ survives, beside the per-bin $n$.

#### Motion

Step rate, straightness and within-window dispersion are per-recording scalars, so level (a) covers
them.

#### A whole-curve test

To test whether two cohorts' mean positions differ at all, the recording-level object is a
$d_z$-vector. Hotelling's $T^2$ needs $n > d_z$ to invert the pooled covariance, so it is out of
scope. Use a distance-based test instead: build the $n \times n$ Euclidean distance matrix over
recordings, form a PERMANOVA-style pseudo-$F$ or the energy distance, and get the $p$-value from the
same recording-label permutation. This is the one new statistical routine in this specification.

### 7.5 Cohort against stage

Per recording and per §5 readout, take $\delta_g = m_g^{(\text{stage 2})} - m_g^{(\text{stage 1})}$.
It is defined only for recordings with at least `MIN_GROUP_SIZE` points in both stages. Compare
$\delta$ across cohorts with Mann-Whitney and Cliff's delta, one test per readout, with Holm across
readouts. When the population falls below the minimum, record a skip.

### 7.6 What a cohort comparison carries

* $n_{\text{recordings}}$ per cohort, overall and per time bin;
* the out-of-distribution statement, with the side of the pair it applies to (§7.1);
* the bg-axis confound for any `hie_*` versus `healthy_no_bg_*` contrast;
* the span distributions and the duration-matched result;
* Cliff's delta with its magnitude label beside every $p$-value;
* a note that the tests ran in $d_z$ dimensions while the figure shows three, with
  $\mathrm{EVR}_{1:3}$.

---

## 8. The figures

### 8.1 Colour channels

The figures use the existing palette machinery in `figures_seam`. `_blend(color, amount)` shades a
class colour toward white or black, and `SUBGROUP_TINT_RANGE` uses it for subgroups.

| Channel | Carries |
|---|---|
| hue | clinical class, from `CLINICAL_CLASS_COLORS`, unchanged |
| luminance (via `_blend` along the path) | time before delivery, dark early to light late |
| marker and line style | stage: open marker and thin line for the first, filled marker and heavier line for the second |
| marker size (optional) | $K_t$, §5.5 |

### 8.2 The four figures

**`latent_path.pdf`: one recording.** A $2 \times 2$ panel:

* PC1–PC2 with the luminance ramp and the stage markers, the onset annotated, structural joins as a
  lighter connector and real breaks not drawn;
* PC1–PC3, the same;
* PC1, PC2 and PC3 against hours before delivery, as three stacked lines;
* $K_t$ and `mc_pred_gap` against the same axis.

The default recording is the longest one, as in `trajectory`. The stratified draw of `samples` is
the model for emitting several.

**`latent_path_by_class.pdf`: the cohort overlay.** One panel per stage. Each recording's path is
drawn faintly in its class colour, with the cohort mean path over it. Its guide entry carries the
population statement of §2.3.

**`latent_separation.pdf`: the pair contrast of §7.** Three stacked rows share one x axis of hours
before delivery:

* the standardised separation $\hat d(h)$, with the permutation null band and the surviving bins
  marked;
* $n_{\text{recordings}}$ per cohort per bin, as a stepped line;
* the duration-matched curve, drawn faintly over the full-population curve.

The filename must not end in `_by_clinical_class` or `_by_subgroup`. The runner's grouped fan-out
reserves those stems and would overwrite the figure.

**`latent_features.pdf`: the coordinates.** Two rows: the $d_z \times$ time heatmap of $\mu^q$ (and
of $\Delta\mu$ on the same colour scale), and the $M$ per-group KL curves.

### 8.3 3-D panels

A static PDF does not show depth, so the pairwise planes are the primary panels and the 3-D panel is
an orientation aid. Label them so, and put the plane panels first. Implementation notes:

* `figures.new_figure` returns 2-D axes from `plt.subplots`. A 3-D panel needs
  `fig.add_subplot(..., projection="3d")` on a figure built for it.
* Do not call `figures_seam.style_axes` on a 3-D axes. It sets spine widths and a 2-D grid.
* Set the 3-D margins explicitly instead of with `tight_layout`.

---

## 9. Where this lands in the pipeline

### 9.1 Extract the assembly before writing the second consumer

`analyses/trajectory.py` owns the whole-delivery assembly: the $t_{\mathrm{abs}}$ coordinate, the
overlap averaging, `gap_before_s` and the segment boundaries. The latent path needs the same
assembly at window resolution. An analysis may not import another analysis, so move the assembly
one layer down into `cohort.py`, which already owns the time axis, `SECONDS_PER_HOUR` and the
binning. A possible signature is `assemble_on_absolute_time(frame, *, time_column, value_columns)`.
Repoint `trajectory.whole_delivery` at it. The structural-versus-real break classification (§2.1)
and any trim-offset correction (§2.2) also go there.

### 9.2 The collection-pass change

Compute the window means in `metrics.py`, where `outputs["mu_prior"]`, `outputs["mu_post"]` and the
`contributing` indicator are available. Do not add a second forward pass over the split.

Emit them on `BatchReadout` as $(B, k \cdot d_z)$ arrays, flattened window-major, as the
cross-spectral vectors are. Write them into `per_sample_vectors.npz` under their own prefix, beside
`SPECTRAL_VECTOR_PREFIX`.

### 9.3 Keep them out of `VECTOR_READOUTS`

`VECTOR_READOUTS` drives `aggregate_by_recording`, which takes per-recording means. The
per-recording mean of a trajectory is its centroid, which removes the trajectory. Keep the window
means under their own prefix, and let the analysis do its own grouping.

Two rules that `collect.py` already applies also apply here:

* The vectors sidecar is aligned to `per_sample.csv` by position. A batch without the key must be
  padded with `NaN` rows, never skipped.
* Apply the blanking rule `np.where(scored[:, None], rows, np.nan)`.

Read any CSV that the analysis writes with `float_precision="round_trip"`, and put no absolute path
in anything that reaches `results`.

### 9.4 Files a new analysis touches

1. `eval/analyses/latent_trajectory.py`: the module, layer 2.
2. `eval/run.py`: the registry entry and the launch-table comment block.
3. `eval/EVAL.md`: a `### latent_trajectory` heading, by exact slug equality.
4. `eval/FIGURE_GUIDE.md`: one entry per emitted PDF.
5. `eval/figure_manifest.json`: the committed manifest of a run's figures.
6. `tests/test_eval_latent_trajectory.py`: the analysis tests.
7. `teb_vae/lag_attn_transformer_rws/eval/FIGURE_GUIDE.md` and its `figure_manifest.json`. The
   transformer sibling runs the same registry through the same runner.

The fixture in `tests/conftest.py` already writes `second_stage_onset`, and its epochs use the
$1200$ s stride, so a test can check the structural-break classifier of §2.1.

### 9.5 Offline re-runnability

All inputs except the window means come from files. This command must work with no checkpoint and
no GPU:

```bash
python -m teb_vae.lag_attn_rws.eval.run --output-dir <a finished run> --only latent_trajectory
```

It reads `per_sample.csv`, `per_sample_vectors.npz` and `per_anchor.parquet`. The analysis
therefore has no layer-1 import: no `metrics`, no `collect`, no task and no model.

`cross_subgroup` is the last entry in the `run.py` registry and reads finished per-recording CSVs
from disk. Register `latent_trajectory` before it, for example beside `trajectory`. Otherwise its
`MetricSource` entries resolve to a file that does not exist yet, and `cross_subgroup` records them
as missing.

---

## 10. Order of work

1. **Plumbing.** Add the window means in `metrics.py` and write them to the vectors sidecar under
   their own prefix. Test that the sidecar is row-aligned, that a window with no contributing
   anchors is `NaN`, and that the stage boundary reproduces the anchor rule of §6.2. This is the
   only step that needs a new collection pass.
2. **Shared assembly.** Extract `assemble_on_absolute_time` into `cohort.py`, add the
   structural-versus-real break classification, and repoint `trajectory.whole_delivery` at it. Test
   that the `trajectory` outputs do not change, that a synthetic $1200$ s-stride recording gives
   only structural joins, and that a dropped segment gives a real break.
3. **Projection-free readouts.** Write the §5 per-recording CSV, with $\rho$. No figure yet.
4. **Cohort contrast, level (a).** Declare the step-3 CSV in `cross_subgroup.METRIC_SOURCES`
   (§7.2), with a test.
5. **PCA and the primary figure.** Implement §4.2 and §8.2, with the recorded basis and the
   $\mathrm{EVR}$ triple.
6. **Stage split and paired test.** Implement §6, with the matched-window control.
7. **Curve and geometric contrasts.** Implement §7.3 and §7.4: the per-bin separation with its
   permutation null, the per-bin $n$, the duration-matched view and the distance-based test.
8. **Second-order views.** Add the per-head-group projections, `kld_per_t_per_head` on the
   per-anchor table, the cross-fitted supervised axis, the 3-D panel and the cohort-by-stage
   comparison of §7.5.

Step 1 blocks all other steps. Steps 4 and 5 are independent of each other.
