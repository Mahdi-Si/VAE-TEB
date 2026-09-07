# The latent trajectory through labour

How to summarise, project, draw and *test* the path a recording's latent state takes from the
start of monitoring to delivery — and how to split that path at the second-stage onset.

This is a design note, not a contract. `EVAL.md` and `FIGURE_GUIDE.md` are bound to the code by
test; this document is bound to nothing yet, and it exists to be argued with before any of it is
built. Everything below is stated against the shipped geometry: $T = 300$, $T_{\mathrm{valid}} =
270$, $w = 30$, $H = 30$, $L = 91$, $\Delta = 4$ s, $d_z = 48$, $M = 4$ heads of $g = 12$
coordinates.

---

## Table of contents

0. [What is being asked, and what exists today](#0-what-is-being-asked-and-what-exists-today)
1. [What a trajectory is here, and what it is not](#1-what-a-trajectory-is-here-and-what-it-is-not)
2. [The geometry that will masquerade as physiology](#2-the-geometry-that-will-masquerade-as-physiology)
3. [Summarising a segment: the window scheme](#3-summarising-a-segment-the-window-scheme)
4. [From 48 dimensions to a picture](#4-from-48-dimensions-to-a-picture)
5. [The readouts that need no projection](#5-the-readouts-that-need-no-projection)
6. [First stage and second stage](#6-first-stage-and-second-stage)
7. [Comparing two cohorts' trajectories](#7-comparing-two-cohorts-trajectories)
8. [The figures](#8-the-figures)
9. [Where this lands in the pipeline](#9-where-this-lands-in-the-pipeline)
10. [How this output will be misread](#10-how-this-output-will-be-misread)
11. [Suggested order of work](#11-suggested-order-of-work)

---

## 0. What is being asked, and what exists today

The request is four things, and they have different costs:

| Ask | Status today |
|---|---|
| $K_t$ against absolute time, per recording | **Done.** `analyses/trajectory.py` assembles it on $t_{\mathrm{abs}}$, with `pred_gap` beside it. |
| $K_t$ binned by time-before-delivery, by class, with tests | **Done.** `analyses/time_to_delivery.py`, on a $0.5$ h grid with Holm across windows. |
| The **latent** path — where $\mu^q_t$ goes, drawn in 2-D/3-D | **Absent, and no artifact carries the input.** |
| First stage versus second stage | **Absent, and the field is not loaded.** |

So the KL half of the request is largely satisfied and the work is the other two. Both are
blocked on the same fact, and it is worth stating plainly before any design: **no file a run
leaves behind carries a latent mean.** `per_sample.csv` carries `mu_prior_rms`, `delta_mu_rms`
and `mu_post_prior_gap_sq` — *norms*, not vectors. `per_anchor.parquet` carries $K_t$, both score
pairs, `argmax_lag`, `coverage` and `seconds_since_contraction`. `per_sample_vectors.npz` carries
the per-dimension KL, four lag profiles and the cross-spectral sums. `retained_arrays.npz` carries
waveforms and attention weights under `caps`, and `RETAINED_QUANTITIES` names no latent tensor.

The single load-bearing change in everything that follows is therefore **one addition to the
collection pass**. Every other decision here is downstream of it and can be revised offline
against the tables; that one cannot, because it costs a re-collection.

### The latent is a deliverable, and this is the analysis that treats it as one

`MODEL_DOCUMENTATION.md` §11.2 already says so:

| Quantity | Meaning |
|---|---|
| $\mu^p_t$ | the target-only predictive state of the fetal heart rate at $t$ |
| $\mu^q_t - \mu^p_t$ | the additional predictive information the source contributed |
| $\mu^q_t$ | the complete state after the source has been incorporated |

Because there is **no decoder bypass** — the decoder's forward accepts exactly one tensor — the
latent is not a side-channel code that happens to correlate with the forecast. It *is* the
forecast's sufficient statistic within this model family. A path through $\mathbb{R}^{48}$ is
therefore a path through "what this model believes the next two minutes of the heart rate will
look like", and that is what makes drawing it worth doing at all. The pipeline has fifteen
analyses that reduce that belief to a scalar; none of them looks at the belief.

Use the **means**, never the samples $z^p_t, z^q_t$. That table states it, and the reason is that the
samples carry the reparameterisation noise, so two runs of one checkpoint would draw two different
paths and the difference would be $\epsilon$.

Store $\mu^p$ and $\mu^q$, and **not** $\Delta\mu$ separately. The posterior is a bounded residual
on the prior in the prior's own coordinates (model documentation §5.6), so $\Delta\mu = \mu^q -
\mu^p$ holds exactly
and a third stored array is a third thing that can disagree with the other two.

---

## 1. What a trajectory is here, and what it is not

Three properties of this architecture decide what the picture can mean. All three cut against the
naive reading, and a document that draws the path without stating them is drawing a dynamical
system that does not exist.

### 1.1 There is no latent transition density

The model documentation's §4.6 is explicit: this is **not** a latent state-space model. There is no
$p(z_t \mid z_{t-1})$,
no filtering recursion, and the objective factorises over anchors. $z_t$ and $z_{t+1}$ are
correlated because $H^y_t$ and $H^y_{t+1}$ are — through the dilated convolution stack's fixed
receptive field and the LSTM's carried state — not because the latents are linked.

Consequence: the "trajectory" is a **sequence of independent readouts of a smoothly varying
input**, not an orbit. Its smoothness is inherited from the encoders, so smoothness is not
evidence of anything. What *is* meaningful is where the sequence goes, how fast, and whether the
speed or the direction changes at a clinical landmark. Velocity and displacement are the readable
quantities; "the dynamics" is not.

### 1.2 The encoder resets at every segment boundary

Each 20-minute segment is a separate forward pass. The LSTM state starts from zero, the
convolution stack's left padding is zeros, and this is exactly why $w = 30$ anchors of warm-up
exist. So the latent at the start of segment $k$ does **not** continue from the end of segment
$k-1$; it is a fresh encoding of a fresh window.

Consequence: a whole-delivery path is $N$ short arcs stitched together, and every stitch carries a
re-encoding discontinuity that is a property of the harness rather than of the fetus. §5.2 below
turns that liability into a measurement — the join gives a free estimate of the re-encoding noise
floor, and any claim about trajectory speed has to clear it.

### 1.3 The coordinates are the decoder's, and are not orthogonal, ordered or scaled

$\mu^p$ is tanh-bounded into $[-5, 5]^{48}$ and $\Delta\mu$ into $[-3, 3]^{48}$, so the state
lives in a bounded box and Euclidean distance is well-behaved. But nothing orthogonalises the
coordinates, nothing orders them by importance, and a collapsed dimension sits at a near-constant
value rather than at zero. There is no canonical "first three dimensions". A projection is
mandatory, and §4 is about choosing it honestly.

One piece of structure *is* canonical and is worth exploiting: the posterior is **head-structured**
— latent group $m$ (coordinates $12m \dots 12m+11$) is written only by attention head $m$. That
partition is architectural, so a per-group view of the path is a decomposition rather than an
arbitrary slice, and it is the natural answer to "how the latent *features* change".

Read `mu_prior_sat_frac` and `delta_mu_sat_frac` before quoting any distance. A state pinned on
the tanh bound has its motion clipped, and a flat stretch of trajectory would then be saturation
rather than stability. `latent`'s `prior_variance_not_pinned` verdict is the companion check for
anything KL-shaped.

---

## 2. The geometry that will masquerade as physiology

Six structural facts. Each produces a feature in the drawn path that a reader will attribute to
labour. They are stated with their exact arithmetic because approximations here become "the state
moves faster near delivery".

| # | Fact | What it looks like on the path |
|---|---|---|
| 1 | Anchors $t < w = 30$ carry no loss term | the first $120$ s of every segment is an untrained transient |
| 2 | Anchors $t \ge T - H = 270$ are never decoded | the last $120$ s of every segment does not exist |
| 3 | Lag support is truncated until $t \ge L - 1 = 90$ | $\Delta\mu$ and $K_t$ are biased over the first $6$ min of every segment |
| 4 | Segment stride is $1200$ s, trained span is $960$ s | a **$244$ s hole at every segment join**, in every recording |
| 5 | `epoch` is not trim-corrected | every $t_{\mathrm{abs}}$ is $60$ s early, uniformly |
| 6 | The coverage floor drops whole anchors | a signal-loss stretch removes points without removing time |

### 2.1 The trained anchor set, and the join arithmetic

The trained set is $\mathcal{A} = [30, 270)$: $240$ anchors, $960$ s, $16$ minutes of every
$20$-minute segment. On the absolute axis $t_{\mathrm{abs}} = \mathrm{epoch} + 4t$ a segment
starting at $e$ contributes $[e + 120,\ e + 1076]$.

The dataset writes stored segments of $5280$ raw samples ($22$ min) at a step of $4800$ samples
($1200$ s), and the loader's symmetric $1$-minute trim removes exactly the $2$-minute stored
overlap. So consecutive segments' *trimmed* grids abut, and consecutive segments' *trained* spans
do not:

$$
\underbrace{e + 1076}_{\text{last trained anchor of segment } k}
\quad\longrightarrow\quad
\underbrace{e + 1200 + 120 = e + 1320}_{\text{first trained anchor of segment } k+1},
\qquad
\text{gap} = 244\ \mathrm{s}.
$$

Three things follow, and the third is the one that matters.

* **A trained-anchor delivery is $80\%$ covered by construction** ($960 / 1200$). The missing fifth
  is warm-up plus undecodable tail, not signal loss.
* **`trajectory.whole_delivery`'s overlap averaging is defensive rather than routinely exercised.**
  Trimmed segments abut; they do not overlap. `n_overlapping` will normally be $0$, and that is
  correct behaviour rather than a bug.
* **`BREAK_TOLERANCE_S = 6.0` s marks every segment join as a break.** It cannot distinguish the
  structural $244$ s join from a real missing segment. For a latent path that distinction is the
  whole difference between "the recording continues" and "an hour is missing", so a latent
  trajectory must classify a join as **structural** when `gap_before_s` lands within one anchor of
  the expected $\mathrm{stride} - (T_{\mathrm{valid}} - w)\Delta = 1200 - 960 = 240$ s, and as a
  **real break** otherwise. Draw the first as a lighter connector and the second not at all.

### 2.2 The $60$-second offset, stated rather than silently fixed

`epoch` is the start of the **stored** $22$-minute segment. The loader trims one minute from each
end of the signal arrays and passes `epoch` through untouched, so anchor $0$ of the trimmed grid is
at absolute time $\mathrm{epoch} + 60$ s, while `trajectory.whole_delivery` places it at
$\mathrm{epoch}$.

It is a **constant** offset. Every point moves by the same minute, so no shape, no distance, no
speed and no cohort contrast changes. Two things do inherit it: the hours-before-delivery axis, and
the second-stage boundary of §6. One minute against a second stage lasting $0.4$–$6$ h is
immaterial, which is why the recommendation is to *state* it rather than to fix it in a new
analysis and leave `trajectory` disagreeing by a minute with its neighbour. If it is fixed, fix it
in the shared assembly (§8.1) so the two cannot diverge.

### 2.3 The population caveat that applies to every cohort statement

The checkpoint trains on `pre_training_dataset/*`, which is healthy-**with-background** only. On
the holdout split that makes ACIDOSIS, HIE and both healthy no-background subgroups
out-of-distribution. A latent trajectory of an HIE recording is therefore partly a picture of the
encoder extrapolating, and a class separation visible in the projection is not evidence of
clinical discrimination. `cohort.build_cohort_block` already computes and carries this; the
analysis must repeat it in its own record rather than assume a reader will look.

---

## 3. Summarising a segment: the window scheme

The request phrases it exactly right — *"summarize the 20 minute segments into couple of points"*.
The question is how many, where the boundaries go, and what is averaged.

### 3.1 Four windows, and why the boundaries are 30 / 90 / 150 / 210 / 270

Split $\mathcal{A} = [30, 270)$ into four windows of $60$ anchors:

| Window $j$ | Anchors | Duration | Lag support |
|---|---|---|---|
| $0$ | $[30, 90)$ | $240$ s | **truncated** — every anchor has $t < L - 1$ |
| $1$ | $[90, 150)$ | $240$ s | full |
| $2$ | $[150, 210)$ | $240$ s | full |
| $3$ | $[210, 270)$ | $240$ s | full |

The boundary at $90$ is not a round number chosen for tidiness: $L - 1 = 90$ is exactly the anchor
at which the attention window stops being truncated, so **window $0$ is precisely the
truncated-support region and windows $1$–$3$ are precisely the clean one**. Every source-dependent
quantity — $\Delta\mu$, $K_t$, the lag profile — is biased short in window $0$ and unbiased in the
rest. A four-window split with any other boundary mixes the two.

Practical consequence: report every window, and make window $0$ **excludable by a single filter**
in every downstream statistic. The recommended default is to keep it in the drawn path (dropping
it would leave a $240$ s hole per segment on top of the $244$ s join) and to exclude it from every
$\Delta\mu$-derived or KL-derived number.

Window centres land at $t_{\mathrm{abs}} = e + 238 + 240j$, i.e. $e + 238,\ 478,\ 718,\ 958$. So:

$$
\underbrace{240\ \mathrm{s}}_{\text{within a segment}}
\qquad\text{versus}\qquad
\underbrace{(e + 1438) - (e + 958) = 480\ \mathrm{s}}_{\text{across a join}} .
$$

**The join step is exactly twice the within-segment step in time.** That exact factor of two is
what makes §5.2's noise-floor test clean, and it is the reason to fix $k = 4$ rather than treat it
as a tunable: at $k = 4$ the two spacings are commensurate, and at $k = 3$ or $k = 5$ they are not.

### 3.2 What is averaged, and what is `NaN`

Per window $j$ of each segment, over the window's **contributing** anchors only — the same
indicator the reconstruction and the KL support are derived from (§9.3 of the model
documentation), not "every anchor in the range":

$$
\bar\mu^{q}_{j} = \frac{\sum_{t \in W_j} c_t\, \mu^q_t}{\sum_{t \in W_j} c_t},
\qquad
\bar\mu^{p}_{j} = \frac{\sum_{t \in W_j} c_t\, \mu^p_t}{\sum_{t \in W_j} c_t},
\qquad
n_j = \sum_{t \in W_j} c_t .
$$

Three rules, each with a named failure it prevents:

* **The latent exists at every anchor; the mean must not.** $\mu^q_t$ is produced for anchors the
  coverage floor rejected and for anchors inside a signal-loss gap, where the encoder is reading
  $0$ bpm at roughly $-11\sigma$. Averaging those in produces a state excursion that looks like a
  physiological event and is a gap.
* **$n_j$ travels with every point.** A window backed by $3$ of $60$ anchors is a different object
  from one backed by $60$, and only the count says so.
* **$n_j = 0$ is `NaN` for that window, not for the segment, and never $0.0$.** This is the same
  rule the collection pass applies to a segment that scored no anchors, applied one level down. A
  zeroed latent is the origin of the space, which on a centred projection is the population mean —
  the most plausible-looking wrong answer available.

Carry one more scalar per window, because it is what says whether a window mean is a summary at
all: the **within-window dispersion**

$$
s_j = \sqrt{\frac{\sum_{t \in W_j} c_t \lVert \mu^q_t - \bar\mu^q_j \rVert^2}{\sum_{t \in W_j} c_t}} .
$$

If $s_j$ is comparable to the distance between consecutive window means, the path is a random walk
inside a cloud and the connecting line is decoration. That comparison is the first thing to
compute and the first thing to report.

### 3.3 What it costs

Per segment, at $k = 4$ windows, $d_z = 48$, two latents, `float64` (the sidecar's dtype):

$$
4 \times 48 \times 2 \times 8\ \mathrm{B} = 3{,}072\ \mathrm{B} \approx 3\ \mathrm{KiB},
$$

plus $4 \times 2$ scalars for $n_j$ and $s_j$. That is about **$0.15\%$ of one retained waveform**
($\approx 2.0$ MiB per sample) and about $3$ MiB per thousand segments. It is small enough to be
**unconditional** — no `caps` entry, no retention plan, retained for every segment of the split —
which is the right call: a cap here would mean the trajectory figure silently describes a seeded
subsample of recordings while every neighbouring analysis describes all of them.

For contrast, the full per-anchor latent is $240 \times 48 \times 2 \times 4$ B $= 90$ KiB per
segment in `float32`, i.e. $88$ MiB per thousand segments. That is retention-plan territory, and
the recommendation is **not** to take it now:

```text
lean-limit: the latent path is summarised at 4-minute windows; retain the full per-anchor
mu under a `caps.latents` entry when an analysis needs sub-window structure — e.g. a
contraction-triggered latent response, which needs anchor resolution around each onset.
```

That upgrade condition is real rather than hypothetical: `seconds_since_contraction` is already on
the per-anchor table, so a contraction-locked latent average is one retention away.

---

## 4. From 48 dimensions to a picture

### 4.1 What a projection has to satisfy *here*

Not "which embedding is best" in the abstract — five requirements come from this pipeline, and
they eliminate most of the field before any comparison of embedding quality:

1. **Out-of-sample.** A second checkpoint, a second arm, a second cohort must be projectable into
   the *same* picture, or the architecture comparison this repository exists to make cannot be
   drawn.
2. **Deterministic.** Two runs of one checkpoint at one seed compare byte-identical on `results`.
   A stochastic objective breaks that contract.
3. **Metric.** §5's readouts are distances and speeds. An embedding whose distances are not
   distances turns them into decoration.
4. **No new dependency.** `EVAL.md`'s dependency section says "no new ones", and `scikit-learn` is
   not among them (it appears once in a sibling script and nowhere in `lag_attn_rws`).
5. **Recording-fair.** The healthy cohort contributes roughly an order of magnitude more segments
   than HIE. A basis fitted on pooled points is a basis fitted on the healthy cohort.

### 4.2 Recommended: recording-weighted PCA, by SVD, with a recorded basis

Fit once, on the whole split, on the per-(segment, window) posterior means $m_i \in
\mathbb{R}^{48}$, with weights $w_i = 1 / n_{g(i)}$ where $n_g$ is the number of points recording
$g$ contributed, normalised so $\sum_i w_i = 1$. Then

$$
\bar m = \sum_i w_i m_i,
\qquad
A_{i\cdot} = \sqrt{w_i}\,(m_i - \bar m),
\qquad
A = U S V^{\!\top},
$$

and the components are the first three columns of $V$, with

$$
\mathrm{EVR}_j = \frac{s_j^2}{\sum_{j'} s_{j'}^2}.
$$

Nine lines of `numpy.linalg.svd`. Five decisions inside it, each with a reason:

**Weight by $1/n_g$, not uniformly.** This is `frames.per_recording_means`' rule — "a recording
contributing thirty-seven segments and one contributing two count the same" — applied to the basis
rather than to a mean. Weighting uniformly fits the basis to whichever recordings are longest, and
the longest recordings are the ones with the cleanest signal.

**Fit on per-window points, not on per-recording means.** The within-recording variation *is* the
object being drawn. Fitting on recording centroids finds the directions in which recordings differ
from each other, which is a cohort axis, not a trajectory axis.

**Covariance, not correlation.** The $48$ coordinates share one unit and one bound. Standardising
each to unit variance would promote a collapsed dimension — near-constant, tiny variance, no
information — to equal footing with a live one, which is exactly backwards.

**Pin the sign.** SVD signs are arbitrary and flip between runs and between platforms. For each
component $v_j$ let $d^\star = \arg\max_d |v_{j,d}|$ (lowest $d$ on a tie) and negate $v_j$ when
$v_{j,d^\star} < 0$. Without this the same checkpoint draws a mirror-image figure on a rerun and
nothing fails.

**Record the basis.** Write $\bar m$ (48 floats) and $V_{1:3}$ ($48 \times 3$) to the analysis's
own CSV, beside the three $\mathrm{EVR}$ values. That is what makes requirement 1 real: a second
run is projected into the first's basis by reading a file, not by refitting and hoping the two
agree.

**Report $\mathrm{EVR}_{1:3}$ next to the figure, always.** If the first three components carry
$30\%$ of the variance, the picture is showing $30\%$ of the motion and the parts of the path that
appear to cross may be far apart. This is the single number that decides whether the 3-D figure is
evidence or an illustration.

Fit on $\mu^q$ (the complete state). Project $\mu^p$ into the *same* basis rather than fitting a
second one — the two live in one coordinate system by construction (model documentation §5.6), so a
shared basis makes
$\Delta\mu$ a visible arrow from the prior point to the posterior point on the same axes. Two
separately fitted bases would make that arrow meaningless.

### 4.3 The per-head-group view — "how the latent features change"

The head-structured posterior gives four disjoint blocks of $12$ coordinates, and group $m$ is
written by attention head $m$ alone. So a second, architecture-grounded view costs one more fit:
project each group's $12$ coordinates onto its own first two components, and draw four small
panels. A finding of the form "group $2$ moves in the second stage and the others do not" is
directly interpretable — it says *which lag structure* the change came through, because head $2$'s
attention profile is a separate, already-emitted readout (`analyses/attention.py`).

The scalar companion is the per-group KL $K^{(m)}_t$, which the model already computes as
`kld_per_t_per_head` and the trainer already logs. It is **not** on the per-anchor table today;
adding it is four `float32` columns over roughly $240$ rows per segment ($\approx 38$ MB per
$10^4$ segments) and it is the cheapest way to get "which head carried the change, when".

### 4.4 A supervised axis, only if it is cross-fitted

A "labour axis" — the direction in latent space that best predicts hours-before-delivery — is far
more interpretable than PC1, and it is the natural axis for the figure the request describes.

It is also the easiest way to draw a circular result. **An axis fitted to predict time, applied to
the same points, will show a trajectory that trends with time whether or not one exists.** The only
honest construction is the one `sufficiency` already uses: split at **GUID** level into two halves,
fit ridge on half A and project half B, fit on B and project A, and every plotted coordinate is an
out-of-fold prediction. Assert the disjointness at runtime rather than testing it once.

Ship it with its **held-out $R^2$**. An axis that predicts nothing out of fold is a random
direction, and a path along a random direction is noise drawn as a trend. Below a stated threshold
the analysis should record a skip rather than emit the panel.

Recommendation: build PCA first and treat the supervised axis as a second sprint. PCA answers "does
the state move at all, and how much of the motion is in three dimensions"; the supervised axis
answers "does it move *toward delivery*", which is only worth asking once the first answer is yes.

### 4.5 What is rejected, and why

| Method | Why not here |
|---|---|
| **t-SNE, UMAP** | No out-of-sample map, so no two runs share a picture (requirement 1). Stochastic, so no byte-stability (2). Distances are not distances and the "path" is an artifact of the neighbourhood graph (3). UMAP is not installed (4). t-SNE's perplexity and UMAP's `n_neighbors` are exactly the kind of knob `EVAL.md` keeps out of `eval_config` — an operator could tune until a separation appeared. |
| **Diffusion maps, PHATE** | Designed for trajectories, and that is the problem: they *assume* the samples lie on a manifold traversed at varying density, which is the hypothesis under test. Same out-of-sample and dependency objections. |
| **An autoencoder of the latent** | A second learned model, with its own training run, its own collapse modes and its own checkpoint, introduced to explain the first one. |
| **Raw coordinate triples** ($\mu_0, \mu_1, \mu_2$) | Cheap and tempting. The coordinates are not ordered by importance, so this draws three arbitrary directions and calls them the latent space. |
| **Per-recording PCA** | Every recording gets its own basis, so no two paths can be laid side by side and every cohort comparison is between different spaces. |

The rejection of the nonlinear methods is not a claim that they would find nothing. It is that what
they found could not be checked, compared across arms, or turned into the per-recording scalars of
§5 — and this pipeline's currency is per-recording scalars.

---

## 5. The readouts that need no projection

**This is where the science is.** A projection is for looking; these are for testing. Each is a
per-recording (or per-recording-per-stage) scalar, which is the unit the whole aggregation chain
and `cross_subgroup` already consume — so each can be bootstrapped over recordings and tested
without any new statistical machinery. All are rotation-invariant, so none depends on §4's choices.

Let $x_1, \dots, x_N$ be a recording's window means in chronological order, at times $\tau_1 <
\dots < \tau_N$.

### 5.1 Displacement from the healthy centroid

$$
D_i = \lVert x_i - c_{\mathrm{healthy}} \rVert_2,
\qquad
c_{\mathrm{healthy}} = \operatorname{mean}_{g \in \text{healthy}} \big(\operatorname{mean}_i x_i^{(g)}\big),
$$

the outer mean over healthy **recordings** (not segments), and leave-one-out when the recording
being scored is itself healthy. A one-dimensional "how far from normal is this belief" curve
against time before delivery, with no projection anywhere in it.

Caveat to state in the record: a dimension that is inactive but sits at a large constant offset
contributes a constant to every $D_i$. That shifts the whole curve and leaves its shape intact, so
it is harmless for trends and matters for absolute values.

```text
lean-limit: Euclidean distance in the raw latent box; replace with Mahalanobis against the
healthy cohort's ridge-regularised covariance when the healthy cohort carries more than
2 * d_z = 96 recordings, below which the 48x48 covariance is not estimable.
```

### 5.2 Step rate, and the free noise floor

$$
v_i = \frac{\lVert x_{i+1} - x_i \rVert_2}{\tau_{i+1} - \tau_i}\ \ \big[\text{latent units per second}\big].
$$

Normalising by $\Delta\tau$ is not optional: §3.1 showed the step is $240$ s within a segment and
$480$ s across a join, so an unnormalised speed is $2\times$ larger at every join by construction.

And that same fact is the gift. Within a segment, consecutive windows come from **one** encoder
pass; across a join they come from **two**. So

$$
\boxed{\ \rho = \frac{\operatorname{median}\ v_i\ \text{across joins}}{\operatorname{median}\ v_i\ \text{within segments}}\ }
$$

isolates the re-encoding contribution of §1.2. If $\rho \approx 1$ the encoder reset costs nothing
and the whole-delivery path is a single object. If $\rho \gg 1$, the excess is the noise floor, and
**any trajectory motion smaller than it is not a measurement.** Report $\rho$ beside every speed
number, in the same way `coherence` reports both its estimators side by side so the size of its bias
is measured on every run rather than assumed.

Only join steps whose `gap_before_s` is *structural* (§2.1) enter the numerator. A step that
bridges a missing hour is not a step.

### 5.3 Straightness

$$
\Sigma = \frac{\lVert x_N - x_1 \rVert_2}{\sum_{i=1}^{N-1} \lVert x_{i+1} - x_i \rVert_2} \in (0, 1].
$$

Near $1$ the state marches; near $0$ it wanders in place. This distinguishes "the belief drifts
steadily toward delivery" from "the belief is agitated", which is a distinction a speed cannot
make and which is plausibly the clinically interesting one. It is undefined on a path with a real
break, so compute it per unbroken run and report the longest, with the run's own span.

### 5.4 Within-segment dispersion

The $s_j$ of §3.2, averaged per recording. It is both a diagnostic (is the window mean a summary?)
and a readout in its own right: a rising within-segment dispersion as delivery approaches is
"the belief is becoming less stable over four minutes", which is a different statement from
"the belief is moving".

### 5.5 The KL and the gap, on the same axis

$K_t$ and `mc_pred_gap` are already per anchor on `per_anchor.parquet`, so their window means are a
`groupby` and cost nothing. Carry both, as `analyses/trajectory.py`'s `READOUTS` already does, and
for its stated reason — the KL is inflated by an arbitrary factor whenever the prior variance sits
on its clamp, and `pred_gap` is not.

The recommended encoding is to make them the **third channel of the trajectory figure**: point size
or marker fill from $K$, so "where in latent space is the source most informative" is one figure
rather than two that a reader has to align by eye. Only the unfloored `source_conditioned_kl_raw`
may be drawn — never `source_conditioned_kl_train`.

### 5.6 Summary of the emitted per-recording table

| Column | Meaning |
|---|---|
| `n_points`, `n_segments`, `span_hours` | the population behind the row |
| `mean_step_rate`, `median_step_rate` | §5.2, within-segment only |
| `join_within_step_rate_ratio` | $\rho$, §5.2 — the noise floor |
| `straightness`, `straightness_span_hours` | §5.3, on the longest unbroken run |
| `mean_within_window_dispersion` | §5.4 |
| `mean_distance_from_healthy_centroid`, `final_distance_...` | §5.1 |
| `mean_kld_per_t`, `mean_mc_pred_gap` | §5.5 |
| … and the same block again, suffixed `_stage1` / `_stage2` | §6 |

That table is what `cross_subgroup` reads through `METRIC_SOURCES`, which is what makes "do the
cohorts' trajectories differ" a question the pipeline already knows how to answer properly.

---

## 6. First stage and second stage

### 6.1 The field exists, and it is not currently loaded

`second_stage_onset` is a per-segment scalar written by the **new** dataset pipeline
(`hdf5_dataset/new_pipeline/create_new_pipeline.py`), from `second_stage_onset_hours` in the
labour-onset CSV:

$$
\texttt{second\_stage\_onset} = \mathrm{epoch} - 3600 \cdot \texttt{second\_stage\_onset\_hours},
$$

in seconds, with `epoch`'s sign convention: **negative before the onset, zero at it, positive
after.** `NaN` where the recording is absent from the table. It is the exact analogue of
`time_from_labor_onset`, which the evaluation already loads.

It is **not** in `eval/configs/eval_overrides.yaml`'s `load_fields`, so it does not reach a batch,
so it is on no table. Adding it is one list entry — and the loader **skips a field a shard does not
carry, silently**, so presence must be asserted after the first batch rather than assumed from the
list. That is exactly what `probe.py` and the loader probe already do for the other five.

### 6.2 It must not become a preflight refusal

`preflight.REQUIRED_EVAL_LOAD_FIELDS` raises when a field is missing, and the temptation is to add
a sixth entry. **Do not.** Shards built by the legacy pipeline genuinely do not carry the field,
and refusing an entire evaluation — every score, every verdict, the acceptance gate — because one
optional figure cannot be split by stage is the wrong trade.

The right shape is the one `band_partition` already uses for a shard with no `sel_*` attributes:
record the absence and continue. Concretely — add it to `load_fields`, add an **informational**
entry to `preflight.json` saying whether it arrived, and have the stage half of the analysis emit
a recorded `skipped` with a reason naming the pipeline that writes it. The trajectory itself still
draws; only the colour-by-stage and the paired test are absent.

### 6.3 The stage boundary, exactly

The onset in absolute coordinates is recoverable per segment and is constant across a recording:

$$
\theta_g = \mathrm{epoch} - \texttt{second\_stage\_onset}
\qquad\text{(seconds relative to delivery, negative)} .
$$

Resolve it per GUID with the **one-value-or-`None`** rule `frames.per_recording_labels` uses: if a
recording's segments disagree, carry no onset rather than the first or the commonest. A
disagreement means the field was mis-joined or the GUID appears in two shards, and inventing a
value hides a fault the loader probe would otherwise raise on.

An anchor is in the second stage iff its own absolute time is at or past the onset, which reduces
to a test on quantities the tables already carry:

$$
\boxed{\ \text{anchor } t \text{ of a segment is second-stage} \iff \texttt{second\_stage\_onset} + \Delta t \ \ge\ 0 .\ }
$$

**Assign the stage per anchor, not per segment.** A $20$-minute segment can straddle the onset, and
labelling by segment start puts up to $20$ minutes of second stage into the first. Per anchor it is
exact and costs one column.

A summary window then takes a stage only when **all** its contributing anchors agree; otherwise it
is `crossing`. At $240$ s per window, at most one window per recording is `crossing`, and it is
excluded from both stages rather than assigned to either — it is the one point that is genuinely
both.

### 6.4 What must be reported before any stage claim

Second stage is short. In the labour-onset table's own sample it runs $0.42$–$6.25$ h before
delivery, so a recording contributes between **one and nineteen segments** to it — four to
seventy-six window points, against potentially hundreds in the first stage. Three counts therefore
travel with every stage number:

* `n_recordings_with_onset` and `n_recordings_missing_onset`, with the missing fraction — the
  `cohort.labor_onset_readout` pattern, which should be **generalised to take a column name**
  rather than copied, since it will then serve both fields;
* per recording, `n_points_stage1` and `n_points_stage2`;
* how many recordings clear the minimum for a paired test.

A recording with fewer than `stats.MIN_GROUP_SIZE` points in either stage is **excluded and
counted**, never entered — the pipeline's standing rule, and here it bites hard, because the
recordings with the shortest second stage are not a random sample of recordings.

### 6.5 The test that makes it a finding

Every §5 readout, computed twice per recording — once over its first-stage points, once over its
second-stage points — gives a **paired** vector over recordings, and `stats.wilcoxon_paired` is
already in `_reuse`. Paired is the right design and matters: it removes the between-recording
variance that would otherwise swamp a stage effect, and each recording is its own control.

Holm across the readouts as one family, in the way `cross_subgroup` adjusts across metrics. The
unpaired alternative — pooling all first-stage points against all second-stage points — is
anticonservative by the recording count *and* confounded by which recordings happen to have a long
second stage, and should not be emitted at all.

One confound to state rather than adjust away: **the second stage is also the part of the recording
nearest delivery**, so a stage difference and a time-before-delivery trend are not separable on
this data. `time_to_delivery` already flags its `pooled` row `confounded_by_time` for the same
reason. The honest partial control is to compare each recording's second stage against **the
equally long stretch of first stage immediately preceding the onset**, which holds the distance to
delivery roughly fixed and turns "second stage differs" into "second stage differs from the hour
before it". Emit both; the matched one is the one to quote.

---

## 7. Comparing two cohorts' trajectories

The worked example is the one asked for: **`hie_no_cs` against `healthy_no_bg_no_cs`**. Everything
below is written for a subgroup pair, and applies unchanged to a class pair
(`hie` against `healthy`) with larger $n$ on both sides.

### 7.1 Read the pair before comparing it

That specific pair is a better-chosen contrast than it looks, and it also carries a confound that
the cohort table cannot resolve. Both facts should be stated in the record rather than discovered
by a reader.

**It holds the caesarean axis fixed.** Both cohorts are `no_cs`, so mode of delivery is not free to
explain a difference. That is the right way to pick a pair here, and it is the reason to prefer it
over the bare class contrast.

**It does not hold the blood-gas axis fixed, and cannot.** `labels.CANONICAL_SUBGROUPS` resolves
`bg` only inside the healthy class: healthy has four subgroups ($\text{bg} \times \text{cs}$), while
acidosis and HIE have two each (`cs` only). So `hie_no_cs` is *unresolved* on the background axis —
in practice largely background-present, since blood gas is how the diagnosis is confirmed — and
`healthy_no_bg_no_cs` is explicitly background-absent. The contrast therefore mixes the class
difference with a recording-context difference, and no stratification available in this dataset
separates them.

**But the pair is out-of-distribution-balanced, and the bg-matched pair is not.** The pretraining
split is `healthy_bg_cs` and `healthy_bg_no_cs` only (`cohort.PRETRAINING_SUBGROUPS`). So:

| Pair | bg axis | Encoder saw… |
|---|---|---|
| `hie_no_cs` vs `healthy_no_bg_no_cs` | **unmatched** | **neither** cohort — OOD on both sides |
| `hie_no_cs` vs `healthy_bg_no_cs` | matched | only the healthy side — OOD on one side |

That is a genuine trade, not a ranking. In the second pair, any latent separation is partly the
encoder extrapolating on HIE and interpolating on healthy — an asymmetry that manufactures exactly
the finding being looked for. In the first, the extrapolation penalty is paid by both cohorts, so a
residual separation is more likely to be about the fetus and less likely to be about the training
set. **Emit both pairs and label which is which**; quoting either alone is quoting half the answer.

**Check $n$ before anything else.** The HIE subgroups are the smallest in the split, and every
statistic below is over **recordings**. `cohort.cohort_counts` already reports segments and
recordings per cohort, and the second is the one that matters — a subgroup with hundreds of segments
and six recordings has $n = 6$. At that size use an exact or permutation $p$-value rather than the
asymptotic one, and read **Cliff's delta with its Romano magnitude** as the primary number: a
two-sided Mann–Whitney at $n = 6$ versus $n = 30$ cannot resolve anything but a large effect, and
reporting "not significant" from it as evidence of no difference is the standard misreading.

### 7.2 Three levels of comparison, and only one of them is new

| Level | Question | Machinery |
|---|---|---|
| **(a) Scalar** | Do the cohorts' trajectory *properties* differ? | `cross_subgroup`, unchanged |
| **(b) Curve** | *When* do they differ, against time before delivery? | `time_to_delivery`'s shape, reused |
| **(c) Geometric** | Do they occupy different regions, or move differently? | one new permutation test |

**Level (a) costs nothing and should be built first.** The §5 per-recording table — step rate,
straightness, dispersion, distance from the healthy centroid, and their `_stage1`/`_stage2`
variants — is declared in `cross_subgroup.METRIC_SOURCES` as one more `MetricSource`, and the
existing three-layer protocol then runs: a Kruskal omnibus per metric over all eight subgroups,
Holm **across metrics as one family**, and pairwise Mann–Whitney with Cliff's delta on the
survivors only. `hie_no_cs` versus `healthy_no_bg_no_cs` falls out as one of those pairs, correctly
adjusted, with no new statistics anywhere.

Two conventions inherited with it, worth restating because they decide how a sign reads.
`pairwise_comparisons` names the pair **in the cohort order it receives, least severe
first**, so for this contrast
`left = healthy_no_bg_no_cs` and `right = hie_no_cs`, and a **positive** Cliff's delta means the
healthy cohort's values run higher. And a cohort with fewer than `stats.MIN_GROUP_SIZE` finite
recordings is *excluded and recorded*, never silently entered.

**Level (b) is `time_to_delivery` applied to a different column set.** Bin on the shared $0.5$ h
grid (`cohort.add_time_bins`), reduce to one value per recording *inside* each window
(`cohort.per_recording_in_bins`), test per window with Holm across windows as one family, and
pairwise on the survivors. The only new thing is which columns are binned — the §5 scalars and the
PC coordinates rather than $K_t$ and `pred_gap`. Do not invent a second binning; `TRAJECTORY_BIN_HOURS`
is a module constant precisely so that two analyses cannot cut time differently and disagree.

Level (c) is §7.4.

### 7.3 The alignment problem, which comes before any test

Two recordings' trajectories are curves of different length, starting at different times, sampled
at the same $240$ s pitch. They cannot be averaged or compared until they are put on a common axis,
and the choice of axis *is* the scientific question being asked.

| Axis | Landmark at zero | Answers | Cost |
|---|---|---|---|
| **Time before delivery** | delivery | "how does the state approach the endpoint?" | left-censoring, §7.3.1 |
| Time from labour onset | onset | "how does the state evolve through labour?" | `time_from_labor_onset` is `NaN` for part of the cohort |
| Time from second-stage onset | second-stage onset | "what does the second stage do?" | smallest population of the three |
| Fractional time, $0 \to 1$ | both ends | — | **rejected** |

**Delivery-anchored is primary**, for three reasons: it is the landmark every recording has, it is
the axis the pipeline already bins on, and it is the axis on which the clinical question — does the
state diverge *before* the outcome, and how early — is actually posed. The two onset alignments are
worth emitting as secondary views because they control for different things: labour-onset alignment
removes "HIE labours are longer" as an explanation, delivery alignment does not.

**Fractional time is rejected.** Warping every recording onto $[0, 1]$ makes a $2$-hour and a
$12$-hour recording the same length, which destroys duration — and duration is one of the few
things that plausibly *does* differ between these cohorts. It also makes "the state moves faster"
untestable, because the time axis has absorbed the rate.

#### 7.3.1 Left-censoring is the largest trap in this comparison

On the delivery-anchored axis, the set of recordings contributing to a bin at $-10$ h is the set
that was monitored for at least ten hours. That is not a random subset of either cohort, and if the
two cohorts have different monitoring-duration distributions — which is exactly the kind of thing
that differs between HIE and healthy — then the cohort *composition* changes with the bin and a
"divergence at $-8$ h" can be a change in who is still in the sample.

Three requirements follow, and none is optional:

* **Draw $n_{\text{recordings}}$ per cohort per bin on the figure**, not only in the CSV, and refuse
  to draw or test a bin below `MIN_GROUP_SIZE` for either cohort. A mean over two recordings joined
  to a mean over twenty by a line is the single most misleading mark this figure can carry.
* **Report the two cohorts' span distributions** (`span_hours` per recording is already in §5.6). If
  they differ, say so before any early-divergence claim.
* **Ship a duration-matched sensitivity view**: restrict both cohorts to the last $X$ hours, with
  $X$ chosen as the largest window in which both clear `MIN_GROUP_SIZE`, and re-run the contrast
  there. If the finding survives it is about the state; if it disappears it was about who was still
  being recorded.

The uncomfortable corollary is worth writing down: **separation is best estimated near delivery and
worst estimated far from it**, which is the opposite of what an early-warning reading wants.

### 7.4 The geometric comparison, and the one test worth adding

Two cohorts can differ in **where** they sit and in **how** they move, and those are different
findings that must never be merged. A cohort can occupy a distinct region and move identically, or
share a region and move differently.

#### Where: separation against time

Per time bin $h$, form each cohort's mean position from one value per recording:

$$
\bar x_A(h) = \frac{1}{n_A(h)} \sum_{g \in A} x_g(h),
\qquad
d(h) = \big\lVert \bar x_A(h) - \bar x_B(h) \big\rVert_2 ,
$$

with $x_g(h)$ recording $g$'s mean window point in that bin. **Compute $d(h)$ in the full $48$
dimensions, not in the PCA coordinates.** The basis was fitted on the pooled data (§4.2), so it is
partly a function of the very class difference being tested — fine for drawing, circular for
testing. *Draw in PCA, test in 48-d* is the rule.

Standardise it so the number means something across bins:

$$
\hat d(h) = \frac{d(h)}{\sigma_{\mathrm{pooled}}(h)},
\qquad
\sigma^2_{\mathrm{pooled}}(h) = \frac{\sum_{c \in \{A,B\}} \sum_{g \in c} \lVert x_g(h) - \bar x_c(h)\rVert^2}{n_A(h) + n_B(h) - 2} ,
$$

a multivariate effect size in units of within-cohort spread. Raw $d(h)$ in latent units is not
comparable between bins whose cohorts have different internal spread.

**The null is a label permutation over recordings.** Permute the cohort labels among the recordings
$2000$ times (reusing `eval_config.bootstrap_resamples`), recompute $\hat d(h)$ each time, and take
the bin-wise null band. Recording-level permutation is the correct null because it preserves each
recording's own trajectory entirely and breaks only the cohort assignment — which is precisely the
hypothesis. Permuting *points* would destroy the within-recording correlation and produce a null
band far too narrow, reporting significance everywhere.

Holm across bins as one family, and report **the earliest bin whose $\hat d(h)$ survives**. That
single number — "the cohorts' latent states become distinguishable $X$ hours before delivery" — is
the deliverable, and it is only readable beside §7.3.1's per-bin $n$.

#### How: motion, already covered

Step rate, straightness and within-window dispersion are per-recording scalars and go through level
(a) unchanged. Nothing new is needed, and that is the point of having defined them
projection-free in §5.

#### A whole-curve test, when one is wanted

For "do these two cohorts' latent states differ at all", collapsed over time, the recording-level
object is a $48$-vector (the recording's mean position) and $n_{\text{HIE}} \approx 6 \ll d_z = 48$.

**Hotelling's $T^2$ does not work here and is the obvious thing to reach for.** It needs
$n > d$ to invert a pooled covariance; at these sizes the covariance is singular and the statistic
is undefined or, worse, silently regularised into something with no stated null.

The tool that does work is **distance-based**: build the $n \times n$ Euclidean distance matrix over
recordings, form the between-group over within-group ratio (a PERMANOVA-style pseudo-$F$, or the
energy distance — either is fine and both are a few lines of `numpy`), and get the $p$-value from
the same recording-label permutation as above. It needs no covariance inverse, no $n > d$, no new
dependency, and its null is exact by construction rather than asymptotic. This is **the one new
statistical routine** this whole document proposes; everything else reuses `_reuse.stats`.

### 7.5 Cohort against stage

The natural follow-up — *does the cohort gap widen in the second stage?* — is a difference in
differences, and it should be built as one rather than as four separate comparisons.

Per recording, for each §5 readout, take $\delta_g = m_g^{(\text{stage 2})} - m_g^{(\text{stage 1})}$
(defined only for recordings clearing `MIN_GROUP_SIZE` points in **both** stages), then compare
$\delta$ across cohorts with the same Mann–Whitney and Cliff's delta as level (a). One test per
readout, Holm across readouts.

This is worth the extra structure because it removes the between-recording offset that the two
separate stage-wise cohort comparisons would each carry, and because it asks the question that is
actually clinically interesting: not "is HIE different" but "does the second stage do something
different to HIE". State the population loss plainly — with HIE subgroups already small and both
stages required, this analysis will routinely fall below the minimum and record a skip, and a skip
is the honest output.

### 7.6 What a cohort comparison must carry with it

Every number in §7 ships beside these, or it is not readable:

* $n_{\text{recordings}}$ per cohort — overall and per time bin;
* the out-of-distribution statement, with **which side of the pair** it applies to (§7.1);
* the bg-axis confound for any `hie_*` versus `healthy_no_bg_*` contrast;
* the span distributions and the duration-matched sensitivity result;
* Cliff's delta with its magnitude label, always, beside every $p$-value;
* the note that the tests ran in $48$ dimensions while the figure shows three, with
  $\mathrm{EVR}_{1:3}$.

---

## 8. The figures

### 8.1 The colour budget, and how it is spent

Three things want colour and there are not three channels: time before delivery (continuous),
clinical class (categorical, and the palette is already committed), and stage (binary). Spending
hue on time would collide with the class palette that every other figure in the package uses; a
figure where green means "healthy" on one page and "eight hours out" on the next is a figure that
will be misread.

The resolution uses machinery `figures_seam` already has. `_blend(color, amount)` shades a class
colour toward white or black, and `SUBGROUP_TINT_RANGE` already uses it to make a subgroup a shade
of its class. So:

| Channel | Carries |
|---|---|
| **hue** | clinical class — `CLINICAL_CLASS_COLORS`, unchanged |
| **luminance** (via `_blend` along the path) | time before delivery, dark early to light late |
| **marker and line style** | stage — open marker / thin line for first, filled marker / emphasis line for second |
| **marker size** (optional) | $K_t$, §5.5 |

Hue still means what it means everywhere else in the package; the severity reading survives; and
the time ramp is a within-class luminance ramp, which also survives greyscale, as the class palette
was already designed to.

### 8.2 The four figures

**`latent_path.pdf` — one recording, the primary.** A $2 \times 2$ panel:

* PC1–PC2 with the ramp and the stage markers, the onset point annotated, structural joins drawn
  as a lighter connector and real breaks not drawn at all;
* PC1–PC3, same;
* each of PC1–PC3 against hours before delivery, as three stacked lines — this is the panel that
  actually answers "how does the latent change as delivery approaches", because a plane plot hides
  the time axis inside a colour;
* $K_t$ and `mc_pred_gap` against the same axis, so the coupling and the geometry are read in one
  column.

The recording drawn is the longest by default, as `trajectory` already chooses; the stratified draw
of `samples` is the model for emitting several.

**`latent_path_by_class.pdf` — the cohort overlay.** One panel per stage, every recording's path
drawn faintly in its class colour with the cohort's mean path over it. This is where a class
difference in trajectory shape would be visible, and it must carry the out-of-distribution sentence
of §2.3 in its guide entry.

**`latent_separation.pdf` — the pair contrast of §7.** Three stacked rows against hours before
delivery, sharing one x axis:

* the two cohorts' mean paths, one PC per row or the standardised separation $\hat d(h)$ as a single
  curve, with the permutation null band behind it and the surviving bins marked;
* $n_{\text{recordings}}$ per cohort per bin as a stepped line — **on the figure, not only in the
  CSV**, because §7.3.1's censoring is invisible without it and it is what stops a two-recording bin
  being read as a trend;
* the duration-matched sensitivity curve, drawn faintly over the full-population one, so agreement
  or disagreement between them is on the page rather than in a paragraph.

Its filename must not end in `_by_clinical_class` or `_by_subgroup`: those stems are reserved by the
runner's grouped fan-out and a collision would have the fan-out overwrite this figure.

**`latent_features.pdf` — the coordinates themselves.** Two rows: the $48 \times$ time heatmap of
$\mu^q$ (and of $\Delta\mu$ on the same colour scale, as the training figure's row 3 already does,
so their relative size is visible), and the four per-group KL curves. Projection-free, so it is the
figure to check when the PCA panel looks surprising.

### 8.3 On drawing it in 3-D

The request asks for 3-D and it is worth being direct about what it buys. In a static PDF, depth is
not recoverable — there is no rotation, occlusion hides ordering, and two points that look adjacent
may be far apart along the view axis. A 3-D path is an orientation aid; the pairwise planes are the
evidence. Draw both, label them as such, and put the plane panels first.

Four practical notes, since they are all things that fail quietly:

* `figures.new_figure` returns 2-D axes from `plt.subplots`. A 3-D panel needs
  `fig.add_subplot(..., projection="3d")` on a figure built for it, not a converted axes.
* Do **not** call `figures_seam.style_axes` on a 3-D axes: it sets spine widths and a 2-D grid, and
  a `mplot3d` axes' spines are not the same objects.
* `tight_layout` clips 3-D axes at the pipeline's proportions; set the margins explicitly.
* This environment's matplotlib mathtext rejects `\mathcal`, `\star` and `\operatorname` in labels
  and titles, and the failure happens at `savefig` — after the whole analysis has run. Use plain
  `L`, `K`, `PC 1` in axis labels.

---

## 9. Where this lands in the pipeline

### 9.1 Extract the assembly before writing the second consumer

`analyses/trajectory.py` owns the whole-delivery assembly: the $t_{\mathrm{abs}}$ coordinate, the
overlap averaging, `gap_before_s`, the segment boundaries. A latent path needs the same assembly at
window resolution. An analysis may not import another analysis, so the choice is to copy it or to
move it down — and `EVAL.md` states the rule: *"Anything two analyses share moves one layer down —
that rule is why `frames`, `lag_axis`, `cohort` and `events` exist."*

Move it into `cohort.py`, which already owns the time axis, `SECONDS_PER_HOUR` and the binning, as
something like `assemble_on_absolute_time(frame, *, time_column, value_columns)`. Have
`trajectory.whole_delivery` call it. This is also where §2.1's structural-versus-real break
classification and §2.2's trim offset belong, so the two analyses cannot disagree about where a
recording's points are.

### 9.2 The collection-pass change

Compute the window means in `metrics.py`, where `outputs["mu_prior"]` and `outputs["mu_post"]` are
already in hand and where the `contributing` indicator is already built — never in a second pass,
which would be a second forward over the split and would double the only expensive part of a run.

Emit them on `BatchReadout` as $(B, k \cdot d_z)$ arrays, flattened **window-major**, which is the
convention the cross-spectral vectors already follow. Carry them into `per_sample_vectors.npz`
under their own prefix, alongside `SPECTRAL_VECTOR_PREFIX`.

### 9.3 The trap: do not put them in `VECTOR_READOUTS`

`VECTOR_READOUTS` drives `aggregate_by_recording`, which takes per-recording **means**. The
cross-spectral sums are deliberately excluded from it because a mean of two segments' sums is
meaningless. Here the quantity *is* a mean, so membership would type-check — and would still be
wrong, for a different reason: **the per-recording mean of a trajectory is its centroid, which is
precisely the reduction that erases the trajectory.** Keep them out, under their own prefix, and
let the analysis do its own grouping.

Two more traps from the same neighbourhood, both already solved in `collect.py` and both easy to
reintroduce:

* the vectors sidecar is aligned to `per_sample.csv` **by position**, so a batch missing the key
  must be padded with `NaN` rows, never skipped — a skipped batch shifts every row below it onto a
  different recording;
* the blanking rule `np.where(scored[:, None], rows, np.nan)` applies here too.

And two pipeline-wide ones that a new analysis re-encounters: read any CSV it wrote back with
`float_precision="round_trip"`, and put no absolute path in anything that reaches `results`.

### 9.4 The places a new analysis touches

Registering `latent_trajectory` is more than a module. From the current bindings:

1. `eval/analyses/latent_trajectory.py` — the module itself, layer 2.
2. `eval/run.py` — the registry entry, **and** the launch-table comment block, which
   `test_eval_docs.py` asserts lists every selectable analysis and only those.
3. `eval/EVAL.md` — a `### latent_trajectory` heading, by exact slug equality, asserted by test.
4. `eval/FIGURE_GUIDE.md` — one entry per emitted PDF, asserted by test.
5. `eval/figure_manifest.json` — the committed manifest, kept equal to a real run's figures.
6. `tests/test_eval_figures.py` — the second, hand-kept figure table (`("trajectory",
   ("PROFILE_FIGURE",))` is the pattern).
7. `tests/test_eval_latent_trajectory.py` — the analysis's own tests.
8. **The transformer sibling.** `teb_vae/lag_attn_transformer_rws/eval` runs this same registry
   through this same runner, so it carries its own `FIGURE_GUIDE.md` and `figure_manifest.json`
   that must gain the same entries. Missing this is how the two packages' summaries quietly stop
   meaning the same thing — the exact failure the `ModelBinding` seam exists to prevent.

Plus the configuration and fixture edits: `load_fields` gains `second_stage_onset`, and
`tests/conftest.py`'s shard writer must write it. The fixture's epochs are already spaced at the
real $1200$ s stride, so §2.1's join arithmetic reproduces there — which means the structural-break
classifier can be tested rather than only reasoned about.

### 9.5 Offline re-runnability

Everything except the window means is derived from files. So

```bash
python -m teb_vae.lag_attn_rws.eval.run --output-dir <a finished run> --only latent_trajectory
```

must work with no checkpoint and no GPU, reading `per_sample.csv`, `per_sample_vectors.npz` and
`per_anchor.parquet`. That is the property that makes the PCA choices of §4 revisable — refitting
a basis is seconds against a multi-hour collection pass — and it is why §8.2's change has to be
right the first time while nothing else here does.

The analysis must therefore hold **no** layer-1 import: no `metrics`, no `collect`, no task, no
model. It reads tables.

One consequence for §7: because `cross_subgroup` runs **last** in the registry and reads finished
per-recording CSVs off disk, the latent-trajectory analysis must be registered **before** it, or its
`MetricSource` entries resolve to a file that does not exist yet and are recorded as missing rather
than raising — a silent loss of exactly the cohort contrast §7.2 depends on. Registering it beside
`trajectory` satisfies this with room to spare.

---

## 10. How this output will be misread

In the house style, because these are the readings the figure invites and does not support.

**A smooth path is not evidence of smooth physiology.** §1.1: the latent has no transition density.
Consecutive points are smooth because the encoders are smooth, and a straight-line interpolation
between two window means is drawing, not data.

**A break at a segment join is geometry.** §2.1: there is a $244$ s hole at every join in every
recording, and `BREAK_TOLERANCE_S` cannot tell it from a lost hour. A path drawn without the
structural/real distinction reports every recording as heavily fragmented.

**A step smaller than the join noise floor is not a step.** §5.2: report $\rho$, and read it before
reading a speed.

**A projection is not the space.** §4.2: quote $\mathrm{EVR}_{1:3}$ beside every PC figure. Two
paths that appear to cross in three dimensions are not near each other in forty-eight.

**A supervised axis fitted in-sample proves itself.** §4.4: cross-fit at GUID level or do not draw
it.

**A class separation in latent space is not clinical discrimination.** §2.3 and §7.1: acidosis, HIE
and both healthy no-background subgroups are out of distribution for this checkpoint. The
separation may be the encoder extrapolating — and in a pair where only one side is unseen, the
extrapolation itself is the asymmetry.

**A separation tested in the PCA plane is circular.** §7.4: the basis was fitted on the pooled data
and is therefore partly a function of the difference being tested. Draw in PCA, test in $48$-d.

**"The cohorts diverge at $-8$ h" may be "the cohorts stop being the same recordings at $-8$ h".**
§7.3.1: on a delivery-anchored axis the population in each bin is duration-selected, and the
selection differs between cohorts. Read the per-bin $n$ and the duration-matched view before the
divergence time.

**A non-significant Mann–Whitney at $n = 6$ is not evidence of no difference.** §7.1: the HIE
subgroups are the smallest in the split. Cliff's delta with its magnitude is the number to read;
the $p$-value at that size answers a question about power.

**A permutation null built over points rather than recordings reports significance everywhere.**
§7.4: within-recording correlation is the whole reason the unit is the recording.

**A stage difference is confounded with time before delivery.** §6.5: the second stage is also the
end of the recording. Quote the matched-window comparison, not the pooled one.

**The KL along the path carries every caveat it carries elsewhere.** Only
`source_conditioned_kl_raw` may be read as a rate, and only once `prior_variance_not_pinned` has
passed — a prior variance on its clamp inflates it by an arbitrary factor while every decoder-side
diagnostic stays healthy.

**Nothing here is causal.** The coupling readout is not causal under the shipped
`causal_reach_budget_s: null`, and drawing it against time does not make it so. Every artifact this
analysis writes carries the run's disclosure sentence like every other.

**A window mean is a summary only if the dispersion says so.** §3.2 and §5.4: if $s_j$ is
comparable to the step length, the path is a random walk inside a cloud.

---

## 11. Suggested order of work

Ordered so that the expensive, hard-to-revise step is first and everything after it is offline.

**Step 1 — the plumbing.** Window means in `metrics.py`, into the vectors sidecar under their own
prefix; `second_stage_onset` in `load_fields`, in `IDENTITY_COLUMNS`, and in the fixture; the
informational preflight entry. Tested by: the sidecar is row-aligned; a window with no contributing
anchors is `NaN`; the recorded stage boundary reproduces the anchor rule. **This is the only step
that costs a re-collection.**

**Step 2 — the shared assembly.** Extract `assemble_on_absolute_time` into `cohort.py`, add the
structural-versus-real break classification, repoint `trajectory.whole_delivery` at it. Tested by:
`trajectory`'s existing outputs are unchanged; a synthetic $1200$ s-stride recording classifies
every join as structural and a dropped segment as a real break.

**Step 3 — the projection-free readouts.** §5, straight to a per-recording CSV, with $\rho$. No
figure yet. This is the step that decides whether the rest is worth drawing: if $\rho \gg 1$ or the
within-window dispersion swamps the step length, the honest deliverable is that finding, not a
picture.

**Step 4 — the cohort contrast, level (a).** Declare the step-3 per-recording CSV in
`cross_subgroup.METRIC_SOURCES` (§7.2). This is a table entry and a test, and it delivers
`hie_no_cs` versus `healthy_no_bg_no_cs` — correctly Holm-adjusted, on per-recording values — before
any figure exists. It is the cheapest real answer in this document and it should not wait behind
the projection.

**Step 5 — PCA and the primary figure.** §4.2 and §8.2, with the recorded basis and the
$\mathrm{EVR}$ triple.

**Step 6 — the stage split and the paired test.** §6, including the matched-window control.

**Step 7 — the curve and geometric contrasts.** §7.3 and §7.4: the per-bin separation with its
recording-label permutation null, the per-bin $n$ and the duration-matched sensitivity view, and
the distance-based whole-curve test. This is where the one new statistical routine lands.

**Step 8 — the second-order views.** Per-head-group projections, `kld_per_t_per_head` on the
per-anchor table, the cross-fitted supervised axis, the 3-D panel, the cohort-by-stage
difference-in-differences of §7.5.

Step 3 gates steps 4, 5 and 7 — if the join noise floor $\rho$ swamps the step length, the honest
deliverable is that finding and steps 5 and 7 measure noise. Steps 4 and 5 are independent of each
other, and step 4 is the one to build first because it answers the cohort question with machinery
that already exists. Only step 1 blocks anything.
