# Feature-domain lag-attention forecaster - Spec and Roadmap

Status: DRAFT
Last updated: 2026-08-08
Owner: Mahdi Si

`lag_attn_fs` is the third model in the lag-attention family: the `lag_attn_rws`
architecture and rationale exactly - no decoder bypass, one latent, one shared decoder
invoked twice, head-structured posterior, source purity, exact zero-KL initialisation,
four-term nats-per-anchor objective - forecasting the **scattering and phase-harmonic
feature future** instead of the raw FHR future.

Companion documents: `teb_vae/lag_attn_rws/DESIGN.md` is the as-built record of the
model this one forks; `teb_vae/lag_attn_rws/model_explained.md` states the latent
factorisation both share; `teb_vae/lag_attn/DESIGN.md` records the feature-target model
that predates the bypass removal and is the reference this one is compared against.

---

## 1. Context

### 1.1 What exists, and what is confounded

Three lag-attention models are in the tree.

| Package | Target | Latent role |
|---|---|---|
| `lag_attn` | 109 feature channels, $H_d = 30$ steps | incremental correction on a `decoder_state` bypass |
| `lag_attn_rws` | 480 raw FHR samples | complete predictive state, no bypass |
| `lag_attn_transformer_rws` | 480 raw FHR samples | the same, conv-Transformer encoders |

The move from the first to the second changed **two things at once**: the latent
factorisation (removing the `decoder_state` path around the latent, so $z$ must carry
the whole predictive state) and the target domain (features to raw). Nothing in the tree
separates them. `lag_attn_fs` is the missing cell: feature target, no bypass. Against
`lag_attn` it isolates the bypass removal; against `lag_attn_rws` it isolates the target
domain.

### 1.2 The finding that shapes the scope

`lag_attn_rws/PRIOR_SCALE_AND_PREDICTIVE_GAP.md` section 5 establishes that the raw
models' held-out predictive gain is negative because **the source pathway does not
generalise**: at the same epoch of the same run the full branch beats the base branch by
$+7.14$ nats on training data and loses by $-7.37$ nats held out, and the excess
shortfall roughly doubles between epoch $120$ and epoch $400$.

That failure lives in the source encoder, the lag attention and the posterior fusion -
none of which a target-domain swap touches. **This roadmap is therefore an experiment,
not a remedy.** It is expected to reproduce the negative held-out gap. Its value is that
it removes a confound from the comparison, and it must not be scoped or sold as a fix.

### 1.3 The reuse surface, measured

The rws objective is nearly domain-generic already. `masked_raw_likelihood`,
`masked_raw_block_per_anchor`, `raw_sample_score`, `masked_source_kl`,
`masked_prior_rate` and `kld_tensor` all operate on `(B, T_valid, H, X)` against a
`(B, T_valid, H)` mask; only their names say raw. `masked_raw_likelihood`
(`lag_attn_rws/nets/losses.py:199`) already computes `d_sample` as
`d_block / (mu.shape[2] * mu.shape[3])`, which is generic.

In `compute_loss`, **four sites are raw-tied**, of which two carry the meaning:

- `:419` - `target = build_future_target(fhr_raw, geometry, future_index=future_index)`
- `:465` - `elem_denom = (elem_mask.sum() * float(geometry.r)).clamp_min(1.0)`
- `:417` - `device, dtype = fhr_raw.device, fhr_raw.dtype`, feeding the `kld_beta` and
  `beta_prior` tensors and the empty-support zeros. Taking them from `target` instead is
  bitwise identical.
- `:515` - `coverage_frac[:, geometry.warmup:]`. `geometry` therefore **stays** in the
  signature; a version of this refactor that drops it cannot compute
  `anchor_coverage_frac`.

`block_width` is a trap worth stating once: it feeds **only** `mean_logvar_full`,
`mean_logvar_base`, `logvar_full_floor_frac` and `logvar_full_ceil_frac` (`:465-486`) -
**not the loss**. Passing `geometry.r` where `C_keep` belongs changes no loss, fails no
shape check, and rescales by $4.9\times$ exactly the four metrics that acceptance
criterion 4 and the `logvar_clamp` re-derivation are read from.

`lag_attn_rws/nets/raw_masks.py` is reusable **unchanged**: `forecast_mask`,
`contributing_anchors` and `kl_mask` read only `geometry.t`, `geometry.t_valid`,
`geometry.horizon` and `geometry.warmup`, never `raw_len`, `decimation` or `r`.

Four claims were verified by direct execution rather than by reading, and each is a
premise the plan rests on:

1. `BaselineFutureDecoder(core, d_model=48, out_channels=109)` emits
   `(B, 270, 30, 109)` with **no new decoder class**, for **+23,994 parameters** over
   the raw variant. At the shipped $78$-channel target the delta is **+15,996**.
2. `forecast_mask` and `kl_mask` run unchanged on a feature model and produce
   `(B, 270, 30)` and `(B, 300)`.
3. The `unfold` future target has shape `(B, 270, 30, 109)` and satisfies the index
   identity `future[:, t, tau, :] == feat[:, t + 1 + tau, :]`.
4. `masked_raw_likelihood` consumes that block unchanged, `d_sample` equals
   `d_block / (H * C)`, and a value of `1e9` planted at a masked position moves the
   loss by exactly zero.

Output activation cost at batch $128$, fp32: $1.68$ GiB across the four output tensors
against $0.25$ GiB for the raw variant. Not a constraint - `lag_attn` already emits
`(B, 300, 30, 109)` from **two** decoders at that batch size, and this model emits from
one decoder over $270$ anchors.

### 1.4 The subclassing precedent

`lag_attn_transformer_rws` is the template for adding a sibling. Its `task.py` is $125$
lines against the rws $682$, and its `trainer.py` is $273$ against $799$, because it
subclasses rather than copies:

- `task.py:32` subclasses `SeqVaeLagAttnRwsTask` and defines exactly one method.
- `trainer.py:71` subclasses `LagAttnRwsTrainer` and sets three class attributes
  (`MODEL_CLS`, `TASK_CLS`, `CHECKPOINT_STEM`) plus four small overrides.
- `trainer.py:204` - `main(config_path)` is one line delegating to
  `lag_attn_rws.trainer.main(config_path, trainer_cls=...)`.

`lag_attn_fs` follows that shape. **Four** things in the shared driver are bound to the
raw target by name, and all four need a seam before the new package can inherit
`train_model` and `main`:

- `trainer.py:535` - `_check_raw_target_normalized(config)`, called by name from `main`.
- `trainer.py:438` - `MetricsLoggingCallback(tracked_metrics=_TRACKED_METRICS)`, reading
  a **module global**, so a subclass that does not override `train_model` cannot change
  the tracked set.
- `trainer.py:483` - `from teb_vae.lag_attn_rws.plotting import LagAttnRwsPlotCallback`,
  hard-coded inside `train_model`.
- `trainer.py:478` - the literal config block name `"lag_attn_rws_plotting"`.

A fourth package makes the first of these load-bearing in a way that is easy to miss:
`lag_attn_transformer_e2e/trainer.py:103-134` overrides `preflight` with a **bare body**
and its docstring records that it relies on `_check_raw_target_normalized` being called
by name from `main`. Moving that guard into `preflight` would silently disable it there,
and `preflight`'s documented no-op property - *a subclass that forgets `super()` cannot
drop an inherited check, there are none to drop* - is what makes a bare-bodied override
safe. The guard is parameterised instead; see 4.5.

### 1.5 The target is smeared, and that is not a leak

A stored coefficient at decimated step $s$ is not a value *at* $s$. It is a weighted
average of raw signal over a window **centred** at raw index $16s$ with half-width
$\rho_c$, the channel's $L_{95}$ energy reach. So the forecast target at short horizons
is partly a deterministic function of raw signal the model has already observed.

Measured on the production filter bank
(`python -m teb_vae.lag_attn.channel_reach`):

```
     block    n      min   median       max
    fhr_st   43      1.5     45.8     965.5
    fhr_ph   66     13.5     50.9     266.8
```

Horizon step $\tau$ targets a coefficient centred approximately $4\tau$ seconds after
the anchor, so the fraction of its support lying in observed history is

$$
b(\tau, \rho_c) \;=\; \max\!\left(0,\; \frac{\rho_c - 4\tau}{2\rho_c}\right),
$$

which is exactly $0.5$ at $\tau = 0$ for every channel and falls linearly to zero at
step $\rho_c / 4$.

**This is not a causality violation.** No future information enters the model;
part of the answer is computable from what the model legitimately already observed,
which is the opposite of leakage. The input-side two-sidedness is a separate matter,
identical in both target domains, and handled by the existing
`causal_reach_budget_s` gate.

**It does not bias the readout either.** Writing the target as $Y^+ = (A, B)$ with $A$
the component fixed by the observed history, $A$ is $\sigma(Y^-)$-measurable, so
$I(U^-; A \mid Y^-) = 0$ and $I(U^-; Y^+ \mid Y^-) = I(U^-; B \mid Y^-)$. In the
estimator, $D_0$ and $D_1$ come from one shared decoder under one shared $\epsilon$, so
$A$'s contribution is near-identical in both branches and largely cancels in the
difference.

What it does affect is **optimisation** (a share of the summed NLL is a component the
source cannot help with, so the source pathway competes against an easy reconstruction
task) and **interpretation** (a 30-step feature forecast is not the same physical claim
as a 30-step raw forecast).

### 1.6 Why the target is the gated subset

Restricting the forecast target to the channels surviving `causal_reach_budget_s` halves
the mean blend and makes the far horizon genuinely clean. Measured over the $H = 30$
horizon:

```
 tau   t(s) | ALL 109: mean blend  clean ch | KEPT 78: mean blend  clean ch
   0      0 |         0.500         0       |        0.500         0
   3     12 |         0.317        14       |        0.254        14
   6     24 |         0.229        33       |        0.141        33
  12     48 |         0.153        55       |        0.055        55
  18     72 |         0.115        63       |        0.020        63
  24     96 |         0.090        71       |        0.005        71
  29    116 |         0.075        75       |        0.000        75

mean blend over the whole horizon: all = 0.173   kept = 0.091
rho (s):  all median 47.2 max 965.5   |   kept median 28.0 max 117.2
```

The kept set's worst channel becomes clean at step $29.3$, just past the horizon. The
full set never reaches a fully clean step, because it retains channels reaching
$965.5$ s. Restricting the target also makes one statement true that is otherwise false:
input and target live under one causal budget.

---

## 2. Goals

1. A `teb_vae/lag_attn_fs` package that trains end to end under the `lag_attn_rws`
   architecture, forecasting the $78$-channel gated feature future over $H = 30$
   decimated steps.
2. One objective definition shared by all three models. The generalisation of
   `lag_attn_rws/nets/losses.py` must leave both shipped models **bitwise unchanged**,
   proved by regression test rather than asserted.
3. A training-time diagnostic figure that shows whether the model works, since the
   evaluation pipeline is deferred.
4. A short run on the committed HIE shard showing the four-term loss recomposing, a KL
   that opens off zero, and a prior that stays off its floor.
5. A recalibrated $\beta$, because the reconstruction term grows by a factor of
   $H \cdot C / (H \cdot R) = 3270 / 480 \approx 6.8$ against an unchanged KL.
6. The preprint at `lag_attn_rws/doc/latex_template/` documenting the feature-domain
   variant and the backward half of two-sidedness, which it currently does not cover.
7. A `lag_attn_transformer_fs` sibling swapping in the conv-Transformer encoders.

---

## 3. Non-goals

- **The evaluation pipeline.** No `eval/` package, no `ModelBinding`, no additions to
  `lag_attn_rws/tests/test_eval_launch.py::ENTRY_POINTS`. Deferred whole. The
  consequence is explicit: this model will report `pred_gap`, `source_conditioned_kl_raw`
  and the lag map in its training metrics and its diagnostic figure, and will have no
  verdicts, no bootstrap confidence intervals, no trivial-predictor baselines, no
  calibration and no per-recording tables until a later effort.
- **Fixing the source-pathway generalisation failure** of section 1.2.
- **Raw-domain forecasting** from this package. `lag_attn_rws` owns that.
- **A production run.** The prod box is a separate 8x A6000 Linux machine; runs there
  are a manual step outside this roadmap.
- **Eval stubs or placeholder seams** built now against a later port.
- **Cross-domain nats comparability.** The factorized Gaussian over $2340$ correlated
  coefficients overcounts independent information; the resulting nats rank arms of this
  model and are not comparable to the raw model's. Recorded, not fixed. The same applies
  **across reach budgets within this model**: each budget changes $C_{\mathrm{keep}}$,
  hence the decoder width, hence the block the NLL sums over, so two fs arms at different
  budgets have non-comparable `pred_gap` and mutually unloadable checkpoints.
- **A second normalisation of the target.** The loader's `normalize_fields` output is
  used as delivered.

---

## 4. Design

### 4.1 What changes and what does not

Held identical to `lag_attn_rws`, by import rather than by convention: the input
adapters, both encoders, the channel gate, `FullLatentPriorHead`, `LagCrossAttention`,
the query projection, `PosteriorHead` (head-structured), `TEAnalysisHead`,
`HorizonDecoderCore`, `BaselineFutureDecoder`, the paired reparameterisation,
`nets/controls.py`, `collapse.py`, and the four-term objective.

Changed:

| | `lag_attn_rws` | `lag_attn_fs` |
|---|---|---|
| forecast target | `(B, 270, 30, 16)` raw samples | `(B, 270, 30, 78)` feature coefficients |
| target source | `batch.fhr`, gathered by raw index | `cat(fhr_st, fhr_ph)`, gathered by channel then unfolded |
| decoder `out_channels` | `raw_per_step = 16` | `target_gate.out_channels = 78` |
| block width | `H * R = 480` | `H * C = 2340` |

`raw_per_step` **stays** in the config. `TrimmedRawGeometry` requires `decimation` and
validates the raw index identities in `__post_init__`, and the diagnostic page's row-1
time axis is sourced from the raw grid, so the key remains a geometry input; it simply
stops being the decoder width. The decoder width comes from
`self.target_gate.out_channels`, or `self.c_y` when no budget is configured.

### 4.2 The objective generalisation

`compute_loss` takes the target as an argument instead of building it, and takes the
last-axis width explicitly instead of reading `geometry.r`:

```
compute_loss(forward_outputs, target, *, mask, kl_support, block_width,
             coverage_frac, logvar_clamp, beta, beta_prior, lambda_full,
             lambda_base, likelihood, free_bits)
```

The two shipped models keep thin `compute_loss` methods that build their raw target and
masks and then call it, so `model.compute_loss(...)` still works at every existing call
site and the arithmetic has exactly one home. This is the reason the module is free
functions in the first place - its docstring says what two architectures optimise must
never diverge, and three make the argument stronger.

The change is inert by construction: `block_width` receives `geometry.r` from the raw
callers, and the target they pass is the tensor `build_future_target` was returning.
S0-T01 proves it bitwise. **Three** models keep such a method, not two -
`lag_attn_transformer_e2e` delegates verbatim as well.

### 4.3 The target: gathered, never delayed

The input `ChannelGate` (`lag_attn/nets/delays.py`) applies **two** operations - a gather
of the surviving channels and a per-channel delay $\delta_c = \lceil \rho_c / \Delta
\rceil$ that pushes each channel's forward reach behind the anchor's causal endpoint.

The target takes the gather and **not** the delay. Delaying the target would change
which future the model is asked to forecast - anchor $t$'s target would silently become
the future of anchor $t - \delta_c$ - and nothing downstream would fail. This is the
sharpest correctness trap in the build and gets a dedicated test asserting that the
target at anchor $t$, horizon step $\tau$, equals the *undelayed* feature block at step
$t + 1 + \tau$ restricted to the kept channels.

At `causal_reach_budget_s: null` the survivor set is all $109$ channels and the decoder
width follows, so the unguarded arm is well defined.

### 4.4 Geometry

`TrimmedRawGeometry` is reused unchanged. The raw grid remains a true fact about the
batch - the loader delivers `fhr` at $4800$ and `weight` at $300$, and
`raw_masks._validate_weight` checks `weight.size(1) == geometry.t`. Only `t`, `t_valid`,
`horizon` and `warmup` are consulted; `r`, `n_raw()` and `future_block_start()` are
never called by this model. A second geometry type and a shared protocol for
`raw_masks` to accept both would be new code buying nothing the explicit `block_width`
argument does not already buy.

### 4.5 The entry-point guard, and the three other seams

`_check_raw_target_normalized` (`lag_attn_rws/trainer.py:692-723`) gains a `fields`
keyword defaulting to `("fhr",)`, and `main` passes `trainer_cls.TARGET_FIELDS` - one
class attribute beside the three that already exist for exactly this purpose
(`MODEL_CLS`, `TASK_CLS`, `CHECKPOINT_STEM`). `LagAttnFsTrainer.TARGET_FIELDS` is
`("fhr_st", "fhr_ph")`.

The guard stays unconditional in `main`, so no subclass can drop it; `preflight` keeps
its no-op property; the three modules that import the guard individually
(`lag_attn_rws/eval/preflight.py:56-59`, and the order assertions in both transformer
packages' `test_trainer.py`) keep their one-argument calls and their assertion that the
four guards run in a fixed order.

The other three seams follow the same shape: a `TRACKED_METRICS` class attribute
defaulting to the module global, and a `plot_callback_cls()` classmethod plus a
`PLOT_CONFIG_KEY` returning today's values. Each is an exact no-op for all three shipped
packages and is covered by the regression proof of S0-T01.

**No `lag_attn_fs/plotting.py` is written.** The rws callback is already model-agnostic -
`plotting.py:249-300` routes through `pl_module._build_forward_inputs`,
`pl_module._build_raw_target`, `model.compute_loss`, `model.geometry` and
`_source_delay_steps(model)`, and its docstring states that a subclass changing its net's
input signature gets a correct figure with nothing edited there. The work is in
`sample_page`, not in the callback.

### 4.6 The diagnostic page

`sample_page.build_diagnostic_figure` (`:144`) is **one function** with all seven rows
inline and nested closures, and its shared scaffolding - the time axis, the tile window
edges, the caption - reads `geometry.r`, `geometry.raw_len` and
`geometry.future_block_start(t)`. There is no row-level seam to inherit, so "rows 3-7
verbatim" is only achievable by first *creating* one. S0-T06 extracts a forecast-row
seam in the shipped module, leaving rows 3-7 and the whole layout machinery in one place;
Sprint 4 fills it.

Rows 3-7 - prior mean over posterior displacement, per-step per-dimension KL, $K_t$, the
lag-attention matrix, the per-lag KL attribution - are then genuinely shared. Rows 1-2
are rebuilt:

- Row 1 keeps the raw FHR and UP traces as physiological context. The model does not
  read them; they orient a reader judging whether a forecast is plausible, and they are
  in the batch regardless.
- Row 2 becomes the forecast over three representative target channels (true, base and
  full with $\pm 2\sigma$ bands) plus an $(H \times C)$ absolute-error heatmap for one
  anchor, so a per-channel failure is visible without choosing channels by hand.

Channels are selected by a stated rule, not by hard-coded indices. `lag_attn`'s
`forecast_channels` key was removed precisely because inherited indices silently changed
meaning when `fhr_ph` went from $44$ to $66$ channels.

The config block keeps the inherited name `lag_attn_rws_plotting`. The transformer
package documents why at its `configs/default.yaml:490-491`: renaming it to match the
package silently disables the figure with no error anywhere.

### 4.7 Configuration

`default.yaml` is written out in full rather than inheriting, following the transformer
precedent and its stated reason, and is pinned leaf-for-leaf against the rws file
outside a declared allow-list. Relative to `lag_attn_rws/configs/default.yaml`:

- removed: **nothing**. `raw_per_step` stays, per 4.1 - it is a geometry input and simply
  stops being the decoder width. The pin is therefore *total* rather than schema-limited:
  the constructor schema is unchanged, so every key means the same thing in both files and
  every key is comparable, which is a stronger statement than the transformer package can
  make about its own pin.
- unchanged: every width, the attention block, the horizon block, the init policies,
  `logvar_clamp: [-5.0, 3.0]`, `coverage_floor: 0.9`, `base_decode: mean`,
  `posterior_logvar_mode: independent`, `causal_reach_budget_s: 120`, `causal_norm: true`,
  the whole `dataset_config`, the spike breaker
- retuned: `beta_schedule.end`, by the sweep of section 4.8

`logvar_clamp` stays `(-5, 3)`. `doc/latex_template/sections/architecture.tex:368-371`
records that this interval was *inherited from a decoder that emitted feature
coefficients rather than standardized raw samples* - in this model it is going home.

### 4.8 The beta recalibration

The reconstruction term is summed over $H \cdot C = 30 \times 78 = 2340$ coefficients
against the raw model's $H \cdot R = 480$ samples, while the KL is summed over
$d_z = 48$ either way. At the inherited $\beta$ the reconstruction therefore applies
roughly $4.9\times$ more pressure against the KL than it did (or $6.8\times$ at the
ungated $C = 109$).

**The direction matters and is easy to get backwards.** A larger reconstruction at fixed
$\beta$ makes $\beta \cdot \mathrm{KL}$ relatively *weaker*, so the KL opens **wider**,
not narrower. Holding the raw model's ratio of KL weight to reconstruction scale puts the
scale-matched value near $\beta \approx 4.9$, so the arms bracket that rather than the
inherited $1.0$: **`1.0 / 2.5 / 5.0 / 10.0`**.

$\beta_p$ moves with it. `lag_attn_rws/configs/default.yaml:83-92` argues that
`beta_prior: 0.1` is a threshold rather than a dial, because the anchor's restoring force
**saturates** at $\beta_p / 2$ per dimension while the reconstruction's opposing pressure
grows as the decoder sharpens. Multiplying the reconstruction by $4.9$ attacks exactly
that threshold, and `logvar_prior_floor_frac < 0.2` is an acceptance criterion. The
$\beta_p / \beta$ ratio is therefore held fixed across the arms rather than $\beta_p$
being frozen. Nothing else is swept until a pipeline exists to score it.

### 4.9 Observability, and the one thing the deferred eval still owes

With the evaluation deferred, every readout is a scalar summed over $2340$ coefficients.
A model forecasting three easy low-frequency channels well is indistinguishable from one
that is uniformly mediocre, and - given 1.5 - a model reconstructing the blended
component is indistinguishable from one forecasting the clean tail.

Two additions to the tracked metrics close that, and both are nearly free because the
block score already reduces over $(H, C)$:

- `pred_gap` **resolved by horizon step**, at minimum as the two scalars
  `pred_gap_tau_first` and `pred_gap_tau_last`. This is what separates real forecasting
  from reconstruction of the smeared component, since by 1.6 the blend at $\tau = 29$ is
  $0.000$ on the kept set and $0.500$ at $\tau = 0$.
- `pred_gap` **split by feature block**, `fhr_st` against `fhr_ph`, whose blends differ.

They also discharge an obligation the plan would otherwise carry silently: S7-T02 states
a horizon-resolved reporting rule in the preprint, and without these metrics no code in
the tree produces it.

### 4.10 Alternatives considered

- **A full eval port.** Nine of eighteen analyses are domain-agnostic and four need only
  a unit seam, but `collect`, `metrics`, `spectra`, `oracle`, `events`, `coherence` and
  `samples` are structurally raw, and `report_seam.HEADLINE_SCALARS` requires every
  registered path to resolve on every run. A feature-domain `events` or `coherence` is a
  new scientific construction rather than a port, and inventing one to fill a column is
  worse than an absent column. Deferred whole.
- **A new `FeatureGridGeometry`.** Rejected in 4.4.
- **Copying `losses.py`.** Three copies of the objective that must never drift is the
  exact failure the free-function structure exists to prevent.
- **Restructuring the preprint to a 2x2 grid** (encoder x target domain). Honest to what
  will exist once `lag_attn_transformer_fs` lands, but it rewrites the title, the abstract,
  `tab:scope`, `tab:heldfixed`, `tab:whatchanges`, `tab:parametercompare`,
  `tab:config-encoders`, `tab:config-runtime`, `tab:sym-encoders` and roughly thirty
  "both architectures" passages in a $75$-page document that currently builds clean at
  zero bad boxes. A self-contained variant section is the smaller correct change;
  the restructure is available later if the 2x2 fills in.
- **Per-channel target standardisation.** Would make the `head_init_calibration`
  trivial-predictor argument exactly true, but adds a second normalisation over the
  loader's and a statistics blob that must travel with the checkpoint. The decoder's
  per-coefficient log-variance head absorbs residual scale differences; `lag_attn`
  already trains this way.
- **Predicting all 109 channels.** Rejected on the measurements of section 1.6.

---

## 5. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| The `losses.py` generalisation changes a shipped model's numbers | medium | high - invalidates every prior run's comparability | S0-T01 pins all **three** models' full metric dicts bitwise, by re-running one seeded script before and after rather than against committed constants |
| The target is accidentally delayed with the input | medium | high - silently forecasts the wrong window, nothing fails | S1-T02 asserts the index identity against the undelayed block explicitly |
| Moving `_check_raw_target_normalized` behind `preflight` silently disables it | low | high - a run trains on an unnormalised target | S0-T04 keeps the existing guard test and adds one asserting `preflight` refuses |
| Inherited $\beta$ leaves the KL over-open at the larger reconstruction scale, so the latent carries target information it should not | high | medium - inflated coupling that a permutation control may not catch | The scale-matched arms of 4.8; S5 reads `kld_raw` and `shuffle_penalty` together |
| The clip and the spike breaker are inherited unchanged at a $4.9\times$ loss scale | high | high - `gradient_clip_val: 5000.0` was **measured** from this objective's own grad-norm percentiles, and `additive_margin: 1.0e+3` was derived as $\approx 2 \cdot 480$; at $2340$ they either clip every step or never bind | S3-T05 re-derives both before the first long run, from a short instrumented run |
| The $\beta_p$ threshold is overrun by the larger reconstruction, collapsing the prior scale | medium | high - `logvar_prior_floor_frac` is an acceptance criterion | The $\beta_p / \beta$ ratio is held across arms (4.8) rather than $\beta_p$ frozen |
| An rws checkpoint is loaded into an fs model | medium | high - every tensor but the decoder head aligns, so `load_checkpoint_strict` returns non-`None` and the run starts on a partially-initialised model | S2-T05 pins the checkpoint contract, including that `target_keep_index` is stamped and `raw_per_step` is absent |
| Nothing distinguishes "forecasting well" from "reconstructing the blended component" during a run | high | medium - a multi-day run cannot be judged from its scalars | S3-T04 tracks `pred_gap` resolved by horizon step and split by feature block |
| The init NLL is dominated by the widest channels until the log-variance head catches up | medium | low | `head_init_calibration` already centres the log-variance head at $\sigma = 1$; watched via `mean_logvar_full` in the first local run |
| The blend of section 1.5 is read as leakage by a later reader | medium | medium - a correct result gets discarded | The preprint subsection of Sprint 7 states the argument in full; `DESIGN.md` cross-references it |
| Config drift between the three `default.yaml` files | medium | low | A leaf-for-leaf pin against the rws file outside a declared allow-list, following `lag_attn_transformer_rws/tests/test_config_load.py:34` |

---

## 6. Acceptance

1. `pytest teb_vae/lag_attn_rws/tests -q -m "not slow"` and
   `pytest teb_vae/lag_attn_transformer_rws/tests -q -m "not slow"` are green after the
   objective generalisation, with both models' metric dicts bitwise unchanged.
2. `pytest teb_vae/lag_attn_fs/tests -q -m "not slow"` is green.
3. A `tiny.yaml` run completes one epoch through the real framework.
4. On the committed HIE shard, pre-registered so the result cannot be read after the
   fact: `total_loss` recomposes from its four weighted terms to $10^{-6}$ **relative**;
   `source_conditioned_kl_raw` exceeds $0.05$ nats per anchor by epoch $20$ and is not
   falling over the last five; `logvar_prior_floor_frac` stays below $0.2$;
   `mean_logvar_full` sits at least $0.5$ above the clamp floor. **A negative `pred_gap`
   is a PASS** - section 1.2 predicts it - and what would instead indicate a build error
   is `pred_gap` identically zero, or `abs(pred_gap)` exceeding `nll_base_block` in
   magnitude.
   The recomposition tolerance is **relative and has to be**, which the draft left
   ambiguous: `main_loss` here is a few thousand nats and the run records float32, whose
   spacing at $4 \times 10^{3}$ is already $\approx 5 \times 10^{-4}$, so an absolute
   $10^{-6}$ is below one representable step and no correct implementation could meet it.
   The shipped unit test has always asserted `rtol=1e-6`; the run-level reading is the same
   quantity.
5. `gradient_clip_val` and `spike_breaker.additive_margin` are re-derived at the new loss
   scale from a short instrumented run, and the config comment names the run they came
   from.
6. The four structural invariants hold under test: source purity, no decoder bypass,
   exact zero KL at initialisation, and the shared decoder invoked twice.
7. The lag map sums over lags to `kld_per_t` exactly - to float round-off, $\approx 10^{-6}$ at
   these geometries, not bitwise: the two sides sum over different axes in different orders. See
   S4-T03.
8. The diagnostic figure renders for a real batch and its lag axis agrees with the
   training figure's.
9. The preprint builds clean with `latexmk` - no undefined reference or citation, no
   "Label(s) may have changed", no overfull box wider than about $15$ pt - and the page
   count is reported.
10. `lag_attn_transformer_fs` reaches the same bar 1-8.

---

## 7. Sprint overview

| Sprint | Goal | Demoable at the end | Tasks |
|---|---|---|---|
| 0 | Seams in the shipped packages | All three shipped suites green, every metric provably unchanged | 6 |
| 1 | The feature target | The 78-channel block builds from the committed shard and matches a hand-written index | 3 |
| 2 | The model | `SeqVaeLagAttnFs(**tiny)` forwards, returns 20 keys, KL exactly zero at init | 7 |
| 3 | Task, trainer, configuration | `python -m teb_vae.lag_attn_fs.trainer` completes a tiny epoch | 7 |
| 4 | The diagnostic page | A seven-row PDF for a real batch, lag axis agreeing with the metrics | 3 |
| 5 | Local validation on real data | **Landed.** Four HIE-shard arms meeting every pre-registered criterion, arm table filled, shipped weights moved to the sweep's winner | 5 |
| 6 | The package record | **Landed.** `DESIGN.md` bound to the module by test, in both directions | 1 |
| 7 | The preprint | `latexmk` clean, page count reported against the 75-page baseline | 7 |
| 8 | `lag_attn_transformer_fs` | **Discharged elsewhere** - see the note at the end | - |

**Implementation note that applies to every task below.** The code, comments, docstrings
and commit messages must not mention sprints, tasks, phases, or this document. A
`SN-TNN` identifier is a planning artifact and has no place in the tree. Write each
change as though it were always meant to be there. All tests go in the package's own
flat `tests/` directory, never beside the module under test.

---

## Sprint 0 - Seams in the shipped packages

**Goal.** Four by-name raw dependencies in the shared driver and one in the shared
objective become parameters, with every shipped number provably unchanged.

Every task here edits code that three production models depend on. S0-T01 comes first
for that reason: it is the evidence the rest are inert.

**Landed 2026-08-08.** All three shipped suites green at their baseline counts, and all
three models' full metric dicts bitwise unchanged under both likelihoods, verified by
re-running one seeded script before and after the objective edit rather than by comparing
against committed constants. Six things this document got wrong. Each is **already corrected
in place** at the task, section or table it appears in - what follows is the record of what
changed and why, because a criterion that quietly became satisfiable is worth knowing about:

1. **`compute_loss` has three callers, not two.** `lag_attn_transformer_e2e/nets/model.py`
   is a third verbatim delegation; S0-T02's file list omitted it, and it breaks without the
   same edit.
2. **S0-T04's "the three order assertions pass unchanged" is false.** All three stub the
   guard with a one-positional lambda, which a `fields=` keyword cannot survive; passing it
   positionally instead binds it to the stub's label parameter and corrupts the recorded
   order. Each stub absorbs the keyword (`lambda config, **_`); the assertions themselves
   are byte-identical.
3. **S0-T01 cannot "pass unmodified" through S0-T02 literally.** Before the change nothing
   can hand the objective a target. What survives untouched is the assertion block; the
   three-line call helper is the part that moves, and a second test reassembles every
   metric from the primitives and so depends on no signature at all.
4. **No pre-change figure reference existed.** `test_plotting.py` asserted per-row content,
   never the assembled page, so "compares equal to a pre-change reference under the existing
   figure test" had nothing to compare against. A row inventory was written and committed
   green first; the numeric proof is the same before/after script technique.
5. **The dispatch route was unspecified.** The shared callback binds the page builder as a
   module global, and S4-T02 forbids both a `lag_attn_fs/plotting.py` and a new callback
   class -- so nothing reached the seam. The callback now reads a `forecast_rows` attribute
   off the task, defaulting to `None`, and passes the batch through beside it. **Sprint 4's
   only remaining figure work is `lag_attn_fs/sample_page.py` and a `forecast_rows` property
   on the fs task.**
6. **Cross-references.** Section 4.2's "S0-T03 proves it bitwise" and the risk table's
   "S0-T03 pins both models' full metric dicts" both meant **S0-T01**; Sprint 8's "under
   S0-T03 the model is a subclass" meant **S2-T01**. All three now name the right task.

### S0-T01 - The equivalence harness, captured on the unmodified tree

**Description.** Before any other change, add a test asserting that
`model.compute_loss(...)` equals a direct call to the free `compute_loss` fed the target
and masks built explicitly from `build_future_target`, `forecast_mask` and `kl_mask`.
Commit it green against the current implementation.

**Acceptance criteria.**
- Compares **every** key of the returned `metrics` dict, not a chosen subset, with
  `torch.equal`.
- Both `likelihood: gaussian_nll` and `likelihood: mse`.
- Computed **in-process**, both sides in the same run - not against committed decimal
  constants, which do not survive a move between the RTX 4080 dev box and the A6000
  prod box, and which no rule in the repo could legally regenerate.
- Covers `lag_attn_rws` only. `lag_attn_transformer_rws/nets/model.py:1000-1014` is a
  verbatim delegation with identical arguments, so its metrics are unchanged by
  construction; a second copy is a second copy of one piece of evidence.

**Files.** `teb_vae/lag_attn_rws/tests/test_objective.py`.

**Validation.** `pytest teb_vae/lag_attn_rws/tests/test_objective.py -q`.

### S0-T02 - `compute_loss` takes the target and the block width

**Description.** Swap the two sites 1.3 measured: `fhr_raw` plus `future_index` become a
`target` argument, and `geometry.r` becomes a required `block_width: int`. `weight`,
`geometry` and `coverage_floor` **stay** - `geometry.warmup` is needed at `:515` and the
masks are already domain-neutral, so moving their construction would duplicate two lines
into three model classes for nothing.

**Acceptance criteria.**
- Signature is `compute_loss(forward_outputs, target, *, weight, geometry, block_width,
  coverage_floor, logvar_clamp, beta, beta_prior, lambda_full, lambda_base, likelihood,
  free_bits)`.
- `device, dtype` come from `target`; the import of `build_future_target` is gone.
- The alias imports of `forecast_mask`/`kl_mask` and the plain import of
  `contributing_anchors` (`losses.py:32-35`) all **stay** - `masked_raw_block_per_anchor`
  needs the last one.
- All three shipped models keep their existing public `compute_loss` signatures, so no call
  site outside `nets/` changes.
- S0-T01's **assertions** pass unmodified. Its three-line call helper is the part that moves,
  and must be the only part: before this task nothing can hand the objective a target, so a
  harness that survived here literally could not have existed beforehand.

**Files.** `teb_vae/lag_attn_rws/nets/losses.py`,
`teb_vae/lag_attn_rws/nets/model.py`, `teb_vae/lag_attn_transformer_rws/nets/model.py`,
`teb_vae/lag_attn_transformer_e2e/nets/model.py` -- **three** delegations, not two.

**Validation.** `pytest teb_vae/lag_attn_rws/tests/test_objective.py teb_vae/lag_attn_rws/tests/test_losses.py teb_vae/lag_attn_transformer_rws/tests -q -m "not slow"`.

### S0-T03 - `decoder_out_channels` on the shipped model

**Description.** Add `decoder_out_channels: Optional[int] = None` to
`SeqVaeLagAttnRws.__init__`, used at `nets/model.py:499` and defaulting to
`self.raw_per_step`.

**Acceptance criteria.**
- A default-argument model is bitwise the pre-change one, including its RNG stream:
  the decoder must be built at the same point in `__init__`, **before**
  `initialization(self)`, `_zero_init_film_generators()` and `_calibrate_output_heads()`
  at `:519-542`.
- A test constructs at a non-default width and asserts the output shape follows and the
  three init passes still applied.
- The keyword is forwarded by `_build_model_kwargs`'s signature sweep without a special
  case.

**Files.** `teb_vae/lag_attn_rws/nets/model.py`,
`teb_vae/lag_attn_rws/tests/test_construct.py`.

**Validation.** `pytest teb_vae/lag_attn_rws/tests/test_construct.py teb_vae/lag_attn_rws/tests/test_decoder.py -q`.

### S0-T04 - `TARGET_FIELDS` parameterises the normalisation guard

**Description.** Give `_check_raw_target_normalized` a `fields` keyword defaulting to
`("fhr",)`, add `TARGET_FIELDS = ("fhr",)` to `LagAttnRwsTrainer`, and have `main` pass
`trainer_cls.TARGET_FIELDS` at `:535`.

**Acceptance criteria.**
- The guard stays in `main`'s by-name list, so no subclass can drop it and
  `preflight` keeps its documented no-op property.
- `lag_attn_transformer_e2e`, whose `preflight` has a bare body and whose docstring
  relies on this guard being called by name, is unaffected.
- The three order assertions (`lag_attn_transformer_rws/tests/test_trainer.py:405-441`,
  `lag_attn_transformer_e2e/tests/test_trainer.py:477-513`, and
  `test_the_base_preflight_is_a_no_op_so_a_subclass_cannot_drop_an_inherited_check`)
  still assert the same order. The first two stub the guard with a one-positional lambda
  and must absorb the new keyword; only their stub signatures change.
- `lag_attn_rws/eval/preflight.py:56-59,728-734` keeps its one-argument call.
- The refusal message names the offending field, whichever it is.

**Files.** `teb_vae/lag_attn_rws/trainer.py`,
`teb_vae/lag_attn_rws/tests/test_main.py`.

**Validation.** `pytest teb_vae/lag_attn_rws/tests/test_main.py teb_vae/lag_attn_transformer_rws/tests/test_trainer.py teb_vae/lag_attn_transformer_e2e/tests/test_trainer.py -q`.

### S0-T05 - `TRACKED_METRICS` and the plot-callback seam

**Description.** Three more class attributes on `LagAttnRwsTrainer`, each defaulting to
today's literal: `TRACKED_METRICS = _TRACKED_METRICS`, `PLOT_CONFIG_KEY =
"lag_attn_rws_plotting"`, and a `plot_callback_cls()` classmethod performing today's
lazy import. `train_model` reads the attributes instead of the globals.

**Acceptance criteria.**
- All three shipped packages behave identically; the lazy import stays lazy.
- A subclass can add tracked metrics without overriding `train_model`.
- `test_metric_tracking.py`'s `collector.tracked_metrics == _TRACKED_METRICS` assertion
  still holds for the shipped packages.

**Files.** `teb_vae/lag_attn_rws/trainer.py`,
`teb_vae/lag_attn_rws/tests/test_trainer.py`.

**Validation.** `pytest teb_vae/lag_attn_rws/tests/test_trainer.py teb_vae/lag_attn_rws/tests/test_metric_tracking.py teb_vae/lag_attn_transformer_rws/tests/test_metric_tracking.py -q`.

### S0-T06 - A forecast-row seam in `sample_page`

**Description.** `build_diagnostic_figure` (`sample_page.py:144`) is one function with
all seven rows inline and nested closures. Extract rows 1-2 behind a seam - a
`forecast_rows` callable, or a method on a small class - leaving the GridSpec, the row
cuts, rows 3-7 and the caption in one place.

**Acceptance criteria.**
- The rws figure is unchanged. **No pre-change reference exists** - `test_plotting.py`
  asserts per-row content and never the assembled page - so one is built first: a row
  inventory (per row, its artist counts and labels) committed green before the extraction,
  and a numeric before/after comparison of every drawn artist by the same seeded-script
  technique S0-T01 uses, over all three render variants (with statistics, without, no UP).
- The seam receives everything a feature-domain implementation needs, including the
  batch, so a caller can read `batch.fhr` directly rather than through
  `_build_raw_target`.
- Rows 3-7 and the layout are not duplicated by the seam. Asserted by driving the page with
  a seam that draws nothing and checking the other five rows still have data.
- **The dispatch route is decided and implemented here, not in Sprint 4.** The shared
  callback binds the page builder as a module global, so without one the seam exists and
  nothing can reach it. `plotting._generate_plots` reads a `forecast_rows` attribute off the
  task -- `None` for the raw models, which is what the builder turns back into its own
  implementation -- and passes `batch` beside it.

**Files.** `teb_vae/lag_attn_rws/sample_page.py`, `teb_vae/lag_attn_rws/plotting.py`,
`teb_vae/lag_attn_rws/tests/test_plotting.py`.

**Validation.** `pytest teb_vae/lag_attn_rws/tests/test_plotting.py -q`.

---

## Sprint 1 - The feature target

**Goal.** The 78-channel future block exists, is provably not delayed, and is proved on
real data.

**Landed 2026-08-08.** The package scaffold, the planted-pattern batch and forty-six tests --
forty-two fast, four `slow` against the committed shard -- all green, and both shipped suites
unchanged (nothing shared was edited; Sprint 1 adds files and touches none). Three things this
document got wrong or left underspecified. Each is **already corrected in place** at the task it
appears in:

1. **There are no shard helpers for this package to import.** S1-T01 asked the conftest to import
   "the shard helpers from the sibling conftests". The sibling's shard writer exists for the
   evaluation pipeline, which section 3 defers whole, and the two session-wide budget shrinkers
   that travel beside it are `autouse` fixtures whose bodies import `eval` to rebind its module
   globals -- importing either would pull the evaluation package into every run of this suite to
   no effect. The committed shard is reached the way `test_data_contract.py` reaches it instead:
   `absolutize_dataset_paths` over the sibling's `tiny.yaml`.
2. **The kept set's worst reach is $117.25$ s, not $117.2$.** Section 1.6's block is a dump at one
   decimal; S1-T03 restated it as an acceptance criterion, where the rounded value cannot be
   asserted. Same for the full set's median, $47.25$ rather than $47.2$.
3. **All 78 survivors carry a non-zero delay**, the fastest one step and the slowest thirty. S1-T02
   asked for the count without naming it, and the answer is what makes the negative test specific:
   a target built through the gate is wrong in *every* channel it contains, not in some of them.

### S1-T01 - Package scaffold

**Description.** `teb_vae/lag_attn_fs/{__init__.py,nets/__init__.py,tests/__init__.py}`
and `tests/conftest.py`.

**Acceptance criteria.**
- `conftest.py` **imports** the `utils` pre-import pin, `perturb_posterior`,
  `make_stub_batch`, `absolutize_dataset_paths` and both constructor keyword sets from the
  sibling conftests rather than restating them; a fourth verbatim copy of a load-order hack is
  the duplication the rest of this plan avoids. The generated-shard writer and the two
  session-wide budget shrinkers are **not** imported -- both exist for the evaluation pipeline
  this roadmap defers, and the shrinkers are `autouse`, so importing them would drag `eval` into
  every run of this suite for no effect.
- `tiny_kwargs` keeps `raw_per_step` (4.1) and adds no `decoder_out_channels`, so the
  default path is exercised.
- A stub batch carries a **known per-`(t, c)` pattern** in `fhr_st`/`fhr_ph`; on random
  data a transposed gather passes every shape check in S1-T02.
- A trivial test collects and passes.

**Files.** the four new files.

**Validation.** `pytest teb_vae/lag_attn_fs/tests -q`.

### S1-T02 - The future block: reuse the unfold, add the gather

**Description.** No new module. `teb_vae/lag_attn/figure_primitives.py:56`
`future_target(y_st, y_ph, horizon)` already concatenates and unfolds to
`(B, T - H, H, c_y)` and is what `lag_attn`'s eval and plotting are pinned against; the
fs model gathers the kept channels from its output with `torch.index_select`.

**Acceptance criteria.**
- Shape is `(B, 270, 30, 78)` at the shipped geometry; `T_valid = 300 - 30 = 270`.
- The index identity holds at the first and last valid anchor and at one interior
  `(t, tau)`, against the known pattern of S1-T01.
- **The no-delay negative test.** `ChannelGate.forward` is
  `self.delay(torch.index_select(x, -1, self.keep_index))` (`delays.py:233`) - there is
  no gather-only method, so the target must call `index_select` directly. The test
  asserts the target differs from `self.target_gate(...)` output, and names how many of
  the 78 kept channels carry a non-zero delay at the shipped budget, so the disagreement
  is specific rather than "not equal". The answer is **all 78** -- one step at the
  fastest survivor, thirty at the slowest -- so a gate-built target is wrong in every
  channel it contains.
- The `permute` is load-bearing: `unfold` yields `(B, T_valid, C, H)`. Assert equality
  against `future_target` rather than reimplementing it.
- No new file under `nets/`. `test_nets_are_framework_free.py` forbids the strings
  `fhr_st`/`fhr_ph` anywhere in `nets/*.py` **including docstrings**, which a feature
  target builder cannot honestly satisfy. That same rule means the model cannot *call*
  `future_target` either - its signature is those two block names - so the model unfolds its
  already-concatenated stream itself and S2-T04 pins the two equal.

**Files.** `teb_vae/lag_attn_fs/tests/test_feature_target.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_feature_target.py -q`.

### S1-T03 - Masks, and the budget-to-width binding on real data

**Description.** A test importing `forecast_mask`, `kl_mask` and `contributing_anchors`
unchanged and exercising them on a feature-shaped block, plus a `slow` check against the
committed shard.

**Acceptance criteria.**
- Masks broadcast over the channel axis, producing `(B, 270, 30)` and `(B, 300)`.
- A gap in `weight` removes the expected anchors.
- The resolved target keep-index has length **78** at `causal_reach_budget_s: 120` and
  **109** at `null`, and the kept set's reach is median $28.0$ s, max $117.25$ s. Section
  1.6's block prints at one decimal; the criterion needs the unrounded value, as it does
  for the full set's median $47.25$ s.
- Marked `slow` where it loads the shard.

**Files.** `teb_vae/lag_attn_fs/tests/test_masks.py`,
`teb_vae/lag_attn_fs/tests/test_budget_width.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_masks.py teb_vae/lag_attn_fs/tests/test_budget_width.py -q`.

---

## Sprint 2 - The model

**Goal.** A subclass, its contract, and the invariants that make its readout mean
anything.

**Landed 2026-08-08.** `SeqVaeLagAttnFs` forwards, forecasts and optimises at both geometries and
at every budget; 168 tests in this package (164 fast, 4 `slow`) and all four shipped suites green,
with the shipped parameter count unchanged at **3,371,725**. Six things this document got wrong or
left unplaced. Each is **already corrected in place** at the task it appears in:

1. **The width seam had no home, and Sprint 0 did not build it.** S2-T01's
   `decoder_out_channels = target_gate.out_channels if target_gate is not None else c_y` reads an
   attribute the *base* constructor creates, so nothing outside it can compute the value
   beforehand -- and a subclass that narrowed `__init__` to intercept the keyword would break the
   `inspect.signature` sweep in `trainer._build_model_kwargs`, which then forwards **no**
   configuration at all and silently builds an all-defaults model. Resolved by a
   `_default_decoder_out_channels()` method on `SeqVaeLagAttnRws`, called at the decoder's existing
   construction site: an exact no-op there (no RNG consumed, parameter count unchanged), and the
   one place the roadmap's own expression can be evaluated. The subclass therefore owns **three**
   callables, not one.
2. **`DESIGN.md`'s 3,371,725 is the *unguarded* model.** At the shipped $120$ s budget the raw
   model holds **3,377,997** -- a guarded run builds availability input adapters, worth $6{,}272$
   -- so the feature model at that budget is **3,393,993** and the $+15{,}996$ delta is against
   $3{,}377{,}997$. The $+23{,}994$ figure of 1.3 is the *unguarded* pair.
3. **`future_index` is inherited, not absent.** The base constructor registers it and the subclass
   cannot drop it without overriding `__init__`. It is non-persistent, so it reaches no checkpoint,
   and the test asserts the stronger thing instead: zeroing it moves no reported metric, so the raw
   index grid reaches nothing this model produces.
4. **`load_checkpoint_strict` returns `None`, not non-`None`.** It evaluates a candidate's
   alignment *before* loading anything and skips it on any shape mismatch, so an rws blob's four
   decoder-head tensors make it refuse outright and write no weight. The class guard still earns
   its place -- it fires first and names the model that wrote the blob rather than naming misaligned
   keys -- and the same all-or-nothing property is what refuses a **cross-budget `fs` checkpoint**,
   which the class guard cannot separate at all because both arms stamp the same `model_class`.
5. **`figure_primitives.future_target` cannot be called from `nets/`.** Its signature is the two
   *named stored blocks*, which is exactly the schema knowledge the framework-free guard forbids
   there. The model unfolds its already-concatenated stream itself -- three tokens, gathering
   before the unfold so the copy stays $(B, T, C)$ rather than $(B, T_{\mathrm{valid}}, H, C)$ --
   and `test_objective.py` pins the result equal to the shared helper.
6. **Three sibling copies of the import guard, not two**, and two of them carry an assertion that
   enumerates the package list, so those move too.

One measurement worth carrying into Sprint 3: the step-wise causality claim holds **only under
`causal_norm: true`**. The tiny keyword set does not set it, and without it the encoders' time-
pooling normaliser mixes the whole sequence, so a forward probe at the tiny fixture is not
bit-stable at the cut. Both directions are pinned in `test_smear.py`.

### S2-T01 - `SeqVaeLagAttnFs`

**Description.** `class SeqVaeLagAttnFs(SeqVaeLagAttnRws)` resolving
`decoder_out_channels = target_gate.out_channels if target_gate is not None else c_y`
and overriding `compute_loss`. The trf precedent's docstring justifies a standalone class
by the constructor building components it replaces and validating a keyword schema it
lacks; neither applies here, so a subclass it is.

**Acceptance criteria.**
- Overrides `compute_loss` and nothing else structural. Three own callables in total:
  `compute_loss`, the width hook `_default_decoder_out_channels` (which names a number and
  builds nothing) and `_build_forecast_target` (a new method, overriding none). No `forward`
  and **no `__init__`** - the constructor signature must stay the sibling's, because
  `trainer._build_model_kwargs` sweeps it with `inspect.signature` and a narrowed one forwards
  no configuration at all.
- Constructs at tiny and shipped geometry; decoder width is 78 at the shipped budget,
  109 at `null`.
- Records the parameter counts as constants. `lag_attn_rws/DESIGN.md:41`'s **3,371,725** is the
  **unguarded** shipped model; at the shipped 120 s budget the raw model is **3,377,997**
  (availability adapters, $+6{,}272$) and this one **3,393,993**. The decoder delta is
  $258 \times (C - 16)$: **+15,996** at 78 channels, **+23,994** at 109.
- Refuses the same geometry errors with the same messages - compared as strings, not merely as
  "both raise".

**Files.** `teb_vae/lag_attn_fs/nets/model.py`,
`teb_vae/lag_attn_fs/tests/test_construct.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_construct.py -q`.

### S2-T02 - The forward contract

**Description.** A test pinning the exact forward key set and every shape.

**Acceptance criteria.**
- The key set is compared by equality and has 20 entries.
- `mu_base`, `logvar_base`, `mu_full`, `logvar_full` are `(B, 270, 30, 78)`.
- No `decoder_state` and no `delta_mu_src`. The `future_index` buffer is **inherited and
  present**: the base constructor registers it and a subclass can only drop it by overriding
  `__init__`, which the width hook exists to avoid. It is non-persistent, so it reaches no
  checkpoint, and what is asserted instead is stronger - zeroing it moves no reported metric.

**Files.** `teb_vae/lag_attn_fs/tests/test_forward_contract.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_forward_contract.py -q`.

### S2-T03 - The four structural invariants

**Description.** Source purity, exact zero KL at initialisation, no decoder bypass, and
the shared decoder. Parametrise the sibling's tests over both classes where the fixture
allows rather than writing four new files.

**Acceptance criteria.**
- Resampling the source leaves `mu_prior`, `logvar_prior`, `raw_logvar_prior`,
  `target_state`, `z_prior`, `mu_base`, `logvar_base` bitwise unchanged.
- Gradient reaches the decoder only through `z`; the decoder's forward takes one tensor.
- Every KL assertion uses `perturb_posterior`.
- **The zero-KL claim states its fixture's flags.** It holds under `TINY_KWARGS`, which
  sets none of them; the shipped config ships `base_decode: mean` (forecasts not bitwise
  identical) and `posterior_logvar_mode: independent` (zero init KL only with
  `head_init_calibration: true`). Say which is being asserted.

**Files.** `teb_vae/lag_attn_fs/tests/test_invariants.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_invariants.py -q`.

### S2-T04 - `compute_loss` wiring, and the `block_width` trap

**Description.** The model's `compute_loss` builds the feature target and delegates with
`block_width = C_keep`.

**Acceptance criteria.**
- `total_loss` recomposes from its four weighted terms to `1e-6`.
- `nll_full_sample == nll_full_block / 2340` at the shipped budget, pinned against a
  **hand-computed** constant rather than against the implementation - a self-consistent
  ratio passes for any wrong-but-consistent width.
- `mean_logvar_full` checked against a hand computation. This is the only thing that
  catches `block_width` being passed as `geometry.r`: that mistake changes no loss, fails
  no shape check, and rescales the four log-variance diagnostics by $4.9\times$.
- The metric key set equals the sibling's exactly.
- The masked-plant assertion: `1e9` at a masked position moves the loss by exactly zero.

**Files.** `teb_vae/lag_attn_fs/nets/model.py`,
`teb_vae/lag_attn_fs/tests/test_objective.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_objective.py -q`.

### S2-T05 - The checkpoint contract

**Description.** A test that a checkpoint carries `model_class` and `model_kwargs` and
that a foreign blob is refused.

**Acceptance criteria.**
- `target_keep_index` is stamped in `model_kwargs`; the decoder width is not otherwise
  recoverable, and `decoder_out_channels` is deliberately absent so no second field can
  disagree with the gate.
- `raw_per_step` is present (4.1), unlike the draft's earlier assumption.
- **An rws blob is refused**, and the mechanism is not the one drafted here.
  `load_checkpoint_strict` evaluates a candidate's alignment *before* loading anything and skips
  it on any shape mismatch, so the four decoder-head tensors make it return `None` and write no
  weight - it does not partially align. The class check still fires first, and what it buys is
  the **message**: without it the failure names misaligned keys rather than the model that wrote
  the blob.
- **A cross-budget `fs` blob is refused too**, and the class check cannot help there at all:
  both arms stamp `SeqVaeLagAttnFs`, and only the width the stamped keep-index implies separates
  them.

**Files.** `teb_vae/lag_attn_fs/tests/test_checkpoint_contract.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_checkpoint_contract.py -q`.

### S2-T06 - The smear argument, made falsifiable

**Description.** Section 1.5's "this is not leakage" currently exists only as prose. Two
assertions give it a test.

**Acceptance criteria.**
- The forward at anchor `t` is bitwise unchanged when raw signal past the anchor's causal
  endpoint is perturbed, at the shipped reach budget.
- The blend fraction `b(tau, rho)` and the section 1.6 table are recomputed from the
  shipped filter bank and matched to the recorded numbers, so the preprint of S7-T02
  prints figures some test reproduces.

**Files.** `teb_vae/lag_attn_fs/tests/test_smear.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_smear.py -q`.

### S2-T07 - `nets/` import purity

**Description.** Port `test_nets_are_framework_free.py` and extend the
`_FRAMEWORK_PREFIXES` tuples in the sibling copies to name `lag_attn_fs`.

**Acceptance criteria.**
- `nets/*.py` imports only torch, stdlib, entmax and sibling nets.
- No batch field name appears in `nets/*.py`, docstrings included. This is what forbids calling
  `figure_primitives.future_target` from `nets/`: its signature *is* the two stored block names.
- **Three** sibling copies are extended, not two - `lag_attn_rws`,
  `lag_attn_transformer_rws` and `lag_attn_transformer_e2e` each carry a package tuple, and the
  last two also carry an assertion enumerating it, so the count in those moves as well.

**Files.** `teb_vae/lag_attn_fs/tests/test_nets_are_framework_free.py`, and the three
sibling copies.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_nets_are_framework_free.py -q`.

---

## Sprint 3 - Task, trainer, configuration

**Goal.** A tiny epoch runs, with the loss-scale constants re-derived rather than
inherited.

**Landed 2026-08-08.** `python -m teb_vae.lag_attn_fs.trainer` completes a tiny epoch through the
real framework, writes a checkpoint that rebuilds at its own width, and records every tracked
column; 336 fast tests and 20 `slow` in this package, all green, and no shipped file was edited --
Sprint 3 adds files to `lag_attn_fs` and touches nothing else, so all three shipped suites are
unchanged by construction.

The two loss-scale constants were re-derived from a 120-epoch instrumented run on the committed
HIE shard, and **one of them did not move**: the pre-clip gradient-norm distribution came out close
enough to the raw sibling's that the same rule returns the same $5000$, so `gradient_clip_val` left
the parity allow-list rather than staying in it at a value that no longer differs. The margin did
move, to $5\times 10^{3}$. S3-T06 records both, with the run and the percentiles.

Five things this document got wrong or left underspecified. Each is **already corrected in place**
at the section or task it appears in:

1. **Section 4.7's "removed: `raw_per_step`" contradicted 4.1**, which says it stays. It stays, and
   the consequence is larger than one key: with no schema difference at all, the parity pin is
   *total* rather than schema-limited, so the fs allow-list is nine declared leaves against every
   other leaf in both files.
2. **There is no `test_main.py`.** S3-T02's file list asks for one, but the entry point here is a
   one-line delegation whose guards live in the shared module, and the subclassing precedent this
   package follows -- `lag_attn_transformer_rws` -- has no such file either: its guard-order, its
   `RUN_CONFIG` and its resolved-config assertions all live in `test_trainer.py`. So do these.
3. **S3-T05's file list omits `task.py`, and the split it adds could not work without it.** The
   per-block gap needs the boundary between the two stored blocks, which is not derivable from
   $c_y$ -- it is their *sum* -- so the net declares it as `TARGET_BLOCK_SPLIT` and the task, which
   is the only layer that sees the two blocks separately, refuses a batch that disagrees. Nothing
   else depends on the number, which is exactly why it needs a check: a stale value mislabels two
   reported columns and breaks nothing.
4. **Two of Sprint 2's landed assertions had to move**, and neither was wrong when it was written:
   `test_objective.py`'s "the metric key set is the sibling's **exactly**" and
   `test_construct.py`'s "the subclass defines **two** methods". S3-T05 adds four metrics and one
   method by design. Both are now stated against a declared addition rather than relaxed.
5. **S3-T07's `broadcast_buffers` justification named a buffer this model was said not to have.**
   Sprint 2 already recorded the opposite: `future_index` is inherited and present, merely never
   read. The setting is safe for the reason it was always safe -- every buffer is a deterministic
   function of the config and there is no `BatchNorm` anywhere -- and an unread buffer only makes
   the broadcast more wasteful.

**One thing this sprint leaves visibly unfinished, deliberately.** The diagnostic figure does not
render. The shared page's first two rows read the target as a raw trace and plot it against the raw
time axis; handed a $(B, T, c_y)$ feature block they raise on the length mismatch, the callback
catches it as designed, and the run writes no figure. S4-T02's note predicted the *softer* failure
-- "silently draws the raw page against a feature target" -- and the real one is louder in the log
and quieter on disk. `test_train_smoke.py` asserts both halves: the callback is attached and never
fails the fit, and no figure appears until the `forecast_rows` seam is filled. **Sprint 4 filled
it**, and that assertion -- written to fail the moment it was -- is now one counting figures per
plotted epoch.

### S3-T01 - The task subclass

**Description.** `SeqVaeLagAttnFsTask(SeqVaeLagAttnRwsTask)` overriding
`_build_raw_target` alone, returning the concatenated `(B, T, c_y)` feature tensor and
`weight` in the same positional slot `fhr_raw` occupies.

**Acceptance criteria.**
- **Exactly one** override. `_mu_gap_rms` needs none: `task.py:539-568` reads only
  `model.geometry`, `model.coverage_floor`, the two masks and `mu_post`/`mu_prior`, all
  present here. If this is asserted by set equality over the subclass's own callables,
  write it so S4-T02's `forecast_rows` **addition** does not read as a second override -
  it overrides nothing, and the shared callback resolves it with `getattr(..., None)`.
- Doing the concatenation task-side keeps `plotting.py:249` working unchanged.
- `main_loss` is still set, unprefixed, so the spike breaker watches the right name.
- The permutation control still emits its three metrics on validation steps only, is
  gated at `batch_size >= 2`, and leaves the prior and base branch bitwise unchanged
  under a derangement.

**Files.** `teb_vae/lag_attn_fs/task.py`, `teb_vae/lag_attn_fs/tests/test_task.py`,
`teb_vae/lag_attn_fs/tests/test_perm_control.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_task.py teb_vae/lag_attn_fs/tests/test_perm_control.py -q`.

### S3-T02 - The trainer subclass

**Description.** `LagAttnFsTrainer(LagAttnRwsTrainer)` setting `MODEL_CLS`, `TASK_CLS`,
`CHECKPOINT_STEM`, `TARGET_FIELDS = ("fhr_st", "fhr_ph")`, and a one-line `main`
delegating to the shared runner.

**Acceptance criteria.**
- The subclass's own callables are asserted by set equality, following
  `lag_attn_transformer_rws/tests/test_task.py:64` - **not** a line count, which passes a
  subclass that overrides `training_step` in 140 lines and silently disables the spike
  breaker.
- A guard-order test asserts `lag_attn_fs.trainer.main` reaches the **fs** `TARGET_FIELDS`,
  not the rws one - the failure a `trainer_cls=` wiring mistake produces.
- The `RUN_CONFIG` module-constant convention is honoured: runs from the IDE Run button
  with no command line, no `required=True`, and a refusal message naming the constant.
- **No `test_main.py`.** The entry point is a one-line delegation and every guard it runs belongs
  to the shared module; the subclassing precedent this package follows carries its guard-order,
  `RUN_CONFIG` and resolved-config assertions in `test_trainer.py` and has no `test_main.py` at
  all. A second file would be a second place for the same four assertions to drift.

**Files.** `teb_vae/lag_attn_fs/trainer.py`, `teb_vae/lag_attn_fs/tests/test_trainer.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_trainer.py -q`.

### S3-T03 - `default.yaml` and its parity pin

**Description.** The production config, written out in full, pinned leaf-for-leaf against
the rws file outside a declared allow-list.

**Acceptance criteria.**
- The allow-list names its contents rather than being open-ended: the retuned
  `beta_schedule.end` and `beta_prior`, the re-derived `additive_margin`, and the four
  **IDENTITY** exemptions the transformer package marks for the same reason -
  `folders_config.out_dir_base`, `mlflow.experiment_name`, `mlflow.run_name`,
  `mlflow.tags.variant`, plus `general_config.tag`. Inheriting those writes fs runs into the rws
  output tree and MLflow experiment.
- **Eight entries, not nine.** `gradient_clip_val` was drafted as an exemption, was genuinely
  re-derived (S3-T06), and came out **equal** to the comparison model's - so it is not exempt, and
  a separate test says that equality is a measurement rather than an oversight. An exemption for a
  key that no longer differs is a permission that outlived its reason.
- The comparison is **total**, not schema-limited: the constructor schema is unchanged, so every
  key means the same thing in both files and every leaf outside the allow-list is compared. That is
  a stronger pin than the transformer package's, which must exclude twelve encoder keys.
- `raw_per_step: 16` present, with a comment saying it is a geometry input and no longer
  the decoder width.
- The plotting block keeps the name `lag_attn_rws_plotting`, with a comment saying why.
- The retuned $\beta$ and $\beta_p$ hold the ratio 4.8 fixes, and the **direction** is asserted
  rather than described: a larger reconstruction at fixed $\beta$ makes $\beta\,\mathrm{KL}$
  relatively weaker, so the scale-matched value is *above* the inherited one. Shipped
  $\beta = 5.0$, $\beta_p = 0.5$; the sweep of S5-T03 brackets that rather than the inherited $1.0$.
  **Superseded by S5-T05.** The sweep ran and chose its lower edge, so the shipped pair is now
  $\beta = 1.0$, $\beta_p = 0.1$ - the comparison model's own values, reached by measurement. Two
  consequences land back here: the allow-list is **six** entries rather than eight, because an
  exemption for a key that no longer differs is refused by the very test this task added; and the
  direction assertion above is no longer a claim about the shipped value, only about where the
  scale-matched point sits. The ratio $\beta_p / \beta = 0.1$ still holds and is still asserted.

**Files.** `teb_vae/lag_attn_fs/configs/default.yaml`,
`teb_vae/lag_attn_fs/tests/test_config_load.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_config_load.py -q`.

### S3-T04 - `tiny.yaml` as a delta

**Description.** `base: default.yaml` plus only its overrides, following
`lag_attn_rws/configs/tiny.yaml:18`.

**Acceptance criteria.**
- Names only its deltas, under the sibling's non-comment line budget.
- Ships `likelihood: mse` so the smoke path exercises the DDP fallback.
- Keeps the real geometry.

**Files.** `teb_vae/lag_attn_fs/configs/tiny.yaml`,
`teb_vae/lag_attn_fs/tests/test_config_load.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_config_load.py -q`.

### S3-T05 - The observability metrics

**Description.** Per 4.9: `pred_gap` resolved by horizon step and split by feature block,
added through the `TRACKED_METRICS` seam of S0-T05.

**Acceptance criteria.**
- Exactly `pred_gap_tau_first`, `pred_gap_tau_last`, `pred_gap_st` and `pred_gap_ph`.
- Computed by reducing the existing block score over `C` alone, not by a second forward. The
  per-element term is `losses.raw_sample_score` and the mask is rebuilt through the objective's own
  two functions, so all four stay **partial sums of the `pred_gap` beside them** -- the only
  property that makes them worth reporting. Both splits recompose, and that is asserted.
- Reach `metrics_history.csv` and the loss-curve HTML.
- Two-direction reachability: every emitted metric tracked, every tracked metric
  reachable.
- **`task.py` too, which the drafted file list omits.** The block split needs the boundary between
  the two stored blocks, and $c_y$ is their sum, so the net declares `TARGET_BLOCK_SPLIT` and the
  task -- the only layer that sees the two blocks separately -- refuses a batch that disagrees.
  Nothing else depends on the number, so a stale one would mislabel two columns silently.
- **Two landed Sprint-2 assertions move**, and neither was wrong when written: S2-T04's "the metric
  key set is the sibling's exactly" becomes "the sibling's plus these four, in both directions",
  and S2-T01's "the subclass defines two methods" becomes four. Both stay set equalities.

**Files.** `teb_vae/lag_attn_fs/trainer.py`, `teb_vae/lag_attn_fs/nets/model.py`,
`teb_vae/lag_attn_fs/task.py`, `teb_vae/lag_attn_fs/tests/test_metric_tracking.py`,
`teb_vae/lag_attn_fs/tests/test_objective.py`, `teb_vae/lag_attn_fs/tests/test_construct.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_metric_tracking.py teb_vae/lag_attn_fs/tests/test_objective.py -q`.

### S3-T06 - The loss-scale constants, re-derived

**Description.** `gradient_clip_val` and `spike_breaker.additive_margin` were both
derived from the 480-sample block and the shipped config says the clip does **not**
transfer across loss scales. Re-derive both from a short instrumented run at the new
scale.

**Acceptance criteria.**
- The clip is set from observed `train/grad_norm` percentiles, and the config comment
  records the run and the percentiles, as `lag_attn_rws/configs/default.yaml:357-371`
  does.
- `additive_margin` is re-derived from observed `main_loss` fluctuation; the old value
  was `~2 * 480` and the same derivation at 2340 lands near `5e+3`.
- `ema_floor: 1.0e+9` is confirmed to still exceed any reachable loss, and the comment
  says so rather than leaving it assumed.
- `test_spike_breaker.py` is ported, since the loss is sign-indefinite and summed here
  more than anywhere.

**Measured.** 120 epochs over the committed `output/hie_cs.hdf5` shard (339 windows, batch 32, one
device, `gaussian_nll`, $\beta$ ramping to $5.0$ over 20 epochs), with the clip itself at `1.0e+9`
so nothing rescaled the steps the norms were drawn from. **The step count first recorded here was
wrong** and is corrected in the config comment: Lightning's own log for that run reports **11**
training batches per epoch, not $\approx 33$, so the sample is $120$ steps thinned from $1{,}320$
rather than from $\approx 3{,}900$. The percentiles are unaffected - the CSV records one step per
epoch either way - and so is the chosen threshold.

- **The clip did not move.** Pre-clip `train/grad_norm` over the 120 sampled steps: min $1187$,
  q50 $2147$, q90 $3156$, q95 $3416$, q99 $4421$, q99.9 $4706$, max $4741$ -- close enough to the
  raw sibling's (q50 $2775$, q99 $4681$) that the same rule, *the smallest round value above q99*,
  returns the same $5000$. So the exemption drafted for this key in S3-T03's allow-list is
  **removed**: an exemption for a value that no longer differs would wave through the next
  accidental divergence. The reason it did not move belongs beside it: the reconstruction sums over
  $78$ channels, but the decoder's output head is per-channel, so the extra terms land on disjoint
  rows of two `Linear` layers rather than accumulating onto one shared parameter -- the *loss*
  scales with the block and the norm of its gradient does not.
- **The margin did.** Over the 100 post-warm-up epochs `train/main_loss` sat at $4562 \pm 404$
  (min $3493$, max $5683$) with epoch-to-epoch $|\Delta|$ of median $323$, q90 $745$, q99 $1316$,
  max $1500$. `5e+3` is $3.3\times$ the largest ordinary movement and $\approx 12\times$ the
  standard deviation; the scaling argument that produced the provisional value lands in the same
  place, which is a confirmation rather than the reason.
- **The floor holds with five orders of magnitude to spare.** The per-coefficient Gaussian NLL is
  bounded below by $\tfrac{1}{2}(\log 2\pi + \ell_{\min}) \approx -1.6$ at the shipped clamp, so
  the two reconstruction terms cannot exceed $\approx 7.5 \times 10^{3}$ in magnitude; the KL and
  the anchor are nonnegative. Both the bound and the observed range are asserted.

**Not recorded in `RESULTS.md`.** S5-T01 requires that file to be written *before* anything writes
into it, so the derivation lives where the acceptance criterion puts it -- in the config comment --
and S5-T01's "the gradient-clipping threshold - measured" section quotes it rather than the reverse.

**Files.** `teb_vae/lag_attn_fs/configs/default.yaml`,
`teb_vae/lag_attn_fs/tests/test_spike_breaker.py`,
`teb_vae/lag_attn_fs/tests/test_config_load.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_spike_breaker.py teb_vae/lag_attn_fs/tests/test_config_load.py -q`.

### S3-T07 - DDP and the tiny smoke

**Description.** The DDP strategy test and one real fit through the real framework.

**Acceptance criteria.**
- Under `tiny.yaml`'s `likelihood: mse` the starved parameter set is **exactly** the
  decoder log-variance heads, so `find_unused_parameters=True` is justified rather than
  assumed.
- `ddp_kwargs`'s `broadcast_buffers=False` justification is restated, and **not** as the draft
  had it: that buffer list includes the raw-target index grid, and this model *does* still carry
  it -- Sprint 2 recorded that the base constructor registers it and a subclass can only drop it by
  overriding `__init__`. It is simply never read. The setting is safe for the reason it always was
  (every buffer is a deterministic function of the config; there is no `BatchNorm` anywhere), and
  an unread buffer only makes the broadcast more wasteful.
- One epoch completes, a checkpoint is written, `metrics_history.csv` has the expected
  columns -- including the four of S3-T05, whose two splits are checked to recompose from the CSV
  itself. Marked `slow`.
- **The diagnostic figure does not render, and that is asserted rather than left unstated.** The
  shared page's raw first two rows raise on a feature target and the callback catches it, so the
  run writes no figure until S4-T02 fills the seam. What the smoke asserts is that the callback is
  attached and never fails the fit.

**Files.** `teb_vae/lag_attn_fs/tests/test_ddp_strategy.py`,
`teb_vae/lag_attn_fs/tests/test_train_smoke.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_ddp_strategy.py -q`;
`pytest teb_vae/lag_attn_fs/tests/test_train_smoke.py -q -m slow`.

---

## Sprint 4 - The diagnostic page

**Goal.** A seven-row page for a real batch, filling the S0-T06 seam.

**Landed 2026-08-08.** The tiny fit through the real entry point now writes a page per plotted epoch
per drawn sample -- seven rows, the forecast row carrying three data-chosen target channels and a
$78 \times 30$ error map, the lag axis reading $120$-$155$ s at the shipped delay. 384 tests in this
package (364 fast, 20 `slow`) all green, and so are the suites that own the two shipped files this
sprint edited: `lag_attn_rws/tests/test_plotting.py` and `test_eval_self_contained.py` (101 with
`test_eval_samples.py`) and the whole 435-test `lag_attn` suite. The rws row inventory passing
unchanged is what says the shared row-1 extraction below drew nothing differently.

Six things this document got wrong or left underspecified. Each is **already corrected in place** at
the section or task it appears in:

1. **`torch.equal` is unreachable for the lag-map identity, and always was.** S4-T03 asked for it.
   The map is contracted with `einsum` over the *head* axis while $K_t$ is summed over the
   *dimension* axis, so the two reach the same number by different summation orders: measured at
   both normalisers the gap is $\approx 10^{-6}$ on values of order $10$. The sibling's own
   `test_lag_map.py` has always used `allclose(atol=1e-5)`, and so does this one. Acceptance
   criterion 6's "exactly" means round-off, not bitwise.
2. **The error map cannot be a panel of its own**, and the same task says why. Its x-axis is the
   horizon step $\tau$ of one anchor, not physical time, so a side-by-side split of the forecast row
   would leave the curves narrower than the other six rows and break S4-T02's own criterion that a
   column of the page stays one instant across all seven. It is an **untitled inset** in the
   forecast axes -- untitled deliberately, so "every titled axes spans the recording" stays exactly
   checkable -- using the row's reserved colorbar column, which the raw page leaves hidden.
3. **Two of the four primitives S4-T02 names do not fit, and one had to be written.**
   `average_forecast_per_channel` averages overlapping anchors, blending $H$ latents into one curve
   -- the sibling page rejects that for its own forecast row and the reason carries over, so this
   row tiles with `concat_single_forecasts` as that one does. `stack_feature_blocks` would be a
   no-op concatenation of two slices of one array purely to recover an index the block split
   already gives. `safe_vabs` is used, for the map's `vmax`. And S4-T01 is a **new** primitive
   rather than a lift: `lag_attn/plotting.py:701-732` picks the single worst-calibrated channel
   inline and returns nothing reusable.
4. **Row 1 is shared, not rebuilt.** Section 4.6 says rows 1-2 "are rebuilt"; row 2 is, and row 1 is
   the sibling's own implementation, extracted out of `raw_forecast_rows` into a
   `raw_context_row(rows, fhr_values)` both pages call. The only thing that differs between the two
   pages is where the trace comes from -- the raw page's target, this one's batch -- and a copy of
   twenty lines of twin-axis and denormalisation handling is the duplication the rest of this plan
   avoids. The extraction is inert, proved by the row inventory S0-T06 built for exactly this.
   `_BAND_SIGMAS` became public `BAND_SIGMAS` for the same reason: two pages of one family quoting
   different intervals under the same $\pm k\sigma$ caption is a difference nobody would look for.
5. **The seam needs two facts the page cannot derive from what it is handed**, so `forecast_rows` is
   a property returning a `functools.partial` rather than a bare function. The keep-index says which
   declared channel each decoder output *is* -- needed both to gather the truth a lane is judged
   against and to label that lane with a number that survives a change of reach budget -- and the
   block split is where the two stored blocks meet on the error map's channel axis. A plain class
   attribute would also have bound `self` into the row builder's first parameter.
6. **S4-T01's "deterministic given `(keep_index, batch)`" names the wrong inputs.** The rule ranks on
   the predictive calibration of the drawn block, so it is a function of the truth, the mean and the
   sigma; the keep-index enters at the *label*, which is precisely where the channel-count failure
   that removed `lag_attn`'s `forecast_channels` key actually bit.

### S4-T01 - The channel-selection rule

**Description.** A named, tested function choosing which target channels the forecast row
draws. `lag_attn/plotting.py:701-732` already selects the worst-calibrated channel and
draws its band, commented "Data-driven rather than configured" - generalise that rule into
`figure_primitives.py`, the module whose stated purpose is to hold exactly this. It is a **new**
function rather than a lift: that code picks one channel inline, mid-figure, and returns nothing a
second caller could reach.

**Acceptance criteria.**
- Deterministic given the block it ranks -- the truth, the predictive mean and the predictive
  sigma. **Not `(keep_index, batch)`, as first drafted:** the rule reads calibration, and the
  keep-index enters one step later, at the label, which is where the failure below actually bit.
  Ties break by channel index, so equal coverage -- the common case early in training, where every
  channel sits at $0$ or at $1$ -- cannot make the drawn lanes hop between epochs.
- Stable across a channel-count change - this is the failure that removed `lag_attn`'s
  `forecast_channels` key when `fhr_ph` went from 44 to 66.
- No hard-coded indices anywhere.
- **Three channels, not one:** the worst, the middle and the best of the calibration ranking. A
  panel showing only the worst reads as a broken model on every run and one showing only the best
  as a working one; the shipped single-channel rule is this one's `count=1` case.
- Positions with no finite element are ignored rather than counted as misses -- the tiled forecast
  is `NaN` wherever no window covers the step, and counting those would rank a channel by how much
  of the recording the tiling happened to reach. A channel with no scorable element at all scores
  $0$ and sorts worst, which is the honest outcome.

**Files.** `teb_vae/lag_attn/figure_primitives.py`,
`teb_vae/lag_attn_fs/tests/test_sample_page.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_sample_page.py -q`.

### S4-T02 - The feature forecast rows

**Description.** Fill the S0-T06 seam: row 1 the raw FHR/UP context, row 2 the
three-channel forecast with $\pm 2\sigma$ plus the `(H x C)` error heatmap. Built from
`concat_single_forecasts` and `safe_vabs`, both already framework-free and already imported by the
rws page, plus S4-T01's new selector. **Two of the four primitives first named here do not fit:**
`average_forecast_per_channel` averages overlapping anchors, blending $H$ latents into a curve
neither the model nor the objective ever produces -- the sibling's forecast row rejects that and
this one inherits the reason -- and `stack_feature_blocks` would be a no-op concatenation of two
slices of one array, purely to recover an index the block split already supplies. Neither was
imported by the rws page either.

As landed, the seam is a callable taking one
`lag_attn_rws.sample_page.ForecastRowInputs` and returning nothing, drawing into
`rows.row_axes(RAW_ROW)` and `rows.row_axes(FORECAST_ROW)` through
`rows.finalise_time_axis`. **Two pieces of work, not one:** the function, and a
`forecast_rows` property on `SeqVaeLagAttnFsTask` returning it - that attribute is what the
shared callback reads. It is a `functools.partial`, because the page needs two facts it cannot
derive from the arrays it is handed -- the target gate's keep-index and `TARGET_BLOCK_SPLIT` -- and
because a bare function assigned as a class attribute would bind `self` into the row builder's
first parameter.

Without it the fs run draws **no page at all**, which is not what this note first predicted. The
raw rows plot the target against the raw time axis, so a $(B, T, c_y)$ feature block raises on the
length mismatch, the callback catches it and warns, and nothing reaches disk - louder in the log
and quieter on disk than "silently draws the raw page". Sprint 3 asserts that state; the assertion
is written to fail the moment this task lands, and is replaced here by one counting figures per
plotted epoch.

**Acceptance criteria.**
- Row 1 reads `rows.batch.fhr` **directly**, not `rows.target`, which is now the feature
  block. It is drawn by the *sibling's* implementation, not a copy: section 4.6's "rebuilt" holds
  for row 2 and not for row 1, so `raw_forecast_rows`'s first row is extracted into a shared
  `raw_context_row(rows, fhr_values)` that both pages call and the rws row inventory still pins.
- The row axes come from `rows.row_axes` and the limits from `rows.finalise_time_axis`, so
  a column of the page stays one instant across all seven rows. **That is what makes the error map
  an inset** rather than a second panel: its x-axis is the horizon step of one anchor, so a
  side-by-side split of the row would leave the forecast curves narrower than every other row. It
  carries no title, so "every titled axes spans the recording" stays checkable, and it takes the
  row's reserved colorbar column, which the raw page leaves hidden.
- Renders with `normalization_stats_of` returning `None`, and says so on the figure. The forecast
  row says so unconditionally: the target is the loader's `normalize_fields` output used as
  delivered (3), so there is no second normalisation to invert and no physical unit to claim.
- Renders at both tiny and shipped channel counts.
- No `lag_attn_fs/plotting.py` and no new callback class - the rws callback is reused
  through the `plot_callback_cls()` seam, and reaches these rows through the task's
  `forecast_rows` attribute.

**Files.** `teb_vae/lag_attn_fs/sample_page.py`, `teb_vae/lag_attn_fs/task.py`,
`teb_vae/lag_attn_fs/tests/test_sample_page.py`, and -- for the shared row-1 extraction --
`teb_vae/lag_attn_rws/sample_page.py`, whose figure the rws suite proves unchanged.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_sample_page.py -q`.

### S4-T03 - Lag consistency and the lag map

**Description.** The figure's lag axis against the model's, and the conservation identity.

**Acceptance criteria.**
- The two lag axes agree exactly, including the `source_delay_steps` offset. With the evaluation
  deferred whole (3) the two consumers are the ones that exist: the model, which reports $\delta$
  through one accessor, and the page, which converts a lag index to seconds on **both** of its lag
  panels. Both panels are checked against `lag_compensated_seconds` of their own primary limits, at
  the guarded delay and at zero, and against each other. The secondary axis must be read **after a
  draw** -- matplotlib defers its limits, so an assertion made before one passes against the
  default $(0, 1)$ whatever the transform is, which is the bug this file exists to catch.
- `te_analysis` output sums over lags to `kld_per_t`, asserted at the **nets** level so acceptance
  criterion 6 does not depend on Sprint 4 shipping. **Not under `torch.equal`, as drafted:** the
  map is contracted with `einsum` over the head axis while $K_t$ is summed over the dimension axis,
  so the two agree to round-off and not bitwise -- $\approx 10^{-6}$ measured, against the
  $10^{-5}$ the sibling's own copy of this test has always used.

**Files.** `teb_vae/lag_attn_fs/tests/test_lag_map.py`,
`teb_vae/lag_attn_fs/tests/test_lag_consistency.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_lag_map.py teb_vae/lag_attn_fs/tests/test_lag_consistency.py -q`.

---

## Sprint 5 - Local validation on real data

**Goal.** Evidence, against criteria registered before the run.

**Landed 2026-08-08.** Four arms over the committed HIE shard - $200$ epochs, $2{,}200$ optimizer
steps, one device, $\approx 25$ minutes each - meet **every** pre-registered criterion, and the sweep
they ran to settle changed the shipped configuration. 416 tests in this package (396 fast, 20 `slow`)
and all four shipped suites green. `RESULTS.md` carries the tables; the headline is that the four
arms are **monotone in $\beta$ on every column the selection rule reads** and the lower bracket wins
all of them, so `default.yaml` moved from the scale-matched $\beta = 5.0 / \beta_p = 0.5$ to
$\beta = 1.0 / \beta_p = 0.1$.

Seven things this document got wrong or left underspecified. Each is **already corrected in place**
at the section or task it appears in:

1. **Section 4.8's inference was refuted, though its direction was right.** A larger reconstruction
   at fixed $\beta$ does make $\beta\,\mathrm{KL}$ relatively weaker and does open the latent wider -
   the rate falls from $3.46$ to $0.23$ across the bracket. What 4.8 inferred from that, that the
   inherited $\beta$ would leave the latent carrying target information a permutation control might
   miss, is what the arms refute: at $\beta = 1.0$ the control fires **hardest** (`shuffle_penalty`
   $13.9$ against $2.9$ at $5.0$), the base forecast is best, and the gap is positive at the far
   horizon step where none of the target is determined by observed history. The scale-matching
   argument bracketed the right axis and put the shipped value on the wrong side of it.
2. **S3-T03's allow-list is six entries, not eight.** Two of its three retuned exemptions were
   measured back to parity with the comparison model, joining `gradient_clip_val`, which S3-T06
   already had. The consequence is the opposite of what "the allow-list still covers exactly these
   keys" reads like: satisfying S5-T05 meant **deleting** two exemptions, because
   `test_every_declared_parity_exemption_is_a_real_divergence` refuses one for a key that no longer
   differs. The three are now named in a `MEASURED_TO_MATCH_PATHS` tuple whose test asserts the
   equality, so parity reads as a measurement rather than as an oversight.
3. **Section 6 item 4's $10^{-6}$ is unattainable as an absolute tolerance**, and always was. See
   the corrected item.
4. **S5-T04 is four runs, not five.** `sweep_beta_5p0.yaml` restated the shipped weights when the
   sweep was written, so "the baseline run plus the four arms" is one run counted twice. After
   S5-T05 the restating arm is `sweep_beta_1p0.yaml`; the two swapped roles without either file
   changing a number, which is exactly what the restating arm exists for.
5. **S3-T06's step count was wrong** - $\approx 33$ per epoch against the run log's $11$. Corrected
   at that task.
6. **The clip does bind, rarely, and S3-T06's "does not bind at all" needed weakening.** Across the
   four arms' $800$ sampled steps it fired on three, for whole-run `grad_clip_frac` means of $0.000$
   to $0.010$ - twenty-five-fold inside the hold band, and precisely the thinner-tail caveat the
   config comment predicted for a $120$-step sample. No revision to the threshold: the median is
   stable to within $2\%$ across all four arms and a tenfold change in $\beta$.
7. **S5-T03's epoch-count criterion was nearly vacuous** as drafted - any production config has
   $5{,}000$ epochs against a patience of $5$ - so the lint additionally requires the run to outlast
   its own beta ramp *plus* the patience, which is the bound that makes a tail readable.

**One result that was not predicted at all.** The prior anchor holds at **every** weight in the
bracket, including $\beta_p = 0.1$ - the raw sibling's value carried across unchanged against a
$4.9\times$ larger reconstruction, finishing at a floor fraction of $0.0121$ where 4.8 predicted it
would be the arm most at risk. The threshold argument is not wrong; the pressure it describes reaches
the prior's log-variance through `base_decode` and `posterior_logvar_mode`, and this package ships
the configuration that removes both paths. **This configuration is not anchor-limited anywhere in the
bracket**, which makes `beta_prior` a far less delicate key here than the sibling's history suggests.

**What the local runs do not establish, stated once so no row is over-read.** Both splits are the
same $339$ windows, so every `val/` column is in-sample, and the sibling package saw exactly this
sign flip between its in-sample arms and its held-out production run. The positive `pred_gap` at
$\beta = 1.0$ is **not** a generalisation result and section 1.2's expectation is untouched. In-sample
runs reward capacity, so if the optimum moves at production scale it is more likely to move *up* than
down, and `sweep_beta_2p5.yaml` - second on every column - is the arm that would inherit.

### S5-T01 - The `RESULTS.md` skeleton

**Description.** The document and its column set, written **before** anything writes into
it.

**Acceptance criteria.**
- Sections: arm inventory, the headline selection rule stated before any table, "the
  gradient-clipping threshold - measured", "the prior-anchor weight", **"what reverts,
  and when to stop"** and **"go/no-go while a run is in flight"**. Both siblings carry the
  last two, and with no eval pipeline they are more load-bearing here, not less.
- Every backticked column name is in `TRACKED_METRICS` or a declared derived-quantities
  block.

**Files.** `teb_vae/lag_attn_fs/RESULTS.md`.

**Validation.** The binding test of S6-T01.

### S5-T02 - The local config

**Description.** `smoke_hie.yaml` over `output/hie_cs.hdf5`.

**Acceptance criteria.**
- Resolves with its base consumed; a model builds through the real driver.
- Points at the existing `output/hie_cs_stats.hdf5` - it is already generated from that
  shard, so no regeneration task exists.
- `normalize_fields` contains `fhr_st` and `fhr_ph`, which
  `lag_attn_rws/configs/default.yaml:318` already does.
- Runs on one GPU at a batch the dev box holds.

**Files.** `teb_vae/lag_attn_fs/configs/smoke_hie.yaml`,
`teb_vae/lag_attn_fs/tests/test_config_load.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_config_load.py -q`.

### S5-T03 - The arms

**Description.** Four `beta` arms bracketing the scale-matched value, each holding the
`beta_prior / beta` ratio fixed per 4.8.

**Acceptance criteria.**
- Values `1.0 / 2.5 / 5.0 / 10.0`, not the inherited bracket - the scale-matched value is
  near 4.9 and three of the drafted arms sat at or below the inherited 1.0. Two arms below that
  point and two at or above it is what let the sweep locate an optimum rather than only bound one,
  and the optimum turned out to be at the lower edge.
- Each differs from `default.yaml` by `beta_schedule.end` and `beta_prior` alone, asserted
  by a lint test. Two keys is the axis, not a second delta: the anchor's restoring force saturates
  at $\beta_p / 2$ per dimension while the reconstruction it opposes is what this domain multiplied,
  so the ratio is pinned too and a pinning prior has one explanation rather than two.
- Each arm's epoch count exceeds `collapse.KL_COLLAPSE_PATIENCE_EPOCHS`, imported rather
  than restated. **As drafted that is nearly vacuous** - a production config has $5{,}000$ epochs
  against a patience of $5$ - so the lint additionally requires the count to outlast the arm's own
  beta ramp *plus* the patience, which is the bound under which the criterion's tail is readable.

**Files.** `teb_vae/lag_attn_fs/configs/sweep_beta_*.yaml`,
`teb_vae/lag_attn_fs/tests/test_sweep_configs.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_sweep_configs.py -q`.

### S5-T04 - The validation run

**Description.** The baseline run plus the four arms, against the pre-registered criteria
of section 6.

**Acceptance criteria.** Exactly section 6 item 4, including that a **negative
`pred_gap` is a PASS** and what would instead indicate a build error. Additionally
`logvar_full_floor_frac` and `logvar_full_ceil_frac` are read, since they are what
re-derives `logvar_clamp` at this target.

**Four runs, not five.** `sweep_beta_5p0.yaml` restated the shipped weights when this was written, so
"the baseline run plus the four arms" counted one configuration twice. The local arms are
`smoke_hie.yaml` plus the arm's two keys - the sweep files inherit `default.yaml` and with it the
production shard paths, the seven-device list and the production epoch count - and the resolved
config each run writes beside its own checkpoints is the record that keys the rows.

**Met, all six criteria on all four arms.** `logvar_clamp` is **confirmed, not revised**: the
decoder's mean log-variance sits at $-1.11$ to $-1.13$ across a tenfold change in $\beta$, $3.88$
above the floor with $1.4\%$ of coefficients on it and $0.03\%$ on the ceiling - a clamp catching a
thin tail at both ends rather than shaping the distribution.

**Files.** `teb_vae/lag_attn_fs/RESULTS.md`.

**Validation.** Manual run; criteria read from `metrics_history.csv`.

### S5-T05 - Ship the chosen weights

**Description.** Fold the selected `beta_schedule.end` and `beta_prior` into
`default.yaml`.

**Acceptance criteria.**
- The comment names the arm that set each value.
- The parity allow-list of S3-T03 still covers exactly these keys.

**Shipped $\beta = 1.0$, $\beta_p = 0.1$, from `sweep_beta_1p0.yaml`**, which now restates the
shipped pair and whose local run is the baseline's.

**The second criterion is satisfied by deleting, not by keeping**, which is the opposite of what it
reads like. The chosen values equal the comparison model's, so both exemptions became stale and
`test_every_declared_parity_exemption_is_a_real_divergence` - added by S3-T03 itself - refuses them.
The allow-list drops to six entries, five of which are identity and one a number; the three
loss-scale constants that were drafted as divergences and measured back to parity are named in a
`MEASURED_TO_MATCH_PATHS` tuple whose own test asserts the equality, so a future retune has to leave
that tuple deliberately. The parity pin is now **total apart from identity and one margin**, which is
a stronger statement than 4.7 claimed for it.

Two literals elsewhere had to stop being literals rather than move. `test_trainer.py`'s
resolved-config provenance assertion and its loss-hyperparameter forwarding assertion both pinned
$5.0$; each now reads the value from `default.yaml`, because what they are about is the config
*reaching* the record and the task, and a second copy of the number is a second place for it to go
stale. `test_train_smoke.py`'s echoed-anchor assertion is the same fix with a different guard: it
compares against the run's own resolved config and separately asserts the configured anchor is
non-zero, since the task's default is $0$ and the test would otherwise pass vacuously if the key were
ever dropped from `tiny.yaml`.

**Files.** `teb_vae/lag_attn_fs/configs/default.yaml`,
`teb_vae/lag_attn_fs/configs/sweep_beta_1p0.yaml`,
`teb_vae/lag_attn_fs/configs/sweep_beta_5p0.yaml`,
`teb_vae/lag_attn_fs/tests/test_config_load.py`,
`teb_vae/lag_attn_fs/tests/test_trainer.py`,
`teb_vae/lag_attn_fs/tests/test_train_smoke.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_config_load.py -q`.

---

## Sprint 6 - The package record

**Landed 2026-08-09.** `DESIGN.md` (14 sections) and `tests/test_docs.py` (49 tests) in one commit,
the house pattern. 465 tests in this package, all green. The configuration surface is driven in both
directions - all **44** shipped `VAE_model` keys documented, **16** recorded as deliberately absent
and verified absent - and the parameter arithmetic is pinned against constructed models rather than
against literals.

Five things worth recording, four of them about what a *subclass*'s record has to bind that the
siblings' do not:

1. **The preprint cross-reference has no target yet, and inventing one would be worse than none.**
   The criterion asks for a cross-reference to the preprint subsection stating the backward half of
   two-sidedness. That subsection is Sprint 7's work and does not exist. `DESIGN.md` §8 therefore
   references `doc/latex_template/sections/reach.tex` - which exists, and covers the *input*-side
   half - and says in the same sentence that the backward half is not yet written there, so this
   record is the one until it is. The test asserts the referenced file exists, which a `\cref` to an
   unwritten label could not have satisfied.
2. **The parameter claim is a delta, so it needs four totals and an identity.** The siblings pin one
   absolute number; here the interesting statement is that this model differs from its comparison by
   the decoder's output head *and nothing else*. That is four measurements - both classes at both
   reach budgets - plus $258 \times (C - R)$, all checked against `sum(p.numel() ...)`. If the two
   models ever diverge anywhere but that head, the identity fails rather than letting §1 keep
   asserting a decomposition that stopped holding.
3. **`RESULTS.md` is bound on shape, not on values, and here that is forced rather than chosen.**
   The siblings' tables are generated by an evaluation pass; this package has none, so its tables are
   transcribed. What a test can still guarantee is that no column names a metric nothing emits, that
   every arm has exactly one inventory row, and that every launch line names a config that exists -
   which is what stops a run being recorded against a name that would be NaN in every row.
4. **Two notations for the same integer.** §1 writes totals both as markdown (`3,393,993`) and inside
   maths spans (`3{,}377{,}997`), because a number in a maths span must brace its separators to keep
   its spacing. A pattern matching only one notation left half of §1's arithmetic unchecked and
   passing.
5. **Phrase assertions must run on flattened text.** The document is hard-wrapped at 100 columns, so
   any phrase long enough to be worth pinning is eventually split across a line by an edit elsewhere
   in its paragraph - and a `lean-limit` note is a blockquote, so a wrapped phrase inside one has a
   stray `>` in the middle of it after a naive whitespace collapse. Both are reflows rather than lost
   claims, and a test that fails on them is noise.

### S6-T01 - `DESIGN.md` and its binding test

**Description.** The as-built record and the test that binds it, in one commit - the
house pattern.

**Acceptance criteria.**
- Covers: what the model is, the input contract, the geometry, the forward return dict,
  the loss, the four structural constraints, the smear argument of 1.5 with a
  cross-reference to the preprint subsection, the target-gathered-never-delayed rule, and
  the deviation record. **The preprint cross-reference names `sections/reach.tex`**, which exists
  and covers the input-side half; the subsection for the backward half is Sprint 7's and does not
  exist yet, which the record says rather than pointing at an unwritten label.
- States that the nats are comparable neither to the raw model's **nor across reach
  budgets within this model**.
- The test drives the config-key lists in both directions, including that keys recorded
  as deliberately absent really are.
- The test asserts no module or test in the package mentions this roadmap, enforcing the
  implementation note.
- `RESULTS.md` is bound too: every backticked column resolves, and every launch line
  names a config that exists.

**Files.** `teb_vae/lag_attn_fs/DESIGN.md`,
`teb_vae/lag_attn_fs/tests/test_docs.py`.

**Validation.** `pytest teb_vae/lag_attn_fs/tests/test_docs.py -q`.

---

## Sprint 7 - The preprint

Paths below are relative to `teb_vae/lag_attn_rws/doc/latex_template/`. Every task runs
the build itself; S7-T07 is the final clean build and report.

### S7-T01 - Notation and macros

**Description.** The symbols the new material needs, in `math_commands.tex`'s Project
block and `appendix/notation.tex`'s tables.

**Acceptance criteria.**
- The collisions are resolved explicitly: `R` is already both the decimation factor and
  raw-samples-per-token, `H` is the horizon, `\nheads` is $H_e$, `C` is not currently a
  symbol, and `\gA`/`\gB` are bound to the anchor set and the reach budget. The kept-channel
  count needs a name that collides with none of them; `parameters.tex:111` already uses
  `c^{kept}` informally.
- Every new macro is used at least once.
- Rows added to `tab:sym-signals` and `tab:sym-objective`, not only macros defined.
- No `Command ... already defined`.

**Files.** `math_commands.tex`, `appendix/notation.tex`.

**Validation.** `latexmk`; grep that each new macro appears in a section file.

### S7-T02 - `ss:filtersupport` in `reach.tex`

**Description.** A subsection between `ss:reachtrainability` and `ss:reachlimits`
covering the backward half of two-sidedness.

**Acceptance criteria.**
- Contains: the blend fraction with `\highlight` at its definition, the measured table of
  1.6, the not-a-leak argument, the zero-conditional-MI display naming the shared-decoder
  cancellation, and the horizon-resolved reporting rule.
- Does **not** restate the eight things `reach.tex` already covers: the two-sidedness
  itself, the $L_{95}$ definition and its numbers, the quantile-not-support caveat, the
  delay construction, `tab:budgets`, the trainability finding, the one-sided-support
  requirement, and the mixture-of-coordinates consequence.
- Also amends `rem:lagreference`, which attributes the neighbourhood width entirely to the
  encoder's receptive field and never to the filter's own support - a second, independent
  source of the same blur.
- `\cref{ss:filtersupport}` resolves.

**Files.** `sections/reach.tex`, `sections/readouts.tex`, `sections/limits.tex`.

**Validation.** `latexmk`; grep for the eight restated phrases.

### S7-T03 - `fig_filter_support`

**Description.** A TikZ figure showing the two-sided coefficient window straddling the
anchor at several horizon steps, with the blend falling to zero.

**Acceptance criteria.**
- Stem is `fig_filter_support`; `fig_support` is taken by the anchor-support cascade.
- `\input{figstyle}`, the shared palette, no per-figure style.
- Ships `.tex`, `.pdf` and `.png`, with the `.pdf` no older than the `.tex`.
- Caption self-contained; the float is interpreted in the text.

**Files.** `assets/figures/fig_filter_support.{tex,pdf,png}`.

**Validation.** Build the figure standalone, then `latexmk`.

### S7-T04 - `s:featuredomain`, and the front matter

**Description.** The self-contained variant section, plus the one abstract sentence, the
`tab:scope` row and the body-order comment.

**Acceptance criteria.**
- States what does not change by naming the shared modules, and what does: the target,
  the decoder width, the block cardinality.
- States the $H \cdot C$ against $H \cdot R$ unit consequence explicitly, since `ss:units`
  currently claims a cross-architecture comparability this variant breaks.
- States that `\archA`/`\archB` remain **encoder** names and this is a target-domain axis,
  so the section does not contradict `notation.tex:17-20`.
- Notes the `logvar_clamp` provenance from `architecture.tex:368-371`.
- The abstract's "predicts raw FHR at 4 Hz" becomes conditional; the title is unchanged.
- `s:featuredomain` appears in the ToC; the `\cref` to `ss:units` resolves.

**Files.** `sections/featuredomain.tex`, `main.tex`, `sections/introduction.tex`.

**Validation.** `latexmk`; grep the abstract for the stale claim.

### S7-T05 - `appendix/configuration.tex`

**Description.** The config tables gain the variant's keys.

**Acceptance criteria.**
- `tab:config-geometry` records that `raw_per_step` remains a geometry input and no longer
  the decoder width.
- `a:absentkeys` gains nothing about `raw_per_step`, since it is not absent.
- The per-architecture `\midrule` blocks take a variant block additively.

**Files.** `appendix/configuration.tex`.

**Validation.** `latexmk`.

### S7-T06 - The citation, and the `filemap` decision

**Description.** `references.bib:227` holds `schreiber2000measuring`, uncited, while
"transfer entropy" is discussed with no citation at `main.tex:57`,
`introduction.tex:128`, `readouts.tex:323` and `limits.tex:23,87,418`. Cite it. Separately,
decide `appendix/filemap.tex`: it is `\input` nowhere, carries two dangling `\cref`s, and
is nonetheless the only place listing module names.

**Acceptance criteria.**
- Schreiber cited at the definition of transfer entropy.
- `filemap` either revived - dangling refs fixed, a row added for the new package, wired
  into `main.tex` - or deleted, with the choice recorded.
- No invented citations.

**Files.** `sections/limits.tex` or `sections/readouts.tex`, `appendix/filemap.tex`,
`main.tex`.

**Validation.** `latexmk`; the bibliography lists Schreiber.

### S7-T07 - The clean build

**Description.** A full rebuild and report.

**Acceptance criteria.**
- `latexmk -C && latexmk` from the template directory, per that directory's `CLAUDE.md` -
  a partial rebuild after adding a citation reports a stale "Label(s) may have changed".
- No undefined reference or citation, no overfull box wider than about 15 pt.
- Page count reported **as a delta against the 75-page baseline**, and any non-`latexfont`
  warning reported.
- `-shell-escape` not enabled.

**Files.** none.

**Validation.** `latexmk -C && latexmk` in `teb_vae/lag_attn_rws/doc/latex_template/`.

---

## Sprint 8 - `lag_attn_transformer_fs`, gated

**Not scheduled.** Section 1.2 states this roadmap is an experiment expected to reproduce
a negative held-out gap, and section 1.1's missing cell - feature target, no bypass - is
delivered by Sprints 2 to 5. A conv-Transformer feature model isolates no additional
confound; it is the fourth cell of a grid whose third cell has not reported.

**Gate.** Start only if S5-T04 shows a `pred_gap` whose sign or magnitude makes the
encoder axis worth re-opening in this domain.

**Gate reading, 2026-08-08: not opened, and the local runs cannot open it.** S5-T04's shipped arm
reports `pred_gap` $+1.20$ with the horizon split positive at the far step - which would be exactly
the kind of sign worth re-opening the encoder axis for - but every one of those runs is **in-sample**,
both splits being the same $339$ windows, and the sibling package saw this same sign flip between its
in-sample arms and its held-out production run. The gate asks a question only a held-out run answers.
It therefore moves to the production runs `RESULTS.md` still owes, and stays shut until then.

**Discharged by direction, 2026-08-09, with the gate reading above left standing.** The work is
scheduled, and not because the gate opened. The reading above is preserved rather than overwritten
because it was correct when written and is still correct about the only thing it claims: a local
in-sample run cannot decide whether the encoder axis is worth re-opening, and this sprint's own
runs are local and in-sample too. What the work buys instead is that the $2\times2$ grid becomes
complete, so the encoder axis is measurable in the feature domain at all and the two axes stop
being confounded with each other. It is **not** a remedy for the source pathway's generalisation
failure and must not be read as one. The reasoning is set out in
`teb_vae/lag_attn_transformer_fs/SPEC_AND_SPRINTS.md` section 1.2, which is where this sprint now
lives; nothing below is scheduled from this document.

**The two questions are settled, against the tree rather than by estimate**, and both answers
turned out to be load-bearing. They are recorded in that roadmap's section 1.3; in summary:

- **Neither parent can be subclassed for both halves, because the MRO forbids it.**
  `SeqVaeLagAttnFs` derives from `SeqVaeLagAttnRws` while `SeqVaeLagAttnTrfRws` derives from
  `nn.Module` directly, so `class SeqVaeLagAttnTrfFs(SeqVaeLagAttnFs, SeqVaeLagAttnTrfRws)`
  linearises through the **conv-LSTM** constructor and builds the wrong model with nothing raising.
  The answer is a mixin, and S0-T02 of that roadmap moves this package's five target-domain members
  into `lag_attn_fs/nets/feature_target.py` to supply it - which is why this package's model class
  is now a composition with an empty body rather than the subclass S2-T01 built.
- **Both `default.yaml` pins are references and the square closes on three pins, not one.** The
  transitivity is real but weaker than it looks: it gives equality for any leaf outside every
  allow-list, and what it does not give is that no *further* leaf appears on the target edge - so
  the fs-to-rws pin this package already ships stays load-bearing and is named there as one of two
  sibling tests the argument leans on.

**Two estimates in the paragraph this replaces were wrong.** The work is $23$ tasks over five
sprints, not "roughly ten mirroring S2, S3, S5 and S6" - the evaluation pipeline is deferred whole,
as it is here, but three configuration pins, two inheritance diamonds and a re-derivation of the
loss-scale constants at the new encoder have no analogue in those sprints. And the framework-prefix
tuple goes to **six** packages, not four: it already covered five, the sixth is the new package,
and the extension that mattered was not a package at all but `sample_page` joining the banned
module list, which had been missing while both `sample_page.py` modules import matplotlib.
`test_lr_warmup.py` and `test_ddp_reachability.py` are re-run over the new nets as expected.

---

## Full task list

- S0-T01: The equivalence harness, captured on the unmodified tree
- S0-T02: `compute_loss` takes the target and the block width
- S0-T03: `decoder_out_channels` on the shipped model
- S0-T04: `TARGET_FIELDS` parameterises the normalisation guard
- S0-T05: `TRACKED_METRICS` and the plot-callback seam
- S0-T06: A forecast-row seam in `sample_page`
- S1-T01: Package scaffold
- S1-T02: The future block: reuse the unfold, add the gather
- S1-T03: Masks, and the budget-to-width binding on real data
- S2-T01: `SeqVaeLagAttnFs`
- S2-T02: The forward contract
- S2-T03: The four structural invariants
- S2-T04: `compute_loss` wiring, and the `block_width` trap
- S2-T05: The checkpoint contract
- S2-T06: The smear argument, made falsifiable
- S2-T07: `nets/` import purity
- S3-T01: The task subclass
- S3-T02: The trainer subclass
- S3-T03: `default.yaml` and its parity pin
- S3-T04: `tiny.yaml` as a delta
- S3-T05: The observability metrics
- S3-T06: The loss-scale constants, re-derived
- S3-T07: DDP and the tiny smoke
- S4-T01: The channel-selection rule
- S4-T02: The feature forecast rows
- S4-T03: Lag consistency and the lag map
- S5-T01: The `RESULTS.md` skeleton
- S5-T02: The local config
- S5-T03: The arms
- S5-T04: The validation run
- S5-T05: Ship the chosen weights
- S6-T01: `DESIGN.md` and its binding test
- S7-T01: Notation and macros
- S7-T02: `ss:filtersupport` in `reach.tex`
- S7-T03: `fig_filter_support`
- S7-T04: `s:featuredomain`, and the front matter
- S7-T05: `appendix/configuration.tex`
- S7-T06: The citation, and the `filemap` decision
- S7-T07: The clean build

---

## Todo checklist

### Sprint 0: Seams in the shipped packages
- [x] S0-T01: The equivalence harness, captured on the unmodified tree
- [x] S0-T02: `compute_loss` takes the target and the block width
- [x] S0-T03: `decoder_out_channels` on the shipped model
- [x] S0-T04: `TARGET_FIELDS` parameterises the normalisation guard
- [x] S0-T05: `TRACKED_METRICS` and the plot-callback seam
- [x] S0-T06: A forecast-row seam in `sample_page`

### Sprint 1: The feature target
- [x] S1-T01: Package scaffold
- [x] S1-T02: The future block: reuse the unfold, add the gather
- [x] S1-T03: Masks, and the budget-to-width binding on real data

### Sprint 2: The model
- [x] S2-T01: `SeqVaeLagAttnFs`
- [x] S2-T02: The forward contract
- [x] S2-T03: The four structural invariants
- [x] S2-T04: `compute_loss` wiring, and the `block_width` trap
- [x] S2-T05: The checkpoint contract
- [x] S2-T06: The smear argument, made falsifiable
- [x] S2-T07: `nets/` import purity

### Sprint 3: Task, trainer, configuration
- [x] S3-T01: The task subclass
- [x] S3-T02: The trainer subclass
- [x] S3-T03: `default.yaml` and its parity pin
- [x] S3-T04: `tiny.yaml` as a delta
- [x] S3-T05: The observability metrics
- [x] S3-T06: The loss-scale constants, re-derived
- [x] S3-T07: DDP and the tiny smoke

### Sprint 4: The diagnostic page
- [x] S4-T01: The channel-selection rule
- [x] S4-T02: The feature forecast rows
- [x] S4-T03: Lag consistency and the lag map

### Sprint 5: Local validation on real data
- [x] S5-T01: The `RESULTS.md` skeleton
- [x] S5-T02: The local config
- [x] S5-T03: The arms
- [x] S5-T04: The validation run
- [x] S5-T05: Ship the chosen weights

### Sprint 6: The package record
- [x] S6-T01: `DESIGN.md` and its binding test

### Sprint 7: The preprint
- [ ] S7-T01: Notation and macros
- [ ] S7-T02: `ss:filtersupport` in `reach.tex`
- [ ] S7-T03: `fig_filter_support`
- [ ] S7-T04: `s:featuredomain`, and the front matter
- [ ] S7-T05: `appendix/configuration.tex`
- [ ] S7-T06: The citation, and the `filemap` decision
- [ ] S7-T07: The clean build

### Sprint 8: `lag_attn_transformer_fs`
- [x] Discharged elsewhere: scheduled by direction with the gate reading left standing, and
      both subclassing questions settled against the tree. Tracked in
      `teb_vae/lag_attn_transformer_fs/SPEC_AND_SPRINTS.md`; check its boxes, not this one.

---

This document is the living roadmap and guidebook for the project. As each task is
completed, check its box above. Keep this file as the single source of truth - update
status here rather than tracking progress elsewhere.

**In the implementation phase, the code must not mention these sprints, tasks, phases, or
this document.** Tests belong in the package's own flat `tests/` directory, never beside
the module under test.

