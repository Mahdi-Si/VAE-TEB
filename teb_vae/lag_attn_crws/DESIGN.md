# `lag_attn_crws` — design record

This document describes the causal-input raw-target lag-attention VAE-TEB: what it consumes, what it
returns, what it optimises, what it reads out, and which of its members are objects of another
package reached by reference.

Companion documents, not restated here:

- `teb_vae/lag_attn_rws/DESIGN.md` records the raw target, the objective and the architecture that
  every raw-target cell shares. `teb_vae/lag_attn_rws/model_explained.md` records the latent
  factorisation.
- `teb_vae/lag_attn_cfs/DESIGN.md` records the causal input machinery that this package composes:
  the warm-up budget, the anchor tiling, the derived phase, the channel alignment and the
  availability-clock control.
- `teb_vae/lag_attn_transformer_rws/DESIGN.md` records the conv-Transformer encoder of the twin cell
  `lag_attn_transformer_crws`.
- `hdf5_dataset/dataset_explained_research.md` defines the causal dataset variant, and
  `documents/docs-md-files/datasets/CAUSAL_SCATTERING_PHASE_HARMONIC_MATH.md` carries the warm-up
  and group-delay mathematics.

Almost every member of this package is an object of another package. A change in a sibling package
can therefore change this model with no file in this directory changing. §6 lists those members.

---

## 1. What the model is

The model is one cell of an encoder-by-target grid:

```
                          conv-LSTM encoders          conv-Transformer encoders
  raw FHR target          lag_attn_rws                lag_attn_transformer_rws
  two-sided feature       lag_attn_fs                 lag_attn_transformer_fs
  causal feature          lag_attn_cfs                lag_attn_transformer_cfs
  causal in / raw out     lag_attn_crws  <- this      lag_attn_transformer_crws
```

At each decoded 4-second anchor $t$ the model forecasts the next $H \cdot R = 30 \times 16 = 480$
raw FHR samples (two minutes). It makes this forecast twice, through one shared decoder: once from a
target-only latent and once from a source-conditioned latent. The gap between the two forecasts, and
the KL between the two latents resolved across lags, are the coupling readout.

The inputs are the one-sided scattering and phase-harmonic coefficients of the **causal** dataset
variant. An input at decimated step $t$ is therefore a function of $\{x(s) : s \le t\}$ only. The
target is the raw signal itself, so it has no warm-up, no group delay and no channel selection.
Neither side of the objective contains its own future.

Structurally this model is `lag_attn_rws` with one-sided inputs. The two cells share the target, the
objective and the horizon family. They differ in the input representation.

The whole model is

```python
class SeqVaeLagAttnCrws(CausalRawInputs, SeqVaeLagAttnRws): ...
```

`vars(SeqVaeLagAttnCrws)` holds `__init__` and no other callable. The method resolution order is
`SeqVaeLagAttnCrws -> CausalRawInputs -> CausalWarmupInputs -> SeqVaeLagAttnRws -> Module -> object`.
§6 explains why the constructor is the one member, and why there is one mixin where the
causal-feature cells have two.

At the shipped configuration the model holds **4,589,907 parameters**. §13 gives the breakdown.

Four mechanisms of this cell are configuration keys, and the off-state of each key reproduces the
pre-revision model bitwise. The keys control where the lag attention's keys and values come from,
whether the prior receives the source arrival clock, whether the reconstruction weights the horizon
axis, and how the lag bias is seeded. §16 is the inventory. Two mechanisms of the feature-target
cells are not part of this row: the decoder's persistence residual and the second alignment
reference (§14).

## 2. Input contract

`train/data_module.py::GraphDataModule` reads the **causal** HDF5 shards at `trim_minutes: 1.0`. The
three input tensors are the same tensors that the causal-feature cells read. The target is the same
raw target that the raw-signal cells read.

| Field | Shape | Role |
| --- | --- | --- |
| `fhr` | $(B, 4800)$ | **the reconstruction target**, z-scored; $T \cdot R = 300 \times 16$ raw samples after the loader's symmetric trim |
| `fhr_st`, `fhr_ph` | $(B, 300, 36)$, $(B, 300, 66)$ | the target-feature **input** stream, concatenated to $(B, 300, 102)$; an input here, not a target |
| `up_st`, `up_ph` | $(B, 300, 36)$, $(B, 300, 15)$ | the source stream, concatenated to $(B, 300, 51)$ |
| `weight` | $(B, 300)$ | per-step validity on the decimated grid; every mask of the objective reads it |
| `up` | $(B, 4800)$ | **not read by the model**; the raw context row of the diagnostic page only |
| `guid`, `epoch` | — | figure titles, run provenance, **and the tile phase** (§4) |

$c_y = 36 + 66 = 102$ and $c_u = 36 + 15 = 51$. The two-sided widths are $109$ and $58$: the
dataset builder drops seven scattering channels per block because their one-sided warm-up is longer
than the stored segment. The causal transform does not change the raw arrays, so `fhr` trims to the
$4800$ samples that the raw geometry expects.

The driver's pre-flight (`LagAttnCrwsTrainer.preflight`) enforces these loader rules:

- **`fhr` must be in `load_fields` and in `normalize_fields`.** The shared guard
  `_check_raw_target_normalized` from `lag_attn_rws/trainer.py` checks this against
  `LagAttnCrwsTrainer.TARGET_FIELDS = ("fhr",)`, which is inherited from the raw-signal driver. An
  unnormalised raw target makes the Gaussian NLL meaningless against a unit-scale variance model.
- **`weight` must be in `load_fields`.** No shared guard covers this field, because the target is not
  built from it. Gaps in the raw trace are stored as $0$ bpm, which is approximately $-11\sigma$
  after z-scoring and is not a detectable sentinel. Without `weight` the objective scores padding as
  signal.
- **`guid` and `epoch` must be in `load_fields`.** The tile phase is keyed on this pair, and
  `load_fields` is used literally with no forced additions. Without them every segment is decoded on
  one tile grid, and no shape or metric shows it.
- **`fhr_up_ph` must be absent from every list.** A coefficient that mixes both signals puts the
  source signal into the target-only branch's inputs and breaks the target-only / source-conditioned
  separation. The causal variant does not store this block.

**The stored warm-up region holds real values, not zeros and not NaN.** The normalisation
constants exclude exactly that region. The availability adapter masks it (§7), and
`lag_attn_cfs/DESIGN.md` §2 describes the convention.

## 3. Geometry and the channel alignment

`TrimmedRawGeometry` from `lag_attn_rws/nets/geometry.py` is reused unchanged:

$$T = 300, \qquad H = 30, \qquad R = 16, \qquad T_{\mathrm{valid}} = T - H = 270, \qquad F = 134,
\qquad S = H = 30.$$

Anchors live in $[F, T_{\mathrm{valid}})$, which holds $136$ anchors. The tiling admits
$A_{\max} = \lceil (T_{\mathrm{valid}} - F)/S \rceil = 5$ anchors per sample at phase
$\varphi \le 15$ and $4$ otherwise.

**The target does not force these values.** In the causal-feature cells the floor must satisfy
$F \ge B - 1$ against the resolved warm-up budget $B$, because the target there is a stored
coefficient that is valid only after its own warm-up $W'_c$. A raw sample is valid at every step, so
the objective here is sound at any floor. The values are held at the values of `lag_attn_cfs`, so
that the two cells differ only in what the decoder emits.

### The alignment reference of this row

The alignment machinery is the one of `lag_attn_cfs`, reached by import. Every surviving channel of
**both** input streams is gathered with a shift

$$d_c = \operatorname{round}\!\Bigl(\kappa\,\frac{\tau_{\mathrm{ref}} - \tau_c}{\Delta}\Bigr) \ge 0,
\qquad \kappa \;=\; 1 - \frac{1}{2\gamma} \;=\; 0.875, \qquad \Delta = 4\ \mathrm{s},$$

so that a channel vector at one encoder step describes one instant. Here $\tau_c$ is the composed
group delay of channel $c$ (`causal_delay_s` in the shards) and $\gamma = 4$ is the gammatone order.
$\kappa$ is the ratio between the energy centroid of the one-sided impulse response and the
envelope-mean group delay $\tau_g$; `lag_attn_cfs/DESIGN.md` §3 derives it.

This row ships `causal_align_reference: 42.21`. The four feature-target cells ship `target_max`
($402.1604$ s).

**The reason is that a raw target has $\tau^y \equiv 0$, so the reference does not cancel.** For a
lag-attention peak at lag $\ell$ and horizon element $h$, the lead time between the source content
and the forecast target content is

$$\tau_{\mathrm{phys}}(\ell, h) \;=\; \Delta\,(\ell + 1 + h) \;+\; \kappa\,\tau^u_{\mathrm{ref}}.$$

`teb_vae/lag_attn/nets/lag_report.py::physical_lag_seconds` implements the general form
$\Delta(\ell + 1 + h) + \kappa(\tau^u_{\mathrm{ref}} - \tau^y_{\mathrm{ref}})$, with $\kappa$ passed
as `realised_delay_factor` and $\tau^y_{\mathrm{ref}} = 0$ by default. The lead is smallest at
$\ell = h = 0$ and grows with the lag. At $\tau^u_{\mathrm{ref}} = 402.1604$ s the smallest
expressible lead is $355.9$ s, so the $20$ to $120$ s contraction-to-deceleration band is outside
the lag axis at every lag index. In the feature-target cells the same reference is on both sides of
the identity and cancels.

The configured value is a **name for a channel**. The resolver snaps it to the shard's own float32
delay, $42.2066$ s, and refuses a value more than $\Delta/2$ from every kept channel. The effective
common delay of the aligned stream is $\kappa\,\tau^u_{\mathrm{ref}} = 36.93$ s. The reachable lead
is then $[40.9, 516.9]$ s for $\ell \in [0, 90]$ and $h \in [0, 29]$: $\ell + h = 0$ gives $40.9$ s
and $\ell + h = 20$ gives $120.9$ s.

### Two rules that drop channels

The warm-up budget and the alignment are different rules.

**The budget** (`causal_warmup_budget_steps: 134`) drops a channel whose one-sided warm-up $W'_c$ is
longer than the budget. Applied alone, it keeps $98$ of $102$ target-stream channels (it drops four
`fhr_st` channels) and all $51$ source channels.

**The alignment** then drops every channel whose composed delay is larger than the reference,
because such a channel would need a negative shift, which reads the channel's own future. This is a
correctness rule. At $42.2066$ s the resolved budget on the committed causal fixture is:

| Stream | Kept | Warm-up $W'_c$ | Shift $d_c$ | Valid from $W'_c + d_c$ |
| --- | --- | --- | --- | --- |
| target-feature input | `fhr_st` $17/36$, `fhr_ph` $21/66$; $38/102$ | $0$–$1$ steps | $0$–$6$ steps | $0$–$6$ steps |
| source | `up_st` $17/36$, `up_ph` $0/15$; $17/51$ | $0$–$1$ steps | $0$–$6$ steps | $0$–$6$ steps |

**The whole `up_ph` block is dropped.** Every `up_ph` channel has a composed delay above any
reference that puts the coupling band inside the lag axis. The source stream therefore carries only
`up_st`, the contraction envelope. Without `up_st` the source stream would have no channels, and the
driver refuses `use_up_st: false` together with a warm-up budget (§7).

### The anchor floor is a policy

`CausalRawInputs._check_anchor_floor` keeps the floor inequality as a declared **input-warmth
policy** over the kept **target-stream** input channels:

$$F \;\ge\; \max\Bigl(B - 1,\; \max_c\bigl(W'_c + d_c\bigr)\Bigr),
\qquad B = \max_{c \in \mathrm{kept}} W'_c .$$

- The first half says that every kept channel is warm by the first forecast step $t + 1$.
- The second half applies only when the stream is shifted. It says that every kept channel is warm
  at the anchor itself, because an aligned channel vector describes one instant only after every
  entry has arrived.

The constructor calls this check from `_validate_causal_geometry`, with the target gate's shift
vector. The driver's pre-flight (`_check_floor_pairs_with_budget`) calls the same function with the
resolved budget, so the two cannot disagree. The refusal message names the half that binds and both
numbers.

At the shipped reference $B = 1$ and $\max_c(W'_c + d_c) = 6$, so the policy requires $F \ge 6$. The
shipped `warmup_period: 134` is held so that this cell and `lag_attn_cfs` share one geometry. A lower
floor increases the anchor count: a floor equal to the model's $30$-step warm-up gives $240$ anchors
against $136$ (§14).

**The horizon also sets the anchor count.** $T_{\mathrm{valid}} = T - H$, so $H = 30$ admits $136$
dense anchors and $H = 15$ admits $151$. `sweep_horizon_15.yaml` is that arm.

## 4. Anchor tiling, and the anchored raw target

The tiled anchor set, the derived phase and their refusals are inherited from the causal-feature
cell; `lag_attn_cfs/DESIGN.md` §4 is the record. In summary, the model decodes

$$\mathcal{A}(\varphi) = \{\, F + \varphi + kS \;:\; k \ge 0,\; F + \varphi + kS < T_{\mathrm{valid}} \,\},
\qquad S = H = 30.$$

- Windows partition the timeline, so no raw sample is scored twice in one step.
- The phase is
  $\mathrm{blake2b}(\texttt{guid} \Vert \lfloor \texttt{domain\_start} \rfloor \Vert
  \texttt{train\_epoch} \Vert \texttt{seed}) \bmod S$. It is derived per segment and never drawn, so
  it consumes no RNG, needs no collective and survives a resume.
- Validation and test decode every valid anchor at stride $1$ and phase $0$.
- The stride and the phase are **forward arguments**, resolved from the stage string and not from
  `self.training`.
- Short rows repeat their last valid anchor and mark the copy invalid. `anchor_index` is
  $(B, A_{\max})$ with `anchor_valid` beside it, so every rank has the same shape.

A padded slot must contribute nothing. If it held a distinct legal anchor, the raw gather below
would pull one raw window twice. `tests/test_anchors.py` asserts that a padded slot contributes zero
to the loss.

### The anchored raw target

`nets/causal_raw_inputs.py::gather_anchored_future_target` is the one new piece of arithmetic in the
package. `lag_attn_rws/nets/raw_targets.py::build_future_index` returns the
$(T_{\mathrm{valid}}, H, R)$ grid of raw sample indices, and the architecture caches it as the
non-persistent `future_index` buffer. The gather indexes that grid at the anchor set:

$$\mathrm{idx}[b, a, \tau, r] = \texttt{future\_index}\big[\,\mathcal{A}[b, a],\, \tau,\, r\,\big],
\qquad X^{+} = \texttt{fhr}.\texttt{gather}\big(1,\ \mathrm{idx}.\texttt{reshape}(B, -1)\big).\texttt{reshape}(B, A, H, R).$$

**The function uses `gather`, not `index_select`.** The dense builder `build_future_target` can use
`index_select` because its index is the shared grid, identical for every sample. Here the anchor set
is per sample, because the tile phase is derived per segment. The index is therefore
$(B, A, H, R)$, and an `index_select` on dimension $1$ would return
$(B, B \cdot A \cdot H \cdot R)$.

Three tests pin the function: at `anchors = arange(t_valid)` it equals `build_future_target` under
`torch.equal`; two different per-sample anchor rows give two different windows; and it reads the
model's own cached buffer (asserted by `data_ptr` identity).

The gather checks that every anchor is in $[0, T_{\mathrm{valid}})$. This check is necessary because
advanced indexing on a negative index wraps instead of raising. Uniqueness among valid entries is
not checked there, because a padded slot is a duplicate by design. `forecast_mask` refuses
duplicates on the same call.

## 5. Loss

$$\mathcal{L} = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0
+ \beta(e)\,\mathrm{KL}_{\mathrm{train}} + \beta_p\,R_p
+ \lambda_{\mathrm{ms}} \mathcal{L}_{\mathrm{ms}}
+ \lambda_{\mathrm{deriv}} \mathcal{L}_{\mathrm{deriv}}
+ \lambda_{\mathrm{boundary}} \mathcal{L}_{\mathrm{boundary}}$$

The loss is in nats per anchor and is computed by `lag_attn_rws/nets/losses.py::compute_loss` — the
same code, not a copy. `CausalRawInputs.compute_loss` is the architecture's adapter with one change:
it gathers the raw window at `anchor_index` (§4) instead of at every anchor. Every term, every
reduction, `block_width = geometry.r`, the coverage floor and the log-variance clamp come from the
shared objective. `compute_loss` also accepts `lambda_event_gate` and `full_anchor_weight` and
forwards them to the shared objective.

The shared objective reads `anchor_index` and `anchor_valid` from the forward dict and passes them
to `forecast_mask` and `kl_mask`. The reconstruction support and the KL support are therefore the
decoded set by construction.

**A forward dict with no anchor set falls back to the dense builder.** The dense target then
carries $T_{\mathrm{valid}} = 270$ anchors against a forecast with $A_{\max}$, and the objective
raises a shape error. `tests/test_objective.py` asserts two things: at `anchor_stride: 1` the model
equals the shared objective given the explicit dense anchor index $[F, T_{\mathrm{valid}})$, key by
key under `torch.equal`; and stripping the anchor keys raises.

**`block_width` is `geometry.r` $= 16$** at every anchor set, because a horizon token emits $R$ raw
samples. It feeds only the four log-variance diagnostics.

**The shape weights.** `lambda_ms` and `lambda_deriv` ship at $0.1$, the raw-signal sibling's value.
They act on the envelope and slope of a raw waveform, and the coarsest pooling rate of $16$ divides
the block. `lambda_boundary` must be $0.0$. The boundary term is a slicing identity over
**adjacent** anchors, and this cell always decodes anchors $S$ apart, so a non-zero weight would
join two windows a whole horizon apart. The shared objective raises on that combination, and the
driver's pre-flight (`_check_boundary_term_is_off`) refuses it before a run directory exists.

### Horizon weighting

`horizon_weight_halflife_steps` resolves a geometric decay over the horizon index $\tau$:

$$w_\tau \;\propto\; 2^{-\tau/\lambda}, \qquad \sum_\tau w_\tau = H, \qquad \lambda = 15.0
\text{ shipped}.$$

The vector is a non-persistent buffer `horizon_weight`. It passes through the objective in the same
way as `channel_weight` in the feature-target cells (`raw_sample_score` ->
`masked_raw_block_per_anchor` -> `masked_raw_likelihood` -> `compute_loss`), on the horizon axis. At
$\lambda = 15.0$ and $H = 30$ the resolved weights go from $1.8063$ at $\tau = 0$ to $0.4729$ at
$\tau = 29$. This row has no channel weight, because the last axis of the block is $R = 16$ raw
samples of one trace. The horizon weight is the only weight in this objective.

The renormalisation to $\sum_\tau w_\tau = H$ keeps the block magnitude unchanged, so
`gradient_clip_val`, `additive_margin` and the balance of $\beta$ against the reconstruction do not
change scale. The off-state `null` registers no buffer, and the block score is bitwise the
unweighted one. A vector of the wrong length is refused by name.

A weighted block score is not a log-density, so at $\beta = 1$ the objective is not the exact ELBO.
Every block-scored metric in `metrics_history.csv` is weighted. This package has no evaluation
pipeline that reports an unweighted reading.

**Checkpoints at two warm-up budgets cannot load into each other**, because the input adapters are
built at the surviving widths. The class stamp does not separate them; the stamped keep-indices do.

## 6. Composition: one mixin, one architecture, one constructor

The causal-feature cells use two mixins, because a stored-coefficient target changes the decoder
width, the gather and the readouts. A raw target changes none of these, so this cell composes one
mixin, and that mixin is a subclass:

```python
class CausalRawInputs(CausalWarmupInputs): ...
```

`CausalRawInputs` inherits the input-side members of `CausalWarmupInputs` unchanged, among them
`_set_causal_inputs`, `_validate_causal_geometry`, `_build_adapter`, `_combined_source_steps`,
`build_lag_mask`, `_build_anchor_index` and `forward`. Not one line of the tiled forward, the warm-up
adapter or the floored lag mask is copied. `vars(CausalRawInputs)` holds exactly eight names:

- `_check_anchor_floor` — the floor policy of §3. `_validate_causal_geometry` calls it; in the
  causal-feature cells `CausalFeatureForecastTarget` supplies it.
- `_resolve_warmup_readout_constants` — resolves the two **source**-block warmth patterns and
  nothing else. The target warm fraction and the warm tertiles of the causal-feature cells partition
  kept target channels, and a raw target has none.
- `compute_loss` — the anchored objective of §5.
- Five members bound by reference (table below).

The gather is a module-level function, so that the dense-equivalence test can call it directly.

**The width hook is absent on purpose.** Neither `CausalRawInputs` nor `SeqVaeLagAttnCrws` defines
`_default_decoder_out_channels`. It resolves to the architecture's, which returns `raw_per_step`, so
the decoder is built at $16$ against a $(B, A, H, 16)$ target. If `CausalFeatureForecastTarget` were
composed in by mistake, the decoder would be built at $C_{\mathrm{keep}}$, and `raw_sample_score`
would fail on shapes that do not broadcast.

**The order of the bases is load-bearing.** The mixin comes first. This makes the tiled `forward`,
the `_build_adapter` that builds availability terms from $W'_c + d_c$, and the anchored
`compute_loss` win method resolution over the architecture's members. In the reversed order the
model would decode the dense range, return no anchor set, and score a
$(B, T_{\mathrm{valid}}, H, R)$ target. A test pins the linearisation as a list of class names.

**The constructor exists because of the driver.** `trainer._build_model_kwargs` builds the kwargs by
sweeping `inspect.signature(MODEL_CLS.__init__)`, and it drops every config key that is not a
parameter name. A `**kwargs` signature would therefore build an all-defaults model. The signature is
the architecture's, written out in full, with these differences:

| Removed from the `SeqVaeLagAttnRws` signature | Added |
| --- | --- |
| `target_delays`, `source_delays` | `target_warmup_steps`, `source_warmup_steps` |
| `persistence_residual` | `target_align_delays`, `source_align_delays` |
| `posterior_event_gate` | `anchor_stride`, `lag_floor` |

The forwarded set is captured from `locals()` minus `FORWARDED_EXCLUSIONS`
(`lag_attn_cfs/nets/causal_inputs.py`). A keyword added to the architecture therefore reaches it.
`target_align_delays` and `source_align_delays` are forwarded to the base as `target_delays` and
`source_delays`, so they reach `ChannelDelay` ($\mathrm{out}[t, c] = x[t - d_c, c]$). The warm-up
vectors do not go under a delay name, because a warm-up is a leading *mask* and `ChannelDelay` is a
*shift*.

**`decoder_out_channels` is still a keyword of this signature.** A config with
`decoder_out_channels: 98` builds the decoder at $98$ against a $(B, A, H, 16)$ target. This fails on
the first batch inside `raw_sample_score`, not at the config. `SeqVaeLagAttnTrfCrws` does not have
this keyword. `lag_attn_rws` has the same exposure.

### The binding record

Every member that this package needs from a sibling and does not change is **bound by reference in a
class body**, or imported. A bound member is read once at class creation, and there is no second
definition to drift. A `staticmethod` must be re-wrapped when bound: `Owner.some_staticmethod`
returns the plain function, and a plain function assigned in a class body becomes an instance method
that receives `self` as its first argument.

| Bound member | Owner |
| --- | --- |
| `SOURCE_BLOCK_SPLIT`, `TARGET_BLOCK_SPLIT` | `CausalFeatureForecastTarget` |
| `_resolve_block_warm_steps` (re-wrapped as `staticmethod`) | `CausalFeatureForecastTarget` |
| `_anchors_per_sample`, `_source_lag_warmth` | `CausalFeatureForecastTarget` |
| `anchor_phase`, `_phase_field` (re-wrapped as `staticmethod`), `resolve_anchor_geometry`, `_build_forward_inputs` | `SeqVaeLagAttnCfsTask` |
| `_mu_gap_rms`, `_added_metrics` | `SeqVaeLagAttnCfsTask` |
| `input_stream_panels` (a `property`; the binding carries the descriptor), `input_budget_figure` | `SeqVaeLagAttnCfsTask` |
| `warmup_model_kwargs` | `lag_attn_cfs/model_kwargs.py` |
| `resolve_warmup_budget`, `WarmupBudget` | `lag_attn_cfs/causal_warmup.py` |
| `_tiling_anchors`, `_draw_anchor_overlay` | `lag_attn_cfs/sample_page.py` |
| `_horizon_receptive_field` | `lag_attn_cfs/trainer.py` |
| `_check_raw_target_normalized`, the shared `main` | `lag_attn_rws/trainer.py` |

`TARGET_BLOCK_SPLIT` is needed because the target feature block is an **input** here, and the
page's stream panels split it.

`SeqVaeLagAttnCrwsTask` derives directly from `SeqVaeLagAttnRwsTask`, so the raw target builder
`_build_raw_target` is the parent's. The task writes out three members of its own: `__init__`
(adds the run `seed` to the hyperparameters, because the tile phase depends on it),
`compute_loss_and_metrics` (records the stage for `_build_forward_inputs`), and the `forecast_rows`
property (§11). The first two call zero-argument `super()`, which closes over the defining class.
Bound onto a class outside that hierarchy, such a member raises `TypeError` on the first step.

## 7. The warm-up mask in the availability adapter

The stored coefficients inside $[0, W'_c)$ are zeroed after normalisation, and an availability
channel carries that fact. Both happen inside `AvailabilityInputAdapter`:

$$m_{t,c} = \mathbb{1}[t \ge W'_c + d_c], \qquad
e_t = W_x (x_t \odot m_t) + W_m (m_t - \mathbf{1}) + \mathbb{1}\Big[\textstyle\sum_c m_{t,c} = 0\Big] e_{\mathrm{start}}.$$

`lag_attn_cfs/DESIGN.md` §7 is the record. The inherited `_build_adapter` announces a channel at
$W'_c + d_c$, its warm-up plus its alignment shift, because a gathered and delayed channel is valid
only after both have passed.

**Both streams build a start embedding $e_{\mathrm{start}}$.** The adapter builds it only when every
channel of a stream waits at least one step. At the shipped reference
$\min_c(W'_c + d_c) = 1$ on both streams, so both vectors exist, and each indicator is true on step
$0$ only.

**The driver refuses `use_up_st: false` together with a warm-up budget**
(`_check_source_block_kept`). Dropping `up_st` changes a construction-time decision: the source
stream's minimum warm-up is then above zero, which creates the start embedding, a parameter that only
the leading steps of a segment reach. Under `find_unused_parameters=False` that is a DDP hazard. At
the shipped reference, the alignment also keeps no `up_ph` channel, so without `up_st` the source
stream would be empty.

**No gradient flows from inside the warm-up.** The masked positions are exactly zero before the
input linear. `tests/test_construct.py` asserts this.

## 8. The source stream

Lag attention searches $L = 91$ lags (`max_lag: 90`). From the anchor floor $F = 134$ it reads
source states back to step $F - (L - 1) = 44$.

**The warm-up budget gates no source channel.** All $51$ survive it, and its resolved source
keep-index is the identity. The alignment then removes the $34$ channels whose composed delay is
larger than the $42.2066$ s reference. It keeps $17$: `up_st` $17/36$ and `up_ph` $0/15$.

**Every lag that the attention can reach is warm.** The kept source channels are valid from step
$\max_c(W'_c + d_c) = 6$. The deepest step that a lag reaches from the floor is $44$, and the warmth
readouts are evaluated at the decoded anchors, all of which are at or above $F$.

`lag_floor` generalises the lag validity mask from $\mathbb{1}[t - \ell \ge 0]$ to
$\mathbb{1}[t - \ell \ge F_u]$. It ships at $0$, where the mask is bitwise the sibling's.

### The availability clock and the source-null control

The source availability pattern $m^u_{t,c}$ is a deterministic function of $t$ and is the same for
every sample. It enters $q(z \mid Y, U)$ but not $p(z \mid Y)$, so the clock alone can move the
posterior away from the prior and raise `source_conditioned_kl_raw` with no source information. The
permutation control cannot detect this, because it permutes `source_state` across the batch and
every row carries the same pattern.

The source-null arm, `kld_source_null`, runs the source gate, the adapter and the source K/V path on
a zeroed stream and reports the KL that remains. Zero is the channel mean over the region that the
model reads, because the normalisation constants exclude the warm-up. The arm costs one source
encode and no decode, draws no noise, and its adapter output depends on $m_t$ alone. The null state
is therefore one $(1, T, d_{\mathrm{model}})$ tensor broadcast over the batch.
`source_conditioned_kl_raw - kld_source_null` is the part of the KL that comes from source
variation.

**Both controls use the module that `lag_kv_source` selects** (§8.1), not a deep source encoder
that may not exist. Each control resolves the path from model attributes, so no call site in the
task layer changes.

### 8.1 Where the keys and values come from

**`lag_kv_source` selects the source representation that the lag attention builds its keys *and*
values from.**

- `encoder` (the constructor default) uses the deep source history state. The LSTM makes this state
  a function of the whole causal prefix, so the lag-$0$ entry already contains whatever a later-lag
  entry carries.
- `conv_stem` (shipped) uses a `CausalConvStem` over the availability adapter's output.
- `adapter` uses the adapter's output directly: one step of content.

Keys and values move together. Local keys with deep values would leave the lag-$0$ value with the
most information.

Under `conv_stem` or `adapter` the deep source encoder is **not built**, because nothing else uses
it. The stem reuses this parent encoder's convolution schedule and block classes, with an output
norm, and `causalize_norms` applies to the stem as it applies to the encoder. At the shipped
`encoder_extra_dilations: [8, 16]` and `encoder_extra_kernel: 15` the schedule is kernels
$(3, 5, 11, 15, 15)$ at dilations $(1, 2, 4, 8, 16)$, so the stem's receptive field is
$1 + \sum_b (k_b - 1) r_b = 387$ steps. This is longer than the $91$-lag window and longer than the
$300$-step sequence. On this parent, `conv_stem` therefore removes the recurrence but keeps a
convolution memory over the whole prefix. The conv-Transformer twin's stem reaches $21$ steps.

**`alibi_slope_scale` sets the seed of the lag bias.** `lag_bias_init: alibi_decay` builds a
learnable $(\text{num heads}, L)$ bias parameter. The scale seeds it: the shipped $0.0$ seeds it
flat, and $1.0$ seeds a monotone decay towards lag $0$. `lag_bias_init: normal` builds no bias
parameter.

### 8.2 The prior receives the arrival clock

**`prior_availability_input` gives the same clock to the prior**, so the clock term cancels in the
divergence. `FullLatentPriorHead.forward(h_y, clock=None)` passes the clock through the head's own
`LayerNorm` and a zero-initialised, bias-free linear projection. This projection is not shared with
the adapter's `mask_proj`, because sharing the map would couple the gradients of the two pathways.
The parent's post-initialisation zeroing block re-zeroes the projection beside
`_zero_init_delta_heads`, because the generic `initialization(self)` pass runs after module
construction. At initialisation the prior is therefore bitwise the flag-off prior, and the KL starts
at exactly zero.

**The clock is the encode of a zeroed source stream.** The per-channel arrival pattern is constant
on every scored anchor, because the anchor floor is past the last step at which it changes, and a
constant is an offset that the prior head's biases already span. What remains past the floor is the
source pathway's memory of the arrivals. The clock is therefore the encode of a **zeroed** source
stream through the configured K/V path, computed at batch $1$ and broadcast. It is the same tensor
that the source-null control feeds the posterior. It is **detached**, so no gradient couples the
prior to the source pathway, and it is computed **out of train mode**, so dropout does not change it
per step.

**The invariant becomes "the prior sees no function of the source *values*."** The source-purity
tests assert this. `lag_attn_cfs/DESIGN.md` §8.2 gives the bound on what a prior-side clock can
reach; it is a property of the shared posterior parameterisation and holds here too (§14).

## 9. Forward return dict

`SeqVaeLagAttnCrws.forward(y_st, y_ph, u_stream, anchor_phase=None, anchor_stride=None,
source_valid=None)` is inherited from `CausalWarmupInputs`. It returns **twenty-two keys**: the
twenty keys of the raw-signal architecture, plus the two keys of the anchor axis.
`tests/test_forward_contract.py` asserts the key set.

- `mu_base`, `logvar_base`, `mu_full`, `logvar_full` — $(B, A_{\max}, H, R)$: $(B, 5, 30, 16)$ at the
  shipped training geometry and $(B, 136, 30, 16)$ at the dense evaluation geometry.
- **`anchor_index`** $(B, A_{\max})$ `long` and **`anchor_valid`** $(B, A_{\max})$ `bool` — the
  decoded anchors and which of them are real. The first gathers and scatters; the second multiplies
  into a float mask.
- `mu_prior`, `logvar_prior`, `raw_logvar_prior`, `mu_post`, `logvar_post`, `z_prior`, `z_post` —
  $(B, 300, d_z)$, with $d_z = 64$ shipped.
- `target_state`, `source_state` — $(B, 300, 128)$. **`source_state` is the lag attention's K/V
  tensor** (§8.1): under `lag_kv_source: encoder` it is the deep source state, and under the shipped
  `conv_stem` it is the stem's output.
- `attended_source_heads` $(B, 300, 4, 32)$ and `attn_weights` $(B, 300, 4, 91)$.
- `kld_per_t`, `kld_per_t_per_head`, `source_kl_lag_map`, `mu_prior_sat_frac`, `delta_mu_sat_frac`.

The dict has no `decoder_state` and no `delta_mu_src`, because the decoder receives only the
latent. It has no `persistence`, because this row has no persistence residual. A three-argument call
and a four-argument call with a zero phase agree bitwise at stride $1$.

## 10. Metrics

This package adds four readouts to the objective's metric dict.

| Metric | Stages | Definition |
| --- | --- | --- |
| `anchors_per_sample` | train, val | decoded anchors per sample; $[4, 5]$ in train and $136$ in val at the shipped geometry. A **guard**: a value off that band means the tiling is not the configured one |
| `source_lag_warmth_frac_st` | train, val | attention mass on lags where the first stored source block (`up_st`) is warm |
| `source_lag_warmth_frac_ph` | train, val | the same for the second block (`up_ph`) |
| `kld_source_null` | val | the KL that remains when the source stream is zeroed (§8) |

`compute_loss` merges the first three into the shared objective's metric dict, through the two
readouts bound from the causal-feature cell (§6). `kld_source_null` comes from the task, through the
bound `_added_metrics`, because it needs the source stream, which is not in the forward dict.

**Both warmth patterns are built from $W'_c + d_c$**, the warm-up plus the alignment shift, through
`_combined_source_steps`, the same vector that the availability mask reads.
`lag_attn_cfs/tests/test_metrics.py::test_the_shipped_source_warmth_pattern_is_not_vacuous` checks
the pattern. At the shipped geometry no reachable lag is cold (§8), so
`source_lag_warmth_frac_st` is $1.0$ by construction. The `up_ph` block has no kept channel, and
`_resolve_block_warm_steps` reports an empty block as warm at every step, so
`source_lag_warmth_frac_ph` is $1.0$ over zero channels. Both columns can vary only on the unaligned
arm.

**Eight readouts of the causal-feature cell are not tracked here**, on both stages:

- `pred_gap_tau_first`, `pred_gap_tau_last` — the gap by horizon step, which the raw-signal family
  does not track.
- `pred_gap_st`, `pred_gap_ph` — the gap by stored target block; a raw target has no blocks.
- `pred_gap_warm_lo`, `pred_gap_warm_mid`, `pred_gap_warm_hi` — tertiles of $W'_c$ over kept target
  channels; a raw target has no target channels.
- `target_warm_frac` — the share of scored target coefficients past their own warm-up; it would be a
  constant $1.0$.

`tests/test_metric_tracking.py` asserts the tracked surface in both directions.

The parent's `pred_gap` is unchanged. It is the same quantity that `lag_attn_rws` reports, over the
same raw target, computed from one-sided inputs.

**`_mu_gap_rms` is re-pointed onto the decoded anchor set.** The inherited version rebuilds
`forecast_mask` and `kl_mask` with no anchors. Under tiling, `mu_post_prior_gap_rms` would then
average over all $136$ anchors while `source_conditioned_kl_raw` averages over the decoded set. The
bound override from `SeqVaeLagAttnCfsTask` takes the same anchor set as the KL.

`LagAttnCrwsTrainer.TRACKED_METRICS` is the raw-signal driver's `_TRACKED_METRICS`, plus the three
per-stage columns on both stages, plus `val/kld_source_null` (89 entries in total). There is no
`train/kld_source_null`, because the readout never enters the objective and that column would be NaN
in every row.

## 11. Diagnostic page

The page is the raw-signal sibling's. `build_diagnostic_figure` owns the GridSpec, the shared
physical-time axis, the caption and the latent-and-attention rows. This cell replaces one seam and
binds two.

| Row | Source |
| --- | --- |
| `raw` | the shared raw-context row; this model's target is the raw trace |
| `forecast` | **this cell's** `sample_page.causal_raw_forecast_rows`: the base and full forecasts tiled into non-overlapping $480$-sample windows, walked from `anchor_index` and `anchor_valid`; window edges; the anchor overlay (floor, decoded anchors, training tile grid); the lag caveat as a footnote |
| `input_target`, `input_source` | the causal-feature cell's `input_stream_panels`, bound |
| `latent`, `kld_dims`, `kld_total`, `lag_attn`, `kl_lag_map` | the shared builder's, unchanged |

**The forecast rows are re-pointed because the forecast is indexed by position in the decoded set.**
The shipped raw rows tile through `concat_single_forecasts`, which walks
`range(warmup, t_valid, horizon)` and reads the forecast at each **anchor**. This model's forecast is
$(A_{\max}, H, R)$. At the shipped geometry, $136$ positions read at anchors $134 \dots 269$ go out
of range on the first read. `tests/test_sample_page.py` compares the drawn curve at the first tile
with the forward's own `mu_full` at the anchor that `anchor_index` names. Raw samples that no drawn
window covers render as gaps. `row_axes` and `finalise_time_axis` are reused, and the anchor walk
and the overlay are the causal-feature page's functions, imported. Window edges are deduplicated
before drawing, because consecutive windows at the evaluation resolution share a boundary.

The page has two forecast rows, not the causal-feature page's eight, because the decoder emits
$R = 16$ raw samples of **one** signal. Nothing is reserved through `forecast_extra_rows`.

**The run-level warm-up budget figure** is the causal-feature cell's, reached through the bound
`input_budget_figure`. It is written once per run under the stem `causal_warmup_budget`, which is
different from the shipped `causal_input_budget` of the two-sided reach guard. The figure is drawn
from the resolved budget, because its subject includes the channels that the budget **dropped**,
whose $W'_c$ the checkpoint does not carry. The driver gives the task the budget after it builds the
model (`self.pl_model.warmup_budget`). A task with no budget refuses the figure by name.

**The lag footnote is this cell's own string.** The two lag panels belong to the shared builder, so
the caveat is a page footnote. Lag axes are stored-coefficient time on the input side: the raw
target has no group delay, so the anchor is exact, but a causal input coefficient lags by its own
composed group delay. Under the channel alignment the footnote gives the identity of §3,
$\Delta(\ell + 1 + h) + \kappa\tau^u_{\mathrm{ref}}$, and interpolates $\Delta$ and $\kappa$ from
`physical_lag_seconds`.

The footnote gives the identity, not a number, because $\tau^u_{\mathrm{ref}}$ is a decision of the
run. The model's `source_delay_steps` is the largest stored-step shift ($6$ at the shipped
reference), not the reference. The resolved reference is in the run's resolved-config dump and in
the pre-flight disclosure as `source_reference_delay_s`, which the footnote names.

`tests/test_lag_consistency.py` asserts that the page's lag axis, the model's reported lag and a
hand-computed physical lag agree; that filling the panel's delay slot with the warm-up does not move
the axis; and that a non-zero `lag_floor` moves the mask and not the axis.

## 12. Configuration, and DDP

`configs/` ships `default.yaml`, `tiny.yaml`, `smoke_causal.yaml`, `sweep_anchor_stride_1.yaml` and
`sweep_horizon_15.yaml`. `default.yaml` is written out in full. The other four files inherit from it
through `base: default.yaml` and override a small set of keys.

**`default.yaml` is pinned leaf-for-leaf against `lag_attn_rws/configs/default.yaml`** outside a
declared allow-list of **twenty-eight** exemptions (`PARITY_EXEMPT_PATHS` in
`tests/test_config_load.py`), in both directions. An exemption that is no longer a divergence fails
as a divergence that is not exempt does. The exemptions are:

- **Identity (5):** the run tag, the output tree, the MLflow experiment, the run name and the
  variant tag.
- **Dataset (4):** the causal shards, their statistics, and `epoch` in `load_fields`.
- **Geometry (7):** `c_y`, `c_u`, `warmup_period`, `causal_reach_budget_s` (present and required
  `null`), `causal_warmup_budget_steps`, `anchor_stride`, `lag_floor`. Both configs ship
  `horizon: 30`, so `horizon` is not exempt.
- **Channel alignment (2):** `causal_align_reference` (`42.21`) and `causal_leg_alignment`.
- **Objective (1):** `lambda_boundary` ($0.05 \to 0.0$).
- **Mechanisms (7):** the four architecture switches (`lag_kv_source`, `prior_availability_input`,
  `horizon_weight_halflife_steps`, `alibi_slope_scale`) and three training controls
  (`early_stopping.enabled`, its `patience`, and `model_checkpoint.secondary_monitor`).
- **Loss-scale constants (2):** `gradient_clip_val` ($12000.0$ against $5000.0$) and
  `spike_breaker.additive_margin` ($2.5 \times 10^{3}$ against $10^{3}$), declared `RETUNED`. `ema_floor` ($10^{9}$) and `horizon_embed_std` ($0.8$)
  equal the sibling's values and are held by the parity check.

`tests/test_config_load.py` also checks that every key reaches a constructor argument or a
task-level consumer. `tests/test_sweep_configs.py` checks `anchor_stride == horizon` in every arm
that moves the horizon.

`causal_reach_budget_s` must be `null`. It prunes channels by the forward reach of a two-sided
Morlet, measured on a bank that did not produce these coefficients, and the resolver refuses the two
guards together. The key is present so that the parity check against the raw-signal sibling stays
leaf-for-leaf.

The other configs:

- `tiny.yaml` points at the committed causal fixture, sets `d_model: 32`, `d_z: 8`, `d_head: 8`,
  `max_lag: 8`, `lstm_layers: 1`, `dropout: 0.0` and `likelihood: mse`, and runs one epoch on one
  device. It keeps the real sequence length, channel counts, budget, floor, horizon and trim.
- `smoke_causal.yaml` keeps the shipped widths and `gaussian_nll` against the same fixture. It sets
  `gradient_clip_val: 1.0e+9`, a batch of $4$ (the whole fixture, so one epoch is one optimizer
  step) and $600$ epochs. It is the configuration for collecting gradient-norm and loss-excursion
  statistics at the shipped widths.
- `sweep_anchor_stride_1.yaml` sets `anchor_stride: 1`, so training decodes the dense range.
- `sweep_horizon_15.yaml` sets `horizon: 15` and `anchor_stride: 15`. The block is then
  $15 \times 16 = 240$ raw samples and the dense anchor count is $151$. `horizon_depth` stays at
  $4$, which meets the criterion $\mathrm{RF} \ge H + 1$ at both horizons.

There is no floor arm (§3). `emit_validity_mask` is in no shipped config.

**Training controls.** Early stopping is on, at patience $50$ on `val/total_loss`. A second
`ModelCheckpoint` monitors `val/nll_full_block` through `model_checkpoint.secondary_monitor`, so a
run writes checkpoints under two stems.

**DDP.** Production runs use plain `"ddp"` with `find_unused_parameters=False` under `gaussian_nll`.
`mse` leaves the decoder's log-variance heads without a gradient and selects
`ddp_find_unused_parameters_true`.

- The availability terms are unconditional in the forward, and every branch is a construction-time
  decision about whether a module exists.
- Each rank derives the per-segment phase from its own samples, with no collective.
- $A_{\max}$ is a geometry constant, so no rank can disagree on shape.
- `broadcast_buffers=False`, because every buffer is a deterministic function of the config. The two
  source-warmth patterns are non-persistent buffers.
- `static_graph` is absent, because the spike breaker's skipped-batch backward is structurally
  different from the recorded one.

`tests/test_train_smoke.py` asserts that two evaluations of one checkpoint on one shard produce an
identical `metrics_history.csv` row set. It runs two CPU subprocesses under different
`PYTHONHASHSEED` values at `OMP_NUM_THREADS=1`, because a reduction split across workers accumulates
in the order they finish. Under `benchmark: true` the drift bound is
$\mathrm{atol} + \mathrm{rtol} \cdot |x|$ per column.

**A run's artifacts state the shard variant and the alignment reference.** `causal_leg_alignment`
names the phase-harmonic operator that built the phase blocks and is compared with the shards' root
attribute. `causal_align_reference` names the clock that the input channels are shifted onto. Both
are leaves of `model_config.VAE_model`, so both are written into the resolved-config artifact beside
the checkpoints. The resolved consequences are in the startup log's budget summary and in the
checkpoint's `model_kwargs`, which carry the two shift vectors. At the shipped config the budget
summary reads:

```
causal warm-up budget 134 steps (quantile 0.95, trim_minutes 1.0, leg alignment envelope,
phase operator ratio_power_v0, reference 42.2066 s): target: fhr_st 17/36, fhr_ph 21/66;
38/102 channels, warm-up 0-1 steps, shift 0-6 steps, honest by 0-6; source: up_st 17/36,
up_ph 0/15; 17/51 channels, warm-up 0-1 steps, shift 0-6 steps, honest by 0-6
```

`create_model` also logs the resolved anchor geometry ($H$, $S$, $F$, $T_{\mathrm{valid}}$,
$A_{\max}$, tiles per sample, the raw block width $H \cdot R$, and the horizon refine stack's
receptive field against $H + 1$). It adds the run `seed` to the task hyperparameters, because the
tile phase depends on it.

## 13. Parameter budget

Counts on constructed models, at the shipped `default.yaml` keywords, with the warm-up budget
resolved against the committed causal fixture:

| Configuration | Parameters |
| --- | ---: |
| **shipped** (local K/V, prior clock, weighted horizon, flat bias seed; budget $134$, aligned to $42.2066$ s; $38/102$ target, $17/51$ source) | $\mathbf{4{,}589{,}907}$ |
| shipped, unaligned (`causal_align_reference: null`; $98/102$ target, $51/51$ source) | $4{,}613{,}715$ |
| shipped, ungated ($102$ and $51$) | $4{,}595{,}155$ |
| shipped, `lag_kv_source: encoder` | $5{,}097{,}786$ |
| every switch of §16 at its off-state, aligned | $5{,}081{,}146$ |

"Ungated" means the whole guard: the warm-up mask and the alignment together. A shift vector is
positional over the surviving channels, so a stream with no keep-index cannot carry one.

**The four switches cost $-491{,}239$**, and each term is a module:

| Term | Parameters | What it is |
| --- | ---: | --- |
| the deep source encoder, not built | $-1{,}312{,}231$ | `lag_kv_source` is not `encoder`, and nothing else uses it |
| the local K/V stem (`source_kv_stem`), built | $+804{,}352$ | this parent encoder's convolution blocks plus an output norm |
| the prior clock | $+16{,}640$ | $2 \times 128$ for its own `LayerNorm` and $128 \times 128$ for the bias-free projection |
| the horizon weight | $0$ | a non-persistent buffer, not a parameter |
| the flat lag-bias seed | $0$ | the same $(\text{num heads}, L)$ parameter, seeded differently |

**The alignment costs $-23{,}808$.** At $42.2066$ s it drops $60$ target-stream and $34$ source
channels relative to the unaligned budget:

$$\underbrace{-\,60 \times 128 \times 2}_{\text{target adapter}}
\;\underbrace{-\,34 \times 128 \times 2}_{\text{source adapter}}
\;+\; \underbrace{2 \times 128}_{\text{start embeddings}} \;=\; -23{,}808.$$

The two linears per adapter are the input linear and the availability projection. Both adapters
build a start embedding because the shifted warm-up starts above zero (§7). The alignment narrows
adapters and the switches replace a source module, so the two effects are independent.

## 14. Limitations and planned work

- **Two delay conventions.** `hdf5_dataset/causal_scattering.py::leg_alignment_shift`, the shift
  baked into the stored coefficients, uses the unscaled $\tau_g$. The model-side alignment
  (`channel_alignment_delays` and `_align_stream`) applies $\kappa$. Applying $\kappa$ on the
  transform side would change the stored coefficients and require a dataset rebuild.
- **The lag reading needs the effective reference.** The aligned stream's common delay is
  $\kappa\,\tau^u_{\mathrm{ref}}$, not the $\tau^u_{\mathrm{ref}}$ that the run logs as
  `source_reference_delay_s`. Call `physical_lag_seconds` with
  `realised_delay_factor` $= \kappa$; its default $1.0$ treats the reference as a realised delay.
- **`dataset_kwargs` is shared between the two loaders** and cannot carry a per-split GUID filter.
  A dev-box run on the committed fixture therefore validates and trains on the same shard.
- **No mixed precision** (`precision: "32-true"`) and **`compile: false`**. The `nn.LSTM` encoders,
  the checkpointed attention region and the data-dependent boolean mask indexing behind
  `kld_active_frac` each break TorchInductor.
- **No warm start.** `core_model_checkpoint` stays `null`. A checkpoint from a sibling carries a
  different `model_class` stamp, and a checkpoint from a raw-signal cell has adapters at other
  widths. `load_checkpoint_strict` and `check_model_class` refuse each case by name.
- **Pre-revision checkpoints do not load into the shipped configuration.** The constructor and
  state-dict changes of §16 change the state dict. The off-state arm rebuilds the pre-revision model.
- **No run checker and no `eval/` package ship.** `tests/test_train_smoke.py` asserts
  `anchors_per_sample` on the fixture fit.
- **The K/V stem is not local on this parent.** Its receptive field is $387$ steps (§8.1). A
  separate stem schedule would be a new design decision, not a reuse.
- **No persistence residual.** `persistence_residual` is not on this cell's signature, so the
  driver's signature sweep drops the key. The architecture parent `SeqVaeLagAttnRws` refuses
  `persistence_residual=True` by name, because a raw block has no per-channel level to carry
  forward.
- **No second alignment reference.** The target is raw with zero delay, so the single
  `causal_align_reference` already acts on the source clock only. No shipped config of this row
  sets `causal_align_reference_source`.

Planned or deferred code changes:

> lean-limit: no persistence residual in this row; a candidate replacement is an anchored raw
> persistence input. The $(H, C)$ weight of the feature-target cells does not apply, because the
> last axis of this block is $R = 16$ raw samples of one trace, and the anchor's own value is a
> $16$-sample vector, not a per-channel scalar.

> lean-limit: the prior clock cancels no part of the KL's **mean** term, because the posterior is a
> bounded residual on the prior and that term is a function of the delta head alone. A replacement
> is a delta defined as $D(a) - D(a^\varnothing)$, with `posterior_logvar_mode` set back to
> `residual` for the variance half. This changes the posterior parameterisation that the coupling
> readout is defined on. `lag_attn_cfs/DESIGN.md` §8.2 gives the decomposition.

> lean-limit: the anchor floor is $134$ by policy, not by validity. A floor of $F = 30$ (the model's
> own warm-up) gives $240$ anchors against $136$. At the shipped reference the constructor admits any
> floor at or above $6$, so this change needs only a config edit and a re-derived stride.

> lean-limit: four config-shaped pre-flight refusals of the driver (`_check_no_cross_channel_block`,
> `_check_boundary_term_is_off`, `_check_phase_key_fields`, `_check_source_block_kept`) and the two
> `super()`-calling task members are written out here rather than shared with the causal-feature
> cell. Move them to a shared home when a third consumer appears.

> lean-limit: no `eval/` package. A replacement is a `ModelBinding` against
> `teb_vae/lag_attn_rws/eval`. `eval/binding.py::ModelBinding` is a frozen dataclass that names no
> model class, and `teb_vae/lag_attn_transformer_rws/eval/` is a working precedent. The known
> obstacle: `eval/metrics.py::model_inputs` assembles the net inputs from the task's stream builders
> and does not call `_build_forward_inputs`, so it passes no phase and no stride. A binding for this
> cell must supply the anchor geometry and pin its lag read site.

## 15. Running it

From the repository root:

```bash
# Production, 7 ranks. TEB_RUN_STAMP is required so ranks 1..N-1 share rank 0's run directory,
# and the rank count must equal len(general_config.cuda_devices). The shard paths in default.yaml
# are REPOINT_ME placeholders until the causal production shards exist.
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_crws.trainer --config teb_vae/lag_attn_crws/configs/default.yaml

# Local smoke: one epoch, one device, the committed causal fixture.
python -m teb_vae.lag_attn_crws.trainer --config teb_vae/lag_attn_crws/configs/tiny.yaml

# Shipped widths on the committed fixture, the clip parked at 1e9.
python -m teb_vae.lag_attn_crws.trainer --config teb_vae/lag_attn_crws/configs/smoke_causal.yaml

# The two arms.
python -m teb_vae.lag_attn_crws.trainer --config teb_vae/lag_attn_crws/configs/sweep_anchor_stride_1.yaml
python -m teb_vae.lag_attn_crws.trainer --config teb_vae/lag_attn_crws/configs/sweep_horizon_15.yaml
```

`default.yaml` points `vae_train_datasets`, `vae_test_datasets` and `stat_path` at `REPOINT_ME`
placeholders, so a production launch fails on a missing file until the causal production shards
exist.

`RUN_CONFIG` near the bottom of `trainer.py` names the config used when the module is launched with
no command line. The entry point therefore works from an IDE Run button, and the only operator
action is to edit that value. A `--config` on the command line always wins, and a relative path
resolves against the repository root. A Run-button launch of `default.yaml` is a single process whose
seven `cuda_devices` make the framework spawn DDP workers. There is no `eval` entry point and no
`check_run` entry point.

Tests:

```bash
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_crws/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_crws/tests -q -m slow
```

`scripts/print_objective_metrics.py` prints every objective metric of the four shipped forecasters
in its `MODELS` tuple. This cell is not in that tuple: the script checks that shared-seam edits leave
those four unchanged.

---

## 16. The switch inventory

Four mechanisms, each a key whose **off-state reproduces the pre-revision model bitwise**. A single
YAML line therefore removes any one mechanism, and the construction tests assert the off-state
state dict key for key and parameter for parameter.

| Key | Shipped | Off-state | What it decides |
|---|---|---|---|
| `lag_kv_source` | `conv_stem` | `encoder` | which source representation the lag attention's keys **and** values come from, and whether the deep source encoder is built (§8.1) |
| `prior_availability_input` | `true` | `false` | whether the prior head receives the source pathway's encode of a zeroed stream (§8.2) |
| `horizon_weight_halflife_steps` | `15.0` | `null` | whether the reconstruction weights the horizon axis, and with what half-life (§5) |
| `alibi_slope_scale` | `0.0` | `1.0` | whether the learnable lag bias is seeded flat or with a monotone decay towards lag $0$ (§8.1) |

All four are constructor keywords, so all four are in the checkpoint's `model_kwargs`, and a
checkpoint is rebuilt through them. The constructor defaults are the off-states.

**An unknown value is refused by name.** The refusal for `lag_kv_source` names its three admissible
values.

**A key absent from this cell's constructor is dropped without a message.** The driver builds the
kwargs by sweeping `inspect.signature(MODEL_CLS.__init__)`. All four keys are therefore written out
in this cell's signature as well as in the parent's. The resolved-config artifact carries the
configured values, and the startup log carries the resolved consequences.

**Every new parameter is reachable in the graph.** The prior head's forward always applies the
clock projection when the flag is on, and the lag bias exists whether or not its seed is flat, so
`find_unused_parameters=False` holds. The horizon weight is a **non-persistent buffer** and is not
in the state dict, so a checkpoint loads across horizons.
