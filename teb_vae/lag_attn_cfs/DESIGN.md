# `lag_attn_cfs` — design record

This document describes the causal-feature-domain lag-attention VAE-TEB as it is built. It covers
what the model consumes, what it returns, what it optimises, what it measures, and how to run it.

Companion documents, not restated here:

- `teb_vae/lag_attn_rws/DESIGN.md` records the architecture that all models of the family share.
- `teb_vae/lag_attn_rws/model_explained.md` records the latent factorisation.
- `teb_vae/lag_attn_fs/DESIGN.md` records the feature target and the four gap readouts that this
  package inherits.
- `hdf5_dataset/dataset_explained_research.md` section 8.1 defines the causal dataset variant.
- `eval/EVAL.md`, `eval/FIGURE_GUIDE.md` and `eval/ATTRIBUTION.md` document the evaluation package.

---

## 1. What the model is

The package is one cell of an encoder-by-target grid:

```
                          conv-LSTM encoders          conv-Transformer encoders
  raw FHR target          lag_attn_rws                lag_attn_transformer_rws
  two-sided feature       lag_attn_fs                 lag_attn_transformer_fs
  causal feature          lag_attn_cfs   <- this      lag_attn_transformer_cfs
```

At each admitted 4-second anchor $t$, the model forecasts the next $H$ stored steps of the one-sided
FHR feature stream: $H \cdot C_{\mathrm{keep}}$ coefficients per anchor. The shipped `default.yaml`
sets $H = 10$ (40 s) and keeps $C_{\mathrm{keep}} = 76$ target channels, so the block has
$760$ coefficients. The model forecasts the block twice: once from a target-only latent and once
from a source-conditioned latent. One shared decoder runs both forecasts under one noise draw. The
gap between the two forecasts, and the KL between the two latents resolved across lags, form the
coupling readout.

Structurally the model is `lag_attn_fs` with one change: the transform that produced its features.
The two-sided cells read coefficients that average raw samples on both sides of step $t$. The causal
variant runs the same cascade through a strictly one-sided gammatone bank. Thus a coefficient at
$t$ is a function of $\{x(s) : s \le t\}$ only, and the coupling readout is measured on inputs that
do not contain their own future.

The whole model is

```python
class SeqVaeLagAttnCfs(CausalWarmupInputs, CausalFeatureForecastTarget, SeqVaeLagAttnRws): ...
```

in `nets/model.py`, with a constructor and no other member. The method resolution order is
`SeqVaeLagAttnCfs -> CausalWarmupInputs -> CausalFeatureForecastTarget -> FeatureForecastTarget ->
SeqVaeLagAttnRws -> Module -> object`. Section 6 records why the constructor is the one member.

Several mechanisms of this cell are configuration keys whose off-state reproduces the model as it
was before the key existed. Section 15 lists them with their shipped values and off-states.

## 2. Input contract

The model reads the **causal** HDF5 shards through `train/data_module.py::GraphDataModule`, at
`trim_minutes: 1.0`. The field names and dtypes are those of the two-sided variant. Only the widths
and the transform differ.

The shipped configuration reads shards built with the integer phase operator
(`causal_phase_operator: integer_harmonic_v1`). The legacy operator `ratio_power_v0` gives wider
phase blocks; `sweep_legacy_dualref_physclock.yaml`, `smoke_hie.yaml` and `planted.yaml` read
legacy shards.

| Field | Shape (integer operator) | Shape (legacy operator) | Role |
| --- | --- | --- | --- |
| `fhr_st`, `fhr_ph` | $(B, 300, 36)$, $(B, 300, 44)$ | $(B, 300, 36)$, $(B, 300, 66)$ | target stream, concatenated to $(B, 300, c_y)$; both the network input and the reconstruction target |
| `up_st`, `up_ph` | $(B, 300, 36)$, $(B, 300, 10)$ | $(B, 300, 36)$, $(B, 300, 15)$ | source stream, concatenated to $(B, 300, c_u)$ |
| `weight` | $(B, 300)$ | $(B, 300)$ | per-step validity on the decimated grid |
| `fhr`, `up` | $(B, 4800)$ | $(B, 4800)$ | raw traces; not read by the model, used by the diagnostic page only |
| `guid`, `epoch` | — | — | figure titles, run provenance and the tile phase (section 4) |

The declared widths are $c_y = 80$, $c_u = 46$ under the integer operator and $c_y = 102$,
$c_u = 51$ under the legacy operator. `causal_leg_alignment` and `causal_phase_operator` name the
operator that built the phase blocks. The resolver compares both against the shards' own root
attributes and refuses a mismatch.

**`fhr_st` and `fhr_ph` must be in `normalize_fields`.** They are the reconstruction target. An
unnormalised block makes the Gaussian NLL meaningless against a unit-scale variance model, and the
loader raises nothing. The shared entry point refuses a config that omits either field, by name,
from `LagAttnCfsTrainer.TARGET_FIELDS`.

**`guid` and `epoch` must be in `load_fields`.** The loader honours `load_fields` literally with no
forced additions, and the tile phase is keyed on the pair. Without them every segment is decoded at
one tile grid, and no shape, count or metric shows it. `trainer.preflight` refuses the config.

**`fhr_up_ph` must stay absent from every config.** A coefficient that mixes both signals would put
the source signal into the forecast target. The preflight refuses it in `load_fields` and in
`normalize_fields`.

**Quality fields are opt-in.** On a causal build with `signal_quality=True`, `fhr`/`up` hold the
bridged signals, and the coefficients come from them. The build also writes `up_quality`,
`fhr_quality`, `*_raw`, `<block>_rho` and per-segment shares
(`hdf5_dataset/dataset_explained_research.md` §8.3, §10.5). The shipped `load_fields` names none
of them, and no quality field is ever normalised. Such a build needs a statistics file computed over
it with the quality mask on; the loader refuses an older statistics file by its `quality_version`.
The model-side source key mask is the `source_quality_mask` keyword (section 15). The preflight
refuses `source_quality_mask` without a matching `ua_step_mask`, and the reverse.

**The coefficients inside $[0, W'_c)$ are real float values.** The writer does not zero them and
does not set them to NaN. It attaches the boundary as an attribute and leaves the array untouched.
The normalisation constants were accumulated *excluding* that region, so the values there are on no
defined scale. The binary validity mask `weight` is independent of the warm-up.

## 3. Geometry

### 3.1 The budget and the floor

`TrimmedRawGeometry` is reused unchanged. At the shipped configuration,

$$T = 300, \qquad H = 10, \qquad T_{\mathrm{valid}} = T - H = 290, \qquad F = 134.$$

A forecast at anchor $t$, horizon step $\tau$, reads target time $t + 1 + \tau$. Channel $c$ is
honest there only from its rebased warm-up $W'_c$ onwards. $W'_c$ is the leading delay that encloses
95% of the kernel energy, read from the shards and never from YAML. If every kept channel must be
valid across every anchor's whole window, one inequality results:

$$t + 1 \ge W'_c \quad \forall\, c \in \mathrm{kept}, \; \forall\, t \ge F.$$

It holds exactly when the budget $B = \max_{c \in \mathrm{kept}} W'_c$ and the floor $F \ge B - 1$.
`causal_warmup_budget_steps` sets the budget, and `causal_warmup.resolve_warmup_budget` keeps the
target channels whose warm-up fits it. `warmup_period` sets the floor. The constructor refuses a
floor that is too low and names the requirement it violates.

At budget $134$ on integer-operator shards, the resolver keeps `fhr_st` $32/36$ and `fhr_ph` $44/44$,
so $C_{\mathrm{keep}} = 76$. The source stream is not budget-gated (section 8).

The floor is a causal-validity boundary, not a clinical policy. The pairing requires
$F \ge B - 1$, not equality, so `sweep_floor_150.yaml` sets a higher floor as a one-key change.

**The mask stays three-dimensional.** The alternative is a per-$(\text{anchor}, \tau, c)$ validity
mask that uses partially-valid coefficients at low anchors. Every loss denominator in the family is
an anchor count, so a valid-channel count that varied per anchor would make nats-per-anchor change
with mask density. Under the pairing, the mask stays $(B, A, H)$ and broadcasts over channels as in
every sibling.

**The horizon also sets the anchor count.** Anchors live in $[F, T_{\mathrm{valid}})$, so a shorter
horizon gives more anchors. `sweep_horizon_15.yaml` sets $H = 15$ with stride $7$.

### 3.2 Optional input-channel alignment

Every one-sided filter has its own composed group delay $\tau_c$, stored as `causal_delay_s`. The
encoder reads each stream as one vector per step. `causal_align_reference` can re-index each kept
channel onto a common clock. Channel $c$ is gathered with a shift

$$d_c = \operatorname{round}\!\Bigl(\kappa\,\frac{\tau_{\mathrm{ref}} - \tau_c}{\Delta}\Bigr) \ge 0,
\qquad \Delta = 4\ \mathrm{s}.$$

The shipped `default.yaml` sets `causal_align_reference: null`. Every channel is then read at its
own stored index, $d_c \equiv 0$, and the gate is a pure gather. `sweep_align_target_max.yaml` and
`sweep_target_clock_input.yaml` set `target_max`, which resolves $\tau_{\mathrm{ref}}$ to the slowest
kept target channel. A float is also admitted. The resolver snaps it to the nearest stored channel
delay and returns that channel's own value.

**The factor $\kappa$.** The stored attribute is the gammatone envelope *mean*
$\tau_g = \gamma / (2\pi b)$. The delay that a gather must cancel is the energy centroid of the
impulse response, $(2\gamma - 1)/(4\pi b)$. Their ratio is

$$\kappa = \frac{2\gamma - 1}{2\gamma} = 1 - \frac{1}{2\gamma} = 0.875 \quad \text{at } \gamma = 4,$$

shipped as `ALIGNMENT_DELAY_FACTOR` in `causal_warmup.py`. $\kappa$ scales only the difference
$\tau_{\mathrm{ref}} - \tau_c$, so the reference channel keeps shift $0$. The keep-index does not
depend on $\kappa$.

> The wavelet's zero-mean correction $\kappa_k = \bigl(b_k/(b_k - i\xi_k)\bigr)^{\gamma}$ is a
> different, per-filter complex constant. It always carries its subscript. The unsubscripted
> $\kappa$ above is the single real alignment factor.

A channel slower than the reference would need $d_c < 0$, which reads a later stored step and
therefore raw signal after the anchor. The resolver drops such a channel instead.

**A separate source reference.** `causal_align_reference_source` aligns the source stream onto its
own clock $\tau^u_{\mathrm{ref}}$, while the target keeps $\tau^y_{\mathrm{ref}}$. The key admits
`null` (the source shares the target's reference) or a number of seconds, snapped against the
source stream's own stored delays. There is no `source_max` option. The resolver refuses a source
reference on an unaligned target. The inter-stream offset $\tau^u_{\mathrm{ref}} -
\tau^y_{\mathrm{ref}}$ is carried on `WarmupBudget`, written to the startup log and the preflight
causality record, and printed on the evaluation console. `sweep_legacy_dualref_physclock.yaml` sets
`causal_align_reference_source: 288.2672`.

**The floor with alignment.** A shifted channel is honest only from $W'_c + d_c$, so the floor
must also clear that on both input streams:

$$F \;\ge\; \max\Bigl(\underbrace{B - 1}_{\text{scored target}},\;
\underbrace{\max_c\bigl(W'_c + d_c\bigr)}_{\text{shifted inputs}}\Bigr).$$

With $W_c \approx \rho\tau_c$ and $\rho > \kappa$, $W_c + \Delta d_c \le \rho\tau_{\mathrm{ref}}$, so
the second term costs no warm-up beyond the reference channel's own. The constructor refuses a lower
floor and names which of the two requirements it violates.

**What a lag readout means.** A coefficient at $t$ is a function of the past, so a forecast of
$t + 1 + \tau$ from history up to $t$ is a genuine forecast whatever the filter latency. A lag
readout is an attribution over stored-coefficient time. On an aligned run, the lag between the
streams carries one constant offset:

$$\mathrm{lag}_{\mathrm{physical}} = \Delta\ell + \kappa\bigl(\tau^u_{\mathrm{ref}} -
\tau^y_{\mathrm{ref}}\bigr).$$

`lag_report.physical_lag_seconds` takes the factor through its `realised_delay_factor` argument. On
an unaligned run, each channel keeps its own delay and no single constant applies. The stored
UP/FHR timeline is canonical; the identity uses only the feature grid and the filter-delay terms.

### 3.3 The forecast-target clock

`causal_target_forecast_clock` selects the clock of the scored element. The resolver turns it into
`target_forecast_shift`, one signed integer $s_c$ per kept target channel. The scored element at
anchor $t$, horizon step $\tau$, kept channel $c$ reads stored step $t + 1 + \tau + s_c$.

| Value | $s_c$ | Notes |
| --- | --- | --- |
| `stored` (shipped, and the absent key) | $0$, no vector | the model is built as before the key existed |
| `physical` | $\operatorname{round}(\kappa(\tau_c - \tau_{\min})/\Delta) \ge 0$ | $\tau_{\min}$ is the fastest kept channel; every horizon step scores content at one physical future instant |
| `input` | $-d_c$ | the continuation of the encoder's own aligned stream; requires an aligned target |

Advancing is admissible on a target and refused on an input. A later stored target step is a harder
question, not a leak.

With a shift, four things change in code:

- The scored-target half of the floor becomes $F \ge \max_c(W'_c - s_c) - 1$.
- The anchor ceiling becomes $T_{\mathrm{valid}} - \max(0, \max_c s_c)$ (`anchor_ceiling`).
- `_build_forecast_target` gathers per unique shift value. With no shift it delegates to the parent.
- Every mask is built from `scored_weight`, the validity signal min-pooled over the shift span.
  `compute_loss` substitutes it for `weight` before the parent objective runs.

The shift tuple goes into `model_kwargs` only when the clock is not `stored`
(`FORECAST_ALIGN_MODEL_KWARGS`). The evaluation preflight compares it between config and checkpoint.
`planted.yaml` pins `stored`, and `lag_recovery_check.py` refuses any other clock.

## 4. Anchor tiling

Every sibling decodes all anchors in $[0, T_{\mathrm{valid}})$ every step. This model decodes a
**tiled** set during training:

$$\mathcal{A}(\varphi) = \{\, F + \varphi + kS \;:\; k \ge 0,\; F + \varphi + kS < T_{\mathrm{ceil}} \,\},$$

where $T_{\mathrm{ceil}}$ is the anchor ceiling of section 3.3 and $S$ is `anchor_stride`. The phase
rotates with the epoch, so every anchor in $[F, T_{\mathrm{ceil}})$ is used over epochs. At the
shipped $S = 5$ the dense span $[134, 290)$ has $156$ anchors, and a training row has $32$ tiles at
phase $0$ and $31$ at every other phase. Consecutive windows overlap by half a horizon.

**Validation and test decode every valid anchor**, at stride $1$ and phase $0$.
`SeqVaeLagAttnCfsTask.resolve_anchor_geometry` returns $(\varphi_b, S)$ on `train` and $(0, 1)$ on
`val` and `test`.

**The stride and the phase are forward arguments.** The signature is
`forward(y_st, y_ph, u_stream, anchor_phase=None, anchor_stride=None, source_valid=None)`. The
geometry is not derived from `self.training`, because the diagnostic callback calls `eval()` during
training and then the objective. A mode-derived geometry would make `total_loss` a function of the
dropout switch. Two guards apply: an absent phase raises once the resolved stride exceeds $1$, and a
non-zero phase raises at stride $1$.

**The phase is derived, never drawn.**

$$\varphi_b = \mathrm{blake2b}\big(\texttt{guid}_b \,\Vert\, \lfloor\texttt{domain\_start}_b\rfloor
\,\Vert\, \texttt{epoch} \,\Vert\, \texttt{seed}\big) \bmod S$$

- `hashlib.blake2b` replaces Python's `hash()`, which is salted per process and is stable neither
  across DDP ranks nor across a resume.
- `domain_start` is in the key as well as `guid`, because `guid` identifies the recording, and
  consecutive segments of one recording can share a batch.
- A `torch.randint` inside `forward` would consume the global RNG stream and move the
  reparameterisation noise $\epsilon$.

Each rank hashes its own samples, so the phase needs no cross-rank collective.

**The anchor tensor has a validity companion.** $A_{\max} = \lceil (T_{\mathrm{ceil}} - F)/S
\rceil$ is a geometry constant, so no rank can disagree on shape. Short rows repeat their last valid
anchor and mark it invalid in `anchor_valid`. A padded slot that held a distinct legal anchor would
be scored twice by the reconstruction while `kl_mask` deduplicated it, so the two denominators would
diverge.

**Shape contract.** Everything on the anchor axis is sparse, $(B, A_{\max}, \ldots)$: the decoder
input, the four forecast tensors, the target, `forecast_mask`, `contributing_anchors` and every
per-anchor metric. `kl_mask` is the only place the anchor axis is scattered back to dense. It returns
$(B, T)$ because the latent tensors it gates are $(B, T, d_z)$.

`anchor_stride: 1` recovers the dense range and ships as `sweep_anchor_stride_1.yaml`.

## 5. Loss

$$\mathcal{L} = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0
+ \beta(e)\,\mathrm{KL}_{\mathrm{train}} + \beta_p\,R_p
+ \lambda_{\mathrm{ms}} \mathcal{L}_{\mathrm{ms}}
+ \lambda_{\mathrm{deriv}} \mathcal{L}_{\mathrm{deriv}}
+ \lambda_{\mathrm{boundary}} \mathcal{L}_{\mathrm{boundary}}$$

The loss is in nats per anchor and is computed by `lag_attn_rws/nets/losses.py`, through the
inherited objective. The anchor set reaches it through `forward_outputs['anchor_index']`.
`CausalFeatureForecastTarget.compute_loss` changes one thing: it replaces `weight` with
`scored_weight(weight)` (section 3.3), which is the identity object under the stored clock.

**`lambda_boundary` must be $0$.** The boundary-continuity term joins anchor $t$'s first forecast
sample to anchor $t-1$'s last. This model decodes a tiled set whose entries are $S$ apart, so the term
would join two windows that are not adjacent. The trainer preflight refuses a non-zero value.
`lambda_ms` and `lambda_deriv` ship at $0.0$, because they price the envelope and slope of a raw
waveform, and the block's last axis is a set of wavelet channels.

**`block_width` is $C_{\mathrm{keep}}$, not `geometry.r`.** It feeds only `mean_logvar_full`,
`mean_logvar_base`, `logvar_full_floor_frac` and `logvar_full_ceil_frac`, not the loss.

### 5.1 Per-block channel weight

Both reconstruction terms carry a per-channel weight $w_c$, resolved at construction from
`target_weight_st` and `target_weight_ph` and registered as a non-persistent buffer
(`target_channel_weight`) positional against `target_keep_index`. The shipped pair is $(1.0, 0.1)$.

The vector is renormalised so that $\sum_c w_c = C_{\mathrm{keep}}$. The weighted block therefore
sums to the same magnitude as the uniform block, and only the distribution over channels moves. The
configuration states a ratio: $(10.0, 1.0)$ describes the same objective. At $(1.0, 1.0)$ the
resolved vector is all ones, and `raw_sample_score` skips the multiplication.

### 5.2 Per-horizon-step weight

`horizon_weight_halflife_steps` resolves a geometric decay on the horizon axis
(`losses.horizon_decay_weight`):

$$w_\tau \;\propto\; 2^{-\tau/\lambda}, \qquad \sum_{\tau} w_\tau = H, \qquad \lambda = 5.0
\text{ shipped}.$$

It is a non-persistent buffer and threads through the objective as the channel weight does:
`raw_sample_score` -> `masked_raw_block_per_anchor` -> `masked_raw_likelihood` -> `compute_loss`.
The two weights compose as $w_c w_\tau$. `null` registers no buffer. The resolver refuses a vector
of the wrong length or of shape $(H, 1)$.

**A weighted block is not a log-density**, so $\beta = 1$ is no longer the exact ELBO. The
evaluation pipeline applies neither weight, so its `nll_*` and `pred_gap` are true log-densities.
A `pred_gap` from `metrics_history.csv` and one from `summary.json` are therefore different
quantities. The channel-axis splits (`pred_gap_st`, `pred_gap_ph`, the warm and novelty tertiles)
and `pred_gap_tau_first` / `pred_gap_tau_last` apply both weights, so each is a partial sum of the
`pred_gap` printed beside it.

### 5.3 Target-only persistence residual

$$\mu[t, \tau, c] \;=\; w[\tau, c]\, y^{0}[t, c] \;+\; f_\theta(z_t)[\tau, c].$$

$w$ is a learnable $(H, C_{\mathrm{keep}})$ raw `nn.Parameter`, seeded from a fixed geometric decay
in $\tau$. It applies to the mean head only. `persistence_residual: false` does not build the
parameter.

$y^{0}$ is `_anchor_target_values`, the target's own stored vector at the anchor, on the scored
clock:

$$y^{0}[b, a, k] = Y[b,\, t_a + \min(s_k, 0),\, \mathrm{keep}[k]] \cdot
\mathbb 1\bigl[t_a + \min(s_k, 0) \ge W'_k\bigr].$$

- The gather comes from the target stream alone, before the gate's shift, over the keep-index.
- The clamp $\min(s_k, 0)$ stops an exact per-channel copy under the `input` clock and stops a
  future read under the `physical` clock.
- The warm-up mask uses the non-persistent buffer `target_warm_steps`. At every floor that satisfies
  $F \ge B$ the mask is all ones.

Both decoder invocations receive the same tensor, so $\mu^{\mathrm{full}} - \mu^{\mathrm{base}}$
equals the residual-free gap. The residual therefore opens no source bypass. The forward returns the
tensor as `persistence`, so the permutation control and the Monte Carlo estimator decode with the
same tensor.

### 5.4 Optional likelihood structure

Two constructor keywords are off in every shipped config of this cell:

- `target_scored_horizon`: per declared channel, the number of leading horizon steps scored. It is
  held as the $(H, C_{\mathrm{keep}})$ buffer `target_cell_mask` and resolved from
  `target_phase_fast_cutoff_hz` / `target_phase_fast_horizon` by `scored_horizon.py`.
- `forecast_ar_residual`: a per-channel AR(1) residual along the horizon, $\phi_c = \tanh(a_c)$,
  seeded at $0$, scored on innovations so the block is the joint density.

Both reach `raw_sample_score` through `FeatureForecastTarget.forecast_likelihood_kwargs()`, in the
objective and in the evaluation.

### 5.5 Comparability of the nats

- The block has $H \cdot C_{\mathrm{keep}}$ coefficients. A loss level is comparable only between
  runs with the same horizon, the same kept channels and the same objective weights.
- Two runs at two warm-up budgets have different decoder widths, so their checkpoints cannot load
  into each other. Both stamp `SeqVaeLagAttnCfs`; only the width that `target_keep_index` implies
  separates them.

## 6. Two mixins and one constructor

| Mixin | What it owns |
| --- | --- |
| `CausalWarmupInputs` (`nets/causal_inputs.py`) | the input warm-up mask, the lag validity floor, the tiled anchor set, the prior clock and the forward that decodes at the anchor set |
| `CausalFeatureForecastTarget` (`nets/causal_feature_target.py`) | the one-sided channel layout, the budget-and-floor refusal, the forecast-clock gather, the persistence gather and the resolved readouts of section 10 |

Neither mixin names an encoder, so `lag_attn_transformer_cfs` composes the same two objects over a
different history stack. `CausalFeatureForecastTarget` subclasses `lag_attn_fs`'s
`FeatureForecastTarget`. It sets `TARGET_BLOCK_SPLIT = 36` and `SOURCE_BLOCK_SPLIT = 36`, and it
overrides `_build_forecast_target`, `compute_loss` and `_resolved_forecast_gaps`.

**The order of the bases matters.** The mixins come first, so the tiled forward, the warm-up adapter,
the floored lag mask, the width hook and the resolved gaps win method resolution over the raw-target
members. Reversed, the decoder would be built at $R = 16$ and a $C_{\mathrm{keep}}$-wide block would
be scored against it.

**The constructor is the one member, because of the driver.** `trainer._build_model_kwargs` sweeps
`inspect.signature(MODEL_CLS.__init__)`, so a `**kwargs` signature would forward few keys and
silently build an all-defaults model. The signature is the base's written out in full, with
`target_delays` and `source_delays` removed. The causal-only keywords (`CAUSAL_ONLY_KEYWORDS` in
`causal_inputs.py`) take their place:

- `target_warmup_steps`, `source_warmup_steps`: $W'_c$ per surviving channel.
- `target_align_delays`, `source_align_delays`: $d_c$ per surviving channel. The constructor
  forwards them to the base as `target_delays` / `source_delays`, so they reach `ChannelDelay`.
- `anchor_stride`, `lag_floor`.
- `target_weight_st`, `target_weight_ph`, `target_novelty_frac`, `target_forecast_shift`.
- `target_scored_horizon`, `forecast_ar_residual`, `source_quality_mask`.

A warm-up is a leading mask and `ChannelDelay` is a shift ($\mathrm{out}[t,c] = x[t - \delta_c,
c]$). Separate names keep a checkpoint's `model_kwargs` unambiguous between the two quantities.
`model_kwargs.warmup_model_kwargs` translates the resolved budget into these tuples, so they land in
every checkpoint.

## 7. The warm-up mask lives in the availability adapter

The stored coefficients inside $[0, W'_c)$ are zeroed after normalisation, and an availability
channel announces the fact. Both happen inside `AvailabilityInputAdapter`:

$$m_{t,c} = \mathbb{1}[t \ge \delta_c], \qquad
e_t = W_x (x_t \odot m_t) + W_m (m_t - \mathbf{1}) + \mathbb{1}\Big[\textstyle\sum_c m_{t,c} = 0\Big] e_{\mathrm{start}}$$

The $\odot\, m_t$ term is inert for the two-sided models. A gated model reaches the adapter through
`ChannelGate.forward`, which returns `gathered * available` with the same $\delta$. An ungated model
builds no availability buffer and no mask.

`CausalWarmupInputs._build_adapter` passes $\delta_c = W'_c + d_c$ rather than reading
`gate.delay.delay_steps`. A gathered-and-shifted channel is clean only when the step index has
reached both the warm-up and the shift. On an unaligned run $d_c = 0$.

$e_{\mathrm{start}}$ exists only when every channel of a stream waits at least one step. On the
shipped unaligned configuration each stream has a channel with $W' = 0$, so neither vector is built.
Under alignment both streams can have $\min_c(W'_c + d_c) > 0$, and both vectors are then built.
The preflight refuses `use_up_st: false` with a warm-up budget, because dropping `up_st` would build
the source start embedding as a construction-time change.

## 8. The source stream

Lag attention searches $L = $ `max_lag` $+ 1 = 91$ lags. At anchor $F = 134$ it reads source states
back to step $44$, where part of the source is still inside its own warm-up.

**The warm-up budget gates no source channel.** `causal_source_channels` is the one place the source
is narrowed by choice. It names declared indices into the concatenated `up_st ++ up_ph` stream; `null`
keeps every declared channel. The shipped list keeps 42 of 46 channels: every `up_st` channel except
the four whose rebased warm-up exceeds the floor, and the whole `up_ph` block. After this selection,
every kept source channel is warm by the floor, and the availability staircase is constant on every
scored anchor. Configs that read 51-wide legacy shards restate the list in that layout or set
`null`.

**`lag_floor`** generalises the lag validity mask from $\mathbb{1}[t - \ell \ge 0]$ to
$\mathbb{1}[t - \ell \ge F_u]$. It ships at $0$, where the mask is the sibling's.

**`source_lag_warmth_frac_st` / `_ph`** report the attention mass on lags where at least half
(`WARM_BLOCK_FRACTION = 0.5`) of a stored source block's kept channels are past $W'_c + d_c$.
`CausalWarmupInputs` adds the shift before it splits the stream into its two blocks, so the readout,
the availability mask and the anchor floor use the same quantity. `_resolve_block_warm_steps` treats
an empty block as warm at every step, so read either fraction beside its block's kept width.

### 8.1 Where the keys and values come from

`lag_kv_source` selects the source representation that the lag attention's keys **and** values are
built from (`LAG_KV_SOURCE_CHOICES` in `lag_attn_rws/nets/model.py`).

| Value | K/V representation | Receptive field |
| --- | --- | --- |
| `encoder` (constructor default) | the deep source history state from `CausalConvLstmEncoder` | the whole prefix, through the LSTM |
| `conv_stem` (shipped) | `CausalConvStem` over the availability adapter output | $1 + \sum_b (k_b - 1) r_b$; $387$ steps at kernels $(3, 5, 11, 15, 15)$ and dilations $(1, 2, 4, 8, 16)$ |
| `adapter` | the availability adapter output directly | $1$ step |

**The mechanism argument.** Under `encoder`, the key and the value at lag $0$ are functions of every
earlier source step. By the data-processing inequality, the lag-$0$ entry already contains whatever
a later-lag entry carries. A local K/V representation removes that property, and both K and V move
together for that reason. `CausalConvStem.receptive_field` reports the stem reach. On this cell the
stem reuses the parent encoder's convolution schedule, and $387$ steps is longer than both the lag
window and the sequence. On this cell `conv_stem` therefore removes unbounded recurrence, not
whole-prefix memory.

**Under a local arm the deep source encoder is not built.** Nothing else consumes it, so building it
would leave a parameter block with no gradient under DDP. In this cell's parent, `causal_norm`
(`causalize_norms`) follows the stem where it followed the encoder.

**`source_state` is the tensor the attention reads** under every arm. Both controls follow it:
`perm_forward_outputs` permutes it, and `source_null_forward_outputs` re-encodes gate -> adapter ->
the configured K/V path. Queries, the posterior fusion, `entmax15`, the per-lag key bias, the
$W_o$ convention and the lag mask do not depend on the arm.

**`alibi_slope_scale`** sets the seed of the learnable $(\text{num\_heads}, L)$ lag bias that
`lag_bias_init: alibi_decay` builds. The shipped $0.0$ seeds it flat. At $1.0$ it is seeded with
$-\text{slope}_h \cdot \ell$, a monotone penalty in the lag (`sweep_lag_bias_decay.yaml`).
`lag_bias_init: normal` builds no bias parameter, so it is not the flat arm.

### 8.2 The prior clock and the source-null control

**The hazard.** The source availability pattern $m^u_{t,c}$ is a deterministic function of $t$,
identical for every sample. It enters $q(z \mid Y, U)$ but not $p(z \mid Y)$. It can therefore push
the posterior off the prior with no source information in it. The permutation control cannot detect
that, because every row of the batch carries the same pattern.

**The source-null control.** `kld_source_null` re-runs `source_gate`, `source_adapter` and the module
that `lag_kv_source` selected, from a zeroed source stream. It costs one source encode and no
decode: it needs only $(\mu^{q,\mathrm{null}}, \ell^{q,\mathrm{null}})$ and the existing `kl_mask`
support, and it draws no `torch.randn_like`. With $x \equiv 0$ the adapter output depends on $m_t$
alone, so $h_u^{\mathrm{null}}$ is one $(1, T, d_{\mathrm{model}})$ tensor broadcast over the batch.
Zero is the channel mean over the region the model reads, because the normalisation constants
exclude the warm-up region. `source_conditioned_kl_raw - kld_source_null` is the part of the
coupling readout attributable to source variation.

**`prior_availability_input`** gives the prior the same function of $t$.
`FullLatentPriorHead.forward(h_y, clock=None)` takes it through the head's own `LayerNorm` and a
bias-free linear projection, not shared with the adapter's `mask_proj`. The projection is
zero-initialised and re-zeroed after the parent's generic `initialization(self)` pass. At
initialisation the prior therefore equals the flag-off prior, and the KL starts at exactly zero.

The clock is `CausalWarmupInputs._prior_clock`:

$$h^{u,\varnothing} = \mathrm{Enc}_u\bigl(\mathrm{Adapt}_u(\mathrm{Gate}_u(\mathbf 0))\bigr)
\in \mathbb R^{1 \times T \times d_{\mathrm{model}}}.$$

- It is the encode of a zeroed source stream through the configured K/V path, the same tensor the
  source-null control feeds the posterior.
- It is computed at batch $1$ and broadcast.
- It is detached and computed under no-grad, so no gradient couples the prior to the source
  pathway.
- The source modules are forced into eval mode for the call, so dropout does not make the clock a
  random draw.

The staircase $\mathbb 1[t \ge W'_c + d_c]$ itself is constant on every scored anchor once every kept
source channel is warm by the floor. What varies past the floor is the source pathway's memory of the
arrivals, which is what $h^{u,\varnothing}$ carries. Under `adapter` the pathway has no memory, so
the clock is constant on scored anchors.

**The invariant.** The prior sees no function of the source *values*. The source-purity tests
assert this form.

**The reach of the mechanism.** The posterior is a bounded residual on the prior,
$\mu^q = \mu^p + \Delta_\mu$. The mean half of $\mathrm{KL}(q^\varnothing \Vert p)$ is therefore a
function of the delta head alone, and the prior's input does not appear in it. Conditioning the prior
can reach only the variance half. Section 13 records this limitation.

## 9. Forward return dict

`SeqVaeLagAttnCfs.forward(y_st, y_ph, u_stream, anchor_phase=None, anchor_stride=None,
source_valid=None)` returns the family's twenty keys, `anchor_index` and `anchor_valid`, plus
optional keys.

- `mu_base`, `logvar_base`, `mu_full`, `logvar_full`: $(B, A_{\max}, H, C_{\mathrm{keep}})$.
- `anchor_index` $(B, A_{\max})$ `long` and `anchor_valid` $(B, A_{\max})$ `bool`: the decoded
  anchors and which of them are real.
- `mu_prior`, `logvar_prior`, `raw_logvar_prior`, `mu_post`, `logvar_post`, `z_prior`, `z_post`:
  $(B, T, d_z)$.
- `target_state`, `source_state`: $(B, T, d_{\mathrm{model}})$; `attended_source_heads`
  $(B, T, \text{num\_heads}, d_{\mathrm{head}})$; `attn_weights` $(B, T, \text{num\_heads}, L)$.
  `source_state` is the lag attention's K/V tensor (section 8.1).
- `kld_per_t`, `kld_per_t_per_head`, `source_kl_lag_map`, `mu_prior_sat_frac`, `delta_mu_sat_frac`.
- `persistence` $(B, A_{\max}, C_{\mathrm{keep}})$: only under `persistence_residual`. It is indexed
  by anchor position.
- `event_gate` $(B, T, 1)$: only under `posterior_event_gate`.
- `source_key_mask` $(B, T, L)$ and `source_open` $(B, T)$: only under `source_quality_mask`.

There is no `decoder_state` and no `delta_mu_src`. The decoder receives the latent and the
target-only persistence vector, and nothing else.

The forward builds the anchor set and returns it, so the forecasts, the target, the objective and the
diagnostic page all read one set.

## 10. Readouts

`_resolved_forecast_gaps` is the family's per-package metric hook; the parent `compute_loss` merges
what it returns. It recomputes the four inherited gap splits (`pred_gap_tau_first`,
`pred_gap_tau_last`, `pred_gap_st`, `pred_gap_ph`) on the tiled anchor mask, with the parent's own
reduction `_forecast_gaps_from_mask`. It also emits ten readouts of this package. The task adds two
more on the validation stages only.

| Metric | Stages | What it reports |
| --- | --- | --- |
| `target_warm_frac` | train, val | the budget-and-floor pairing; a constant resolved at construction, $1.0$ by the constructor refusal |
| `anchors_per_sample` | train, val | the number of decoded valid anchors per sample |
| `source_lag_warmth_frac_st` | train, val | attention mass on lags where the first stored source block is warm |
| `source_lag_warmth_frac_ph` | train, val | the same for the second block |
| `pred_gap_warm_lo` / `_mid` / `_hi` | train, val | the forecast gap over the kept target channels, split into three rank tertiles of $W'_c$ |
| `pred_gap_novel_lo` / `_mid` / `_hi` | train, val | the forecast gap split into three rank tertiles of the novelty $\nu_c$ |
| `kld_source_null` | val | the KL from a zeroed source stream (section 8.2) |
| `pred_gap_mean` | val | the forecast gap with both branches decoded at their latent means, $D_0(\mu^p) - D_1(\mu^q)$, under the same weights and masks as `pred_gap` |

**Guards.** `target_warm_frac` and `anchors_per_sample` are geometry guards. `target_warm_frac` is
computed once at construction and emitted as a constant column; a value other than $1.0$ means the
checkpoint was built by code that predates the constructor refusal. `anchors_per_sample` must sit at
the value the run's own geometry implies: $[\lceil (T_{\mathrm{ceil}} - F - (S - 1))/S \rceil,
A_{\max}]$ in training and $T_{\mathrm{ceil}} - F$ on the dense stages.

**The tertiles.** Both tertile families partition the kept target channels by rank and recompose to
`pred_gap` over the same denominator. They cut across the block boundary, so they are not a
restatement of `pred_gap_st` / `pred_gap_ph`.

- The warm tertiles rank channels by $W'_c$. Rank boundaries move with the budget.
- The novelty tertiles rank channels by $\nu_c$, the share of a coefficient's composed envelope mass
  that lies after the anchor at the last scored element. It is an envelope-mass proxy, not an exact
  value fraction. The shard stores the horizon-free curve `causal_novelty_curve`, $N_c(w)$, and the
  resolver looks up $\nu_c = N_c(H + s_c)$ for this run's horizon and each kept channel's forecast
  advance. Legacy shards carry the scalar `causal_novelty_frac` instead, labelled `legacy_scalar`.

The novelty vector travels as `target_novelty_frac` in declared channel coordinates and is gathered
through the keep-index. `warmup_model_kwargs` refuses a gated feature-target model whose shards carry
no novelty vector, rather than defaulting one.

**`_mu_gap_rms` is re-pointed onto the tiled anchor set** in `task.py`. The parent version builds
its masks with no anchors, so under tiling `mu_post_prior_gap_rms` would average over a different
anchor set from the `source_conditioned_kl_raw` printed beside it.

`kld_source_null` and `pred_gap_mean` come from `SeqVaeLagAttnCfsTask._added_metrics`, which runs
only on `val` and `test`. They are absent, not zero-filled, on training steps.
`LagAttnCfsTrainer.TRACKED_METRICS` is the shared surface, plus the four inherited gap splits and the
ten readouts on both stages, plus the two validation-only readouts.

## 11. Figures

1. **The warm-up staircase**, inside the `input_target` / `input_source` page rows. The row builder
   draws each channel's $W'_c$ as a step function. It replaces the shipped builder, which calls
   `describe_streams` on the two-sided Morlet bank and refuses these widths.
2. **The run-level warm-up budget figure**, `causal_warmup_budget` (`warmup_budget.py`). It reuses
   `_budget_panel` by object identity. With $\delta = W'$ and $\rho = \Delta W'$ each bar is
   $[-\Delta W'_c, 0]$: a backward settling length that ends at the anchor's causal endpoint. It is
   drawn from the resolved budget, so it also shows the channels the budget dropped.
3. **The anchor overlay** on the forecast row: the floor, the decoded anchors read off
   `anchor_index`, and their windows. The page is produced at the dense set and phase $0$, and the
   row states the training stride beside it.
4. **The budget tradeoff curve**, `causal_warmup_tradeoff`, produced offline by `causal_warmup.py`'s
   entry point. It plots kept channels, admitted anchors and tiles against every candidate budget,
   with the shipped one marked. Anchors are computed from the survivors' own maximum warm-up, not
   from the threshold.

### 11.1 The diagnostic page rows

The shared builder reserves seven rows. `input_stream_panels` adds two. `forecast_extra_rows` adds
the six of `CAUSAL_EXTRA_ROWS` (`sample_page.py`) directly below the forecast row.

| # | Row | What it is |
|---|-----|------------|
| 1 | `raw` | the raw FHR and UP traces |
| 2 | `forecast` | three channels with $2\sigma$ bands as offset lanes, plus the anchor overlay and one anchor's error map |
| 3–5 | `pred_truth`, `pred_base`, `pred_full` | $Y^{+}$, $\mu^p$ and $\mu^q$ over all kept channels, one colour scale |
| 6 | `pred_skill` | $\lvert Y^{+}-\mu^p\rvert - \lvert Y^{+}-\mu^q\rvert$, the gap resolved per channel and per step |
| 7 | `pred_sigma` | $\sigma^q$ |
| 8 | `pred_gap` | each drawn window's block score, base against full, plus two per-channel profiles |
| 9–10 | `input_target`, `input_source` | the streams as the encoders receive them |
| 11–15 | `latent`, `kld_dims`, `kld_total`, `lag_attn`, `kl_lag_map` | the shared builder's rows |

`COMPACT_PAGE_ROWS` defines a reduced page with `raw`, `forecast`, `input_target`, `input_source`,
`latent`, `kld_total`, `lag_attn` and `kl_lag_map`.

- **Rows 3–8 draw one tiling.** The lane row resolves `_tiling_anchors` once and passes the result
  down as a `_Stitched`.
- **Row 8 uses the objective's own functions.** Its curves go through `raw_sample_score` under
  `forecast_mask`, so a window's height is in the same nats as `nll_base_block` and
  `nll_full_block`. The task binds the likelihood and the coverage floor into the seam. A batch with
  no `weight` gets an annotation instead of the row.
- **Every channel axis reads top-down**, through `top_down_extent`: coefficient $0$ at the top, the
  scattering block above the phase block.
- **Every inset sits in the warm-up prefix**, through `_prefix_boxes`. No anchor exists below $F$,
  so that span of the forecast row is blank by construction. `_draw_error_map` takes the box as an
  argument.

**Raw overlay on the input rows.** Every model row is at the anchor step, and only the raw row is in
physical time. On an aligned run, each input row overlays the raw signal it was computed from,
delayed by $\kappa\tau^y_{\mathrm{ref}}$ (target row) or $\kappa\tau^u_{\mathrm{ref}}$ (source
row), on a twin axis (`InputStreamPanel.raw_field` / `raw_delay_s`). An unaligned run has no single
constant and draws no overlay. Rows that are not on the raw clock state their constant on the x axis;
the forecast rows take theirs from the budget's `target_forecast_clock_delay_s`.
`eval/run.py::attach_warmup_budget` attaches the resolved budget to the task that the evaluation
loads from a checkpoint, after preflight.

## 12. DDP and provenance

Production runs under plain `"ddp"` with `find_unused_parameters=False` under `gaussian_nll`, so every
parameter must be reachable. `mse` leaves the decoder's log-variance heads without gradient and
selects `ddp_find_unused_parameters_true`.

A parameter multiplied by an identically-zero tensor is reachable. What breaks
`find_unused_parameters=False` is a parameter left out of the graph by a Python-level branch on a
tensor value. The availability terms are therefore added unconditionally in the forward, and every
branch is a construction-time decision on a module's existence. The per-segment phase needs no
collective, $A_{\max}$ is a geometry constant, and neither the stride nor the phase reads
`self.training`. The prior clock projection, the persistence weight and the lag bias are each in the
graph whenever they are built. The horizon weight and the channel weight are non-persistent buffers,
so they are absent from the state dict.

**A run's own artifacts state every switch.** Each mechanism is a leaf of `model_config.VAE_model`,
so it lands in the resolved-config artifact beside the checkpoints. The constructor keys land again
in the checkpoint's `model_kwargs`. The startup log's budget summary states the resolved references
in seconds, the inter-stream offset, the shift range and the surviving counts per block. The
evaluation console prints the configured `causal_align_reference` label, the source reference, both
resolved clocks, the offset and `lag_kv_source`, and `summary.json` carries the configured, built and
resolved readings separately. No tracked metric is added for any of them.

The driver drops any config key that the class signature does not list, without an error. The
configured-against-built disclosure and `check_run.py`'s arm criterion (section 14) make such a drop
visible.

## 13. Deliberate limitations

- **The nats are local to the budget, the horizon and the objective weights** (section 5.5).
- **No mixed precision and no compilation.** `precision: "32-true"` and `compile: false`. The
  `nn.LSTM` encoders, the checkpointed attention region and the data-dependent boolean mask indexing
  behind `kld_active_frac` each break TorchInductor.
- **No warm start.** `core_model_checkpoint` stays `null`. A blob from a sibling carries a different
  `model_class` stamp and a decoder head of a different width, and the guard refuses it.
- **Checkpoint compatibility across switch settings.** A checkpoint built at one setting of the
  constructor switches does not load into a model built at another. `load_checkpoint_strict` refuses
  rather than partially loads, and `check_model_class` guards the class.
- **Local measurements are in-sample.** `dataset_kwargs` is shared between the two loaders and cannot
  carry a per-split GUID filter.
- **`causal_reach_budget_s` must be `null`.** It prunes channels by the forward reach of a two-sided
  Morlet, measured on a bank that did not produce these coefficients. The resolver refuses it
  together with a warm-up budget.
- **The K/V receptive field bounds the lag resolution.** A local K/V bounds the encoder's memory, not
  the transform's. Each stored coefficient is a causal integral with its own group delay.

> lean-limit: the prior clock cancels no part of the KL's **mean** term, because the posterior is a
> bounded residual on the prior and that term is a function of the delta head alone; replace with a
> delta defined as $D(a) - D(a^\varnothing)$ -- and `posterior_logvar_mode` back to `residual` for
> the variance half -- when the owner accepts a change to the posterior parameterisation the whole
> coupling readout is defined on.

> lean-limit: the persistence residual ships for the feature-target cells only. The raw grid has 16
> samples per decimated step and no channel axis for a positional weight. The two raw-target cells
> refuse the key by name.

> lean-limit: the causal warm-up is paid once per segment rather than once per recording. A
> left-context rebuild that prepends history before the transform would drive every $W'$ to zero.
> The model side needs no change for such a dataset: the budget resolves to every channel and the
> floor to the model's own warm-up.

> lean-limit: `lag_floor` is configuration for a value no shipped arm varies; replace with a
> constructor-level constant, or with a resolved source floor, when a source floor is needed. One
> bitwise-equality test holds its cost.

**Model side and transform side use different delay conventions.** Only the model-side alignment
applies $\kappa$: `channel_alignment_delays` / `_align_stream` re-index what the shards hold. The
transform-side `causal_scattering.leg_alignment_shift`, which puts the two legs of a phase-harmonic
pair on one clock, uses $\tau_g$ unscaled; applying $\kappa$ there would change stored coefficient
values. `causal_delay_s` stores $\tau_g$, which is the value to report a channel's staleness with.

**The evaluation package** (`eval/`, documented in `eval/EVAL.md`) is a fork of
`teb_vae/lag_attn_rws/eval` edited for this target domain. It keeps the model-free primitives
shared, re-derives the shared arithmetic through both packages in a sibling-agreement test, and
classifies every module of the sibling in `eval/divergences.json`. `GEOMETRY_KEYS` in
`eval/binding.py` includes `prior_availability_input`, `lag_kv_source` and `persistence_residual`,
and excludes `horizon_weight_halflife_steps`, because no evaluated readout applies the horizon
weight.

## 14. Running it

From the repository root:

```bash
# Production, 7 ranks. TEB_RUN_STAMP is required so ranks 1..N-1 share rank 0's run directory,
# and the rank count must equal len(general_config.cuda_devices).
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_cfs.trainer --config teb_vae/lag_attn_cfs/configs/default.yaml

# Local smoke: one epoch, one device, the committed integer-operator causal fixture.
python -m teb_vae.lag_attn_cfs.trainer --config teb_vae/lag_attn_cfs/configs/tiny.yaml

# Dev-box run over a causal HIE sample shard (legacy operator).
python -m teb_vae.lag_attn_cfs.trainer --config teb_vae/lag_attn_cfs/configs/smoke_hie.yaml

# Lag recovery check: fit the committed planted-delay fixture and read the lag profile back
# through the evaluation code. --mode manifest builds the model and stops.
python teb_vae/lag_attn_cfs/lag_recovery_check.py \
    --config teb_vae/lag_attn_cfs/configs/planted.yaml

# Price every candidate source alignment reference for a config's budget.
python teb_vae/lag_attn_cfs/warmup_budget.py --config <config>

# Resolve RUN_CONFIG's budget, print it, and draw the tradeoff curve.
python teb_vae/lag_attn_cfs/causal_warmup.py

# Score a run's training CSV and resolved config, while it is still in flight.
python -m teb_vae.lag_attn_cfs.check_run --run-dir <run>
python -m teb_vae.lag_attn_cfs.check_run --run-dir <a> --second-run-dir <b>

# Evaluate a finished checkpoint on the held-out population: one reviewable directory.
python -m teb_vae.lag_attn_cfs.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt

# Gate that directory offline, with no torch installed; or tabulate several runs.
python -m teb_vae.lag_attn_cfs.eval.verify <run>/eval_results/summary.json
python -m teb_vae.lag_attn_cfs.eval.verify --runs <dir-of-runs> --out <table>.md
```

**IDE Run button.** `RUN_CONFIG` near the bottom of `trainer.py` (and of `causal_warmup.py`) names
the config used when the module runs with no command line. A `--config` on the command line always
wins, and a relative path resolves against the repository root. `check_run.py`, `eval/run.py`,
`eval/probe.py`, `eval/verify.py`, `lag_recovery_check.py` and `warmup_budget.py` use their own
`RUN_ARGS` dicts. A Run-button launch of `default.yaml` is a single process; its seven
`cuda_devices` make the framework spawn DDP workers.

**Checkpoints and stopping.** `default.yaml` enables early stopping on `val/total_loss` at patience
$50$. `model_checkpoint.monitor` is `val/total_loss`, and `secondary_monitor: val/nll_full_block`
builds a second `ModelCheckpoint` under a distinct stem. Record which of the two checkpoints an
evaluation reads.

**Evaluation precondition.** `eval/configs/eval_overrides.yaml` points `vae_test_datasets` and
`stat_path` at `REPOINT_ME` placeholders. Preflight refuses the run until they name a causal holdout
build and statistics computed from those shards at `trim_minutes: 1.0`. The shipped training configs
also point at `REPOINT_ME_causal_int` shard paths. Build such shards with `create_new_pipeline.py`
under `phase_operator=integer_harmonic_v1`; they carry `causal_novelty_curve`.

### 14.1 Configs

| File | Delta from `default.yaml` |
|---|---|
| `tiny.yaml` | one-epoch smoke at tiny widths (`d_model` 32, `max_lag` 8, `likelihood: mse`) on `tiny_shard_causal_int.hdf5` |
| `smoke_hie.yaml` | one device, legacy operator ($c_y = 102$, $c_u = 51$) on a causal HIE sample shard |
| `planted.yaml` | based on `tiny.yaml`; the planted-delay fixture, legacy operator, `horizon: 30`, `horizon_weight_halflife_steps: 15.0`, `anchor_stride: 1`, `causal_align_reference: target_max`, stored clock |
| `sweep_lag_bias_decay.yaml` | `alibi_slope_scale: 1.0` |
| `sweep_lag_kv_adapter.yaml` | `lag_kv_source: adapter` |
| `sweep_align_target_max.yaml` | `causal_align_reference: target_max` |
| `sweep_target_clock_input.yaml` | `causal_align_reference: target_max`, `causal_target_forecast_clock: input` |
| `sweep_legacy_dualref_physclock.yaml` | legacy operator and shards, `causal_source_channels: null`, `causal_align_reference: target_max`, `causal_align_reference_source: 288.2672`, `causal_target_forecast_clock: physical` |
| `sweep_anchor_stride_1.yaml` | `anchor_stride: 1` |
| `sweep_horizon_15.yaml` | `horizon: 15`, `anchor_stride: 7` |
| `sweep_floor_150.yaml` | `warmup_period: 150` |
| `sweep_horizon_depth_3.yaml` | `horizon_depth: 3` |

`sweep_lag_bias_decay.yaml`, `sweep_lag_kv_adapter.yaml`, `sweep_align_target_max.yaml`,
`sweep_target_clock_input.yaml` and `sweep_legacy_dualref_physclock.yaml` also set
`advanced_config.tracking.mlflow.run_name` and `tags.variant`, so the run directory names the arm.

`planted.yaml` is the lag recovery check's geometry, not an arm. Its planted delay $\delta$ is stamped
in stored steps, so the informative lags are $[\delta - H, \delta - 1]$ at its pinned $H = 30$.

### 14.2 `check_run.py` and `eval/verify.py`

The two scripts answer two different questions.

`check_run.py` reads `train_results/metrics_history.csv` and `resolved_config.yaml`, and needs no
checkpoint, no shard and no `torch`. Its exit code follows the tier-1 criteria alone:

1. `target_warm_frac` is exactly $1.0$ on every logged row.
2. `anchors_per_sample` sits in the band derived from the run's own geometry (`resolve_anchor_band`).
3. The loss is finite on every row, and the spike breaker never latches.
4. The warm-tertile sum and `pred_gap_st + pred_gap_ph` agree on every row, within a relative
   tolerance of $10^{-6}$ (`RECOMPOSITION_TOLERANCE`). The check compares split against split, not
   against `pred_gap`, because `pred_gap` is a difference of two large sums.
5. Two runs of one checkpoint produce an identical metric row set (with `--second-run-dir`).
6. The resolved config states every key in `ARM_KEYS`, and `run_name` / `tags.variant` are set.
7. The event mechanisms' columns match the configured arm (`event_weight_floor`,
   `posterior_event_gate`).

It also prints the tier-2 columns in `TIER_TWO_METRICS` without gating them.

`eval/verify.py` reads `eval_results/summary.json` and needs no `torch`. It gates a finished
checkpoint on a held-out population, per recording, with bootstrap intervals. `eval/EVAL.md` is its
contract.

### 14.3 Tests

```bash
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_cfs/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_cfs/tests -q -m slow
```

## 15. Switch inventory

Each key below has an off-state that reproduces the model as it was before the key existed. The
construction tests in `tests/test_construct.py` assert the off-states state-dict key for state-dict
key.

| Key | Shipped | Off-state | Level | What it decides |
|---|---|---|---|---|
| `lag_kv_source` | `conv_stem` | `encoder` | constructor | which source representation builds the lag attention's keys and values, and whether the deep source encoder is built (section 8.1) |
| `prior_availability_input` | `true` | `false` | constructor | whether the prior head is conditioned on the source pathway's encode of a zeroed stream (section 8.2) |
| `persistence_residual` | `true` | `false` | constructor | whether the decoder mean head carries a target-only persistence term (section 5.3) |
| `horizon_weight_halflife_steps` | `5.0` | `null` | constructor | whether the reconstruction weights the horizon axis, and the half-life (section 5.2) |
| `alibi_slope_scale` | `0.0` | `1.0` | constructor | whether the learnable lag bias is seeded flat or with a monotone decay towards lag $0$ (section 8.1) |
| `target_weight_st`, `target_weight_ph` | `1.0`, `0.1` | `1.0`, `1.0` | constructor | the per-block channel weight (section 5.1) |
| `causal_align_reference` | `null` | `null` | task | whether the input streams are aligned onto a common clock (section 3.2) |
| `causal_align_reference_source` | `null` | `null` | task | whether the source stream has its own clock (section 3.2) |
| `causal_target_forecast_clock` | `stored` | `stored` | task | the clock of the scored element (section 3.3) |
| `causal_source_channels` | 42 of 46 | `null` | task | which declared source channels the model reads (section 8) |
| `target_scored_horizon` (via `target_phase_fast_*`) | off | off | constructor | per-channel scored horizon (section 5.4) |
| `forecast_ar_residual` | `false` | `false` | constructor | the AR(1) horizon residual (section 5.4) |
| `source_quality_mask` | `false` | `false` | constructor | the source key mask from `source_valid` (section 2) |
| `posterior_event_gate` | `false` | `false` | constructor | the posterior event gate, charged by `lambda_event_gate` |

**An unknown value is refused by name.** `lag_kv_source` names its admissible values in the refusal.
A source reference that snaps to no stored source delay names the delay it was compared against. A
source reference set against an unaligned target is refused.

**A key absent from this cell's constructor is dropped in silence** by the driver. Every constructor
key is therefore written out in this cell's signature as well as the parent's, and the evaluation
console prints the configured value beside the built one.

**The two levels differ.** Constructor keys land in the checkpoint's `model_kwargs`, and a checkpoint
is rebuilt through them. Task keys are resolution: they decide which channels exist and how each is
shifted, before any constructor runs. They reach a run through the resolved warm-up budget, and their
consequences are stamped as the keep-index, the warm-up vectors and the shift vectors.

**Where each off-state is pinned.** The construction pins live in `tests/test_construct.py` and the
shared owners' stability file. The horizon weight's off-state is pinned as loss equality, because it
builds no parameter. The prior head's `clock=None` equality is pinned in the heads owner file.
