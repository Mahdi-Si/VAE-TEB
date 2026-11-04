# `lag_attn_fs`: the as-built design record

This document records the feature-domain lag-attention VAE-TEB: what it is, what it consumes, what
it returns and what it optimises. It also records every place where the built model differs from
the model it subclasses.

`teb_vae/lag_attn_rws/DESIGN.md` is the as-built record of the parent model. Everything that
document states about the encoders, the heads, the lag attention, the shared decoder, the
initialisation policies and the objective's arithmetic holds here **by inheritance**, and this
document does not restate it. `teb_vae/lag_attn_rws/model_explained.md` states the latent
factorisation that both models share. `teb_vae/lag_attn/DESIGN.md` records the older feature-target
model that has a `decoder_state` bypass.

This document also explains what a reported number of this model means and what it does not mean.
A subclass makes that question easy to get wrong.

---

## 1. What the model is

At every 4-second anchor $t$, the model forecasts the next **two minutes of the stored FHR feature
future**. At the shipped reach budget, that block is $H \cdot C_{\mathrm{keep}} = 30 \times 78 =
2340$ coefficients. The model makes the forecast twice: once from a target-only latent and once from
a source-conditioned latent. The gap between the two forecasts, and the KL divergence between the
two latents, are the coupling readout.

The model fills the missing cell of a $2 \times 2$ grid. `lag_attn` forecasts features through a
`decoder_state` bypass around the latent. `lag_attn_rws` forecasts raw FHR with no bypass. A move
from the first to the second changes the latent factorisation *and* the target domain at once. This
model keeps the factorisation of `lag_attn_rws` and changes only the target domain. A comparison
against `lag_attn` therefore isolates the bypass removal, and a comparison against `lag_attn_rws`
isolates the target domain.

The class is `class SeqVaeLagAttnFs(FeatureForecastTarget, SeqVaeLagAttnRws)` in `nets/model.py`. It
is a subclass of `SeqVaeLagAttnRws` by the MRO, and **its class body is empty**. The members that
make it a feature-domain forecaster live in the mixin `FeatureForecastTarget` in
`nets/feature_target.py`:

| Member of `FeatureForecastTarget` | What it is |
| --- | --- |
| `TARGET_BLOCK_SPLIT` | the channel index where the two stored blocks meet; used only by the block split of §9 |
| `target_channel_weight` | an annotation only; this model sets no value, so every reduction site reads `None` |
| `forecast_likelihood_kwargs` | returns `cell_mask` and `ar_coef`; both are `None` for this model |
| `_default_decoder_out_channels` | names the decoder's width; builds nothing |
| `_build_forecast_target` | gathers the surviving channels and builds each anchor's future window |
| `_resolved_forecast_gaps` | builds the forecast mask and returns the four added readouts of §9 |
| `_forecast_gaps_from_mask` | the reduction behind `_resolved_forecast_gaps`, given a mask |
| `compute_loss` | builds the target, delegates to the shared objective, merges the four readouts |

**Why a mixin.** No member mentions an encoder. Each member reads only `target_gate`, `c_y`,
`geometry`, `horizon`, `coverage_floor`, `logvar_clamp` and `decoder_out_channels`, and every
constructor in this family sets those attributes before it builds the decoder. A second forecaster
that reaches this target through different encoders therefore composes the same mixin instead of
copying it. The mixin is a plain object that derives from `object`. It has no `__init__`, no
`Protocol` and no `__init_subclass__`.

**The order of the bases is load-bearing.** The mixin comes first, so its width hook wins method
resolution over the base's `raw_per_step` hook. If the order is reversed, the base builds the
decoder at $R = 16$ and the objective scores a feature block against it, and nothing raises.

**The empty body is a guarantee.** Because the class defines nothing, the twenty forward keys, the
posterior's structure, the lag map and the objective's metric set are the base's own code objects.
The base's own test suite pins them.

A static type checker cannot see that a *base* constructor sets the seven attributes that the mixin
reads, so it reports them unresolved in `nets/feature_target.py`. The class docstring states the
contract instead of a `Protocol`. The repository has no type-check gate.

**The constructor signature is the parent's, down to the keyword order.**
`trainer._build_model_kwargs` sweeps the signature with `inspect.signature` to decide which config
keys it forwards. A narrowed signature forwards **nothing** and silently builds an all-defaults
model. For this reason the width is a method override and not a constructor keyword, and the mixin
defines no `__init__`.

**Parameter budget.** At `configs/default.yaml` the model holds **5,126,326 parameters**. The
comparison model `SeqVaeLagAttnRws` holds **5,094,458** at the same reach budget, so the delta is
$+31{,}868$. The whole delta is the decoder's output head, at $514 \times (C_{\mathrm{keep}} - R)$
parameters. The value $514$ is $2(d_{\mathrm{hidden}} + 1)$ at `decoder_hidden: 256`, because the
head is $\mathrm{Linear}(d_{\mathrm{hidden}}, X)$ twice (mean and log-variance).

| Model | Reach budget | Decoder width | Parameters | Delta against the comparison model |
| --- | --- | ---: | ---: | ---: |
| `SeqVaeLagAttnFs` (`default.yaml`) | 120 s | 78 | 5,126,326 | +31,868 |
| `SeqVaeLagAttnFs`, unguarded | `null` | 109 | 5,135,988 | +47,802 |
| `SeqVaeLagAttnRws` | 120 s | 16 | 5,094,458 | — |
| `SeqVaeLagAttnRws`, unguarded | `null` | 16 | 5,088,186 | — |

Read the budget column before the delta. A run with a finite budget builds availability input
adapters worth $6{,}272$ parameters. The unguarded `SeqVaeLagAttnRws` total is therefore not the
baseline for either delta.

## 2. Input contract

The model reads its input from the HDF5 shards through `train/data_module.py::GraphDataModule`, at
`trim_minutes: 1.0`. Every field is the same as for the comparison model. Only the choice of target
changes.

| Field | Shape | Role |
| --- | --- | --- |
| `fhr_st`, `fhr_ph` | $(B, 300, 43)$, $(B, 300, 66)$ | target stream, concatenated to $(B, 300, 109)$; **both the network input and the reconstruction target** |
| `up_st`, `up_ph` | $(B, 300, 43)$, $(B, 300, 15)$ | source stream, concatenated to $(B, 300, 58)$ |
| `weight` | $(B, 300)$ | per-step validity on the decimated grid |
| `fhr`, `up` | $(B, 4800)$ | raw traces; **the model does not read them**; row 1 of the diagnostic page only |
| `guid` | — | figure titles and run provenance |

`SeqVaeLagAttnFsTask._build_raw_target` concatenates `fhr_st` and `fhr_ph` in the task, not in
`nets/`, because `nets/` must not contain batch field names. The task also checks
`TARGET_BLOCK_SPLIT` against the width of the `fhr_st` block that it reads.

**`fhr_st` and `fhr_ph` must be in `normalize_fields`.** They are the reconstruction target. If a
block is not normalised, the Gaussian NLL is meaningless against a unit-scale variance model, and the
loader raises nothing. The entry point refuses a config that omits either field. It checks each
field in `LagAttnFsTrainer.TARGET_FIELDS`.

**`fhr_up_ph` is absent from every config and must stay absent.** A coefficient that mixes both
signals breaks the separation between the target-only and source-conditioned paths. Here it would
also put the source's own signal into the forecast target.

When the task flag `target_quality_mask` is on, `SeqVaeLagAttnFsTask._target_step_valid`
concatenates `fhr_st_step_valid` and `fhr_ph_step_valid`, gathers them to the target gate's kept
channels, and marks a step valid only where every kept channel is valid.

## 3. Geometry

The model reuses `TrimmedRawGeometry` **unchanged**. There is no second geometry type and no shared
protocol. The model reads only `t`, `t_valid`, `horizon` and `warmup`. It never calls `r`, `n_raw()`
or `future_block_start()`.

$$T = 300, \qquad H = 30, \qquad T_{\mathrm{valid}} = T - H = 270, \qquad \text{warm-up} = 30.$$

The raw grid stays a true fact about the batch. The loader delivers `fhr` at $4800$ samples and
`weight` at $300$ steps, and `raw_masks._validate_weight` checks `weight.size(1) == geometry.t`. So
`raw_per_step` remains a required geometry input, but it is not the decoder width. If you delete it,
the geometry breaks. If you change it, the decoder width does not change.

The model reuses `forecast_mask`, `contributing_anchors` and `kl_mask` **unchanged**. They broadcast
over the channel axis and produce $(B, 270, 30)$ and $(B, 300)$, because they read only the four
geometry fields above and never `raw_len`, `decimation` or `r`.

## 4. Forward return dict

The forward pass returns the parent's twenty keys. Only the four forecast tensors change shape.

- `mu_base`, `logvar_base`, `mu_full`, `logvar_full`: $(B, 270, 30, C_{\mathrm{keep}})$, against the
  comparison model's $(B, 270, 30, 16)$.
- `mu_prior`, `logvar_prior`, `raw_logvar_prior`, `mu_post`, `logvar_post`, `z_prior`, `z_post`:
  unchanged.
- `target_state`, `source_state`, `attended_source_heads`, `attn_weights`: unchanged.
- `kld_per_t`, `kld_per_t_per_head`, `source_kl_lag_map`: unchanged.
- `mu_prior_sat_frac`, `delta_mu_sat_frac`: unchanged.

The parent adds a twenty-first key, `event_gate`, only when `posterior_event_gate` is true. No config
in this package sets it.

**There is no `decoder_state` and no `delta_mu_src`**, as in the parent: there is no bypass.

**The forward pass emits no `anchor_index`**, so `compute_loss` builds the dense target over all
$T_{\mathrm{valid}}$ anchors. `_build_forecast_target` also accepts an anchor index $(B, A)$ for
subclasses that decode a gathered anchor set.

**`future_index` is inherited and present.** It reads as a raw-target artefact, so it needs a
statement. The base constructor registers it, and a subclass can drop it only by overriding
`__init__`, which the width hook exists to avoid. The buffer is non-persistent, so it reaches no
checkpoint, and this model never reads it. `tests/test_forward_contract.py` zeroes it and asserts
that no reported metric moves.

## 5. Loss, and what its nats are summed over

$$\mathcal{L} = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0
+ \beta(e)\,\mathrm{KL}_{\mathrm{train}} + \beta_p\,R_p
+ \lambda_{\mathrm{ms}} \mathcal{L}_{\mathrm{ms}}
+ \lambda_{\mathrm{deriv}} \mathcal{L}_{\mathrm{deriv}}
+ \lambda_{\mathrm{boundary}} \mathcal{L}_{\mathrm{boundary}}$$

The loss is in nats per anchor. `lag_attn_rws/nets/losses.py::compute_loss` computes it, and this
model calls **that code, not a copy**. The shared module is domain-neutral: it reduces a
$(B, T_{\mathrm{valid}}, H, X)$ block against a $(B, T_{\mathrm{valid}}, H)$ mask and takes both the
target and the block width $X$ as arguments. This model supplies its gathered target and its block
width and delegates. A copy of the objective could drift, and that drift would make every comparison
between the models partly a comparison of losses.

`FeatureForecastTarget.compute_loss` also forwards `lambda_event_gate` and `full_anchor_weight` to the
shared objective. The defaults are `0.0` and `None`, which add nothing. It also forwards
`channel_weight`, `horizon_weight`, `cell_mask` and `ar_coef`. These are `None` for this model unless
`horizon_weight_halflife_steps` is set, and no config in this package sets it.

**The three shape weights ship at $0.0$ because of the target domain.** The last three terms score
the *shape* of a raw waveform: its envelope under pooling, its slope under first differences, and its
first sample against the anchor's last observed sample. In this model the block's last axis counts
$C_{\mathrm{keep}} = 78$ surviving wavelet channels, not $16$ consecutive raw samples. "Adjacent" on
that axis means "the next filter", pooling mixes unrelated scales, and no last observed sample
exists for a boundary term. The config keeps the keys so that the task plumbing and the config-parity
test stay uniform. A term at weight $0.0$ **is not computed**, so its metric is exactly $0.0$.

**`block_width` is $C_{\mathrm{keep}}$, not `geometry.r`.** The mixin passes
`self.decoder_out_channels`. This argument feeds only `mean_logvar_full`, `mean_logvar_base`,
`logvar_full_floor_frac` and `logvar_full_ceil_frac`, **not the loss**. If a caller passes the raw
grid's $R$, no loss changes and no shape check fails, but those four diagnostics change by a factor
of $4.9$. For this reason, `tests/test_objective.py` reassembles every metric from the objective's
primitives with a hand-written block width.

### What the nats are, and are not, comparable to

The reconstruction is a sum over $H \cdot C_{\mathrm{keep}} = 2340$ coefficients. The comparison
model sums over $H \cdot R = 480$ samples. In both models, the factorised Gaussian over correlated
quantities overcounts independent information. Two consequences follow:

- **The nats are not comparable to the raw model's nats.** The two sums run over different blocks
  of differently correlated quantities.
- **The nats are not comparable across reach budgets within this model.** `causal_reach_budget_s`
  sets $C_{\mathrm{keep}}$, which sets the decoder width and the block that every nat is summed over.
  Two arms at different budgets have non-comparable `pred_gap` values and checkpoints that cannot
  load into each other. The class guard cannot separate them, because both stamp `SeqVaeLagAttnFs`.
  Only the width that the stamped `target_keep_index` implies separates them.

Arms of this model at one reach budget can be ranked against each other, and against nothing else.

## 6. Structural constraints

Every reported nat depends on four inherited properties. The tests re-assert them on this subclass.

| Property | What enforces it | Test |
| --- | --- | --- |
| **No decoder bypass**: the gradient reaches the decoder only through $z$ | `BaselineFutureDecoder.forward` takes exactly one tensor; no `decoder_state` head, no second decoder | `tests/test_invariants.py` |
| **Source purity**: the source pathway never sees a target tensor, and the prior never sees the source | separate adapters and encoders; the posterior is a residual on the prior | `tests/test_invariants.py` |
| **Exact zero KL at initialisation** | posterior deltas zeroed after initialisation; one shared $\epsilon$ | `tests/test_invariants.py` |
| **One decoder, invoked twice** | the same module object on $z^p$ and $z^q$ | `tests/test_invariants.py` |

**The zero-KL property depends on the flags.** It holds under the tiny keyword set of the test
fixtures, which sets none of the flags below. The shipped config sets `base_decode: mean`. Under that
flag the two *forecasts* are not bitwise identical at initialisation, but the KL is still exactly
zero, because the KL depends on the two distributions and not on the samples. The shipped config also
sets `posterior_logvar_mode: independent`. Under that mode the initial KL is exactly zero only with
`head_init_calibration: true`, because that flag puts the prior at $\sigma_p = 1$, where the
independent posterior log-variance head starts. An arm that sets `head_init_calibration: false`
starts with a small non-zero KL.

The delta heads are zero-initialised, so **any KL assertion on a freshly built model passes
vacuously**. Every test that checks KL behaviour therefore uses the `perturb_posterior` fixture.

**Step-wise causality holds only under `causal_norm: true`.** Without it, the encoders' time-pooling
normaliser mixes the whole sequence, and a forward probe is not bit-stable at the causal cut.
`tests/test_smear.py` pins both directions.

## 7. The target is gathered, never delayed

The input `ChannelGate` applies **two** operations: a gather of the surviving channels, and a
per-channel delay $\delta_c = \lceil \rho_c / \Delta \rceil$. The delay moves each channel's forward
reach behind the anchor's causal endpoint.

**The target takes the gather and not the delay.** A delayed target would silently ask anchor $t$ to
forecast the future of anchor $t - \delta_c$, per channel, and nothing downstream would fail, because
every shape is identical. `ChannelGate.forward` is `self.delay(index_select(x, -1,
self.keep_index))` and has no gather-only method, so the target builder calls `index_select`
directly:

$$Y^{+}[b, t, \tau, k] = Y[b,\, t + 1 + \tau,\, \mathrm{keep}[k]], \qquad \tau \in [0, H).$$

At the shipped budget, **all 78 surviving channels have a non-zero delay**, from one step for the
fastest channel to thirty steps for the slowest. A target built through the gate would therefore be
wrong in *every* channel. This makes the negative test specific, not only "not equal".

The gather runs **before** the unfold. The two operations commute. Gathering first keeps the copy at
$(B, T, C_{\mathrm{keep}})$ instead of $(B, T_{\mathrm{valid}}, H, C_{\mathrm{keep}})$, which saves a
factor of $H$.

At `causal_reach_budget_s: null`, the survivor set is all $109$ channels and the decoder width
follows. The unguarded arm is therefore well defined, and its target is the ungathered stream.

## 8. The target is smeared, and that is not a leak

A stored coefficient at decimated step $s$ is **not** a value *at* $s$. It is a weighted average of
the raw signal over a window **centred** at raw index $16s$, with half-width $\rho_c$, the channel's
$L_{95}$ energy reach. At short horizons, the forecast target is therefore partly a deterministic
function of raw signal that the model has already observed. The fraction of the support of horizon
step $\tau$ that lies in observed history is

$$b(\tau, \rho_c) = \max\!\left(0,\; \frac{\rho_c - 4\tau}{2\rho_c}\right).$$

This fraction is exactly $0.5$ at $\tau = 0$ for every channel, and it falls linearly to zero at step
$\rho_c / 4$.

**This is not a causality violation.** No future information enters the model. Part of the answer
can be computed from what the model has legitimately observed, which is the opposite of leakage. The
two-sidedness on the *input* side is a separate matter. It is the same in both target domains, and
the `causal_reach_budget_s` gate handles it.

**It does not bias the readout either.** Write the target as $Y^+ = (A, B)$, where $A$ is the
component that the observed history fixes. Then $A$ is $\sigma(Y^-)$-measurable, so
$I(U^-; A \mid Y^-) = 0$ and $I(U^-; Y^+ \mid Y^-) = I(U^-; B \mid Y^-)$. In the estimator, $D_0$ and
$D_1$ come from one shared decoder under one shared $\epsilon$, so the contribution of $A$ is nearly
identical in both branches and largely cancels in the difference.

The smear does affect **optimisation**: a share of the summed NLL is a component that the source
cannot help with. It also affects **interpretation**: a 30-step feature forecast is not the same
physical claim as a 30-step raw forecast.

**This is why the target is the gated subset and not all 109 channels.** The values below come from
the production filter bank (`python -m teb_vae.lag_attn.channel_reach`). The kept set has a maximum
reach of $117.2$ s, and the full set has a maximum reach of $965.5$ s. Over the $H = 30$ horizon, the
mean blend is $0.091$ on the kept set and $0.173$ on the full set. The blend on the kept set reaches
$0.000$ by the last horizon step, because its slowest channel becomes clean at step $29.3$. The full
set never reaches a fully clean step. The gated target also makes the input and the target live
under one causal budget.

> lean-limit: the reach budget bounds the blend but never removes it; replace the bank with a
> one-sided filter bank when one exists and the stored features are regenerated under it.

## 9. Observability without an evaluation pipeline

This package has **no `eval/` package**: no `ModelBinding`, no verdicts, no bootstrap confidence
intervals, no trivial-predictor baselines, no calibration and no per-recording tables. Every readout
is therefore a scalar in the run's own `train_results/metrics_history.csv`.

A scalar summed over $2340$ coefficients cannot separate a model that forecasts three easy channels
well from a model that is uniformly mediocre. Because of §8, it also cannot separate forecasting from
reconstruction of the component that history already fixes. `_resolved_forecast_gaps` adds four
metrics to separate these cases. `LagAttnFsTrainer.TRACKED_METRICS` adds them on both the `train/`
and `val/` stages:

| Metric | What it isolates |
| --- | --- |
| `pred_gap_tau_first` | the horizon step $\tau = 0$, whose target is half-determined by observed history |
| `pred_gap_tau_last` | the step $\tau = H - 1$, whose target on the kept set is not determined by it |
| `pred_gap_st`, `pred_gap_ph` | the surviving channels of each of the two stored blocks, whose filters have different reaches |

**All four are partial sums of `pred_gap`**, over the same denominator. The sum over all horizon
steps recomposes `pred_gap`, and so does the sum of the two block terms. To keep this property, the
per-element term is the objective's own `raw_sample_score`, and the mask is rebuilt through the
objective's own `forecast_mask` and `contributing_anchors`. The reduction processes the two branches
one at a time, because one branch's score is a $(B, T_{\mathrm{valid}}, H, C_{\mathrm{keep}})$
tensor.

The block split needs `TARGET_BLOCK_SPLIT = 43`, which **cannot be derived**, because $c_y$ is the
*sum* of the two block widths. It is a class attribute and not a constructor keyword. A keyword that
changes only which of two diagnostics counts a coefficient would look like an architecture decision
in every checkpoint. No loss term, shape or parameter depends on it, so a wrong value mislabels two
columns and breaks nothing. The task checks it against the data from which it assembles the target.

> lean-limit: the four added readouts are training-path scalars with no uncertainty; replace them
> with per-recording paired statistics when a feature-domain `eval` package exists.

## 10. The diagnostic page

**There is no `lag_attn_fs/plotting.py` and no callback class of its own.** The shared callback is
model-agnostic: it routes through the task's builders, `model.compute_loss` and `model.geometry`. The
trainer reaches it through the inherited `plot_callback_cls()` seam. The domain-specific work is in
`sample_page.py`.

`lag_attn_rws.sample_page.build_diagnostic_figure` owns the layout and draws rows 3 to 7: the prior
mean over the posterior displacement, the per-step per-dimension KL, $K_t$, the lag-attention matrix
and the per-lag KL attribution. `SeqVaeLagAttnFsTask.forecast_rows` supplies rows 1 and 2 through the
`forecast_rows` seam, as `sample_page.feature_forecast_rows` bound to the target gate's keep-index and
`TARGET_BLOCK_SPLIT`.

- **Row 1** shows the raw FHR and UP traces as physiological context. It calls the shared
  `raw_context_row`. The model does not read these traces, but a reader who judges whether a
  forecast is plausible needs them.
- **Row 2** shows three target channels as offset lanes, each with the truth and the base and full
  forecasts with $\mu \pm 2\sigma$ bands. An untitled inset shows the $(C_{\mathrm{keep}} \times H)$
  absolute-error map for one anchor.

**The channels are selected by a stated rule, not by index.** `lag_attn` had a `forecast_channels`
key, which was removed because inherited indices silently changed meaning when `fhr_ph` went from 44
to 66 channels. `select_forecast_channels` ranks the channels by $2\sigma$ calibration and draws the
worst, the middle and the best channel.

**The error map is an inset and not a separate panel.** Its x-axis is the horizon step of one anchor,
not physical time. A side-by-side split would make the forecast curves narrower than the other six
rows, so one column of the page would no longer be one instant across all seven rows. The inset is
untitled, so the property "every titled axes spans the recording" stays exactly checkable.

The forecast row carries no physical unit. The target is the loader's `normalize_fields` output, used
as delivered, so there is no second normalisation to invert.

**The config block keeps the inherited name `lag_attn_rws_plotting`.** If the block is renamed to
match this package, the figure is permanently off, `enabled: true` still looks correct in the config,
and nothing in the log says why.

## 11. Deliberate limitations

- **The nats are local to one budget and one model.** See §5.
- **There is no evaluation pipeline.** See §9. In `lag_attn_rws/eval`, the modules `collect`,
  `metrics`, `spectra`, `oracle` and `events` and the analyses `coherence`, `events` and `samples`
  are structurally raw. A feature-domain `events` or `coherence` analysis is a new scientific
  construction, not a port.
- **The target has no second normalisation.** The loader's output is used as delivered. Per-channel
  target standardisation would make the `head_init_calibration` trivial-predictor argument exactly
  true, but it adds a statistics blob that must travel with the checkpoint. The decoder's
  per-coefficient log-variance head absorbs residual scale differences, and `lag_attn` trains the
  same way.
- **`smoke_hie.yaml` trains and validates on the same shard.** It lists `output/hie_cs.hdf5` as both
  the train and the test dataset, and `dataset_kwargs` is shared between the two loaders, so it
  cannot carry a per-split GUID filter. Runs of this config exercise the optimisation path only.

## 12. Deviation record

This section records where the built model differs from the design it was built from, and why.

**Architecture and construction**

- **The target domain is a mixin and not methods on this class.** The target-domain members live in
  `nets/feature_target.py` as `FeatureForecastTarget`, and the class body of `SeqVaeLagAttnFs` is
  empty. The reason is a second forecaster that reaches the same target through the
  conv-Transformer encoders. `SeqVaeLagAttnTrfRws` derives from `nn.Module` directly and not from
  `SeqVaeLagAttnRws`. A class that inherits from both `SeqVaeLagAttnFs` and `SeqVaeLagAttnTrfRws`
  therefore linearises through the **conv-LSTM** constructor and silently builds the wrong model. See
  §1.
- **The decoder width is a method and not a keyword.** `_default_decoder_out_channels` overrides a
  hook on the base class. The base constructor builds the gate that the hook reads, so nothing
  outside the constructor can compute the width beforehand. Also, a subclass that narrows `__init__`
  to intercept the keyword breaks the `inspect.signature` sweep. On the base class the hook is an
  exact no-op: it consumes no RNG and adds no parameters.
- **`decoder_out_channels` is absent from the config and from `model_kwargs`.** The width can be
  recovered from the stamped `target_keep_index`, and a second field could disagree with the gate.
- **The model builds its own future window** and does not call `future_target` in
  `lag_attn/figure_primitives.py`. The signature of that helper contains the two stored block names,
  which `nets/` must not know. `tests/test_objective.py` pins the model's window to the index
  identity $Y^{+}[t, \tau, k] = Y[t + 1 + \tau, \mathrm{keep}[k]]$.
- **`future_index` is inherited and present.** See §4.

**Objective and metrics**

- **The shared `compute_loss` takes the target and the block width as arguments.** The raw-target
  models pass their raw target and `geometry.r`, so the generalisation is inert for them.
- **The four resolved gaps are merged after the delegation and not computed inside the objective.**
  The raw models' block is one physical channel over thirty horizon steps, so neither split means
  anything there, and their metric dicts stay unchanged.
- **The lag-map conservation identity is `allclose` and not `torch.equal`.** `tests/test_lag_map.py`
  asserts that `source_kl_lag_map` summed over the lag axis equals `kld_per_t`. The map contracts
  over the head axis with `einsum`, and $K_t$ sums over the dimension axis, so the two reach the same
  number in different summation orders.

**Configuration and driver**

- **`raw_per_step` stays.** See §3.
- **The normalisation guard is parameterised and not moved.** `_check_raw_target_normalized` has a
  `fields` keyword, and `main` passes `trainer_cls.TARGET_FIELDS`. The guard stays in `main`'s
  by-name list, so no subclass can drop it. `preflight` keeps its no-op property, which makes a
  bare-bodied `preflight` override safe in the packages that have one.
- **The plotting config block keeps the parent's name.** See §10.
- **`default.yaml` is written out in full and does not inherit.** `tests/test_config_load.py` pins it
  leaf for leaf against `lag_attn_rws/configs/default.yaml`, outside a declared allow-list. The pin
  is **total**: the constructor schema is unchanged, so every key means the same thing in both
  files.
- **The parity allow-list `PARITY_EXEMPT_PATHS` has nine entries.** Five are identity entries (the
  run tag, the output directory and three MLflow names). One is a retuned number,
  `advanced_config.spike_breaker.additive_margin`, which is stated in nats of the summed block and
  is listed in `RETUNED_PATHS`. Three are the shape weights `lambda_ms`, `lambda_deriv` and
  `lambda_boundary`, which ship at $0.0$ for the reason in §5.

## 13. Running it

Run these commands from the repository root.

```bash
# Production, 7 ranks. TEB_RUN_STAMP is required so ranks 1..N-1 share rank 0's run directory.
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_fs.trainer --config teb_vae/lag_attn_fs/configs/default.yaml

# Local smoke: one epoch, one device, the committed four-sample shard.
python -m teb_vae.lag_attn_fs.trainer --config teb_vae/lag_attn_fs/configs/tiny.yaml

# Dev-box run: the shipped geometry over the committed HIE sample shard.
python -m teb_vae.lag_attn_fs.trainer --config teb_vae/lag_attn_fs/configs/smoke_hie.yaml

# A KL-weight arm: the same launch line against any configs/sweep_beta_*.yaml.
```

The rank count must equal `len(general_config.cuda_devices)`. `default.yaml` ships seven devices.

`RUN_CONFIG` near the bottom of `trainer.py` names the config that the module uses when it starts
with no command line. You can therefore launch it from the IDE's Run button and only edit that value.
A `--config` on the command line always wins. A relative path is resolved against the repository
root, and the module changes its working directory to the repository root before it loads the
config.

This package has no `eval` entry point. See §9.

### The shipped configs

| Config | What it is |
| --- | --- |
| `default.yaml` | the production config, written out in full |
| `tiny.yaml` | a one-epoch smoke over `teb_vae/lag_attn/tests/fixtures/tiny_shard.hdf5`, with small widths and `likelihood: mse` |
| `smoke_hie.yaml` | `default.yaml` at local scale over `output/hie_cs.hdf5`; `tests/test_config_load.py` pins its delta to the declared `SMOKE_HIE_DELTA_PATHS` |
| `sweep_beta_*.yaml` | the KL-weight arms below |

**The KL-weight arms.** Each arm is `default.yaml` plus two keys, and `tests/test_sweep_configs.py`
checks that it differs in exactly those two keys. Each arm holds the ratio `beta_prior` /
`beta_schedule.end` at $0.1$, so the two weights move as one axis. The prior anchor's restoring force
saturates: $\partial R_p / \partial \mathrm{lv}_p = \tfrac{1}{2}(e^{\mathrm{lv}_p} - 1)$ per latent
dimension, which tends to $-\tfrac{1}{2}$ as the log-variance falls. The anchor therefore never
pushes back harder than $\tfrac{1}{2}\,$`beta_prior` per dimension.

| Config | `beta_schedule.end` | `beta_prior` |
| --- | ---: | ---: |
| `sweep_beta_1p0.yaml` | 1.0 | 0.1 |
| `sweep_beta_2p5.yaml` | 2.5 | 0.25 |
| `sweep_beta_5p0.yaml` | 5.0 | 0.5 |
| `sweep_beta_10p0.yaml` | 10.0 | 1.0 |

`sweep_beta_1p0.yaml` restates the weights that `default.yaml` ships. The sweep files inherit
`default.yaml`, and with it the production shard paths, the seven devices and the production epoch
count. To run an arm locally, write a two-key overlay on `smoke_hie.yaml`:

```yaml
base: <repo>/teb_vae/lag_attn_fs/configs/smoke_hie.yaml
general_config:
  tag: lag_attn_fs_hie_b1p0
  folders_config: {out_dir_base: output/lag_attn_fs_hie_b1p0}
model_config:
  VAE_model:
    beta_schedule: {end: 1.0}
    beta_prior: 0.1
```

Give each local arm its own tag and output directory, so that runs started in the same minute stay
distinct. Each run writes its fully resolved `resolved_config.yaml` beside its checkpoints. Key a
result by the swept values in that file, not by the directory name.

The shared collapse criterion is in `teb_vae/lag_attn_rws/collapse.py`. A completed run is collapsed
when `val/source_conditioned_kl_raw` is below `KL_COLLAPSE_THRESHOLD_NATS` ($0.02$ nats per anchor)
at each of its final `KL_COLLAPSE_PATIENCE_EPOCHS` ($5$) epochs, or when its final
`val/kld_active_frac` is below $2 / d_z$.

### Re-deriving the loss-scale constants

`advanced_config.trainer.gradient_clip_val` ships at $5000$, and
`advanced_config.spike_breaker.additive_margin` ships at $5 \times 10^{3}$ nats. Both values depend on
the loss scale and do not transfer across block sizes. A change of `causal_reach_budget_s` changes
the block, so re-derive both after such a change:

1. Launch with `gradient_clip_val` set far above any reachable norm, for example $10^{9}$.
2. Read `train/grad_norm` from `metrics_history.csv`.
3. Record its quantiles, its maximum and the fraction of steps above the current threshold.
4. Set the clip to the smallest round value above $q_{99}$.
5. Read the epoch-to-epoch movement of `main_loss` after the warm-up.
6. Set `additive_margin` well above the largest ordinary movement.
7. Write each value and its quantiles into the comment above its key in `configs/default.yaml`.

The task logs `train/grad_norm` *before* clipping, in `on_before_optimizer_step`, and has no `val/`
variant. `train/grad_clip_frac` is a per-step $0$/$1$ exceedance indicator. `MetricsLoggingCallback`
reads `trainer.callback_metrics` at `on_validation_epoch_end`, so `metrics_history.csv` holds one
sampled step per epoch for these two columns, not an epoch aggregate. The mean of `grad_clip_frac`
over epochs estimates the per-step exceedance fraction. Every `val/` column is a true epoch mean.

`ema_floor` ships at $10^{9}$, which switches off the spike breaker's relative test. The breaker gates
the training path only. On a skipped batch, it replaces `train/total_loss` and `train/main_loss` with
its own EMA before they reach the logger, so read `spike_skipped` and the `val/` columns to see
skipped batches.

## 14. Configuration keys

This list names the keys that the claims in this document depend on.

**Required**

- `general_config.tag`
- `general_config.seed`
- `general_config.lr`
- `general_config.lr_milestone`
- `general_config.epochs`
- `general_config.plot_frequency`
- `general_config.accumulate_grad_batches`
- `general_config.folders_config.out_dir_base`
- `model_config.core_model_checkpoint`
- `model_config.VAE_model.beta_schedule`
- `model_config.VAE_model.free_bits`
- `model_config.VAE_model.beta_prior`
- `model_config.VAE_model.likelihood`
- `model_config.VAE_model.lambda_full`
- `model_config.VAE_model.lambda_base`
- `model_config.VAE_model.lambda_ms`
- `model_config.VAE_model.lambda_deriv`
- `model_config.VAE_model.lambda_boundary`
- `model_config.VAE_model.d_model`
- `model_config.VAE_model.d_z`
- `model_config.VAE_model.horizon`
- `model_config.VAE_model.raw_per_step`
- `model_config.VAE_model.warmup_period`
- `model_config.VAE_model.sequence_length`
- `model_config.VAE_model.c_y`
- `model_config.VAE_model.c_u`
- `model_config.VAE_model.use_up_st`
- `model_config.VAE_model.lstm_layers`
- `model_config.VAE_model.dropout`
- `model_config.VAE_model.source_dropout`
- `model_config.VAE_model.decoder_hidden`
- `model_config.VAE_model.logvar_clamp`
- `model_config.VAE_model.mu_scale`
- `model_config.VAE_model.delta_mu_scale`
- `model_config.VAE_model.delta_logvar_scale`
- `model_config.VAE_model.coverage_floor`
- `model_config.VAE_model.base_decode`
- `model_config.VAE_model.posterior_logvar_mode`
- `model_config.VAE_model.causal_norm`
- `model_config.VAE_model.causal_reach_budget_s`
- `model_config.VAE_model.max_lag`
- `model_config.VAE_model.num_heads`
- `model_config.VAE_model.d_head`
- `model_config.VAE_model.use_entmax`
- `model_config.VAE_model.attention_grad_checkpoint`
- `model_config.VAE_model.lag_bias_init`
- `model_config.VAE_model.query_uses_logvar`
- `model_config.VAE_model.horizon_depth`
- `model_config.VAE_model.horizon_kernel`
- `model_config.VAE_model.horizon_film`
- `model_config.VAE_model.horizon_attention_blocks`
- `model_config.VAE_model.horizon_embed_std`
- `model_config.VAE_model.head_init_calibration`
- `model_config.VAE_model.a_head_gain`
- `model_config.VAE_model.encoder_extra_dilations`
- `model_config.VAE_model.encoder_extra_kernel`
- `model_config.VAE_model.conv_norm_groups`
- `advanced_config.trainer.precision`
- `advanced_config.trainer.compile`
- `advanced_config.trainer.num_sanity_val_steps`
- `advanced_config.trainer.use_distributed_sampler`
- `advanced_config.trainer.gradient_clip_val`
- `advanced_config.spike_breaker.ema_floor`
- `advanced_config.spike_breaker.additive_margin`
- `advanced_config.spike_breaker.comparison_metric`
- `advanced_config.spike_breaker.max_consecutive_skips`
- `advanced_config.callbacks.lag_attn_rws_plotting.enabled`
- `dataset_config.stat_path`
- `dataset_config.dataloader_config.normalize_fields`

**Deliberately absent**

- `model_config.VAE_model.decoder_out_channels`
- `model_config.VAE_model.target_block_split`
- `model_config.VAE_model.forecast_channels`
- `model_config.VAE_model.sigma_obs`
- `model_config.VAE_model.head_structured_latent`
- `model_config.VAE_model.freeze_unused_attn_proj`
- `model_config.VAE_model.plain_conv_stack`
- `model_config.VAE_model.horizon_film_per_block`
- `model_config.VAE_model.plain_residual_seams`
- `model_config.VAE_model.lambda_perm`
- `model_config.VAE_model.lag_smoothness_lambda`
- `model_config.VAE_model.detach_baseline_in_full`
- `model_config.VAE_model.kld_support`
- `model_config.VAE_model.lag_band_mask`
- `advanced_config.callbacks.lag_attn_fs_plotting.enabled`
- `advanced_config.callbacks.lag_attn_rws_plotting.plot_frequency`
