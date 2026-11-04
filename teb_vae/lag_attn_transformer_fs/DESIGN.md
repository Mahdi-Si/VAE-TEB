# `lag_attn_transformer_fs` — design record

This document describes the conv-Transformer feature-domain lag-attention VAE-TEB. It states which
parent supplies which half of the model, what the model consumes and returns, what it optimises,
and how to run it.

Companion documents, not restated here:
`teb_vae/lag_attn_transformer_rws/DESIGN.md` describes the encoders, the adapters, the attention
blocks and the wiring that this model imports unchanged.
`teb_vae/lag_attn_fs/DESIGN.md` describes the feature target, the smear argument and the four added
readouts. `teb_vae/lag_attn_rws/DESIGN.md` describes the architecture that all four models share,
and `teb_vae/lag_attn_rws/model_explained.md` describes the latent factorisation.

---

## 1. What the model is

The model is the fourth cell of an encoder-by-target grid:

```
                        conv-LSTM encoders          conv-Transformer encoders
  raw FHR target        lag_attn_rws                lag_attn_transformer_rws
  feature target        lag_attn_fs                 lag_attn_transformer_fs   <- this
```

At every 4-second anchor $t$, the model forecasts the next **two minutes of the stored FHR feature
future**. At the shipped reach budget that is $H \cdot C_{\mathrm{keep}} = 30 \times 78 = 2340$
coefficients. It makes the forecast twice: once from a target-only latent and once from a
source-conditioned latent. One shared decoder runs on both latents under one noise draw. The gap
between the two forecasts, and the KL between the two latents resolved across lags, are the
coupling readout. The model is the `lag_attn_fs` target reached through the
`lag_attn_transformer_rws` encoders.

The whole model is

```python
class SeqVaeLagAttnTrfFs(FeatureForecastTarget, SeqVaeLagAttnTrfRws): ...
```

with **an empty class body**: `vars(SeqVaeLagAttnTrfFs)` holds no callable and no class constant.
The method resolution order is
`SeqVaeLagAttnTrfFs -> FeatureForecastTarget -> SeqVaeLagAttnTrfRws -> Module -> object`.

`FeatureForecastTarget` lives in `teb_vae/lag_attn_fs/nets/feature_target.py`:

| Member of `FeatureForecastTarget` | What it is |
| --- | --- |
| `TARGET_BLOCK_SPLIT` | where the two stored blocks meet ($43$); used only for the block split of §14 |
| `target_channel_weight` | declared without a value; this model never registers it, so every read returns `None` |
| `forecast_likelihood_kwargs` | the scored-cell mask and the AR(1) coefficient; both are `None` on this model |
| `_default_decoder_out_channels` | returns the decoder width; builds nothing |
| `_build_forecast_target` | gathers the surviving channels and builds each anchor's future window |
| `_resolved_forecast_gaps`, `_forecast_gaps_from_mask` | the four added readouts of §14 |
| `compute_loss` | builds the target, delegates to the shared objective, merges the four readouts |

**Why the class body is empty.** With nothing defined here, the forward keys, the absent
`decoder_state` and `delta_mu_src`, every latent shape, the head-structured posterior, the lag map
and the objective's metric set are the parents' own code objects. The parents' own test suites pin
them over the same functions. So this model differs from `lag_attn_fs` only in the encoder, and from
`lag_attn_transformer_rws` only in the target domain.

At the shipped configuration the model holds **5,034,984 parameters**. §13 gives the arithmetic.

## 2. Input contract

The model reads the HDF5 through `train/data_module.py::GraphDataModule`, at `trim_minutes: 1.0`.
The fields are the same as in every other cell of the grid. The target domain changes only which
field is the reconstruction target.

| Field | Shape | Role |
| --- | --- | --- |
| `fhr_st`, `fhr_ph` | $(B, 300, 43)$, $(B, 300, 66)$ | target stream, concatenated to $(B, 300, 109)$ — **both the network input and the reconstruction target** |
| `up_st`, `up_ph` | $(B, 300, 43)$, $(B, 300, 15)$ | source stream, concatenated to $(B, 300, 58)$ |
| `weight` | $(B, 300)$ | per-step validity on the decimated grid |
| `fhr`, `up` | $(B, 4800)$ | raw traces; **not read by the model** — row 1 of the diagnostic page only |
| `guid` | — | figure titles and run provenance |

**`fhr_st` and `fhr_ph` must be in `normalize_fields`.** They are the reconstruction target, so an
unnormalised block makes the Gaussian NLL meaningless against a unit-scale variance model, and the
loader raises nothing. The shared entry point refuses a config that omits either field. It reads the
fields from `LagAttnTrfFsTrainer.TARGET_FIELDS`, which resolves to the **feature** parent through
the diamond of §7.

**`fhr_up_ph` is absent from every config and must stay absent.** A coefficient that mixes both
signals breaks the target-only / source-conditioned separation. Here it would also put the source
signal into the forecast target.

## 3. Geometry

The model reuses `TrimmedRawGeometry` **unchanged**, from the same module that both parents use:

$$T = 300, \qquad H = 30, \qquad T_{\mathrm{valid}} = T - H = 270, \qquad \text{warm-up} = 30.$$

The target builder and the masks read only `t`, `t_valid`, `horizon` and `warmup`. This model never
calls `r`, `n_raw()` or `future_block_start()`.

`raw_per_step` stays a required geometry input. The loader delivers `fhr` at $4800$ and `weight` at
$300$, and `raw_masks._validate_weight` checks `weight.size(1) == geometry.t`. The value is no
longer the decoder width. Removing it breaks the geometry, and changing it does not change the
decoder width.

`forecast_mask`, `contributing_anchors` and `kl_mask` are reused **unchanged** and broadcast over
the channel axis. They produce $(B, 270, 30)$ and $(B, 300)$.

## 4. Forward return dict

`SeqVaeLagAttnTrfFs.forward(y_st, y_ph, u_stream)` returns the conv-Transformer parent's twenty
keys. **Only the four forecast tensors change shape**, in their last axis.

- `mu_base`, `logvar_base`, `mu_full`, `logvar_full` — $(B, 270, 30, C_{\mathrm{keep}})$: that is
  $(B, 270, 30, 78)$ at the shipped budget and $(B, 270, 30, 109)$ ungated, against the raw
  variant's $(B, 270, 30, 16)$.
- `mu_prior`, `logvar_prior`, `raw_logvar_prior`, `mu_post`, `logvar_post`, `z_prior`, `z_post` —
  $(B, 300, 64)$.
- `target_state`, `source_state` — $(B, 300, 128)$; `attended_source_heads` $(B, 300, 4, 32)$;
  `attn_weights` $(B, 300, 4, 91)$.
- `kld_per_t`, `kld_per_t_per_head`, `source_kl_lag_map`.
- `mu_prior_sat_frac`, `delta_mu_sat_frac`.

A twenty-first key, `event_gate` $(B, T, 1)$, is present only under `posterior_event_gate: true`.
The shipped configs do not set it.

**There is no `decoder_state` and no `delta_mu_src`**, as in both parents: no bypass exists.

**`future_index` is inherited and present**, although it belongs to the raw target. The
conv-Transformer constructor registers it, and a subclass can drop it only by overriding
`__init__`. The width hook exists to avoid that override. The buffer is non-persistent, so it does
not go into a checkpoint, and nothing reads it.

## 5. Loss, and what its nats are summed over

$$\mathcal{L} = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0
+ \beta(e)\,\mathrm{KL}_{\mathrm{train}} + \beta_p\,R_p
+ \lambda_{\mathrm{ms}} \mathcal{L}_{\mathrm{ms}}
+ \lambda_{\mathrm{deriv}} \mathcal{L}_{\mathrm{deriv}}
+ \lambda_{\mathrm{boundary}} \mathcal{L}_{\mathrm{boundary}}$$

The loss is in nats per anchor. `lag_attn_rws/nets/losses.py` computes it — **the same code, not a
copy** — through the mixin's `compute_loss`. That module is domain-neutral: it reduces a
$(B, T_{\mathrm{valid}}, H, X)$ block against a $(B, T_{\mathrm{valid}}, H)$ mask, and it takes
both the target and $X$ as arguments. This model supplies its gathered target and its block width,
and delegates.

$R_p$ is the prior's scale rate,
$\mathrm{KL}\!\left(\mathcal N(\mu^p, \operatorname{diag} e^{\ell^p}) \Vert \mathcal N(\mu^p,
I)\right)$, reduced on the KL's own anchor support and in the same units. No other term of the
objective penalises a *narrow* prior.

**The three shape weights ship at $0.0$**, as in the feature-target sibling. $\mathcal{L}_{\mathrm{ms}}$,
$\mathcal{L}_{\mathrm{deriv}}$ and $\mathcal{L}_{\mathrm{boundary}}$ price the envelope, slope and
opening level of a raw *waveform*. Here the block axis is $C_{\mathrm{keep}} = 78$ surviving wavelet
channels. "Adjacent" there means "the next filter", pooling mixes unrelated scales, and no
last-observed sample exists for a boundary term. The keys stay in the config so the task plumbing
and the parity comparison against the raw sibling stay uniform. A term at weight $0.0$ is **not
computed**, and its metric is exactly $0.0$.

**`block_width` is $C_{\mathrm{keep}}$, not `geometry.r`.** It feeds only `mean_logvar_full`,
`mean_logvar_base`, `logvar_full_floor_frac` and `logvar_full_ceil_frac` — **not the loss**. If the
raw grid's $R$ is passed instead, no loss changes and no shape check fails, but those four
diagnostics are rescaled by $78/16 \approx 4.9$. `tests/test_objective.py` reassembles every metric
at a hand-written block width for this reason.

### What the nats are summed over

The reconstruction is a sum over $H \cdot C_{\mathrm{keep}} = 2340$ coefficients, against
$H \cdot R = 480$ samples in the raw models. A factorised Gaussian over correlated wavelet
coefficients also overcounts independent information. Two consequences follow:

- **The nats are not comparable to the raw models' nats.** The two sums run over different blocks
  of differently correlated quantities.
- **The nats are not comparable across reach budgets within this model.** `causal_reach_budget_s`
  changes $C_{\mathrm{keep}}$, hence the decoder width, hence the block. Two runs at different
  budgets also have **mutually unloadable checkpoints**. The class guard cannot separate them,
  because both stamp `SeqVaeLagAttnTrfFs`; only the width that the stamped `target_keep_index`
  implies differs.

## 6. The mixin

**The obvious construction runs the wrong constructor, silently.** `SeqVaeLagAttnFs` subclasses
`SeqVaeLagAttnRws`, while `SeqVaeLagAttnTrfRws` derives from `nn.Module` directly. So
`class SeqVaeLagAttnTrfFs(SeqVaeLagAttnFs, SeqVaeLagAttnTrfRws)` linearises as
`TrfFs -> Fs -> Rws -> TrfRws -> Module` and **runs the conv-LSTM constructor**. That model builds
and trains, but it is not this architecture.

So the members that make a model a feature forecaster live in
`teb_vae/lag_attn_fs/nets/feature_target.py`, and both feature models are their own architecture
plus that mixin. **No member mentions an encoder.** The mixin reads `target_gate`, `c_y`,
`geometry`, `horizon`, `coverage_floor`, `logvar_clamp` and `decoder_out_channels`, and both
constructors set all of them before they build the decoder. It also reads `target_channel_weight`,
`horizon_weight`, `target_cell_mask` and `target_ar_logit` through `getattr` with a `None` default.

The mixin is a plain object that derives from `object`. It has no `__init__`, no `Protocol` and no
`__init_subclass__`. One copy of the target logic serves both feature models, so the target-delay
error of §8 and the `block_width` error of §5 cannot drift apart between them.

**The order of the bases is load-bearing.** The mixin comes first, so its width hook wins method
resolution over the conv-Transformer parent's `raw_per_step` hook. If the order is reversed, the
decoder is built at $R = 16$ and a $C_{\mathrm{keep}}$-wide feature block is scored against it.
Because the objective takes the block width as an argument, nothing raises.

**The width is a method, not a constructor keyword.** `_default_decoder_out_channels` overrides a
hook on `SeqVaeLagAttnTrfRws`, which builds its decoder at `out_channels=self.decoder_out_channels`.
There are two reasons for a method:

1. The hook reads the target gate, and that constructor builds the gate, so nothing outside can
   compute the value beforehand.
2. A subclass that narrows `__init__` to intercept a keyword breaks the `inspect.signature` sweep in
   `trainer._build_model_kwargs`. The sweep then forwards **no configuration at all** and silently
   builds an all-defaults model.

The constructor signature here is therefore the conv-Transformer parent's, keyword for keyword.

A static type checker cannot see that a *base* constructor sets the mixin's read attributes, so it
reports them unresolved in `lag_attn_fs/nets/feature_target.py`. The class docstring states the
contract instead. The repository has no type-check gate.

## 7. The two diamonds, and their resolution order

```python
class SeqVaeLagAttnTrfFsTask(SeqVaeLagAttnFsTask, SeqVaeLagAttnTrfRwsTask):
    ...

class LagAttnTrfFsTrainer(LagAttnFsTrainer, LagAttnTrfRwsTrainer):
    MODEL_CLS = SeqVaeLagAttnTrfFs
    TASK_CLS = SeqVaeLagAttnTrfFsTask
    CHECKPOINT_STEM = "lag-attn-trf-fs"
```

The task defines **no** callable. The driver re-points three class attributes and defines no
method. The resolution orders are

`SeqVaeLagAttnTrfFsTask -> SeqVaeLagAttnFsTask -> SeqVaeLagAttnTrfRwsTask -> SeqVaeLagAttnRwsTask
-> LightningModelBase`, and
`LagAttnTrfFsTrainer -> LagAttnFsTrainer -> LagAttnTrfRwsTrainer -> LagAttnRwsTrainer ->
GraphModelBase`.

`tests/test_task.py` and `tests/test_trainer.py` assert each resolution below by the behaviour it
decides.

**Both diamonds are well-formed because both parents descend from a common base.** The task
parents' overrides are disjoint today — $\{$`_build_raw_target`, `forecast_rows`$\}$ against
$\{$`build_lr_scheduler`$\}$ — but the tests assert each behaviour against its named class, so a
future overlap fails a test. The driver parents' overrides are **not** disjoint: both set
`MODEL_CLS`, `TASK_CLS` and `CHECKPOINT_STEM`, and resolution order alone takes the feature side.
So the driver re-points all three. Each omission fails silently:

- Without `MODEL_CLS`, the driver builds `SeqVaeLagAttnFs`, a conv-LSTM model, with no error.
- Without `TASK_CLS`, the same happens one layer up.
- Without `CHECKPOINT_STEM`, the driver writes `lag-attn-fs-*.ckpt` and mixes two models'
  checkpoints in a shared output tree.

All other members arrive by resolution order:

| From the **feature** parent | From the **conv-Transformer** parent | From the shared driver |
| --- | --- | --- |
| `TARGET_FIELDS = ("fhr_st", "fhr_ph")` | `compile_model_requested` | `PLOT_CONFIG_KEY` |
| `TRACKED_METRICS`, 90 entries | `_build_model_kwargs`, `create_model` | `preflight` |
| `_build_raw_target`, `forecast_rows` | `_build_trainer_kwargs`, `build_lr_scheduler` | the DDP strategy selection |

**`compile_model_requested` resolves to the conv-Transformer side.** `LagAttnFsTrainer` does not
define it, so lookup passes through to `LagAttnTrfRwsTrainer`. That parent turns the shared
driver's hard refusal into a live key: `torch.compile` is permitted on this model. Because this
outcome comes from resolution order and not from code in this package, `tests/test_trainer.py`
asserts it explicitly. The shipped configs keep `compile: false` (§15).

**`PLOT_CONFIG_KEY` stays `"lag_attn_rws_plotting"`**, and the config block keeps that name. The
shared callback assembly reads the literal. A sibling that renames it gets no figure, no error and
no log line.

## 8. The target is gathered, never delayed

The input `ChannelGate` applies **two** operations. It gathers the channels that survive the reach
budget. Then it applies a per-channel delay $\delta_c = \lceil \rho_c / \Delta \rceil$ that pushes
each channel's forward reach behind the anchor's causal endpoint.

**The target takes the gather and not the delay.** A delayed target silently asks anchor $t$ to
forecast the future of anchor $t - \delta_c$, per channel, and nothing downstream fails because every
shape is identical. `ChannelGate.forward` is `self.delay(index_select(x, -1, self.keep_index))` and
has no gather-only method, so the target builder calls `index_select` directly.

**This architecture's own constructor builds the gate.** `tests/test_objective.py` asserts the
index identity $Y^{+}[b,t,\tau,k] = Y[b,\,t+1+\tau,\,\mathrm{keep}[k]]$ against a **hand-written
slice**, not against the shared `figure_primitives.future_target` helper. A target and a reference
that both wrongly applied the delay would agree.

At the shipped budget, **all 78 surviving channels carry a non-zero delay**, from one step to thirty
steps. So a target built through the gate is wrong in every channel, and the negative test is
specific.

The gather runs **before** the window is built. The two operations commute. Gathering first keeps
the copy at $(B, T, C_{\mathrm{keep}})$ and not $(B, T_{\mathrm{valid}}, H, C_{\mathrm{keep}})$, a
factor of $H$ smaller.

At `causal_reach_budget_s: null` the survivor set is all $109$ channels and the decoder width
follows. The target is then the ungathered stream.

### The target is also smeared, and that is not a leak

A stored coefficient at decimated step $s$ is a weighted average of raw signal over a window
*centred* at raw index $16s$. So a share of the short-horizon target is a deterministic function of
signal the model has already observed. The fraction is $\max(0, (\rho_c - 4\tau) / 2\rho_c)$: $0.5$
at $\tau = 0$ for every channel, and $0$ by $\tau = 29$ on the surviving set.
`lag_attn_fs/DESIGN.md` gives the full argument. The property belongs to the target and the filter
bank, so the encoder does not change it. It affects optimisation and interpretation, not causality.
The readouts of §14 separate forecasting from reconstruction of the already-determined component.

## 9. Structural constraints

The table lists the properties that every reported nat depends on. Where an invariant is the
parents' own, the conv-Transformer parent's suite pins it over this same forward. This package runs
the composition once at the production budget.

| Property | What enforces it | Test |
| --- | --- | --- |
| **No decoder bypass** — gradient reaches the decoder only through $z$ | `BaselineFutureDecoder.forward` takes exactly one tensor, at $d_z$ in-features; no `decoder_state` head, no second decoder | the conv-Transformer parent's suite, over this same forward |
| **Source purity** — the prior never sees the source, the source state never sees the target | separate gates, adapters and encoders; the posterior is a residual on the prior | `tests/test_invariants.py`, at the production budget |
| **Exact zero KL at initialisation** | posterior deltas zeroed **after** the generic init; one shared $\epsilon$ | `tests/test_invariants.py`, at the production budget |
| **One decoder, invoked twice** | the same module object on $z^p$ and $z^q$ | the conv-Transformer parent's suite, over this same forward |
| **Token causality**, unconditionally | §12 | `tests/test_causality.py` |
| **The lag attribution identity**, $\sum_\ell \widetilde K_{t,\ell} = K_t$ | the lag attention is built at `dropout=0.0` | the conv-Transformer parent's suite, over this same forward |
| **`lag_attn.W_o` frozen** | the head-structured posterior consumes the per-head summaries, so `W_o` receives no gradient | the conv-Transformer parent's suite, over this same constructor |
| **No recurrence and no time-pooling normaliser** on a history path | none is constructed; the remaining `GroupNorm`s are all under `horizon_core.` | the conv-Transformer parent's suite, over this same constructor |

**The zero-KL property depends on the flags.** It holds under the conv-Transformer suite's tiny
keyword set, which sets none of them. The shipped config sets `base_decode: mean`. Under that
setting the two *forecasts* are not bitwise identical at initialisation, but the KL is still exactly
zero, because the KL depends on the two distributions and not on the samples. The shipped config
also sets `posterior_logvar_mode: independent`. Under that setting the initial KL is zero only with
`head_init_calibration: true`.

Because the delta heads are zero-initialised, **any KL assertion on a freshly constructed model
passes vacuously**. Every test in this package that checks KL behaviour therefore uses the
`perturb_posterior` fixture.

**The decoder subtree holds no dropout module**, which is stronger than $p = 0$. The projection MLP
builds its `nn.Dropout` layers only at a positive rate. At the decoder's hard-coded rate of $0$
there is no layer that `model.train()` or a config rate can re-enable.

## 10. Initialisation order

The order in the conv-Transformer parent's `__init__` is load-bearing, top to bottom, and this model
reaches it unchanged:

1. The constructor builds the horizon core, sets
   `self.decoder_out_channels = self._default_decoder_out_channels()`, and builds the decoder at
   that width. **This is the only step the mixin reaches**, and it comes before all steps below.
2. `initialization(self)` — the shared generic pass, Xavier over every `nn.Linear` and `nn.Conv1d`.
3. `init_depthwise_(self)` — immediately after, never before. Xavier on a $(C, 1, k)$ depthwise
   weight reads $\mathrm{fan\_in} = k$ against $\mathrm{fan\_out} = Ck$, so the standard deviation
   is a factor $\sqrt{(1 + C)/2} = 8.03$ too small at $C = 128$, independent of $k$. The model
   records the returned count as `n_depthwise_init`.
4. `_zero_init_delta_heads()` — the generic pass refills the posterior delta heads, and this step
   restores the exact zero-KL start.
5. `prior_head.zero_init_clock()` and `posterior_head.zero_init_event_gate()` — the same repair for
   the prior's clock projection and the posterior's event gate. Each is a no-op on a head built
   without that path.
6. `_zero_init_film_generators()` — the horizon core zero-initialises the FiLM generators, and the
   generic pass refills them. Re-zeroing here makes the identity at initialisation true.
7. The three zero-parameter policies — `horizon_embed_std`, `head_init_calibration`, `a_head_gain`.
   Each runs only when its config value differs from the constructor default, so a default-flag
   model is bitwise unchanged.

**`head_init_calibration` centres the log-variance head at $\sigma = 1$ across $78$ output
channels, not $16$.** It is the one initialisation policy that the width change reaches.
`tests/test_invariants.py` checks that the calibration ran on every channel of a gate-wide head.

## 11. DDP reachability

Production runs under plain `"ddp"` with `find_unused_parameters=False` under `gaussian_nll`, so
every parameter must be reachable. Under `mse` the decoder's log-variance heads receive no gradient,
and the driver selects `ddp_find_unused_parameters_true`. `tests/test_ddp_strategy.py` checks both
cases, and checks that under `mse` the starved set is **exactly** those heads.

A parameter multiplied by an identically-zero tensor **is** reachable: its `AccumulateGrad` node
fires, and DDP's reducer marks it ready. `find_unused_parameters=False` breaks only when a
Python-level branch on a tensor value leaves a parameter *out of the graph*. So the forward adds the
availability terms unconditionally, and the branching happens at construction time only. The two
availability parameters have **different** conditions: $W_m$ (`mask_proj`) exists when
$\max_c \delta_c > 0$, and $e_{\mathrm{start}}$ (`start_embed`) exists when $\min_c \delta_c > 0$.
The start indicator is non-zero for some $t$ exactly when *every* channel is delayed.

**This package does not port the tensor-branch AST walk.** Every `forward` that executes here
belongs to an imported module: the encoders and blocks from the conv-Transformer parent, and
`AvailabilityInputAdapter` from `teb_vae/lag_attn/nets/encoders.py`. Those packages walk them. The
empty class body, asserted in `tests/test_construct.py`, means this package defines **no
`forward`**.

The reachability probe's guarded arm resolves the **production** budget at `sequence_length: 64`,
`warmup_period: 30`. It does not use the hand-made tiny guard, because that guard's delay tuple
starts at $0$ and so builds no start embedding.

**`broadcast_buffers=False`** is safe because every buffer is a deterministic function of the
config: rotary tables, causal masks, the gate's keep-index and delays, the adapters' availability
patterns and the raw-target index grid. The model has no `BatchNorm`. `static_graph` is not set,
because the loss-spike breaker substitutes a zero-weighted sum over every parameter on a skipped
batch, which is a different backward from the one the first iteration records.

## 12. Causality, unconditionally

**Step-wise causality holds through the whole model with no qualifying flag.** In the conv-LSTM
cell, step-wise causality needs `causal_norm: true`, because otherwise a time-pooling normaliser
mixes the whole sequence (`lag_attn_fs/DESIGN.md`). These encoders have no such normaliser.
`RMSNorm` reduces over channels only, convolutions pad left before a `padding == 0` `Conv1d`, and
attention is causal by kernel flag (target, full prefix) or by an explicit band mask (source,
$W_U = 16$). **`causal_norm` is not a constructor keyword of this model.**

`tests/test_causality.py` checks causality through the assembled model, at the shipped reach budget
and at the tiny fixture. It also checks prefix equivalence,
$\mathcal{E}(X_{0:T-1})_t = \mathcal{E}(X_{0:t})_t$.

**Prefix equivalence applies to the latent and the mean-decoded branch, not to a draw.**
`randn_like` fills row-major over $(B, T, d_z)$, so the $\epsilon$ of sample $b > 0$ at anchor $t$
sits at offset $bTd_z + td_z$, which changes with $T$. The test therefore checks the base forecast
under `base_decode: mean`. This is a property of the test harness, not of the model.

**Token causality is not raw-signal causality**, and this model delivers the first. The
architecture guarantees $H_t = f(X_{\le t})$. Raw-signal causality needs one-sided feature
transforms, but the stored features are two-sided. `causal_reach_budget_s` bounds the leak: it
prunes channels whose analytic reach exceeds the budget and delays the survivors. It does not remove
the leak, because $L_{95}$ is an energy *quantile*. So the KL is called
`source_conditioned_kl_raw` / `_train`, and its decomposition is `source_kl_lag_map`. It is **not**
called transfer entropy.

> lean-limit: token-causal features; replace with genuinely one-sided scattering and phase-harmonic
> transforms when a causal front end exists and the stored features are regenerated under it.

## 13. Parameter budget

Parameter counts of constructed models at the shipped $120$ s reach budget:

| | conv-LSTM encoders | conv-Transformer encoders |
| --- | ---: | ---: |
| raw target, unguarded | — | $4{,}996{,}844$ |
| raw target, guarded ($C = 16$) | — | $5{,}003{,}116$ |
| **feature target, guarded** ($C_{\mathrm{keep}} = 78$) | $5{,}126{,}326$ | $\mathbf{5{,}034{,}984}$ |
| feature target, unguarded ($C = 109$) | $5{,}135{,}988$ | $5{,}044{,}646$ |

Every module outside the encoders is shared, so the deltas decompose cleanly.

**The encoder axis: $-91{,}342$**, guarded against guarded. The whole delta is in the two history
encoders.

**The target axis: $+31{,}868 = 514 \times (78 - 16)$.** This is the decoder's output head and
nothing else: two per-channel output rows plus their biases, at $d_{\mathrm{hidden}} + 1 = 257$
each. Ungated it is $+47{,}802 = 514 \times (109 - 16)$.

**The guard adds $+6{,}272$** on the raw-target model. The availability projections alone are
$128 \times (78 + 29) = 13{,}696$, because `target_adapter.mask_proj.weight` is $(128, 78)$ and the
source's is $(128, 29)$. The net delta adds $256$ for the two `start_embed` vectors and subtracts
$7{,}680 = 128 \times 60$, because the input linears narrow from $109$ and $58$ channels to $78$ and
$29$.

## 14. Readouts while the evaluation is deferred

There is **no `eval/` package for this model**: no `ModelBinding`, no verdicts, no bootstrap
confidence intervals, no trivial-predictor baselines, no calibration and no per-recording tables.
Every readout is a scalar in the run's own `train_results/metrics_history.csv`.

A scalar summed over $2340$ coefficients cannot separate a model that forecasts three easy channels
well from one that is uniformly mediocre. Given §8, it also cannot separate forecasting from
reconstruction of the already-determined component. Four metrics address this. The mixin computes
them, and the feature parent's `TRACKED_METRICS` reports them:

| Metric | What it isolates |
| --- | --- |
| `pred_gap_tau_first` | the horizon step whose target is half-determined by observed history |
| `pred_gap_tau_last` | the step whose target is not determined by it at all |
| `pred_gap_st`, `pred_gap_ph` | the two stored blocks, whose filters have different reaches |

**All four are partial sums of `pred_gap`**, over the same denominator, and both splits recompose
to it. The per-element term is the objective's own `raw_sample_score`, and the mask is rebuilt
through the objective's own two functions. This package adds no metric of its own.

The block split needs `TARGET_BLOCK_SPLIT = 43`, which **cannot be derived**, because $c_y$ is the
sum of the two block widths. It is a class attribute and not a constructor keyword: a key that only
moves a coefficient between two diagnostics would look like an architecture decision in every
checkpoint. Nothing else depends on it, so a wrong value mislabels two columns and breaks nothing.
The task checks it against the data that it assembles the target from.

**The diagnostic page is inherited whole**, through the task's `forecast_rows` property and the
shared model-agnostic callback. This package ships no `plotting.py` and no `sample_page.py`.

### Training diagnostics in `metrics_history.csv`

- `train/grad_norm` is the **pre-clip** gradient $L_2$ norm. The shared task logs it in
  `on_before_optimizer_step`. There is no `val/` variant.
- `train/grad_clip_frac` is a per-step $0$/$1$ indicator of `grad_norm > gradient_clip_val`. It is
  absent when no clipping is configured.
- For these two columns, `metrics_history.csv` records **one optimizer step per epoch**, not an
  epoch aggregate. `MetricsLoggingCallback` reads `trainer.callback_metrics` in
  `on_validation_epoch_end`, before the training epoch is reduced. So the mean of `grad_clip_frac`
  over epochs estimates the per-step exceedance fraction. Every `val/` column is a true epoch mean.
- The shared collapse criterion is `is_collapsed` in `teb_vae/lag_attn_rws/collapse.py`. A
  completed run is collapsed when `source_conditioned_kl_raw` is below $0.02$ nats per anchor at
  every one of its last $5$ epochs, or when its final `kld_active_frac` is below $2 / d_z$.

### Setting the two loss-scale constants

Both constants are in nats of the summed block, so they depend on the block size and on the
encoder. To re-derive `advanced_config.trainer.gradient_clip_val`:

1. Copy `configs/smoke_hie.yaml` to a scratch overlay.
2. Set `gradient_clip_val` to $10^{9}$ in the overlay, so the clip rescales no step.
3. Run the overlay.
4. Read `train/grad_norm` from the run's `metrics_history.csv`.
5. Record $q_{50}$, $q_{90}$, $q_{95}$, $q_{99}$ and the maximum.
6. Set the threshold to the smallest round value above $q_{99}$.
7. Write the value and its quantiles into the comment above the key in `configs/default.yaml`.

A threshold below $q_{99}$ rescales ordinary steps, so the run trains at a fraction of its
configured learning rate. A threshold far above the maximum leaves the non-finite guard as the only
protection.

`advanced_config.spike_breaker.additive_margin` is the active spike test: a batch is skipped when
`main_loss` exceeds the EMA plus the margin. Derive it from the post-warm-up epoch-to-epoch
movement of `main_loss` in the same kind of run. `ema_floor` ships at $10^{9}$, which switches off
the relative test $\ell > m \cdot \max(\mathrm{EMA}, \mathrm{floor})$. The loss is sign-indefinite,
and once the EMA is negative the relative test skips every positive batch.

## 15. Deliberate limitations

- **The nats are budget-local and model-local.** See §5.
- **No evaluation pipeline.** See §14. The `collect`, `metrics`, `spectra`, `oracle`, `events`,
  `coherence` and `samples` stages are structurally raw-signal. A feature-domain `events` or
  `coherence` stage is a new scientific construction, not a port. The `lean-limit:` marker for it
  is in this package's `__init__.py`.
- **No sweep configs ship.** `configs/` holds exactly `default.yaml`, `tiny.yaml` and
  `smoke_hie.yaml`, and `tests/test_config_load.py` checks that. An encoder or $\beta$ sweep needs
  scratch overlays.
- **No mixed precision.** `precision: "32-true"`. Mixed precision needs float32 islands around the
  log-variances, the closed-form KL and the Gaussian NLL reduction, and this model's reduction runs
  over $2340$ terms.
- **`compile: false`, but the key is live** (§7). Inductor may reassociate float arithmetic, and
  `pred_gap` is a small difference between two large block NLLs.
- **No warm start.** `core_model_checkpoint` stays `null`. A checkpoint from any sibling carries a
  different `model_class` stamp, and either different encoder tensors or a decoder head of a
  different width, so the guard refuses it.
- **`smoke_hie.yaml` is in-sample.** It uses `output/hie_cs.hdf5` for both splits, because
  `dataset_kwargs` is shared between the two loaders and cannot carry a per-split GUID filter.

> lean-limit: the four added readouts are training-path scalars with no uncertainty; replace with
> the evaluation's per-recording paired statistics when a feature-domain `eval` package exists.

## 16. Code notes

**Architecture and construction**

- **`SeqVaeLagAttnTrfRws` has a width hook.** It sets
  `self.decoder_out_channels = self._default_decoder_out_channels()` and builds the decoder at that
  width. The hook returns `raw_per_step` by default. It is called after the gates and before
  `initialization`, `init_depthwise_` and the calibration passes, so an override changes the width
  and not the RNG stream or the init order. No constructor keyword accompanies it.
- **`SeqVaeLagAttnTrfE2E` has no width hook.** That model derives from `nn.Module` directly and
  builds its own decoder inline for a raw target. Its tests construct `SeqVaeLagAttnTrfRws` as a
  reference, so a change to that constructor's RNG stream also shows in that suite.
- **The target-domain mixin also serves `SeqVaeLagAttnFs`**, whose class body is empty too:
  `class SeqVaeLagAttnFs(FeatureForecastTarget, SeqVaeLagAttnRws)`.
- **This package writes no encoder code and copies none.** `lag_attn_transformer_rws/nets/blocks.py`
  and `nets/encoders.py` contain no `raw`, `fhr` or `raw_per_step`. The seven block-level test
  modules of that parent — `test_attention_block`, `test_blocks`, `test_encoders`,
  `test_encoder_causality`, `test_rope`, `test_source_window`, `test_prefix_equivalence` — are not
  ported, because they test imported objects. The model-level causality tests are local (§12).

**Objective, metrics and tests**

- **Two tests build their comparison model from the other suite's keyword set.** The metric-set
  comparison and the foreign-checkpoint fixtures need a conv-LSTM model beside this one, and the two
  constructors' schemas differ, so each builds its comparison from that suite's own set.
- **No `test_lag_map.py` and no forward-contract module.** The $\sum_\ell \mathrm{map} = K_t$
  identity is latent-side and the decoder width does not touch it. The twenty-key set and every
  latent shape follow from the empty `vars(SeqVaeLagAttnTrfFs)`. The parent's suite tests both over
  this exact forward.

**Configuration and driver**

- **`raw_per_step` stays.** See §3.
- **`decoder_out_channels` is absent from the config and from `model_kwargs`**, and is not a
  keyword of this constructor. The stamped `target_keep_index` gives the width, and a second field
  could disagree with the gate.
- **`default.yaml` is written out in full**, with no `base:` chain, so a reader sees every setting
  it shares with the comparison models. `tests/test_config_load.py` holds three pins against drift:
  every leaf equals the conv-Transformer raw sibling's value outside declared exemptions; every
  comparable leaf equals the `lag_attn_fs` sibling's value outside declared exemptions; and the
  edges of the encoder-by-target square carry the same key sets.
- **The plotting config block keeps the shared driver's name.** See §7.
- **`smoke_hie.yaml` changes two schedule lengths.** `beta_schedule.warmup_epochs` is $20$, and
  `lr_warmup_steps` is $200$. At $339$ windows and batch $32$ an epoch is $11$ optimizer steps, so
  $200$ epochs is about $2{,}200$ steps, and the inherited $2{,}000$-step ramp would cover most of
  the run.
- **`tiny.yaml` shrinks the widths under two independent constraints.** The constructor validates
  `num_heads * d_head == d_model` for the **lag-attention** heads. Separately, `d_model /
  encoder_num_heads` must be even. Both head counts are $4$ in the shipped set, but the two
  constraints are independent.

## 17. Running it

Run from the repository root.

```bash
# Production, 7 ranks. TEB_RUN_STAMP is required so ranks 1..N-1 share rank 0's run directory,
# and the rank count must equal len(general_config.cuda_devices).
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_transformer_fs.trainer \
    --config teb_vae/lag_attn_transformer_fs/configs/default.yaml

# Local smoke: one epoch, one device, the committed four-sample shard.
python -m teb_vae.lag_attn_transformer_fs.trainer \
    --config teb_vae/lag_attn_transformer_fs/configs/tiny.yaml

# Dev-box run: the shipped geometry and reach budget over the committed HIE sample shard.
python -m teb_vae.lag_attn_transformer_fs.trainer \
    --config teb_vae/lag_attn_transformer_fs/configs/smoke_hie.yaml
```

`smoke_hie.yaml` needs `output/hie_cs_stats.hdf5`, generated from the shard by
`hdf5_dataset/calculate_dataset_stats.py` at `trim_minutes=1.0`.

`RUN_CONFIG` near the bottom of `trainer.py` names the config for a launch with no command line, so
the module runs from an IDE's Run button. A `--config` on the command line always wins. A relative
path resolves against the repository root, and the module changes the working directory to the
repository root before it runs. A Run-button launch of `default.yaml` is a *single* process; its
seven `cuda_devices` make the framework spawn DDP workers underneath it.

The test gate:

```bash
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_fs/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_fs/tests -q -m slow
```

There is no `eval` entry point for this package. See §14.

## 18. Configuration keys

The first list holds the keys this document depends on. Outside `VAE_model` it is not a complete
inventory of the framework's own settings.

**Required**

- `general_config.tag`
- `general_config.seed`
- `general_config.cuda_devices`
- `general_config.lr`
- `general_config.lr_milestone`
- `general_config.lr_warmup_steps`
- `general_config.epochs`
- `general_config.plot_frequency`
- `general_config.accumulate_grad_batches`
- `general_config.batch_size.train`
- `general_config.batch_size.test`
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
- `model_config.VAE_model.causal_reach_budget_s`
- `model_config.VAE_model.encoder_conv_kernels`
- `model_config.VAE_model.encoder_conv_dilations`
- `model_config.VAE_model.encoder_num_heads`
- `model_config.VAE_model.encoder_d_ff`
- `model_config.VAE_model.target_attention_blocks`
- `model_config.VAE_model.source_attention_blocks`
- `model_config.VAE_model.source_attention_window`
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
- `advanced_config.trainer.precision`
- `advanced_config.trainer.gradient_clip_val`
- `advanced_config.trainer.gradient_clip_algorithm`
- `advanced_config.trainer.compile`
- `advanced_config.trainer.log_every_n_steps`
- `advanced_config.trainer.num_sanity_val_steps`
- `advanced_config.trainer.use_distributed_sampler`
- `advanced_config.spike_breaker.enabled`
- `advanced_config.spike_breaker.ema_floor`
- `advanced_config.spike_breaker.additive_margin`
- `advanced_config.spike_breaker.comparison_metric`
- `advanced_config.spike_breaker.max_consecutive_skips`
- `advanced_config.tracking.mlflow.experiment_name`
- `advanced_config.tracking.mlflow.run_name`
- `advanced_config.tracking.mlflow.tags.variant`
- `advanced_config.callbacks.lag_attn_rws_plotting.enabled`
- `advanced_config.callbacks.lag_attn_rws_plotting.num_examples`
- `dataset_config.stat_path`
- `dataset_config.dataloader_config.normalize_fields`
- `dataset_config.dataloader_config.dataset_kwargs.trim_minutes`

**Deliberately absent**

These five keys describe the conv-LSTM encoder. None names an argument of this constructor, so the
signature sweep drops each one without a warning, and the config then builds a different model from
the one it appears to describe:

- `model_config.VAE_model.lstm_layers`
- `model_config.VAE_model.encoder_extra_dilations`
- `model_config.VAE_model.encoder_extra_kernel`
- `model_config.VAE_model.conv_norm_groups`
- `model_config.VAE_model.causal_norm`

The target domain derives or declares these two values elsewhere. A key for either would be a second
value that can disagree with the first:

- `model_config.VAE_model.decoder_out_channels`
- `model_config.VAE_model.target_block_split`

These keys are structural, derived or inert in this architecture. Each would look to a maintainer
like a control that exists:

- `model_config.VAE_model.forecast_channels`
- `model_config.VAE_model.encoder_head_width`
- `model_config.VAE_model.encoder_dropout`
- `model_config.VAE_model.layer_scale_init`
- `model_config.VAE_model.rope_base`
- `model_config.VAE_model.target_attention_window`
- `model_config.VAE_model.use_availability_embedding`
- `model_config.VAE_model.encoder_grad_checkpoint`
- `model_config.VAE_model.sigma_obs`
- `model_config.VAE_model.head_structured_latent`
- `model_config.VAE_model.freeze_unused_attn_proj`
- `advanced_config.callbacks.lag_attn_transformer_fs_plotting.enabled`
- `advanced_config.callbacks.lag_attn_rws_plotting.plot_frequency`

The encoder head width is `d_model // encoder_num_heads`. Encoder dropout is the existing `dropout`.
The LayerScale initialisation and the rotary base are constructor defaults. The target attention
window is the full causal prefix. `forecast_channels` was removed from the feature-domain family
because inherited indices change meaning when a stored block changes width. The plot cadence is
`general_config.plot_frequency`, the same key that drives the loss curves, so the figure and the
curves exist for the same epochs.
