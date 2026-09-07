# `lag_attn_transformer_fs` — the as-built design record

The conv-Transformer feature-domain lag-attention VAE-TEB: what it is, which parent supplies which
half, what it consumes, what it returns, what it optimises, and every place the built package
differs from the design it was built from.

Companion documents, none of them restated here:
`teb_vae/lag_attn_transformer_rws/DESIGN.md` records the encoders, the adapters, the attention
blocks and the wiring this model imports **by import and unchanged**;
`teb_vae/lag_attn_fs/DESIGN.md` records the feature target, the smear argument and the four added
readouts; `teb_vae/lag_attn_rws/DESIGN.md` records the architecture all four models share, and
`teb_vae/lag_attn_rws/model_explained.md` the latent factorisation. `RESULTS.md` in this directory
carries the measurements.

**What this document is for.** Reading it should leave a reader able to say what a reported number
of this model means, what it does not, and which of two comparisons a given number belongs to —
which is a different question from how the architecture works, and the one a model assembled
entirely out of two parents makes easy to get wrong.

---

## 1. What the model is

The fourth cell of an encoder-by-target grid, and the one that closes it:

```
                        conv-LSTM encoders          conv-Transformer encoders
  raw FHR target        lag_attn_rws                lag_attn_transformer_rws
  feature target        lag_attn_fs                 lag_attn_transformer_fs   <- this
```

At every 4-second anchor $t$ the model forecasts the next **two minutes of the stored FHR feature
future** — $H \cdot C_{\mathrm{keep}} = 30 \times 78 = 2340$ coefficients at the shipped reach
budget — twice: once from a target-only latent and once from a source-conditioned one, through one
shared decoder invoked twice under one noise draw. The gap between the two forecasts, and the KL
between the two latents resolved across lags, are the coupling readout. That is the
`lag_attn_fs` target reached through the `lag_attn_transformer_rws` encoders, and nothing else.

The whole model is

```python
class SeqVaeLagAttnTrfFs(FeatureForecastTarget, SeqVaeLagAttnTrfRws): ...
```

with **an empty class body** — `vars(SeqVaeLagAttnTrfFs)` carries no callable and no class
constant — linearising as
`SeqVaeLagAttnTrfFs -> FeatureForecastTarget -> SeqVaeLagAttnTrfRws -> Module -> object`.

| Member of `FeatureForecastTarget` | What it is |
| --- | --- |
| `TARGET_BLOCK_SPLIT` | where the two stored blocks meet, for §14's block split only |
| `_default_decoder_out_channels` | names the decoder's width; builds nothing |
| `_build_forecast_target` | gathers the surviving channels and unfolds each anchor's future |
| `_resolved_forecast_gaps` | the four added readouts of §14 |
| `compute_loss` | builds the target, delegates, merges those four |

**Why the empty body is the point rather than an economy.** With nothing defined here, the twenty
forward keys, the absent `decoder_state` and `delta_mu_src`, every latent shape, the
head-structured posterior, the lag map and the objective's metric set cannot have moved: they are
the parents' own code objects, pinned by the parents' own suites over the same functions. So a
difference in results against `lag_attn_fs` is attributable to the encoder alone, and a difference
against `lag_attn_transformer_rws` to the target domain alone. That is what neither of the two
three-cell configurations allowed, and it is the entire value of this cell.

**It is an experiment, not a remedy.** `lag_attn_rws/PRIOR_SCALE_AND_PREDICTIVE_GAP.md` §5
establishes that the raw models' held-out predictive gain is negative because the source pathway
does not generalise — a failure that lives in the source encoder, the lag attention and the
posterior fusion, none of which an encoder swap of this kind touches. This model is expected to
reproduce it. Its value is that it removes a confound, and it must not be read as a fix for one.

At the shipped configuration the model holds **5,034,984 parameters**, against **5,126,326** for
the encoder-axis comparison and **5,003,116** for the target-axis one, at the same reach budget.
§13 carries the arithmetic and both deltas.

## 2. Input contract

Read from the HDF5 through `train/data_module.py::GraphDataModule`, at `trim_minutes: 1.0`.
Identical to every other cell of the grid; what the target domain changes is which of these fields
is the reconstruction target, and what the encoder change touches is nothing here at all.

| Field | Shape | Role |
| --- | --- | --- |
| `fhr_st`, `fhr_ph` | $(B, 300, 43)$, $(B, 300, 66)$ | target stream, concatenated to $(B, 300, 109)$ — **both the network input and the reconstruction target** |
| `up_st`, `up_ph` | $(B, 300, 43)$, $(B, 300, 15)$ | source stream, concatenated to $(B, 300, 58)$ |
| `weight` | $(B, 300)$ | per-step validity on the decimated grid |
| `fhr`, `up` | $(B, 4800)$ | raw traces; **not read by the model** — row 1 of the diagnostic page only |
| `guid` | — | figure titles and run provenance |

**`fhr_st` and `fhr_ph` must be in `normalize_fields`.** They are this model's reconstruction
target, so an unnormalised block makes the Gaussian NLL meaningless against a unit-scale variance
model, with the loader raising nothing. The shared entry point refuses a config omitting either,
field by field, from `LagAttnTrfFsTrainer.TARGET_FIELDS` — and that guard reads the driver it was
handed, so it reaches this package only because `TARGET_FIELDS` resolves to the **feature** parent
through the diamond of §7. A resolution the other way would leave it checking the raw model's
`fhr`, which these configs satisfy.

**`fhr_up_ph` is absent from every config and must stay absent.** A coefficient mixing both signals
would break the target-only / source-conditioned separation the whole design rests on — and here it
would additionally put the source's own signal into the forecast target.

## 3. Geometry

`TrimmedRawGeometry` is reused **unchanged**, from the same module both parents use, with no second
geometry type and no shared protocol:

$$T = 300, \qquad H = 30, \qquad T_{\mathrm{valid}} = T - H = 270, \qquad \text{warm-up} = 30.$$

Only `t`, `t_valid`, `horizon` and `warmup` are consulted by the target builder and the masks;
`r`, `n_raw()` and `future_block_start()` are never called by this model.

The raw grid stays a true fact about the batch — the loader delivers `fhr` at $4800$ and `weight`
at $300$, and `raw_masks._validate_weight` checks `weight.size(1) == geometry.t` — so
`raw_per_step` remains a required geometry input. It simply stops being the decoder width.
Deleting it would break the geometry rather than narrow the decoder, and setting it would not
widen the decoder by one channel.

`forecast_mask`, `contributing_anchors` and `kl_mask` are reused **unchanged** and broadcast over
the channel axis, producing $(B, 270, 30)$ and $(B, 300)$.

## 4. Forward return dict

`SeqVaeLagAttnTrfFs.forward(y_st, y_ph, u_stream)` returns twenty keys — the conv-Transformer
parent's exactly, which are in turn the conv-LSTM family's exactly. **Only the four forecast
tensors change shape**, and that single last axis is the entire difference this class introduces to
the forward.

- `mu_base`, `logvar_base`, `mu_full`, `logvar_full` — $(B, 270, 30, C_{\mathrm{keep}})$, so
  $(B, 270, 30, 78)$ at the shipped budget and $(B, 270, 30, 109)$ ungated, against the raw
  variant's $(B, 270, 30, 16)$.
- `mu_prior`, `logvar_prior`, `raw_logvar_prior`, `mu_post`, `logvar_post`, `z_prior`, `z_post` —
  $(B, 300, 64)$, unchanged.
- `target_state`, `source_state` — $(B, 300, 128)$; `attended_source_heads` $(B, 300, 4, 32)$;
  `attn_weights` $(B, 300, 4, 91)$ — unchanged.
- `kld_per_t`, `kld_per_t_per_head`, `source_kl_lag_map` — unchanged.
- `mu_prior_sat_frac`, `delta_mu_sat_frac` — unchanged.

**No `decoder_state` and no `delta_mu_src`**, as in both parents: there is no bypass to carry one.

**`future_index` is inherited and present**, and that is worth stating because it reads as a
raw-target artefact. The conv-Transformer constructor registers it and a subclass could only drop
it by overriding `__init__`, which the width hook exists to avoid. It is non-persistent, so it
reaches no checkpoint, and it is simply never read.

## 5. Loss, and what its nats are summed over

$$\mathcal{L} = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0
+ \beta(e)\,\mathrm{KL}_{\mathrm{train}} + \beta_p\,R_p
+ \lambda_{\mathrm{ms}} \mathcal{L}_{\mathrm{ms}}
+ \lambda_{\mathrm{deriv}} \mathcal{L}_{\mathrm{deriv}}
+ \lambda_{\mathrm{boundary}} \mathcal{L}_{\mathrm{boundary}}$$

in nats per anchor, computed by `lag_attn_rws/nets/losses.py` — **the same code, not a copy of
it**, reached through the mixin's `compute_loss`. That module is domain-neutral: it reduces a
$(B, T_{\mathrm{valid}}, H, X)$ block against a $(B, T_{\mathrm{valid}}, H)$ mask and takes both
the target and $X$ as arguments. This model supplies its gathered target and its block width and
delegates. Four models keep a thin `compute_loss` of this shape, and four copies of an objective
that must never diverge is exactly the failure the free-function structure exists to prevent.

$R_p$ is the prior's scale rate,
$\mathrm{KL}\!\left(\mathcal N(\mu^p, \operatorname{diag} e^{\ell^p}) \Vert \mathcal N(\mu^p,
I)\right)$, reduced on the KL's own anchor support and in the same units. Nothing else in the
objective penalises a *narrow* prior.

**The three shape weights ship at $0.0$ here, as they do in the feature-target sibling, and for the
same domain reason.** $\mathcal{L}_{\mathrm{ms}}$, $\mathcal{L}_{\mathrm{deriv}}$ and
$\mathcal{L}_{\mathrm{boundary}}$ price the envelope, slope and opening level of a raw *waveform*.
This model's block axis is $C_{\mathrm{keep}} = 78$ surviving wavelet channels: "adjacent" there
means "the next filter", pooling it mixes unrelated scales, and there is no last-observed sample for
a boundary to be continuous with. The keys are carried rather than omitted so the task plumbing and
the parity comparison against the raw sibling stay uniform, and a term at weight $0.0$ is **not
computed** — its metric is exact $0.0$, so the three columns are honest zeros rather than raw-domain
formulas evaluated over a channel axis. The three exemptions in the parity table say so.

**`block_width` is $C_{\mathrm{keep}}$, not `geometry.r`, and this is the sharpest silent trap in
the module.** It feeds only `mean_logvar_full`, `mean_logvar_base`, `logvar_full_floor_frac` and
`logvar_full_ceil_frac` — **not the loss**. Passing the raw grid's $R$ changes no loss, fails no
shape check, and rescales by $4.9\times$ exactly the four diagnostics `logvar_clamp` is re-derived
from. `tests/test_objective.py` pins `mean_logvar_full` against a hand computation for this reason
alone.

### What the nats are, and are not, comparable to

The reconstruction is summed over $H \cdot C_{\mathrm{keep}} = 2340$ coefficients against the raw
models' $H \cdot R = 480$ samples, and the factorised Gaussian over correlated wavelet coefficients
overcounts independent information in both. So:

- **Not comparable to the raw models'.** A nat here and a nat in either raw-target cell are sums
  over different blocks of differently-correlated quantities. The encoder axis is readable in nats
  only against `lag_attn_fs`; against `lag_attn_transformer_rws` only the parameter budget, the
  throughput, the bottleneck-health columns and the *ordering* of arms carry across.
- **Not comparable across reach budgets within this model.** `causal_reach_budget_s` moves
  $C_{\mathrm{keep}}$, hence the decoder width, hence the block every nat is summed over. Two arms
  of this model at different budgets have non-comparable `pred_gap` and **mutually unloadable
  checkpoints** — and the class guard cannot separate them, because both stamp
  `SeqVaeLagAttnTrfFs`. Only the width the stamped `target_keep_index` implies does.

This is recorded, not fixed. Arms of this model at this budget rank against each other; nothing
else.

## 6. The mixin, and why it is a move rather than an abstraction

**The obvious construction does not work, and the failure is silent.** `SeqVaeLagAttnFs` subclasses
`SeqVaeLagAttnRws`, while `SeqVaeLagAttnTrfRws` derives from `nn.Module` directly. So
`class SeqVaeLagAttnTrfFs(SeqVaeLagAttnFs, SeqVaeLagAttnTrfRws)` linearises as
`TrfFs -> Fs -> Rws -> TrfRws -> Module` and so **runs the conv-LSTM constructor** — a model that
builds, trains and reports, and is not this architecture. That was measured against the tree rather
than inferred.

So the five members that make a model a feature forecaster live in
`teb_vae/lag_attn_fs/nets/feature_target.py`, and both feature models are their own architecture
plus that. **None of the five mentions an encoder**: each reads only `target_gate`, `c_y`,
`geometry`, `horizon`, `coverage_floor`, `logvar_clamp` and `decoder_out_channels`, all of which
both constructors set before building the decoder.

It is a **move, not an abstraction**: a plain object deriving from `object`, with no `__init__`, no
`Protocol`, no `__init_subclass__` and no member that was not lifted verbatim. What it prevents is
a second copy of ≈230 lines whose drift would be silent — the delay trap of §8 and the
`block_width` trap of §5 both change no shape and raise nothing.

**The order of the bases is load-bearing.** The mixin comes first, which is what makes its width
hook win method resolution over the conv-Transformer parent's `raw_per_step` one; reversed, the
decoder would be built at $R = 16$ and a $C_{\mathrm{keep}}$-wide feature block scored against it,
and since the objective takes the block width as an argument, nothing would raise.

**The width is a method, not a constructor keyword.** `_default_decoder_out_channels` overrides a
hook, and the hook had to be *added* to the conv-Transformer parent, which built its decoder at
`out_channels=self.raw_per_step` inline — §16. Two reasons, each sufficient: the gate it reads is
built *by* that constructor, so nothing outside can compute the value beforehand; and a subclass
narrowing `__init__` to intercept a keyword breaks the `inspect.signature` sweep in
`trainer._build_model_kwargs`, which then forwards **no configuration at all** and silently builds
an all-defaults model. The constructor signature here is therefore the conv-Transformer parent's,
keyword for keyword, asserted directly.

One consequence a reader will meet: a static type checker cannot see that the mixin's seven read
attributes are set by a *base* constructor, and reports them unresolved in
`lag_attn_fs/nets/feature_target.py`. Declaring them would mean the `Protocol` the move
deliberately does not have, so the contract is stated in that class's docstring instead. There is
no repository type-check gate.

## 7. The two diamonds, and their measured resolution order

```python
class SeqVaeLagAttnTrfFsTask(SeqVaeLagAttnFsTask, SeqVaeLagAttnTrfRwsTask):
    pass

class LagAttnTrfFsTrainer(LagAttnFsTrainer, LagAttnTrfRwsTrainer):
    MODEL_CLS = SeqVaeLagAttnTrfFs
    TASK_CLS = SeqVaeLagAttnTrfFsTask
    CHECKPOINT_STEM = "lag-attn-trf-fs"
```

The task defines **zero** callables. The driver re-points three class attributes and defines no
method. Both linearisations are asserted as lists of class names in `tests/test_task.py` and
`tests/test_trainer.py`:

`SeqVaeLagAttnTrfFsTask -> SeqVaeLagAttnFsTask -> SeqVaeLagAttnTrfRwsTask -> SeqVaeLagAttnRwsTask
-> LightningModelBase`, and
`LagAttnTrfFsTrainer -> LagAttnFsTrainer -> LagAttnTrfRwsTrainer -> LagAttnRwsTrainer ->
GraphModelBase`.

**Both diamonds are well-formed because both parents descend from a common base, not because their
overrides are disjoint.** The task's happen to be — $\{$`_build_raw_target`, `forecast_rows`$\}$
against $\{$`build_lr_scheduler`$\}$ — but that is a fact about today's code rather than a property
of the construction, so each of the three behaviours is asserted against the class named here. The
driver's are **not** disjoint: both parents set `MODEL_CLS`, `TASK_CLS` and `CHECKPOINT_STEM`, and
resolution order alone would take the feature side. All three are therefore re-pointed, and each
failure it prevents is silent:

- omit `MODEL_CLS` and the driver builds `SeqVaeLagAttnFs`, a conv-LSTM model, with no error
  anywhere — a run that looks like this package and is not;
- omit `TASK_CLS` and the same, one layer up;
- omit `CHECKPOINT_STEM` and it writes `lag-attn-fs-*.ckpt`, interleaving two models' checkpoints
  in whichever output tree they share.

Everything else arrives by resolution order, across both layers at once — the task's members and
the driver's — and where each comes from is a decision rather than an accident:

| From the **feature** parent | From the **conv-Transformer** parent | From the shared driver |
| --- | --- | --- |
| `TARGET_FIELDS = ("fhr_st", "fhr_ph")` | `compile_model_requested` | `PLOT_CONFIG_KEY` |
| `TRACKED_METRICS`, 78 entries | `_build_model_kwargs`, `create_model` | `preflight` |
| `_build_raw_target`, `forecast_rows` | `_build_trainer_kwargs`, `build_lr_scheduler` | the DDP strategy selection |

**`compile_model_requested` resolves to the conv-Transformer side, and that is a decision.**
`LagAttnFsTrainer` does not define it, so lookup passes through to `LagAttnTrfRwsTrainer`, which
flips the shared driver's hard refusal to live: `torch.compile` becomes permitted on a model whose
feature-domain ancestor never exercised it. That is the right outcome — it is the transformer
encoder that makes compilation worth having, and the LSTM that defeated inductor is gone — but it
arrives by resolution order rather than by anything written down, so `tests/test_trainer.py`
asserts it explicitly, together with the guard refusing `compile` and `attention_grad_checkpoint`
in the same config. Shipped configs keep `compile: false` regardless, for the numerical reason in
§15.

**`PLOT_CONFIG_KEY` stays `"lag_attn_rws_plotting"`**, and the config block keeps that name. The
shared callback assembly reads the literal, so a sibling that renames it to match its own package
gets no figure, no error and nothing in the log saying why.

## 8. The target is gathered, never delayed

The input `ChannelGate` applies **two** operations: a gather of the channels surviving the reach
budget, and a per-channel delay $\delta_c = \lceil \rho_c / \Delta \rceil$ that pushes each
channel's forward reach behind the anchor's causal endpoint.

**The target takes the gather and not the delay.** Delaying it would silently ask anchor $t$ to
forecast the future of anchor $t - \delta_c$, per channel, and nothing downstream would fail: every
shape is identical. `ChannelGate.forward` is `self.delay(index_select(x, -1, self.keep_index))` and
offers no gather-only method, so the target builder calls `index_select` directly.

**The gate is built at *this* architecture's own construction site**, so that the mixin reaches an
un-delayed keep-index here is a fact this package owns rather than one it inherits.
`tests/test_objective.py` asserts the index identity
$Y^{+}[b,t,\tau,k] = Y[b,\,t+1+\tau,\,\mathrm{keep}[k]]$ against a **hand-written slice** rather
than against the shared `figure_primitives.future_target` helper: a target that wrongly applied the
delay and a reference that wrongly applied it would agree.

At the shipped budget **all 78 surviving channels carry a non-zero delay** — one step at the
fastest, thirty at the slowest — so a gate-built target would be wrong in *every* channel it
contains, not in some of them. That is what makes the negative test specific rather than "not
equal".

The gather runs **before** the unfold. The two commute; doing it first keeps the copy at
$(B, T, C_{\mathrm{keep}})$ rather than $(B, T_{\mathrm{valid}}, H, C_{\mathrm{keep}})$ — a factor
of $H$, and a third of a gigabyte at the production batch.

At `causal_reach_budget_s: null` the survivor set is all $109$ channels and the decoder width
follows, so the unguarded arm is well defined and its target is the ungathered stream.

### The target is also smeared, and that is not a leak

A stored coefficient at decimated step $s$ is a weighted average of raw signal over a window
*centred* at raw index $16s$, so a share of the short-horizon target is a deterministic function of
signal the model has legitimately already observed. **The argument, the blend fraction and the
measured table are `lag_attn_fs/DESIGN.md` §8 and are not restated here**, because it is a property
of the target and the filter bank and is therefore **unaffected by the encoder**: the two
feature-domain models blend identically, and `lag_attn_fs/tests/test_smear.py` recomputes the
figures from the shipped filter bank for both. What it affects is optimisation and interpretation,
not causality — and §14 is the readout that separates forecasting from reconstruction of the
already-determined component.

## 9. Structural constraints that are not preferences

The properties every reported nat rests on, re-asserted against *this* class rather than assumed to
have survived the composition. Where an invariant is the parents' own, this package's suite
imports and re-parametrises the sibling module that owns it rather than restating the assertions.

| Property | What enforces it | Test |
| --- | --- | --- |
| **No decoder bypass** — gradient reaches the decoder only through $z$ | `BaselineFutureDecoder.forward` takes exactly one tensor, at $d_z$ in-features; no `decoder_state` head, no second decoder | `tests/test_invariants.py` |
| **Source purity** — the prior never sees the source, the source state never sees the target | separate gates, adapters and encoders; the posterior is a residual on the prior | `tests/test_invariants.py` |
| **Exact zero KL at initialisation** | posterior deltas zeroed **after** the generic init; one shared $\epsilon$ | `tests/test_invariants.py` |
| **One decoder, invoked twice** | the same module object on $z^p$ and $z^q$ | `tests/test_invariants.py` |
| **Token causality**, unconditionally | §12 | `tests/test_causality.py` |
| **The lag attribution identity**, $\sum_\ell \widetilde K_{t,\ell} = K_t$ | the lag attention is built at `dropout=0.0` | the conv-Transformer parent's suite, over this same forward |
| **`lag_attn.W_o` frozen** | the head-structured posterior consumes the per-head summaries, so `W_o` receives no gradient | `tests/test_construct.py` |
| **No recurrence and no time-pooling normaliser** on a history path | none is constructed; the surviving `GroupNorm`s are enumerated and each asserted to be under `horizon_core.` | `tests/test_construct.py` |

**The zero-KL claim states its fixture's flags**, because it is conditional. It holds under the
conv-Transformer suite's tiny keyword set, which sets none of them. The shipped config ships
`base_decode: mean` — under which the two *forecasts* are no longer bitwise identical, though the
KL is still exactly zero, since that depends on the two distributions and not on the samples drawn
from them — and `posterior_logvar_mode: independent`, under which the init KL is zero only with
`head_init_calibration: true`. Any arm setting that flag false inherits a broken zero-KL start.

Because the delta heads are zero-initialised, **any KL assertion on a freshly constructed model
passes vacuously**; the `perturb_posterior` fixture is load-bearing for every test in this package
that claims to check KL behaviour.

**The decoder subtree holds no dropout *module***, which is a stronger statement than $p = 0$: the
projection MLP builds its `nn.Dropout` layers only at a positive rate, so at the decoder's
hard-coded $0$ there is nothing there for a later `model.train()` or a config rate to re-enable.
Asserted as the absence, on a model built at $0.1$ everywhere else so the absence is a choice.

## 10. Initialisation order

Load-bearing, top to bottom, in the conv-Transformer parent's `__init__` — and reached unchanged
here, which is the whole safety argument for the width seam of §16:

1. the horizon core, then `self.decoder_out_channels = self._default_decoder_out_channels()`, then
   the decoder at that width. **This is the only step the mixin reaches**, and it happens *before*
   everything below it.
2. `initialization(self)` — the shared generic pass, Xavier over every `nn.Linear` and `nn.Conv1d`.
3. `init_depthwise_(self)` — immediately after, never before. Xavier on a $(C, 1, k)$ depthwise
   weight reads $\mathrm{fan\_in} = k$ against $\mathrm{fan\_out} = Ck$, a factor
   $\sqrt{(1 + C)/2} = 8.03$ too small at $C = 128$ and independent of $k$. The model records
   `n_depthwise_init`, the count the pass returns, so a test can tell "the correction ran" from
   "the correction was a no-op".
4. `_zero_init_delta_heads()` — the generic pass would otherwise refill the posterior delta heads
   and destroy the exact zero-KL start.
5. `_zero_init_film_generators()` — the horizon core zero-initialises them itself and the generic
   pass refills them, so re-zeroing here is what makes the identity-at-init actually true.
6. The three zero-parameter policies — `horizon_embed_std`, `head_init_calibration`, `a_head_gain`
   — each applied only when its config value leaves the constructor default, so a default-flag
   model is bitwise the pre-bundle one.

**`head_init_calibration` centres the log-variance head at $\sigma = 1$ across $78$ output
channels, not $16$.** That is the one initialisation policy the target domain's width change
actually reaches, and it is asserted at the wide head rather than assumed:
`tests/test_invariants.py` subclasses the width hook and checks the calibration ran on the *wide*
head while
`n_depthwise_init` held — which dates the decoder's construction against the init block rather than
merely describing it. The FiLM re-zeroing and the still-zero delta heads are asserted alongside.

## 11. DDP reachability, and the two parameters it governs

Production runs under plain `"ddp"` with `find_unused_parameters=False` under `gaussian_nll`, so
every parameter must be reachable; `mse` starves the decoder's log-variance heads and selects
`ddp_find_unused_parameters_true`. `tests/test_ddp_strategy.py` measures both, and measures that
under `mse` the starved set is **exactly** those heads, so the fallback is justified rather than
assumed.

A parameter multiplied by an identically-zero tensor **is** reachable: its `AccumulateGrad` node
fires and DDP's reducer marks it ready. What breaks `find_unused_parameters=False` is a parameter
left *out of the graph* by a Python-level branch on a tensor value. So the availability terms are
added unconditionally in the forward and the branching happens at construction time only, with the
two terms carrying **different** conditions because they become non-trivial at different points:
$W_m$ exists when $\max_c \delta_c > 0$; $e_{\mathrm{start}}$ exists when $\min_c \delta_c > 0$,
since the indicator is non-zero for some $t$ exactly when *every* channel is delayed.

**The tensor-branch AST walk is deliberately not ported, and the premise that makes that sound is
asserted instead.** Every `forward` that executes here belongs to a module this package imports —
the encoders and blocks from the conv-Transformer parent, the `AvailabilityInputAdapter` from the
shared net layer — and both are walked where they live. `tests/test_ddp_reachability.py` asserts
that this package's `nets/`, and the mixin, define **no `forward` at all**, and that the sibling
module carrying the walk still exists and still reaches the shared net layer.

The reachability probe's guarded arm resolves the **production** budget at `sequence_length: 64`,
`warmup_period: 30` rather than using the hand-made tiny guard, because that guard's delay tuple
starts at $0$ and so builds no start embedding: a probe on it would assert two availability
parameters where only one exists. It is also the one arm where the adapters and a $78$-wide output
head are exercised together.

**`broadcast_buffers=False`** is justified as it always was: every buffer — rotary tables, causal
masks, the gate's keep-index and delays, the adapters' availability patterns, the raw-target index
grid — is a deterministic function of the config, and there is no `BatchNorm` anywhere. The
`future_index` buffer is present and never read (§4), which only makes the broadcast more wasteful.
`static_graph` is deliberately absent: the loss-spike breaker substitutes a zero-weighted sum over
every parameter on a skipped batch, which is a structurally different backward from the one the
first iteration recorded.

## 12. Causality, unconditionally

**Step-wise causality holds through the whole model with no flag to qualify it**, and that is the
one claim of this package that is genuinely stronger than the feature-domain sibling's rather than
merely inherited. `lag_attn_fs/DESIGN.md` §6 records that the conv-LSTM cell's step-wise causality
holds **only under `causal_norm: true`** — without it a time-pooling normaliser mixes the whole
sequence. These encoders have no such normaliser: `RMSNorm` reduces over channels only,
convolutions pad left before a `padding == 0` `Conv1d`, and attention is causal by kernel flag
(target, full prefix) or by an explicit band mask (source, $W_U = 16$). **`causal_norm` is not a
constructor keyword of this model at all**, so there is no flag to condition the claim on, and a
reader should not go looking for the one the sibling's copy names.

`tests/test_causality.py` measures it through the assembled model — not at the encoder level, which
is where the parent's own seven block-level modules measure it — at the shipped reach budget **and
at the tiny fixture**, which is the half that fails on the conv-LSTM sibling without the flag.
Prefix equivalence, $\mathcal{E}(X_{0:T-1})_t = \mathcal{E}(X_{0:t})_t$, is measured there too.

**Prefix equivalence is a claim about the latent and the mean-decoded branch, not about a draw.**
`randn_like` fills row-major over $(B, T, d_z)$, so sample $b > 0$'s $\epsilon$ at anchor $t$ sits
at offset $bTd_z + td_z$ and moves with $T$. The base forecast is therefore asserted under the
shipped `base_decode: mean`, and a second test locates the difference in the draw rather than in
the positions — prefix-equivalent in batch element $0$, not in element $1$, while `mu_prior` is
prefix-equivalent in every element. That is a harness property, not a model one.

**Token causality is not raw-signal causality**, and this model delivers and reports the first.
$H_t = f(X_{\le t})$ is what the architecture guarantees. Raw-signal causality would need genuinely
one-sided feature transforms; the stored features are two-sided, and `causal_reach_budget_s` bounds
the leak by pruning channels whose analytic reach exceeds the budget and delaying the survivors
rather than eliminating it, because $L_{95}$ is an energy *quantile*. The KL is
`source_conditioned_kl_raw` / `_train` and its decomposition is `source_kl_lag_map`; it is **not**
called transfer entropy, for that reason.

> lean-limit: token-causal features; replace with genuinely one-sided scattering and phase-harmonic
> transforms when a causal front end exists and the stored features are regenerated under it.

## 13. Parameter budget

Measured on constructed models at the shipped $120$ s reach budget, not predicted.
`tests/test_docs.py` re-measures every total below by constructing the models rather than comparing
against literals, so a legitimate change to a shared imported component re-costs this table instead
of failing an unrelated assertion.

| | conv-LSTM encoders | conv-Transformer encoders |
| --- | ---: | ---: |
| raw target, unguarded | $5{,}088{,}186$ | $4{,}996{,}844$ |
| raw target, guarded ($C = 16$) | $5{,}094{,}458$ | $5{,}003{,}116$ |
| **feature target, guarded** ($C_{\mathrm{keep}} = 78$) | $5{,}126{,}326$ | $\mathbf{5{,}034{,}984}$ |
| feature target, unguarded ($C = 109$) | $5{,}135{,}988$ | $5{,}044{,}646$ |

Two deltas, and both decompose cleanly because every module outside the encoders is shared.

**The encoder axis: $-91{,}342$, a $1.8\%$ reduction** on the feature-domain sibling, guarded
against guarded. It is **identical** to the reduction the same two encoders buy in the raw domain
at the same budget, which is what a difference living entirely in the two history encoders must
look like. That reduction was $1{,}304{,}782$ — $38.4\%$ — before the capacity revision, which
raised the conv-Transformer encoders (six target blocks at $d_{\mathrm{ff}} = 512$ against four at
$256$) and left the conv-LSTM ones alone. The two encoder families are now near parity in budget,
which makes the encoder axis a comparison of *structure* rather than of size.

**The target axis: $+31{,}868 = 514 \times (78 - 16)$**, the decoder's output head and nothing
else: two per-channel output rows plus their biases, at $d_{\mathrm{hidden}} + 1 = 257$ each.
Ungated it is $+47{,}802 = 514 \times (109 - 16)$. Identical to the conv-LSTM pair's delta, because
both pairs build the same `BaselineFutureDecoder`.

**The guarded delta is $+6{,}272$ in both encoder families, and the conv-Transformer design
record's $13{,}696$ is the same measurement of a different quantity.** Stated here so the next
reader does not treat two correct numbers as a contradiction, and **no edit to that document is
needed**. Its §5 describes the availability *projections* alone: `target_adapter.mask_proj.weight`
is $(128, 78)$ and the source's $(128, 29)$, so $128 \times (78 + 29) = 13{,}696$. The net
constructor delta is that, plus $256$ for the two `start_embed` vectors, minus $7{,}680$ because
the input linears **narrow** from $109$ and $58$ channels to $78$ and $29$ — $128 \times 60$. The
three terms give $6{,}272$.

## 14. Observability while the evaluation is deferred

There is **no `eval/` package for this model** — no `ModelBinding`, no verdicts, no bootstrap
confidence intervals, no trivial-predictor baselines, no calibration, no per-recording tables.
Deferred whole, and the consequence is structural: every readout is a scalar read from a run's own
`train_results/metrics_history.csv`, and no reported difference carries an uncertainty.

That leaves the problem this target domain creates and the raw cells do not have. A scalar summed
over $2340$ coefficients cannot separate a model forecasting three easy channels well from one that
is uniformly mediocre — and, given §8, cannot separate forecasting from reconstruction of the
already-determined component. Four metrics close it, inherited through the mixin and reported
through the feature parent's `TRACKED_METRICS`:

| Metric | What it separates |
| --- | --- |
| `pred_gap_tau_first` | the horizon step whose target is half-determined by observed history |
| `pred_gap_tau_last` | the step whose target is not determined by it at all |
| `pred_gap_st`, `pred_gap_ph` | the two stored blocks, whose filters have different reaches |

**All four are partial sums of the `pred_gap` beside them**, over the same denominator, and both
splits recompose to it. That is the only property that makes them worth reporting, so the
per-element term is the objective's own `raw_sample_score` and the mask is rebuilt through the
objective's own two functions rather than restated. No metric is added by this package: an encoder
swap raises no question the feature domain had not already raised.

The block split needs `TARGET_BLOCK_SPLIT = 43`, which **cannot be derived** — $c_y$ is the two
blocks' *sum*. It is a class attribute rather than a constructor keyword because a key that changed
only which of two diagnostics a coefficient was counted in would look like an architecture decision
in every checkpoint. Nothing else depends on it, so a stale value would mislabel two reported
columns and break nothing; the task, which is the only layer that sees the two blocks separately,
checks it against the data it assembles the target from.

**The diagnostic page is inherited whole**, through the task's `forecast_rows` property and the
shared model-agnostic callback. This package ships no `plotting.py` and no `sample_page.py`, which
`tests/test_sample_page.py` asserts as a directory check — near-vacuous the day it was written, and
the thing that fails when someone later reaches for a local copy.

## 15. Deliberate limitations

- **The nats are budget-local and model-local.** §5. Recorded, not fixed.
- **No evaluation pipeline.** §14. Deferred for the reason the feature-domain sibling defers it:
  `collect`, `metrics`, `spectra`, `oracle`, `events`, `coherence` and `samples` are structurally
  raw-signal, and a feature-domain `events` or `coherence` is a new scientific construction rather
  than a port. Inventing one to fill a column is worse than an absent column. The module-level
  `lean-limit:` marker for it lives in this package's `__init__.py`; this section is its record.
- **No encoder arms and no $\beta$ arms ship.** `configs/` holds exactly `default.yaml`,
  `tiny.yaml` and `smoke_hie.yaml`, linted by `tests/test_config_load.py`. The encoder values were
  swept in the raw domain and the encoders are reached by import; the block cardinality is
  unchanged at $2340$, so the feature domain's four-arm $\beta$ sweep answered that axis. Re-opening
  either means a scratch overlay, deliberately.
- **No mixed precision.** `precision: "32-true"`. Mixed precision needs float32 islands around the
  log-variances, the closed-form KL and the Gaussian NLL reduction — and this model's reduction is
  over $2340$ terms, the largest in the family. Any of those moving is a difference the two
  comparisons would attribute to an encoder or to a target domain.
- **`compile: false`, but the key is live rather than inert** (§7). It ships off for a numerical
  reason: inductor may reassociate float arithmetic, and `pred_gap` is a difference of order
  $10^{-1}$ between two block NLLs of order $10^{3}$. Adopting it means one run each way from the
  same seed, comparing `pred_gap` on a fixed batch.
- **No warm start.** `core_model_checkpoint` stays `null`: a blob from any sibling carries a
  different `model_class` stamp and either different encoder tensors or a decoder head of a
  different width, and the guard refuses it — correctly.
- **Every local measurement is in-sample.** `dataset_kwargs` is shared between the two loaders and
  cannot carry a per-split GUID filter, so the dev-box runs in `RESULTS.md` validate the
  objective's optimisation behaviour and say nothing about generalisation.

> lean-limit: the four added readouts are training-path scalars with no uncertainty; replace with
> the evaluation's per-recording paired statistics when a feature-domain `eval` package exists.

## 16. Deviation record

Where the built package differs from the design it was built from, and why.

**Architecture and construction**

- **A width seam was added to a *shipped* sibling's net.** `SeqVaeLagAttnTrfRws` built its decoder
  with `out_channels=self.raw_per_step` inline and had no width hook at all; it now sets
  `self.decoder_out_channels = self._default_decoder_out_channels()` and builds at that. No
  constructor keyword was added, so the schema difference between the two families is untouched.
  At default arguments the same value is passed at the same point, so the decoder draws identical
  parameters in identical order and **the RNG stream cannot move** — proved by one seeded script
  run before and after, comparing both state dicts by `torch.equal`, both parameter totals,
  `n_depthwise_init` and the constructor signature. This is the only package in the family that
  edited another package's `nets/` to exist; `RESULTS.md` carries the revert path.
- **`SeqVaeLagAttnTrfE2E` did *not* gain the seam**, and nothing needs it there. That model is
  standalone — it derives from `nn.Module` directly rather than from `SeqVaeLagAttnTrfRws` — and
  builds its own decoder inline for a raw target it is the only consumer of. The asymmetry is
  recorded rather than resolved. It is still the **third** suite an RNG perturbation would surface
  in: it constructs `SeqVaeLagAttnTrfRws` as a reference in four test modules, one of which asserts
  an `n_depthwise_init` *difference* against it.
- **The target domain is a mixin, and it was moved rather than written.** The five members were
  `SeqVaeLagAttnFs`'s own until this package needed them; they moved verbatim into
  `lag_attn_fs/nets/feature_target.py`, leaving that class body empty too. Nothing about the built
  conv-LSTM model changed — proved the same way, in-process on one box, comparing every key of the
  metric dict under both likelihoods and every state-dict tensor with `torch.equal`. §6 records why
  the alternatives were rejected: multiple inheritance at the model layer runs the wrong
  constructor, and borrowing the five as class attributes would forfeit the empty-`vars` assertion
  half the test economy here rests on.
- **This package writes no encoder code and copies none.** A search of the parent's `nets/blocks.py`
  and `nets/encoders.py` for `raw`, `fhr` or `raw_per_step` returns nothing, so there is no
  raw-domain assumption in either that a copy would have to edit. The seven block-level test
  modules that own the encoder-level claims — `test_attention_block`, `test_blocks`,
  `test_encoders`, `test_encoder_causality`, `test_rope`, `test_source_window`,
  `test_prefix_equivalence` — are deliberately not ported: they exercise objects this package
  imports rather than copies. The model-level causality claim is new and is tested here (§12).

**Objective, metrics and tests**

- **Three tests build their comparison model from the *other* suite's keyword set.** The parameter
  comparison, the key-set comparison and the foreign-blob fixtures all need a conv-LSTM model beside
  this one, and the two constructors' schemas differ by six keywords — so each builds its comparison
  from that suite's own set and asserts the geometry the two share (horizon, $T$, resolved
  keep-index, decoder width). `tests/test_fixtures.py` records the failure directly: the feature
  suite's shipped set raises `TypeError` here and its tiny set does not, which is why the mistake
  would surface only at the shipped geometry.
- **The adapter-identity probe is an *ungated* claim.** Under a gate the forward hands the source
  adapter the gate's output rather than the source object itself, so the identity assertion runs on
  the ungated tiny set, following the parent's own copy; a second test covers the gated form, where
  each adapter sees a tensor at *its own* stream's surviving width.
- **Prefix equivalence is asserted on the mean-decoded branch**, for the `randn_like` layout reason
  in §12, with a second test that locates the difference in the draw rather than in the positions.
- **No `test_lag_map.py` and no forward-contract module.** The $\sum_\ell \mathrm{map} = K_t$
  identity is latent-side and untouched by the decoder width, and the twenty-key set and every
  latent shape are consequences of `vars(SeqVaeLagAttnTrfFs)` being empty — both are proved by the
  parent's suite over this exact forward, which is a stronger guard than a local copy.

**Configuration and driver**

- **`raw_per_step` stays.** §3.
- **`decoder_out_channels` is deliberately absent from the config and from `model_kwargs`**, and is
  not a keyword of this constructor at all. The width is recoverable from the stamped
  `target_keep_index`, and a second field could disagree with the gate.
- **`default.yaml` is written out in full rather than inheriting**, following both precedents and
  their stated reason: a `base:` chain hides which settings this run shares with the models it is
  compared against, and that sharing is the whole value of the comparison. The cost is drift, so
  three pins hold it — total against the conv-Transformer sibling, schema-limited against the
  feature sibling outside twelve config keys, and a key-set closure over the square. The exclusion
  list is **twelve** config keys and not the thirteen constructor keywords, because
  `decoder_out_channels` is named in no YAML in the repository.
- **The plotting config block keeps the shared driver's name.** §7.
- **`smoke_hie.yaml` carries two schedule-length deltas, not one.** The beta ramp is the one the
  feature sibling also scales; the second is `lr_warmup_steps`, which that sibling has no analogue
  for. The shipped $2{,}000$ steps is $91\%$ of that run's $\approx 2{,}200$: inherited, the
  learning rate would still be climbing when the criteria are read, and the pre-clip gradient norms
  the clip is derived from would come almost entirely from a model frozen near its initialisation.
  It ships at $200$.
- **`tiny.yaml` shrinks the widths under two independent constraints.** The constructor validates
  `num_heads * d_head == d_model` for the **lag-attention** heads, while `encoder_num_heads` is
  unrelated to `num_heads` and carries its own requirement that `d_model / encoder_num_heads` be
  even. The two products coincide in the shipped set only because both head counts happen to be
  $4$; a variant shrinking `d_head` while treating them as one constraint raises at construction.

**Where a stated rationale did not survive measurement**

- **The gradient clip moved, and it is the one constant both comparison models would have handed
  over wrong.** They arrived at $5000$ independently — the conv-Transformer raw model from a
  $1018$-epoch run at $q_{99} = 4683$, the conv-LSTM feature model from a $120$-epoch instrumented
  run at this same block size at $q_{99} = 4421$ — and this stack's $q_{99}$ is $3047.8$. The
  shared rule, the smallest round value above $q_{99}$, returns $4000$. The inherited value was not
  dangerous, and that is the point: nothing was being clipped at it, so no log line and no metric
  would ever have said the threshold was set for a different model. `RESULTS.md` carries the
  percentiles and the run.
- **`additive_margin` did *not* move, and the equality is now a measurement.** It was drafted as a
  pending measurement — equal to the feature sibling and different from the conv-Transformer one —
  so both a declared divergence and an asserted equality would have pre-committed the experiment.
  The instrumented run put this stack's post-warm-up `main_loss` fluctuation at $4.0\times$ inside
  the shipped margin against $3.3\times$ for the value's own derivation, so it is asserted equal on
  the encoder edge and stays a declared divergence on the target edge, where the block is $480$
  samples rather than $2340$ coefficients.
- **The throughput claim does not reproduce at dev-box scale.** At batch $32$ and $T = 300$ on one
  RTX 4080 the conv-Transformer is $3\%$ *slower* per step than the conv-LSTM at $38\%$ fewer
  parameters, and holds $3.4\%$ more memory. Recorded as measured rather than explained away. It
  neither supports nor refutes the production-scale claim, which is at batch $128$ on an A6000
  across seven ranks — and `RESULTS.md` says so where the row is.
- **The step-wise causality claim got *stronger*, not weaker.** It was expected to be inherited with
  the sibling's `causal_norm` qualification; there is no such flag here and the claim is
  unconditional (§12), including at the tiny fixture, which is the half that fails on the sibling.

## 17. Running it

From the repository root.

```bash
# Production, 7 ranks. TEB_RUN_STAMP is required so ranks 1..N-1 share rank 0's run directory,
# and the rank count must equal len(general_config.cuda_devices).
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_transformer_fs.trainer \
    --config teb_vae/lag_attn_transformer_fs/configs/default.yaml

# Local smoke: one epoch, one device, the committed four-sample shard.
python -m teb_vae.lag_attn_transformer_fs.trainer \
    --config teb_vae/lag_attn_transformer_fs/configs/tiny.yaml

# Dev-box validation: the shipped geometry and reach budget over the committed HIE sample shard.
python -m teb_vae.lag_attn_transformer_fs.trainer \
    --config teb_vae/lag_attn_transformer_fs/configs/smoke_hie.yaml
```

`RUN_CONFIG` near the bottom of `trainer.py` names the config used when the module is launched with
no command line, so the entry point works from an IDE's Run button with the only operator action
being to edit a value inside the file; a `--config` on the command line always wins, and a relative
path resolves against the repository root rather than the working directory. Note that a Run-button
launch of `default.yaml` is a *single* process whose seven `cuda_devices` make the framework spawn
DDP workers underneath it.

The gate:

```bash
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_fs/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_fs/tests -q -m slow
```

There is no `eval` entry point for this package. §14.

## 18. Configuration keys

Keys this document's claims depend on. `tests/test_docs.py` drives this section against
`configs/default.yaml` in both directions, so it cannot drift: every key in the first list must
exist, every key in the second must not, and every `model_config.VAE_model` key the shipped config
carries must appear in the first. Outside `VAE_model` the first list is the set this document's
claims rest on rather than an exhaustive inventory of the framework's own settings.

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

The five that describe the encoder being replaced. Each names no argument of this constructor, so
each would be dropped by the signature sweep without a word, leaving a config that reads correct
and builds a different model:

- `model_config.VAE_model.lstm_layers`
- `model_config.VAE_model.encoder_extra_dilations`
- `model_config.VAE_model.encoder_extra_kernel`
- `model_config.VAE_model.conv_norm_groups`
- `model_config.VAE_model.causal_norm`

The two the target domain derives or declares elsewhere. A key for either would be a second value
free to disagree with the first — and in the second case the disagreement breaks nothing, it
mislabels two reported columns:

- `model_config.VAE_model.decoder_out_channels`
- `model_config.VAE_model.target_block_split`

And the ones this architecture or this target domain made structural, derived or inert — each would
read to a maintainer as a control that exists:

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

The encoder head width is derived as `d_model // encoder_num_heads`; encoder dropout is the
existing `dropout`; the LayerScale initialisation and the rotary base are constructor defaults no
arm varies; the target attention window is the full causal prefix by design; `forecast_channels`
was removed from the feature-domain family precisely because inherited indices silently changed
meaning when `fhr_ph` went from $44$ to $66$ channels; and the plot cadence is
`general_config.plot_frequency`, the same key that drives the loss curves, so the figure and the
curves always exist for the same epochs.
