# `lag_attn_fs` — the as-built design record

The feature-domain lag-attention VAE-TEB: what it is, what it consumes, what it returns, what it
optimises, and every place the built model differs from the model it subclasses.

Companion documents: `teb_vae/lag_attn_rws/DESIGN.md` is the as-built record of the model this one
subclasses, and everything it states about the encoders, the heads, the lag attention, the shared
decoder, the initialisation policies and the objective's arithmetic holds here **by inheritance
rather than by restatement**. `teb_vae/lag_attn_rws/model_explained.md` states the latent
factorisation both share. `teb_vae/lag_attn/DESIGN.md` records the feature-target model that
predates the removal of the decoder bypass. `RESULTS.md` in this directory carries the measurements.
None of them is restated here.

**What this document is for.** Reading it should leave a reader able to say what a reported number
of this model means and what it does not — which is a different question from how the architecture
works, and the one a subclass makes easy to get wrong.

---

## 1. What the model is

At every 4-second anchor $t$ the model forecasts the next **two minutes of the stored FHR feature
future** — $H \cdot C_{\mathrm{keep}} = 30 \times 78 = 2340$ coefficients at the shipped reach
budget — twice: once from a target-only latent and once from a source-conditioned one. The gap
between the two forecasts, and the KL between the two latents, are the coupling readout.

It is the missing cell of a $2 \times 2$. `lag_attn` forecasts features through a `decoder_state`
bypass around the latent; `lag_attn_rws` forecasts raw FHR with no bypass. The move between them
changed the latent factorisation *and* the target domain at once, and nothing in the tree separated
them. Against `lag_attn` this model isolates the bypass removal; against `lag_attn_rws` it isolates
the target domain.

**It is an experiment, not a remedy.** `lag_attn_rws/PRIOR_SCALE_AND_PREDICTIVE_GAP.md` §5
establishes that the raw models' held-out predictive gain is negative because the source pathway
does not generalise — a failure that lives in the source encoder, the lag attention and the
posterior fusion, none of which a target-domain swap touches. This model is expected to reproduce
it. Its value is that it removes a confound, and it must not be read as a fix for one.

`SeqVaeLagAttnFs` is `class SeqVaeLagAttnFs(FeatureForecastTarget, SeqVaeLagAttnRws)` — still a
subclass of `SeqVaeLagAttnRws` by the MRO, with **an empty class body**. Everything that makes it a
feature-domain forecaster is one class constant and four callables, and they live in
`nets/feature_target.py` rather than here:

| Member of `FeatureForecastTarget` | What it is |
| --- | --- |
| `TARGET_BLOCK_SPLIT` | where the two stored blocks meet, for §9's block split only |
| `_default_decoder_out_channels` | names the decoder's width; builds nothing |
| `_build_forecast_target` | gathers the surviving channels and unfolds each anchor's future |
| `_resolved_forecast_gaps` | the four added readouts of §9 |
| `compute_loss` | builds the target, delegates, merges those four |

**Why a mixin rather than five methods here.** None of the five mentions an encoder — each reads
only `target_gate`, `c_y`, `geometry`, `horizon`, `coverage_floor`, `logvar_clamp` and
`decoder_out_channels`, all of which any of this family's constructors sets before building the
decoder. So a second forecaster reaching this same target through different encoders composes the
same two pieces in the same order instead of copying ≈230 lines whose drift would be silent. It is a
**move, not an abstraction**: a plain object deriving from `object`, with no `__init__`, no
`Protocol`, no `__init_subclass__` and no member that was not lifted verbatim.

The order of the bases is load-bearing. The mixin comes first, which is what makes its width hook
win method resolution over the base's `raw_per_step` one; reversed, the decoder would be built at
$R = 16$ and a feature block scored against it, with nothing raising.

The empty body is itself the guarantee. With nothing defined here, the twenty forward keys, the
posterior's structure, the lag map and the objective's metric set cannot have moved — they are the
base's own code objects, pinned by the base's own suite — which is why `test_construct.py` asserts
`vars(SeqVaeLagAttnFs)` carries no callable at all rather than counting methods.

One consequence is worth recording because a reader will meet it: a static type checker cannot see
that the mixin's seven read attributes are set by a *base* constructor, and reports them unresolved
in `nets/feature_target.py`. Declaring them would mean the `Protocol` the move deliberately does not
have, so the contract is stated in that class's docstring instead. There is no repository
type-check gate.

The constructor signature is deliberately the sibling's, down to the keyword order:
`trainer._build_model_kwargs` sweeps it with `inspect.signature` to decide what a config forwards,
and a narrowed one forwards **nothing at all** and silently builds an all-defaults model. That is
why the width is a method override rather than a constructor keyword, and why the mixin defines no
`__init__`.

At the shipped configuration the model holds **5,126,326 parameters**, against **5,094,458** for the
comparison model at the same reach budget — a delta of $+31{,}868$, entirely the decoder's output
head at $514 \times (C_{\mathrm{keep}} - R)$. Unguarded the pair is **5,135,988** and $5{,}088{,}186$,
a delta of $+47{,}802$. Read the budget before the delta: a guarded run builds availability input
adapters worth $6{,}272$, so the number `lag_attn_rws/DESIGN.md` §1 states is its *unguarded* model
and is not the baseline either delta is against.

The per-channel cost is $514$ rather than the $258$ this document once stated because the decoder
core's hidden width moved from $128$ to $256$: the head is $\mathrm{Linear}(d_{\mathrm{hidden}}, X)$
twice, so it is $2(d_{\mathrm{hidden}} + 1)$ per emitted channel.

## 2. Input contract

Read from the HDF5 through `train/data_module.py::GraphDataModule`, at `trim_minutes: 1.0`.
Identical to the comparison model's in every field; what changes is which of them is the target.

| Field | Shape | Role |
| --- | --- | --- |
| `fhr_st`, `fhr_ph` | $(B, 300, 43)$, $(B, 300, 66)$ | target stream, concatenated to $(B, 300, 109)$ — **both the network input and the reconstruction target** |
| `up_st`, `up_ph` | $(B, 300, 43)$, $(B, 300, 15)$ | source stream, concatenated to $(B, 300, 58)$ |
| `weight` | $(B, 300)$ | per-step validity on the decimated grid |
| `fhr`, `up` | $(B, 4800)$ | raw traces; **not read by the model** — row 1 of the diagnostic page only |
| `guid` | — | figure titles and run provenance |

**`fhr_st` and `fhr_ph` must be in `normalize_fields`.** They are this model's reconstruction
target, so an unnormalised block makes the Gaussian NLL meaningless against a unit-scale variance
model, with the loader raising nothing. The entry point refuses a config omitting either, field by
field, from `LagAttnFsTrainer.TARGET_FIELDS`.

**`fhr_up_ph` is absent from every config and must stay absent.** A coefficient mixing both signals
would break the target-only / source-conditioned separation the whole design rests on — and here it
would additionally put the source's own signal into the forecast target.

## 3. Geometry

`TrimmedRawGeometry` is reused **unchanged**, with no second geometry type and no shared protocol.
Only `t`, `t_valid`, `horizon` and `warmup` are consulted; `r`, `n_raw()` and `future_block_start()`
are never called by this model.

$$T = 300, \qquad H = 30, \qquad T_{\mathrm{valid}} = T - H = 270, \qquad \text{warm-up} = 30.$$

The raw grid stays a true fact about the batch — the loader delivers `fhr` at $4800$ and `weight` at
$300$, and `raw_masks._validate_weight` checks `weight.size(1) == geometry.t` — so `raw_per_step`
remains a required geometry input. It simply stops being the decoder width. Deleting it would break
the geometry rather than narrow the decoder, and setting it would not widen the decoder by one
channel.

`forecast_mask`, `contributing_anchors` and `kl_mask` are reused **unchanged** and broadcast over
the channel axis, producing $(B, 270, 30)$ and $(B, 300)$: they read only the four geometry fields
above and never `raw_len`, `decimation` or `r`.

## 4. Forward return dict

Twenty keys, the sibling's exactly. Only the four forecast tensors change shape.

- `mu_base`, `logvar_base`, `mu_full`, `logvar_full` — $(B, 270, 30, C_{\mathrm{keep}})$, against
  the comparison model's $(B, 270, 30, 16)$.
- `mu_prior`, `logvar_prior`, `raw_logvar_prior`, `mu_post`, `logvar_post`, `z_prior`, `z_post` —
  unchanged.
- `target_state`, `source_state`, `attended_source_heads`, `attn_weights` — unchanged.
- `kld_per_t`, `kld_per_t_per_head`, `source_kl_lag_map` — unchanged.
- `mu_prior_sat_frac`, `delta_mu_sat_frac` — unchanged.

**No `decoder_state` and no `delta_mu_src`**, as in the model this subclasses: there is no bypass to
carry one.

**`future_index` is inherited and present**, and that is worth stating because it reads as a
raw-target artefact. The base constructor registers it and a subclass can only drop it by overriding
`__init__`, which the width hook exists to avoid. It is non-persistent, so it reaches no checkpoint,
and it is simply never read — the stronger property is asserted instead: zeroing it moves no reported
metric.

## 5. Loss, and what its nats are summed over

$$\mathcal{L} = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0
+ \beta(e)\,\mathrm{KL}_{\mathrm{train}} + \beta_p\,R_p
+ \lambda_{\mathrm{ms}} \mathcal{L}_{\mathrm{ms}}
+ \lambda_{\mathrm{deriv}} \mathcal{L}_{\mathrm{deriv}}
+ \lambda_{\mathrm{boundary}} \mathcal{L}_{\mathrm{boundary}}$$

in nats per anchor, computed by `lag_attn_rws/nets/losses.py` — **the same code, not a copy of it**.
That module is already domain-neutral: it reduces a $(B, T_{\mathrm{valid}}, H, X)$ block against a
$(B, T_{\mathrm{valid}}, H)$ mask and takes both the target and $X$ as arguments. This model supplies
its gathered target and its block width and delegates. Three models keep a thin `compute_loss`
of this shape, and three copies of an objective that must never diverge is exactly the failure the
free-function structure exists to prevent.

**The three shape weights ship at $0.0$ here, and that is a domain judgement rather than an
oversight.** The last three terms price the *shape* of a raw waveform — its envelope under pooling,
its slope under first differences, and its first sample against the anchor's last observed one. This
model's block axis is $C_{\mathrm{keep}} = 78$ surviving wavelet channels, not $16$ consecutive raw
samples, so "adjacent" along that axis means "the next filter", pooling it mixes unrelated scales,
and there is no last-observed sample for a boundary to be continuous with. Evaluating those formulas
here would produce numbers, and every one of them would be meaningless. The keys are carried rather
than omitted so the task plumbing and the cross-package parity comparison stay uniform, and because a
term at weight $0.0$ is **not computed** — its metric is exact $0.0$, so the columns are honest zeros
rather than raw-domain formulas evaluated over a channel axis. The three exemptions in the parity
table state this reason.

**`block_width` is $C_{\mathrm{keep}}$, not `geometry.r`, and this is the sharpest silent trap in the
module.** It feeds only `mean_logvar_full`, `mean_logvar_base`, `logvar_full_floor_frac` and
`logvar_full_ceil_frac` — **not the loss**. Passing the raw grid's $R$ changes no loss, fails no shape
check, and rescales by $4.9\times$ exactly the four diagnostics `logvar_clamp` is re-derived from.
A test pins `mean_logvar_full` against a hand computation for this reason alone.

### What the nats are, and are not, comparable to

The reconstruction is summed over $H \cdot C_{\mathrm{keep}} = 2340$ coefficients against the
comparison model's $H \cdot R = 480$ samples, and the factorised Gaussian over correlated wavelet
coefficients overcounts independent information in both. So:

- **Not comparable to the raw model's.** A nat here and a nat there are sums over different blocks
  of differently-correlated quantities.
- **Not comparable across reach budgets within this model.** `causal_reach_budget_s` moves
  $C_{\mathrm{keep}}$, hence the decoder width, hence the block every nat is summed over. Two `fs`
  arms at different budgets have non-comparable `pred_gap` and mutually unloadable checkpoints —
  and the class guard cannot separate them, because both stamp `SeqVaeLagAttnFs`. Only the width the
  stamped `target_keep_index` implies does.

This is recorded, not fixed. Arms of this model at this budget rank against each other; nothing else.

## 6. Structural constraints that are not preferences

The four properties every reported nat rests on, inherited and re-asserted here rather than assumed
to have survived subclassing.

| Property | What enforces it | Test |
| --- | --- | --- |
| **No decoder bypass** — gradient reaches the decoder only through $z$ | `BaselineFutureDecoder.forward` takes exactly one tensor; no `decoder_state` head, no second decoder | `tests/test_invariants.py` |
| **Source purity** — the source pathway never sees a target tensor and the prior never sees the source | separate adapters and encoders; the posterior is a residual on the prior | `tests/test_invariants.py` |
| **Exact zero KL at initialisation** | posterior deltas zeroed after `initialization`; one shared $\epsilon$ | `tests/test_invariants.py` |
| **One decoder, invoked twice** | the same module object on $z^p$ and $z^q$ | `tests/test_invariants.py` |

**The zero-KL claim states its fixture's flags**, because it is conditional. It holds under the tiny
keyword set, which sets none of them. The shipped config ships `base_decode: mean` — under which the
two *forecasts* are no longer bitwise identical, though the KL is still exactly zero, since that
depends on the two distributions and not on the samples drawn from them — and
`posterior_logvar_mode: independent`, under which the init KL is zero only with
`head_init_calibration: true`. Any arm setting that flag false inherits a broken zero-KL start.

Because the delta heads are zero-initialised, **any KL assertion on a freshly constructed model
passes vacuously**; the `perturb_posterior` fixture is load-bearing for every test that claims to
check KL behaviour.

One further measurement belongs here: **the step-wise causality claim holds only under
`causal_norm: true`.** Without it the encoders' time-pooling normaliser mixes the whole sequence and
a forward probe is not bit-stable at the causal cut. Both directions are pinned in
`tests/test_smear.py`.

## 7. The target is gathered, never delayed

The input `ChannelGate` applies **two** operations: a gather of the surviving channels, and a
per-channel delay $\delta_c = \lceil \rho_c / \Delta \rceil$ that pushes each channel's forward
reach behind the anchor's causal endpoint.

**The target takes the gather and not the delay.** Delaying it would silently ask anchor $t$ to
forecast the future of anchor $t - \delta_c$, per channel, and nothing downstream would fail: every
shape is identical. `ChannelGate.forward` is `self.delay(index_select(x, -1, self.keep_index))` and
offers no gather-only method, so the target builder calls `index_select` directly.

At the shipped budget **all 78 surviving channels carry a non-zero delay** — one step at the fastest,
thirty at the slowest — so a gate-built target would be wrong in *every* channel it contains, not in
some of them. That is what makes the negative test specific rather than "not equal".

The gather runs **before** the unfold. The two commute; doing it first keeps the copy at
$(B, T, C_{\mathrm{keep}})$ rather than $(B, T_{\mathrm{valid}}, H, C_{\mathrm{keep}})$ — a factor of
$H$, and a third of a gigabyte at the production batch.

At `causal_reach_budget_s: null` the survivor set is all $109$ channels and the decoder width follows,
so the unguarded arm is well defined and its target is the ungathered stream.

## 8. The target is smeared, and that is not a leak

A stored coefficient at decimated step $s$ is **not** a value *at* $s$. It is a weighted average of
raw signal over a window **centred** at raw index $16s$ with half-width $\rho_c$, the channel's
$L_{95}$ energy reach. So the forecast target at short horizons is partly a deterministic function of
raw signal the model has already observed. The fraction of horizon step $\tau$'s support lying in
observed history is

$$b(\tau, \rho_c) = \max\!\left(0,\; \frac{\rho_c - 4\tau}{2\rho_c}\right),$$

exactly $0.5$ at $\tau = 0$ for every channel, falling linearly to zero at step $\rho_c / 4$.

**This is not a causality violation.** No future information enters the model; part of the answer is
computable from what the model legitimately already observed, which is the opposite of leakage. The
*input*-side two-sidedness is a separate matter, identical in both target domains, and handled by the
existing `causal_reach_budget_s` gate.

**It does not bias the readout either.** Writing the target as $Y^+ = (A, B)$ with $A$ the component
fixed by the observed history, $A$ is $\sigma(Y^-)$-measurable, so $I(U^-; A \mid Y^-) = 0$ and
$I(U^-; Y^+ \mid Y^-) = I(U^-; B \mid Y^-)$. In the estimator, $D_0$ and $D_1$ come from one shared
decoder under one shared $\epsilon$, so $A$'s contribution is near-identical in both branches and
largely cancels in the difference.

What it does affect is **optimisation** — a share of the summed NLL is a component the source cannot
help with, so the source pathway competes against an easy reconstruction task — and
**interpretation**: a 30-step feature forecast is not the same physical claim as a 30-step raw
forecast.

**This is why the target is the gated subset rather than all 109 channels.** Restricting it halves
the mean blend and makes the far horizon genuinely clean. Measured over the $H = 30$ horizon on the
production filter bank, the mean blend is $0.091$ on the kept set against $0.173$ on the full set,
reaching exactly $0.000$ by the last horizon step; the kept set's worst channel becomes clean at step
$29.3$, just past the horizon, while the full set never reaches a fully clean step because it retains
channels reaching $965.5$ s. Restricting it also makes one statement true that is otherwise false:
**input and target live under one causal budget.**

The blend fraction and that table are recomputed from the shipped filter bank in
`tests/test_smear.py`, so the figures a reader meets are ones a test reproduces. The preprint at
`teb_vae/lag_attn_rws/doc/latex_template/` covers the *input*-side half of two-sidedness in
`sections/reach.tex`; a dedicated subsection for the backward half stated above is not yet written
there, and until it is, this section is the record.

> lean-limit: the blend is bounded by the reach budget and never removed; replace with a one-sided
> filter bank when one exists and the stored features are regenerated under it.

## 9. Observability while the evaluation is deferred

There is **no `eval/` package for this model** — no `ModelBinding`, no verdicts, no bootstrap
confidence intervals, no trivial-predictor baselines, no calibration, no per-recording tables.
Deferred whole, and the consequence is structural: every readout is a scalar read from a run's own
`train_results/metrics_history.csv`.

That leaves one problem this target domain creates and the raw models do not have. A scalar summed
over $2340$ coefficients cannot separate a model forecasting three easy channels well from one that
is uniformly mediocre — and, given §8, cannot separate forecasting from reconstruction of the
already-determined component. Four added metrics close it, and all four are nearly free because the
block score already reduces over $(H, C)$:

| Metric | What it separates |
| --- | --- |
| `pred_gap_tau_first` | the horizon step whose target is half-determined by observed history |
| `pred_gap_tau_last` | the step whose target is not determined by it at all |
| `pred_gap_st`, `pred_gap_ph` | the two stored blocks, whose filters have different reaches |

**All four are partial sums of the `pred_gap` beside them**, over the same denominator, and both
splits recompose to it. That is the only property that makes them worth reporting, so the per-element
term is the objective's own `raw_sample_score` and the mask is rebuilt through the objective's own
two functions rather than restated. The two branches are reduced one at a time: one branch's score is
a third of a gigabyte at the production batch, and holding two plus their difference would triple
that for four scalars.

The block split needs `TARGET_BLOCK_SPLIT = 43`, which **cannot be derived** — $c_y$ is the two
blocks' *sum*. It is a class attribute rather than a constructor keyword because a key that changed
only which of two diagnostics a coefficient was counted in would look like an architecture decision
in every checkpoint. Nothing else depends on it, so a stale value would mislabel two columns and break
nothing; the task, which is the only layer that sees the two blocks separately, checks it against the
data it assembles the target from.

## 10. The diagnostic page

**There is no `lag_attn_fs/plotting.py` and no callback class of its own.** The shared callback is
already model-agnostic — it routes through the task's builders, `model.compute_loss` and
`model.geometry` — and it is reached through the trainer's `plot_callback_cls()` seam. The work is in
the page, not the callback.

Rows 3–7 (prior mean over posterior displacement, per-step per-dimension KL, $K_t$, the
lag-attention matrix, the per-lag KL attribution) are the sibling's, drawn by the sibling's code.
Row 1 is also the sibling's, extracted into a shared `raw_context_row` both pages call: it keeps the
raw FHR and UP traces as physiological context, which the model does not read but a reader judging
whether a forecast is plausible does. Only **row 2 is rebuilt** — three target channels with
$\pm 2\sigma$ bands plus a $(H \times C)$ absolute-error heatmap for one anchor.

Two things about that row are decisions rather than details. **Channels are selected by a stated
rule**, not by index: `lag_attn`'s `forecast_channels` key was removed precisely because inherited
indices silently changed meaning when `fhr_ph` went from 44 to 66 channels. The rule ranks by
predictive calibration and draws the worst, the middle and the best — a panel showing only the worst
reads as a broken model on every run, and one showing only the best as a working one. **The error map
is an untitled inset**, not a second panel: its x-axis is the horizon step of one anchor rather than
physical time, so a side-by-side split would leave the curves narrower than the other six rows and
break the property that a column of the page is one instant across all seven. Untitled deliberately,
so "every titled axes spans the recording" stays exactly checkable.

The forecast row states unconditionally that it carries no physical unit: the target is the loader's
`normalize_fields` output used as delivered, so there is no second normalisation to invert.

**The config block keeps the inherited name `lag_attn_rws_plotting`.** Renaming it to match this
package would leave the figure permanently off, with `enabled: true` still reading correct to anyone
looking at the config and nothing in the log saying why.

## 11. Deliberate limitations

- **The nats are budget-local and model-local.** §5. Recorded, not fixed.
- **No evaluation pipeline.** §9. Nine of eighteen sibling analyses are domain-agnostic, but
  `collect`, `metrics`, `spectra`, `oracle`, `events`, `coherence` and `samples` are structurally
  raw, and a feature-domain `events` or `coherence` is a new scientific construction rather than a
  port. Inventing one to fill a column is worse than an absent column.
- **The target carries a second normalisation nowhere.** The loader's output is used as delivered.
  Per-channel target standardisation would make the `head_init_calibration` trivial-predictor
  argument exactly true, but it adds a statistics blob that must travel with the checkpoint; the
  decoder's per-coefficient log-variance head absorbs residual scale differences, and `lag_attn`
  already trains this way.
- **Every local measurement is in-sample.** `dataset_kwargs` is shared between the two loaders and
  cannot carry a per-split GUID filter, so the dev-box runs in `RESULTS.md` validate the objective's
  optimisation behaviour and say nothing about generalisation.

> lean-limit: the four added readouts are training-path scalars with no uncertainty; replace with the
> evaluation's per-recording paired statistics when a feature-domain `eval` package exists.

## 12. Deviation record

Where the built model differs from the design it was built from, and why.

**Architecture and construction**

- **The target domain is a mixin, not four methods on this class.** Built as a subclass carrying
  four callables and one constant; they were later moved verbatim into
  `nets/feature_target.py` as `FeatureForecastTarget`, leaving this class body empty. Nothing about
  the built model changed — proved by one seeded script run before and after, comparing every key of
  the metric dict under both likelihoods and every state-dict tensor with `torch.equal`, plus both
  parameter totals and the constructor signature. The reason is a second forecaster reaching this
  same target through the conv-Transformer encoders: `SeqVaeLagAttnTrfRws` derives from `nn.Module`
  directly rather than from `SeqVaeLagAttnRws`, so a class inheriting from both models linearises
  through the **conv-LSTM** constructor and silently builds the wrong one. See §1.
- **The decoder width is a method, not a keyword.** `_default_decoder_out_channels` overrides a hook
  on the base class rather than passing `decoder_out_channels` into it. Two reasons, each sufficient:
  the gate it reads is built *by* the base constructor, so nothing outside can compute the value
  beforehand; and a subclass narrowing `__init__` to intercept the keyword breaks the
  `inspect.signature` sweep, which then forwards no configuration at all. The hook is an exact no-op
  on the base — no RNG consumed, parameter count unchanged.
- **`decoder_out_channels` is deliberately absent from the config and from `model_kwargs`.** The
  width is recoverable from the stamped `target_keep_index`, and a second field could disagree with
  the gate.
- **The model unfolds its own stream** rather than calling `lag_attn/figure_primitives.py`'s
  `future_target`. That helper's signature *is* the two stored block names, which is exactly the
  schema knowledge `tests/test_nets_are_framework_free.py` forbids inside `nets/`. A test pins the
  two equal so the duplication is proven inert.
- **`future_index` is inherited and present**, not absent. §4.

**Objective and metrics**

- **`compute_loss` takes the target and the block width as arguments** — a generalisation applied to
  the shared objective, proved bitwise inert for all three shipped models by re-running one seeded
  script before and after rather than against committed constants.
- **The four resolved gaps are merged after the delegation, not computed inside the objective.** The
  raw models' block is one physical channel over thirty horizon steps, so neither split says anything
  there, and their metric dicts are pinned bitwise.
- **The lag-map conservation identity is `allclose`, not `torch.equal`.** The map is contracted with
  `einsum` over the head axis while $K_t$ is summed over the dimension axis, so the two reach the
  same number by different summation orders: $\approx 10^{-6}$ on values of order $10$.

**Configuration and driver**

- **`raw_per_step` stays.** §3.
- **The normalisation guard is parameterised, not moved.** `_check_raw_target_normalized` gained a
  `fields` keyword and `main` passes `trainer_cls.TARGET_FIELDS`. It stays in `main`'s by-name list
  so no subclass can drop it, and `preflight` keeps its documented no-op property — which is what
  makes a bare-bodied override safe in the packages that have one.
- **The plotting config block keeps the sibling's name.** §10.
- **`default.yaml` is written out in full rather than inheriting**, then pinned leaf-for-leaf against
  the comparison config outside a declared allow-list. The pin is **total** rather than
  schema-limited, because the constructor schema is unchanged and every key therefore means the same
  thing in both files.
- **The parity allow-list is six entries: five identity and one number.** Three loss-scale constants
  were drafted as divergences on the argument that this objective sums a $4.9\times$ larger block,
  and when each was measured at that scale only `additive_margin` had moved. The other two live in
  `MEASURED_TO_MATCH_PATHS`, whose test asserts the equality so parity reads as a measurement rather
  than an oversight.

**Where a stated rationale did not survive measurement**

- **`gradient_clip_val` did not move.** Re-derived from a 120-epoch instrumented run at this
  objective's own scale: $q_{99}$ $4421$ against the comparison model's $4681$, so the same rule —
  the smallest round value above $q_{99}$ — returns the same $5000$. The reconstruction sums over 78
  channels but the decoder's output head is per-channel, so the extra terms land on disjoint rows of
  two `Linear` layers rather than accumulating onto one shared parameter: the *loss* scales with the
  block and the norm of its gradient does not. `additive_margin` did move, $10^{3} \to 5 \times 10^{3}$.
- **The scale-matched $\beta$ was wrong, and the sweep says so.** The direction argument is correct —
  a larger reconstruction at fixed $\beta$ makes $\beta\,\mathrm{KL}$ relatively weaker, and the
  measured rate falls from $3.46$ to $0.23$ across the bracket — but what it inferred, that the
  inherited $\beta$ would leave the latent carrying target information a permutation control might
  miss, is false here: at $\beta = 1.0$ the control fires *hardest*. Four arms are monotone in
  $\beta$ on every column of the selection rule and the lower edge wins all of them, so the shipped
  pair is $\beta = 1.0$, $\beta_p = 0.1$ — the comparison model's own values, reached by measurement.
  `RESULTS.md` carries the tables and the in-sample caveat.

  **That sweep ran at $d_z = 48$ and the shipped latent is now $64$**, so the KL it selected against
  is summed over a third more dimensions while the reconstruction block is unchanged at $2340$
  coefficients — the same direction argument the sweep already refuted, one step further. The pair is
  **not** rescaled on that reasoning: the sweep's finding was that the reconstruction-versus-KL scale
  argument does not predict this model's behaviour, and re-applying it now would be repeating the
  error the entry records. It stands as measured, and is re-examined against the same four columns
  when fs training resumes at the new geometry.
- **The prior anchor is not the delicate key the comparison model's history suggests.**
  `logvar_prior_floor_frac` is $0.0121$ at $\beta_p = 0.1$ and exactly $0$ at every stronger anchor,
  where the threshold argument predicted the weakest anchor would be overrun. The argument is not
  wrong; the pressure it describes reaches the prior's log-variance through `base_decode` and
  `posterior_logvar_mode`, and this configuration ships the settings that remove both paths.
- **`logvar_clamp` is confirmed, not revised.** The decoder's mean log-variance sits at $-1.11$ to
  $-1.13$ across a tenfold change in $\beta$ — $3.88$ above the floor — with $1.4\%$ of coefficients
  on the floor and $0.03\%$ on the ceiling. Both ends live, neither binding. The interval was
  inherited by the raw models *from* a feature-coefficient decoder, so here it is going home.

## 13. Running it

From the repository root.

```bash
# Production, 7 ranks. TEB_RUN_STAMP is required so ranks 1..N-1 share rank 0's run directory.
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_fs.trainer --config teb_vae/lag_attn_fs/configs/default.yaml

# Local smoke: one epoch, one device, the committed four-sample shard.
python -m teb_vae.lag_attn_fs.trainer --config teb_vae/lag_attn_fs/configs/tiny.yaml

# Dev-box validation: the shipped geometry over the committed HIE sample shard.
python -m teb_vae.lag_attn_fs.trainer --config teb_vae/lag_attn_fs/configs/smoke_hie.yaml

# A calibration arm: the same launch line against any configs/sweep_beta_*.yaml.
```

`RUN_CONFIG` near the bottom of `trainer.py` names the config used when the module is launched with
no command line, so every runnable entry point works from an IDE's Run button with the only operator
action being to edit a value inside the file. A `--config` on the command line always wins.

There is no `eval` entry point for this package. §9.

## 14. Configuration keys

Keys this document's claims depend on. `tests/test_docs.py` asserts each required key exists in
`configs/default.yaml` and each absent key does not, in both directions, so this section cannot drift
from the config.

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
