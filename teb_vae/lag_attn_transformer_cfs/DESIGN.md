# `lag_attn_transformer_cfs` — the as-built design record

The conv-Transformer causal-feature-domain lag-attention VAE-TEB: which parent supplies which half,
what it consumes, what it returns, what it optimises, and every place the built package differs from
the design it was built from.

Companion documents, none of them restated here: `teb_vae/lag_attn_cfs/DESIGN.md` records the target
domain, the warm-up budget, the anchor tiling, the source compromise and the eight added readouts
this model composes **by import and unchanged**; `teb_vae/lag_attn_transformer_rws/DESIGN.md` records
the encoders, the adapters, the attention blocks and the wiring it composes them over;
`teb_vae/lag_attn_rws/DESIGN.md` records the architecture all six models share and
`teb_vae/lag_attn_rws/model_explained.md` the latent factorisation. `RESULTS.md` in this directory
carries the pre-registered criteria and the measurements.

**What this document is for.** Reading it should leave a reader able to say which of two comparisons
a given number of this model belongs to — which is a different question from how the architecture
works, and the one a model assembled entirely out of imported parts makes easy to get wrong.

---

## 1. What the model is

The sixth cell of an encoder-by-target grid, and the one that closes it:

```
                          conv-LSTM encoders          conv-Transformer encoders
  raw FHR target          lag_attn_rws                lag_attn_transformer_rws
  two-sided feature       lag_attn_fs                 lag_attn_transformer_fs
  causal feature          lag_attn_cfs                lag_attn_transformer_cfs   <- this
```

At each admitted 4-second anchor $t$ the model forecasts the next **two minutes of the stored
one-sided FHR feature future** — $H \cdot C_{\mathrm{keep}} = 30 \times 98 = 2940$ coefficients at
the shipped warm-up budget — twice, from a target-only latent and from a source-conditioned one,
through one shared decoder invoked twice under one noise draw. That is the `lag_attn_cfs` target and
input handling reached through the `lag_attn_transformer_rws` encoders, and nothing else.

**Its value is that it closes the square.** With all six cells present either axis can be read at a
fixed value of the other: against `lag_attn_cfs` the configurations differ in the **encoder** alone,
and against `lag_attn_transformer_fs` in the **transform** alone. Neither of the two three-cell
configurations allowed that, and it is the entire reason this cell exists.

The whole model is

```python
class SeqVaeLagAttnTrfCfs(CausalWarmupInputs, CausalFeatureForecastTarget, SeqVaeLagAttnTrfRws): ...
```

with **a constructor and nothing else** — `vars(SeqVaeLagAttnTrfCfs)` carries `__init__` and no other
callable and no class constant — linearising as
`SeqVaeLagAttnTrfCfs -> CausalWarmupInputs -> CausalFeatureForecastTarget -> FeatureForecastTarget ->
SeqVaeLagAttnTrfRws -> Module -> object`. §7 records why the constructor is the one exception.

**It is an experiment, not a remedy.** `lag_attn_rws/PRIOR_SCALE_AND_PREDICTIVE_GAP.md` §5
establishes that the held-out predictive gain is negative because the source pathway does not
generalise — a failure in the source encoder, the lag attention and the posterior fusion, and an
encoder swap of this kind removes a confound rather than that failure. This model is expected to
reproduce it. **The sign of `pred_gap` is a criterion nowhere in this document.**

At the shipped configuration the model holds **4,284,556 parameters**, against **4,655,987** for the
encoder-axis comparison `lag_attn_cfs`. With every architecture switch of
`lag_attn_cfs/DESIGN.md` §17 at its off-state and the single shared alignment clock restored it
holds **5,054,992**, which is bitwise the model that shipped before this revision and is the row on
which the target-axis comparison against `lag_attn_transformer_fs` (**5,034,984**) and the
pre-revision encoder-axis comparison (**5,146,334**) are still readable. §13 carries the arithmetic
and decomposes every delta.

**Six mechanisms of this cell are configuration rather than architecture**, each gated by a key
whose off-state reproduces the pre-revision model bitwise. The inventory, the off-states and where
each is pinned are `lag_attn_cfs/DESIGN.md` §17's and are not restated; both cells take the same six
keys and both configs ship the same six values, because five of them are the shared causal parent's
and the sixth is the shared resolver's. What is **not** shared is what one of them builds: under
`lag_kv_source: conv_stem` this cell's local K/V stem is a `GatedCausalConvStem` over this
architecture's own `encoder_conv_kernels` / `encoder_conv_dilations` schedule, and §13 is where the
difference that makes is priced.

## 2. Input contract

Identical to `lag_attn_cfs` in every field, width and refusal — the causal HDF5 shards through
`train/data_module.py::GraphDataModule` at `trim_minutes: 1.0`, $c_y = 36 + 66 = 102$ and
$c_u = 36 + 15 = 51$, with `fhr_up_ph` absent by construction. `lag_attn_cfs/DESIGN.md` §2 is the
record and it is not restated, because the input contract is a property of the *dataset and the
target domain* and neither is a property of an encoder. What the encoder change touches here is
nothing at all.

Three requirements are repeated because a config for this package is a separate file that can lose
them independently:

- **`fhr_st` and `fhr_ph` must be in `normalize_fields`.** They are the reconstruction target, so an
  unnormalised block makes the Gaussian NLL meaningless with the loader raising nothing. The shared
  entry point's guard reads the driver it was handed and resolves `TARGET_FIELDS` through the
  **causal** parent (§7); a `trainer_cls=` wiring mistake would leave it checking the raw model's
  `fhr`, which these configs satisfy.
- **`guid` and `epoch` must be in `load_fields`**, because the tile phase is keyed on the pair and
  `load_fields` is honoured literally.
- **The shards must be the causal variant.** The two variants share every field name and every dtype;
  only the root `transform` attribute and the channel counts tell them apart, and the warm-up
  resolver refuses a two-sided shard by name before a run directory exists.

## 3. Geometry, tiling and the warm-up

All three are the causal parent's, reached by import: the budget-and-floor pairing $B = 134$,
$F = 134$ keeping $98$ of $102$ target channels and, under the shipped pair of alignment clocks,
$39$ of $51$ source (`up_st` $30/36$, `up_ph` $9/15$) at an inter-stream offset of $-113.8932$ s;
the tiled anchor set
$\mathcal{A}(\varphi) = \{F + \varphi + kS\}$ at $S = H = 30$ with the phase derived per segment from
a `blake2b` of `guid`, `domain_start`, the epoch and the seed; $A_{\max} = 5$ at the training
geometry and $136$ at the dense evaluation one; and the input warm-up mask applied inside
`AvailabilityInputAdapter`. `lag_attn_cfs/DESIGN.md` §§3, 4 and 7 are the record.

Two facts about *this* architecture are worth stating where a reader will look for them.

**`TrimmedRawGeometry` is reused unchanged and `raw_per_step` stays a required geometry input.** The
loader delivers `fhr` at $4800$ and `weight` at $300$, and `raw_masks._validate_weight` checks
`weight.size(1) == geometry.t`, so the raw grid is a true fact about the batch. It simply stops being
the decoder width.

**The lag validity floor is the parent's too.** `lag_floor` ships at $0$, where the mask is bitwise
the architecture parent's, and exists so the source compromise of `lag_attn_cfs/DESIGN.md` §8 is
measurable rather than argued about.

## 4. Forward return dict

`SeqVaeLagAttnTrfCfs.forward(y_st, y_ph, u_stream, anchor_phase=None, anchor_stride=None)` returns
**twenty-three keys** at the shipped configuration — the causal parent's exactly, which are the
family's twenty, the two the anchor axis needs, and `persistence` under `persistence_residual`
alone. With that key off it returns twenty-two, which is bitwise the contract that predates the
mechanism.

- `mu_base`, `logvar_base`, `mu_full`, `logvar_full` — $(B, A_{\max}, H, C_{\mathrm{keep}})$, so
  $(B, 5, 30, 98)$ at the shipped training geometry and $(B, 136, 30, 98)$ at the dense one.
- `anchor_index` $(B, A_{\max})$ `long` and `anchor_valid` $(B, A_{\max})$ `bool`.
- `mu_prior`, `logvar_prior`, `raw_logvar_prior`, `mu_post`, `logvar_post`, `z_prior`, `z_post` —
  $(B, 300, 64)$; `target_state`, `source_state` — $(B, 300, 128)$; `attended_source_heads`
  $(B, 300, 4, 32)$; `attn_weights` $(B, 300, 4, 91)$; `kld_per_t`, `kld_per_t_per_head`,
  `source_kl_lag_map`, `mu_prior_sat_frac`, `delta_mu_sat_frac` — all unchanged in shape.
  **`source_state` is the lag attention's K/V tensor** rather than the deep source state by
  definition: under `lag_kv_source: encoder` those are the same object, and under the shipped
  `conv_stem` it is the stem's output. Both intervention controls follow it.
- `persistence` $(B, A_{\max}, C_{\mathrm{keep}})$ — the anchor's own target vector, under
  `persistence_residual` alone, indexed by anchor **position** rather than by sequence step.

**No `decoder_state` and no `delta_mu_src`**, as in both parents: the decoder receives the latent
and a target-only persistence vector, so there is no *source* bypass to carry one.

**`future_index` is inherited from the architecture parent and never read**, and that is worth
stating because it reads as a raw-target artefact. The constructor registers it and a subclass could
only drop it by overriding `__init__`, which the width hook exists to avoid. It is non-persistent, so
it reaches no checkpoint.

## 5. Loss, metrics and what the nats are summed over

The objective, its $\beta$ schedule, its metric surface, the validation-only permutation control, the
source-null arm, the spike-breaker wiring and the checkpoint contract are the causal parent's — the
same code, not a copy. `LagAttnTrfCfsTrainer.TRACKED_METRICS` carries **99** entries and resolves to
the causal parent's tuple; `lag_attn_cfs/DESIGN.md` §10 records what the eight added readouts
separate, why `target_warm_frac` and `anchors_per_sample` are guards rather than results, and why
`_mu_gap_rms` is overridden onto the tiled anchor set.

**`lambda_boundary` is refused at any non-zero value**, inherited from the causal parent's pre-flight:
the boundary term is a slicing identity over *adjacent* anchors, and this family always decodes a set
whose entries are $S$ apart.

### The reconstruction is weighted per stored block

Both reconstruction terms carry a per-channel weight $w_c$, resolved at construction from
`target_weight_st` and `target_weight_ph` and registered as a non-persistent buffer positional
against `target_keep_index`. It ships at $(1.0, 0.1)$.

**Uniform was never neutral.** At the shipped budget $66$ of the $98$ survivors are phase-harmonic
against $32$ scattering, so an unweighted objective spent $67.3\%$ of itself on `fhr_ph` by nothing
more deliberate than channel count. The shipped pair resolves to $2.5389$ and $0.25389$ and moves
scattering's share to $82.9\%$.

**The vector is renormalised so $\sum_c w_c = C_{\mathrm{keep}}$**, which is the part that keeps the
rest of the configuration valid: the weighted block sums to the same magnitude the uniform one did,
so `gradient_clip_val` and `additive_margin` are not invalidated a second time and $\beta$ keeps its
standing against the reconstruction. Only the *distribution* moves. What the configuration states is
therefore a **ratio** — $(10.0, 1.0)$ describes the same objective.

At $(1.0, 1.0)$ the resolved vector is exactly ones and `raw_sample_score` skips the multiplication
entirely, so an unweighted cell scores a block that is **bitwise** the one it scored before the
mechanism existed. `tests/test_metrics.py` asserts that rather than approximating it, because every
other cell of the grid depends on it.

**What it costs, and it is a unit rather than a scale.** A weighted reconstruction is not a
log-density, so $\beta = 1$ is no longer the exact ELBO. **The evaluation pipeline is deliberately
left unweighted** — `eval/` reduces through the same functions with no vector, so its `nll_*` and
`pred_gap` remain true log-densities and `pred_gap_mc_likelihood_pct` remains a probability
statement. Training optimises the weighted objective and the evaluation measures the unweighted one,
which is the right split for a device that decides where capacity goes rather than what a good
forecast is — but it means a `pred_gap` from `metrics_history.csv` and one from `summary.json` are
no longer the same quantity.

The four channel-axis splits (`pred_gap_st`, `pred_gap_ph`, the three warm-up tertiles) apply the
same vector, so each remains a partial sum of the `pred_gap` printed beside it. Reading
`pred_gap_st` against `pred_gap_ph` is what judges the choice.

### And per horizon step, and the decoder mean carries a persistence residual

Two further mechanisms, both the causal parent's and both shipped here at the same values:
`horizon_weight_halflife_steps: 15.0` weights the reconstruction on the *horizon* axis by
$w_\tau \propto 2^{-\tau/\lambda}$ renormalised to $\sum_\tau w_\tau = H$, and
`persistence_residual: true` adds a learnable $(H, C_{\mathrm{keep}})$ target-only persistence term
to the decoder's **mean** head alone. `lag_attn_cfs/DESIGN.md` §5 is the record for both: why the
renormalisation is what keeps `gradient_clip_val`, `additive_margin` and $\beta$'s standing valid,
why the residual opens no source bypass, and why both decoder invocations must receive the same
persistence tensor.

Two consequences belong here rather than there, because they are properties of *this* edge:

- **The encoder edge stays readable only because both cells ship both mechanisms at the same
  values.** Had either shipped a different half-life or a different residual state, the two cells
  would optimise different criteria and a loss level would carry nothing — the same failure mode the
  horizon move already had to avoid.
- **The caution about weighted scores now applies on two axes.** A block score weighted on either is
  not a log-density, so $\beta = 1$ is not the exact ELBO; the evaluation applies **neither** weight,
  and `horizon_weight_halflife_steps` is deliberately absent from the eval binding's `GEOMETRY_KEYS`
  for exactly that reason. `persistence_residual` **is** in that tuple, because it changes what the
  predictor is and therefore what every `nll_*` measures.

### What the nats are, and are not, comparable to

This is the cell where the distinction earns its place, because the two edges are not symmetric:

- **The encoder edge, against `lag_attn_cfs`: a loss *level* is comparable.** Both cells sum the same
  $2940$ coefficients over the same anchor count under the same objective, so `total_loss`,
  `nll_base_block` and `pred_gap` may be read against each other directly. **This survived the
  horizon move only because both causal cells moved together**; had this one gone to $H = 30$ alone,
  three non-encoder keys would differ and the edge — the reason this cell exists — would carry
  nothing.
- **The target edge, against `lag_attn_transformer_fs`: a loss level is still *not* comparable, but
  for one reason now rather than two.** The block is $2940$ against $2340$, because
  $C_{\mathrm{keep}}$ is what the warm-up budget decides and no configuration change restores it.
  What the horizon move *did* buy is that both sides now forecast $30$ steps, so a **per-horizon-step**
  reading — does the gap survive to the far step? — is made along the same axis on both sides, where
  before it compared a one-minute question to a two-minute one. A summed level still carries only a
  sign, a trajectory, the bottleneck-health columns, the parameter budget and the *ordering* of arms.
- **Not comparable across warm-up budgets within this model.** $C_{\mathrm{keep}}$ is what the budget
  decides, hence the decoder width, hence the block. Two arms at two budgets have non-comparable
  `pred_gap` and **mutually unloadable checkpoints**, and the class stamp cannot separate them —
  only the width the stamped `target_keep_index` implies does.

## 6. Why two mixins and not two inheritances

**The obvious construction does not work, and the failure is silent.** `SeqVaeLagAttnCfs` subclasses
`SeqVaeLagAttnRws`, while `SeqVaeLagAttnTrfRws` derives from `nn.Module` directly. So
`class X(SeqVaeLagAttnCfs, SeqVaeLagAttnTrfRws)` linearises as
`X -> Cfs -> ... -> Rws -> TrfRws -> Module` and **runs the conv-LSTM constructor**: a model that
builds, trains and reports, and is not this architecture.

So the target domain lives in two plain objects in `teb_vae/lag_attn_cfs/nets/`, **neither of which
mentions an encoder**, and both cells are their own architecture plus those two. It is a *move*, not
an abstraction: no `Protocol`, no `__init_subclass__`, no member that was not lifted verbatim. What
it prevents is a second copy of the target domain whose drift would be silent — the delay trap of §7
and the `block_width` trap of `lag_attn_cfs/DESIGN.md` §5 both change no shape and raise nothing.

**The order of the bases is load-bearing.** The mixins come first, which is what makes the width hook
win method resolution over the architecture parent's `raw_per_step` one and the tiled forward win
over the dense one. Reversed, the decoder is built at $R = 16$ and a $98$-channel block is scored
against it. That failure is loud, but not where a reader would look for it: `block_width` would not
catch it, since it feeds only the four log-variance diagnostics and no shape check, while
`raw_sample_score` computes $(\text{target} - \mu)^2$ on $(B, A, H, 98)$ against $(B, A, H, 16)$,
which is not broadcastable. `tests/test_construct.py` pins the `__mro__` and constructs the reversed
order to check `out_features`, so the failure names its cause rather than arriving as a broadcast
error three frames down.

**The width is a method, not a constructor keyword.** `_default_decoder_out_channels` overrides a
hook the architecture parent carries. Two reasons, each sufficient: the gate it reads is built *by*
that constructor, so nothing outside can compute the value beforehand; and a subclass narrowing
`__init__` to intercept a keyword breaks the `inspect.signature` sweep in
`trainer._build_model_kwargs`, which then forwards **no configuration at all** and silently builds an
all-defaults model.

## 7. The three diamonds, and their measured resolution order

```python
class SeqVaeLagAttnTrfCfsTask(SeqVaeLagAttnCfsTask, SeqVaeLagAttnTrfRwsTask):
    pass

class LagAttnTrfCfsTrainer(LagAttnCfsTrainer, LagAttnTrfRwsTrainer):
    MODEL_CLS = SeqVaeLagAttnTrfCfs
    TASK_CLS = SeqVaeLagAttnTrfCfsTask
    CHECKPOINT_STEM = "lag-attn-trf-cfs"
```

The task defines **zero** callables. The driver re-points three class attributes and defines no
method. All three linearisations are asserted as lists of class names against the real `__mro__`:

`SeqVaeLagAttnTrfCfs -> CausalWarmupInputs -> CausalFeatureForecastTarget -> FeatureForecastTarget ->
SeqVaeLagAttnTrfRws -> Module`,

`SeqVaeLagAttnTrfCfsTask -> SeqVaeLagAttnCfsTask -> SeqVaeLagAttnFsTask -> SeqVaeLagAttnTrfRwsTask ->
SeqVaeLagAttnRwsTask -> LightningModelBase`, and

`LagAttnTrfCfsTrainer -> LagAttnCfsTrainer -> LagAttnTrfRwsTrainer -> LagAttnRwsTrainer ->
GraphModelBase`.

**The driver's three attributes are re-pointed because all three collide**, and each failure is
silent: both parents set all three, and resolution order alone would take the causal side —

- omit `MODEL_CLS` and the driver builds a conv-LSTM model with no error anywhere: a run that looks
  like this package and is not;
- omit `TASK_CLS` and the same, one layer up;
- omit `CHECKPOINT_STEM` and it writes `lag-attn-cfs-*.ckpt`, interleaving two models' checkpoints in
  whichever output tree they share.

Everything else arrives by resolution order, across both layers at once, and where each comes from is
a decision rather than an accident:

| From the **causal** parent | From the **conv-Transformer** parent | From the shared driver |
| --- | --- | --- |
| `TARGET_FIELDS = ("fhr_st", "fhr_ph")` | `compile_model_requested` | `PLOT_CONFIG_KEY` |
| `TRACKED_METRICS`, 99 entries | `_build_trainer_kwargs`, the step-granular LR monitor | the DDP strategy selection |
| `preflight`, five refusals | the five encoder constructor refusals | the callback assembly |
| the three page seams and the budget figure | | |

**`_build_model_kwargs` and `create_model` are defined on *both* parents, and both run.** Each calls
`super()`, so the linearisation threads the conv-Transformer's contributions — re-admitting
`source_attention_window: null`, applying `lr_warmup_steps` — underneath the causal one's: the four
resolved warm-up tuples, the geometry log line, the seed the tile phase is derived from, and the
resolved budget handed to the task. A reader who assumes "resolves to the causal side" also loses the
transformer half is wrong, and `tests/test_trainer.py` asserts both halves fire rather than only the
outermost.

**`compile_model_requested` resolves to the conv-Transformer side, and that is a decision.** The
causal parent does not define it, so lookup passes through and `torch.compile` becomes permitted on a
model whose causal ancestor never exercised it. That is the right outcome — it is the transformer
encoder that makes compilation worth having, and the LSTM that defeated inductor is gone — but it
arrives by resolution order rather than by anything written down, so `tests/test_trainer.py` asserts
it explicitly. Shipped configs keep `compile: false` regardless.

**`PLOT_CONFIG_KEY` stays `"lag_attn_rws_plotting"`**, and the config block keeps that name. The
shared callback assembly reads the literal, so a sibling that renames it to match its own package
gets no figure, no error and nothing in the log saying why.

## 8. Step-wise causality, unconditionally

**This is the one claim of this package that is genuinely stronger than the conv-LSTM causal cell's
rather than merely inherited.** That cell needs `causal_norm: true` to make step-wise causality hold
— without it a time-pooling normaliser mixes the whole sequence, the "prior" conditions on the
future, and the source-conditioned KL is not a coupling readout at all. These encoders have no such
normaliser: `RMSNorm` reduces over channels only, convolutions pad left before a `padding == 0`
`Conv1d`, and attention is causal by kernel flag (target, full prefix) or by an explicit band mask
(source, $W_U = 16$). **`causal_norm` is not a constructor keyword of this model at all**, so there
is no flag to condition the claim on, and a reader should not go looking for the one the sibling's
config names.

`tests/test_causality.py` measures it through the assembled model at the shipped warm-up budget and
at the tiny fixture, together with prefix equivalence
$\mathcal{E}(X_{0:T-1})_t = \mathcal{E}(X_{0:t})_t$.

**Two causalities meet in this cell and they are independent.** Token causality — $H_t = f(X_{\le t})$
— is what the architecture guarantees, and it is what every two-sided cell delivers. Raw-signal
causality is what the *transform* delivers here: a stored coefficient at $t$ is a function of
$\{x(s) : s \le t\}$. The transform can be causal and the encoder not, which is exactly what
`causal_norm: false` would produce on the conv-LSTM cell, and the reverse is what every two-sided
cell is. This cell has both, which is why its coupling readout is the one measurement in the grid
that is not qualified by the inputs containing their own future — though it is still named
`source_conditioned_kl_raw` and still not called a transfer entropy, because §14's group-delay note
stands.

## 9. Structural constraints that are not preferences

Re-asserted against *this* class rather than assumed to have survived the composition. Where an
invariant is a parent's own, this package's suite imports and re-parametrises the sibling module that
owns it rather than restating the assertions.

| Property | What enforces it | Test |
| --- | --- | --- |
| **No decoder bypass** — gradient reaches the decoder only through $z$ | `BaselineFutureDecoder.forward` takes exactly one tensor, at $d_z$ in-features | `tests/test_invariants.py` |
| **Source purity** — the prior never sees the source, the source state never sees the target | separate gates, adapters and encoders; the posterior is a residual on the prior | `tests/test_invariants.py` |
| **Exact zero KL at initialisation** | posterior deltas zeroed **after** the generic init; one shared $\epsilon$ | `tests/test_invariants.py` |
| **One decoder, invoked twice** | the same module object on $z^p$ and $z^q$ | `tests/test_invariants.py` |
| **Token causality**, unconditionally | §8 | `tests/test_causality.py` |
| **No target coefficient scored twice in a step** | the tiled anchor set partitions the timeline; padded slots repeat and are marked invalid | `tests/test_forward_contract.py` |
| **The lag attribution identity**, $\sum_\ell \widetilde K_{t,\ell} = K_t$ | the lag attention is built at `dropout=0.0` | the architecture parent's suite, over this same forward |

**Because the delta heads are zero-initialised, any KL assertion on a freshly constructed model
passes vacuously**; the `perturb_posterior` fixture is load-bearing for every test in this package
that claims to check KL behaviour.

**The zero-KL claim states its fixture's flags**, because it is conditional. The shipped config ships
`base_decode: mean` — under which the two *forecasts* are no longer bitwise identical, though the KL
is still exactly zero, since that depends on the two distributions and not on the samples drawn from
them — and `posterior_logvar_mode: independent`, under which the init KL is zero only with
`head_init_calibration: true`.

## 10. DDP reachability

Production runs under plain `"ddp"` with `find_unused_parameters=False` under `gaussian_nll`, so
every parameter must be reachable; `mse` starves the decoder's log-variance heads and selects
`ddp_find_unused_parameters_true`. `tests/test_ddp_strategy.py` measures both, on the guarded *and*
the ungated arm, and measures that under `mse` the starved set is **exactly** those heads.

A parameter multiplied by an identically-zero tensor **is** reachable: its `AccumulateGrad` node
fires and the reducer marks it ready. What breaks `find_unused_parameters=False` is a parameter left
out of the graph by a Python-level branch on a tensor value. So the availability terms are added
unconditionally in the forward and the branching happens at construction time only.

**Under the shipped alignment both streams build a start embedding, and that is a change from the
unaligned arm.** The adapter branches on $\min_c \delta_c$, and in the causal cells it is told
$\delta_c = W'_c + d_c$ rather than $W'_c$ alone; the alignment lifts that minimum from $0$ to $80$
steps on the target *and* on the source, so each adapter gains one learned $d_{\mathrm{model}}$-wide
start-of-record vector — the $+2 \times 128$ term §13's alignment delta is built from. It is a
construction-time decision either way, the term is added unconditionally in the forward, and every
sample carries the full $300$-step sequence from step $0$, so the vector is reached on every batch of
every rank. `use_up_st: false` together with a warm-up budget stays refused, but its stated premise —
that `up_st` is the block reaching $W' = 0$ and dropping it is what flips the indicator into
existence — now describes the **unaligned** arm only; on the aligned arm the indicator already
exists, and the refusal is a retained guard rather than the binding one.

This package writes no `forward`: the encoders and blocks come from the architecture parent, the
adapters from the shared net layer and the tiled forward from the causal parent.
`tests/test_ddp_reachability.py` therefore asserts the rule where it is *reachable* — it walks
`AvailabilityInputAdapter.forward` and requires every conditional in it to test whether a module was
built (`is None` / `is not None`) rather than to read a tensor value. Its two remaining
start-embedding tests have **not** caught up with the paragraph above: they still assert the
pre-alignment claim — no indicator on either stream at the shipped budget, and a source-only
negative control whose shift vector no longer matches the channel count it rebuilds at — and both
fail on the aligned geometry. That is a stale assertion about the alignment, not a defect in the
model.

The per-segment phase is derived per rank from that rank's own samples and introduces no collective —
asserted by searching the phase derivation for `all_reduce`, `all_gather`, `broadcast`, `barrier` and
`dist.` rather than by describing it — and $A_{\max}$ is a geometry constant at every phase, so no
rank can disagree on shape and no shape is a function of the data. **`broadcast_buffers=False`** is
justified as it always was: every buffer — rotary tables, causal masks, the gates' keep-indices, the
adapters' availability patterns, the raw-target index grid — is a deterministic function of the
config, and there is no `BatchNorm` anywhere. `static_graph` is deliberately absent: the loss-spike
breaker substitutes a zero-weighted sum over every parameter on a skipped batch, which is a
structurally different backward from the one the first iteration recorded.

## 11. What is drawn

The three per-run figures and the offline tradeoff curve are the causal parent's, reached through the
task's page seams — `forecast_rows`, `input_stream_panels`, `forecast_extra_rows` and the
`input_budget_figure` method; `lag_attn_cfs/DESIGN.md` §11 is the record, and §11.1 is the page's
fifteen-row inventory. This package ships no `plotting.py` and no `sample_page.py`, which
`tests/test_sample_page.py` asserts as a directory check — near-vacuous the day it was written, and
the thing that fails when someone later reaches for a local copy.

What the test does exercise is that the page is reached **through two levels of inheritance**: the
seams resolve off the task, the task resolves them off the causal parent, and the shipped input-row
builder — welded to the production two-sided Morlet bank — must not be the one that runs. Its failure
is inside a handler that warns and continues, so the assertion is on the *absence of the warning*
rather than on the presence of the rows.

## 12. Configuration

`configs/` ships exactly `default.yaml`, `tiny.yaml`, `smoke_hie.yaml`, `planted.yaml` and eight
sweep arms — `sweep_anchor_stride_1.yaml`, `sweep_align_target_max.yaml`,
`sweep_lag_bias_decay.yaml`, `sweep_lag_kv_adapter.yaml`, `sweep_source_dropout_02.yaml`,
`sweep_source_dropout_03.yaml`, `sweep_target_clock_input.yaml` and
`sweep_legacy_dualref_physclock.yaml` (the pre-2026-09-05 default, kept as a comparator) — and the directory listing
itself is asserted. Each is written out in full rather than inheriting: a `base:` chain would be the smaller
file and the worse record, because it hides which settings this run shares with the models it is
compared against, and that sharing is the whole value of the square.

**The arm inventory is closed, one axis per file.** Each sweep config is the default plus its one
named leaf plus the two identity keys (`run_name`, `tags.variant`) that put the arm in the run's own
directory name. The unaligned arm is the one exception and its second leaf is *forced* rather than a
second delta: the resolver refuses a source reference against an unaligned target by name, so an arm
moving only `causal_align_reference` would not launch. The two source-dropout arms exist here and
not on the conv-LSTM cell, because that seam — the one that regularises the source map without
touching the target pathway — is named on this architecture.

**The planted delta is an instrument rather than an arm.** It is the identifiability check's own
geometry: tiny channel widths at the **production** lag window, `max_lag: 90`, because the planted
delay has to sit strictly inside $(H, L-1)$ and the smoke delta's `max_lag: 8` makes that interval
empty. Its alignment arm is pinned — `causal_align_reference: target_max` with
`causal_align_reference_source: null` — and held fixed, deliberately: the shipped source clock would
shift the plant's readable band $25$ steps earlier and censor half of it below lag $0$, so a default
flip must not be able to move the instrument.

The cost is drift, so the shipped config is pinned **leaf-for-leaf against both neighbours** outside
a declared per-edge allow-list, in both directions — an exemption that is no longer a divergence
fails as loudly as a divergence that is not exempt. The square closes iff the two allow-lists are
disjoint outside the identity keys and the one constant both edges move: a key exempt on **both**
edges is one neither comparison can read, and it would be exempt for two reasons nothing forces to
agree. The one deliberate asymmetry is that `additive_margin` must *equal* the conv-LSTM causal
cell's, because the encoder edge changes neither the block nor the anchor count, while
`gradient_clip_val` is a gradient statistic and moves on both edges.

The five conv-LSTM-only keys — `lstm_layers`, `encoder_extra_dilations`, `encoder_extra_kernel`,
`conv_norm_groups`, `causal_norm` — are absent from every config here and name no argument of this
constructor, so each would be dropped by the signature sweep **without a word**, leaving a config
that reads correct and builds a different model. The seven encoder keys this architecture adds —
`encoder_conv_kernels`, `encoder_conv_dilations`, `encoder_num_heads`, `encoder_d_ff`,
`target_attention_blocks`, `source_attention_blocks`, `source_attention_window` — are inherited from
the raw domain where they were swept.

**Two of those seven apply to the `encoder` K/V arm alone**, and they are kept rather than removed.
`source_attention_blocks` and `source_attention_window` describe the deep source encoder, which the
shipped `lag_kv_source: conv_stem` does not build; the shipped `conv_stem` reads
`encoder_conv_kernels` and `encoder_conv_dilations` instead. They stay because they are exactly what
the `encoder` comparison arm needs and because removing them would make the leaf-for-leaf parity
comparison against both neighbours read a divergence where there is none — with a comment at each
saying when it applies, so a reader editing one under `conv_stem` is told it changes nothing.

`causal_reach_budget_s` is present and required `null`, and `causal_warmup_budget_steps: 134` is the
guard this dataset actually needs; the resolver refuses the two together by name.

**A run's own artifacts state every independently-toggleable mechanism.** Each is its own
configuration key and a leaf of `model_config.VAE_model`, so all of them land verbatim in the
resolved-config artifact the run writes beside its checkpoints, and the five constructor ones land
again in the checkpoint's `model_kwargs`. `causal_leg_alignment` names which phase-harmonic operator
built the phase blocks and is compared against the shards' own root attribute;
`causal_align_reference` and `causal_align_reference_source` name the target's clock and the
source's, resolving to $\tau^y_{\mathrm{ref}} = 402.1604$ s and $\tau^u_{\mathrm{ref}} = 288.2672$ s
from the data at an offset of $-113.8932$ s; `lag_kv_source`, `prior_availability_input`,
`persistence_residual`, `horizon_weight_halflife_steps` and `alibi_slope_scale` name the five
architecture switches. The resolved *consequences* land in the startup log's budget summary — both
references in seconds, the offset, the shift range and the surviving counts per block, per stream —
and the evaluation console prints the configured arm label, both resolved clocks, the offset and
`lag_kv_source` on the same block as the delay, with the same readings in `summary.json`. No tracked
metric is added for any of them: they are constants of the configuration, so a per-step column would
be the same value in every row.

**Printing the configured value beside the built one is a guard rather than a courtesy.** The
signature sweep above **silently drops** any config key the class does not re-list, so an arm whose
key never reached this constructor would train as the baseline with nothing in its log saying so.
The sweep-arm configs additionally carry the arm in `run_name` and `tags.variant`, so a run's own
directory name states which arm it is.

**$\tau_{\mathrm{ref}}$ is the clock the shift is *reported* against; $\kappa\,\tau_{\mathrm{ref}}$
is the one it *aligns* on.** The shards' stored `causal_delay_s` is the envelope mean
$\tau_g = \gamma / (2\pi b)$, which is the right number to report as staleness and the wrong one to
align on: the lag a channel actually realises is the energy centroid of its impulse response,
$(2\gamma - 1)/(4\pi b) = \kappa\,\tau_g$ with $\kappa = 1 - 1/(2\gamma) = 0.875$ at the bank's
$\gamma = 4$. The resolver therefore scales the *difference* by `ALIGNMENT_DELAY_FACTOR` before
quantising it,

$$d_c = \operatorname{round}\!\left(\frac{\kappa\,(\tau_{\mathrm{ref}} - \tau_c)}{\Delta}\right),
\qquad \Delta = 4\ \mathrm{s},$$

and scales the difference rather than the reference so that the reference channel keeps shift $0$
and the keep-index does not move at all — $\tau_c \le \tau_{\mathrm{ref}}$ is scale-invariant, so the
same four source channels are dropped and every count in §13 is exactly what it was. The bias the
factor removes is **one-sided rather than symmetric jitter**: within a channel's own passband
$\tau_g(\nu) = \tau_g(\xi)\, b^2 / \big(b^2 + (\nu - \xi)^2\big)$ is *maximal* at the centre
frequency, so the spectrum-weighted average can only pull the realised lag **down**. Measured on the
aligned shard, the median realised-to-reported ratio is $0.9032$ over $30$ resolved channels and
$0.882$ over the nine slow ones the $4$ s grid barely quantises, against $0.875$ predicted by the
centroid and $1.000$ predicted by $\tau_g$.

What moves is the shift *magnitude*, and only that. On both streams $d_c$ now spans $0$–$85$ steps
rather than $0$–$97$; $\min_c (W'_c + d_c)$ falls from $91$ to $80$; the worst quantisation residual
is $1.9865$ s, still inside the $\Delta/2 = 2$ s half-step bound and now measured against the scaled
difference. $\max_c (W'_c + d_c) = 134 = \max_c W'_c$ still holds **exactly** on both streams, which
is the zero-marginal-warm-up lemma the anchor floor stands on, so the floor is unchanged and no total
in §13 moved.

The clock the aligned stream lands on is therefore $\kappa\,\tau_{\mathrm{ref}} = 0.875 \times
402.1604 \approx 351.9$ s of *realised* delay rather than the $402.2$ s the reference channel
reports. **The transform side is deliberately not on this convention.**
`hdf5_dataset/causal_scattering.py`'s `leg_alignment_shift`, which aligns the two legs of a phase
pair before the coefficients are stored, still uses the unscaled $\tau_g$: applying $\kappa$ there
would change stored coefficients and require a dataset rebuild, so it was left alone. Only the
model-side alignment carries the factor. That is a stated limitation rather than an oversight, and
it is why a reader comparing a stored `causal_delay_s` against a model-side shift finds them a
factor of $0.875$ apart.

## 13. Parameter budget

Measured on constructed models at the shipped warm-up budget, not predicted. `tests/test_docs.py`
re-measures every total below by constructing the models rather than comparing against literals.

**Two rows per cell, and both are the record.** The **shipped** row is the revised default: local
K/V, the prior clock, the persistence residual, the weighted horizon axis, the flat lag-bias seed
and the two alignment clocks. The **off-state** row is every switch at its inert value with the
single shared clock restored, which is bitwise the model that shipped before this revision — and it
is the row on which the target-axis comparison against `lag_attn_transformer_fs` is still readable,
because that cell is two-sided and never takes the new keys.

| | conv-LSTM encoders | conv-Transformer encoders |
| --- | ---: | ---: |
| **causal feature, shipped** ($C_{\mathrm{keep}} = 98$, $c_u^{\mathrm{keep}} = 39$) | $4{,}655{,}987$ | $\mathbf{4{,}284{,}556}$ |
| causal feature, shipped but one shared clock ($c_u^{\mathrm{keep}} = 47$) | $4{,}658{,}035$ | $4{,}286{,}604$ |
| causal feature, shipped but unaligned ($c_u^{\mathrm{keep}} = 51$) | $4{,}658{,}803$ | $4{,}287{,}372$ |
| causal feature, shipped but ungated ($C = 102$, $c_u = 51$) | $4{,}642{,}419$ | $4{,}270{,}988$ |
| causal feature, `lag_kv_source: encoder` at the shipped clocks | $5{,}163{,}866$ | $5{,}072{,}524$ |
| causal feature, `lag_kv_source: adapter` at the shipped clocks | $3{,}851{,}635$ | $4{,}183{,}564$ |
| **off-state, guarded and aligned** ($c_u^{\mathrm{keep}} = 47$) | $5{,}146{,}334$ | $\mathbf{5{,}054{,}992}$ |
| off-state, guarded and unaligned ($c_u^{\mathrm{keep}} = 51$) | $5{,}147{,}102$ | $5{,}055{,}760$ |
| off-state, ungated ($C = 102$) | $5{,}130{,}598$ | $5{,}039{,}256$ |
| two-sided feature, guarded ($C_{\mathrm{keep}} = 78$) | $5{,}126{,}326$ | $5{,}034{,}984$ |

**Ungated means the whole guard**, the warm-up mask and both common clocks together: a shift vector
is positional over the *survivors*, so a stream with no keep-index has no width for one to be
positional against. **The alignment costs $-768$**, the same in every one of the four causal cells —
the source adapter loses four channels from two $d_{\mathrm{model}}$-wide linears at $-1{,}024$, and
both adapters gain a start-of-record vector at $+256$ (§10). **The second clock costs a further
$-2{,}048$**, eight more source channels off the same two linears at $-8 \times 128 \times 2$, with
the target stream contributing nothing. Both the unaligned and the single-clock rows are named
comparison arms rather than history: they are what `causal_align_reference: null` and
`causal_align_reference_source: null` still build.

### What the revision costs on this cell, factorised

**At fixed clocks the five architecture switches cost $-768{,}388$ here** against $-488{,}299$ on
the conv-LSTM sibling, and the whole of that difference is the two stems:

| Term | conv-Transformer | conv-LSTM | What it is |
| --- | ---: | ---: | --- |
| the deep source encoder, not built | $-888{,}960$ | $-1{,}312{,}231$ | the whole windowed attention stack, and nothing else consumed it |
| the local K/V stem, built | $+100{,}992$ | $+804{,}352$ | `GatedCausalConvStem` at $(5, 9)$ / $(1, 2)$ here; five conv blocks at the conv-LSTM schedule there |
| the prior's clock | $+16{,}640$ | $+16{,}640$ | $2 \times 128$ for its own `LayerNorm` and $128 \times 128$ for the bias-free projection |
| the persistence weight | $+2{,}940$ | $+2{,}940$ | $H \times C_{\mathrm{keep}} = 30 \times 98$, one raw parameter |
| the horizon weight | $0$ | $0$ | a non-persistent buffer, so not a parameter and not a state-dict key |
| the flat lag-bias seed | $0$ | $0$ | the same $(\text{num heads}, L)$ parameter, seeded differently |

$-888{,}960 + 100{,}992 + 16{,}640 + 2{,}940 = -768{,}388$. A delta that does not decompose into
exactly these means something else moved.

**This cell's stem is genuinely local and the conv-LSTM cell's is not, and that is what makes the
encoder edge worth reading here.** Each stem reuses *its own* parent encoder's convolution schedule,
so the two arms differ in what is removed rather than in two independently chosen front ends. Here
that schedule is $(5, 9)$ at dilations $(1, 2)$, reaching
$1 + \sum_b (k_b - 1) r_b = 21$ steps — $84$ s, against a $91$-lag window. On the conv-LSTM cell it
is $(3, 5, 11, 15, 15)$ at $(1, 2, 4, 8, 16)$ and reaches **387 steps**, longer than the lag window
and longer than the sequence, so there the arm bounds unbounded recurrence rather than whole-prefix
memory. Any statement about what localising the K/V does to a lag readout has to be read on **this**
cell.

### The guard, at the shipped configuration

**The guard costs parameters here and saves them on the two-sided sibling, and both numbers are
right.** Guarded minus ungated is $+13{,}568$ on the shipped model against $-9{,}662$ on
`lag_attn_transformer_fs`, because the two guards drop very different numbers of channels. Here the
budget drops $4$ target channels of $102$ and the alignment $12$ source channels of $51$:

$$\underbrace{128 \times 98}_{\text{target availability}}
+ \underbrace{128 \times 39}_{\text{source availability}}
+ \underbrace{2 \times 128}_{\text{start embeddings}}
- \underbrace{514 \times 4}_{\text{decoder head}}
- \underbrace{128 \times 4}_{\text{target input linear}}
- \underbrace{128 \times 12}_{\text{source input linear}}
- \underbrace{30 \times 4}_{\text{persistence weight}} \;=\; +13{,}568 .$$

On the off-state row, where the source keeps $47$ channels and the persistence weight is not built,
the same identity is the six-term one and sums to $+15{,}736$:

$$\underbrace{128 \times 98}_{\text{target availability}}
+ \underbrace{128 \times 47}_{\text{source availability}}
+ \underbrace{2 \times 128}_{\text{start embeddings}}
- \underbrace{514 \times 4}_{\text{decoder head}}
- \underbrace{128 \times 4}_{\text{target input linear}}
- \underbrace{128 \times 4}_{\text{source input linear}} \;=\; +15{,}736 .$$

The reach budget drops $31$ of $109$, so on the two-sided sibling the narrowing dominates instead.

### The two axes

**The encoder axis: $-371{,}431$** at the shipped configuration, guarded against guarded, and
identical across the shipped, single-clock, unaligned and ungated rows — so it is still the two
history stacks alone and nothing else. Both stacks contribute: the target encoders differ by
$+331{,}929$ as they always did, and the two local stems by $-703{,}360$. **At the off-state row the
same axis is $-91{,}342$**, a $1.8\%$ reduction, and *that* is the number identical to the reduction
the same two encoders buy in the two-sided feature domain and in the raw domain at the same budget.
The comparison against the other two rows of the grid is therefore only meaningful on the off-state
row, where every cell carries the same source module; that is stated rather than left for a reader
to trip over. That reduction was $1{,}304{,}782$ — $38.4\%$ — before the capacity revision, which
raised the conv-Transformer encoders and left the conv-LSTM ones alone.

**The target axis: $+20{,}008$** against `lag_attn_transformer_fs` on the **off-state** row, and it
is the same number the conv-LSTM pair shows there, because every module outside the encoders is
shared. It decomposes into exactly **two** terms, each measured parameter by parameter:

| Term | Value | What it is |
| --- | ---: | --- |
| the decoder's output head | $+10{,}280 = 514 \times (98 - 78)$ | two per-channel output rows plus their biases, at $d_{\mathrm{hidden}} + 1 = 257$ each |
| the two input adapters | $+9{,}728$ | $128 \times (98 - 78)$ and $128 \times (47 - 29)$ on the input linear *and* the availability projection |

$10{,}280 + 9{,}728 = 20{,}008$. `horizon_depth` is **not** a term — it stays at $4$ across both —
so a delta that does not decompose into exactly these two means something else moved. **It is read
on the off-state row deliberately:** `lag_attn_transformer_fs` is two-sided and never takes the new
keys, so at the shipped configuration the difference between the two models is dominated by
mechanisms one of them does not have.

**The start-embedding term used to be the adapters' third and is now exactly zero.** It contributed
$-256$ while this cell built neither vector and `lag_attn_transformer_fs` built both; the alignment
builds both here too (§10), so the two models agree and the term vanishes — which, together with the
source stream narrowing to $47$ on that row, is why the target-axis delta fell by precisely $768$.

**The horizon embedding used to be the third term and is now exactly zero.** It contributed
$-3{,}840 = -15 \times 256$ while this cell forecast one minute against the two-sided cell's two;
both now forecast $30$ steps, so the two embeddings are the same size and the term vanishes. That is
why the target-axis delta grew by precisely $3{,}840$. `tests/test_docs.py` still computes the term
as $(30 - 30) \times 256$ and asserts it is zero, so a future horizon divergence reappears as a
failing sum rather than as a silently wrong total.

Both causal off-state rows gained exactly $+3{,}840 = 15 \times 256$ when the horizon moved to $30$,
and that was the **whole** parameter cost of doubling the forecast: the horizon embedding was the
only tensor in the model whose shape carried $H$. The persistence weight of §5 is now a second one,
at $30 \times 98$. Everything else in the decoder — the projection MLP, the dilated refine stack,
the FiLM generators, the two horizon-attention blocks, the mean and log-variance heads — is shared
across horizon tokens. The two-sided row did not move, because it was already at $30$.

## 14. Deliberate limitations

- **The nats are edge-dependent, budget-local and horizon-local.** §5. Recorded, not fixed.
- **`_mu_gap_rms` is overridden onto the tiled anchor set**, inherited from the causal parent.
  `lag_attn_cfs/DESIGN.md` §10 records why: left dense, `mu_post_prior_gap_rms` would average the
  latent belief shift over all $136$ anchors while the `source_conditioned_kl_raw` printed beside it
  averages over $\approx 4.6$, and the two are read against each other. The longer horizon widened
  that ratio from $15\times$ to $30\times$, so the override matters more than it did.
- **Group-delay compensation is per channel, not per channel pair.** The second `lean-limit:` note
  below.
- **No encoder arms and no $\beta$ arms ship here.** The encoder values were swept in the raw domain
  and the encoders are reached by import; the $\beta$ pair is the causal parent's and is under the
  same open question there. Re-opening either means a scratch overlay, deliberately.
- **No mixed precision** (`precision: "32-true"`), and **`compile: false`** although the key is live
  rather than inert (§7): inductor may reassociate float arithmetic, and `pred_gap` is a difference
  of order $10^{-1}$ between two block NLLs of order $10^{3}$.
- **Every local measurement is in-sample.** `dataset_kwargs` is shared between the two loaders and
  cannot carry a per-split GUID filter.
- **Checkpoint compatibility with the pre-revision model is deliberately broken.** The constructor
  and state-dict changes mean a pre-revision blob does not load into a shipped-configuration model.
  `load_checkpoint_strict` refuses rather than partially loads and `check_model_class` still guards
  the class, so the failure is by name; no migration shim is built, because the off-state arm exists
  for exactly the case where the old weights are wanted.
- **The two cells' local K/V stems are not the same size and cannot be made so without breaking the
  edge.** Each reuses its own parent encoder's schedule, which is what keeps the arm "the encoder
  with its deep stage removed" on both sides; giving one of them a schedule of its own would make
  the encoder edge a comparison of two chosen front ends rather than of two architectures. §13
  prices the consequence: this cell's stem reaches $21$ steps and the conv-LSTM cell's $387$.

> lean-limit: the prior's clock cancels no part of the KL's **mean** term, because the posterior is
> a bounded residual on the prior and that term is a function of the delta head alone; replace with
> a delta defined as $D(a) - D(a^\varnothing)$ -- and `posterior_logvar_mode` back to `residual` for
> the variance half -- when the owner accepts a change to the posterior parameterisation the whole
> coupling readout is defined on.

`lag_attn_cfs/DESIGN.md` §8.2 carries the decomposition, and it is a property of the shared
parameterisation rather than of either encoder, so it holds identically on both cfs cells.

> lean-limit: the persistence residual ships for the feature-target cells only; replace with an
> anchored raw persistence input when the raw-target cells' fast-step NLL shows the same
> suppression signature on a trained revised run.

> lean-limit: the causal warm-up is paid once per $22$-minute segment rather than once per recording,
> costing roughly half the available forecast supervision; replace with a left-context rebuild when a
> measured run shows the anchor floor rather than the source pathway is the binding constraint on
> `pred_gap`.

Prepending left-context before the transform would drive every $W'$ to zero and return $c_y$ to $109$
and $c_u$ to $58$, making the causal-versus-two-sided comparison exact. **The model side needs no
change against such a dataset**: the budget resolves to every channel and the floor to the model's
own $30$. `lag_attn_cfs/DESIGN.md` §14 carries the cost.

> lean-limit: the lag map is an attribution over stored-coefficient time whose bias is now the single
> constant $\kappa(\tau^u_{\mathrm{ref}} - \tau^y_{\mathrm{ref}})$ plus a one-sided intra-band
> residual, rather than a $1167$ s pair-dependent range; replace with a physical-lag readout when a
> nonzero target reference can be divided out, which in this cell it cannot — the target is itself a
> stored coefficient on the aligned clock.

**The residual is one-sided, not $\pm$ anything.** An earlier form of this note called it a symmetric
$\pm 16\%$ intra-band spread; that framing is wrong. $\tau_g(\nu)$ is maximal at a channel's centre
frequency (§12), so a channel's own passband can only pull its realised lag **below** the $\tau_g$
the shard reports — never above it. The alignment now removes the deterministic part of that bias by
scaling every shift by $\kappa = 0.875$, and what is left is the spread of the *residual* ratios
around $\kappa$, which the measurement in §12 puts at a median of $0.9032$ across resolved channels.

One-sidedness and zero latency are different properties and this family buys only the first. The
**forecast** claim is exact under that — a coefficient at $t$ is a function of the past, so
forecasting $t + 1 + \tau$ from history up to $t$ is a genuine forecast whatever the internal latency
— but $\mathrm{lag}_{\mathrm{physical}} = \Delta\ell + \kappa(\tau^u_{\mathrm{ref}} -
\tau^y_{\mathrm{ref}}) + \mathrm{shift}_{\mathrm{preprocessing}}$. The alignment (§3) collapsed both
$\tau$ terms from channel-dependent quantities of the same order as the $364$ s lag search itself to
one reference each, and the dual scheme makes their difference a **known non-zero constant**,
$\kappa \cdot (-113.8932) = -99.65$ s, which travels through the preflight causality record and is
printed on the evaluation console beside the delay. What it did not do is make the target reference
zero, which is what a lag between *signals* would need. Every lag-resolved caption says so.

> lean-limit: `lag_floor` is configuration for a value no shipped arm varies; replace with a
> constructor-level constant, or with a resolved source floor, when a run's
> `source_lag_warmth_frac_ph` falls below the fraction of searched lags that are warm at all.

### What the evaluation closed, and what it deliberately did not

There is now an evaluation pipeline for this cell, and — like everything else here — almost none of
it is in this package. `eval/` supplies four files: `TRF_CFS_BINDING`, an override delta that is the
causal parent's key for key, a runner that names this model's registry, and a gate that delegates in
full. **This package defines no numeric function**, asserted about the classes in
`tests/test_eval_run.py` rather than described here, because that is exactly what makes a difference
against `lag_attn_cfs` attributable to the encoder rather than to two implementations.

Everything the evaluation closes it closes for both cfs cells and is recorded once, in
`lag_attn_cfs/DESIGN.md` §14: per-recording quantities with bootstrap intervals in place of
per-epoch scalars, the availability-clock control no permutation control can see, the anchor
geometry as a FAIL-able verdict, feature-space trivial predictors, the measured lag support, the
band-resolved skill readout, the input-level occlusion analysis, the near edge read as censoring
rather than as inertness, the per-head profiles on the page, the run's arm on the page, and the
availability-clock margin now set at $0.15$ nats so its verdict decides. `eval/EVAL.md` here is
short by design and defers to
`teb_vae/lag_attn_cfs/eval/EVAL.md` for all of it. The deliberate absences are that document's too —
no phase-domain analysis, no deceleration skill or triggered response, no clinical unit, no
physical-lag readout — and they are absences of the shared pipeline rather than of this binding.

Two things are this cell's own:

- **`geometry_keys` is the causal parent's minus `causal_norm`**, which is not a constructor
  parameter here because these encoders carry no time-pooling normaliser to causalise, **plus** the
  seven encoder keys §12 lists. Each must be both a constructor parameter and a config key or
  `preflight.reconcile` skips it silently, which is a reconciliation that never happens and never
  says so. The revision adds three to both cells' tuples — `prior_availability_input`,
  `lag_kv_source` and `persistence_residual` — because the evaluation rebuilds the architecture from
  the checkpoint's own `model_kwargs`, so a config disagreeing about one of them would not fail: it
  would report one architecture's numbers under another's stated name.
  `horizon_weight_halflife_steps` is deliberately **not** there, on the same ground as the objective
  weights, because no evaluated readout applies it.
- **The comparability rule is asymmetric and both halves are stated.** A loss level *is* comparable
  against `lag_attn_cfs` — same $2940$-coefficient block, same $136$ dense anchors, same objective,
  same shards, same warm-up budget, same horizon, only the encoders differ — and is *not* comparable
  against `lag_attn_transformer_fs`, whose blocks are $2340$ coefficients at the *same* horizon of
  $30$. The cross-cell table carries the first edge only. The target edge's single remaining
  divergence is $C_{\mathrm{keep}}$, which is what the warm-up budget decides.

**`clock_margin_min_nats` is set, at $0.15$ nats**, so `coupling_exceeds_availability_clock` is a
real gate and the acceptance set is ten criteria rather than nine. Both cfs cells' override files
carry the same value and the same provenance comment — the diagnosed unaligned run's observed
$\Delta_{\mathrm{clock}} = 0.160$, interval $[0.157, 0.164]$ — because a margin set independently
per cell would gate two architectures the two cfs cells exist to compare against two different bars.
The margin's provenance is the unaligned arm; the gated quantity is right on both arms.

## 15. Deviation record

Where the built package differs from the design it was built from, and why.

- **The alignment reference was silently reverted to `null` and has been restored, and the window is
  recorded rather than erased.** Between the commit that reverted it and the one that put it back,
  this cell's config carried `causal_align_reference: null` while every document here — §3, §10,
  §12, §13 and `RESULTS.md`'s parameter table — described an aligned model. Nothing in the documents
  was wrong about the *design*; the configuration had stopped implementing it, so a reader who
  checked one against the other during that window found them contradictory and had no way to tell
  which side was stale. `tests/test_config_load.py` was the visible symptom: three of its cases
  failed for the whole window and all $45$ pass now. The entry stays here because the failure mode
  is not the revert but the silence — the aligned/unaligned pair is a *named comparison arm*, so a
  config sitting on the unaligned value looks exactly like a deliberate arm selection unless the
  record says otherwise.
- **This package writes no network code and copies none.** The model is a constructor, the task is
  empty, and the driver is three class attributes. That is what makes a difference against
  `lag_attn_cfs` attributable to the encoder alone and a difference against
  `lag_attn_transformer_fs` to the transform alone — and it is the reason the empty bodies are
  asserted as facts about the classes rather than described in prose.
- **The conftest is spliced from two siblings, and which half comes from which is not
  interchangeable.** The constructor keyword sets are written here at the conv-Transformer schema;
  the data half — the committed causal shard, the config builder, the tiny warm-up staircase, the
  budget resolver, the stub batch carrying `guid` and `epoch`, the seeded streams at the one-sided
  widths — is imported from the causal sibling, because it describes the *dataset* and the *target
  domain*, neither of which is a property of an encoder. The two halves meet at a named tuple of
  geometry keys that `tests/test_fixtures.py` asserts agree; a disagreement would build a model
  neither parent's suite tests, with no shape differing, because $A_{\max}$ and the block width are
  geometry constants either way.
- **Taking the causal suite's keyword sets instead fails *asymmetrically*, which is why it is named.**
  That suite's tiny set carries one of the five conv-LSTM-only keys and its shipped set carries four
  more, so the tiny path would fail on one keyword and the shipped path on five — and every failure
  would name a keyword rather than the conftest.
- **The gradient clip moved on the encoder edge and the additive margin did not, and both are
  measurements.** The constants stated in nats of the summed block must not move across an edge that
  changes neither the block nor the anchor count; a gradient statistic must. `RESULTS.md` carries the
  table with its "moved on the encoder edge?" column.
- **`raw_per_step` stays**, for the geometry reason in §3, and `decoder_out_channels` is deliberately
  absent from the config and from `model_kwargs` — it is not a keyword of this constructor at all.
  The width is recoverable from the stamped `target_keep_index`, and a second field could disagree
  with the gate.
- **`tiny.yaml` shrinks the widths under two independent constraints.** The constructor validates
  `num_heads * d_head == d_model` for the **lag-attention** heads, while `encoder_num_heads` is
  unrelated to `num_heads` and carries its own requirement that `d_model / encoder_num_heads` be
  even. The two products coincide in the shipped set only because both head counts happen to be $4$.

### What moved in this revision, against the record it replaces

The six mechanisms are the shared parents' and `lag_attn_cfs/DESIGN.md` §15 lists them once. Three
consequences are **this** cell's and are recorded here, because they are statements the earlier
record of this package made:

- **"`geometry_keys` is twenty-two rather than sixteen" no longer names the count**, because the
  revision added three keys to both cells' tuples. §14 states which three and why the fourth is
  deliberately absent; the count is left to the code, which is where a reader can check it.
- **Two of the seven encoder keys now apply to a comparison arm rather than to the shipped model.**
  `source_attention_blocks` and `source_attention_window` describe a module the shipped
  `lag_kv_source: conv_stem` does not build. They are kept, commented and still compared
  leaf-for-leaf (§12), so the parity check keeps reading a genuine divergence rather than an absence.
- **The encoder edge is measured on a different number now, and the earlier one is still the right
  one for a different question.** $-91{,}342$ is the two *deep* encoder stacks and is read on the
  off-state row; $-371{,}431$ is the two shipped history stacks, stems included, and is read on the
  shipped row. §13 states both and says which comparison each belongs to, because quoting one where
  the other belongs is the easiest mistake this table now admits.

## 16. Running it

From the repository root.

```bash
# Production, 7 ranks. TEB_RUN_STAMP is required so ranks 1..N-1 share rank 0's run directory,
# and the rank count must equal len(general_config.cuda_devices).
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_transformer_cfs.trainer \
    --config teb_vae/lag_attn_transformer_cfs/configs/default.yaml

# Local smoke: one epoch, one device, the committed causal fixture.
python -m teb_vae.lag_attn_transformer_cfs.trainer \
    --config teb_vae/lag_attn_transformer_cfs/configs/tiny.yaml

# Dev-box validation: the shipped geometry over a causal HIE sample shard.
python -m teb_vae.lag_attn_transformer_cfs.trainer \
    --config teb_vae/lag_attn_transformer_cfs/configs/smoke_hie.yaml

# Can this architecture recover a delay it is known to be looking at? Fits the committed
# planted-delay fixture for a few epochs and reads the lag profile back through the evaluation's
# own code. This cell is the one whose local K/V stem is genuinely local, so it is the cell that
# argument gets tested on. Accepts either cfs cell's planted config.
python teb_vae/lag_attn_cfs/lag_recovery_check.py \
    --config teb_vae/lag_attn_transformer_cfs/configs/planted.yaml

# Score a finished run against the criteria RESULTS.md registers, while it is still in flight.
python -m teb_vae.lag_attn_cfs.check_run --run-dir <run>

# Evaluate a finished checkpoint on the held-out population: one reviewable directory.
python -m teb_vae.lag_attn_transformer_cfs.eval.run \
    --checkpoint <run>/model_checkpoints/<name>.ckpt

# Gate that directory offline, on a box with no torch installed.
python -m teb_vae.lag_attn_transformer_cfs.eval.verify <run>/eval_results/summary.json
```

`RUN_CONFIG` near the bottom of `trainer.py` names the config used when the module is launched with
no command line, so the entry point works from an IDE's Run button with the only operator action
being to edit a value inside the file; `eval/run.py` and `eval/verify.py` follow the same convention
through their own `RUN_ARGS` dicts.

`check_run.py` and `eval/verify.py` answer two different questions: the first reads a run's
`metrics_history.csv` and says whether the fit behaved, in-sample and per epoch, while it is still
going; the second reads `eval_results/summary.json` and says whether the finished checkpoint is
acceptable, on a held-out population, per recording, with intervals. Neither substitutes for the
other, and `eval/EVAL.md` carries the same cross-reference.

The gate:

```bash
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_cfs/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_cfs/tests -q -m slow
```

---

## Amendment (2026-09): the forecast-target clock

This cell inherits the forecast clock exactly as it inherits the rest of the target domain: by
composing the same two mixins. `causal_target_forecast_clock` re-indexes the *question* — the
scored element at anchor $t$, horizon step $\tau$, kept channel $c$ reads stored step
$t + 1 + \tau + s_c$, with the signed per-channel shift resolved by
`teb_vae.lag_attn_cfs.causal_warmup` and applied in
`teb_vae.lag_attn_cfs.nets.causal_feature_target`. The full statement — the three clocks, the
generalised floor, the anchor ceiling, the persistence clamp, the pooled validity, the
checkpoint/kwargs contract, and the lag-axis re-registration under `physical` — lives in the
conv-LSTM cell's `DESIGN.md` amendment; nothing about it is encoder-specific, which is what lets
one copy of it be correct for both cells.

What this cell's own records must carry:

- The shipped default was `physical` at `anchor_stride: 5` (ceiling $270 - 85 = 185$, dense
  anchors 51, ~10 training tiles per sample at $A_{\max} = 11$) until 2026-09-05; that
  configuration is now `sweep_legacy_dualref_physclock.yaml`, and the default scores the `stored`
  clock at `anchor_stride: 13` (ceiling 270, dense anchors 136, the same 11 tiles; see the
  amendment at the end of this document). `sweep_target_clock_input.yaml` is the
  aligned-continuation arm; `planted.yaml` pins `stored`.
- The pre-registered reading in §9.2 of `DIAGNOSIS.md` and in the config header — "alignment
  should NOT improve the forecast gap" — is **amended**: it holds only within stored-clock arms,
  because re-clocking the question changes what every arm is scored on.
- **Deviation from `LAG_READOUT_DIAGNOSIS.md` §11, recorded rather than silent.** That section
  says a change re-reading the cell's pre-registered edges lands as a new versioned package. This
  change lands **in place, config-gated**, by explicit decision: the stored clock remains
  reachable at one key and resolves byte-for-byte the historical cell, so both edges of the
  six-cell square stay readable on the stored-clock arms, which is the property §11 exists to
  protect.
- The diagnostic page's input rows overlay their own raw signal delayed onto the row's clock, as
  the causal parent's do (the conv-LSTM cell's `DESIGN.md` amendment *the raw overlay on the
  input rows*), and the shared evaluation runner attaches the resolved budget to the loaded task
  so the delays reach the page on this cell as well.


## Amendment (2026-09-05): the corrected representation is the shipped default

`default.yaml` now carries what the CFS review recommended and the task list CFS-09 built as a
baseline: the integer phase operator (`causal_phase_operator: integer_harmonic_v1`, so the phase
blocks are $44$ `fhr_ph` and $10$ `up_ph` and the declared widths are $c_y = 80$, $c_u = 46$), **no**
input-channel alignment (`causal_align_reference: null`, `causal_align_reference_source: null`:
every channel is read at its own availability time), the **stored** forecast clock (labels are the
next $H = 30$ stored coefficients, exact availability time, ceiling $T_{\mathrm{valid}} = 270$, dense
span $136$) and `anchor_stride: 13`, which tiles that span into the same $A_{\max} = 11$ tiles the
physical-clock geometry had at stride $5$. At the shipped budget the kept widths are $76/80$ target
($32$ `fhr_st` + $44$ `fhr_ph`) and $46/46$ source, so the block is $30 \times 76 = 2280$
coefficients; no nat from this default is comparable to a $2940$-coefficient row above. The scattering
coefficients themselves are unchanged by the operator.

The reasons are mathematical rather than empirical: the fractional $2^{3/2}$ phase family is
discontinuous at the principal-angle branch; the $0.875$ input-alignment convention is not an exact
content clock and withholds up to $340$ s of the fastest channels' trajectories from the encoder;
and the `physical` clock is an approximate delay compensation that costs $85$ trailing anchors. The
configuration this file shipped before is preserved verbatim as `sweep_legacy_dualref_physclock.yaml`
(legacy shards, `REPOINT_ME_causal`); `sweep_align_unaligned.yaml` and `sweep_target_clock_stored.yaml`
were deleted because the default now is them, and `tiny.yaml` reads the committed integer-operator
fixture. `sweep_align_target_max.yaml` now *adds* the single-reference alignment and
`sweep_target_clock_input.yaml` carries that alignment with it, since the `input` clock copies an
input shift. The guard bands above that read `anchors_per_sample` $\in [10, 11]$ train / $51$ val
are the legacy arm's; the promoted default's are $[10, 11]$ train / $136$ val. Shards for the default
are built with `create_new_pipeline.py` under `phase_operator=integer_harmonic_v1`
(`REPOINT_ME_causal_int`); they also carry the horizon-free `causal_novelty_curve`.

## Amendment (2026-09-05, later the same day): the horizon moves to 10 steps, the tiling to stride 5

`default.yaml` now forecasts $H = 10$ stored steps ($40$ s) instead of $30$, and so does
`teb_vae/lag_attn_cfs/configs/default.yaml` (mirrored the same day, so the encoder edge still differs in
the encoder alone); the two-sided cells still forecast $30$. The change is a configuration decision
and is stated in the config; the constructor default stays the architecture parent's $30$ and no
network code moved.

**What follows from it, and was re-derived rather than left behind.** On the stored clock the anchor
ceiling is $T_{\mathrm{valid}} = 300 - H = 290$, so the dense span $[134, 290)$ holds $156$ anchors
(against $136$). `anchor_stride` moves $13 \to 5 = H/2$, which tiles that span into
$A_{\max} = 32$ tiles at phase $0$ and $31$ at every other phase (mean $31.2$, against $10$–$11$
before); consecutive windows overlap by exactly half a horizon, so every scored coefficient is seen
from two anchors per step, and the per-step training tensors do not grow —
$(B, 32, 10, 76)$ is $320$ anchor-steps per sample against $(B, 11, 30, 76)$ at $330$, while the
dense validation decode falls from $136 \times 30$ to $156 \times 10$. `horizon_weight_halflife_steps`
moves $15.0 \to 5.0$, keeping the $H/2$ rule: the resolved weights run $1.7260$ at the first step to $0.4957$ at the last (a $3.5\times$ spread) where they ran $1.8063$ to $0.4729$ before. `horizon_depth`
stays $4$: the criterion is $\mathrm{RF} \ge H + 1 = 11$ and $\mathrm{RF} = 31$ clears it by twenty tokens; depth $3$ ($\mathrm{RF} = 15$) now clears it too, so depth $4$ is kept for parity rather than necessity. The block is $10 \times 76 = 760$
coefficients, so the two loss-scale constants stated in nats of the block were **scaled by
$760/2940$ from the recorded $H = 30$ measurement rather than re-measured**: `gradient_clip_val`
$14000 \to 3500$ (the same rounding rule on the scaled quantiles) and `additive_margin`
$9.0 \times 10^{3} \to 2.2 \times 10^{3}$, which is what keeps the additive test *live* — the
reachable magnitude of the two reconstruction terms at this block is $\approx 2.4 \times 10^{3}$,
and the previous margin had already exceeded the $\approx 7.2 \times 10^{3}$ bound of the promoted
$2280$-coefficient block. Both are to be re-derived from the headline run's own `train/grad_norm`
and `main_loss` columns.

**What it costs the square.** Nothing on the encoder edge: both causal cells sum the same
$760$-coefficient block over the same $156$ anchors under the same weights, so §5's statement that a
loss level is comparable there still holds, and `tests/test_config_load.py` pins the four horizon
leaves (`horizon`, `anchor_stride`, `horizon_weight_halflife_steps`, `additive_margin`) leaf-for-leaf
against the conv-LSTM cell. The target edge, against `lag_attn_transformer_fs`, now differs in the
horizon as well as the block: `horizon` is exempt on that edge, and a per-horizon-step reading shares only its first $10$ steps with the two-sided cell. The guard bands for `anchors_per_sample` are now
$[31, 32]$ train / $156$ val.

**What does not move.** The diagnostic page derives everything it draws from the model —
`geometry.horizon`, `anchor_stride`, the anchor ceiling and the decoded anchor set — so the forecast
rows, the training-tile fan and the anchor overlay follow the new geometry with no code change; the
smoke fit in `tests/test_train_smoke.py` renders it at $H = 10$, $S = 5$. `planted.yaml` **pins**
`horizon: 30` and `horizon_weight_halflife_steps: 15.0` explicitly, because the planted delay of $45$
stored steps makes the readable band $[\delta - H, \delta - 1] = [15, 44]$ only at $H = 30$; both
twins pin the pair. `sweep_legacy_dualref_physclock.yaml` deliberately
does *not* pin the horizon: it compares the representation, the input references and the clock
against the default, which is only readable with the horizon held equal — under $H = 10$ its
physical-clock ceiling is $205$, its span $71$ anchors and its stride-$5$ tiling $14$–$15$ tiles.
