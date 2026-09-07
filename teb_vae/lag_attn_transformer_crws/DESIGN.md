# `lag_attn_transformer_crws` — the as-built design record

The conv-Transformer causal-input raw-target lag-attention VAE-TEB: which parent supplies which half,
what it consumes, what it returns, what it optimises, and every place the built package differs from
the design it was built from.

Companion documents, none of them restated here: `teb_vae/lag_attn_crws/DESIGN.md` records the
input domain, the anchored raw target, the input-warmth policy, the source compromise and the four
added readouts this model composes **by import and unchanged**, and its §6 carries the binding record
of every member reached by reference; `teb_vae/lag_attn_transformer_rws/DESIGN.md` records the
encoders, the adapters, the attention blocks and the wiring it composes them over;
`teb_vae/lag_attn_rws/DESIGN.md` records the raw target, the objective and the architecture every
cell of the raw-target row shares, and `teb_vae/lag_attn_rws/model_explained.md` the latent
factorisation; `teb_vae/lag_attn_cfs/DESIGN.md` records the causal input machinery both cells of
this row inherit. `RESULTS.md` in this directory carries the pre-registered criteria and, once there
are runs, the measurements.

**What this document is for.** Reading it should leave a reader able to say which of two comparisons
a given number of this model belongs to — which is a different question from how the architecture
works, and the one a model assembled entirely out of imported parts makes easy to get wrong.

---

## 1. What the model is

The eighth cell of an encoder-by-target grid, and the conv-Transformer half of the one row in which
neither side of the objective contains its own future:

```
                          conv-LSTM encoders          conv-Transformer encoders
  raw FHR target          lag_attn_rws                lag_attn_transformer_rws
  two-sided feature       lag_attn_fs                 lag_attn_transformer_fs
  causal feature          lag_attn_cfs                lag_attn_transformer_cfs
  causal in / raw out     lag_attn_crws               lag_attn_transformer_crws   <- this
```

At each admitted 4-second anchor $t$ the model forecasts the next **two minutes of raw FHR** —
$H \cdot R = 30 \times 16 = 480$ raw samples — twice, from a target-only latent and from a
source-conditioned one, through one shared decoder invoked twice. That is the `lag_attn_crws` input
handling and anchored objective reached through the `lag_attn_transformer_rws` encoders, and nothing
else.

**Its value is that it closes the row.** With both cells present either axis can be read at a fixed
value of the other: against `lag_attn_crws` the configurations differ in the **encoder** alone, and
against `lag_attn_transformer_rws` in the **input representation** alone. Neither of the two
three-cell configurations allowed that, and it is the entire reason this cell exists.

The whole model is

```python
class SeqVaeLagAttnTrfCrws(CausalRawInputs, SeqVaeLagAttnTrfRws): ...
```

with **a constructor and nothing else** — `vars(SeqVaeLagAttnTrfCrws)` carries `__init__` and no
other callable and no class constant — linearising as
`SeqVaeLagAttnTrfCrws -> CausalRawInputs -> CausalWarmupInputs -> SeqVaeLagAttnTrfRws -> Module ->
object`. §7 records why the constructor is the one exception.

**It is an experiment, not a remedy.** `lag_attn_rws/PRIOR_SCALE_AND_PREDICTIVE_GAP.md` §5
establishes that the held-out predictive gain is negative because the source pathway does not
generalise — a failure in the source encoder, the lag attention and the posterior fusion, and an
encoder swap of this kind removes a confound rather than that failure. This model is expected to
reproduce it. **The sign of `pred_gap` is a criterion nowhere in this document.**

At the shipped configuration the model holds **4,218,476 parameters**, against **4,589,907** for the
encoder-axis comparison `lag_attn_crws`. With every architecture switch of `lag_attn_crws/DESIGN.md`
§17 at its off-state it holds **4,989,804** against that cell's **5,081,146**, which is bitwise the
pair that shipped before this revision and is the row on which the input-representation comparison
against `lag_attn_transformer_rws` (**5,003,116**) is still readable. Both causal off-state totals
fell when this row moved its alignment reference off the feature row's `target_max` and down to
$42.21$ s (§3), which drops most of both input streams: at `target_max` the same two off-state
models hold **5,012,844** and **5,104,186**, and that is the arm the family's shared test fixture
still resolves. §13 carries the arithmetic and decomposes every delta.

**Four mechanisms of this cell are configuration rather than architecture**, each gated by a key
whose off-state reproduces the pre-revision model bitwise. The inventory, the off-states and where
each is pinned are `lag_attn_crws/DESIGN.md` §17's and are not restated; both cells of this row take
the same four keys at the same four values, because all four are the shared causal parent's, and
both decline the same two — the decoder's persistence residual and a second alignment reference.
What is **not** shared is what one of them builds: under `lag_kv_source: conv_stem` this cell's
local K/V stem is a `GatedCausalConvStem` over this architecture's own `encoder_conv_kernels` /
`encoder_conv_dilations` schedule, reaching $21$ steps against the conv-LSTM cell's $387$, and §13
is where the difference that makes is priced.

## 2. Input contract

Identical to `lag_attn_crws` in every field, width and refusal — the causal HDF5 shards through
`train/data_module.py::GraphDataModule` at `trim_minutes: 1.0`; `fhr` at $(B, 4800)$ as the target,
$c_y = 36 + 66 = 102$ and $c_u = 36 + 15 = 51$ as the inputs, `weight`, `guid` and `epoch` loaded,
and `fhr_up_ph` absent by construction. `lag_attn_crws/DESIGN.md` §2 is the record and it is not
restated, because the input contract is a property of the *dataset and the target* and neither is a
property of an encoder. What the encoder change touches here is nothing at all.

Three requirements are repeated because a config for this package is a separate file that can lose
them independently:

- **`fhr` must be in `load_fields` and `normalize_fields`, and `weight` in `load_fields`.** The
  target must arrive z-scored or the Gaussian NLL is meaningless with the loader raising nothing, and
  the validity signal is the only trustworthy gap marker for a trace whose gaps are stored as $0$
  bpm. The shared entry point's guard reads the driver it was handed and resolves `TARGET_FIELDS`
  through the **causal** parent (§7), which inherits it unchanged from the shared ancestor as
  `("fhr",)`.
- **`guid` and `epoch` must be in `load_fields`**, because the tile phase is keyed on the pair and
  `load_fields` is honoured literally.
- **The shards must be the causal variant.** The two variants share every field name and every dtype;
  only the root `transform` attribute and the channel counts tell them apart, and the warm-up
  resolver refuses a two-sided shard by name before a run directory exists.

## 3. Geometry, tiling and the warm-up

All three are the conv-LSTM cell of this row's, reached by import: the input-warmth policy $F \ge
B - 1$ and $F \ge \max_c(W'_c + d_c)$, which at the shipped reference resolves to $B = 1$ and
$\max_c(W'_c + d_c) = 6$ on **both** streams and so requires no more than $F \ge 6$; the shipped
$F = 134$ is kept anyway, so the floor is now a retained anchor-cost policy rather than a
constraint. $38$ of $102$ target-stream input channels survive and $17$ of $51$ source ones; the
tiled anchor set $\mathcal{A}(\varphi) = \{F + \varphi + kS\}$ at $S = H = 30$ with the phase
derived per segment from a `blake2b` of `guid`, `domain_start`, the epoch and the seed;
$A_{\max} = 5$ at the training geometry and $136$ at the dense evaluation one; the anchored raw
gather; and the input warm-up mask applied inside `AvailabilityInputAdapter`.
`lag_attn_crws/DESIGN.md` §§3, 4 and 7 are the record.

**Why this row's reference is $42.21$ s and not the grid's `target_max`, and why that is forced.**
A raw target passes through no filter, so $\tau^y \equiv 0$ and there is nothing on the target side
to cancel the source reference out of the physical-lag identity

$$\tau^{\mathrm{phys}}(\ell, h) \;=\; \Delta\,(\ell + 1 + h) \;+\; \kappa\,\tau_{\mathrm{ref}}
\;-\; \tau_{\mathrm{pre}}, \qquad \Delta = 4\ \mathrm{s}, \quad \tau_{\mathrm{pre}} = 20\ \mathrm{s}.$$

That expression is *minimised* at $\ell = h = 0$ and grows with the lag. At the feature row's
`target_max` reference, $\tau_{\mathrm{ref}} = 402.1604$ s, the smallest lead the attention can
express is $335.9$ s, so the $20$–$120$ s contraction-to-deceleration band is unreachable at
**every lag index** — and raising `max_lag` cannot help, because it only makes the lead longer.
The causal-feature cells do not have this problem: there the same reference appears on both sides
of the identity and cancels, so they keep `target_max` and reach the band at
$\ell + h \in [9, 34]$. `configs/default.yaml` therefore names $42.21$ s, and the resolver snaps
it to the shard's own `float32` $42.206562$ — refusing a literal further than $\Delta / 2$ from
any kept channel, so the key stays a *name for a channel* rather than a number that would keep
resolving shifts against a transform the shards no longer have.

**What that buys and what it costs.** The shift carries the centroid factor
$\kappa = 1 - 1/(2\gamma) = 0.875$ at gammatone order $\gamma = 4$ (`lag_attn_cfs/DESIGN.md` §3
is the derivation and the measurement), so the aligned stream's common *effective* delay is
$\kappa\,\tau_{\mathrm{ref}} = 36.93$ s rather than $42.21$ s, and the reachable lead runs
$[20.9,\ 496.9]$ s: $99$ of the $100$ s band is inside the lag axis, reached at
$\ell + h \in [0, 24]$, with the next index landing at $120.9$ s just outside it. The price is most
of both input streams. The target keeps $17$ `fhr_st` and $21$ `fhr_ph` channels of $36 + 66$; the
source keeps $17$ `up_st` and **none** of the $15$ `up_ph`, whose fastest channel sits at
$150.79$ s — every band-reachable reference loses that whole block, so contraction morphology
leaves the source stream entirely. Both streams' shifts span $d_c \in [0, 6]$ and both survivor
sets wait $W'_c \in \{0, 1\}$, which is where $B = 1$ above comes from.

**`warmup_period` is deliberately left at $134$.** It clears the requirement of $6$ more than
twenty times over, so nothing in this cell's validity depends on it any more; lowering it toward
$\sim 7$ would roughly double the anchors per sample, which is a training-cost change to be taken
with a run rather than with a key, and no arm ships for it.

Two facts about *this* architecture are worth stating where a reader will look for them.

**`TrimmedRawGeometry` is reused unchanged and `raw_per_step` is the decoder's width.** The loader
delivers `fhr` at $4800$ and `weight` at $300$, `raw_masks._validate_weight` checks
`weight.size(1) == geometry.t`, and the decoder emits $R = 16$ raw samples per horizon token — so
unlike the causal-feature cell of this encoder family, `future_index` is inherited **and read**: the
anchored gather indexes into it, asserted by `data_ptr` identity.

**The lag validity floor is the causal-input cell's too.** `lag_floor` ships at $0$, where the mask
is bitwise the architecture parent's, and exists so the source compromise of
`lag_attn_crws/DESIGN.md` §8 is measurable rather than argued about.

## 4. Forward return dict

`SeqVaeLagAttnTrfCrws.forward(y_st, y_ph, u_stream, anchor_phase=None, anchor_stride=None)` returns
**twenty-two keys** — the causal-input parent's exactly, which are in turn the raw-signal family's
twenty plus the two the anchor axis needs.

- `mu_base`, `logvar_base`, `mu_full`, `logvar_full` — $(B, A_{\max}, H, R)$, so $(B, 5, 30, 16)$
  at the shipped training geometry and $(B, 136, 30, 16)$ at the dense one, and $16$ wide at every
  budget.
- `anchor_index` $(B, A_{\max})$ `long` and `anchor_valid` $(B, A_{\max})$ `bool`.
- `mu_prior`, `logvar_prior`, `raw_logvar_prior`, `mu_post`, `logvar_post`, `z_prior`, `z_post` —
  $(B, 300, 64)$; `target_state`, `source_state` — $(B, 300, 128)$; `attended_source_heads`
  $(B, 300, 4, 32)$; `attn_weights` $(B, 300, 4, 91)$; `kld_per_t`, `kld_per_t_per_head`,
  `source_kl_lag_map`, `mu_prior_sat_frac`, `delta_mu_sat_frac` — all unchanged in shape.
  **`source_state` is the lag attention's K/V tensor** rather than the deep source state by
  definition: under `lag_kv_source: encoder` those are the same object, and under the shipped
  `conv_stem` it is the stem's output. Both intervention controls follow it, resolving the path off
  model attributes, so neither signature moved.

**No `decoder_state` and no `delta_mu_src`**, as in both parents: there is no bypass to carry one.
**And no `persistence`**, because this row declines the feature-target cells' residual — so the
twenty-two keys are the whole contract at every configuration, where in those cells the count moves
with a switch.

## 5. Loss, metrics and what the nats are summed over

The objective, its $\beta$ schedule, its metric surface, the validation-only permutation control, the
source-null arm, the spike-breaker wiring and the checkpoint contract are the causal-input parent's —
the same code, not a copy. `LagAttnTrfCrwsTrainer.TRACKED_METRICS` carries **77** entries and
resolves to the causal parent's tuple by identity; `lag_attn_crws/DESIGN.md` §10 records what the
four added readouts separate, why `anchors_per_sample` is a guard rather than a result, which eight
of the causal-feature cell's columns are dropped and why, and why `_mu_gap_rms` is overridden onto
the tiled anchor set.

**`lambda_boundary` is refused at any non-zero value**, inherited from the causal-input parent's
pre-flight: the boundary term is a slicing identity over *adjacent* anchors, and this family always
decodes a set whose entries are $S$ apart. `lambda_ms` and `lambda_deriv` are inherited at $0.1$
from the raw-signal side, because they are within-block raw-waveform quantities on the same grid.

**`horizon_weight_halflife_steps: 15.0` weights the reconstruction on the horizon axis**, threaded
through the objective exactly as the feature cells' channel weight is and renormalised to
$\sum_\tau w_\tau = H$ so the block score's scale — and with it `gradient_clip_val`,
`additive_margin` and $\beta$'s standing — survives. `lag_attn_crws/DESIGN.md` §5 is the record.
Two consequences belong here rather than there: the encoder edge stays readable only because both
cells of the row ship the same half-life, and the caution that a weighted block score is not a
log-density has no unweighted second reading beside it, since this row has no evaluation pipeline.

### What the nats are, and are not, comparable to

This is the cell where the distinction earns its place, because the two edges are not symmetric:

- **The encoder edge, against `lag_attn_crws`: a loss *level* is comparable.** Both cells sum the
  same $480$ raw samples over the same anchor count under the same objective from the same shards at
  the same warm-up budget, so `total_loss`, `nll_base_block` and `pred_gap` may be read against each
  other directly.
- **The input-representation edge, against `lag_attn_transformer_rws`: a loss level is *not*
  comparable.** The block is $240$ against $480$ and the decoded anchors per training step are about
  $10.1$ against $240$, so only a sign, a trajectory, the bottleneck-health columns, the parameter
  budget and the *ordering* of arms carry across. `sweep_horizon_15.yaml` exists one package over
  on the conv-LSTM cell for exactly this reading; no horizon arm ships here (§12).
- **Comparable across warm-up budgets within this model**, because a raw decoder is $R$ wide at every
  budget — though two arms at two budgets still have **mutually unloadable checkpoints**, since the
  input adapters are built at the surviving widths.

## 6. Why one mixin and not an inheritance

**The obvious construction does not work, and the failure is silent.** `SeqVaeLagAttnCrws`
subclasses `SeqVaeLagAttnRws`, while `SeqVaeLagAttnTrfRws` derives from `nn.Module` directly. So
`class X(SeqVaeLagAttnCrws, SeqVaeLagAttnTrfRws)` linearises as
`X -> Crws -> ... -> Rws -> TrfRws -> Module` and **runs the conv-LSTM constructor**: a model that
builds, trains and reports, and is not this architecture.

So the input domain lives in one plain object, `CausalRawInputs` in `teb_vae/lag_attn_crws/nets/`,
**which mentions no encoder**, and both cells of this row are their own architecture plus that one
object. It is a *move*, not an abstraction: no `Protocol`, no `__init_subclass__`, no member that
was not lifted verbatim. What it prevents is a second copy of the input domain whose drift would be
silent — the delay trap of §7 changes no shape and raises nothing.

**One mixin, not two, and the missing one is the point.** The causal-feature cells compose a target
mixin beside the input one because a stored-coefficient target changes the decoder's width, its
gather and its readouts. A raw target changes none of them: `_default_decoder_out_channels` already
returns `raw_per_step` on the architecture parent, which is exactly this cell's width, so the correct
thing to do about the width hook is to not define one — and neither `CausalRawInputs` nor this class
does. Compose the causal-feature cell's `CausalFeatureForecastTarget` in by mistake and the decoder
is built at $C_{\mathrm{keep}}$ — the *surviving target-input* width, $38$ under the shipped
reference (§3) — against a $(B, A, H, 16)$ target, and `raw_sample_score` computes
$(\text{target} - \mu)^2$ on shapes that do not broadcast, three frames below the decision that
caused it. `tests/test_construct.py` builds that wrong composition and reads `mean_head.out_features`
back as whatever target keep width its own fixture resolved, so the trap is pinned rather than the
number.

**The order of the bases is load-bearing.** The mixin comes first, which is what makes the tiled
forward win method resolution over the architecture parent's dense one, the warm-up adapter win over
the gate's shift vector alone -- all zeros only on an unaligned arm -- and the anchored
`compute_loss` win over the dense raw one.
Reversed, the model would decode the dense anchor range, return no anchor set at all, and score a
$(B, T_{\mathrm{valid}}, H, R)$ target against a $(B, A_{\max}, H, R)$ forecast.
`tests/test_construct.py` pins the `__mro__` as a list.

**There is no `decoder_out_channels` keyword, and that is the one property this cell has that its
conv-LSTM twin does not.** The architecture parent declares none — the decoder emits $R$ raw samples
per horizon token, full stop — so no configuration can put the decoder and the raw target on
different widths. On `lag_attn_crws` the keyword survives from its architecture parent's signature
and a config setting it fails at the first batch rather than at the config;
`lag_attn_crws/DESIGN.md` §6 records the limit.

## 7. The three diamonds, and their measured resolution order

```python
class SeqVaeLagAttnTrfCrwsTask(SeqVaeLagAttnCrwsTask, SeqVaeLagAttnTrfRwsTask):
    pass

class LagAttnTrfCrwsTrainer(LagAttnCrwsTrainer, LagAttnTrfRwsTrainer):
    MODEL_CLS = SeqVaeLagAttnTrfCrws
    TASK_CLS = SeqVaeLagAttnTrfCrwsTask
    CHECKPOINT_STEM = "lag-attn-trf-crws"
```

The task defines **zero** callables. The driver re-points three class attributes and defines no
method. All three linearisations are asserted as lists of class names against the real `__mro__`:

`SeqVaeLagAttnTrfCrws -> CausalRawInputs -> CausalWarmupInputs -> SeqVaeLagAttnTrfRws -> Module`,

`SeqVaeLagAttnTrfCrwsTask -> SeqVaeLagAttnCrwsTask -> SeqVaeLagAttnTrfRwsTask -> SeqVaeLagAttnRwsTask
-> LightningModelBase`, and

`LagAttnTrfCrwsTrainer -> LagAttnCrwsTrainer -> LagAttnTrfRwsTrainer -> LagAttnRwsTrainer ->
GraphModelBase`.

**The driver's three attributes are re-pointed because all three collide**, and each failure is
silent: both parents set all three, and resolution order alone would take the causal side —

- omit `MODEL_CLS` and the driver builds a conv-LSTM model with no error anywhere: a run that looks
  like this package and is not;
- omit `TASK_CLS` and the same, one layer up — and the step-granular learning-rate ramp this
  architecture needs would never be reached, because that task does not define it;
- omit `CHECKPOINT_STEM` and it writes `lag-attn-crws-*.ckpt`, interleaving two models' checkpoints in
  whichever output tree they share. All nine drivers of the family carry distinct stems, asserted.

Everything else arrives by resolution order, across both layers at once, and where each comes from is
a decision rather than an accident:

| From the **causal-input** parent | From the **conv-Transformer** parent | From the shared driver |
| --- | --- | --- |
| `TRACKED_METRICS`, 77 entries | `compile_model_requested` | `TARGET_FIELDS = ("fhr",)` — neither parent re-points it |
| `preflight`, six refusals | `_build_trainer_kwargs`, the step-granular LR monitor | `PLOT_CONFIG_KEY` |
| `causal_standing_message` | `build_lr_scheduler`, on the task | the DDP strategy selection |
| the tiling phase, the source-null readout and the three page seams, on the task | | the callback assembly |

**`_build_model_kwargs` and `create_model` are defined on *both* parents, and both run.** Each calls
`super()`, so the linearisation threads the conv-Transformer's contributions — re-admitting
`source_attention_window: null`, applying `lr_warmup_steps` — underneath the causal one's: the four
resolved warm-up tuples, the geometry log line with the horizon receptive field, the seed the tile
phase is derived from, and the resolved budget handed to the task. A reader who assumes "resolves to
the causal side" also loses the transformer half is wrong, and `tests/test_trainer.py` asserts both
halves fire rather than only the outermost.

**`compile_model_requested` resolves to the conv-Transformer side, and that is a decision.** The
causal parent does not define it, so lookup passes through and `torch.compile` becomes permitted on a
model whose causal ancestor never exercised it. That is the right outcome — it is the transformer
encoder that makes compilation worth having, and the LSTM that defeated inductor is gone — but it
arrives by resolution order rather than by anything written down, so `tests/test_trainer.py` asserts
it explicitly. Shipped configs keep `compile: false` regardless.

**`TARGET_FIELDS` is the shared ancestor's object and neither parent re-points it.** `("fhr",)` on
every driver of the raw-target row; the causal-feature drivers re-point it to the two feature blocks,
and that is exactly the edit this row must not make. Asserted as the ancestor's object rather than as
"comes from the causal parent", which would have been true and unfalsifiable.

**`PLOT_CONFIG_KEY` stays `"lag_attn_rws_plotting"`**, and the config block keeps that name. The
shared callback assembly reads the literal, so a sibling that renames it to match its own package
gets no figure, no error and nothing in the log saying why.

**The task's diamond is well-formed today because its two branches are disjoint** — everything the
causal-input cell adds against `{build_lr_scheduler}` — and that is a fact about today's code rather
than a property of the construction. A future member defined on both sides would resolve to the
causal side by order alone, silently; `tests/test_task.py` asserts the linearisation as a list and
each behaviour against the class the design names, so a reorder fails rather than trains something
else.

## 8. Step-wise causality, unconditionally

**This is the one claim of this package that is genuinely stronger than the conv-LSTM cell of this
row's rather than merely inherited.** That cell needs `causal_norm: true` to make step-wise
causality of the *history states* hold — without it a time-pooling normaliser mixes the whole
sequence, the "prior" conditions on the future, and the source-conditioned KL is not a coupling
readout at all. These encoders have no such normaliser: `RMSNorm` reduces over channels only,
convolutions pad left before a `padding == 0` `Conv1d`, and attention is causal by kernel flag
(target, full prefix) or by an explicit band mask (source, $W_U = 16$). **`causal_norm` is not a
constructor keyword of this model at all**, so there is no flag to condition the claim on, and a
reader should not go looking for the one the sibling's config names.

`tests/test_causality.py` measures it through the assembled model at the shipped warm-up budget and
at the tiny fixture, together with prefix equivalence
$\mathcal{E}(X_{0:T-1})_t = \mathcal{E}(X_{0:t})_t$ — which the warm-up mask cannot break, being a
function of $t$ alone.

**Two causalities meet in this cell and they are independent, and here they meet a third fact.**
Token causality — $H_t = f(X_{\le t})$ — is what the architecture guarantees. Raw-signal causality of
the *inputs* is what the transform delivers: a stored coefficient at $t$ is a function of
$\{x(s) : s \le t\}$. And the **target** is the raw signal itself, so the objective scores samples
that no input has seen and no anchor mis-times. This cell has all three, which is why its forecast
claim and its lag claim are simultaneously exact on the target side — though the coupling readout is
still named `source_conditioned_kl_raw` and still not called a transfer entropy, because the inputs
carry their own group delay and §14's note stands.

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
| **No raw sample scored twice in a step** | the tiled anchor set partitions the timeline; padded slots repeat and are marked invalid, and the anchored gather honours the repeat so the mask can remove it | `tests/test_forward_contract.py`, `tests/test_objective.py` |
| **The anchored target equals the dense builder at the dense set** | `gather_anchored_future_target` against `build_future_target` under `torch.equal` | `tests/test_objective.py` |
| **The lag attribution identity**, $\sum_\ell \widetilde K_{t,\ell} = K_t$ | the lag attention is built at `dropout=0.0` | `tests/test_invariants.py`, after `perturb_posterior` |

**Because the delta heads are zero-initialised, any KL assertion on a freshly constructed model
passes vacuously**; the `perturb_posterior` fixture is load-bearing for every test in this package
that claims to check KL behaviour, and the permutation control's vacuity at initialisation is itself
a passing test rather than a comment.

**The zero-KL claim states its fixture's flags**, because it is conditional. The shipped config ships
`base_decode: mean` — under which the two *forecasts* are no longer bitwise identical, though the KL
is still exactly zero, since that depends on the two distributions and not on the samples drawn from
them — and `posterior_logvar_mode: independent`, under which the init KL is zero only with
`head_init_calibration: true`.

## 10. DDP reachability

Production runs under plain `"ddp"` with `find_unused_parameters=False` under `gaussian_nll`, so
every parameter must be reachable; `mse` starves the decoder's log-variance heads and selects
`ddp_find_unused_parameters_true`. `tests/test_ddp_strategy.py` measures both, on the guarded *and*
the ungated arm, and measures that under `mse` the starved set is **exactly** those heads — at width
$R$, which is the one width no budget can move, so both guard states starve the same tensor at the
same size.

A parameter multiplied by an identically-zero tensor **is** reachable: its `AccumulateGrad` node
fires and the reducer marks it ready. What breaks `find_unused_parameters=False` is a parameter left
out of the graph by a Python-level branch on a tensor value. So the availability terms are added
unconditionally in the forward and the branching happens at construction time only.

**Under the alignment both streams build a start embedding, and unaligned neither does.** The
indicator exists when a stream's earliest honest step $\min_c (W'_c + d_c)$ is above zero. Unaligned
that minimum is $0$ on both streams — each has a channel at $W' = 0$ — so neither embedding is
built; under any reference the shift pushes it up, to $1$ on both streams at the shipped $42.21$ s
and to $80$ at `target_max`, so both are built. Either way the branch is at construction time and
the parameter, once built, is reached on every batch, which is what `find_unused_parameters=False`
requires. It is also why `use_up_st: false` together with a warm-up budget is refused: it would flip
the indicator into existence on a *stored-block* decision rather than on the shift, giving a
parameter reached only by the leading steps of a segment.

This package writes no `forward`: the encoders and blocks come from the architecture parent, the
adapters from the shared net layer and the tiled forward from the causal-input parent.
`tests/test_ddp_reachability.py` therefore asserts the rule where it is *reachable* — it walks
`AvailabilityInputAdapter.forward` and requires every conditional in it to test whether a module was
built (`is None` / `is not None`) rather than to read a tensor value. Two of its cases carried a
**pre-alignment premise** and were corrected alongside the reference change. One asserted the
*absence* of a start embedding on both streams at the shipped budget; the shift had made that false
as soon as the adapter began reading $W'_c + d_c$, which was before this row moved to $42.21$ s, so
the case had been failing against the feature cells' geometry too. The other narrowed the source
stream to build the negative control without narrowing its shift vector alongside, so `ChannelDelay`
refused the length mismatch by name and the control measured nothing at all. Both now state what the
model does.

This row's shipped minimum of $1$ is pinned rather than assumed: `tests/conftest.py` reaches the
causal-feature package's `causal_config` through the conv-LSTM sibling's wrapper, which applies
**this row's** `SHIPPED_ALIGN_REFERENCE` of $42.21$ s rather than the feature cells' `target_max`.
The sibling's
`tests/test_ddp_strategy.py::test_the_shipped_aligned_budget_builds_a_start_embedding_on_both_streams`
pins the same property at the same reference, so the two rows cannot drift on it.

The per-segment phase is derived per rank from that rank's own samples and introduces no collective —
asserted by searching the phase derivation for `all_reduce`, `all_gather`, `broadcast`, `barrier` and
`dist.` rather than by describing it — and $A_{\max}$ is a geometry constant at every phase, so no
rank can disagree on shape and no shape is a function of the data. **`broadcast_buffers=False`** is
justified as it always was: every buffer — rotary tables, causal masks, the gates' keep-indices, the
adapters' availability patterns, the two source-warmth patterns, the raw-target index grid — is a
deterministic function of the config, and there is no `BatchNorm` anywhere. `static_graph` is
deliberately absent: the loss-spike breaker substitutes a zero-weighted sum over every parameter on a
skipped batch, which is a structurally different backward from the one the first iteration recorded.

## 11. What is drawn

The nine-row page and the run-level warm-up figure are the conv-LSTM cell of this row's, reached
through the task's page seams — `forecast_rows`, `input_stream_panels` and the
`input_budget_figure` method, the last two of which are themselves the causal-feature cell's, bound;
`lag_attn_crws/DESIGN.md` §11 is the record. This package ships no `plotting.py` and no
`sample_page.py`, which `tests/test_sample_page.py` asserts as a directory check and by
`ModuleNotFoundError` — near-vacuous the day it was written, and the thing that fails when someone
later reaches for a local copy. The forecast rows and the input panel builder resolve, by object
identity, to the conv-LSTM cell's.

What the test does exercise is that the page is reached **through two levels of inheritance**: the
seams resolve off the task, the task resolves them off the causal-input parent, and the shipped raw
forecast rows — which walk a dense block at an anchor index this model's forecast does not have —
must not be the one that runs. Their failure is inside a handler that warns and continues, so the
assertion is on the *absence of the warning* rather than on the presence of the rows. The lag caveat
the page carries is the one-sided one, and it is the conv-LSTM cell's own string — one string, not
two: this cell differs from that one in the encoder, and what a lag *means* is a property of the
target domain and the transform rather than of the architecture. Under the channel alignment that
caveat states the content lead time outright, $\Delta(\ell + 1 + h) + \kappa\tau^u_{\mathrm{ref}}$ s
on the canonical stored timeline (no dataset-shift term),
because a raw target has $\tau^y \equiv 0$ and the alignment collapses the remaining bias to one
constant; `lag_attn_crws/DESIGN.md` §11 carries the derivation and the reason the number itself is
not printed on the page.

**The constant to substitute is the *effective* reference, not the configured one.** A shift carries
the centroid factor $\kappa = 0.875$ (§3), so the delay every aligned source channel actually lands
on is $\kappa\,\tau_{\mathrm{ref}}$ — $36.93$ s at the shipped $42.21$ s, not $42.21$ s. The
resolved-config dump and the caption both name the *configured* reference, which is $\tau_g$; a
reader completing the arithmetic from that number alone overstates every lead by $5.3$ s here, and
by $50.3$ s at `target_max`. At the shipped reference the whole lag axis then spans leads of
$[20.9,\ 496.9]$ s.

## 12. Configuration

`configs/` ships exactly `default.yaml`, `tiny.yaml`, `smoke_causal.yaml` and
`sweep_anchor_stride_1.yaml`, and the directory listing itself is asserted. Each is written out in
full rather than inheriting: a `base:` chain would be the smaller file and the worse record, because
it hides which settings this run shares with the models it is compared against, and that sharing is
the whole value of the row.

The cost is drift, so the shipped config is pinned **leaf-for-leaf against `lag_attn_crws`'s**
outside a declared allow-list of **nineteen** exemptions, in both directions — an exemption that is no
longer a divergence fails as loudly as a divergence that is not exempt. Five are identity; twelve are
the encoder — the five conv-LSTM keys this architecture does not have and the seven it adds, which is
the whole declared content of the edge; one is the encoder's optimisation, `lr_warmup_steps`, which
exists in every conv-Transformer sibling and in no conv-LSTM one; and **one is a measurement**,
`gradient_clip_val`, declared `RETUNED`. `additive_margin` is in `MEASURED_TO_MATCH_PATHS`
instead, beside `ema_floor` and `horizon_embed_std`: it was re-measured on this encoder and came back
to the conv-LSTM cell's value, and the list makes that equality read as a measurement rather than as
an oversight.

The five conv-LSTM-only keys — `lstm_layers`, `encoder_extra_dilations`, `encoder_extra_kernel`,
`conv_norm_groups`, `causal_norm` — are absent from every config here and name no argument of this
constructor, so each would be dropped by the signature sweep **without a word**, leaving a config
that reads correct and builds a different model; the driver refuses a config carrying any of them
with the key named. The seven encoder keys this architecture adds — `encoder_conv_kernels`,
`encoder_conv_dilations`, `encoder_num_heads`, `encoder_d_ff`, `target_attention_blocks`,
`source_attention_blocks`, `source_attention_window` — are inherited from the raw domain where they
were swept.

**No horizon arm ships**, and that is a decision. The conv-LSTM cell's `sweep_horizon_15.yaml`
exists because at $H = 30$ a nat crosses the input-representation axis unchanged, and that
comparison already exists one package over on this same encoder — `lag_attn_transformer_rws` *is*
the $H = 30$ raw-target model on these encoders. `smoke_causal.yaml` carries a fourth delta the
conv-LSTM cell's instrumented config does not need: `lr_warmup_steps: 100`, because the shipped
$2000$-step ramp outlasts the whole $600$-step run and a gradient-norm distribution measured under a
ramp is the ramp's.

`causal_reach_budget_s` is present and required `null`, and `causal_warmup_budget_steps: 134` is the
guard this dataset actually needs; the resolver refuses the two together by name.

**A run's own artifacts state both of the two independently-toggleable mechanisms.** The shard
variant and the stream reference are separate decisions and each is its own configuration key —
`causal_leg_alignment` names which phase-harmonic operator built the phase blocks and is compared
against the shards' own root attribute; `causal_align_reference` names the clock the input
channels are shifted onto and resolves to $\tau_{\mathrm{ref}}$ from the data. Both are leaves of
`model_config.VAE_model`, so both land verbatim in the resolved-config artifact the run writes
beside its checkpoints, and the resolved *consequences* land in the startup log's budget summary
(the reference in seconds, the shift range and the surviving counts per block) and in the
checkpoint's own `model_kwargs`, which carry the two shift vectors. No tracked metric is added
for the reference: it is a constant of the configuration, so a per-step column of it would be the
same number in every row.

**`causal_align_reference` is $42.21$ here and `target_max` in every feature cell, and the split is
not a preference.** §3 carries the reason — a raw target cannot cancel the source reference out of
the physical-lag identity, so `target_max` puts the entire coupling band outside the lag axis. The
key therefore reads as a literal in both cells of this row and as a name in the other six, and both
forms go through the same resolver: a literal is snapped to the nearest kept channel and refused
beyond $\Delta / 2$, so neither form can outlive a rebuilt bank silently. `null` remains the
unaligned arm and is bitwise the model that shipped before the key existed.

## 13. Parameter budget

Measured on constructed models in one process, not predicted: both cells of this row at the shipped
warm-up budget and ungated, and the two raw-signal cells they are compared against at the shipped
reach budget and ungated. `tests/test_docs.py` re-measures every total below by constructing the
models rather than comparing against literals, and attributes the stated decomposition parameter name
by parameter name.

**Two rows per configuration, and both are the record.** The **shipped** rows carry the four
architecture switches of `lag_attn_crws/DESIGN.md` §17 at their revised defaults; the **off-state**
rows carry every one of them at its inert value, which is bitwise the pair that shipped before this
revision and is the row on which the input-representation comparison against
`lag_attn_transformer_rws` is still readable, because that cell is a raw-input model and never takes
the new keys.

| | conv-LSTM encoders | conv-Transformer encoders |
| --- | ---: | ---: |
| causal in / raw out, **shipped**, budget $134$, reference $42.21$ s ($38$ of $102$ target, $17$ of $51$ source) | $4{,}589{,}907$ | $\mathbf{4{,}218{,}476}$ |
| causal in / raw out, shipped but unaligned ($51$ of $51$ source) | $4{,}613{,}715$ | $4{,}242{,}284$ |
| causal in / raw out, shipped but ungated ($102$) | $4{,}595{,}155$ | $4{,}223{,}724$ |
| causal in / raw out, shipped but `lag_kv_source: encoder` | $5{,}097{,}786$ | $5{,}006{,}444$ |
| causal in / raw out, **off-state**, budget $134$, reference $42.21$ s | $5{,}081{,}146$ | $4{,}989{,}804$ |
| causal in / raw out, off-state, `target_max` reference $402.1604$ s ($98$ of $102$, $47$ of $51$) | $5{,}104{,}186$ | $5{,}012{,}844$ |
| causal in / raw out, off-state and unaligned ($51$ of $51$ source) | $5{,}104{,}954$ | $5{,}013{,}612$ |
| causal in / raw out, off-state and ungated ($102$) | $5{,}086{,}394$ | $4{,}995{,}052$ |
| raw target, reach budget $120$ s ($78$ of $109$) | $5{,}094{,}458$ | $5{,}003{,}116$ |
| raw target, ungated ($109$) | $5{,}088{,}186$ | $4{,}996{,}844$ |

**Three named arms, not a history.** The `target_max` row is the reference every *feature* cell of
the grid still uses and this row cannot (§3); it stays here because it is the arm the family's shared
test fixture resolves and the one the encoder-axis and grid-wide comparisons were first written
against. The unaligned rows are `causal_align_reference: null`, bitwise the model that shipped before
the key existed. **Ungated means the whole guard**, the warm-up mask and the common clock together: a
shift vector is positional over the *survivors*, so a stream with no keep-index has no width for one
to be positional against.

**The four architecture switches cost $-771{,}328$ here** and $-491{,}239$ on the conv-LSTM cell of
this row, and the whole of that difference is the two stems:

| Term | conv-Transformer | conv-LSTM | What it is |
| --- | ---: | ---: | --- |
| the deep source encoder, not built | $-888{,}960$ | $-1{,}312{,}231$ | the whole windowed attention stack, and nothing else consumed it |
| the local K/V stem, built | $+100{,}992$ | $+804{,}352$ | `GatedCausalConvStem` at $(5, 9)$ / $(1, 2)$ here; five conv blocks at the conv-LSTM schedule there |
| the prior's clock | $+16{,}640$ | $+16{,}640$ | $2 \times 128$ for its own `LayerNorm` and $128 \times 128$ for the bias-free projection |
| the horizon weight | $0$ | $0$ | a non-persistent buffer, so not a parameter and not a state-dict key |
| the flat lag-bias seed | $0$ | $0$ | the same $(\text{num heads}, L)$ parameter, seeded differently |

$-888{,}960 + 100{,}992 + 16{,}640 = -771{,}328$. There is no persistence term on this row, because
this row does not take that key at all. **This cell's stem is genuinely local at $21$ steps against a
$91$-lag window and the conv-LSTM cell's is not, at $387$**: each reuses its own parent encoder's
schedule, so the arms differ in what is *removed* rather than in two chosen front ends, and any
statement about what localising the K/V does to a lag readout has to be read on this cell.
**Every number below is unchanged by the four switches**, which is the check that they and the
alignment are independent: the alignment narrows adapters and the switches replace a source module,
and neither touches the other's tensors.

**Moving the reference from `target_max` down to $42.21$ s costs $-23{,}040$**, identically in both
cells of this row, and it is four matrices and nothing else: the target adapter loses $60$ channels
and the source adapter $30$, each from an input linear and an availability projection of width
$d_{\mathrm{model}} = 128$, at $128 \times 60 \times 2 + 128 \times 30 \times 2 = 23{,}040$.

**The alignment costs $-23{,}808$ at the shipped reference** and $-768$ at `target_max` — the latter
being the number the other three causal cells share, since there the reference is the target's own
maximum and only four source channels sit above it. At $42.21$ s the target loses $60$ of its $98$
and the source $34$ of its $51$, at $-15{,}360$ and $-8{,}704$ across two $128$-wide matrices each,
against which both adapters gaining a start-of-record vector at $+256$ (§10) is a rounding error.

**The encoder axis: $-91{,}342$, a $1.8\%$ reduction, on the off-state rows** against the conv-LSTM
cell of this row, guarded against guarded. It is **identical** at every reference and at every guard,
and identical to the reduction the same two encoders buy in the raw-signal pair and in the two-sided
and causal-feature pairs, which is what a difference living entirely in the two history encoders must
look like — and it is what "the grid's two axes are independent" means numerically. The two encoder
families are near parity in budget, which makes the encoder axis a comparison of *structure* rather
than of size.

**At the shipped configuration the same axis is $-371{,}431$**, identical across the shipped,
unaligned and ungated rows, so it is still the two history stacks alone and nothing else. It moved
because *both* stacks moved: the target encoders differ by $+331{,}929$ as they always did, and the
two local stems by $-703{,}360$ where the two deep source encoders differed by $-423{,}271$. The
comparison against the other three rows of the grid is therefore only meaningful on the off-state
row, where every cell carries the same source module; that is stated rather than left for a reader
to trip over.

**The input-representation axis: $-13{,}312$** against `lag_attn_transformer_rws` at the shipped
reference **on the off-state row**, and it is the same number the conv-LSTM pair shows, because
every module outside the encoders is shared. **It is read there deliberately:**
`lag_attn_transformer_rws` never takes the new keys, so at the shipped configuration the difference
between the two is dominated by mechanisms one of them does not have. It **changed sign** with the
reference: at `target_max` the axis is $+9{,}728$,
because there the causal streams are the wider pair. It decomposes into exactly **one surviving
term**, measured parameter by parameter, and the decoder head is deliberately not one of them:

| Term | Value | What it is |
| --- | ---: | --- |
| the horizon embedding | $0$ | `nn.Parameter(torch.zeros(horizon, decoder_hidden))`; both cells forecast $30$ steps |
| the two input adapters | $-13{,}312$ | $128 \times (38 - 78)$ and $128 \times (17 - 29)$ on the input linear *and* the availability projection of each stream |
| the two start embeddings | $0$ | the reach guard builds both and, under the alignment, so does this cell (§10) |
| the decoder's output head | $0$ | `raw_per_step` in both cells |

$0 - 13{,}312 = -13{,}312$, and every parameter name whose count differs between the two guarded
models is one of the four adapter weights; a delta that does not decompose into exactly them means
something else moved. Ungated against ungated the axis is $-1{,}792$ on the two input linears at the
narrower stored widths, and that one does **not** move with the reference — an ungated model has no
shift vector at all.

**Two of those terms used to be nonzero, and both went to zero for stated reasons.** The horizon
embedding contributed $-3{,}840 = -15 \times 256$ while this cell forecast one minute against the
raw-signal sibling's two. The start embeddings contributed $-256$ while the reach guard built both
and a warm-up whose fastest channel waits zero steps built neither; the alignment's shift makes the
combined minimum $\min_c (W'_c + d_c) = 1$ on both streams, so this cell builds both too. They are
computed rather than deleted so that either divergence reappears as a failing sum.

**The guard costs**

$$\underbrace{128 \times 38}_{\text{target availability}}
+ \underbrace{128 \times 17}_{\text{source availability}}
+ \underbrace{2 \times 128}_{\text{start embeddings}}
- \underbrace{128 \times 64}_{\text{target input linear}}
- \underbrace{128 \times 34}_{\text{source input linear}} \;=\; -5{,}248$$

**here against $+6{,}272$ on `lag_attn_transformer_rws`**, and both are right, and the sign is the
whole of the difference. The reference drops $64$ of $102$ target-stream channels and $34$ of $51$
source ones, so here the narrowing of the two input linears *outruns* everything the guard adds;
the reach budget drops $31$ of $109$ and $29$ of $58$, so there the two availability projections and
the two start embeddings still dominate. At `target_max` this cell sat on the raw-signal side of that
line too, at $+17{,}792$ over $98$ and $47$ surviving channels. Unlike the feature cells, nothing in
this target domain widens a head — every parameter the guard adds is under an adapter, asserted by
name.

## 14. Deliberate limitations

- **The nats are edge-dependent, row-local and horizon-local.** §5. Recorded, not fixed.
- **`_mu_gap_rms` is overridden onto the tiled anchor set**, inherited from the causal-input parent.
  `lag_attn_crws/DESIGN.md` §10 records why.
- **Group-delay compensation is per channel, not yet in the readout.** The alignment (§3) puts
  every kept input channel of both streams on one clock, and the target carries no group delay at
  all, so the residual bias on a lag reading is the single known constant
  $\kappa\,\tau^u_{\mathrm{ref}} - \tau_{\mathrm{pre}}$; no readout divides it out yet.
  `lag_attn_crws/DESIGN.md` §14 is the record, and no attention peak may yet be read as a
  physiological delay.
- **The model-side alignment and the transform-side one are on different conventions, on purpose.**
  `channel_alignment_delays` scales the difference $\tau_{\mathrm{ref}} - \tau_c$ by the energy
  centroid factor $\kappa = 0.875$ (§3), while `causal_scattering.py::leg_alignment_shift`, which
  runs at write time inside the phase-harmonic legs, still shifts by the envelope mean $\tau_g$.
  Bringing the transform side onto $\kappa$ would change stored coefficients and require the dataset
  to be rebuilt, so it was not done. The consequence is bounded and known rather than unmeasured:
  the shards' own `causal_delay_s` remains $\tau_g$, every model-side shift and every physical-lag
  reading is $\kappa\,\tau_g$, and a reader who mixes the two overstates a lead by
  $(1 - \kappa)\,\tau_{\mathrm{ref}} = 5.3$ s at this cell's reference.
- **The two source warmth columns are saturated at the shipped reference and measure nothing there.**
  `source_lag_warmth_frac_ph` is $1.0$ over **zero** channels, because the reference keeps none of
  `up_ph` (§3) and `_resolve_block_warm_steps` reports an empty block as warm at every step by
  deliberate design — a zero there would read as a measurement rather than as an absence.
  `source_lag_warmth_frac_st` is $1.0$ for a different reason: the alignment's survivors are honest
  from step $\max_c(W'_c + d_c) = 6$, the anchor floor is $134$ and `max_lag` is $90$, so the coldest
  source step any decoded anchor can read is $44$ and every reachable lag is warm. Both columns
  become informative again only at a floor near the requirement, which is the arm §3 declines to
  ship. Read them beside the kept width per block, never alone.
- **No encoder arms and no $\beta$ arms ship here.** The encoder values were swept in the raw domain
  and the encoders are reached by import; the $\beta$ pair is the raw-signal family's and is under
  the same open question one package over. Re-opening either means a scratch overlay, deliberately.
- **No mixed precision** (`precision: "32-true"`), and **`compile: false`** although the key is live
  rather than inert (§7): inductor may reassociate float arithmetic, and `pred_gap` is a difference
  of order $10^{-1}$ between two block NLLs of order $10^{2}$.
- **Every local measurement is in-sample.** `dataset_kwargs` is shared between the two loaders and
  cannot carry a per-split GUID filter.
- **No run checker ships**, for the conv-LSTM cell's reason: no production run is in scope, so
  `anchors_per_sample` is asserted on the fixture fit by `tests/test_train_smoke.py` and read by
  hand from `metrics_history.csv` until one exists.
- **Checkpoint compatibility with the pre-revision model is deliberately broken.** The constructor
  and state-dict changes mean a pre-revision blob does not load into a shipped-configuration model.
  `load_checkpoint_strict` refuses rather than partially loads and `check_model_class` still guards
  the class, so the failure is by name; no migration shim is built, because the off-state arm exists
  for exactly the case where the old weights are wanted.
- **The persistence residual and the second alignment reference are both declined on this row.** The
  first is a feature-domain answer to a feature-domain measurement, and this block's last axis is
  $R = 16$ raw samples of one trace with no channel axis for an $(H, C)$ weight to be positional
  against; the second is unnecessary, because a raw target has $\tau^y \equiv 0$ and this row's
  single `causal_align_reference` is therefore already source-only. Both keys are refused by name at
  construction rather than silently ignored. `lag_attn_crws/DESIGN.md` §14 carries both, with their
  triggers.

> lean-limit: the prior's clock cancels no part of the KL's **mean** term, because the posterior is
> a bounded residual on the prior and that term is a function of the delta head alone; replace with
> a delta defined as $D(a) - D(a^\varnothing)$ -- and `posterior_logvar_mode` back to `residual` for
> the variance half -- when the owner accepts a change to the posterior parameterisation the whole
> coupling readout is defined on.

`lag_attn_cfs/DESIGN.md` §8.2 carries the decomposition. It is a property of the shared posterior
parameterisation rather than of any encoder or any target domain, so it holds identically here — and
on this row it binds a mechanism that had little to do in the first place, since every kept source
channel arrives by step $6$.

> lean-limit: the anchor floor is $134$ by policy rather than by validity, costing $104$ of $240$
> available anchors; replace with $F = 30$ and a re-derived stride when a run shows the anchor count
> rather than the source pathway is the binding constraint on `pred_gap`.

Inherited whole from the conv-LSTM cell of this row: a raw target is honest at every step, so the
floor constrains nothing about the target and is retained as the declared input-warmth policy over
the target-stream channels. `lag_attn_crws/DESIGN.md` §3 carries the cost.

**The note got weaker, not stronger, when the reference moved.** Under `target_max` the floor was
the input-warmth requirement exactly — $\max_c(W'_c + d_c) = 134$ — so the only floors below $134$
were ones the constructor and the pre-flight both refused, and "policy rather than validity" was a
statement about the *target* alone. At $42.21$ s the requirement is $6$ (§3), so every floor from
$6$ upwards now builds and runs: the $F = 30$ arm the trigger names is reachable today, and nothing
refuses it. It still does not ship, because roughly doubling the anchors per sample changes the
optimisation regime and is a decision to be taken with a run rather than with a key.

> lean-limit: the driver's four config-shaped pre-flight refusals and the task's two
> `super()`-calling members are written out in the conv-LSTM cell of this row rather than shared
> with the causal-feature cell, each pinned by its own refusal or behaviour test rather than by
> identity; replace with a shared home when a third consumer of any of them appears.

This package is not that third consumer: it reaches all six through its two parents by resolution
order and writes none of them. What it does write — the two three-line entry-point helpers of
`trainer.py`, `_resolve_cli_config_path` and `main` — is the family's Run-button boilerplate, present
in every driver of the grid, and it is not this note's subject.

> lean-limit: no `eval/` package, so every number is a scalar from one run's own
> `metrics_history.csv` and no reported difference carries an uncertainty; replace with a
> `ModelBinding` against `teb_vae/lag_attn_rws/eval` when a result from these cells is to be
> reported as a measurement rather than as a demonstration.

Deferred whole for both cells of this row, and stated here so no number is read as though it had a
confidence interval. The binding is genuinely available — the target is raw, which is exactly what
that pipeline evaluates, and `teb_vae/lag_attn_transformer_rws/eval/` is the working precedent on
these very encoders — and the one known obstacle, that `eval/metrics.py::model_inputs` bypasses
`_build_forward_inputs` and would silently score the default geometry, is recorded in
`lag_attn_crws/DESIGN.md` §14 rather than solved.

## 15. Deviation record

Where the built package differs from the design it was built from, and why.

- **This package writes no network code and copies none.** The model is a constructor, the task is
  empty, and the driver is three class attributes. That is what makes a difference against
  `lag_attn_crws` attributable to the encoder alone and a difference against
  `lag_attn_transformer_rws` to the input representation alone — and it is the reason the empty
  bodies are asserted as facts about the classes rather than described in prose.
- **The conftest is spliced from two siblings, and which half comes from which is not
  interchangeable.** The constructor keyword sets are written here at the conv-Transformer schema;
  the data half — the committed causal shard, the config builder, the tiny warm-up staircase, the
  budget resolver, the stub batch, the seeded streams and the raw signal at the one-sided widths — is
  imported from the conv-LSTM cell of this row, whose own imports are the causal-feature cell's, so
  the objects here are the family's single copies rather than a second hop's worth of copies. The two
  imported-name lists are literal tuples asserted disjoint, because a name reachable from both would
  resolve by import order rather than by intention. The two halves meet at a named tuple of nine
  geometry keys that `tests/test_fixtures.py` asserts agree, **and** whose completeness has its own
  test with a named allow-list of the shared-but-not-geometry keys — without it the list would pass
  on whatever the two sets happen to hold today, which is exactly the drift it exists to catch.
- **The five conv-LSTM keywords are asserted refused *and* asserted to be keywords of
  `SeqVaeLagAttnCrws`.** `conv_norm_groups` is the fifth, which that cell declares but its shipped
  config leaves unset; without the second half the refusal test would pass on any misspelling.
- **The distinct-stem check walks nine drivers rather than six.** The stem is a filename, and a
  filename collides with whatever else is written beside it, so every driver of the grid is in the
  set.
- **The gradient clip moved on the encoder edge and the additive margin did not, and only a run could
  say which.** The encoder edge changes neither the block ($240$) nor the anchor count ($\approx
  10.1$), so no arithmetic predicted either. `gradient_clip_val` $1000 \to 1100$: the instrumented
  run through `smoke_causal.yaml` read pre-clip $q_{99} = 1055$ and $\max = 1333$ with the whole
  shard in one batch, so the conv-LSTM cell's $1000$ sits *below* this encoder's $q_{99}$ and would
  rescale more than one step in a hundred, while $1500$ and $2000$ both sit above the observed
  maximum and would not have bound on a single step of the run they came from — which is why the
  clip is rounded to $100$ rather than to $500$, and the reason is falsifiable rather than aesthetic.
  `additive_margin` stays at $5.0\mathrm{e}{+2}$: the worst excursion above the breaker's own EMA
  in the noisiest regime was $283$ against the conv-LSTM cell's $248$, so the value sits at
  $1.8\times$ the worst excursion against $2.0\times$ there, inside the same $(283, 759)$ bracket —
  a measurement with a stated distance from its floor rather than an inheritance. The `RETUNED` test
  asserts divergence but **not** direction, deliberately: across the input-representation axis the
  block halves and a larger threshold would describe a model with more to clip, but across the
  encoder edge nothing predicts a direction at all. `RESULTS.md` carries the percentiles.
- **The cross-budget checkpoint refusal lands on the input adapter rather than on the decoder
  head.** Both decoders are $R$ wide and align perfectly, which is precisely why the adapter has to
  be the thing that refuses.
- **The permutation control's vacuity at initialisation is about `mu_post` / `logvar_post`, not
  `mu_full`.** The control decodes a fresh $\epsilon$, so the shuffled forecast moves at
  initialisation for a reason unrelated to the source; the test names the tensors the claim is about.
- **`raw_per_step` stays**, for the geometry reason in §3, and `decoder_out_channels` is not a keyword
  of this constructor at all (§6).
- **`tiny.yaml` shrinks the widths under two independent constraints.** The constructor validates
  `num_heads * d_head == d_model` for the **lag-attention** heads, while `encoder_num_heads` is
  unrelated to `num_heads` and carries its own requirement that `d_model / encoder_num_heads` be
  even. The two products coincide in the shipped set only because both head counts happen to be $4$.

### What moved in this revision, against the record it replaces

The four mechanisms are the shared parents' and `lag_attn_crws/DESIGN.md` §15 lists them once,
along with the two this row declines. Two consequences are **this** cell's:

- **Two of the seven encoder keys now apply to a comparison arm rather than to the shipped model.**
  `source_attention_blocks` and `source_attention_window` describe the deep source encoder, which
  the shipped `lag_kv_source: conv_stem` does not build; the stem reads `encoder_conv_kernels` and
  `encoder_conv_dilations` instead. They are kept, commented and still compared leaf-for-leaf (§12),
  because they are exactly what the `encoder` arm needs and removing them would make the parity check
  read a divergence where there is none.
- **The encoder edge is measured on a different number now, and the earlier one is still the right
  one for a different question.** $-91{,}342$ is the two *deep* encoder stacks and is read on the
  off-state rows; $-371{,}431$ is the two shipped history stacks, stems included, and is read on the
  shipped rows. §13 states both and says which comparison each belongs to.

## 16. Running it

From the repository root.

```bash
# Production, 7 ranks. TEB_RUN_STAMP is required so ranks 1..N-1 share rank 0's run directory,
# and the rank count must equal len(general_config.cuda_devices). The shard paths in default.yaml
# are REPOINT_ME placeholders until the causal production shards exist.
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_transformer_crws.trainer \
    --config teb_vae/lag_attn_transformer_crws/configs/default.yaml

# Local smoke: one epoch, one device, the committed causal fixture.
python -m teb_vae.lag_attn_transformer_crws.trainer \
    --config teb_vae/lag_attn_transformer_crws/configs/tiny.yaml

# The instrumented run the clip was re-measured on: shipped widths, the committed fixture, the clip
# parked at 1e9, the LR ramp shortened to 100 steps.
python -m teb_vae.lag_attn_transformer_crws.trainer \
    --config teb_vae/lag_attn_transformer_crws/configs/smoke_causal.yaml

# The one arm: dense training anchors.
python -m teb_vae.lag_attn_transformer_crws.trainer \
    --config teb_vae/lag_attn_transformer_crws/configs/sweep_anchor_stride_1.yaml
```

`RUN_CONFIG` near the bottom of `trainer.py` names the config used when the module is launched with
no command line, so the entry point works from an IDE's Run button with the only operator action
being to edit a value inside the file; a `--config` on the command line always wins, and a relative
path resolves against the repository root rather than the working directory. There is no `eval` entry
point and no `check_run` entry point (§14).

The gate:

```bash
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_crws/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_crws/tests -q -m slow
```
