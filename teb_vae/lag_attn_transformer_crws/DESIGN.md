# `lag_attn_transformer_crws` — design record

This package is the conv-Transformer causal-input raw-target lag-attention VAE-TEB. This document
records which parent supplies which half of the model, what the model consumes and returns, what it
optimises, and how its configuration is built.

Companion documents carry the parts this model reaches by import, and this document does not restate
them:

- `teb_vae/lag_attn_crws/DESIGN.md` records the input domain, the anchored raw target, the
  input-warmth policy, the source compromise, the added readouts and the architecture switches.
- `teb_vae/lag_attn_transformer_rws/DESIGN.md` records the encoders, the adapters, the attention
  blocks and the wiring.
- `teb_vae/lag_attn_rws/DESIGN.md` records the raw target, the objective and the shared
  architecture of the raw-target row. `teb_vae/lag_attn_rws/model_explained.md` records the latent
  factorisation.
- `teb_vae/lag_attn_cfs/DESIGN.md` records the causal input machinery.

---

## 1. What the model is

The model is one cell of an encoder-by-target grid. It is the conv-Transformer half of the row in
which the inputs are causal and the target is the raw FHR signal:

```
                          conv-LSTM encoders          conv-Transformer encoders
  raw FHR target          lag_attn_rws                lag_attn_transformer_rws
  two-sided feature       lag_attn_fs                 lag_attn_transformer_fs
  causal feature          lag_attn_cfs                lag_attn_transformer_cfs
  causal in / raw out     lag_attn_crws               lag_attn_transformer_crws   <- this
```

At each decoded 4-second anchor $t$, the model forecasts the next two minutes of raw FHR. That is
$H \cdot R = 30 \times 16 = 480$ raw samples. It makes the forecast twice: once from a target-only
latent and once from a source-conditioned latent, through one shared decoder that it calls twice.

Against `lag_attn_crws`, the configuration differs in the encoder alone. Against
`lag_attn_transformer_rws`, it differs in the input representation alone.

The whole model is

```python
class SeqVaeLagAttnTrfCrws(CausalRawInputs, SeqVaeLagAttnTrfRws): ...
```

The class defines a constructor and nothing else: `vars(SeqVaeLagAttnTrfCrws)` holds `__init__` and
no other callable. The method resolution order is
`SeqVaeLagAttnTrfCrws -> CausalRawInputs -> CausalWarmupInputs -> SeqVaeLagAttnTrfRws -> Module ->
object`. §7 gives the reason for the constructor.

The model takes the same architecture switches as `lag_attn_crws` (`lag_kv_source`,
`prior_availability_input`, `horizon_weight_halflife_steps`, `alibi_slope_scale`), and
`lag_attn_crws/DESIGN.md` records their inventory and off-states. Both cells of this row decline two
mechanisms of the feature-target cells: the decoder's persistence residual and a second alignment
reference. One switch builds a different module here. Under `lag_kv_source: conv_stem`, the local
K/V stem is a `GatedCausalConvStem` over this architecture's own `encoder_conv_kernels` /
`encoder_conv_dilations` schedule. At the shipped $(5, 9)$ / $(1, 2)$ schedule, its reach is
$1 + (5-1) \cdot 1 + (9-1) \cdot 2 = 21$ steps.

## 2. Input contract

The input contract is identical to that of `lag_attn_crws` in every field, width and refusal. The
model reads the causal HDF5 shards through `train/data_module.py::GraphDataModule` at
`trim_minutes: 1.0`. The target is `fhr` at $(B, 4800)$. The inputs are $c_y = 36 + 66 = 102$
target-stream channels (`fhr_st`, `fhr_ph`) and $c_u = 36 + 15 = 51$ source channels (`up_st`,
`up_ph`). The loader also reads `weight`, `guid` and `epoch`. The field `fhr_up_ph` is absent by
construction. `lag_attn_crws/DESIGN.md` is the full record.

Three requirements are repeated here, because each config file of this package can lose them
independently:

- **`fhr` must be in `load_fields` and `normalize_fields`, and `weight` must be in `load_fields`.**
  The target must arrive z-scored, or the Gaussian NLL is meaningless while the loader raises
  nothing. The validity signal is the only reliable gap marker, because gaps are stored as $0$ bpm.
  The shared entry point checks the fields that `TARGET_FIELDS` names. The driver resolves
  `TARGET_FIELDS` through the causal parent (§7), which inherits `("fhr",)` from the shared
  ancestor.
- **`guid` and `epoch` must be in `load_fields`.** The tile phase is keyed on this pair, and the
  loader honours `load_fields` literally.
- **The shards must be the causal variant.** The two variants share every field name and dtype.
  Only the root `transform` attribute and the channel counts tell them apart. The warm-up resolver
  refuses a two-sided shard by name before it creates a run directory.

## 3. Geometry, tiling, warm-up and channel alignment

The geometry comes from the conv-LSTM cell of this row by import. `lag_attn_crws/DESIGN.md` is the
record. In summary:

- The input-warmth policy requires $F \ge B - 1$ and $F \ge \max_c(W'_c + d_c)$. Here $F$ is
  `warmup_period`, $W'_c$ is the rebased warm-up of channel $c$, $B = \max_c W'_c$ over the kept
  channels, and $d_c$ is the alignment shift of channel $c$.
- The tiled anchor set is $\mathcal{A}(\varphi) = \{F + \varphi + kS\}$ with $S = H = 30$. The
  phase $\varphi$ is derived per segment from a `blake2b` hash of `guid`, `domain_start`, the epoch
  and the seed.
- With $F = 134$ and $T = 300$, the valid anchor range is $[134, 270)$. Thus $A_{\max} = 5$ at the
  training geometry and $A_{\max} = 136$ at the dense evaluation geometry (stride 1, phase 0).
- The raw target is gathered at the anchors.
- `AvailabilityInputAdapter` applies the input warm-up mask.

**The budget.** `causal_warmup_budget_steps: 134` keeps a target-stream channel when
$W'_c \le 134$. On the configured shards it keeps 98 of 102 target-stream channels. The budget does
not gate the source.

**The channel alignment.** `causal_align_reference` names a reference delay $\tau_{\mathrm{ref}}$.
The resolver shifts each kept channel by
$d_c = \mathrm{round}\big(\kappa\,(\tau_{\mathrm{ref}} - \tau_c) / \Delta\big)$, with
$\Delta = 4$ s and the energy-centroid factor $\kappa = 1 - 1/(2\gamma) = 0.875$ at gammatone order
$\gamma = 4$. It drops every channel slower than the reference, because that channel could only
reach the reference by reading a later stored step. A literal reference snaps to the shard's own
nearest `float32` delay, and the resolver refuses a literal further than $\Delta / 2$ from any kept
channel. `null` builds the unaligned model.

**`default.yaml` ships `causal_align_reference: null`**, which is the unaligned arm. Its comment
names $42.21$ s as the aligned value. At that reference, the resolver snaps to $42.206562$ s and
keeps 38 of 102 target-stream channels (17 `fhr_st`, 21 `fhr_ph`) and 17 of 51 source channels
(17 `up_st`, no `up_ph`). The shifts span $d_c \in [0, 6]$ and the survivors wait
$W'_c \in \{0, 1\}$, so the policy requires only $F \ge 6$.

**Why the aligned value for this row is a fast channel and not `target_max`.** A raw target passes
through no filter, so its delay is $\tau^y \equiv 0$. Thus nothing on the target side cancels the
source reference out of the lag identity

$$\tau^{\mathrm{lead}}(\ell, h) \;=\; \Delta\,(\ell + 1 + h) \;+\; \kappa\,\tau_{\mathrm{ref}}.$$

The lead is smallest at $\ell = h = 0$ and grows with the lag. At the feature row's `target_max`
reference ($402.1604$ s), the term $\kappa\,\tau_{\mathrm{ref}}$ alone is above $350$ s, and a
larger `max_lag` only makes the lead longer. In the causal-feature cells, the same reference
appears on both sides of the identity and cancels, so those cells keep `target_max`.

**`TrimmedRawGeometry` is reused unchanged.** The loader delivers `fhr` at $4800$ and `weight` at
$300$. `raw_masks._validate_weight` checks `weight.size(1) == geometry.t`. The decoder emits
$R = 16$ raw samples per horizon token, and the anchored gather indexes into the inherited
`future_index`.

**The lag validity floor.** `lag_floor` ships at $0$. At that value the lag mask is bitwise the
architecture parent's mask.

## 4. Forward return dict

`SeqVaeLagAttnTrfCrws.forward(y_st, y_ph, u_stream, anchor_phase=None, anchor_stride=None)` returns
22 keys. These are the raw-signal family's 20 keys plus the two anchor keys. Shapes below are at the
shipped config ($d_z = 64$, $d_{\mathrm{model}} = 128$, 4 lag heads of width 32, $L = 91$ lags).

- `mu_base`, `logvar_base`, `mu_full`, `logvar_full`: $(B, A_{\max}, H, R)$. That is
  $(B, 5, 30, 16)$ at the training geometry and $(B, 136, 30, 16)$ at the dense geometry.
- `anchor_index`: $(B, A_{\max})$, `long`. `anchor_valid`: $(B, A_{\max})$, `bool`.
- `mu_prior`, `logvar_prior`, `raw_logvar_prior`, `mu_post`, `logvar_post`, `z_prior`, `z_post`:
  $(B, 300, 64)$.
- `target_state`, `source_state`: $(B, 300, 128)$.
- `attended_source_heads`: $(B, 300, 4, 32)$. `attn_weights`: $(B, 300, 4, 91)$.
- `kld_per_t`: $(B, 300)$. `kld_per_t_per_head`: $(B, 300, 4)$. `source_kl_lag_map`:
  $(B, 300, 91)$.
- `mu_prior_sat_frac`, `delta_mu_sat_frac`: scalars.

`source_state` is the K/V tensor of the lag attention. Under `lag_kv_source: encoder` it is the
deep source state. Under the shipped `conv_stem` it is the stem output. Both intervention controls
resolve the path from model attributes, so their signatures did not change.

The dict has no `decoder_state` and no `delta_mu_src`, because there is no decoder bypass. It has no
`persistence` key, because this row declines the persistence residual. Thus the 22 keys are the
whole contract at every configuration.

## 5. Loss and metrics

The objective, its $\beta$ schedule, the metric surface, the validation-only permutation control,
the source-null arm, the spike-breaker wiring and the checkpoint contract are the causal-input
parent's code, not a copy. `LagAttnTrfCrwsTrainer.TRACKED_METRICS` is the causal parent's tuple by
identity and holds 89 entries. `lag_attn_crws/DESIGN.md` records the added readouts and the
overridden `_mu_gap_rms`.

The readouts that the causal-input row adds:

| Metric | Stages | What it measures |
| --- | --- | --- |
| `anchors_per_sample` | train, val | the number of decoded anchors per sample: 4 or 5 in training at stride 30, 136 in validation |
| `source_lag_warmth_frac_st` | train, val | the attention mass on lags where the `up_st` source block is warm |
| `source_lag_warmth_frac_ph` | train, val | the same for the `up_ph` source block |
| `kld_source_null` | val | the KL that the availability clock induces when the source carries no content |

`_resolve_block_warm_steps` reports an empty block as warm at every step. Thus, at a reference that
keeps no `up_ph` channel, `source_lag_warmth_frac_ph` reads $1.0$ over zero channels.

The causal-feature cell's eight target-side columns are not tracked here: the four gap splits by
horizon step and stored block, the three warm-up tertiles and the target-side warm fraction. A raw
target has no blocks, no kept channels and no warm-up.

**`lambda_boundary` is refused at any non-zero value** by the causal-input parent's pre-flight. The
boundary term is a slicing identity over adjacent anchors, and this family decodes anchors that are
$S$ apart. `lambda_ms` and `lambda_deriv` ship at $0.1$, as on the raw-signal side, because they are
within-block raw-waveform terms on the same grid.

**`horizon_weight_halflife_steps: 15.0` weights the reconstruction on the horizon axis.**
`lag_attn_rws/nets/losses.py::horizon_decay_weight` builds $w_\tau \propto 0.5^{\tau/\lambda}$ and
renormalises it to $\sum_\tau w_\tau = H$. The renormalisation keeps the block score's scale, and
with it the scale that `gradient_clip_val`, `additive_margin` and $\beta$ act on. The weight is a
non-persistent buffer.

**Units.** The reconstruction is summed over $H \cdot R = 480$ raw samples per anchor. `lag_attn_crws`
uses the same block, the same anchor set, the same objective and the same half-life, so a loss level
of the two cells is on one scale. `lag_attn_transformer_rws` sums the same 480-sample block, but it
decodes the dense anchor range and uses an unweighted horizon, so a loss level is not on the same
scale. Two arms at two warm-up budgets have mutually unloadable checkpoints, because the input
adapters are built at the surviving widths.

## 6. Why one mixin and not an inheritance

**The obvious construction runs the wrong constructor.** `SeqVaeLagAttnCrws` subclasses
`SeqVaeLagAttnRws`, while `SeqVaeLagAttnTrfRws` derives from `nn.Module` directly. Thus
`class X(SeqVaeLagAttnCrws, SeqVaeLagAttnTrfRws)` linearises as
`X -> Crws -> ... -> Rws -> TrfRws -> Module` and runs the conv-LSTM constructor. That model builds,
trains and reports, but it is not this architecture.

So the input domain lives in one plain object, `CausalRawInputs` in `teb_vae/lag_attn_crws/nets/`,
which names no encoder. Both cells of this row are their own architecture plus that one object. The
object has no `Protocol` and no `__init_subclass__`, and every member was lifted verbatim.

**One mixin, not two.** The causal-feature cells compose a target mixin beside the input mixin,
because a stored-coefficient target changes the decoder's width, its gather and its readouts. A raw
target changes none of them. `_default_decoder_out_channels` already returns `raw_per_step` on the
architecture parent, so neither `CausalRawInputs` nor this class defines a width hook. If
`CausalFeatureForecastTarget` were composed in, the decoder would be built at the surviving
target-input width against a $(B, A, H, 16)$ target, and `raw_sample_score` would fail on shapes
that do not broadcast. `tests/test_construct.py` asserts that the shipped decoder is
`raw_per_step` wide.

**The order of the bases matters.** The mixin comes first. Thus the tiled forward overrides the
architecture parent's dense forward, the warm-up adapter overrides the gate's shift vector alone,
and the anchored `compute_loss` overrides the dense raw one. With the order reversed, the model
would decode the dense anchor range, return no anchor set, and score a
$(B, T_{\mathrm{valid}}, H, R)$ target against a $(B, A_{\max}, H, R)$ forecast. The tests read the
order through its effects: the adapter built at the warm-up (`tests/test_construct.py`), the anchor
set in the forward (`tests/test_forward_contract.py`) and the anchored objective's metric set
(`tests/test_objective.py`).

**There is no `decoder_out_channels` keyword.** The architecture parent declares none, because the
decoder emits $R$ raw samples per horizon token. Thus no configuration can put the decoder and the
raw target on different widths.

## 7. The three diamonds and their resolution order

```python
class SeqVaeLagAttnTrfCrwsTask(SeqVaeLagAttnCrwsTask, SeqVaeLagAttnTrfRwsTask):
    pass

class LagAttnTrfCrwsTrainer(LagAttnCrwsTrainer, LagAttnTrfRwsTrainer):
    MODEL_CLS = SeqVaeLagAttnTrfCrws
    TASK_CLS = SeqVaeLagAttnTrfCrwsTask
    CHECKPOINT_STEM = "lag-attn-trf-crws"
```

The task defines no callable. The driver re-points three class attributes and defines no method.
The three linearisations are:

- `SeqVaeLagAttnTrfCrws -> CausalRawInputs -> CausalWarmupInputs -> SeqVaeLagAttnTrfRws -> Module`
- `SeqVaeLagAttnTrfCrwsTask -> SeqVaeLagAttnCrwsTask -> SeqVaeLagAttnTrfRwsTask ->
  SeqVaeLagAttnRwsTask -> LightningModelBase`
- `LagAttnTrfCrwsTrainer -> LagAttnCrwsTrainer -> LagAttnTrfRwsTrainer -> LagAttnRwsTrainer ->
  GraphModelBase`

**The driver re-points its three attributes because both parents set all three.** Resolution order
alone would take the causal side, and each failure would be silent:

- Without `MODEL_CLS`, the driver builds a conv-LSTM model with no error.
- Without `TASK_CLS`, the driver builds the conv-LSTM task, which does not define the step-granular
  learning-rate ramp.
- Without `CHECKPOINT_STEM`, the driver writes `lag-attn-crws-*.ckpt` and mixes two models'
  checkpoints in a shared output tree. `tests/test_trainer.py` asserts that the stems of all nine
  drivers in the family are distinct.

All other members arrive by resolution order:

| From the causal-input parent | From the conv-Transformer parent | From the shared driver |
| --- | --- | --- |
| `TRACKED_METRICS` (89 entries) | `compile_model_requested` | `TARGET_FIELDS = ("fhr",)` |
| `preflight` and its refusals | `_build_trainer_kwargs` (the step-granular LR monitor) | `PLOT_CONFIG_KEY` |
| `causal_standing_message` | `build_lr_scheduler`, on the task | the DDP strategy selection |
| the tiling phase, the source-null readout and the three page seams, on the task | | the callback assembly |

**Both parents define `_build_model_kwargs` and `create_model`, and both run.** Each calls
`super()`, so the conv-Transformer contributions run underneath the causal ones. The
conv-Transformer half re-admits `source_attention_window: null` and applies `lr_warmup_steps`. The
causal half adds the four resolved warm-up tuples, the geometry log line, the tile-phase seed and the
resolved budget for the task. `tests/test_trainer.py` asserts that both halves run.

**`compile_model_requested` resolves to the conv-Transformer side.** The causal parent does not
define it, so `torch.compile` is permitted on this model. The shipped configs keep `compile: false`.

**`TARGET_FIELDS` is the shared ancestor's object.** It is `("fhr",)` on every driver of the
raw-target row. `tests/test_config_load.py` asserts that the shipped config loads and normalizes
every field that the tuple names.

**`PLOT_CONFIG_KEY` stays `"lag_attn_rws_plotting"`**, and the config block keeps that name. The
shared callback assembly reads the literal. A block under another name gives no figure and no
error.

**The task diamond is well-formed because the two branches are disjoint today.** The conv-Transformer
branch adds only `build_lr_scheduler`. A future member defined on both sides would resolve to the
causal side by order alone. `tests/test_task.py` reads one behaviour from each side through the
diamond, so a reorder fails the test.

## 8. Step-wise causality

**Step-wise causality of the history states holds at every configuration.** These encoders have no
time-pooling normaliser. `RMSNorm` reduces over channels only. The convolutions pad on the left
before a `padding == 0` `Conv1d`. Attention is causal by kernel flag on the target (full prefix) and
by an explicit band mask on the source ($W_U = 16$). `causal_norm` is not a constructor keyword of
this model, so no flag conditions the claim.

`tests/test_causality.py` measures causality through the assembled model at the shipped warm-up
budget and at the tiny fixture. It also measures prefix equivalence
$\mathcal{E}(X_{0:T-1})_t = \mathcal{E}(X_{0:t})_t$. The warm-up mask cannot break prefix
equivalence, because it is a function of $t$ alone.

**Three properties meet in this cell, and they are independent.** Token causality,
$H_t = f(X_{\le t})$, comes from the architecture. Input causality comes from the transform: a
stored coefficient at $t$ is a function of $\{x(s) : s \le t\}$. The target is the raw signal, so
the objective scores samples that no input has seen. The coupling readout is still named
`source_conditioned_kl_raw` and is not called a transfer entropy, because the inputs carry their own
group delay (§13).

## 9. Structural constraints

The tests assert these constraints against this class. Where an invariant belongs to a parent, this
package's suite imports and re-parametrises the sibling test module that owns it.

| Property | What enforces it | Test |
| --- | --- | --- |
| **No decoder bypass**: gradient reaches the decoder only through $z$ | `BaselineFutureDecoder.forward` takes exactly one tensor, at $d_z$ in-features | `tests/test_invariants.py` |
| **Source purity**: the prior never sees the source, and the source state never sees the target | separate gates, adapters and encoders; the posterior is a residual on the prior | `tests/test_invariants.py` |
| **Exact zero KL at initialisation** | posterior deltas zeroed after the generic init; one shared $\epsilon$ | `tests/test_invariants.py` |
| **One decoder, called twice** | the same module object on $z^p$ and $z^q$ | `tests/test_invariants.py` |
| **Token causality** | §8 | `tests/test_causality.py` |
| **No raw sample scored twice in a step** | the tiled anchor set partitions the timeline; padded slots repeat and are marked invalid, and the anchored gather honours the repeat so the mask can remove it | `tests/test_forward_contract.py`; `lag_attn_crws/tests/test_objective.py` |
| **The anchored target equals the dense builder at the dense set** | `gather_anchored_future_target` against `build_future_target` under `torch.equal` | `lag_attn_crws/tests/test_raw_target.py` |
| **The lag attribution identity** $\sum_\ell \widetilde K_{t,\ell} = K_t$ | the lag attention is built at `dropout=0.0` | `tests/test_invariants.py`, after `perturb_posterior` |

Because the delta heads are zero-initialised, a KL assertion on a freshly constructed model passes
vacuously. Thus every KL test in this package uses the `perturb_posterior` fixture.

The zero-KL claim depends on the fixture's flags. The shipped config sets `base_decode: mean`. Under
it, the two forecasts are not bitwise identical, but the KL is still exactly zero, because the KL
depends on the two distributions and not on the drawn samples. The shipped config also sets
`posterior_logvar_mode: independent`. Under it, the init KL is zero only with
`head_init_calibration: true`.

## 10. DDP reachability

Production runs use plain `"ddp"` with `find_unused_parameters=False` under `gaussian_nll`, so every
parameter must be reachable. `mse` starves the decoder's log-variance heads and selects
`ddp_find_unused_parameters_true`. `tests/test_ddp_strategy.py` checks both likelihoods, on the
guarded and the ungated arm.

A parameter multiplied by an identically-zero tensor is reachable: its `AccumulateGrad` node fires
and the reducer marks it ready. What breaks `find_unused_parameters=False` is a parameter that a
Python branch on a tensor value leaves out of the graph. Thus the forward adds the availability
terms unconditionally, and branching happens at construction time only.

**Start embeddings.** A stream builds a start embedding when its earliest valid step
$\min_c (W'_c + d_c)$ is above zero. Unaligned, that minimum is $0$ on both streams, because each
stream has a channel at $W' = 0$, so neither embedding is built. Under an alignment reference, the
shift raises the minimum above zero, so both embeddings are built. In both cases the branch happens
at construction time, and a built parameter is reached on every batch. For the same reason, a
driver pre-flight refuses `use_up_st: false` together with a warm-up budget: it would create the indicator
from a stored-block decision instead of from the shift.

This package writes no `forward`. `tests/test_ddp_reachability.py` checks the consequence on this
composition: under `gaussian_nll`, every parameter gets a gradient on both guard states and at the
shipped switches. Under `mse`, the starved set is exactly the decoder's log-variance head.

Each rank derives the per-segment phase from its own samples, with no collective, and $A_{\max}$ is a
geometry constant at every phase. Thus no rank can disagree on shape. **`broadcast_buffers=False`**
is safe, because every buffer is a deterministic function of the config and there is no
`BatchNorm`. The buffers are the rotary tables, causal masks, gate keep-indices, adapter
availability patterns, the two source-warmth patterns and the raw-target index grid. `static_graph`
is not set: on a skipped batch, the loss-spike breaker substitutes a zero-weighted sum over every
parameter, which is a different backward from the one the first iteration recorded.

## 11. Diagnostic plots

The nine-row page and the run-level warm-up figure are those of the conv-LSTM cell of this row. They
are reached through the task's page seams: `forecast_rows`, `input_stream_panels` and the
`input_budget_figure` method. This package ships no `plotting.py` and no `sample_page.py`.
`lag_attn_crws/DESIGN.md` is the record.

The seams resolve on the task, and the task resolves them from the causal-input parent. The shipped
raw forecast rows walk a dense block at an anchor index that this model's forecast does not have, so
they must not run. A failure inside the plotting handler warns and continues. Thus the fixture fit
in `tests/test_train_smoke.py` asserts only that the callback is attached and that the fit
completes.

The page carries the one-sided lag caveat, as the same string as the conv-LSTM cell. Under the
channel alignment, the caveat gives the content lead time as
$\Delta(\ell + 1 + h) + \kappa\,\tau^u_{\mathrm{ref}}$ seconds on the stored timeline, because a raw
target has $\tau^y \equiv 0$. The effective reference is $\kappa\,\tau_{\mathrm{ref}}$, not the
configured $\tau_{\mathrm{ref}}$ that the resolved-config dump and the caption name.

## 12. Configuration

`configs/` holds exactly `default.yaml`, `tiny.yaml`, `smoke_causal.yaml` and
`sweep_anchor_stride_1.yaml`. `tests/test_sweep_configs.py` asserts that set, and the framework's
validator checks every file. `default.yaml` is written out in full and has no `base:` chain. The
three other files use `base: default.yaml`.

| Config | Purpose |
| --- | --- |
| `default.yaml` | production: 7 CUDA devices, the causal production shards |
| `tiny.yaml` | local smoke: one epoch, one device, the committed causal fixture, shrunken widths, `likelihood: mse`, `lr_warmup_steps: 4` |
| `smoke_causal.yaml` | instrumented run at the shipped widths on the committed causal fixture: 600 epochs, `gradient_clip_val: 1.0e+9`, `lr_warmup_steps: 100`, whole-shard batch |
| `sweep_anchor_stride_1.yaml` | dense training anchors: `anchor_stride: 1` |

**Parity with `lag_attn_crws`.** `tests/test_config_load.py` compares the shipped config
leaf-for-leaf with the conv-LSTM cell's config, outside the allow-list `PARITY_EXEMPT_PATHS` of 19
exemptions. The check works in both directions: an exemption that is no longer a divergence fails,
and a divergence that is not exempt fails. The exemptions are:

- Five identity paths: the tag, the output directory and three MLflow identifiers.
- Seven encoder keys that this architecture adds: `encoder_conv_kernels`, `encoder_conv_dilations`,
  `encoder_num_heads`, `encoder_d_ff`, `target_attention_blocks`, `source_attention_blocks`,
  `source_attention_window`.
- Five conv-LSTM keys that this architecture removes: `lstm_layers`, `encoder_extra_dilations`,
  `encoder_extra_kernel`, `conv_norm_groups`, `causal_norm`.
- `lr_warmup_steps`, the step-granular learning-rate ramp of the conv-Transformer encoders.
- `gradient_clip_val`, declared `RETUNED`. `default.yaml` sets it to $11100.0$.

`additive_margin` ($2.5 \times 10^{3}$) is under the parity check. `tests/test_spike_breaker.py`
checks it and the clip against fixed bounds.

**The five conv-LSTM keys are not constructor keywords of this model.** Passed to the constructor
directly, each raises `TypeError`, and `tests/test_construct.py` asserts this. The driver's
`inspect.signature` sweep drops any config key that names no constructor argument, so a config that
carries one would still build. `tests/test_config_load.py` asserts that every key in the shipped
configs reaches the constructor or the task.

`source_attention_blocks` and `source_attention_window` describe the deep source encoder. The
shipped `lag_kv_source: conv_stem` does not build it; the stem reads `encoder_conv_kernels` and
`encoder_conv_dilations`. The two keys stay in the config for the `encoder` arm and for the parity
check.

**No horizon arm ships in this package.** `lag_attn_transformer_rws` is the raw-target model on the
same encoders.

**Causal keys.** `causal_reach_budget_s` is present and must be `null`, and
`causal_warmup_budget_steps: 134` is the guard for the causal dataset. The resolver refuses the two
together by name. `causal_leg_alignment: envelope` names the phase-harmonic operator that built the
shards' phase blocks, and the resolver compares it with the shards' root attribute.
`causal_align_reference` names the reference clock for the input channels (§3). Both keys are leaves
of `model_config.VAE_model`, so both land verbatim in the resolved-config artifact beside the
checkpoints. The startup log's budget summary records the resolved reference in seconds, the shift
range and the surviving counts per block. The checkpoint's `model_kwargs` carry the two shift
vectors. No tracked metric records the reference.

**Training controls.** `default.yaml` enables early stopping on `val/total_loss` with patience 50.
`model_checkpoint` monitors `val/total_loss` with `save_top_k: 3`, and
`secondary_monitor: "val/nll_full_block"` adds a second checkpoint criterion.

## 13. Parameter structure

`tests/test_construct.py` asserts the structural facts about parameters:

- The warm-up budget and the alignment change only input-adapter parameters.
- The ungated model is parameter-for-parameter the architecture parent.
- Outside the two history encoders, this model is parameter-for-parameter the conv-LSTM twin.
- The prior clock builds no parameter when `prior_availability_input` is off.
- The horizon weight is a non-persistent buffer, not a parameter, or is absent.
- A local K/V arm does not build the deep source encoder.

The architecture switches change these modules:

| Switch | Effect on the parameter set |
| --- | --- |
| `lag_kv_source: conv_stem` | removes the deep source encoder (the windowed attention stack) and adds a `GatedCausalConvStem` at $(5, 9)$ / $(1, 2)$ |
| `prior_availability_input: true` | adds a `LayerNorm` ($2 \times 128$) and a bias-free $128 \times 128$ projection for the prior clock |
| `horizon_weight_halflife_steps` | none: a non-persistent buffer |
| `alibi_slope_scale` | none: the same $(\text{num heads}, L)$ lag-bias parameter with a different seed |

A change of the alignment reference changes, in each stream's adapter, the input linear and the
availability projection, both of width $d_{\mathrm{model}} = 128$. The guard adds one start-of-record
vector of width 128 to each adapter whose stream builds a start embedding (§10). Nothing in this
target domain widens the decoder head.

## 14. Limitations and planned work

- **Group-delay compensation is per channel and is not in the readout.** The alignment (§3) puts
  every kept input channel of both streams on one clock, and the target carries no group delay. Thus
  the remaining offset on a lag reading is the single constant $\kappa\,\tau^u_{\mathrm{ref}}$. No
  readout divides it out yet, so an attention peak is not a physiological delay.
  `lag_attn_crws/DESIGN.md` is the record.
- **The model side and the transform side use different conventions.** `channel_alignment_delays`
  scales the difference $\tau_{\mathrm{ref}} - \tau_c$ by $\kappa = 0.875$. At write time,
  `causal_scattering.py::leg_alignment_shift` shifts the phase-harmonic legs by the envelope mean
  $\tau_g$. Moving the transform side onto $\kappa$ would change the stored coefficients and requires
  a dataset rebuild. Thus the shards' `causal_delay_s` stays $\tau_g$, while every model-side shift is
  $\kappa\,\tau_g$.
- **The two source warmth columns can saturate.** At a reference that keeps no `up_ph` channel,
  `source_lag_warmth_frac_ph` reads $1.0$ over zero channels (§5). When the coldest source step that
  any decoded anchor can read is warm, $F - \max_\ell \ell \ge \max_c(W'_c + d_c)$,
  `source_lag_warmth_frac_st` also reads $1.0$. Read both columns beside the kept width per block.
- **No encoder arms and no $\beta$ arms ship here.** The encoder values come from the raw domain,
  and the $\beta$ pair comes from the raw-signal family.
- **No mixed precision** (`precision: "32-true"`), and **`compile: false`**, although the key is
  live (§7). Inductor may reassociate float arithmetic.
- **`dataset_kwargs` is shared between the two loaders** and cannot carry a per-split GUID filter.
  A config with one shard for both splits has in-sample validation columns.
- **No run checker ships.** `tests/test_train_smoke.py` asserts `anchors_per_sample` on the fixture
  fit.
- **Checkpoints from before the architecture switches do not load into a shipped-configuration
  model.** `load_checkpoint_strict` refuses instead of partially loading, and `check_model_class`
  guards the class. No migration shim exists; the off-state arm builds the old architecture.
- **The persistence residual and the second alignment reference are not part of this row.** The
  persistence residual carries a per-channel level, and this block's last axis is $R = 16$ raw
  samples of one trace, with no channel axis. A second reference is unnecessary, because a raw
  target has $\tau^y \equiv 0$, so the single `causal_align_reference` is already source-only.
  `persistence_residual` is not a constructor keyword, and passing it directly raises `TypeError`.

Planned changes, each with the condition that triggers it:

> lean-limit: the prior's clock cancels no part of the KL's **mean** term, because the posterior is
> a bounded residual on the prior and that term is a function of the delta head alone. Replace with
> a delta defined as $D(a) - D(a^\varnothing)$, and set `posterior_logvar_mode` back to `residual`
> for the variance half, when the owner accepts a change to the posterior parameterisation.

`lag_attn_cfs/DESIGN.md` carries the decomposition. It is a property of the shared posterior
parameterisation, so it holds here unchanged.

> lean-limit: the anchor floor $F = 134$ is a policy and not a validity requirement at an aligned
> reference. Replace with $F = 30$ and a re-derived stride when the owner decides to change the
> anchor count per sample.

A raw target is valid at every step, so the floor constrains nothing about the target. At the
unaligned default, the input-warmth requirement $\max_c W'_c = 134$ equals the floor. At $42.21$ s
the requirement is $6$, so every floor from $6$ upwards builds and runs. A lower floor roughly
doubles the anchors per sample.

> lean-limit: the driver's config-shaped pre-flight refusals and the task's two `super()`-calling
> members are written out in the conv-LSTM cell of this row and are not shared with the
> causal-feature cell. Each has its own refusal or behaviour test. Replace with a shared home when a
> third consumer of any of them appears.

This package is not that third consumer: it reaches all of them through its two parents. The two
entry-point helpers in `trainer.py`, `_resolve_cli_config_path` and `main`, are the family's
Run-button boilerplate.

> lean-limit: no `eval/` package, so every number comes from one run's own `metrics_history.csv`
> and carries no uncertainty. Replace with a `ModelBinding` against `teb_vae/lag_attn_rws/eval`.

The target is raw, which is what that pipeline evaluates, and `teb_vae/lag_attn_transformer_rws/eval/`
binds the same encoders. One known obstacle is recorded in `lag_attn_crws/DESIGN.md`:
`eval/metrics.py::model_inputs` bypasses `_build_forward_inputs` and would score the default
geometry.

## 15. Implementation notes

- **This package writes no network code and copies none.** The model is a constructor, the task is
  empty, and the driver is three class attributes.
- **The test `conftest.py` comes from two siblings.** The constructor keyword sets are written here
  at the conv-Transformer schema. The data half comes from the conv-LSTM cell of this row: the
  committed causal shard, the config builder, the tiny warm-up staircase, the budget resolver, the
  stub batch, the seeded streams and the raw signal at the one-sided widths. The two halves meet at
  a named tuple of geometry keys. `tests/test_fixtures.py` asserts that every key that both
  production sets declare agrees.
- **The constructor refuses the five conv-LSTM keywords and `persistence_residual` with a
  `TypeError`.** `conv_norm_groups` is the fifth; the conv-LSTM cell declares it, but its shipped
  config leaves it unset.
- **The distinct-stem check walks all nine drivers of the grid**, because a checkpoint stem is a
  filename and collides with anything written beside it.
- **The cross-budget checkpoint refusal lands on the input adapter, not on the decoder head.** Both
  decoders are $R$ wide and align, so the adapter is the module that refuses.
- **The permutation control is vacuous at initialisation for `mu_post` / `logvar_post`, not for
  `mu_full`.** The control decodes a fresh $\epsilon$, so the shuffled forecast moves at
  initialisation for a reason unrelated to the source.
- **`tiny.yaml` shrinks the widths under two independent constraints.** The constructor requires
  `num_heads * d_head == d_model` for the lag-attention heads. `encoder_num_heads` is separate from
  `num_heads` and requires `d_model / encoder_num_heads` to be even.

## 16. Running it

Run from the repository root.

```bash
# Production, 7 ranks. TEB_RUN_STAMP lets ranks 1..N-1 share rank 0's run directory, and the rank
# count must equal len(general_config.cuda_devices).
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_transformer_crws.trainer \
    --config teb_vae/lag_attn_transformer_crws/configs/default.yaml

# Local smoke: one epoch, one device, the committed causal fixture.
python -m teb_vae.lag_attn_transformer_crws.trainer \
    --config teb_vae/lag_attn_transformer_crws/configs/tiny.yaml

# Instrumented run: shipped widths, the committed fixture, the clip parked at 1e9, the LR ramp
# shortened to 100 steps.
python -m teb_vae.lag_attn_transformer_crws.trainer \
    --config teb_vae/lag_attn_transformer_crws/configs/smoke_causal.yaml

# Dense training anchors.
python -m teb_vae.lag_attn_transformer_crws.trainer \
    --config teb_vae/lag_attn_transformer_crws/configs/sweep_anchor_stride_1.yaml
```

`RUN_CONFIG` near the bottom of `trainer.py` names the config for a launch with no command line, so
the IDE's Run button works after an edit of that one value. A `--config` on the command line always
wins, and a relative path resolves against the repository root. A Run-button launch of
`default.yaml` is a single process; Lightning spawns the DDP workers for its seven `cuda_devices`.
There is no `eval` entry point and no `check_run` entry point (§14).

Training writes `train_results/metrics_history.csv` with one column per entry of `TRACKED_METRICS`,
`model_checkpoints/resolved_config.yaml`, and checkpoints named `lag-attn-trf-crws-*.ckpt`.

Tests:

```bash
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_crws/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_crws/tests -q -m slow
```
