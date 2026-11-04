# `lag_attn_transformer_rws` — design record

This record describes the causal conv-Transformer raw-signal lag-attention VAE-TEB: what it is,
what it replaces, what it leaves unchanged, and every place where the built module differs from
the design it was built from.

Companion documents: `new_architecture.md` in this directory is the architecture reference, with
the full forward pass and its equations. `teb_vae/lag_attn_rws/DESIGN.md` records the comparison
model, whose objective, data contract, geometry and metric surface this model shares. This record
covers the encoders and the wiring around them.

---

## 1. What the model is

The model is `teb_vae/lag_attn_rws` with **both history encoders replaced**. At every 4-second
anchor $t$ the model forecasts the next two minutes of raw normalized FHR —
$H \cdot R = 30 \times 16 = 480$ samples — twice: once from a target-only latent and once from a
source-conditioned latent. One shared decoder produces both forecasts. The KL between the two
latents, resolved across lags, is the coupling readout.

The change is in how the two history states $H^Y$ and $H^U$ are computed. The replaced encoder is
a five-block dilated causal convolution stack in parallel with a two-layer unidirectional LSTM,
fused by a wide MLP. In its place, per stream:

$$
X^s \longrightarrow \mathcal G_s \longrightarrow \operatorname{InputAdapter}_s
\longrightarrow \operatorname{CausalConvStem}_s
\longrightarrow \operatorname{CausalTransformer}_s \longrightarrow H^s
\in \mathbb R^{B \times T \times 128}.
$$

The design intent follows from the geometry. A recurrent bottleneck serialises training and mixes
adjacent source steps into one state, and the late lag cross-attention needs adjacent source lags to
stay separable. Self-attention gives the target prior a content-dependent direct path from step $j$
to step $t$, in place of a fixed dilation schedule plus a fixed-width recurrent state.

Everything downstream of the two encoder outputs is imported from `teb_vae.lag_attn_rws.nets` and
`teb_vae.lag_attn.nets` without modification: the target-only conditional prior head, the
lag-restricted cross-attention over $L = 91$ lags with $M = 4$ heads, the head-structured bounded
posterior residual, the shared horizon decoder, the lag-resolved KL attribution, the masks and the
objective in nats per anchor. A change to any of them belongs in those packages, so that both
models get it. The shared decoder core runs `horizon_attention_blocks: 2` bidirectional
self-attention blocks over the $H = 30$ horizon tokens after its dilated refine stack.

**The objective is shared.** `compute_loss` in `teb_vae/lag_attn_rws/nets/losses.py` optimises

$$
\mathcal L = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0 + \beta(e)\, K_{\mathrm{train}}
  + \beta_p\, R_p
  + \lambda_{\mathrm{ms}} \mathcal L_{\mathrm{ms}}
  + \lambda_{\mathrm{deriv}} \mathcal L_{\mathrm{deriv}}
  + \lambda_{\mathrm{boundary}} \mathcal L_{\mathrm{boundary}} .
$$

$R_p$ is the prior's scale rate,
$\mathrm{KL}\!\left(\mathcal N(\mu^p, \operatorname{diag} e^{\ell^p}) \Vert \mathcal N(\mu^p, I)\right)$.
It is reduced on the KL's own anchor support, in the same nats-per-anchor units. It is the only term
that penalises a *narrow* prior. `beta_prior` ships at $0.1$; `new_architecture.md` §12.5 carries
the term in full.

The last three terms are the **auxiliary shape terms**. A factorized Gaussian NLL scores every raw
sample independently, so its optimum is the conditional mean, and a fully parallel decoder can emit
an over-smoothed mean. Each shape term is computed on the forecast *means* of both branches and
summed over them: a pooled L1 at rates $(1, 4, 16)$ for the envelope, a Huber on first differences
for the slope, and the gap between the first forecast sample and the anchor's last observed one for
the starting level. The first two use the coverage-floored forecast mask that the NLL uses. The
boundary term uses the anchor's own thresholded `weight`, because the sample it reads belongs to that
anchor. The weights ship at $0.1 / 0.1 / 0.05$ and are marked provisional in `configs/default.yaml`.

`compute_loss` also accepts `lambda_event_gate`, the sparsity weight for the optional posterior event
gate (`posterior_event_gate`, §10a). The gate is not built in the shipped configuration, and at weight
$0$ the term adds nothing.

**A term at weight $0$ is not computed**, and its metric reports exact $0.0$. The branch reads a
config-constant float, identical on every rank and every batch, so DDP graph identity is not
affected.

**`total_loss` is a mixed-unit criterion.** The shape terms are L1 and Huber quantities on z-scored
raw samples, not nats. The pure-nats readouts are `nll_full_block` and `nll_base_block`.

## 2. The two encoders

### 2.1 The input adapter

`AvailabilityInputAdapter` lives in `teb_vae/lag_attn/nets/encoders.py` and is re-exported from
`nets/encoders.py`. It reproduces the comparison model's
`InputAdapter(post_residual_activation=False)` stack — `Linear -> LayerNorm -> GELU -> Dropout ->
ResidualMLP`, submodule for submodule and name for name — with two terms added to the first
linear's output:

$$
e_t = W_x \left(x_t \odot m_t\right) + W_m\!\left(m_t - \mathbf 1\right)
    + \mathbb 1\!\left[\textstyle\sum_c m_{t,c} = 0\right] e_{\mathrm{start}},
\qquad m_{t,c} = \mathbb 1[t \ge \delta_c].
$$

$m$ depends only on the per-channel delays $\delta_c$ that the causal input guard resolves. It is
therefore a constant $(T, C)$ pattern, not a function of the batch. The forward needs **no new
argument**, so the three-tensor forward signature, the imported plotting callback and the imported
permutation control stay unchanged.

The projection reads $m_t - \mathbf 1$ rather than $m_t$. The two differ by the constant
$W_m \mathbf 1$, which the first linear's bias already spans, so they define the same model. In this
form the term is *exactly* zero wherever every channel is available, which is everywhere past the
delayed prefix.

The adapter exists because, under a finite reach budget, the first $\max_c \delta_c$ steps of every
surviving channel are exact zeros. A zero token that enters repeated normalisation layers produces
derivatives of order $1/\sqrt{\epsilon}$. The availability terms give those positions a non-zero
input.

### 2.2 The stem

`nets/blocks.py::GatedCausalConvBlock`, block $b$ at kernel $k_b$ and dilation $r_b$:

$$
N_b = \operatorname{RMSNorm}(C^{b-1}), \qquad
[G^{(1)}_b, G^{(2)}_b] = W^b_{\mathrm{in}} N_b, \qquad
G_b = G^{(1)}_b \odot \sigma\!\left(G^{(2)}_b\right),
$$
$$
D_b = \operatorname{DWConv}^{\mathrm{causal}}_{k_b, r_b}(G_b), \qquad
R_b = W^b_{\mathrm{out}} \operatorname{SiLU}\!\left(\operatorname{RMSNorm}(D_b)\right),
\qquad
C^b = C^{b-1} + \gamma^{\mathrm{conv}}_b \odot \operatorname{Dropout}(R_b),
$$

with $W^b_{\mathrm{in}}: \mathbb R^{128} \to \mathbb R^{256}$, all projections bias-free, the
depthwise convolution in $128$ groups, and $\gamma^{\mathrm{conv}}_b \in \mathbb R^{128}$ a
LayerScale vector at $10^{-2}$.

Causality is explicit. The left padding $P_b = (k_b - 1) r_b$ is applied to the input *before* the
convolution, and the underlying `Conv1d` has `padding == 0`. A non-zero `padding` argument would pad
both ends and read the future. `tests/test_source_window.py` asserts this structurally over every
convolution in both encoders.

A stem of **zero** blocks is legal and builds a working module. The stem-free architecture arm uses
this, so that arm is a config delta and not a code path.

`nets/encoders.py::GatedCausalConvStem` wraps the same block stack and the same closing `RMSNorm`
without the attention stack. The model builds it only when `lag_kv_source: conv_stem` (§2.5).

### 2.3 The attention blocks

`nets/blocks.py::CausalTransformerBlock`, pre-normalised, with LayerScale on both sublayers:

$$
S^{n,\mathrm{attn}} = S^{n-1} + \gamma^{\mathrm{attn}}_n \odot
\operatorname{Dropout}\!\left(\operatorname{MHSA}\!\left(\operatorname{RMSNorm}(S^{n-1})\right)\right),
$$
$$
S^n = S^{n,\mathrm{attn}} + \gamma^{\mathrm{ffn}}_n \odot
\operatorname{Dropout}\!\left(F_n\!\left(\operatorname{RMSNorm}\!\left(S^{n,\mathrm{attn}}\right)\right)\right),
$$

with rotary position encoding on queries and keys, $H_e = 4$ heads of width $32$, and
$F_n(x) = W^o_n\left[\operatorname{SiLU}(W^g_n x) \odot W^v_n x\right]$ at
$d_{\mathrm{ff}} = 512$.

**The encoder self-attention heads are unrelated** to the $M = 4$ lag-attention heads and to the
$d_z / M = 16$ latent groups. The counts are equal at the shipped configuration, but nothing couples
them. `tests/test_construct.py` builds the model at `encoder_num_heads != num_heads` and asserts that
the posterior structure and the lag-map identity do not change.

**Masks.** The target uses the full causal prefix, through `F.scaled_dot_product_attention` with
`is_causal=True`. The source uses a bounded causal window $0 \le t - j < W_U$, through an explicit
boolean mask held as a non-persistent buffer. The constructor refuses a stated `is_causal` that
contradicts the window, in both directions. Neither form can produce a fully masked row, because
both always admit $j = t$ and encoder self-attention does no data-driven validity masking (§11).
`tests/test_attention_block.py` pins the exact masks.

### 2.4 The final normalisation

Each encoder ends in an `RMSNorm(128)`, which adds $128$ parameters per encoder. The comparison
model's `CausalConvLstmEncoder` ends in `output_norm = LayerNorm(128)`, so the prior head, the
posterior fusion and the lag attention's key-value normalisation all expect a normalised $H$. A
pre-norm residual stack without a final norm exports a residual stream whose scale grows with depth.

### 2.5 The lag-attention key-value source

`lag_kv_source` selects which source representation the lag attention's keys and values are built
from:

- `encoder` (the default and the shipped value) uses $H^U$ from the windowed source encoder;
- `conv_stem` uses `GatedCausalConvStem` over the same stem schedule;
- `adapter` uses the projected source stream directly, with one step of reach.

Under the two local options the deep source encoder is not built at all, so
`source_attention_blocks` and `source_attention_window` apply to the `encoder` option only.

## 3. Receptive fields, and why the source is bounded

$$
R_{\mathrm{conv}} = 1 + \sum_b (k_b - 1) r_b = 1 + 4 \cdot 1 + 8 \cdot 2 = 21 \text{ steps } (84\text{ s}),
$$
$$
R_U = \min\!\left(R_{\mathrm{conv}} + N_U (W_U - 1),\ T\right) = 21 + 3 \cdot 15 = 66 \text{ steps } (264\text{ s}).
$$

The encoder's `receptive_field` property reports the target's reach as `None` (unbounded) rather
than as $T$. The bound is *absent*, and a number would not show that.

$R_U = 264$ s is shorter than the lag search range — $\ell \in \{0, \ldots, 90\}$ spans $90$ steps,
$360$ s. The source encoder characterises a *local* neighbourhood, and the lag attention selects
which neighbourhood matters. An encoder whose reach exceeded the lag range would already do the
alignment that the lag attention exists to do.

`tests/test_source_window.py` checks the bound directly. It perturbs the source encoder's input at
$t - R_U$ and at $t - R_U + 1$, and it requires the output at $t$ to be bitwise identical in the first
case and bitwise different in the second. The test runs in float64 with a large amplitude and uses
`torch.equal`. The edge-most path traverses three attention blocks, each scaled by LayerScale
$10^{-2}$ times an attention weight of order $1/W_U$, so in float32 the change would round to zero.

To print the receptive-field table, run:

```
python -m teb_vae.lag_attn_transformer_rws.nets.encoders
```

## 4. Forward return dict

`nets/model.py::SeqVaeLagAttnTrfRws.forward(y_st, y_ph, u_stream)` returns the twenty keys that the
comparison model returns, at the same shapes: `mu_prior`, `logvar_prior`, `raw_logvar_prior`,
`mu_post`, `logvar_post`, `z_prior`, `z_post`, `target_state`, `source_state`,
`attended_source_heads`, `attn_weights`, `mu_base`, `logvar_base`, `mu_full`, `logvar_full`,
`kld_per_t`, `kld_per_t_per_head`, `source_kl_lag_map`, `mu_prior_sat_frac`, `delta_mu_sat_frac`.
There is no `decoder_state` and no `delta_mu_src`, because neither pathway exists. A model built with
`posterior_event_gate: true` adds a twenty-first key, `event_gate`.

`tests/test_forward_contract.py` compares the key set and every shape against `SeqVaeLagAttnRws`
itself on the same stub batch.

## 5. Parameter budget

The counts below come from `sum(p.numel() for p in model.parameters())` on a model built from the
`VAE_model` block of `configs/default.yaml` with no reach budget. They include the frozen lag-attention
$W_o$.

| Component | Parameters |
| --- | ---: |
| Convolution block, $k = 5$ ($3d^2 + 3d + dk$) | $50{,}176$ |
| Convolution block, $k = 9$ | $50{,}688$ |
| Attention block ($4d^2 + 3 d\,d_{\mathrm{ff}} + 4d$) | $262{,}656$ |
| Target encoder, $2$ conv $+$ $6$ attention $+$ final norm | $1{,}676{,}928$ |
| Source encoder, $2$ conv $+$ $3$ attention $+$ final norm | $888{,}960$ |
| Both encoders | $2{,}565{,}888$ |
| Everything else, adapters and the shared decoder included | $2{,}430{,}956$ |
| **Total** | $\mathbf{4{,}996{,}844}$ |

No test pins these totals. A shared change to an imported downstream component changes the last two
rows.

Inside the last row, each of the two horizon self-attention blocks has $262{,}657$ parameters: four
bias-free $256 \times 256$ projections, one `LayerNorm` and one scalar residual gain, so
$4 d_{\mathrm{hidden}}^2 + 2 d_{\mathrm{hidden}} + 1$ at `decoder_hidden: 256`.
`horizon_attention_blocks: 0` builds no module at all, so `sweep_horizon_attn_off.yaml` removes
exactly $525{,}314$ parameters.

Under a finite reach budget each adapter adds a bias-free availability projection $W_m$ of
$128\, c^{\mathrm{kept}}$ parameters, where $c^{\mathrm{kept}}$ is the stream's surviving channel
count, and a $128$-wide $e_{\mathrm{start}}$ when every channel is delayed (§8). The shipped
`causal_reach_budget_s: 120` therefore adds these terms to the total above.

## 6. Structural constraints

Each constraint is enforced by construction and checked by a test.

| Constraint | What enforces it | Test |
| --- | --- | --- |
| **Token causality**, $H_t = f(X_{\le t})$, per block and per encoder | left-only convolution padding; `is_causal` or windowed SDPA masks; every other primitive is position-wise | `tests/test_blocks.py`, `tests/test_attention_block.py`, `tests/test_encoder_causality.py`, `tests/test_source_purity.py` |
| **Prefix equivalence**, $\mathcal E(X_{0:T-1})_t = \mathcal E(X_{0:t})_t$ | rotary positions are absolute and start at zero; no right padding anywhere | `tests/test_prefix_equivalence.py` |
| **Source purity** — the prior never sees the source, the source state never sees the target | separate adapters and encoders; the posterior is a residual on the prior | `tests/test_source_purity.py` |
| **No decoder bypass** — the decoder receives no source-derived tensor except through $z$ | the decoder takes $z$ at $d_z$ in-features; its optional persistence input is refused by this architecture (`persistence_residual` must be `false`) | `tests/test_no_bypass.py` |
| **Exact zero KL at initialisation** | posterior deltas zeroed **after** the generic init; one shared $\epsilon$; decoder and attention dropout fixed at $0$ | `tests/test_zero_kl_init.py` |
| **The lag attribution identity**, $\sum_\ell \widetilde K_{t,\ell} = K_t$ | the lag attention is built at `dropout=0.0`, so the returned probabilities are the ones the posterior consumed | `tests/test_lag_map.py` |
| **The bounded source reach** $R_U$ | the window mask | `tests/test_source_window.py` |
| **No recurrence and no time-pooling normaliser** in the history path | none is constructed on either gate, adapter or encoder; the horizon core's `GroupNorm`s pool the *forecast* axis of one anchor | `tests/test_construct.py` |
| **`lag_attn.W_o` frozen** | the head-structured posterior consumes the per-head summaries, so `W_o` receives no gradient; freezing removes it from DDP's expectation set | `tests/test_ddp_reachability.py` |

`tests/test_zero_kl_init.py` uses the constructor defaults (`base_decode: sample`,
`posterior_logvar_mode: residual`). Under them the base and full forecasts are also bitwise identical
at initialisation, in train mode and in eval mode. §10a gives the shipped-configuration case.

Two conventions apply to the whole suite. **Every positive invariant test has a negative test that
shows the probe is not vacuous.** The KL is identically zero at initialisation, so a KL assertion on
a fresh model passes on a broken one. The same trap has three instances in this architecture: a dead
layer passes every per-token invariant, a no-op rotary encoding passes every positional one, and a
zero-initialised $W_m$ passes every availability one. And **the causality negative control needs no
test hook in production code**: perturbing strictly after $t_0$ must leave $H_{t_0}$ bit-stable *and*
must move $H_{T-1}$.

## 7. Initialisation order

The order in `nets/model.py::__init__` is load-bearing, top to bottom:

1. `initialization(self)` — the shared generic pass, Xavier over every `nn.Linear` and `nn.Conv1d`.
2. `init_depthwise_(self)` — **immediately after**, never before. Xavier on a $(C, 1, k)$ depthwise
   weight reads $\mathrm{fan\_in} = k$ against $\mathrm{fan\_out} = Ck$, which gives
   $\sigma = \sqrt{2/(k + Ck)}$ against the variance-preserving $1/\sqrt k$. That is a factor
   $\sqrt{(1 + C)/2} = 8.03$ too small at $C = 128$, **independent of $k$**. At $k = 5$ the values
   are $0.0557$ against the target $0.447$.
3. `_zero_init_delta_heads()` — the generic pass would otherwise refill the posterior delta heads
   and remove the exact zero-KL start.
4. `prior_head.zero_init_clock()` and `posterior_head.zero_init_event_gate()` — the same repair for
   the prior's availability-clock projection and the posterior's event gate. Each is a no-op on a
   head built without that path.
5. `_zero_init_film_generators()` — the horizon core zero-initialises the FiLM generators itself and
   the generic pass refills them, so re-zeroing here makes the identity at initialisation true.
6. The three zero-parameter init policies — `horizon_embed_std`, `head_init_calibration`,
   `a_head_gain` — each applied only when its config value differs from the constructor default.
   They touch disjoint parameters, so their order among themselves does not matter.

`init_depthwise_` runs only when `init_weights` is true. It repairs what Xavier does to a depthwise
weight, and the torch `Conv1d` default already reads the depthwise fan correctly. The model records
`n_depthwise_init`, the count the pass returns, so a test can tell "the correction ran" from "the
correction was a no-op". The stem-free arm has zero.

## 8. DDP reachability

Production runs under DDP with `find_unused_parameters=False` (for `gaussian_nll`), so every
parameter must be reachable.

A parameter multiplied by an identically-zero tensor **is** reachable: its `AccumulateGrad` node
fires, it receives a zero gradient rather than `None`, and DDP's reducer marks it ready. What breaks
`find_unused_parameters=False` is a parameter left *out of the graph* — for example, a Python-level
`if all_zero.any(): e = e + e_start` that skips the addition on some ranks or batches.

So the rule is: **the availability terms are added unconditionally in the forward; branching happens
at construction time only.** `tests/test_ddp_reachability.py` asserts both halves. Every
`requires_grad` parameter has a gradient after one backward, under the unguarded configuration and
under a guarded one, with a deliberately dangling parameter as the negative control. An AST walk over
`nets/blocks.py`, `nets/encoders.py` and `nets/model.py` finds no `if` or conditional expression
inside a `forward` whose test reads a tensor *value*.

The walk admits shape metadata, because three legitimate conditionals exist: two shape guards that
raise, and `self.left_padding > 0`. It accepts an expression built only from constants, names,
attributes, `.shape` subscripts, comparisons and boolean operators, recursively. It rejects any
call, any element subscript and any arithmetic. Its one known gap is that `self.some_tensor > 0`
would be admitted; such a test raises on the first forward. The adapter's two `is None` tests read
module attributes fixed at construction, so they are identical on every rank at every step.

The two availability parameters have **different** construction conditions:

* $W_m$ exists when $\max_c \delta_c > 0$. Below that $m \equiv 1$, the term is identically zero,
  and the projection could never receive a gradient.
* $e_{\mathrm{start}}$ exists when $\min_c \delta_c > 0$. The indicator is non-zero for some $t$
  only when *every* channel is delayed, so a mixed delay vector such as $(0, 3, 5)$ satisfies
  $\max > 0$ while the start token stays inert.

Under `causal_reach_budget_s: null` there is no gate object, both conditions are false, and neither
parameter is constructed.

## 9. Token causality is not raw-signal causality

**Token causality**, $H_t = f(X_{\le t})$, is what the architecture guarantees and §6 tests.

**Raw-signal causality**, $H_t = f(Y_{\le n_{\mathrm{raw}}(t)}, U_{\le n_{\mathrm{raw}}(t)})$,
requires one-sided feature transforms. The stored features are two-sided wavelet transforms, so a
feature step reads raw signal from its own future. No encoder can repair that.
`causal_reach_budget_s` bounds the leak by pruning channels whose analytic reach exceeds the budget
and delaying the survivors. It does not eliminate the leak, because $L_{95}$ is an energy *quantile*
and $5\%$ of every filter's energy lies beyond its stated reach. The evaluation records the
longest-reaching channel as `max_channel_reach_s`.

`tests/test_train_smoke.py` runs real CPU fits, unguarded and guarded, in the slow suite.

> lean-limit: token-causal features; replace with genuinely one-sided scattering and phase-harmonic
> transforms when a causal front end exists.

The KL is named `source_conditioned_kl_raw` / `_train`, and its decomposition is
`source_kl_lag_map`. It is **not** called transfer entropy, for the reason above.

## 10. Deliberate limitations

> lean-limit: primitives imported from `teb_vae/lag_attn/nets` and `teb_vae/lag_attn_rws/nets`;
> promote to a common package when this model outlives its comparison, or when a consumer needs to
> modify one of the shared primitives rather than only import it.

Every package that uses the shared primitives imports them without modification, so the promotion is
not done.

Also deliberate:

- **No mixed precision.** `precision: "32-true"`. Mixed precision needs float32 islands around the
  log-variances, the closed-form KL and the Gaussian NLL reduction.
- **Compilation is available.** `LagAttnTrfRwsTrainer.compile_model_requested` honours
  `advanced_config.trainer.compile`, which `configs/default.yaml` sets to `true`. Only the net's
  **forward** is compiled. `compute_loss_and_metrics` reaches the objective through `orig_model`, so
  the data-dependent mask indexing behind `kld_active_frac` does not enter the graph. The driver
  refuses `compile` together with `attention_grad_checkpoint`, and it names both keys. Inductor may
  reassociate float arithmetic, and `pred_gap` is a small difference between two large block NLLs.
  The driver logs that compilation is on and asks the operator to compare `pred_gap` against an
  eager run on one fixed batch.
- **The DDP strategy is a configured `DDPStrategy`, not a shorthand string.** It is inherited from
  `LagAttnRwsTrainer.ddp_kwargs`: `find_unused_parameters` keyed on the likelihood, plus
  `broadcast_buffers=False` (every buffer here is a deterministic function of the config, and there
  is no `BatchNorm`) and `gradient_as_bucket_view=True`. `static_graph` is absent: the loss-spike
  breaker substitutes a zero-weighted sum over every parameter on a skipped batch, which is a
  different backward from the one the first iteration recorded.
- **Warm start only from this model's own checkpoints.** A comparison-model checkpoint carries a
  different `model_class` stamp and different encoder tensors, and `check_model_class` refuses it.
  `configs/default.yaml` sets `core_model_checkpoint` to a checkpoint of this model.
- **The evaluation pipeline has its own contract.** It ships in `eval/`, documented in
  `eval/EVAL.md`. During training the readouts are `train/grad_norm`, the tracked metric surface and
  the per-epoch diagnostic figure (§12.2).

## 10a. The bottleneck keys

Four keys act on two mechanisms. They are separate keys because they are separate mechanisms, and
each has an ablate-one arm.

**The prior's scale.** Two gradient paths push the prior log-variance $\ell^p$ down, and each key
removes one:

| Key | Path it removes | Shipped | Arm |
| --- | --- | --- | --- |
| `base_decode: mean` | $D_0$ decodes a *sample* from the prior; sampling noise can only degrade the forecast, so its gradient on $\ell^p$ points down without limit | `mean` | `sweep_base_sample.yaml` |
| `posterior_logvar_mode: independent` | $\ell^q = \mathrm{sb}(\tilde\ell^p + \Delta\ell)$ routes $D_1$'s pressure to sharpen $\sigma^q$ onto the **prior's** tensor | `independent` | `sweep_logvar_residual.yaml` |

With both shipped, $\ell^p$ receives gradient from $\mathrm{KL}(q \Vert p)$, which pulls it *up* to
cover $q$, and from the scale anchor $\beta_p R_p$. The posterior branch is still sampled; only the
base branch decodes at $\mu^p$. `beta_prior: 0.0` removes the fourth term and leaves the `prior_rate`
metric reported.

**The source pathway.** `source_dropout` regularises the source pathway alone: the source adapter,
the source encoder, and the attended summary inside the posterior fusion. It does not reach the lag
attention or the shared decoder. `null` keeps the adapter and encoder at the global `dropout` and the
posterior-fusion site at $0$; any explicit value applies to all three sites.
`sweep_source_dropout_0p2/0p3` set it.

**The causal input guard.** `causal_reach_budget_s: 120` ships. `sweep_reach_null.yaml` sets it to
`null`, where the two-sided features reach further into their own future.

**At initialisation under the shipped keys**, the base and full forecasts are not bitwise identical,
because the full branch still carries the posterior's noise. The KL is still exactly $0$, because it
is a function of the two distributions and not of the samples: `head_init_calibration` pins the
prior's raw log-variance to a constant and writes the same constant into the independent head, so
both log-variances start at $0$. `head_init_calibration` also calibrates the decoder's output heads.
The calibration is written into the weights at construction, so a resumed run carries it.
`tests/test_bottleneck_keys.py` covers this section.

`posterior_event_gate` is a further, unshipped option. It builds a scalar gate
$g_t \in (0, 1)$ per step, read from the attended source summary, that multiplies both bounded
posterior residuals; at $g_t = 0$ the posterior equals the prior. `lambda_event_gate` weights its
sparsity term.

## 11. Deviation record

Every intentional difference between the built module and the design it was built from.

**Architecture**

- **A final `RMSNorm(128)` per encoder**, $128$ parameters each (§2.4).
- **The availability projection reads $m_t - \mathbf 1$, not $m_t$** (§2.1).
- **The encoder attention ignores the loader's `weight`.** The design's masks also excluded invalid
  keys; this build does not. Masking data gaps out of encoder self-attention would make this model
  condition on strictly *less* than the comparison model. It also makes the fully-masked-row case
  unreachable.
- **The availability parameters are constructed conditionally**, on different conditions for the two
  terms (§8). The *forward* is unconditional.
- **The model is a standalone `nn.Module`, not a subclass** of the comparison model. The comparison
  model's constructor builds the replaced encoders and validates a keyword schema that this
  architecture does not have. The **objective** is shared: `compute_loss` and `kld_tensor` live in
  `teb_vae/lag_attn_rws/nets/losses.py`, and both models call them. `LOGVAR_FLOOR_MARGIN_FRAC` is
  re-exported from `nets/model.py`.
- **Five constructor keys of the comparison model are absent**: `lstm_layers`,
  `encoder_extra_dilations`, `encoder_extra_kernel`, `conv_norm_groups` and `causal_norm`. Passing
  any of them raises `TypeError`.
- **Two keys exist only for composing cells and are refused here.** `prior_availability_input` and
  `persistence_residual` are on the constructor signature so that a subclass can reach them through
  the driver's `inspect.signature` sweep. This architecture raises `ValueError` when either is true.

**Configuration and driver**

- **`configs/default.yaml` is written out in full**, not inherited from the comparison model's. The
  two `VAE_model` blocks do not share a schema. `tests/test_config_load.py` walks every leaf of both
  files and requires equality outside the encoder schema, against an exemption table that is itself
  asserted to contain only real divergences. Four exemptions are identity — the output tree, the
  MLflow experiment, the run name and the variant tag.
- **`lr_warmup_steps` lives in `general_config`**, beside `lr` and `lr_milestone`. The driver's
  `create_model` forwards it through the framework's `apply_config_hyperparameters`, so it lands in
  `self.hparams` and in every checkpoint.
- **The learning-rate monitor is *replaced*, not added.** The framework attaches
  `LearningRateMonitor(logging_interval='epoch')`. The driver swaps it for a step-granular monitor,
  so exactly one monitor logs.
- **One `LambdaLR` rather than `SequentialLR`.** `LinearLR` rejects `start_factor=0.0`, and
  `SequentialLR` restarts the second scheduler's counter at the switch. At `lr_warmup_steps: 0` the
  task delegates to the framework.
- **The driver re-admits one `null`.** The inherited config sweep drops every `VAE_model` key set to
  `null`. For `source_attention_window`, `null` is a value: the unbounded source encoder.
  `NULLABLE_MODEL_KEYS` in `trainer.py` names the exception, and `tests/test_trainer.py` pins both
  directions.
- **`tiny.yaml` shrinks the widths but keeps the encoder shape.** `encoder_d_ff` is $64$; the stem
  kernels and dilations, both block counts and the source window keep the shipped values at the
  shipped $300$ steps, so the smoke fit exercises the bounded source reach.
- **`gradient_clip_val: 5000.0`**, marked `PROVISIONAL AT THIS CAPACITY` in `configs/default.yaml`.
  The spike breaker's `additive_margin` ($10^{3}$) carries a provisional marker too; its watched
  `main_loss` is mixed-unit (§1).

**Test-design facts**

- **An end-relative rotary encoding is not a prefix-equivalence negative control.** With positions
  running $T-1-t$, a rotary score depends only on $\operatorname{pos}(t) - \operatorname{pos}(j)$, so
  the length-dependent part cancels and the encoding is still exactly prefix-equivalent. The working
  control is a **length-normalised** rotary encoding, which places two steps a fixed distance apart
  at different displacements depending on sequence length. Both are in
  `tests/test_prefix_equivalence.py`, the first as a recorded non-control.
- **The lag-attribution tolerance is relative.** On a perturbed tiny model $K_t$ can be large, so an
  absolute $10^{-6}$ would depend on the perturbation's scale. `tests/test_lag_map.py` asserts
  `rtol=1e-6` with a $10^{-6}$ absolute floor for near-zero anchors.
- **The FiLM-zeroing negative control needs the generic pass on.** The horizon core zero-initialises
  the FiLM generators itself, and the refill *is* the generic pass, so with `init_weights=False` they
  stay zero. `tests/test_init_policies.py` asserts the FiLM half only under `init_weights=True`.
- **The ban on time-pooling normalisers is scoped to the history path.** The horizon core carries one
  `GroupNorm` per refine block — four at `horizon_depth: 4` — plus one `LayerNorm` per horizon
  self-attention block. All of them pool over the *forecast* axis of a single anchor.
  `tests/test_construct.py` bans the family on both gates, both adapters and both encoders, bans
  recurrence everywhere, and asserts that every remaining `GroupNorm` is under `horizon_core.`.

## 12. Running it

From the repository root:

```
# Production, 7 ranks. TEB_RUN_STAMP is required so ranks 1..N-1 share rank 0's run directory.
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_transformer_rws.trainer \
    --config teb_vae/lag_attn_transformer_rws/configs/default.yaml

# Local smoke: one epoch, one device, the committed four-sample shard.
python -m teb_vae.lag_attn_transformer_rws.trainer \
    --config teb_vae/lag_attn_transformer_rws/configs/tiny.yaml

# A sweep arm: the same launch line against any configs/sweep_*.yaml.
```

The rank count must equal `len(general_config.cuda_devices)`. To launch from the IDE Run button,
set `RUN_CONFIG` at the bottom of `trainer.py`; `--config` on the command line overrides it. A
relative path resolves against the repository root, and the entry point changes the working
directory to the repository root.

The test gate:

```
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_rws/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_rws/tests -q -m slow
```

### 12.1 Sweep arms

Every arm is `base: default.yaml` plus one declared delta. `tests/test_sweep_configs.py` lints each
arm against a declared-delta table. The arms share the baseline's tag, MLflow experiment and run
name, so a run is identified by its `TEB_RUN_STAMP` directory and the `resolved_config.yaml` written
beside its checkpoints. The comparison model `teb_vae/lag_attn_rws` is not a config here.

| Config | Delta against `default.yaml` | $\Delta$ params |
| --- | --- | ---: |
| `sweep_arch_a1.yaml` | `encoder_conv_kernels: []`, `encoder_conv_dilations: []`; `source_attention_blocks: 6`, `source_attention_window: null` | $+586{,}240$ |
| `sweep_arch_a2.yaml` | `source_attention_blocks: 6`, `source_attention_window: null` | $+787{,}968$ |
| `sweep_window_8.yaml` | `source_attention_window: 8` ($R_U = 42$) | $0$ |
| `sweep_window_32.yaml` | `source_attention_window: 32` ($R_U = 114$) | $0$ |
| `sweep_window_64.yaml` | `source_attention_window: 64` ($R_U = 210$) | $0$ |
| `sweep_window_full.yaml` | `source_attention_window: null` (unbounded) | $0$ |
| `sweep_reach_null.yaml` | `causal_reach_budget_s: null` | removes the availability terms |
| `sweep_base_sample.yaml` | `base_decode: sample` | $0$ |
| `sweep_logvar_residual.yaml` | `posterior_logvar_mode: residual` | $0$ |
| `sweep_source_dropout_0p2.yaml` | `source_dropout: 0.2` | $0$ |
| `sweep_source_dropout_0p3.yaml` | `source_dropout: 0.3` | $0$ |
| `sweep_target_blocks_4.yaml` | `target_attention_blocks: 4` | $-525{,}312$ |
| `sweep_target_blocks_8.yaml` | `target_attention_blocks: 8` | $+525{,}312$ |
| `sweep_source_blocks_2.yaml` | `source_attention_blocks: 2` | $-262{,}656$ |
| `sweep_source_blocks_4.yaml` | `source_attention_blocks: 4` | $+262{,}656$ |
| `sweep_ff_384.yaml` | `encoder_d_ff: 384` | $-442{,}368$ |
| `sweep_aux_off.yaml` | `lambda_ms`, `lambda_deriv`, `lambda_boundary` all `0.0` | $0$ |
| `sweep_horizon_attn_off.yaml` | `horizon_attention_blocks: 0` | $-525{,}314$ |
| `sweep_beta_prior_0p001.yaml` | `beta_prior: 1.0e-3` | $0$ |
| `sweep_beta_prior_0p01.yaml` | `beta_prior: 1.0e-2` | $0$ |
| `sweep_beta_prior_0p1.yaml` | restates the shipped `beta_prior: 0.1` | $0$ |
| `sweep_beta_prior_1p0.yaml` | `beta_prior: 1.0` | $0$ |

$d = 128$ is fixed across every arm. A1 and A2 differ by exactly the two stems' $201{,}728$
parameters.

`sweep_aux_off.yaml` optimises a different criterion, so its `total_loss` is not comparable with the
baseline's; compare `nll_full_block` and `nll_base_block` instead.

### 12.2 Training readouts

Every run writes `train_results/metrics_history.csv`, which carries every tracked metric per epoch.

- `train/grad_norm` is logged **pre-clip** in `on_before_optimizer_step`. It exists on training steps
  only, with no `val/` variant. `train/grad_clip_frac` is the per-step indicator that the norm
  exceeded `gradient_clip_val`.
- Both are **sampled, not reduced**. `MetricsLoggingCallback` reads `trainer.callback_metrics` at
  validation end, before the training epoch is reduced, so each CSV row carries one optimizer step.
  `train/spike_skipped` reaches the CSV the same way. Every `val/` column is a true epoch mean.
- The spike breaker gates the training path only. On a skipped batch it replaces `train/total_loss`
  and `train/main_loss` with its EMA before they reach the logger. `configs/default.yaml` ships
  `spike_breaker.enabled: false`.
- The shared collapse criterion is in `teb_vae/lag_attn_rws/collapse.py`. A completed run is
  collapsed when `val/source_conditioned_kl_raw` is below $0.02$ nats per anchor at every one of its
  last $5$ epochs, or when its final `val/kld_active_frac` is below $2 / d_z$.

## 13. Configuration keys

`configs/default.yaml` sets every key in the first list and none in the second.
`tests/test_config_load.py` checks that every `model_config.VAE_model` key in the shipped config
reaches the constructor or the task. Outside `VAE_model`, the first list names the keys this
document refers to, not every framework setting.

**Required**

- `general_config.seed`
- `general_config.lr`
- `general_config.lr_milestone`
- `general_config.lr_warmup_steps`
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
- `model_config.VAE_model.decoder_hidden`
- `model_config.VAE_model.logvar_clamp`
- `model_config.VAE_model.mu_scale`
- `model_config.VAE_model.delta_mu_scale`
- `model_config.VAE_model.delta_logvar_scale`
- `model_config.VAE_model.coverage_floor`
- `model_config.VAE_model.base_decode`
- `model_config.VAE_model.posterior_logvar_mode`
- `model_config.VAE_model.source_dropout`
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
- `advanced_config.trainer.compile`
- `advanced_config.trainer.num_sanity_val_steps`
- `advanced_config.trainer.use_distributed_sampler`
- `advanced_config.spike_breaker.ema_floor`
- `advanced_config.spike_breaker.additive_margin`
- `advanced_config.spike_breaker.comparison_metric`
- `dataset_config.stat_path`
- `dataset_config.dataloader_config.normalize_fields`

**Deliberately absent**

The five keys that describe the replaced encoder:

- `model_config.VAE_model.lstm_layers`
- `model_config.VAE_model.encoder_extra_dilations`
- `model_config.VAE_model.encoder_extra_kernel`
- `model_config.VAE_model.conv_norm_groups`
- `model_config.VAE_model.causal_norm`

The keys this architecture makes structural, derived or inert:

- `model_config.VAE_model.horizon_attention_heads`
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

The horizon attention's head count is a constructor argument on the shared core, with default $4$.
The encoder head width is derived as `d_model // encoder_num_heads`. Encoder dropout is the existing
`dropout`. The LayerScale initialisation and the rotary base are constructor defaults that no arm
varies. The target attention window is the full context in every arm. Encoder gradient checkpointing
has no key, because `batch_size` with `accumulate_grad_batches` are the memory levers.

Constructor keys that `configs/default.yaml` leaves at their defaults include `lag_kv_source`,
`posterior_event_gate`, `alibi_slope_scale` and `horizon_weight_halflife_steps`.
