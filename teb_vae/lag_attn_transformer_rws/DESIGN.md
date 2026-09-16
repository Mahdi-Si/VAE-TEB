# `lag_attn_transformer_rws` — the as-built design record

The causal conv-Transformer raw-signal lag-attention VAE-TEB: what it is, what it replaces, what it
deliberately leaves alone, and every place the built module differs from the design it was built
from.

Companion documents: `new_architecture.md` in this directory is the as-built architecture reference
— the full forward pass, its equations, and what the original proposal specified but this build did
not do; `teb_vae/lag_attn_rws/DESIGN.md` records the model this one is compared against, whose
objective, data contract, geometry and metric surface are shared unchanged. Neither is restated
here — this record covers the encoders and the wiring around them.

---

## 1. What the model is

`teb_vae/lag_attn_rws` with **both history encoders replaced** and nothing else changed. At every
4-second anchor $t$ the model still forecasts the next two minutes of raw normalized FHR —
$H \cdot R = 30 \times 16 = 480$ samples — twice: once from a target-only latent and once from a
source-conditioned one, through one shared decoder invoked twice under one noise draw. The KL
between the two latents, resolved across lags, is the coupling readout.

What changed is how the two history states $H^Y$ and $H^U$ are computed. The encoder being
replaced is a five-block dilated causal convolution stack running in parallel with a two-layer
unidirectional LSTM, fused by a wide MLP — $2{,}657{,}230$ parameters across the two streams,
$78.8\%$ of that model. In its place, per stream:

$$
X^s \longrightarrow \mathcal G_s \longrightarrow \operatorname{InputAdapter}_s
\longrightarrow \operatorname{CausalConvStem}_s
\longrightarrow \operatorname{CausalTransformer}_s \longrightarrow H^s
\in \mathbb R^{B \times T \times 128}.
$$

Three arguments motivate the replacement, and each is a property of the geometry rather than a
preference. The convolution stack's receptive field ($391$ steps) already exceeds the segment
($300$ steps), so it duplicates the recurrent branch's role. The recurrent bottleneck both
serialises training and blurs adjacent source lags, which is precisely the resolution the late lag
cross-attention depends on. And a content-dependent direct path from $j$ to $t$ is a better
conditional prior than a fixed dilation schedule plus a fixed-width recurrent state.

At the shipped configuration the model holds **4,996,844 parameters** (measured, §5), a $1.8\%$
reduction against $5{,}088{,}186$. That margin used to be $38.7\%$: the capacity revision raised
this architecture's encoders (six target blocks at $d_{\mathrm{ff}} = 512$ against four at $256$)
and left the comparison model's alone, so the two are now near parity in budget. The comparison is
better posed for it -- an encoder axis read at matched capacity attributes a forecast difference to
the encoder's structure rather than to its size.

Everything downstream of the two encoder outputs is imported, unmodified, from
`teb_vae.lag_attn_rws.nets` and `teb_vae.lag_attn.nets`: the target-only conditional prior head,
the lag-restricted cross-attention over $L = 91$ lags with $M = 4$ heads, the head-structured
bounded posterior residual, the shared horizon decoder, the lag-resolved KL attribution, the masks
and the objective in nats per anchor. A change to any of them belongs in those packages, where both
models get it. Two such changes landed with the capacity revision — the shape terms below, and
`horizon_attention_blocks: 2`, which runs two bidirectional self-attention blocks over the $H = 30$
horizon tokens inside the shared decoder core after its dilated refine stack (the core's own record
is `teb_vae/lag_attn_rws/DESIGN.md`; §5 here carries what the blocks cost). Both are shared, so the
encoder comparison this package exists to make still holds everything but the encoders fixed.

**The objective is inherited, and it has seven terms** — stated here rather than left implicit,
because "unchanged" is not the same claim as "unchanged and still four terms". `compute_loss` in
`teb_vae/lag_attn_rws/nets/losses.py` optimises

$$
\mathcal L = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0 + \beta(e)\, K_{\mathrm{train}}
  + \beta_p\, R_p
  + \lambda_{\mathrm{ms}} \mathcal L_{\mathrm{ms}}
  + \lambda_{\mathrm{deriv}} \mathcal L_{\mathrm{deriv}}
  + \lambda_{\mathrm{boundary}} \mathcal L_{\mathrm{boundary}} ,
$$

where $R_p$ is the prior's scale rate,
$\mathrm{KL}\!\left(\mathcal N(\mu^p, \operatorname{diag} e^{\ell^p}) \Vert \mathcal N(\mu^p, I)\right)$,
reduced on the KL's own anchor support and in the same nats-per-anchor units. It was added because
nothing else in the objective penalises a *narrow* prior, and **this architecture's first production
run is where that was measured**: `logvar_prior_floor_frac` reached $0.992$ inside one epoch, at
which point $K$ stops being a rate. `new_architecture.md` §12.5 carries the term and the numbers;
`beta_prior` ships at $0.1$ and is swept here.

The last three are the **auxiliary shape terms**, and the same production run is why they exist: a
factorized Gaussian NLL scores every raw sample independently, so its optimum is the conditional
mean and a fully parallel decoder is free to emit an over-smoothed one. Each is computed on the
forecast *means* of both branches and summed over them — pooled L1 at
rates $(1, 4, 16)$ for the envelope, a Huber on first differences for the slope, and the gap between
the first forecast sample and the anchor's last observed one for the starting level. The first two
carry the same coverage-floored forecast mask the NLL uses; the boundary term carries the anchor's
own thresholded `weight` instead, because the sample it reaches for belongs to that anchor and not
to the one whose forecast window supplied it. They ship at
$0.1 / 0.1 / 0.05$, all three **provisional**: no run has measured their magnitudes at this scale,
and `sweep_aux_off.yaml` prices the bundle as a whole. The sibling's `DESIGN.md` §5.1 is the full
record.

**A term at weight $0$ is not computed** and reports its metric as exact $0.0$ rather than as the
value it would have had. That keeps a term-off arm's CSV honest and keeps the full-block
intermediates out of that run's graph; the branch reads a config-constant float, identical on every
rank and every batch, so DDP graph identity is untouched.

**`total_loss` is therefore a mixed-unit criterion.** The shape terms are L1 and Huber quantities on
z-scored raw samples, not nats. The pure-nats readouts are `nll_full_block` and `nll_base_block`,
and `RESULTS.md`'s go/no-go section says which of the two to read for what.

## 2. The two encoders

### 2.1 The input adapter, and what it adds

`nets/encoders.py::AvailabilityInputAdapter` reproduces the comparison model's
`InputAdapter(post_residual_activation=False)` stack — `Linear -> LayerNorm -> GELU -> Dropout ->
ResidualMLP`, submodule for submodule and name for name — with two terms added to the first
linear's output:

$$
e_t = W_x \bar x_t + W_m\!\left(m_t - \mathbf 1\right)
    + \mathbb 1\!\left[\textstyle\sum_c m_{t,c} = 0\right] e_{\mathrm{start}},
\qquad m_{t,c} = \mathbb 1[t \ge \delta_c].
$$

$m$ depends only on the per-channel delays $\delta_c$ the causal input guard resolves, so it is a
constant $(T, C)$ pattern rather than a function of the batch: **no new forward argument**, and
with it the three-tensor forward signature, the imported plotting callback and the imported
permutation control all survive untouched.

The projection reads $m_t - \mathbf 1$ rather than $m_t$. The two differ by the constant
$W_m \mathbf 1$, which the first linear's own bias already spans, so they are the same model —
but written this way the term is *exactly* zero wherever every channel is available, which is
everywhere past the delayed prefix. The availability mechanism therefore cannot move the
representation on the part of the sequence where nothing is missing, and cannot confound the
comparison this package exists to make.

Why it exists at all: under a finite reach budget the first $\max_c \delta_c$ steps of every
surviving channel are exact zeros. A zero token entering repeated normalisation layers produces
derivatives of order $1/\sqrt{\epsilon}$, and the comparison model's guarded configurations reach
global gradient norms around $10^{26}$ that way at *every* finite budget — a switch, not a
gradient, so raising the warm-up does not help and the clip coefficient leaves the run optimising
nothing but weight decay. §9 records what the same measurement gives here.

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

Causality is explicit rather than argued: the left padding $P_b = (k_b - 1) r_b$ is applied to the
input *before* the convolution, and the underlying `Conv1d` carries `padding == 0`. A non-zero
`padding` argument would pad both ends and read the future. `tests/test_source_window.py` asserts
that structurally over every convolution in both encoders.

A stem of **zero** blocks is legal and builds a working module. That is what the stem-free
architecture arm needs, and it is why the arm is a config delta rather than a code path.

### 2.3 The attention blocks

`nets/blocks.py::CausalTransformerBlock`, pre-normalised with LayerScale on both sublayers:

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
$d_{\mathrm{ff}} = 512$, the conventional $4 d_{\mathrm{model}}$.

**The encoder self-attention heads are unrelated** to the $M = 4$ lag-attention heads and to the
$d_z / M = 16$ latent groups. They happen to number the same at the shipped configuration; nothing
may couple them, and `tests/test_construct.py` builds at
`encoder_num_heads != num_heads` and asserts the posterior structure and the lag-map identity are
untouched.

**Masks.** The target uses the full causal prefix, through `F.scaled_dot_product_attention` with
`is_causal=True`. The source uses a bounded causal window $0 \le t - j < W_U$, through an explicit
bool mask held as a non-persistent buffer. Never both: the constructor refuses a window together
with an explicit `is_causal`, in both directions. Neither form can produce a fully masked row,
because both always admit $j = t$ and there is no data-driven validity masking in encoder
self-attention (§10, third bullet) — so `tests/test_attention_block.py` asserts
`mask.any(-1).all()` structurally rather than testing a NaN path the architecture cannot enter.

### 2.4 The final normalisation

Each encoder ends in an `RMSNorm(128)`. This is a **deliberate addition** to the proposal's
parameter arithmetic — $128$ parameters per encoder — and it is a contract rather than a
preference. `CausalConvLstmEncoder` ends in `output_norm = LayerNorm(128)`, so the prior head, the
posterior fusion and the lag attention's key-value normalisation are all calibrated to a normalised
$H$. A pre-norm residual stack without a final norm exports an unnormalised residual stream whose
scale grows with depth; dropping it would move the downstream operating point for reasons that have
nothing to do with the encoder architecture.

## 3. Receptive fields, and why the source is bounded

$$
R_{\mathrm{conv}} = 1 + \sum_b (k_b - 1) r_b = 1 + 4 \cdot 1 + 8 \cdot 2 = 21 \text{ steps } (84\text{ s}),
$$
$$
R_U = \min\!\left(R_{\mathrm{conv}} + N_U (W_U - 1),\ T\right) = 21 + 3 \cdot 15 = 66 \text{ steps } (264\text{ s}).
$$

The target's reach is reported as **unbounded** rather than as $T$: the bound is *absent*, and a
caller reading a number could not tell the two apart.

$R_U = 264$ s is shorter than the lag search range — $\ell \in \{0, \ldots, 90\}$ spans $90$ steps,
$360$ s — and that is the point. The source encoder characterises a *local* neighbourhood and the
lag attention selects which neighbourhood matters. An encoder whose reach exceeded the lag range
would already be doing the alignment the lag attention exists to do, and the reported lag would
stop being a statement about where the coupling came from.

The bound is **measured, not computed**: `tests/test_source_window.py` perturbs the source
encoder's input at $t - R_U$ and at $t - R_U + 1$ and requires the output at $t$ to be bitwise
identical in the first case and bitwise different in the second. It runs in float64 with a large
amplitude and asserts with `torch.equal` rather than against a magnitude threshold — the edge-most
path traverses three attention blocks each scaled by LayerScale $10^{-2}$ times an attention weight
of order $1/W_U$, so the movement is around $10^{-7}$ and rounds to exactly zero in float32.

Print the table for the shipped architecture:

```
python -m teb_vae.lag_attn_transformer_rws.nets.encoders
```

## 4. Forward return dict

`nets/model.py::SeqVaeLagAttnTrfRws.forward(y_st, y_ph, u_stream)` returns exactly the twenty keys
the comparison model returns, at the same shapes — `mu_prior`, `logvar_prior`, `raw_logvar_prior`,
`mu_post`, `logvar_post`, `z_prior`, `z_post`, `target_state`, `source_state`,
`attended_source_heads`, `attn_weights`, `mu_base`, `logvar_base`, `mu_full`, `logvar_full`,
`kld_per_t`, `kld_per_t_per_head`, `source_kl_lag_map`, `mu_prior_sat_frac`, `delta_mu_sat_frac`.
There is deliberately no `decoder_state` and no `delta_mu_src`: neither pathway exists.

`tests/test_forward_contract.py` compares the key set and every shape against
`SeqVaeLagAttnRws` itself on the same stub batch, rather than against a table written here — so
"the same twenty keys at the same shapes" is a fact about the two models rather than about two
copies of one list.

## 5. Parameter budget

Measured on a constructed shipped-geometry model, not predicted.

| Component | Parameters |
| --- | ---: |
| Convolution block, $k = 5$ | $50{,}176$ |
| Convolution block, $k = 9$ | $50{,}688$ |
| Attention block ($4d^2 + 3 d\,d_{\mathrm{ff}} + 4d$) | $262{,}656$ |
| Target encoder, $2$ conv $+$ $6$ attention $+$ final norm | $1{,}676{,}928$ |
| Source encoder, $2$ conv $+$ $3$ attention $+$ final norm | $888{,}960$ |
| Both encoders | $2{,}565{,}888$ |
| Everything else, adapters and the shared decoder included, imported unchanged | $2{,}430{,}956$ |
| **Total** | $\mathbf{4{,}996{,}844}$ |

That total is the **one place the absolute number is pinned**, and `tests/test_docs.py` checks it
against `sum(p.numel() ...)` rather than against a literal in a test — so a legitimate shared change
to an imported downstream component re-costs this line rather than failing a test in this package.
The per-block and per-encoder subtotals are pinned separately in `tests/test_construct.py`, as
deltas.

Inside that last row, the two horizon self-attention blocks are $262{,}657$ each — four bias-free
$256 \times 256$ projections, one `LayerNorm` and one scalar residual gain, so
$4 d_{\mathrm{hidden}}^2 + 2 d_{\mathrm{hidden}} + 1$ at the shipped `decoder_hidden: 256`.
`horizon_attention_blocks: 0` builds no module at all rather than an inert one, which is what makes
`sweep_horizon_attn_off.yaml` an exact $-525{,}314$ against the shipped configuration and leaves the
reverted core parameter-for-parameter the one that shipped before the blocks existed.

Under a finite reach budget the availability projections add $128\, c^{\mathrm{kept}}$ per stream —
and only then. The widths are the *surviving* channel counts, not $109$ and $58$: at the $120$ s
budget that is $128 \times (78 + 29) = 13{,}696$.

## 6. Structural constraints that are not preferences

Each is enforced by construction and measured by a test, never asserted by convention.

| Constraint | What enforces it | Test |
| --- | --- | --- |
| **Token causality**, $H_t = f(X_{\le t})$, per block and per encoder | left-only convolution padding; `is_causal` / windowed SDPA masks; every other primitive is position-wise | `tests/test_encoder_causality.py`, `tests/test_source_purity.py` |
| **Prefix equivalence**, $\mathcal E(X_{0:T-1})_t = \mathcal E(X_{0:t})_t$ | rotary positions are absolute and start at zero; no right padding anywhere | `tests/test_prefix_equivalence.py` |
| **Source purity** — the prior never sees the source, the source state never sees the target | separate adapters and encoders; the posterior is a residual on the prior | `tests/test_source_purity.py` |
| **No decoder bypass** — gradient reaches the decoder only through $z$ | `BaselineFutureDecoder.forward` takes exactly one tensor, at $d_z$ in-features | `tests/test_no_bypass.py` |
| **Exact zero KL at initialisation**, and bitwise identical base and full *in train mode* | posterior deltas zeroed **after** the generic init; one shared $\epsilon$; decoder and attention dropout fixed at $0$ | `tests/test_zero_kl_init.py` |
| **The lag attribution identity**, $\sum_\ell \widetilde K_{t,\ell} = K_t$, exactly | the lag attention is built at `dropout=0.0`, so the returned probabilities are the ones the posterior consumed | `tests/test_lag_map.py` |
| **The bounded source reach** $R_U$ | the window mask, measured rather than computed | `tests/test_source_window.py` |
| **No recurrence and no time-pooling normaliser** in the history path | none is constructed; the surviving `GroupNorm`s are enumerated and each asserted to be under `horizon_core.`, where they pool the *forecast* axis of one anchor | `tests/test_construct.py` |
| **`lag_attn.W_o` frozen** | the head-structured posterior consumes the per-head summaries, so `W_o` receives no gradient; freezing drops it from DDP's expectation set | `tests/test_construct.py` |

Two conventions run through the whole suite and are not optional here. **Every positive invariant
test is paired with a probe-is-not-vacuous negative test** — the KL is identically zero at
initialisation, so any KL assertion on a fresh model passes on a completely broken one, and the same
trap has three instances specific to this architecture: a dead layer passes every per-token
invariant, a no-op rotary encoding passes every positional one, and a zero-initialised $W_m$ passes
every availability one. And **the causality negative control needs no test hook in production
code**: the control is positional — perturbing strictly after $t_0$ must leave $H_{t_0}$ bit-stable
*and* must visibly move $H_{T-1}$ — so there is no switch in the model that exists only for tests.

## 7. Initialisation order

Load-bearing, top to bottom, in `nets/model.py::__init__`:

1. `initialization(self)` — the shared generic pass, Xavier over every `nn.Linear` and `nn.Conv1d`.
2. `init_depthwise_(self)` — **immediately after**, never before. Xavier on a $(C, 1, k)$ depthwise
   weight reads $\mathrm{fan\_in} = k$ against $\mathrm{fan\_out} = Ck$, giving
   $\sigma = \sqrt{2/(k + Ck)}$ against the variance-preserving $1/\sqrt k$ — a factor
   $\sqrt{(1 + C)/2} = 8.03$ too small at $C = 128$, **independent of $k$**, so the stem would start
   an order of magnitude too quiet and no kernel sweep could reveal it. Numerically at $k = 5$:
   $0.0557$ against the target $0.447$.
3. `_zero_init_delta_heads()` — the generic pass would otherwise refill the posterior delta heads
   and destroy the exact zero-KL start.
4. `_zero_init_film_generators()` — the horizon core zero-initialises them itself and the generic
   pass refills them, so re-zeroing here is what makes the identity-at-init actually true.
5. The three zero-parameter init policies — `horizon_embed_std`, `head_init_calibration`,
   `a_head_gain` — each applied only when its config value leaves the constructor default, so a
   default-flag model is bitwise the pre-bundle one. They touch disjoint parameters, so their order
   among themselves is immaterial.

`init_depthwise_` is gated on `init_weights` and the reason is **not** symmetry with the generic
pass: it exists to repair what Xavier does to a depthwise weight, and torch's own `Conv1d` default
already reads the depthwise fan correctly — so with no generic pass there is nothing to repair and
running it anyway would be a second, unrequested initialisation policy. The model records
`n_depthwise_init`, the count the pass returns, so a test can tell "the correction ran" from "the
correction was a no-op": the stem-free arm legitimately has zero.

## 8. DDP reachability, and the two parameters it governs

Production runs under plain `"ddp"` with `find_unused_parameters=False`, so every parameter must be
reachable. The intuitive reading of what that requires is wrong, and the difference decides this
adapter's shape.

A parameter multiplied by an identically-zero tensor **is** reachable: its `AccumulateGrad` node
fires, it receives a zeros gradient rather than `None`, and DDP's reducer marks it ready. What
breaks `find_unused_parameters=False` is a parameter left *out of the graph* — a Python-level
`if all_zero.any(): e = e + e_start` that skips the addition entirely, on some ranks and not
others, on some batches and not others.

So the rule is: **the availability terms are added unconditionally in the forward; the branching
happens at construction time only.** `tests/test_ddp_reachability.py` asserts both halves — every
`requires_grad` parameter has a gradient after one backward under both the unguarded and a guarded
configuration, with a deliberately dangling parameter as the negative control; and an AST walk over
`nets/blocks.py`, `nets/encoders.py` and `nets/model.py` finds no `if` or conditional expression
inside a `forward` whose test reads a tensor *value*.

That walk admits shape metadata deliberately, because three legitimate conditionals exist: two shape
guards that raise, and `self.left_padding > 0`. The rule it applies is "an expression built only
from constants, names, attributes, `.shape` subscripts, comparisons and boolean operators",
recursively — any call, any element subscript, any arithmetic is rejected. Its one stated gap is
that `self.some_tensor > 0` would be admitted, and would raise on the first forward rather than
diverge silently. The adapter's own two `is None` tests read module attributes fixed at
construction, so they are identical on every rank at every step.

Construction is conditional for parameter economy and honesty rather than for DDP, and the two
terms have **different** conditions because they become non-trivial at different points:

* $W_m$ exists when $\max_c \delta_c > 0$. Below that $m \equiv 1$, the term is identically zero,
  and the projection would be a parameter that can never receive a gradient.
* $e_{\mathrm{start}}$ exists when $\min_c \delta_c > 0$. The indicator is non-zero for some $t$
  exactly when *every* channel is delayed, so a mixed delay vector such as $(0, 3, 5)$ satisfies
  $\max > 0$ while leaving the start token permanently inert.

Under the shipped `causal_reach_budget_s: null` there is no gate object at all, both conditions are
false, and neither parameter is constructed — which is correct: without delays there is no all-zero
prefix for them to repair. Absent rather than `None`, following the same convention the model uses
for the gate itself, so `named_buffers()` on an unguarded adapter lists neither.

## 9. Token causality is not raw-signal causality

Two distinct properties, and this module delivers the first and reports it as the first.

**Token causality**, $H_t = f(X_{\le t})$, is what the architecture guarantees and §6 measures.

**Raw-signal causality**, $H_t = f(Y_{\le n_{\mathrm{raw}}(t)}, U_{\le n_{\mathrm{raw}}(t)})$,
requires genuinely one-sided feature transforms. The stored features are two-sided wavelet
transforms: a feature step reads raw signal from its own future, up to $974$ s of it. No encoder
can repair that. `causal_reach_budget_s` bounds the leak by pruning channels whose analytic reach
exceeds the budget and delaying the survivors; it does not eliminate it, because $L_{95}$ is an
energy *quantile* and $5\%$ of every filter's energy lies beyond its stated reach.

What this build changes is that a finite budget is now **trainable**. Measured per epoch on the
committed four-sample shard, pre-clip global gradient norm:

| | epoch 0 | epoch 1 | epoch 2 | max |
| --- | ---: | ---: | ---: | ---: |
| unguarded (`null`) | $95.0$ | $107$ | $113$ | $113.3$ |
| guarded, $120$ s | $122$ | $87.5$ | $76.4$ | $122.3$ |

Against the comparison model's $\approx 10^{26}$ at every finite budget. The guarded arm is now
inside the same order of magnitude as the unguarded one, so a reach arm trained here optimises its
own gradient rather than AdamW's weight decay. `tests/test_train_smoke.py` re-measures this on
every run of the slow suite; the equivalent test in the comparison package is a `strict` xfail and
stays one, because that defect is that model's.

> lean-limit: token-causal features; replace with genuinely one-sided scattering and phase-harmonic
> transforms when a causal front end exists.

The KL is `source_conditioned_kl_raw` / `_train` and its decomposition is `source_kl_lag_map`. It is
**not** called transfer entropy, for the reason above.

## 10. Deliberate limitations

> lean-limit: primitives imported from `teb_vae/lag_attn/nets` and `teb_vae/lag_attn_rws/nets`;
> promote to a common package when this model outlives its comparison, or when a consumer needs to
> modify one of the shared primitives rather than only import it.

Both triggers this note has carried have now fired. This package is the third consumer named by the
note in `teb_vae/lag_attn_rws/nets/__init__.py`; the fourth consumer that this package's own note
named as *its* trigger exists as well. Acting on either would refactor a model that is currently
training and touch all $81$ of its test files, and every one of the four packages imports the
primitives without modifying any of them — so the promotion is deferred again, with the trigger
restated as a condition that is not yet met rather than left silent.

Also deliberate:

- **No mixed precision.** `precision: "32-true"` for the first controlled comparison. Mixed
  precision needs float32 islands around the log-variances, the closed-form KL and the Gaussian NLL
  reduction, and any of those moving is a difference the comparison would attribute to the encoder.
- **`compile: false`, but the key is live rather than inert.** With the recurrence gone,
  `LagAttnTrfRwsTrainer.compile_model_requested` honours `advanced_config.trainer.compile`; the
  raw-signal base still refuses it outright, because *its* LSTM encoders defeat TorchInductor
  unconditionally. The other blocker once recorded here was never one for the compiled region:
  only the net's **forward** is compiled, and `compute_loss_and_metrics` reaches the objective
  through `orig_model`, so the data-dependent mask indexing behind `kld_active_frac` cannot enter
  the graph. `attention_grad_checkpoint` *is* a genuine blocker and is still reachable, so the two
  are refused together by name rather than one being silently dropped.

  It ships **off** for a numerical reason rather than a mechanical one: inductor may reassociate
  float arithmetic, and `pred_gap` is a difference of order $10^{-1}$ between two block NLLs of
  order $10^{2}$ — a relative scale of about $10^{-4}$, which is the one quantity in this model
  with no tolerance for reassociation. Adopting it means one epoch each way from the same seed on
  the same shards, comparing `pred_gap` between the two `metrics_history.csv` files. A run that
  compiled says so in its startup log, so the two are distinguishable afterwards.

- **The DDP strategy is a configured `DDPStrategy`, not a shorthand string.** Inherited from
  `LagAttnRwsTrainer.ddp_kwargs`: `find_unused_parameters` keyed on the likelihood as before, plus
  `broadcast_buffers=False` (every buffer here — rotary tables, causal masks, the raw-target index
  grid — is a deterministic function of the config, and there is no `BatchNorm` to carry a
  per-rank running statistic) and `gradient_as_bucket_view=True`. `static_graph` is deliberately
  absent: the loss-spike breaker substitutes a zero-weighted sum over every parameter on a skipped
  batch, which is a structurally different backward from the one the first iteration recorded.
- **No warm start.** `core_model_checkpoint` stays `null`: a comparison-model checkpoint carries a
  different `model_class` stamp and different encoder tensors, and `check_model_class` refuses it —
  correctly.
- **The gradient-clipping threshold was measured, and the capacity revision made it provisional
  again.** §11, last bullet, which also carries the spike breaker's margin.
- **The evaluation pipeline is not part of this record.** It ships in `eval/` and has its own
  contract in `eval/EVAL.md`. What this module ships *during training* is unchanged and is still
  the only readout available while a run is in flight: `train/grad_norm`, the tracked metric
  surface and the per-epoch diagnostic figure.
- **`MODEL_DOCUMENTATION.md` is deferred** until the architecture stops moving.

## 10a. The bottleneck bundle

Four keys added together, each aimed at one of two measurements from the $125$-epoch production
run (`output/2026-08-05/metrics_history_rr.csv`). They are separate keys because they are separate
mechanisms, and each ships with an ablate-one arm so the next run stays attributable.

**Measurement one — the prior's scale is still collapsing.** `logvar_prior_floor_frac` climbs
$0.017 \to 0.496$ and is still rising at epoch $125$. The `beta_prior` anchor slowed it and did
not stop it, and §7 says why: the anchor's restoring force saturates at $\beta_p/2$ per dimension
while the reconstruction's opposing pressure grows as the decoder sharpens. There are **two**
gradient paths pushing $\ell^p$ down, and each needs its own key:

| Key | Path it removes | Shipped | Arm |
| --- | --- | --- | --- |
| `base_decode: mean` | $D_0$ decodes a *sample* from the prior, so sampling noise can only degrade the forecast and its gradient on $\ell^p$ points down without limit | `mean` | `sweep_base_sample.yaml` |
| `posterior_logvar_mode: independent` | $\ell^q = \mathrm{sb}(\tilde\ell^p + \Delta\ell)$ routes $D_1$'s pressure to sharpen $\sigma^q$ onto the **prior's** tensor | `independent` | `sweep_logvar_residual.yaml` |

The second is the one that **survives** the first, which is why turning off the base branch's
noise alone would not have been enough. Measured on the tiny model: the full branch's
reconstruction alone puts a gradient of $8.7 \times 10^{-2}$ on `prior_head.logvar_prior_head`
under `residual` and **exactly $0$** under `independent`. With both shipped, $\ell^p$ receives
gradient from $\mathrm{KL}(q \Vert p)$ — which pulls it *up*, to cover $q$ — and from the scale
anchor, and from nothing that wants it narrow.

This does **not** make the model an autoencoder. The posterior branch, which is the one the
coupling is read from, is still sampled; what is removed is the pressure making the *prior*
deterministic.

**Measurement two — the source pathway memorises at constant rate.** Between epochs $55$ and
$120$ the KL is unchanged ($1.444 \to 1.442$ nats) and the displacement is unchanged
(`delta_mu_rms` $0.084 \to 0.083$), while the held-out gain falls from $+3.25$ to $-0.70$. Same
rate, same magnitude, degraded *content* — so a rate lever ($\beta$) cannot reach it.
`source_dropout` regularises $\Delta\mu(h^y, a)$ alone: the source adapter, the source encoder and
the attended summary inside the posterior fusion, and nothing on the target side. It ships `null`
(= the global `dropout`) deliberately — no measurement supports a particular rate, and an
unvalidated one in the headline run would confound it; `sweep_source_dropout_0p2/0p3` measure it.

**And the guard is now on.** `causal_reach_budget_s: 120`. Every number quoted above was measured
at `null`, where the two-sided features let the target branch read up to $974$ s of its own future,
so $D_0$ was a target-only forecast *plus lookahead*. Expect $D_0$ to get **worse** under the
guard: that is the leak being removed, not the guard failing.

**What the bundle costs.** Base and full are no longer bitwise identical at initialisation — the
full branch still carries the posterior's noise. The KL is still exactly $0$ there, because it is
a function of the two distributions and not of the samples: `head_init_calibration` pins the
prior's raw log-variance to $\log(5/3)$ and the independent head is seeded at the same constant,
so both start at log-variance $0$. `tests/test_bottleneck_keys.py` pins every claim in this
section, including the two gradient measurements.

## 11. Deviation record

Every intentional difference between the built module and the design it was built from.

**Architecture**

- **A final `RMSNorm(128)` per encoder**, which the proposal's parameter arithmetic
  ($2 \times 50{,}000 + 4 \times 164{,}000$, no final norm) does not carry. $128$ parameters each.
  The reason is the downstream calibration contract in §2.4, not a preference.
- **The availability projection reads $m_t - \mathbf 1$, not $m_t$** (§2.1). The same model; written
  so the term is exactly zero wherever nothing is missing.
- **The encoder attention ignores the loader's `weight`.** The proposal's masks say "and key $j$ is
  valid"; this build does not implement that half. Masking data gaps out of encoder self-attention
  would make this model condition on strictly *less* than the model it is compared against does, and
  a forecast difference would no longer be attributable to the encoder. It is also what makes the
  fully-masked-row case unreachable, so no NaN path exists to test.
- **The availability parameters are constructed conditionally, and the justification is not the one
  the DDP framing suggests.** A zero-multiplied parameter is reachable; the conditions are parameter
  economy and honesty, and they differ between the two terms (§8). The *forward* is unconditional,
  which is the half that matters for DDP.
- **The model is a standalone `nn.Module`, not a subclass** of the comparison model. Its
  $384$-line `__init__` constructs the encoders being replaced and validates a keyword schema this
  architecture does not have. What must never diverge is the **objective**, so `compute_loss` and
  `kld_tensor` were extracted into `teb_vae/lag_attn_rws/nets/losses.py` — which already owns the
  reduction and its units — and both models call the same functions. `LOGVAR_FLOOR_MARGIN_FRAC`
  moved with them and is re-exported from `nets/model.py`, so every existing consumer's import
  still resolves.
- **Five constructor keys of the comparison model are absent**, not defaulted: `lstm_layers`,
  `encoder_extra_dilations`, `encoder_extra_kernel`, `conv_norm_groups` and `causal_norm`. There is
  no recurrent branch, no extra dilation schedule and no time-pooling normaliser left to causalise.
  Passing any of them raises `TypeError`, so a copy-pasted config is caught rather than silently
  dropped.

**Configuration and driver**

- **`configs/default.yaml` is written out in full**, not inherited from the comparison model's. The
  two `VAE_model` blocks do not share a schema and a merge would leave the dead half silently
  dropped by the signature sweep. The cost of that choice is drift, so `tests/test_config_load.py`
  walks every leaf of both files and requires equality outside the encoder schema, against a
  six-entry exemption table that is itself asserted to contain only *real* divergences. Four of the
  six are identity — the output tree, the MLflow experiment, the run name and the variant tag — and
  copying any of them would write these runs into the other model's tree.
- **`lr_warmup_steps` lives in `general_config`**, beside `lr` and `lr_milestone`, because it is the
  third term of the same schedule rather than a property of the network. In `VAE_model` it would be
  a key the signature sweep silently drops. The driver's `create_model` forwards it through the
  framework's own `apply_config_hyperparameters`, so it lands in `self.hparams` and therefore in
  every checkpoint.
- **The learning-rate monitor is *replaced*, not added.** The framework attaches
  `LearningRateMonitor(logging_interval='epoch')` unconditionally; a second one would leave two
  callbacks logging the same key at two resolutions. The driver swaps the instance so exactly one
  monitor logs, at the resolution a step-granular warm-up operates on.
- **One `LambdaLR` rather than `SequentialLR`.** `LinearLR` rejects `start_factor=0.0` and so cannot
  express a ramp from zero at all, and `SequentialLR` restarts the second scheduler's counter at the
  switch, requiring a compensating milestone shift a single lambda does not need. At
  `lr_warmup_steps: 0` the task delegates to the framework, keeping the epoch-granularity path
  reachable from configuration at no cost.
- **The driver re-admits one `null`.** The inherited config sweep drops every `VAE_model` key set to
  `null`, reading it as "use the constructor's own default" — right for a key whose null means
  *unset*, and wrong for the one key here whose null is a value. An unbounded source encoder **is**
  `source_attention_window: null`; dropped, the sweep would rebuild the shipped $16$-step window
  while the arm still reported under the unbounded arm's name. `NULLABLE_MODEL_KEYS` in `trainer.py`
  names the exception, and `tests/test_trainer.py` pins both directions.
- **`tiny.yaml` shrinks one encoder key and no others.** `encoder_d_ff` drops to $64$; the stem
  kernels and dilations, both block counts and the source window are the shipped values at the
  shipped $300$ steps — so the smoke fit exercises the production encoder *shape*, including the
  bounded source reach, at miniature width.
- **`gradient_clip_val: 5000.0` was measured, and is provisional again at this capacity.** The reason for
  carrying the sibling's value was sound — the objective, its reduction and its units are
  identical, so the gradient scale is dominated by the summed $480$-sample Gaussian NLL rather
  than by the encoder — but the scale it was carried at was not. The $1018$-epoch baseline run
  logged `train/grad_norm` pre-clip: $q_{50} = 2775$, $q_{99} = 4681$, $q_{99.9} = 5866$,
  maximum $7313$, minimum $703$. **Every** recorded step exceeded $250$, so that run performed
  normalised-gradient descent at roughly a eleventh of its configured learning rate. $5000$ is
  the smallest round value above $q_{99}$. The CSV carries one optimizer step per epoch for this
  metric rather than the epoch's aggregate, so those are per-step percentiles over a thinned
  sample — the right distribution for a per-step threshold; `train/grad_clip_frac` samples the
  exceedance indicator the same way and is read as a mean over epochs.

  **That run predates this capacity**, so the value ships marked `PROVISIONAL AT THIS CAPACITY` in
  `configs/default.yaml`: the revision widened the latent, widened and deepened the decoder, added
  its horizon attention, doubled the encoder feed-forward width, added two target blocks and put
  three shape terms on the criterion — each of which moves the gradient this threshold is set
  against. It is carried across rather than rescaled by a guess, and `RESULTS.md`'s five-step
  procedure re-derives it from the first run at this geometry. The spike breaker's
  `additive_margin` ($10^{3}$) carries the same marker for the same reason, with the added wrinkle
  that its watched `main_loss` is now mixed-unit.

**Where a stated rationale did not survive measurement**

- **The obvious prefix-equivalence negative control is not a control.** An end-relative rotary
  encoding, positions running $T-1-t$, is still prefix-equivalent — and not marginally, *bitwise*: a
  rotary score depends only on $\operatorname{pos}(t) - \operatorname{pos}(j) = j - t$, so the
  length-dependent part is a uniform shift that cancels in every difference and only the sign of the
  displacement flips. Measured on the built encoder it moves the property by exactly $0.0$. The
  working control is a **length-normalised** rotary encoding — positions stretched to fill the
  table, so two steps a fixed distance apart sit at different displacements depending on sequence
  length — which moves it by $4 \times 10^{-4}$ against a $10^{-5}$ tolerance. Both are in
  `tests/test_prefix_equivalence.py`, the second as a recorded finding so the obvious control is not
  re-added and believed.
- **The lag-attribution tolerance is *relative*.** On a perturbed tiny model $K_t$ reaches $10^2$
  nats, so an absolute $10^{-6}$ would be a statement about the perturbation's scale rather than
  about the decomposition, and float32 cannot meet it. The measured worst-case relative residual is
  $2 \times 10^{-7}$ — float32 eps — so the file asserts `rtol=1e-6` with a $10^{-6}$ absolute floor
  for near-zero anchors.
- **Zeroing the FiLM generators needs the generic pass *on* to have a negative control.** The
  posterior delta heads become non-zero under `init_weights=False` with the re-zeroing disabled; the
  FiLM generators do not, because the horizon core zero-initialises them itself and the refill *is*
  the generic pass. `tests/test_init_policies.py` runs the control both ways and asserts the FiLM
  half only under `init_weights=True`.
- **The whole-model ban on time-pooling normalisers is scoped to the history path.** The horizon
  core carries one `GroupNorm` per refine block — four at the shipped `horizon_depth: 4` — plus one
  `LayerNorm` per horizon self-attention block, and deliberately: all of them pool over the
  *forecast* axis of a single anchor, not across input time.
  `tests/test_construct.py` bans the family on both gates, both adapters and
  both encoders, bans recurrence everywhere, then enumerates the surviving `GroupNorm`s and asserts
  each is under `horizon_core.` — so a new one anywhere else fails there rather than slipping past a
  scoped check.

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

# An architecture arm: the same launch line against any configs/sweep_*.yaml.
```

The gate:

```
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_rws/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_rws/tests -q -m slow
```

`RESULTS.md` states what every arm must record and the procedure that re-derives the clipping
threshold. The arms themselves are `configs/sweep_*.yaml`, each linted by
`tests/test_sweep_configs.py` against a declared-delta table, so an arm carrying a second change
fails on the development box rather than confounding its own result days later.

## 13. Configuration keys

`tests/test_docs.py` drives this section against `configs/default.yaml` in both directions, so it
cannot drift: every key in the first list below must exist, every key in the second must not, and
every `model_config.VAE_model` key the shipped config carries must appear in the first. Outside
`VAE_model` the first list is the set this document's claims depend on rather than an exhaustive
inventory of the framework's own settings.

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

The five that describe the encoder being replaced, and would each be dropped by the signature sweep
without a word:

- `model_config.VAE_model.lstm_layers`
- `model_config.VAE_model.encoder_extra_dilations`
- `model_config.VAE_model.encoder_extra_kernel`
- `model_config.VAE_model.conv_norm_groups`
- `model_config.VAE_model.causal_norm`

And the ones this architecture made structural, derived, or inert — each would read to a maintainer
as a control that exists:

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

The horizon attention's head count is a constructor argument on the shared core, defaulting to $4$,
and stays one: no arm varies it, and a key for a value nothing sweeps is a knob a maintainer would
budget attention for. `horizon_attention_blocks` is the axis that moves, and it has a key. The
encoder head width is derived as `d_model // encoder_num_heads`; encoder dropout is the existing
`dropout`, documented as applying inside the encoders; the LayerScale initialisation and the rotary
base are constructor defaults that no arm varies; the target attention window is full context in
every arm by design; an availability switch would be a flag for a mechanism already off wherever it
would be inert; and encoder gradient checkpointing sits behind two cheaper memory levers that are
already config keys (`batch_size` with `accumulate_grad_batches`).
