# `lag_attn_transformer_e2e` — design record

This record describes the end-to-end causal raw-signal lag-attention VAE-TEB. It states what the
model is, what it replaces, what it leaves unchanged, and the design decisions in the built module.

Companion documents: `teb_vae/lag_attn_transformer_rws/DESIGN.md` records the comparison model.
This package shares that model's encoders, objective, geometry, data contract and metric surface
without change. `teb_vae/lag_attn_rws/DESIGN.md` records the package that owns the objective, the
masks and the trimmed-grid geometry. This record does not restate them. It covers the two encoder
**inputs** and the wiring around them, because only those are different.

---

## 1. What the model is

The model is `teb_vae/lag_attn_transformer_rws` with **both encoder inputs replaced** and nothing
else changed. At every 4-second anchor $t$ the model forecasts the next two minutes of raw
normalized FHR, $H \cdot R = 30 \times 16 = 480$ samples. It makes this forecast twice: once from a
target-only latent and once from a source-conditioned latent. One shared decoder makes both
forecasts under one noise draw. The KL between the two latents, resolved across lags, is the
coupling readout.

The change is in what the two encoders read. In the comparison model, each stream arrives as a
block of **stored wavelet and phase-harmonic coefficients**. An availability-aware input adapter
projects the block onto the backbone width:

$$
[\,\texttt{fhr\_st} \mid \texttt{fhr\_ph}\,] \in \mathbb R^{B \times 300 \times 109}
\;\xrightarrow{\ \operatorname{AvailabilityInputAdapter}\ }\;
\mathbb R^{B \times 300 \times 128} \longrightarrow \text{encoder} .
$$

In this model, each stream arrives as the **raw 4 Hz signal**. A learned, strictly one-sided front
end maps it onto the token grid:

$$
\texttt{fhr} \in \mathbb R^{B \times 4800},\ \texttt{weight} \in \mathbb R^{B \times 300}
\;\xrightarrow{\ \operatorname{CausalRawFrontend}\ }\;
\mathbb R^{B \times 300 \times 128} \longrightarrow \text{encoder} .
$$

The reason is a property of the stored features. A scattering or phase-harmonic coefficient at
decimated step $t$ is a weighted average over raw samples on **both** sides of $t$. A model that
conditions on "the past up to $t$" therefore conditions on part of the interval it must forecast.
On the source side this leak enters the posterior alone, so it changes $D_0 - D_1$, $K$ and the lag
map. Those are the three readouts the model exists to produce. §9 states what the raw input gives
and what it does not give.

Everything downstream of the two front-end outputs is imported without change:

- both `CausalConvTransformerEncoder`s from `teb_vae.lag_attn_transformer_rws.nets`;
- the target-only conditional prior head, the masks, the raw-target extraction, the trimmed-grid
  geometry and the seven-term objective from `teb_vae.lag_attn_rws.nets`;
- the lag-restricted cross-attention over $L = 91$ lags, the head-structured bounded posterior
  residual, the shared horizon decoder and the lag-resolved KL attribution from
  `teb_vae.lag_attn.nets`.

A change to any of them belongs in those packages, where every model gets it.

**The objective is inherited, and it has seven terms.** `compute_loss` in
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
reduced on the KL's own anchor support and in nats per anchor. Nothing else in the objective
penalises a *narrow* prior, so this term exists for that purpose. It ships at the comparison
model's value, `beta_prior: 0.1`, so that the two runs differ only in their inputs.

The last three terms are auxiliary shape terms on the forecast means: pooled L1 for the envelope, a
derivative Huber for the slope, and a boundary gap for the starting level. They ship at the
comparison model's $0.1 / 0.1 / 0.05$ for the same reason. They make `total_loss` a **mixed-unit**
criterion, because the shape terms are L1 and Huber quantities on z-scored raw samples, not nats.
`nll_full_block` and `nll_base_block` are therefore the pure-nats readouts. A term at weight $0$ is
not computed and reports exactly $0.0$.

`tests/test_losses_delegation.py` feeds one real forward dict to **both** models' `compute_loss`
and compares every metric tensor with `torch.equal`. Before that, it asserts that the two
geometries, index grids, coverage floors and log-variance bounds are equal.

## 2. The causal raw front end

`nets/frontend.py` is the only module in this package that is written rather than imported. No
module under `teb_vae` does causal anti-aliased decimation of a raw signal. The front end uses no
dependency beyond `torch` and the standard library; the filter coefficients are a binomial
expression.

Per stream:

$$
\text{featurise} \rightarrow
\underbrace{\text{stage}_1 \rightarrow \text{stage}_2 \rightarrow \text{stage}_3 \rightarrow
            \text{stage}_4}_{\text{stride } 2 \text{ each, widths } 32 \rightarrow 64 \rightarrow 96 \rightarrow 128}
\rightarrow \operatorname{RMSNorm}(d_{\mathrm{model}}) \rightarrow (B, T, d_{\mathrm{model}}) .
$$

The two streams use **two independently parameterised instances at identical settings**, never one
shared module. A shared module would make the source state a function of the target, and the KL
readout depends on source purity.

### 2.1 Featurisation

$$
\left[\; \bar x_n\, m_n, \quad m_n, \quad (\bar x_n - \bar x_{n-1})\, m_n m_{n-1} \;\right],
\qquad
m_n = \mathbb 1\!\left[\,\mathrm{weight}_{\lfloor n/r \rfloor} \ge \theta\,\right]
      \wedge \operatorname{isfinite}(x_n),
$$

Here $\bar x$ is the signal **as the loader produced it**. It is already z-scored, so the front end
owns no statistics. The threshold $\theta$ is `VALID_THRESHOLD`, imported from
`teb_vae/lag_attn_rws/nets/raw_masks.py`. The loss masks use the same constant, so the front-end
mask cannot drift from the mask the objective uses.

`tests/test_frontend_featurize.py` tests three properties:

* **The validity mask is a channel.** A gap is therefore different from a genuine normalised zero:
  the value channel is the same, but the mask channel is different.
* **Invalid positions are set to zero in standardised space**, not in raw space. A zero raw value
  before the z-score would map a gap to $-\mu/\sigma$, a large negative constant that the low-pass
  then spreads across the following tokens. The operation is `torch.where`, not a multiply, because
  a non-finite sample times a zero mask is still non-finite.
* **The first difference is gated on both endpoints.** The first valid sample after a gap therefore
  adds no false slope. The replicate pad makes $\Delta x_0$ exactly zero for the same reason.

`featurize` derives the ratio $r$ from the two input lengths and checks only divisibility. The front
end itself checks that $r$ equals its own total stride, at construction and again on each call.

### 2.2 One stage

$$
P = W x + b, \qquad G = \operatorname{GatedCausalConvBlock}_k(P), \qquad
\mathrm{out} = \operatorname{Decimate}_2(G) .
$$

The convolution block is `GatedCausalConvBlock` from `teb_vae/lag_attn_transformer_rws/nets/blocks.py`,
imported and used whole in its $(B, T, C)$ layout. The block adds its residual at the **full** rate,
and one decimator then runs on the sum. There is no separate skip path that a different operator
could decimate out of sample alignment. Every front-end depthwise convolution is also a
`CausalDepthwiseConv1d`, the class that `init_depthwise_` detects (§7).

The widening is in the pointwise projection before the block, because the block keeps its width
constant. That projection carries a **bias**, the only bias in the front end. A fully invalid
window featurises to an exactly zero vector. An exactly zero token that enters repeated
pre-normalisation produces derivatives of order $1/\sqrt{\epsilon}$. The bias makes "this window is
empty" a learnable constant instead. It costs $320$ parameters per front end at the production
width ($32 + 64 + 96 + 128$).

### 2.3 The decimator

`CausalAntiAliasDecimate` is a fixed causal FIR low-pass followed by right-offset subsampling:

$$
\tilde x[n] = \sum_{i=0}^{\tau-1} h_i\, x[n-i], \qquad \mathrm{out}[t] = \tilde x[s t + s - 1],
\qquad h_i = \binom{\tau-1}{i} \Big/ 2^{\tau-1} .
$$

Three details are structural:

* **The offset is right**, not left or centre. Four stride-2 stages give a total stride of $16$,
  and the newest input sample of token $t$ is raw index $16t + 15$. This is exactly
  `TrimmedRawGeometry.n_raw(t) = 16(t+1) - 1`, so the front end and the model use the same anchor
  convention. A centred offset would still pass a causality probe but would discard the newest
  quarter-second of every token. A left offset would read the future. `tests/test_causality.py`
  perturbs raw $16t+16$ and requires the state at anchor $t$ to stay bitwise identical. It then
  perturbs $16t+15$ and requires the state to move.
* **The coefficients are a non-persistent buffer applied with `F.conv1d`**, never an `nn.Conv1d`.
  `teb_vae/lag_attn/nets/blocks.py::initialization` Xavier-fills every `nn.Conv1d` weight in the
  model, so a fixed kernel held as a layer would be replaced by random values with no error.
  `tests/test_frontend_decimate.py` runs a full `initialization()` pass over a module that holds one
  decimator and requires the buffer to stay bitwise identical. As a negative control, the test also
  holds an `nn.Conv1d` with the same coefficients, which the same pass must change.
* **There is no `antialias=False` switch.** A decimation without the filter is a different model.

The filter has $\tau = 5$ taps (`ANTI_ALIAS_TAPS`). A binomial kernel has $H(\pi) = 0$ **exactly**
for every $\tau \ge 2$. The old Nyquist frequency, which a factor-2 decimation folds onto DC, is
therefore removed completely. For this reason the alias test asserts against $10^{-12}$ in float64.
More taps give more attenuation at the fold point: $|H(\pi/2)| = (\cos \pi/4)^{\tau-1}$ is $0.5$ at
three taps and $0.25$ at five. Aliased $0.5$–$2$ Hz energy would look like fetal heart-rate
variability, so the front end uses five taps and pays the extra reach.

### 2.4 The final normalisation

Each front end ends in `RMSNorm(d_model)`. This mirrors the encoder's own final norm. The encoder
stem, the prior head and the lag attention's key-value normalisation all expect a normalised
state, and a pre-norm residual stack without a final norm exports a stream whose scale grows with
depth.

**Normalisation is on the channel axis only.** `nn.GroupNorm` reduces over $(C/G, T)$ within a
group, so it pools across time. The `BatchNorm` family also makes the output of each sample depend
on the rest of the batch. `refuse_time_pooling_norms` refuses both families, `SyncBatchNorm` (which
does not subclass `BatchNorm1d`) and the `InstanceNorm` family. Every front end calls it at the end
of its `__init__`. `tests/test_frontend.py` tests it in both directions. It plants an `nn.GroupNorm`
by monkeypatching the name the stage looks up, because the public constructor offers no other
route.

## 3. Reach

### 3.1 The front end's own reach, and its budget

Reach is a **count** of raw samples in the support, the sample itself included. This matches the
`receptive_field` convention of the sibling's blocks. The front end accumulates the reach from the
*built* modules, not from the constructor arguments:

$$
R \;=\; 2 \;+\; \sum_{i=1}^{4} \left(k_i + \tau - 2\right) \prod_{j<i} s_j .
$$

Each stage adds $(k_i - 1)$ samples from its depthwise kernel and $(\tau - 1)$ from the anti-alias
filter, scaled by the stride below it. The leading $2$ is the one-sample first difference of the
featurisation plus the sample itself.

The front end **refuses at construction** when $R$ exceeds its `reach_budget`, and the message names
both numbers. The model resolves that budget in `_resolve_frontend_budget`:

- With `frontend_reach_budget_s: null`, the budget is $\texttt{warmup\_period} \times
  \texttt{raw\_per\_step}$ raw samples.
- With a value in seconds, the model converts it to raw samples (floored). The value must be
  positive and may not exceed $\texttt{warmup\_period} \times \texttt{raw\_per\_step}$; otherwise the
  model raises `ValueError`.

The warm-up is the hard ceiling because only an anchor inside the warm-up may see the zero-padded
convolution transient at the start of the segment. The shipped config sets
`frontend_reach_budget_s: 120.0`, which is $480$ raw samples, exactly the warm-up ceiling.
`tests/test_construct.py` asserts that the model passes $\texttt{warmup\_period} \times
\texttt{raw\_per\_step}$ when no budget is configured. `tests/test_trainer.py` asserts that the
shipped config resolves to the same value.

`python -m teb_vae.lag_attn_transformer_e2e.nets.frontend` prints:

```
production (warmup_period 30): d_model=128, total stride 16
  stage  width  kernel  stride               reach    params
      1     32      65       2       70 raw /   17.5 s     5,376
      2     64      15       4      106 raw /   26.5 s    15,552
      3     96      15       8      178 raw /   44.5 s    35,616
      4    128      15      16      322 raw /   80.5 s    63,872
   norm    128       -       -                   -       128
  reach 322 raw samples (80.5 s) against a budget of 480 (120.0 s)
  total 120,544 parameters per stream, 241,088 for both
```

The kernels $(65, 15, 15, 15)$ are the constructor default `FRONTEND_KERNELS`. They are not a config
key, which follows the precedent of `ROPE_BASE` in the sibling's blocks; the reach guard bounds any
future choice. The widths are derived from $d_{\mathrm{model}}$ as $(d/4,\, d/2,\, 3d/4,\, d)$, so
there is no width configuration.

The probe tests the *safety* of the bound, not its tightness. A perturbation of raw at $n - R$
leaves the token bitwise identical, and a perturbation at $n$ moves it. A test that required
$n - R + 1$ to move would pin the bound as tight, which nothing requires.

At the smoke geometry, `tests/conftest.py` uses kernels $(5, 3, 3, 3)$ against a budget of
$6 \times 16 = 96$, and the reach is $94$. `configs/tiny.yaml` does **not** use those kernels. It
keeps `warmup_period: 30` and the production kernels, so the smoke fit runs the front ends at their
production reach of $322/480$ (§11).

### 3.2 The composed raw receptive field

The source encoder's token reach is
$R_U = R_{\mathrm{conv}} + N_U(W_U - 1) = 21 + 3 \cdot 15 = 66$ steps. The source state at anchor
$t$ therefore reads tokens $t - 65 \ldots t$, and each of those tokens reads $322$ raw samples that
end at its own anchor. Composed:

$$
R^{\mathrm{raw}}_U \;=\; R_{\mathrm{frontend}} + r\,(R_U - 1) \;=\; 322 + 16 \cdot 65
\;=\; \mathbf{1362} \text{ raw samples} \;=\; 340.5 \text{ s},
$$

ending at raw index $16t + 15$. The formula is $R + r(R_U - 1)$, not $R + rR_U$, because both
reaches are counts and the two supports overlap on the $16$ samples of the anchor token.
`tests/test_frontend_reach.py` computes this value from a model at the shipped geometry and requires
it to stay inside the lag search range.

The lag search range $\ell \in \{0, \ldots, 90\}$ spans $90$ steps, which is $360$ s. The source
encoder's reach must stay shorter than that range. An encoder whose reach exceeded the lag range
would already do the alignment that the lag attention exists to do, and the reported lag would no
longer show where the coupling came from. The margin is $19.5$ s. Each extra step of
`source_attention_window` adds $N_U = 3$ steps to $R_U$, which is $12$ s of raw reach. A window of
$17$ reaches $352.5$ s and fits; a window of $18$ reaches $364.5$ s and does not. Any change that
widens the source attention window, the encoder stem or the front-end kernels must be checked
against this margin.

The target encoder's reach is reported as **unbounded**, not as $T$, because the target attention
has no window. Its composed raw reach is the whole trimmed prefix, ending at $16t + 15$.

## 4. Forward return dict

`nets/model.py::SeqVaeLagAttnTrfE2E.forward(y_raw, u_raw, weight)` takes the two raw signals at
shape $(B, T \cdot R)$ and the decimated validity signal at shape $(B, T)$. It returns the twenty
keys that the comparison model returns, at the same shapes: `mu_prior`, `logvar_prior`,
`raw_logvar_prior`, `mu_post`, `logvar_post`, `z_prior`, `z_post`, `target_state`, `source_state`,
`attended_source_heads`, `attn_weights`, `mu_base`, `logvar_base`, `mu_full`, `logvar_full`,
`kld_per_t`, `kld_per_t_per_head`, `source_kl_lag_map`, `mu_prior_sat_frac`, `delta_mu_sat_frac`.
There is no `decoder_state` and no `delta_mu_src`, because neither pathway exists.

`tests/test_forward_contract.py` compares the key set and every shape against a constructed
`SeqVaeLagAttnTrfRws` on the same stub batch.

The arguments are named `y_raw` / `u_raw`, not `fhr_raw` / `up_raw`, because a `nets/` module takes
tensors and does not know the on-disk field names. §6 records how a swap of the two arguments is
caught.

## 5. Parameter budget

The front-end figures below come from the built module (§3.1):

| Component | Parameters |
| --- | ---: |
| Front-end stage 1, $32$ wide, $k = 65$ | $5{,}376$ |
| Front-end stage 2, $64$ wide, $k = 15$ | $15{,}552$ |
| Front-end stage 3, $96$ wide, $k = 15$ | $35{,}616$ |
| Front-end stage 4, $128$ wide, $k = 15$ | $63{,}872$ |
| Final `RMSNorm(128)` | $128$ |
| One front end | $120{,}544$ |
| Both front ends | $241{,}088$ |

`tests/test_frontend.py` checks the per-stage figures against their arithmetic. No test pins the
whole-model total, so a shared change to an imported downstream component does not fail a test in
this package.

The eight anti-alias filters, one per stage per stream, contribute **zero** parameters.
`tests/test_frontend_decimate.py` asserts this, and `tests/test_construct.py` counts them.

## 6. Structural constraints

Each constraint is enforced by construction or measured by a test.

| Constraint | What enforces it | Test |
| --- | --- | --- |
| **Raw-signal causality**, $H_t = f(x_{\le 16t+15})$, per front end and per assembled state | right-offset decimation; left-only convolution padding; every other front-end primitive is position-wise on the channel axis | `tests/test_frontend_causality.py`, `tests/test_causality.py` |
| **The warm-up covers the front end's reach** | the front end refuses at construction against its resolved budget, which may not exceed `warmup_period * raw_per_step` | `tests/test_frontend_reach.py`, `tests/test_construct.py` |
| **No time-pooling or batch-coupling normaliser** in the front ends | `refuse_time_pooling_norms` at the end of every `__init__` | `tests/test_frontend.py` |
| **No time-pooling normaliser elsewhere, and no recurrence** | none is constructed; the remaining `GroupNorm`s sit under `horizon_core.`, where they pool the *forecast* axis of one anchor | `tests/test_causality.py`, `tests/test_construct.py` |
| **Source purity**: the prior never sees the source, the source state never sees the target | separate front ends and encoders; the posterior is a residual on the prior | `tests/test_source_purity.py` |
| **The two raw arguments are not interchangeable** | forward pre-hooks in the test assert object identity at the two front ends, because a swap of two same-shaped raw tensors is otherwise invisible | `tests/test_source_purity.py` |
| **No decoder bypass**: gradient reaches the decoder only through $z$ | `BaselineFutureDecoder.forward` takes exactly one tensor, at $d_z$ in-features | by construction |
| **Exact zero KL at initialisation**, and bitwise identical base and full *in train mode* | posterior deltas zeroed **after** the generic init; one shared $\epsilon$; decoder and lag-attention dropout fixed at $0$ | `tests/test_zero_kl_init.py` |
| **The lag attribution identity**, $\sum_\ell \widetilde K_{t,\ell} = K_t$, exactly | the lag attention is built at `dropout=0.0`, so the returned probabilities are the ones the posterior consumed | `tests/test_lag_map.py` |
| **`lag_attn.W_o` frozen** | the head-structured posterior consumes the per-head summaries, so `W_o` gets no gradient; `requires_grad_(False)` drops it from DDP's expectation set | `tests/test_ddp_reachability.py` |
| **The fixed FIRs survive initialisation** | non-persistent buffers applied with `F.conv1d`, never `nn.Conv1d` weights | `tests/test_frontend_decimate.py` |
| **`nets/` reaches no framework layer** | import-graph walk over every `nets/*.py` | `tests/test_nets_are_framework_free.py` |

Two conventions apply to the whole test suite.

**Every positive invariant is paired with a negative control that shows the probe is not vacuous.**
The KL is exactly zero at initialisation, so any KL assertion on a fresh model passes on a broken
one. This architecture adds three more cases: a dead stage passes every per-token invariant, an
all-ones mask channel passes every gap test, and a replaced FIR passes every shape test. The planted
defect for causality is `_SymmetricallyPaddedDecimate`, a subclass of the real decimator with one
method replaced. It has the same shapes, the same reach and the same parameter count, so only a
causality probe can find it. A centred *offset* would not be a valid control: it makes the token
depend on raw $\le 16t$, which is strictly more conservative, so the planted model would pass the
bitwise half.

**The causality control is positional and needs no test hook in production code.** A perturbation
strictly after the cut must leave the output at the cut bit-stable *and* must move the output at the
end.

**The movement half of a boundary probe uses a local tolerance.** Each stage's filter puts the
newest sample on its leading tap $h_0 = 1/16$. After four stages that sample carries only
$16^{-4} \approx 1.5 \times 10^{-5}$ of the token, and its effect through the encoder depends on how
much history the anchor has. The boundary tests in `tests/test_causality.py` therefore assert
movement against a local $10^{-9}$ (`_ANCHOR_MOVEMENT_TOL`), not the shared `MOVEMENT_TOL` of
$10^{-3}$. They pair this with a perturbation of the anchor's **whole step**, which must clear the
shared tolerance. A low-pass may weight the edge of its window lightly; it may not weight anything
past that edge, which is the bitwise half.

## 7. Initialisation order

The order in `nets/model.py::__init__` is load-bearing, top to bottom:

1. `initialization(self)` is the shared generic pass. It applies Xavier to every `nn.Linear` and
   `nn.Conv1d`. The anti-alias filters are buffers, not layers, so the pass does not touch them
   (§2.3).
2. `init_depthwise_(self)` runs **immediately after**, never before. Xavier on a $(C, 1, k)$
   depthwise weight reads $\mathrm{fan\_in} = k$ against $\mathrm{fan\_out} = Ck$. This gives
   $\sigma = \sqrt{2/(k + Ck)}$ against the variance-preserving $1/\sqrt k$, a factor
   $\sqrt{(1+C)/2}$ too small, **independent of $k$**.
3. `_restore_frontend_stage_bias()` also depends on the generic pass. `initialization` sets every
   `nn.Linear` bias to zero, which includes the four stage projections per front end. Those are the
   *only* biases in either front end (§2.2). With them at zero, a fully invalid window maps to an
   exactly zero token. The method restores torch's own `nn.Linear` default,
   $\mathcal U(-1/\sqrt{C_{\mathrm{in}}}, 1/\sqrt{C_{\mathrm{in}}})$.
4. `_zero_init_delta_heads()` sets the posterior delta heads to zero. Without it the generic pass
   would refill them and remove the exact zero-KL start.
5. `_zero_init_film_generators()` sets the per-block FiLM generators to zero. The horizon core
   zero-initialises them, the generic pass refills them, and this step makes the identity at
   initialisation true again.
6. The three zero-parameter init policies (`horizon_embed_std`, `head_init_calibration`,
   `a_head_gain`) run last. Each runs only when its config value differs from the constructor
   default, so a default-flag model is bitwise the model without them.

Steps 2 and 3 run only when `init_weights` is true, because they repair the generic pass. Torch's
own defaults already read the depthwise fan correctly and give a non-zero bias.

`n_depthwise_init` records how many depthwise convolutions step 2 re-initialised. It counts the
encoder stem convolutions plus $2 \times \texttt{NUM\_STAGES} = 8$ front-end convolutions, one per
stage per stream. `tests/test_init_policies.py` asserts this count. It also asserts that every
depthwise bank sits at the variance-preserving $\sigma = 1/\sqrt k$ within a $20\%$ band
(`_STD_BAND`). A correction that ran before the generic pass would not pass this check.

**At initialisation each front end is approximately a linear mix of the decimated
`[value, mask, delta]` channels.** `LayerScale` starts the residual branch of every convolution
block at $10^{-2}$ (`LAYER_SCALE_INIT`), so each stage is close to its pointwise projection composed
with the fixed low-pass.

## 8. DDP reachability

Production runs under DDP with `find_unused_parameters=False` under `gaussian_nll`, so every
parameter must be reachable in every backward.

A parameter multiplied by an identically zero tensor **is** reachable. Its `AccumulateGrad` node
fires, it gets a zeros gradient rather than `None`, and DDP's reducer marks it ready. What breaks
`find_unused_parameters=False` is a parameter left *out of the graph*: for example, a Python-level
`if mask.any(): ...` that skips an operation on some ranks or some batches.

The rule is therefore: **front-end masking is multiplicative and unconditional.** No `forward` in
this package branches on a tensor value.

- `tests/test_ddp_strategy.py` tests the backward half. After one training step, every
  `requires_grad` parameter has a gradient. A separate test requires every front-end parameter to
  get a gradient on a **fully masked** batch, where the featurisation emits an exactly zero vector
  and only the stage-projection biases (§2.2) keep the tokens off zero.
- `tests/test_ddp_reachability.py` tests the static half. An AST walk over `nets/frontend.py` and
  `nets/model.py`, with machinery imported from the sibling's test file, finds no `if` or
  conditional expression inside a `forward` whose test reads a tensor value. The walk admits an
  expression built only from constants, names, attributes, `.shape` subscripts, comparisons and
  boolean operators. It rejects any call, element subscript or arithmetic. Its known gap is that it
  admits `self.some_tensor > 0`, which would raise on the first forward rather than diverge
  silently.
- `tests/test_frontend.py` asserts that both front ends of the model emit a finite, non-zero token
  on a fully invalid window. Only a stage-projection bias that survived initialisation allows this.

The walk rejects calls, so the front end's shape guards use `x.ndim != 3` and
`x.shape[1] != self.channels`, not `x.dim()` and `int(x.shape[1])`.

The zero-bias failure has a specific mechanism. An exactly zero token that enters repeated
pre-normalisation gives derivatives of order $1/\sqrt{\epsilon}$. Under
`gradient_clip_val` applied as a global norm, one such batch rescales the gradient of every
parameter towards zero, while the loss moves too little for the spike breaker to fire.

**The DDP strategy settings** come from `LagAttnRwsTrainer.ddp_kwargs`, inherited without change.
`select_ddp_strategy` returns a configured `DDPStrategy`, not the `'ddp'` /
`'ddp_find_unused_parameters_true'` shorthand, because those strings can express only
`find_unused_parameters`:

| setting | value | reason |
| --- | --- | --- |
| `find_unused_parameters` | `False` under `gaussian_nll` | everything above; `True` under `mse`, which starves the decoder log-variance heads |
| `broadcast_buffers` | `False` | otherwise DDP re-broadcasts every buffer from rank $0$ on **each forward** |
| `gradient_as_bucket_view` | `True` | `param.grad` points at the reduction bucket instead of a separate allocation |

`broadcast_buffers=False` is safe for this model. Its buffers are the anti-alias filter banks
(§2.3), the rotary tables, the causal attention masks and the raw-target index grid. Each is a
deterministic function of the config, built identically in the constructor of each rank. A running
statistic is the one kind of buffer that diverges per rank, and **there is no `BatchNorm` in this
model**; `sync_batchnorm: false` states the same fact in the config. A model that gained one would
need the broadcast back.

**`static_graph` is not set.** It promises DDP an identical autograd graph on every iteration. The
loss-spike circuit breaker breaks that promise: on a skipped batch it substitutes a loss summed over
every trainable parameter times zero, which is a structurally different backward. The breaker ships
enabled.

## 9. Raw-signal causality: what this model has, and what it does not claim

**Token causality**, $H_t = f(X_{\le t})$, is a property of the *encoder*. Both this model and the
comparison model have it.

**Raw-signal causality**,
$H_t = f\!\left(Y_{\le n_{\mathrm{raw}}(t)},\, U_{\le n_{\mathrm{raw}}(t)}\right)$ with
$n_{\mathrm{raw}}(t) = 16t + 15$, is a property of the *input representation*. No encoder can supply
it. The comparison model's inputs are two-sided wavelet and phase-harmonic transforms, so a
token-causal model over them still conditions on raw signal after the anchor. This model has
raw-signal causality by construction. `tests/test_causality.py` perturbs the raw input at
$16t + 16$ and requires `target_state`, `source_state` and `mu_prior` at anchor $t$ to be bitwise
identical. It then perturbs at $16t + 15$ and requires movement. The test runs in float64 with a
large amplitude and uses `torch.equal` at the cut.

**What this does not license.** The KL is still named `source_conditioned_kl_raw` /
`source_conditioned_kl_train`, and its decomposition is still `source_kl_lag_map`. It is **not**
called transfer entropy. The model reports a KL between two of *its own* distributions, fitted by
one architecture on one dataset. The raw input makes that quantity a statement about the source's
*past*, which is a precondition for the interpretation, not the interpretation.

The trainer's `causal_standing_message` writes this line to the startup log:

```
causal standing: raw-signal inputs through strictly one-sided front ends -- the history state at
step t is a function of raw samples at index <= 16t + 15 and no further, so the source-conditioned
KL reads no source content recorded after its own anchor. Front-end reach <R> raw samples (<s> s)
against a budget of <B> (<s> s).
```

The base trainer logs this line before it assigns `self.pytorch_model`, so a launch that fails in
the constructor still records what it was about to build. The reach in the line therefore comes
from a *throwaway* front end that `_reference_frontend` builds from the resolved constructor
kwargs. The budget in the line is $\texttt{warmup\_period} \times \texttt{raw\_per\_step}$; it does
not read `frontend_reach_budget_s`. At the shipped config the line reads $322$ against $480$.
`tests/test_trainer.py::test_the_logged_reach_cannot_drift_from_the_front_end_that_was_built` pins
the logged reach against both front ends of the model that `create_model` produces.

## 10. Deliberate limitations

> lean-limit: no eval package, so this model cannot run through the shared evaluation pipeline,
> and the training-time metrics are the only readout; replace with a `ModelBinding` and an `eval/`
> package when the separate evaluation specification lands.

> lean-limit: no architecture arms; replace with `configs/sweep_*.yaml` and a declared-delta lint
> when a baseline run exists to sweep against.

> lean-limit: the front-end width schedule (32/64/96/128) and kernel schedule (65/15/15/15) are
> chosen by analogy, not by a sweep; replace with a swept schedule when the first arms compare at
> least two of them. The reach guard bounds any choice.

> lean-limit: primitives are imported from `teb_vae/lag_attn/nets`, `teb_vae/lag_attn_rws/nets` and
> `teb_vae/lag_attn_transformer_rws/nets` rather than promoted to a common package. Promote them
> when this model outlives its comparison, or when a consumer needs to modify a shared primitive
> rather than only import it.

> lean-limit: `teb_vae/lag_attn_rws/eval/metrics.py::model_inputs` still calls
> `_build_target_streams` and `_build_source_stream` directly and returns
> `(y_st, y_ph, u_stream, fhr_raw, weight)`. It is the one call site in that package **not** routed
> through the `_build_forward_inputs` hook that the task and the plotting callback use; replace
> with the hook when this package gains an evaluation entry point.

The last call site stays un-routed while evaluation is out of scope. Routing it would change the
shape of a tuple that its evaluation callers unpack, in a package this one does not otherwise touch,
and a raw-input task cannot reach it today.

Also deliberate:

- **No warm start.** `core_model_checkpoint` stays `null`. A checkpoint from either sibling carries
  a different `model_class` stamp and holds input-adapter tensors this architecture does not have,
  so `check_model_class` refuses it. The front ends train from scratch.
- **No mixed precision.** `precision: "32-true"`. $\mathrm{pred\_gap} = D_0 - D_1$ is a small
  difference between two large block NLLs. `mu_base` and `mu_full` come from one decoder invoked
  twice, so under reduced precision each picks up its own rounding, and that noise enters the
  difference. A float32 island around the loss cannot remove it, because the rounding happens in the
  decoder.
- **`compile: false`, and the key is live.** There is no LSTM in this architecture.
  `compile_model_requested`, inherited from `LagAttnTrfRwsTrainer`, refuses `compile: true`
  together with `attention_grad_checkpoint` and names both. The data-dependent mask indexing behind
  `kld_active_frac` lives in `compute_loss`, which `compute_loss_and_metrics` reaches through
  **`orig_model`**. Only the forward is compiled, so that indexing cannot enter the graph. The hook
  lives in `LagAttnTrfRwsTrainer` and is not restated here. The key ships off because inductor may
  reassociate float arithmetic, and `pred_gap` is sensitive to rounding. A run that compiles says so
  in its startup log. The adoption procedure is in §12.
- **No inter-stream weight sharing and no per-stream front-end settings** (§2).
- **No gap sentinel.** The validity mask is the loader's `weight`, expanded, plus finiteness. The
  pipeline's sanitisation interpolates non-finite values before it writes, and segment quality
  filtering requires a mean weight of at least $0.90$. The finiteness term stays, because a single
  NaN would otherwise propagate through the low-pass into every following token.
- **The front end owns no z-score.** This is safe only because the pre-flight requires `up` in
  `normalize_fields` (§11).
- **`horizon`, `warmup_period` and the whole geometry are inherited without change.**

## 11. Design decisions

**Architecture**

- **The anti-alias filter has five taps, and the kernel schedule is $(65, 15, 15, 15)$.** Five taps
  give $|H(\pi/2)| = 0.25$ against $0.5$ at three taps (§2.3). Both choices cost reach, and both fit:
  $322$ against $480$ (§3.1).
- **The stage projection carries a bias**, the only exception to the bias-free convention of the
  package (§2.2). It costs $320$ parameters per front end at production.
- **A stage is `Linear` → `GatedCausalConvBlock` → `CausalAntiAliasDecimate`.** The stage reuses the
  sibling's block whole. There is therefore exactly one FIR per stage and the residual is
  sample-aligned by construction, and `init_depthwise_` walks the front ends with no extension.
- **The front end computes its reach from the built modules**, not from the constructor arguments.
- **`raw_per_step` is checked against the front end's own total stride**, at construction and again
  on each forward call.
- **The model is a standalone `nn.Module`, not a mode or subclass** of the comparison model. The
  constructor schemas differ: `c_y`, `c_u`, `use_up_st`, `causal_reach_budget_s` and the four
  channel tuples have no meaning here, and `frontend_kernels` and `frontend_reach_budget_s` are new.
  The two models share their *objective* and everything under it, imported.
- **The constructor has no `**kwargs`**, so an unknown keyword raises `TypeError` that names it.

**Testing**

- **`test_encoder_causality.py` and `test_prefix_equivalence.py` are not mirrored.** Both test
  imported code over $(B, T, d)$ inputs whose behaviour cannot change here, and prefix equivalence
  has no consumer while evaluation is out of scope. The assembled raw-resolution probe in
  `tests/test_causality.py` tests this package's own wiring.
- **`tests/test_source_purity.py` carries an object-identity check.** Both inputs are $(B, 4800)$
  raw signals, so a swapped argument pair gives correctly shaped output and a source-conditioned KL
  of the target against itself. Every other probe in the file misses it, because a swapped model
  keeps both streams *pure*. Forward pre-hooks that assert object identity catch it.
- **The AST machinery in `tests/test_ddp_reachability.py` is imported from the sibling**, and its
  self-tests are not ported, because they would only test the import.
- **The movement half of every boundary probe asserts against a local $10^{-9}$**, not the shared
  `MOVEMENT_TOL`, and is paired with a whole-step perturbation that clears the shared tolerance
  (§6).

**Configuration and driver**

- **`configs/default.yaml` is written out in full**, not inherited from the comparison model's. The
  two `VAE_model` blocks do not share a schema, and a merge would leave dead keys that the signature
  sweep drops in silence. `tests/test_config_load.py` walks every leaf of both files and requires
  equality outside the seven-leaf input block `INPUT_PATHS`: `c_y`, `c_u`, `use_up_st`,
  `causal_reach_budget_s`, `frontend_reach_budget_s`, `load_fields` and `normalize_fields`. These
  leaves are *excluded* from the comparison. The test checks the rest against a five-entry exemption
  table `PARITY_EXEMPT_PATHS`: the run tag, plus four identity entries (the output tree, the MLflow
  experiment, the run name and the variant tag). A copy of any identity entry would write these runs
  into another model's tree. Both sets are also checked for stale entries.
- **The front end has one configuration key, `frontend_reach_budget_s`.** Widths are derived from
  `d_model`, and kernels are the constructor default `FRONTEND_KERNELS`. `frontend_reach_budget_s`
  bounds the backward reach (§3.1). An arm that changes the kernels must move it with them.
- **The pre-flight refuses the eight inert keys by name, before the run starts.** `INERT_MODEL_KEYS`
  in `trainer.py` lists them. Shared `main` calls the four inherited module-level guards by name
  before this hook. `_check_declared_widths_against_shard` returns early unless the config carries
  **both** `c_y` and `c_u`, and this pre-flight refuses a config that carries either. A copied
  sibling config with *correct* widths therefore passes that guard and then gets the inert-key
  message. `_check_causal_budget_resolves` is a no-op, because `causal_reach_budget_s` is absent.
- **The pre-flight also requires `up` in both `load_fields` and `normalize_fields`**
  (`_check_raw_source_normalized`). The inherited `_check_raw_target_normalized` already covers
  `fhr`. Without `up` in `load_fields`, the task fails on the first batch, after every rank has
  initialised. Without `up` in `normalize_fields`, nothing fails: the source arrives in raw
  contraction units and goes straight into the source encoder. The front end owns no statistics, so
  this guard is the only protection.
- **The shard-side pre-flight applies `trim_minutes` before it compares**
  (`_check_raw_length_against_shard`). Shards store the untrimmed geometry, `fhr` and `up` at $5280$
  and `weight` at $330$. The loader trims $\texttt{int}(4 \cdot 60 \cdot \texttt{trim\_minutes})$
  samples from **each** end, so $5280 - 2 \cdot 240 = 4800 = 300 \cdot 16$. The guard reproduces that
  one line, because the loader computes it in its constructor and not as a function. The guard is
  non-fatal on a missing, field-less or unreadable shard; the data module reports those cases.
- **`tiny.yaml` changes only the declared smoke deltas** (`TINY_DELTA_PATHS` in
  `tests/test_config_load.py`). In `VAE_model` these are `d_model: 32`, `d_z: 8`, `d_head: 8`,
  `max_lag: 8`, `encoder_d_ff: 64`, `dropout: 0.0` and `likelihood: mse`. `sequence_length`,
  `raw_per_step`, `warmup_period`, `horizon`, the conv stem, both block counts, the source window and
  `trim_minutes` keep the shipped values. `warmup_period` *cannot* shrink, because it sets the reach
  ceiling; a smaller value would build a narrower front end than the production run. The smoke fit
  therefore runs the production front-end kernels at the same $322/480$ reach.
- **`gradient_clip_val: 5000.0` is carried over from the comparison model**, and the config comment
  says so. §12 gives the procedure that replaces it.

## 12. Running it

From the repository root:

```
# Production, 7 ranks. TEB_RUN_STAMP is required so ranks 1..N-1 share rank 0's run directory.
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_transformer_e2e.trainer \
    --config teb_vae/lag_attn_transformer_e2e/configs/default.yaml

# Local smoke: one epoch, one device, the committed four-sample shard.
python -m teb_vae.lag_attn_transformer_e2e.trainer \
    --config teb_vae/lag_attn_transformer_e2e/configs/tiny.yaml
```

Both also launch from an IDE Run button with no command line. `RUN_CONFIG` near the bottom of
`trainer.py` names the config, and `--config` always wins over it. A relative path resolves against
the repository root, not the working directory. The entry point `chdir`s to the repository root,
because the paths *inside* a config are also relative to the repository root.

The tests:

```
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_e2e/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_e2e/tests -q -m slow
```

The smoke fit writes under `output/teb_vae_trf_e2e_tiny/<stamp>-lag_attn_trf_e2e_tiny/`:

- a checkpoint under this package's stem `lag-attn-trf-e2e` (`CHECKPOINT_STEM`);
- `resolved_config.yaml` beside it;
- `train_results/metrics_history.csv` with the tracked columns;
- `train_results/full.log`;
- `train_results/loss_plot_epoch.html`;
- the per-epoch diagnostic figures under `train_results/lag_attn_rws_diagnostics/`.

The slow tier (`tests/test_train_smoke.py`) runs the same path for three epochs (`SMOKE_EPOCHS`).
It uses three epochs because `lr` is logged at train-epoch start, so its first CSV cell is always
NaN, and the step warm-up needs more than one epoch to change visibly. The test asserts that
`train/grad_norm` is finite and non-zero. It also compares every front-end tensor after the fit
against a freshly built model and requires each one to differ, which catches a gradient that exists
but is never applied.

**Observability during a run.** All of it is inherited:

- the tracked metric surface in `metrics_history.csv`: the objective terms, `pred_gap`, both KL
  readouts, `prior_rate` and `beta_prior`, the log-variance diagnostics including
  `logvar_prior_floor_frac`, `anchor_coverage_frac`, the latent saturation fractions and the
  validation-only permutation control;
- `train/grad_norm` and `train/grad_clip_frac`;
- `train/spike_skipped` and `train/spike_ema_loss`, the only columns that show the loss-spike
  breaker skipping batches;
- the per-epoch diagnostic figure.

`profiler: simple` is enabled in `configs/default.yaml` and writes a `FIT Profiler Report` into
`train_results/full.log`. The row `[_TrainingEpochLoop].train_dataloader_next`, as a percentage of
`run_training_epoch`, gives the data-loading share of a step.

**Re-deriving `gradient_clip_val`.** Use the first production run:

1. Read `train/grad_norm` from `train_results/metrics_history.csv`. It is the **pre-clip** global
   norm.
2. Read it as **one optimizer step per epoch**, not as an epoch aggregate. The metrics collector
   reads `callback_metrics` at validation end, before the training epoch is reduced.
3. Take $q_{99}$ over at least a few thousand epochs.
4. Choose the smallest round value above $q_{99}$.
5. Read `train/grad_clip_frac` as a **mean over epochs** to see how often the threshold bound.
6. Write the chosen value into `configs/default.yaml`.
7. Replace the provisional comment with the percentiles that gave the value.

**Adopting `torch.compile`.** The key is live (§10) and ships off. Inductor needs Triton, so use a
Linux box with Triton installed:

1. Set `advanced_config.trainer.compile: true`.
2. Confirm that the startup log carries `torch.compile is ON`.
3. Run **one** epoch with compilation and one without, from the same seed on the same shards.
4. Compare `pred_gap`, `source_conditioned_kl_raw` and `nll_full_block` between the two
   `metrics_history.csv` files.
5. Abandon compilation if `pred_gap` moves by a noticeable fraction of itself.
6. Expect the first step of a compiled run to include the inductor compile time.

The trainer refuses `compile: true` together with `attention_grad_checkpoint`, and the message names
both.

## 13. Configuration keys

The first list gives the required keys, and the second the keys that are deliberately absent.
`tests/test_config_load.py` requires every `model_config.VAE_model` key in the shipped config to
reach the constructor or the task. Outside `VAE_model`, the first list holds the keys this document
refers to; it is not an inventory of the framework's settings.

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
- `model_config.VAE_model.frontend_reach_budget_s`
- `model_config.VAE_model.sequence_length`
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
- `advanced_config.trainer.profiler`
- `advanced_config.spike_breaker.ema_floor`
- `advanced_config.spike_breaker.additive_margin`
- `advanced_config.spike_breaker.comparison_metric`
- `advanced_config.callbacks.lag_attn_rws_plotting.enabled`
- `dataset_config.stat_path`
- `dataset_config.dataloader_config.normalize_fields`
- `dataset_config.dataloader_config.dataset_kwargs.load_fields`
- `dataset_config.dataloader_config.dataset_kwargs.trim_minutes`

Four of these keys need care:

- `dataset_config.dataloader_config.dataset_kwargs.load_fields` must carry `fhr`, `up`, `weight`
  and `guid`, and **none** of the four stored feature blocks. `fhr_up_ph` stays absent, because a
  coefficient that mixes both signals would remove the target-only / source-conditioned separation.
- `normalize_fields` must carry both `fhr` and `up` (§11).
- `trim_minutes` must match the stats file's value. It makes the raw grid $4800$ rather than $5280$.
- `advanced_config.callbacks.lag_attn_rws_plotting` keeps the **inherited driver's spelling**. A
  rename to match this package would disable the diagnostic figure with no error.

**Deliberately absent**

The input representation that this model replaces. None of these keys names a constructor argument
here, so the signature sweep would drop each one without a word. The pre-flight refuses each by
name instead:

- `model_config.VAE_model.c_y`
- `model_config.VAE_model.c_u`
- `model_config.VAE_model.use_up_st`
- `model_config.VAE_model.causal_reach_budget_s`
- `model_config.VAE_model.target_keep_index`
- `model_config.VAE_model.target_delays`
- `model_config.VAE_model.source_keep_index`
- `model_config.VAE_model.source_delays`

The front end's *shape*, which is derived or constant (§3.1). There is no width key, and the
kernels are the constructor default `frontend_kernels`. The backward reach budget is the one
front-end quantity that is configured, as `frontend_reach_budget_s` in the required list:

- `model_config.VAE_model.frontend_kernels`

The encoder keys of the LSTM-based ancestor, which are still absent:

- `model_config.VAE_model.lstm_layers`
- `model_config.VAE_model.encoder_extra_dilations`
- `model_config.VAE_model.encoder_extra_kernel`
- `model_config.VAE_model.conv_norm_groups`
- `model_config.VAE_model.causal_norm`

Keys that are structural, derived or inert. Each would look to a maintainer like a control that
exists:

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

The encoder head width is derived as `d_model // encoder_num_heads`. Encoder and front-end dropout
both use the existing `dropout`. The LayerScale initialisation and the rotary base are constructor
defaults. The target attention window is the full causal prefix by design. An availability
embedding belongs to the adapter that this model does not have. Encoder gradient checkpointing is
behind two memory levers that are already config keys (`batch_size` with
`accumulate_grad_batches`).
