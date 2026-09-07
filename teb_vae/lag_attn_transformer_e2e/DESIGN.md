# `lag_attn_transformer_e2e` — the as-built design record

The end-to-end causal raw-signal lag-attention VAE-TEB: what it is, what it replaces, what it
deliberately leaves alone, and every place the built module differs from the design it was built
from.

Companion documents: `teb_vae/lag_attn_transformer_rws/DESIGN.md` records the model this one is
compared against, whose encoders, objective, geometry, data contract and metric surface are shared
unchanged; `teb_vae/lag_attn_rws/DESIGN.md` records the package that owns the objective, the masks
and the trimmed-grid geometry all three models are built on. Neither is restated here — this record
covers the two encoder **inputs** and the wiring around them, because those are the only things
that changed.

---

## 1. What the model is

`teb_vae/lag_attn_transformer_rws` with **both encoder inputs replaced** and nothing else changed.
At every 4-second anchor $t$ the model still forecasts the next two minutes of raw normalized FHR —
$H \cdot R = 30 \times 16 = 480$ samples — twice: once from a target-only latent and once from a
source-conditioned one, through one shared decoder invoked twice under one noise draw. The KL
between the two latents, resolved across lags, is the coupling readout.

What changed is what the two encoders read. In the model being compared against, each stream
arrives as a block of **stored wavelet and phase-harmonic coefficients** and is projected onto the
backbone width by an availability-aware input adapter:

$$
[\,\texttt{fhr\_st} \mid \texttt{fhr\_ph}\,] \in \mathbb R^{B \times 300 \times 109}
\;\xrightarrow{\ \operatorname{AvailabilityInputAdapter}\ }\;
\mathbb R^{B \times 300 \times 128} \longrightarrow \text{encoder} .
$$

Here each stream arrives as the **raw 4 Hz signal itself** and is mapped onto the token grid by a
learned, strictly one-sided front end:

$$
\texttt{fhr} \in \mathbb R^{B \times 4800},\ \texttt{weight} \in \mathbb R^{B \times 300}
\;\xrightarrow{\ \operatorname{CausalRawFrontend}\ }\;
\mathbb R^{B \times 300 \times 128} \longrightarrow \text{encoder} .
$$

The reason is a property of the stored features rather than a preference. A scattering or
phase-harmonic coefficient at decimated step $t$ is a weighted average over raw samples on **both**
sides of $t$, so a model conditioning on "the past up to $t$" is conditioning on part of the
interval it is asked to forecast. Measured on the production filter bank, at the shipped two-minute
horizon, $31$ of the $109$ target channels and $29$ of the $58$ source channels have already seen
the *whole* forecast window, and every one of the $15$ `up_ph` channels reaches at least $100$ s
forward. The two leaks are not equally damaging: a target-stream leak improves the base and full
forecasts roughly equally and largely cancels in `pred_gap`, so it corrupts the *forecasting* claim;
a source-stream leak enters the posterior alone, so it inflates $D_0 - D_1$, inflates $K$ and shifts
the lag map — which are the three readouts the model exists to produce. §9 states what closing it
buys and what it does not.

At the shipped configuration the model holds **5,081,644 parameters** (measured, §5), $84{,}800$
more than the $4{,}996{,}844$ of the model it is compared against — and that difference is
*exactly* the two front ends against the two adapters they replace. Nothing else moved.

Everything downstream of the two front-end outputs is imported, unmodified: both
`CausalConvTransformerEncoder`s from `teb_vae.lag_attn_transformer_rws.nets`; the target-only
conditional prior head, the masks, the raw-target extraction, the trimmed-grid geometry and the
seven-term objective from `teb_vae.lag_attn_rws.nets`; the lag-restricted cross-attention over
$L = 91$ lags, the head-structured bounded posterior residual, the shared horizon decoder and the
lag-resolved KL attribution from `teb_vae.lag_attn.nets`. A change to any of them belongs in those
packages, where every model gets it.

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
reduced on the KL's own anchor support and in the same nats-per-anchor units. It exists because
nothing else in the objective penalises a *narrow* prior, and it ships here at the value the
comparison model ships — `beta_prior: 0.1` — deliberately: the prior-collapse threshold is a
property of the objective and the decoder, both of which are shared code, so a different value
would be a second difference between the two runs.

The last three are the auxiliary shape terms on the forecast means — pooled L1 for the envelope, a
derivative Huber for the slope, a boundary gap for the starting level — and they ship at the
comparison model's own $0.1 / 0.1 / 0.05$ for exactly the same reason, all three provisional. They
make `total_loss` a **mixed-unit** criterion: the shape terms are L1 and Huber quantities on z-scored
raw samples, not nats, so `nll_full_block` and `nll_base_block` are the pure-nats readouts. A term at
weight $0$ is not computed and reports exact $0.0$.
`tests/test_losses_delegation.py` feeds one real forward dict to **both** models' `compute_loss`
and compares every metric tensor with `torch.equal`, after first asserting that the two geometries,
index grids, coverage floors and log-variance bounds are equal — so "the same objective" is a fact
about the two models rather than about two one-line delegations.

## 2. The causal raw front end

`nets/frontend.py` is the one module in this package that is written rather than imported. Nothing
under `teb_vae` performs causal anti-aliased decimation of a raw signal, and the requirement — a
strictly one-sided map from the 4 Hz raw grid onto the 4 s token grid — cannot be met by composing
what exists. It uses no dependency beyond `torch` and the standard library; the filter coefficients
are a three-line binomial expression, not a signal-processing package.

Per stream:

$$
\text{featurise} \rightarrow
\underbrace{\text{stage}_1 \rightarrow \text{stage}_2 \rightarrow \text{stage}_3 \rightarrow
            \text{stage}_4}_{\text{stride } 2 \text{ each, widths } 32 \rightarrow 64 \rightarrow 96 \rightarrow 128}
\rightarrow \operatorname{RMSNorm}(d_{\mathrm{model}}) \rightarrow (B, T, d_{\mathrm{model}}) .
$$

The two streams are **two independently parameterised instances at identical settings**, never one
shared module: sharing would make the source state a function of the target and destroy the purity
the KL readout rests on, and differing per-stream settings would add a difference nobody has
measured.

### 2.1 Featurisation

$$
\left[\; \bar x_n\, m_n, \quad m_n, \quad (\bar x_n - \bar x_{n-1})\, m_n m_{n-1} \;\right],
\qquad
m_n = \mathbb 1\!\left[\,\mathrm{weight}_{\lfloor n/r \rfloor} \ge \theta\,\right]
      \wedge \operatorname{isfinite}(x_n),
$$

with $\bar x$ the signal **as the loader produced it** — already z-scored, so the front end owns no
statistics of its own — and $\theta$ the repository's own `VALID_THRESHOLD`, imported from
`teb_vae/lag_attn_rws/nets/raw_masks.py` rather than restated, because it is what the loss masks
use and a second float comparison here could drift from the mask the objective scores against.

Three properties are load-bearing, and `tests/test_frontend_featurize.py` measures each:

* **The validity mask is a channel.** A gap is then representable rather than indistinguishable
  from a genuine normalised zero — same value channel, different mask channel.
* **Invalid positions are neutralised in standardised space**, not raw space. Zeroing the raw value
  *before* a z-score would map a gap to $-\mu/\sigma \approx -7$ for FHR, an
  extreme-bradycardia-looking constant that the low-pass then smears across the following tokens.
  The neutralisation is `torch.where`, not a multiply: a non-finite sample times a zero mask is
  still non-finite.
* **The first difference is gated on both endpoints**, so the first valid sample after a gap
  injects no spurious slope; the replicate pad makes $\Delta x_0$ exactly zero for the same reason.

The ratio $r$ is derived from the two input lengths rather than passed, and `featurize` checks only
*divisibility* — it has no opinion about what the ratio should be. That the ratio is the front
end's own total stride is a different claim, and the front end asserts it against itself, at
construction and again per call.

### 2.2 One stage

$$
P = W x + b, \qquad G = \operatorname{GatedCausalConvBlock}_k(P), \qquad
\mathrm{out} = \operatorname{Decimate}_2(G) .
$$

The convolution block is the sibling's `nets/blocks.py::GatedCausalConvBlock`, imported and used
whole in the $(B, T, C)$ layout it already presents. That settles a hazard rather than defending
against it: the block adds its residual at the **full** rate and one decimator then runs on the sum,
so there is no separate skip path that could be decimated by a different operator and drift out of
sample alignment. It also means every front-end depthwise convolution is a `CausalDepthwiseConv1d`
— the exact class `init_depthwise_` detects — by construction rather than by remembering to use it
(§7).

The widening lives in the pointwise projection ahead of the block, because the block holds its width
constant. That projection carries a **bias**, the one place in this stack that does, and it is
load-bearing: a fully invalid window featurises to an exactly zero vector, and an exactly zero token
entering repeated pre-normalisation is the numerical accident the sibling's own adapter records
reaching gradient norms around $10^{26}$. The bias makes "this window is empty" a learnable
constant instead. It costs $320$ parameters at production.

### 2.3 The decimator

`CausalAntiAliasDecimate` is a fixed causal FIR low-pass followed by right-offset subsampling:

$$
\tilde x[n] = \sum_{i=0}^{\tau-1} h_i\, x[n-i], \qquad \mathrm{out}[t] = \tilde x[s t + s - 1],
\qquad h_i = \binom{\tau-1}{i} \Big/ 2^{\tau-1} .
$$

Three details are not stylistic, and each is a defect the module is shaped to make impossible:

* **The offset is right**, not left or centre. Composed over four stride-2 stages the total stride
  is $16$ and token $t$'s newest input sample is raw index $16t + 15$, which is exactly
  `TrimmedRawGeometry.n_raw(t) = 16(t+1) - 1`. The front end's decimation convention and the
  model's anchor convention agree with no off-by-one to negotiate. A centred offset would be *more*
  conservative and would still pass a causality probe while silently discarding the newest
  quarter-second of every token; a left offset would read the future.
  `tests/test_frontend_causality.py` perturbs raw $16t+16$ and requires token $t$ bitwise
  identical, and perturbs $16t+15$ and requires it to move.
* **The coefficients are a non-persistent buffer applied with `F.conv1d`**, never an `nn.Conv1d`.
  `teb_vae/lag_attn/nets/blocks.py::initialization` Xavier-fills every `nn.Conv1d` weight in the
  model, so a fixed kernel held as a layer would be replaced by random values with no error and no
  symptom beyond aliased high-frequency energy appearing as real variability.
  `tests/test_frontend_decimate.py` runs a full `initialization()` pass over a module holding one
  and requires the buffer bitwise identical — paired with a test-local `nn.Conv1d` holding the same
  coefficients, which the same pass must visibly change, or the assertion would also pass against an
  `initialization` that walked nothing.
* **There is no `antialias=False` switch.** A decimation without the filter is a different model,
  not a setting of this one.

$\tau = 5$ taps (`ANTI_ALIAS_TAPS`). A binomial kernel has $H(\pi) = 0$ **exactly** for every
$\tau \ge 2$, so the old Nyquist — what a factor-2 decimation folds onto DC — is annihilated rather
than attenuated, which is why the alias test asserts against $10^{-12}$ in float64 instead of
against a threshold somebody had to choose. More taps buy stopband depth at the point the fold
actually lands on: $|H(\pi/2)| = (\cos \pi/4)^{\tau-1}$ is $0.5$ at three taps and $0.25$ at five.
Aliased $0.5$–$2$ Hz energy would appear as fetal heart-rate variability that is not there, so the
depth is worth the reach it costs at both shipped geometries.

### 2.4 The final normalisation

Each front end ends in `RMSNorm(d_model)`. This mirrors the encoder's own final norm and is a
contract rather than a preference: the encoder stem, the prior head and the lag attention's
key-value normalisation are all calibrated to a normalised state, and a pre-norm residual stack
without a final norm exports an unnormalised stream whose scale grows with depth.

**Normalisation is channel-axis only, throughout.** `nn.GroupNorm` reduces over $(C/G, T)$ within a
group and therefore pools across time — the exact leak this package exists to remove — and the
`BatchNorm` family additionally makes every sample's output depend on the rest of the batch. Both
families, plus `SyncBatchNorm` (which does not subclass `BatchNorm1d`) and the `InstanceNorm`
family, are refused by `refuse_time_pooling_norms`, called at the end of every front end's
`__init__` so it is a standing guard rather than something a test happens to notice.
`tests/test_frontend.py` runs it in both directions, planting an `nn.GroupNorm` by monkeypatching
the name the stage looks up — which is the only route a real edit has, since the public constructor
offers none.

## 3. Reach

### 3.1 The front end's own reach, and the budget it is refused against

Reach is a **count** of raw samples in the support, itself included, matching the `receptive_field`
convention the sibling's blocks use. Accumulated from the *built* modules rather than recomputed
from the constructor arguments, so the reported number cannot disagree with the stack that produced
it:

$$
R \;=\; 2 \;+\; \sum_{i=1}^{4} \left(k_i + \tau - 2\right) \prod_{j<i} s_j ,
$$

each stage contributing $(k_i - 1)$ samples from its depthwise kernel and $(\tau - 1)$ from the
anti-alias filter, scaled by the stride already accumulated below it, and the leading $2$ being the
featurisation's one-sample first difference plus the sample itself.

The front end **refuses at construction** when $R$ exceeds
$\texttt{warmup\_period} \times \texttt{raw\_per\_step}$, naming both numbers. That budget is not a
configuration key and not a caller's choice — the model passes it, because an anchor inside the
warm-up is the only one allowed to see the zero-padded convolution transient at the segment's start.
`tests/test_construct.py` asserts the model passes exactly that product; nothing else connects the
guard to this geometry.

Printed by `python -m teb_vae.lag_attn_transformer_e2e.nets.frontend`:

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

The $322$ is pinned as `SHIPPED_REACH_SAMPLES` in `tests/test_frontend_reach.py` and checked against
this document by `tests/test_docs.py`; the $480$ is `warmup_period * raw_per_step` and is derived,
never configured. The kernels $(65, 15, 15, 15)$ are the
constructor's own default (`FRONTEND_KERNELS`), following the precedent `ROPE_BASE` sets in the
sibling's blocks: no arm varies them, so a config key would be a configuration surface with nothing
behind it, and the reach guard bounds any future choice. Widths are derived from $d_{\mathrm{model}}$
as $(d/4,\, d/2,\, 3d/4,\, d)$, so there is no width configuration to get wrong and no
"the last stage must equal $d_{\mathrm{model}}$" invariant to violate.

The probe pins the *safety* claim rather than tightness: perturbing raw at $n - R$ leaves the token
bitwise identical, and perturbing at $n$ moves it. Asserting that $n - R + 1$ moves would pin the
bound as tight, which nothing requires and which would break the first time a kernel change made
the formula conservative.

At the smoke geometry `tests/conftest.py` ships $(5, 3, 3, 3)$ against a budget of
$6 \times 16 = 96$, reaching $94$ (`TINY_REACH_SAMPLES`). Two samples of margin — enough that the
reach guard would refuse a seven-tap filter there, which is the guard doing its job rather than a
problem with it. `configs/tiny.yaml` is a different case and does **not** use those kernels: it
keeps `warmup_period: 30` and the production kernels and shrinks widths only, so the smoke fit runs
the front ends at their production reach of $322/480$ (§12).

### 3.2 The composed raw receptive field

A derived quantity neither sibling has recorded, because neither could: it only means something once
the input is the raw signal. The source encoder's token reach is
$R_U = R_{\mathrm{conv}} + N_U(W_U - 1) = 21 + 3 \cdot 15 = 66$ steps, so the source state at anchor
$t$ reads tokens $t - 65 \ldots t$, and each of those tokens reads $322$ raw samples ending at its
own anchor. Composed:

$$
R^{\mathrm{raw}}_U \;=\; R_{\mathrm{frontend}} + r\,(R_U - 1) \;=\; 322 + 16 \cdot 65
\;=\; \mathbf{1362} \text{ raw samples} \;=\; 340.5 \text{ s},
$$

ending at raw index $16t + 15$ and reaching no further forward. (The formula is
$R + r(R_U - 1)$, not $R + rR_U$, because both reaches are counts: the two supports overlap on the
anchor token's own $16$ samples.) `tests/test_docs.py` computes it from a constructed
shipped-geometry model and checks it against this section.

$340.5$ s is still **shorter than the lag search range** — $\ell \in \{0, \ldots, 90\}$ spans $90$
steps, $360$ s — which is the property the sibling's §3 argues the source encoder must have: an
encoder whose reach exceeded the lag range would already be doing the alignment the lag attention
exists to do, and the reported lag would stop being a statement about where the coupling came from.
The margin is $19.5$ s and it is the tightest structural constraint in the model. Anything that
widens either the source attention window, the encoder stem or the front-end kernels must be
checked against it: each extra step of `source_attention_window` adds $N_U = 3$ steps to $R_U$ and
so $12$ s of raw reach, which leaves exactly one step of headroom — $17$ reaches $352.5$ s and
still fits, $18$ reaches $364.5$ s and does not.

The target encoder's reach is reported as **unbounded** rather than as $T$ — the bound is *absent*,
and a caller reading a number could not tell the two apart — so its composed raw reach is the whole
trimmed prefix, ending at $16t + 15$.

## 4. Forward return dict

`nets/model.py::SeqVaeLagAttnTrfE2E.forward(y_raw, u_raw, weight)` returns exactly the twenty keys
both comparison models return, at the same shapes — `mu_prior`, `logvar_prior`, `raw_logvar_prior`,
`mu_post`, `logvar_post`, `z_prior`, `z_post`, `target_state`, `source_state`,
`attended_source_heads`, `attn_weights`, `mu_base`, `logvar_base`, `mu_full`, `logvar_full`,
`kld_per_t`, `kld_per_t_per_head`, `source_kl_lag_map`, `mu_prior_sat_frac`, `delta_mu_sat_frac`.
There is deliberately no `decoder_state` and no `delta_mu_src`: neither pathway exists.

`tests/test_forward_contract.py` compares the key set and every shape against a constructed
`SeqVaeLagAttnTrfRws` on the same stub batch, rather than against a table written here — so "the
same twenty keys at the same shapes" is a fact about the two models rather than about two copies of
one list.

The parameters are named `y_raw` / `u_raw` rather than `fhr_raw` / `up_raw`: a `nets/` module takes
tensors and may not know what anything was called on disk. That naming has a cost this architecture
alone pays, and §6 records how it is caught.

## 5. Parameter budget and measured cost

Measured on a constructed shipped-geometry model, not predicted.

| Component | Parameters |
| --- | ---: |
| Front-end stage 1, $32$ wide, $k = 65$ | $5{,}376$ |
| Front-end stage 2, $64$ wide, $k = 15$ | $15{,}552$ |
| Front-end stage 3, $96$ wide, $k = 15$ | $35{,}616$ |
| Front-end stage 4, $128$ wide, $k = 15$ | $63{,}872$ |
| Final `RMSNorm(128)` | $128$ |
| One front end | $120{,}544$ |
| Both front ends | $241{,}088$ |
| Both encoders, imported unchanged | $2{,}565{,}888$ |
| Everything else, unchanged | $2{,}274{,}668$ |
| **Total** | $\mathbf{5{,}081{,}644}$ |

That total is the **one place the absolute number is pinned**, and `tests/test_docs.py` checks it
against `sum(p.numel() ...)` rather than against a literal in a test — so a legitimate shared change
to an imported downstream component re-costs this line rather than failing a test elsewhere in the
package. `tests/test_construct.py` pins the front-end figure as a *delta* instead:
$241{,}088$ against the $156{,}288$ of the two adapters replaced, so the entire architectural cost
of reading the raw signal rather than a two-sided transform of it is $84{,}800$ parameters, about
$1.7\%$ of the model — and
$5{,}081{,}644 - 4{,}996{,}844 = 241{,}088 - 156{,}288$ exactly, which is the arithmetic statement
that nothing else moved.

The eight anti-alias filters — one per stage per stream — contribute **zero** parameters, which
`tests/test_construct.py` asserts alongside their count. Held as `nn.Conv1d` layers they would be
both counted here and Xavier-overwritten at initialisation.

**Step cost and activation memory.** Measured on the development box (RTX 4080 Laptop), batch $16$,
fp32, one forward plus backward of the **whole model** at the shipped geometry, against the model
this one is compared against under the same conditions:

| | ms per step | peak allocated |
| --- | ---: | ---: |
| feature-input model, the one compared against | $104.0$–$104.3$ | $3{,}256$ MiB |
| this model | $127.0$–$128.5$ | $4{,}092$ MiB |
| difference | $+22\%$ | $+836$ MiB |

The two front ends in isolation account for essentially all of it: timed alone, forward plus
backward at the same batch, they cost $23.8$–$25.1$ ms at a peak of $864$ MiB. Nothing downstream of
them changed, so nothing downstream of them moved.

**Where the step actually goes.** Forward only, same box and batch, fp32, so the parts sum to the
forward rather than to the full step:

| component | ms | note |
| --- | ---: | --- |
| decoder, $\times 2$ per step | $37.1$ | $270$ anchors $\times$ $H$ $\times$ $R$, base and full |
| target encoder | $12.4$ | $4$ blocks, full causal prefix |
| source encoder | $9.9$ | $3$ blocks, window $16$ |
| target front end | $6.4$ | $4{,}800$ raw samples |
| source front end | $6.4$ | |
| `compute_loss` | $6.0$ | |
| posterior head | $5.5$ | |
| lag attention | $4.5$ | $3.7$ with `softmax` instead of `entmax15` |
| prior head | $2.5$ | |

The decoder dominates, and it dominates because **every valid anchor is decoded every batch** while
adjacent anchors overlap in $29$ of their $30$ horizon steps. Timed alone at batch $16$, forward
plus backward, one invocation: $45.5$ ms at $270$ anchors, $20.0$ ms at $135$, $11.8$ ms at $90$.
That is the upside behind the standing `lean-limit` in `teb_vae/lag_attn_rws/nets/losses.py`, and
it is a change to the gradient estimator rather than a free speed-up, so it is not made here.

Two levers that look attractive and are not, both measured rather than argued: throughput is still
climbing at batch $32$ ($38.4 \to 66.4 \to 90.5$ samples/s at batch $8 \to 16 \to 32$), so the
model is launch-bound at small batch and the shipped batch $128$ is already the right side of that
curve; and the loss-spike breaker's two `.item()` calls per step cost $5.4$ ms of a $200$ ms step
($2.7\%$), which is real but is not where a first optimisation belongs.

These are hardware measurements and are the one class of number in this document that **no test
pins** — the same is true of the smoke gradient norms and the smoke wall time in §12. Every
structural number here is driven from the code by `tests/test_docs.py` or by the test named beside
it; a timing is reproduced by re-running the measurement, not by a gate.

**Is the run data-loading bound?** Unresolved on this box, and that is the finding. Dropping the
four feature blocks removes $(109 + 58) \times 300 = 50{,}100$ elements of read and host-to-device
traffic per sample against the $2 \times 4800 + 300 = 9{,}900$ retained — about $5\times$ less data
per sample, whatever the stored dtype — so whether the net wall-clock
effect of this architecture is *faster* despite the step cost above depends entirely on the
data-loading share of the comparison model's production step. `profiler: simple` is already enabled
in both packages' `configs/default.yaml` and writes a `FIT Profiler Report` into each run's
`train_results/full.log`; the row to read is `[_TrainingEpochLoop].train_dataloader_next` as a
percentage of `run_training_epoch`. The production baseline's log is on the production box: only
its `metrics_history.csv` and a copied checkpoint were brought across, under
`output/lag_attn_rws_transformer/`, and neither carries timing. The local smoke run's own report
puts that row at $0.29\%$, which is **not** evidence about production — four cached samples,
`num_workers: 0`, CPU, and a step dominated by the per-epoch plotting callback. Answering the
question needs one `grep -A40 'FIT Profiler Report'` over the baseline run's `full.log` on the
production box; no code, and no new run.

## 6. Structural constraints that are not preferences

Each is enforced by construction and measured by a test, never asserted by convention.

| Constraint | What enforces it | Test |
| --- | --- | --- |
| **Raw-signal causality**, $H_t = f(x_{\le 16t+15})$, per front end and per assembled state | right-offset decimation; left-only convolution padding; every other front-end primitive position-wise on the channel axis | `tests/test_frontend_causality.py`, `tests/test_causality.py` |
| **The warm-up covers the front end's reach** | the front end refuses at construction against `warmup_period * raw_per_step` | `tests/test_frontend_reach.py`, `tests/test_construct.py` |
| **No time-pooling or batch-coupling normaliser** in the front ends | `refuse_time_pooling_norms` at the end of every `__init__` | `tests/test_frontend.py` |
| **No time-pooling normaliser elsewhere, and no recurrence anywhere** | none is constructed; the three surviving `GroupNorm`s are enumerated and each asserted under `horizon_core.`, where they pool the *forecast* axis of one anchor | `tests/test_construct.py` |
| **Source purity** — the prior never sees the source, the source state never sees the target | separate front ends and encoders; the posterior is a residual on the prior | `tests/test_source_purity.py` |
| **The two raw arguments are not interchangeable** | a forward-pre-hook identity check, because two same-shaped raw tensors make a swap otherwise invisible | `tests/test_source_purity.py` |
| **No decoder bypass** — gradient reaches the decoder only through $z$ | `BaselineFutureDecoder.forward` takes exactly one tensor, at $d_z$ in-features | `tests/test_construct.py` |
| **Exact zero KL at initialisation**, and bitwise identical base and full *in train mode* | posterior deltas zeroed **after** the generic init; one shared $\epsilon$; decoder and both attention dropouts fixed at $0$ | `tests/test_zero_kl_init.py` |
| **The lag attribution identity**, $\sum_\ell \widetilde K_{t,\ell} = K_t$, exactly | the lag attention is built at `dropout=0.0`, so the returned probabilities are the ones the posterior consumed | `tests/test_lag_map.py` |
| **`lag_attn.W_o` frozen** | the head-structured posterior consumes the per-head summaries, so `W_o` receives no gradient; freezing drops it from DDP's expectation set | `tests/test_construct.py` |
| **The fixed FIRs survive initialisation** | non-persistent buffers applied with `F.conv1d`, never `nn.Conv1d` weights | `tests/test_frontend_decimate.py`, `tests/test_init_policies.py` |
| **`nets/` reaches no framework layer** | import-graph walk over every `nets/*.py` | `tests/test_nets_are_framework_free.py` |

Two conventions run through the whole suite and are not optional here.

**Every positive invariant is paired with a probe-is-not-vacuous negative control.** The KL is
identically zero at initialisation, so any KL assertion on a fresh model passes on a completely
broken one — and this architecture adds three instances of its own: a dead stage passes every
per-token invariant, an all-ones mask channel passes every gap test, and a clobbered FIR passes
every shape test. The planted defect for causality is `_SymmetricallyPaddedDecimate`, a subclass of
the real decimator with one method replaced, chosen because it has the same shapes, the same reach
and the same parameter count — so nothing but a causality probe could find it, which is exactly the
property a control needs. A centred *offset* would not be a valid control: it makes the token depend
on raw $\le 16t$, which is strictly more conservative, so the planted-broken model would pass the
bitwise half.

**The causality control is positional and needs no test hook in production code.** Perturbing
strictly after the cut must leave the output at the cut bit-stable *and* must visibly move the
output at the end.

One measurement is worth recording because it surprised. A token's **single newest** raw sample
moves it by only $1.3$–$1.9 \times 10^{-5}$: each stage's filter puts the newest sample on its
leading tap $h_0 = 1/16$, so after four stages that sample carries $16^{-4} \approx 1.5 \times
10^{-5}$ of the token. Through the encoder it becomes $1.5 \times 10^{-3}$ at the first trained
anchor and $6.9 \times 10^{-5}$ at the last, because how much an edge sample matters depends on how
much history the anchor has. No single threshold near the suite's shared `MOVEMENT_TOL` of
$10^{-3}$ fits both, so the boundary tests assert movement against a local $10^{-9}$ — nine orders
above float64 round-off — and pair it with the anchor's **whole step**, which moves it by
$4.7 \times 10^{-3}$ to $3.8 \times 10^{-1}$ and clears the shared tolerance. A low-pass is meant to
weight the edge of its window lightly; what it may not do is weight anything past that edge, which
is the bitwise half.

## 7. Initialisation order

Load-bearing, top to bottom, in `nets/model.py::__init__`:

1. `initialization(self)` — the shared generic pass, Xavier over every `nn.Linear` and `nn.Conv1d`.
   The front ends' anti-alias filters are untouched, because they are buffers rather than layers
   (§2.3).
2. `init_depthwise_(self)` — **immediately after**, never before. Xavier on a $(C, 1, k)$ depthwise
   weight reads $\mathrm{fan\_in} = k$ against $\mathrm{fan\_out} = Ck$, giving
   $\sigma = \sqrt{2/(k + Ck)}$ against the variance-preserving $1/\sqrt k$ — a factor
   $\sqrt{(1+C)/2}$ too small, **independent of $k$**, so the affected convolutions would start an
   order of magnitude too quiet and no kernel sweep could reveal it.
3. `_restore_frontend_stage_bias()` — also gated on the generic pass, and for the same kind of
   reason: `initialization` zeros every `nn.Linear` bias, which includes the four stage projections
   per front end. Those are the *only* biases in either front end (§2.2), so zeroed, a fully
   invalid window's exactly-zero feature vector maps to an exactly-zero token and the mechanism the
   bias exists to provide is gone. Restores torch's own `nn.Linear` default rather than inventing a
   scale.
4. `_zero_init_delta_heads()` — the generic pass would otherwise refill the posterior delta heads
   and destroy the exact zero-KL start.
5. `_zero_init_film_generators()` — the horizon core zero-initialises them itself and the generic
   pass refills them, so re-zeroing here is what makes the identity-at-init actually true.
6. The three zero-parameter init policies — `horizon_embed_std`, `head_init_calibration`,
   `a_head_gain` — each applied only when its config value leaves the constructor default, so a
   default-flag model is bitwise the pre-bundle one.

Steps 2 and 3 are the same lesson twice: the generic pass is right for the model as a whole and
wrong for two specific things inside the front end, and both repairs must run *after* it. The
symptom in each case is silent — a stage an order of magnitude too quiet, and a token that is
exactly zero only on the windows where it matters — which is why both are measured rather than
asserted. Step 3's consequence is quantified in §8.

`n_depthwise_init` is **12** here against the sibling's **4** at equal stem settings: $4$ encoder
stem convolutions plus $8$ front-end ones, one per stage per stream.
`tests/test_init_policies.py` asserts the difference is exactly $2 \times \texttt{NUM\_STAGES}$
rather than merely positive, and proves the *ordering* by running `initialization` again afterwards
and requiring the measured standard deviation to visibly drop — so "after, never before" is
measured rather than asserted. This is where the risk of a front-end convolution starting $8\times$
too quiet is actually retired; a count alone would not catch a wrong standard deviation.

Two tolerances there are wider than the sibling's, and the reason is **sampling spread over twelve
convolutions**, not a narrower bank. At the shipped geometry the narrowest depthwise bank is still
the encoder stem's $Ck = 640$, exactly the sibling's; what changed is that there are now twelve
banks to satisfy rather than four, and the worst relative deviation of $\sigma$ from
$1/\sqrt k$ across twenty-five seeds reaches $9.7\%$. The sibling's $10\%$ band would therefore
flake here, and the band is $20\%$ instead. The correction-factor bar is $3\times$ rather than the
sibling's higher one because the predicted factor $\sqrt{(1+C)/2}$ is $8.03$ at $C = 128$ but only
$4.1$ at the front end's narrowest $C = 32$; for the same reason the ordering counterfactual is
measured on the **last** front-end stage, where the generic pass has a factor large enough to see.

**At initialisation each front end is approximately a linear mix of the decimated
`[value, mask, delta]` channels.** `LayerScale` starts every convolution block's residual branch at
$10^{-2}$, so each stage is close to its pointwise projection composed with the fixed low-pass. That
is a sane start rather than a defect, but it is stated rather than left to be discovered, and
`tests/test_frontend.py` pins it — the first epochs are the stages finding temporal structure that
is not there yet.

## 8. DDP reachability

Production runs under plain `"ddp"` with `find_unused_parameters=False`, so every parameter must be
reachable in every backward. The intuitive reading of what that requires is wrong, and the
difference decides the front end's shape.

A parameter multiplied by an identically-zero tensor **is** reachable: its `AccumulateGrad` node
fires, it receives a zeros gradient rather than `None`, and DDP's reducer marks it ready. What
breaks `find_unused_parameters=False` is a parameter left *out of the graph* — a Python-level
`if mask.any(): ...` that skips an operation entirely, on some ranks and not others, on some batches
and not others.

So the rule is: **the front end's masking is multiplicative and unconditional.** No `forward` in
this package branches on a tensor value. `tests/test_ddp_reachability.py` asserts both halves — every
`requires_grad` parameter has a gradient after one backward at both keyword sets, with a
deliberately dangling parameter as the negative control; and an AST walk over `nets/frontend.py` and
`nets/model.py`, whose machinery is **imported** from the sibling's file rather than retyped, finds
no `if` or conditional expression inside a `forward` whose test reads a tensor value. The walk finds
six conditionals across the two modules and flags none: five are shape-metadata guards that raise,
and the sixth is `forward`'s `... if self.query_uses_logvar else ...`, which reads a Python bool
fixed at construction and is therefore identical on every rank at every step. A local non-vacuity
check requires the walk to find *some* conditional, or a renamed `forward` would make it pass on
anything. The rule admits "an expression built only from constants, names, attributes, `.shape`
subscripts, comparisons and boolean operators" and rejects any call, element subscript or
arithmetic; its one stated gap, inherited with the machinery, is that `self.some_tensor > 0` would
be admitted — and would raise on the first forward rather than diverge silently.

Two behavioural halves sit beside it that the siblings have no reason to carry: every front-end
parameter receives a gradient on a batch carrying a planted weight gap, and again on a **fully
masked** batch, where the featurisation emits an exactly zero vector and the stage projections'
biases (§2.2) are the only thing keeping the tokens off zero. That fully-masked case is the one an
`if mask.any():` optimisation would have been written for.

**What the fully-masked case costs when the bias is missing**, measured, because it is the reason
§7 step 3 exists. With the stage biases at zero a $26$-step ($104$ s) validity gap — routine in this
dataset — takes the global pre-clip gradient norm from $2.5 \times 10^4$ to $2.1 \times 10^{17}$,
essentially all of it on the first stage's bias, because an exactly zero token entering repeated
pre-normalisation produces derivatives of order $1/\sqrt{\epsilon}$. Under
`gradient_clip_val: 5000` applied as a global norm, that rescales *every* parameter's gradient for
the batch by around $10^{-13}$: the batch contributes nothing, and the loss moves too little for the
spike breaker to notice. Restoring the bias returns the norm to $2.3 \times 10^4$.
`tests/test_frontend.py` asserts both halves — that the model's own front end emits a non-zero token
on a fully invalid window, and that every stage projection's bias survives initialisation.

This shaped one production-code detail. The front end's three shape guards were originally written
`x.dim() != 3` and `int(x.shape[1]) != self.channels`; the walk rejects **any call** inside a
conditional, because a call is how a forward reads a tensor's content. They are now `x.ndim != 3`
and `x.shape[1] != self.channels` — the same claim in the form the walk admits — with the reason
recorded at the first of them.

**The three settings the strategy carries**, from `LagAttnRwsTrainer.ddp_kwargs`, inherited
unchanged. `select_ddp_strategy` returns a configured `DDPStrategy` rather than the `'ddp'` /
`'ddp_find_unused_parameters_true'` shorthand, because those strings can express
`find_unused_parameters` and nothing else:

| setting | value | why |
| --- | --- | --- |
| `find_unused_parameters` | `False` under `gaussian_nll` | everything above; `True` under `mse`, which starves the decoder log-variance heads |
| `broadcast_buffers` | `False` | DDP re-broadcasts every buffer from rank $0$ on **each forward**, which here is $1.5$ MiB |
| `gradient_as_bucket_view` | `True` | `param.grad` points at the reduction bucket instead of a separate allocation |

`broadcast_buffers=False` is worth its own justification because it is the one that could be
wrong. This model's buffers are the eight fixed anti-alias filter banks (§2.3), fourteen rotary
tables, three causal attention masks and the raw-target index grid — every one a deterministic
function of the config, built identically in each rank's constructor, so the broadcast restored
values that were never going to differ. It is safe **because there is no `BatchNorm` anywhere in
this model**, a running statistic being the one kind of buffer that genuinely diverges per rank;
`sync_batchnorm: false` is the same fact stated in the config. A model that gained one would need
this back.

**`static_graph` is deliberately absent**, and that is a correctness call rather than an omission.
It promises DDP an identical autograd graph on every iteration, and the loss-spike circuit breaker
breaks exactly that promise: on a skipped batch it substitutes a loss summed over every trainable
parameter times zero, which is a structurally different backward from the one iteration $1$
recorded. The breaker ships enabled, so the promise would be false on precisely the batches that
already went wrong. `tests/test_ddp_strategy.py` carries it as a negative control on the bundle.

## 9. Raw-signal causality: what this model has, and what it still does not claim

The sibling's §9 draws the distinction and delivers the weaker half. This model delivers the
stronger one.

**Token causality**, $H_t = f(X_{\le t})$, is a property of the *encoder*. Both models have it.

**Raw-signal causality**,
$H_t = f\!\left(Y_{\le n_{\mathrm{raw}}(t)},\, U_{\le n_{\mathrm{raw}}(t)}\right)$ with
$n_{\mathrm{raw}}(t) = 16t + 15$, is a property of the *input representation*, and no encoder can
supply it. The sibling's inputs are two-sided wavelet and phase-harmonic transforms — a feature step
reads raw signal from its own future, up to $974$ s of it — so a token-causal model over them still
conditions on part of the interval it forecasts. This model has raw-signal causality by
construction, and it is **measured, not argued**: `tests/test_causality.py` perturbs the raw input
at $16t + 16$ and requires `target_state`, `source_state` and `mu_prior` at anchor $t$ to be bitwise
identical, then perturbs at $16t + 15$ and requires movement, in float64 with a large amplitude and
`torch.equal` at the cut.

The two partial remedies available on the sibling do not close it, and both remain available there,
unaffected by this package. `causal_reach_budget_s` prunes channels whose analytic reach exceeds a
budget and delays the survivors, but the delay may not exceed `warmup_period`, which caps the budget
at $120$ s in the shipped geometry — keeping $78/109$ target and $29/58$ source channels, losing
$13$ of the $15$ `up_ph` channels, and still leaking each survivor's $5\%$ energy tail, because
$L_{95}$ is an energy *quantile* rather than a hard support. A longer forecast horizon dilutes the
contaminated *fraction* but leaves $15$ channels fully leaked and does nothing at all for the
coupling readout, because the source encoder still reads UP recorded after the anchor no matter how
far ahead the decoder forecasts.

**What this does not license.** The KL is still `source_conditioned_kl_raw` / `_train` and its
decomposition is still `source_kl_lag_map`. It is **not** called transfer entropy, and the reason
has changed rather than disappeared: what the model reports is a KL between two of *its own*
distributions, fitted by one particular architecture on one particular dataset, and renaming a
quantity on the strength of an architectural argument is how a readout starts being read as
something nobody measured. What the input change buys is that the quantity is now a statement about
the source's *past* — which is a precondition for the interpretation, not the interpretation.

The startup log states this run's standing, replacing the sibling's sentence about features reading
$974$ s into their own future — which would otherwise appear in every production run of a package
whose central claim is its negation:

```
causal standing: raw-signal inputs through strictly one-sided front ends -- the history state at
step t is a function of raw samples at index <= 16t + 15 and no further, so the source-conditioned
KL reads no source content recorded after its own anchor. Front-end reach 322 raw samples (80.5 s)
against a budget of 480 (120.0 s).
```

The reach in that line is measured on a *throwaway* front end built from the resolved constructor
kwargs, because the base logs this before `self.pytorch_model` is assigned — deliberately, so a
launch that dies in the constructor has still said what it was about to build. The obvious hazard is
that second construction disagreeing with the real one, so
`tests/test_trainer.py::test_the_logged_reach_cannot_drift_from_the_front_end_that_was_built` pins
the logged number against both front ends of the model a real `create_model` produces.

**The run's `resolved_config.yaml` records `model_config.resolved_causal_budget: null` and carries
no front-end reach at all.** That key is the inherited driver's, and `null` is the correct value —
there is no forward reach to prune channels against. The front end's *backward* reach is derived
rather than configured, so it appears in no config artifact; the two places it is recorded are the
startup log above and `SHIPPED_REACH_SAMPLES` in `tests/test_frontend_reach.py`.

## 10. Deliberate limitations

> lean-limit: no eval package, so this model cannot be run through the shared evaluation pipeline
> and what it emits during training is the only readout; replace with a `ModelBinding` and an
> `eval/` package when the separate evaluation specification lands.

> lean-limit: no architecture arms, so every front-end choice here is reasoned rather than measured;
> replace with `configs/sweep_*.yaml` and a declared-delta lint once the baseline run has produced
> numbers to sweep against.

> lean-limit: the front-end width schedule (32/64/96/128) and kernel schedule (65/15/15/15) are
> chosen by analogy to a design that worked rather than measured here; replace with a measured
> schedule when the first arms compare at least two of them, and note the reach guard bounds any
> choice.

> lean-limit: primitives imported from `teb_vae/lag_attn/nets`, `teb_vae/lag_attn_rws/nets` and
> `teb_vae/lag_attn_transformer_rws/nets` rather than promoted to a common package; this package is
> the fourth consumer, so both siblings' promotion triggers have now fired and each is deferred
> again for the reason recorded there. Promote to a common package when this model outlives its
> comparison, or when a consumer needs to modify one of the shared primitives rather than only
> import it.

> lean-limit: `teb_vae/lag_attn_rws/eval/metrics.py::model_inputs` still calls
> `_build_target_streams` and `_build_source_stream` directly and returns
> `(y_st, y_ph, u_stream, fhr_raw, weight)`, so it is the one call site in that package **not**
> routed through the `_build_forward_inputs` hook the task and the plotting callback now use;
> replace with the hook when this package gains an evaluation entry point, which is what would
> first make that call site reachable with a raw-input task.

The last one is worth stating rather than leaving to be rediscovered: leaving it un-routed was
deliberate while evaluation is out of scope, because routing it would change the shape of a tuple
its evaluation callers unpack, in a package this one does not otherwise touch — and a raw-input
task cannot reach it today, so the divergence costs nothing until the evaluation record exists to
carry it.

Also deliberate:

- **No warm start.** `core_model_checkpoint` stays `null`: a checkpoint from either sibling carries
  a different `model_class` stamp and holds input-adapter tensors this architecture does not have,
  so `check_model_class` refuses it — correctly. The front ends are the thing being measured and
  must be trained from scratch.
- **No mixed precision, and now for a measured reason rather than an inherited one.** The sibling
  records that mixed precision "needs float32 islands around the log-variances, the closed-form KL
  and the Gaussian NLL reduction". Measured here, that framing understates it: the term bf16
  destroys is **`pred_gap`**, and no island around the loss can save it, because the damage is done
  in the decoder before the loss sees anything. On one batch at the shipped geometry, against fp32:

  | metric | fp32 | bf16 rel. error | fp16 rel. error |
  | --- | ---: | ---: | ---: |
  | `nll_full_block` | $689.333$ | $7.9\times10^{-5}$ | $3.8\times10^{-5}$ |
  | `nll_base_block` | $689.246$ | $8.9\times10^{-8}$ | $6.1\times10^{-5}$ |
  | **`pred_gap`** | $-0.0867$ | $\mathbf{6.3\times10^{-1}}$ | $1.8\times10^{-1}$ |
  | `source_conditioned_kl_raw` | $6.457$ | $9.4\times10^{-6}$ | $1.3\times10^{-4}$ |
  | `prior_rate` | $3.304$ | $1.9\times10^{-3}$ | $1.3\times10^{-4}$ |

  $\mathrm{pred\_gap} = D_0 - D_1$ is a difference of order $10^{-1}$ between two block NLLs of
  order $10^{2}$ — a relative scale of $1.3\times10^{-4}$ against bf16's $\epsilon$ of
  $3.9\times10^{-3}$. `mu_base` and `mu_full` come from one decoder invoked twice, so each picks up
  its *own* rounding, and the noise in their difference swamps the signal. Confining bf16 to the
  front ends and encoders would leave the decoder — roughly half the step — in fp32 and buy about
  $1.05\times$; applied to everything it measured $1.13\times$ at batch $32$ and was *slower* than
  fp32 at batch $16$. The model is not matmul-bound, so there is little there to win.

  **The measurement has a trap in it, recorded because the first attempt fell into it.** At
  initialisation the KL, `pred_gap` and `prior_rate` are all exactly $0$ by construction (§7), so a
  bf16-versus-fp32 comparison on a fresh model compares $0.0$ against $0.0$ and passes on a
  completely broken precision path. The table above is measured with the posterior delta heads and
  the prior log-variance head perturbed off their zero start — the same
  probe-is-not-vacuous discipline §6 applies to every other invariant here.

- **`compile: false`, but the key is now live rather than inert.** Two of the three blockers once
  recorded against this family do not exist here, and the third never applied to the compiled
  region at all. There is no LSTM in this architecture. `attention_grad_checkpoint` is reachable
  and *is* a genuine blocker, so `compile_model_requested` refuses the two together by name rather
  than silently dropping one. And the data-dependent mask indexing behind `kld_active_frac` lives
  in `compute_loss`, which `compute_loss_and_metrics` reaches through **`orig_model`** — only the
  forward is ever compiled, so that indexing cannot enter the graph.
  `tests/test_trainer.py::test_the_objective_is_never_the_thing_compiled` pins that routing,
  because it is the single line that makes the key safe to honour.

  **The hook is `LagAttnTrfRwsTrainer`'s, inherited rather than restated here**, and that placement
  is the point: the blocker it clears is the *LSTM*, which both conv-Transformer packages replaced,
  so the decision belongs to the driver where it first becomes true. A copy here would be a second
  copy of one decision on a question that has nothing to do with the input representation this
  package exists to change, free to diverge from it.
  `tests/test_trainer.py::test_the_compile_decision_is_inherited_rather_than_restated` pins it.

  It ships **off** for a numerical reason rather than a mechanical one: inductor may reassociate
  float arithmetic, and per the table above `pred_gap` is the one number in this model with no
  tolerance for that. The adoption procedure is in §12. A run that compiles says so in its startup
  log, so an eager and a compiled run are distinguishable after the fact.
- **No inter-stream weight sharing and no per-stream front-end settings** (§2).
- **No gap sentinel.** The validity mask is the loader's `weight`, expanded, plus finiteness. The
  pipeline's own sanitisation interpolates non-finite values before writing and segment quality
  filtering requires mean weight at least $0.90$, so a sentinel refinement would guard a case the
  writer does not produce. The finiteness term is kept regardless, because a single NaN would
  otherwise propagate through the low-pass into every following token.
- **The front end owns no z-score**, which is simpler than carrying fixed statistic buffers and is
  safe only because the pre-flight makes the missing case impossible rather than merely unlikely
  (§11).
- **`horizon`, `warmup_period` and the whole geometry are inherited unchanged.** The four-minute
  horizon question is real and separate; mixing it in would make a bad result unattributable.

## 11. Deviation record

Every intentional difference between the built module and the design it was built from.

**Architecture**

- **The anti-alias filter is five taps, and the kernel schedule is $(65, 15, 15, 15)$.** The design
  left both open. Five taps rather than three because $|H(\pi/2)|$ — the response at the point a
  stride-2 decimation folds *onto* — is $0.25$ against $0.5$, and aliased $0.5$–$2$ Hz energy would
  read as fetal heart-rate variability that is not there. Both cost reach, which is the budgeted
  resource, and both fit: $322$ against $480$ (§3.1).
- **The stage projection carries a bias**, the one deviation from the package's otherwise bias-free
  convention, and it is load-bearing rather than incidental (§2.2). $320$ parameters at production.
- **A stage is `Linear` → `GatedCausalConvBlock` → `CausalAntiAliasDecimate`**, reusing the
  sibling's block whole rather than building a gated depthwise convolution here. That is what makes
  "there is exactly one FIR per stage and the residual is sample-aligned" true by construction
  rather than by test, and what makes `init_depthwise_` walk the front ends with no extension.
- **The front end computes its reach from the built modules**, not from the constructor arguments,
  so the number it reports and the number it is refused against cannot disagree with the stack that
  produced them.
- **`raw_per_step` is checked against the front end's own total stride**, at construction and again
  per forward call — the front end has an opinion about the grid it emits, and states it, rather
  than trusting the caller's.
- **The model is a standalone `nn.Module`, not a mode or subclass** of the model it is compared
  against. The constructor schemas genuinely differ: `c_y`, `c_u`, `use_up_st`,
  `causal_reach_budget_s` and the four channel tuples all become meaningless, and a front-end kernel
  schedule appears that the other constructor has never heard of. Absorbing the difference behind a
  flag would leave half the keyword surface dead on every run. What the two share is their
  *objective* and everything under it, imported rather than retyped.
- **The eight inert keys are refused by absence**: there is no `**kwargs`, so `TypeError` names the
  key. The seven **encoder** keys are asserted *live* in the same test file, because the ban and the
  admission are one decision and a sweep that silently caught an encoder key would be a divergence
  between two models that are supposed to differ in one thing.

**Testing**

- **`test_encoder_causality.py` and `test_prefix_equivalence.py` are not mirrored.** Both test
  imported code over $(B, T, d)$ inputs whose behaviour cannot change here, and prefix equivalence
  has no consumer in this package while evaluation is out of scope. The assembled raw-resolution
  probe in `tests/test_causality.py` is the one that catches this package's own wiring.
- **`tests/test_source_purity.py` carries a planted `SwappedModel`**, which the sibling's equivalent
  file has no reason to. Both inputs here are $(B, 4800)$ raw signals, so a transposed argument pair
  produces correctly shaped output, a plausible loss curve, and a source-conditioned KL of the
  target against itself — and it is invisible to every other probe in the file, because a swapped
  model keeps both streams *pure*. Only the forward-pre-hook identity check catches it. In the
  sibling this mistake dies on the first forward against a width check.
- **The AST machinery in `tests/test_ddp_reachability.py` is imported from the sibling**, and its
  own self-tests are deliberately not ported: porting them would test the import.
- **The movement half of every boundary probe asserts against a local $10^{-9}$**, not the shared
  `MOVEMENT_TOL`, and is paired with a whole-step perturbation that does clear the shared tolerance
  (§6). A single threshold cannot fit a quantity that varies by four orders of magnitude across
  anchors.
- **The startup-log assertions do not use `caplog`.** Loguru does not route through the standard
  library's `logging`, so a `caplog.at_level` assertion against these lines passes on a driver that
  logs nothing at all. A loguru sink fixture is used instead.

**Configuration and driver**

- **`configs/default.yaml` is written out in full**, not inherited from the comparison model's. The
  two `VAE_model` blocks do not share a schema and a merge would leave the dead half silently
  dropped by the signature sweep. The cost of that choice is drift, so `tests/test_config_load.py`
  walks every leaf of both files and requires equality outside a six-leaf **input block** —
  `c_y`, `c_u`, `use_up_st`, `causal_reach_budget_s`, `load_fields` and `normalize_fields`, which
  are *excluded* from the comparison rather than exempted from it — against a five-entry exemption
  table, four of them identity: the output tree, the MLflow experiment, the run name and the variant
  tag. Copying any of those four would write these runs into another model's tree. Both sets are
  asserted stale-free in the other direction, and all seven encoder keys get their own explicit
  equality assertion, because *same encoder, different input* is the whole claim.
- **There is no front-end configuration key at all.** Widths are derived from `d_model`, kernels are
  a module constant, and the reach budget is derived from `warmup_period`. §13 lists
  `frontend_kernels` as deliberately absent so the choice reads as a choice.
- **The pre-flight refuses the eight inert keys by name, before the run starts.** "Refuses before
  the inherited width guard" cannot be arranged by ordering — shared `main` calls the four
  module-level guards by name before this hook, and moving them would break three other call sites
  that depend on their being individually callable. It does not need to be:
  `_check_declared_widths_against_shard` returns early unless the config carries **both** `c_y` and
  `c_u`, and a config carrying either is refused here anyway, so a copy-pasted sibling config with
  *correct* widths passes that guard in silence and the operator sees the inert-key message.
  `tests/test_trainer.py::test_the_inherited_width_guard_stays_silent_on_this_packages_configs`
  pins the mechanism.
- **The pre-flight also requires `up` in both `load_fields` and `normalize_fields`.** `fhr` is
  already covered by the inherited `_check_raw_target_normalized`. Without `up` in `load_fields` the
  task fails on the first batch, late, after every rank has initialised; without it in
  `normalize_fields` **nothing fails at all** — the source arrives in raw contraction units, the
  front end feeds that scale straight into the source encoder, and every source-side readout is
  measured at an operating point nobody chose. This guard is the only thing standing between a
  missing entry and a silently unnormalised run, because the front end owns no statistics itself.
- **The shard-side pre-flight applies `trim_minutes` before comparing.** Shards store the untrimmed
  geometry — `fhr` and `up` at $5280$, `weight` at $330$ — and the loader trims
  $\texttt{int}(4 \cdot 60 \cdot \texttt{trim\_minutes})$ samples from **each** end, so
  $5280 - 2 \cdot 240 = 4800 = 300 \cdot 16$. Comparing the stored length directly would fail on
  every real shard. The guard reproduces that one line rather than importing it, because the loader
  computes it in its constructor rather than as a function. Non-fatal on a missing, field-less or
  unreadable shard: the data module reports those better than a pre-flight peek can.
- **`tiny.yaml` shrinks widths only** — `sequence_length`, `raw_per_step`, `warmup_period`,
  `horizon`, the conv stem, both block counts, the source window and `trim_minutes` are all the
  shipped values. `warmup_period` in particular *could not* be shrunk: it **is** the reach budget,
  so a smaller one would build a narrower front end than the production run's and the smoke would
  exercise a stack nobody ships. The smoke model is $2{,}041{,}919$ parameters against the production
  $5{,}081{,}644$, running the same front-end kernels at the same $322/480$ reach.
- **`gradient_clip_val: 5000.0` is carried over, not measured**, and the config says so in the
  comment above it. §12 states the procedure that replaces it.

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

Both also launch from an IDE Run button with no command line: `RUN_CONFIG` near the bottom of
`trainer.py` names the config, `--config` always wins over it, a relative path resolves against the
repository root rather than the working directory, and the entry point `chdir`s there because the
paths *inside* a config are repo-root-relative too.

The gate:

```
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_e2e/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_e2e/tests -q -m slow
```

The smoke fit completes in about $0.15$ minutes on the development box and leaves, under
`output/teb_vae_trf_e2e_tiny/<stamp>-lag_attn_trf_e2e_tiny/`: a checkpoint under this package's own
stem `lag-attn-trf-e2e`, `resolved_config.yaml` beside it, `train_results/metrics_history.csv`
carrying the tracked columns, `train_results/full.log`, `train_results/loss_plot_epoch.html` and the per-epoch
diagnostic figures under `train_results/lag_attn_rws_diagnostics/`. The slow tier runs the same path
at three epochs — three rather than one, because `lr` is logged at train-epoch start so its first
CSV cell is always NaN, and the step warm-up needs more than one epoch to be visibly non-constant.

**Observability during a run**, all inherited and all already wired: the full tracked metric surface
in `metrics_history.csv` (the four objective terms, `pred_gap`, both KL readouts, `prior_rate` and
`beta_prior`, the log-variance diagnostics including `logvar_prior_floor_frac`,
`anchor_coverage_frac`, the latent saturation fractions and the validation-only permutation
control); `train/grad_norm` and `train/grad_clip_frac`; `train/spike_skipped` and
`train/spike_ema_loss`, which are the only columns that can show the skip-forever breaker failure
this repository has already lost a run to; and the per-epoch diagnostic figure.

**Re-deriving `gradient_clip_val`.** The shipped $5000.0$ is the comparison model's measured value
carried over, and the comment above it in `configs/default.yaml` says so. The objective, its
reduction and its units are identical, so the gradient scale should still be dominated by the summed
$480$-sample Gaussian NLL rather than by the front end — but the front end is a new gradient path,
so the value is provisional here rather than measured. The procedure, from the first production run:

1. Read `train/grad_norm` from `train_results/metrics_history.csv`. It is the **pre-clip** global
   norm.
2. Read it as **one optimizer step per epoch**, not as an epoch aggregate: the metrics collector
   reads `callback_metrics` at validation end, before the training epoch is reduced. Percentiles
   over that column are therefore per-step percentiles over a thinned sample, which is the right
   distribution for a per-step threshold.
3. Take $q_{99}$ over at least a few thousand epochs and choose the smallest round value above it.
   For reference, the comparison model's $1018$-epoch baseline gave $q_{50} = 2775$,
   $q_{99} = 4681$, $q_{99.9} = 5866$, maximum $7313$, minimum $703$ — every recorded step exceeded
   the $250$ it had shipped at, so that run performed normalised-gradient descent at roughly an
   eleventh of its configured learning rate.
4. Read `train/grad_clip_frac` as a **mean over epochs** to say how often the threshold actually
   bound. It samples the exceedance indicator the same way.
5. Write the chosen value into `configs/default.yaml` and replace the provisional note with the
   percentiles it came from.

The three-epoch smoke fit logs `train/grad_norm` at $155.2$, $121.9$, $138.0$ (final
`callback_metrics` $130.9$) with `train/grad_clip_frac` at $0.0$ in every row.
`tests/test_train_smoke.py` asserts only that it is finite and non-zero, deliberately: the smoke
config ships `likelihood: mse` and a $32$-wide model, so the number is not the production gradient
scale in either package, and a comparison against the sibling's smoke value could fail a correct
implementation while carrying no information. Finite-and-non-zero catches the two failures that
matter — a NaN path, and a front end receiving no gradient. A stronger form of the second sits
beside it: every front-end tensor is compared against a freshly built model's after the fit, and
none is identical, because a gradient that exists but is never applied would leave every downstream
number looking exactly like a run with a frozen input stage.

**Adopting `torch.compile`.** The key is live here (§10) and ships off. It is not a config edit to
make casually, because inductor may reassociate float arithmetic and `pred_gap` is the one readout
with no tolerance for that. The procedure:

1. On **Linux**. Triton is not installed on the Windows development box, so `compile: true` there
   fails with `TritonMissing` rather than doing anything — the flag cannot be smoke-tested locally
   and must be exercised on the production box.
2. Set `advanced_config.trainer.compile: true` and confirm the startup log carries
   `torch.compile is ON`. A run that did not compile is otherwise indistinguishable from one that
   did.
3. Run **one** epoch each way from the same seed on the same shards, and compare `pred_gap`,
   `source_conditioned_kl_raw` and `nll_full_block` between the two `metrics_history.csv` files.
   The NLLs may move in their last digits; `pred_gap` moving by a noticeable fraction of itself is
   the signal to abandon compilation, because that is the number the model exists to produce.
4. Only then keep it. Note that the first step of a compiled run pays the inductor compile,
   which is minutes and is not a regression.

The key is refused outright alongside `attention_grad_checkpoint`, which genuinely does defeat
inductor; the message names both.

**What to expect of the first run.** The front end is likely to converge more slowly than the
feature model, and that is an expected outcome rather than a defect: the stored features handed the
model a 42-scale decomposition for free, and the encoder must now build long-range structure itself
from $300$ tokens. Reading a slower first result as "raw does not work" is the misreading this
paragraph exists to prevent.

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

Four of those carry more weight here than their presence suggests.
`dataset_config.dataloader_config.dataset_kwargs.load_fields` must carry `fhr`, `up`, `weight` and
`guid` and **none** of the four stored feature blocks — dropping them is the change — while
`fhr_up_ph` stays absent for the reason it always was: a coefficient mixing both signals would
destroy the target-only / source-conditioned separation the design rests on.
`normalize_fields` must carry both `fhr` and `up` (§11). `trim_minutes` must match the stats file's
and is what makes the raw grid $4800$ rather than $5280$. And
`advanced_config.callbacks.lag_attn_rws_plotting` keeps the **inherited driver's literal spelling**:
renaming that block to match this package would silently disable the diagnostic figure with no error
anywhere.

**Deliberately absent**

The input representation this model replaces. Every one names no constructor argument here, so the
signature sweep would drop it without a word and the run would not be the one the operator believes
they launched — so the pre-flight refuses each by name instead:

- `model_config.VAE_model.c_y`
- `model_config.VAE_model.c_u`
- `model_config.VAE_model.use_up_st`
- `model_config.VAE_model.causal_reach_budget_s`
- `model_config.VAE_model.target_keep_index`
- `model_config.VAE_model.target_delays`
- `model_config.VAE_model.source_keep_index`
- `model_config.VAE_model.source_delays`

The front end's own *shape*, which is derived or constant rather than configured (§3.1) — there
is no width key at all, and the kernels are the constructor's default. Its backward reach is the
one thing about it that *is* configured, as `frontend_reach_budget_s` in the required list above:
that bound is what an arm changing the kernels has to move alongside them, and deriving it
silently would make such an arm fail at construction against a number nobody chose.

- `model_config.VAE_model.frontend_kernels`

The encoder this architecture's ancestor replaced, absent two models back and still absent:

- `model_config.VAE_model.lstm_layers`
- `model_config.VAE_model.encoder_extra_dilations`
- `model_config.VAE_model.encoder_extra_kernel`
- `model_config.VAE_model.conv_norm_groups`
- `model_config.VAE_model.causal_norm`

And the ones that are structural, derived or inert — each would read to a maintainer as a control
that exists:

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

The encoder head width is derived as `d_model // encoder_num_heads`; encoder and front-end dropout
are both the existing `dropout`; the LayerScale initialisation and the rotary base are constructor
defaults that no arm varies; the target attention window is the full causal prefix in every arm by
design; an availability embedding is a mechanism of the adapter this model does not have; and
encoder gradient checkpointing sits behind two cheaper memory levers that are already config keys
(`batch_size` with `accumulate_grad_batches`).
