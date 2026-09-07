# `lag_attn_rws` — the as-built design record

The raw-signal lag-attention VAE-TEB: what it is, what it consumes, what it returns, what it
optimises, and every place the built model differs from the design it was built from.

Companion documents: `model_explained.md` in this directory states the architecture and its
motivation; `teb_vae/lag_attn/DESIGN.md` records the feature-target sibling this model is
compared against. Neither is restated here.

---

## 1. What the model is

At every 4-second anchor $t$ the model forecasts the next **two minutes of raw normalized FHR** —
$H \cdot R = 30 \times 16 = 480$ samples — twice: once from a target-only latent and once from a
source-conditioned one. The gap between the two forecasts, and the KL between the two latents, are
the coupling readout.

$$
p_\theta(z_t \mid Y_{\le t}) = \mathcal{N}\!\left(\mu^p_t, \operatorname{diag} e^{\ell^p_t}\right),
\qquad
q_\phi(z_t \mid Y_{\le t}, U_{\le t}) = \mathcal{N}\!\left(\mu^q_t, \operatorname{diag} e^{\ell^q_t}\right)
$$

with $d_z = 64$, one shared decoder $D_\psi$ invoked on $z^p_t$ and on $z^q_t$ and receiving
nothing else, and one noise draw $\epsilon_t$ serving both.

Three properties are enforced structurally rather than by convention. Each has its own test,
named beside it.

| Property | What enforces it | Test |
| --- | --- | --- |
| **No decoder bypass** — gradient reaches the decoder only through $z$ | `BaselineFutureDecoder.forward` takes exactly one tensor; there is no `decoder_state` head and no second decoder | `tests/test_no_bypass.py`, `tests/test_decoder.py` |
| **Source purity** — the source pathway never sees a target tensor and the prior never sees the source | separate adapters and encoders; the posterior is a residual on the prior | `tests/test_source_purity.py` |
| **Exact zero at initialization** — $\mathrm{KL}(q_t \Vert p_t) = 0$ and the two forecasts are bitwise identical, *in train mode* | posterior deltas zeroed **after** `initialization`; one shared $\epsilon$; decoder and attention dropout fixed at $0$ | `tests/test_zero_kl_init.py` |

The last is the one that makes every reported nat of coupling *earned*. It also means any KL
assertion on a freshly constructed model passes vacuously, which is why the `perturb_posterior`
fixture is load-bearing for every test that claims to check KL behaviour.

At the shipped configuration the model holds **5,088,186 parameters** (measured, unguarded), of which
16,512 are the frozen attention output projection `lag_attn.W_o` (§6); the shipped
`causal_reach_budget_s: 120` adds $6{,}272$ for the two availability adapters, giving $5{,}094{,}458$
as built. The count follows from the baseline
architecture bundle in §12: the plain conv stack ($-2{,}048$), per-block FiLM ($+394{,}752$) and
plain residual seams ($-1{,}536$) against the pre-bundle $4{,}697{,}018$; the three init policies
add no parameters. Of the total, $525{,}314$ are the shared decoder's two horizon self-attention
blocks (§12), which `horizon_attention_blocks: 0` removes entirely rather than making inert.

## 2. Input contract

Read from the HDF5 through `train/data_module.py::GraphDataModule`, at `trim_minutes: 1.0`.

| Field | Shape per sample | Role |
| --- | --- | --- |
| `fhr` | $(4800,)$ | **the reconstruction target**, z-scored |
| `fhr_st` | $(300, 43)$ | target scattering |
| `fhr_ph` | $(300, 66)$ | target phase-harmonic |
| `up_st` | $(300, 43)$ | source scattering |
| `up_ph` | $(300, 15)$ | source phase-harmonic |
| `weight` | $(300,)$ | the authoritative validity signal |
| `up`, `guid` | | diagnostic figures only |

So $c_y = 43 + 66 = 109$ and $c_u = 43 + 15 = 58$ (or $15$ under `use_up_st: false`).

`fhr_up_ph` is **never loaded**. A cross-channel coefficient mixes both signals in one number and
would destroy the separation between $p(z_t \mid Y_{\le t})$ and $q(z_t \mid Y_{\le t}, U_{\le t})$
that the whole design rests on. `tests/test_config_load.py` asserts the string appears in no
config at all.

Two contract points fail silently rather than loudly and are therefore guarded at the entry point:

- **`fhr` must be in `normalize_fields`, not only in `load_fields`.** Without it the target
  arrives at ~140 bpm while the decoder's learned log-variance models a z-scale, and the Gaussian
  NLL is meaningless with nothing raising. Guarded by `trainer.py::_check_raw_target_normalized`.
- **Gaps are stored as $0.0$ bpm**, which after z-scoring is roughly $-11\sigma$ — not a
  detectable sentinel. The decimated `weight` is the only trustworthy validity signal, and both
  it and `fhr` are hard requirements in `task.py::_build_raw_target`.

Channel widths are dataset facts, not model constants: they are checked against the first shard
before the fit (`_check_declared_widths_against_shard`) and against every batch at the data
boundary (`task.py::_build_target_streams` / `_build_source_stream`).

## 3. Geometry

`nets/geometry.py::TrimmedRawGeometry`, validated in `__post_init__` so an unvalidated instance
cannot exist. The loader has already applied the crop, so there is **no** `CROP` offset anywhere:

$$
n_{\mathrm{raw}}(t) = 16\,(t+1) - 1, \qquad \mathrm{future\_block\_start}(t) = 16\,(t+1).
$$

$T = 300$, $T_{\mathrm{valid}} = T - H = 270$, trained anchors $[30, 270)$, anchor $0$ forecasts
`fhr[16:496]`, anchor $269$ forecasts `fhr[4320:4800]`.

**The most error-prone point in the build.** A model that loads *untrimmed* data and crops 15
tokens inside the model has $16(t+16) - 1$, whose anchor-0 forecast starts at raw $256$, not $16$
— off by exactly one minute, and nothing downstream fails loudly on the shift. Each formula is
wrong on the other grid. `tests/test_geometry.py` pins $16$ as distinct from both $256$ and $0$.

Masks (`nets/raw_masks.py`) exploit a consequence of there being no crop: future sample
$(t, \tau, r)$ maps to `weight[t + 1 + \tau]` for **every** $r$, so the forecast mask is a
$(B, 270, 30)$ decimated tensor broadcast over $r$, not a raw-resolution stack. The equivalence is
pinned by test against a naive per-raw-index gather rather than assumed.

The KL is masked to the **same** anchor support the reconstruction uses, and "same" is meant
literally: `kl_mask` takes the *forecast mask* and reduces it with `contributing_anchors`, the
one indicator the loss's per-anchor denominator also uses. It is not a second expression over
`weight` that happens to agree. Charging $\beta \cdot \mathrm{KL}$ on an anchor with no
reconstruction term leaves nothing pulling the posterior off the prior, so it is regularised
onto it for free. On the tail 30 anchors that shows up as an end-of-sequence droop resembling
fading coupling; on anchors dropped by a gap or by `coverage_floor` it shows up immediately
before every signal-loss gap, which is the same artifact in the place it is hardest to
recognise. Deriving the support rather than restating it is what forecloses both.

## 4. Forward return dict

`nets/model.py::SeqVaeLagAttnRws.forward(y_st, y_ph, u_stream)` returns exactly twenty keys.
There is deliberately no `decoder_state` and no `delta_mu_src`: neither pathway exists.

| Key | Shape | |
| --- | --- | --- |
| `mu_prior`, `logvar_prior`, `raw_logvar_prior` | $(B, T, d_z)$ | the target-only prior; the raw form is pre-`smooth_bound`, which is what the posterior residual is applied to |
| `mu_post`, `logvar_post` | $(B, T, d_z)$ | the source-conditioned posterior |
| `z_prior`, `z_post` | $(B, T, d_z)$ | paired samples under one $\epsilon$ |
| `target_state`, `source_state` | $(B, T, d_{model})$ | encoder history states |
| `attended_source_heads` | $(B, T, m, d_{head})$ | per-head attended summaries |
| `attn_weights` | $(B, T, m, L)$ | attention probabilities in lag order, $L = 91$ |
| `mu_base`, `logvar_base` | $(B, 270, 30, 16)$ | the target-only raw forecast |
| `mu_full`, `logvar_full` | $(B, 270, 30, 16)$ | the source-conditioned raw forecast |
| `kld_per_t` | $(B, T)$ | per-step KL, summed over $d_z$ |
| `kld_per_t_per_head` | $(B, T, m)$ | its additive per-latent-group split |
| `source_kl_lag_map` | $(B, T, L)$ | the KL attributed across lags |
| `mu_prior_sat_frac`, `delta_mu_sat_frac` | scalar | bound-saturation diagnostics |

Decoding covers the valid anchor range only. The tail $H$ anchors have no fully observed raw
future and the loss would discard them; at this output size the ~10% saved activation memory is a
decision, not an accident.

`source_kl_lag_map` sums over lags to `kld_per_t` **exactly**, and only because the attention
probabilities carry no dropout. That identity is why `LagCrossAttention` is constructed at
`dropout=0.0` — a correctness requirement, not a style choice.

## 5. Loss

`nets/model.py::compute_loss`, with the reductions in `nets/losses.py`:

$$
\mathcal{L} = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0
  + \beta(e)\,\mathrm{KL}_{\mathrm{train}} + \beta_p R_p
  + \lambda_{\mathrm{ms}} \mathcal{L}_{\mathrm{ms}}
  + \lambda_{\mathrm{deriv}} \mathcal{L}_{\mathrm{deriv}}
  + \lambda_{\mathrm{boundary}} \mathcal{L}_{\mathrm{boundary}}, \qquad
  \lambda_{\mathrm{full}} = \lambda_{\mathrm{base}} = 1 .
$$

**Units are the point.** The NLL is *summed* over the 480-sample block and the KL is *summed* over
$d_z$; both are then averaged over batch and contributing anchors only. Both are therefore in nats
per anchor, which is what makes $\beta$ mean anything. `mse` is summed over the horizon too, so
$\beta$ keeps its meaning across likelihoods. A test asserts
$D_{\mathrm{block}} = 480 \cdot D_{\mathrm{sample}}$ and that a masked sample contributes exactly
zero.

**The last three terms are not nats**, so the total is not either — see §5.1. The pure-nats readouts
are `nll_full_block` and `nll_base_block`, and every cross-arm comparison uses those rather than
`total_loss`.

**The fourth term anchors the prior's scale, and nothing else did.**

$$
R_p = \sum_{d} \tfrac{1}{2}\left(e^{\ell^p_{t,d}} - 1 - \ell^p_{t,d}\right)
  = \mathrm{KL}\!\left(\mathcal{N}(\mu^p, \operatorname{diag} e^{\ell^p})
    \,\Vert\, \mathcal{N}(\mu^p, I)\right),
$$

summed over $d_z$, masked by the KL's own anchor support and divided by the same
contributing-anchor count — so it is in nats per anchor and adds to the other three without
rescaling. It is nonnegative, convex in $\ell^p$, and exactly zero at $\sigma_p = 1$. The prior
*mean* does not appear, which is the property that makes it safe to weight: it compresses nothing
$D_0$ depends on.

Without it the incentive on the prior's scale is one-sided. The reconstruction strictly prefers a
deterministic latent, since sampling noise can only degrade a forecast, and $\mathrm{KL}(q \Vert p)$
measures the posterior *against* the prior without constraining the prior's own scale — so unlike a
fixed-prior VAE, in which $\mathrm{KL}(q \Vert \mathcal{N}(0, I))$ anchors it, $\ell^p$ was free to
fall until it met its clamp. It did: the first production run of this objective finished with
`logvar_prior_floor_frac` at $0.992$ against a floor of $-5$, reached inside **one epoch**
($0.118$ at epoch $0$, $0.812$ at epoch $1$), at which point the divergence stops being a rate —
the KL carries $(\mu^q - \mu^p)^2/\sigma_p^2$, so a floored $\sigma_p^2$ multiplies it by an
arbitrary factor.

$\beta_p$ is a **constant, not a schedule**: the collapse completes between epoch $0$ and epoch $1$,
so a warm-up would arrive after the damage. It is also a **threshold rather than a dial**. The
weighted restoring force $\beta_p\,\partial R_p / \partial \ell^p = \tfrac12\beta_p(e^{\ell^p} - 1)$
saturates at $-\tfrac12\beta_p$ per dimension as $\ell^p \to -\infty$, while the reconstruction's opposing
pressure *grows* as the decoder sharpens; below the crossing weight the prior pins anyway, merely
later. Measured on the committed HIE shard: at $\beta_p = 10^{-2}$ the floor fraction still finished
at $0.955$, the collapse delayed $6.7\times$ in optimizer steps and no more, while at $0.1$ it held
at $0.046$ with `kld_active_frac` $0.69$ against $0.24$–$0.25$ in every collapsed arm. The shipped
value is $0.1$; `configs/default.yaml` carries the measurement and its one caveat, that it is one
in-sample shard on this architecture.

$R_p$ is computed and logged **unconditionally**, under every likelihood and whatever $\beta_p$ is
set to, so a collapsing prior is visible in any run's `metrics_history.csv` whether or not that run
opted into paying for it.

The precise claim: $\lambda_{\mathrm{full}} D_1 + \beta\,\mathrm{KL}$ at $\beta = 1$ is the exact
ELBO of the source-conditioned branch. Adding $D_0$ at unit weight doubles the reconstruction
pressure against the KL, so **$\beta = 1$ is a principled starting point, not a distinguished
optimum**.

$\beta$ follows a linear warm-up from exactly $0$. Removing the decoder bypass makes $z$ the only
route to the decoder, and a nonzero $\beta$ before the decoder can use the latent at all is the
standard route to posterior collapse.

Two scale consequences do **not** transfer from the sibling and are carried in config instead:

- **The spike breaker's relative test is off.** `watched > multiplier * max(EMA, ema_floor)`
  assumes a loss bounded below by zero; a learned-variance Gaussian NLL summed over 480 samples
  goes negative, and once the EMA is negative that test degenerates to "skip every positive
  batch". So `ema_floor: 1.0e+9` disables it outright and `additive_margin` — compared against the
  **raw** EMA, so it survives the negative regime — carries finite-spike detection alongside the
  non-finite guard.
- **`gradient_clip_val` was measured; `additive_margin` never was; both are provisional at this
  capacity.** Both began as
  values scaled from the sibling rather than derived. The clip has since been re-derived from the
  first production run of this objective, which logged `train/grad_norm` pre-clip:
  $q_{50} = 2775$, $q_{99} = 4681$, $q_{99.9} = 5866$, maximum $7313$, minimum $703$, and
  **every** recorded step above the old $250$ — the clip was rescaling every step, so the run
  trained at roughly a eleventh of its configured learning rate. It now ships at $5000$, the
  smallest round value above $q_{99}$. Note the CSV records one step per epoch for this metric
  rather than the epoch's aggregate, so those are per-step percentiles over a thinned sample;
  `train/grad_clip_frac` is the same sample of the exceedance indicator and is read as a mean over
  epochs. `additive_margin` remains scaled rather than measured; that run recorded
  zero skips and a maximum loss-to-EMA distance of $669$ nats against its $1000$ nat margin, which
  bounds it as loose-but-working rather than deriving it.

  **The capacity revision made both provisional again**, and `configs/default.yaml` marks them so:
  that run predates the wider latent, the wider and deeper decoder, its horizon attention and the
  three shape terms, each of which moves the gradient the threshold is set against, and
  `additive_margin` now watches a mixed-unit `main_loss`. Both ship unchanged — a measured value
  carried across a scale change is a known quantity and a guessed rescaling is not — and both are
  re-derived from the first run at this geometry.

### 5.1 The auxiliary shape terms

The last three terms of §5's criterion price the *shape* of the forecast mean, which the
factorized Gaussian NLL cannot: it scores every raw sample independently, so its optimum is the
conditional mean and a fully parallel block decoder is free to emit an over-smoothed one. Each is
computed on `mu_base` and `mu_full` and **summed** over the two, under the module's own masks, and
reduced by the **same**
contributing-anchor count as $D_k$ so their per-anchor scale is comparable to the reconstruction's.
Summing over the branches is the convention $\lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0$
already uses at unit weights, so one shape weight prices both forecasts equally.

| Term | What it prices | Definition |
| --- | --- | --- |
| $\mathcal{L}_{\mathrm{ms}}$ | the block's envelope | mask-weighted $L_1$ between forecast mean and target, average-pooled over the flattened $H \cdot R$ block at `MS_RATES = (1, 4, 16)` |
| $\mathcal{L}_{\mathrm{deriv}}$ | slope, which an over-smoothed mean loses first | Huber at $\delta = 1$ between the first differences of the two, a pair valid only when both its samples are |
| $\mathcal{L}_{\mathrm{boundary}}$ | the starting level | $\lvert \hat\mu_t[0] - Y[n_t] \rvert$: the first forecast sample against the anchor's last observed one |

Three implementation points are load-bearing rather than incidental:

- **The multiscale mask is applied before pooling**, not after. Pooling mixes neighbours, so a gap
  left in until afterwards leaks its sentinel into every pool it touches. This is the one place the
  module's multiplicative-mask convention runs early.
- **The boundary term is a slicing identity**, `mu[:, 1:, 0, 0]` against
  `target[:, :-1, 0, -1]`, exact because $X = R = D$ on the raw grid — so no new tensor is plumbed
  into the free objective. It is computed over $t \in [1, T_{\mathrm{valid}})$ *structurally*, so
  anchor $0$ is excluded by construction rather than by assuming a warm-up. Its validity is anchor
  $t$'s own `weight` at threshold times $t$'s contributing indicator, deliberately **not** anchor
  $t-1$'s forecast mask, which would import a different anchor's coverage-floor decision.
- **A term whose weight is $0.0$ is not computed**, and its metric is reported as exact $0.0$ rather
  than as the value it would have had. That keeps a term-off arm's CSV honest, keeps the feature-target
  siblings' columns from carrying raw-domain formulas evaluated over a channel axis, and keeps the
  full-block intermediates out of the graph of any run with a term off. The branch reads a
  config-constant float, identical on every rank and every batch, so DDP graph identity is untouched.

The weights ship at $0.1 / 0.1 / 0.05$, all three **provisional**: no run has measured their
magnitudes at this scale. The boundary weight is half the others because it constrains **one** sample
per anchor against $480$ for the other two; at parity it would out-weigh them per-sample by that
factor.

Three `lean-limit` notes belong to these terms — the provisional weights, `MS_RATES` as a constant,
and the missing slope variant of the boundary condition — and are recorded in §11 with the rest.

## 6. Structural constraints that are not preferences

| Setting | Why it is required | Where |
| --- | --- | --- |
| decoder `dropout=0.0` | one module invoked twice draws two independent masks: base and full would differ at initialization even with $z^p = z^q$, and independent noise would enter `pred_gap` every step | `nets/model.py`, decoder construction |
| attention `dropout=0.0` | dropout is applied to the probabilities *before* they are returned, breaking the KL/lag-map identity | `nets/model.py`, attention construction |
| `lag_attn.W_o` frozen | the head-structured posterior consumes the per-head summaries, so `W_o` receives no gradient; freezing drops it from DDP's expectation set | `nets/model.py`, and `select_ddp_strategy` |
| `causal_norm: true` | a non-causal `GroupNorm` pools statistics across time, so the "prior" conditions on the future and the KL is not a coupling readout at all | `configs/default.yaml`; `create_model` warns loudly when off |
| `compile: false` | the LSTM encoders and the checkpointed attention region each break TorchInductor independently, so `compile_model_requested` refuses the key outright rather than reading it | `configs/default.yaml`; `trainer.py::compile_model_requested`, and `task.py` where `compile_model` **defaults** to `False` |
| `num_sanity_val_steps: 0` | `MetricsLoggingCallback` has no sanity guard; a sanity pass shifts every epoch number against MLflow and the checkpoint filenames | `configs/default.yaml` |
| posterior deltas zeroed **after** `initialization` | `initialization` xavier-fills every `nn.Linear` and would otherwise undo the zeroing — and with it the exact zero-KL start | `nets/model.py::_zero_init_delta_heads` |

## 7. The causal input guard

The stored features are two-sided wavelet transforms: a feature step reads raw signal from its own
future, up to 974 s of it. `causal_reach_budget_s` bounds that.

Resolution (`channel_reach.py`) keeps channel $c$ when $\mathrm{reach}_c \le \mathrm{budget}$ —
inclusively — and reads each survivor at $t - \delta_c$ with
$\delta_c = \lceil \mathrm{reach}_c / 4 \rceil$ steps, applied by `nets/delays.py::ChannelDelay`.
Per channel rather than one uniform band because the two dominate very differently at equal
guarantee: at 120 s the fastest survivors are one step stale where a uniform band would make every
channel 30 steps stale.

| budget | $c_y$ | $c_u$ | max $\delta$ | per-block survivors |
| --- | --- | --- | --- | --- |
| `null` | 109 | 58 | 0 | everything |
| 240 s | 94 | 43 | 57 | — |
| **120 s** | **78** | **29** | **30** | `fhr_st` 27/43, `fhr_ph` 51/66, `up_st` 27/43, `up_ph` 2/15 |
| 60 s | 59 | 23 | 14 | `up_ph` 0/15 |
| 32 s | 43 | 19 | 8 | `up_ph` 0/15 |

Regenerate with `python -m teb_vae.lag_attn.channel_reach`.

Design points worth knowing before changing anything here:

- **The model is built with the full declared $c_y$ / $c_u$** and gathers survivors *inside* the
  forward, after the data boundary and before the input adapters. Otherwise the task's width check
  and the shard-width pre-flight would both reject any nonzero budget.
- **The guard is one object per stream** — `ChannelGate`, holding the keep-index and its
  `ChannelDelay`. The two are meaningless apart, because the delay vector is positional *against*
  the keep-index, so anything holding one must hold the other. The model exposes
  `target_gate` / `source_gate` (`None` when unguarded) and the derived
  `source_delay_steps`, which is the single accessor every lag consumer reads (§8).
- **The delay must fit inside the warm-up.** The first $\max_c \delta_c$ steps of a delayed stream
  are partly zero-filled, so they must fall inside the steps the loss already discards. The
  comparison is strictly greater-than: at 120 s the maximum delay is exactly 30, which is also the
  shipped `warmup_period`, and that configuration must be allowed.
- **Both gate buffers are non-persistent.** Their length is the surviving-channel count, so a
  persistent copy would make a checkpoint trained at one budget fail to load at another —
  surfacing as "checkpoint keys did not align" rather than as anything about budgets. The four
  resolved tuples do live in `model_kwargs`, because the adapters' widths depend on them and a
  checkpoint recording only the budget in seconds could not be rebuilt without re-running the
  resolution.
- **Arms at different budgets cannot share checkpoints.** Different adapter input widths.
- **The unguarded default builds no gather and no delay at all**, so it is structurally the model
  that existed before the guard did — the clean architecture comparison against the sibling.

**No finite budget is currently trainable.** Measured on the tiny fixture in float64, the global
gradient norm is $\approx 1 \times 10^{26}$ at every budget (32 / 60 / 120 / 240 s) against
$\approx 98$ unguarded, and overflows fp32 to $\infty$. The mechanism: at step $0$ *every*
surviving channel is zero-filled — the fastest survivor is already one step stale — so the input
adapter's norm and each causal conv pre-norm receive a zero-variance vector, and the
$1/\sqrt{\epsilon} = 316\times$ backward amplification of the $\approx 10$ stacked norms compounds.
It is a switch, not a gradient: the magnitude does **not** scale with $\max_c \delta_c$, so raising
`warmup_period` does not fix it. With the shipped `gradient_clip_val: 5000` the clip coefficient
is $\approx 5 \times 10^{-23}$ — it was $\approx 10^{-24}$ at the old $250$, and the re-derivation
moves the number without touching the conclusion, since a $20\times$ larger threshold against a
$10^{26}$ norm is still annihilation — so a reach arm completes normally having optimised nothing
but AdamW's weight decay. `tests/test_train_smoke.py::test_the_guarded_runs_gradient_stays_finite` is a
`strict` xfail recording this; it flips to a failure when the defect is fixed. The unguarded
default builds no gate and is unaffected.

**The guard bounds the leak; it does not remove it.** $L_{95}$ is an energy *quantile*, not a
support: 5% of every filter's energy lies beyond its stated reach. Measured against a severe
perturbation (`tests/test_causal_leak.py`), the 120 s budget suppresses the movement of the read
features by roughly a factor of 20 — from ≈12–17 channel spreads to ≈0.5–0.8 — rather than to
numerical noise. The residual is larger on the phase-harmonic block than on the scattering block,
consistent with a phase coefficient normalising by its own envelope and so amplifying exactly the
low-energy tail the quantile discounts. Only genuinely causal transforms remove the leak.

## 8. Figure and interpretation traps

Three ways to read a correct number wrongly. Each has cost a result in this repository's history
or is one step away from doing so.

**There is one lag quantity, and the stored timeline is canonical.** `nets/lag_report.py`
computes it from the attention lag index $\ell$:

$$
\tau_{\mathrm{compensated}} = 4\,(\ell + \delta).
$$

The stored UP/FHR timeline is canonical: the dataset builder shifts the UP channel when it writes the shards, that shift is part of how the stored signals are, and nothing downstream adds it back, subtracts it, budgets it or interprets it. An earlier revision of `lag_report.py` carried a
`MECHANICAL_SHIFT_SECONDS = 20` constant and a `lag_original_sensor_seconds` helper that
"undid" the builder's shift; both were removed on purpose (2026-09-05) and must not return
under another name. Figure axes use `COMPENSATED_LAG_AXIS_LABEL` so a plot and the number
beside it cannot disagree about what is shown.

The $\delta$ term is the causal input delay, and **every consumer reads it from one place**:
`SeqVaeLagAttnRws.source_delay_steps`. Both the training figure and the evaluation take it from
there, because the model is what was trained. That single accessor exists because the alternative
was tried and failed silently: the plotting callback and the evaluation each probed the model's
internals for a delay under different names, and the two reports of the same run disagreed by up
to 30 steps — two minutes — with nothing raising. `tests/test_lag_consistency.py` pins them
together. With per-channel delays there is no single $\delta$, so the **maximum** over source
channels is used, making the reported lag an upper bound; recorded as
`source_delay_is_max_over_channels` in `summary.json` beside the number it produced.

**Only the unfloored KL may be read as an information rate.** `source_conditioned_kl_raw` is the
readout; `source_conditioned_kl_train` has free bits applied and is the quantity the optimiser
sees. Watching the floored one hides a collapsed source pathway. The progress bar and every
interpretive claim use the raw one. (The shipped `free_bits` is $0.0$, so they currently coincide
— which is exactly why the distinction has to be documented rather than observed.)

**A prior variance on its clamp inflates the coupling readout while everything else looks
healthy.** The KL carries $(\mu^q - \mu^p)^2 / \sigma_p^2$, and removing the decoder bypass puts
direct downward pressure on $\sigma_p^2$. `mean_logvar_full` and `mean_logvar_base` are *decoder
output* variances and do not detect this. `mean_logvar_prior` and `logvar_prior_floor_frac` do, and
are logged from epoch 0.

## 9. Naming

The KL is `source_conditioned_kl_raw` / `_train`, and its decomposition is `source_kl_lag_map`. It
is **not** called transfer entropy. The inputs are not causal (§7), and the label would assert a
property the measurement in §10 shows the data does not have. `pred_gap` is $D_0 - D_1$.

## 10. Measured evidence (cited, not copied)

Cited so there is one source of truth and the numbers cannot drift.

- **Forward reach of the production filter bank.**
  `teb_vae/lag_attn/eval/representation_capacity_probe.py` — $\phi$ is essentially causal at
  8.75 s, 27 of 43 `fhr_st` channels stay within the 120 s horizon, and the lowest-frequency
  channel reads 965 s into its own future. `tests/test_channel_reach.py` checks this module's
  reaches against those pinned figures rather than against itself.
- **Causal normalization.** The sibling's `DESIGN.md` §7 records an 11.5% relative future leak
  through a non-causal `GroupNorm`, falling to 0.0 with the causal replacement.
  `tests/test_causal_encoder.py` re-measures it here, in both directions.
- **The measured effect of the channel delay.** `tests/test_causal_leak.py`, summarised in §7.

## 11. Deliberate limitations

Each `lean-limit` annotation lives where the engineer who would trip over it is already reading.

| Where | Limitation |
| --- | --- |
| `nets/losses.py`, beside the KL | the KL is a provisional source-conditioned rate, not transfer entropy |
| `nets/losses.py`, beside the Gaussian NLL | a factorized Gaussian estimates TE within that model family, not the data-generating TE |
| `nets/__init__.py` | shared primitives are imported from the sibling rather than promoted to a common package |
| `nets/model.py`, beside `compute_loss` | all valid anchors are decoded every batch; no anchor subsampling |
| `channel_reach.py` | the reach bound is an energy quantile, not a hard support |

And here, because they have no single home in the code:

> lean-limit: the auxiliary shape weights $0.1 / 0.1 / 0.05$ (§5.1) are provisional single points;
> replace with values re-derived from the per-term magnitudes when the first production run at this
> geometry has written its `aux_*` columns.

> lean-limit: `MS_RATES = (1, 4, 16)` is a module constant rather than configuration; replace with a
> config key when a sweep over pooling rates is actually wanted.

> lean-limit: the boundary term uses the level identity only; replace with the slope variant of
> `model_explained.md` 15.3 when the derivative metric shows the transition shape is still wrong at
> converged weights.

> lean-limit: the context-sufficiency gap is **measured, as an estimate**, by the evaluation's
> oracle probe (`eval/oracle.py`, reported by the `sufficiency` analysis): a decoder of the same
> capacity reading `target_state` instead of $z$, fitted on half the evaluation recordings and
> scored on the other half, gives $\Delta_{\mathrm{suff}} = D_{\mathrm{base}} - D_{\mathrm{oracle}}$
> — what the latent bottleneck costs the forecast. What remains unmeasured is the **direction of
> its error**: conditioning on `target_state` rather than on the raw FHR history omits the
> encoder's own information loss and biases the gap down, while fitting the probe on the evaluation
> population when $D_{\mathrm{base}}$ comes from a model trained on the disjoint, healthier
> pretraining cohort biases it up by a domain shift the probe does not suffer. The two oppose and
> neither is quantified, so the number is an estimate and not a bound. Replace with a one-sided
> bound by fitting a second probe on a held-out split of the *training* cohort, which removes the
> second bias.

> lean-limit: the prior anchor $R_p$ of §5 is the **scale half** of the context rate
> $\mathrm{KL}(p_\theta \Vert \mathcal{N}(0, I))$, not the whole term. The two halves separate
> exactly — $\mathrm{KL}(p \Vert \mathcal{N}(0,I)) = R_p + \tfrac12\sum_d (\mu^p_{t,d})^2$ — and
> only the first was adopted, because the second compresses the prior *mean*, which is the entire
> content of the base forecast: weighting it trades $D_0$ away to buy the fix the collapse needed,
> and turns a scale pathology into a target-side rate–distortion question about how much of the
> target's predictive state the latent is allowed to carry. That is a different study, not a
> constant. Replace with the full context rate when a measured target-side rate budget exists to
> set its weight against — the scale half is a strict subset, so adopting it later is an addition
> rather than a revision. The half that ships is also the half the failure demanded: the collapse
> was in $\sigma_p$, at a mean the run had no complaint about.

> lean-limit: the initialisation-policy defaults (`horizon_embed_std`, `head_init_calibration`,
> `a_head_gain`) are backed by synthetic-data training probes only — except the prior-scale half
> of the calibration, which answers a prior-variance collapse measured in production; confirm or
> revise them from the `sweep_init_off.yaml` arm run against the new baseline. Each is
> config-reversible without a code change.

> lean-limit: `horizon_embed_std: 0.8` was calibrated at $d_z = 48$ (projected-latent RMS
> $\approx 0.83$) and the shipped latent is now $64$; by xavier arithmetic the dz-sweep arms drift
> ($\approx 0.56$ at dz $24$ to $\approx 0.82$ at the shipped $64$ and lower again at $96$), which
> symmetry breaking tolerates. Replace with a
> measured-at-construction match when a dz arm's results show the mis-scale matters; the drift is
> recorded in `RESULTS.md` rather than silently inherited.

Also deliberate, and not annotated because nothing in the code would prompt the question:

- The attention query is a projection of the prior belief. By default that is $\mu^p_t$ alone;
  the `query_uses_logvar` key widens it to $[\mu^p_t \,\|\, \ell^p_t]$ (`query_proj` in-width
  $2 d_z$), so an arm can A/B whether the query benefits from also reading the prior's certainty.
  Both inputs are target-only, so source purity holds either way.
- `lag_band_mask` is **not** a forward parameter. In `LagCrossAttention` a band mask *replaces*
  rather than intersects the causal-validity mask, so a naive band mask would admit lags that do
  not exist, and an all-masked row makes `entmax15` raise.

### What the evaluation closed, and what it deliberately did not

The evaluation is complete: fifteen analyses, two durable per-run tables, an offline acceptance
gate and the arm tables that fill `RESULTS.md`. `eval/EVAL.md` is its contract and
`eval/FIGURE_GUIDE.md` documents every figure; both are bound to the code by test.

What it **closed**, each having been a standing gap in this section:

- The context-sufficiency gap is measured, as an estimate — the bullet above.
- The prior-variance inflation is detected and FAIL-able (`prior_variance_not_pinned`), where
  before the model computed the detectors and nothing read them.
- The lag readout is reported unbiased: per-lag support corrected, truncated support accounted
  for against the attainable entropy ceiling, and per head rather than head-averaged.
- The observation model's calibration is checked, so a block score is a log density of something.
- The forecast is scored against trivial baselines, in bpm, and resolved by horizon step.
- The coupling readout carries uncertainty on per-recording units, with class and subgroup cuts
  and trajectories against time before delivery.
- The raw target is exploited: deceleration forecast skill, contraction-triggered response, and
  contraction-conditioned coupling.

What it deliberately does **not** do, so each is a known absence rather than an oversight:

- **Lag ablation** (band-restricted sufficiency ranking) is not implemented, for the model-side
  reason stated directly above: a band mask replaces rather than intersects the causal-validity
  mask, and `lag_band_mask` is not a forward parameter. Replace with the full ranking when
  `LagCrossAttention` gains an intersecting band-mask forward argument.
- **Necessity is not measured anywhere.** The sibling's band ranking measures *sufficiency* and
  reads exactly backwards if taken for a removal ablation, so this pipeline emits neither rather
  than emitting the one that is routinely misread.
- **The oracle conditions on `target_state`, not on the raw target history**, which is what makes
  the sufficiency number an estimate rather than a bound; the bullet above states both bias
  directions and the path to a one-sided bound.
- **Spectral analysis below the window's own resolution, and time-resolved coherence.** The
  deferral's reason is now *answered* rather than overridden. It was: at $f_s = 4$ Hz with
  `nperseg = 64` the band $[0, 0.04)$ contains only the DC bin that `detrend='constant'` has already
  removed, so the most clinically interesting band is the one the geometry cannot deliver. The
  $\tau$-slice construction defeats it — fixing a horizon step and concatenating over consecutive
  anchors yields a contiguous $960$ s series per lead time, so `nperseg = 512` gives
  $\Delta f = 7.8$ mHz and four bins below $0.03$ Hz — and the `coherence` analysis reports
  magnitude-squared coherence, spectral gain, cross-spectral phase and an exact three-way split of
  the residual spectrum on it, resolved by frequency and by lead time. Three absences remain and are
  deliberate. (i) **Below $\Delta f$**: nothing under $7.8$ mHz is resolved, so a $0.003$ Hz VLF
  floor is unreachable — a 20-minute segment cannot deliver it at any window length. (ii)
  **Non-stationary coherence** (STFT or wavelet *within* a recording): every estimate here pools
  windows, so a coherence that varies through a recording reads as a lower constant one. (iii)
  **Absolute band power against a clinical norm**: what is measured is truth-versus-forecast
  agreement, not the trace's own spectral content.
- **No distributional distances between per-class latent populations** (FID via `sqrtm`, MMD-RBF).
  The checkpoint has never seen ACIDOSIS or HIE, so a distance between class-conditional latent
  distributions measures input distribution shift rather than anything the model learned. Revisit
  only if a classification-cohort checkpoint is trained.
- **No held-out clinical discrimination.** The pretraining cohort is healthy-only, so every class
  contrast is out-of-distribution and is declared as one. Building a classifier on the latent is a
  different project.

## 12. Deviation record

Every intentional difference between the built module and the design it was built from.

**Architecture and configuration**

- **`sigma_obs`, `head_structured_latent` and `freeze_unused_attn_proj` are not config keys.** All
  three are unconditional in the net: the learned observation variance, the head-structured
  posterior and the frozen attention projection are structural facts, and a key would read to a
  maintainer as a control that exists. `tests/test_config_load.py` asserts their absence.
- **`select_ddp_strategy` keys on the likelihood alone**, and returns a configured `DDPStrategy`
  rather than a `'ddp'` / `'ddp_find_unused_parameters_true'` shorthand string. `W_o` is frozen
  unconditionally by the constructor, so it is never in the reducer's expectation set; the only
  config-dependent starvation left is the decoder log-variance heads under `likelihood: mse`, and
  `find_unused_parameters` is the only entry in `ddp_kwargs` that reads config at all.

  The instance replaces the string because the strings can carry that one setting and none of the
  others: `broadcast_buffers=False`, since DDP re-broadcasts every buffer from rank $0$ on each
  forward and every buffer in this family is a deterministic function of the config — safe only
  because there is no `BatchNorm` anywhere, a running statistic being the one buffer that genuinely
  diverges per rank; and `gradient_as_bucket_view=True`, which points `param.grad` at the reduction
  bucket instead of a separate allocation. **`static_graph` is deliberately absent**: it promises an
  identical autograd graph every iteration, and the loss-spike breaker substitutes a zero-weighted
  sum over every parameter on a skipped batch, which is a structurally different backward from the
  one iteration $1$ recorded. `tests/test_ddp_strategy.py` carries that as a negative control.
- **`tiny.yaml` ships `likelihood: mse`.** A deliberate delta, not a simplification: it is the
  configuration that starves those heads, so the smoke path exercises the DDP fallback where it is
  cheap to observe.
- **`FullLatentPriorHead` is written fresh** rather than reusing the sibling's `PriorHead` and
  discarding its `decoder_state` output, which would leave dead parameters that a DDP run must
  then be told to tolerate.
- **No decoder class was written.** `BaselineFutureDecoder` already takes one shared core and one
  input tensor; at `d_model = d_z` and `out_channels = 16` it emits $(B, T, 30, 16)$ directly, and
  its single-tensor `forward` is what *forbids* the bypass.
- **The encoder conv stack is a single plain residual, not the sibling's double residual.** Each
  `CausalMultiChannelConvBlock` is already a pre-norm residual (`return output + residual`); the
  sibling then adds a *second*, `GroupNorm`-rescaled copy of the stream at every inter-block seam,
  a constant-magnitude injection onto an un-renormalised, growing stream. Measured on the
  constructed model: that double form inflated the conv-stack activation RMS $\approx 4.27\times$
  across the five shipped blocks and diluted the input skip $x_{\mathrm{lin}}$ to $\approx 18\%$ of
  the stack output; the plain stack holds the growth to $\approx 1.41\times$ and restores the skip
  to $\approx 53\%$, so deepening the encoder now improves conditioning instead of fighting itself.
  Built through the shared `CausalConvLstmEncoder(stack_skip_connection=False)` at both encoders;
  the shared-class default keeps the sibling's double stack, so the comparison sibling is bitwise
  unchanged. One residual point is documented but *not* acted on: the outer `+ x_lin` at the stack
  exit still double-counts the input skip once the inner term is gone (measured cos $0.818$
  alignment, $\approx 2{:}1$ weight), but the review found the downstream fusion learns around it,
  so it is left as-is. Consequence for the norm count: with the four inter-block skip `GroupNorm`s
  per encoder gone at the shipped five-block geometry, `n_causalized_norms` drops by $8$ ($2{,}048$
  parameters).
- **The horizon core re-injects the latent via zero-init FiLM at *every* refine block, not once.**
  Measured on the constructed core: the learned per-token step embedding is $\approx 2.4\%$ the
  magnitude of the broadcast latent, so the $30$ horizon tokens enter the refine stack
  $\approx 97.6\%$ identical, and the single-FiLM core leaves the post-refine tokens
  $\approx 0.986$ correlated -- room for the stack to synthesise the trajectory shape in a
  $z$-independent direction and use the latent only for a coarse offset. Per-block FiLM
  (`film_per_block=True` on the shared `HorizonDecoderCore`, with `horizon_film` still feeding
  `film`) makes every block read the projected latent $h$, the only $z$ entry point, so the
  decoder still consumes nothing but the latent. $+394{,}752$ parameters at the shipped decoder
  geometry (four $\text{Linear}(256, 512)$ generators replacing one). The generators are
  zero-initialised for an identity at init, but `initialization` xavier-refills every `nn.Linear`
  afterwards and undoes it -- so the identity is only *actually* true because the model re-zeros
  the generators in its post-init block, unconditionally. This is not cosmetic: the shipped
  single-FiLM sibling was measured running FiLM *random* at init ($\gamma$ RMS $0.68$, $8\%$
  negative gains). The bitwise base $=$ full at init survives regardless, because it derives from
  $z^p = z^q$ under one $\epsilon$, not from the FiLM structure. A one-off z-uptake probe ($80$
  steps on a temporally structured synthetic target, $\beta = 0$) showed the $z$-driven decoder
  variance fraction holding at $0.393 \to 0.328$ (no collapse) while the per-block FiLM weights
  moved off zero (RMS $0 \to 0.007$). One rollback asymmetry is deliberate: reverting the per-block
  construction alone would *not* restore the historical random-FiLM init, because the re-zero is
  unconditional.
- **The horizon core mixes its $H$ tokens with self-attention after the refine stack.** The dilated
  convolutions reach the whole block — at `horizon_depth: 4` the receptive field is $31$ over $30$
  tokens — but only through a chain of fixed local windows with a schedule set at construction. Two
  pre-norm bidirectional blocks (`horizon_attention_blocks: 2`, $4$ heads, `attention_heads` a
  constructor argument with no config key because no arm varies it) mix all $30$ at once with
  content-dependent weights, so a forecast is shaped as a whole rather than assembled from
  overlapping neighbourhoods. Symmetric attention is legitimate on this axis for the same reason the
  refine stack pads symmetrically: every horizon step is predicted from the same anchor $t$, so
  there is no future *there* to leak. $+525{,}314$ parameters at the shipped decoder width —
  $4 d_{\mathrm{hidden}}^2 + 2 d_{\mathrm{hidden}} + 1$ per block, four bias-free projections, one
  `LayerNorm` and one scalar residual gain. Three details are correctness rather than taste: the
  block is hand-rolled rather than `nn.MultiheadAttention`, whose attention dropout is functional and
  therefore invisible to the dropout-zero scans that guard the twice-invoked decoder and whose packed
  `in_proj_weight` the generic `initialization` pass would xavier-fill at three times the true
  fan-in; the residual gain is a bare `nn.Parameter` at $10^{-2}$, which the generic pass ignores, so
  the stack starts near-identity but every projection carries gradient from step $0$; and
  `attention_blocks: 0` constructs **no module**, so a core that does not ask for them is
  parameter-for-parameter and bitwise the core that existed before they did. There is no positional
  encoding of its own — `horizon_embedding` is already the tokens' identity.
- **The adapter, front and fusion seams carry no post-residual activation.** Each of those seams is
  a `ResidualMLP` that ended in a normalise + GELU; the GELU gated the backward gradient through the
  seam (measured $\approx \times 0.39$ stacked across the three, restored to $\approx \times 1.0$
  once removed) and left a fraction of the exported units persistently compressed on the forward
  side ($27\%$ of the fusion-exit units had mean pre-activation $< -0.5$ and $11\%$ were
  persistently compressed before the change; removing the gate raised the fusion-exit effective
  rank by $\approx 9\%$). A three-seed synthetic A/B improved by $\approx 7.5\%$ mid-training and
  $\approx 2.6\%$ at convergence. Built with `final_activation=False` on all four seams (two
  `InputAdapter.res_mlp`, two `CausalConvLstmEncoder.front_mlp`/`fusion`) via the shared-class flag
  `post_residual_activation=False`; the shared-class default keeps the sibling's gated seams. Each
  affected seam drops its final `LayerNorm`, $-256$ parameters at $d_{model} = 128$, $-1{,}536$
  across the two adapters and two encoders.

**Initialisation policies (zero-parameter, config-reversible)**

Three post-init re-initialisations, applied in the rws post-init block after the generic
`initialization` xavier-fills. Each is a config key (`horizon_embed_std`, `head_init_calibration`,
`a_head_gain`) that the `sweep_init_off.yaml` arm reverts together, and each default is an exact
no-op — so a default-flag model is bitwise the pre-policy one, and the constructor gates the
re-init on the value leaving its default. They add no parameters. The evidence is from synthetic
training probes only (recorded in §11 as a `lean-limit`); the init-off arm isolates the bundle on
production data.

- **The horizon-step embedding is re-seeded at a larger scale.** The core seeds
  `horizon_embedding` at $\mathcal{N}(0, 0.02^2)$, $\approx 2.4\%$ the magnitude of the broadcast
  projected latent, so the $30$ horizon tokens enter the refine stack $\approx 0.999$ correlated at
  init and per-block FiLM has almost no token-specific structure to modulate (token-specific
  variance fraction $\approx 0.06\%$). Re-seeding at `horizon_embed_std` $= 0.8$ drops the
  horizon-token correlation to $\approx 0.45$ and raises the token-specific variance fraction to
  $\approx 48\%$, giving per-block FiLM distinct per-step offsets to shape from step $0$; a
  synthetic A/B reached $z$-informed forecasting $2$–$3\times$ sooner (the quantity racing the
  $50$-epoch $\beta$ warm-up against posterior collapse). The posterior deltas are untouched, so
  the exact zero-KL start is preserved.
- **The shared decoder's output heads are calibrated onto the trivial predictor.** Xavier-filled
  heads emit a high-variance mean and an over-confident low log-variance, so the init factorized
  Gaussian NLL of the raw z-scored target sits far above the trivial $\mu = 0, \sigma = 1$
  predictor's (measured at the shipped geometry: $\approx 15.8$ vs the trivial $\approx 1.42$
  nats/raw-sample under the model's own per-sample convention, with a log-variance tail as low as
  $q_{05} \approx -4.3$). `head_init_calibration` shrinks the mean head by $0.02$ so
  $\hat\mu \approx 0$ (scaled, not zeroed, so a perturbed posterior still moves the two forecasts
  apart — `test_zero_kl_init` depends on it), sets the log-variance bias to $\log(5/3)$ so
  `smooth_bound(-5, 3)` maps it to exactly $0$ ($\sigma = 1$; $\mathrm{sigmoid}(\log(5/3)) = 5/8$,
  $-5 + 8 \cdot 5/8 = 0$), and shrinks the log-variance weight by $0.1$; the init NLL then lands at
  $\approx 1.44$, within $\approx 1.5\%$ of the trivial predictor, log-variance centred at $0$.
  Base and full share this one decoder, so both stay calibrated identically and every
  bitwise-at-init contract holds.
- **The posterior fusion's attended-source norm carries a gain.** The head-structured posterior
  fuses a $d_{model}$-wide target state with a $d_{head}$-wide attended source summary; at unit
  gain the summary is out-columned $128 : 32$ and (measured) explains only $9$–$20\%$ of the fused
  representation's variance, receives $2$–$4.5\times$ less gradient than `h_y` while the KL is open,
  and moves the source-carrying delta $2$–$9\times$ less than the target state does.
  `a_head_gain` $= 2.0 = \sqrt{d_{model}/d_{head}}$ rescales the summary up so the two inputs enter
  the fusion at comparable magnitude, halving the imbalance. The deltas are still zero at init, so
  the KL is exactly zero.
- **The prior head's log-variance starts at unit scale, under the same calibration key.** The
  same policy as the decoder calibration — start every distribution head at the trivial
  $\mathcal{N}(0, I)$ predictor — so it lives under `head_init_calibration` rather than a fourth
  key, which also keeps the uncalibrated prior start (the configuration in which the prior-variance
  collapse was measured, mean log-variance $\approx -3.081$ at epoch $0$) expressible by config and
  keeps `sweep_init_off.yaml` a whole-bundle revert. The mechanism is the posterior deltas' own
  zero-weight recipe rather than the decoder's shrink-plus-bias, because the head is a
  `ResidualMLP` returning `body(x) + skip_proj(x)` with no single bias governing its output level:
  the final body `Linear`'s weight and the whole skip projection are zeroed and the final bias set
  to the pre-image of log-variance $0$ under `smooth_bound(*logvar_clamp)` — $\log(5/3)$ at the
  shipped $(-5, 3)$ — so the raw output is input-independent and the bounded output is exactly $0$
  (asserted at $10^{-6}$, float rounding only). A shrink recipe cannot be exact here:
  `smooth_bound` is a sigmoid, so the mean of the bound is not the bound of the mean and a
  shrunk-but-input-dependent raw value starts near zero only approximately. The zeroed layers
  still receive gradient (the final layer against its activations, the skip against its input),
  so the head trains off the constant exactly as the zero-initialised posterior deltas do, and the
  posterior builds its residual on the same raw tensor, so the exact zero-KL start is untouched.

**The causal guard**

- **`channel_reach.py` carries more than the specified `resolve_channel_budget`.** A `ChannelBudget`
  dataclass and `resolve_stream_budgets` sit above it, because two callers need the resolution
  (the model kwargs and the run record) and the startup log needs per-block counts that the
  per-stream function cannot produce. `resolve_channel_budget` remains the pure function it was
  specified as.
- **The resolved guard is recorded at `model_config.resolved_causal_budget`**, not inside
  `VAE_model`. Written into `VAE_model` it would become a competing source of truth, and
  re-running from the written file would both forward the record and re-resolve the budget.
- **A fourth pre-flight guard, `_check_causal_budget_resolves`, was added.** Without it an
  unsatisfiable budget surfaces during model construction — after directories, log sinks and an
  MLflow run exist — rather than in the pre-flight where the message is the only thing on screen.
- **The model exposes `source_delay_steps`, and every lag consumer reads it.** Adding the guard
  activated the "lag misreport" risk, and the first attempt at closing it — having the evaluation
  compute the delay itself — left the plotting callback still reporting zero, so the figure and
  the summary for one run disagreed by two minutes. The accessor lives on the model because the
  model is what was trained; `tests/test_lag_consistency.py` compares the two consumers directly.
  It is deliberately *not* in `nets/lag_report.py`, which is model-free arithmetic and stays that
  way.

**Where the design document's stated numbers or rationale did not survive measurement**

- **The reach table differs from the design's §4.5 in its fast rows.** Computed here: `fhr_st` min
  1.5 s, median 45.8 s, max 965.5 s (§4.5 states 8.8 / 54.5 / 974.2). The 120 s and 240 s budget
  rows match the design exactly (78/29 and 94/43); the 32 s and 60 s rows each keep one more
  channel per stream than §4.5 states, which is what an inclusive `reach <= budget` comparison
  gives. Inclusivity is deliberate and load-bearing: the slowest `up_ph` channel sits at exactly
  100.0 s, so the boundary decides whether a 100 s budget keeps a channel or none.
- **"Building the bank at 4800 would silently produce wrong reaches" is false at this geometry.**
  Reaches follow the *padded* length, and both 4800 and 5280 pad to $2^{13}$, so the two banks are
  identical channel for channel. The bank is still built at the stored 5280 — that is the correct
  length and the coincidence does not generalise across trims, $J$ or $Q$ — but the test pins the
  property that actually holds (the module's reaches equal an independently rebuilt 5280 bank's)
  plus a demonstration at a length that does move $J_{\mathrm{pad}}$, rather than asserting a
  difference that does not exist.
- **The leak test's guarded arm is a suppression threshold, not a numerical tolerance.** See §7.
  The design assumed the delayed features would hold still to within round-off; they do not,
  because $L_{95}$ is a quantile. The two arms are separated by a factor of 4 in threshold and
  roughly 20 in measurement, with a module-level assertion that the floor exceeds the tolerance
  four-fold so a later loosening cannot make both arms pass.

**Test placement**

- **The run-record test lives in `tests/test_main.py`**, not in `test_train_smoke.py`. The record
  is written by `_persist_resolved_config`, which `main()` calls and the smoke fixture bypasses.
- **The model-side gate tests live in `tests/test_construct.py`**, beside the other construction
  invariants, rather than in a file of their own.

## 13. Closeout — authoring checklist

`train/MODEL_MIGRATION_GUIDE.md` §3.9, verified item by item.

| Item | Verdict | Where |
| --- | --- | --- |
| Files placed per §1.1; nothing added to the §1.2 deviation table | yes | `teb_vae/lag_attn_rws/` — `nets/`, `task.py`, `trainer.py`, `configs/`, `eval/`, `tests/` |
| Raw `nn.Module` has no Lightning, no config, no I/O | yes | `nets/*.py`, enforced by `tests/test_nets_are_framework_free.py`, which also forbids the sibling's `task`/`trainer`/`config`/`eval` by dotted prefix |
| Wrapper implements `compute_loss_and_metrics` and nothing else it did not have to | yes | `task.py` — no `training_step`, `forward` or `configure_optimizers` override |
| `super().__init__(base_model, ...)` — no grandparent bypass | yes | `task.py`, forwarding its own `compile_model` kwarg (default `False`) |
| `self.model` for forward, `self.orig_model` everywhere else | yes | `task.py::compute_loss_and_metrics` |
| `on_save_checkpoint` (overridden) calls `super()` first | yes | `task.py::on_save_checkpoint` — `super()` stamps `model_class`, then `model_kwargs` is added |
| `create_model` checks the `load_checkpoint_strict` return value | yes | `trainer.py::create_model` — `None` raises rather than training from random weights |
| `train_model` calls `build_trainer`; no hand-rolled `pl.Trainer` | yes | `trainer.py::train_model` |
| `ModelCheckpoint(dirpath=self.model_checkpoint_dir, filename=...)` | yes | `trainer.py` — `filename="lag-attn-rws-{epoch:02d}"` |
| DDP strategy via the un-prefixed `select_ddp_strategy` override | yes | `trainer.py::select_ddp_strategy`, sourced from config, not from the wrapper |
| Config has all fourteen effectively-required keys | yes | `configs/default.yaml`, asserted by `tests/test_config_load.py` |
| No hand seeding; `general_config.seed` is set | yes | `configs/default.yaml`; asserted by `tests/test_train_smoke.py`, which greps the package |
| `use_distributed_sampler: true` and no self-built sampler | yes | `configs/default.yaml`; data via `GraphDataModule` |
| Entry point order ctor → `setup_config()` → `create_model()` → `train_model()` | yes | `trainer.py::main`, asserted by `tests/test_main.py` |
| The four verifications of §3.7 exist and pass | yes | the module suite, including `tests/test_train_smoke.py` |

Run the gate from the repository root:

```
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_rws/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_rws/tests -q -m slow
.venv/Scripts/python.exe -m pytest train/tests -q
```

Evaluate a checkpoint, then check the finished run against the pre-registered criteria — the
second command reads `summary.json` and nothing else, so it needs no model, no shard and no
`torch`:

```
python -m teb_vae.lag_attn_rws.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt
python -m teb_vae.lag_attn_rws.eval.verify <run>/eval_results/summary.json
```

`eval/EVAL.md` is the evaluation's contract: the output layout, the layer rules, the
configuration reference, one section per analysis, the interpretation rules and the operations
guide, including the recovery table for every way preflight refuses a run.

## 14. Configuration keys

Keys this document's claims depend on. `tests/test_docs.py` asserts each required key exists in
`configs/default.yaml` and each absent key does not, so this section cannot drift from the config.

**Required**

- `general_config.seed`
- `general_config.lr_milestone`
- `model_config.VAE_model.beta_schedule`
- `model_config.VAE_model.free_bits`
- `model_config.VAE_model.beta_prior`
- `model_config.VAE_model.likelihood`
- `model_config.VAE_model.lambda_full`
- `model_config.VAE_model.lambda_base`
- `model_config.VAE_model.lambda_ms`
- `model_config.VAE_model.lambda_deriv`
- `model_config.VAE_model.lambda_boundary`
- `model_config.VAE_model.d_z`
- `model_config.VAE_model.horizon`
- `model_config.VAE_model.raw_per_step`
- `model_config.VAE_model.warmup_period`
- `model_config.VAE_model.sequence_length`
- `model_config.VAE_model.c_y`
- `model_config.VAE_model.c_u`
- `model_config.VAE_model.use_up_st`
- `model_config.VAE_model.logvar_clamp`
- `model_config.VAE_model.coverage_floor`
- `model_config.VAE_model.base_decode`
- `model_config.VAE_model.posterior_logvar_mode`
- `model_config.VAE_model.source_dropout`
- `model_config.VAE_model.causal_norm`
- `model_config.VAE_model.causal_reach_budget_s`
- `model_config.VAE_model.max_lag`
- `model_config.VAE_model.horizon_attention_blocks`
- `model_config.VAE_model.horizon_embed_std`
- `model_config.VAE_model.head_init_calibration`
- `model_config.VAE_model.a_head_gain`
- `model_config.VAE_model.encoder_extra_kernel`
- `model_config.VAE_model.conv_norm_groups`
- `model_config.VAE_model.query_uses_logvar`
- `advanced_config.trainer.compile`
- `advanced_config.trainer.num_sanity_val_steps`
- `advanced_config.trainer.use_distributed_sampler`
- `advanced_config.trainer.gradient_clip_val`
- `advanced_config.spike_breaker.ema_floor`
- `advanced_config.spike_breaker.additive_margin`
- `advanced_config.spike_breaker.comparison_metric`
- `dataset_config.stat_path`
- `dataset_config.dataloader_config.normalize_fields`

**Deliberately absent**

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
