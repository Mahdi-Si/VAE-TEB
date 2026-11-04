# `lag_attn_rws` — the as-built design record

The raw-signal lag-attention VAE-TEB: what it is, what it consumes, what it returns, what it
optimises, and where the built model differs from the design it was built from.

Companion documents: `MODEL_DOCUMENTATION.md` in this directory describes the architecture layer by
layer; `model_explained.md` states the design it was built from; `teb_vae/lag_attn/DESIGN.md`
records the feature-target sibling. Neither is restated here.

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

The shipped configuration uses $d_z = 64$. One shared decoder $D_\psi$ is invoked on $z^p_t$ and on
$z^q_t$ and receives nothing else. Under `base_decode: sample` one noise draw $\epsilon_t$ serves
both latents; under the shipped `base_decode: mean` the base branch decodes $z^p_t = \mu^p_t$ and
only $z^q_t$ is sampled.

The code enforces three properties structurally. Each has its own test.

| Property | What enforces it | Test |
| --- | --- | --- |
| **No decoder bypass** — gradient reaches the decoder only through $z$ | `BaselineFutureDecoder.forward` takes the latent alone on this architecture (`persistence_residual` is refused by `_check_persistence_target`); there is no `decoder_state` head and no second decoder | `tests/test_no_bypass.py`, `tests/test_decoder.py` |
| **Source purity** — the source pathway never sees a target tensor and the prior never sees the source | separate adapters and encoders; the posterior mean is a residual on the prior mean | `tests/test_source_purity.py` |
| **Exact zero at initialization** — $\mathrm{KL}(q_t \Vert p_t) = 0$ *in train mode* | posterior deltas zeroed **after** `initialization`; decoder and attention dropout fixed at $0$; under `base_decode: sample` one shared $\epsilon$ also makes the two forecasts bitwise identical | `tests/test_zero_kl_init.py` |

Because the KL is exactly zero at initialization, any KL assertion on a freshly constructed model
passes vacuously. The `perturb_posterior` fixture in `tests/conftest.py` moves the posterior off the
prior, and every test that checks KL behaviour uses it.

Under `posterior_logvar_mode: independent` (shipped) the posterior log-variance has its own head.
The exact zero-KL start then depends on `head_init_calibration: true`, which pins the prior's raw
log-variance to the constant the independent head is seeded at.

## 2. Input contract

The model reads the HDF5 through `train/data_module.py::GraphDataModule`, at `trim_minutes: 1.0`.

| Field | Shape per sample | Role |
| --- | --- | --- |
| `fhr` | $(4800,)$ | **the reconstruction target**, z-scored |
| `fhr_st` | $(300, 43)$ | target scattering |
| `fhr_ph` | $(300, 66)$ | target phase-harmonic |
| `up_st` | $(300, 43)$ | source scattering |
| `up_ph` | $(300, 15)$ | source phase-harmonic |
| `weight` | $(300,)$ | the authoritative validity signal |
| `up`, `guid` | | diagnostic figures (and the optional contraction weight, §5.2) |

So $c_y = 43 + 66 = 109$ and $c_u = 43 + 15 = 58$ (or $15$ under `use_up_st: false`).

`fhr_up_ph` is **never loaded**. A cross-channel coefficient mixes both signals in one number and
would destroy the separation between $p(z_t \mid Y_{\le t})$ and $q(z_t \mid Y_{\le t}, U_{\le t})$.
`tests/test_config_load.py` asserts it is not among the shipped config's `load_fields`.

Two contract points fail silently rather than loudly, so the entry point guards them:

- **`fhr` must be in `normalize_fields`, not only in `load_fields`.** Without it the target
  arrives in bpm while the decoder's learned log-variance models a z-scale, and the Gaussian NLL
  is meaningless with nothing raising. `trainer.py::_check_raw_target_normalized` guards it.
- **Gaps are stored as $0.0$ bpm**, which after z-scoring is not a detectable sentinel. The
  decimated `weight` is the only trustworthy validity signal, and both it and `fhr` are hard
  requirements in `task.py::_build_raw_target`.

Channel widths are dataset facts, not model constants. The trainer checks them against the first
shard before the fit (`_check_declared_widths_against_shard`), and the task checks them against
every batch at the data boundary (`task.py::_build_target_streams` / `_build_source_stream`).

## 3. Geometry

`nets/geometry.py::TrimmedRawGeometry` validates itself in `__post_init__`, so an unvalidated
instance cannot exist. The loader has already applied the crop, so there is **no** `CROP` offset
anywhere:

$$
n_{\mathrm{raw}}(t) = 16\,(t+1) - 1, \qquad \mathrm{future\_block\_start}(t) = 16\,(t+1).
$$

$T = 300$, $T_{\mathrm{valid}} = T - H = 270$, trained anchors $[30, 270)$. Anchor $0$ forecasts
`fhr[16:496]` and anchor $269$ forecasts `fhr[4320:4800]`.

**This is the most error-prone point in the build.** A model that loads *untrimmed* data and crops
15 tokens inside the model has $16(t+16) - 1$, whose anchor-0 forecast starts at raw $256$, not
$16$. Each formula is wrong on the other grid, and nothing downstream fails loudly on the
difference. `tests/test_geometry.py` pins $16$ as distinct from both $256$ and $0$.

The masks in `nets/raw_masks.py` use a consequence of there being no crop: future sample
$(t, \tau, r)$ maps to `weight[t + 1 + \tau]` for **every** $r$. The forecast mask is therefore a
$(B, 270, 30)$ decimated tensor broadcast over $r$, not a raw-resolution stack. A test pins the
equivalence against a naive per-raw-index gather.

The KL is masked to the **same** anchor support the reconstruction uses. `kl_mask` takes the
*forecast mask* and reduces it with `contributing_anchors`, the one indicator the loss's per-anchor
denominator also uses. It is not a second expression over `weight`. If $\beta \cdot \mathrm{KL}$
were charged on an anchor with no reconstruction term, nothing would pull the posterior off the
prior there, and the KL would be regularised to zero for free on the tail anchors and before every
signal-loss gap.

## 4. Forward return dict

`nets/model.py::SeqVaeLagAttnRws.forward(y_st, y_ph, u_stream)` returns twenty keys, plus
`event_gate` when the model is built with `posterior_event_gate`. There is no `decoder_state` and
no `delta_mu_src`: neither pathway exists.

| Key | Shape | |
| --- | --- | --- |
| `mu_prior`, `logvar_prior`, `raw_logvar_prior` | $(B, T, d_z)$ | the target-only prior; the raw form is pre-`smooth_bound`, which is what the `residual` posterior log-variance is applied to |
| `mu_post`, `logvar_post` | $(B, T, d_z)$ | the source-conditioned posterior |
| `z_prior`, `z_post` | $(B, T, d_z)$ | the two decoded latents; under `base_decode: mean`, `z_prior` **is** the `mu_prior` tensor (same object) |
| `target_state` | $(B, T, d_{model})$ | target encoder history state |
| `source_state` | $(B, T, d_{model})$ | the source representation the attention reads as keys and values (selected by `lag_kv_source`) |
| `attended_source_heads` | $(B, T, m, d_{head})$ | per-head attended summaries |
| `attn_weights` | $(B, T, m, L)$ | attention probabilities in lag order, $L = 91$ |
| `mu_base`, `logvar_base` | $(B, 270, 30, 16)$ | the target-only raw forecast |
| `mu_full`, `logvar_full` | $(B, 270, 30, 16)$ | the source-conditioned raw forecast |
| `kld_per_t` | $(B, T)$ | per-step KL, summed over $d_z$ |
| `kld_per_t_per_head` | $(B, T, m)$ | its additive per-latent-group split |
| `source_kl_lag_map` | $(B, T, L)$ | the KL attributed across lags |
| `mu_prior_sat_frac`, `delta_mu_sat_frac` | scalar | bound-saturation diagnostics |
| `event_gate` | $(B, T, 1)$ | only under `posterior_event_gate` |

Decoding covers the valid anchor range only. The tail $H$ anchors have no fully observed raw future
and the loss would discard them.

`source_kl_lag_map` sums over lags to `kld_per_t` **exactly**, and only because the attention
probabilities carry no dropout. That identity is why `LagCrossAttention` is constructed at
`dropout=0.0`.

`lag_kv_source` selects the key/value memory: `'encoder'` (default) is the deep source encoder,
`'conv_stem'` is a `CausalConvStem` over the same convolution schedule, and `'adapter'` is the
projected stream itself. Under the two local arms the deep source encoder is not built.
`SeqVaeLagAttnRws.source_kv_body` is the one place the arm is resolved into a module.

## 5. Loss

`nets/model.py::compute_loss` delegates to `nets/losses.py::compute_loss`:

$$
\mathcal{L} = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0
  + \beta(e)\,\mathrm{KL}_{\mathrm{train}} + \beta_p R_p
  + \lambda_{\mathrm{ms}} \mathcal{L}_{\mathrm{ms}}
  + \lambda_{\mathrm{deriv}} \mathcal{L}_{\mathrm{deriv}}
  + \lambda_{\mathrm{boundary}} \mathcal{L}_{\mathrm{boundary}}, \qquad
  \lambda_{\mathrm{full}} = \lambda_{\mathrm{base}} = 1 .
$$

**Units.** The NLL is *summed* over the 480-sample block and the KL is *summed* over $d_z$; both
are then averaged over batch and contributing anchors only. Both are therefore in nats per anchor,
which is what gives $\beta$ its meaning. `mse` is summed over the horizon too, so $\beta$ keeps its
meaning across likelihoods. A test asserts $D_{\mathrm{block}} = 480 \cdot D_{\mathrm{sample}}$ and
that a masked sample contributes exactly zero.

**The last three terms are not nats**, so the total is not either (§5.1). The pure-nats readouts
are `nll_full_block` and `nll_base_block`.

**The fourth term anchors the prior's scale.**

$$
R_p = \sum_{d} \tfrac{1}{2}\left(e^{\ell^p_{t,d}} - 1 - \ell^p_{t,d}\right)
  = \mathrm{KL}\!\left(\mathcal{N}(\mu^p, \operatorname{diag} e^{\ell^p})
    \,\Vert\, \mathcal{N}(\mu^p, I)\right),
$$

summed over $d_z$, masked by the KL's own anchor support and divided by the same
contributing-anchor count. It is therefore in nats per anchor. It is nonnegative, convex in
$\ell^p$, and exactly zero at $\sigma_p = 1$. The prior *mean* does not appear, so the term
compresses nothing $D_0$ depends on.

Without $R_p$ nothing in the objective penalises a narrow prior. The reconstruction prefers a
deterministic latent, since sampling noise can only degrade a forecast, and $\mathrm{KL}(q \Vert p)$
measures the posterior *against* the prior without constraining the prior's own scale. A prior
log-variance on its clamp floor would multiply the KL term $(\mu^q - \mu^p)^2/\sigma_p^2$ by an
arbitrary factor.

$\beta_p$ (`beta_prior`, shipped $0.1$) is a constant, not a schedule. The weighted restoring force
$\beta_p\,\partial R_p / \partial \ell^p = \tfrac12\beta_p(e^{\ell^p} - 1)$ saturates at
$-\tfrac12\beta_p$ per dimension as $\ell^p \to -\infty$. $R_p$ is computed and logged as
`prior_rate` **unconditionally**, under every likelihood and at any $\beta_p$.

$\lambda_{\mathrm{full}} D_1 + \beta\,\mathrm{KL}$ at $\beta = 1$ is the exact ELBO of the
source-conditioned branch. Adding $D_0$ at unit weight doubles the reconstruction pressure against
the KL, so $\beta = 1$ is a starting point, not a distinguished optimum.

$\beta$ follows a linear warm-up from exactly $0$ (`beta_schedule`, $0 \to 1$ over 50 epochs).
Because $z$ is the only route to the decoder, a nonzero $\beta$ before the decoder can use the
latent pushes the posterior onto the prior.

The base branch's latent is set by `base_decode`. Under `'mean'` $D_0$ decodes $\mu^p$, so $D_0$
applies no gradient to $\ell^p$. Under `'mean'` only the full branch is sampled, so the
training-path `pred_gap` is not a common-random-numbers contrast; the offline evaluation rebuilds
the pairing from the latent distributions and does not read `base_decode`.

Two loss-scale settings are carried in config:

- **The spike breaker's relative test is off.** `watched > multiplier * max(EMA, ema_floor)`
  assumes a loss bounded below by zero. A learned-variance Gaussian NLL summed over 480 samples can
  go negative, and once the EMA is negative that test would skip every positive batch. So
  `ema_floor: 1.0e+9` disables it, and `additive_margin` (compared against the **raw** EMA, so it
  works in the negative regime) carries finite-spike detection beside the non-finite guard.
- **`gradient_clip_val: 5000.0`** with `gradient_clip_algorithm: norm`. `train/grad_norm` is logged
  pre-clip and `train/grad_clip_frac` logs the exceedance indicator.

### 5.1 The auxiliary shape terms

The last three terms of the criterion score the *shape* of the forecast mean. The factorized
Gaussian NLL scores every raw sample independently, so its optimum is the conditional mean, and a
fully parallel block decoder can emit an over-smoothed one. Each term is computed on `mu_base` and
`mu_full` and **summed** over the two, under the module's own masks. Each is reduced by the
**same** contributing-anchor count as $D_k$.

| Term | What it scores | Definition |
| --- | --- | --- |
| $\mathcal{L}_{\mathrm{ms}}$ | the block's envelope | mask-weighted $L_1$ between forecast mean and target, average-pooled over the flattened $H \cdot R$ block at `MS_RATES = (1, 4, 16)` |
| $\mathcal{L}_{\mathrm{deriv}}$ | slope | Huber at $\delta = 1$ (`DERIVATIVE_HUBER_DELTA`) between the first differences of the two; a pair is valid only when both its samples are |
| $\mathcal{L}_{\mathrm{boundary}}$ | the starting level | $\lvert \hat\mu_t[0] - Y[n_t] \rvert$: the first forecast sample against the anchor's last observed one |

Three implementation points matter:

- **The multiscale mask is applied before pooling**, not after. Pooling mixes neighbours, so a gap
  left in until afterwards would leak its sentinel into every pool it touches.
- **The boundary term is a slicing identity**, `mu[:, 1:, 0, 0]` against `target[:, :-1, 0, -1]`,
  exact because $X = R = D$ on the raw grid. It is computed over $t \in [1, T_{\mathrm{valid}})$, so
  anchor $0$ is excluded by construction. Its validity is anchor $t$'s own `weight` at threshold
  times $t$'s contributing indicator, not anchor $t-1$'s forecast mask.
- **A term whose weight is $0.0$ is not computed**, and its metric is reported as exact $0.0$. The
  branch reads a config-constant float, identical on every rank and every batch, so DDP graph
  identity is unaffected.

The weights ship at $0.1 / 0.1 / 0.05$. The boundary weight is half the others because it
constrains **one** sample per anchor against $480$ for the other two.

### 5.2 Optional objective terms

These are off in the shipped configuration and leave the objective bitwise unchanged when off:

- `posterior_event_gate` with `lambda_event_gate` — a scalar gate $g_t \in (0, 1)$ per step, read
  off the attended source summary, multiplies both posterior residuals; the objective charges its
  sparsity.
- `event_weight_floor` with `event_weight_reach_steps` (and `event_weight_ua_valid`) — a dense
  $(B, T)$ weight on the full branch's reconstruction alone, built by `event_weight.py` from the
  contractions detected on `up`.
- `horizon_weight_halflife_steps` — a decaying horizon weight $w_\tau \propto 2^{-\tau/\lambda}$,
  normalised to sum to $H$, on both reconstruction terms. A weighted block score is not a
  log-density, so $\beta = 1$ is then no longer the exact ELBO.

## 6. Structural constraints that are not preferences

| Setting | Why it is required | Where |
| --- | --- | --- |
| decoder `dropout=0.0` | one module invoked twice draws two independent masks: base and full would differ at initialization even with $z^p = z^q$, and independent noise would enter `pred_gap` every step | `nets/model.py`, decoder construction |
| attention `dropout=0.0` | dropout is applied to the probabilities *before* they are returned, which would break the KL/lag-map identity | `nets/model.py`, attention construction |
| `lag_attn.W_o` frozen | the head-structured posterior consumes the per-head summaries, so `W_o` receives no gradient; freezing drops it from DDP's expectation set | `nets/model.py`, and `select_ddp_strategy` |
| `causal_norm: true` | a non-causal `GroupNorm` pools statistics across time, so the "prior" conditions on the future and the KL is not a coupling readout | `configs/default.yaml`; `create_model` warns when off |
| `compile: false` | the LSTM encoders break TorchInductor, so `compile_model_requested` returns `False` and does not read the key | `configs/default.yaml`; `trainer.py::compile_model_requested` |
| `num_sanity_val_steps: 0` | `MetricsLoggingCallback` has no sanity guard; a sanity pass shifts every epoch number against MLflow and the checkpoint filenames | `configs/default.yaml` |
| posterior deltas zeroed **after** `initialization` | `initialization` xavier-fills every `nn.Linear` and would otherwise undo the zeroing, and with it the exact zero-KL start | `nets/model.py::_zero_init_delta_heads` |

## 7. The causal input guard

The stored features are two-sided wavelet transforms: a feature step reads raw signal from its own
future, up to 974 s of it. `causal_reach_budget_s` bounds that.

The resolution in `teb_vae/lag_attn/channel_reach.py` keeps channel $c$ when
$\mathrm{reach}_c \le \mathrm{budget}$ (inclusive) and reads each survivor at $t - \delta_c$ with
$\delta_c = \lceil \mathrm{reach}_c / 4 \rceil$ steps. `teb_vae/lag_attn/nets/delays.py::ChannelDelay`
applies the delay. The delay is per channel rather than one uniform band, so the fastest survivors
stay only a few steps stale.

| budget | $c_y$ | $c_u$ | max $\delta$ | per-block survivors |
| --- | --- | --- | --- | --- |
| `null` | 109 | 58 | 0 | everything |
| 240 s | 94 | 43 | 57 | — |
| **120 s** (shipped) | **78** | **29** | **30** | `fhr_st` 27/43, `fhr_ph` 51/66, `up_st` 27/43, `up_ph` 2/15 |
| 60 s | 59 | 23 | 14 | `up_ph` 0/15 |
| 32 s | 43 | 19 | 8 | `up_ph` 0/15 |

Regenerate with `python -m teb_vae.lag_attn.channel_reach`.

Design points to know before changing anything here:

- **The model is built with the full declared $c_y$ / $c_u$** and gathers survivors *inside* the
  forward, after the data boundary and before the input adapters. Otherwise the task's width check
  and the shard-width pre-flight would both reject any nonzero budget.
- **The guard is one object per stream** — `ChannelGate`, holding the keep-index and its
  `ChannelDelay`. The delay vector is positional *against* the keep-index, so the two are
  meaningless apart. The model exposes `target_gate` / `source_gate` (`None` when unguarded) and
  the derived `source_delay_steps`, which every lag consumer reads (§8).
- **The input adapters are `AvailabilityInputAdapter`.** Under a guard the first $\delta_c$ steps
  of each surviving channel are exact zeros. The adapter adds availability terms built from
  $m_{t,c} = \mathbb 1[t \ge \delta_c]$, read back off the constructed gate, so a zero-filled token
  does not enter the normalisation layers as a zero-variance vector. Unguarded, neither
  availability term is constructed. `tests/test_train_smoke.py::test_the_guarded_runs_gradient_stays_finite_and_small`
  asserts that a guarded fit trains with a finite, bounded gradient.
- **The delay must fit inside the warm-up.** The first $\max_c \delta_c$ steps of a delayed stream
  are partly zero-filled, so they must fall inside the steps the loss already discards. The
  comparison is strictly greater-than: at 120 s the maximum delay is exactly 30, which is also the
  shipped `warmup_period`, and that configuration is allowed. `sweep_reach_240.yaml` raises
  `warmup_period` to 60 for this reason.
- **Both gate buffers are non-persistent.** Their length is the surviving-channel count, so a
  persistent copy would make a checkpoint trained at one budget fail to load at another. The four
  resolved tuples live in `model_kwargs`, because the adapters' widths depend on them.
- **Arms at different budgets cannot share checkpoints**, because their adapter input widths
  differ.
- **The unguarded model builds no gather and no delay at all**, so it is structurally the model
  that existed before the guard.

**The guard bounds the leak; it does not remove it.** $L_{95}$ is an energy *quantile*, not a
support: 5% of every filter's energy lies beyond its stated reach. `tests/test_causal_leak.py`
therefore tests the guarded arm against a suppression threshold, not a round-off tolerance. Only
genuinely causal transforms remove the leak.

## 8. Lag and KL reading rules

**There is one lag quantity, and the stored timeline is canonical.**
`teb_vae/lag_attn/nets/lag_report.py::lag_compensated_seconds` computes it from the attention lag
index $\ell$:

$$
\tau_{\mathrm{compensated}} = 4\,(\ell + \delta).
$$

Figure axes use `COMPENSATED_LAG_AXIS_LABEL`, so a plot and the number beside it show the same
quantity.

The $\delta$ term is the causal input delay, and **every consumer reads it from one place**:
`SeqVaeLagAttnRws.source_delay_steps`. The training figure and the evaluation both take it from
there. `tests/test_lag_consistency.py` pins the two consumers together. With per-channel delays
there is no single $\delta$, so the **maximum** over source channels is used, which makes the
reported lag an upper bound. `summary.json` records this as `source_delay_is_max_over_channels`.

**Only the unfloored KL is an information rate.** `source_conditioned_kl_raw` is the readout;
`source_conditioned_kl_train` has free bits applied and is the quantity the optimiser sees. The
floored one can hide a collapsed source pathway, so the progress bar
(`SeqVaeLagAttnRwsTask.prog_bar_metrics`) shows the raw one. At the shipped `free_bits: 0.0` the two coincide.

**A prior variance on its clamp inflates the coupling readout.** The KL carries
$(\mu^q - \mu^p)^2 / \sigma_p^2$. `mean_logvar_full` and `mean_logvar_base` are *decoder output*
variances and do not detect a narrow prior. `mean_logvar_prior` and `logvar_prior_floor_frac` do,
and are logged from epoch 0.

## 9. Naming

The KL is `source_conditioned_kl_raw` / `_train`, and its decomposition is `source_kl_lag_map`. It
is **not** called transfer entropy: the inputs are not fully causal (§7), and a factorized Gaussian
decoder estimates a rate within its model family only. `pred_gap` is $D_0 - D_1$.

## 10. Deliberate limitations

Each `lean-limit` annotation lives in the code where an engineer would trip over it.

| Where | Limitation |
| --- | --- |
| `nets/losses.py`, beside the KL | the KL is a provisional source-conditioned rate, not transfer entropy |
| `nets/losses.py`, beside the Gaussian NLL | a factorized Gaussian estimates TE within that model family, not the data-generating TE |
| `nets/__init__.py` | shared primitives are imported from the sibling rather than promoted to a common package |
| `nets/model.py`, beside `compute_loss` | all valid anchors are decoded every batch; no anchor subsampling |
| `channel_reach.py` | the reach bound is an energy quantile, not a hard support |

These limitations have no single home in the code:

> lean-limit: `MS_RATES = (1, 4, 16)` is a module constant rather than configuration; replace it
> with a config key when a sweep over pooling rates is wanted.

> lean-limit: the boundary term uses the level identity only; the slope variant of
> `model_explained.md` is not implemented.

> lean-limit: the evaluation's oracle probe (`eval/oracle.py`, reported by the `sufficiency`
> analysis) fits a decoder of the same capacity on `target_state` instead of $z$, on half the
> evaluation recordings, and scores it on the other half:
> $\Delta_{\mathrm{suff}} = D_{\mathrm{base}} - D_{\mathrm{oracle}}$. Two biases of unknown size
> oppose each other. Conditioning on `target_state` rather than on the raw FHR history omits the
> encoder's own information loss and biases the gap down. Fitting the probe on the evaluation
> population while $D_{\mathrm{base}}$ comes from a model trained on a disjoint pretraining cohort
> biases it up. The number is therefore an estimate, not a bound. A second probe fitted on a
> held-out split of the training cohort would remove the second bias.

> lean-limit: the prior anchor $R_p$ of §5 is the **scale half** of the context rate
> $\mathrm{KL}(p_\theta \Vert \mathcal{N}(0, I))$. The two halves separate exactly:
> $\mathrm{KL}(p \Vert \mathcal{N}(0,I)) = R_p + \tfrac12\sum_d (\mu^p_{t,d})^2$. Only the first
> is adopted, because the second compresses the prior *mean*, which is the entire content of the
> base forecast. Adding the mean half later is an addition, not a revision.

> lean-limit: `horizon_embed_std: 0.8` is a fixed value, not matched to the projected-latent RMS
> at construction, so its relative scale changes across the `d_z` sweep arms.

These choices are also deliberate:

- The attention query is a projection of the prior belief. By default that is $\mu^p_t$ alone;
  `query_uses_logvar: true` widens it to $[\mu^p_t \,\|\, \ell^p_t]$ (`query_proj` in-width
  $2 d_z$). Both inputs are target-only, so source purity holds either way.
- `lag_band_mask` is **not** a forward parameter. In `LagCrossAttention` a band mask *replaces*
  rather than intersects the causal-validity mask, so a naive band mask would admit lags that do
  not exist, and an all-masked row makes `entmax15` raise.

### What the evaluation does not do

`eval/EVAL.md` is the evaluation's contract and `eval/FIGURE_GUIDE.md` documents every figure. The
evaluation deliberately omits these:

- **Lag ablation** (band-restricted sufficiency ranking), because `lag_band_mask` is not a forward
  parameter (above). It needs an intersecting band-mask forward argument on `LagCrossAttention`.
- **Necessity.** A band ranking measures *sufficiency* and reads backwards if taken for a removal
  ablation, so the pipeline emits neither.
- **Spectral resolution below $\Delta f$.** The `coherence` analysis fixes a horizon step and
  concatenates over consecutive anchors, which yields a contiguous 960 s series per lead time;
  `nperseg = 512` at $f_s = 4$ Hz gives $\Delta f = 7.8$ mHz. Nothing below that is resolved.
- **Non-stationary coherence** (STFT or wavelet *within* a recording). Every estimate pools
  windows.
- **Absolute band power against a clinical norm.** The analysis measures truth-versus-forecast
  agreement, not the trace's own spectral content.
- **Distributional distances between per-class latent populations** (FID, MMD). The pretraining
  checkpoint has not seen the clinical classes, so such a distance would measure input
  distribution shift.
- **Held-out clinical discrimination.** The pretraining cohort is healthy-only, so every class
  contrast is out-of-distribution.

## 11. Deviation record

Every intentional difference between the built module and the design it was built from.

**Architecture and configuration**

- **`sigma_obs`, `head_structured_latent` and `freeze_unused_attn_proj` are not config keys.** All
  three are unconditional in the net: the learned observation variance, the head-structured
  posterior and the frozen attention projection are structural facts.
- **`select_ddp_strategy` returns a configured `DDPStrategy`**, not a `'ddp'` /
  `'ddp_find_unused_parameters_true'` shorthand string. `trainer.py::ddp_kwargs` builds its
  arguments. `find_unused_parameters` is the only entry that reads config: the decoder log-variance
  heads are consumed only under `likelihood: gaussian_nll`, so `mse` starves them. The other
  settings are `broadcast_buffers=False` (every buffer in this family is a deterministic function
  of the config, and there is no `BatchNorm`) and `gradient_as_bucket_view=True`. **`static_graph`
  is absent**: on a skipped batch the loss-spike breaker substitutes a zero-weighted sum over every
  parameter, which is a structurally different backward. `tests/test_ddp_strategy.py` carries that
  as a negative control.
- **`tiny.yaml` ships `likelihood: mse`**, so the smoke path exercises the DDP fallback.
- **`FullLatentPriorHead` is written fresh** rather than reusing the sibling's `PriorHead` and
  discarding its `decoder_state` output, which would leave dead parameters.
- **No decoder class was written.** `BaselineFutureDecoder` already takes one shared core and one
  input tensor; at `d_model = d_z` and `out_channels = 16` it emits $(B, T, 30, 16)$ directly.
- **The encoder conv stack is a single plain residual, not the sibling's double residual.** Each
  `CausalMultiChannelConvBlock` is already a pre-norm residual. The sibling adds a second,
  `GroupNorm`-rescaled copy of the stream at every inter-block seam onto an un-renormalised,
  growing stream. Both encoders here use `CausalConvLstmEncoder(stack_skip_connection=False)`; the
  shared-class default keeps the sibling's double stack, so the sibling is unchanged. The outer
  `+ x_lin` at the stack exit is kept.
- **The horizon core re-injects the latent via zero-init FiLM at *every* refine block**
  (`film_per_block=True` on the shared `HorizonDecoderCore`; `horizon_film` still feeds `film`, so
  `horizon_film: false` fails at construction). Every block reads the projected latent $h$, the only
  $z$ entry point. `initialization` xavier-refills every `nn.Linear`, so the model re-zeros the FiLM
  generators unconditionally in `_zero_init_film_generators`; at step 0 the per-block-FiLM decoder
  is bitwise the FiLM-free decoder.
- **The horizon core mixes its $H$ tokens with self-attention after the refine stack.** At
  `horizon_depth: 4` the dilations are $(1, 2, 4, 8)$ and the receptive field is $31$ over $30$
  tokens, but only through fixed local windows. `horizon_attention_blocks: 2` adds two pre-norm
  bidirectional blocks with 4 heads (a constructor argument with no config key). Symmetric attention
  is legitimate on this axis because every horizon step is predicted from the same anchor $t$. Each
  block has $4 d_{\mathrm{hidden}}^2 + 2 d_{\mathrm{hidden}} + 1$ parameters: four bias-free
  projections, one `LayerNorm` and one scalar residual gain. The block is hand-rolled rather than
  `nn.MultiheadAttention`, whose functional attention dropout is invisible to the dropout-zero scans
  and whose packed `in_proj_weight` the generic `initialization` would xavier-fill at three times
  the true fan-in. The residual gain is a bare `nn.Parameter` at $10^{-2}$, which the generic pass
  ignores. `attention_blocks: 0` constructs **no module**. There is no separate positional
  encoding; `horizon_embedding` is the tokens' identity.
- **The adapter, front and fusion seams carry no post-residual activation.** Each seam is a
  `ResidualMLP` that ended in a normalise + GELU, and the GELU gates the backward gradient through
  the seam. The model builds them with `final_activation=False` on all four seams (two
  `InputAdapter.res_mlp`, two `CausalConvLstmEncoder.front_mlp`/`fusion`) through the shared-class
  flag `post_residual_activation=False`. Each affected seam drops its final `LayerNorm`. The
  shared-class default keeps the sibling's gated seams.

**Initialisation policies (zero-parameter, config-reversible)**

Three post-init re-initialisations run after the generic `initialization`. Each is a config key
(`horizon_embed_std`, `head_init_calibration`, `a_head_gain`) that `sweep_init_off.yaml` reverts
together. Each constructor default is an exact no-op, and the constructor applies the re-init only
when the value leaves its default. They add no parameters.

- **The horizon-step embedding is re-seeded at a larger scale** (`_reinit_horizon_embedding`). The
  core seeds `horizon_embedding` at $\mathcal{N}(0, 0.02^2)$, small against the broadcast projected
  latent, so the $30$ horizon tokens enter the refine stack nearly identical. Re-seeding at
  `horizon_embed_std` $= 0.8$ gives per-block FiLM distinct per-step offsets from step $0$. The
  posterior deltas are untouched, so the exact zero-KL start is preserved.
- **The shared decoder's output heads are calibrated onto the trivial predictor**
  (`_calibrate_output_heads`). `head_init_calibration` shrinks the mean head by $0.02$ so
  $\hat\mu \approx 0$ (scaled, not zeroed, so a perturbed posterior still moves the two forecasts
  apart; `test_zero_kl_init` depends on it). It sets the log-variance bias to $\log(5/3)$ so
  `smooth_bound(-5, 3)` maps it to exactly $0$ ($\sigma = 1$; $\mathrm{sigmoid}(\log(5/3)) = 5/8$,
  $-5 + 8 \cdot 5/8 = 0$), and shrinks the log-variance weight by $0.1$. Base and full share this
  decoder, so both stay calibrated identically.
- **The posterior fusion's attended-source norm carries a gain** (`_set_a_head_gain`). The
  head-structured posterior fuses a $d_{model}$-wide target state with a $d_{head}$-wide attended
  source summary. `a_head_gain` $= 2.0 = \sqrt{d_{model}/d_{head}}$ rescales the summary so the two
  inputs enter the fusion at comparable magnitude. The deltas are still zero at init, so the KL is
  exactly zero.
- **The prior head's log-variance starts at unit scale, under the same calibration key**
  (`_calibrate_prior_scale`). The head is a `ResidualMLP` returning `body(x) + skip_proj(x)` with no
  single bias governing its output level. So the final body `Linear`'s weight and the whole skip
  projection are zeroed, and the final bias is set to the pre-image of log-variance $0$ under
  `smooth_bound(*logvar_clamp)` — $\log(5/3)$ at $(-5, 3)$. The raw output is then
  input-independent and the bounded output is exactly $0$. A shrink recipe cannot be exact here,
  because `smooth_bound` is a sigmoid. The zeroed layers still receive gradient, so the head trains
  off the constant as the zero-initialised posterior deltas do.

**The causal guard**

- **`channel_reach.py` carries more than `resolve_channel_budget`.** A `ChannelBudget` dataclass
  and `resolve_stream_budgets` sit above it, because the model kwargs and the run record both need
  the resolution and the startup log needs per-block counts. `resolve_channel_budget` remains a
  pure function.
- **The resolved guard is recorded at `model_config.resolved_causal_budget`**, not inside
  `VAE_model`. Inside `VAE_model` it would become a competing source of truth, and re-running from
  the written file would both forward the record and re-resolve the budget.
- **A pre-flight guard, `_check_causal_budget_resolves`, runs before the fit.** An unsatisfiable
  budget then fails before directories, log sinks and an MLflow run exist.
- **The model exposes `source_delay_steps`, and every lag consumer reads it.** The accessor lives
  on the model because the model is what was trained; `tests/test_lag_consistency.py` compares the
  two consumers directly. It is not in `lag_report.py`, which is model-free arithmetic.
- **Reach resolution is inclusive** (`reach <= budget`). The slowest `up_ph` channel sits at
  exactly 100.0 s, so the boundary decides whether a 100 s budget keeps a channel or none.
- **The filter bank is built at the stored length 5280.** Reaches follow the *padded* length, and
  4800 and 5280 both pad to $2^{13}$, so the two banks coincide at this geometry. The test pins the
  module's reaches against an independently rebuilt 5280 bank, plus a length that does move
  $J_{\mathrm{pad}}$.

**Test placement**

- **The run-record test lives in `tests/test_main.py`**, not in `test_train_smoke.py`.
  `_persist_resolved_config` writes the record; `main()` calls it and the smoke fixture bypasses it.
- **The model-side gate tests live in `tests/test_construct.py`**, beside the other construction
  invariants.

## 12. Framework contract

The package follows `train/MODEL_MIGRATION_GUIDE.md` §3.9.

| Item | Where |
| --- | --- |
| Layout | `teb_vae/lag_attn_rws/` — `nets/`, `task.py`, `trainer.py`, `configs/`, `eval/`, `tests/` |
| The raw `nn.Module` has no Lightning, no config and no I/O | `nets/*.py`, enforced by `tests/test_nets_are_framework_free.py`, which also forbids the sibling's `task`/`trainer`/`config`/`eval` by dotted prefix |
| The wrapper implements `compute_loss_and_metrics` and no `training_step`, `forward` or `configure_optimizers` override | `task.py` |
| `super().__init__(base_model, ...)`, forwarding `compile_model` (default `False`) | `task.py` |
| `self.model` for forward, `self.orig_model` everywhere else | `task.py::compute_loss_and_metrics` |
| `on_save_checkpoint` calls `super()` first, then adds `model_kwargs` | `task.py::on_save_checkpoint` |
| `create_model` checks the `load_checkpoint_strict` return value; `None` raises | `trainer.py::create_model` |
| `train_model` calls `build_trainer`; no hand-rolled `pl.Trainer` | `trainer.py::train_model` |
| Checkpoints are named `lag-attn-rws-{epoch:02d}` (`CHECKPOINT_STEM`) | `trainer.py` |
| DDP strategy via the `select_ddp_strategy` override | `trainer.py::select_ddp_strategy` |
| The shipped configs validate with no unknown-key warnings | `tests/test_config_load.py` |
| No hand seeding; `general_config.seed` is set | `configs/default.yaml`; `tests/test_train_smoke.py` |
| `use_distributed_sampler: true` and no self-built sampler | `configs/default.yaml`; data via `GraphDataModule` |
| Entry-point order: constructor → `setup_config()` → `create_model()` → `train_model()` | `trainer.py::main`, `tests/test_main.py` |

Run the tests from the repository root:

```
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_rws/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_rws/tests -q -m slow
.venv/Scripts/python.exe -m pytest train/tests -q
```

Train, evaluate a checkpoint, then check the finished run against the pre-registered criteria. The
last command reads `summary.json` and nothing else, so it needs no model, no shard and no `torch`:

```
python -m teb_vae.lag_attn_rws.trainer --config teb_vae/lag_attn_rws/configs/default.yaml
python -m teb_vae.lag_attn_rws.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt
python -m teb_vae.lag_attn_rws.eval.verify <run>/eval_results/summary.json
```

`eval/EVAL.md` is the evaluation's contract: the output layout, the layer rules, the configuration
reference, one section per analysis, the interpretation rules and the operations guide.

## 13. Configuration

### 13.1 Keys this document depends on

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

**Optional, off in the shipped config** (§4, §5.2): `lag_kv_source`, `posterior_event_gate`,
`lambda_event_gate`, `event_weight_floor`, `event_weight_reach_steps`, `event_weight_ua_valid`,
`horizon_weight_halflife_steps`.

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

### 13.2 Sweep configurations

Each sweep file sets `base: default.yaml` and changes one key under `model_config.VAE_model`,
except the two multi-key arms noted below. All arms share the default tag, so a run is identified
by its `TEB_RUN_STAMP` directory and the `resolved_config.yaml` written beside its checkpoints.
Launch an arm from the repository root:

```bash
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_rws.trainer --config teb_vae/lag_attn_rws/configs/<arm>.yaml
```

| Config | Swept key | Value | Note |
|---|---|---|---|
| `sweep_beta_0p1.yaml` | `beta_schedule.end` | 0.1 | |
| `sweep_beta_0p3.yaml` | `beta_schedule.end` | 0.3 | |
| `sweep_beta_1p0.yaml` | `beta_schedule.end` | 1.0 | equals the default |
| `sweep_beta_3p0.yaml` | `beta_schedule.end` | 3.0 | |
| `sweep_dz_24.yaml` | `d_z` | 24 | |
| `sweep_dz_32.yaml` | `d_z` | 32 | |
| `sweep_dz_48.yaml` | `d_z` | 48 | |
| `sweep_dz_64.yaml` | `d_z` | 64 | equals the default |
| `sweep_dz_96.yaml` | `d_z` | 96 | |
| `sweep_reach_null.yaml` | `causal_reach_budget_s` | null | unguarded |
| `sweep_reach_240.yaml` | `causal_reach_budget_s` | 240 | also `warmup_period: 60` (§7) |
| `sweep_reach_120.yaml` | `causal_reach_budget_s` | 120 | equals the default |
| `sweep_reach_60.yaml` | `causal_reach_budget_s` | 60 | `up_ph` fully pruned |
| `sweep_reach_32.yaml` | `causal_reach_budget_s` | 32 | |
| `sweep_enc_kernel_7.yaml` | `encoder_extra_kernel` | 7 | |
| `sweep_norm_groups_1.yaml` | `conv_norm_groups` | 1 | per-timestep channel norm; no parameter change |
| `sweep_query_logvar.yaml` | `query_uses_logvar` | true | query reads $[\mu^p \,\|\, \ell^p]$ |
| `sweep_horizon_depth_5.yaml` | `horizon_depth` | 5 | |
| `sweep_init_off.yaml` | (init bundle) | off | multi-key: `horizon_embed_std` 0.02, `head_init_calibration` false, `a_head_gain` 1.0 |

`tests/test_sweep_configs.py` checks the arm files. Arms of the reach sweep build different
input-adapter widths, so they cannot share checkpoints.

Three further configs are not sweep arms. `tiny.yaml` is the one-epoch local smoke variant.
`smoke_hie.yaml` runs the shipped geometry on the committed HIE sample shard on one device.
`smoke_hie_control.yaml` is its control, with `beta_prior: 0.0` and `head_init_calibration: false`.

The collapse criterion is defined in `collapse.py` (`is_collapsed`). `eval/verify.py` applies it
to the per-epoch tail of a run's training `metrics_history.csv`.
