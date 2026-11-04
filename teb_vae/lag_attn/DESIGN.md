# `teb_vae/lag_attn` — design contract

This document is the standalone contract for the lag-attention VAE. It gives the tensor interface,
the geometry, the forward return dict, the loss, and the design choices that the code fixes. The
long-form description of each module is
[`nets/vae-teb-lag-attn-model.md`](nets/vae-teb-lag-attn-model.md).

This is the contract for the *model*. The evaluation pipeline, which reads a trained checkpoint
back, has its own contract: [`eval/EVAL.md`](eval/EVAL.md) describes what each analysis measures
and which interpretation rules it enforces, and [`eval/FIGURE_GUIDE.md`](eval/FIGURE_GUIDE.md)
describes how to read what a run emits.

## 1. What the model is

`SeqVaeLagAttn` (`teb_vae/lag_attn/nets/model.py`) is a source-pure, causal, residual VAE for
fetal-monitoring feature streams. It reads a target stream $Y$ (FHR features) and a source stream
$U$ (uterine-pressure features). At every decimated step $t$ it forecasts the target's near future
$Y^{+}_t = Y_{t+1:t+1+H_d}$ as a baseline prediction plus a source-conditioned residual:

$$\hat{Y}^{\mathrm{full}}_t = \hat{Y}^{\mathrm{base}}_t + \Delta\hat{Y}^{\mathrm{src}}_t .$$

The latent $z_t$ carries only what the source adds. Its per-step KL,

$$K_t = \mathrm{KL}\!\left(q(z_t \mid Y_{\le t}, U_{\le t}) \;\|\; p(z_t \mid Y_{\le t})\right),$$

is reported as a transfer-entropy surrogate. The delta heads of the posterior are zero-initialised,
so at initialisation $q \equiv p$ and $K_t = 0$; any later nonzero $K_t$ comes from source
conditioning that the model learned. A lag cross-attention over a sliding window of past source
states supplies the residual, and the attention-weighted attribution of $K_t$ across lags is
exposed as `te_lag_map`.

The model is *source-pure*: the source pathway sees only $U$, never a cross-channel field. A
permutation of $U$ across the batch is therefore a clean negative control (see
`teb_vae/lag_attn/nets/controls.py`).

## 2. Input contract

`forward(y_st, y_ph, u_stream, *, lag_band_mask=None)` takes floating tensors:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `y_st` | $(B, T, 43)$ | target FHR scattering features |
| `y_ph` | $(B, T, 66)$ | target FHR phase-harmonic features; concatenated to $Y \in \mathbb{R}^{c_y}$, $c_y = 109$ |
| `u_stream` | $(B, T, c_u)$ | source UP stream. $c_u = 58$ as `[up_st(43), up_ph(15)]` when `use_up_st=True`; $c_u = 15$ (`up_ph` only) when `False` |
| `lag_band_mask` | $(L,)$ or $(T, L)$, bool | optional keep-mask in lag order; `True` keeps. `None` is a bit-exact no-op |

The task checks the $c_y$ / $c_u$ widths **against every batch**, at the data boundary rather than
at construction. `SeqVaeLagAttnTask._build_source_stream` (via `_checked_source`) and
`_build_target_streams` compare the configured values with the batch's per-field channel counts
and raise `RuntimeError` naming both. The check runs on every batch because a run can read a list
of shards, and the shards can come from different pipeline revisions. The widths are properties
of the dataset, and the constructor cannot see the dataset.

> **The number 58 is ambiguous.** It is the current `use_up_st=true` width ($43 + 15$) and was
> the width of the old phase-only selection. Decide from `use_up_st` first, then set the width;
> never migrate `c_u` by pattern-matching the number.

Both UP fields are first-class dataset entries; there is no slicing fallback. The constructor runs
the guards that do not depend on the dataset: `num_heads * d_head == d_model`, `max_lag >= 0`,
`d_z % num_heads == 0` when `head_structured_latent=True`, and `c_y, c_u >= 1`. The last guard
matters because `nn.Linear(0, d_model)` is legal and returns its bias, so a zero width would build
a model that never reads that stream. `forward` carries no runtime shape assertion.

Assembling `u_stream` from a batch is the task's job, not the caller's.
`SeqVaeLagAttnTask._build_source_stream` (`teb_vae/lag_attn/task.py`) concatenates `[up_st, up_ph]`
or returns `up_ph` alone. It raises a clear error that names the missing field and the config key
that fixes it.

**Causal input guard.** The stored features are two-sided wavelet transforms, so the value at step
$t$ contains raw signal from after $t$. The config key `causal_reach_budget_s` sets a per-channel
reach budget in seconds. The trainer resolves it (`channel_reach.resolve_stream_budgets`) into four
constructor arguments: `target_keep_index`, `target_delays`, `source_keep_index` and
`source_delays`. Inside `forward`, a `ChannelGate` (`nets/delays.py`) keeps the channels whose
forward reach fits the budget and delays each survivor by
$\delta_c = \lceil \mathrm{reach}_c / \Delta \rceil$ steps. The resolver raises if the maximum delay
exceeds `warmup_period`. The batch must still carry the full declared widths, because the gather
runs after the data-boundary check. With no budget, no gate is built and the model reads every
channel with no delay. `AvailabilityInputAdapter` (`nets/encoders.py`) projects each gated stream
and adds an availability term for the zero-filled delayed prefix.

## 3. Geometry

The preprocessing and the shipped config fix these values:

| Symbol | Value | Meaning |
| --- | --- | --- |
| $f_s$ | $4$ Hz | raw sampling rate |
| decimation | $16$ | raw $\to$ decimated; $\Delta t = 4$ s |
| $T$ | $300$ | decimated steps ($20$ min) |
| $H_d$ | $30$ | forecast horizon ($120$ s) |
| warmup | $30$ | leading steps excluded from the KL and the feature loss |
| `max_lag` | $90$ | attention window is $L = \mathrm{max\_lag} + 1 = 91$ lags |
| $c_y$ | $109$ | target channels ($43$ scattering $+\ 66$ phase-harmonic) |
| $c_u$ | $58$ or $15$ | source channels (see §2) |
| $d_{\mathrm{model}}$ | $128$ | encoder / attention width |
| $d_z$ | $24$ | latent width |
| $M$ | $4$ | attention heads ($d_{\mathrm{head}} = 32$) |

The tiny test geometry (`teb_vae/lag_attn/tests/conftest.py`) shrinks $T$, $d_{\mathrm{model}}$,
$d_z$, $H_d$ and `max_lag`. It keeps every channel count and every invariant.

## 4. Forward return dict

`forward` returns a dict of $24$ tensors ($B$ batch, $T$ steps, $L = 91$ lags, $H_d$ horizon, $M$
heads):

| Key | Shape | Meaning |
| --- | --- | --- |
| `mu_prior` | $(B, T, d_z)$ | prior mean $\mu^p$ |
| `logvar_prior` | $(B, T, d_z)$ | prior log-variance $\ell^p$ (smooth-bounded) |
| `raw_logvar_prior` | $(B, T, d_z)$ | pre-bound prior log-variance (smooth bounding is not idempotent) |
| `mu_post` | $(B, T, d_z)$ | posterior mean $\mu^q$ |
| `logvar_post` | $(B, T, d_z)$ | posterior log-variance $\ell^q$ (residual around $\ell^p$ under `posterior_logvar_mode='residual'`) |
| `z` | $(B, T, d_z)$ | sampled posterior latent |
| `target_state` | $(B, T, d_{\mathrm{model}})$ | causal target encoder states $H^y$ |
| `source_state` | $(B, T, d_{\mathrm{model}})$ | causal source encoder states $H^u$ |
| `decoder_state` | $(B, T, d_{\mathrm{model}})$ | prior-head decoder state |
| `attended_source` | $(B, T, d_{\mathrm{model}})$ | fused attended source summary $A$ |
| `attended_source_heads` | $(B, T, M, d_{\mathrm{head}})$ | per-head source summaries |
| `attn_weights` | $(B, T, M, L)$ | lag attention $\alpha$, in lag order |
| `mu_base` | $(B, T, H_d, c_y)$ | baseline forecast mean |
| `logvar_base` | $(B, T, H_d, c_y)$ | baseline forecast log-variance |
| `delta_mu_src` | $(B, T, H_d, c_y)$ | source residual mean |
| `mu_full` | $(B, T, H_d, c_y)$ | full forecast mean $= \hat{Y}^{\mathrm{base}} + \Delta\hat{Y}^{\mathrm{src}}$ |
| `logvar_full` | $(B, T, H_d, c_y)$ | full forecast log-variance |
| `kld_per_t` | $(B, T)$ | per-step KL $K_t$ |
| `kld_per_t_per_head` | $(B, T, M)$ | per-head KL groups (an additive decomposition only with a head-structured latent) |
| `te_lag_map` | $(B, T, L)$ | attention-weighted lag attribution of $K_t$ |
| `warmup_mask` | $(T,)$, bool | `True` outside the warmup region |
| `mu_prior_sat_frac` | scalar | fraction of the prior mean near its $\tanh$ bound |
| `delta_mu_sat_frac` | scalar | fraction of the posterior-delta mean near its bound |
| `kld_active_frac` | scalar | fraction of latent dims with meaningful KL |

`encode_only(y_st, y_ph, u_stream, sample_z=True, *, lag_band_mask=None)` runs the encoders, the
attention and the latent heads. It skips the decoders and the diagnostics, and returns the $11$-key
subset: `mu_prior`, `logvar_prior`, `mu_post`, `logvar_post`, `z`, `target_state`, `source_state`,
`decoder_state`, `attended_source`, `attended_source_heads`, `attn_weights`. With
`sample_z=False`, `z` is the posterior mean rather than a reparameterized sample.

## 5. Loss

`compute_loss(forward_outputs, y_st, y_ph, *, weight, compute_kld_loss, beta, lambda_full,
lambda_base, likelihood, sigma_obs, free_bits, detach_baseline_in_full, lambda_lag)` returns

$$L = \lambda_{\mathrm{full}} L_{\mathrm{feat}} + \lambda_{\mathrm{base}} L_{\mathrm{base}}
     + \beta L_{\mathrm{KL}} + \lambda_{\mathrm{lag}} L_{\mathrm{smooth}}.$$

The two feature terms run over the valid anchor support $t \in [\mathrm{warmup}, T - H_d)$. The KL
term runs over the support that `kld_support` selects. $L_{\mathrm{base}}$ is load-bearing: without
it, the model could push target-explainable variance through the latent and inflate $K_t$ with
information the source never supplied. The KL comes back as both `kld_train` (the free-bit-floored
quantity that enters the loss) and `kld_raw` (the un-floored KL over the same support). Only
`kld_raw` / `kld_per_t` may be read as the transfer-entropy surrogate. The task schedules $\beta$
per epoch (`SeqVaeLagAttnTask._resolve_beta`) and drives the whole objective through the
framework's `compute_loss_and_metrics` seam.

## 6. Fixed design choices

**Smooth bounding is unconditional.** Every log-variance passes through `smooth_bound`. The
constructor has no `logvar_bound` or `posterior_logvar` argument; it is keyword-only with no
`**kwargs`, so passing either raises `TypeError`.

**The posterior log-variance has two modes.** `posterior_logvar_mode='residual'` (the constructor
default and the shipped value) adds a bounded delta to the prior's pre-bound raw log-variance, so
$q \equiv p$ exactly at initialisation. `'independent'` gives the posterior its own head. That
head is seeded at the pre-image of log-variance $0$, which requires $0$ inside `logvar_clamp`. This
model's prior head is not calibrated to log-variance $0$ at initialisation, so under
`'independent'` the KL at initialisation is not exactly zero.

**Research axes stay as constructor arguments.** These are `kld_support`
$\in \{\text{full}, \text{anchor}\}$, `lambda_perm` and `perm_every_n_batches` (the
source-permutation control; the shipped config sets `lambda_perm = 0`, so it is a readout only),
`use_entmax` (constructor default `False`; the shipped config sets `true`), `source_dropout`, and
`causal_norm`.

**`compile_model = False` is a hard requirement, not a default.** The task forces it in its
constructor. Three things in this net defeat TorchInductor independently: the `nn.LSTM` encoders,
the `torch.utils.checkpoint` region in the attention (under `attention_grad_checkpoint`), and the
data-dependent boolean-mask indexing behind `kld_active_frac`, which runs on every forward.

**The spike breaker runs a non-finite guard and an additive test, with the relative test off.**
The framework breaker's relative test is
$\text{watched} > \text{multiplier} \cdot \max(\mathrm{EMA}, \mathrm{ema\_floor})$, which assumes a
loss bounded below by zero. This model's `main_loss` is a Gaussian NLL with a learned observation
variance and can be negative. Once the EMA is negative, $\max(\mathrm{EMA}, 0)$ is $0$, so every
positive batch reads as a spike. The shipped config therefore sets `ema_floor: 1.0e+9`, which
switches the relative test off. The non-finite guard never consults the threshold, so it stays
active and stops a NaN loss from writing NaN gradients into every weight. The shipped
`additive_margin: 3.0` adds a sign-agnostic test: a batch is skipped when
$\text{watched} > \mathrm{EMA} + \mathrm{margin}$, compared against the raw EMA.
`max_consecutive_skips: 25` force-accepts the next finite batch after that many consecutive skips.

## 7. Causal normalisation and the permutation control

- **Causal normalisation.** A non-causal `GroupNorm` over a $(B, C, T)$ tensor pools statistics
  across time, so $H^y[t]$ becomes a function of $Y_{>t}$. The "prior" then conditions on the
  future, and $K_t$ is not a transfer-entropy surrogate. `causal_norm=True` replaces every encoder
  `GroupNorm` with `CausalGroupNorm`, which normalises each timestep independently.
- **The permutation control.** The shipped config runs the control as a *readout* at
  $\lambda_{\mathrm{perm}} = 0$ rather than as a training term. `perm_kl_from_forward` reuses the
  states that a completed forward already computed. The whole control therefore stays inside one
  forward and one backward, so automatic optimisation (and with it gradient clipping,
  accumulation, LR scheduling and the spike breaker) is kept.

## 8. Differences from the predecessor architecture

`SeqVaeLagAttn` is a flattened rebuild of an earlier three-level inheritance chain. It is not a
subclass of anything and is not checkpoint-compatible with the predecessor. The rebuild made these
intentional changes:

- **Encoder wrappers removed.** Each encoder was wrapped in a class whose only content was a `body`
  attribute. State-dict keys changed from `{target,source}_encoder.body.X` to
  `{target,source}_encoder.X`. The model passes the encoder kernels explicitly. Parameters are
  unchanged.
- **Latent-statistics buffers removed.** `mu_post_running_{mean,var,count}` backed a
  latent-normalisation mechanism with no consumer. The three buffers are gone; the parameter count
  is unchanged.
- **`raw_future_pred` removed.** The forward dict no longer carries this `None` entry, which came
  from a stub decoder that was never constructed.
- **Construction flags removed.** `posterior_logvar` and `logvar_bound` are gone (see §6).
- **Dead lag bank removed.** `LagMemoryBankBuilder` was constructed and never called. Strided views
  over the projected keys replace it. The `max_lag >= 0` check it carried now lives in the model's
  constructor.

No test asserts structural parity with the predecessor. A transposed weight or a reordered
residual preserves every shape and count, so a shape-level comparison could not detect it.

## 9. Framework conventions

The framework's authoring checklist (`train/MODEL_MIGRATION_GUIDE.md` §3.9) maps onto this tree as
follows:

| Item | Where |
| --- | --- |
| Files placed per §1.1 | `teb_vae/lag_attn/` — `nets/`, `task.py`, `trainer.py`, `configs/`, `tests/` |
| Raw `nn.Module` has no Lightning, no config, no I/O | `nets/*.py`, enforced by `tests/test_nets_are_framework_free.py` |
| Wrapper implements `compute_loss_and_metrics` | `task.py` — no `training_step` / `forward` / `configure_optimizers` override |
| `super().__init__(base_model, ...)` — no grandparent bypass | `task.py` (`compile_model=False` passed as a kwarg) |
| `self.model` for forward, `self.orig_model` everywhere else | `task.py` — `orig_model.compute_loss(...)` in the loss |
| `on_save_checkpoint` calls `super()` first | `task.py::on_save_checkpoint` — `super()` then adds `model_kwargs` |
| `create_model` checks the `load_checkpoint_strict` return value | `trainer.py::LagAttnTrainer.create_model` (`None` raises) |
| `train_model` calls `build_trainer` | `trainer.py::LagAttnTrainer.train_model` |
| `ModelCheckpoint(dirpath=self.model_checkpoint_dir, filename=...)` | `trainer.py` — `filename="lag-attn-{epoch:02d}"` |
| DDP strategy via the un-prefixed `select_ddp_strategy` override | `trainer.py::LagAttnTrainer.select_ddp_strategy` — reads its inputs from config, not from the wrapper |
| Required config keys | `configs/default.yaml`, checked by `tests/test_config_load.py` |
| No hand seeding; `general_config.seed` is set | `configs/default.yaml` sets `seed` |
| `use_distributed_sampler: true` and no self-built sampler | `configs/default.yaml`; data via `GraphDataModule` |
| Entry point order ctor -> `setup_config()` -> `create_model()` -> `train_model()` | `trainer.py:main`, checked by `tests/test_main.py` |

Run the two suites from the repository root:

```
.venv/Scripts/python.exe -m pytest train/tests -q
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn/tests -q
```

`scripts/make_tiny_shard.py` regenerates the committed shard and stats fixtures. Use it when the
dataset's channel widths change.
