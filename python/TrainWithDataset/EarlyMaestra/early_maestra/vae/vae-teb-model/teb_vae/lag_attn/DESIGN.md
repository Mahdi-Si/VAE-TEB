# `teb_vae/lag_attn` — design record and contract

This document is the standalone contract for the lag-attention VAE. It is written to be read on its
own: the tensor interface, the geometry, the forward return dict, and what the flatten changed are
all here. Measured evidence is cited to `knowledge/`, not copied. The architecture-deviation list
is [in §8](#8-the-deviation-record); it used to live in code, and moved here when the structural
snapshot it annotated was retired.

It is the contract for the *model*. Reading a trained checkpoint back and asking whether it works
is a separate pipeline with its own contract: [`eval/EVAL.md`](eval/EVAL.md) for what each analysis
measures and which interpretation rules it enforces, and
[`eval/FIGURE_GUIDE.md`](eval/FIGURE_GUIDE.md) for how to read what a run emits — starting with the
seven traps that make a figure say something other than what it looks like.

## 1. What the model is

`SeqVaeLagAttn` (`teb_vae/lag_attn/nets/model.py`) is a source-pure, causal, residual VAE for
fetal-monitoring feature streams. It reads a target stream $Y$ (FHR features) and a source stream
$U$ (uterine-pressure features), and at every decimated step $t$ forecasts the target's near future
$Y^{+}_t = Y_{t+1:t+1+H_d}$ as a baseline prediction plus a source-conditioned residual:

$$\hat{Y}^{\mathrm{full}}_t = \hat{Y}^{\mathrm{base}}_t + \Delta\hat{Y}^{\mathrm{src}}_t .$$

The latent $z_t$ carries only what the source adds. Its per-step KL,

$$K_t = \mathrm{KL}\!\left(q(z_t \mid Y_{\le t}, U_{\le t}) \;\|\; p(z_t \mid Y_{\le t})\right),$$

is reported as a transfer-entropy surrogate: at initialization $q \equiv p$ so $K_t = 0$, and every
nat it later reports is earned by source conditioning rather than inherited from a random
prior/posterior mismatch. A lag cross-attention over a sliding window of past source states supplies
the residual, and the attention-weighted attribution of $K_t$ across lags is exposed as
`te_lag_map`.

The model is *source-pure*: the source pathway sees only $U$, never a cross-channel field, so a
permutation of $U$ across the batch is a clean negative control (see
`teb_vae/lag_attn/nets/controls.py`).

## 2. Input contract

`forward(y_st, y_ph, u_stream, *, lag_band_mask=None)` takes floating tensors:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `y_st` | $(B, T, 43)$ | target FHR scattering features |
| `y_ph` | $(B, T, 66)$ | target FHR phase-harmonic features; concatenated to $Y \in \mathbb{R}^{c_y}$, $c_y = 109$ |
| `u_stream` | $(B, T, c_u)$ | source UP stream. $c_u = 58$ as `[up_st(43), up_ph(15)]` when `use_up_st=True`; $c_u = 15$ (`up_ph` only) when `False` |
| `lag_band_mask` | $(L,)$ or $(T, L)$, bool | optional keep-mask in lag order; `True` keeps. `None` is a bit-exact no-op |

The $c_y$ / $c_u$ widths are checked **against every batch**, at the data boundary rather than
at construction: `SeqVaeLagAttnTask._build_source_stream` (via `_checked_source`) and
`_build_target_streams` compare the configured values with the batch's actual per-field channel
counts and raise `RuntimeError` naming both. Every batch, not just the first, because a run can
be given a *list* of shards and they are not guaranteed to come from one pipeline revision.
The widths are properties of the dataset, and the constructor cannot see one — an earlier
version compared them against module constants, which went stale the moment the pipeline's
phase-harmonic selection changed, and failed as an opaque shape error inside an `InputAdapter`.

> **The number 58 is ambiguous across that change.** It is the current `use_up_st=true` width
> ($43 + 15$) and was the old `use_up_st=false` width. Decide from `use_up_st` first, then set the
> width; never migrate `c_u` by pattern-matching the number.

Both UP fields are first-class dataset entries; there is no slicing fallback. Four guards still
run at construction — the ones that do not depend on the dataset: `num_heads * d_head ==
d_model`, `max_lag >= 0`, `d_z % num_heads == 0` when `head_structured_latent=True`, and
`c_y, c_u >= 1`. The last is not cosmetic: `nn.Linear(0, d_model)` is legal and returns its
bias, so a zero width would build a model that trains to completion having never read that
stream. `forward` carries no runtime shape assertion.

Assembling `u_stream` from a batch is the task's job, not the caller's:
`SeqVaeLagAttnTask._build_source_stream` (`teb_vae/lag_attn/task.py`) concatenates `[up_st, up_ph]`
or returns `up_ph` alone and raises a clear error naming the missing field and the config key that
fixes it.

## 3. Geometry

Fixed by the preprocessing and the shipped config:

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
$d_z$, $H_d$ and `max_lag` while keeping every channel count and every invariant, so it exercises
each code path a production-scale model does.

## 4. Forward return dict

`forward` returns a dict of $24$ tensors ($B$ batch, $T$ steps, $L = 91$ lags, $H_d$ horizon, $M$
heads):

| Key | Shape | Meaning |
| --- | --- | --- |
| `mu_prior` | $(B, T, d_z)$ | prior mean $\mu^p$ |
| `logvar_prior` | $(B, T, d_z)$ | prior log-variance $\ell^p$ (smooth-bounded) |
| `raw_logvar_prior` | $(B, T, d_z)$ | pre-bound prior log-variance (smooth bounding is not idempotent) |
| `mu_post` | $(B, T, d_z)$ | posterior mean $\mu^q$ |
| `logvar_post` | $(B, T, d_z)$ | posterior log-variance $\ell^q$ (residual around $\ell^p$) |
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
| `kld_per_t_per_head` | $(B, T, M)$ | per-head KL groups (head-structured latent only) |
| `te_lag_map` | $(B, T, L)$ | attention-weighted lag attribution of $K_t$ |
| `warmup_mask` | $(T,)$, bool | `True` outside the warmup region |
| `mu_prior_sat_frac` | scalar | fraction of the prior mean near its $\tanh$ bound |
| `delta_mu_sat_frac` | scalar | fraction of the posterior-delta mean near its bound |
| `kld_active_frac` | scalar | fraction of latent dims with meaningful KL |

`encode_only(y_st, y_ph, u_stream, *, sample_z=True)` runs the encoders, attention and latent heads
but skips the decoders and diagnostics, returning the $11$-key subset: `mu_prior`, `logvar_prior`,
`mu_post`, `logvar_post`, `z`, `target_state`, `source_state`, `decoder_state`, `attended_source`,
`attended_source_heads`, `attn_weights`. With `sample_z=False`, `z` is the posterior mean rather
than a reparameterized sample.

## 5. Loss

`compute_loss(forward_outputs, y_st, y_ph, *, beta, lambda_full, lambda_base, likelihood, sigma_obs,
free_bits, detach_baseline_in_full, lambda_lag)` returns

$$L = \lambda_{\mathrm{full}} L_{\mathrm{feat}} + \lambda_{\mathrm{base}} L_{\mathrm{base}}
     + \beta L_{\mathrm{KL}} + \lambda_{\mathrm{lag}} L_{\mathrm{smooth}},$$

over the valid anchor support $t \in [\mathrm{warmup}, T - H_d)$. $L_{\mathrm{base}}$ is
load-bearing: without it, target-explainable variance could be pushed through the latent, inflating
$K_t$ with information the source never supplied. The reported KL comes back as both `kld_train` (the
free-bit-floored quantity that enters the loss) and `kld_raw` (the un-floored KL over the same
support) — only `kld_raw` / `kld_per_t` may be read as the transfer-entropy surrogate. The task
schedules $\beta$ per epoch (`SeqVaeLagAttnTask._resolve_beta`) and drives the whole objective
through the framework's `compute_loss_and_metrics` seam.

## 6. What the flatten changed

This tree was flattened out of a three-level inheritance chain in the now-deprecated
`model/vae_teb_prediction/` tree. Behaviour is preserved; the differences are deliberate and
recorded. That predecessor is retained only as a frozen historical record — this model is
standalone, is not a subclass of anything, and is not checkpoint-compatible with it.

**Parity flags now unconditional (branches and flags deleted).** Smooth log-variance bounding and a
residual posterior log-variance are the design, not options — passing `logvar_bound` or
`posterior_logvar` to the constructor now raises `TypeError`. This supersedes §22 of
`knowledge/vae-teb-lag-attn-v3-implemented-model.md`, which recorded every predecessor flag as
defaulting to its parity value; that record describes the old tree and stays frozen.

**Parity axes kept as constructor kwargs** because they are real research axes, not parity toggles:
`kld_support` $\in \{\text{full}, \text{anchor}\}$, `lambda_perm` and `perm_every_n_batches` (the
source-permutation control; ships at `lambda_perm = 0` as a *readout*, see §7), and `use_entmax`
(the entmax15-vs-softmax normalizer, which ships **on** — it is not optional, and the stale "entmax
remains optional" claim in that knowledge doc must not be carried forward).

**`compile_model = False` is a hard requirement, not a default.** The task forces it in its
constructor. Three things in this net defeat TorchInductor independently: the `nn.LSTM` encoders, the
`torch.utils.checkpoint` region in the attention (under `attention_grad_checkpoint`), and the
data-dependent boolean-mask indexing behind `kld_active_frac`, which runs on every forward.

**The spike breaker ships as a non-finite guard only.** The framework breaker's relative test is
$\text{watched} > \text{multiplier} \cdot \max(\mathrm{EMA}, \mathrm{ema\_floor})$, which assumes a
loss bounded below by zero. This model's `main_loss` is a Gaussian NLL with a learned observation
variance and goes negative routinely; once the EMA is negative, $\max(\mathrm{EMA}, 0)$ is $0$ and
every positive batch reads as a spike, so the run silently drops its hardest batches and no value of
`ema_floor` rescues the relative test. The shipped config therefore sets `ema_floor: 1.0e9` — far
above any reachable loss — which switches the relative test off while leaving the non-finite guard
(which never consults the threshold) intact. That guard is the part worth keeping: it stops a NaN
loss writing NaN gradients into every weight. Revisit if `main_loss` is measured positive at
production scale.

## 7. Measured evidence (cited, not copied)

Two design choices rest on measurements recorded in
`knowledge/vae-teb-lag-attn-v3-implemented-model.md`; they are cited rather than duplicated, so there
is one source of truth:

- **Causal normalization (§8.5).** A non-causal `GroupNorm` pools statistics across time, leaking the
  future into $H^y[t]$: a measured relative leak of $11.5\%$ at $T{=}32$, falling to $0.0$ after the
  causal `GroupNorm` replaces it. Without this, $K_t$ is not a transfer-entropy surrogate at all.
- **The permutation control (§20.1).** A positive $\lambda_{\mathrm{perm}}$ collapsed the source
  pathway in half the seeds tried, so the control ships as a *readout* at $\lambda_{\mathrm{perm}} =
  0$ rather than as a training term. `perm_kl_from_forward` reuses the states a completed forward
  already computed, keeping the whole control inside one forward and backward so automatic
  optimization (and therefore gradient clipping, accumulation, LR scheduling and the spike breaker)
  survives.

## 8. The deviation record

This is the complete list of every intentional difference between the predecessor architecture
and this rebuild. Behaviour is preserved; each entry says what changed and why it is not a
defect.

- **Encoder wrappers removed.** The two encoders were each wrapped in a 31-line class whose only
  content was a `body` attribute holding the real encoder. The wrappers existed to shape
  state-dict keys, which stopped mattering once checkpoint compatibility was dropped. Their
  differing default kernels were never load-bearing: the model computes and passes kernels
  explicitly. Effect: `{target,source}_encoder.body.X` → `{target,source}_encoder.X`. Parameters
  unchanged.
- **Latent-statistics buffers removed.** `mu_post_running_{mean,var,count}` backed a
  latent-normalisation mechanism with no consumer in this tree. It was also the only thing in the
  model that imported `loguru` or `torch.distributed`, the only place a batch field name was read,
  and it cost four device syncs per training step. Effect: 3 buffers gone. They are buffers, not
  parameters, so the parameter count is unchanged.
- **`raw_future_pred` removed.** The forward dict carried `raw_future_pred: None` — a non-tensor
  in a dict of tensors — produced by a decoder that was a stub raising `NotImplementedError` and
  was never constructed. Effect: one forward key gone. No parameters were ever involved.
- **Construction flags removed.** `posterior_logvar` and `logvar_bound` selected between a parity
  branch and the shipped behaviour. The shipped values (`"residual"` and `"smooth"`) are now the
  only behaviour, so the flags are gone and passing either raises `TypeError`.
- **Dead lag bank removed.** `LagMemoryBankBuilder` was constructed on every model and never
  called; strided views over the projected keys replaced it. It held no parameters and no
  buffers. Note it also carried the only `max_lag >= 0` check, which the rebuild re-established in
  the model's constructor.

**The structural assertion behind this list has been retired.** It was
`teb_vae/lag_attn/tests/test_architecture_snapshot.py`, which compared the rebuild against
captured snapshots of the predecessor (sorted state-dict keys and shapes, total and trainable
parameter counts, and the forward and loss key sets) with everything not named above required to
match exactly. Those snapshots were captured at the predecessor's input geometry — $c_y = 87$,
$c_u = 101$ — and the model's input widths have since changed with the dataset (§3), so the
comparison no longer has a common geometry to make. The capture script read the deprecated
`model/vae_teb_prediction/` tree, which is frozen as a historical record and is not to be edited
to follow the new widths. The test, its two fixtures, and `scripts/capture_v3_architecture.py`
were therefore removed, and this section is now the record.

What that test could never catch, and what nothing catches now: a transposed weight or a
reordered residual, both of which preserve every shape and count. The predecessor tree remains on
disk, so a golden-tensor comparison against it stays available if a retrain ever diverges — at
the old widths.

## 9. Closeout — authoring checklist

The framework's authoring checklist (`train/MODEL_MIGRATION_GUIDE.md` §3.9) verified item by item
against this tree:

| Item | Verdict | Where |
| --- | --- | --- |
| Files placed per §1.1; nothing added to the §1.2 deviation table | yes | `teb_vae/lag_attn/` — `nets/`, `task.py`, `trainer.py`, `configs/`, `tests/` |
| Raw `nn.Module` has no Lightning, no config, no I/O | yes | `nets/*.py`, enforced by `tests/test_nets_are_framework_free.py` |
| Wrapper implements `compute_loss_and_metrics` and nothing else it did not have to | yes | `task.py` — no `training_step` / `forward` / `configure_optimizers` override |
| `super().__init__(base_model, ...)` — no grandparent bypass | yes | `task.py` (`compile_model=False` passed as a kwarg) |
| `self.model` for forward, `self.orig_model` everywhere else | yes | `task.py` — `orig_model.compute_loss(...)` in the loss |
| `on_save_checkpoint` (overridden) calls `super()` first | yes | `task.py::on_save_checkpoint` — `super()` then adds `model_kwargs` |
| `create_model` checks the `load_checkpoint_strict` return value | yes | `trainer.py::LagAttnTrainer.create_model` (`None` -> raises rather than training silently untrained) |
| `train_model` calls `build_trainer`; no hand-rolled `pl.Trainer` | yes | `trainer.py::LagAttnTrainer.train_model` — first production caller of `build_trainer` |
| `ModelCheckpoint(dirpath=self.model_checkpoint_dir, filename=...)` | yes | `trainer.py` — `dirpath=self.model_checkpoint_dir`, model-specific `filename="lag-attn-{epoch:02d}"` |
| DDP strategy via the un-prefixed `select_ddp_strategy` override | yes | `trainer.py:191` — sources its inputs from config, not the wrapper |
| Config has all fourteen effectively-required keys | yes | `configs/default.yaml`, asserted by `tests/test_config_load.py` |
| No hand seeding; `general_config.seed` is set | yes | `configs/default.yaml` sets `seed`; no `manual_seed` / `seed_everything` in the tree |
| `use_distributed_sampler: true` and no self-built sampler | yes | `configs/default.yaml`; data via `GraphDataModule` (no `world_size`) |
| Entry point order ctor -> `setup_config()` -> `create_model()` -> `train_model()` | yes | `trainer.py:main`, asserted by `tests/test_main.py` |
| The four verifications of §3.7 exist and pass | yes | the new-tree suite, incl. `tests/test_train_smoke.py` |

Both live suites pass:

```
.venv/Scripts/python.exe -m pytest train/tests -q
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn/tests -q
```

(At closeout a third suite, the predecessor tree's `model/vae_teb_prediction/model/tests`, also
passed. That tree is deprecated and its suite is no longer part of this model's gate.)

`scripts/make_tiny_shard.py` is kept: it keeps the committed shard and stats fixtures reproducible,
and is the only supported way to regenerate them when the dataset's channel widths change. Its
companion `scripts/capture_v3_architecture.py` was removed along with the structural snapshot (§8).
