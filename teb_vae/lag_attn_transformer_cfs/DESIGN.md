# `lag_attn_transformer_cfs` — design record

This package is the conv-Transformer, causal-feature lag-attention VAE-TEB. This document states
which parent supplies each part of the model, what the model reads, what it returns, what it
optimises, and how to configure and run it.

Companion documents, not restated here:

- `teb_vae/lag_attn_cfs/DESIGN.md` records the causal target domain, the warm-up budget, the anchor
  tiling and the readouts that this model composes by import.
- `teb_vae/lag_attn_transformer_rws/DESIGN.md` records the encoders, the adapters, the attention
  blocks and the wiring.
- `teb_vae/lag_attn_rws/DESIGN.md` records the architecture that the whole family shares.
- `eval/EVAL.md` records what the evaluation binding of this package supplies.
- `LATENT_CLASS_FINETUNING_PILOT.md` records the `latent_pilot/` sub-package.

---

## 1. What the model is

The package is one cell of an encoder-by-target grid:

```
                          conv-LSTM encoders          conv-Transformer encoders
  raw FHR target          lag_attn_rws                lag_attn_transformer_rws
  two-sided feature       lag_attn_fs                 lag_attn_transformer_fs
  causal feature          lag_attn_cfs                lag_attn_transformer_cfs   <- this
```

At each admitted 4-second anchor $t$, the model forecasts the next $H$ stored steps of the
one-sided FHR feature block. It makes two forecasts: one from a target-only latent (the base
branch) and one from a source-conditioned latent (the full branch). One shared decoder makes both
forecasts under one noise draw.

The whole model is one class:

```python
class SeqVaeLagAttnTrfCfs(CausalWarmupInputs, CausalFeatureForecastTarget, SeqVaeLagAttnTrfRws): ...
```

- `SeqVaeLagAttnTrfRws` (`teb_vae/lag_attn_transformer_rws/nets/model.py`) supplies the
  architecture: the causal conv-Transformer target encoder, the channel gates, the
  availability-aware input adapters, the source key/value path, the prior and posterior heads, the
  lag cross-attention, the horizon decoder and the paired reparameterisation.
- `CausalWarmupInputs` (`teb_vae/lag_attn_cfs/nets/causal_inputs.py`) supplies the input warm-up
  mask, the lag validity floor, the tiled anchor set, the UA quality key mask and the forward that
  decodes at the anchors.
- `CausalFeatureForecastTarget` (`teb_vae/lag_attn_cfs/nets/causal_feature_target.py`) supplies the
  decoder width, the gathered one-sided target block, the per-block channel weights, the scored
  horizon mask, the AR(1) residual likelihood and the causal readouts.

The class defines a constructor and nothing else. The constructor exists because the experiment
driver builds the model kwargs from `inspect.signature(MODEL_CLS.__init__)`, and this architecture
has a keyword schema that differs from the conv-LSTM cell's.

## 2. Package layout

```
nets/model.py              SeqVaeLagAttnTrfCfs: two imported mixins over an imported architecture
nets/arch_viz/             traced, interactive architecture page (see nets/arch_viz/README.md)
task.py                    SeqVaeLagAttnTrfCfsTask: a diamond of two parent tasks, empty body
trainer.py                 LagAttnTrfCfsTrainer: three re-pointed class attributes, and main()
adverse_finetune.py        premise check, fine-tune, evaluate and classify on one k-fold split
configs/                   YAML configs (default, fixtures, sweep arms, production overlays)
eval/                      evaluation binding, runner, gate and the quality-arm scripts
latent_pilot/              outcome-separation pilot on the pretrained latents
docs/architecture/         the architecture note (.tex and .pdf)
docs/latex_template/       the LaTeX report of this model and its TikZ figures
tests/                     pytest suite
```

## 3. Class composition

### 3.1 The model

The model composes two mixins and not the conv-LSTM causal model. `SeqVaeLagAttnCfs` subclasses
`SeqVaeLagAttnRws`, while `SeqVaeLagAttnTrfRws` derives from `nn.Module` directly. Thus
`class X(SeqVaeLagAttnCfs, SeqVaeLagAttnTrfRws)` would run the conv-LSTM constructor. The two
mixins in `teb_vae/lag_attn_cfs/nets/` do not refer to an encoder, so both causal cells compose
them.

The method resolution order is
`SeqVaeLagAttnTrfCfs -> CausalWarmupInputs -> CausalFeatureForecastTarget -> FeatureForecastTarget
-> SeqVaeLagAttnTrfRws -> Module -> object`.

**The order of the bases is load-bearing.** The mixins come first, so their width hook
(`_default_decoder_out_channels`) and their tiled forward take precedence over the architecture
parent's. With the order reversed, the decoder is built at $R = 16$ raw samples per step and
`raw_sample_score` fails on a shape mismatch. `tests/test_construct.py` asserts that the decoder's
`out_features` follow the target gate.

**The decoder width is a method, not a constructor keyword.** The gate that sets the width is
built inside the base constructor. Also, a subclass that narrows `__init__` to intercept a keyword
breaks the signature sweep in `trainer._build_model_kwargs`. There is no `decoder_out_channels`
keyword; the width follows from the stamped `target_keep_index`.

**The constructor holds no validation of its own.** It captures its keyword arguments, calls
`_set_causal_inputs`, `_set_channel_weights`, `_set_target_novelty` and
`_set_likelihood_structure` before the base constructor, and calls `_validate_causal_geometry`,
`_register_channel_weights` and `_register_likelihood_structure` after it. The target-domain
keywords are:

| Keyword | Meaning |
| --- | --- |
| `target_keep_index`, `source_keep_index` | declared channel indices that survive the warm-up budget |
| `target_warmup_steps`, `source_warmup_steps` | $W'_c$ per surviving channel; `None` builds no warm-up mask |
| `target_align_delays`, `source_align_delays` | optional alignment shift $d_c$ per surviving channel, forwarded to the base as `target_delays` / `source_delays` |
| `anchor_stride` | $S$, the spacing of decoded training anchors, in $[1, H]$ |
| `lag_floor` | $F_u$, the earliest source step that lag attention may read |
| `target_weight_st`, `target_weight_ph` | relative reconstruction weight of the two stored target blocks |
| `target_novelty_frac` | per-channel novelty share, a readout only |
| `target_forecast_shift` | per-channel signed re-indexing of the scored element under a non-stored forecast clock |
| `target_scored_horizon` | $H_c$ per declared target channel |
| `forecast_ar_residual` | score the horizon under a per-channel AR(1) residual |
| `source_quality_mask` | apply the UA quality mask to the lag keys |

The trainer resolves the warm-up tuples, the alignment shifts, the forecast shift and
`target_scored_horizon` from the config and the shards. They reach the checkpoint through
`model_kwargs`.

### 3.2 The task and the driver

```python
class SeqVaeLagAttnTrfCfsTask(SeqVaeLagAttnCfsTask, SeqVaeLagAttnTrfRwsTask):
    pass

class LagAttnTrfCfsTrainer(LagAttnCfsTrainer, LagAttnTrfRwsTrainer):
    MODEL_CLS = SeqVaeLagAttnTrfCfs
    TASK_CLS = SeqVaeLagAttnTrfCfsTask
    CHECKPOINT_STEM = "lag-attn-trf-cfs"
```

The task resolution order is `SeqVaeLagAttnTrfCfsTask -> SeqVaeLagAttnCfsTask ->
SeqVaeLagAttnFsTask -> SeqVaeLagAttnTrfRwsTask -> SeqVaeLagAttnRwsTask -> LightningModelBase`.
The causal parent supplies the tile phase, the stage-to-geometry resolution, the source-null KL
readout, the anchor-aware latent-gap diagnostic and the diagnostic-page seams. The
conv-Transformer parent supplies `build_lr_scheduler`, a step-granular learning-rate warm-up. The
task defines no `training_step`, so the framework's step runs the loss-spike breaker.
`tests/test_task.py` asserts that the two parent branches are disjoint.

The driver resolution order is `LagAttnTrfCfsTrainer -> LagAttnCfsTrainer -> LagAttnTrfRwsTrainer
-> LagAttnRwsTrainer -> GraphModelBase`. Both parents set `MODEL_CLS`, `TASK_CLS` and
`CHECKPOINT_STEM`, so the driver re-points all three. Without the re-pointing, the driver would
build the conv-LSTM model or write `lag-attn-cfs-*.ckpt` without an error.

| From the causal parent | From the conv-Transformer parent | From the shared driver |
| --- | --- | --- |
| `TARGET_FIELDS = ("fhr_st", "fhr_ph")` | `compile_model_requested` | `PLOT_CONFIG_KEY = "lag_attn_rws_plotting"` |
| `TRACKED_METRICS` | `_build_trainer_kwargs`, the step-granular LR monitor | the DDP strategy selection |
| `preflight` refusals | the encoder constructor refusals | the callback assembly |

`_build_model_kwargs` and `create_model` are defined on both parents, and both run through
`super()`. The conv-Transformer half re-admits `source_attention_window: null` and applies
`lr_warmup_steps`. The causal half adds the resolved warm-up tuples, the scored horizon, the
geometry log line and the seed for the tile phase. `tests/test_trainer.py` asserts that both halves
run and that `compile_model_requested` resolves to the conv-Transformer side.

## 4. Input contract

The model reads the causal HDF5 shards through `train/data_module.py::GraphDataModule` at
`trim_minutes: 1.0`. `teb_vae/lag_attn_cfs/DESIGN.md` §2 is the full record.

**Timing.** A stored segment holds $5280$ raw samples at $4$ Hz and $330$ feature steps at a
decimation factor of $16$, so one feature step is $\Delta = 4$ s. The one-minute trim removes $240$
raw samples and $15$ feature steps from each end, which leaves $T = 300$ steps and $4800$ raw
samples. The loader keeps `epoch` as the untrimmed segment start in seconds relative to delivery.
Thus trimmed step $t$ is at

$$\operatorname{time}(t) = \mathrm{epoch} + 60 + 4t \ \text{s}.$$

The causal transform runs separately on each segment, so each segment pays its own filter warm-up.

**Channel layout under the shipped integer phase operator** (`causal_phase_operator:
integer_harmonic_v1`):

| Block | Stored width | Stream |
| --- | ---: | --- |
| `fhr_st` | $36$ | target $Y$ |
| `fhr_ph` | $44$ | target $Y$ |
| `up_st` | $36$ | source $U$ |
| `up_ph` | $10$ | source $U$ |

So $c_y = 80$ and $c_u = 46$. The legacy operator (`ratio_power_v0`) stores $66$ `fhr_ph` and $15$
`up_ph` channels, so $c_y = 102$ and $c_u = 51$. The `fhr_up_ph` block is not read.

**Normalisation is pointwise.** The loader applies fixed dataset statistics: scattering channel
$0$ is standardised directly, the other scattering channels are log-transformed and standardised,
and the phase channels are asinh-transformed and standardised. The statistics file must match the
transform variant, the phase operator, the leg alignment, the trim and the quality version.

**Config requirements.**

- `fhr_st` and `fhr_ph` must be in `normalize_fields`, because they are the reconstruction target.
  The shared entry point checks this through the driver's `TARGET_FIELDS`.
- `guid` and `epoch` must be in `load_fields`, because the tile phase is keyed on the pair.
- The shards must be the causal variant. The warm-up resolver refuses a two-sided shard by name.

**Quality fields.** A quality build (`signal_quality=True`) stores bridged `fhr` / `up`, the raw
signals, per-sample quality bits and, per coefficient, the share $\rho$ of filter mass on bridged
samples. Such a build needs statistics computed over it; the loader refuses statistics of another
`quality_version`. `dataset_kwargs.ua_step_mask` and `dataset_kwargs.fhr_step_mask` turn the
quality fields into per-step validity (§8.5).

## 5. Geometry

### 5.1 Warm-up budget and keep indices

A one-sided filter's output depends on assumed pre-segment history until its warm-up $W_c$ has
passed. The loader rebases the warm-up to the trimmed window, $W'_c = \max(W_c - 15, 0)$. This is a
mask boundary, not a shift: the adapter zeroes a channel before $W'_c$ and gives the encoder an
availability representation.

`causal_warmup_budget_steps` ($B = 134$ at the shipped configuration) keeps every target channel
with $W'_c \le B$. At the shipped shards the gate keeps $76$ of the $80$ target channels (declared
indices $32$ to $35$ of `fhr_st` drop). `causal_source_channels` names the source channels
explicitly; the shipped list keeps $42$ of $46$ (it drops `up_st` channels $32$ to $35$).
`causal_reach_budget_s` must be `null`; the resolver refuses it together with the warm-up budget.

### 5.2 Anchors and tiling

The anchor floor is $F = $ `warmup_period` $= 134$, the largest $W'_c$ of the kept target
channels. Under the stored forecast clock the anchor ceiling is $T_{\mathrm{valid}} = T - H$, so
the dense anchor set is $[F, T_{\mathrm{valid}})$. At $H = 30$ that is $[134, 270)$, $136$ anchors.

Training decodes a tiled subset,

$$\mathcal A(\varphi) = \{F + \varphi + kS\} \cap [F, T_{\mathrm{valid}}), \qquad \varphi \in [0, S),$$

with $A_{\max} = \lceil (T_{\mathrm{valid}} - F) / S \rceil$ slots. A shorter phase pads its last
slot with a repeat and marks it invalid in `anchor_valid`. The phase $\varphi_b$ is derived per
sample from a `blake2b` hash of `guid`, `epoch`, the training epoch and the seed, so it draws no
random number. At the shipped $S = 15$, $A_{\max} = 10$. Validation and test decode the dense set
at phase $0$ and stride $1$.

`lambda_boundary` must be $0$. The boundary term compares adjacent anchors, and this family
decodes anchors $S$ steps apart; the pre-flight refuses a non-zero value.

### 5.3 Lag window

The lag attention searches $L = $ `max_lag` $+ 1$ lags, $\ell = 0, \dots, \ell_{\max}$. At the
shipped `max_lag: 37` that is $38$ lags and $\Delta(L - 1) = 148$ s. Lag $\ell$ at anchor $t$ reads
the source key at step $t - \ell$, masked where $t - \ell < F_u$ (`lag_floor`, shipped $0$). A lag is
a distance on the feature grid, $\Delta \ell$. The composed filter delays of the stored channels
separate it from a physiological delay.

### 5.4 Optional alignment and forecast clocks

The shipped configuration reads every channel at its stored index
(`causal_align_reference: null`, `causal_align_reference_source: null`) and scores the stored
clock (`causal_target_forecast_clock: stored`): anchor $t$, horizon step $\tau$, channel $c$ is
scored against stored step $t + 1 + \tau$.

Two arms change this:

- **Input alignment** (`causal_align_reference: target_max`, optionally with a source reference).
  The resolver shifts each kept channel onto a common reference clock with a non-negative delay
  $$d_c = \operatorname{round}\!\left(\frac{\kappa\,(\tau_{\mathrm{ref}} - \tau_c)}{\Delta}\right),
  \qquad \kappa = 1 - \frac{1}{2\gamma} = 0.875,$$
  where $\tau_c$ is the stored `causal_delay_s`, $\gamma = 4$ is the gammatone order and
  $\kappa$ is `ALIGNMENT_DELAY_FACTOR` in `teb_vae/lag_attn_cfs/causal_warmup.py`. $\kappa\tau_g$ is
  the energy centroid of a gammatone impulse response. The resolver drops channels above the
  reference rather than advance them. The input then is
  $X_c[t] = \mathbf 1_{t \ge W'_c + d_c}\, X_c[t - d_c]$.
- **Forecast clock** (`causal_target_forecast_clock: physical | input`). The resolver sets
  `target_forecast_shift` $s_c$, so the scored element reads stored step $t + 1 + \tau + s_c$. The
  `physical` clock advances the labels; the `input` clock copies the input alignment shift.

The dataset's phase-leg alignment (`causal_leg_alignment: envelope`) is a separate operation. It
acts inside each phase pair before the coefficients are stored, and the model settings above do not
change it.

## 6. Architecture

### 6.1 Target path

$Y_{\le t}$ $(B, T, c_y)$ passes the target channel gate ($80 \to 76$ at the shipped budget) and
the availability adapter (a `LayerNorm` and a linear map to $d_{\mathrm{model}} = 128$, plus a
learned availability projection). The target encoder then applies a convolution stem of two gated
causal depthwise blocks (`encoder_conv_kernels: [5, 9]`, `encoder_conv_dilations: [1, 2]`,
receptive field $1 + \sum_b (k_b - 1) r_b = 21$ steps) and `target_attention_blocks` ($6$)
causal self-attention blocks over the full prefix (`encoder_num_heads: 4`, rotary positions,
SwiGLU feed-forward of width `encoder_d_ff: 512`, `RMSNorm`). The output is the target state
$h^y_t$ $(B, T, d_{\mathrm{model}})$.

### 6.2 Source key/value path

`lag_kv_source` selects what the lag attention scores as keys and values:

| Value | Key/value representation | Reach |
| --- | --- | --- |
| `adapter` (shipped) | the source adapter output $h^u_t$, position-wise | $1$ step |
| `conv_stem` | a `GatedCausalConvStem` on the encoder's own convolution schedule | $21$ steps at the shipped schedule |
| `encoder` (constructor default) | the windowed source attention encoder (`source_attention_blocks`, `source_attention_window`) | $R_{\mathrm{conv}} + N_U (W_U - 1)$ steps |

`source_attention_blocks` and `source_attention_window` apply only under `lag_kv_source: encoder`.
They stay in the config so that the `encoder` arm and the config-parity tests can read them.
`source_dropout` ($0.2$ shipped) applies dropout on the source pathway only.

### 6.3 Prior, lag attention and posterior

**Prior.** The prior head maps $h^y_t$ to $p_\theta(z_t \mid Y_{\le t}) = \mathcal N(\mu^p_t,
\mathrm{diag}\, e^{\eta^p_t})$ with $d_z = 64$. With `prior_availability_input: true`, the prior also
reads the source availability clock $h^{\varnothing}_t$: the source adapter applied to an all-zero
source, detached. This term carries no source values. The prior never reads $h^u$.

**Lag attention.** The query comes from the prior mean, $q_t = W_Q \mathrm{LayerNorm}_q(W_{qp}
\mu^p_t)$ (and from the log-variance under `query_uses_logvar`). The score per head $m$ is

$$\zeta_{t,m,\ell} = \frac{\langle q_t, k_{t-\ell} \rangle + \langle q_t, e^{\mathrm{lag}}_{\ell,m}
\rangle}{\sqrt{d_{\mathrm{head}}}} + b_{m,\ell},$$

with `num_heads: 4`, `d_head: 32` and a learned lag embedding $e^{\mathrm{lag}}$.
`use_entmax: true` normalises with $1.5$-entmax, which gives exact zeros. `lag_bias_init:
alibi_decay` seeds $b_{m,\ell} = -s_m \ell$ with slopes scaled by `alibi_slope_scale`; the shipped
scale $0.0$ seeds a flat bias. The attended summary is $a^{(m)}_t = \sum_\ell \alpha_{t,m,\ell}
v^{(m)}_{t-\ell}$. The lag attention runs at `dropout=0.0`, so the lag attribution identity
$\sum_\ell \widetilde K_{t,\ell} = K_t$ holds.

**Posterior.** The posterior head reads $\mu^p_t$, $\eta^p_t$, $h^y_t$ and the attended summaries.
It is a bounded residual on the prior: $\mu^q_t = \mu^p_t + \Delta\mu_t$ with
$|\Delta\mu| < $ `delta_mu_scale` ($3$). Under `posterior_logvar_mode: independent` (shipped) the
posterior log-variance has its own head; under `residual` it is a bounded residual too. The latent
is head-structured: $d_z$ splits into `num_heads` groups, and group $m$ is written by head $m$ only.

**Event gate.** With `posterior_event_gate: true` (shipped), one scalar
$g_t = \sigma(w^\top \tilde a_t + b)$ per step, read from the attended summary before its dropout,
multiplies both residuals. At $g_t = 0$ the posterior equals the prior and $K_t = 0$. Under
`independent`, the posterior raw log-variance moves towards the prior's detached raw value. The
gate bias is seeded at $b = 2$. A strict load of a checkpoint without the gate fails;
`teb_vae/lag_attn_rws/seed_event_gate_checkpoint.py` writes a copy with the two gate tensors at
their constructor seed.

**Initialisation.** The posterior delta heads are zeroed after the generic initialisation, and
both branches share one $\epsilon$, so the KL at initialisation is exactly zero. Under
`independent`, this holds only with `head_init_calibration: true` (shipped).

### 6.4 Decoder

One `BaselineFutureDecoder` maps a latent $z_t \in \mathbb R^{d_z}$ to an $(H, C_{\mathrm{keep}})$
mean and log-variance. It applies a residual projection MLP to `decoder_hidden` ($256$), adds a
horizon embedding (`horizon_embed_std: 0.8`), runs `horizon_depth` ($4$) dilated refine blocks of
kernel `horizon_kernel` ($3$) with FiLM from the latent (`horizon_film: true`), runs
`horizon_attention_blocks` ($2$) unmasked horizon self-attention blocks, and applies the mean and
log-variance heads (clamped to `logvar_clamp`, shipped $[-6, 3]$). The refine stack's receptive
field is $31$ tokens at depth $4$, kernel $3$.

The decoder takes exactly one latent tensor, so no gradient reaches it from the source except
through $z$. It is invoked twice per anchor, on $z^p$ (base) and $z^q$ (full).
`base_decode: mean` (shipped) decodes the base branch at $\mu^p_t$ rather than at a sample.

**Persistence residual.** With `persistence_residual: true` (shipped), the mean head adds
$w^{\mathrm p}_{\tau,c}\, y_{t, c}$, the anchor's own kept target vector, weighted by a learned
$(H, C_{\mathrm{keep}})$ parameter seeded at $2^{-\tau/5}$. The gather is masked by each channel's
own warm-up. Both decoder invocations receive the same persistence tensor, and the tensor holds
target values only.

## 7. Forward return dict

`SeqVaeLagAttnTrfCfs.forward(y_st, y_ph, u_stream, anchor_phase=None, anchor_stride=None,
source_valid=None)` returns:

- `mu_base`, `logvar_base`, `mu_full`, `logvar_full` — $(B, A_{\max}, H, C_{\mathrm{keep}})$.
- `anchor_index` $(B, A_{\max})$ `long` and `anchor_valid` $(B, A_{\max})$ `bool`.
- `mu_prior`, `logvar_prior`, `raw_logvar_prior`, `mu_post`, `logvar_post`, `z_prior`, `z_post` —
  $(B, T, d_z)$.
- `target_state`, `source_state` — $(B, T, d_{\mathrm{model}})$. `source_state` is the lag
  attention's key/value tensor, so its meaning follows `lag_kv_source`.
- `attended_source_heads` $(B, T, \text{heads}, d_{\mathrm{head}})$, `attn_weights`
  $(B, T, \text{heads}, L)$, `kld_per_t`, `kld_per_t_per_head`, `source_kl_lag_map`,
  `mu_prior_sat_frac`, `delta_mu_sat_frac`.
- `persistence` $(B, A_{\max}, C_{\mathrm{keep}})$, only under `persistence_residual`.
- `event_gate` $(B, T, 1)$, only under `posterior_event_gate`.
- `source_key_mask` and `source_open`, only under `source_quality_mask`.

There is no `decoder_state` and no `delta_mu_src` key. `future_index` is a non-persistent buffer
inherited from the raw-target architecture; this model does not read it.

## 8. Objective

The objective, its $\beta$ schedule, the validation-only permutation control, the source-null arm,
the spike-breaker wiring and the checkpoint contract are the causal parent's code
(`teb_vae/lag_attn_rws/nets/losses.py`, `teb_vae/lag_attn_rws/task.py`). Each reconstruction term
is a Gaussian block NLL over the scored cells of the anchor's forecast block.

### 8.1 Channel weights

Both reconstruction terms carry a per-channel weight $w_c$, resolved from `target_weight_st` and
`target_weight_ph` and renormalised so that $\sum_c w_c = C_{\mathrm{keep}}$. The configuration
therefore states a ratio. At $(1.0, 1.0)$ the vector is exactly ones and `raw_sample_score` skips
the multiplication. The shipped pair is $(1.0, 0.1)$; with $32$ kept `fhr_st` and $44$ `fhr_ph`
channels it resolves to $2.0879$ and $0.20879$.

### 8.2 Horizon weights

`horizon_weight_halflife_steps` $\lambda$ weights the horizon axis by $w_\tau \propto
2^{-\tau/\lambda}$, renormalised to $\sum_\tau w_\tau = H$. The shipped $\lambda = 30$. The
renormalisation keeps the block magnitude, so `gradient_clip_val`, `additive_margin` and the
standing of $\beta$ do not change with the weights.

### 8.3 Scored horizon $H_c$

Each target channel is scored over its first $H_c$ horizon steps only:

$$H_c = \begin{cases} H_{\mathrm{fast}} & c \in \texttt{fhr\_ph},\ \xi_{i(c)} > f_{\mathrm{cut}} \\
H & \text{otherwise,} \end{cases}$$

where $\xi_{i(c)}$ is the slow-leg centre frequency of the phase pair (`sel_xi_i_hz` in the shard),
$f_{\mathrm{cut}}$ is `target_phase_fast_cutoff_hz` ($0.03$) and $H_{\mathrm{fast}}$ is
`target_phase_fast_horizon` ($5$). `teb_vae/lag_attn_cfs/scored_horizon.py::resolve_target_scored_horizon`
resolves the vector from the shards. The network holds the non-persistent buffer
`target_cell_mask`, $m_{\tau,c} = \mathbb 1[\tau < H_c]$, of shape $(H, C_{\mathrm{keep}})$.

The rationale is structural. A first-order scattering coefficient $S_1 = \phi * |x * \psi_k|$ is a
band power. A phase product $\mathrm{Re}\{\phi * ([z_i]^k \bar z_j)\}$ depends on the relative
phase of two band-limited processes, which decorrelates within about the inverse bandwidth.

The mask is not a weight: the scored block stays a log-density, and the evaluation applies the
mask too. Setting both keys to `null` scores every cell.

### 8.4 AR(1) residual likelihood

With `forecast_ar_residual: true`, each cell is scored on its innovation,

$$e_{\tau,c} = r_{\tau,c} - \phi_c\, r_{\tau-1,c}, \qquad r = x - \mu, \quad r_{-1} = 0,
\qquad \phi_c = \tanh(a_c),$$

under the decoder's log-variance. The map $r \mapsto e$ is unit lower-triangular, so the summed
innovation NLL is the exact joint negative log-density of the horizon block. A gap step's residual
is zeroed before it is lagged. `target_ar_logit` $a_c$ is a $(C_{\mathrm{keep}},)$ parameter
seeded at $0$, so the model starts at the factorised score. Both decoder invocations share
$\phi_c$, so `pred_gap` stays a difference of two log-densities of one family.

Both likelihood terms enter through one seam, `FeatureForecastTarget.forecast_likelihood_kwargs()`.
The objective, the in-training gap splits and every evaluation scoring site pass it to
`raw_sample_score`.

### 8.5 Contraction weighting, event gate and quality masks

These mechanisms live in the shared parents (`teb_vae/lag_attn/nets/heads.py`,
`teb_vae/lag_attn_rws/nets/losses.py`, `teb_vae/lag_attn_rws/task.py`,
`teb_vae/lag_attn_rws/event_weight.py`). Their constructor and task defaults are off; the off
state is the model and objective before the keys existed. `default.yaml` turns them on.

- **Contraction weighting** (`event_weight_floor: 0.0`, `event_weight_reach_steps: 55`, task
  keys). The task runs the cell's evaluation contraction detector on the batch's raw UP, floors the
  onsets onto the feature grid, and weights the full branch's per-anchor block NLL at $1$ within
  $R$ steps after an onset and at the floor elsewhere. The base branch and the KL are not weighted.
  `nll_full_block` and `pred_gap` stay unweighted; `nll_full_event_block` is the term that enters
  the total, and `event_anchor_frac` is the share of scored anchors at weight $1$.
- **Event gate penalty** (`lambda_event_gate: 0.05`). The objective adds
  $\lambda_g\, \beta\, \mathbb E_t[g_t]$ over the KL's anchor support. A weight on a gate that the
  model did not build is refused by name. `event_gate_mean` is the mean gate over the open anchors
  ($1.0$ without a gate).
- **UA quality mask** (`source_quality_mask: true`, constructor; `dataset_kwargs.ua_step_mask`,
  loader). The loader rule takes `bits` (quality bits, or `"invalid"`), `rho_epsilon` and
  `hold_steps`. Stored step $k$ is bad when a raw sample $16k - 15, \dots, 16k$ carries a named bit,
  or when the coefficient's $\rho > \varepsilon$; `hold_steps` extends a bad bit-step forward. The
  loader emits `up_st_step_valid` and `up_ph_step_valid`, and the task concatenates them in
  `u_stream` order. A source step is usable when every kept source channel is valid. The key mask
  is $K_{b,t,\ell} = \mathrm{lag\_mask}_{t,\ell} \wedge m_{b,t-\ell}$, and the gate closes where no
  key is left: $g^{\mathrm{eff}}_{b,t} = g_{b,t}\, o_{b,t}$ with $o_{b,t} = \exists \ell :
  K_{b,t,\ell}$. A model without the learned gate uses $o_t$ as its gate. Under `conv_stem` and
  `encoder`, `source_key_mask` widens the bad indicator over the key's receptive field with a
  max-pool. The contraction weight returns to its floor at a closed anchor. `source_mask_fraction`
  is the share of KL-scored anchors that the mask closed, and `source_conditioned_kl_open` is the KL
  over the open anchors. The pre-flight `_check_source_quality_pairing` in
  `teb_vae/lag_attn_cfs/trainer.py` refuses one key without the other.
- **Detector validity** (`event_weight_ua_valid: true`, task key). The contraction detector treats
  a sample that the loader's `up_valid` marks invalid as a gap and drops an event that touches one.
  It needs `dataset_kwargs.emit_quality_mask: true` and an `event_weight_floor`.
- **Target quality mask** (`target_quality_mask: true`, task key; `dataset_kwargs.fhr_step_mask`).
  The task multiplies the loss weight by the loader's `fhr_step_mask` after the detector has read
  the unmasked weight. The shipped rule is `{bits: invalid, rho_epsilon: null}`. The coverage floor
  and the KL support use the same weight. The pre-flight refuses the key without an active
  `fhr_step_mask`, and the reverse.
- **Source event mask** (`source_event_mask`, task key, off in `default.yaml`). It ANDs the detected
  contraction spans into `source_valid`, so the posterior reads UA only inside a contraction. It
  needs `source_quality_mask` and a floor.

None of these keys adds a parameter or a buffer except the gate's two tensors.

### 8.6 What the evaluation applies

The evaluation scores every block without the channel weights, the horizon weights and the
contraction weights, so its `nll_*` and `pred_gap` are log-densities of the scored block. It applies
the scored-horizon mask and the AR(1) term, because these define the density.

## 9. Metrics this target domain adds

`LagAttnTrfCfsTrainer.TRACKED_METRICS` is the causal parent's tuple: the shared metric surface,
the two-sided feature cell's forecast-gap splits, and the causal readouts below.

| Metric | Stages | Definition |
| --- | --- | --- |
| `pred_gap_tau_first`, `pred_gap_tau_last` | train, val | the forecast gap at the first and the last horizon step |
| `pred_gap_st`, `pred_gap_ph` | train, val | the forecast gap over each stored target block; they sum to `pred_gap` |
| `target_warm_frac` | train, val | the share of scored target cells past their warm-up; resolved at construction |
| `anchors_per_sample` | train, val | decoded valid anchors per sample |
| `source_lag_warmth_frac_st`, `source_lag_warmth_frac_ph` | train, val | attention mass on lags where each stored source block is past its warm-up ($W'_c + d_c$) |
| `pred_gap_warm_lo` / `_mid` / `_hi` | train, val | the forecast gap over the slowest, middle and fastest third of the kept target channels |
| `pred_gap_novel_lo` / `_mid` / `_hi` | train, val | the forecast gap over the thirds ranked by the share of a scored coefficient drawn from raw samples the anchor has not seen |
| `kld_source_null` | val | the KL with the source values removed and the availability clock kept |
| `pred_gap_mean` | val | the forecast gap with both branches decoded at their latent means |

The novelty share comes from the shard's horizon-free `causal_novelty_curve` at the run's horizon
(legacy shards: `causal_novelty_frac`). Each family of splits recomposes to `pred_gap` over the same
denominator. The event and quality metrics of §8.5 come from the shared surface.

## 10. Causality

**Token causality holds without a configuration flag.** The target encoder uses `RMSNorm` over
channels only, left-padded causal convolutions and causal attention. The source path is
position-wise under `adapter`, causal under `conv_stem`, and causal with an explicit band mask under
`encoder`. `causal_norm` is not a constructor keyword of this model. `tests/test_causality.py`
measures causality through the assembled model, with the prefix equivalence
$\mathcal E(X_{0:T-1})_t = \mathcal E(X_{0:t})_t$, and enumerates every normaliser on a history
path.

**The transform is causal too.** A stored coefficient at $t$ is a function of
$\{x(s) : s \le t\}$, so a forecast of $t + 1 + \tau$ from history up to $t$ uses no future sample.
The stored coefficients still lag their content by the composed filter delay, so a lag-resolved
readout is an attribution over stored-coefficient time.

## 11. Structural invariants

| Property | What enforces it | Test |
| --- | --- | --- |
| No decoder bypass: gradient reaches the decoder only through $z$ | `BaselineFutureDecoder.forward` takes one latent tensor at $d_z$ in-features | `tests/test_invariants.py` |
| Source purity: the prior never sees the source, the source state never sees the target | separate gates, adapters and encoders; the posterior is a residual on the prior | `tests/test_invariants.py` |
| Exact zero KL at initialisation | posterior deltas zeroed after the generic init; one shared $\epsilon$ | `tests/test_invariants.py` |
| One decoder, invoked twice | the same module on $z^p$ and $z^q$ | `tests/test_invariants.py` |
| Token causality | §10 | `tests/test_causality.py` |
| Padded slots are marked invalid | the tiled anchor set | `tests/test_forward_contract.py` |
| Likelihood structure | scored mask and AR(1) term | `tests/test_likelihood_structure.py` |
| Event mechanisms and their off state | gate, weighting, penalty | `tests/test_event_mechanisms.py` |
| UA quality mask | model, controls, objective, task, pre-flight | `tests/test_source_quality_mask.py` |

The delta heads are zero-initialised, so a KL assertion on a freshly built model passes trivially.
The `perturb_posterior` fixture is necessary for every test that checks KL behaviour.

## 12. DDP

Production runs use plain `"ddp"` with `find_unused_parameters=False` under `gaussian_nll`. Under
`mse` the decoder log-variance heads get no gradient, so the driver selects
`ddp_find_unused_parameters_true`. `tests/test_ddp_strategy.py` checks both cases, and
`tests/test_ddp_reachability.py` checks that one backward per guard state reaches every trainable
parameter.

A parameter multiplied by a zero tensor is reachable; a parameter left out of the graph by a
Python branch on a tensor value is not. So the availability terms are added unconditionally in the
forward, and branches happen at construction only. The tile phase is derived per rank from that
rank's samples and adds no collective, and $A_{\max}$ is a geometry constant. Every buffer is a
deterministic function of the config and there is no `BatchNorm`, so `broadcast_buffers=False` is
safe. `static_graph` is not set, because the loss-spike breaker substitutes a zero-weighted sum
over every parameter on a skipped batch.

## 13. Diagnostic figures

The per-run figures come from the causal parent through the task's page seams
(`forecast_rows`, `input_stream_panels`, `forecast_extra_rows`, `input_budget_figure`).
`teb_vae/lag_attn_cfs/DESIGN.md` §11 records them. This package ships no `plotting.py` and no
`sample_page.py`. The config block keeps the name `lag_attn_rws_plotting`, because the shared
callback assembly reads that literal. `tests/test_sample_page.py` checks that the forecast rows are
the causal builder, and `tests/test_heatmap_timestamps.py` renders the page. The page draws the
dense anchor set at phase $0$.

## 14. Configuration

### 14.1 Config files

Every config except `default.yaml` names a parent with `base:` and states only its differences.

| File | Base | What it is |
| --- | --- | --- |
| `default.yaml` | — | the shipped configuration |
| `tiny.yaml` | `default.yaml` | one-epoch smoke on the committed fixture `tiny_shard_causal_int.hdf5`; small widths, `likelihood: mse`, quality keys off |
| `smoke_hie.yaml` | `default.yaml` | the shipped geometry on a local causal HIE shard, legacy operator widths, quality keys off |
| `planted.yaml` | `tiny.yaml` | geometry of the planted-delay identifiability check (`max_lag: 90`, `horizon: 30`, `lag_kv_source: conv_stem`, alignment `target_max`, event and likelihood mechanisms off) |
| `ua_s0_only.yaml` | `default.yaml` | one source channel, physical clock, `warmup_period: 48`, `anchor_stride: 13` |
| `prod_q4_masked.yaml` | `default.yaml` | the default on the 2026-10-27 quality-version-4 build, 4 GPUs |
| `prod_q4_no_quality_mask.yaml` | `sweep_no_quality_mask.yaml` | the no-mask arm on the same build, 4 GPUs |
| `prod_q4_smoke.yaml` | `prod_q4_masked.yaml` | two epochs on one shard, MLflow off |
| `sweep_align_target_max.yaml` | `default.yaml` | input alignment `target_max` |
| `sweep_target_clock_input.yaml` | `default.yaml` | alignment `target_max` and the `input` forecast clock |
| `sweep_legacy_dualref_physclock.yaml` | `default.yaml` | legacy operator, two alignment references, physical clock, `anchor_stride: 5`, legacy shards |
| `sweep_anchor_stride_1.yaml` | `default.yaml` | dense training anchors |
| `sweep_lag_bias_decay.yaml` | `default.yaml` | `alibi_slope_scale: 1.0` |
| `sweep_lag_kv_conv_stem.yaml` | `default.yaml` | `lag_kv_source: conv_stem` |
| `sweep_source_dropout_03.yaml` | `default.yaml` | `source_dropout: 0.3` |
| `sweep_factorised_likelihood.yaml` | `default.yaml` | `forecast_ar_residual: false` |
| `sweep_all_cells_scored.yaml` | `default.yaml` | no scored-horizon mask |
| `sweep_no_event_gate.yaml` | `default.yaml` | no posterior event gate |
| `sweep_no_event_weight.yaml` | `default.yaml` | no contraction weighting, `event_weight_ua_valid: false` |
| `sweep_no_quality_mask.yaml` | `default.yaml` | the three quality switches off, both loader step masks unset |
| `sweep_source_event_mask.yaml` | `default.yaml` | `source_event_mask: true` |

Each sweep arm sets `run_name` and `tags.variant`, so the run directory name states the arm.
`tests/test_sweep_configs.py` checks the arms. `tests/test_config_load.py` compares `default.yaml`
leaf for leaf against the two neighbour cells, outside declared per-edge allow-lists, in both
directions.

### 14.2 Keys that the signature sweep drops

The driver forwards a config key only if it names a constructor argument; it drops every other key
without a message. Thus:

- The conv-LSTM-only keys `lstm_layers`, `encoder_extra_dilations`, `encoder_extra_kernel`,
  `conv_norm_groups` and `causal_norm` are absent from every config here.
- The seven encoder keys `encoder_conv_kernels`, `encoder_conv_dilations`, `encoder_num_heads`,
  `encoder_d_ff`, `target_attention_blocks`, `source_attention_blocks` and
  `source_attention_window` are constructor arguments.
- `causal_align_reference`, `causal_align_reference_source`, `causal_target_forecast_clock`,
  `causal_warmup_budget_steps`, `causal_source_channels`, `target_phase_fast_cutoff_hz` and
  `target_phase_fast_horizon` name no constructor argument. The driver resolves them into the
  constructor's tuples.
- `tiny.yaml` must satisfy two width constraints: `num_heads * d_head == d_model` for the lag
  attention, and an even `d_model / encoder_num_heads` for the encoder.

### 14.3 What a run records

Every model key is a leaf of `model_config.VAE_model`, so it lands in the resolved config the run
writes beside its checkpoints; the constructor keys land again in the checkpoint's `model_kwargs`.
The startup log prints the resolved budget: the references in seconds, the shift range and the
kept counts per block and stream. The evaluation console and `summary.json` print the configured
arm, the resolved clocks and `lag_kv_source`.

## 15. Checkpoints

- The driver writes `lag-attn-trf-cfs-*.ckpt`. `model_checkpoint` monitors `val/total_loss`, and a
  secondary checkpoint monitors `val/nll_full_block`.
- `load_checkpoint_strict` refuses missing or unexpected keys, and `check_model_class` checks the
  stamped class.
- The decoder width, the AR(1) parameter and the persistence weight follow $C_{\mathrm{keep}}$, so
  checkpoints of two warm-up budgets cannot load into each other.
- The quality-mask keys add no parameter, so a checkpoint loads under either value.
- `core_model_checkpoint` in `model_config` names a checkpoint to warm-start from.

## 16. Training step and data loading

- The loader reads one random sample per call. The shard writer
  (`hdf5_dataset/new_pipeline/create_new_pipeline.py::create_initial_hdf5`) chunks every dataset of
  two or more dimensions one sample deep. `hdf5_dataset/rechunk_hdf5.py` converts an older shard to
  that layout.
- `SeqVaeLagAttnCfsTask.transfer_batch_to_device` leaves `epoch` on the host, so the tile phase is
  derived without a device synchronisation. The objective validates the anchor index once per step
  and passes `validate=False` to the other mask builds. `tests/test_sync_free_step.py` (CUDA only)
  checks that those sites cause no synchronisation.
- `precision: "32-true"` and `compile: false` are shipped.

## 17. Running it

From the repository root:

```bash
# Production, 7 ranks. TEB_RUN_STAMP makes ranks 1..N-1 share rank 0's run directory;
# the rank count must equal len(general_config.cuda_devices).
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_transformer_cfs.trainer \
    --config teb_vae/lag_attn_transformer_cfs/configs/default.yaml

# Local smoke: one epoch, one device, the committed causal fixture.
python -m teb_vae.lag_attn_transformer_cfs.trainer \
    --config teb_vae/lag_attn_transformer_cfs/configs/tiny.yaml

# The shipped geometry over a local causal HIE shard.
python -m teb_vae.lag_attn_transformer_cfs.trainer \
    --config teb_vae/lag_attn_transformer_cfs/configs/smoke_hie.yaml

# A sweep arm: the same launch line with the arm's config.
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_transformer_cfs.trainer \
    --config teb_vae/lag_attn_transformer_cfs/configs/sweep_lag_kv_conv_stem.yaml

# Planted-delay check: fit the planted fixture and read the lag profile back.
python teb_vae/lag_attn_cfs/lag_recovery_check.py \
    --config teb_vae/lag_attn_transformer_cfs/configs/planted.yaml

# Read a run's metrics_history.csv and resolved config while it trains.
python -m teb_vae.lag_attn_cfs.check_run --run-dir <run>

# Evaluate a finished checkpoint, then gate the result offline (no torch needed).
python -m teb_vae.lag_attn_transformer_cfs.eval.run \
    --checkpoint <run>/model_checkpoints/<name>.ckpt
python -m teb_vae.lag_attn_transformer_cfs.eval.verify <run>/eval_results/summary.json

# Fine-tune on one k-fold split and evaluate / classify the result.
python -m teb_vae.lag_attn_transformer_cfs.adverse_finetune --steps premise finetune \
    --pretrained-checkpoint <run>/model_checkpoints/<name>.ckpt \
    --kfold-root <k_fold_cross_validation_dataset> --fold 1 --output-dir <dir>
```

Every entry point also runs from the IDE Run button. `RUN_CONFIG` near the bottom of `trainer.py`
names the config for a launch with no command line; `eval/run.py`, `eval/verify.py`,
`eval/run_quality_arms.py`, `eval/compare_quality_arms.py` and `adverse_finetune.py` read a
`RUN_ARGS` dictionary. A command-line flag wins over the file value. A Run-button launch of a
multi-device config is a single process, and Lightning spawns the DDP workers under it.

`check_run.py` reads the training metrics of one run, in-sample and per epoch. `eval/verify.py`
reads `eval_results/summary.json` of a finished checkpoint on a held-out population, per
recording, with intervals.

Tests:

```bash
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_cfs/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_cfs/tests -q -m slow
```

## 18. Limitations of the design

- **The nats depend on the block.** A block score sums $\sum_c \min(H_c, H)$ cells, and its scale
  depends on $H$, $H_c$, $C_{\mathrm{keep}}$ and the AR(1) term. Two runs compare as levels only
  when these match. The training objective also applies channel, horizon and contraction weights,
  so a training `pred_gap` from `metrics_history.csv` and an evaluation `pred_gap` from
  `summary.json` are different quantities.
- **The lag axis is stored-coefficient time.** The composed group delay of a source channel is not
  divided out, and the target is itself a stored coefficient on its own clock.
- **Group-delay compensation is per channel, not per channel pair.** The optional alignment scales
  each channel's stored $\tau_g$ by $\kappa$. The dataset's phase-leg alignment uses the unscaled
  $\tau_g$; changing it would need a dataset rebuild.
- **The warm-up is paid once per $22$-minute segment.** The transform runs per segment, so every
  segment loses its first $F$ steps of forecast supervision. A dataset built with left context
  before the transform would make every $W'_c$ zero; the model would need no change.
- **The prior's clock does not cancel the KL's mean term.** The posterior is a bounded residual on
  the prior, so the mean half of the divergence at a silent source is a function of the delta head
  alone.
- **The residual model is AR(1) per channel**, with no cross-channel term.
- **$H_c$ is a two-key rule** (one cutoff frequency, one short horizon).
- **The two cfs cells' local stems differ in size.** Each `conv_stem` reuses its own parent
  encoder's convolution schedule.
- **No mixed precision and no `torch.compile`** in the shipped configs.
