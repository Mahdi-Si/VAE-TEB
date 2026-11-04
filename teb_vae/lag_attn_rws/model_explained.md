# Raw-Domain Predictive Transfer-Entropy Bottleneck for FHR–UP

This file states the design of `SeqVaeLagAttnRws` (`nets/model.py`) in information-theoretic
terms, and it lists the planned extensions that the code does not implement yet. Sections 1–7
describe the code as built. Section 8 lists planned work. For the layer-by-layer reference, see
`MODEL_DOCUMENTATION.md` and `DESIGN.md`.

## Table of contents

1. [Transfer entropy in this model](#1-transfer-entropy-in-this-model)
2. [Raw-domain geometry](#2-raw-domain-geometry)
3. [The latent model as built](#3-the-latent-model-as-built)
4. [Structural invariants](#4-structural-invariants)
5. [Input causality](#5-input-causality)
6. [The training objective](#6-the-training-objective)
7. [Readouts and controls](#7-readouts-and-controls)
8. [Planned extensions](#8-planned-extensions)
9. [References](#9-references)

---

# 1. Transfer entropy in this model

Let $Y$ be the fetal heart rate (FHR, the target) and $U$ the uterine pressure (UP, the source).
For a future block of $H_{\mathrm{raw}}$ raw samples, the block transfer entropy is

$$
\operatorname{TE}_{U \rightarrow Y} = I\!\left(U_t^-; Y_t^+ \mid Y_t^-\right)
= \mathbb{E}\left[\log \frac{p^\star(Y_t^+ \mid Y_t^-, U_t^-)}{p^\star(Y_t^+ \mid Y_t^-)}\right],
$$

where $p^\star$ is the true data distribution, $Y_t^-$ and $U_t^-$ are the histories up to anchor
$t$, and $Y_t^+$ is the raw future block.

Transfer entropy is directed predictive information. Hidden common causes, unrecorded clinical
interventions and higher-order interactions can produce it without an interventional causal
effect. The documentation therefore calls it directed predictive information unless stronger
causal assumptions are stated explicitly [2].

The model follows the transfer-entropy bottleneck (TEB) idea [1]. A target-only latent
distribution and a source-conditioned latent distribution share one coordinate system and one
decoder. The KL divergence between them is the latent readout of what the source added. The
difference of the two predictive log-scores is the predictive readout.

---

# 2. Raw-domain geometry

The class `TrimmedRawGeometry` (`nets/geometry.py`) owns the anchor-to-raw index map.

| Quantity | Symbol | Value |
| --- | --- | --- |
| Raw sampling rate | $f_s$ | $4$ Hz |
| Raw samples per decimated step | $D = R$ | $16$ |
| Decimated step | $\Delta t$ | $4$ s |
| Decimated steps per segment | $T$ | `sequence_length` $= 300$ |
| Raw samples per trimmed segment | $L_{\mathrm{raw}}$ | $T \cdot D = 4800$ |
| Horizon tokens | $H$ | `horizon` $= 30$ |
| Raw samples per forecast block | $H \cdot R$ | $480$ ($120$ s) |
| Anchors with a full forecast window | $T_{\mathrm{valid}}$ | $T - H = 270$ |
| Warm-up steps excluded from every loss | $w$ | `warmup_period` $= 30$ |

The loader applies `trim_minutes: 1.0` before the model sees the data. On this trimmed grid, the
causal endpoint of anchor $t$ and the first sample of its forecast block are

$$
n_t = D\,(t+1) - 1, \qquad \mathrm{future\_block\_start}(t) = n_t + 1 = D\,(t+1).
$$

The forecast target of anchor $t$ is therefore $Y_t^+ = Y[n_t+1 : n_t+H R]$, with
$Y_t^+ \in \mathbb{R}^{480}$. The model decodes the anchor range $[0, T_{\mathrm{valid}})$ and the
loss reads the range $[w, T_{\mathrm{valid}})$. The functions `build_future_index` and
`build_future_target` (`nets/raw_targets.py`) gather the target from the raw `fhr` signal.

---

# 3. The latent model as built

The model has one latent space with two distributions over it:

$$
p_\theta(z_t \mid Y_{\le t}) = \mathcal{N}\!\left(\mu_t^p, \operatorname{diag} e^{\ell_t^p}\right),
\qquad
q_\phi(z_t \mid Y_{\le t}, U_{\le t}) = \mathcal{N}\!\left(\mu_t^q, \operatorname{diag} e^{\ell_t^q}\right).
$$

The prior $p_\theta$ is the complete target-only predictive state. The posterior $q_\phi$ is the
same state after the model adds the source information. The decoder receives only $z$, so the
prior latent must carry the target state itself.

## 3.1 Inputs and encoders

The inputs are the stored feature blocks, not raw signals:

* Target stream: `fhr_st` ($43$) and `fhr_ph` ($66$), concatenated to $c_y = 109$ channels.
* Source stream: `up_st` ($43$) and `up_ph` ($15$), $c_u = 58$ channels; $c_u = 15$ when
  `use_up_st: false`.

The cross-channel FHR×UP phase-harmonic block is not loaded. A coefficient that mixes both signals
would break the separation between $p_\theta$ and $q_\phi$.

Each stream passes through an optional `ChannelGate` (Section 5), an `AvailabilityInputAdapter`,
and a `CausalConvLstmEncoder` of width `d_model` $= 128$. The encoder has a causal dilated
convolution branch (dilations $(1, 2, 4)$ plus `encoder_extra_dilations`) and a unidirectional
LSTM branch of `lstm_layers` layers. The target encoder output is $H_t^y$. The source keys and
values come from the module that `lag_kv_source` selects:

* `encoder` (default): the deep source encoder state $H_t^u$.
* `conv_stem`: a `CausalConvStem` on the same convolution schedule, without the LSTM.
* `adapter`: the projected source stream, with one step of reach.

## 3.2 Prior head

`FullLatentPriorHead` (`nets/heads.py`) reads $H_t^y$ and produces

$$
\mu_t^p = a_\mu \tanh\!\left(\frac{\widetilde\mu_t^p}{a_\mu}\right), \qquad
\ell_t^p = \operatorname{smoothbound}\!\left(\widetilde\ell_t^p\right)
= \ell_{\min} + (\ell_{\max}-\ell_{\min})\,\sigma\!\left(\widetilde\ell_t^p\right),
$$

with $a_\mu$ = `mu_scale` and $(\ell_{\min}, \ell_{\max})$ = `logvar_clamp`. The head also returns
the pre-bound value $\widetilde\ell_t^p$ (`raw_logvar_prior`). The posterior needs this value
because the sigmoid bound is not idempotent.

## 3.3 Lag cross-attention

`LagCrossAttention` (`teb_vae/lag_attn/nets/attention.py`) uses a query that is a projection of
the prior belief, $Q_t = W_Q \mu_t^p$, or $W_Q [\mu_t^p \Vert \ell_t^p]$ when
`query_uses_logvar: true`. Both inputs are target-only. For head $m$ and lag
$\ell = 0, \ldots, L-1$ with $L$ = `max_lag` $+ 1$, the score is

$$
a_{t,m,\ell} = \frac{\left\langle q_t^{(m)}, k_{t-\ell}^{(m)} \right\rangle
+ \left\langle q_t^{(m)}, r_{m,\ell} \right\rangle}{\sqrt{d_h}} + b_{m,\ell},
$$

where $r_{m,\ell}$ is a learned Shaw-style lag embedding (`lag_embeddings`) and $b_{m,\ell}$ is a
learnable score bias (`lag_score_bias`). The bias exists only under `lag_bias_init: alibi_decay`,
which seeds it at $-m_h \ell$ with ALiBi-style geometric slopes $m_h$. The mask
$\mathbf{1}[t-\ell \ge 0]$ removes lags before the segment start. The weights are

$$
\alpha_{t,m,\ell} = \operatorname{entmax15}_\ell(a_{t,m,\ell})
\quad\text{or}\quad \operatorname{softmax}_\ell(a_{t,m,\ell}),
$$

selected by `use_entmax`. The per-head summary is $A_t^{(m)} = \sum_\ell \alpha_{t,m,\ell}
v_{t-\ell}^{(m)}$. The attention dropout is fixed at $0$, so the returned `attn_weights` are the
weights that the posterior consumed. The output projection `W_o` is frozen because the posterior
reads the per-head summaries directly.

## 3.4 Posterior head

`PosteriorHead` (`teb_vae/lag_attn/nets/heads.py`) is head-structured. The latent has `num_heads`
groups of $d_z / M$ dimensions, and group $m$ reads only $[\operatorname{LN}(H_t^y) \Vert
\operatorname{LN}(A_t^{(m)})]$. The mean is a bounded residual on the prior mean:

$$
\mu_t^q = \mu_t^p + a_{\Delta\mu} \tanh\!\left(\frac{\widetilde{\Delta\mu}_t}{a_{\Delta\mu}}\right),
$$

with $a_{\Delta\mu}$ = `delta_mu_scale`. The key `posterior_logvar_mode` selects the
log-variance:

* `residual`: $\ell_t^q = \operatorname{smoothbound}\!\left(\widetilde\ell_t^p + a_{\Delta\ell}
  \tanh(\widetilde{\Delta\ell}_t / a_{\Delta\ell})\right)$, with $a_{\Delta\ell}$ =
  `delta_logvar_scale`. The full-branch gradient then reaches the prior log-variance.
* `independent` (shipped): the posterior has its own log-variance head, and
  $\ell_t^q = \operatorname{smoothbound}(\widetilde\ell_t^q)$. No gradient from the full branch
  reaches the prior log-variance through this head.

`_zero_init_delta_heads` zeroes the delta heads after the generic initialisation. It seeds the
independent log-variance head with zero weight and the bias $\log\!\left((0 - \ell_{\min}) /
(\ell_{\max} - 0)\right)$, so $\ell^q = 0$ at step $0$. Under `head_init_calibration: true`,
`_calibrate_prior_scale` puts the prior log-variance at the same value, $\ell^p = 0$. With both
settings, $q_t = p_t$ at step $0$ in both modes. Under `independent` without the calibration, the
KL at step $0$ is not zero.

With `posterior_event_gate: true`, the head also builds a gate $g_t \in (0, 1)$ from the attended
summary. The gate multiplies both bounded residuals, so $g_t = 0$ gives $q_t = p_t$. The shipped
configuration does not build the gate.

## 3.5 Latent sampling

`_reparameterize_shared` draws one $\epsilon_t \sim \mathcal{N}(0, I)$ per step:

$$
z_t^q = \mu_t^q + \sigma_t^q \odot \epsilon_t, \qquad
z_t^p =
\begin{cases}
\mu_t^p + \sigma_t^p \odot \epsilon_t & \texttt{base\_decode: sample}, \\
\mu_t^p & \texttt{base\_decode: mean (shipped)}.
\end{cases}
$$

Under `sample`, the shared draw is a common-random-number coupling: $p_t = q_t$ gives
$z_t^p = z_t^q$ sample by sample. Under `mean`, only the posterior branch is sampled.

## 3.6 Shared raw decoder

One `BaselineFutureDecoder` with one `HorizonDecoderCore` serves both branches:

$$
(\mu_t^{\mathrm{base}}, \ell_t^{\mathrm{base}}) = D_\psi(z_t^p), \qquad
(\mu_t^{\mathrm{full}}, \ell_t^{\mathrm{full}}) = D_\psi(z_t^q).
$$

The decoder works as follows:

1. A residual MLP projects $z_t$ ($d_z$) to `decoder_hidden`.
2. The core adds a learned embedding $e_k$ to each of the $H = 30$ horizon tokens.
3. `horizon_depth` dilated refine blocks (dilations $1, 2, 4, \ldots$, kernel `horizon_kernel`)
   mix the tokens. Each block has its own FiLM generator (`film_per_block=True`), so every block
   reads the latent.
4. `horizon_attention_blocks` bidirectional self-attention blocks mix all $H$ tokens at once.
5. The heads `mean_head` and `logvar_head` emit $R = 16$ raw samples per token. The log-variance
   passes through `smoothbound`.

The output shapes are $(B, T_{\mathrm{valid}}, H, R)$. The decoder has zero dropout, because one
module that runs twice would draw two different dropout masks. The horizon-core normalisers and
self-attention are not causal; they pool over the forecast axis of one anchor, which every token
predicts from the same $t$.

The decoder is non-autoregressive. An autoregressive decoder with teacher forcing could predict
each raw sample from the preceding samples and use little information from $z_t$. A parallel
decoder makes the latent specify the full trajectory.

The architecture has no term of the form $\mu^{\mathrm{full}} = \mu^{\mathrm{base}} +
\Delta\mu^{\mathrm{src}}$. The source effect on the forecast, $\mu_t^{\mathrm{full}} -
\mu_t^{\mathrm{base}}$, is a consequence of the change in the latent.

## 3.7 The three representations

| Representation | Meaning |
| --- | --- |
| $\mu_t^p$ | target-only predictive state |
| $\mu_t^q - \mu_t^p$ | source-derived update |
| $\mu_t^q$ | target state after the source update |

---

# 4. Structural invariants

The code enforces four properties, and a test checks each one.

| Invariant | Statement | Test |
| --- | --- | --- |
| Source purity | $H^y$ and $p_\theta$ are functions of the target only; the source pathway receives no target tensor. Replacing the source leaves the prior and the base forecast bitwise unchanged in `eval()`. | `tests/test_source_purity.py` |
| No decoder bypass | The decoder receives only $z$. There is no `decoder_state`. Gradient from the base forecast reaches the target encoder only through $z$. | `tests/test_no_bypass.py` |
| Shared decoder | The base and full forecasts use the same decoder parameters. | `tests/test_decoder.py` |
| Exact zero at init | $D_{\mathrm{KL}}(q_t \Vert p_t) = 0$ at construction under `posterior_logvar_mode: residual`, or under `independent` with `head_init_calibration: true` (Section 3.4). Under `base_decode: sample`, $z^p = z^q$ and the two forecasts are bitwise equal in train mode. | `tests/test_zero_kl_init.py` |

The per-lag attribution identity of Section 7.3 is checked in `tests/test_lag_map.py`.

---

# 5. Input causality

The KL is a source-conditioned rate only if every input at step $t$ depends on raw samples up to
$n_t$ alone. The required invariant is

$$
\frac{\partial X_t}{\partial Y[j]} = 0 \quad \forall j > n_t,
$$

and the same for $U$. The neural layers satisfy it when `causal_norm: true`: `causalize_norms`
replaces every `GroupNorm` on a history path with `CausalGroupNorm`. The probe in
`tests/test_causal_encoder.py` checks that `target_state` at $t_0$ does not move when the input
changes after $t_0$.

The stored scattering and phase-harmonic features are two-sided wavelet transforms. Some
coefficients at step $t$ therefore depend on raw samples after $n_t$. A causal network layer cannot
remove information that is already in its input.

The causal reach guard limits this leak. The trainer resolves `causal_reach_budget_s` into a list
of surviving channels and a delay $\delta_c$ per channel. The channel survives only if its analytic
forward reach is within the budget, and $\delta_c = \lceil \mathrm{reach}_c / \Delta t \rceil$.
`ChannelGate` (`teb_vae/lag_attn/nets/delays.py`) applies both inside the forward pass. The
adapters are `AvailabilityInputAdapter` instances, because a guard zero-fills the first
$\delta_c$ steps of each surviving channel. `null` disables the guard. The test
`tests/test_causal_leak.py` perturbs the raw signal after the causal endpoint, recomputes the
features, and compares the values that the model reads.

---

# 6. The training objective

`nets/losses.py::compute_loss` implements

$$
\mathcal{L} = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0
+ \beta\,\mathrm{KL}_{\mathrm{train}} + \beta_p R_p
+ \lambda_{\mathrm{ms}} \mathcal{L}_{\mathrm{ms}} + \lambda_\Delta \mathcal{L}_\Delta
+ \lambda_{\mathrm{boundary}} \mathcal{L}_{\mathrm{boundary}}
\; [+\, \lambda_{\mathrm{event\_gate}} \mathcal{L}_{\mathrm{gate}}].
$$

## 6.1 Reconstruction terms

$D_1 = -\log p_\psi(Y_t^+ \mid z_t^q)$ and $D_0 = -\log p_\psi(Y_t^+ \mid z_t^p)$. With
`likelihood: gaussian_nll`, the factorised Gaussian score per anchor is

$$
D_t = \frac12 \sum_{h=1}^{H R} \left[\log 2\pi + \ell_t[h] + \frac{(Y_t^+[h]-\mu_t[h])^2}{e^{\ell_t[h]}}\right].
$$

With `likelihood: mse`, the decoder log-variance heads receive no gradient; this mode is for
debugging. The $D_0$ term makes the prior latent a forecaster on its own.

A factorised likelihood ignores the correlation between raw samples in one block. Its log-score
difference is therefore a transfer-entropy estimate within the model family, not the exact
data-generating value.

## 6.2 Source-conditioned KL

For diagonal Gaussians, per step and per dimension (`kld_tensor`):

$$
\mathrm{KL}_{t,j} = \frac12 \left[\ell_{t,j}^p - \ell_{t,j}^q
+ \frac{e^{\ell_{t,j}^q} + (\mu_{t,j}^q-\mu_{t,j}^p)^2}{e^{\ell_{t,j}^p}} - 1\right].
$$

$\mathrm{KL}_{\mathrm{train}}$ applies the `free_bits` floor per dimension and step. The raw,
unfloored KL is reported separately and is the only one read as a rate. The KL uses the same anchor
support $[w, T_{\mathrm{valid}})$ as the reconstruction terms.

## 6.3 Prior scale rate

`masked_prior_rate` computes

$$
R_p(t) = \sum_d \tfrac12\left(e^{\ell^p_{t,d}} - 1 - \ell^p_{t,d}\right)
= D_{\mathrm{KL}}\!\left(\mathcal{N}(\mu^p_t, \operatorname{diag} e^{\ell^p_t}) \,\Big\|\, \mathcal{N}(\mu^p_t, I)\right).
$$

This is the scale half of $D_{\mathrm{KL}}(p_\theta \Vert \mathcal{N}(0, I)) = R_p +
\tfrac12 \sum_d (\mu^p_{t,d})^2$. The term does not contain the prior mean, so it does not compress
the content of the base forecast. It is the only term that penalises a narrow prior. The weight
$\beta_p$ (`beta_prior`) is a constant. The function reports `prior_rate` at every weight,
including $0$.

## 6.4 Auxiliary shape terms

The three terms act on the forecast means of both branches. Each is computed only when its weight
is nonzero; at weight $0$ its metric is exactly $0.0$. The terms are $L_1$ and Huber quantities on
z-scored samples, so a total that contains them is a mixed-unit criterion.

* `masked_multiscale_l1` ($\lambda_{\mathrm{ms}}$, `lambda_ms`): $L_1$ between the average-pooled
  forecast mean and target at `MS_RATES` $= (1, 4, 16)$, with uniform weights. The function applies
  the forecast mask before it pools.
  $$
  \mathcal{L}_{\mathrm{ms}} = \sum_{k \in \{0,1\}} \sum_{r \in \{1,4,16\}}
  \left\|\operatorname{Pool}_r(Y_t^+) - \operatorname{Pool}_r(\mu_t^k)\right\|_1.
  $$
* `masked_derivative_huber` ($\lambda_\Delta$, `lambda_deriv`): Huber loss with
  $\delta = 1$ (`DERIVATIVE_HUBER_DELTA`) between first differences. A difference pair is valid
  only when both samples are valid.
  $$
  \mathcal{L}_\Delta = \sum_{k \in \{0,1\}} \operatorname{Huber}\!\left(\Delta Y_t^+, \Delta \mu_t^k\right).
  $$
* `masked_boundary_gap` ($\lambda_{\mathrm{boundary}}$, `lambda_boundary`): the level gap between
  the first forecast sample and the last observed sample.
  $$
  \mathcal{L}_{\mathrm{boundary}} = \sum_{k \in \{0,1\}} \left|\mu_t^k[1] - Y[n_t]\right|.
  $$
  On the raw grid $Y[n_t]$ is a slice of the gathered target, `target[:, :-1, 0, -1]`, so the term
  covers $t \in [1, T_{\mathrm{valid}})$. Its validity is the thresholded `weight` of anchor $t$
  times the contributing indicator of anchor $t$.

## 6.5 Units and the KL weight

The reconstruction is summed over the $H R$ samples of a block, and the KL is summed over $d_z$.
Both are in nats per anchor and are averaged over the contributing anchors only. The code logs the
block value (`nll_full_block`, `nll_base_block`) and the per-sample value (block divided by
$H R$). With these units, $D_1 + \beta\,\mathrm{KL}$ at $\beta = 1$ is the ELBO of the
source-conditioned branch.

The `beta_schedule` key sets $\beta$ per epoch. The kind `constant` holds one value. The kind
`linear_warmup` ramps from `start` to `end` over `warmup_epochs`:

$$
\beta(e) = \beta_{\mathrm{start}} + (\beta_{\mathrm{end}} - \beta_{\mathrm{start}})
\min\!\left(1, \frac{e}{E_{\mathrm{warm}}}\right).
$$

## 6.6 Anchor coverage floor

An anchor enters the loss only if the valid fraction of its forecast window is at least
`coverage_floor`. The masks come from `nets/raw_masks.py`.

---

# 7. Readouts and controls

## 7.1 Predictive readout

The primary readout is the held-out log-score improvement:

$$
\widehat{\operatorname{TE}}_{\mathrm{pred}}(t) = \log p_1(Y_t^+) - \log p_0(Y_t^+) = D_0(t) - D_1(t).
$$

If $p_0$ and $p_1$ equal the true conditional distributions, its expectation is
$I(U_t^-; Y_t^+ \mid Y_t^-)$. A single anchor can have a negative value; the recording-level mean is
the aggregate. The training path logs it as `pred_gap`.

## 7.2 Latent readout

The secondary readout is the per-step KL, $K_t = D_{\mathrm{KL}}(q_t \Vert p_t)$, returned as
`kld_per_t` and logged as `source_conditioned_kl_raw`. The name is a source-conditioned KL, not
transfer entropy. It approaches transfer entropy only when all of these conditions are true:

1. The latent is predictively sufficient for the target history.
2. The bottleneck is at its conditional minimum-necessary-information point.
3. The decoder represents the future conditional distribution accurately.
4. The input construction is strictly causal.
5. The relevant confounders are addressed.

The evaluation module `eval/analyses/sufficiency.py` estimates the gap for condition 1,
$\Delta_{\mathrm{suff}} = D_{\mathrm{base}} - D_{\mathrm{oracle}}$, with an evaluation-only oracle
decoder that reads `target_state`. The oracle is not part of the model.

## 7.3 Head- and lag-resolved attribution

Because group $m$ of the latent is written only by head $m$, the per-head KL is an additive split:

$$
K_t^{(m)} = D_{\mathrm{KL}}\!\left(q^{(m)} \Vert p^{(m)}\right), \qquad K_t = \sum_{m=1}^M K_t^{(m)}.
$$

`TEAnalysisHead` returns `kld_per_t_per_head` and the lag map

$$
\widetilde K_{t,\ell} = \sum_{m=1}^M K_t^{(m)} \alpha_{t,m,\ell}, \qquad
\sum_{\ell=0}^{L-1} \widetilde K_{t,\ell} = K_t,
$$

as `source_kl_lag_map`. The identity is exact because the attention weights have no dropout. The
attention is an attribution mechanism; it does not prove a physical effect at a given lag.

A lag index $\ell$ maps to the source content $\ell + \delta$ steps back, where
$\delta$ = `source_delay_steps()` is the maximum source channel delay of the reach guard. In
seconds, $\tau = 4(\ell + \delta)$ s. Because the channel delays differ, $\delta$ is a maximum and
$\tau$ is an upper bound.

## 7.4 Monte Carlo predictive likelihood

The evaluation estimates the marginal predictive likelihood with $K$ latent draws:

$$
\log \widehat p_k(Y_t^+) = \operatorname{logsumexp}_{r=1}^K
\left[\log d_\psi\!\left(Y_t^+ \mid z_t^{k,(r)}\right)\right] - \log K.
$$

$K$ is `num_mc_samples` in the evaluation configuration (schema default $8$). The implementation
is in `eval/metrics.py` and is checked in `tests/test_mc_likelihood.py`.

## 7.5 Permutation control

A deranged source is out of distribution and can move the posterior strongly. The KL under a
shuffled source is therefore not a useful criterion. The control is predictive:

$$
D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}}.
$$

`nets/controls.py::perm_forward_outputs` builds the shuffled branch as follows:

1. Keep the prior and the base forecast unchanged.
2. Derange `source_state` along the batch axis.
3. Rebuild the query from the prior, as the main forward does.
4. Recompute the attention and the posterior.
5. Draw the shuffled latent with a new $\epsilon$.
6. Decode it with the same shared decoder and score the raw future NLL.

The task runs this control on validation batches only, and it never enters the loss. Its metrics
include `nll_shuffled_block` and `kld_shuffled`.

---

# 8. Planned extensions

The code does not implement the items in this section.

## 8.1 Two-block latent with a target bottleneck

A later version may split the latent into a target block $C_t$ and a source block $S_t$:

$$
Z_t^0 = [C_t, S_t^0], \qquad Z_t^1 = [C_t, S_t^1] = Z_t^0 + [0,\, S_t^1 - S_t^0],
$$

with $C_t \sim e_C(c_t \mid Y_t^-)$, a target-only source prior $S_t^0 \sim r_S(s_t \mid C_t)$ and
a source-conditioned distribution $S_t^1 \sim q_S(s_t \mid C_t, U_t^-)$. Because $e_C$ is common
to both,

$$
D_{\mathrm{KL}}(P_1 \Vert P_0) = \mathbb{E}_{C \sim e_C}\!\left[D_{\mathrm{KL}}\!\left(q_S(S \mid C, U_t^-) \Vert r_S(S \mid C)\right)\right] = R_U,
$$

so the source rate charges only the source block. The target block would use a Conditional
Entropy Bottleneck [3] with a training-only future encoder $b_C(c_t \mid Y_t^+)$:

$$
R_C(t) = D_{\mathrm{KL}}\!\left(e_C(c_t \mid Y_t^-) \Vert b_C(c_t \mid Y_t^+)\right).
$$

The future encoder would be a small non-causal convolutional network over $Y_t^+$
($480 \to 240 \to 120 \to 60 \to 30$ positions), pooled to $128$ features, with mean and
log-variance heads. It would not run at inference. The full objective would add $\beta_C R_C$ and
use $\beta_U R_U$ in place of the single KL term. The source-block dimension would satisfy
$d_S < d_C$ and $d_S = M d_{S,h}$.

## 8.2 Causal raw input stem

Each signal would get its own raw stem with four anti-aliased causal downsampling stages
($4 \to 2 \to 1 \to 0.5 \to 0.25$ Hz). One stage is

$$
\widetilde x^{(\ell)}[n] = \sum_{r=0}^{K_\ell-1} g_\ell[r]\, x^{(\ell)}[n-r], \qquad
x^{(\ell+1)}[k] = \rho\!\left(W_\ell \widetilde x^{(\ell)}[2k] + b_\ell\right),
$$

with a causal low-pass filter $g_\ell$, left-only padding, a stride-two convolution, a per-step
LayerNorm or RMSNorm, GELU, a pointwise projection and a residual connection. The widths would be
$(1 + \text{mask channels}) \to 16 \to 24 \to 32 \to 64$. Each signal would carry a validity mask
and an artifact mask. Imputation and normalisation would use past-only statistics or training-set
global statistics.

The proposed input interface adds `y_raw`, `u_raw` $(B, N)$, `y_valid`, `u_valid` $(B, N)$ and an
explicit `anchor_raw_index` $(T,)$. A comparison would run three arms: stored features only, raw
only, and raw plus features.

## 8.3 Causal scattering and phase harmonics

The transforms would use one-sided filters. For a causal wavelet $\psi_\lambda[r]$, $r \ge 0$:

$$
v_\lambda[n] = \sum_{r=0}^{R_\lambda} \psi_\lambda[r]\, x[n-r], \qquad
S_\lambda[t] = \sum_{r=0}^{R_\phi} g[r]\, \left|v_\lambda[n_t-r]\right|,
$$

with $R_\lambda \le R_{\max}$. For causal analytic coefficients $v_i, v_j$ and $p_{ij} = \xi_j / \xi_i$:

$$
\Phi_{ij}[t] = \sum_{r=0}^{R_\phi} g[r]\, \operatorname{Re}\!\left([v_i[n_t-r]]^{p_{ij}}\, \overline{v_j[n_t-r]}\right),
\qquad [v]^{p} = |v|\, e^{i p \arg v}.
$$

Decimation would follow a causal low-pass filter. Each feature family would be added and checked
separately.

## 8.4 Target-only past reconstruction

A past decoder would reconstruct $Y_t^{\mathrm{rec}} = Y[n_t-P+1 : n_t]$, with $P = 600$, from the
target state only. With the current single latent, that input is $\mu_t^p$; with the two-block
latent, it is $C_t$. The source pathway must receive no gradient from this term:
$\partial \mathcal{L}_{\mathrm{past}} / \partial \theta_{\mathrm{source}} = 0$. A multiscale form is

$$
\mathcal{L}_{\mathrm{past}} = \sum_{r \in \{1,4,16\}} w_r
\left\|\operatorname{LPDown}_r(Y_t^{\mathrm{rec}}) - \operatorname{LPDown}_r(\widehat Y_t^{\mathrm{rec}})\right\|_1,
$$

with $\lambda_{\mathrm{past}}$ much smaller than the future weights. Masked-patch reconstruction is
an alternative.

## 8.5 Further objective terms

* Slope boundary term:
  $\left|(\mu_t^k[2] - \mu_t^k[1]) - (Y[n_t] - Y[n_t-1])\right|$.
* State reconstruction: $\mathcal{L}_{\mathrm{state}} = |D_{\mathrm{level}}(\mu^p_t) - Y[n_t]| +
  \operatorname{Huber}(D_{\mathrm{slope}}(\mu^p_t), Y[n_t] - Y[n_t-1])$. The heads would read
  the latent only. They must not feed the future decoder, or they would form a bypass.
* Lag smoothness on the learned Shaw lag embeddings $r_{m,\ell}$ (`lag_embeddings`):
  $$
  \mathcal{L}_{\mathrm{lag}} = \frac{1}{M(L-1)} \sum_{m,\ell} \left\|r_{m,\ell+1} - r_{m,\ell}\right\|_2^2.
  $$
  This is a parameter regulariser, not a measure of physiological smoothness.

## 8.6 Constrained rate–distortion training

The fixed weight $\beta$ could become a constraint, as in GECO and ControlVAE [4]:

$$
\min\ R \quad \text{subject to} \quad D_0 \le \tau_0,\ D_1 \le \tau_1, \qquad
\lambda_k \leftarrow \left[\lambda_k + \eta_\lambda (D_k-\tau_k)\right]_+.
$$

## 8.7 Richer likelihoods

Options are a Student-$t$ output, a low-rank plus diagonal covariance, a banded Cholesky
covariance, a fixed learned temporal correlation kernel, and a conditional normalising flow over
the future residuals.

## 8.8 Anchor subsampling

Training would decode a random subset of $K_A = 16$ to $32$ valid anchors per recording per batch.
The latent would still exist at every step. Validation and test would decode all valid anchors.
The current code decodes all $T_{\mathrm{valid}}$ anchors in every batch.

## 8.9 Additional controls and tests

* Source controls: circularly shifted UP, time-reversed UP, spectrum-matched UP surrogates and
  lag-band exclusions. In each, the base representation and base forecast must be bit-exact.
* Future Jacobian: $|\partial \mu^p_t / \partial Y[j]| < \epsilon$ and
  $|\partial \mu^q_t / \partial U[j]| < \epsilon$ for $j > n_t$.
* Latent dependence: score the raw future NLL with $\mu^p$ set to zero, $\mu^p$ shuffled across
  the batch, $z^q = z^p$, and individual latent groups removed.
* Synthetic transfer-entropy systems: no coupling, $U \to Y$, $Y \to U$, increasing coupling
  strength, fixed lag, variable lag, a common driver, instantaneous correlation without delay,
  nonlinear coupling, and coupling limited to one frequency band. Under the null,
  $K_t \approx 0$ and $D_0 - D_1 \approx 0$.

## 8.10 Representation probes and segment pooling

Frozen-latent probes from $\mu^p_t$, $\mu^q_t$ and $\mu^q_t - \mu^p_t$ would predict future mean,
variance, slope, deceleration depth and duration, recovery time, low-frequency power, current
baseline, short-term variability and clinical labels. The effective rank of the latent covariance,
with normalised eigenvalues $\widetilde\lambda_i = \lambda_i / \sum_j \lambda_j$, is

$$
r_{\mathrm{eff}} = \exp\!\left(-\sum_i \widetilde\lambda_i \log \widetilde\lambda_i\right).
$$

For a per-recording classifier, an exporter would pool the per-step latent, for example as
$\operatorname{concat}(\operatorname{mean}_t, \operatorname{std}_t, \operatorname{max}_t)$ or with
attention pooling $w_t \propto \exp g(\bar z_t)$. It would export three pooled vectors:
$\operatorname{Pool}(\mu^p_t)$, $\operatorname{Pool}(\mu^q_t - \mu^p_t)$ and
$\operatorname{Pool}(\mu^q_t)$.

---

# 9. References

1. Transfer-entropy bottleneck: <https://arxiv.org/pdf/2211.16607>
2. Transfer entropy and causal interpretation: <https://arxiv.org/pdf/1512.06479.pdf>
3. Conditional Entropy Bottleneck: <https://arxiv.org/pdf/2002.05379>
4. Constrained VAE training (GECO): <https://arxiv.org/abs/1810.00597>
