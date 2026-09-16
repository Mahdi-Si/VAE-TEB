# `SeqVaeLagAttn`: the lag-attention VAE-TEB model

**Code:** `teb_vae/lag_attn/nets/model.py` and the modules it imports from
`teb_vae/lag_attn/nets/` (`attention.py`, `blocks.py`, `decoders.py`, `encoders.py`, `heads.py`).
**Training:** `teb_vae/lag_attn/task.py` (objective and controls) and
`teb_vae/lag_attn/trainer.py` (experiment driver), configured by `teb_vae/lag_attn/configs/default.yaml`.
**Companion documents:** `teb_vae/lag_attn/DESIGN.md` is the short standalone contract (tensor
interface, geometry, deviation record). This document is the long-form, code-grounded description
of the network and how it is trained.

`SeqVaeLagAttn` is a single self-contained `nn.Module`. It is the flattened rebuild of the earlier
three-level inheritance chain that lived under the now-deprecated `model/vae_teb_prediction/model/`.
That older tree, and its historical framing of each level as a set of deltas on the one below, are
described in `knowledge/vae-teb-lag-attn-v3-implemented-model.md`, which is kept frozen as a record
of where this model came from. **This model is not a subclass of anything and is not
checkpoint-compatible with that tree**; read this document, not the frozen one, for what the code
does today. The flatten preserved the model's structure exactly — the parameter count, every
state-dict key and shape, and the forward/loss contracts. The complete list of intentional
differences from the original is §8 of `DESIGN.md`. The structural test that once asserted this
against a captured snapshot has been retired: its fixtures were captured at the predecessor's
input widths ($c_y = 87$, $c_u = 101$), which the dataset has since changed, so there is no
longer a common geometry to compare at.

---

## 1. Conceptual summary

The model answers one question at every decimated step $t$: *how much does the source's recent past
tell us about the target's near future that the target's own past did not already say — and at what
delay?*

The target is a fetal-heart-rate (FHR) feature stream $Y$; the source is a uterine-pressure (UP)
feature stream $U$. A uterine contraction does not move the fetal heart rate at the same instant —
the response arrives some lag $\ell$ later, the lag is not known in advance, and it is not the same
for every recording. The machinery, per step $t$, is:

1. Two causal encoders build history states $H^y_t$ and $H^u_t$ from the target and the source.
2. A target-only **prior** $p(z_t \mid Y_{\le t})$ is read off $H^y$.
3. **Lag cross-attention** lets $H^y_t$ look back over $\{H^u_{t-\ell}\}_{\ell=0}^{L-1}$ and pick
   the delay that matters.
4. A **posterior** $q(z_t \mid Y_{\le t}, U_{\le t})$ is built as a bounded residual on the prior,
   conditioned on what the attention found.
5. Two decoders forecast the next $H_d$ steps: a **baseline** from the target alone, and a
   latent-driven **correction**.

The per-step KL between posterior and prior,

$$K_t = \mathrm{KL}\!\big(q(z_t \mid Y_{\le t}, U_{\le t}) \,\big\|\, p(z_t \mid Y_{\le t})\big),$$

is the reported transfer-entropy (TE) surrogate, and the attention weights split it across lags.
Every design choice exists to keep that reading honest:

- **The prior never sees the source.** It is target-only by construction, so $K_t$ can only measure
  what the source added.
- **The encoders never see their own future.** Every component of both encoders is causal, so
  $H^y_t$ and $H^u_t$ depend on inputs at $t' \le t$ only.
- **The baseline decoder is trained to be good on its own.** Without a strong baseline the model
  could route target-explainable variance through the latent, inflating $K_t$ with information the
  source never supplied.
- **The model starts at $K_t \equiv 0$.** At initialisation the posterior *equals* the prior
  exactly, so any coupling it later reports had to be learned against a null, not inherited from a
  random prior/posterior mismatch.

The model is *source-pure*: the source pathway sees only $U$, never a cross-channel field, which is
what makes a batch-wise permutation of $U$ a clean negative control (§20).

## 2. Data interface

`forward(y_st, y_ph, u_stream, *, lag_band_mask=None)` takes three floating tensors and an optional
boolean mask (`lag_band_mask` is keyword-only):

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `y_st` | $(B, T, 43)$ | target FHR scattering features |
| `y_ph` | $(B, T, 66)$ | target FHR phase-harmonic features |
| `u_stream` | $(B, T, c_u)$ | source UP stream |
| `lag_band_mask` | $(L,)$ or $(T, L)$ bool, optional | keep-mask over lags; `True` keeps. `None` is a bit-exact no-op |

Inside `forward` the target is assembled as $Y = \mathrm{concat}(y_{\mathrm{st}}, y_{\mathrm{ph}})$
of width $c_y = 109$. The source stream is either $\mathrm{concat}(u_{\mathrm{st}}, u_{\mathrm{ph}})$
of width $c_u = 58$ (when `use_up_st=True`) or $u_{\mathrm{ph}}$ alone of width $c_u = 15$ (when
`use_up_st=False`). Both UP fields are first-class HDF5 datasets; there is no slicing fallback.

The widths are checked **against the first batch**, not at construction. $c_y$ and $c_u$ are
properties of the dataset, and the constructor cannot see one: an earlier version compared them
against module constants, which went stale the moment the pipeline's phase-harmonic selection
changed — and, because a checkpoint stores its constructor kwargs, would have made every
pre-change checkpoint un-rebuildable. Assembling and checking the streams is the *task's* job:
`SeqVaeLagAttnTask._build_source_stream` and `_build_target_streams` concatenate the fields and
raise a clear error naming the actual per-field widths, the configured value, and the config key
that fixes it.

> **The number $58$ is ambiguous across that change**: it is the current `use_up_st=True` width
> ($43 + 15$) and was the old `use_up_st=False` width. Decide from `use_up_st` first, then set the
> width — never migrate $c_u$ by pattern-matching the number.

Three geometry guards — the ones the model genuinely owns — still run at construction:
$\mathrm{num\_heads} \cdot d_{\mathrm{head}} = d_{\mathrm{model}}$, $d_z \bmod
\mathrm{num\_heads} = 0$ when `head_structured_latent=True`, and `max_lag >= 0`. A fourth checks
only that $c_y$ and $c_u$ are positive — a dataset-independent fact, and not a cosmetic one:
`nn.Linear(0, d_model)` is legal and returns its bias, so a zero width would build a model that
trains to completion having never read that stream.

## 3. Notation and geometry

Fixed by the preprocessing and the shipped config (`configs/default.yaml`):

| Symbol | Value | Meaning |
| --- | --- | --- |
| $f_s$ | $4$ Hz | raw sampling rate |
| decimation | $16$ | raw $\to$ decimated; $\Delta t = 4$ s per step |
| $B$ | — | batch |
| $T$ | $300$ | decimated steps ($\approx 20$ min) |
| $H_d$ | $30$ | forecast horizon ($\approx 120$ s) |
| $w_{\mathrm{warm}}$ | $30$ | leading steps excluded from the KL and the feature loss |
| `max_lag` | $90$ | attention window is $L = \mathrm{max\_lag} + 1 = 91$ lags ($\approx 6$ min of UP history) |
| $c_y$ | $109$ | target channels ($43$ scattering $+\ 66$ phase-harmonic) |
| $c_u$ | $58$ or $15$ | source channels (see §2) |
| $d_{\mathrm{model}}$ | $128$ | encoder / attention width |
| $d_z$ | $24$ | latent width |
| $M$ | $4$ | attention heads ($d_{\mathrm{head}} = 32$) |

$T = 300$ is what the **loader** delivers, not what the HDF5 stores. The stored geometry is $330$
decimated steps; `CombinedHDF5Dataset` trims $15$ steps from each end under `trim_minutes: 1.0`
(the shipped `dataset_kwargs` value) to remove the transform's reflection-padding edge artifacts.
**The model performs no trimming and must not** — omit `trim_minutes` and the loader silently
yields $330$ steps, which every module here will happily accept.

> **Lag readouts are on the stored timeline, and it is canonical.** The pipeline's MIMO adaptor
> is built with `up_shift_secs=-20`, so the UP trace is advanced $80$ raw samples ($5$ decimated
> steps) earlier than FHR before any transform is taken
> (`hdf5_dataset/dataset_explained_research.md` §3.4). That shift is part of how the stored
> signals are. Every lag this model reports is measured on the stored grid and stays there: an
> attention peak at lag $\ell$ is $4\ell$ seconds, and no downstream figure, column or reading
> adds the builder's shift back or subtracts it. The `up_shift_secs` evaluation key that once did
> so was removed on 2026-09-05.

The anchor support is $\mathcal{A} = [w_{\mathrm{warm}}, T - H_d)$: the steps whose $H_d$-long
forecast window is fully inside the observed sequence. The tiny test geometry
(`teb_vae/lag_attn/tests/conftest.py::TINY_KWARGS`) shrinks $T$, $d_{\mathrm{model}}$, $d_z$, $H_d$,
$d_{\mathrm{head}}$, `max_lag` and `warmup_period` ($30 \to 2$) and turns dropout off, while keeping
every channel count and invariant. It is the *geometry*, not the production feature set: `TINY_KWARGS`
leaves `causal_norm`, `kld_support`, `head_structured_latent`, `use_entmax`, `lag_bias_init`,
`horizon_film` and `encoder_extra_dilations` at their constructor defaults, so only `SHIPPED_KWARGS`
exercises the paths a production model takes (§25).

## 4. End-to-end forward pipeline

`forward` runs the following, all in the $(B, T, \cdot)$ layout unless noted:

1. **Adapt.** $Y$ and $U$ are projected to $d_{\mathrm{model}}$ by their own `InputAdapter` (§6).
2. **Encode.** `target_encoder` and `source_encoder` produce the causal history states
   $H^y, H^u \in \mathbb{R}^{B \times T \times d_{\mathrm{model}}}$ (§7).
3. **Prior.** `prior_head(H^y)` returns $\mu^p$, $\log\sigma^{2,p}$ (bounded), the raw pre-bound
   $\widetilde{\log\sigma^{2,p}}$, and a target-only `decoder_state` (§8).
4. **Attend.** The causal lag-validity mask is intersected with any `lag_band_mask`, then
   `lag_attn(H^y, H^u, m_lag)` returns the fused attended source $A$, the per-lag weights $\alpha$
   in lag order, and the per-head summaries $a^{(m)}$ (§9). Dead anchors (every valid lag masked
   away) are ablated: $\alpha$ and $a^{(m)}$ go to zero, while $A$ takes $W_o$'s bias (§9.4).
5. **Posterior.** `posterior_head` builds $q$ as a bounded residual on the prior, conditioned on
   $A$ (flat mode) or on the per-head summaries $a^{(m)}$ (head-structured mode) (§10).
6. **Sample.** $z = \mu^q + \sigma^q \epsilon$, $\epsilon \sim N(0, I)$ (§11).
7. **Decode.** `baseline_decoder(decoder_state)` forecasts $\hat{Y}^{\mathrm{base}}$;
   `residual_decoder(decoder_state, z)` forecasts the correction $\Delta\hat{Y}^{\mathrm{src}}$;
   $\hat{Y}^{\mathrm{full}} = \hat{Y}^{\mathrm{base}} + \Delta\hat{Y}^{\mathrm{src}}$ (§12).
8. **Read out.** `kld_tensor` gives the per-$(t,j)$ KL; `te_analysis` collapses it to $K_t$, the
   per-head KL groups, and the lag attribution `te_lag_map` (§13).

`forward` returns a $24$-key dict (§17). `encode_only(...)` runs steps 1–6 only and returns an
$11$-key subset — it skips the decoders and the diagnostics, which are most of the compute and none
of the latent, so any analysis that needs only $z$ or $K_t$ need not pay for them. With
`sample_z=False`, `encode_only` returns the posterior mean as `z` instead of a sample.

## 5. Shared building blocks (`blocks.py`)

Small, generic components used by more than one module. They are *vendored* into this package rather
than imported from the tree it replaces, deliberately: that older module pulls in `numpy`, `loguru`,
`logging` and `torch._dynamo` at import time, none of which a network component needs, and all of
which would make a net impossible to construct without a configured logger. A test
(`test_nets_are_framework_free.py`) enforces that the nets stay framework-free.

- **`ResidualMLP`** — a per-timestep MLP applied to the channel axis only, so it mixes features
  without ever mixing time and needs no causal mask. Optional input `LayerNorm`, a projected skip
  connection, and an explicit `hidden_dims` tuple — the geometric funnel is not built in; call sites
  pass a schedule produced by `geometric_schedule`. It is the workhorse: the adapters, the encoders'
  front/fusion stages, the prior/posterior heads, and the decoder projections are all built from it.
- **`CausalMultiChannelConvBlock`** — a pre-norm residual 1-D convolution block. Normalisation and
  activation come *before* the convolution, leaving the residual path an unmodified identity through
  the whole stack — while `in_channels == out_channels`, which is how the encoders build every
  block; a width change inserts a $1\times1$ projection on the skip. Causality comes from
  left-padding by $(\text{filter}-1)\cdot\text{dilation}$, so the output at $t$ reads inputs at
  $t' \le t$ only. Its `pre_norm` is a `GroupNorm` — see §7.1.
- **`CausalGroupNorm`** — a drop-in `GroupNorm` replacement that normalises over the channels of
  each group **at each timestep independently**, with no pooling across time. It registers exactly
  the parameters (`weight`, `bias`, both $(C,)$) an *affine* `nn.GroupNorm` does, under the same
  names, so swapping it in leaves a state-dict aligned key-for-key and shape-for-shape. (It has no
  `affine` flag and always creates the pair; every encoder `GroupNorm` takes the `affine=True`
  default, so the alignment holds in this model.) `causalize_norms` performs the in-place swap and
  copies the affine parameters, so the swap is a no-op for the affine transform and changes only
  which elements the statistics pool over.
- **`validate_choice(value, allowed, name)`** — the shared enum guard behind the constructor's
  string arguments (`kld_support`, `lag_bias_init`), raising with the offending value and the
  permitted set rather than failing later on an unrecognised branch.
- **`smooth_bound(r, lo, hi)`** $= lo + (hi - lo)\,\sigma(r)$ — a scaled sigmoid mapping a raw value
  into the open interval $(lo, hi)$. Unlike `torch.clamp` its gradient is strictly positive
  everywhere, so a saturated log-variance can still recover (under a hard clamp the recovering
  gradient is exactly zero). It is **not idempotent**, which is why every head that uses it returns
  its pre-bound raw value alongside the bounded one — the posterior must build its residual on the
  raw value (§10).
- **`geometric_schedule(in, out, n)`** — interpolates $n{+}1$ layer widths so each layer changes
  width by the same *ratio*, keeping the per-layer compression even across a funnel.
- **`initialization(model)`** — Xavier-uniform for linear/conv weights, orthogonal for LSTM
  recurrent/input weights, zeros for biases, ones for `LayerNorm` scales, and — the one non-obvious
  step — the LSTM `bias_hh` forget-gate slice set to $1$ (the `bias_ih` slice stays zero, so the
  *effective* forget bias $b_{ih} + b_{hh}$ starts at $1$) so the gate starts open and gradients
  survive the early steps of a long sequence.

## 6. Input adapters (`encoders.py::InputAdapter`)

One class serves both streams; they differ only in `in_dim`. The adapter is
`Linear` $\to$ `LayerNorm` $\to$ `GELU` $\to$ `Dropout` $\to$ `ResidualMLP`, projecting
$(B, T, \text{in\_dim}) \to (B, T, d_{\mathrm{model}})$. `in_dim` is required rather than defaulted
because the two streams have different widths and a default would silently fit only one of them.

## 7. Causal CNN–LSTM encoders (`encoders.py::CausalConvLstmEncoder`)

Each encoder turns a projected stream into a per-step history state. Two branches run in parallel
over the projected input and are then fused:

- a **dilated causal convolution stack** — `CausalMultiChannelConvBlock`s with an exponential
  dilation schedule, buying a wide but *fixed* receptive field without buying depth. Each block
  after the first is combined with its own input through a skip `GroupNorm`,
  $\mathrm{out}_i = \mathrm{conv}_i(\mathrm{out}_{i-1}) + \mathrm{norm}_{i-1}(\mathrm{out}_{i-1})$,
  so there are $n_{\mathrm{blocks}} - 1$ of these. The stack exit adds the front MLP's output back
  as a long residual and normalises:
  $\mathrm{conv\_out} = \mathrm{LayerNorm}(\mathrm{out}_{n-1} + x_{\mathrm{front}})$;
- a **unidirectional LSTM** — carrying unbounded history in a fixed-width recurrent state
  (`hidden_size = d_model`, so the branch neither widens nor narrows), followed by a `LayerNorm`.

They fail in different directions — the conv stack cannot remember past its receptive field, the
LSTM blurs what it does remember — so concatenating them $(2 d_{\mathrm{model}})$ and fusing back
down through a `ResidualMLP` lets each cover the other. A front `ResidualMLP` precedes both branches
and a final `output_norm` `LayerNorm` caps the exit to roughly per-step $N(0, I)$; without it the
exit drifts unbounded, which downstream shows up as a single latent dimension sitting at an absurd
prior mean.

The two encoders share this architecture and differ only in their base convolution kernel schedule
(target `(3, 7, 11)`, source `(3, 5, 11)`); both run at $d_{\mathrm{model}}$, because the streams'
differing channel counts have already been equalised by their own `InputAdapter` upstream.
`encoder_extra_dilations`
appends one extra block per requested dilation, each at kernel size $15$, extending both stacks'
receptive fields; the base dilation schedule is $(1, 2, 4)$ and the production config appends
$(8, 16)$.

### 7.1 Causal normalisation (`causal_norm`)

Every component above is causal — left-padded convolutions, a unidirectional LSTM, and per-step
`LayerNorm`s — **except** the `GroupNorm`s inside the conv blocks and between them. `nn.GroupNorm`
on a $(B, C, T)$ tensor reduces over every non-batch dimension inside a group, i.e. over $(C/G, T)$,
so its statistics **pool across time**: the normalised value at $t$ becomes a function of the whole
sequence, including $t' > t$. Then $H^y_t$ carries a low-bandwidth image of its own future, the
"prior" $p(z_t \mid Y_{\le t})$ secretly conditions on that future, and $K_t$ is no longer a
transfer-entropy surrogate at all. The leak is small and completely invisible in a loss curve — it
corrupts only the quantity the model exists to measure. Measured relative leak into
`target_state[t]` from resampling $Y_{>t}$: $\approx 11.5\%$ at $T{=}32$, $\approx 3.4\%$ at
$T{=}300$.

`causal_norm=True` (production) calls `causalize_norms` on both encoders, swapping every encoder
`GroupNorm` for a `CausalGroupNorm`. Each encoder holds one `pre_norm` per conv block plus
$n_{\mathrm{blocks}} - 1$ inter-block skip norms, so the total across both encoders is
$2(2 n_{\mathrm{blocks}} - 1)$: **10** in the base 3-block geometry, **18** in production, where
`encoder_extra_dilations: [8, 16]` takes each stack to 5 blocks. `n_causalized_norms` records the
actual count. Measured
leak after the swap: $0.0$. Because the swap preserves parameter names and shapes, a state-dict
still aligns key-for-key. The decoders and the shared horizon core are deliberately **not**
causalised — their `GroupNorm`s pool over the forecast-horizon axis of a *single anchor*, mixing
that anchor's outputs rather than reaching across input time (§12). `causal_norm=False` restores the
leaky non-causal behaviour and the trainer emits a warning, because `kld_raw` is then not a TE
surrogate.

## 8. Target-only prior head (`heads.py::PriorHead`)

From $H^y$ the prior head produces four outputs from **three** heads — mean, log-variance and
decoder state — each head fed through its own input `LayerNorm` so the raw encoder state cannot
drift unbounded through any of them:

- `mu_prior` $\mu^p \in \mathbb{R}^{B \times T \times d_z}$, bounded by
  $\mu^p = \mu\_scale \cdot \tanh(\text{raw}/\mu\_scale)$ so $|\mu^p| \le \mu\_scale$;
- `logvar_prior` $\log\sigma^{2,p}$, smooth-bounded to `logvar_clamp` $= [-5, 3]$;
- `decoder_state` — the target-only conditioning $(B, T, d_{\mathrm{model}})$ consumed by the
  baseline decoder (and, concatenated with $z$, by the residual decoder);
- `raw_logvar_prior` $\widetilde{\log\sigma^{2,p}}$ — the *pre-bound* log-variance. It is not a
  fourth pathway: it is the output of the same `LayerNorm` and the same log-variance head as
  `logvar_prior`, captured before `smooth_bound` is applied.

The fourth output is not a diagnostic. Because `smooth_bound` is a sigmoid and therefore not
idempotent, the posterior cannot build an exact residual from the already-bounded value; it needs
the raw one, or the zero-KL-at-init property is lost (§10). The prior head is where the model's
source-purity lives: it is a function of $H^y$ alone and never touches $H^u$.

## 9. Lag cross-attention (`attention.py::LagCrossAttention`)

Multi-head cross-attention from the target state (queries) to a sliding window of lagged source
states (keys/values). The query at step $t$ attends over $\{H^u_{t-\ell}\}_{\ell=0}^{L-1}$ and the
attention weights are themselves a readout: they say *which lag* the model found informative. It is
pre-norm (`q_norm` on $H^y$, `kv_norm` on $H^u$), with the standard $1/\sqrt{d_{\mathrm{head}}}$
scaling.

### 9.1 The window is a view, not a bank

Keys and values are projected from $H^u$ exactly once, then the lag window is formed with strided
`unfold` views over the projected tensors (`F.pad` by $L-1$ on the left of the time axis, then
`unfold`). Projecting once rather than once per lag is the real saving, and the window itself is a
genuine view — but **do not read this as an $L\times$ activation-memory saving**: the two consuming
`einsum` calls materialise contiguous copies of the non-contiguous windows and retain them for the
backward pass, so peak retained activation is of the same order as materialising the window
outright. `attention_grad_checkpoint=True`, which recomputes the attention in the backward pass, is
therefore the lever that actually trades compute for attention memory, not the `unfold`.

The window is indexed oldest-first, so every lag-indexed tensor is flipped on the small $L$ axis
before or after use — the Shaw bias, the ALiBi score bias, the mask, and the returned $\alpha$;
the large windows stay views rather than copies.

### 9.2 Lag biases

Two learned lag biases shape the scores:

- a **Shaw-style per-lag key bias** `lag_embeddings` $(L, M, d_{\mathrm{head}})$, added to the
  content score as $\langle q_t, r_\ell\rangle$;
- an optional **per-(head, lag) scalar score bias** `lag_score_bias` $(M, L)$, present only when
  `lag_bias_init='alibi_decay'`.

The two enter at different points relative to the $1/\sqrt{d_{\mathrm{head}}}$ scaling, which is
worth knowing before tuning either:

$$s_{t,\ell}^{(m)} = \underbrace{\big(\langle q_t^{(m)}, k^{(m)}_{t-\ell}\rangle
+ \langle q_t^{(m)}, r_\ell^{(m)}\rangle\big)}_{\text{scaled}} \big/ \sqrt{d_{\mathrm{head}}}
\;+\; \underbrace{b^{(m)}_\ell}_{\text{not scaled}}.$$

The Shaw bias is scaled with the content score because it is a dot product in the same space; the
ALiBi score bias is added afterwards, in raw score units. So `alibi_slope_scale` is calibrated
against post-scaling logits — at $d_{\mathrm{head}} = 32$ a unit of `lag_score_bias` is
$\sqrt{32} \approx 5.7\times$ larger, relative to the content term, than the same number would be
inside the parenthesis.

With `lag_bias_init='alibi_decay'` (production) the score bias is seeded to a negative slope in
$\ell$: $-m_h\,\ell$, where the per-head slopes $m_h$ follow the geometric ALiBi power-of-two
schedule (Press et al., 2022) scaled by `alibi_slope_scale`. This penalises long lags at init, so
the model begins biased toward short lags and must *earn* a long-lag reading — the physiologic prior
that recent UP history matters more, and a guard against a randomly-initialised head latching onto
spurious long-lag structure early. `alibi_slope_scale` below $1$ softens the penalty so a genuinely
long coupling is still reachable; $0$ gives a flat but still learnable bias. The published schedule
is defined for a power-of-two head count (production has $M = 4$); the code carries the reference
implementation's interpolation for other counts, which no shipped configuration exercises.
`lag_bias_init='normal'` creates no score-bias `Parameter` — the attribute is registered as `None`,
so nothing appears in the state-dict. The Shaw key bias is present in both modes and is the target
of the optional lag-smoothness penalty (§15).

### 9.3 Masking and normalisation

Scores are masked by lag validity ($t - \ell \ge 0$; early steps have fewer valid lags) and
normalised over the lag axis by either `softmax` or `entmax15` (`use_entmax`). `entmax15` is worth
the dependency: unlike `softmax` it can assign a lag *exactly* zero weight rather than merely a
small one, and when the output is read as "which lag mattered", the difference between $0$ and
$10^{-4}$ across $91$ lags is the difference between a clean answer and a smear. Production sets
`use_entmax=True`. Under `softmax`, a row whose every valid lag is masked normalises to `NaN` and
`nan_to_num` maps it to $0$, which is the correct reading — no lag was attended because none was
available. `entmax15` never reaches that path: it *raises* on an empty support, which is why the
model forces lag $0$ back on at those anchors before calling the attention and ablates them
afterwards (§9.4).

**The normalised weights then pass through attention dropout** (rate = the model's `dropout`, $0.1$
in production) *before* being returned and before forming `head_out`. So in `train()` mode neither
the returned $\alpha$ nor the per-head summaries are a normalised distribution — rows sum to a
random value near $1$. Every reading of $\alpha$ as "which lag mattered", and the
$\sum_\ell \widetilde{TE}_{t,\ell} = K_t$ identity of §13, holds in `eval()` mode only. This is why
`measure_transfer_entropy` switches modes rather than trusting the training-time tensors.

The attention returns three tensors: the fused attended source $A = W_o(\text{head\_out})$
$(B, T, d_{\mathrm{model}})$; the weights $\alpha$ $(B, T, M, L)$ in lag order (index $0$ is the
current step); and the per-head summaries `head_out` $(B, T, M, d_{\mathrm{head}})$ taken *before*
$W_o$. The head-structured posterior consumes `head_out`; the flat posterior consumes $A$.

### 9.4 Band masking and dead anchors

The optional `lag_band_mask` is an **ablation** tool. The model intersects it with the causal
validity mask in `_combined_lag_mask` — a bare band mask passed straight to the attention would
*replace* the internally-built validity mask and silently destroy the causal constraint, so the
model would attend to lags that do not exist and report their KL as a TE measurement.

Be precise about what the band does to the *distribution*. Masked lags are set to $-\infty$ and the
normaliser runs over the survivors, so a partially masked row still sums to $1$ and the removed
mass **is** redistributed across the lags that remain — the ablation removes a *pathway*, not a
quantity of attention. What is never rescaled is a **dead** anchor, where no valid lag survives at
all: there the row is forced to $0$ rather than renormalised.

When the kept band excludes lag $0$, an anchor can have *no* surviving valid lag — that dead anchor.
`softmax` degrades gracefully there, but `entmax15` raises (its support size is $0$). To
keep the activation well-posed, lag $0$ (always causally valid) is forced back on at those anchors;
`_ablate_dead_anchors` then overwrites the result to exactly what `softmax`'s all-$-\infty$ path
would give — $\alpha = 0$, per-head summary $a^{(m)} = 0$, and fused $A = W_o(0) = $ $W_o$'s *bias*
(not necessarily zero) — so the two normalisers agree at precisely the anchors an ablation creates.
Dead anchors are exactly the steps $t < \min(\text{kept lags})$; the test is computed from the mask
alone and makes no reference to `warmup_period`. **They fall entirely inside the warm-up prefix, and
so out of every loss, only when $\min(\text{kept lags}) \le w_{\mathrm{warm}}$** — a band starting
beyond the warm-up (e.g. keeping lags $40\ldots50$ at $w_{\mathrm{warm}} = 30$) puts dead anchors
into the supervised support, carrying $\alpha = 0$ and $A = W_o$'s bias. Check that bound when
choosing an ablation band. When `lag_band_mask`
is `None`, `_combined_lag_mask` returns `(None, None)` and the attention is called exactly as it
would be without the feature, so the no-mask path is **bit-exact**, not merely equivalent.

## 10. Source-conditioned posterior head (`heads.py::PosteriorHead`)

The posterior is a *bounded residual* around the prior:

$$\mu^q_t = \mu^p_t + s_\mu \tanh\!\big(\widetilde{\Delta\mu}_t / s_\mu\big), \qquad
\log\sigma^{2,q}_t = \mathrm{smoothbound}\!\big(\widetilde{\log\sigma^{2,p}_t}
+ s_\ell \tanh\!\big(\widetilde{\Delta\ell}_t / s_\ell\big),\, -5,\, 3\big),$$

with $s_\mu = $ `delta_mu_scale` $= 3$ and $s_\ell = $ `delta_logvar_scale` $= 2$. Two properties are
load-bearing:

- **Earned from zero.** Both delta heads are zero-initialised, and $\tanh(0) = 0$, so at
  initialisation $\Delta\mu_t = \Delta\ell_t = 0$, giving $\mu^q_t = \mu^p_t$ and
  $\log\sigma^{2,q}_t = \log\sigma^{2,p}_t$ **exactly**, hence $K_t \equiv 0$. The model begins
  asserting that the source says nothing, and every later nat of $K$ is earned by source
  conditioning rather than produced by random mismatch between two independent variance heads.
  `_zero_init_delta_heads` (called once at construction, after the generic init, never on a trained
  model) zeroes the posterior's `delta_mu_head` and `delta_logvar_head` and the residual decoder's
  `mean_head`. `test_zero_kl_init.py` pins $\max|K_t| < 10^{-6}$ at init.
- **Residual on the raw prior.** The log-variance delta is added to the prior's *pre-bound raw*
  value before bounding, not to the already-bounded output. Because `smooth_bound` is not idempotent,
  bounding the prior and then adding a zero delta would leave $\log\sigma^{2,q}_t \neq
  \log\sigma^{2,p}_t$ at init and break the exact zero. The forward therefore threads
  `raw_logvar_prior` through, and `posterior_head` raises if it is not supplied.

Two latent structures are supported:

- **Flat** (`head_structured_latent=False`): the posterior concatenates $[LN(H^y) \,\|\, LN(A)]$,
  fuses through a `ResidualMLP`, and produces the whole $d_z$-vector of deltas at once.
- **Head-structured** (`head_structured_latent=True`, production): the latent is partitioned into
  $M$ contiguous groups of $d_z/M$ dims, and group $m$'s deltas come from its own fusion
  `ResidualMLP` and delta pair over $[LN(H^y) \,\|\, LN(a^{(m)})]$ — the *shared* target state
  concatenated with **only** head $m$'s summary. (The two input `LayerNorm`s are shared across
  groups; the fusion MLP and the delta pair are per-group `ModuleList` entries. A shared affine over
  disjoint inputs preserves the decomposition.) The target state is common to every group; what is
  head-local is the source. That is exactly the property the decomposition needs: any nat of
  $K_t^{(m)}$ is a divergence from a prior built on the same $H^y$, so it can only have been
  produced by head $m$'s view of the source. Hence $K_t = \sum_m K_t^{(m)}$ is a genuine additive
  decomposition rather than an arbitrary slice of a shared vector. It requires $d_z \bmod M = 0$ and
  consumes `attended_source_heads`. The prior stays shared and target-only in both modes.

## 11. Latent sampling

$z = \mu^q + \sigma^q \epsilon$ with $\epsilon \sim N(0, I)$ and $\sigma^q = \exp(\tfrac{1}{2}
\log\sigma^{2,q})$ (`reparameterize`). The noise is a separate factor rather than a draw from
$N(\mu^q, \sigma^{2,q})$ directly, which keeps $\mu^q$ and $\sigma^q$ on the gradient path.

`forward` samples **unconditionally** — there is no `self.training` gate — so two `eval()` passes on
identical inputs return different `z`, `mu_full` and `logvar_full`. The KL readouts are unaffected
(they are functions of $\mu$ and $\log\sigma^2$ only). Only `encode_only` offers `sample_z=False`,
which returns $\mu^q$ in place of a sample; it defaults to `True`. (The
older tree carried a latent-statistics EMA-normalisation mechanism; it had no consumer in this tree
and was removed in the flatten, along with its three running buffers — see `DESIGN.md` §8.)

## 12. Future decoders (`decoders.py`)

The model forecasts the next $H_d$ target steps twice, and both decoders share one
`HorizonDecoderCore`:

- **`HorizonDecoderCore`** holds a per-step forecast-step embedding `horizon_embedding`
  $(H_d, d_{\mathrm{hidden}})$ (which step of the horizon this is, not what time it was in the
  recording), an optional FiLM modulation, a dilated `_HorizonRefine` stack, and an output
  `LayerNorm`. `decode` expands a per-step state $(B, T, d_{\mathrm{hidden}})$ over the horizon to
  $(B, T, H_d, d_{\mathrm{hidden}})$, adds the step embedding, optionally FiLM-modulates, and
  refines. The refine stack's convolutions run along the *horizon* axis with symmetric padding and
  doubling dilations ($1, 2, 4, \dots$ over `horizon_depth` blocks); this axis is not time — every
  step of it is predicted from the same anchor $t$ — so symmetric padding and its `GroupNorm`s leak
  nothing. The stack's output is added back to its pre-refine input before the output norm,
  $\mathrm{LayerNorm}(\mathrm{refine}(f) + f)$. Note this composite is **not** an identity at init
  and cannot be made one: `_HorizonRefine` is itself internally residual, so zeroing its
  convolutions gives $\mathrm{refine}(f) = f$ and the outer add yields
  $\mathrm{LayerNorm}(2f)$ — and the output `LayerNorm` runs unconditionally in any case. The refine
  stack is *not* zero-initialised. FiLM is the part that genuinely starts as an identity: its
  generator is zero-initialised and the modulation is $f \cdot (1 + \gamma) + \beta$, so
  $\gamma = \beta = 0$ leaves $f$ untouched. That $(\gamma, \beta)$ pair is generated from the
  anchor's own state and broadcast across the horizon, i.e. it is one modulation per anchor, not per
  horizon step.
- **`BaselineFutureDecoder`** consumes `decoder_state` only, projects it through a `ResidualMLP`,
  runs the shared core, and emits $\hat{Y}^{\mathrm{base}}$ (`mu_base`) and a heteroscedastic
  `logvar_base`, both $(B, T, H_d, c_y)$. This is the branch that must be good on its own.
- **`ResidualFutureDecoder`** consumes `decoder_state` *and* the latent $z$ (concatenated), runs the
  *same* core instance, and emits the correction $\Delta\hat{Y}^{\mathrm{src}}$ (`delta_mu_src`) and
  `logvar_full`. Its `mean_head` is zero-initialised — by the *model*, in `_zero_init_delta_heads`,
  not by the decoder's own constructor, so a `ResidualFutureDecoder` built standalone does not have
  this property — giving $\Delta\hat{Y}^{\mathrm{src}} = 0$ at init and a full forecast equal to the
  baseline exactly; divergence is learned, not assumed.

The full forecast is $\hat{Y}^{\mathrm{full}} = \hat{Y}^{\mathrm{base}} +
\Delta\hat{Y}^{\mathrm{src}}$. Sharing the core makes the correction a genuine correction *in the
baseline's own representation space*: learning the horizon dynamics twice would cost parameters and,
worse, let the two forecasts drift into different spaces, at which point their difference stops being
a correction and becomes a comparison of strangers.

## 13. KL and transfer-entropy readouts

**Closed-form KL.** `kld_tensor` computes the per-$(B, T, d_z)$ KL between the two diagonal
Gaussians in closed form:

$$K_{t,j} = \tfrac{1}{2}\!\left[\log\sigma^{2,p}_{t,j} - \log\sigma^{2,q}_{t,j}
+ \frac{\sigma^{2,q}_{t,j} + (\mu^q_{t,j}-\mu^p_{t,j})^2}{\sigma^{2,p}_{t,j}} - 1\right],
\qquad K_t = \sum_{j=1}^{d_z} K_{t,j}.$$

It is closed-form rather than Monte-Carlo because this quantity is the model's *output*, not an
intermediate, and a sampled estimate would inject variance straight into the number being reported.
`kld_tensor` returns the raw, full-$T$, warmup-unmasked KL — masking is the caller's job because
every caller wants a different window (the training term masks to `kld_support`, the reported curve
masks the warm-up prefix to `NaN`, the permutation control masks nothing).

**Lag attribution.** `TEAnalysisHead` is a pure, parameterless function that turns $K_t$ and
$\alpha$ into a lag-resolved attribution. $K_t$ says *how much* the source contributed; $\alpha$
says *which lag* it came from. In head-structured mode the attribution is the rigorous
decomposition

$$\widetilde{TE}_{t,\ell} = \sum_m K_t^{(m)}\,\alpha^{(m)}_{t,\ell},$$

because $K_t^{(m)}$ is the KL of a group that head $m$ alone produced. Without head structure the
per-head split is an arbitrary slice, and the fallback — $K_t$ times the head-mean attention — is a
diagnostic, not a decomposition. The head returns `kld_per_t` $(B, T)$, `te_lag_map` $(B, T, L)$,
and `kld_per_t_per_head` $(B, T, M)$.

`kld_per_t_per_head` is computed the same way in **both** modes: $d_z$ is cut into $M$ contiguous
in-order blocks of $d_z/M$ and each is summed, whenever $d_z \bmod M = 0$. (When it is not — which
`PosteriorHead` rejects at construction, so this is reachable only by driving `TEAnalysisHead`
directly — the head falls back to $K_t / M$ in *every* slot, a uniform split that is more
misleading than a partition, not an omission.) `te_lag_map`'s definition switches on
`head_structured_latent` **and** that same divisibility. So the key is always populated and always
looks reasonable — under a flat latent it is a partition of a shared vector that no head owns, and
reading it as a per-head attribution is the specific mistake this paragraph exists to prevent.

> **Note.** The model exposes *no* per-sample scalar TE distinct from $K_t$: `te_lag_map` summed
> over lags equals $K_t$ — **in `eval()` mode, at anchors that are not dead**. Attention dropout
> perturbs the row sums in `train()` (§9.3), and an ablation's dead anchors carry $\alpha = 0$
> against a nonzero $K_t$ (§9.4), so the identity is not unconditional. Per-sample TE-versus-KLD
> analysis must in any case use a ground-truth injected/empirical TE, not a "model TE".

**KL support.** `_kld_support_mask` builds the $(T,)$ time-support mask for the training KL.
`kld_support='full'` masks only the warm-up prefix; `kld_support='anchor'` (production) additionally
masks the final $H_d$ steps, whose forecast windows run off the end of the sequence and so receive
no supervised gradient. Left in, those anchors' KL is regularised toward the prior by $\beta$ with
nothing pulling the other way, and the resulting tail collapse is easy to misread as a real drop in
coupling. The same mask is shared by the training KL, `measure_transfer_entropy`, and
`kld_active_frac`, so $K_{\mathrm{true}}$ and $K_{\mathrm{shuffled}}$ stay comparable.

**Collapse diagnostic.** `kld_active_frac` (in $[0, 1]$) is the fraction of latent dimensions whose
mean KL — over the batch *and* the support, strictly greater — exceeds $10^{-2}$. A model can post a
healthy total KL while routing all of it through one dimension; this is the headline diagnostic that
distinguishes that from a latent that is actually being used. Unlike the training KL it does not
apply the per-sample validity `weight`, so it treats padded and real steps alike.

**Measurement helper.** `measure_transfer_entropy(y_st, y_ph, u_stream, reduce_mean=False)`
estimates the surrogate for analysis. It switches to `eval` mode (so dropout does not add noise to a
measurement) and **restores the previous mode afterwards** — it is routinely called from a plotting
callback mid-training, and a method that silently left the module in `eval` would disable dropout
for the rest of the run. Both return modes share one support, so an `anchor` model's plotted curve
cannot show a KL spike across the untrained tail that the reported scalar does not contain:
`reduce_mean=False` gives the per-step *per-dimension* tensor $(B, T, d_z)$ — not $(B, T)$ — with
`NaN` outside the support; sum the last axis to get a $K_t$ curve comparable with `kld_per_t`.

> **The two return modes share a support but not a scale.** `reduce_mean=True` averages over batch,
> support steps **and latent dimensions** — it is $\overline{K_t}/d_z$, not $\overline{K_t}$. To
> compare the scalar against the mean of the plotted curve, multiply it by $d_z$ ($24$ in
> production).

## 14. Training objective (`model.py::compute_loss`)

$$\mathcal{L} = \lambda_{\mathrm{full}}\,L_{\mathrm{feat}}
+ \lambda_{\mathrm{base}}\,L_{\mathrm{base}}
+ \beta\,L_{\mathrm{KL}} + \lambda_{\mathrm{lag}}\,L_{\mathrm{smooth}},$$

The two feature terms are evaluated over the valid anchor support
$t \in [w_{\mathrm{warm}}, T - H_d)$. **The other two terms do not share that window**:
$L_{\mathrm{KL}}$ uses `_kld_support_mask`, which is $[w_{\mathrm{warm}}, T)$ under the constructor
default `kld_support='full'` and only narrows to the anchor support under `'anchor'` (production);
$L_{\mathrm{smooth}}$ is a pure parameter penalty with no time support at all.

- **Future target.** For anchor $t$ the target is $Y^{+}_t = Y_{t+1:t+1+H_d}$, built for every
  anchor at once with `unfold` rather than by slicing per anchor in a loop. The `unfold` itself is a
  view, but the subsequent `permute(...).contiguous()` does materialise one
  $(B, T - H_d, H_d, c_y)$ tensor — the saving is the loop, not the memory.
- **$L_{\mathrm{feat}}$** — reconstruction of $Y^{+}$ from the *full* forecast.
- **$L_{\mathrm{base}}$** — the same from the *baseline* forecast. This term is what stops the model
  cheating: without it the baseline could be left weak so that target-explainable variance gets
  pushed through the latent, inflating $K_t$ with information the source never supplied.
- **$L_{\mathrm{KL}}$** — the KL over its own support, deliberately independent of the feature
  window.
- **$L_{\mathrm{smooth}}$** — an optional smoothness penalty on the successive differences of the
  Shaw lag embeddings along the lag axis, encouraging a physiologically plausible (smooth) lag
  profile. When $\lambda_{\mathrm{lag}} = 0$ (the constructor default) the term is not computed at
  all and the reported `lag_smoothness` is a hard $0$ placeholder, **not** a measurement of the
  embeddings' actual smoothness — so that logged series is only informative on a run that enables
  the penalty (production sets $10^{-3}$).

**Likelihood.** With `likelihood='mse'` the per-element loss is squared error. With
`likelihood='gaussian_nll'` it is the per-element Gaussian NLL in nats (constant dropped),
$\tfrac{1}{2}\big(\log\sigma^2_{\mathrm{obs}} + (Y^+ - \mu)^2 / \sigma^2_{\mathrm{obs}}\big)$, where
the observation variance is either a fixed positive scalar `sigma_obs` (homoscedastic debug run) or,
with `sigma_obs='learned'` (production), the decoders' own `logvar_full` / `logvar_base` heads —
giving a per-point predictive Gaussian $\mathcal{N}(\mu_{\mathrm{full}}, \sigma^2_{\mathrm{full}})$.
The masking counts the channel axis in the denominator, so the loss scale matches a mean over
$(B, T_{\mathrm{valid}}, H_d, c_y)$ and does not drift with the mask density.

**Gradient separation.** `detach_baseline_in_full=True` (production) stop-gradients the baseline
inside the full term, so $L_{\mathrm{feat}}$ trains only the residual and source path while
$L_{\mathrm{base}}$ alone shapes the baseline — keeping the baseline an honest source-free reference.

**Two KL scalars.** `compute_loss` returns `kld_train` (the free-bit-floored KL that actually enters
the loss) and `kld_raw` (the un-floored KL over the same support, detached). `kld_train >= kld_raw`
always, because free-bits clamps each per-dimension KL upward before masking. **Only `kld_raw` /
`kld_per_t` may be read as the transfer-entropy surrogate**; `kld_train` is optimisation-only. The
full loss dict has 12 keys: `feat_loss`, `base_loss`, `kld_loss`, `total_loss`, `beta`,
`likelihood`, `mean_logvar_full`, `mean_logvar_base`, `lag_smoothness`, `kld_raw`, `kld_train`,
`kld_active_frac`. `kld_loss` and `kld_train` are the *same tensor* under two names — the first is
the loss term, the second the reporting alias — so only `kld_raw` is a second quantity.
(`likelihood` echoes the input string — a caller forwarding the dict to a metric logger should drop
that key; the task builds its metric dict key by key for exactly this reason, since the logger
coerces a non-numeric value to a clean $0.0$ rather than raising.)

## 15. Bounds, free bits, and lag smoothness

- **Log-variance bounds.** Every log-variance in the model — prior, posterior, and both decoders —
  is smooth-bounded to `logvar_clamp` $= [-5, 3]$, i.e. $\sigma^2 \in (e^{-5}, e^{3})$ — an *open*
  interval, since `smooth_bound`'s endpoints are asymptotes — which is well-conditioned for the
  closed-form Gaussian KL. The lower bound guards against decoder variance collapse
  (`test_logvar_floor.py`). `mean_logvar_full` / `mean_logvar_base` are tracked to watch for that
  collapse, but they are only *meaningful* under `sigma_obs='learned'`: with `likelihood='mse'` or a
  fixed scalar `sigma_obs` the decoder log-variance heads receive no gradient, so the two metrics
  report untrained heads.
- **Free bits.** `free_bits` (production $0.1$ nats per dim per step) floors each per-dimension KL
  before masking. The closed-form Gaussian KL is already non-negative, so a $0$ floor is a no-op;
  the floor exists to keep the *optimised* KL (`kld_train`) from being driven all the way to zero,
  while `kld_raw` still reports the true value.
- **Saturation diagnostics.** `mu_prior_sat_frac` and `delta_mu_sat_frac` report the fraction of
  entries at or beyond $99\%$ of their $\tanh$ bound ($|\mu^p| \ge 0.99\,\mu\_scale$ and
  $|\mu^q - \mu^p| \ge 0.99\,\delta\mu\_scale$). Both are means over *every* $(B, T, d_z)$ entry
  with **no warm-up or support masking**, unlike the other diagnostics here, and both are returned
  by `forward` rather than by `compute_loss`. A bound that is always active is a bound that is
  binding, and a binding bound is a silently mis-set hyperparameter (its gradient vanishes and the
  latent stops responding to the source).

## 16. `mu_scale`, `delta_mu_scale`, `delta_logvar_scale`

These three set the saturation magnitudes of the tanh/smooth bounds on the latent means and the
posterior log-variance delta. `mu_scale` ($=5$) bounds the prior mean; `delta_mu_scale` ($=3$)
bounds the posterior mean's residual on the prior; `delta_logvar_scale` ($=2$) bounds the posterior
log-variance's residual. They are stabilisers: large enough not to restrict the model around the
$N(0, I)$ reference, small enough to prevent a runaway head. All three must be positive — checked at
construction, in the heads rather than in the model, both of which the model builds in its own
`__init__`.

## 17. Forward return dictionary (24 keys)

| Key | Shape | Meaning |
| --- | --- | --- |
| `mu_prior` | $(B, T, d_z)$ | prior mean $\mu^p$ |
| `logvar_prior` | $(B, T, d_z)$ | prior log-variance (smooth-bounded) |
| `raw_logvar_prior` | $(B, T, d_z)$ | pre-bound prior log-variance |
| `mu_post` | $(B, T, d_z)$ | posterior mean $\mu^q$ |
| `logvar_post` | $(B, T, d_z)$ | posterior log-variance (residual on the raw prior) |
| `z` | $(B, T, d_z)$ | sampled posterior latent |
| `target_state` | $(B, T, d_{\mathrm{model}})$ | causal target state $H^y$ |
| `source_state` | $(B, T, d_{\mathrm{model}})$ | causal source state $H^u$ |
| `decoder_state` | $(B, T, d_{\mathrm{model}})$ | target-only decoder conditioning |
| `attended_source` | $(B, T, d_{\mathrm{model}})$ | fused attended source $A$ |
| `attended_source_heads` | $(B, T, M, d_{\mathrm{head}})$ | per-head source summaries |
| `attn_weights` | $(B, T, M, L)$ | lag attention $\alpha$, in lag order |
| `mu_base` | $(B, T, H_d, c_y)$ | baseline forecast mean |
| `logvar_base` | $(B, T, H_d, c_y)$ | baseline forecast log-variance |
| `delta_mu_src` | $(B, T, H_d, c_y)$ | source-driven correction mean |
| `mu_full` | $(B, T, H_d, c_y)$ | full forecast mean $= $ `mu_base` $+$ `delta_mu_src` |
| `logvar_full` | $(B, T, H_d, c_y)$ | full forecast log-variance |
| `kld_per_t` | $(B, T)$ | per-step KL $K_t$ (raw, full-$T$) |
| `kld_per_t_per_head` | $(B, T, M)$ | per-head KL groups |
| `te_lag_map` | $(B, T, L)$ | attention-weighted lag attribution of $K_t$ |
| `warmup_mask` | $(T,)$ bool | `True` outside the warm-up prefix |
| `mu_prior_sat_frac` | scalar | prior-mean saturation fraction |
| `delta_mu_sat_frac` | scalar | posterior-delta saturation fraction |
| `kld_active_frac` | scalar | fraction of latent dims with meaningful KL |

`encode_only` returns the 11-key subset: `mu_prior`, `logvar_prior`, `mu_post`, `logvar_post`, `z`,
`target_state`, `source_state`, `decoder_state`, `attended_source`, `attended_source_heads`,
`attn_weights`.

## 18. Training task (`task.py::SeqVaeLagAttnTask`)

The task is the only place that knows both the net and the data. It assembles the target streams and
the source stream — width-validating both against $c_y$ and $c_u$ — and the per-step validity
`weight` from the batch, runs one forward, and calls `compute_loss`. Its full override set over the
framework base is small: `prog_bar_metrics`, `__init__`, `setup`, `compute_loss_and_metrics` and
`on_save_checkpoint`; everything else it adds is a new private helper. `setup` seeds the
derangement generator from `1234 + global_rank`, deliberately not in `__init__`, where the module is
unattached and `global_rank` is $0$ on every rank, so a rank-seeded generator would produce
identical shuffles everywhere while claiming otherwise. The optimizer, scheduler, `training_step`,
metric logging, and spike breaker all come from the framework base (`LightningModelBase`).
`prog_bar_metrics` is `('total_loss', 'feat_loss', 'kld_raw')` — `kld_raw` rather than `kld_loss`,
since watching the free-bit-floored quantity hides a collapsed source pathway.
`compile_model=False` is forced permanently in its constructor: the `nn.LSTM` encoders, the
checkpointed attention region, and the data-dependent boolean-mask indexing behind
`kld_active_frac` each defeat TorchInductor independently.

**Beta schedule** (`_resolve_beta`). $\beta$ is resolved per epoch. `constant` returns the schedule's
own `value` when it has one and otherwise falls back to the `kld_beta` hyperparameter (as does a
malformed, non-dict schedule); `linear_warmup` (production) ramps linearly from `start` to `end`
over the first `warmup_epochs` epochs, then holds:

$$\beta(e) = \mathrm{start} + (\mathrm{end} - \mathrm{start})\,
\min\!\left(1, \max\!\left(0, \tfrac{e}{\mathrm{warmup\_epochs}}\right)\right),$$

with $\mathrm{warmup\_epochs} \le 0$ short-circuiting to `end`. $e$ is `current_epoch`, which is
$0$-based: $\beta(0) = \mathrm{start}$, and `end` is first reached at epoch index `warmup_epochs`.

A weak early $\beta$ lets the residual decoder learn to use $z$ before the bottleneck tightens for a
calibrated TE reading. Because the KL starts at exactly $0$ and the anchor support changes the
natural KL scale, `beta_schedule.end` is retuned empirically against the logged `kld_raw`
trajectory. An unknown `kind` raises rather than silently falling back to a constant.

**Diagnostics** (`_compute_residual_diagnostics`). `delta_mu_rms` (the masked RMS of the
source-driven mean shift, under the feature mask) and `mu_post_prior_gap_rms` (the masked RMS of the
latent mean gap, under the KL's *own* support taken from the model rather than rebuilt). Rebuilding
the KL support here is how the two would drift apart under `kld_support='anchor'`, so it is read from
the model. The task also logs `pred_gap` $= L_{\mathrm{base}} - L_{\mathrm{feat}}$ (positive when the
source helps).

**Checkpoint contract.** `on_save_checkpoint` calls `super()` first (which stamps `model_class`) and
then writes the exact constructor `model_kwargs`, so a checkpoint is self-describing: the
architecture can be rebuilt with no config file, and `check_model_class` can refuse a blob written by
a different model before that rebuild is attempted.

## 19. Experiment driver (`trainer.py::LagAttnTrainer`)

Builds the net from `model_config.VAE_model` (one `inspect.signature` sweep forwards any flat key
naming a real constructor argument; the two nested blocks `horizon_refine` and `encoder`, and the
`logvar_clamp` list, are the only translations), optionally warm-starts it, wraps it in the task, and
runs the fit through the framework's `build_trainer`. Notable behaviours:

- **Warm-start is checked.** `load_checkpoint_strict` returns `None` rather than raising when nothing
  aligns, so an unchecked call would train a randomly-initialised model that was supposed to be
  warm-started and report nothing; the driver raises instead. `check_model_class` runs *before* the
  load.
- **`causal_norm=False` warns** that `kld_raw` is not a TE surrogate.
- **DDP strategy selection** (`select_ddp_strategy`). A single-device run returns `'auto'` before any
  of the following is considered; the choice below is what happens from two devices up. Plain `'ddp'`
  implies
  `find_unused_parameters=False`, under which the reducer expects *every* parameter marked ready in
  every backward. Two parameter groups can go unused, both decided by config: the decoder
  log-variance heads (consumed exactly when `likelihood='gaussian_nll'` and `sigma_obs='learned'`),
  and — under `head_structured_latent=True` — the attention output projection $W_o$, which then feeds
  only the diagnostic `attended_source` key and receives no gradient. The selector returns plain
  `'ddp'` when the log-variance heads are consumed and $W_o$ is not starved (i.e.
  `freeze_unused_attn_proj` has taken it out of the expectation set), else
  `'ddp_find_unused_parameters_true'`. It sources every input from `config`, deliberately not from
  the passed Lightning module, whose attributes would not answer the question.
- **Spike breaker as a non-finite guard only.** The framework breaker's relative test is
  $\text{watched} > \text{multiplier} \cdot \max(\mathrm{EMA}, \mathrm{ema\_floor})$, which assumes a
  loss bounded below by zero. This model's `main_loss` is a Gaussian NLL with learned observation
  variance and goes negative routinely; once the EMA is negative, $\max(\mathrm{EMA}, 0)$ is $0$ and
  every positive batch reads as a spike, so the run silently drops its hardest batches and no value
  of `ema_floor` rescues the relative test. The shipped config sets `ema_floor: 1.0e9` — far above
  any reachable loss — which switches the relative test off while leaving the non-finite guard
  intact. That guard is the part worth keeping: it stops a `NaN` loss from writing `NaN` gradients
  into every weight.

`freeze_unused_attn_proj=True` clears `requires_grad` on $W_o$'s parameters when
`head_structured_latent` is also set (a numerical no-op — AdamW never updated them), which is what
lets the run use plain `'ddp'`. `test_ddp_strategy.py` licenses `find_unused_parameters=False` by a
grad-coverage assertion over *every* parameter on both perm and non-perm steps, not by the strategy
string.

## 20. Source-permutation control (`nets/controls.py`)

A high $K_t$ on its own proves nothing: the posterior sees the source, so it reacts to *any* source,
including one from a different recording. The control deranges the batch so every target is paired
with a stranger's source and asks what happens. It ships as a **readout** (`lambda_perm=0`) and
produces two readouts that are not equally useful:

- **KL space** — $K_{\mathrm{shuffled}} = \mathrm{KL}(q(z \mid Y, \pi(U)) \,\|\, p(z \mid Y))$,
  against the same prior. This answers "did the source move my belief?", which a mismatched source
  does too — often *more* strongly, since the posterior only ever trained on matched pairs and a
  stranger's source is out of distribution. So $K_{\mathrm{shuffled}} \gtrsim K_{\mathrm{true}}$ is
  routine even on a model that plainly uses the source. **`kld_raw` alone is therefore not a
  source-specific TE surrogate**, and the spec's original acceptance
  ($K_{\mathrm{shuffled}} < K_{\mathrm{true}} - \text{margin}$) is not achievable. A positive
  `lambda_perm` collapsed the source pathway in half the seeds tried, so it stays $0$.
- **Prediction space** — the control that discriminates, and it needs no auxiliary loss. Feeding the
  decoder a latent drawn from $q(z \mid Y, \pi(U))$ and re-scoring against the true future lands
  *above* the target-only baseline loss on a model genuinely using the source: a wrong source is
  worse than no source. The ordering that means something is

  $$L_{\mathrm{feat}} < L_{\mathrm{base}} < L_{\mathrm{feat, shuffled}},$$

  logged as `feat_loss`, `base_loss`, and `feat_loss_shuffled` with `shuffle_penalty` $=
  L_{\mathrm{feat, shuffled}} - L_{\mathrm{feat}}$.

**When it runs** (`SeqVaeLagAttnTask._should_run_perm`). Not every step: every **non-training** stage
(the gate is `stage != 'train'`, so test as well as validation) runs it on **every** batch —
`kld_shuffled` and `feat_loss_shuffled` are the headline diagnostics and are cheap under `no_grad` —
while training subsamples it to every `perm_every_n_batches`-th batch ($4$ in
production), on the rank-invariant `batch_idx`. A batch with $B < 2$ cannot be deranged and is
skipped, and the decision is `MIN`-reduced across ranks (below). Its metrics are `kld_shuffled`,
`kld_shuffled_ratio` ($K_{\mathrm{shuffled}} / K_{\mathrm{raw}}$), `feat_loss_shuffled`,
`shuffle_penalty` and `perm_loss` — the last of which is identically zero under the shipped
`lambda_perm: 0.0`, so it is a flat series in every production run and carries no information there.

On the steps the control does not run those keys are **omitted, not zero-filled** — deliberately.
The framework logs every metric with `on_epoch=True`, whose epoch value is the mean over the steps
that reported it. Zeros on 3 of every 4 training steps would scale the epoch-aggregated
`train/feat_loss_shuffled` and `train/shuffle_penalty` to a quarter of their real value, inverting
the very ordering the control exists to check and making a healthy model read as a collapsed source
pathway. Omitted, the mean is taken over the perm steps alone and is right.

**Fused single backward.** The source path contains no batch-coupled operator (only causal
convolutions, an LSTM and LayerNorm), so
$\mathrm{Encoder}(\mathrm{Adapter}(\pi(U)))_i = H^u[\pi(i)]$. `perm_kl_from_forward` therefore
permutes the already-computed `source_state` along the batch axis — exactly equivalent to re-encoding
$\pi(U)$, verified to $\approx 1.6\times10^{-6}$ — and only the attention and posterior are re-run.
This keeps the whole control inside the *single* main forward and backward, which is what lets the
model keep automatic optimisation; under plain `'ddp'` a second backward that does not touch every
parameter would deadlock or force `find_unused_parameters`. Gradient clipping, accumulation, LR
scheduling and the spike breaker all ride on that. The prediction-space control
(`perm_forward_outputs`) runs under `no_grad`.

**`detach_prior`.** The control detaches the prior before it enters the posterior's residual base or
the KL, so the control's gradient flows only through source, attention and posterior — otherwise the
objective could be satisfied by dragging the *prior* toward $q$ instead of by collapsing the
source-driven deltas, which would destroy the quantity being measured. **That failure mode is real
only on the variance path.** On the mean path the detach is a mathematical no-op: $\mu^q = \mu^p +
\Delta\mu$ identically, so the KL's $(\mu^q - \mu^p)^2$ term is $(\Delta\mu)^2$ and carries no
$\mu^p$ dependence to detach. The variance path does not cancel — $\log\sigma^{2,q}$ is a bound
applied to $\widetilde{\log\sigma^{2,p}} + \Delta\ell$, and $\sigma^{2,p}$ appears alone in the
denominator — which is where the detach does work. A test of `detach_prior` written against the mean
head passes vacuously. The derangement uses
Sattolo's algorithm (fixed-point-free by construction, not by rejection), and the per-step decision
is `MIN`-reduced across ranks so no rank runs the control alone (a rank whose local batch is too
small to derange would otherwise build a divergent autograd graph and deadlock the all-reduce).

## 21. Production configuration summary (`configs/default.yaml`)

Backbone: `d_model: 128`, `d_z: 24`, `horizon: 30`, `warmup_period: 30`, `sequence_length: 300`,
`c_y: 109`, `c_u: 58`, `use_up_st: true`, `lstm_layers: 2`, `dropout: 0.1`, `decoder_hidden: 128`,
`logvar_clamp: [-5.0, 3.0]`, `mu_scale: 5.0`, `delta_mu_scale: 3.0`, `delta_logvar_scale: 2.0`.

Correctness-critical / research toggles: `causal_norm: true`, `kld_support: anchor`,
`head_structured_latent: true`, `freeze_unused_attn_proj: true`, `lambda_perm: 0.0`,
`perm_every_n_batches: 4`, `use_entmax: true`, `lag_bias_init: alibi_decay`,
`lag_smoothness_lambda: 1.0e-3`.

Attention: `max_lag: 90`, `num_heads: 4`, `d_head: 32`. Horizon core: `horizon_refine:
{depth: 3, kernel: 3, film: true}`. Encoders: `encoder.extra_dilations: [8, 16]`.

Loss (task/`compute_loss` arguments, not constructor kwargs): `likelihood: gaussian_nll`,
`sigma_obs: learned`, `detach_baseline_in_full: true`, `lambda_full: 1.0`, `lambda_base: 0.5`,
`free_bits: 0.1`, `beta_schedule: {kind: linear_warmup, start: 1.0e-4, end: 0.1, warmup_epochs: 50}`
(`end` is a starting value to retune against `kld_raw`), and `kld_beta: 0.001` — the fallback used
only when `beta_schedule.kind` is `constant`, and therefore inert in the shipped configuration.

## 22. Constructor arguments

Every constructor flag has a sensible default; the production values above are supplied by the
config. The ones that change behaviour materially:

| Argument | Default | Production | Effect |
| --- | --- | --- | --- |
| `use_up_st` | `True` | `True` | source stream includes UP scattering ($c_u=58$) vs phase-harmonic only ($c_u=15$) |
| `causal_norm` | `False` | `True` | swap encoder `GroupNorm`s for `CausalGroupNorm`; **required** for the TE reading |
| `head_structured_latent` | `False` | `True` | per-head latent groups $\Rightarrow$ additive per-head KL and a rigorous `te_lag_map` |
| `freeze_unused_attn_proj` | `False` | `True` | freeze $W_o$ so the run can use plain `'ddp'`. Conditioned on `head_structured_latent`: with a flat latent $W_o$ is live and the flag is a no-op |
| `kld_support` | `'full'` | `'anchor'` | training-KL time support: all post-warmup steps vs supervised anchors only |
| `use_entmax` | `False` | `True` | exact-zero lag weights vs merely small ones |
| `lag_bias_init` | `'normal'` | `'alibi_decay'` | flat vs seeded short-lag ALiBi score bias |
| `alibi_slope_scale` | `1.0` | `1.0` (unset) | multiplier on the ALiBi slopes; $<1$ softens the long-lag penalty, $0$ is flat but learnable. Ignored under `'normal'` |
| `horizon_film` | `False` | `True` | FiLM-condition each horizon step (identity at init) |
| `horizon_depth` / `horizon_kernel` | `2` / `3` | `3` / `3` | dilated blocks in the shared horizon core |
| `encoder_extra_dilations` | `()` | `(8, 16)` | extra kernel-15 conv blocks per encoder for a longer receptive field |
| `mu_scale` / `delta_mu_scale` / `delta_logvar_scale` | `5` / `3` / `2` | same | latent bound saturation magnitudes |
| `logvar_clamp` | `(-5, 3)` | same | effective log-variance range for every smooth bound |
| `lambda_perm` / `perm_every_n_batches` | `0` / `4` | same | permutation-control weight (read by the task) and schedule |
| `attention_grad_checkpoint` | `False` | `False` | recompute attention in backward to save memory |
| `init_weights` | `True` | `True` | apply standard init before the delta heads are zeroed. **Not settable from config** — it is in the driver's `_NON_CONSTRUCTOR_KEYS`, because weight initialisation is not a config decision |

`likelihood`, `sigma_obs`, `beta`, `lambda_full`, `lambda_base`, `free_bits`,
`detach_baseline_in_full` and `lambda_lag` are **`compute_loss` arguments, not constructor kwargs** —
the task reads them from the loss config and passes them through per step.

**Two loss arguments do not appear in the YAML under their own names.** `lambda_lag` is read from
`model_config.VAE_model.lag_smoothness_lambda` (`trainer.py::create_model`); there is no
`lambda_lag:` key in the YAML and adding one would be silently ignored. `beta` has no config key at
all — it is derived per epoch by the task from `beta_schedule` (falling back to `kld_beta`). The
remaining six loss arguments keep their names. The constructor kwargs are otherwise resolved by a single `inspect.signature`
sweep, so a flat `VAE_model` key is forwarded exactly when it names a real constructor argument,
is not in `_NON_CONSTRUCTOR_KEYS` (`horizon_refine`, `encoder`, `init_weights`), and is not `None`
— a misspelled key is dropped without complaint and the constructor default silently applies.

## 23. What the model does not use / out of scope

Circular, reversed-time, or spectrum-matched source surrogates; leave-one-lag-band-out
interventional TE beyond the `lag_band_mask` ablation hook; and any alternative-architecture stage
(SSM / multi-scale encoders, flow posteriors, Student-$t$ likelihood) are out of scope. The entmax
*normaliser* is optional (softmax fallback) and ships **on**; the `entmax` *dependency* is not
optional — `attention.py` imports it unconditionally at module level, so `use_entmax=False` still
requires the package installed. The latent-statistics EMA normalisation,
the dead lag-memory-bank builder, and the stub raw-future decoder that existed in the source tree
were all removed in the flatten (see the deviation record, `DESIGN.md` §8).

## 24. Interpretation rules

- `kld_raw` / `kld_per_t` is the reported TE surrogate; `kld_train` is optimisation-only.
- `kld_raw` measures source *influence*, not source *correctness* — it does **not** separate under a
  deranged source. Use the prediction-space control (`feat_loss < base_loss < feat_loss_shuffled`)
  for correctness.
- The TE reading is only valid with `causal_norm=True`. With it off, the prior conditions on the
  future and `kld_raw` is not a transfer-entropy surrogate.
- The lag decomposition `te_lag_map` is a rigorous attribution only with `head_structured_latent=True`;
  otherwise it is a diagnostic.
- Read $\alpha$ and `te_lag_map` from an `eval()`-mode pass. Attention dropout is applied to the
  normalised weights, so in `train()` the rows do not sum to $1$ and
  $\sum_\ell \widetilde{TE}_{t,\ell} \neq K_t$ (§9.3). `measure_transfer_entropy` handles this;
  a callback reading `forward` outputs mid-training does not.
- Watch `kld_active_frac` (dimensions actually used), `pred_gap` and `shuffle_penalty` (source is
  used and specific in prediction space), and the saturation fractions (bounds not binding).

## 25. Tests

The suite lives under `teb_vae/lag_attn/tests/`. Structural: `test_construct.py` (geometry
guards, and that the caller's channel widths are honoured rather than derived),
`test_nets_are_framework_free.py`. Behavioural: `test_zero_kl_init.py`
(exact $K \equiv 0$ at init, flat and head-structured), `test_smooth_bound.py`,
`test_residual_head.py`, `test_logvar_floor.py`, `test_causal_encoder.py` (zero future leak),
`test_alibi.py`, `test_lag_band_mask.py`, `test_kl_support.py`, `test_kl_report.py`,
`test_derangement.py`, `test_perm_kl.py`, `test_perm_schedule.py`, `test_spike_breaker.py`,
`test_ddp_strategy.py`, `test_checkpoint_contract.py`, `test_data_contract.py`, `test_task.py`,
`test_trainer.py`, `test_main.py`, `test_config_load.py`, `test_config_merge.py`,
`test_train_smoke.py`, `test_fixtures.py`, and the plotting tests
(`test_plotting_callback.py`, `test_plotting_figure.py`).

`conftest.py` ships three constructor-kwarg sets, and the distinction matters when adding a test:
`TINY_KWARGS` is the bare geometry (§3), `PROD_KWARGS` adds the six keys the predecessor's suite
called "prod" (`causal_norm`, `kld_support`, `lag_bias_init`, `lambda_perm`, `perm_every_n_batches`,
`freeze_unused_attn_proj`), and `SHIPPED_KWARGS` is what `default.yaml` actually sets — `use_entmax`,
`head_structured_latent`, `horizon_depth: 3`, `horizon_film` and `encoder_extra_dilations`. Only
the last builds the per-head posterior, the FiLM generator, the third refine block and the two
extra conv blocks per encoder, so a test written against `PROD_KWARGS` leaves roughly a third of
the production model's parameters unexercised and `freeze_unused_attn_proj` inert. The
`perturb_posterior` fixture is equally load-bearing: the delta heads are zero-initialised, so every
KL assertion on an untouched model passes vacuously — including on a model that is entirely wrong.

## 26. One-sentence definition

`SeqVaeLagAttn` is a source-pure, strictly-causal, residual sequential VAE that forecasts a fetal
FHR feature stream from its own past plus a lag-attended uterine-pressure source, and reports the
per-step KL between its source-conditioned posterior and its target-only prior — earned from an exact
zero at initialisation, anchor-aligned, honestly split into optimised and raw readouts, and
attributed across lags — as a defensible transfer-entropy surrogate for source$\to$target
information flow.
