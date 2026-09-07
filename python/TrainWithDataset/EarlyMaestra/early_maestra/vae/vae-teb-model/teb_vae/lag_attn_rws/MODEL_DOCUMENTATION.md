# A Lag-Attentive Sequential Variational Autoencoder for Directed Coupling between Uterine Pressure and Raw Fetal Heart Rate

**Complete model documentation.**

This document describes one model end to end: what it consumes, what it computes, every layer it
is built from, the probability model it implements, the objective it optimises, how it is
trained, what it reports, and what those reports may and may not be read as. It is written to be
self-contained — every symbol used is defined here, and no external construction is assumed.

---

## Table of contents

0. [Notation](#0-notation)
1. [What the model is](#1-what-the-model-is)
2. [Signals, sampling and the anchor grid](#2-signals-sampling-and-the-anchor-grid)
3. [Input representation](#3-input-representation)
4. [The probabilistic model](#4-the-probabilistic-model)
5. [Architecture, layer by layer](#5-architecture-layer-by-layer)
6. [Parameter budget](#6-parameter-budget)
7. [Structural invariants](#7-structural-invariants)
8. [Initialisation](#8-initialisation)
9. [Masks, supports and validity](#9-masks-supports-and-validity)
10. [The objective](#10-the-objective)
11. [Readouts and diagnostics](#11-readouts-and-diagnostics)
12. [Training procedure](#12-training-procedure)
13. [Evaluation protocol](#13-evaluation-protocol)
14. [Causal-reach control of the inputs](#14-causal-reach-control-of-the-inputs)
15. [Assumptions and limits](#15-assumptions-and-limits)
16. [Configuration reference](#16-configuration-reference)
17. [File map and commands](#17-file-map-and-commands)

---

## 0. Notation

Every symbol used anywhere in this document, with the shipped value where it is a constant.

### Signals and indices

| Symbol | Meaning | Value |
| --- | --- | --- |
| $Y[n]$ | raw fetal heart rate (the **target**), z-scored | — |
| $U[n]$ | raw uterine pressure (the **source**) | — |
| $f_s$ | raw sampling rate | $4$ Hz |
| $L_{\mathrm{raw}}$ | raw samples per segment | $4800$ |
| $D = R$ | decimation factor; also raw samples per horizon token | $16$ |
| $\Delta$ | seconds per decimated step, $D/f_s$ | $4$ s |
| $T$ | decimated steps per segment, $L_{\mathrm{raw}}/D$ | $300$ |
| $t$ | **anchor** index on the decimated grid | $0 \dots T-1$ |
| $n_{\mathrm{raw}}(t)$ | anchor $t$'s raw causal endpoint | $16(t+1)-1$ |
| $H$ | forecast horizon in decimated steps | $30$ |
| $\tau$ | forecast-step index within the horizon | $0 \dots H-1$ |
| $r$ | raw-sample index within a step | $0 \dots R-1$ |
| $T_{\mathrm{valid}}$ | anchors with a fully observed future, $T-H$ | $270$ |
| $w$ | warm-up steps excluded from every loss | $30$ |
| $\mathcal{A}$ | trained anchor set, $[w,\ T-H)$ | $[30, 270)$ |
| $X^{+}_{t,\tau,r}$ | the raw future target of anchor $t$ | $(B,270,30,16)$ |
| $Y^{-}_t,\ U^{-}_t$ | target / source history up to $n_{\mathrm{raw}}(t)$ | — |
| $v_t$ | decimated validity weight | $\{0,1\}$ |
| $B$ | batch size | $128$ |

### Features

| Symbol | Meaning | Value |
| --- | --- | --- |
| $\phi,\ \psi_\lambda$ | low-pass filter, analytic wavelet at scale $\lambda$ | — |
| $\xi_\lambda$ | centre frequency of $\psi_\lambda$ | — |
| $p = 2^{k/Q}$ | phase-harmonic ratio, $k \in \{4,6,8\}$ | — |
| $X^Y_t,\ X^U_t$ | assembled target / source feature vectors | $\mathbb{R}^{109},\ \mathbb{R}^{58}$ |
| $c_y,\ c_u$ | their channel counts | $109,\ 58$ |

### Model

| Symbol | Meaning | Value |
| --- | --- | --- |
| $d_{\mathrm{model}}$ | backbone width | $128$ |
| $d_z$ | latent width | $64$ |
| $M$ | attention heads = latent groups | $4$ |
| $d_{\mathrm{head}}$ | per-head width | $32$ |
| $g = d_z/M$ | latent coordinates per group | $16$ |
| $L$ | attention lags, $\mathrm{max\_lag}+1$ | $91$ |
| $\ell$ | lag index | $0 \dots L-1$ |
| $H^y_t,\ H^u_t$ | target / source history states | $\mathbb{R}^{128}$ |
| $\mu^p_t,\ \ell^p_t$ | prior mean and log-variance | $\mathbb{R}^{64}$ |
| $\tilde\ell^p_t$ | prior **pre-bound** raw log-variance | $\mathbb{R}^{64}$ |
| $\mu^q_t,\ \ell^q_t$ | posterior mean and log-variance | $\mathbb{R}^{64}$ |
| $z^p_t,\ z^q_t$ | the two latent samples | $\mathbb{R}^{64}$ |
| $\epsilon_t$ | the single shared noise draw | $\mathcal{N}(0,I)$ |
| $a_{t,m,\ell}$ | attention score before normalisation | — |
| $\rho_{m,\ell}$ | learned per-lag key bias | $\mathbb{R}^{32}$ |
| $b_{m,\ell}$ | learned per-(head, lag) scalar score bias | — |
| $\omega_m$ | initial lag-decay slope of head $m$ | $(0.25 \dots 0.0039)$ |
| $\mathcal{M}^{\mathrm{lag}}_{t,\ell}$ | lag-validity mask, $\mathbb{1}[t-\ell \ge 0]$ | $\{0,1\}$ |
| $\nu_{t,m}$ | entmax normalising threshold | — |
| $\alpha_{t,m,\ell}$ | attention weight, head $m$, lag $\ell$ | $\sum_\ell \alpha = 1$ |
| $A^{(m)}_t$ | attended source summary of head $m$ | $\mathbb{R}^{32}$ |
| $h_t$ | latent projected to decoder width | $\mathbb{R}^{256}$ |
| $e_\tau$ | learned horizon-step embedding | $\mathbb{R}^{256}$ |
| $\gamma_b, \beta_b$ | FiLM modulation of refine block $b$ | $\mathbb{R}^{256}$ |
| $\hat\mu_{t,\tau,r},\ \hat\ell_{t,\tau,r}$ | decoder forecast mean and log-variance | — |
| $\theta,\phi,\psi$ | prior / posterior / decoder parameters | — |

### Bounds and objective

| Symbol | Meaning | Value |
| --- | --- | --- |
| $\mathrm{sb}(\cdot)$ | smooth bound, $\ell_{\min} + (\ell_{\max}-\ell_{\min})\sigma(\cdot)$ | — |
| $(\ell_{\min}, \ell_{\max})$ | log-variance clamp | $(-5, 3)$ |
| $a_\mu$ | prior-mean tanh saturation magnitude | $5$ |
| $s_\mu,\ s_\ell$ | posterior mean / log-variance delta magnitudes | $3,\ 2$ |
| $g_A$ | posterior-fusion attended-source LayerNorm gain | $2$ |
| $\varsigma$ | tanh-bound saturation fraction | $0.99$ |
| $\varrho_{\mathrm{margin}}$ | log-variance "on the clamp" margin fraction | $0.05$ |
| $m_{t,\tau}$ | forecast mask | $\{0,1\}$ |
| $\bar c_t$ | anchor forecast-window coverage | $[0,1]$ |
| $c_{\min}$ | coverage floor | $0.9$ |
| $c_t$ | contributing-anchor indicator | $\{0,1\}$ |
| $m^{\mathrm{KL}}_t$ | KL anchor support | $\{0,1\}$ |
| $D_0,\ D_1$ | base / full block NLL, nats per anchor | — |
| $\Delta_t$ | predictive gain, $D_0 - D_1$ | — |
| $\kappa_{t,d}$ | per-dimension KL | — |
| $K_t = \sum_d \kappa_{t,d}$ | per-anchor source-conditioned rate | — |
| $K^{(m)}_t$ | its group-$m$ part | — |
| $\widetilde K_{t,\ell}$ | its lag-resolved attribution | — |
| $\eta$ | free-bits floor | $0$ |
| $\epsilon_{\mathrm{act}}$ | per-dimension activity threshold | $10^{-2}$ |
| $\beta(e)$ | KL weight at epoch $e$ | $0 \to 1$ |
| $R_p$ | prior scale rate, nats per anchor | — |
| $\beta_p$ | prior-anchor weight — a constant, not a schedule | $0.1$ |
| $\lambda_{\mathrm{full}},\ \lambda_{\mathrm{base}}$ | reconstruction weights | $1,\ 1$ |
| $\mathcal{L}_{\mathrm{ms}},\ \mathcal{L}_{\mathrm{deriv}},\ \mathcal{L}_{\mathrm{boundary}}$ | auxiliary shape terms on the forecast mean | — |
| $\lambda_{\mathrm{ms}},\ \lambda_{\mathrm{deriv}},\ \lambda_{\mathrm{boundary}}$ | their weights, all provisional | $0.1,\ 0.1,\ 0.05$ |
| $\delta_c$ | causal input delay of channel $c$, in steps | $0$ by default |
| $S$ | Monte Carlo latent draws at evaluation | $8$ |

Three conventions hold throughout.

* **The prior is written $p_\theta$ and the posterior $q_\phi$**, and the KL is always taken in
  the order $D_{\mathrm{KL}}(q \Vert p)$.
* **"Base" always means the prior-latent branch and "full" the posterior-latent branch**, in every
  equation, metric name and figure.
* Two symbols are reused in their standard senses and are distinguished by their arguments:
  $\beta(e)$ is always the KL weight at epoch $e$, whereas $(\gamma_b, \beta_b)$ with a block
  subscript is always a FiLM modulation pair and $(\gamma_c, \beta_c)$ with a channel subscript is
  always a normaliser's affine parameters. Likewise $\ell$ without a superscript is the attention
  lag index, while $\ell^p$, $\ell^q$ and $\hat\ell$ are log-variances.

### The residual-MLP primitive

One block recurs at every seam in the model, so it is defined once here. For an input
$x \in \mathbb{R}^{c_{\mathrm{in}}}$ applied per timestep, and widths
$(h_1, \dots, h_n)$:

$$
\bar x = \mathrm{LayerNorm}(x), \qquad
y_0 = \bar x,
$$

$$
y_i = \begin{cases}
\mathrm{Dropout}\big(\mathrm{GELU}\big(\mathrm{LayerNorm}(W_i y_{i-1} + b_i)\big)\big) & i < n,\\[4pt]
W_n y_{n-1} + b_n \ \ (\text{plus } \mathrm{LayerNorm} \text{ iff the seam is activated}) & i = n,
\end{cases}
$$

$$
\mathrm{ResMLP}(x) = y_n + P\,\bar x,
$$

where $P$ is the identity when $c_{\mathrm{in}} = h_n$ and a learned projection otherwise. The
skip is taken from the **normalised** input, so both branches of the sum are on one scale. It acts
on the channel axis independently at each timestep, so it mixes features without ever mixing time
— which is what allows it to sit inside a causal stack with no mask.

Two variants appear, and the difference is *two* operations rather than one:

| Variant | Last body layer | After the sum |
| --- | --- | --- |
| **activated** | $W_n y_{n-1} + b_n$, then $\mathrm{LayerNorm}$ | $\mathrm{GELU}$ |
| **plain** | $W_n y_{n-1} + b_n$, no normaliser | nothing |

So a plain seam drops both the body's closing $\mathrm{LayerNorm}$ and the post-sum
$\mathrm{GELU}$, which is $-2 d$ parameters per seam ($-256$ at $d = 128$). Which variant is used
where, and why, is stated at each site.

### The geometric width schedule

Hidden widths are never hand-picked. Between an input width $s_{\mathrm{in}}$ and an output width
$s_{\mathrm{out}}$ with $n$ hidden layers, the widths follow

$$
s_i = s_{\mathrm{in}}\, \varrho^{\,i}, \qquad \varrho = \left(\frac{s_{\mathrm{out}}}{s_{\mathrm{in}}}\right)^{1/(n+1)},
\qquad i = 1, \dots, n+1,
$$

rounded to integers. Each layer therefore changes width by the same *ratio* rather than the same
amount, which keeps the per-layer compression even instead of dropping most of it in one step.
Every width tuple quoted later in this document — $(111, 97, 84, 74, 64)$, $(215, 181, 152, 128)$,
$(94, 55, 32)$, $(91, 128, 181, 256)$ — is this formula evaluated at the stated endpoints.

---

## 1. What the model is

Two physiological signals are recorded simultaneously during labour: the fetal heart rate,
written $Y$, and the uterine pressure, written $U$. The scientific question is directional and
has a delay in it:

> How much does the recent past of the uterine pressure tell us about the near future of the
> fetal heart rate that the fetal heart rate's own past did not already say — and at what
> delay?

The model answers this by forecasting the same quantity twice. At every anchor time $t$ it
produces two probability distributions over the **next two minutes of the raw fetal heart rate
signal**, $480$ samples at $4$ Hz:

* a **base** forecast, conditioned on a latent drawn from a distribution that has seen only the
  target's own history;
* a **full** forecast, conditioned on a latent drawn from a distribution that has additionally
  seen the source's lagged history.

Both forecasts come out of **one shared decoder**, invoked twice, receiving nothing but the
latent vector. The two readouts of interest are therefore

$$
\underbrace{\Delta_t = D_0(t) - D_1(t)}_{\text{predictive gain, in nats per anchor}},
\qquad
\underbrace{K_t = D_{\mathrm{KL}}\!\left(q_\phi(z_t \mid Y_{\le t}, U_{\le t}) \,\Vert\, p_\theta(z_t \mid Y_{\le t})\right)}_{\text{source-conditioned rate, in nats per anchor}}
$$

where $D_0$ and $D_1$ are the negative log-likelihoods of the true raw future under the base and
full forecasts. A lag-resolved attribution of $K_t$ across candidate delays is produced
alongside.

The design decision that carries the whole model is the **absence of a decoder bypass**. The
decoder's forward accepts exactly one tensor, $z$. There is no separate target-conditioning
path around the latent. Consequently the latent cannot be a small side-channel correction: it
must *be* the predictive state of the fetal heart rate, and the shift the source induces in it
must be the additional predictive information the source carries.

Three properties hold by construction rather than by convention, and each is enforced in the
module and pinned by a test:

| Property | Statement | Mechanism |
| --- | --- | --- |
| **No decoder bypass** | Gradient reaches the decoder only through $z$ | the decoder's `forward` takes one tensor; no second decoder and no conditioning head exist |
| **Source purity** | The source pathway never sees a target tensor; the prior never sees the source | separate adapters and encoders; the posterior is a residual on the prior |
| **Exact zero at initialisation** | $K_t \equiv 0$ and the two forecasts are bitwise identical | posterior residual heads zeroed *after* generic initialisation; one shared noise draw; zero dropout in the decoder and in the attention probabilities |

The third property is what makes every reported nat of coupling *earned*: the model starts from
the assertion "the source says nothing", and any departure from it had to be produced by
training against data.

At the shipped configuration the model holds **5,088,186 parameters**, of which $16{,}512$ are
frozen. The shipped `causal_reach_budget_s: 120` adds $6{,}272$ more for the two availability
adapters, so the model as launched holds $5{,}094{,}458$.

---

## 2. Signals, sampling and the anchor grid

### 2.1 Raw signals

Both signals are sampled at

$$
f_s = 4\ \mathrm{Hz}, \qquad \Delta_{\mathrm{raw}} = 0.25\ \mathrm{s}.
$$

A training example is one segment of one recording. After a symmetric one-minute trim applied by
the loader, a segment holds

$$
L_{\mathrm{raw}} = 4800 \text{ raw samples} = 1200\ \mathrm{s} = 20\ \text{minutes}
$$

of fetal heart rate $Y[n]$, $n = 0, \dots, 4799$, and the same span of uterine pressure $U[n]$.
The raw fetal heart rate is **z-scored** by training-set statistics before it reaches the model;
it is the reconstruction target, so this normalisation is load-bearing rather than cosmetic and
is checked at the entry point.

### 2.2 Decimated grid and anchors

Features and latents live on a decimated grid with decimation factor

$$
D = R = 16, \qquad \Delta = \frac{D}{f_s} = 4\ \mathrm{s},
$$

giving

$$
T = \frac{L_{\mathrm{raw}}}{D} = 300 \text{ decimated steps per segment}.
$$

Decimated step $t$ covers raw samples $[Dt,\ D(t+1))$. Its **causal endpoint** — the last raw
sample an anchor at $t$ may depend on — and the first raw sample of its forecast block are

$$
\boxed{\;n_{\mathrm{raw}}(t) = D\,(t+1) - 1 = 16(t+1) - 1, \qquad
\mathrm{start}(t) = n_{\mathrm{raw}}(t) + 1 = 16(t+1).\;}
$$

This map is held in a validated, frozen geometry object whose constructor asserts its own index
identities, so no unvalidated instance can exist. In particular $\mathrm{start}(0) = 16$, and the
last trained anchor's forecast window ends exactly at raw index $L_{\mathrm{raw}} - 1$.

### 2.3 Horizon and valid anchors

The forecast horizon is

$$
H = 30 \text{ decimated steps}, \qquad H \cdot R = 480 \text{ raw samples} = 120\ \mathrm{s}.
$$

An anchor has a fully observed forecast window only if its window fits inside the segment, so

$$
T_{\mathrm{valid}} = T - H = 270,
$$

and the model decodes anchors $t \in [0, T_{\mathrm{valid}})$ only. A **warm-up** prefix of

$$
w = 30 \text{ steps} = 2 \text{ minutes}
$$

is excluded from every loss term, because the encoders' history states are not meaningful before
they have accumulated history. The trained anchor set is therefore

$$
\mathcal{A} = [w,\ T - H) = [30,\ 270),
$$

$240$ anchors per segment, each carrying a $480$-sample forecast target.

### 2.4 The future target tensor

For anchor $t$, forecast step $\tau \in [0, H)$ and intra-step sample $r \in [0, R)$, the raw
future target is

$$
X^{+}_{t,\tau,r} = Y\big[\,\mathrm{start}(t) + D\tau + r\,\big] = Y\big[\,D(t+1) + D\tau + r\,\big],
$$

a tensor of shape $(B, T_{\mathrm{valid}}, H, R) = (B, 270, 30, 16)$. The integer index grid is
built once from the geometry and cached as a non-persistent buffer, and the target is produced by
a single gather on the training hot path.

A consequence used throughout: future raw sample $(t, \tau, r)$ lies in decimated step
$t + 1 + \tau$ for **every** $r$. Validity is therefore constant within a decimated step, and all
masks live on the decimated $(B, T_{\mathrm{valid}}, H)$ grid and broadcast over $r$.

---

## 3. Input representation

The encoders do not read the raw signals. They read time–frequency features computed from them
on the decimated grid, in four blocks.

### 3.1 Scattering block

Let $\phi$ be a low-pass filter and $\{\psi_\lambda\}$ a bank of analytic Morlet wavelets with
centre frequencies $\xi_\lambda$, built with $J = 11$ octaves, $Q = 4$ wavelets per octave, and a
low-pass averaging scale of $T_\phi = 16$ samples ($4$ s). The stored scattering block of a
signal $x$ is the zeroth-order average followed by the first-order envelopes:

$$
S_0(t) = (x \star \phi)(t), \qquad
S_\lambda(t) = \big(\,|x \star \psi_\lambda|\, \star \phi\,\big)(t),
$$

decimated to the $4$ s grid. Forty-two wavelets survive the pipeline's frequency selection, so
each scattering block has

$$
1 + 42 = 43 \text{ channels}.
$$

Both signals carry one: `fhr_st` $\in \mathbb{R}^{300 \times 43}$ and
`up_st` $\in \mathbb{R}^{300 \times 43}$.

### 3.2 Phase-harmonic block

The modulus in the scattering transform destroys phase relations between scales. The
phase-harmonic block restores them. Writing $z_i = x \star \psi_i$ for the analytic response at
scale $i$, and for a pair $(i, j)$ with $\xi_i \le \xi_j$,

$$
C_{i,j,p}(t) = \phi \star \Big(\,|z_i|\,e^{\mathrm{i}\,p\,\arg z_i}\;\overline{z_j}\,\Big)(t),
\qquad p = \frac{\xi_j}{\xi_i} = 2^{k/Q}.
$$

The operator $z \mapsto |z|\,e^{\mathrm{i} p \arg z}$ multiplies the instantaneous phase by $p$;
the product with $\overline{z_j}$ is large and stable only when the phase at scale $j$ advances
$p$ times as fast as at scale $i$ — that is, when the two scales are **phase-locked at ratio
$p$**. The harmonic steps retained are $k \in \{4, 6, 8\}$, i.e. $p \in \{2,\ 2\sqrt{2},\ 4\}$:
one octave (waveform asymmetry — a non-sinusoidal wave carries harmonic content at $2f$ locked to
$f$), one and a half octaves, and two octaves (coupling between well-separated rhythms). The
diagonal $k = 0$ ($p = 1$) is dropped, being redundant with the scattering block already stored
alongside.

The two streams retain different frequency bands, matched to their physiology:

| Block | Band | Channels |
| --- | --- | --- |
| `fhr_ph` | $0.008$ – $1.00$ Hz | $66$ |
| `up_ph` | $0.008$ – $0.05$ Hz | $15$ |

The uterine-pressure band is capped at $0.05$ Hz because contractions carry no energy above it.

### 3.3 Assembled streams

The target stream and the source stream are the concatenations

$$
X^Y_t = \big[\,\texttt{fhr\_st}_t \;\Vert\; \texttt{fhr\_ph}_t\,\big] \in \mathbb{R}^{c_y},
\qquad c_y = 43 + 66 = 109,
$$

$$
X^U_t = \big[\,\texttt{up\_st}_t \;\Vert\; \texttt{up\_ph}_t\,\big] \in \mathbb{R}^{c_u},
\qquad c_u = 43 + 15 = 58 .
$$

An ablation switch reduces the source stream to its phase block alone ($c_u = 15$).

**No cross-signal feature block is ever loaded.** A coefficient formed from $Y$ and $U$ jointly
would mix both signals into one number and would destroy the separation between "target-only" and
"source-conditioned" that the entire measurement rests on. Its absence is asserted by test across
every configuration file.

### 3.4 Validity signal

Gaps in the recording are stored as $0$ bpm, which after z-scoring is approximately $-11\sigma$ —
a value that is *not* a detectable sentinel and would silently dominate a summed $480$-sample
log-likelihood. The authoritative validity signal is therefore a separate decimated weight
vector $v \in \{0,1\}^{300}$ accompanying each segment. A decimated step counts as valid only
when it is *fully* valid:

$$
\mathrm{valid}(t) = \mathbb{1}\big[\,v_t \ge 1\,\big].
$$

A partially valid step still contains raw samples near $-11\sigma$, so partial steps are excluded
rather than admitted.

### 3.5 Batch contract

| Field | Shape per sample | Role |
| --- | --- | --- |
| `fhr` | $(4800,)$ | the reconstruction target, z-scored |
| `fhr_st` | $(300, 43)$ | target scattering |
| `fhr_ph` | $(300, 66)$ | target phase-harmonic |
| `up_st` | $(300, 43)$ | source scattering |
| `up_ph` | $(300, 15)$ | source phase-harmonic |
| `weight` | $(300,)$ | validity |
| `guid` | — | recording identifier, for grouping and diagnostics |

The declared widths $c_y$ and $c_u$ are dataset facts, not model constants: they are checked
against the first shard before the fit and against every batch at the data boundary, and a
mismatch raises with the config key that fixes it.

---

## 4. The probabilistic model

### 4.1 Two distributions over one latent

At each anchor $t$ the model defines two diagonal-Gaussian distributions over the **same** latent
vector $z_t \in \mathbb{R}^{d_z}$, $d_z = 64$:

$$
p_\theta(z_t \mid Y_{\le t}) = \mathcal{N}\!\left(\mu^p_t,\ \operatorname{diag} e^{\ell^p_t}\right),
\qquad
q_\phi(z_t \mid Y_{\le t}, U_{\le t}) = \mathcal{N}\!\left(\mu^q_t,\ \operatorname{diag} e^{\ell^q_t}\right).
$$

The first is the **prior**: it is a function of the target history alone. The second is the
**posterior**: it additionally reads a lag-attended summary of the source history. Both live in
the same coordinate system, because the posterior is parameterised as a bounded residual on the
prior.

### 4.2 Why the prior is learned and conditional

This is the point at which the model departs from an ordinary variational autoencoder, and the
departure is what makes the KL readable.

In a standard autoencoder the prior is a **fixed** $\mathcal{N}(0, I)$, and
$D_{\mathrm{KL}}(q \Vert \mathcal{N}(0,I))$ measures the total information the code carries about
its input — target information and source information indiscriminately. That is the wrong
quantity here. The model is asked what the source added *given the target's own past*, so a fixed
prior would charge the code for everything the target history already explains, and the number
would be dominated by exactly the term the question conditions away.

Making the prior a **learned function of the target history** removes that term at the source. The
two distributions differ in one respect and one only — whether the source was read — so their
divergence

$$
K_t = D_{\mathrm{KL}}\!\left(q_\phi(z_t \mid Y_{\le t}, U_{\le t})\ \Vert\ p_\theta(z_t \mid Y_{\le t})\right)
$$

prices the source's contribution and nothing else. Target information common to both cancels
because it is present in both arguments.

Three consequences follow, and each is load-bearing:

* **The prior is not a regulariser; it is a forecaster.** $p_\theta$ must be good enough that
  $D_0$ is a strong forecast in its own right, which is why the objective trains it explicitly
  (§10.7). A weak prior would make any posterior look informative.
* **The posterior must be a residual on the prior, in the prior's own coordinates.** Two
  independently parameterised heads could place the "same" belief at different points of latent
  space, and their divergence would then measure parameterisation mismatch rather than
  information. §5.6 builds $q_\phi$ as $p_\theta$ plus a bounded delta for exactly this reason.
* **There is no free lunch at initialisation.** With the delta at zero the two distributions
  coincide and $K_t = 0$ exactly, so the model begins by asserting that the source is
  uninformative and must be driven off that assertion by data.

### 4.3 One decoder, two invocations

A single decoder $D_\psi$ maps a latent to a factorised Gaussian over the anchor's raw future:

$$
(\hat\mu^k_t,\ \hat\ell^k_t) = D_\psi(z^k_t), \qquad
d_\psi(X^+_t \mid z) = \prod_{\tau=0}^{H-1}\prod_{r=0}^{R-1}
\mathcal{N}\!\left(X^{+}_{t,\tau,r};\ \hat\mu_{t,\tau,r},\ e^{\hat\ell_{t,\tau,r}}\right).
$$

The base forecast is $D_\psi(z^p_t)$ and the full forecast is $D_\psi(z^q_t)$ — the *same*
parameters $\psi$, the same call, differing only in which distribution the latent was drawn from.
There is no residual decoder and no equation of the form
$\hat\mu_{\mathrm{full}} = \hat\mu_{\mathrm{base}} + \Delta\hat\mu$ anywhere in the architecture:
the source's effect on the forecast is a *consequence* of moving the latent, not a separate
output pathway.

### 4.4 Common random numbers

One standard-normal tensor $\epsilon_t \sim \mathcal{N}(0, I)$ is drawn per anchor per forward
and used for **both** samples:

$$
z^p_t = \mu^p_t + e^{\ell^p_t/2}\odot\epsilon_t,
\qquad
z^q_t = \mu^q_t + e^{\ell^q_t/2}\odot\epsilon_t.
$$

This coupling has a precise consequence: if $p_t = q_t$ then $z^p_t = z^q_t$ *sample by sample*,
not merely in distribution, so the base-minus-full readout carries no independent sampling noise
and is exactly zero when the two distributions coincide.

### 4.5 The two coupling quantities

**Predictive gain.** With $D_0(t)$ and $D_1(t)$ the negative log-likelihoods of the true raw
future under the base and full forecasts,

$$
\boxed{\ \Delta_t \;=\; D_0(t) - D_1(t) \;=\; \log \frac{p_1(X^+_t)}{p_0(X^+_t)}\ }
$$

is a held-out predictive log-score improvement in nats per anchor. Under a correctly specified
model family and an exact posterior its expectation is the conditional mutual information
$I(U^-_t; X^+_t \mid Y^-_t)$; in practice it is a model-based estimate within the chosen family.
It may be negative for individual anchors; the meaningful aggregate is the per-recording mean.

**Source-conditioned rate.** The closed-form Kullback–Leibler divergence between the two
diagonal Gaussians, per step and per latent dimension:

$$
\boxed{\
\kappa_{t,d} = \tfrac{1}{2}\left[\ \ell^p_{t,d} - \ell^q_{t,d}
+ \frac{e^{\ell^q_{t,d}} + \big(\mu^q_{t,d} - \mu^p_{t,d}\big)^2}{e^{\ell^p_{t,d}}} - 1\ \right],
\qquad K_t = \sum_{d=1}^{d_z} \kappa_{t,d}.
\ }
$$

It is computed in closed form rather than sampled, because it is the model's *output* and a Monte
Carlo estimate would put variance directly into the reported number.

$K_t$ measures how much the source moved the model's belief about the target's predictive state.
It is deliberately **not** called a transfer entropy; see [§15](#15-assumptions-and-limits).

### 4.6 Sequence factorisation, and what is amortised

The model is **not** a latent state-space model: there is no transition density
$p(z_t \mid z_{t-1})$ and no filtering recursion over the latent. The latent at each anchor is
directly from that anchor's history, and the objective factorises over anchors:

$$
\mathcal{L} = \frac{1}{|\mathcal{A}|}\sum_{t \in \mathcal{A}} \mathcal{L}_t .
$$

Temporal structure enters in two places instead, both inside the encoders rather than in the
latent chain: the dilated convolution stack, whose receptive field is a fixed window, and the
unidirectional recurrence, which carries state across the whole prefix. So $z_t$ and $z_{t+1}$ are
correlated because $H^y_t$ and $H^y_{t+1}$ are, not because the latents are linked.

This is a deliberate simplification with one clear benefit and one clear cost. The benefit is
that $K_t$ is a *local* quantity: it prices the source at anchor $t$ with no contribution
propagated from a neighbouring anchor's code, which is what allows the per-anchor and per-lag
attributions of §11 to be read at all. The cost is that anchors are scored as though independent,
which they are not — adjacent forecast windows overlap by $29/30$ — and this is why aggregate
uncertainty is estimated by bootstrapping recordings rather than anchors (§13.5).

**Everything in the model is used at inference.** There is no training-only module to discard:
the prior, the attention, the posterior and the decoder are all evaluated at test time. Only the
inputs decide which branch is available — with the source present both branches run and both
readouts are defined; with the source withheld the prior branch alone still produces a complete
forecast, because the prior is a forecaster and not a regulariser.

---

## 5. Architecture, layer by layer

The forward pass, in order. All widths below are the shipped configuration:
$d_{\mathrm{model}} = 128$, $d_z = 64$, $M = 4$ attention heads, $d_{\mathrm{head}} = 32$,
$L = 91$ lags, $H = 30$ horizon tokens, $R = 16$ raw samples per token.

```
  fhr_st (B,300,43) ┐                                  up_st (B,300,43) ┐
  fhr_ph (B,300,66) ┴─ concat ─ (B,300,109)            up_ph (B,300,15) ┴─ concat ─ (B,300,58)
                         │                                                    │
                  [channel gate]                                       [channel gate]
                         │                                                    │
                  InputAdapter                                         InputAdapter
                         │ (B,300,128)                                        │ (B,300,128)
             CausalConvLstmEncoder                             CausalConvLstmEncoder
                         │ H^y (B,300,128)                             H^u (B,300,128)
                 FullLatentPriorHead                                          │
              μ^p, ℓ^p, ℓ̃^p (B,300,64)                                       │
                         │                                                    │
                    query_proj ──────────► LagCrossAttention ◄────────────────┘
                                            α (B,300,4,91)
                                            a (B,300,4,32)
                         │                        │
                         └──── PosteriorHead ◄────┘
                              μ^q, ℓ^q (B,300,64)
                                     │
                       paired reparameterisation (one ε)
                            z^p, z^q (B,300,64)
                                     │
                    ┌────────────────┴────────────────┐
             D_ψ(z^p[:, :270])                 D_ψ(z^q[:, :270])
          μ_base, ℓ_base (B,270,30,16)     μ_full, ℓ_full (B,270,30,16)
```

### 5.1 Channel gate (optional causal input guard)

Applied between the batch boundary and the input adapters. When a causal reach budget is
configured, it selects the surviving channels of a stream and delays each survivor by its own
integer number of steps:

$$
\big(\mathcal{G}x\big)_{t,c} =
\begin{cases}
x_{\,t - \delta_c,\ \pi(c)} & t \ge \delta_c,\\[2pt]
0 & t < \delta_c,
\end{cases}
$$

where $\pi$ is the strictly ascending keep-index and $\delta_c$ the per-channel delay. With no
budget configured, **no gate module is built at all**, so the default model is structurally
identical to one that has no delay mechanism. The rationale and the resolution of $\delta_c$ are
in [§14](#14-causal-reach-control-of-the-inputs).

### 5.2 Input adapter

One per stream, differing only in input width. It lifts a feature stream to the model width:

$$
u_t = \mathrm{Dropout}\Big(\mathrm{GELU}\big(\mathrm{LayerNorm}(W_{\mathrm{in}} x_t + b)\big)\Big),
\qquad W_{\mathrm{in}} \in \mathbb{R}^{128 \times c},
$$

followed by the residual MLP of §0 at constant width, $128 \to (128, 128, 128, 128)$, whose skip
is therefore an identity:

$$
x^{\mathrm{adapt}}_t = \mathrm{ResMLP}_{128 \to 128}(u_t).
$$

**The seam is plain, not activated.** Neither the body's closing $\mathrm{LayerNorm}$ nor the
post-sum $\mathrm{GELU}$ is present. A gate there attenuates the backward gradient flowing through
the seam and leaves a fraction of the exported units persistently compressed on the forward side,
for no measured benefit; removing it un-gates the seam and raises the effective rank of what it
exports. **Six** seams are plain for this reason — the two input adapters, and inside each encoder
the front MLP and the fusion — for $6 \times 256 = 1{,}536$ fewer parameters than the activated
form.

Two further residual MLPs are unactivated for an unrelated and more basic reason: the prior head's
mean and log-variance branches (§5.4) emit distribution *parameters*, which must be free to take
either sign, and a closing GELU would gate them. The remaining five — the four per-group posterior
fusions and the decoder's latent projection — are activated.

Output: $(B, 300, 128)$.

### 5.3 Causal convolutional–recurrent encoder

One per stream. It converts the adapted stream into a per-step **history state** $H_t$ that must
be a function of $t' \le t$ only. Two branches run in parallel and are fused.

**Stage A — front residual MLP.** The §0 primitive at constant width $128$, plain seam,
producing $x^{\mathrm{lin}}$.

**Stage B — dilated causal convolution stack.** Five **pre-norm** residual blocks. Block $b$
computes

$$
\mathrm{Block}_b(x) = x + \mathrm{Dropout}\Big(\mathrm{Conv1d}_{k_b, d_b}\big(\mathrm{GELU}\big(\mathrm{Norm}(x)\big)\big)\Big),
$$

with **left-only padding** of $(k_b - 1)d_b$ samples, so output $t$ reads inputs
$t - (k_b-1)d_b, \dots, t$ and nothing later. Normalisation and activation come *before* the
convolution rather than after, which leaves the residual path an unmodified identity all the way
through the stack — gradients reach early blocks without passing through a normaliser at every
hop. The convolutions carry no bias. The schedules are

| Block | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| dilation $d_b$ | 1 | 2 | 4 | 8 | 16 |
| kernel $k_b$ (target) | 3 | 7 | 11 | 15 | 15 |
| kernel $k_b$ (source) | 3 | 5 | 11 | 15 | 15 |
| left pad | 2 | 12 | 40 | 112 | 224 |

The stack's receptive field is

$$
1 + \sum_b (k_b - 1)\,d_b = 391 \text{ steps (target)},\qquad 387 \text{ steps (source)},
$$

i.e. roughly $26$ minutes of history at $4$ s per step — longer than a segment, so the
convolutional branch is not the binding constraint on memory.

The stack is a **single plain residual chain**: each block already adds its own residual, and no
second stack-level skip is injected between blocks. A second, rescaled copy of the stream added
at every seam inflates the activation scale through depth and dilutes the input skip, so
deepening the encoder would degrade conditioning instead of improving it. The stack exits
through

$$
x^{\mathrm{conv}} = \mathrm{LayerNorm}\big(\mathrm{stack}(x^{\mathrm{lin}}) + x^{\mathrm{lin}}\big).
$$

**Causal normalisation.** The pre-norm inside each convolution block is a group normaliser. The
standard operator on a $(B, C, T)$ tensor reduces over $(C/G, T)$ — its statistics at step $t$
are functions of the whole sequence, *including* $t' > t$. Every "history" state would then carry
a low-bandwidth image of its own future, which invalidates the entire measurement while being
completely invisible in a loss curve. The model therefore replaces every such normaliser inside
both encoders with a **time-local** variant that reduces over channels within a group at each
timestep independently:

$$
\hat{x}_{b,c,t} = \frac{x_{b,c,t} - \mu_{b,g(c),t}}{\sqrt{\sigma^2_{b,g(c),t} + \varepsilon}}\,\gamma_c + \beta_c,
\qquad
\mu_{b,g,t} = \frac{G}{C}\sum_{c \in g} x_{b,c,t}.
$$

It registers exactly the same parameters under the same names, so it is numerically a no-op for
the affine transform and changes only which elements the statistics pool over. This is a
**correctness requirement**, not a preference; the driver warns loudly when it is switched off.

**Stage C — recurrent branch.** A two-layer **unidirectional** LSTM at hidden size $128$ over
$x^{\mathrm{lin}}$, followed by LayerNorm. Bidirectionality would read the future. The recurrent
branch carries unbounded history through a bottleneck; the convolution branch sees a wide but
fixed window at full resolution. They fail in different directions, which is why both are
present.

**Stage D — fusion.** Concatenate and project back down:

$$
H_t = \mathrm{LayerNorm}\Big(\mathrm{ResMLP}_{256 \to 128}\big([\,x^{\mathrm{conv}}_t \,\Vert\, x^{\mathrm{lstm}}_t\,]\big)\Big),
$$

through hidden widths $(215, 181, 152, 128)$, again a plain seam. The
closing LayerNorm caps the encoder exit at roughly per-step $\mathcal{N}(0, I)$; without it the
exit drifts unbounded, which surfaces downstream as a single latent dimension sitting at an
absurd prior mean.

Output: $H^y, H^u \in \mathbb{R}^{B \times 300 \times 128}$.

### 5.4 Prior head — the complete target-only latent

The prior head reads $H^y$ and produces the whole latent distribution. It has **no
decoder-conditioning output**: a target-only path around the latent would turn $z$ back into a
side-channel code.

Two independent residual MLPs, each preceded by its own LayerNorm (so the two heads are decoupled
from shared drift in $H^y$), map $128 \to 64$ through hidden widths $(111, 97, 84, 74, 64)$. Both
are **unactivated** seams: they emit distribution parameters, which must be free to take either
sign, so a closing GELU would gate exactly the values being parameterised.

$$
\tilde\mu_t = f_\mu\big(\mathrm{LayerNorm}(H^y_t)\big), \qquad
\tilde\ell^p_t = f_\ell\big(\mathrm{LayerNorm}(H^y_t)\big).
$$

The mean is **tanh-bounded**, and the log-variance is **smoothly bounded** into an open interval:

$$
\mu^p_t = a_\mu \tanh\!\left(\frac{\tilde\mu_t}{a_\mu}\right), \qquad a_\mu = 5,
$$

$$
\ell^p_t = \mathrm{sb}(\tilde\ell^p_t), \qquad
\mathrm{sb}(r) = \ell_{\min} + (\ell_{\max} - \ell_{\min})\,\sigma(r),
\qquad (\ell_{\min}, \ell_{\max}) = (-5, 3).
$$

A smooth bound is used rather than a hard clamp because $\mathrm{sb}$ has a strictly positive
gradient everywhere: a log-variance that saturates can still recover, whereas under a hard clamp
the gradient that would pull it back is exactly zero.

The head returns **three** tensors: $\mu^p_t$, the bounded $\ell^p_t$, and the *pre-bound* raw
value $\tilde\ell^p_t$. The third is not a diagnostic. $\mathrm{sb}$ is a sigmoid and therefore
not idempotent, so the posterior's log-variance residual must be applied to the raw value; built
on the bounded one, the exact zero-KL initialisation would silently fail to hold.

### 5.5 Lag cross-attention — deciding *when* the source mattered

A uterine contraction does not move the fetal heart rate at the same instant. The response
arrives some lag $\ell$ later, the lag is not known in advance, and it differs between
recordings. Rather than fixing a lag, the model attends over a window of past source states and
learns where to look; the attention weights are then themselves a readout.

**Query.** Posed from the *prior belief*, not from the encoder state. The prior mean is first
lifted to the model width by a dedicated projection $W_{\mathrm{qp}} \in \mathbb{R}^{128 \times 64}$,
then pre-normed and projected again inside the attention:

$$
q_t = W_Q\,\mathrm{LayerNorm}\big(W_{\mathrm{qp}}\,\mu^p_t\big) \in \mathbb{R}^{128},
\qquad
q_t \ \longrightarrow\ \big(q^{(1)}_t, \dots, q^{(M)}_t\big),\ q^{(m)}_t \in \mathbb{R}^{32},
$$

the split into heads being a reshape of the $128$-vector, not $M$ separate projections. Under a
configuration switch the lifted quantity is $W_{\mathrm{qp}}[\mu^p_t \Vert \ell^p_t]$ with
$W_{\mathrm{qp}} \in \mathbb{R}^{128 \times 96}$ instead, so the query can also condition on how
*certain* the prior belief is. Both inputs are target-only, so source purity holds either way.

Using the mean rather than a sample keeps the lag weights free of latent sampling noise. The
question asked of the source memory is therefore "what would move *this* belief?", posed from the
very latent the posterior will then correct.

**Keys and values.** Projected once from the source state through a shared pre-norm, again as
single $128 \times 128$ maps reshaped into heads:

$$
k_{t'} = W_K\,\mathrm{LayerNorm}\big(H^u_{t'}\big), \qquad
v_{t'} = W_V\,\mathrm{LayerNorm}\big(H^u_{t'}\big),
\qquad
k_{t'}, v_{t'} \ \longrightarrow\ \big(k^{(m)}_{t'}, v^{(m)}_{t'}\big)_{m=1}^{M},
$$

for every source step $t'$, with $k^{(m)}_{t'}, v^{(m)}_{t'} \in \mathbb{R}^{32}$. The projections
are computed once over the whole sequence and the lag window is then taken as a view over them,
never recomputed per lag.

**Scores.** For lag $\ell = 0, \dots, L-1$ with $L = \mathrm{max\_lag} + 1 = 91$:

$$
\boxed{\;
a_{t,m,\ell} =
\frac{\big\langle q^{(m)}_t,\ k^{(m)}_{t-\ell}\big\rangle + \big\langle q^{(m)}_t,\ \rho_{m,\ell}\big\rangle}{\sqrt{d_{\mathrm{head}}}}
\;+\; b_{m,\ell}\;}
$$

where $\rho_{m,\ell} \in \mathbb{R}^{32}$ is a learned per-lag key bias (a relative-position
encoding, seeded $\mathcal{N}(0, 0.02^2)$) and $b_{m,\ell}$ a learned per-(head, lag) scalar
score bias. The window covers

$$
L \cdot \Delta = 91 \times 4\ \mathrm{s} = 364\ \mathrm{s} \approx 6 \text{ minutes of source history}.
$$

**Long lags start penalised.** The score bias is seeded with a negative slope in $\ell$,

$$
b_{m,\ell}\big|_{\text{init}} = -\,\omega_m\,\ell,
\qquad \omega = (0.25,\ 0.0625,\ 0.015625,\ 0.00390625)
$$

for $M = 4$, a geometric per-head schedule. The steepest head sees essentially only the current
step, the shallowest is nearly flat across the window, and the set spans short, medium and long
lags without any head being told which to take. Without this seeding a randomly initialised head
can settle on spurious long-lag structure early and never leave it. The bias is a free parameter
and can be learned away.

**Validity mask.** Lag $\ell$ at step $t$ refers to source step $t - \ell$, which does not exist
for $\ell > t$:

$$
\mathcal{M}^{\mathrm{lag}}_{t,\ell} = \mathbb{1}[\,t - \ell \ge 0\,],
\qquad a_{t,m,\ell} \leftarrow -\infty \ \text{ where } \mathcal{M}^{\mathrm{lag}}_{t,\ell} = 0.
$$

**Normalisation over lag.** Scores are turned into weights along the lag axis. Two normalisers are
available. The shipped one is the sparse $1.5$-entmax,

$$
\alpha_{t,m,\ell} = \Big[\tfrac{1}{2}\,a_{t,m,\ell} - \nu_{t,m}\Big]_+^{\,2},
\qquad [x]_+ = \max(x, 0),
$$

with $\nu_{t,m}$ the unique threshold making $\sum_\ell \alpha_{t,m,\ell} = 1$. The alternative is
the ordinary softmax,

$$
\alpha_{t,m,\ell} = \frac{\exp a_{t,m,\ell}}{\sum_{\ell'} \exp a_{t,m,\ell'}} .
$$

The choice is not cosmetic. Softmax weights are strictly positive, so every one of the $91$ lags
receives some mass; the $1.5$-entmax drives every lag whose score falls below $2\nu_{t,m}$ to
**exactly zero**. When the output is read as "which lag mattered", the difference between $0$ and
$10^{-4}$ across $91$ lags is the difference between a clean answer and a smear — and, because
the lag attribution of §11.4 multiplies these weights by the KL, exact zeros mean exact zeros in
the attribution too.

A row with no valid lag is all $-\infty$ and normalises to $\mathrm{NaN}$; it is mapped to zero,
which is the correct reading — no lag was attended because none was available.

**Attended summaries.** Per head,

$$
A^{(m)}_t = \sum_{\ell=0}^{L-1} \alpha_{t,m,\ell}\; v^{(m)}_{t-\ell} \in \mathbb{R}^{32}.
$$

**Attention dropout is exactly zero.** Dropout is applied to the probabilities *before* they are
returned, and the per-lag attribution in [§11.4](#114-lag-resolved-attribution) is exact only if
the returned weights are the ones the posterior actually consumed. This is a correctness
requirement.

**Memory.** The lag window is formed with strided views over the projected key and value
tensors rather than materialised as a real $(B, T, L, d)$ tensor, which would cost roughly $L$
times the activation memory. The result is numerically identical.

**The output projection $W_o$ is frozen.** The head-structured posterior consumes the per-head
summaries $A^{(m)}$ directly, so the fused projection feeds nothing and receives no gradient.
Clearing `requires_grad` makes that explicit and removes it from the distributed reducer's
expectation set. It accounts for the model's $16{,}512$ frozen parameters.

### 5.6 Posterior head — a head-structured bounded residual

The latent is partitioned into $M = 4$ **groups** of

$$
g = \frac{d_z}{M} = \frac{64}{4} = 16
$$

coordinates each, and group $m$ is written **only** by attention head $m$. This is what makes
the per-group KL a genuine additive decomposition rather than an arbitrary slice of a shared
vector.

Per group $m$:

$$
F^{(m)}_t = \mathrm{ResMLP}^{(m)}\Big(\big[\ \mathrm{LayerNorm}(H^y_t)\ \Vert\ \mathrm{LayerNorm}_{g_A}\big(A^{(m)}_t\big)\ \big]\Big),
\qquad 160 \to (94, 55, 32),
$$

an **activated** seam, where $\mathrm{LayerNorm}_{g_A}$ is a LayerNorm whose scale is initialised
to the constant $g_A$ rather than to $1$. The group's two delta heads are then plain linear maps,

$$
\widetilde{\Delta\mu}^{(m)}_t = W^{(m)}_{\Delta\mu} F^{(m)}_t \in \mathbb{R}^{16},
\qquad
\widetilde{\Delta\ell}^{(m)}_t = W^{(m)}_{\Delta\ell} F^{(m)}_t \in \mathbb{R}^{16}.
$$

The attended summary's LayerNorm carries a **gain**
$g_A = \sqrt{d_{\mathrm{model}}/d_{\mathrm{head}}} = \sqrt{128/32} = 2$. At unit gain the
$32$-wide source summary is out-columned $128 : 32$ by the target state in the fusion, explains
only a small fraction of the fused representation's variance, and receives correspondingly less
gradient while the bottleneck is open. The gain rescales it so both inputs enter the fusion at
comparable magnitude.

Concatenating the groups gives
$\widetilde{\Delta\mu}_t, \widetilde{\Delta\ell}_t \in \mathbb{R}^{64}$, and the posterior is

$$
\boxed{\;
\mu^q_t = \mu^p_t + s_\mu \tanh\!\left(\frac{\widetilde{\Delta\mu}_t}{s_\mu}\right),
\qquad
\ell^q_t = \mathrm{sb}\!\left(\tilde\ell^p_t + s_\ell \tanh\!\left(\frac{\widetilde{\Delta\ell}_t}{s_\ell}\right)\right),\;}
$$

with $s_\mu = 3$ and $s_\ell = 2$. Note the log-variance residual is added to the prior's
**pre-bound raw** value and the bound is applied once to the sum.

Since $\tanh(0) = 0$ and both delta heads are zero-initialised, at initialisation
$\mu^q_t = \mu^p_t$ and $\ell^q_t = \ell^p_t$ *exactly*, hence $K_t \equiv 0$.

### 5.7 Shared horizon decoder

One decoder, invoked on $z^p[:, :270]$ and on $z^q[:, :270]$, receiving nothing else.

The slice is deliberate. The latent exists at all $T = 300$ anchors, but the tail $H = 30$ have no
fully observed raw future, so the loss would discard them; decoding only $[0, T_{\mathrm{valid}})$
saves roughly $10\%$ of the decoder's activation memory at this output size. It is a decision, not
an accident, and it is the reason the two forecast tensors have an anchor axis of $270$ while
every latent tensor has $300$.

**Latent projection.** A residual MLP lifts the latent to the decoder width,

$$
h_t = \mathrm{ResMLP}_{64 \to 256}(z_t), \qquad 64 \to (91, 128, 181, 256),
$$

with a closing normalise-and-GELU. This $h_t$ is the **only** point at which the latent enters
the decoder.

**Horizon expansion.** The projected state is broadcast across $H = 30$ horizon tokens and given
a learned per-step embedding $e_\tau \in \mathbb{R}^{256}$:

$$
F^{(0)}_{t,\tau} = h_t + e_\tau, \qquad \tau = 0, \dots, 29.
$$

Token $\tau$ is "which step of the forecast this is", not "what time it was in the recording".

**Refine stack with per-block FiLM.** Four dilated residual blocks along the *horizon* axis,
kernel $3$, dilations $(1, 2, 4, 8)$ — a receptive field of $31$ over the $30$ horizon tokens, so
every token reaches the whole block through the convolutions alone — each

$$
F^{(b+1)} = F^{(b)} + \Big[\big(1 + \gamma_b(h_t)\big) \odot \mathrm{GELU}\big(\mathrm{Norm}(\mathrm{Conv}_b F^{(b)})\big) + \beta_b(h_t)\Big],
$$

where $(\gamma_b, \beta_b) = \mathrm{Linear}_b(h_t)$ are per-channel modulation parameters
generated **separately for every block**. Convolution along this axis is symmetric rather than
causal, and legitimately so: every position of the horizon axis is predicted from the same anchor
$t$, so pooling across it cannot reach information the anchor did not already have. The same
argument makes the group normalisers here safe to leave non-causal — they pool over the forecast
axis of a single anchor, never across input time.

Per-block FiLM is structural. The learned step embedding is roughly $2.4\%$ the magnitude of the
broadcast latent, so the $30$ horizon tokens enter the stack nearly identical; with a single
top-of-stack modulation the refine stack can synthesise the trajectory shape in a
$z$-independent direction and use the latent only for a coarse offset. Reading $h_t$ at every
block forecloses that, while keeping the decoder's input set unchanged — every modulation is a
function of the latent alone.

**Horizon self-attention.** Two pre-norm bidirectional self-attention blocks then mix the $30$
horizon tokens directly, at $4$ heads:

$$
G^{(a+1)} = G^{(a)} + \varrho_a\, W^{\mathrm{out}}_a\,
  \mathrm{MHSA}_a\!\left(\mathrm{LayerNorm}_a\big(G^{(a)}\big)\right),
\qquad G^{(0)} = F^{(4)},
$$

with $q$, $k$, $v$ and $W^{\mathrm{out}}$ all bias-free and $\varrho_a$ a **scalar** residual gain
initialised at $10^{-2}$, so the stack begins near-identity and every projection still carries
gradient from step $0$. The attention is unmasked, and symmetric for exactly the reason the
convolutions are: the horizon is not time. The convolutions mix through a fixed local window with a
schedule set at construction; these mix all $30$ at once with content-dependent weights, so a
forecast is shaped as a whole rather than assembled from overlapping neighbourhoods. There is no
positional encoding of their own — $e_\tau$ is already what tells the tokens apart.
`horizon_attention_blocks: 0` constructs no module, leaving the core bitwise the one that existed
before them, and `sweep_horizon_attn_off.yaml` is that arm.

The block is hand-rolled rather than `nn.MultiheadAttention` for two correctness reasons: MHA
applies its attention dropout functionally, where the dropout-zero scans guarding the twice-invoked
decoder cannot see it, and the generic initialisation pass would xavier-fill MHA's *packed*
projection as one $3d \times d$ matrix, giving $q$, $k$ and $v$ three times their true fan-in. The
residual gain is a bare `nn.Parameter`, which that pass ignores, so it stays where the constructor
put it.

The stack exits through a residual over the whole refine chain and a LayerNorm:

$$
F_t = \mathrm{LayerNorm}\big(G^{(2)}_t + F^{(0)}_t\big) \in \mathbb{R}^{30 \times 256}.
$$

The skip is the input to the refine stack, not to the attention: the attention sits inside the
residual the skip closes over, so turning it off restores the pre-attention path exactly.

**Output heads.** Each horizon token emits $R = 16$ raw samples:

$$
\hat\mu_{t,\tau} = W_\mu F_{t,\tau} \in \mathbb{R}^{16},
\qquad
\hat\ell_{t,\tau} = \mathrm{sb}\big(W_\ell F_{t,\tau}\big) \in \mathbb{R}^{16},
$$

so the forecast tensors are $(B, 270, 30, 16)$, which reshapes to $480$ raw samples per anchor.

**Decoder dropout is exactly zero.** Invoking one module twice would otherwise draw two
independent dropout masks: base and full would differ at initialisation even with
$z^p = z^q$, and independent noise would enter the base-minus-full readout on every training
step. This is a correctness requirement.

**Why not autoregressive.** A powerful autoregressive decoder can predict each raw sample from
preceding generated or teacher-forced samples and may then need very little from $z$ — which
would defeat the entire purpose. The parallel block decoder forces the latent to specify the
complete future trajectory.

### 5.8 Forward return contract

The forward returns exactly twenty tensors. There is deliberately no decoder-conditioning state
and no source-correction tensor, because neither pathway exists.

| Key | Shape |
| --- | --- |
| `mu_prior`, `logvar_prior`, `raw_logvar_prior` | $(B, 300, 64)$ |
| `mu_post`, `logvar_post` | $(B, 300, 64)$ |
| `z_prior`, `z_post` | $(B, 300, 64)$ |
| `target_state`, `source_state` | $(B, 300, 128)$ |
| `attended_source_heads` | $(B, 300, 4, 32)$ |
| `attn_weights` | $(B, 300, 4, 91)$ |
| `mu_base`, `logvar_base` | $(B, 270, 30, 16)$ |
| `mu_full`, `logvar_full` | $(B, 270, 30, 16)$ |
| `kld_per_t` | $(B, 300)$ |
| `kld_per_t_per_head` | $(B, 300, 4)$ |
| `source_kl_lag_map` | $(B, 300, 91)$ |
| `mu_prior_sat_frac`, `delta_mu_sat_frac` | scalars |

### 5.9 Construction-time validation

The constructor refuses inconsistent geometry before anything is built. Each rejected case would
otherwise produce a model that is *wrong* rather than one that *fails*, and none of them would
surface in a loss curve:

| Refused | Why it must not be allowed to build |
| --- | --- |
| $c_y < 1$ or $c_u < 1$ | a zero-width linear layer is legal and simply returns its bias, so the model would train to completion having never read that stream |
| $M \cdot d_{\mathrm{head}} \ne d_{\mathrm{model}}$ | the head reshape would silently mis-slice the projections |
| $\mathrm{max\_lag} < 0$ | an empty attention window collapses the attended source to a bias — the model never reads the source, then reports its KL as a measurement of it |
| $d_z \bmod M \ne 0$ | the per-group KL would no longer be an additive decomposition (§11.2) |
| a keep-index that is empty, out of range, or not strictly ascending | the delay vector is positional *against* the keep-index, so a reordered index delays the wrong channels with no other failure signal |
| $\max_c \delta_c > w$ | the zero-filled prefix of a delayed stream would reach the scored anchors |
| a degenerate raw geometry | $T_{\mathrm{valid}} < 1$, a warm-up outside $[0, T-H)$, or a raw length not divisible by $D$ |

The geometry object additionally asserts its own index identities at construction —
$\mathrm{start}(0) = D$, $n_{\mathrm{raw}}(0) = D - 1$, and that the last anchor's window ends
exactly at $L_{\mathrm{raw}} - 1$ — so an unvalidated geometry cannot exist anywhere in the
program.

---

## 6. Parameter budget

Measured on the constructed model at the shipped configuration.

| Component | Parameters | Share |
| --- | ---: | ---: |
| Horizon decoder core (shared) | 1,849,346 | 36.35% |
| Target encoder | 1,344,999 | 26.43% |
| Source encoder | 1,312,231 | 25.79% |
| Posterior head | 116,484 | 2.29% |
| Decoder projection + output heads | 113,936 | 2.24% |
| Prior head | 108,010 | 2.12% |
| Target input adapter | 81,408 | 1.60% |
| Lag cross-attention | 78,572 | 1.54% |
| Source input adapter | 74,880 | 1.47% |
| Attention query projection | 8,320 | 0.16% |
| **Total** | **5,088,186** | 100% |
| — of which frozen (attention $W_o$) | 16,512 | 0.32% |
| — trainable | 5,071,674 | 99.68% |

The shipped `causal_reach_budget_s: 120` adds $6{,}272$ more for the two availability adapters, so
the model as launched holds $5{,}094{,}458$; the table is the unguarded build, which is the number
the design records quote.

**The capacity revision reordered this table.** The two history encoders were nearly $79\%$ of the
model and are now $52.2\%$, because the decoder went from $128$ wide and three refine blocks to
$256$ and four, and gained the horizon self-attention — it is now the single largest component. That
was the point: the decoder was the only thing standing between one latent vector and $480$ raw
samples, and at $128$ it held roughly a fifth of what the encoders did. Within each encoder the
convolution stack still dominates: the five blocks cost $49{,}408$, $114{,}944$, $180{,}480$,
$246{,}016$ and $246{,}016$ parameters respectively, and the two-layer LSTM $264{,}192$.

The horizon core's $1{,}849{,}346$ parameters break down as four horizon convolutions
($4 \times 196{,}864$), their group norms ($4 \times 512$), four FiLM generators
($4 \times 131{,}584$), two horizon self-attention blocks ($2 \times 262{,}657$), the horizon-step
embedding ($30 \times 256 = 7{,}680$) and the output LayerNorm ($512$). Each attention block is four
bias-free $256 \times 256$ projections, one `LayerNorm` and one scalar residual gain, so
$4 d_{\mathrm{hidden}}^2 + 2 d_{\mathrm{hidden}} + 1$; `horizon_attention_blocks: 0` builds none of
it rather than building it inert.

---

## 7. Structural invariants

These are properties of the architecture, not of the training run. Each is enforced at
construction and pinned by a test.

### 7.1 No decoder bypass

The decoder's `forward` signature accepts exactly one tensor. There is no second decoder, no
conditioning head, and no path from either encoder state to the decoder. Formally, for the
decoder parameters $\psi$,

$$
\frac{\partial\,\hat\mu_t}{\partial\,H^y_{t'}} = 0
\quad\text{except through } z_t,
$$

so the only route by which target information reaches the forecast is the latent. This is what
forces $\mu^p_t$ to *be* the predictive state of the target and makes $\mu^q_t - \mu^p_t$ the
additional source-derived predictive information.

### 7.2 Source purity

$$
\mu^p_t, \ \ell^p_t = f\big(Y_{\le t}\big) \quad\text{only},
\qquad
H^u_t = g\big(U_{\le t}\big) \quad\text{only}.
$$

The two streams pass through separate adapters and separate encoders; the prior reads $H^y$
alone; the attention query is a projection of prior quantities; and no feature that mixes both
signals is ever loaded. Without this, $K_t$ would not measure what the source added.

### 7.3 Exact zero at initialisation

At construction, in training mode,

$$
K_t \equiv 0 \quad\text{and}\quad
\hat\mu^{\mathrm{base}} = \hat\mu^{\mathrm{full}},\quad
\hat\ell^{\mathrm{base}} = \hat\ell^{\mathrm{full}}
\quad\text{bitwise.}
$$

Three mechanisms together produce this: the posterior residual heads are zeroed (so $q \equiv p$
exactly), one shared $\epsilon$ makes $z^q = z^p$ sample by sample rather than merely in
distribution, and the twice-invoked decoder and the attention probabilities carry no dropout.

A corollary worth stating explicitly: **any KL assertion on a freshly constructed model passes
vacuously.** Tests that claim to check KL behaviour must first perturb the posterior.

### 7.4 Zero dropout where it would corrupt a readout

Dropout is $0.1$ in the adapters, encoders and latent heads. It is exactly $0$ in the decoder
(§5.7) and in the attention probabilities (§5.5). Both zeros are correctness requirements with
named failure modes, not stylistic choices.

---

## 8. Initialisation

Initialisation runs in a fixed order, and the order is load-bearing.

**Step 1 — generic initialisation.** Xavier-uniform for every linear and convolutional weight;
orthogonal for every LSTM input and recurrent weight; zeros for biases; ones for LayerNorm
scales. The LSTM forget-gate bias slice is set to $1$, which starts the forget gate open so
gradients survive the early steps of a long sequence instead of decaying through a near-zero
gate.

**Step 2 — zero the posterior residual heads.** *After* step 1, never before: the generic
initialisation would otherwise Xavier-refill them and destroy the exact zero-KL start.

**Step 3 — zero the FiLM generators.** Same reasoning. The horizon core zero-initialises its
generators for an identity at construction, and step 1 undoes it; re-zeroing here is what makes
the identity actually true. At step $0$ the per-block-FiLM decoder is bitwise the FiLM-free
decoder, and the latent enters the trajectory shape only as training drives the generators off
zero.

**Step 4 — three zero-parameter initialisation policies.** Each is applied only when its
configured value departs from a no-op default, so a default-flag model is bitwise the pre-policy
one. None adds a parameter.

* **Horizon-embedding rescale.** The core seeds $e_\tau \sim \mathcal{N}(0, 0.02^2)$, roughly
  $2.4\%$ of the broadcast latent's magnitude, leaving the $30$ horizon tokens $\approx 0.999$
  correlated at initialisation and per-block FiLM with almost no token-specific structure to
  modulate. Re-seeding at $\mathcal{N}(0, 0.8^2)$ drops the token correlation to $\approx 0.45$
  and raises the token-specific variance fraction from $\approx 0.06\%$ to $\approx 48\%$,
  giving every block distinct per-step offsets to shape from step $0$. The posterior deltas are
  untouched, so the exact zero-KL start survives.

* **Output-head calibration.** Xavier-filled output heads emit a high-variance mean and an
  over-confident low log-variance, so the initial factorised Gaussian NLL of a z-scored target
  sits far above the trivial $\mu = 0, \sigma = 1$ predictor's. Three edits move the decoder onto
  that predictor at step $0$: scale the mean head's weight by $0.02$ so $\hat\mu \approx 0$
  (scaled, not zeroed, so a perturbed posterior still moves the two forecasts apart), set the
  log-variance bias to $\log(5/3)$ so that

  $$
  \mathrm{sb}\big(\log(5/3)\big) = -5 + 8\cdot\sigma\big(\log(5/3)\big) = -5 + 8\cdot\tfrac58 = 0,
  $$

  i.e. $\sigma = 1$ exactly, and scale the log-variance weight by $0.1$ so the initial spread
  around that centre is small. Both forecasts share this one decoder, so both stay calibrated
  identically and every bitwise-at-initialisation contract is preserved.

  The same key calibrates the **prior head's log-variance** onto unit scale ($\sigma_p = 1$, the
  scale-anchor term's optimum) — the same trivial-predictor policy, applied to the other
  distribution head. The mechanism is the posterior deltas' zero-weight recipe rather than a
  shrink: the head is a residual MLP with no single bias governing its output level, so the final
  body layer's weight and the whole skip projection are zeroed and the final bias set to the
  pre-image of log-variance $0$ under the bound — $\log(5/3)$ at the shipped $(-5, 3)$ — making
  the raw output input-independent and the bounded output exactly $0$. The posterior's residual
  is built on that same raw tensor, so the exact zero-KL start is untouched, and the zeroed
  layers still receive gradient, so the head trains off the constant like the posterior deltas
  do. Without this half, nothing in the initialisation places the prior's scale, and the first
  production run measured it collapsing onto the clamp floor within one epoch.

* **Attended-source gain.** Set the posterior fusion's attended-source LayerNorm gain to
  $\sqrt{d_{\mathrm{model}}/d_{\mathrm{head}}} = 2$ (see §5.6). The deltas are still zero, so the
  KL is still exactly zero.

---

## 9. Masks, supports and validity

### 9.1 Forecast mask

An anchor's forecast step $\tau$ is scored only when four factors are simultaneously nonzero: the
anchor is past the warm-up, the anchor's own step is valid, the forecast step is valid, and the
anchor clears the coverage floor of §9.2.

$$
m_{t,\tau} = \mathbb{1}[\,t \ge w\,]\ \cdot\ \mathrm{valid}(t)\ \cdot\ \mathrm{valid}(t + 1 + \tau)\ \cdot\ \mathbb{1}\!\left[\,\bar c_t \ge c_{\min}\,\right],
$$

shape $(B, 270, 30)$, broadcast over the $R = 16$ raw samples of each step.

### 9.2 Coverage floor

$$
\bar c_t = \frac{1}{H}\sum_{\tau=0}^{H-1} \mathrm{valid}(t + 1 + \tau)
$$

is the fraction of an anchor's forecast window that is observed. An anchor below
$c_{\min} = 0.9$ is zeroed *entirely* rather than partially scored. The reason is arithmetic: a
half-masked anchor's *summed* block NLL covers half a window, so the base-minus-full gap read off
it is spuriously small, and mixing such anchors into the average biases the coupling readout
downward. The pre-floor distribution of $\bar c_t$ is logged so the threshold can be re-derived
from data.

### 9.3 Contributing anchors and the KL support

$$
c_t = \mathbb{1}\!\left[\max_\tau m_{t,\tau} > 0\right]
$$

is the single indicator used by *both* the reconstruction's per-anchor denominator and the KL
mask:

$$
m^{\mathrm{KL}}_t = \begin{cases} c_t & t < T - H,\\ 0 & \text{otherwise.}\end{cases}
$$

**The KL support is derived from the forecast mask, not restated from the validity signal.** The
distinction matters. Charging $\beta K_t$ on an anchor that carries no reconstruction term leaves
nothing pulling the posterior off the prior, so it is regularised onto the prior for free. On the
tail $H$ anchors that appears as an end-of-sequence droop resembling fading coupling; on anchors
dropped by a gap or by the coverage floor it appears immediately before every signal-loss gap —
the same artifact, in the place it is hardest to recognise. Deriving the support rather than
restating it forecloses both, and guarantees that the reconstruction and KL terms are averaged
over one anchor set rather than two that agree only by coincidence.

---

## 10. The objective

### 10.1 Units, and why they are the point

Every quantity in the objective is in **nats per anchor**. The reconstruction NLL is *summed*
over the $H \cdot R = 480$-sample block, with its full constant so the value is a true
log-density; the KL is *summed* over $d_z$; and both are then averaged over batch and
**contributing anchors only**. This is what makes the KL weight $\beta$ mean anything at all: at
$\beta = 1$, reconstruction-plus-KL is the exact evidence lower bound of the source-conditioned
branch.

If the reconstruction were averaged over the horizon while the KL stayed summed over latent
dimensions, changing the horizon from $480$ to $600$ samples would silently change the
operational meaning of $\beta$.

### 10.2 Per-sample score

$$
s_{t,\tau,r} =
\begin{cases}
\dfrac{1}{2}\left[\log 2\pi + \hat\ell_{t,\tau,r} + \big(X^{+}_{t,\tau,r} - \hat\mu_{t,\tau,r}\big)^2 e^{-\hat\ell_{t,\tau,r}}\right] & \text{Gaussian NLL},\\[10pt]
\big(X^{+}_{t,\tau,r} - \hat\mu_{t,\tau,r}\big)^2 & \text{squared error}.
\end{cases}
$$

The shipped likelihood is the Gaussian, which consumes the decoder's learned log-variance heads —
the observation model is the decoder's own heteroscedastic output, unconditionally, with no
separate observation-noise hyperparameter anywhere in the model.

### 10.3 Block reduction

Per anchor,

$$
d_t = \sum_{\tau=0}^{H-1}\sum_{r=0}^{R-1} m_{t,\tau}\, s_{t,\tau,r},
$$

and over the batch,

$$
D = \frac{\sum_{b,t} d_{b,t}}{\max\!\left(1, \sum_{b,t} c_{b,t}\right)},
\qquad
D_{\mathrm{sample}} = \frac{D}{H \cdot R} = \frac{D}{480}.
$$

The mask multiplies, so a masked position contributes exactly zero — a finite planted value at a
masked position cannot move the loss at all — and a fully masked anchor leaves both numerator and
denominator, so the per-anchor scale does not drift with mask density. The squared-error variant
sums over the same block, so $\beta$ keeps its meaning across likelihoods.

Applying this to the two forecasts gives

$$
D_1 = D\big(\hat\mu^{\mathrm{full}}, \hat\ell^{\mathrm{full}}\big),
\qquad
D_0 = D\big(\hat\mu^{\mathrm{base}}, \hat\ell^{\mathrm{base}}\big).
$$

### 10.4 The KL terms

From the per-dimension divergence $\kappa_{t,d}$ of §4.5, two scalars are produced and they are
**not interchangeable**:

$$
K_{\mathrm{raw}} = \frac{\sum_{b,t} m^{\mathrm{KL}}_{b,t}\sum_d \kappa_{b,t,d}}{\max\!\left(1, \sum_{b,t} m^{\mathrm{KL}}_{b,t}\right)},
\qquad
K_{\mathrm{train}} = \frac{\sum_{b,t} m^{\mathrm{KL}}_{b,t}\sum_d \max\big(\kappa_{b,t,d},\ \eta\big)}{\max\!\left(1, \sum_{b,t} m^{\mathrm{KL}}_{b,t}\right)}.
$$

$K_{\mathrm{raw}}$ is computed without gradient and is the **only** quantity that may be read as
an information rate. $K_{\mathrm{train}}$ applies a per-dimension per-step free-bits floor $\eta$
*before* masking and is what enters the loss; with $\eta > 0$ it exceeds the raw value by
construction, which is exactly why it must never be reported as a measurement. The shipped floor
is $\eta = 0$, so the two currently coincide — which is precisely why the distinction has to be
documented rather than observed.

A per-dimension activity indicator is also produced:

$$
\mathrm{active} = \frac{1}{d_z}\sum_{d=1}^{d_z} \mathbb{1}\!\left[\ \overline{\kappa_{\cdot,\cdot,d}} > \epsilon_{\mathrm{act}}\ \right],
\qquad \epsilon_{\mathrm{act}} = 10^{-2},
$$

the mean taken over the masked support. It is the fraction of latent dimensions carrying
information, and the primary early-warning signal for a collapsing bottleneck.

### 10.5 The prior scale rate

The three terms above leave the prior's own scale unconstrained from below, so a fourth is needed.
Per anchor and per dimension it is

$$
\rho_{t,d} = \tfrac12\left(e^{\ell^p_{t,d}} - 1 - \ell^p_{t,d}\right),
\qquad
R_p = \frac{\sum_{b,t} m^{\mathrm{KL}}_{b,t}\sum_d \rho_{b,t,d}}
           {\max\!\left(1, \sum_{b,t} m^{\mathrm{KL}}_{b,t}\right)},
$$

which is $\mathrm{KL}\!\left(\mathcal{N}(\mu^p_t, \operatorname{diag} e^{\ell^p_t})
\,\Vert\, \mathcal{N}(\mu^p_t, I)\right)$ — the **scale half** of the divergence from the prior to a
unit-scale Gaussian at the same mean. It is nonnegative, convex in $\ell^p$, and exactly zero at
$\sigma_p = 1$. The reduction is the KL's, term for term: summed over $d_z$, masked by the same
$m^{\mathrm{KL}}$, divided by the same denominator — so $R_p$ is in nats per anchor and adds to
$K_{\mathrm{train}}$ without rescaling. Unlike $K_{\mathrm{raw}}$ it carries gradient; weighted, it
is an objective term rather than only a diagnostic.

The prior **mean** does not appear in $\rho$, and that is the property that makes the term safe to
weight. The full context rate $\mathrm{KL}(p_\theta \Vert \mathcal{N}(0, I))$ splits exactly into
$R_p + \tfrac12\sum_d (\mu^p_{t,d})^2$, and the second half compresses the state $D_0$ is built
from — so only the first is adopted, and $D_0$ is not traded away to buy the fix. §15 records that
choice as a deliberate limitation.

### 10.6 The auxiliary shape terms

The factorized Gaussian of §10.2 scores every raw sample independently, so its optimum is the
conditional mean — and a fully parallel block decoder is free to emit an over-smoothed one. The
per-sample likelihood cannot tell a forecast with the right envelope, slope and starting level from
one with merely the right average. Three terms on the forecast **means** can, and each is computed
for both branches $k \in \{\mathrm{base}, \mathrm{full}\}$ and **summed** over them — the same
convention $\lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0$ uses at unit weights, so a
shape weight prices both forecasts equally and the formulas below are written per branch.

Write $\bar\mu^{(k)}_{t} \in \mathbb{R}^{HR}$ for the anchor's forecast block flattened over its
horizon and raw axes, $\bar X^{+}_{t}$ for the target flattened the same way, and $\bar m_t$ for the
forecast mask of §9 expanded to that axis. All three divide by $\sum_{b,t} c_{b,t}$ — the *same*
contributing-anchor count as $D$ — so their per-anchor scale is comparable to the reconstruction's
even though their unit is not.

$$
\mathcal{L}_{\mathrm{ms}} = \frac{\sum_{r \in \mathcal{R}} \sum_{b,t} \sum_{j}
    P_r(\bar m)_j \,\big| P_r(\bar m \odot \bar\mu)_j - P_r(\bar m \odot \bar X^{+})_j \big|}
  {\max\!\left(1, \sum_{b,t} c_{b,t}\right)},
\qquad \mathcal{R} = \{1, 4, 16\},
$$

with $P_r$ average pooling at rate $r$, $j$ indexing the pools, and the pooled mask — that pool's
valid *fraction* — carried alongside as the weight. The rates are summed rather than averaged, so
$\mathcal{R}$'s size is part of the term's scale and therefore part of what $\lambda_{\mathrm{ms}}$
is calibrated against. **The
mask is applied before pooling**, which is the one place in the objective the multiplicative-mask
convention runs early: pooling mixes neighbours, so a gap left in until afterwards would leak its
sentinel into every pool it touches. Both sides are scaled identically, so a partially covered pool
still compares like with like. A block length that is not a multiple of a rate drops its trailing
remainder — `avg_pool1d`'s own behaviour, and what keeps every pool a full-width average rather than
a boundary special case. The flattened block must be at least $\max \mathcal{R}$ long, and the
function raises naming the geometry otherwise.

$$
\mathcal{L}_{\mathrm{deriv}} = \frac{\sum_{b,t} \sum_i
  \bar m_{t,i}\,\bar m_{t,i+1}\,\mathrm{Huber}_{\delta=1}\!\big(\Delta_i \bar\mu,\ \Delta_i \bar X^{+}\big)}
  {\max\!\left(1, \sum_{b,t} c_{b,t}\right)},
$$

a difference pair being valid only when **both** of its samples are — the mask is the product of the
two, not either one.

$$
\mathcal{L}_{\mathrm{boundary}} = \frac{\sum_{b,t \ge 1}
  v_{b,t}\, c_{b,t}\,\big|\hat\mu_{b,t,0,0} - X^{+}_{b,t-1,0,R-1}\big|}
  {\max\!\left(1, \sum_{b,t} c_{b,t}\right)},
$$

the first forecast sample against the anchor's own last observed one, with $v_{b,t}$ the thresholded
`weight`. No new tensor is plumbed into
the objective for it: anchor $t-1$'s horizon step $0$ *is* decimated step $t$, so on the raw grid,
where $X = R = D$, $Y[n_t]$ is a slicing identity on the gathered
target — `mu[:, 1:, 0, 0]` against `target[:, :-1, 0, -1]`. The sum runs over
$t \in [1, T_{\mathrm{valid}})$ **structurally**, so anchor $0$ is excluded by construction rather
than by assuming a warm-up, and the validity factor is anchor $t$'s own thresholded `weight` times
its contributing indicator — deliberately *not* anchor $t-1$'s forecast mask, which would import a
different anchor's coverage-floor decision.

**A term whose weight is $0.0$ is not computed**, and its metric is reported as exact $0.0$ rather
than the value it would have had. That keeps a term-off arm's CSV honest, keeps the feature-target
siblings from reporting raw-waveform formulas evaluated over a channel axis, and keeps the
full-block intermediates out of that run's graph. The branch reads a config-constant float,
identical on every rank and every batch, so DDP graph identity is unaffected.

### 10.7 The total objective

$$
\boxed{\;
\mathcal{L} \;=\; \lambda_{\mathrm{full}}\, D_1 \;+\; \lambda_{\mathrm{base}}\, D_0 \;+\; \beta(e)\, K_{\mathrm{train}} \;+\; \beta_p\, R_p
\;+\; \lambda_{\mathrm{ms}}\, \mathcal{L}_{\mathrm{ms}}
\;+\; \lambda_{\mathrm{deriv}}\, \mathcal{L}_{\mathrm{deriv}}
\;+\; \lambda_{\mathrm{boundary}}\, \mathcal{L}_{\mathrm{boundary}},
\qquad \lambda_{\mathrm{full}} = \lambda_{\mathrm{base}} = 1 .\;}
$$

**The total is mixed-unit, and §10.1 does not extend to it.** The first four terms are nats per
anchor; the last three are $L_1$ and Huber quantities on z-scored raw samples. So `total_loss`'s
*level* is not readable as nats, and comparing it across arms carrying different $\lambda$ weights
compares two different quantities. Its direction is still the training signal, `nll_full_block` and
`nll_base_block` remain the pure-nats readouts every comparison uses, and the three `aux_*` metrics
carry each shape term on its own.

Four remarks make the rest of the objective readable.

**The full branch alone is an ELBO.** $\lambda_{\mathrm{full}} D_1 + \beta K$ at $\beta = 1$ is
the exact evidence lower bound of the source-conditioned branch: reconstruction under
$q_\phi$ plus the divergence of $q_\phi$ from the conditional prior $p_\theta$.

**Why the base term exists.** Without $D_0$ the model could route target information through the
latent and post an excellent forecast while telling us nothing about the source. Training the
base branch as a *strong forecaster in its own right* is what forces the source-conditioned
branch to earn its advantage by carrying information the target's past did not already have.
Adding $D_0$ at unit weight doubles the reconstruction pressure against the KL, so **$\beta = 1$
is a principled starting point, not a distinguished optimum** — it is a sweep axis.

**Why $\beta$ warms up from exactly zero.** Because $z$ is the only route to the decoder,
charging a nonzero KL before the decoder can use the latent at all is the standard route to
posterior collapse. The schedule is linear from $0$:

$$
\beta(e) = \beta_{\mathrm{start}} + \big(\beta_{\mathrm{end}} - \beta_{\mathrm{start}}\big)\,
\min\!\left(1,\ \frac{e}{E_{\mathrm{warm}}}\right),
\qquad \beta_{\mathrm{start}} = 0,\ \beta_{\mathrm{end}} = 1,\ E_{\mathrm{warm}} = 50 .
$$

An unrecognised schedule name raises rather than falling back to a constant: silently training a
different objective than the configuration describes is worse than failing.

**Why $\beta_p$ does *not* warm up.** The incentive $R_p$ opposes is one-sided and immediate: the
reconstruction strictly prefers a deterministic latent, since sampling noise can only degrade a
forecast, and $K$ measures $q$ *against* $p$ without constraining $p$'s own scale. Unweighted, the
first production run of this objective drove `logvar_prior_floor_frac` from $0.118$ at epoch $0$ to
$0.812$ at epoch $1$ and finished at $0.992$ — the collapse completes inside one epoch, so any
warm-up would arrive after the damage. $\beta_p$ is therefore a constant.

It is also a **threshold rather than a dial**, which is why the shipped value is bracketed by a
sweep rather than tuned by interpolation. The weighted restoring force
$\beta_p\,\partial\rho/\partial\ell^p = \tfrac12\beta_p(e^{\ell^p} - 1)$ saturates at
$-\tfrac12\beta_p$ per
dimension as $\ell^p \to -\infty$, while the reconstruction's opposing pressure *grows* as the
decoder sharpens: below the crossing weight the prior still pins, only later. Measured on the
committed HIE shard, $\beta_p = 10^{-2}$ delayed the collapse $6.7\times$ in optimizer steps and
finished at $0.955$ anyway, while $\beta_p = 0.1$ held at $0.046$ with `kld_active_frac` at $0.69$
against $0.24$–$0.25$ in every collapsed arm. $0.1$ ships; the four `sweep_beta_prior_*` arms of the
transformer package bracket it on the architecture the collapse was first measured on.

$R_p$ is computed and logged **unconditionally** — under every likelihood, whatever $\beta_p$ is —
so a collapsing prior is visible in any run's metrics whether or not that run paid for the anchor.

---

## 11. Readouts and diagnostics

### 11.1 The two coupling numbers

| Name | Definition | Reading |
| --- | --- | --- |
| `pred_gap` | $D_0 - D_1$ | predictive gain in nats per anchor; the primary readout |
| `source_conditioned_kl_raw` | $K_{\mathrm{raw}}$ | source-conditioned latent rate; the secondary readout |
| `source_conditioned_kl_train` | $K_{\mathrm{train}}$ | the optimiser's quantity; **never** an information rate |

### 11.2 The latent as an exported representation

Because the decoder receives nothing but $z$, the latent is itself a deliverable and not only an
intermediate. Three quantities are exportable per anchor, and they mean different things:

| Quantity | Meaning |
| --- | --- |
| $\mu^p_t$ | the target-only predictive state of the fetal heart rate at $t$ |
| $\mu^q_t - \mu^p_t$ | the additional predictive information the source contributed |
| $\mu^q_t$ | the complete state after the source has been incorporated |

The means are used rather than the samples, so a downstream consumer gets a deterministic
representation. The decomposition is exact — $\mu^q_t = \mu^p_t + (\mu^q_t - \mu^p_t)$ by
construction, since the posterior is a residual on the prior in the prior's own coordinates
(§5.6) — which is what makes the middle row usable on its own.

### 11.3 Per-head decomposition

Because latent group $m$ is written only by attention head $m$,

$$
K^{(m)}_t = \sum_{d \in \text{group } m} \kappa_{t,d},
\qquad
K_t = \sum_{m=1}^{M} K^{(m)}_t
$$

is a genuine additive decomposition, not an arbitrary slice.

### 11.4 Lag-resolved attribution

$K_t$ says *how much* the source moved the belief; the attention weights say *from which lag*.
Their product is the attribution:

$$
\boxed{\;\widetilde K_{t,\ell} = \sum_{m=1}^{M} K^{(m)}_t\; \alpha_{t,m,\ell}\;}
$$

and, because the attention probabilities sum to one over $\ell$ and carry no dropout,

$$
\sum_{\ell=0}^{L-1} \widetilde K_{t,\ell} = K_t \qquad \textbf{exactly}.
$$

This identity is the reason the attention is constructed at zero dropout.

### 11.5 From a lag index to seconds

Two different quantities can be computed from a lag index $\ell$, and conflating them is a real
reporting error rather than a hypothetical one.

$$
\boxed{\;\tau_{\mathrm{compensated}} = \Delta\,(\ell + \delta) = 4(\ell + \delta)\ \mathrm{s}.\;}
$$

* $\delta$ is the causal input delay applied to the source channels ([§14](#14-causal-reach-control-of-the-inputs)); it is $0$ when no reach budget is configured.
* There is **no dataset-shift term**. The stored UP/FHR timeline is canonical: the dataset builder shifts the UP channel when it writes the shards, that shift is part of how the stored signals are, and nothing downstream adds it back, subtracts it, budgets it or interprets it.
  An earlier revision of `lag_report.py` subtracted $20$ s to reach a "sensor timeline"; that
  helper was removed on 2026-09-05.

**$\tau_{\mathrm{compensated}}$ is the quantity to report as *the* lag.** Figure axes use a shared label constant so a plot and the
number beside it cannot disagree about which is shown, and every consumer reads $\delta$ from one
accessor on the model — because the model is what was trained, and two consumers each guessing
the delay is how two reports of the same run came to disagree by two minutes with nothing raising.

### 11.6 Health diagnostics

These exist because specific failure modes are invisible in a loss curve.

| Metric | Watches for |
| --- | --- |
| `mean_logvar_prior`, `logvar_prior_floor_frac` | **A prior variance pinned on its lower clamp.** The KL carries $(\mu^q - \mu^p)^2/\sigma_p^2$, so a floored $\sigma_p^2$ inflates the coupling readout by orders of magnitude while every decoder-side diagnostic looks healthy. Logged from epoch $0$. |
| `prior_rate` | **The same pathology as a distance rather than a fraction**, and the objective term itself: $R_p$ in nats per anchor (§10.5). Zero means $\sigma_p = 1$ exactly; it is the *only* one of these that is bounded below by its own optimum, so a **rising** `prior_rate` is the prior's scale walking away from unit scale in either direction — downward towards the clamp in every case observed so far, which the floor fraction then confirms. It leads that fraction: $R_p$ grows continuously from the first step, while `logvar_prior_floor_frac` reads $0$ until the mass actually reaches the margin. Logged whatever `beta_prior` is, so it diagnoses unanchored runs too. |
| `beta_prior` | The weight in force, echoed like `kld_beta` so a `metrics_history.csv` identifies its own arm and every weighted term can be recomposed from the file alone. |
| `aux_multiscale`, `aux_derivative`, `aux_boundary` | The three shape terms of §10.6, each on its own, and what the three provisional weights are re-derived from — read them against `main_loss` to see what fraction of the criterion each is buying. Their weights `lambda_ms`, `lambda_deriv` and `lambda_boundary` are echoed beside them on the same recomposition principle. A term whose weight is $0.0$ reports exact $0.0$ because it is **not computed**, so a zero column means the term was off, not that the forecast satisfied it. |
| `grad_clip_frac` | **Whether the clip threshold is binding.** `grad_norm` gives the norm; whether it exceeded `gradient_clip_val` on a given step is not recoverable from an epoch aggregate. $1.0$ is normalised-gradient descent in disguise — the first production run recorded exactly that — and $0.0$ is a clip that never fires. Omitted entirely when the trainer configures no clipping. |
| `logvar_full_floor_frac`, `logvar_full_ceil_frac` | **A binding decoder variance bound**, at each end separately. Pinned at the floor the decoder is over-confident and the NLL's squared term explodes — this is what a loss spike looks like from the inside. Pinned at the ceiling it has given up and is predicting noise, which reads as a healthy falling NLL while `pred_gap` goes to zero. A single mean cannot distinguish either from a well-spread distribution. |
| `kld_active_frac` | Collapse onto one or two latent dimensions. |
| `mu_prior_sat_frac`, `delta_mu_sat_frac` | Fraction of the tanh-bounded means within $1\%$ of their bound. A bound that is always active is a silently mis-set hyperparameter. |
| `delta_mu_rms`, `mu_post_prior_gap_rms` | The size of the source-induced belief shift, per coordinate and per step respectively. |
| `anchor_coverage_frac` | The pre-floor distribution the coverage floor is re-derived from. |
| `grad_norm` | The **pre-clip** global gradient $L_2$ norm, the only quantity the clip threshold can be derived from. |

**The two gradient columns are sampled, not reduced.** Every other column here is a true epoch
mean. These two are logged from `on_before_optimizer_step`, and `MetricsLoggingCallback` reads
`trainer.callback_metrics` at validation-epoch end — before the *training* epoch is reduced — so
what reaches `metrics_history.csv` is one optimizer step per epoch. That is a usable sample rather
than a defect, since a clip threshold is a per-step question and one step per epoch is an unbiased
draw from the per-step distribution, but it governs how the two are read: quantiles of `grad_norm`
are per-step quantiles over a thinned sample, and `grad_clip_frac` is $0$ or $1$ per row whose
**mean over epochs** estimates the exceedance fraction.

**The two thresholds, exactly.** Both are module constants read by the training log *and* the
offline evaluation, because two copies of a threshold are two thresholds.

*Saturation* of a tanh-bounded quantity means within $1\%$ of its own bound, at
$\varsigma = 0.99$:

$$
\mathrm{sat}\big(\mu^p\big) = \operatorname{mean}\Big(\mathbb{1}\big[\,|\mu^p_{t,d}| \ge \varsigma\, a_\mu\,\big]\Big),
\qquad
\mathrm{sat}\big(\Delta\mu\big) = \operatorname{mean}\Big(\mathbb{1}\big[\,|\mu^q_{t,d} - \mu^p_{t,d}| \ge \varsigma\, s_\mu\,\big]\Big).
$$

*On the clamp* for a log-variance means within a fixed fraction $\varrho_{\mathrm{margin}} = 0.05$
of the clamp **range** of an asymptote:

$$
\text{on the floor: } \ell \le \ell_{\min} + \varrho_{\mathrm{margin}}(\ell_{\max} - \ell_{\min}),
\qquad
\text{on the ceiling: } \ell \ge \ell_{\max} - \varrho_{\mathrm{margin}}(\ell_{\max} - \ell_{\min}),
$$

i.e. $\ell \le -4.6$ and $\ell \ge 2.6$ at the shipped clamp. A margin is required rather than an
equality test because the smooth bound never reaches an asymptote exactly — an equality test would
report $0.0$ forever while the variance sat pinned.

### 11.7 Reading the two numbers together

$\Delta_t$ and $K_t$ answer different questions and only their combination is interpretable. The
four regimes:

| $K$ | $\Delta$ | Reading |
| --- | --- | --- |
| $\approx 0$ | $\approx 0$ | the source pathway never opened — a collapsed bottleneck, not an absence of coupling in the data |
| large | $\approx 0$ | the posterior moves but not usefully: the code is paying for source information the decoder cannot convert into a better forecast. Check the prior variance first (§11.6) — a floored $\sigma_p^2$ manufactures exactly this signature |
| $\approx 0$ | $> 0$ | arithmetically near-impossible: with $q \approx p$ the two forecasts share a latent. A nonzero gap here indicates a leak, not a finding |
| $> 0$ | $> 0$ | the intended regime — and still only meaningful once the permutation control (§13.3) shows the gain is specific to *this* recording's source |

$\Delta_t$ is the primary readout precisely because it is the one that cannot be inflated by a
degenerate variance: it is measured in prediction space against the observed future.

### 11.8 Per-epoch diagnostic figure

One figure per selected validation sample, from a single forward pass, in seven rows:

1. the raw target FHR **in bpm** and the raw source UP **in mmHg**, on one time axis with one
   $y$-axis each — UP shares the row because rows 6 and 7 are statements *about* it, and a
   contraction has to be findable in the same column of the page as the response it is claimed to
   drive;
2. the forecast over the whole recording, tiled into **consecutive non-overlapping** windows: the
   true future against the base and full forecasts with their $\hat\mu \pm 2\hat\sigma$ bands and a
   thin dashed vertical at every window edge — the panel the model exists to produce;
3. $\mu^p_t$ over $\mu^q_t - \mu^p_t$ on one colour scale, so their relative size is visible (a
   delta as large as the state itself means the posterior is doing the prior's job);
4. the per-step per-dimension KL, where a collapse onto one or two dimensions shows up;
5. the total per-step $K_t$;
6. the head-averaged lag-attention matrix with its per-step argmax overlaid;
7. the lag-attributed KL $\widetilde K_{t,\ell}$.

**The tiling is a property of the figure and of nothing else.** The forward pass decodes *every*
valid anchor, at stride $1$, and §10's objective and every readout in §11 are computed over all
$240$ trained ones — the model, the training and the metrics are exactly as described above and
below. What the row does is choose which of those already-computed forecasts to *draw*: one anchor
forecasts $H \cdot R = 480$ raw samples, so the anchors whose windows abut without overlapping are
spaced exactly $H$ apart, and the row plots
$t \in \{w,\ w + H,\ w + 2H,\ \ldots\} \cap [w,\ T - H)$ — at the shipped geometry, $8$ windows of
$120$ s covering $124$ s to $1084$ s. Drawn at the model's own stride $1$ instead, adjacent windows
would overlap by $29/30$ and the panel would show each instant thirty times over, from thirty
latents, with nothing saying which curve was which. Two spans are consequently blank: everything
before the first trained anchor's own window, and the $116$ s tail that cannot be reached without
overlapping the last tiled window.

**Rows 3–7 are drawn over the trained anchors only.** The warm-up prefix $[0, w)$ is cut from all
five, and the tail $[T - H, T)$ from all but the attention. Those columns carry no gradient at all
— the tail is neither decoded nor inside the KL support — and while they stayed in the arrays they
set the colour scale, so a warm-up transient compressed the whole trained region into the bottom of
the colormap. This too is a drawing decision and nothing more: the tensors the forward pass returns
are untouched, and no number anywhere in this document is computed from the cut version. They are
removed from the *data of the panel* rather than shaded over it; every axes still spans the full
recording, so the rows stay column-aligned with rows 1 and 2, and the empty margins are marked in
grey.

Both lag panels carry a secondary axis in **compensated** seconds and say so in the label.

---

## 12. Training procedure

### 12.1 Data flow

Each training example is one $20$-minute trimmed segment. Batches carry the four feature blocks,
the raw z-scored target, the validity weight and the recording identifier. Four pre-flight
guards run *before* any run directory, log sink or experiment-tracking run is created, so a
doomed launch leaves nothing behind:

1. the normalisation statistics file exists — otherwise normalisation is silently disabled and
   the model trains on an unnormalised target while only emitting a warning;
2. the declared $c_y$ and $c_u$ match the first shard's channel counts;
3. the raw target appears in **both** the load list and the normalisation list — absent from the
   second, nothing fails at all: the target arrives at $\approx 140$ bpm while the decoder's
   log-variance models a $z$-scale, and the Gaussian NLL is meaningless;
4. the causal reach budget resolves to a non-empty channel set with a delay that fits inside the
   warm-up.

### 12.2 Optimiser and schedule

| Setting | Value |
| --- | --- |
| Optimiser | AdamW |
| Learning rate | $3 \times 10^{-4}$ |
| Betas | $(0.9,\ 0.95)$ |
| Epsilon | $10^{-8}$ |
| Weight decay | $10^{-4}$ |
| Scheduler | multi-step, $\gamma = 0.1$ at epochs $\{400,\ 800\}$ |
| Epochs | $5000$ |
| Batch size | $128$ per stage |
| Gradient accumulation | $1$ |
| Precision | `32-true` |
| Gradient clipping | global norm, threshold $5000$ |
| Seed | $42$ |

Adam's per-parameter normalisation makes the step size largely invariant to the loss scale, so
the learning rate transfers across objective magnitudes; the **clip threshold does not**. It began
scaled with the objective, at $250$, and has since been re-derived from the first production run's
logged pre-clip `grad_norm`: $q_{50} = 2775$, $q_{99} = 4681$, $q_{99.9} = 5866$, maximum $7313$,
minimum $703$ — **every** recorded step above $250$, so that run performed normalised-gradient
descent at roughly an eleventh of its configured learning rate. $5000$ is the smallest round value
above $q_{99}$, which leaves the top percent of steps clipping and the rest stepping at the
learning rate they were configured with.

The milestones are deliberately late relative to the $\beta$ ramp. The source pathway starts at
exactly zero KL and can only earn coupling once $\beta$ is on, which is epoch $50$. Milestones at
$\{400, 800\}$ leave $400$ epochs at full learning rate with $\beta$ at its endpoint; decaying
earlier would measure the coupling readout on a model that was effectively frozen before the
bottleneck could open.

### 12.3 Distributed training

Seven devices under distributed data-parallel, one process per device. The plain strategy —
which asserts that *every* parameter is marked ready in every backward pass — is selected
whenever the Gaussian likelihood is in use, because that likelihood consumes the decoder's
log-variance heads. Under the squared-error likelihood those heads starve, and the
unused-parameter-tolerant strategy is selected instead. The frozen attention projection is never
in the reducer's expectation set at all, because `requires_grad` was cleared at construction.

The framework injects the distributed sampler and calls its epoch hook; no sampler is hand-built.
Sanity validation steps are disabled, because a sanity pass would shift every epoch number
against the tracking store and the checkpoint filenames.

**Graph compilation is permanently off.** Three independent constructs defeat the compiler: the
LSTM encoders, the optionally checkpointed attention region, and the data-dependent boolean mask
indexing behind the active-dimension count. This is forced in the task's constructor rather than
defaulted.

### 12.4 Loss-spike circuit breaker

The watched quantity is the objective itself. Two independent tests can flag a batch as a spike,
plus a hard non-finite guard:

$$
\text{relative:}\quad \mathcal{L} > \mathrm{multiplier}\cdot\max(\mathrm{EMA},\ \mathrm{floor}),
\qquad
\text{additive:}\quad \mathcal{L} > \mathrm{EMA} + \mathrm{margin}.
$$

The **relative test is deliberately switched off** here, by setting the floor above any reachable
loss. Its form assumes a loss bounded below by zero; a learned-variance Gaussian NLL summed over
$480$ samples goes negative, and once the exponential moving average is negative the relative
test degenerates into "skip every positive batch" — a failure that presents as a run which trains
normally and then skips every batch forever. The **additive margin** carries finite-spike
detection instead, compared against the *raw* moving average so it survives the negative regime.

On a flagged batch the module performs a **zero-gradient step** rather than a true skip: a loss
formed from every trainable parameter times zero. The forward has already armed the distributed
gradient reducer, which expects a hook to fire for each parameter, so returning nothing would
desynchronise it. After a configured number of *consecutive* skips, and only if every rank
agrees, the next finite batch is force-accepted and the moving average is hard re-seeded to it —
an escape hatch against a frozen-average deadlock. The skip decision is reduced with a maximum
across ranks (skip if any rank skips) and the force-accept with a minimum (force only if all
agree), so optimiser steps stay in lockstep.

| Setting | Value |
| --- | --- |
| Watched metric | the objective |
| Multiplier | $5.0$ |
| EMA decay | $0.02$ |
| EMA floor | $10^{9}$ — relative test off |
| Additive margin | $10^{3}$ nats |
| Warm-up batches | $100$ |
| Max consecutive skips | $25$ |

### 12.5 Validation-time permutation control

On **validation batches only**, and never entering the objective, the batch is deranged so every
target is paired with a *different* recording's source, and the full forecast is rebuilt and
re-scored against the true raw future.

The rebuild is exact and cheap. The source pathway contains no batch-coupled operator, so
permuting the already-computed source state along the batch axis is exactly equivalent to
re-encoding a permuted source stream. Only the attention, the posterior and the full forecast are
recomputed; both encoders, the prior and the base forecast are reused untouched — which makes
"the base branch is bitwise identical under permutation" a checkable property rather than a hope.
The permuted latent is drawn with a *fresh* $\epsilon$, because the shuffled branch is scored on
its own rather than differenced sample-by-sample against the matched one.

The control emits `nll_shuffled_block`, `kld_shuffled` and `shuffle_penalty`. On steps where it
did not run its metrics are **absent, never zero-filled**: epoch aggregates are means over the
steps that reported, so a zero placeholder would scale the aggregate toward nothing and invert
the very ordering the control exists to check.

### 12.6 What a run records

* Per-epoch metrics for every quantity in §11, on both train and validation, written to CSV and
  to the tracking store.
* Checkpoints selected on validation objective, top three retained.
* The fully resolved configuration written beside the checkpoints, augmented with the resolved
  causal guard — so a checkpoint directory copied elsewhere carries everything needed to rebuild
  the run's data contract.
* Every checkpoint carries the model class name **and** the exact constructor kwargs, so the
  architecture is rebuildable with no configuration file, and a blob written by a different model
  is refused before any load is attempted.
* Peak device memory after the first training step — the honest high-water mark, covering
  weights, optimiser state and one full forward/backward at the configured batch.

---

## 13. Evaluation protocol

Training reports single-draw quantities. The offline evaluation is a separate pass with a
different estimator and an explicit acceptance gate.

### 13.1 Monte Carlo predictive likelihood

Every branch is scored with $S = 8$ latent draws under **common random numbers**: one $\epsilon$
per replicate, reused by *every* branch, so two branches with identical latent parameters produce
bitwise identical scores and the base-versus-full difference is a difference of predictions rather
than of noise. The estimator draws from an explicit generator rather than the global stream, so
two evaluations of one checkpoint report the same numbers regardless of what else in the process
drew first.

Per-anchor scores are marginalised in the space the quantity lives in. For a log-density,

$$
D = -\left[\operatorname*{logsumexp}_{s=1}^{S}\big(-d^{(s)}\big) - \log S\right]
= -\log\!\left(\frac{1}{S}\sum_{s=1}^{S} e^{-d^{(s)}}\right),
$$

the log of the *average likelihood*. By Jensen this is strictly smaller than the average of the
log scores whenever the draws disagree, and that gap is the entire point of marginalising rather
than averaging. Under the squared-error likelihood a block score is not a log-density and its
exponential means nothing, so the marginal is the plain mean over draws instead. At $S = 1$ the
estimator reduces exactly to the training-path per-anchor score.

### 13.2 Trivial baselines

A summed $480$-sample log-density is a large number under any predictor, so it is only readable
against predictors that know nothing:

| Baseline | Definition |
| --- | --- |
| **persistence** | hold the last *observed* raw sample forward across the block — "observed", because carrying a gap's $-11\sigma$ value forward would measure the gap, not persistence |
| **climatology** | the normalisation's own centre, $\mu = 0$ in $z$ units: the predictor that has seen the population and nothing else |
| **segment mean** | the mean of this segment's own observed samples — deliberately the stronger, **non-causal** form, since it reads the segment's whole future |

Every baseline is a *constant* over the anchor's block, which is what makes it trivial: it says
nothing about the shape of the next two minutes, only about its level. A point predictor has no
variance of its own, so each is handed a fixed observation log-variance of $0$ — that is
$\sigma = 1$ in the loader's $z$ units, matching the decoder's initialisation calibration (§8).
That choice is stated rather than fitted because under a Gaussian likelihood the whole skill score
would otherwise be decided by whatever $\sigma$ the baseline was given. For the same reason a
squared-error-space skill, in which $\sigma$ cancels, is reported **beside** the likelihood-space
one: a learned-variance model otherwise beats a fixed-variance baseline partly on variance
modelling alone, and a single number could not separate the two effects.

### 13.3 Acceptance verdicts

Eight criteria, each reported as `PASS`, `FAIL` or `INCONCLUSIVE`. A criterion that cannot be
evaluated is reported inconclusive, never omitted — a control that could not run and a control
that failed are different facts.

| Verdict | Criterion | Threshold |
| --- | --- | --- |
| `predictive_improvement` | $D_{\mathrm{full}} < D_{\mathrm{base}}$ | — |
| `source_margin_positive` | $D_{\mathrm{full}} < D_{\mathrm{shuffled}}$ | — |
| `source_specificity` | $D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}}$ | — |
| `prior_carries_target_state` | $D_{\mathrm{base}}(\text{shuffled } \mu^p) - D_{\mathrm{base}} \ge$ margin | $1.0$ nats/anchor |
| `latent_not_collapsed` | active latent dimensions $\ge$ minimum | $2$ dimensions at $\epsilon_{\mathrm{act}} = 10^{-2}$ |
| `prior_variance_not_pinned` | fraction of $\ell^p$ on either clamp $<$ maximum | $0.5$ |
| `decoder_variance_not_pinned` | fraction of $\hat\ell$ on either clamp $<$ maximum | $0.5$ |
| `calibration_near_nominal` | observed tail mass within a relative tolerance of nominal, at $1\sigma$, $2\sigma$, $3\sigma$ | factor $1.5$ either way |

Four of these deserve their reasoning stated.

**The middle three are read as a triple, and they are ordered by strength.** `source_specificity`
asks the full chain $D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}}$ and therefore
implies both of its neighbours. `source_margin_positive` drops the base branch entirely and asks
only whether the matched source beat a stranger's — the one comparison in which prior, decoder and
latent geometry are held fixed and *only the source changes*. So `predictive_improvement: FAIL`
beside `source_margin_positive: PASS` and `source_specificity: FAIL` is a real state rather than a
contradiction: the forecast did not improve, and the source pathway is still specific to this
recording. That is exactly what the first production run of this objective reported, at $-7.37$
nats of gain and $+1.06$ nats of margin, and the third verdict exists so a run in that state says
so instead of reading as a flat negative.

**The prior-shuffle criterion is the check that the model is what it claims to be.** It replaces
each recording's prior latent with a stranger's and re-scores the *base* forecast. If a stranger's
prior forecasts this recording as well as its own, the prior is not carrying the target's
predictive state — and then every readout built on the base-versus-full comparison is unsupported,
because the thing the source is supposedly adding to is not there. A degradation below the margin
but above zero is reported inconclusive: it costs something, but less than the stated margin, and
the measured number is what revises that margin.

**The variance criteria check that the numbers mean what they say.** A prior variance pinned on
its lower clamp multiplies every coupling readout by an arbitrary factor while every
decoder-side diagnostic stays healthy; a decoder variance pinned on either clamp means the
observation model has stopped being one. The $0.5$ maximum is deliberately permissive — it is not
a claim that half a distribution on its bound is healthy, it is the point past which the *readout*
built on that variance stops meaning what it says.

**The calibration criterion tests the tails, relatively.** The nominal central coverages are
$\operatorname{erf}(k/\sqrt{2})$ for $k = 1, 2, 3$ — that is $0.6827$, $0.9545$, $0.9973$, computed
rather than quoted, since the two-sigma figure people write as $0.95$ is a different number and
would make a calibrated model look half a point miscalibrated. The tolerance is on the **tail**
mass and is relative, because the three levels' tails span two orders of magnitude: an absolute
tolerance loose enough to admit sampling noise at one sigma would accept a fifty-fold error at
three. An observation model whose coverage misses these nominals is one whose NLL is not a log
density of anything, which makes every nats-per-anchor number in the run uninterpretable.

A structural safeguard applies to the whole set: the verdict list is validated against a registry
that refuses an unregistered name, a duplicate, or a *missing* entry. All three are the same
failure seen from different sides — a silent gap in the list reads as a criterion that passed.

**The specificity criterion reads three losses and no KL, and that is its content rather than a
simplification of it.** The posterior sees the source, so it reacts to *any* source — and a
stranger's source is out of distribution for a posterior trained only on matched pairs, which
routinely moves it *more*. A healthy model therefore has $K_{\mathrm{shuffled}} > K_{\mathrm{true}}$,
so a criterion phrased on the KL would fail exactly the models it should pass. The discriminating
comparison lives in prediction space: a wrong source must be worse than no source at all.

Additional controls available beyond the derangement: circularly shifted source, time-reversed
source, and lag-band exclusions.

### 13.4 Collapse criterion

A completed run is **collapsed** when either

1. `val/source_conditioned_kl_raw` is below $0.02$ nats per anchor at every one of its final
   $5$ epochs, or
2. its final `kld_active_frac` is below $2 / d_z$.

Both clauses say the same thing at the same per-dimension activity epsilon
($\epsilon_{\mathrm{act}} = 10^{-2}$, and $2 \times 10^{-2} = 0.02$): the latent finished
carrying less than two dimensions' worth of source information. The criterion reads the **tail**
of the run only, never an early window — the KL starts at exactly zero by construction and the
$\beta$ warm-up holds it there deliberately, so an any-window reading would classify every
healthy run as collapsed.

### 13.5 Analyses

Twelve analyses run off a single shared collection pass, so no analysis touches the model and any
one of them can be re-run against a finished run directory with no checkpoint and no accelerator:
forecast quality, coupling, permutation control, latent health, lag-resolved KL, attention
structure, calibration, residual structure, trajectory over time, band partition, time to
delivery, and cross-subgroup comparison.

Because forecast windows overlap heavily between adjacent anchors, statistical uncertainty is
estimated by bootstrapping **recordings**, never by treating every anchor as an independent
sample.

---

## 14. Causal-reach control of the inputs

### 14.1 The problem

Forecasting the future does not by itself make a model causal — its *inputs* must be causal too.
The stored features are two-sided wavelet transforms: the value at decimated step $t$ is a
weighted average over raw samples on **both** sides of $t$. A model conditioning on "the past up
to $t$" is therefore partly conditioning on the interval it is being asked to forecast, and the
causal normalisers inside the encoders cannot remove information that has already entered
through the inputs.

The required invariant, if it held, would be

$$
\frac{\partial X_t}{\partial Y[j]} = 0 \quad \forall\, j > n_{\mathrm{raw}}(t),
$$

and it does not hold for the stored transforms.

### 14.2 The measure and the guard

Per channel, the size of the violation is the **forward reach** $L_{95}$: the smallest $D > 0$
enclosing $95\%$ of the filter's energy at taps strictly after $t$. It is computed analytically
from the production filter bank — a scattering channel's reach is its wavelet's; a
phase-harmonic channel multiplies two wavelet responses and then low-pass smooths the product, so
its reach is the slower wavelet's plus the low-pass reach.

Given a budget $\mathcal{B}$ in seconds, channel $c$ **survives** when

$$
\mathrm{reach}_c \le \mathcal{B}
$$

(inclusively — the boundary is load-bearing, since a channel can sit exactly at a round budget),
and each survivor is read $\delta_c$ steps late with

$$
\delta_c = \left\lceil \frac{\mathrm{reach}_c}{\Delta} \right\rceil,
$$

which is the smallest delay for which the channel's forward reach no longer crosses the anchor's
causal endpoint:

$$
\underbrace{\Delta\,(t - \delta_c + 1) - \Delta}_{\text{end of the delayed step}} + \mathrm{reach}_c
\;\le\; \underbrace{\Delta\,(t+1) - \Delta}_{n_{\mathrm{raw}}(t)}
\iff
\mathrm{reach}_c \le \Delta\,\delta_c .
$$

Delays are **per channel** rather than one uniform guard band, because the two dominate very
differently at equal guarantee: at a $120$ s budget the fastest survivors are one step stale,
where a uniform band would make every channel $30$ steps stale.

$\max_c \delta_c$ must fit inside the loss warm-up $w$, because the first $\max_c \delta_c$ steps
of a delayed stream are partly zero-filled and must fall inside the steps the loss already
discards. This is checked at resolution time and raises with both numbers named.

### 14.3 The budget table

| Budget | Target channels $c_y$ | Source channels $c_u$ | $\max_c \delta_c$ |
| --- | ---: | ---: | ---: |
| none (default) | 109 | 58 | 0 |
| 240 s | 94 | 43 | 57 |
| 120 s | 78 | 29 | 30 |
| 60 s | 59 | 23 | 14 |
| 32 s | 43 | 19 | 8 |

Below $100$ s the source phase-harmonic block disappears entirely: its *fastest* channel already
reaches exactly $100.0$ s, so no member of the block clears a shorter budget. (The block's
reaches run $100.0$ s to $266.8$ s; $100.0$ s is its minimum, not its maximum.)

Two rows of that table need reading alongside §14.2's warm-up constraint. At $120$ s the maximum
delay is exactly $30$, which is also the shipped warm-up — the constraint is checked as *strictly
greater than*, so this configuration is admissible by design rather than by luck. At $240$ s the
maximum delay is $57$, which **exceeds** the shipped warm-up, so that budget is only reachable
with the warm-up raised to at least $57$; it is the one budget that cannot be selected by changing
a single key.

### 14.4 Engineering consequences

* The model is **built at the full declared widths** and gathers survivors *inside* the forward,
  after the data boundary. Otherwise the batch-width checks would reject any nonzero budget.
* The keep-index and the delay vector are held in **one object**, because the delay vector is
  positional *against* the keep-index: anything holding one must hold the other.
* Both buffers are **non-persistent**. Their length is the surviving-channel count, so a
  persistent copy would make a checkpoint trained at one budget fail to load at another —
  surfacing as misaligned keys rather than as anything about budgets. The four resolved index
  tuples do live in the checkpoint's constructor kwargs, because the adapters' widths depend on
  them.
* Arms at different budgets build different adapter widths and therefore **cannot share
  checkpoints**.
* The unguarded default builds **no gather and no delay at all**.

### 14.5 What the guard does and does not achieve

$L_{95}$ is an energy *quantile*, not a support: $5\%$ of every filter's energy lies beyond its
stated reach. Measured against a severe perturbation, the $120$ s budget suppresses the movement
of the read features by roughly a factor of $20$ rather than to numerical noise. The residual is
larger on the phase-harmonic block than on the scattering block, consistent with a phase
coefficient normalising by its own envelope and so amplifying exactly the low-energy tail the
quantile discounts.

**The guard bounds the leak; it does not remove it.** Only genuinely causal transforms would.

---

## 15. Assumptions and limits

Stated plainly, because each one bounds what the numbers may be read as.

**The KL is a source-conditioned rate, not a transfer entropy.** The name used throughout is
`source_conditioned_kl`, and the lag decomposition is `source_kl_lag_map`. Reading $K_t$ as a
transfer entropy would require, at minimum: strictly causal inputs (§14 — not currently
achieved at the default budget), a bottleneck at its conditional minimum-necessary-information
point, a decoder that accurately represents the future conditional distribution, and treatment of
confounding. The label would assert a property the input construction does not have.

**A factorised likelihood.** The decoder's Gaussian is factorised over the $480$ future raw
samples, so it ignores conditional correlation among them. Its predictive log-score difference
therefore estimates a directed information quantity *within that model family*, not the
data-generating one. Upgrading the covariance (low-rank plus diagonal, banded Cholesky, or a
conditional flow over residuals) is the natural next step once calibration diagnostics show the
factorised form is the binding error.

**The three shape weights are provisional single points, and the objective is mixed-unit.** No run
has measured $\mathcal{L}_{\mathrm{ms}}$, $\mathcal{L}_{\mathrm{deriv}}$ or
$\mathcal{L}_{\mathrm{boundary}}$ against the reconstruction at this scale, so $0.1 / 0.1 / 0.05$ are
the smallest weights that make the terms visible without letting them dominate rather than a derived
balance. Three consequences bound what may be read: `total_loss` is not in nats and its *level* is
not comparable across arms with different weights (§10.7); an arm that turns a term off loses that
term's readout by design, because a zero-weighted term is not computed; and the boundary term ships
in its **level** form only, without the slope variant, so a forecast whose opening level is right and
whose opening slope is wrong is not penalised for it. Re-derive all three from the first production
run's `aux_*` columns against `main_loss`. `MS_RATES = (1, 4, 16)` is likewise a module constant
rather than a config key, because no arm sweeps pooling rates.

**Context sufficiency is assumed, not shown.** The quantity one would like to condition on is the
full target history $Y^-_t$; the model conditions on $z_t$. These agree only when $z_t$ is
predictively sufficient, i.e. $X^+_t \perp Y^-_t \mid z_t$. Measuring the gap requires an
evaluation-only oracle decoder conditioned on the full target history, compared against the
latent-conditioned decoder:

$$
\Delta_{\mathrm{suff}} = -\log p\big(X^+_t \mid z_t\big) + \log p\big(X^+_t \mid H^y_t\big).
$$

That probe is not currently run, so the sufficiency assumption stands as an assumption.

**Directed predictive information is not interventional causality.** Hidden common causes,
unobserved clinical interventions and higher-order interactions can all produce directed
predictive information without establishing intervention-level causality.

**Attention is attribution, not proof of mechanism.** A peak at lag $\ell$ says the model found
that lag informative. Establishing that a physical effect occurred at that delay requires
synthetic known-lag data, lag-band masking, leave-one-band-out prediction changes, impulse
alignment and source circular shifts.

**With per-channel delays there is no single $\delta$.** The reported lag uses
$\max_c \delta_c$, which makes it an **upper bound**; the fact is recorded in the run summary
beside the number it produced.

**Every valid anchor is decoded in every batch.** No anchor subsampling is implemented, so
activation memory scales with $T_{\mathrm{valid}}$; the documented memory levers are reduced
precision, attention gradient checkpointing, and a smaller batch with gradient accumulation, in
that order.

**The prior anchor is the scale half of the context rate, not the whole term.** The full rate
$\mathrm{KL}(p_\theta \Vert \mathcal{N}(0, I))$ separates exactly into
$R_p + \tfrac12\sum_d (\mu^p_{t,d})^2$ (§10.5), and only the first half is weighted. The second
compresses the prior *mean*, which is the entire content of the base forecast: adopting it trades
$D_0$ away and turns a scale pathology into a target-side rate–distortion question — how much of
the target's own predictive state the latent is permitted to carry — which this model does not
attempt to answer. The half that ships is also the half the failure demanded; the collapse was in
$\sigma_p$, at a mean nothing complained about. The scale half is a strict subset of the full rate,
so adopting the rest later is an addition rather than a revision.

**Both run-measured constants ship provisional at this capacity.** The spike breaker's
additive margin was never measured at all — it is scaled, and is re-derived from an observed
`main_loss` fluctuation distribution; the first production run bounds it as loose-but-working, with
zero skips and a maximum loss-to-EMA distance of $669$ nats against its $1000$ nat margin, and it
now watches a **mixed-unit** `main_loss` (§10.7) rather than a pure-nats one. The gradient clip
**was** re-derived, from that run's logged pre-clip `grad_norm` (§12.2) — but that run predates the
wider latent, the wider and deeper decoder, its horizon attention and the three shape terms, each of
which moves the gradient the threshold is set against, so both values ship unchanged and marked
`PROVISIONAL AT THIS CAPACITY` in `configs/default.yaml` and are re-derived from the first run at
this geometry. Carrying a measured value across a scale change is a known quantity; rescaling it by
a guess is not. The decoder
log-variance clamp $[-5, 3]$ remains inherited and is re-derived from the logged floor/ceiling
fractions — it shares one constant with the latent heads today, which the measurements now pull in
opposite directions: the prior sat on its floor for $99.2\%$ of coordinates while the decoder sat on
its for $22.5\%$ and never reached its ceiling.

---

## 16. Configuration reference

The complete shipped configuration, grouped by what it controls.

### Geometry and widths

| Key | Value | Meaning |
| --- | --- | --- |
| `sequence_length` | 300 | $T$, decimated steps per segment |
| `raw_per_step` | 16 | $R = D$, raw samples per step |
| `horizon` | 30 | $H$, forecast horizon in steps ($= 120$ s) |
| `warmup_period` | 30 | $w$, steps excluded from every loss |
| `d_model` | 128 | backbone width |
| `d_z` | 64 | latent width (sweep axis: 24 / 32 / 48 / 64 / 96) |
| `c_y` | 109 | target channels ($43 + 66$) |
| `c_u` | 58 | source channels ($43 + 15$) |
| `use_up_st` | true | include the source scattering block |

### Encoders

| Key | Value |
| --- | --- |
| `lstm_layers` | 2 |
| `dropout` | 0.1 |
| `encoder_extra_dilations` | $[8, 16]$ |
| `encoder_extra_kernel` | 15 |
| `conv_norm_groups` | null (each block's $\min(8, C)$) |
| `causal_norm` | **true** — a correctness requirement |

### Latent heads

| Key | Value |
| --- | --- |
| `logvar_clamp` | $[-5, 3]$ |
| `mu_scale` | 5.0 |
| `delta_mu_scale` | 3.0 |
| `delta_logvar_scale` | 2.0 |

### Attention

| Key | Value |
| --- | --- |
| `max_lag` | 90 → $L = 91$ ($\approx 6$ min) |
| `num_heads` | 4 |
| `d_head` | 32 |
| `use_entmax` | true — exact zeros in the lag distribution |
| `lag_bias_init` | `alibi_decay` |
| `query_uses_logvar` | false |
| `attention_grad_checkpoint` | false |

### Decoder

| Key | Value |
| --- | --- |
| `decoder_hidden` | 256 |
| `horizon_depth` | 4 |
| `horizon_kernel` | 3 |
| `horizon_film` | true |
| `horizon_attention_blocks` | 2 — heads are the core's own 4, and have no key |

### Objective

| Key | Value |
| --- | --- |
| `likelihood` | `gaussian_nll` |
| `lambda_full` | 1.0 |
| `lambda_base` | 1.0 |
| `beta_schedule` | linear warm-up, $0 \to 1$ over 50 epochs |
| `beta_prior` | 0.1 — a constant, never a schedule (§10.7) |
| `lambda_ms` | 0.1 — PROVISIONAL; multiscale L1 on the forecast mean |
| `lambda_deriv` | 0.1 — PROVISIONAL; derivative Huber |
| `lambda_boundary` | 0.05 — PROVISIONAL; boundary continuity, half the others (one sample against 480) |
| `free_bits` | 0.0 |
| `coverage_floor` | 0.9 |

### Trainer

| Key | Value |
| --- | --- |
| `gradient_clip_val` | 5000.0 — re-derived from the first production run's pre-clip `grad_norm` (§12.2) |

### Initialisation policies

| Key | Value | No-op default |
| --- | --- | --- |
| `horizon_embed_std` | 0.8 | 0.02 |
| `head_init_calibration` | true | false |
| `a_head_gain` | 2.0 | 1.0 |

### Causality

| Key | Value |
| --- | --- |
| `causal_reach_budget_s` | null (unguarded; sweep axis 240 / 120 / 60 / 32) |

### Keys deliberately absent

No configuration key exists for the observation noise, the head-structured latent, the frozen
attention projection, the plain convolution stack, per-block FiLM, or the plain residual seams.
All six are **structural facts** of this architecture; a key would read to a maintainer as a
control that exists.

---

## 17. File map and commands

### Layout

```
nets/                  framework-free torch: no config, no I/O, no logging
  model.py             the network, its KL, and compute_loss
  heads.py             the full-latent target-only prior head
  losses.py            objective terms in nats per anchor
  geometry.py          the validated anchor-to-raw index map
  raw_targets.py       future-target index grid and gather
  raw_masks.py         forecast mask, contributing anchors, KL support
  delays.py            per-channel gather + delay (the causal input guard)
  controls.py          the source-permutation control
  lag_report.py        lag index -> compensated / sensor seconds
task.py                the training task: loss, metrics, permutation control
trainer.py             the experiment driver: config -> model -> fit
channel_reach.py       forward reach per channel; budget resolution
collapse.py            the numeric collapse criterion (no torch dependency)
plotting.py            the per-epoch diagnostic figure
configs/               default.yaml + one file per sweep arm
eval/                  the offline evaluation pipeline and its analyses
tests/                 the invariants above, each pinned
```

The network layer is enforced framework-free by test: nothing under `nets/` may import the
training framework, the configuration layer, or the evaluation package.

### Commands

Training, single device (smoke):

```bash
python -m teb_vae.lag_attn_rws.trainer --config teb_vae/lag_attn_rws/configs/tiny.yaml
```

Training, production (rank count must equal the configured device count):

```bash
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_rws.trainer --config teb_vae/lag_attn_rws/configs/default.yaml
```

Derived tables, printed from the code that produces them:

```bash
python -m teb_vae.lag_attn_rws.nets.geometry     # the anchor-to-raw index table
python -m teb_vae.lag_attn.channel_reach     # per-channel reach and the budget table
```

Test gate:

```bash
python -m pytest teb_vae/lag_attn_rws/tests -q -m "not slow"
python -m pytest teb_vae/lag_attn_rws/tests -q -m slow
```
