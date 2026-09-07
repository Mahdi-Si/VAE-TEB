# Raw-Domain Composite Predictive Transfer-Entropy Bottleneck for FHR–UP

## A complete architecture and implementation roadmap

The recommended new model is **not merely the current model with its feature decoder replaced by a raw-signal decoder**. The latent-variable factorization should also change.

The current model is internally consistent with its original scientific purpose:

* `decoder_state` carries the target-only FHR representation.
* The target-only forecast is produced from `decoder_state`.
* The source-conditioned latent ($z_t$) produces only an incremental forecast correction.
* The KL divergence measures how much UP moves the source-conditioned latent away from the target-only prior.

That is why the current $z_t$ is fundamentally a representation of **additional source information**, not a complete representation of FHR. This is also close to the original TEB formulation: TEB defines its latent as a compressed representation of the source history, conditional on the target history, while the target history or a pretrained target context remains available separately to the decoder. ([arXiv][1])

For the new model, the latent should be divided explicitly into two blocks:

$$
\boxed{
Z_t=
\begin{bmatrix}
C_t \\
S_t
\end{bmatrix}
}
$$

where:

* $C_t$ is a **target-only predictive FHR state**.
* $S_t$ is the **minimum necessary source-conditioned UP information**.
* $Z_t$ is the complete exported representation.

The decoder must receive only $Z_t$, with no high-dimensional `decoder_state` bypass.

The architecture should produce two versions of the complete latent:

$$
Z_t^{0}=
\begin{bmatrix}
C_t \\
S_t^{0}
\end{bmatrix},
\qquad
Z_t^{1}=
\begin{bmatrix}
C_t \\
S_t^{1}
\end{bmatrix},
$$

where:

* $S_t^0$ is drawn from a target-only conditional source prior.
* $S_t^1$ is drawn from the UP-conditioned source distribution.

Therefore,

$$
\boxed{
Z_t^1=
Z_t^0+
\begin{bmatrix}
0 \\
S_t^1-S_t^0
\end{bmatrix}
}
$$

This is the required architectural decomposition:

$$
\boxed{
\text{complete FHR representation}
+
\text{additive minimum necessary UP update}.
}
$$

The split is no longer an evaluation convention. It is encoded directly in the coordinates of the latent vector.

---

# 1. What the new model should represent

At every anchor time $t$, the model should learn three related quantities.

## 1.1 Target-only FHR state

$$
C_t \sim e_C\!\left(c_t \mid Y_{\le t}\right).
$$

$C_t$ should summarize the FHR history in a form that is:

* Sufficient for predicting the future FHR.
* Compressed enough not to memorize irrelevant historical noise.
* Useful as a downstream representation.
* Independent of UP by construction.

Conceptually,

$$
C_t \approx \text{minimum predictive state of the FHR process at time } t.
$$

## 1.2 Target-only source-null state

$$
S_t^0 \sim r_S\!\left(s_t \mid C_t\right).
$$

This is the distribution of the source block when the model knows the FHR context but does not receive UP.

It plays the role of the target-conditioned TEB prior.

## 1.3 Source-conditioned state

$$
S_t^1 \sim q_S\!\left(s_t \mid C_t, U_{\le t}\right).
$$

This is the source block after the model has observed the lagged UP history.

The source-derived additive update is

$$
\Delta S_t^{U \rightarrow Y} = S_t^1 - S_t^0.
$$

The complete source-conditioned representation is therefore

$$
Z_t^{\mathrm{full}} = \begin{bmatrix} C_t \\ S_t^1 \end{bmatrix}.
$$

For deterministic downstream representation learning, use the distribution means:

$$
\bar Z_t^{\mathrm{full}} = \begin{bmatrix} \mu_t^C \\ \mu_t^{S,1} \end{bmatrix},
$$

$$
\bar Z_t^{\mathrm{base}} = \begin{bmatrix} \mu_t^C \\ \mu_t^{S,0} \end{bmatrix},
$$

and

$$
\Delta\bar Z_t^{U \rightarrow Y} = \begin{bmatrix} 0 \\ \mu_t^{S,1}-\mu_t^{S,0} \end{bmatrix}.
$$

This gives three distinct representations for analysis:

| Representation | Meaning |
| --- | --- |
| $\mu_t^C$ | FHR-only predictive state |
| $\Delta\bar Z_t^{U \rightarrow Y}$ | additional UP-derived predictive information |
| $\bar Z_t^{\mathrm{full}}$ | complete FHR state after incorporating UP |

---

# 2. Raw-domain temporal definition

Let:

$$
Y[n] \in \mathbb{R}
$$

be the raw FHR signal, and

$$
U[n] \in \mathbb{R}
$$

be the raw UP signal.

Assume:

$$
f_s = 4\ \mathrm{Hz}.
$$

Let the latent be produced every $D=16$ raw samples:

$$
D = 16, \qquad \Delta t = \frac{16}{4} = 4\ \mathrm{s}.
$$

The raw index corresponding to latent step $t$ should be stored explicitly:

$$
n_t = n_{\mathrm{origin}} + Dt.
$$

Here $n_t$ is the last raw sample that the model is allowed to observe when constructing $Z_t$.

The dataset should store an `anchor_raw_index[t]` tensor rather than reconstructing this mapping implicitly. This prevents silent off-by-one errors from trimming, padding, or shifts.

## 2.1 History variables

For target-history length $L_Y$,

$$
Y_t^- = Y[n_t-L_Y+1 : n_t].
$$

For source-history length $L_U$,

$$
U_t^- = U[n_t-L_U+1 : n_t].
$$

The recurrent target encoder may summarize the entire available prefix rather than a hard fixed window, but the finite-window notation is useful for defining transfer entropy.

## 2.2 Raw future

For a two-minute prediction horizon,

$$
H = 120 \cdot 4 = 480.
$$

Define

$$
Y_t^+ = Y[n_t+1 : n_t+H].
$$

Therefore,

$$
Y_t^+ \in \mathbb{R}^{480}.
$$

For a $2.5$-minute future,

$$
H = 150 \cdot 4 = 600.
$$

For a three-minute future,

$$
H = 180 \cdot 4 = 720.
$$

The first implementation should retain the two-minute horizon:

$$
\boxed{H = 480}
$$

because it provides a direct comparison with the current two-minute feature-domain model.

The current decoder predicts $30 \times 109 = 3270$ feature coefficients per anchor, while the raw two-minute future contains only $480$ samples. Predicting raw FHR therefore removes substantial output redundancy.

## 2.3 Optional recent-past reconstruction

For a $2.5$-minute past reconstruction objective,

$$
P = 150 \cdot 4 = 600
$$

and

$$
Y_t^{\mathrm{rec}} = Y[n_t-P+1 : n_t].
$$

This past target must be reconstructed only from the target latent $C_t$, not from the UP source block.

---

# 3. Transfer entropy in this model

For an $H$-sample future block, define the block transfer entropy as

$$
\boxed{
\operatorname{TE}_{U \rightarrow Y}^{(H)} = I\!\left(U_t^-; Y_t^+ \mid Y_t^-\right).
}
$$

Equivalently,

$$
\operatorname{TE}_{U \rightarrow Y}^{(H)} = \mathbb{E}\left[\log \frac{p^\star(Y_t^+ \mid Y_t^-, U_t^-)}{p^\star(Y_t^+ \mid Y_t^-)}\right].
$$

Here $p^\star$ denotes the true, generally unknown data distribution.

This is a directed predictive-information quantity: source history contributes information about the future target after target history has already been accounted for.

It should not automatically be interpreted as an interventional causal effect. Hidden common causes, unobserved clinical interventions, and higher-order interactions can produce directed predictive information without establishing intervention-level causality. Transfer entropy should therefore be described as directed predictive information unless stronger causal assumptions are explicitly justified. ([arXiv][2])

---

# 4. The information-theoretic structure

The proposed model combines two bottlenecks:

1. A predictive FHR bottleneck for $C_t$.
2. A conditional source bottleneck for $S_t$.

This is the central theoretical change.

## 4.1 Target predictive bottleneck

The FHR latent should retain information from the FHR past that is useful for predicting the FHR future, while discarding historical details that are irrelevant to that future.

The desired minimum-necessary-information relation is approximately

$$
I(Y_t^-; Y_t^+) \approx I(Y_t^-; C_t) \approx I(C_t; Y_t^+).
$$

The Conditional Entropy Bottleneck formalizes this through

$$
\mathcal{J}_C = I(Y_t^-; C_t \mid Y_t^+) - \gamma_C I(C_t; Y_t^+).
$$

The first term removes information in $C_t$ about the past that cannot be explained by the future. The second term rewards future predictiveness.

This is more appropriate than simply placing a standard Gaussian prior on $C_t$, because the target of compression is specifically the information in the past that is unnecessary for the future. CEB introduces a backward encoder from the prediction target to the latent and minimizes a forward-versus-backward latent divergence. Predictive information bottleneck formulations similarly define useful representations as compressed summaries of observations that retain information about future data. ([arXiv][3])

### Variational implementation

Define a target-history encoder

$$
e_C(c_t \mid Y_t^-) = \mathcal{N}\left(\mu_t^C, \operatorname{diag}\left((\sigma_t^C)^2\right)\right).
$$

Define a training-only future encoder

$$
b_C(c_t \mid Y_t^+) = \mathcal{N}\left(\mu_t^F, \operatorname{diag}\left((\sigma_t^F)^2\right)\right).
$$

Then use

$$
R_C(t) = D_{\mathrm{KL}}\left(e_C(c_t \mid Y_t^-) \,\|\, b_C(c_t \mid Y_t^+)\right).
$$

The future encoder $b_C$ is used only during training. It is not used during inference and cannot leak future information into the deployed representation.

The practical target bottleneck objective is

$$
\mathcal{L}_C = \beta_C R_C + \lambda_0 D_0,
$$

where $D_0$ is the target-only raw future negative log-likelihood.

## 4.2 Conditional source bottleneck

Once $C_t$ represents the predictive state of FHR, the source bottleneck asks:

> How much additional information from UP is required to predict the FHR future beyond what is already contained in $C_t$?

Define a target-only conditional prior:

$$
r_S(s_t \mid C_t) = \mathcal{N}\left(\mu_t^{S,0}, \operatorname{diag}\left((\sigma_t^{S,0})^2\right)\right).
$$

Define a UP-conditioned distribution:

$$
q_S(s_t \mid C_t, U_t^-) = \mathcal{N}\left(\mu_t^{S,1}, \operatorname{diag}\left((\sigma_t^{S,1})^2\right)\right).
$$

The source rate is

$$
\boxed{
R_U(t) = D_{\mathrm{KL}}\left(q_S(s_t \mid C_t, U_t^-) \,\|\, r_S(s_t \mid C_t)\right).
}
$$

The corresponding conditional bottleneck objective is

$$
\mathcal{J}_U = I(U_t^-; S_t \mid C_t) - \gamma_U I(S_t; Y_t^+ \mid C_t).
$$

At the ideal conditional minimum-necessary-information point,

$$
I(U_t^-; Y_t^+ \mid C_t) = I(U_t^-; S_t \mid C_t) = I(S_t; Y_t^+ \mid C_t).
$$

This is the conditional MNI principle targeted by TEB. TEB uses a target-conditioned latent prior because a fixed prior would penalize target-explainable information instead of isolating source information. At convergence, its KL upper bound approximates the conditional source information, and at the ideal CMNI point it can coincide with transfer entropy. ([arXiv][1])

The proposed architecture extends this idea: TEB's source latent is retained as $S_t$, while the target context becomes an explicit stochastic representation $C_t$ and both blocks are included in the exported latent.

## 4.3 Why the complete latent KL isolates the source block

Define the target-only complete latent distribution:

$$
P_0(c,s \mid Y_t^-) = e_C(c \mid Y_t^-)\, r_S(s \mid c).
$$

Define the source-conditioned complete latent distribution:

$$
P_1(c,s \mid Y_t^-, U_t^-) = e_C(c \mid Y_t^-)\, q_S(s \mid c, U_t^-).
$$

Then

$$
\begin{aligned}
&D_{\mathrm{KL}}\left(P_1(C,S \mid Y_t^-,U_t^-) \,\|\, P_0(C,S \mid Y_t^-)\right) \\
&= \mathbb{E}_{C \sim e_C}\left[D_{\mathrm{KL}}\left(q_S(S \mid C,U_t^-) \,\|\, r_S(S \mid C)\right)\right].
\end{aligned}
$$

The target-context distribution $e_C$ cancels exactly.

Therefore,

$$
\boxed{
D_{\mathrm{KL}}(P_1 \,\|\, P_0) = R_U.
}
$$

This is the key property:

* The complete latent contains FHR information.
* The KL between full and baseline complete latents charges only the source block.
* FHR information in $C_t$ is not counted as transfer entropy.

## 4.4 The context-sufficiency requirement

The desired transfer entropy is conditioned on the full FHR history:

$$
I(U_t^-; Y_t^+ \mid Y_t^-).
$$

The bottleneck model conditions the source code on $C_t$:

$$
I(U_t^-; Y_t^+ \mid C_t).
$$

These are approximately equal only when $C_t$ is predictively sufficient for the FHR history:

$$
Y_t^+ \perp Y_t^- \mid C_t,
$$

or equivalently,

$$
I(C_t; Y_t^+) \approx I(Y_t^-; Y_t^+).
$$

This assumption must be validated empirically. It cannot simply be declared true.

A practical sufficiency diagnostic is to compare:

$$
D_C = -\log p(Y_t^+ \mid C_t)
$$

against an oracle target-history decoder:

$$
D_H = -\log p(Y_t^+ \mid H_t^Y).
$$

Define

$$
\Delta_{\mathrm{suff}} = D_C - D_H.
$$

A small $\Delta_{\mathrm{suff}}$ indicates that compressing $H_t^Y$ into $C_t$ did not discard substantial future-predictive information.

The oracle decoder is an evaluation-only probe. It must not become a production decoder bypass.

---

# 5. Exact architectural invariants

The model should enforce the following properties structurally.

## 5.1 Source purity

$$
C_t = f(Y_{\le t})
$$

and never receives UP.

$$
H_t^U = g(U_{\le t})
$$

and never receives FHR.

Cross-channel phase-harmonic coefficients must not enter the source encoder because they already combine FHR and UP.

## 5.2 No decoder bypass

The future decoder may receive only

$$
Z_t = [C_t, S_t]
$$

and learned horizon embeddings.

It may not receive:

* $H_t^Y$.
* `decoder_state`.
* Raw FHR history.
* Scattering or phase-harmonic inputs.
* Attention summaries outside the source latent.
* A separate target-only hidden state.

This is what forces the latent to represent the signal.

## 5.3 Shared decoder

The baseline and full forecasts use the exact same decoder parameters:

$$
p_0(Y_t^+ \mid Y_t^-) = \mathbb{E}_{C,S^0}\, d_\theta(Y_t^+ \mid C, S^0),
$$

$$
p_1(Y_t^+ \mid Y_t^-, U_t^-) = \mathbb{E}_{C,S^1}\, d_\theta(Y_t^+ \mid C, S^1).
$$

There are not two separately parameterized decoders.

The only difference is whether the source block is sampled from $r_S$ or $q_S$.

## 5.4 Past reconstruction is target-only

$$
\widehat Y_t^{\mathrm{rec}} = D_{\mathrm{past}}(C_t).
$$

The source block $S_t$ must receive no gradient from past reconstruction.

## 5.5 Exact source-null initialization

At initialization,

$$
q_S(s_t \mid C_t, U_t^-) = r_S(s_t \mid C_t).
$$

Therefore,

$$
R_U(t) = 0
$$

and, when paired sampling is used,

$$
Z_t^1 = Z_t^0,
$$

so

$$
\widehat Y_{t,\mathrm{full}}^+ = \widehat Y_{t,\mathrm{base}}^+
$$

exactly.

This retains one of the strongest properties of the current implementation while making the full latent informative.

---

# 6. Strict causal input construction

Predicting raw FHR does not automatically make the model causal. Inputs to the latent encoder must also be causal.

The current scattering and phase-harmonic implementation uses centered Morlet filters, centered Gaussian averaging, full-segment filtering, reflection padding, and decimation. Consequently, some coefficients at time $t$ depend on raw samples after $t$. Neural `causal_norm` removes future mixing inside the encoder but cannot remove future information that has already entered through its inputs.

The required invariant is

$$
\boxed{
\frac{\partial X_t}{\partial Y[j]} = 0 \quad \forall j > n_t
}
$$

and similarly for UP.

This must hold for:

* Filtering.
* Decimation.
* Normalization.
* Missing-value repair.
* Artifact removal.
* Scattering.
* Phase harmonics.
* Raw neural encoders.
* Signal alignment and shifting.

## 6.1 Recommended implementation order

The safest development order is:

1. Build the first model using raw inputs only.
2. Validate strict causality and TE behavior.
3. Add causal scattering.
4. Add causal phase harmonics.
5. Compare all additions through controlled ablations.

This separates latent-model correctness from transform correctness.

## 6.2 Causal raw downsampling stem

Each signal should have its own raw stem:

$$
Y \longrightarrow E_{\mathrm{raw}}^Y \longrightarrow R_t^Y,
$$

$$
U \longrightarrow E_{\mathrm{raw}}^U \longrightarrow R_t^U.
$$

Use four anti-aliased causal downsampling stages:

$$
4\ \mathrm{Hz} \rightarrow 2\ \mathrm{Hz} \rightarrow 1\ \mathrm{Hz} \rightarrow 0.5\ \mathrm{Hz} \rightarrow 0.25\ \mathrm{Hz}.
$$

A generic causal downsampling block can be written as

$$
\widetilde x^{(\ell)}[n] = \sum_{r=0}^{K_\ell-1} g_\ell[r]\, x^{(\ell)}[n-r],
$$

followed by

$$
x^{(\ell+1)}[k] = \rho\left(W_\ell \widetilde x^{(\ell)}[2k] + b_\ell\right).
$$

Here:

* $g_\ell$ is a causal low-pass filter.
* Only indices $n-r \le n$ are used.
* $W_\ell$ performs channel mixing.
* $\rho$ is GELU, SiLU, or a gated activation.
* A residual path may be added.

A practical block is:

1. Left-only padding.
2. Fixed or learnable causal anti-alias FIR.
3. Stride-two convolution.
4. Per-time-step LayerNorm or RMSNorm.
5. GELU.
6. Pointwise projection.
7. Residual connection.

Do not use a normalization operator that pools statistics over the full time axis.

### Suggested raw stem widths

$$
(1+\text{mask channels}) \rightarrow 16 \rightarrow 24 \rightarrow 32 \rightarrow 64.
$$

The output is

$$
R^Y, R^U \in \mathbb{R}^{B \times T \times 64}.
$$

## 6.3 Missing-data and quality channels

Each raw signal should be accompanied by:

* A validity mask.
* An artifact mask.
* Time since last valid sample, where useful.
* Possibly a quality score.

Imputation must be causal.

Valid choices include:

* Forward fill.
* Learned causal imputation.
* Zero fill after normalization plus a mask.
* A running past-only statistic.

Do not interpolate a missing point using a future valid sample.

Do not normalize each complete recording using statistics computed from its entire duration. Use:

* Training-set global statistics, or
* Running past-only normalization.

## 6.4 Optional causal scattering

For a causal wavelet filter $\psi_\lambda[r]$, $r \ge 0$,

$$
v_\lambda[n] = \sum_{r=0}^{R_\lambda} \psi_\lambda[r]\, x[n-r].
$$

A causal scattering envelope is

$$
S_\lambda[t] = \sum_{r=0}^{R_\phi} g[r]\, \left|v_\lambda[n_t-r]\right|.
$$

Here $g[r]$ is a causal low-pass filter.

Only retain wavelets whose effective support is below a specified history budget:

$$
R_\lambda \le R_{\max}.
$$

Very low-frequency wavelets with supports extending across several minutes should either be removed or treated as explicit long-memory features.

## 6.5 Optional causal phase harmonics

For causal analytic coefficients $v_i[n]$ and $v_j[n]$,

$$
[v_i[n]]^{p_{ij}} = |v_i[n]|\, e^{i p_{ij} \arg v_i[n]},
$$

where

$$
p_{ij} = \frac{\xi_j}{\xi_i}.
$$

Define

$$
\Phi_{ij}[t] = \sum_{r=0}^{R_\phi} g[r]\, \operatorname{Re}\left([v_i[n_t-r]]^{p_{ij}}\, \overline{v_j[n_t-r]}\right).
$$

Then decimate only after causal low-pass filtering.

Do not reproduce the current positive-frequency truncation operation for phase-harmonic decimation. Use ordinary anti-aliased causal downsampling.

---

# 7. Target FHR encoder

The target encoder produces

$$
H_t^Y = E_Y(X_{\le t}^Y),
$$

where $X_t^Y$ contains the causal target modalities.

A target input at time $t$ may be

$$
X_t^Y = \operatorname{concat}\left(R_t^Y, S_t^Y, \Phi_t^Y, M_t^Y\right).
$$

For a raw-only first version,

$$
X_t^Y = \operatorname{concat}\left(R_t^Y, M_t^Y\right).
$$

## 7.1 Modality fusion

Project each modality separately:

$$
r_t = W_R R_t^Y, \qquad s_t = W_S S_t^Y, \qquad p_t = W_P \Phi_t^Y.
$$

Then fuse with

$$
x_t^Y = \operatorname{MLP}\left(\operatorname{concat}(r_t, s_t, p_t, m_t)\right).
$$

Separate projections make it possible to:

* Drop one modality during ablations.
* Apply modality dropout.
* Measure the contribution of scattering and phase harmonics.
* Handle missing feature groups.

## 7.2 Temporal target encoder

The existing causal CNN–LSTM design is suitable for the target pathway after its normalization is made strictly causal.

A recommended target encoder has two branches.

### Dilated causal convolution branch

$$
H_{t,\mathrm{conv}}^Y = E_{\mathrm{TCN}}^Y(x_{\le t}^Y).
$$

Suggested dilations:

$$
(1, 2, 4, 8, 16).
$$

Suggested kernels:

$$
(3, 7, 11, 15, 15).
$$

This branch captures local and multiscale temporal structure.

### Recurrent branch

$$
H_{t,\mathrm{rnn}}^Y = \operatorname{LSTM}\left(x_{\le t}^Y\right)_t.
$$

A unidirectional one- or two-layer LSTM or GRU retains longer history.

### Fusion

$$
H_t^Y = \operatorname{LayerNorm}\left[F_Y\left(H_{t,\mathrm{conv}}^Y \,\Vert\, H_{t,\mathrm{rnn}}^Y\right)\right].
$$

Suggested shape:

$$
H^Y \in \mathbb{R}^{B \times T \times 128}.
$$

---

# 8. FHR context latent $C_t$

The target context head produces a diagonal Gaussian:

$$
e_C(c_t \mid Y_t^-) = \mathcal{N}\left(\mu_t^C, \operatorname{diag} e^{\ell_t^C}\right).
$$

Use separate heads:

$$
\widetilde\mu_t^C = f_\mu(H_t^Y),
$$

$$
\widetilde\ell_t^C = f_\ell(H_t^Y).
$$

Bound the mean smoothly:

$$
\mu_t^C = a_C \tanh\left(\frac{\widetilde\mu_t^C}{a_C}\right).
$$

Bound the log-variance smoothly:

$$
\ell_t^C = \ell_{\min} + (\ell_{\max}-\ell_{\min})\, \sigma(\widetilde\ell_t^C).
$$

A reasonable starting interval is

$$
\ell_{\min} = -5, \qquad \ell_{\max} = 3.
$$

Sample using

$$
C_t = \mu_t^C + \exp\left(\tfrac12 \ell_t^C\right) \epsilon_t^C, \qquad \epsilon_t^C \sim \mathcal{N}(0,I).
$$

Suggested dimension:

$$
d_C = 32.
$$

The correct value should be selected using rate–distortion and downstream-probe analyses, not only forecast loss.

---

# 9. Training-only future encoder

The future encoder provides the backward CEB distribution

$$
b_C(c_t \mid Y_t^+).
$$

Input:

$$
Y_t^+ \in \mathbb{R}^{H}.
$$

A suitable architecture is:

1. Raw future normalization.
2. A small noncausal convolutional encoder.
3. Four downsampling stages.
4. Temporal pooling.
5. Mean and log-variance heads.

Noncausality is allowed here because this module sees a target that is already explicitly the future and is discarded at inference.

For $H=480$, the future encoder can map:

$$
480 \rightarrow 240 \rightarrow 120 \rightarrow 60 \rightarrow 30
$$

temporal positions.

After pooling,

$$
G_t^F \in \mathbb{R}^{128}.
$$

Then

$$
\mu_t^F = f_\mu^F(G_t^F),
$$

$$
\ell_t^F = f_\ell^F(G_t^F).
$$

The CEB rate is

$$
R_C(t) = \frac12 \sum_{j=1}^{d_C} \left[\ell_{t,j}^F - \ell_{t,j}^C + \frac{e^{\ell_{t,j}^C} + (\mu_{t,j}^C-\mu_{t,j}^F)^2}{e^{\ell_{t,j}^F}} - 1\right].
$$

The future encoder serves two purposes:

* It defines the semantics of the target latent in terms of the actual future.
* It penalizes target-history information that does not appear in the future representation.

---

# 10. Source UP encoder and lag attention

The source path should be source-pure:

$$
G_t^U = E_U(U_{\le t}).
$$

However, the source representation used as attention keys and values should preferably have a **bounded receptive field**.

The current source LSTM representation contains unbounded earlier history. If lag attention points to $H^U_{t-\ell}$, that hidden state may itself contain information from much earlier times. An attention peak at $\ell$ would then not correspond cleanly to a physical source delay.

For interpretable lag attribution, use:

$$
G_t^U = E_U^{\mathrm{local}}\left(U[t-R_U:t]\right),
$$

where $R_U$ is known and bounded.

A small causal TCN is preferable to an LSTM before lag attention.

For example:

$$
R_U = 8 \text{ to } 15
$$

decimated steps, corresponding to approximately $32$–$60$ seconds.

The attention window itself then supplies the longer source history.

## 10.1 Lag-attention equations

Let:

* $M$ be the number of attention heads.
* $L$ be the number of candidate lags.
* $\ell = 0, \ldots, L-1$.

Use the target context mean as the query:

$$
q_t^{(m)} = W_Q^{(m)} \mu_t^C.
$$

Using the mean makes the lag weights stable and avoids injecting latent-sampling noise into lag selection.

For each source lag,

$$
k_{t,\ell}^{(m)} = W_K^{(m)} G_{t-\ell}^U,
$$

$$
v_{t,\ell}^{(m)} = W_V^{(m)} G_{t-\ell}^U.
$$

The score is

$$
a_{t,m,\ell} = \frac{\left\langle q_t^{(m)}, k_{t,\ell}^{(m)} \right\rangle + \left\langle q_t^{(m)}, r_{m,\ell} \right\rangle}{\sqrt{d_h}} + b_{m,\ell},
$$

where:

* $r_{m,\ell}$ is a Shaw-style lag embedding.
* $b_{m,\ell}$ is an optional ALiBi-like lag bias.

Apply the strict causal validity mask:

$$
m_{t,\ell} = \mathbf{1}[t-\ell \ge 0].
$$

Normalize over lag:

$$
\alpha_{t,m,\ell} = \operatorname{entmax}_{\ell}\left(a_{t,m,\ell}\right)
$$

or use softmax.

The attended source summary is

$$
A_t^{(m)} = \sum_{\ell=0}^{L-1} \alpha_{t,m,\ell}\, v_{t,\ell}^{(m)}.
$$

Suggested initial geometry:

$$
M = 4, \qquad L = 91,
$$

which covers

$$
91 \times 4\ \mathrm{s} \approx 6\ \mathrm{minutes}.
$$

Return two attention tensors:

* `attn_probs`: normalized probabilities before dropout.
* `attn_used`: weights after training dropout.

Use `attn_probs` for scientific attribution. This avoids the current situation in which training-time attention dropout breaks the identity between KL and its lag decomposition.

## 10.2 Remove the fixed UP shift in the new pipeline

The new raw pipeline should ideally use physically synchronized FHR and UP clocks without a hand-applied $-20$-second source shift.

Let the attention discover the lag.

When a preprocessing shift is unavoidable, store it explicitly as metadata and report

$$
\operatorname{lag}_{\mathrm{physical}} = 4\ell + \operatorname{shift}_{\mathrm{preprocessing}} + \operatorname{group\ delay}_{E_U}.
$$

A lag value should never require undocumented correction.

---

# 11. Conditional source prior and posterior

Let $d_S$ be divisible by the number of heads:

$$
d_S = M\, d_{S,h}.
$$

A suitable starting choice is

$$
d_S = 12, \qquad M = 4, \qquad d_{S,h} = 3.
$$

This deliberately allocates less capacity to the source addition than to the FHR context.

## 11.1 Target-only source prior

The target-only prior is

$$
r_S(S_t \mid C_t).
$$

For each head $m$,

$$
\mu_t^{S,0,(m)} = f_{\mu,0}^{(m)}(C_t),
$$

$$
\widetilde\ell_t^{S,0,(m)} = f_{\ell,0}^{(m)}(C_t).
$$

Concatenating the head groups gives

$$
\mu_t^{S,0} \in \mathbb{R}^{d_S}, \qquad \ell_t^{S,0} \in \mathbb{R}^{d_S}.
$$

This prior sees no UP.

## 11.2 Source-conditioned posterior

Each posterior group $m$ sees:

* The common FHR context $C_t$.
* Only the attended source summary from head $m$.

Define

$$
F_t^{(m)} = \operatorname{MLP}^{(m)}\left(C_t \,\Vert\, A_t^{(m)}\right).
$$

Produce residual parameter updates

$$
\widetilde{\Delta\mu}_t^{(m)} = g_\mu^{(m)}(F_t^{(m)}),
$$

$$
\widetilde{\Delta\ell}_t^{(m)} = g_\ell^{(m)}(F_t^{(m)}).
$$

Then

$$
\mu_t^{S,1,(m)} = \mu_t^{S,0,(m)} + a_\mu \tanh\left(\frac{\widetilde{\Delta\mu}_t^{(m)}}{a_\mu}\right),
$$

and

$$
\widetilde\ell_t^{S,1,(m)} = \widetilde\ell_t^{S,0,(m)} + a_\ell \tanh\left(\frac{\widetilde{\Delta\ell}_t^{(m)}}{a_\ell}\right).
$$

Apply the smooth variance bound only after adding the residual:

$$
\ell_t^{S,1,(m)} = \operatorname{smoothbound}\left(\widetilde\ell_t^{S,1,(m)}\right).
$$

The residual variance update must be applied to the **raw pre-bound prior log-variance**, not to the already bounded value. This preserves exact equality when the residual is zero, as in the current implementation.

Zero-initialize all source residual heads:

$$
g_\mu^{(m)} = 0, \qquad g_\ell^{(m)} = 0.
$$

Therefore,

$$
\mu_t^{S,1} = \mu_t^{S,0},
$$

$$
\ell_t^{S,1} = \ell_t^{S,0}
$$

at initialization.

## 11.3 Paired reparameterization

Draw one source-noise tensor:

$$
\epsilon_t^S \sim \mathcal{N}(0,I).
$$

Use it for both distributions:

$$
S_t^0 = \mu_t^{S,0} + \sigma_t^{S,0} \odot \epsilon_t^S,
$$

$$
S_t^1 = \mu_t^{S,1} + \sigma_t^{S,1} \odot \epsilon_t^S.
$$

This is a common-random-number coupling.

When the two distributions are equal,

$$
S_t^0 = S_t^1
$$

sample by sample, not merely in distribution.

Consequently,

$$
Z_t^0 = Z_t^1
$$

and the baseline and full predictions are exactly equal at initialization.

---

# 12. The complete latent

Construct

$$
Z_t^0 = C_t \,\Vert\, S_t^0,
$$

$$
Z_t^1 = C_t \,\Vert\, S_t^1.
$$

For the suggested dimensions,

$$
d_C = 32, \qquad d_S = 12,
$$

so

$$
d_Z = 44.
$$

The exact update is

$$
\Delta Z_t^{U \rightarrow Y} = Z_t^1 - Z_t^0 = 0_{d_C} \,\Vert\, (S_t^1-S_t^0).
$$

This is the explicit architectural answer to the question:

> Is the new $z_t$ a latent representation of FHR plus minimum necessary information from UP?

Yes:

$$
\boxed{
Z_t^{\mathrm{full}} = \underbrace{C_t}_{\text{predictive FHR state}} \;\Vert\; \underbrace{S_t^1}_{\text{source-conditioned state}}.
}
$$

And relative to the target-only representation,

$$
\boxed{
Z_t^{\mathrm{full}} = Z_t^{\mathrm{base}} + \Delta Z_t^{U \rightarrow Y}.
}
$$

The FHR coordinates are shared and cannot be modified by the UP pathway. The UP update occupies only the source block.

The term "minimum necessary" is encouraged through:

* A small $d_S$.
* Conditional KL minimization.
* Future-prediction requirements.
* Source purity.
* No source contribution to past reconstruction.
* No source bypass around the bottleneck.

It is achieved exactly only at the ideal conditional MNI optimum. In practice, it must be evaluated with rate–distortion, ablation, and source-specificity tests.

---

# 13. Shared raw-future decoder

The new model should contain one future decoder:

$$
D_\theta(Z_t).
$$

It is invoked twice:

$$
(\mu_t^0, \ell_t^0) = D_\theta(Z_t^0),
$$

$$
(\mu_t^1, \ell_t^1) = D_\theta(Z_t^1).
$$

There is no separate baseline decoder and no separate residual decoder.

There is also no equation of the form

$$
\mu_{\mathrm{full}} = \mu_{\mathrm{base}} + \Delta\mu_{\mathrm{src}}
$$

inside the architecture.

The source effect can still be analyzed as

$$
\Delta\mu_t^{U \rightarrow Y} = \mu_t^1 - \mu_t^0,
$$

but this is now a consequence of changing the complete latent, not a separate output pathway.

This prevents the source latent from being defined solely as an output correction.

## 13.1 Nonautoregressive block decoder

For a two-minute future,

$$
H = 480 = 30 \times 16.
$$

Use $K_H = 30$ horizon tokens.

Project the latent:

$$
h_t^Z = W_Z Z_t + b_Z.
$$

For each future block $k = 1, \ldots, 30$,

$$
F_{t,k}^{(0)} = h_t^Z + e_k,
$$

where

$$
e_k \in \mathbb{R}^{d_{\mathrm{dec}}}
$$

is a learned horizon embedding.

Process all horizon tokens jointly:

$$
F_t = \operatorname{HorizonCore}\left(F_{t,1}^{(0)}, \ldots, F_{t,30}^{(0)}\right).
$$

A suitable horizon core contains:

* Three or four residual convolution blocks.
* Kernel size $3$.
* Dilations $1, 2, 4$, optionally $8$.
* LayerNorm or GroupNorm across the horizon of a single anchor.
* Optional FiLM modulation from $Z_t$.

Symmetric convolution across the output-horizon axis is allowed. The core processes future positions that are being generated jointly; it does not consume observed future data.

Each horizon token emits one raw block:

$$
\mu_{t,k} = W_\mu F_{t,k} \in \mathbb{R}^{16},
$$

$$
\ell_{t,k} = W_\ell F_{t,k} \in \mathbb{R}^{16}.
$$

After reshaping,

$$
\mu_t, \ell_t \in \mathbb{R}^{480}.
$$

For $H=600$, retain 30 horizon tokens and emit $600 / 30 = 20$ samples per token.

For $H=720$, emit $720 / 30 = 24$ samples per token.

## 13.2 Why not use a 480-step autoregressive decoder?

A powerful autoregressive decoder can predict each raw sample using preceding generated or teacher-forced future samples. It may then need very little information from $Z_t$.

That would undermine the central objective.

A parallel decoder forces the latent to specify the complete future trajectory rather than relying on sample-by-sample teacher forcing.

## 13.3 Raw likelihood

The simplest initial likelihood is a factorized Gaussian:

$$
p_\theta(Y_t^+ \mid Z_t) = \prod_{h=1}^{H} \mathcal{N}\left(Y_t^+[h];\, \mu_t[h],\, \exp(\ell_t[h])\right).
$$

The per-anchor negative log-likelihood is

$$
D_t = \frac12 \sum_{h=1}^{H} \left[\ell_t[h] + \frac{(Y_t^+[h]-\mu_t[h])^2}{e^{\ell_t[h]}}\right].
$$

A Student-$t$ output is a useful later alternative for robustness to artifacts and heavy-tailed deviations.

A factorized likelihood ignores conditional correlation among future raw samples. Therefore, its predictive log-score difference estimates transfer entropy within the chosen model family, not the exact data-generating TE.

Possible later upgrades include:

* A low-rank plus diagonal covariance.
* A banded Cholesky covariance.
* A fixed learned temporal correlation kernel.
* A conditional normalizing flow over future residuals.

Begin with the Gaussian likelihood and validate calibration before increasing distributional complexity.

## 13.4 Absolute prediction versus endpoint residualization

For the strongest latent requirement, the initial decoder should predict absolute normalized FHR:

$$
\widehat Y_t^+ = D_\theta(Z_t).
$$

Do not feed the last observed FHR value separately to the decoder.

This forces $C_t$ to encode:

* Current baseline.
* Recent slope.
* Current morphology.
* State required to connect history and future.

Add an auxiliary current-state head:

$$
(\widehat Y[n_t], \widehat{\Delta Y}[n_t]) = D_{\mathrm{state}}(C_t).
$$

A later engineering variant may predict residuals relative to $Y[n_t]$, but that introduces an explicit target-information bypass. It should be treated as a documented representational compromise, not as the reference architecture.

---

# 14. Optional past reconstruction

The recent past can be reconstructed as an auxiliary target-only task:

$$
\widehat Y_t^{\mathrm{rec}} = D_{\mathrm{past}}(C_t).
$$

The past decoder receives only $C_t$.

It does not receive:

* $S_t^0$.
* $S_t^1$.
* $\Delta S_t$.
* UP attention.
* Source encoder states.

Therefore,

$$
\frac{\partial \mathcal{L}_{\mathrm{past}}}{\partial \theta_{\mathrm{source}}} = 0.
$$

This is essential. Otherwise, UP could be rewarded for explaining historical FHR–UP correlation, which would contaminate the future-directed TE interpretation.

## 14.1 Appropriate weighting

Future prediction remains the main task.

Use

$$
\lambda_{\mathrm{past}} \ll \lambda_0, \lambda_1.
$$

A reasonable starting range is

$$
\lambda_{\mathrm{past}} \in [0.05, 0.15]
$$

when future losses have unit weight.

The value must be selected empirically.

## 14.2 Multiscale past reconstruction

Do not force $C_t$ to memorize all high-frequency past noise equally.

For $P=600$, define

$$
\mathcal{L}_{\mathrm{past}} = \sum_{r \in \{1,4,16\}} w_r \left|\operatorname{LPDown}_r\left(Y_t^{\mathrm{rec}}\right) - \operatorname{LPDown}_r\left(\widehat Y_t^{\mathrm{rec}}\right)\right|_1.
$$

Possible scales are:

* $4$ Hz.
* $1$ Hz.
* $0.25$ Hz.

Another good option is masked-patch past reconstruction. Randomly mask portions of the recent past and reconstruct only the masked parts. This is less likely to become a copying task.

## 14.3 Decision rule

Retain past reconstruction only when it improves at least one of:

* Future NLL.
* Target latent effective rank.
* Frozen latent probes.
* Downstream classification.
* Stability across random seeds.

Disable or reduce it when it increases $R_C$ substantially without improving future or downstream performance.

---

# 15. Complete training objective

**Implementation status.** Three of these auxiliary terms are built and shipped, and three are not.
`teb_vae/lag_attn_rws/nets/losses.py::compute_loss` implements 15.1 (multiscale, at exactly the
$(1, 4, 16)$ rates below), 15.2 (derivative Huber) and the **level** half of 15.3 (boundary
continuity), each weighted by its own config key — `lambda_ms`, `lambda_deriv`, `lambda_boundary`,
shipping at $0.1 / 0.1 / 0.05$ in the three raw-target packages and at $0.0$ in the two
feature-target ones, where a raw-waveform shape penalty over a feature-channel axis would mean
nothing. A term whose weight is $0.0$ is not computed and reports its metric as exact $0.0$.

**Not implemented**, and each is a decision rather than an omission: the *slope* variant of 15.3
below (the level identity is the cheap half and ships alone until the derivative metric says the
transition shape is still wrong at converged weights); 15.4, the state reconstruction loss, which
would need a decoder-state head this model deliberately does not have — there is no bypass; 15.5,
lag smoothness, which regularises learned lag embeddings this model does not use; and
$\mathcal{L}_{\mathrm{past}}$, the past-reconstruction weight, which never had a term.

For a valid anchor $t$, define:

$$
D_0(t) = -\mathbb{E}_{C,S^0} \log d_\theta(Y_t^+ \mid C, S^0),
$$

$$
D_1(t) = -\mathbb{E}_{C,S^1} \log d_\theta(Y_t^+ \mid C, S^1).
$$

Define the target-context rate:

$$
R_C(t) = D_{\mathrm{KL}}\left(e_C(C_t \mid Y_t^-) \,\|\, b_C(C_t \mid Y_t^+)\right).
$$

Define the source-transfer rate:

$$
R_U(t) = D_{\mathrm{KL}}\left(q_S(S_t \mid C_t, U_t^-) \,\|\, r_S(S_t \mid C_t)\right).
$$

The main objective is

$$
\boxed{
\begin{aligned}
\mathcal{L} ={}& \lambda_0 D_0 + \lambda_1 D_1 + \beta_C R_C + \beta_U R_U \\
&+ \lambda_{\mathrm{past}} \mathcal{L}_{\mathrm{past}} + \lambda_{\mathrm{ms}} \mathcal{L}_{\mathrm{ms}} + \lambda_{\Delta} \mathcal{L}_{\Delta} \\
&+ \lambda_{\mathrm{boundary}} \mathcal{L}_{\mathrm{boundary}} + \lambda_{\mathrm{state}} \mathcal{L}_{\mathrm{state}} + \lambda_{\mathrm{lag}} \mathcal{L}_{\mathrm{lag}}.
\end{aligned}
}
$$

## 15.1 Multiscale future loss — **implemented** (`lambda_ms`)

For both base and full forecasts,

$$
\mathcal{L}_{\mathrm{ms}} = \sum_{k \in \{0,1\}} \sum_{r \in \{1,4,16\}} w_r \left|\operatorname{LPDown}_r(Y_t^+) - \operatorname{LPDown}_r(\mu_t^k)\right|_1.
$$

The probabilistic NLL remains the primary predictive objective. Multiscale losses regularize the mean trajectory.

As built: the anchor's forecast block is flattened over its horizon and raw axes, average-pooled at
`MS_RATES = (1, 4, 16)` with $w_r$ uniform, and the forecast mask is pooled alongside it and applied
as a weight. The mask is applied **before** pooling — pooling mixes neighbours, so a gap left in
until afterwards would leak its sentinel into every pool it touches.

## 15.2 Derivative loss — **implemented** (`lambda_deriv`)

$$
\mathcal{L}_\Delta = \sum_{k \in \{0,1\}} \operatorname{Huber}\left(\Delta Y_t^+, \Delta \mu_t^k\right).
$$

This encourages realistic short-term variability and transition shape.

As built: Huber at $\delta = 1$ over the same flattened block, with a difference pair counted valid
only when **both** of its samples are.

## 15.3 Boundary loss — **level implemented** (`lambda_boundary`); slope not

$$
\mathcal{L}_{\mathrm{boundary}} = \sum_{k \in \{0,1\}} \left|\mu_t^k[1] - Y[n_t]\right|.
$$

An additional slope boundary term may be used:

$$
\left|(\mu_t^k[2] - \mu_t^k[1]) - (Y[n_t] - Y[n_t-1])\right|.
$$

As built, the level identity only. $Y[n_t]$ needs no new tensor in the objective: on the raw grid
$X = R = D$, so it is a slicing identity on the gathered target — `mu[:, 1:, 0, 0]` against
`target[:, :-1, 0, -1]`, computed over $t \in [1, T_{\mathrm{valid}})$ structurally, which excludes
anchor $0$ by construction rather than by assuming a warm-up. Validity is anchor $t$'s own `weight`
at threshold times its contributing indicator, deliberately not anchor $t-1$'s forecast mask. The
weight ships at half the other two because this term constrains **one** sample per anchor against
$480$ for them. The slope variant is not implemented.

## 15.4 State reconstruction loss — **not implemented**

Blocked by a structural decision rather than deferred: $D_{\mathrm{level}}$ and
$D_{\mathrm{slope}}$ read a decoder state, and this model has no `decoder_state` head and no second
decoder — the no-bypass constraint is what makes the latent carry the predictive state at all.

$$
\mathcal{L}_{\mathrm{state}} = \left|D_{\mathrm{level}}(C_t) - Y[n_t]\right| + \operatorname{Huber}\left(D_{\mathrm{slope}}(C_t),\, Y[n_t] - Y[n_t-1]\right).
$$

This keeps endpoint state information inside $C_t$ without providing it directly to the future decoder.

## 15.5 Lag smoothness — **not implemented**

The shipped lag attention carries an ALiBi-style scalar bias per (head, lag) rather than learned lag
embeddings $r_{m,\ell}$, so there is nothing here to smooth. For learned lag embeddings
$r_{m,\ell}$,

$$
\mathcal{L}_{\mathrm{lag}} = \frac{1}{M(L-1)} \sum_{m,\ell} \left|r_{m,\ell+1} - r_{m,\ell}\right|_2^2.
$$

This is a parameter regularizer, not a measurement of physiological smoothness.

---

# 16. Loss scaling and the KL-weight problem

The current objective mixes a mean feature loss over thousands of coefficients with a summed latent KL. This makes the effective meaning of $\beta$ depend strongly on output dimension and reduction conventions.

For the raw model, all information quantities should have explicit units.

## 16.1 Recommended units

Define raw NLL as a sum over the horizon:

$$
D_t = -\sum_{h=1}^{H} \log p(Y_t^+[h] \mid Z_t).
$$

This has units of nats per future block.

Define KL as a sum over latent dimensions:

$$
R_t = \sum_j R_{t,j}.
$$

This also has units of nats per anchor.

Average only over batch and anchor dimensions.

Log both:

$$
D_{\mathrm{block}} \quad \text{in nats/block},
$$

and

$$
D_{\mathrm{sample}} = \frac{D_{\mathrm{block}}}{H} \quad \text{in nats/raw sample}.
$$

Do not average the reconstruction over $H$ while leaving the KL summed over latent dimensions unless the corresponding scaling is made explicit.

Changing the prediction horizon from 480 to 600 samples should not silently change the operational meaning of $\beta$.

## 16.2 Prefer constrained rate–distortion training

Rather than selecting one fixed $\beta$ and hoping it yields the desired latent, formulate:

$$
\min\ R_C + \eta R_U
$$

subject to

$$
D_0 \le \tau_0, \qquad D_1 \le \tau_1.
$$

The Lagrangian is

$$
\mathcal{L}_{\mathrm{constrained}} = R_C + \eta R_U + \lambda_0(D_0-\tau_0) + \lambda_1(D_1-\tau_1),
$$

where

$$
\lambda_0, \lambda_1 \ge 0.
$$

Update the dual variables by

$$
\lambda_k \leftarrow \left[\lambda_k + \eta_\lambda (D_k-\tau_k)\right]_+.
$$

This asks the model to use the smallest target and source rates that satisfy explicit predictive-quality requirements.

Constrained VAE methods such as GECO were developed precisely to replace difficult-to-interpret fixed trade-off coefficients with explicit reconstruction constraints. ControlVAE provides an alternative in which a feedback controller targets a desired KL rate. ([arXiv][4])

A practical development sequence is:

1. Run a target-only rate–distortion sweep.
2. Select a useful $R_C$ range.
3. Add the source block.
4. Sweep or control $R_U$ independently.
5. Never copy the current $\beta=0.1$ into the new model without recalibration.

## 16.3 Raw KL versus optimization KL

Always return separate quantities:

$$
R_{C,\mathrm{raw}}, \qquad R_{C,\mathrm{train}},
$$

$$
R_{U,\mathrm{raw}}, \qquad R_{U,\mathrm{train}}.
$$

Free bits or rate floors may be used for optimization, but only raw unfloored KL may be analyzed as information rate.

---

# 17. Primary and secondary TE measurements

The model should report two different TE-related quantities.

## 17.1 Primary readout: predictive log-score improvement

On held-out data,

$$
\boxed{
\widehat{\operatorname{TE}}_{\mathrm{pred}}(t) = \log p_1(Y_t^+) - \log p_0(Y_t^+)
}
$$

or equivalently,

$$
\boxed{
\widehat{\operatorname{TE}}_{\mathrm{pred}}(t) = D_0(t) - D_1(t).
}
$$

When $p_0$ and $p_1$ match the true conditional distributions, the expectation of this quantity is

$$
I(U_t^-; Y_t^+ \mid Y_t^-).
$$

In practice, it is a model-based held-out estimate.

It may be negative for individual anchors. The patient- or recording-level mean is the meaningful aggregate.

## 17.2 Secondary readout: latent TEB rate

$$
\boxed{
K_t^{\mathrm{TEB}} = R_U(t) = D_{\mathrm{KL}}\left(q_S \,\|\, r_S\right).
}
$$

This measures how much source information changes the latent distribution conditional on the FHR context.

It should be called:

* `teb_rate`,
* `source_conditional_kl`, or
* `latent_te_surrogate`.

It should not automatically be labeled true transfer entropy.

It approaches TE only when:

1. $C_t$ is sufficient for the FHR history.
2. The conditional source bottleneck approaches CMNI.
3. The decoder accurately represents the future conditional distribution.
4. Input construction is strictly causal.
5. Relevant confounding variables have been addressed.

## 17.3 Monte Carlo predictive likelihood

Because $C_t$, $S_t^0$, and $S_t^1$ are stochastic, approximate the marginal predictive likelihood with $K$ samples:

$$
\log \widehat p_k(Y_t^+) = \operatorname{logsumexp}_{r=1}^K \left[\log d_\theta\left(Y_t^+ \mid C_t^{(r)}, S_t^{k,(r)}\right)\right] - \log K.
$$

Use common random numbers:

* The same $C_t^{(r)}$ for base and full.
* The same $\epsilon_S^{(r)}$ for $S_t^0$ and $S_t^1$.

This reduces the variance of the base-versus-full difference.

For routine evaluation, $K=8$ is a reasonable start. More samples may be used for final analysis.

---

# 18. Head- and lag-resolved source information

Partition the source latent into head groups:

$$
S_t = S_t^{(1)} \,\Vert\, \cdots \,\Vert\, S_t^{(M)}.
$$

Then

$$
K_t^{(m)} = D_{\mathrm{KL}}\left(q_S^{(m)} \,\|\, r_S^{(m)}\right).
$$

The total source rate is

$$
K_t = \sum_{m=1}^M K_t^{(m)}.
$$

Define the lag map

$$
\boxed{
\widetilde K_{t,\ell} = \sum_{m=1}^M K_t^{(m)} \alpha_{t,m,\ell}.
}
$$

When the attention probabilities sum to one,

$$
\sum_{\ell=0}^{L-1} \widetilde K_{t,\ell} = K_t.
$$

This gives a rigorous head-to-latent decomposition because source head $m$ owns source latent group $m$.

Attention remains an attribution mechanism, not proof that a physical causal effect occurred at exactly that lag.

Validate lag attribution through:

* Synthetic known-lag data.
* Lag-band masking.
* Leave-one-band-out prediction changes.
* Impulse alignment.
* Source circular shifts.

---

# 19. Permutation and source-specificity controls

A deranged source may produce a large latent KL because it is out of distribution and moves the source posterior strongly. Therefore, do not require

$$
K_{\mathrm{shuffled}} < K_{\mathrm{true}}.
$$

The useful criterion is predictive:

$$
\boxed{
D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}}
}
$$

on held-out data.

This is consistent with the behavior documented in the current model: the prediction-space control is more discriminative than shuffled-source KL.

The new permutation procedure should:

1. Compute $C_t$ once.
2. Keep $r_S(S \mid C)$ unchanged.
3. Derange only the source embeddings $G^U$.
4. Recompute attention.
5. Recompute $q_S(S \mid C, \pi(U))$.
6. Construct $Z_t^{\mathrm{shuffled}}$.
7. Run the same shared decoder.
8. Compare the raw future NLL.

Also evaluate:

* Circularly shifted UP.
* Time-reversed UP.
* Spectrum-matched UP surrogates.
* Lag-band exclusions.

The base representation and base forecast must be bit-exact under all source permutations.

---

# 20. Efficient training over overlapping anchors

The model should produce a latent at every four-second step:

$$
C, S, Z \in \mathbb{R}^{B \times T \times d}.
$$

It is not necessary to decode all valid anchors in every training batch.

For $T=300$, a two-minute horizon leaves approximately

$$
T_{\mathrm{valid}} \approx 270
$$

forecast anchors.

Decode a random subset:

$$
\mathcal{I}_b \subset \mathcal{A}, \qquad |\mathcal{I}_b| = K_A,
$$

with, for example,

$$
K_A = 16 \text{ to } 32
$$

anchors per recording per batch.

Over epochs, all time steps receive forecast supervision.

This reduces decoder activation memory while maintaining a latent at every step.

Compute the future teacher, future decoder, source KL, and predictive losses only on selected valid anchors.

At validation and test time:

* Decode all valid anchors, or
* Use a fixed uniformly spaced anchor grid.

Because future windows overlap heavily, statistical uncertainty should be estimated by bootstrapping patients or complete recordings, not treating every anchor as an independent sample.

---

# 21. Training curriculum

A staged training procedure is safer than end-to-end training from random initialization.

## Stage 0: causal raw-data pipeline

Build:

* Raw FHR and UP alignment.
* Quality masks.
* Causal normalization.
* Causal raw stems.
* Exact anchor-to-raw indexing.
* Future-window extraction.
* Causality tests.

Do not add scattering or phase harmonics yet.

## Stage 1: target-only predictive FHR bottleneck

Train:

* Target raw stem.
* Target temporal encoder.
* Context head ($e_C$).
* Future encoder ($b_C$).
* Target-only source prior ($r_S$).
* Shared raw decoder.
* Optional target-only past decoder.

Set

$$
q_S = r_S.
$$

The stage-one objective is

$$
\mathcal{L}_1 = \lambda_0 D_0 + \beta_C R_C + \lambda_{\mathrm{past}} \mathcal{L}_{\mathrm{past}} + \lambda_{\mathrm{ms}} \mathcal{L}_{\mathrm{ms}} + \lambda_\Delta \mathcal{L}_\Delta + \lambda_{\mathrm{state}} \mathcal{L}_{\mathrm{state}}.
$$

Acceptance criteria:

* Good target-only raw future prediction.
* Noncollapsed $C_t$.
* Small context-sufficiency gap.
* Latent probes show future-relevant information.
* Shuffling or zeroing $C_t$ severely damages prediction.

## Stage 2: source bottleneck

Freeze initially:

* Most target raw-stem weights.
* Most target temporal encoder weights.
* Context head.
* Future encoder.

Train:

* Source raw stem.
* Local source encoder.
* Lag attention.
* Source posterior residual heads.
* Source-input projection of the shared decoder.
* Optionally the decoder core at a lower learning rate.

Initialize

$$
q_S = r_S.
$$

Use

$$
\mathcal{L}_2 = \lambda_0 D_0 + \lambda_1 D_1 + \beta_U R_U + \text{raw auxiliary losses}.
$$

The target-only baseline must remain active throughout this stage.

## Stage 3: joint fine-tuning

Unfreeze the complete model with separate learning rates:

* Lowest learning rate for target context.
* Moderate learning rate for decoder.
* Higher learning rate for source attention and source posterior.

Optimize the complete objective.

Continue to monitor:

$$
D_1 < D_0, \qquad D_0 < D_{\mathrm{shuffled}}, \qquad R_U > 0
$$

only when UP genuinely improves the forecast.

## Stage 4: add causal scattering and phase harmonics

Add one feature family at a time:

1. FHR causal scattering.
2. UP causal scattering.
3. FHR causal phase harmonics.
4. UP causal phase harmonics.

For each addition, compare:

* Future NLL.
* $R_C$.
* $R_U$.
* Context effective rank.
* Source specificity.
* Synthetic-lag recovery.
* Causality tests.

Do not accept a feature family merely because training loss improves.

## Stage 5: likelihood calibration and TE evaluation

Freeze the architecture selection.

Evaluate:

* Raw held-out NLL.
* Calibration coverage.
* CRPS.
* Probability integral transform diagnostics.
* Predictive TE.
* Latent TEB rate.
* Permutation controls.
* Synthetic systems.
* Patient-level confidence intervals.

---

# 22. Recommended tensor interface

A possible model input is:

| Input | Shape | Meaning |
| --- | --- | --- |
| `y_raw` | $(B,N)$ | raw FHR |
| `u_raw` | $(B,N)$ | raw UP |
| `y_valid` | $(B,N)$ | FHR validity mask |
| `u_valid` | $(B,N)$ | UP validity mask |
| `anchor_raw_index` | $(T,)$ | raw cutoff for each latent time |
| `y_st` | $(B,T,C_{YS})$, optional | causal FHR scattering |
| `y_ph` | $(B,T,C_{YP})$, optional | causal FHR phase harmonics |
| `u_st` | $(B,T,C_{US})$, optional | causal UP scattering |
| `u_ph` | $(B,T,C_{UP})$, optional | causal UP phase harmonics |
| `anchor_mask` | $(B,T)$ | valid forecast anchors |
| `lag_band_mask` | $(L,)$ or $(T,L)$, optional | lag ablation |

Core outputs should include:

| Output | Shape | Meaning |
| --- | --- | --- |
| `target_state` | $(B,T,128)$ | causal target state |
| `source_local_state` | $(B,T,d_U)$ | bounded-memory UP state |
| `c_mu`, `c_logvar` | $(B,T,d_C)$ | FHR context distribution |
| `future_c_mu`, `future_c_logvar` | $(B,A,d_C)$ | training-only future encoder |
| `s_prior_mu`, `s_prior_logvar` | $(B,T,d_S)$ | target-only source prior |
| `s_post_mu`, `s_post_logvar` | $(B,T,d_S)$ | UP-conditioned source distribution |
| `z_base` | $(B,T,d_C+d_S)$ | sampled target-only complete latent |
| `z_full` | $(B,T,d_C+d_S)$ | sampled source-conditioned complete latent |
| `z_base_mean` | $(B,T,d_C+d_S)$ | deterministic target-only representation |
| `z_full_mean` | $(B,T,d_C+d_S)$ | deterministic full representation |
| `z_up_update_mean` | $(B,T,d_C+d_S)$ | additive UP update |
| `attn_probs` | $(B,T,M,L)$ | pre-dropout lag probabilities |
| `attn_used` | $(B,T,M,L)$ | training attention weights |
| `pred_base_mu` | $(B,A,H)$ | raw target-only future mean |
| `pred_base_logvar` | $(B,A,H)$ | target-only uncertainty |
| `pred_full_mu` | $(B,A,H)$ | raw full future mean |
| `pred_full_logvar` | $(B,A,H)$ | full uncertainty |
| `context_kl_dim` | $(B,A,d_C)$ | target CEB rate |
| `source_kl_dim` | $(B,A,d_S)$ | source TEB rate |
| `source_kl_head` | $(B,A,M)$ | head-structured source rate |
| `te_lag_map` | $(B,A,L)$ | lag attribution |
| `past_reconstruction` | $(B,A,P)$, optional | target-only past reconstruction |

---

# 23. Suggested starting configuration

These are starting values, not theoretically fixed values.

| Parameter | Suggested value |
| --- | ---: |
| Raw sampling rate | $4$ Hz |
| Anchor stride | $16$ samples |
| Latent interval | $4$ s |
| Future horizon | $480$ samples |
| Future duration | $120$ s |
| Optional past reconstruction | $600$ samples |
| Target/source model width | $128$ |
| FHR context dimension ($d_C$) | $32$ |
| Source dimension ($d_S$) | $12$ |
| Complete latent dimension | $44$ |
| Attention heads | $4$ |
| Source dims per head | $3$ |
| Maximum lag | $90$ steps |
| Lag duration | $360$ s |
| Horizon tokens | $30$ |
| Decoder hidden width | $128$ |
| Horizon depth | $3$ |
| Raw stem stages | $4$ |
| Decoded anchors per record | $16$–$32$ |
| Initial likelihood | learned Gaussian |
| Optional past weight | $0.05$–$0.15$ |

Important dimension ablations are:

$$
d_C \in \{16, 32, 64\},
$$

$$
d_S \in \{4, 8, 12, 16\}.
$$

The expected useful relationship is

$$
d_S < d_C.
$$

---

# 24. Forward-pass pseudocode

```python
def forward(
    y_raw,
    u_raw,
    y_valid,
    u_valid,
    anchor_raw_index,
    *,
    y_st=None,
    y_ph=None,
    u_st=None,
    u_ph=None,
    decode_anchor_idx=None,
    lag_band_mask=None,
    sample_latents=True,
):
    # ---------------------------------------------------------
    # 1. Strictly causal raw feature extraction
    # ---------------------------------------------------------
    y_raw_frames = target_raw_stem(y_raw, y_valid, anchor_raw_index)
    u_raw_frames = source_raw_stem(u_raw, u_valid, anchor_raw_index)

    # Optional causal transform features only
    y_inputs = target_fusion(
        raw=y_raw_frames,
        scattering=y_st,
        phase=y_ph,
        validity=y_valid,
    )

    u_inputs = source_fusion(
        raw=u_raw_frames,
        scattering=u_st,
        phase=u_ph,
        validity=u_valid,
    )

    # ---------------------------------------------------------
    # 2. Target and source encoding
    # ---------------------------------------------------------
    h_y = target_history_encoder(y_inputs)
    g_u = source_local_encoder(u_inputs)

    # ---------------------------------------------------------
    # 3. Target-only predictive FHR context
    # ---------------------------------------------------------
    c_mu, c_logvar = context_head(h_y)

    if sample_latents:
        eps_c = torch.randn_like(c_mu)
        c = c_mu + torch.exp(0.5 * c_logvar) * eps_c
    else:
        c = c_mu

    # ---------------------------------------------------------
    # 4. Lagged source attention
    # Query uses target context, not a decoder bypass
    # ---------------------------------------------------------
    source_heads, attn_probs, attn_used = lag_attention(
        query=c_mu,
        source=g_u,
        lag_band_mask=lag_band_mask,
    )

    # ---------------------------------------------------------
    # 5. Target-only source prior
    # ---------------------------------------------------------
    (
        s0_mu,
        s0_logvar,
        s0_raw_logvar,
    ) = source_prior_head(c)

    # ---------------------------------------------------------
    # 6. UP-conditioned residual posterior
    # ---------------------------------------------------------
    s1_mu, s1_logvar = source_posterior_head(
        context=c,
        attended_source_heads=source_heads,
        prior_mu=s0_mu,
        prior_raw_logvar=s0_raw_logvar,
    )

    # ---------------------------------------------------------
    # 7. Paired source sampling
    # ---------------------------------------------------------
    if sample_latents:
        eps_s = torch.randn_like(s0_mu)

        s0 = s0_mu + torch.exp(0.5 * s0_logvar) * eps_s
        s1 = s1_mu + torch.exp(0.5 * s1_logvar) * eps_s
    else:
        s0 = s0_mu
        s1 = s1_mu

    # ---------------------------------------------------------
    # 8. Complete latent vectors
    # ---------------------------------------------------------
    z_base = torch.cat([c, s0], dim=-1)
    z_full = torch.cat([c, s1], dim=-1)

    z_base_mean = torch.cat([c_mu, s0_mu], dim=-1)
    z_full_mean = torch.cat([c_mu, s1_mu], dim=-1)

    z_up_update_mean = torch.cat(
        [torch.zeros_like(c_mu), s1_mu - s0_mu],
        dim=-1,
    )

    # ---------------------------------------------------------
    # 9. Decode selected valid anchors
    # Same decoder, called twice
    # ---------------------------------------------------------
    z0_decode = gather_anchors(z_base, decode_anchor_idx)
    z1_decode = gather_anchors(z_full, decode_anchor_idx)

    z_pair = torch.cat([z0_decode, z1_decode], dim=0)
    pred_mu_pair, pred_logvar_pair = raw_future_decoder(z_pair)

    pred_base_mu, pred_full_mu = pred_mu_pair.chunk(2, dim=0)
    pred_base_logvar, pred_full_logvar = pred_logvar_pair.chunk(2, dim=0)

    # ---------------------------------------------------------
    # 10. Training-only future CEB encoder
    # ---------------------------------------------------------
    future_raw = gather_raw_future(
        y_raw,
        anchor_raw_index,
        decode_anchor_idx,
        horizon=H,
    )

    future_c_mu, future_c_logvar = future_encoder(future_raw)

    # ---------------------------------------------------------
    # 11. Optional target-only past reconstruction
    # ---------------------------------------------------------
    past_reconstruction = None
    if use_past_decoder:
        c_decode = gather_anchors(c, decode_anchor_idx)
        past_reconstruction = past_decoder(c_decode)

    # ---------------------------------------------------------
    # 12. KL and lag analysis
    # ---------------------------------------------------------
    context_kl_dim = gaussian_kl(
        c_mu_selected,
        c_logvar_selected,
        future_c_mu,
        future_c_logvar,
    )

    source_kl_dim = gaussian_kl(
        s1_mu_selected,
        s1_logvar_selected,
        s0_mu_selected,
        s0_logvar_selected,
    )

    source_kl_head = group_sum(source_kl_dim, num_heads)
    te_lag_map = lag_attribution(source_kl_head, attn_probs_selected)

    return {
        "target_state": h_y,
        "source_local_state": g_u,
        "c_mu": c_mu,
        "c_logvar": c_logvar,
        "s_prior_mu": s0_mu,
        "s_prior_logvar": s0_logvar,
        "s_post_mu": s1_mu,
        "s_post_logvar": s1_logvar,
        "z_base": z_base,
        "z_full": z_full,
        "z_base_mean": z_base_mean,
        "z_full_mean": z_full_mean,
        "z_up_update_mean": z_up_update_mean,
        "attn_probs": attn_probs,
        "attn_used": attn_used,
        "pred_base_mu": pred_base_mu,
        "pred_base_logvar": pred_base_logvar,
        "pred_full_mu": pred_full_mu,
        "pred_full_logvar": pred_full_logvar,
        "future_c_mu": future_c_mu,
        "future_c_logvar": future_c_logvar,
        "context_kl_dim": context_kl_dim,
        "source_kl_dim": source_kl_dim,
        "source_kl_head": source_kl_head,
        "te_lag_map": te_lag_map,
        "past_reconstruction": past_reconstruction,
    }
```

---

# 25. Required tests

These tests should block training when they fail.

## 25.1 Prefix invariance

Construct $x$ and $x'$ such that

$$
x[:n_t+1] = x'[:n_t+1]
$$

but

$$
x[n_t+1:] \ne x'[n_t+1:].
$$

Require

$$
R_t(x) = R_t(x'), \qquad H_t(x) = H_t(x'), \qquad C_t(x) = C_t(x'), \qquad S_t(x) = S_t(x')
$$

within numerical tolerance.

Run this for FHR and UP separately.

## 25.2 Future Jacobian

For every selected $j > n_t$,

$$
\left|\frac{\partial C_t}{\partial Y[j]}\right| < \epsilon, \qquad \left|\frac{\partial S_t}{\partial U[j]}\right| < \epsilon.
$$

## 25.3 Source purity

After changing or permuting UP:

$$
C_t(Y,U) = C_t(Y,\pi(U)).
$$

Also require

$$
S_t^0(Y,U) = S_t^0(Y,\pi(U)), \qquad Z_t^0(Y,U) = Z_t^0(Y,\pi(U)),
$$

and

$$
\widehat Y_{t,\mathrm{base}}^+(Y,U) = \widehat Y_{t,\mathrm{base}}^+(Y,\pi(U)).
$$

These should be bit-exact in evaluation mode.

## 25.4 Zero-source initialization

At construction,

$$
\max_t R_U(t) < 10^{-6}.
$$

With paired source noise,

$$
\max_t \left|Z_t^1 - Z_t^0\right| < 10^{-6},
$$

and

$$
\max_t \left|\mu_t^1 - \mu_t^0\right| < 10^{-6}.
$$

## 25.5 No past-loss source gradient

Backpropagate only

$$
\mathcal{L}_{\mathrm{past}}.
$$

Every source-path parameter must have:

$$
\nabla_\theta \mathcal{L}_{\mathrm{past}} = 0.
$$

## 25.6 No decoder bypass

Programmatically inspect the future decoder call signature and graph.

It must consume only:

* $Z_t$.
* Horizon embeddings.
* Optional fixed metadata that has been explicitly approved.

It must not consume `target_state` or an equivalent hidden state.

## 25.7 Latent dependence

Evaluate raw future NLL under:

* Normal $Z_t$.
* $C_t = 0$.
* Batch-shuffled $C_t$.
* $S_t^1 = S_t^0$.
* Batch-shuffled source block.
* Individual source groups removed.

An informative latent should produce substantial, structured performance degradation under these ablations.

## 25.8 Synthetic TE systems

Test:

1. No coupling.
2. Unidirectional ($U \rightarrow Y$) coupling.
3. Reverse ($Y \rightarrow U$) coupling.
4. Increasing coupling strength.
5. Known fixed lag.
6. Variable lag.
7. Common-driver confounding.
8. Instantaneous correlation without delayed influence.
9. Nonlinear coupling.
10. Source influence limited to one frequency range.

Under the null,

$$
R_U \approx 0, \qquad D_0 - D_1 \approx 0.
$$

As true coupling increases, the held-out predictive uplift and source rate should generally increase.

The recovered lag map should match the known lag after accounting for the source encoder's bounded receptive field.

---

# 26. Measuring whether the new latent is informative

Forecast accuracy is necessary but not sufficient.

## 26.1 Target-context probes

Freeze the model and train small probes from $\mu_t^C$ to predict:

* Future FHR mean.
* Future variance.
* Future slope.
* Deceleration depth.
* Deceleration duration.
* Recovery time.
* Future low-frequency power.
* Current FHR baseline.
* Current short-term variability.
* Available downstream clinical labels.

These tests measure what the FHR context actually represents.

## 26.2 Full-latent probes

Train the same probes from

$$
\bar Z_t^{\mathrm{full}}.
$$

Compare:

$$
\operatorname{Probe}(C_t)
$$

against

$$
\operatorname{Probe}(Z_t^{\mathrm{full}}).
$$

Improvement indicates that UP adds useful predictive or clinical information.

## 26.3 Source-update probes

Use

$$
\Delta\bar Z_t^{U \rightarrow Y}.
$$

This representation should primarily predict:

* Source-related forecast uplift.
* Future response associated with contractions.
* Lag structure.
* Source-conditioned changes in future morphology.

It should not be the primary complete FHR representation.

## 26.4 Latent utilization

Track separately for $C$ and $S$:

* Per-dimension raw KL.
* Active-unit fraction.
* Mean posterior variance.
* Saturation fraction.
* Pairwise correlation.
* Covariance eigenvalues.
* Effective rank.

For covariance eigenvalues $\lambda_i$, define

$$
\widetilde\lambda_i = \frac{\lambda_i}{\sum_j \lambda_j}, \qquad r_{\mathrm{eff}} = \exp\left(-\sum_i \widetilde\lambda_i \log \widetilde\lambda_i\right).
$$

Report:

$$
r_{\mathrm{eff}}^C, \qquad r_{\mathrm{eff}}^S, \qquad r_{\mathrm{eff}}^Z.
$$

---

# 27. Segment-level representation for classification

The model produces a latent at every four-second step:

$$
\bar Z^{\mathrm{full}} \in \mathbb{R}^{B \times T \times d_Z}.
$$

For a per-recording classifier, introduce an explicit representation exporter.

A simple masked statistical pooling is

$$
R_{\mathrm{seg}} = \operatorname{concat}\left(\operatorname{mean}_t \bar Z_t,\ \operatorname{std}_t \bar Z_t,\ \operatorname{max}_t \bar Z_t\right).
$$

An attention-pooling alternative is

$$
w_t = \frac{\exp g(\bar Z_t)}{\sum_{t'} \exp g(\bar Z_{t'})}, \qquad R_{\mathrm{seg}} = \sum_t w_t \bar Z_t.
$$

For interpretable downstream analysis, export three pooled representations:

$$
R_{\mathrm{FHR}} = \operatorname{Pool}(\mu_t^C),
$$

$$
R_{\mathrm{UP}} = \operatorname{Pool}\left(\mu_t^{S,1} - \mu_t^{S,0}\right),
$$

$$
R_{\mathrm{full}} = \operatorname{Pool}\left(\bar Z_t^{\mathrm{full}}\right).
$$

This directly answers:

* How much classification information is present in FHR alone?
* What does UP add?
* What does the complete representation provide?

The new evaluation pipeline should analyze all three. It should no longer treat the posterior-minus-prior gap as the model's only latent-facing representation.

---

# 28. Recommended first implementation

The first scientifically defensible version should use:

$$
\boxed{
\begin{aligned}
&\text{Raw causal FHR input} \\
&\text{Raw causal UP input} \\
&\text{Two-minute raw FHR future} \\
&\text{Target CEB latent } C_t \\
&\text{Conditional TEB source latent } S_t \\
&\text{Complete latent } Z_t = [C_t, S_t] \\
&\text{One shared raw decoder} \\
&\text{No decoder-state bypass} \\
&\text{No source contribution to past reconstruction}.
\end{aligned}
}
$$

Recommended first geometry:

$$
d_C = 32, \qquad d_S = 12, \qquad d_Z = 44,
$$

$$
H = 480, \qquad M = 4, \qquad L = 91.
$$

Use future prediction as the primary objective.

Add the $2.5$-minute target-only past reconstruction only after the target-only raw future model has demonstrated:

* A noncollapsed target latent.
* Good future prediction.
* Good latent probes.
* Causal correctness.

Then add causal scattering and phase harmonics incrementally.

---

# 29. Final model definition

$$
\boxed{
\begin{aligned}
Y_{\le t} &\xrightarrow[\text{strictly causal}]{E_Y} H_t^Y \xrightarrow{e_C} C_t, \\[1mm]
Y_t^+ &\xrightarrow[\text{training only}]{b_C} \text{future-defined target latent distribution}, \\[1mm]
U_{\le t} &\xrightarrow[\text{strictly causal}]{E_U^{\mathrm{local}}} G_{\le t}^U \xrightarrow[\text{query } C_t]{\text{lag attention}} A_t, \\[1mm]
C_t &\xrightarrow{r_S} S_t^0, \\[1mm]
(C_t, A_t) &\xrightarrow{q_S} S_t^1, \\[1mm]
Z_t^0 &= [C_t, S_t^0], \\[1mm]
Z_t^1 &= [C_t, S_t^1] = Z_t^0 + [0, S_t^1-S_t^0], \\[1mm]
Z_t^0 &\xrightarrow{D_\theta} p_0(Y_t^+), \\[1mm]
Z_t^1 &\xrightarrow{\text{same } D_\theta} p_1(Y_t^+).
\end{aligned}
}
$$

The resulting model learns:

$$
\boxed{
C_t = \text{compressed predictive FHR state},
}
$$

$$
\boxed{
S_t^1 - S_t^0 = \text{minimum necessary UP-derived update},
}
$$

and

$$
\boxed{
Z_t^{\mathrm{full}} = \text{complete FHR representation incorporating UP}.
}
$$

The source-conditioned KL is computed only over the source block, while the raw future decoder is forced to use the complete latent. This preserves the conditional transfer-entropy bottleneck interpretation while correcting the central limitation of the current architecture: the FHR representation now lives inside the exported latent rather than in a separate `decoder_state`.

[1]: https://arxiv.org/pdf/2211.16607 "https://arxiv.org/pdf/2211.16607"
[2]: https://arxiv.org/pdf/1512.06479.pdf "https://arxiv.org/pdf/1512.06479.pdf"
[3]: https://arxiv.org/pdf/2002.05379 "https://arxiv.org/pdf/2002.05379"
[4]: https://arxiv.org/abs/1810.00597 "https://arxiv.org/abs/1810.00597"

---

# Amendment for first version

Yes. Your notes suggest a simpler and cleaner **first implementation** than the two-block ($C_t$/$S_t$) architecture I previously proposed.

The best first version is:

$$
\boxed{
p_\theta(z_t \mid Y_{\le t})
\quad\text{and}\quad
q_\phi(z_t \mid Y_{\le t}, U_{\le t})
}
$$

in one shared latent space, with:

* $p_\theta$ representing FHR from its own history.
* $q_\phi$ representing the same FHR state after incorporating the minimum useful information from UP.
* One shared raw-signal decoder.
* No `decoder_state` or other target-history bypass.

This is simpler, closer to your desired interpretation, and closer to the VAE-TEB formulation.

---

# 1. What exactly is $S_t^0$?

In the previously proposed split architecture,

$$
r_S(s_t \mid C_t)
$$

was the target-conditioned prior over the source-specific latent block. Therefore, yes:

$$
S_t^0 \sim r_S(s_t \mid C_t)
$$

depends only on the FHR representation $C_t$.

Consequently,

$$
Z_t^0 = [C_t, S_t^0]
$$

is entirely generated from FHR history and can be regarded as a target-only representation.

However, the semantic interpretation is more subtle:

* $C_t$ is the main FHR state.
* $S_t^0$ is the expected or null state of the source-information slot given FHR.
* $S_t^1-S_t^0$ is the additional UP-derived update.

So $S_t^0$ is FHR-derived, but its main purpose is to define the **reference distribution against which UP information is measured**. It is not necessary to interpret or probe $S_t^0$ separately as an FHR representation.

This produces three objects:

$$
C_t = \text{core FHR state},
$$

$$
Z_t^0 = [C_t, S_t^0] = \text{complete target-only latent},
$$

$$
Z_t^1 = [C_t, S_t^1] = \text{complete target-plus-UP latent}.
$$

That architecture is valid, but it introduces two latent blocks and two information bottlenecks. For your first implementation, I recommend something simpler.

---

# 2. Recommended simplification: one complete latent space

Define a target-only latent distribution:

$$
\boxed{
p_t(z) = p_\theta(z_t \mid Y_{\le t})
}
$$

and a source-conditioned latent distribution:

$$
\boxed{
q_t(z) = q_\phi(z_t \mid Y_{\le t}, U_{\le t}).
}
$$

The target-only distribution is the FHR representation:

$$
z_t^Y \sim p_\theta(z_t \mid Y_{\le t}).
$$

The source-conditioned distribution is the FHR representation after incorporating UP:

$$
z_t^{YU} \sim q_\phi(z_t \mid Y_{\le t}, U_{\le t}).
$$

Conceptually,

$$
\boxed{
z_t^{YU} = z_t^Y + \Delta z_t^{U \rightarrow Y}.
}
$$

Unlike the current architecture, $z_t^Y$ must itself contain enough information to predict future FHR because the decoder will not receive `decoder_state`.

This gives the exact interpretation you want:

$$
\boxed{
z_t^Y = \text{predictive representation of FHR},
}
$$

$$
\boxed{
\Delta z_t^{U \rightarrow Y} = \text{minimum additional predictive information from UP},
}
$$

$$
\boxed{
z_t^{YU} = \text{FHR representation updated by UP}.
}
$$

The current model cannot guarantee this because the target-only FHR content bypasses $z_t$ through `decoder_state`, while $z_t$ is used only by the residual correction pathway.

---

# 3. Proposed first-version architecture

For the first version, use:

* Existing scattering and phase-harmonic inputs.
* Raw FHR as the prediction target.
* One target-only prior latent.
* One residual UP-conditioned posterior latent.
* One shared raw decoder.
* A minimal three-term objective.

The raw-input model can be a later version.

---

# 4. Input definition using existing features

Let

$$
X_t^Y = \operatorname{concat}\left(Y_t^{\mathrm{ST}}, Y_t^{\mathrm{PH}}\right)
$$

be the FHR feature vector.

Let

$$
X_t^U = \operatorname{concat}\left(U_t^{\mathrm{ST}}, U_t^{\mathrm{PH}}\right)
$$

be the source-pure UP feature vector.

Do not place cross-channel phase-harmonic coefficients in either the target-only prior or source-only encoder. Cross-channel coefficients already combine FHR and UP and therefore destroy the clean separation

$$
p(z_t \mid Y_{\le t}) \quad\text{versus}\quad q(z_t \mid Y_{\le t}, U_{\le t}).
$$

For your current configuration, the inputs are approximately

$$
X^Y \in \mathbb{R}^{B \times T \times C_Y}, \qquad X^U \in \mathbb{R}^{B \times T \times C_U},
$$

with

$$
T = 300, \qquad \Delta t = 4\ \mathrm{s}.
$$

The target future is no longer the future feature tensor. It is the raw FHR signal:

$$
Y_t^+ = Y[n_t+1 : n_t+H].
$$

For two minutes,

$$
H = 120 \cdot 4 = 480.
$$

Thus,

$$
Y_t^+ \in \mathbb{R}^{480}.
$$

The dataset must provide an exact mapping from decimated feature time $t$ to raw cutoff index $n_t$.

---

# 5. Important limitation of starting with the current features

You can absolutely begin with the existing scattering and phase-harmonic inputs. This is the fastest way to validate:

* Whether the prior latent becomes an informative FHR representation.
* Whether the shared raw decoder works.
* Whether UP improves the raw forecast.
* Whether removing `decoder_state` changes latent quality.
* Whether the model remains trainable.

However, there are two separate goals:

## Engineering goal

Validate the new latent and decoder architecture.

The current features are sufficient for this purpose.

## Transfer-entropy validity

Claim that

$$
D_{\mathrm{KL}}(q_t \,\|\, p_t)
$$

is a defensible TE surrogate.

The current centered scattering and phase-harmonic features do not yet satisfy this requirement because some features at time $t$ contain raw samples after $t$. Causal neural layers cannot remove information already present in their inputs.

Therefore, the recommended development sequence is:

### Version 1A: architecture validation

$$
\text{existing ST/PH inputs} \rightarrow \text{new complete latent} \rightarrow \text{raw FHR prediction}.
$$

Use this to validate representation learning and raw decoding.

The source KL should be labeled something such as:

* `source_conditioned_kl`,
* `latent_source_rate`, or
* `provisional_teb_rate`.

Do not yet present it as a final TE measurement.

### Version 1B: scientifically valid feature-input model

Replace or adjust the existing transforms so that every feature at time $t$ depends only on raw samples available by its defined cutoff.

### Version 2

Add the raw causal input branches and compare:

$$
\text{ST/PH only}, \qquad \text{raw only}, \qquad \text{raw + ST/PH}.
$$

This progression is reasonable. There is no need to implement the raw-input encoder before validating the new latent architecture.

---

# 6. Target encoder

The target encoder receives only FHR features:

$$
H_t^Y = E_Y\left(X_{\le t}^Y\right).
$$

You can preserve most of your existing causal encoder:

1. FHR input adapter.
2. Causal dilated convolution branch.
3. Unidirectional LSTM branch.
4. Fusion MLP.
5. Per-time-step output normalization.

For example,

$$
\widetilde X_t^Y = \operatorname{InputAdapter}_Y(X_t^Y),
$$

$$
H_{t,\mathrm{conv}}^Y = E_{\mathrm{conv}}^Y\left(\widetilde X_{\le t}^Y\right),
$$

$$
H_{t,\mathrm{rnn}}^Y = E_{\mathrm{LSTM}}^Y\left(\widetilde X_{\le t}^Y\right),
$$

$$
H_t^Y = \operatorname{LayerNorm}\left[F_Y\left(H_{t,\mathrm{conv}}^Y \,\Vert\, H_{t,\mathrm{rnn}}^Y\right)\right].
$$

A suitable width remains

$$
d_{\mathrm{model}} = 128.
$$

`causal_norm=True` must remain enabled. It prevents future mixing inside the neural encoder, although it does not address the upstream transform leakage.

---

# 7. FHR prior latent

The target encoder produces the prior:

$$
p_t(z) = p_\theta(z_t \mid Y_{\le t}) = \mathcal{N}\left(\mu_t^p, \operatorname{diag}\left((\sigma_t^p)^2\right)\right).
$$

Compute

$$
\widetilde\mu_t^p = f_{\mu,p}(H_t^Y),
$$

$$
\widetilde\ell_t^p = f_{\ell,p}(H_t^Y).
$$

Use smooth bounds:

$$
\mu_t^p = a_\mu \tanh\left(\frac{\widetilde\mu_t^p}{a_\mu}\right),
$$

$$
\ell_t^p = \ell_{\min} + (\ell_{\max}-\ell_{\min}) \sigma(\widetilde\ell_t^p),
$$

where

$$
\ell_t^p = \log(\sigma_t^p)^2.
$$

This distribution is now the complete target-only predictive representation.

The deterministic representation used for analysis is

$$
\boxed{
\bar z_t^Y = \mu_t^p.
}
$$

This should replace `decoder_state` as the representation used for:

* Per-time-step representation analysis.
* Segment pooling.
* Downstream classification.
* Latent probes.
* Visualization.
* Similarity and clustering analyses.

---

# 8. Source encoder and lag attention

The source encoder receives only UP features:

$$
H_t^U = E_U\left(X_{\le t}^U\right).
$$

You can initially retain the existing source encoder and lag-attention implementation.

The target query is derived from $H_t^Y$ or $\mu_t^p$:

$$
Q_t = W_Q H_t^Y
$$

or preferably

$$
Q_t = W_Q \mu_t^p.
$$

For lag $\ell$,

$$
K_{t,\ell} = W_K H_{t-\ell}^U, \qquad V_{t,\ell} = W_V H_{t-\ell}^U.
$$

The lag-attention score is

$$
a_{t,m,\ell} = \frac{\langle Q_t^{(m)}, K_{t,\ell}^{(m)} \rangle}{\sqrt{d_h}} + b_{m,\ell}.
$$

After masking invalid lags,

$$
\alpha_{t,m,\ell} = \operatorname{entmax}_\ell\left(a_{t,m,\ell}\right).
$$

The attended source representation is

$$
A_t^{(m)} = \sum_\ell \alpha_{t,m,\ell}\, V_{t,\ell}^{(m)}.
$$

---

# 9. The UP-conditioned posterior

The posterior occupies the same latent coordinate system as the prior:

$$
q_t(z) = q_\phi\left(z_t \mid Y_{\le t}, U_{\le t}\right).
$$

It should remain a residual update around the prior:

$$
\boxed{
\mu_t^q = \mu_t^p + a_{\Delta\mu} \tanh\left(\frac{\widetilde{\Delta\mu}_t}{a_{\Delta\mu}}\right).
}
$$

For variance, add the update to the raw prior log-variance:

$$
\widetilde\ell_t^q = \widetilde\ell_t^p + a_{\Delta\ell} \tanh\left(\frac{\widetilde{\Delta\ell}_t}{a_{\Delta\ell}}\right),
$$

and then apply the smooth bound:

$$
\ell_t^q = \operatorname{smoothbound}\left(\widetilde\ell_t^q\right).
$$

The residual heads receive:

$$
\widetilde{\Delta\mu}_t, \widetilde{\Delta\ell}_t = F_{\mathrm{post}}\left(H_t^Y, A_t\right).
$$

Zero-initialize the two residual output heads:

$$
\widetilde{\Delta\mu}_t = 0, \qquad \widetilde{\Delta\ell}_t = 0
$$

at initialization.

Therefore,

$$
q_t(z) = p_t(z)
$$

and

$$
D_{\mathrm{KL}}(q_t \,\|\, p_t) = 0
$$

at initialization.

The deterministic full representation is

$$
\boxed{
\bar z_t^{YU} = \mu_t^q.
}
$$

The additive UP-derived update is

$$
\boxed{
\Delta\bar z_t^{U \rightarrow Y} = \mu_t^q - \mu_t^p.
}
$$

This produces exactly the three representations needed by evaluation:

| Representation | Interpretation |
| --- | --- |
| $\mu_t^p$ | FHR-only representation |
| $\mu_t^q-\mu_t^p$ | additional UP information |
| $\mu_t^q$ | FHR representation after incorporating UP |

---

# 10. Paired latent sampling

Draw one noise sample:

$$
\epsilon_t \sim \mathcal{N}(0,I).
$$

Use it for both prior and posterior:

$$
z_t^p = \mu_t^p + \sigma_t^p \odot \epsilon_t,
$$

$$
z_t^q = \mu_t^q + \sigma_t^q \odot \epsilon_t.
$$

Using common noise has two benefits:

1. If $p_t = q_t$, then

$$
z_t^p = z_t^q
$$

exactly.

2. The difference between base and full predictions has lower Monte Carlo variance because it is driven by the distributional update rather than unrelated random draws.

---

# 11. One shared raw decoder

The decoder must receive only $z$:

$$
D_\psi(z_t).
$$

There must be no call analogous to

```python
baseline_decoder(decoder_state)
residual_decoder(decoder_state, z)
```

Instead:

$$
(\mu_t^{\mathrm{base}}, \ell_t^{\mathrm{base}}) = D_\psi(z_t^p),
$$

$$
(\mu_t^{\mathrm{full}}, \ell_t^{\mathrm{full}}) = D_\psi(z_t^q).
$$

The exact same decoder parameters are used twice.

This is the architectural change that turns $z_t$ into the FHR representation.

If $z_t^p$ does not contain the target state, the baseline forecast fails. There is no other path through which the decoder can obtain FHR history.

---

# 12. Raw decoder geometry

For a two-minute future,

$$
H = 480 = 30 \times 16.
$$

Project the latent:

$$
h_t = W_z z_t + b_z.
$$

For horizon block $k = 1, \ldots, 30$,

$$
F_{t,k}^{(0)} = h_t + e_k,
$$

where $e_k$ is a learned horizon embedding.

Process the 30 tokens with a small horizon network:

$$
F_t = \operatorname{HorizonCore}\left(F_t^{(0)}\right).
$$

Each token produces 16 raw samples:

$$
\mu_{t,k} = W_\mu F_{t,k} \in \mathbb{R}^{16},
$$

$$
\ell_{t,k} = W_\ell F_{t,k} \in \mathbb{R}^{16}.
$$

After reshaping:

$$
\mu_t, \ell_t \in \mathbb{R}^{480}.
$$

The decoder can reuse most of your current horizon-core ideas:

* Horizon embeddings.
* Dilated convolution over horizon tokens.
* FiLM conditioning.
* Shared output heads.
* Parallel, nonautoregressive prediction.

---

# 13. The UP shift applied at dataset creation is part of the signal

The dataset builder constructs its adaptor with `up_shift_secs=-20`, which moves the UP trace
$20$ s earlier before any transform is taken. **Downstream, that is the end of the story.** The stored UP/FHR timeline is canonical: the dataset builder shifts the UP channel when it writes the shards, that shift is part of how the stored signals are, and nothing downstream adds it back, subtracts it, budgets it or interprets it.

On the stored timeline an attention peak at lag $\ell$, with input delay $\delta$, is

$$
\boxed{
\tau_{\mathrm{compensated}} = 4(\ell + \delta)\ \mathrm{s}.
}
$$

There is no "raw sensor" figure to report beside it. An earlier revision of this section, and
of `lag_report.py`, defined $\tau_{\mathrm{raw\ sensor}} = 4\ell - 20$ s and a
`lag_original_sensor_seconds` helper; both were removed on 2026-09-05 and must not return.

---

# 14. Minimal starting objective

Yes, most loss components should initially be zero.

The recommended initial loss is only:

$$
\boxed{
\mathcal{L} = \lambda_{\mathrm{full}} \mathcal{L}_{\mathrm{full}} + \lambda_{\mathrm{base}} \mathcal{L}_{\mathrm{base}} + \beta D_{\mathrm{KL}}\left(q_t(z) \,\|\, p_t(z)\right) + \beta_p R_p .
}
$$

This is sufficient to test the central hypothesis. The fourth term is the prior's scale anchor, added after the three-term form was built and run: it is what keeps $D_{\mathrm{KL}}(q_t \,\|\, p_t)$ a rate rather than a number divided by a collapsing $\sigma_p^2$. Section 15 gives its form, its measured motivation and why only the scale half of a context rate is taken.

## Full future loss

$$
\mathcal{L}_{\mathrm{full}} = -\log p_\psi\left(Y_t^+ \mid z_t^q\right).
$$

This trains the UP-updated latent to predict raw future FHR.

## Baseline future loss

$$
\mathcal{L}_{\mathrm{base}} = -\log p_\psi\left(Y_t^+ \mid z_t^p\right).
$$

This forces the prior latent itself to contain the FHR predictive representation.

This term is essential. It cannot initially be zero.

Without it, the target-only prior may remain weak, and the posterior may learn to place both target and source information in the source-conditioned update.

## Conditional source KL

$$
\mathcal{L}_{\mathrm{KL}} = D_{\mathrm{KL}}\left(q_\phi(z_t \mid Y_{\le t}, U_{\le t}) \,\|\, p_\theta(z_t \mid Y_{\le t})\right).
$$

For diagonal Gaussians,

$$
\mathcal{L}_{\mathrm{KL}} = \frac12 \sum_{j=1}^{d_z} \left[\ell_{t,j}^p - \ell_{t,j}^q + \frac{e^{\ell_{t,j}^q} + (\mu_{t,j}^q-\mu_{t,j}^p)^2}{e^{\ell_{t,j}^p}} - 1\right].
$$

This encourages UP to change the latent only when the change is useful enough to improve future prediction.

---

# 15. Losses to set to zero initially

For the first implementation, set:

$$
\lambda_{\mathrm{past}} = 0, \qquad
\lambda_{\mathrm{future\ teacher}} = 0, \qquad
\lambda_{\mathrm{multiscale}} = 0,
$$

$$
\lambda_{\mathrm{derivative}} = 0, \qquad
\lambda_{\mathrm{boundary}} = 0, \qquad
\lambda_{\mathrm{state}} = 0,
$$

$$
\lambda_{\mathrm{lag\ smoothness}} = 0, \qquad
\lambda_{\mathrm{permutation}} = 0.
$$

The permutation control should still run during evaluation, but it should not contribute to the training objective.

**$\beta_{\mathrm{target\ prior}} = 0$ was on that list and has been removed from it.** It was wrong, and the correction is recorded here rather than edited away silently, because the instruction was followed and the consequence was measured. With every anchor on the target prior at zero, nothing in the objective penalises a *narrow* prior: the two reconstruction terms strictly prefer a deterministic latent, since sampling noise can only degrade a forecast, and the source-conditioned KL measures the posterior *against* the prior without constraining the prior's own scale. Unlike a fixed-prior VAE — where $D_{\mathrm{KL}}(q \,\|\, \mathcal{N}(0,I))$ anchors the latent scale — the prior log-variance is then free to fall until it meets its clamp. It did. The first production run of the three-term system reached a prior-log-variance floor fraction of $0.992$ against a floor of $-5$, from $0.118$ at epoch $0$ through $0.812$ at epoch $1$: the collapse completes inside one epoch. At that point the coupling readout stops being a rate, because the KL carries $(\mu^q - \mu^p)^2 / \sigma_p^2$ and a floored $\sigma_p^2$ multiplies it by an arbitrary factor; that run reported $1.70$ nats of latent displacement costing $7.37$ nats of forecast score, a $4.3\times$ amplification.

What ships instead is a **scale anchor**, not §4.1's $R_C$ and not the full context rate either:

$$
R_p(t) = \sum_{d} \tfrac{1}{2}\left(e^{\ell^p_{t,d}} - 1 - \ell^p_{t,d}\right)
= D_{\mathrm{KL}}\left(\mathcal{N}\!\left(\mu^p_t, \operatorname{diag} e^{\ell^p_t}\right) \,\Big\|\, \mathcal{N}\!\left(\mu^p_t, I\right)\right),
$$

weighted by a constant $\beta_p$ and reduced on the KL's own anchor support, in the same nats-per-anchor units. Three relations to the terms above place it exactly:

* Against **§4.1's $R_C$**: that term is a Conditional Entropy Bottleneck divergence against a training-only *future* encoder $b_C$, which is the right formulation of "compress the past that the future does not explain" but requires a second encoder that was never built. $R_p$ needs no future encoder at all.
* Against the **full context rate** $D_{\mathrm{KL}}(p_\theta \,\|\, \mathcal{N}(0,I))$: that separates exactly into $R_p + \tfrac{1}{2}\sum_d (\mu^p_{t,d})^2$, and $R_p$ is its **scale half**. Only that half is adopted. The second half compresses the prior *mean*, which is the entire content of the base forecast, so weighting it trades $D_0$ away to buy the fix and turns a scale pathology into a target-side rate–distortion question. Since the mean was not what collapsed, the half that ships is the half the failure demanded — and being a strict subset, the rest can be adopted later as an addition rather than a revision.
* Against **$\beta_C$ and $\beta_U$** of the complete training objective (§15 of the design part, "Complete training objective"): $\beta_p$ is neither. It weights a property of the prior's *scale*, not an information rate between two variables, and it is a **constant rather than a schedule** — a warm-up would arrive after a collapse that completes in one epoch.

This gives the minimal system:

$$
\boxed{
\text{base raw prediction} + \text{full raw prediction} + \text{source-conditioned KL} + \text{prior scale anchor}.
}
$$

Four terms, not three. The fourth is off by default in code ($\beta_p = 0$ leaves the historical three-term sum exactly), so it is an opt-in weight rather than a structural change — but the shipped configuration opts in, at $\beta_p = 0.1$, and $R_p$ is computed and logged whatever the weight, so a collapsing prior is visible in any run.

---

# 16. Recommended initial weights

A reasonable initial relationship is

$$
\lambda_{\mathrm{full}} = 1, \qquad \lambda_{\mathrm{base}} = 1.
$$

Giving the baseline the same initial weight as the full forecast emphasizes that the prior latent must be independently useful.

Do not immediately reuse the existing final KL weight of $0.1$. Its meaning changes because:

* The decoder output changes from features to raw samples.
* The loss reduction changes.
* The latent now carries the complete FHR representation.
* There is no high-dimensional target bypass.

Define both reconstruction losses and KL in consistent units, preferably nats per anchor:

$$
\mathcal{L}_{\mathrm{NLL}} = \sum_{h=1}^{H} -\log p(y_{t,h} \mid z_t),
$$

$$
\mathcal{L}_{\mathrm{KL}} = \sum_{j=1}^{d_z} \operatorname{KL}_j.
$$

Average over batch and valid anchors only.

Then run a small $\beta$ sweep.

A safe procedure is:

$$
\beta(e) = \beta_{\max} \min\left(1, \frac{e}{E_{\mathrm{warm}}}\right),
$$

starting at

$$
\beta(0) = 0.
$$

Select $\beta_{\max}$ based on:

* Raw KL trajectory.
* Base/full NLL.
* Active latent dimensions.
* Source uplift.
* Permutation specificity.

Initially use:

$$
\text{free bits} = 0.
$$

Free bits can be introduced later if posterior collapse is observed.

---

# 17. MSE or probabilistic NLL?

For a basic implementation smoke test, use MSE:

$$
\mathcal{L}_{\mathrm{MSE}} = \frac1H \sum_{h=1}^H \left(Y_t^+[h]-\mu_t[h]\right)^2.
$$

Once the architecture trains correctly, move to a learned Gaussian NLL:

$$
\mathcal{L}_{\mathrm{NLL}} = \frac12 \sum_{h=1}^H \left[\ell_t[h] + \frac{(Y_t^+[h]-\mu_t[h])^2}{e^{\ell_t[h]}}\right].
$$

The Gaussian NLL is preferable for the scientific model because:

* Exact raw high-frequency samples are not fully predictable from decimated ST/PH features.
* The output variance can represent irreducible uncertainty.
* Predictive log-likelihood improvement has a more direct relationship to TE than MSE improvement.

A sensible implementation sequence is:

1. MSE until the architecture passes tests.
2. Fixed-variance Gaussian NLL.
3. Learned heteroscedastic Gaussian NLL.
4. More expressive likelihood only if calibration remains inadequate.

---

# 18. Minimum evaluation outputs

Even with the simplified loss, return and analyze:

$$
\mu_t^p, \qquad \mu_t^q, \qquad \mu_t^q-\mu_t^p,
$$

$$
K_t = D_{\mathrm{KL}}(q_t \,\|\, p_t),
$$

$$
\mathcal{L}_{\mathrm{base}}, \qquad \mathcal{L}_{\mathrm{full}}, \qquad \mathcal{L}_{\mathrm{shuffled}},
$$

$$
\alpha_{t,m,\ell}, \qquad \widetilde K_{t,\ell}.
$$

The key predictive criteria are:

$$
\mathcal{L}_{\mathrm{full}} < \mathcal{L}_{\mathrm{base}}
$$

and

$$
\mathcal{L}_{\mathrm{full}} < \mathcal{L}_{\mathrm{base}} < \mathcal{L}_{\mathrm{shuffled}}.
$$

The key representation criteria are:

* $\mu^p$ supports accurate future prediction.
* Shuffling $\mu^p$ seriously damages the baseline prediction.
* $\mu^q-\mu^p$ is small but useful.
* The full latent $\mu^q$ improves future and downstream probes.
* The KL does not collapse into only one or two dimensions.

---

# 19. Revised implementation roadmap

## Phase 1: feature-input/raw-output architecture

Use the existing:

* FHR scattering.
* FHR phase harmonics.
* UP scattering.
* UP phase harmonics.
* Current target and source encoders.
* Current lag-attention system.

Replace:

* `prior_head` so it produces the full FHR latent.
* `posterior_head` so it updates that same latent.
* Separate baseline/residual decoders with one shared raw decoder.
* Feature future targets with raw 480-sample future targets.
* `decoder_state` with no bypass.

Train with:

$$
\mathcal{L} = \mathcal{L}_{\mathrm{full}} + \mathcal{L}_{\mathrm{base}} + \beta \mathcal{L}_{\mathrm{KL}}.
$$

## Phase 2: latent-quality improvements

After the minimal model works, consider adding one component at a time:

1. Future teacher latent.
2. Weak target-only past reconstruction.
3. Multiscale raw loss.
4. Boundary and derivative losses.
5. Controlled target latent compression.

Each addition should survive an ablation.

## Phase 3: causal input correction

Implement:

* Causal scattering and phase harmonics, or
* A raw causal input encoder.

Only after this phase should the source KL be treated as a defensible TE surrogate.

## Phase 4: raw-input comparison

Compare:

$$
\text{ST/PH input}, \qquad \text{raw input}, \qquad \text{raw + ST/PH input}.
$$

---

# Final recommendation

For the first new model, remove the $C_t$/$S_t$ split and use:

$$
\boxed{
p_\theta(z_t \mid Y_{\le t})
}
$$

as the complete FHR representation and

$$
\boxed{
q_\phi(z_t \mid Y_{\le t}, U_{\le t})
}
$$

as the same representation updated by UP.

Parameterize:

$$
\boxed{
\mu_t^q = \mu_t^p + \Delta\mu_t^{U \rightarrow Y}.
}
$$

Use one shared decoder:

$$
\boxed{
\widehat Y_t^{\mathrm{base}} = D(z_t^p), \qquad \widehat Y_t^{\mathrm{full}} = D(z_t^q).
}
$$

Start with existing scattering and phase-harmonic inputs and predict the next 480 raw FHR samples. Treat this first phase as validation of the latent and raw decoder architecture; causalize or replace the input representation before making final TE claims. The 20-second UP correction should remain as mechanical alignment, while lag attention discovers the remaining physiological delay.

The minimal objective should be:

$$
\boxed{
\mathcal{L} = \mathcal{L}_{\mathrm{full}} + \mathcal{L}_{\mathrm{base}} + \beta D_{\mathrm{KL}}(q_t \,\|\, p_t).
}
$$

Everything else can initially be zero.
