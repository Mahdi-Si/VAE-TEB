# 1. Goal of the first-iteration model

The purpose of the first model is **not** to solve every aspect of the problem at once. It is to build the cleanest possible self-supervised system that learns a transferable latent representation of:

1. the intrinsic fetal state from $\text{FHR}$,
2. the predictive contribution of uterine activity from $\text{UP}$,
3. the evolution of that combined state over the $20$-minute window.

The downstream target is classification of infants into:

* healthy,
* unhealthy,
* and possibly severity levels of unhealthy.

So the representation should preserve structure that is clinically meaningful for those tasks:

* baseline and slow drift,
* variability regime,
* deceleration/recovery patterns,
* sensitivity to uterine activity,
* repeated abnormal response burden,
* failure of recovery or progressive deterioration.

The representation should **not** waste most of its capacity on:

* exact reconstruction of every input coefficient,
* trivial short-term smoothness,
* transform redundancy,
* source-only uterine dynamics that do not change future $\text{FHR}$.

That leads to a very specific first-iteration design:

## First-iteration design choice

Use a **causal multimodal forecasting model** with

* a strong $\text{FHR}$-only predictive pathway,
* a fused $\text{FHR}+\text{UP}$ predictive pathway,
* and a small local TE-style latent that captures only the **incremental predictive contribution of $\text{UP}$ beyond $\text{FHR}$'s own past**.

This is the right first version because it is:

* clean,
* causal,
* aligned with the downstream goal,
* simpler than a full VAE,
* and better controlled than a pure TEB bottleneck.

---

# 2. Data and notation

Each pretraining example is one effective feature window of length

$$
T = 300.
$$

This corresponds to the $20$-minute effective window after decimation and trimming.

Let

$$
Y_{1:T} \in \mathbb{R}^{T \times d_F}
$$

denote the $\text{FHR}$ scattering-transform sequence, and let

$$
U_{1:T} \in \mathbb{R}^{T \times d_U}
$$

denote the $\text{UP}$ scattering-transform sequence.

Here:

* $d_F$ is the number of $\text{FHR}$-ST channels,
* $d_U$ is the number of $\text{UP}$-ST channels.

For a minibatch of size $B$, the tensors are

$$
Y \in \mathbb{R}^{B \times T \times d_F},
\qquad
U \in \mathbb{R}^{B \times T \times d_U}.
$$

The $\text{UP}$ stream is assumed already shifted by $-20$ seconds according to your preprocessing choice.

---

# 3. What representation this model is trying to learn

The learned window representation should contain three conceptual pieces:

$$
e_{\text{win}} = \big[
e^{F}, \;
e^{FU}, \;
e^{TE}
\big].
$$

These mean:

* $e^{F}$: intrinsic $\text{FHR}$ state,
* $e^{FU}$: causal multimodal predictive state,
* $e^{TE}$: uterine-to-fetal incremental predictive influence.

More concretely:

### $e^{F}$ should preserve

* fetal baseline regime,
* variability regime,
* autonomous decelerative behavior,
* recovery patterns,
* slow deterioration visible from $\text{FHR}$ alone.

### $e^{FU}$ should preserve

* how future $\text{FHR}$ evolves when both $\text{FHR}$ and $\text{UP}$ history are known,
* repeated stress-response structure,
* cumulative predictive burden across the window.

### $e^{TE}$ should preserve

* what $\text{UP}$ adds about future $\text{FHR}$ that is **not already predictable from past $\text{FHR}$**.

This decomposition matters. The downstream classification problem is not purely about transfer entropy and not purely about $\text{FHR}$ self-dynamics. It depends on both.

---

# 4. Why this first model is strictly causal

The first version should be strictly causal for two reasons.

### Reason 1: conceptual clarity

You want the self-supervised task to match the intended causal question:

> Given the past, what is the future $\text{FHR}$, and what additional information does uterine history provide?

A bidirectional model blurs that.

### Reason 2: cleaner TE auxiliary branch

The TE-style latent only makes sense if both the self-only baseline and the source-conditioned posterior are built from past information only.

So in the first version:

* no bidirectional encoder,
* no masked reconstruction,
* no global stochastic VAE bottleneck.

Everything is causal.

---

# 5. Input normalization

Use robust channelwise normalization computed on the training split only.

For each $\text{FHR}$ channel $c$, let $m_c^{(F)}$ be the training median and $s_c^{(F)}$ be a robust scale such as IQR or MAD-derived scale. Similarly for $\text{UP}$.

Then normalize:

$$
\tilde Y_{t,c} = \frac{Y_{t,c} - m_c^{(F)}}{s_c^{(F)} + \epsilon},
\qquad
\tilde U_{t,c} = \frac{U_{t,c} - m_c^{(U)}}{s_c^{(U)} + \epsilon}.
$$

This is better than per-window normalization because per-window normalization would erase clinically meaningful slow differences in level and variability regime.

The normalized batch tensors are still

$$
\tilde Y \in \mathbb{R}^{B \times T \times d_F},
\qquad
\tilde U \in \mathbb{R}^{B \times T \times d_U}.
$$

---

# 6. Architecture overview

The first-iteration model has five major components:

1. **$\text{FHR}$ causal stem**
2. **$\text{UP}$ causal stem**
3. **$\text{FHR}$-only causal encoder**
4. **$\text{FHR} \leftarrow \text{UP}$ gated causal fusion encoder**
5. **anchor-based TE auxiliary branch**

The model has three forecasting heads:

* self-only head,
* fused head,
* TE residual head.

This structure is deliberate. These heads are not redundant.

---

# 7. Causal stems

## 7.1 Purpose of the stems

The stems are not there to do long-range reasoning. They are there to build better local tokens before attention.

They should learn local motifs such as:

* local $\text{FHR}$ slope change,
* local variability suppression,
* local contraction onset / rise / fall,
* short recovery patterns.

These are all local temporal patterns that a causal convolutional block can capture efficiently.

## 7.2 Why causal conv does not violate causality

The stems use **causal 1D convolutions**, not symmetric convolutions.

For a kernel size $k$, a causal conv at time $t$ computes

$$
h_t = \sum_{i=0}^{k-1} W_i x_{t-i},
$$

so it depends only on present and past inputs, never future inputs.

If dilation $\delta$ is used, it becomes

$$
h_t = \sum_{i=0}^{k-1} W_i x_{t-i\delta}.
$$

Again, strictly past-only.

So causality is preserved.

## 7.3 Stem architecture

Let the backbone width be

$$
d = 192.
$$

### $\text{FHR}$ stem

First project channels:

$$
H_F^{(0)} = W_F \tilde Y + b_F,
\qquad
H_F^{(0)} \in \mathbb{R}^{B \times T \times d}.
$$

Then apply $L_s=3$ residual causal-convolution blocks:

$$
H_F^{(\ell+1)} = H_F^{(\ell)}
+
\mathcal{C}_F^{(\ell)}\!\big(H_F^{(\ell)}\big),
\qquad \ell=0,1,2.
$$

A good form for each block is

$$
\mathcal{C}(X) = \operatorname{Dropout}
\Big(
W_2 \,
\phi\big(
W_1 \,
\operatorname{DWConv}_{\text{causal}}(\operatorname{LN}(X))
\big)
\Big),
$$

where:

* $\operatorname{DWConv}_{\text{causal}}$ is a depthwise causal 1D convolution,
* $W_1$ is a pointwise projection $d \to 2d$,
* $\phi$ is GELU or SiLU,
* $W_2$ is a pointwise projection $2d \to d$.

Use increasing receptive fields, for example:

* block 1: kernel $3$, dilation $1$,
* block 2: kernel $5$, dilation $2$,
* block 3: kernel $5$, dilation $4$.

Then define the stem output

$$
F = H_F^{(3)} \in \mathbb{R}^{B \times T \times d}.
$$

### $\text{UP}$ stem

Use the same structure:

$$
H_U^{(0)} = W_U \tilde U + b_U,
\qquad
H_U^{(0)} \in \mathbb{R}^{B \times T \times d},
$$

and

$$
H_U^{(\ell+1)} = H_U^{(\ell)}
+
\mathcal{C}_U^{(\ell)}\!\big(H_U^{(\ell)}\big),
\qquad \ell=0,1,2.
$$

Then define

$$
S = H_U^{(3)} \in \mathbb{R}^{B \times T \times d}.
$$

---

# 8. Causal temporal encoders

The stems produce local modality-specific tokens. Next, we need causal encoders that perform longer-range reasoning.

## 8.1 $\text{FHR}$-only causal encoder

Define

$$
H_F^c = E_F^c(F),
\qquad
H_F^c \in \mathbb{R}^{B \times T \times d}.
$$

This encoder sees only the $\text{FHR}$ stream and learns a causal temporal state summarizing past fetal dynamics.

It should answer:

> what does past $\text{FHR}$ alone predict about future $\text{FHR}$?

This pathway is crucial because clinical abnormality is not only about uterine coupling. A great deal of clinically relevant signal is already present in $\text{FHR}$ alone.

## 8.2 $\text{UP}$-only causal encoder

Define

$$
H_U^c = E_U^c(S),
\qquad
H_U^c \in \mathbb{R}^{B \times T \times d}.
$$

This encoder summarizes past uterine dynamics in a form suitable for cross-attention and the TE branch.

## 8.3 Implementation choice

A good first implementation is $4$ causal transformer blocks for each encoder:

* width $d=192$,
* $4$ attention heads,
* pre-norm blocks,
* dropout $0.1$.

This is a good balance: expressive enough to capture minute-scale dependence, but not too large for a first build.

---

# 9. Causal fusion pathway

This is the main multimodal pathway.

Its purpose is to construct a causal state that answers:

> what is the future $\text{FHR}$ if I use both $\text{FHR}$ history and uterine history?

## 9.1 Why not just concatenate $\text{FHR}$ and $\text{UP}$?

Because uterine information should not be injected indiscriminately at every time point. There are many times when the predictive content is mostly intrinsic $\text{FHR}$, and uterine information may add little.

So the model should decide, time by time:

* whether uterine context matters,
* and how strongly it should influence the fused state.

That is why we use **cross-attention plus gating**, not raw concatenation only.

## 9.2 Source-derived context by cross-attention

At each time $t$, use the $\text{FHR}$ state as the query and $\text{UP}$ states as the source memory.

Let

$$
Q_t = W_Q H_{F,t}^c,
\qquad
K_s = W_K H_{U,s}^c,
\qquad
V_s = W_V H_{U,s}^c.
$$

Then compute causal source attention

$$
\alpha_{t,s} = \operatorname{softmax}
\left(
\frac{Q_t K_s^\top}{\sqrt{d_k}} + M_{t,s}
\right),
$$

where $M_{t,s}$ enforces causality, so only $s \le t$ are visible.

Then the attended $\text{UP}$-derived context is

$$
C_t = \sum_{s \le t} \alpha_{t,s} V_s,
\qquad
C_t \in \mathbb{R}^{d}.
$$

### Do we need an explicit lag restriction?

Not strictly. Causality already enforces directionality.

You can optionally restrict attention to the recent past

$$
s \in [t-L_{\text{lag}}, t]
$$

for efficiency and physiological bias, but the first clean implementation can simply use causal attention over all past source steps.

## 9.3 Gated fusion

Now define a gate

$$
G_t = \sigma\!\big(
W_g [H_{F,t}^c \,|\, C_t] + b_g
\big),
\qquad
G_t \in (0,1)^d.
$$

Then construct the fused state by residual gated addition:

$$
\tilde H_t^c = H_{F,t}^c + G_t \odot C_t.
$$

This has the right interpretation:

* $H_{F,t}^c$ = intrinsic fetal state,
* $C_t$ = source-derived uterine context relevant to that state,
* $G_t$ = how much uterine context should matter right now,
* $\tilde H_t^c$ = intrinsic state plus gated uterine contribution.

This is exactly what we want physiologically. The fetus has an intrinsic state, and uterine activity acts as a perturbing or revealing context, but it should not always dominate.

## 9.4 Fused causal encoder

Now pass the gated states through a causal fusion encoder:

$$
H_{FU}^c = E_{FU}^c(\tilde H^c),
\qquad
H_{FU}^c \in \mathbb{R}^{B \times T \times d}.
$$

This encoder integrates the local fused information over time and produces the main multimodal predictive state.

---

# 10. Anchor-based formulation

The model predicts future $\text{FHR}$ blocks from selected anchor times.

## 10.1 What is an anchor?

An anchor is just a time index $a$ where we ask the predictive question:

> given the history up to $a$, what is the future $\text{FHR}$ block?

Anchors are **not** event timestamps. You do not need event annotations.

## 10.2 Valid anchor range

Let

* $L_{\text{ctx}}$ = local context length,
* $g$ = guard gap,
* $h_{\max}$ = largest prediction horizon.

Then the valid anchors are

$$
\mathcal A_{\text{valid}} = \{a: L_{\text{ctx}} \le a \le T-g-h_{\max}\}.
$$

Use the following first-iteration settings:

$$
L_{\text{ctx}} = 30,
\qquad
g = 4,
\qquad
h_{\max} = 30.
$$

Then

$$
\mathcal A_{\text{valid}} = \{30,\dots,266\}.
$$

## 10.3 Should all time steps be used as anchors?

Not during training.

That would be computationally redundant and would overweight smooth easy regions.

Instead, for each window, sample a small number $K$ of anchors, for example

$$
K = 2 \text{ to } 4.
$$

## 10.4 Anchor sampling strategy

Use a mixture of:

* uniform sampling,
* activity-biased sampling.

Define a score

$$
s_t = \alpha |\tilde U_t|_1
+
\beta |\Delta \tilde U_t|_1
+
\gamma |\Delta \tilde Y_t|_1,
\qquad
t \in \mathcal A_{\text{valid}}.
$$

This is a proxy for "interesting" regions:

* high uterine activity,
* changing uterine dynamics,
* changing fetal dynamics.

Then sample anchors from

$$
p(a=t) = \eta \frac{1}{|\mathcal A_{\text{valid}}|}
+
(1-\eta)
\frac{s_t}{\sum_{j \in \mathcal A_{\text{valid}}} s_j}.
$$

A good first choice is

$$
\eta = 0.5.
$$

This means half the anchors are uniform, half are biased toward active regions.

That is the right first unsupervised anchor policy.

---

# 11. Future prediction targets

Use multiple horizons

$$
\mathcal H = \{8, 15, 30\}.
$$

Since each feature step corresponds to about $4$ seconds, these are roughly:

* $32$ s,
* $60$ s,
* $120$ s.

Let the guard gap be

$$
g = 4
$$

steps ($\approx 16$ seconds).

This gap is important because scattering coefficients have temporal support. Without a gap, very near-future prediction can become too easy and partially contaminated by local feature overlap.

For anchor $a$ and horizon $h$, define the target future block

$$
Y_{a,h}^{+} = Y_{a+g+1 : a+g+h}
\in \mathbb{R}^{h \times d_F}.
$$

This is the forecasting target in **$\text{FHR}$-ST space**.

---

# 12. Local summaries at each anchor

For each sampled anchor $a$, define local causal summaries.

## 12.1 Intrinsic $\text{FHR}$ summary

Using recent context pooling:

$$
s_a^F = \operatorname{AttnPool}
\big(
H_{F,a-L_{\text{ctx}}+1:a}^c
\big)
\in \mathbb{R}^{d}.
$$

## 12.2 $\text{UP}$ summary

$$
s_a^U = \operatorname{AttnPool}
\big(
H_{U,a-L_{\text{ctx}}+1:a}^c
\big)
\in \mathbb{R}^{d}.
$$

## 12.3 Fused summary

$$
s_a^{FU} = \operatorname{AttnPool}
\big(
H_{FU,a-L_{\text{ctx}}+1:a}^c
\big)
\in \mathbb{R}^{d}.
$$

These summaries are the inputs to the prediction heads and TE branch.

Why use pooled local context rather than just $H_a$?

Because the coupling question is about recent source/target history, not only the single instant at time $a$.

---

# 13. Prediction heads

The model uses three different forecasting heads. They are not redundant.

---

## 13.1 Self-only baseline head

This head predicts future $\text{FHR}$ using only the intrinsic $\text{FHR}$ summary:

$$
\hat Y_{a,h}^{self} = D_h^{self}(s_a^F),
\qquad
\hat Y_{a,h}^{self} \in \mathbb{R}^{h \times d_F}.
$$

This head defines the baseline predictive content of past $\text{FHR}$ alone.

This is essential because the TE-style auxiliary branch is defined relative to what is already predictable from $\text{FHR}$ history.

Without this head, you cannot cleanly ask what $\text{UP}$ adds beyond target self-history.

---

## 13.2 Main fused forecasting head

This head predicts future $\text{FHR}$ from the fused summary:

$$
\hat Y_{a,h}^{fus} = D_h^{fus}(s_a^{FU}),
\qquad
\hat Y_{a,h}^{fus} \in \mathbb{R}^{h \times d_F}.
$$

This is the **main representation-learning objective**.

This head should become the strongest forecaster, because it uses the full causal multimodal state.

Its purpose is broader than TE:

* it learns the main multimodal predictive latent,
* it captures both intrinsic fetal state and uterine-conditioned future evolution,
* it is the primary source of downstream transfer.

---

## 13.3 TE residual head

This head uses a TE-style local latent $z_a^{TE}$ to model only the **incremental contribution** of $\text{UP}$ beyond the self-only baseline.

It predicts a residual

$$
\hat R_{a,h} = D_h^{TE}([s_a^F \,|\, z_a^{TE}]),
\qquad
\hat R_{a,h} \in \mathbb{R}^{h \times d_F}.
$$

Then define the TE-augmented prediction

$$
\hat Y_{a,h}^{TE} = \hat Y_{a,h}^{self} + \hat R_{a,h}.
$$

This head is not intended to replace the main fused head. It is there specifically to make the TE latent meaningful.

---

# 14. TE-style local coupling latent

This is the part that encodes the uterine-to-fetal incremental predictive influence.

It is **local** because TE is fundamentally about information in recent source history that improves prediction of nearby future target behavior beyond target self-history.

It is **auxiliary** because the full clinically relevant representation should not be bottlenecked to only that information.

## 14.1 Posterior network

At anchor $a$, define the posterior input

$$
x_a^{post} = [s_a^F \,|\, s_a^U]
\in \mathbb{R}^{2d}.
$$

Then pass it through a small MLP:

$$
h_a^{post} = \operatorname{MLP}_{post}\big(\operatorname{LN}(x_a^{post})\big).
$$

Then output posterior parameters

$$
\mu_a = W_\mu h_a^{post},
\qquad
\log \sigma_a^2 = W_\sigma h_a^{post}.
$$

So

$$
q_\phi(z_a^{TE} \mid U_{\le a}, Y_{\le a}) = \mathcal N\!\big(
\mu_a, \operatorname{diag}(\sigma_a^2)
\big).
$$

Use a small latent dimension, e.g.

$$
d_z = 16.
$$

This simple concat-plus-MLP posterior is the right first implementation. The temporal encoders already did the hard sequence modeling. The posterior only needs to parameterize a compact conditional latent.

## 14.2 Conditional prior network

Define the prior from the intrinsic $\text{FHR}$ summary only:

$$
h_a^{prior} = \operatorname{MLP}_{prior}\big(\operatorname{LN}(s_a^F)\big),
$$

then

$$
\mu_a^0 = W_\mu^0 h_a^{prior},
\qquad
\log (\sigma_a^0)^2 = W_\sigma^0 h_a^{prior}.
$$

So

$$
r_\psi(z_a^{TE} \mid Y_{\le a}) = \mathcal N\!\big(
\mu_a^0, \operatorname{diag}((\sigma_a^0)^2)
\big).
$$

This prior is the correct TE-style baseline because it represents what the coupling latent would look like if only $\text{FHR}$ past were available.

## 14.3 Reparameterization

Sample

$$
z_a^{TE} = \mu_a + \sigma_a \odot \epsilon,
\qquad
\epsilon \sim \mathcal N(0, I).
$$

Then use $z_a^{TE}$ in the TE residual head.

---

# 15. Why these three heads are all needed

This is important enough to state explicitly.

## 15.1 Self-only head

Provides the conditional baseline:

$$
\text{future from } Y_{\le a} \text{ only}.
$$

This corresponds to the conditioning term in the TE idea.

## 15.2 Fused head

Provides the main multimodal predictive representation:

$$
\text{future from } (Y_{\le a}, U_{\le a}) \text{ via full multimodal state}.
$$

This is the main self-supervised training signal for the overall encoder.

## 15.3 TE residual head

Provides the source-incremental prediction channel:

$$
\text{extra future information due specifically to } U_{\le a} \text{ beyond } Y_{\le a}.
$$

This is the only head tied directly to the KL bottleneck and the TE idea.

So the relation is:

* self-only head defines the baseline,
* fused head learns the main representation,
* TE residual head gives the local transfer-entropy-like subspace its meaning.

---

# 16. Loss functions

The loss should reflect the structure above.

We do **not** want plain full reconstruction.
We do **not** want a global VAE KL.
We do **not** want the TE branch to dominate.

We want the main pressure to be on future prediction, with a smaller auxiliary pressure on the TE subspace.

---

## 16.1 Robust forecast loss

Use a Huber loss on prediction errors.

Define elementwise Huber with threshold $\delta$:

$$
\operatorname{Huber}_\delta(r) = \begin{cases}
\frac{1}{2}r^2, & |r|\le \delta,\\[4pt]
\delta\left(|r|-\frac{1}{2}\delta\right), & |r|>\delta.
\end{cases}
$$

For tensors $A$ and $\hat A$, define

$$
\rho(\hat A, A) = \frac{1}{|A|}\sum_i \operatorname{Huber}_\delta(\hat A_i - A_i).
$$

Use this instead of plain MSE because:

* the features can be heavy-tailed,
* abnormal regions can generate larger errors,
* Huber is robust while still sensitive to meaningful deviations.

---

## 16.2 Main fused forecasting loss

Define

$$
\mathcal L_{fus} = \frac{1}{|\mathcal A|}
\sum_{a \in \mathcal A}
\sum_{h \in \mathcal H}
w_h \,
\rho(\hat Y_{a,h}^{fus}, Y_{a,h}^{+}).
$$

The horizon weights should slightly emphasize longer horizons, for example:

$$
w_8 = 1,\qquad
w_{15} = 1.5,\qquad
w_{30} = 2.
$$

This is because the short horizon is easiest and least informative.

This is the **main pretraining loss**.

---

## 16.3 Dynamics loss

To prevent the model from succeeding mainly by smooth persistence, also penalize temporal differences.

Within each future block define

$$
\Delta Y_{a,h}^{+}[k] = Y_{a+g+k} - Y_{a+g+k-1},
\qquad
k=2,\dots,h.
$$

Similarly define $\Delta \hat Y_{a,h}^{fus}$.

Then

$$
\mathcal L_{\Delta} = \frac{1}{|\mathcal A|}
\sum_{a \in \mathcal A}
\sum_{h \in \mathcal H}
w_h \,
\rho(\Delta \hat Y_{a,h}^{fus}, \Delta Y_{a,h}^{+}).
$$

This encourages the model to preserve:

* slope changes,
* onset,
* recovery,
* variability changes,

which are highly relevant clinically.

---

## 16.4 Self-only baseline loss

Train the self-only head with

$$
\mathcal L_{self} = \frac{1}{|\mathcal A|}
\sum_{a \in \mathcal A}
\sum_{h \in \mathcal H}
w_h \,
\rho(\hat Y_{a,h}^{self}, Y_{a,h}^{+}).
$$

This ensures the intrinsic $\text{FHR}$-only summary is a strong predictive representation on its own.

This matters because the self-only branch is not just a nuisance baseline. It should genuinely learn clinically meaningful $\text{FHR}$ state.

---

## 16.5 TE residual loss

The TE branch should predict only the incremental part of future $\text{FHR}$ beyond the self-only forecast.

Define the residual target with stop-gradient on the self-only prediction:

$$
R_{a,h}^{*} = Y_{a,h}^{+} - \operatorname{sg}(\hat Y_{a,h}^{self}).
$$

Then the residual loss is

$$
\mathcal L_{TE\text{-}res} = \frac{1}{|\mathcal A|}
\sum_{a \in \mathcal A}
\sum_{h \in \mathcal H}
w_h \,
\rho(\hat R_{a,h}, R_{a,h}^{*}).
$$

This is a crucial design choice. It prevents the TE branch from re-learning the whole forecasting problem. Instead, it must explain only what the source adds beyond target self-history.

---

## 16.6 Conditional KL loss

The TE-style bottleneck is

$$
\mathcal L_{KL} = \frac{1}{|\mathcal A|}
\sum_{a \in \mathcal A}
\mathrm{KL}
\Big(
q_\phi(z_a^{TE} \mid U_{\le a}, Y_{\le a})
\;\|\;
r_\psi(z_a^{TE} \mid Y_{\le a})
\Big).
$$

For diagonal Gaussians, this KL has the usual closed form.

This is the only KL in the model.

That is important: the model is **not** a global VAE.
Only the small TE latent is regularized this way.

This conditional KL is what gives the latent its TE-like meaning:

* posterior sees both $\text{UP}$ and $\text{FHR}$ history,
* prior sees only $\text{FHR}$ history,
* KL penalizes extra information in $z_a^{TE}$ unless it is truly useful.

---

## 16.7 Total loss

The first-iteration objective is

$$
\mathcal L_{total} = \lambda_{fus}\mathcal L_{fus}
+
\lambda_{\Delta}\mathcal L_{\Delta}
+
\lambda_{self}\mathcal L_{self}
+
\lambda_{TE}\mathcal L_{TE\text{-}res}
+
\beta \mathcal L_{KL}.
$$

A good starting setting is:

$$
\lambda_{fus}=1,
\qquad
\lambda_{\Delta}=0.5,
\qquad
\lambda_{self}=0.25,
\qquad
\lambda_{TE}=0.25.
$$

For the KL coefficient, use a small value with warmup:

$$
\beta(t) = \beta_{\max}\min\!\left(1,\frac{t}{T_{\text{warm}}}\right),
$$

with

$$
\beta_{\max} \in [10^{-4}, 10^{-3}].
$$

Why small? Because the TE branch is auxiliary. You want it to be meaningful, not dominant.

---

# 17. How this relates to transfer entropy

This model is not computing closed-form analytical transfer entropy from the raw signals.

Instead, it learns a **representation-level TE proxy**.

The conceptual quantity of interest is

$$
I(U_{\le a}; Y_{a,h}^{+} \mid Y_{\le a}).
$$

The model approximates this logic as follows:

* the self-only head models prediction from $Y_{\le a}$,
* the posterior sees $(U_{\le a}, Y_{\le a})$,
* the prior sees $Y_{\le a}$ only,
* the residual head forces $z_a^{TE}$ to explain only the extra part of future $Y$ not already captured by the self-only prediction.

So the TE branch is a learned bottleneck for **source-incremental predictive information**.

That is exactly the right role for TE in this problem.

It should not be the whole model, because downstream classification also depends on intrinsic fetal abnormalities that may already be visible in $\text{FHR}$ alone.

---

# 18. Training algorithm

This is the training algorithm for one minibatch.

Let the minibatch be

$$
\{(Y^{(n)}, U^{(n)})\}_{n=1}^{B}.
$$

## Step 1: normalize

Compute

$$
\tilde Y^{(n)}, \tilde U^{(n)}.
$$

## Step 2: stems

Compute

$$
F = \operatorname{FHRStem}(\tilde Y),
\qquad
S = \operatorname{UPStem}(\tilde U).
$$

## Step 3: causal modality encoders

Compute

$$
H_F^c = E_F^c(F),
\qquad
H_U^c = E_U^c(S).
$$

## Step 4: causal fusion

Compute cross-attentive source context $C_t$, gates $G_t$, fused states $\tilde H_t^c$, then

$$
H_{FU}^c = E_{FU}^c(\tilde H^c).
$$

## Step 5: anchor sampling

For each window in the batch, sample $K$ anchors from $\mathcal A_{\text{valid}}$ using the mixture anchor distribution.

## Step 6: local summaries

For each anchor $a$, compute

$$
s_a^F,\qquad s_a^U,\qquad s_a^{FU}.
$$

## Step 7: self-only and fused forecasts

For each $h \in \mathcal H$, compute

$$
\hat Y_{a,h}^{self},\qquad \hat Y_{a,h}^{fus}.
$$

## Step 8: posterior / prior / sample

Compute

$$
q_\phi(z_a^{TE}\mid U_{\le a},Y_{\le a}),
\qquad
r_\psi(z_a^{TE}\mid Y_{\le a}),
$$

and sample $z_a^{TE}$.

## Step 9: TE residual forecast

Compute

$$
\hat R_{a,h},
\qquad
\hat Y_{a,h}^{TE} = \hat Y_{a,h}^{self} + \hat R_{a,h}.
$$

## Step 10: future targets

Construct

$$
Y_{a,h}^{+} = Y_{a+g+1:a+g+h}.
$$

## Step 11: losses

Compute

* $\mathcal L_{fus}$,
* $\mathcal L_{\Delta}$,
* $\mathcal L_{self}$,
* $\mathcal L_{TE\text{-}res}$,
* $\mathcal L_{KL}$,

then form

$$
\mathcal L_{total}.
$$

## Step 12: optimization

Backpropagate and update all parameters jointly.

---

# 19. Training schedule

Do not start with a strong KL from the first step.

Use a staged schedule.

## Stage 1: deterministic warm start

Train with

$$
\mathcal L_{fus}
+
\lambda_{\Delta}\mathcal L_{\Delta}
+
\lambda_{self}\mathcal L_{self}
$$

only.

Purpose:

* stabilize stems and encoders,
* learn good intrinsic and fused predictive states,
* avoid early collapse or noise in the TE branch.

## Stage 2: activate TE residual head

Turn on

$$
\lambda_{TE}\mathcal L_{TE\text{-}res}
$$

but keep $\beta$ near zero initially.

Purpose:

* make the TE head useful before applying the bottleneck strongly.

## Stage 3: KL warmup

Gradually increase $\beta$ until $\beta_{\max}$.

Purpose:

* force the TE latent to become compact and source-incremental,
* without killing it early.

This staged schedule is important. If you turn on everything strongly from the start, the TE branch may become unstable or collapse.

---

# 20. Exported latent representation for downstream tasks

After pretraining, you need a window-level representation.

Since we removed the bidirectional pathway in version 1, extract the representation from the causal states.

## 20.1 Fused global representation

From the fused causal sequence $H_{FU}^c$, compute:

### attention pool

$$
g_{attn} = \sum_{t=1}^T \alpha_t H_{FU,t}^c,
\qquad
\alpha_t = \operatorname{softmax}
\big(v^\top \tanh(W H_{FU,t}^c)\big)
$$

### max pool

$$
g_{max} = \max_{t=1,\dots,T} H_{FU,t}^c
$$

### quarter pools

Split the sequence into four temporal quarters and mean-pool each:

$$
q_1, \; q_2, \; q_3, \; q_4 \in \mathbb{R}^{d}.
$$

Then define

$$
e^{FU} = [g_{attn} \,|\, g_{max} \,|\, q_1 \,|\, q_2 \,|\, q_3 \,|\, q_4].
$$

## 20.2 TE summary

For a fixed anchor grid $\mathcal A^*$, for example every $15$ steps across the valid range, compute posterior means $\mu_a$.

Then pool them:

$$
\bar \mu^{TE} = \frac{1}{|\mathcal A^*|}
\sum_{a \in \mathcal A^*}
\mu_a,
$$

$$
\mu_{\max}^{TE} = \max_{a \in \mathcal A^*}
\mu_a.
$$

Then define

$$
e^{TE} = [\bar \mu^{TE} \,|\, \mu_{\max}^{TE}].
$$

## 20.3 Intrinsic $\text{FHR}$ summary

Similarly pool the $\text{FHR}$-only sequence $H_F^c$ to get an intrinsic summary $e^F$, for example using attention pool and max pool.

## 20.4 Final window embedding

Define

$$
e_{\text{win}} = [e^F \,|\, e^{FU} \,|\, e^{TE}].
$$

This is the embedding used for downstream classification.

---

# 21. Baby-level classification downstream

The pretraining learns window-level embeddings, but the clinical label may be baby-level.

So after pretraining:

1. compute window embeddings $e_{i,1}, \dots, e_{i,N_i}$ for each infant $i$,
2. add a time-to-delivery embedding if available,
3. aggregate them with attention and max pooling,
4. classify at the infant level.

For infant $i$, define

$$
\tilde e_{i,j} = [e_{i,j} \,|\, \pi(\tau_{i,j})],
$$

where $\pi(\tau)$ is a learned time-to-delivery embedding.

Then attention pooling gives

$$
b_i = \sum_{j=1}^{N_i}
\alpha_{i,j} \tilde e_{i,j},
\qquad
\alpha_{i,j} = \operatorname{softmax}
\big(
v^\top \tanh(W \tilde e_{i,j})
\big).
$$

Also compute max pooling:

$$
m_i = \max_j \tilde e_{i,j}.
$$

Then use

$$
h_i = [b_i \,|\, m_i]
$$

for the downstream classifier.

That is the right downstream aggregation because:

* attention gives a smooth summary,
* max pooling preserves rare severe windows.

---

# 22. Why this first version is the right one

This first-iteration model makes the right compromises.

## It keeps:

* causal structure,
* a strong $\text{FHR}$-only baseline,
* a strong multimodal predictive objective,
* an interpretable auxiliary TE branch,
* a clean link to downstream transfer.

## It avoids:

* a full reconstruction objective,
* a global VAE bottleneck,
* phase-harmonic complexity,
* mixed cross-feature leakage,
* the extra complexity of a masked bidirectional path.

That is exactly what you want for the first serious implementation: a model whose behavior you can understand and debug.

---

# 23. What not to do in version 1

Do not add these yet:

* full-sequence VAE reconstruction,
* pure TEB as the only objective,
* bidirectional masking,
* raw-signal prediction,
* phase-harmonics and mixed cross features,
* strong disentanglement penalties,
* per-window normalization,
* training with all anchors,
* split by window instead of GUID.

All of these add complexity or risk before the core idea has been validated.

---

# 24. Final concise specification

The first-iteration model is a **strictly causal dual-branch $\text{FHR}$-ST / $\text{UP}$-ST forecasting architecture**. Each modality is first encoded by a causal temporal-convolution stem, then by a modality-specific causal encoder. The $\text{UP}$ history is injected into the $\text{FHR}$ pathway through causal cross-attention followed by gated residual fusion, producing a fused multimodal causal state. At sampled anchor times, the model forms pooled recent summaries and uses three forecasting heads: a self-only $\text{FHR}$ head, a fused multimodal head, and an auxiliary TE residual head. The TE branch uses a small latent
$$
z_a^{TE} \sim q_\phi(z_a^{TE}\mid U_{\le a},Y_{\le a})
$$
with conditional prior
$$
r_\psi(z_a^{TE}\mid Y_{\le a}),
$$
and is trained to predict only the future residual beyond the self-only forecast. The total pretraining loss is
$$
\mathcal L_{total} = \lambda_{fus}\mathcal L_{fus}
+
\lambda_{\Delta}\mathcal L_{\Delta}
+
\lambda_{self}\mathcal L_{self}
+
\lambda_{TE}\mathcal L_{TE\text{-}res}
+
\beta \mathcal L_{KL},
$$
where the main task is multi-horizon future prediction of $\text{FHR}$-ST blocks and the KL regularizes only the local TE latent. The exported window embedding should concatenate pooled intrinsic $\text{FHR}$ state, pooled fused multimodal state, and pooled TE posterior summaries, and this embedding should then be aggregated across windows for infant-level healthy / unhealthy / severity classification.
