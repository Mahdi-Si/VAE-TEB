# FHR-Anchored Lag-Residual Transformer CFS

**Standalone architecture and validation specification — 2026-09-09**

**Proposed class:** SeqVaeLagResidualTrfCfs. **Model kind:** fhr_lag_residual_cfs_v1. **Package:** teb_vae/lag_slot_transformer_cfs. **Status:** design specification; the architecture has not been implemented or trained.

## 1. Purpose and central decision

The model forecasts future fetal-heart-rate (FHR) causal features from FHR history, optionally using uterine activity (UA, named UP in several dataset fields). It should learn an informative FHR latent and use UA when UA contributes predictive evidence beyond the available FHR history.

Keep **one $64$-dimensional FHR-defined latent space**. A target-only Gaussian prior describes the FHR forecast state. UA makes a bounded residual correction to the parameters of a second Gaussian over that same space. Both distributions feed one shared decoder. UA has no independent latent partition, reconstruction objective, or route around the latent.

The main architectural change is to replace the source convolution stem and lag cross-attention with:

1. A pointwise, information-preserving representation of each available UA coefficient.
2. A deterministic proposal from each source lag, conditioned on the anchor's FHR state.
3. An explicit sum at the latent-parameter fusion boundary.

There is **no temporal aggregation or attention inside the UA encoder**. There is aggregation at the final fusion boundary: several source times must combine to update a fixed-width latent. It would be inaccurate to claim no aggregation anywhere in the model.

“FHR-defined” is an architectural property: every latent coordinate is present in the FHR prior, and the full mean is the prior mean plus a correction. “UA contributes only useful, nonredundant information” is a learning and validation objective. Neither additive vectors nor a KL penalty guarantee that semantic property.

### 1.1 What a lag slot means here

A lag slot is an index in a deterministic proposal array:

| Lag index | Source observation read | Mean proposal | Stochastic latent |
|---|---|---|---|
| $0$ | $U_t$ | $r^\mu_{t,0}\in\mathbb R^{64}$ | Same shared $Z_t\in\mathbb R^{64}$ |
| $1$ | $U_{t-1}$ | $r^\mu_{t,1}\in\mathbb R^{64}$ | Same shared $Z_t$ |
| $\ell$ | $U_{t-\ell}$ | $r^\mu_{t,\ell}\in\mathbb R^{64}$ | Same shared $Z_t$ |
| $90$ | $U_{t-90}$ | $r^\mu_{t,90}\in\mathbb R^{64}$ | Same shared $Z_t$ |

A proposal is a suggested update, not a sampled code, probability, or independently identified lag effect. Its width does not increase the sampled latent width. A different design allocating two sampled coordinates to each of $91$ lags plus $32$ target coordinates would have width $214$; that architecture is excluded from this specification. Everything needed for the recommended model is defined below.

## 2. Evidence and relationship to the existing model

The existing lag_attn_transformer_cfs model combines causal feature inputs, a causal FHR conv-Transformer, an FHR-conditioned full-width Gaussian prior, source fusion, a source-conditioned Gaussian, and a shared future-feature decoder. Its executed anchored forward lives in the causal-input mixin, rather than only in the thin CFS wrapper.[^model]

The September 8 diagnosis concerns an epoch-$901$ checkpoint evaluated on $1{,}959$ recordings. It reports a matched Monte Carlo predictive gap of $-14.159$ nats per anchor, with recording-bootstrap interval $[-14.725,-13.586]$, using $K=8$ samples. The source slightly improved point RMSE while worsening the predictive density. Other findings included $70.1\%$ of coupling KL surviving an observed-zero source control, one coordinate carrying $72.8\%$ of KL, tail miscalibration, and no occlusion band with a confidently positive predictive contribution.[^diagnosis]

These observations motivate better uncertainty updates, meaningful source controls, and matched predictive validation. They do not establish a unique cause, prove that true UA information is absent, or demonstrate that the proposed architecture will improve performance.

The evaluated source path used conv_stem: its source self-attention blocks were not built. Its convolutional receptive field was

$$
R_U=1+(5-1)\cdot1+(9-1)\cdot2=21
$$

stored samples, followed by lag cross-attention. The new encoder reduces the additional neural source receptive field to one stored sample. This removes neural temporal mixing before fusion; causal feature extraction itself still mixes raw history.[^model][^diagnosis]

| Component | Existing evaluated model | Recommended model |
|---|---|---|
| FHR feature selection and forecast task | Causal stored features | Preserve |
| FHR encoder | Causal conv-Transformer, width $128$ | Preserve |
| Prior and latent width | FHR-conditioned Gaussian, $64$ | Preserve |
| Source encoder | Availability adapter and temporal convolution stem | Identity coefficients plus explicit masks |
| Lag fusion | Cross-attention and head summaries | Local deterministic proposals and explicit sum |
| Full mean | Prior mean plus bounded residual | Prior mean plus bounded residual in prior-scale units |
| Full variance | Independent posterior head in evaluated arm | Bounded residual in prior log-standard-deviation |
| Decoder | Shared horizon decoder | Preserve |
| FHR persistence | Shared target-only decoder mean term | Preserve for first comparison |
| Training base decode | Prior mean in evaluated configuration | Sample both branches using shared noise |
| Prior availability clock | Zero-source encoding through source network | Explicit value-free position encoding |
| Lag attribution | Attention-weighted latent KL | Fitted-model suppression, source replacement, exclusion refits |

The last three rows and the residual parameterization are real changes, not merely renamings. Controlled experiments in §11 separate their effects.

**Latent bypass contract.** All UA-derived value information must reach predictions through sampled $Z$. The inherited FHR persistence term also carries the anchor's FHR value to the decoder mean. Therefore this main candidate does not make all FHR information latent-mediated. A strict latent-only variant disables that existing term, with a separately trained, equally configured FHR-only comparator. No new decoder side input is introduced.

## 3. Data, notation, causality, and scoring support

### 3.1 Fixed first-experiment task

Use the current integer-phase, unaligned, stored-clock arm. Historical reference defaults include other horizons and channel layouts; the following values define this experiment.[^reference][^config][^diagnosis]

| Symbol or setting | Meaning | Value |
|---|---|---|
| $B$ | Batch size per device for initial configuration | $128$, reduce if memory requires |
| $T$ | Stored steps after trimming | $300$ |
| $\Delta$ | Duration of one stored step | $4$ s |
| $H$ | Future steps per forecast | $10$ |
| $F$ | First eligible anchor | $134$ |
| $L$ | Number of candidate source lags | $91$, indices $0,\ldots,90$ |
| $C_Y^{\rm declared}$ | Input FHR channels before selection | $80=36$ scattering $+44$ phase |
| $C_Y$ | Kept FHR input/output channels | $76=32$ scattering $+44$ phase |
| $C_U$ | UA channels | $46=36$ scattering $+10$ phase |
| $d_h,d_z$ | FHR state width and latent width | $128,64$ |
| $S$ | Training anchor stride | $5$ |
| $N_A$ | Padded number of selected anchors | $32$ for $S=5$; $156$ dense |
| Feature operator | Phase harmonic convention | integer_harmonic_v1 |
| Phase-leg alignment | Within-feature preprocessing convention | envelope |
| Input channel alignment | Additional model-side delays | None for either stream |
| Forecast clock | Label indexing convention | stored |
| Trim | Removed from each segment end | $1$ min, or $15$ stored steps |

The canonical stored UA timeline is binding. Use it directly in every equation and lag axis; apply no downstream timeline correction.

At anchor $t$, the target is the matrix

$$
V_t=Y_t^+=(Y_{t+h,c})_{\substack{h=1,\ldots,H\\c=1,\ldots,C_Y}}
\in\mathbb R^{H\times C_Y}.
$$

Thus dense anchors satisfy $F\le t<T-H$, giving $t=134,\ldots,289$ and $300-10-134=156$ anchors. The last label is at index $299$. The full block has $10\cdot76=760$ coefficients. The maximum source-anchor offset is $90\cdot4=360$ s; $91$ four-second bins have a nominal width of $364$ s. State whether an axis denotes bin edges or index offsets.

Use $b$ for batch index, $a$ for selected-anchor position, $t=t_{b,a}$ for actual time, $\ell$ for source lag, $j$ for source channel, $c$ for target channel, and $d$ for latent coordinate. $A_t$ below denotes metadata, whereas $N_A$ is a count.

### 3.2 Feature preparation and information sets

FHR-only input features must derive exclusively from FHR. UA input features must derive exclusively from UA. Exclude FHR–UA cross-feature blocks from both the target-only input and the forecast labels.

Apply the dataset's fixed transformations before channelwise normalization: scattering channel zero remains linear, other scattering channels use $\log(\max(x,0)+10^{-6})$, and phase channels use $\operatorname{asinh}(x)$. Then apply training-split statistics,

$$
\bar x_c=\frac{g_c(x_c)-m_c^{\rm train}}{s_c^{\rm train}+10^{-8}}.
$$

Statistics must come from the same causal operator and trimming convention and exclude each channel's warm-up region. No samplewise or whole-segment normalization is added. Normalization statistics are fitted on training recordings only.[^reference]

The warm-up boundary after trimming is $W'_c=\max(W_c-15,0)$. Drop FHR scattering indices $32$–$35$, whose rebased waits exceed the $134$-step budget; keep phase indices $36$–$79$ in declared concatenation order. Hence the kept index vector is $0,\ldots,31,36,\ldots,79$. UA retains all $46$ channels, with per-channel masking rather than a common all-warm requirement.

Define

$$
\mathcal H_t=(Y_{\le t}^{\rm permitted},A_t),\qquad
U_t^-=(U_t,\ldots,U_{t-L+1}).
$$

The permitted FHR history includes only kept, causally available coefficients and the FHR availability information used by its adapter. $A_t$ contains declared metadata available at prediction time. For the initial arm, source availability is the deterministic warm-up schedule, determined by stored position and fixed channel metadata. GUID and segment-start fields determine sampling and grouping; they are not predictive inputs. Outcome labels and time-to-delivery are not added to either encoder.

Neural causality means changing inputs after $t$ cannot change the forecast at $t$. A claim of raw-signal causality additionally requires that preprocessing has used only data available by each stored step. One-sided feature filters alone do not certify every upstream resampling, repair, or missing-value operation. Audit raw-prefix equivalence separately; endpoint validity and a $95\%$ warm-up energy criterion do not certify the entire filter support.[^reference]

### 3.3 UA availability and safe indexing

For the initial arm, define $v^U_{s,j}=1$ for accepted finite source observations; there is no assumed source-quality annotation. If a genuine source-quality mask is later added, give its causal metadata to both branches and version that conditioning change. Do not reuse FHR quality as if it were measured UA quality.

At each lag,

$$
m_{t,\ell,j}
=\mathbb{1}[0\le t-\ell<T]\,
 \mathbb{1}[t-\ell\ge W'_j]\,
 v^U_{t-\ell,j}.
$$

Evaluate validity before gathering; use an in-range surrogate index for masked positions and safely fill invalid values before nonlinear operations. Negative indices must never wrap to the sequence end. Multiplying a NaN by zero is not sanitization. Nonfinite values in an otherwise declared-valid input or target are a data error requiring rejection or an explicit, versioned quality policy, not an implicit normalized-zero observation.

A slow UA channel with $W'_j=278$ is unavailable at $t=134$. At $(t,\ell)=(134,90)$ the source index is $44$, even though the lag itself is in bounds. Per-lag exposure must count available channels and scored anchors; index support and feature warm-up are different conditions.

### 3.4 Anchor tiling and forecast masks

For a per-segment phase $\varphi_b\in\{0,\ldots,S-1\}$, select

$$
t_{b,a}=F+\varphi_b+aS,\qquad t_{b,a}<T-H.
$$

Set $\varphi_b$ using the existing stable hash of GUID, segment start, training epoch, and seed. At stride $5$, the five phases contain $32,31,31,31,31$ anchors. Pad to $32$ using a legal repeated final index with anchor_valid false. Never count padding or repeated valid anchors twice. Dense evaluation uses stride $1$.

Let $w_{b,s}$ be the dataset's decimated FHR weight, and $v^Y_{b,s}=\mathbb{1}[w_{b,s}\ge1]$. On the stored clock, define

$$
\begin{aligned}
M^0_{b,a,h}
&=\mathrm{anchorvalid}_{b,a}\,
  \mathbb{1}[t_{b,a}\ge F]\,v^Y_{b,t_{b,a}}v^Y_{b,t_{b,a}+h},\\
\rho_{b,a}&=\frac1H\sum_{h=1}^H M^0_{b,a,h},\\
M_{b,a,h}&=M^0_{b,a,h}\mathbb{1}[\rho_{b,a}\ge0.9],\\
C_{b,a}&=\mathbb{1}\left[\sum_hM_{b,a,h}>0\right],\qquad
N=\sum_{b,a}C_{b,a}.
\end{aligned}
$$

Broadcast $M$ over kept target channels. These same contributing anchors gate reconstruction, source KL, and prior regularization. At $N=0$, return a graph-connected zero contribution and record no scored data; coordinate this across distributed ranks. Do not treat a denominator clamped to one as a real observation.

Future FHR validity is used for loss selection only. It must not enter the predictor or UA head. Scores consequently refer to the selected, observed target population; they do not automatically estimate information in missing outcomes.

## 4. Exact network and starting settings

### 4.1 FHR state and value-free metadata clock

Keep the target availability adapter, two gated causal convolution blocks with kernels $(5,9)$ and dilations $(1,2)$, and six causal Transformer blocks. Width is $128$, with four attention heads, feed-forward width $512$, full causal-prefix attention, rotary positions, residual dropout $0.1$, and final RMS normalization. Convolutions use left padding; normalization reduces features at one time, never future times. Reuse the existing target block implementation and initialization.[^model]

Write its output as $h_t^Y=E_Y(Y_{\le t}^{\rm permitted})\in\mathbb R^{128}$.

Replace the source-network-generated clock with an explicit deterministic $128$-vector:

$$
\chi_{t,2k-2}=\sin(2\pi kt/T),\qquad
\chi_{t,2k-1}=\cos(2\pi kt/T),\quad k=1,\ldots,64.
$$

This clock reads only stored position, not UA values. With fixed warm-up vectors it contains the position needed to infer deterministic availability. Define the prior conditioning state

$$
h_t=h_t^Y+W_A\operatorname{LayerNorm}(\chi_t).
$$

$W_A$ is a bias-free $128\times128$ projection, initialized to zero after generic initialization, matching the existing prior clock-projection initialization policy. Compute $h_t$ once and use it in both the prior heads and lag proposal head. Keep the clock independent of UA-path weights.

This is a new clock parameterization, not a claim that its learned representation is identical to the old zero-source encoding. Both are functions of position and configuration; their representational adequacy can differ. Include a clock-matched attention comparator. If recording-dependent UA missingness is introduced, $\chi_t$ alone is insufficient to share that information: extend the metadata input explicitly and retrain both branches under the same declared $A_t$.

### 4.2 FHR prior

Use the existing full-latent prior mean and log-variance MLP heads on $h_t$, retaining their per-head normalization and residual MLP structure. Let their raw outputs be $u_t^p,\eta_t^p\in\mathbb R^{64}$. Define

$$
\begin{aligned}
\mu_t^p&=5\tanh(u_t^p/5),\\
\lambda_t^p&=-5+8\operatorname{sigmoid}(\eta_t^p),\\
\sigma_t^p&=\exp(\lambda_t^p/2),\\
P_t&=\mathcal N\!\left(\mu_t^p,\operatorname{diag}(\exp\lambda_t^p)\right).
\end{aligned}
$$

The latent parameters $\lambda_t^p,\lambda_t^q$ denote log-variance, not log-standard-deviation; the separately subscripted $\lambda_p,\lambda_q$ in §6 are loss weights. Every coordinate belongs to the FHR prior; no coordinate is reserved for UA. The existence of a coordinate does not guarantee that the trained decoder uses it.

Each prior head has its own input LayerNorm and a residual MLP with body widths $128\rightarrow111\rightarrow97\rightarrow84\rightarrow74\rightarrow64$. The MLP normalizes its input, applies LayerNorm, GELU, and dropout $0.1$ after intermediate linear layers, leaves the final output linear, and adds a learned $128\rightarrow64$ projection of the normalized input. Mean and log-variance heads do not share these MLP weights.

### 4.3 Pointwise UA encoder

The recommended first implementation uses identity plus mask:

$$
x^{\rm safe}_{s,j}=
\begin{cases}\bar U_{s,j},&m_{s,j}=1,\\0,&m_{s,j}=0,\end{cases}
\qquad
e_{s,j}=[x^{\rm safe}_{s,j},m_{s,j}].
$$

Each available standardized coefficient remains recoverable exactly from its own representation. No source encoder parameters are required. Gather $E_{t,\ell}=(e_{t-\ell,j})_{j=1}^{46}$ and flatten channels in declared order, giving width $92$.

An optional scalar-lift ablation appends $\phi_j(x^{\rm safe}_{s,j})\in\mathbb R^2$, using a separate $1\rightarrow8\rightarrow2$ GELU MLP per channel. Retain the identity and mask coordinates, so this lift cannot discard the original available coefficient. It is not part of the minimum starting configuration.

The scalar-lift arm has source width $46\cdot4=184$ per lag and changes the following local MLP input width from $228$ to $320$. All subsequent latent and decoder widths stay fixed.

With masks and metadata fixed,

$$
\frac{\partial e_{s,j}}{\partial \bar U_{r,k}}=0
\quad\text{for }(r,k)\ne(s,j).
$$

Do not introduce source temporal convolutions, recurrence, state-space blocks, attention, pooling, temporal normalization, source dropout, or learned competition across lags. Joint interpretation of channels at the same source time occurs only in the following fusion head.

### 4.4 One local proposal per lag

Let $\zeta_\ell\in\mathbb R^8$ be a learned lag embedding, initialized with independent zero-mean normal entries of standard deviation $0.02$. It identifies the fixed index $\ell$, not source content. Without lag identity, summing a shared lag-blind function would make the representation invariant to permuting source times.

Use one shared MLP,

$$
F_\theta:\mathbb R^{128+92+8}\longrightarrow\mathbb R^{128},
$$

with layers $228\rightarrow128\rightarrow128\rightarrow128$, GELU after the first two layers, linear final output, and no dropout or normalization. Split the final output into two $64$-vectors. Masks already occur in the $92$-vector and are not concatenated twice.

Let $s_{t,\ell}\in\{0,1\}$ be an externally set intervention selector, normally one, and let $v_{t,\ell}=\mathbb{1}[\sum_jm_{t,\ell,j}>0]$. Then

$$
(r^\mu_{t,\ell},r^\sigma_{t,\ell})
=s_{t,\ell}v_{t,\ell}
F_\theta\!\left([h_t,\operatorname{vec}(E_{t,\ell}),\zeta_\ell]\right).
$$

Each proposal reads exactly one UA stored time, although it can combine that time's channels and FHR context. The selector is not learned and is not an attention weight. Signed vector outputs are updates, not a probability distribution over lags. With $h_t$, masks, and metadata fixed, a proposal's derivative with respect to any other UA lag is zero.

Initialize the final output weight and bias to zero after all generic initialization. Initialize hidden layers normally. A mean-only ablation builds a $64$-output final layer and omits the scale proposal parameters entirely.

### 4.5 Fusing proposals into the same FHR latent

For fixed $L$, choose $c_L=L^{-1/2}$, $a_{\max}=3$, and $b_{\max}=1$. These are explicit starting hyperparameters, not empirically optimized values. Elementwise operations give

$$
\begin{aligned}
\bar a_t&=c_L\sum_{\ell=0}^{L-1}r^\mu_{t,\ell},&
a_t&=a_{\max}\tanh(\bar a_t/a_{\max}),\\
\bar b_t&=c_L\sum_{\ell=0}^{L-1}r^\sigma_{t,\ell},&
b_t&=b_{\max}\tanh(\bar b_t/b_{\max}),\\
\mu_t^q&=\mu_t^p+\sigma_t^p\odot a_t,&
\lambda_t^q&=\lambda_t^p+2b_t,\\
\sigma_t^q&=\sigma_t^p\odot\exp(b_t),&
Q_t&=\mathcal N\!\left(\mu_t^q,\operatorname{diag}(\exp\lambda_t^q)\right).
\end{aligned}
$$

The mean correction is bounded in prior-standard-deviation units. With these settings, $|\mu^q_d-\mu^p_d|\le3\sigma^p_d$, $\exp(-1)\le\sigma^q_d/\sigma^p_d\le\exp(1)$, and the implied posterior log-variance range is $[-7,5]$, allowing endpoint rounding in finite precision. It differs from the existing independently bounded posterior's $[-5,3]$ range. Observation log-variance retains its separate $[-5,3]$ setting.

Do not silently reinterpret the old delta_mu_scale as a prior-relative bound: its units differ. Do not pass $\lambda^q$ through the old sigmoid bound again. That map is not idempotent; a second application would change this parameterization and break zero-update equality. If a bounded-raw-log-variance alternative is tested, give it a separate model setting and compute KL from the actual final parameters.

Both signs of $b_t$ are permitted. Extra evidence can increase or decrease conditional uncertainty for a particular observation; useful information does not require pointwise variance reduction.

$c_L$ is a numerical convention. For uncorrelated equal-variance proposals it stabilizes variance, but in general

$$
\operatorname{Var}\!\left(c_L\sum_\ell r_\ell\right)
=c_L^2\sum_{\ell,k}\operatorname{Cov}(r_\ell,r_k).
$$

Shared heads and autocorrelated UA make independence implausible. Identical proposals still accumulate with amplitude $\sqrt L$. Keep $c_L$ fixed when masking lags, and monitor saturation and cancellation. Do not renormalize by the current number of available lags.

The limiter acts after summation. Consequently the posterior mean is an additive correction to FHR, but its final correction is not a sum of separately bounded lag effects. Nonlinear cross-lag effects can arise through this limiter and the decoder; nevertheless the additive parameter structure and finite latent impose a representational restriction. They do not guarantee the ability to represent every possible multi-lag interaction.

### 4.6 Source absence, real zeros, and initialization

If every selector is zero, every proposal is zero and $Q_t=P_t$ for any parameters. The same is true when every source lag is unavailable. At zero-initialized output projections, equality holds for all inputs.

This is an implementation identity, not a learned conditional-independence result. Artificially suppressing UA does not integrate out the true distribution of UA given FHR.

A valid standardized-zero UA observation remains an observation, with mask one. For example, if $U$ takes $-1,0,1$ equally and $V=U^2$, the unconditioned target mean is $2/3$, while observing $U=0$ identifies $V=0$. Do not enforce $F(h,U)-F(h,0)$ centering or equate real zero with absence. Removal rules determine what a removal-based explanation means.[^covert]

At initialization the KL gradient with respect to the residual update is zero. Predictive gradients can still reach the final source output projection through a nonconstant decoder. Hidden source-layer gradients may be zero on the first step because the output projection is zero. Verify learning on a nondegenerate controlled example; no claim is made that every parameter must have a nonzero gradient on every batch.

### 4.7 Paired samples and shared decoder

For each anchor and Monte Carlo draw $k$, use $\epsilon_t^{(k)}\sim\mathcal N(0,I_{64})$, independent across coordinates and draws, and shared between branches:

$$
Z_t^{p,(k)}=\mu_t^p+\sigma_t^p\odot\epsilon_t^{(k)},\qquad
Z_t^{q,(k)}=\mu_t^q+\sigma_t^q\odot\epsilon_t^{(k)}.
$$

The exact paired difference is

$$
Z_t^{q,(k)}-Z_t^{p,(k)}
=\sigma_t^p\odot\left[a_t+(\exp b_t-1)\odot\epsilon_t^{(k)}\right].
$$

Only the mean-only case $b_t=0$ gives a deterministic sample difference. This is not an independent additive UA random variable.

Retain the shared horizon decoder: hidden width $256$, four horizon convolution blocks with kernel $3$ and dilations $(1,2,4,8)$, latent-derived FiLM conditioning in each convolution block, learned horizon embeddings with initialization standard deviation $0.8$, and two horizon attention blocks with four heads each. All future tokens are generated from the latent and positional embeddings; future observed FHR or UA is never fed to them. Horizon attention is permitted because it operates on generated forecast tokens, not the UA history.

Write the decoder output as

$$
\begin{aligned}
\widehat\mu_{t,h,c}(Z)&=g_{\psi,h,c}(Z)+\omega_{h,c}Y_{t,c},\\
\nu_{t,h,c}(Z)&=-5+8\operatorname{sigmoid}(o_{\psi,h,c}(Z)),\\
p_\psi(V_t\mid Z,Y_t)&=\prod_{h,c}
\mathcal N\!\left(V_{t,h,c};\widehat\mu_{t,h,c}(Z),\exp\nu_{t,h,c}(Z)\right).
\end{aligned}
$$

$\nu$ is observation log-variance; it is distinct from latent $\lambda$. Keep the existing learned persistence weights $\omega_{h,c}$, initialized as $\omega_{h,c}=2^{-(h-1)/5}$. In the strict latent-only ablation set $\omega=0$ and do not pass $Y_t$.

No $h_t$, UA value, proposal, source mask, posterior parameter, or encoder summary is passed directly to the decoder. Both calls use the same $Y_t$ and weights. The persistence term cancels from the difference of predicted means, but not algebraically from nonlinear losses; a matching persistence path does not by itself prove an information-theoretic interpretation of score differences.

Compute the FHR state and prior once, sharing target/prior dropout realization. Keep the source head and decoder deterministic. Then equal latent distributions and shared noise produce equal predictions even in training mode.

For fresh FHR pretraining, retain calibrated initialization: set the prior log-variance MLP's final body weight and entire skip projection to zero and its final bias to $\log(5/3)$, giving $\lambda^p=0$. Scale the decoder's initialized mean-head weight by $0.02$, scale its log-variance-head weight by $0.1$, and set the latter bias to $\log(5/3)$. Re-zero FiLM generators after generic initialization so modulation starts at the identity; keep the latent-to-decoder and mean-head paths nonzero. Restore the target stem's depthwise initialization after generic Xavier initialization. Do not apply these fresh-model calibrations to transferred trained FHR/decoder weights. Source output projections and the new metadata projection have their separately specified zero initialization.

### 4.8 Tensor contract and computational cost

| Tensor | Shape |
|---|---|
| Declared FHR, UA inputs | $(B,T,80)$, $(B,T,46)$ |
| Kept FHR and encoded FHR | $(B,T,76)$, $(B,T,128)$ |
| Encoded pointwise UA | $(B,T,46,2)$ |
| Anchor index and validity | $(B,N_A)$ |
| Gathered source input | $(B,N_A,91,46,2)$ |
| Mean and scale proposals, each | $(B,N_A,91,64)$ |
| Prior/full parameters and samples, each | $(B,N_A,64)$ |
| Per-coordinate KL | $(B,N_A,64)$ |
| Total KL | $(B,N_A)$ |
| Base/full output means and log-variances, each | $(B,N_A,10,76)$ |
| Forecast mask | $(B,N_A,10)$ |

The new forward contract is anchor-indexed for latent parameters and proposals. A dense export must explicitly map anchors to time; do not feed an anchor-indexed tensor into an evaluator that assumes its second axis is $T$.

At $(B,N_A,L,d_z)=(128,32,91,64)$, each FP32 proposal array occupies $91$ MiB; mean and scale arrays occupy $182$ MiB before gradients and hidden activations. Dense $156$-anchor construction requires $443.625$ MiB per proposal array. Chunk anchors and optionally lag evaluation; accumulate the sum without detaching gradients. Retain complete proposals only for selected diagnostic batches. Evaluate source MLP cost and peak memory empirically: removing attention does not automatically make this model cheaper.

## 5. Mathematical guarantees and their limits

All logarithms are natural, so information quantities and unweighted log scores use nats. Identities involving differences of expectations assume the quantities are finite and the model densities have appropriate support. Do not subtract divergent terms.

Information-theoretic identities below refer to fixed trained parameters and evaluation mode, with dropout disabled. The Gaussian training KL is conditional on the shared realized encoder/prior dropout. If dropout is marginalized as an additional latent random variable, the resulting distributions are mixtures and their marginal KL is not generally the average conditional Gaussian KL.

### 5.1 Exact diagonal-Gaussian KL

For positive scales,

$$
\begin{aligned}
K_t&=\operatorname{KL}(Q_t\Vert P_t)\\
&=\frac12\sum_d\left[
\log\frac{(\sigma^p_{t,d})^2}{(\sigma^q_{t,d})^2}
+\frac{(\sigma^q_{t,d})^2+(\mu^q_{t,d}-\mu^p_{t,d})^2}
 {(\sigma^p_{t,d})^2}-1
\right].
\end{aligned}
$$

Substituting $\sigma^q/\sigma^p=\exp b$ and $(\mu^q-\mu^p)/\sigma^p=a$ gives

$$
K_t=\frac12\sum_d\left[a_{t,d}^2+\exp(2b_{t,d})-1-2b_{t,d}\right].
$$

This is nonnegative because $\exp x\ge1+x$, and it is zero exactly when $a=b=0$. Use expm1 for $\exp(2b)-1$ and FP32 or higher for the reduction; for extremely small $b$, the variance term has series $2b^2+\frac43b^3+\cdots$. Compare against the general formula using the final parameters. Numerical roundoff near zero needs an explicit tolerance, not an arbitrary large clamp.

The simplified KL has no direct dependence on prior mean or scale when $a,b$ are held fixed. It still affects shared parameters through the residual head's FHR conditioning. Do not assume this tied parameterization independently fits the prior to an aggregate posterior merely because that interpretation is available for an unconstrained variational prior.

The inherited prior regularizer is

$$
R_{p,t}=\frac12\sum_d
\left[\exp(\lambda^p_{t,d})-1-\lambda^p_{t,d}\right].
$$

It regularizes variances only. It is not $\operatorname{KL}(P_t\Vert\mathcal N(0,I))$, which would also include $(\mu^p_{t,d})^2$, and it is not a direct measure of FHR information in the latent.

### 5.2 No exact nonnegative per-lag KL allocation

For a hypothetical mean-only version with the final tanh limiter removed, write $K_t^{\rm linear}$ to distinguish it from the actual bounded model:

$$
\begin{aligned}
K_t^{\rm linear}
&=\frac12\left\|c_L\sum_\ell r^\mu_{t,\ell}\right\|^2\\
&=\frac{c_L^2}{2}\sum_\ell\|r^\mu_{t,\ell}\|^2
+c_L^2\sum_{\ell<k}\langle r^\mu_{t,\ell},r^\mu_{t,k}\rangle.
\end{aligned}
$$

The cross-terms can reinforce or cancel. With two scalar proposals $1,-1$ and $c_L=1/\sqrt2$, the total is zero while the sum of isolated linear-model KLs is $0.5$ nats. Exact cancellation also produces zero in the actual bounded model. Scale updates and final nonlinear bounding do not restore an additive lag decomposition.

The real KL decomposes over latent coordinates, which are not lag labels. Do not report a proposal norm, attention allocation, or sum of isolated KLs as exact per-lag transfer entropy. Remove the old source_kl_lag_map contract from the new model binding.

### 5.3 Local input provenance does not identify unique lag contributions

Even with all masks valid, local proposal functions are not uniquely determined by predictions. Consider functions $k_\ell(h)$ satisfying $\sum_\ell k_\ell(h)=0$, and replace

$$
r^\mu_{t,\ell}
\quad\text{by}\quad
r^\mu_{t,\ell}+k_\ell(h_t).
$$

The sum, posterior, KL, and every full-model prediction stay unchanged. Removing an individual proposal can nevertheless give a different result. Lag embeddings and target-conditioned heads permit such target-only terms. The two-lag example $k_0(h)=g(h)$, $k_1(h)=-g(h)$ already demonstrates the ambiguity on an all-valid support.

Therefore:

- Locality identifies which stored UA time a head can read.
- Proposal suppression measures dependence on that fitted parameterization.
- Neither establishes a unique functional decomposition or a physiological delay.

Correlated source histories add a separate ambiguity: multiple lags may contain the same predictive information. A latent KL penalty on the summed update cannot penalize large proposals that cancel.

Monitor the cancellation ratio

$$
\kappa_t=
\frac{\left\|\sum_\ell r^\mu_{t,\ell}\right\|_2}
{\sum_\ell\|r^\mu_{t,\ell}\|_2+\varepsilon_{\rm num}}
$$

alongside numerator and denominator, using $\varepsilon_{\rm num}=10^{-8}$ for this diagnostic; a near-zero ratio is uninformative when all proposals are near zero. Inspect the scale proposals similarly. Compare multiple seeds, band removals, plausible replacements, and exclusion refits. A proposal-norm regularizer is an optional ablation, not an identifiability theorem: it changes how correlated evidence is allocated and may favor spreading it across lags. Do not add it silently to the main objective.

### 5.4 Desired conditional relevance

The scientific predictive quantity of interest is

$$
I(V_t;U_t^-\mid\mathcal H_t).
$$

This asks what UA adds about the declared future FHR block after accounting for permitted FHR history and metadata. It does not require the UA correction to be independent of FHR or orthogonal to the prior latent.

For independent fair bits $H_0,U_0$ and $V=H_0\mathbin{\mathrm{XOR}}U_0$, neither input alone predicts $V$, while $I(V;U_0\mid H_0)=\log2$. FHR-conditioned corrections should be allowed to retain this kind of synergistic evidence. Conditional information includes synergy and should not be equated with “unique information” in the stricter partial-information-decomposition sense.[^pid]

Subtracting a learned estimate of UA from FHR, decorrelating latent vectors, or forcing a particular coordinate partition does not generally isolate nonlinear conditional relevance.

### 5.5 What the source KL bounds

Under the data distribution and the full stochastic encoder, define the aggregate

$$
Q_{\rm agg}(z\mid\mathcal H)
=\int Q(z\mid\mathcal H,u)\,p_{\rm data}(u\mid\mathcal H)\,du.
$$

Insert $Q_{\rm agg}$ into the log ratio:

$$
\log\frac{Q(z\mid\mathcal H,u)}{P(z\mid\mathcal H)}
=
\log\frac{Q(z\mid\mathcal H,u)}{Q_{\rm agg}(z\mid\mathcal H)}
+\log\frac{Q_{\rm agg}(z\mid\mathcal H)}{P(z\mid\mathcal H)}.
$$

Taking expectations gives the exact conditional rate decomposition,

$$
\mathbb E_{\mathcal H,U}K
=I_Q(Z;U\mid\mathcal H)
+\mathbb E_{\mathcal H}\operatorname{KL}
 \left(Q_{\rm agg}(Z\mid\mathcal H)\Vert P(Z\mid\mathcal H)\right).
$$

It is an upper bound on the UA information encoded given FHR, with an aggregate-prior mismatch term. The identity holds even if the network uses compressed FHR states. The aggregate is usually a Gaussian mixture, not generally a diagonal Gaussian, so mismatch can remain even with a well-trained prior. This application follows the information-bottleneck rate/prediction framework; the displayed conditional decomposition is derived here.[^alemi]

Because the encoder does not observe $V$, and its noise is independent of $V$ given $(\mathcal H,U)$, conditional data processing also gives

$$
I(V;Z\mid\mathcal H)
\le \min\{I(V;U\mid\mathcal H),I_Q(Z;U\mid\mathcal H)\}
\le \mathbb E K.
$$

Thus KL can upper-bound useful additional information represented in $Z$ under these assumptions, but can also pay for irrelevant source details or mismatch. Minimizing prediction loss is what rewards usefulness. Source-disabled equality sets neither the aggregate mismatch nor the true conditional information to zero.

The latent is not required to contain a coordinate equal to TE. The Gaussian KL is a statistic of its two conditional distributions. Calling a coordinate a “TE coordinate” would need an additional definition and validation absent from this model.

### 5.6 Predictive gain includes model approximation error

Let $p^\star_0(V\mid\mathcal H)$ and $p^\star_1(V\mid\mathcal H,U)$ be the true conditional densities on a fixed declared scoring population. Let $\widehat p_0,\widehat p_1$ be normalized model predictive densities with latent uncertainty integrated out. Define

$$
\begin{aligned}
G&=\mathbb E\log\frac{\widehat p_1(V\mid\mathcal H,U)}
{\widehat p_0(V\mid\mathcal H)},\\
\mathcal E_0&=\mathbb E_{\mathcal H}
 \operatorname{KL}(p^\star_0\Vert\widehat p_0),\\
\mathcal E_1&=\mathbb E_{\mathcal H,U}
 \operatorname{KL}(p^\star_1\Vert\widehat p_1).
\end{aligned}
$$

Adding and subtracting $\log p^\star_1$ and $\log p^\star_0$ inside the expectation yields

$$
G=I(V;U\mid\mathcal H)+\mathcal E_0-\mathcal E_1.
$$

Equality to conditional mutual information requires true predictive conditionals or equal approximation errors. A positive gain can arise because an enabled residual head corrects an inadequate FHR baseline, even when UA carries no new information. For example, with $V\sim\operatorname{Bernoulli}(0.8)$ independent of UA, a base prediction of $0.55$ and full prediction of $0.8$ give positive gain $\operatorname{KL}(\operatorname{Bern}(0.8)\Vert\operatorname{Bern}(0.55))\approx0.13757$ nats despite zero true source information.

Conversely a negative fitted gain does not establish zero physiological information. Finite Monte Carlo bias, optimization, capacity, and misspecification can all matter.

For missing targets, apply this identity separately to a fixed observed-coordinate pattern and the declared selected population, or explicitly condition on that pattern and selection. A pooled variable-mask score is not automatically the original complete-block conditional information. Compare both branches on identical coordinates, and include a complete-coverage subset.

## 6. Training objective and optimization

### 6.1 Fully specified Gaussian score

For branch $b_0\in\{p,q\}$, draw its latent as in §4.7 and define the per-coordinate conditional NLL

$$
d^{b_0}_{t,h,c}
=\frac12\left[
\log(2\pi)+\nu^{b_0}_{t,h,c}
+(V_{t,h,c}-\widehat\mu^{b_0}_{t,h,c})^2
 \exp(-\nu^{b_0}_{t,h,c})
\right].
$$

The initial channel weights use relative weights $\widetilde w_c=1$ for the $32$ scattering channels and $0.1$ for the $44$ phase channels:

$$
w_c=C_Y\frac{\widetilde w_c}{\sum_{c'}\widetilde w_{c'}},
\qquad \sum_cw_c=C_Y.
$$

Therefore $w_{\rm st}=76/36.4\approx2.087912$, $w_{\rm ph}\approx0.208791$, and the phase share of total channel weight is $4.4/36.4\approx12.09\%$. The $17.1\%$ figure appearing in the diagnosis's weighting discussion is inconsistent with its stated $32+44$ kept channels and these settings; use the arithmetic and resolved channel weights, not that percentage.[^config][^targetloss]

With horizon half-life $5$ steps,

$$
w_h=H\frac{2^{-(h-1)/5}}{\sum_{r=1}^H2^{-(r-1)/5}},
\qquad \sum_hw_h=H.
$$

The per-anchor weighted score is

$$
D^w_{b_0,t}(Z)=\sum_{h,c}M_{t,h}w_hw_cd^{b_0}_{t,h,c}(Z).
$$

For each batch, optimize

$$
\mathcal L=
\frac1N\sum_{b,a}
\left[
\lambda_q\,\mathbb E D^w_{q,t_{b,a}}
+\lambda_p\,\mathbb E D^w_{p,t_{b,a}}
+C_{b,a}\{\beta K_{t_{b,a}}+\beta_pR_{p,t_{b,a}}\}
\right].
$$

Use one shared-noise sample per branch per anchor initially. Set $\lambda_q=\lambda_p=1$, $\beta_p=0.1$, free_bits $=0$, and all auxiliary reconstruction/derivative/boundary weights to zero. For joint-training epoch $e\ge0$, use the declared ramp $\beta(e)=\min(e/50,1)$. Record epoch origin and resume position.

Sum over coefficients and average over contributing anchors. Do not average over $760$ coefficients while leaving KL unchanged: that changes the effective information penalty by a factor of $760$. Report the coefficient count and mask coverage alongside nats per anchor. Partial coverage is not silently rescaled to a complete block.

For DDP, the target is the global contributing-anchor mean. Use global $N$ and account for DDP gradient averaging; with world size $R$ and averaged gradients, each rank can backpropagate $R$ times its local numerator divided by global $N$. Ranks with no scored anchors must still participate safely. Log global numerators and denominators.

### 6.2 What this objective does and does not optimize

The weighted term is a composite training score, not the unweighted joint log density. Powers of Gaussian densities with non-unit weights do not become a normalized observation density without parameter-dependent normalization.

Also,

$$
-\log\mathbb E_Zp_\psi(V\mid Z,Y_t)
\le\mathbb E_Z[-\log p_\psi(V\mid Z,Y_t)]
$$

by Jensen's inequality. Expected conditional NLL and predictive mixture NLL are different objectives. Paired sampling removes a base/full sampling-policy asymmetry; it does not remove the Jensen gap or fix observation misspecification.

Neither $P$ nor $Q$ observes future labels. This is a conditional predictive bottleneck objective; the presence of VAE in the model family does not by itself make the objective a standard future-conditioned VAE ELBO. Multi-sample predictive training can be tested later. Sampling directly from the predictive latent distribution and averaging likelihoods is not IWAE importance sampling.[^iwae]

### 6.3 FHR foundation and informative latent

Train a competitive FHR-only model on the same task first. Initialize the candidate's target encoder, prior, and decoder from it where tensor semantics match, and initialize the new source outputs to zero. Jointly train with both losses. Gradients flow through the prior and decoder from both branches; do not insert unreported stop-gradient operations.

Use an independent frozen FHR-only predictor as an external reference. The internal base can change during joint training, so an improved internal gap is insufficient if the base deteriorates. Do not use a ranking loss that can improve the gap by making the base worse.

The formula $\mu^q=\mu^p+\Delta\mu$ does not guarantee that each semantic property of the prior representation survives: a correction can overwrite or cancel coordinates. Use bounds, base supervision, KL, and independent latent probes to assess that risk.

Measure latent usefulness by disrupting $Z$ while holding FHR persistence fixed, and by fitting frozen probes on prior/full means, scales, and sampled codes with recording-disjoint training and evaluation. Predict residual future dynamics beyond persistence, not only FHR level. Report whether latent information is accessible from a sample as well as from its distribution parameters. Do not require all $64$ dimensions to be active or reward a high KL as success.

There is no UA reconstruction loss. Information about future FHR mean, tails, calibration, or temporal/band dynamics is relevant to the stated distributional target. A different supervised clinical endpoint would be a separately defined task, not an automatic architecture improvement.

### 6.4 Initial optimization and selection policy

Start with AdamW, learning rate $3\times10^{-4}$, weight decay $10^{-4}$ on all trainable parameters, optimizer betas $(0.9,0.95)$, and epsilon $10^{-8}$, matching the current optimizer builder.[^optimizer] Use a linear LR warm-up over $2000$ optimizer steps from factor $0.1$ to $1$, followed by epoch milestone multipliers of $0.1$ at joint-training epochs $400$ and $800$. Serialize the precise step/epoch boundary convention and scheduler state. This schedule is a proposed explicit run setting; do not infer that a configuration key was executed in a historical run without checking its scheduler artifact. Use the same resolved policy across matched arms.

Use FP32 eager execution initially. Existing clip $3500$ and additive spike margin $2200$ are inherited starting guards for this $760$-coefficient objective, not validated thresholds for the new head. Re-measure pre-clip gradient norms and skipped-batch rates on a training-only pilot. Do not assume Gaussian NLL is nonnegative when configuring a spike detector.

For an initial run, cap training at $5000$ joint epochs and use validation every epoch. Retain the best three checkpoints under weighted validation loss and the best three under matched unweighted full predictive NLL. Use early stopping on the latter with patience $50$ validation epochs and minimum improvement $10^{-4}$ nats per anchor, initially at a fixed $K=32$ and fixed validation noise seeds. Keep the same rule across matched arms. Select finalists after larger-$K$ convergence checks and source-control evaluation, while monitoring the independent FHR-only reference. These are proposed stopping settings, not measured optimal values. Preserve the evaluated run as historical evidence, not as an untouched test set.

## 7. Predictive evaluation and calibration

For branch $b_0$, its fully marginalized predictive density is

$$
\widehat p_{b_0}(V\mid\text{inputs})
=\int p_\psi(V\mid z,Y_t)\,P_{b_0}(dz\mid\text{inputs}),
\qquad P_p=P,\quad P_q=Q.
$$

For the observed target coordinates, compute unweighted conditional log likelihood

$$
\Lambda^{(k)}_{b_0,t}
=-\sum_{h,c}M_{t,h}d^{b_0,(k)}_{t,h,c}.
$$

The direct Monte Carlo predictive score is

$$
D^{(K)}_{b_0,t}
=-\operatorname{logsumexp}_{k=1}^K\Lambda^{(k)}_{b_0,t}+\log K,
\qquad
\widehat G^{(K)}
=\widehat{\mathbb E}[D^{(K)}_{p,t}-D^{(K)}_{q,t}].
$$

Do not replace log-mean-likelihood with mean-NLL or decode only the prior mean. Use the same noise draws, target mask, persistence, and evaluation mode for both branches and all paired interventions.[^metrics]

The likelihood average is unbiased for the model likelihood; its negative logarithm is upward biased for NLL at finite $K$. Branch biases need not cancel. Evaluate $K=8,32,128$ with multiple sampling seeds for finalists, extending $K$ if the conclusion remains sensitive. Increasing $K$ does not guarantee monotonic improvement for an individual realized draw set.

Monitor normalized likelihood weights $\alpha_k=\exp(\Lambda_k)/\sum_r\exp(\Lambda_r)$ and $1/\sum_k\alpha_k^2$ as a concentration diagnostic. They are not source attention weights or proof that Monte Carlo error is negligible.

For a Gaussian observation mixture, predictive mean and variance for a coordinate are

$$
\widehat m=\mathbb E_Z\widehat\mu(Z),\qquad
\widehat v=\mathbb E_Z\exp\nu(Z)+\operatorname{Var}_Z(\widehat\mu(Z)).
$$

The predictive CDF is $\mathbb E_Z\Phi((y-\widehat\mu(Z))/\exp(\nu(Z)/2))$. Use this mixture CDF or predictive sampling for intervals and PIT. A mean of conditional standard deviations omits latent variation, and a Gaussian interval built from total variance is not generally an exact mixture quantile.

Report unweighted joint block NLL, predictive gain, point RMSE, calibration/coverage, source-control margins, KL, prior/full scales, residual saturation, proposal cancellation, per-channel/lag exposure, and recording-level distributions. Give equal-recording and anchor-weighted summaries separately. Bootstrap whole recordings; overlapping windows, coefficients, and anchors are not independent subjects.

For a band or horizon subset $\mathcal I$, rescore its marginal mixture using only those likelihood factors. In general,

$$
\log\mathbb E_Z\prod_i p(V_i\mid Z)
\ne\sum_i\log\mathbb E_Zp(V_i\mid Z).
$$

Marginal subset scores do not sum to the joint mixture score. Conditional per-draw scores do sum. An ordered telescoping decomposition can be additive, but its order must be declared and its terms must not be called independent marginal band contributions.

## 8. Lag readouts and source controls

### 8.1 Proposal suppression

For a lag band $\mathcal B$, set $s_{t,\ell}=0$ for $\ell\in\mathcal B$. Keep FHR, metadata, original masks, remaining local proposals, and $c_L$ fixed. Recompute the sums, limiters, posterior, paired samples, and decoder predictions. Define

$$
J_{\mathcal B}
=\widehat{\mathbb E}
 [D^{(K)}_{q\setminus\mathcal B,t}-D^{(K)}_{q,t}].
$$

Positive $J_{\mathcal B}$ means the fitted model predicts worse when those proposals are suppressed. It can be negative. It is neither an intervention on the biological system nor a uniquely identified contribution; §5.3 gives an explicit parameterization ambiguity. Do not normalize $J$ to sum to total gain or KL.

Start with the diagnosis's inclusive index bands $[0,14]$, $[15,44]$, $[45,67]$, and $[68,90]$. Score whole-band and joint removals before interpreting single-lag peaks. Report usable recording/anchor counts for every band. Unsupported bins are missing measurements, not measured zero effects.

### 8.2 Plausible replacements and refitted exclusions

| Method | Quantity it tests | Necessary qualification |
|---|---|---|
| Proposal suppression | Reliance of the fitted computation on selected proposals | Parameterization dependence and potentially unfamiliar internal states |
| Whole-segment source permutation | Sensitivity to pairing with the correct recording | Breaks FHR–UA dependence; not an exact conditional-independence test |
| Conditional replacement | Sensitivity to plausible alternate UA given retained information | Depends on a fitted conditional source distribution |
| Exclusion refit | Predictive value after the remaining model adapts | Finite-capacity, training, and baseline-comparability errors |

For permutation, use different recordings with compatible acquisition and availability strata, preserve within-source time order, and verify zero same-recording pairings. Do not shuffle individual source samples and destroy their autocorrelation.

For conditional replacement of band $\mathcal B$, the relevant replacement law is approximately $p(U_{\mathcal B}\mid\mathcal H,U_{-\mathcal B})$ on the chosen support. Train that model on training recordings only. Its misspecification can invalidate a conditional-independence interpretation; the conditional permutation literature makes the conditioning-model assumptions explicit.[^berrett]

For Bayes-optimal exclusion refits with normalized predictive densities and common support, the improvement from restoring a band is

$$
I(V;U_{\mathcal B}\mid\mathcal H,U_{-\mathcal B}).
$$

This is information conditional on other source lags. It can be zero for a redundantly informative lag. Finite fitted refits add approximation-error terms analogous to §5.6. Do not equate this refit quantity with proposal suppression.

### 8.3 Required source-specificity controls

Run the following with matched prediction and masks:

1. Explicit source-disabled selectors: verifies the equality invariant only.
2. Observed zeros, observed constants, and mask-only source values with selectors enabled: tests target/clock recalibration through the extra head.
3. Correct source versus compatible cross-recording source: tests recording specificity.
4. Independently trained FHR-only predictor: checks whether internal-base weakness creates the gain.
5. An enabled FHR-only residual head of comparable trainable capacity, with lag identity/masks but no UA values: tests extra nonlinear target capacity.
6. Redundant-UA and no-effect synthetic cases: tests false source claims under known conditional independence.

A negative source-shuffle margin is concerning, but a positive shuffle margin alone is insufficient: both correct and shuffled UA can be worse than the FHR-only model, as the existing diagnosis demonstrates.

Do not subtract the observed-zero control's KL from the true-source KL and call the difference exact source information. Divergences and predictive removal effects have no such general subtraction identity.

## 9. Timing and transfer-entropy interpretation

For the stored feature read at $t-\ell$ and a label at $t+h$,

$$
L_{\rm anchor}=\Delta\ell,\qquad
L_{\rm endpoint}=\Delta(\ell+h).
$$

If source channel $j$ and target channel $c$ have nominal feature delays $\delta_j^U,\delta_c^Y$, their approximate content separation is

$$
L_{j,c}^{\rm approx}(\ell,h)
=\Delta(\ell+h)+\delta_j^U-\delta_c^Y.
$$

This follows by subtracting nominal source content time $(t-\ell)\Delta-\delta_j^U$ from nominal target content time $(t+h)\Delta-\delta_c^Y$. It is a channel-pair/horizon approximation, not a universal correction. Dispersive filters, envelopes, phase operations, and autocorrelation spread dependence across times. Retain the canonical stored timeline throughout.[^critique]

The diagnosis's approximately $1021$ s content-lag spread is a feature-geometry issue. A pointwise neural encoder cannot undo it. Do not report an isolated physiological lag until simulations through the actual raw-to-feature pipeline establish the readout's interpretation on that task.

For a directly planted feature-level process

$$
Y_s=f(Y_{<s},U_{s-d_0})+\eta_s,
$$

the directly referenced source time for the horizon-$h$ label is $t+h-d_0$, corresponding to $\ell=d_0-h$. A pooled direct-support interval is

$$
[d_0-H,d_0-1]\cap[0,L-1].
$$

This is direct structural support, not necessarily the entire predictive-importance profile: FHR propagation, source autocorrelation, and filtering can broaden it. If $d_0<h$, that direct UA value is after the anchor and cannot be read. Earlier source history can still help predict it. Do not expect the pooled profile to peak automatically at $d_0$.[^planted]

One-step TE concerns a next target value conditioned on a specified target past and source past. For $H>1$, $I(V_t;U_t^-\mid\mathcal H_t)$ is future-block conditional information. A forecast of an endpoint from history ending at $t$ differs from conditioning on the observed target past immediately preceding that endpoint. A teacher-forced one-step TE-oriented analysis would be a separately declared analysis, not the deployed multi-step forecast.[^wibral]

Lag-explicit neural Granger models motivate controlling source dependency structure, but predictive lag selection does not establish a clinical causal mechanism. No unconfounded physiological-causality assumption is made here.[^tank]

## 10. Optional improvements, tested separately

### 10.1 Mean-only residual

Set $b_t=0$ and remove the scale head. Then $\sigma^q=\sigma^p$, $K_t=\frac12\|a_t\|^2$, and paired sample differences are deterministic. This tests whether changing latent uncertainty helps. Observation variance can still change because the decoder's log-variance head reads a changed latent sample. A mean-only latent update is therefore not an assumption that UA affects only the predicted FHR mean, although it restricts the Gaussian encoder family.

### 10.2 Student-$t$ observation likelihood

Keep Gaussian latent distributions and their KL. Replace only the observation likelihood after the Gaussian architecture comparison. For degrees of freedom $\nu_T>2$, scale $s>0$, location $\mu$, and residual $e=y-\mu$,

$$
\begin{aligned}
-\log p(y)
&=\log s+\frac12\log(\nu_T\pi)
 +\log\Gamma(\nu_T/2)-\log\Gamma((\nu_T+1)/2)\\
&\quad+\frac{\nu_T+1}{2}
 \log\left(1+\frac{e^2}{\nu_Ts^2}\right).
\end{aligned}
$$

Student-$t$ scale is not standard deviation: $\operatorname{Var}(Y\mid Z)=s^2\nu_T/(\nu_T-2)$. Use $\nu_T=5$ as an explicit fixed starting arm; if learning it later, parameterize $\nu_T=2+\varepsilon+\operatorname{softplus}(\rho)$ with a declared positive $\varepsilon$. The fixed arm avoids initial confounding from a learned tail parameter.[^student]

For a variance-matched initialization relative to Gaussian observation variance $v$, set $s^2=v(\nu_T-2)/\nu_T$. Predictive variance adds the variance of the latent-conditioned location, and predictive intervals must use the Student-$t$ mixture CDF or samples. Change both base and full observation models together. Check whether calibration and held-out joint NLL improve; heavy tails are not a guaranteed fix.

### 10.3 Longer horizons, channel weighting, and FHR persistence

Longer horizons may reveal different conditional information. With $T=300,F=134$, horizons $10,30,45$ give dense anchor counts $156,136,121$. For comparisons, report the common first-$10$-step task on common anchors in addition to each arm's native score. Raw total block nats across different horizons are not directly comparable.

Channel weighting can prioritize dynamic bands, but must use the resolved kept channel set and explicit normalization in §6.1. Do not change weights, horizon, likelihood, and source architecture together and attribute the result to the source encoder.

Disable persistence only in a named strict latent-only arm. Assess whether the latent captures FHR dynamics and whether an equally configured source-free baseline remains competitive. Keeping or removing persistence does not change the requirement that UA must pass through $Z$.

The scalar lift from §4.3, a proposal-norm penalty, alternative residual bounds, and multi-sample predictive training are additional named ablations. Extra UA stochastic slots, source reconstruction, and source attention are not part of the recommended final architecture.

## 11. Experimental design and acceptance

### 11.1 Comparisons that separate mechanisms

| Arm | Purpose | Required matching |
|---|---|---|
| Existing checkpoint | Reassess reported gap with Monte Carlo convergence and larger intervention coverage | Original task and checkpoint |
| Retrained attention reference | Establish comparable training and metadata baseline | Shared-noise base/full, explicit clock, same pretraining and selection budget |
| Pointwise source plus attention | Isolate removal of source temporal convolution | Same downstream attention/prior/decoder as its reference |
| Pointwise source plus local residuals | Main architecture | Same task, $64$ latent, decoder, metadata, sampling, and training budget |
| Local residual, mean only | Isolate variance update | Same source features, mean head, prior, and decoder |
| FHR-only and capacity control | Detect baseline weakness and target-only recalibration | Same support, decoder, optimization and competitive tuning |
| Optional improvements | Test observation family, scalar lift, horizon, weighting, persistence | One declared change at a time |

The proposed full model differs from the historical checkpoint in fusion, residual units, scale parameterization, metadata encoding, and sampling policy. A direct comparison measures that package of changes. To attribute a difference specifically to attention removal, use an attention comparator with the same prior-relative posterior equations and bounded scale range, replacing only how residual inputs are formed. A small factorial comparison of attention versus local fusion and mean-only versus mean/scale updates can separate their interaction.

Arm comparisons should have comparable parameter budgets where practical and report actual parameters, compute, and memory. Equal latent width does not imply equal predictor capacity.

### 11.2 Synthetic tests

Run both simple stored-feature instruments and raw-signal simulations passed through the actual causal feature pipeline:

| Instrument | What it checks |
|---|---|
| Strong FHR autoregression, UA has no effect | False source gain with a capable target baseline |
| UA is a deterministic or redundant function of available FHR | Whether extra target capacity masquerades as novelty |
| One known delay and several delays | Direct lag support, horizon indexing, competing source evidence |
| Broad response kernel and autocorrelated UA | Band recovery instead of an unjustified point delay |
| State-dependent effects and FHR–UA synergy | Whether FHR conditioning is useful |
| Multi-lag source interactions | Limits of additive parameter proposals |
| Informative observed zero versus missing input | Correct absence semantics |
| Constants and availability patterns | Clock and mask confounding |
| Common driver or reverse-direction process | Limits of predictive causal interpretation |
| Feature-filtered raw process | End-to-end delay spread and preprocessing causality |

Predefine recovery bands and error criteria from each generator before fitting. Include off-support delays and source histories with different autocorrelation. Report power and false-positive rate over repeated simulations, rather than displaying only successful examples.

### 11.3 Acceptance gates

Structural gates must pass before training claims are evaluated:

- The prior sees no UA values and retains all $64$ latent coordinates.
- Each source encoder output is pointwise; each local proposal reads one stored source time.
- No future value or future validity mask enters prediction.
- Source-disabled and all-unavailable inputs reproduce the prior; valid observed zeros need not.
- Decoder interventions show that UA has no route around $Z$.
- Shared-noise equality, Gaussian KL, masks, anchor indexing, and gradients behave as specified.

Empirical acceptance requires stable matched held-out improvement over the internal base **and** a competitive independent FHR-only predictor, without base degradation explaining the result. Require Monte Carlo stability, appropriate uncertainty/calibration, informative latent probes, and source-specificity controls. Do not require every patient subgroup to have the same effect; report uncertainty and heterogeneity.

A source-relevance claim needs unforced constant/mask/parameter-capacity controls. A lag-recovery claim additionally needs synthetic support recovery and robust band/replacement/refit results across seeds. Failure of the latter can coexist with useful prediction; it limits the lag claim.

Use at least three training seeds for shortlisted models and recording-level bootstrap intervals, initially $2000$ resamples for continuity. Predeclare primary comparisons and bands; account for multiplicity for exploratory peak searches. More overlapping anchors do not increase the number of independent recordings.

The September 8 test diagnosis informed this design. Confirm selected improvements on an untouched outer fold or reserved final partition. Architecture, hyperparameters, thresholds, and conditional-replacement models must not be selected on that final partition.

## 12. Implementation contract

### 12.1 Module responsibilities

| Proposed file | Responsibility |
|---|---|
| nets/pointwise_source.py | Safe pointwise input construction and optional scalar lift |
| nets/lag_updates.py | Lag gather, embeddings, local MLP, selectors, sum and bounds |
| nets/model.py | FHR/prior composition, explicit clock, anchored forward, shared decoder |
| task.py | Matched stochastic training and predictive validation |
| configs/default.yaml | Fully resolved starting settings and model identity |
| eval/binding.py | Explicit anchor-indexed contract and compatible metric adapters |
| eval/lag_metrics.py | Suppression, replacements, exposure, and refit comparisons |
| tests/ | Structural, mathematical, indexing, and numerical checks below |

These are planned responsibilities, not files claimed to exist.

Reuse channel selection, causal metadata resolution, normalization, target construction, tiling, and decoder components. Override or refactor the inherited causal forward: it calls attention directly and cannot execute this design unchanged. Do not construct an unused source Transformer or unused attention modules merely to satisfy inherited attributes.

Return anchor_index, anchor_valid, prior/full means and log-variances, paired samples when requested, forecast means and log-variances, per-coordinate and total KL, masks, and bounded/unbounded update summaries. Return proposals only when requested for diagnostics. Remove ambiguous attention/per-head KL fields rather than fabricating tensors to satisfy the old evaluator.

Unknown or incompatible constructor keys must raise errors. Source-attention settings, entmax settings, and attention bias options do not apply to this model. The identity encoder has no learned source parameters; the lag update MLP does.

### 12.2 Reproducibility and checkpointing

Persist model kind/version; all architecture and optimizer settings; exact channel order and kept indices; warm-up vectors; operator, leg alignment, forecast clock and trim; normalization provenance; source-quality policy; metadata inputs; dataset split hashes; anchor stride/phase rule; random seeds; sampling policy; score weights; and checkpoint-selection criteria.

Follow [train/graph_models_utils.py](../../train/graph_models_utils.py) for checkpoint loading. Matching dimensions do not make the old source fusion semantically compatible. Warm starts may transfer target/prior/decoder tensors only after validating feature order and task geometry. List every transferred, missing, and reinitialized tensor. Source head, lag embeddings, and the changed clock path require explicit handling. Reinitialize source final projections after transfers unless resuming a checkpoint of this exact model kind.

For joint training after FHR pretraining, start a declared new optimizer/scheduler state and KL ramp; an exact resume instead restores all optimizer, scheduler, epoch, and RNG state. Do not confuse those operations.

Use Google-style code docstrings with LaTeX math. Keep a resolved configuration artifact rather than relying on defaults that can later change.

### 12.3 DDP and execution details

Always evaluate constructed heads on safely filled tensors and apply selector/mask multiplication in the graph. Avoid Python branches that omit learned parameters on ranks with unavailable source. The mean-only model should not construct an unused scale head. Verify all-invalid and mixed-validity ranks without enabling unused-parameter handling merely to hide unreachable modules.

Changing chunk sizes can change floating-point summation order. Accumulate sums and KL in at least FP32, test tolerances explicitly, and record any mixed-precision or compilation change. Do not detach chunks during training. For paired interventions, cache only deterministic original proposals and recompute downstream nonlinear fusion exactly.

## 13. Verification requirements and evidence boundary

### 13.1 Checks required for the implementation

| Check | Expected result |
|---|---|
| General Gaussian KL versus residual formula | Agreement for randomized positive scales and bounded residuals |
| $a=b=0$ | KL zero and prior/full parameters equal |
| All selectors off after nonzero weights | Prior/full parameters and paired predictions equal |
| All-UA-unavailable case | Safe finite outputs; no invalid gather; source residual zero |
| Observed-zero informative toy | No architectural constraint forcing equality to the prior |
| Source locality by Jacobian and perturbation | No dependence on other source times before the sum |
| Future-input perturbation | No forecast change at earlier anchors |
| Shared-noise identity | Agreement with the displayed paired-sample equation |
| Gradient escape from zero initialization | Predictive gradient reaches final source projection in a nondegenerate toy |
| Lag gather boundary tests | Correct index $t-\ell$; no negative wrapping |
| Target endpoint and anchor padding | Last label $299$; padding excluded everywhere |
| Loss mask and KL support | Same contributing anchors and denominators |
| Channel/horizon weights | Sums $76$ and $10$; phase share $12.09\%$ |
| Proposal cancellation and reallocation | Identical total posterior can coexist with different suppression results |
| Chunked versus unchunked forward/backward | Agreement within declared numerical tolerances |
| DDP uneven validity | Same global objective/gradient as combined-batch reference |
| Evaluation density | Log-mean-likelihood, common masks/noise, and latent uncertainty included |
| Checkpoint round-trip | Same metadata, parameters, inference, and source-off invariants |

Use FP64 algebra tests with representative starting tolerance atol $10^{-10}$ and rtol $10^{-8}$, plus separately measured FP32 tolerances for model execution. These values are numerical checks, not predictive acceptance margins. Never demand positive KL at initialization, where zero is intentional.

### 13.2 What this document establishes

The equations establish positive Gaussian scales, a bounded FHR-relative update, exact source-disabled equality, the paired-sample relation, the Gaussian KL, the conditional rate decomposition, and the approximation-error decomposition of predictive gain under the stated assumptions. They also demonstrate why lag proposals, KL, and physiological delays must not be conflated.

Numerical audit results are recorded below. These verify equations and counterexamples using standard-library arithmetic; they do not test an implemented neural model or reproduce production evaluation.

Audit date: September 9, 2026. Random Gaussian checks used seed $20260909$, $2000$ cases of $64$ coordinates, prior log-variances sampled in $[-5,3]$, and residuals passed through the stated limiters. Discrete identities were checked by exact enumeration of binary FHR-context, UA, latent, and target variables.

| Check | Result |
|---|---|
| General versus residual Gaussian KL | Maximum absolute difference $1.14\times10^{-13}$ |
| Paired-sample difference formula | Maximum absolute difference $1.07\times10^{-14}$ |
| Conditional rate identity | Absolute difference $1.05\times10^{-17}$ |
| Predictive-gain/error identity | Absolute difference $3.13\times10^{-17}$ |
| Conditional data-processing example | $I(V;Z\mid\mathcal H)=0.02806\le\min(0.11125,0.13757)\le0.16220=\mathbb E K$ |
| Zero-sum reallocation of two lag proposals | Same full update; suppressed bounded mean changes by $0.70039$ |
| Positive fitted gain with no true source information | $0.13757$ nats |
| FHR–UA XOR synergy | $\log2=0.69315$ nats |
| Training tiling, horizon counts, and weight normalization | Match §§3 and 6 |
| FP32 proposal storage | $91$ MiB tiled and $443.625$ MiB dense, per array |

Numerical agreement supports implementation of the algebra; the proofs, assumptions, and counterexamples above determine its scope.

The remaining scientific questions are empirical: whether the FHR latent is sufficiently informative, whether UA improves prediction beyond a capable FHR model, whether the gain survives controls and calibration checks, and whether the lag readout recovers known dependence under the actual preprocessing pipeline. No architecture-only review can certify those outcomes.

## Sources

Repository behavior was checked against the working tree based on commit **6cfa8be7f6089c625bae64254fdd0700faf92c17** and the current configuration. Reported production metrics come from the September 8 diagnosis; they were not rerun. External papers support general principles and the cited limitations, not performance claims for this proposed model.

[^reference]: [CFS / CRWS models reference](../CFS_CRWS_MODELS_REFERENCE.md), particularly data preparation, normalization, availability, and timing conventions. Its historical geometry is superseded here by the stated current-arm values.
[^diagnosis]: [Evaluation diagnosis, September 8, 2026](../lag_attn_transformer_cfs/EVAL_DIAGNOSIS_2026-09-08.md), source of the reported checkpoint metrics and resolved run geometry.
[^model]: [CFS wrapper](../lag_attn_transformer_cfs/nets/model.py), [Transformer composition](../lag_attn_transformer_rws/nets/model.py), [causal anchored forward](../lag_attn_cfs/nets/causal_inputs.py), [target encoder](../lag_attn_transformer_rws/nets/encoders.py), [full-latent prior](../lag_attn_rws/nets/heads.py), and [shared decoder](../lag_attn/nets/decoders.py).
[^config]: [Current Transformer CFS configuration](../lag_attn_transformer_cfs/configs/default.yaml).
[^targetloss]: [Feature target construction and channel weights](../lag_attn_cfs/nets/causal_feature_target.py), [forecast and KL masks](../lag_attn_rws/nets/raw_masks.py), and [weighted losses](../lag_attn_rws/nets/losses.py).
[^metrics]: [CFS predictive metrics](../lag_attn_cfs/eval/metrics.py), including mc_predictive_block and marginalise_block_scores.
[^critique]: [CFS scattering and phase forecast critique](../CFS_SCATTERING_PHASE_FORECAST_CRITIQUE.md), delay counterexamples and information-theoretic distinctions.
[^planted]: [Planted-lag recovery instrument](../lag_attn_cfs/lag_recovery_check.py), horizon-dependent stored-lag support.
[^alemi]: Alexander A. Alemi, Ian Fischer, Joshua V. Dillon, and Kevin Murphy. [Deep Variational Information Bottleneck](https://arxiv.org/abs/1612.00410), ICLR 2017. Rate/prediction framework.
[^pid]: Paul L. Williams and Randall D. Beer. [Nonnegative Decomposition of Multivariate Information](https://arxiv.org/abs/1004.2515), 2010. Unique, redundant, and synergistic information; no PID estimator is assumed here.
[^covert]: Ian Covert, Scott Lundberg, and Su-In Lee. [Explaining by Removing: A Unified Framework for Model Explanation](https://www.jmlr.org/papers/v22/20-1316.html), JMLR 22(209), 2021. Removal rules and explanation targets.
[^berrett]: Thomas B. Berrett, Yi Wang, Rina Foygel Barber, and Richard J. Samworth. [The Conditional Permutation Test for Independence While Controlling for Confounders](https://academic.oup.com/jrsssb/article/82/1/175/7056014), JRSS Series B 82(1), 2020. Conditional replacement assumptions.
[^wibral]: Michael Wibral et al. [Measuring Information-Transfer Delays](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0055809), PLOS ONE 8(2), e55809, 2013. Target conditioning and delayed information transfer.
[^tank]: Alex Tank, Ian Covert, Nicholas Foti, Ali Shojaie, and Emily B. Fox. [Neural Granger Causality](https://arxiv.org/abs/1802.05842), preprint 2018, revised 2021. Structured predictive source/lag selection.
[^student]: PyTorch contributors. [StudentT distribution](https://docs.pytorch.org/docs/2.14/distributions.html#studentt), official documentation, accessed September 9, 2026. Degrees of freedom, location, and scale parameterization.
[^iwae]: Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. [Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519), ICLR 2016. Importance-weighted latent estimation, distinct from direct predictive sampling.

[^optimizer]: [Optimizer and scheduler construction](../../train/pl_model_base.py), AdamW parameter defaults and epoch milestone scheduler.
