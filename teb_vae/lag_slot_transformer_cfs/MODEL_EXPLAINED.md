# The FHR-anchored lag-residual Transformer: from signals to forecasts

This is a standalone explanation of the model implemented in `teb_vae/lag_slot_transformer_cfs`, its dataset, mathematics, training, and evaluation. It describes the working-tree implementation inspected on **2026-09-10**, at repository commit `63d0623`, together with the supplied configurations. Numerical examples use the production configuration unless explicitly marked otherwise.

The implemented class is **`SeqVaeLagResidualTrfCfs`**, with model kind `fhr_lag_residual_cfs_v1`. Despite the directory name, the current model does **not** create a stochastic latent slot for each lag. That was an earlier proposal preserved in `DESIGN_V1_SUPERSEDED.md`. The current model has one latent space, a target-only Gaussian over that space, and a source-conditioned Gaussian obtained by changing the first Gaussian's parameters. “Lag slots” now refer to deterministic, separately indexed source proposals before they are summed.

The main configuration is a research forecaster of **future FHR features**, not a raw-waveform generator, an outcome classifier, or an established estimator of physiological transfer entropy. Its purpose is to learn useful FHR representations and measure whether uterine-activity history improves a distributional forecast beyond FHR history alone.

## Contents

1. [The question the model asks](#1-the-question-the-model-asks)
2. [Notation and dimensions](#2-notation-and-dimensions)
3. [How raw recordings become the dataset](#3-how-raw-recordings-become-the-dataset)
4. [The causal feature transform](#4-the-causal-feature-transform)
5. [Normalization, channel selection, and availability](#5-normalization-channel-selection-and-availability)
6. [Forecast anchors, labels, and masks](#6-forecast-anchors-labels-and-masks)
7. [The complete forward computation](#7-the-complete-forward-computation)
8. [Encoding FHR history](#8-encoding-fhr-history)
9. [The metadata clock and target-only prior](#9-the-metadata-clock-and-target-only-prior)
10. [Preserving source values and gathering lags](#10-preserving-source-values-and-gathering-lags)
11. [Local proposals and Gaussian residual updates](#11-local-proposals-and-gaussian-residual-updates)
12. [Sampling and the shared forecast decoder](#12-sampling-and-the-shared-forecast-decoder)
13. [The exact KL divergence](#13-the-exact-kl-divergence)
14. [The training objective and its gradients](#14-the-training-objective-and-its-gradients)
15. [Initialization and training orchestration](#15-initialization-and-training-orchestration)
16. [Predictive evaluation](#16-predictive-evaluation)
17. [Source controls and lag interpretation](#17-source-controls-and-lag-interpretation)
18. [What the information-theoretic quantities mean](#18-what-the-information-theoretic-quantities-mean)
19. [Configuration arms and alternatives](#19-configuration-arms-and-alternatives)
20. [Outputs, diagnostics, and computational cost](#20-outputs-diagnostics-and-computational-cost)
21. [A worked example](#21-a-worked-example)
22. [Implementation qualifications and evidence](#22-implementation-qualifications-and-evidence)
23. [Source map and reproduction](#23-source-map-and-reproduction)

## 1. The question the model asks

Fetal heart rate, abbreviated FHR, has substantial temporal structure of its own. A reasonable forecast should first exploit that structure. Uterine activity, abbreviated UA and stored under `up` in the dataset, may provide additional predictive evidence. The architecture explicitly represents these two levels of knowledge.

At an observation time called the **anchor**, the model forms:

- A **base forecast**, using FHR history and deterministic timing metadata.
- A **full forecast**, using the same information plus an available window of UA history.

Both forecasts use the same latent coordinates and the same decoder. This makes their comparison more controlled than comparing unrelated networks with different output mechanisms.

Let $\mathcal H_t$ denote the permitted FHR history and timing metadata available at anchor $t$, and let $U_t^-=(U_t,U_{t-1},\ldots,U_{t-L+1})$ denote the candidate UA history, with its channelwise availability understood. The target is a block $V_t$ of future FHR features. The scientific question is related to

$$
I(V_t;U_t^-\mid\mathcal H_t),
$$

the conditional mutual information between that future block and the source history after accounting for the target history.

The implemented model does not directly calculate this data-distribution quantity. It learns two latent distributions, forecasts through them, and reports a latent divergence and predictive score differences. Their relationships to conditional information are derived in Section 18.

### 1.1 Why use a latent variable?

A latent vector $z\in\mathbb R^{d_z}$ is a compact hidden representation from which the decoder constructs a forecast. Here the latent is stochastic: its distribution represents a family of possible codes rather than one fixed code.

The base distribution is

$$
P_t=\mathcal N(\mu_t^p,\operatorname{diag}((\sigma_t^p)^2)).
$$

The full distribution is

$$
Q_t=\mathcal N(\mu_t^q,\operatorname{diag}((\sigma_t^q)^2)).
$$

The superscripts $p$ and $q$ name branches, not different latent spaces. Every coordinate exists in both. UA changes the location and scale of the full distribution relative to the base distribution.

Penalizing $\operatorname{KL}(Q_t\Vert P_t)$ makes changes costly. Forecast improvement must compete with that cost. This is the conditional predictive bottleneck at the center of the model.

### 1.2 What “prior” and “posterior” mean here

The code calls $P_t$ the prior and $Q_t$ the posterior (`mu_post`, `logvar_post`). **Neither encoder sees the observed future label $V_t$.** The full branch conditions on additional source history, not on future FHR. Consequently, this is not the usual VAE construction in which a recognition network sees the observation being reconstructed and approximates its Bayesian latent posterior. “Full encoder distribution” is often the clearest reading of $Q_t$.

## 2. Notation and dimensions

All time indices below are zero-based unless stated otherwise. Forecast offset $h$ is one-based, so $h=1$ is the first step after an anchor. All logarithms are natural. Vectors use elementwise arithmetic unless a matrix product, sum, or norm is shown.

| Symbol | Meaning | Production value |
|---|---|---|
| $B$ | Batch size on one device | $128$ for training in `default.yaml` |
| $n$ | Raw-sample index | Raw sampling rate $4$ Hz |
| $D$ | Raw samples per stored feature step | $16$ |
| $\Delta$ | Seconds per stored step | $D/4=4$ s |
| $T$ | Stored steps after trimming | $300$ |
| $t,s$ | Anchor/source positions on the trimmed stored grid | $0,\ldots,T-1$ |
| $F$ | Earliest decoded anchor | $134$ |
| $H$ | Number of future steps per forecast | $10$ |
| $h$ | Forecast offset | $1,\ldots,10$ |
| $S$ | Training anchor stride | $5$ |
| $A$ | Padded number of decoded anchors per sample | $32$ in training; $156$ densely |
| $a$ | Position in an anchor array | Not itself a stored time |
| $\varphi_b$ | Training-grid phase for sample $b$ | $0,\ldots,4$ |
| $\ell$ | Lag measured backwards from the anchor | $0,\ldots,90$ |
| $L$ | Number of candidate lags | $91$ |
| $F_u$ | Earliest source index a lag may read | $0$ |
| $C_Y^{\rm dec}$ | Declared FHR width before model gating | $80=36+44$ |
| $C_Y$ | Kept FHR width, also decoder output width | $76=32+44$ |
| $C_U$ | Kept UA width | $46=36+10$ |
| $c,j$ | Target/source channel indices | Positional within their respective streams |
| $d_m$ | Target state and conditioning width | $128$ |
| $d_z$ | Latent width | $64$ |
| $d_D$ | Decoder hidden width | $256$ |
| $W_c,W'_c$ | Channel warm-up before/after trimming | Channel-dependent |
| $d_c$ | Optional input-channel delay in stored steps | Zero in the main configuration |
| $s_c$ | Optional signed forecast-label shift | Zero on the stored forecast clock |
| $\lambda^{p/q}$ | Latent log-variance | $\log((\sigma^{p/q})^2)$ |
| $\nu^{p/q}$ | Observation log-variance emitted by the decoder | Distinct from latent variance |
| $\lambda_{\rm full},\lambda_{\rm base}$ | Reconstruction loss weights | Both $1$ |
| $\beta,\beta_p$ | Source-KL and prior-scale penalties | Ramped $\beta$; $\beta_p=0.1$ |

We write $Y_{t,c}$ and $U_{t,j}$ for standardized stored features, not the raw traces. A kept-channel index is different from a declared-channel index whenever channels have been dropped.

The notation $\odot$ means elementwise multiplication; $[x;y]$ means concatenation; $\operatorname{vec}$ flattens channel coordinates in their existing order; and $\operatorname{diag}(v)$ places a vector on a diagonal matrix.

## 3. How raw recordings become the dataset

### 3.1 Dataset lineage

The configured files are HDF5 shards produced by the repository's recording pipeline. Its main entry is `hdf5_dataset/new_pipeline/create_new_pipeline.py`. It uses the EarlyMaestra MIMO adaptor to read MATLAB EFM recordings, separate continuous sections, resample where necessary, form segments, and compute sample weights.

The production geometry before loader trimming is:

$$
5280\text{ raw samples}=1320\text{ s}=22\text{ min},
\qquad 5280/16=330\text{ stored steps}.
$$

Segments are constructed with a nominal 20-minute step, hence two minutes of overlap. With `trim_minutes: 1.0`, the loader removes one minute from **each** end:

$$
5280-2(240)=4800\text{ raw samples},\qquad
330-2(15)=300\text{ stored steps}.
$$

Trimming slices already-computed coefficients; it does not rerun a filter from a new zero-history boundary. The retained first coefficient therefore already has one minute of within-segment preprocessing history behind it.

The pipeline screens segments using sample-weight and flatness criteria, deduplicates equal segment starts by retaining the higher-weight segment, and stores recording identity and segment timing. Its broader machinery supports pretraining splits and clinical classification folds. Those clinical labels are not the forecast labels used by this model.

### 3.2 The fields that actually matter to this model

| HDF5 field | Stored meaning | Model use |
|---|---|---|
| `fhr_st` | FHR order-zero and first-order scattering | Target input and future target |
| `fhr_ph` | FHR self phase-harmonic coefficients | Target input and future target |
| `up_st` | UA scattering | Source input when `use_up_st: true` |
| `up_ph` | UA self phase-harmonic coefficients | Source input |
| `weight` | Decimated FHR validity/quality weight | Forecast-loss selection |
| `guid` | Recording identifier | Tiling, grouping, and cross-recording controls |
| `epoch` | Segment start/domain time, not the training epoch | Tiling identity and metadata |
| `fhr`, `up` | Raw traces | Context plots; not encoder inputs in this architecture |

HDF5 coefficients have per-sample shape $(C,330)$ before trimming. `CombinedHDF5Dataset` trims and normalizes them, then transposes them to $(300,C)$. Collation adds the batch axis.

The task assembles

$$
Y^{\rm dec}=\operatorname{concat}(\texttt{fhr\_st},\texttt{fhr\_ph}),
\qquad
U^{\rm dec}=\operatorname{concat}(\texttt{up\_st},\texttt{up\_ph}).
$$

With `use_up_st: false`, the source is `up_ph` alone and its declared width must change accordingly. The main configuration keeps both source blocks.

No FHR–UA cross-phase block enters the target stream or the labels. Such a feature would put actual source values into the allegedly target-only branch. Causal-dataset preflight checks reject this incompatible field choice.

### 3.3 What the supplied paths establish

`configs/default.yaml` names two pretraining training shards and two validation/test shards, split by CS status. The paths contain `REPOINT_ME_causal_int`; they are deployment placeholders. `configs/tiny.yaml` instead points at the committed integer-causal fixture and uses the **same fixture file for training and testing**, suitable for integration checks, not held-out scientific evaluation.

The offline evaluation override names eight clinical evaluation shards and a statistics file. That is a separate configured population from the pretraining files. File names and comments alone do not establish cohort sizes, actual recording disjointness, or which records fitted the normalization statistics. Those require the real build provenance and artifacts.

### 3.4 Causality begins before the neural network

The pipeline applies `up_shift_secs=-20`: it shifts UA earlier before feature extraction. The stored source clock already contains this convention. Applying another 20-second correction inside the model would shift it twice.

There are also upstream operations that require care when interpreting a forecast as a real-time raw-signal forecast:

- The adaptor can use `scipy.signal.resample`, a whole-section Fourier resampling operation.
- `interpolate_bad_values` linearly interpolates nonfinite samples using valid knots on both sides where available.
- Segmentation includes padding/availability handling and selection based on segment quality.

Thus the neural model's causal dependence on **stored inputs** and the causal support of the feature filters are concrete structural properties. They do not, by themselves, prove raw-prefix equivalence for the entire recording pipeline. The 20-second UA advance also changes the relationship between stored time and acquisition time. Any deployment-time or biological interpretation must use the actual acquisition/preprocessing convention.

## 4. The causal feature transform

Feature extraction happens before model training. It is not a learned layer of `SeqVaeLagResidualTrfCfs`, and model gradients do not train its filters.

### 4.1 Causal convolution from first principles

For a raw signal $x[n]$ and a finite filter $k[r]$, a causal convolution is

$$
(x*k)[n]=\sum_{r=0}^{R-1}k[r]x[n-r].
$$

Only the present and past are read. At the beginning of a segment, $x[n-r]$ may refer to time before the segment. The feature implementation supplies assumed history, normally edge replication. Early filter outputs consequently depend partly on that assumed history.

A band-pass wavelet is complex-valued. Its response

$$
w_i[n]=(x*\psi_i)[n]=|w_i[n]|e^{\mathrm i\theta_i[n]}
$$

has an amplitude and a phase. Amplitude describes the strength of a frequency component; phase describes its position within an oscillation.

### 4.2 The causal bank

The repository matches a causal complex-gammatone bank to the frequencies and bandwidths of its vendored Kymatio reference bank. Production bank settings are $J=11$, $Q=4$, low-pass scale $16$, and sampling frequency $4$ Hz.

Using raw-sample coordinates, the envelope has order $n_g=4$:

$$
a_i[r]\propto r^{n_g-1}e^{-2\pi b_i r}\mathbb 1[r>0].
$$

If $\sigma_i$ is the reference Gaussian spectral width, bandwidth matching gives

$$
b_i=\frac{\sigma_i\sqrt{\log 2}}{\sqrt{2^{1/n_g}-1}}.
$$

The unnormalized causal wavelet is

$$
\psi_i^{\rm raw}[r]=a_i[r]\big(e^{2\pi\mathrm i\xi_i r}-\kappa_i\big),
\qquad
\kappa_i=\frac{\sum_r a_i[r]e^{2\pi\mathrm i\xi_i r}}{\sum_r a_i[r]}.
$$

Subtracting $\kappa_i a_i$ removes DC response. The wavelet is then divided by $\sum_r|\psi_i^{\rm raw}[r]|$, giving unit $L^1$ norm. The low-pass $\phi$ is a nonnegative gamma envelope normalized to sum to one. The production causal kernels have $2^{15}$ taps. That finite support is much longer than the dominant warm-up intervals discussed below.

The NumPy reference is `causal_scattering.py`; `causal_scattering_torch.py` implements batched execution using the same bank and channel plan. It does not define a different learned feature family.

### 4.3 Scattering channels

The stored causal scattering block contains order zero and order one:

$$
S_0x[t]=(x*\phi)[Dt],
\qquad
S_{1,i}x[t]=(|x*\psi_i|*\phi)[Dt].
$$

$S_0$ retains a smoothed signal level. A first-order channel measures a smoothed amplitude at a selected scale. Modulus removes the wavelet phase, which is why the separate phase block is useful.

The causal storage channel plan removes wavelets whose warm-up exceeds the untrimmed segment. At the production geometry, each scattering block stores $36$ channels: one order-zero channel and $35$ first-order channels. It does not use the two-sided sibling's $43$-channel scattering layout.

### 4.4 Integer phase harmonics

For integer $k$, define the phase harmonic

$$
[w]^k=|w|e^{\mathrm i k\operatorname{Arg}(w)},\qquad [0]^k=0.
$$

This preserves amplitude while multiplying phase. It is **not** the ordinary complex power $w^k$, which also raises amplitude to power $k$.

For a pair of filters with approximate frequency relation $\xi_j\approx k\xi_i$, a phase coefficient compares the accelerated low-frequency phase with the higher-frequency phase:

$$
\Phi_{ij}x[t]
=\operatorname{Re}\left\{\left([w_i]^k\overline{w_j^{\rm al}}*\phi\right)[Dt]\right\}.
$$

Both legs come from the **same raw signal** for these self-phase blocks. The FHR and UA phase selections use different frequency bands: approximately $[0.008,1.00]$ Hz for FHR and $[0.008,0.05]$ Hz for UA.

The configured operator is `integer_harmonic_v1`. It keeps integer harmonics $k\in\{2,4\}$ and produces $44$ FHR phase channels and $10$ UA phase channels. The legacy `ratio_power_v0` used some noninteger frequency ratios as phase exponents. Those produce a discontinuity at the principal-angle branch cut; they are not interchangeable with the configured operator.

### 4.5 Aligning the two phase legs

The main configuration requires `causal_leg_alignment: envelope`. This is a **feature-construction** operation inside a phase pair, distinct from delaying complete feature channels at model input.

The faster response is delayed and its carrier rotation corrected:

$$
w_j^{\rm al}[n]=w_j[n-r_{ij}]e^{2\pi\mathrm i\xi_j r_{ij}},
\qquad r_{ij}\ge0.
$$

Here the shift is $r_{ij}=\operatorname{round}(f_s(\tau_i-\tau_j))$, using the wavelet group delays in seconds and raw sampling frequency $f_s=4$ Hz. The phasor compensates the carrier rotation introduced by the time shift. Delaying a previously causal response remains causal. Leading indices use edge replication.

This within-pair shift uses the full group-delay difference. The separate optional **model-input channel alignment** in Section 5.4 uses an approximate envelope-content factor $\gamma=1-1/(2n_g)=0.875$. Neither operation gives an exact physiological timestamp for every signal passing through the nonlinear feature chain.

### 4.6 Warm-up and filter delay are different quantities

A channel's warm-up describes how long to wait before the dominant filter contribution comes from the observed segment. Its nominal group delay describes how old the represented content is.

For a primitive kernel, the warm-up measurement uses its $95\%$ cumulative energy boundary. The channel plan composes those primitive boundaries: wavelet plus low-pass for scattering, and the maximum of the two legs plus low-pass for phase. It ceiling-rounds the result into stored steps. This is a composed practical boundary, **not** a proof that all pre-segment influence vanishes at that step or that every nonlinear channel has exactly $95\%$ observed information.

For an idealized gamma envelope, nominal group delay scales as

$$
\tau_i\approx\frac{n_g}{2\pi b_i}
$$

in raw samples. The recorded channel delay adds the low-pass delay to the wavelet delay; phase pairs use the slower leg's delay. Smooth features can therefore be causal while summarizing content hundreds of seconds old.

The optional `causal_novelty_curve` records cumulative composed envelope mass as a function of future window length. It supports a horizon-dependent novelty **proxy**. It does not measure an exact fraction of future information in a nonlinear coefficient, and it does not alter the main forward or objective.

## 5. Normalization, channel selection, and availability

### 5.1 Fixed feature transforms and training statistics

The loader applies a field-specific transform followed by channelwise standardization:

$$
g_c(x)=
\begin{cases}
x,&\text{scattering channel zero},\\
\log(\max(x,0)+10^{-6}),&\text{other scattering channels},\\
\operatorname{asinh}(x),&\text{phase channels},
\end{cases}
\qquad
\bar x_c=\frac{g_c(x_c)-m_c}{s_c+10^{-8}}.
$$

The statistics file supplies $m_c$ and variance $s_c^2$. The statistics calculator excludes each causal channel's leading warm-up region and accumulates finite transformed values. Correct experimental use fits these statistics on the declared training population. The calculator itself works on whichever file list it is given; a filename is not proof that the list was training-only.

There is no whole-segment, per-example standardization in the model. Such a transformation could use future values to change an earlier standardized input. Neural LayerNorm and RMSNorm, introduced later, operate across hidden coordinates at a single stored time; they are a different operation.

### 5.2 Rebase the warm-up after trimming

The one-minute leading trim removes $15$ stored steps. Therefore

$$
W'_c=\max(W_c-15,0).
$$

The resolver reads these boundaries and operator metadata from every configured shard. It checks that the files agree and that their trimmed lengths match the model geometry.

`causal_warmup_budget_steps: 134` keeps target channels satisfying $W'_c\le134$. On the production integer-causal channel plan, the kept declared target indices are

$$
\mathcal C_Y=\{0,\ldots,31\}\cup\{36,\ldots,79\}.
$$

Thus scattering indices $32,33,34,35$ are dropped by the **model** gate, while all $44$ phase channels remain. This is a second channel selection after the dataset's earlier storage-level selection.

The unaligned main configuration keeps **all source channels**. It does not apply the target's 134-step cutoff to source channels. Slow source channels are masked until their own waits expire; the largest rebased source wait in the fixture's production channel plan is $278$.

### 5.3 Per-channel availability

With no extra alignment, target/source channel availability at stored step $s$ is

$$
m_{s,c}=\mathbb 1[s\ge W'_c].
$$

If an input channel is delayed by $d_c\ge0$, its gate emits the value from $s-d_c$, so its combined wait becomes $W'_c+d_c$:

$$
X^{\rm aligned}_{s,c}=X_{s-d_c,c},
\qquad m^{\rm aligned}_{s,c}=\mathbb 1[s\ge W'_c+d_c],
$$

with a legal leading fill for delayed positions. A warm-up mask and a channel delay are therefore not interchangeable: one suppresses an early region, the other changes the time read at every step.

The source encoder explicitly rejects nonfinite values in positions declared available. It safely substitutes nonfinite values in unavailable positions before masking, because $0\times\mathrm{NaN}$ is still NaN. This guarantee should not be generalized to all model inputs: the inherited target adapter multiplies by its availability mask without the source encoder's explicit nonfinite-sanitization/refusal path. Targets and target inputs must be finite under the dataset contract.

There is no independent measured UA-quality mask in this model's forward API. The source mask comes from fixed warm-up/alignment metadata. FHR forecast-quality weights must not be reinterpreted as measured UA quality.

### 5.4 Optional channel clocks

The main configuration has no extra target or source alignment, and scores the `stored` clock. For completeness, the resolver also supports channel delays of the form

$$
d_c=\operatorname{round}\left(\gamma\frac{\tau_{\rm ref}-\tau_c}{\Delta}\right),
\qquad \gamma=0.875,
$$

dropping a channel whose nominal delay exceeds the chosen reference rather than advancing that input into its future. Source and target references can differ.

The forecast target may be reindexed separately:

$$
V_{t,h,c}=Y_{t+h+s_c,c}.
$$

On `stored`, $s_c=0$. On `input`, $s_c=-d_c$. On the optional `physical` clock, channels are advanced relative to the fastest kept target's nominal delay, using the same approximate factor and rounding convention. The latter name does not make those nominal content times exact physiological times.

Positive target shifts shorten the usable anchor range. Shifted scoring also pools quality conservatively over the interval spanned by the shifts. These alternatives change the prediction question; the rest of this document's numerical examples use $d_c=s_c=0$.

## 6. Forecast anchors, labels, and masks

### 6.1 Anchor arrays

An anchor identifies where observation stops and prediction begins. On the stored clock, valid anchors satisfy

$$
F\le t<T-H.
$$

The final anchor is $289$, because its last label is $289+10=299$, the final stored step. Dense evaluation uses all anchors $134,\ldots,289$, giving

$$
300-10-134=156\text{ anchors}.
$$

Training uses a sparser grid:

$$
t_{b,a}=F+\varphi_b+aS,\qquad t_{b,a}<T-H.
$$

The task computes $\varphi_b$ using an eight-byte BLAKE2b digest of recording GUID, floored segment-start time, training epoch, and run seed, modulo $S$. This is reproducible across processes and does not consume the latent-sampling RNG. Including the training epoch changes the phase over epochs; it does not guarantee a perfectly balanced deterministic five-epoch cycle.

For $S=5$, the five phases have $32,31,31,31,31$ real anchors. Arrays are padded to $A=32$ by repeating the last legal index with `anchor_valid=False`. That padded entry must never become another scored observation.

### 6.2 Forecast labels

For kept channel $c$, the stored-clock target is

$$
V_{b,a,h,c}=Y_{b,t_{b,a}+h,c},\qquad h=1,\ldots,H.
$$

Its shape is $(B,A,H,C_Y)$. The label builder selects kept columns from the original standardized target stream. It does not accidentally apply input-channel delays to labels. Explicit forecast-clock shifts are handled by a separate operation.

Future target values are accessed by the loss builder, not the prediction encoder.

### 6.3 Scored support

Let $w_{b,s}$ be the dataset's decimated FHR weight. The loss converts it to

$$
v^Y_{b,s}=\mathbb 1[w_{b,s}\ge1].
$$

Thus a nearly valid weight below one does not count as a valid forecast element. A segment-level mean-quality gate upstream is distinct from this element-level rule.

The implemented future coverage is

$$
\rho_{b,a}=\frac1H\sum_{h=1}^{H}v^Y_{b,t_{b,a}+h}.
$$

The forecast mask is

$$
M_{b,a,h}
=\operatorname{anchorvalid}_{b,a}\,
\mathbb 1[t_{b,a}\ge F]\,
v^Y_{b,t_{b,a}}\,
\mathbb 1[\rho_{b,a}\ge0.9]\,
v^Y_{b,t_{b,a}+h}.
$$

At $H=10$, at least nine future steps must be valid, and the anchor itself must also be valid. Accepted windows with one missing step score nine steps; the missing step is not imputed into the objective or rescaled to a complete block.

Define contributing anchors and their count as

$$
C_{b,a}=\mathbb 1\left[\sum_hM_{b,a,h}>0\right],
\qquad N=\sum_{b,a}C_{b,a}.
$$

The same $C$ selects reconstruction, source KL, and prior-scale regularization. The future mask is used only to decide which predictions can be scored. It never enters the neural forward.

The network may still compute a prediction for an anchor subsequently excluded by this quality rule. “Decoded” and “scored” are different sets.

## 7. The complete forward computation

The architecture can be written as the following sequence of operations:

$$
\begin{aligned}
Y^{\rm dec}&\longrightarrow\text{target gate and availability adapter}
\longrightarrow\text{causal conv--Transformer}\longrightarrow h_t^Y,\\
(h_t^Y,\chi_t)&\longrightarrow h_t
\longrightarrow(\mu_t^p,\lambda_t^p),\\
U^{\rm dec}&\longrightarrow\text{source gate and pointwise encoding}
\longrightarrow E_{t,\ell},\\
(h_t,E_{t,\ell},\zeta_\ell)&\longrightarrow(r^\mu_{t,\ell},r^\sigma_{t,\ell})
\longrightarrow(a_t,b_t),\\
(\mu_t^p,\lambda_t^p,a_t,b_t)&\longrightarrow(\mu_t^q,\lambda_t^q),\\
(P_t,Q_t,\epsilon_t)&\longrightarrow(z_t^p,z_t^q),\\
(z_t^p,Y_t)&\longrightarrow(\widehat\mu_t^p,\nu_t^p),\\
(z_t^q,Y_t)&\longrightarrow(\widehat\mu_t^q,\nu_t^q).
\end{aligned}
$$

The last two arrows are calls to the **same decoder object**. $Y_t$ supplies only the configured target-persistence path. UA values, source proposals, and the target encoder state are not passed directly to that decoder.

The target encoder runs over all $T$ stored positions. Its state is gathered at `anchor_index` before the prior and source-proposal computation. From that point onward, the second tensor axis is the anchor axis $A$, not time $T$.

The implementation composes three classes in this order:

1. `CausalWarmupInputs`: adapters, geometry, and anchor building.
2. `CausalFeatureForecastTarget`: feature-label domain, channel weights, and clocks.
3. `LagResidualCore`: actual neural modules.

The final model overrides `forward`, `build_lag_mask`, `_prior_clock`, and `compute_loss`. Reading only an inherited sibling forward would describe the wrong source architecture.

## 8. Encoding FHR history

### 8.1 Basic operations used throughout

A linear layer computes $Wx+b$. GELU is the smooth activation $x\Phi(x)$, where $\Phi$ is the standard normal CDF. SiLU is $x\operatorname{sigmoid}(x)$.

LayerNorm normalizes across the last, hidden-coordinate axis:

$$
\operatorname{LN}(x)=g\odot\frac{x-\operatorname{mean}(x)}{\sqrt{\operatorname{var}(x)+\varepsilon}}+b.
$$

RMSNorm rescales without subtracting the mean:

$$
\operatorname{RMS}(x)=g\odot\frac{x}{\sqrt{d^{-1}\sum_{i=1}^d x_i^2+10^{-5}}}.
$$

Both are time-local when applied to $(B,T,d)$ tensors along their last axis. Dropout randomly removes and rescales hidden activations during training and is disabled in evaluation. A residual connection adds a transformed branch back to its input, making it easier to refine an existing representation.

### 8.2 The availability adapter

Let $y_t\in\mathbb R^{76}$ be the kept target vector and $m_t^Y$ its availability mask. Where availability modules are built, the adapter first forms

$$
e_t=W_y(y_t\odot m_t^Y)+b_y+W_m(m_t^Y-\mathbf1)
+\mathbb 1[\text{all channels unavailable}]e_{\rm start}.
$$

The start embedding is only constructed if every channel has a positive initial wait. It is absent for the production plan, which has immediately available channels. The mask projection is present because some channels have positive waits.

The result passes through

$$
x_t^{(0)}=\operatorname{ResMLP}\left(\operatorname{Dropout}(\operatorname{GELU}(\operatorname{LN}(e_t)))\right).
$$

Its width is $128$. The adapter residual MLP has four linear maps at width $128$, input LayerNorm, normalization/activation/dropout on intermediate body layers, and an identity skip from the normalized input. It has no final post-skip activation in this construction.

The mask distinguishes an unavailable zero from an observed coefficient equal to its standardized mean.

### 8.3 Two gated causal convolution blocks

The target stem uses kernel/dilation pairs $(5,1)$ and $(9,2)$. For one block, the implemented computation is

$$
\begin{aligned}
(v_t,g_t)&=W_{\rm in}\operatorname{RMS}(x_t),\\
p_t&=v_t\odot\operatorname{sigmoid}(g_t),\\
c_t&=\operatorname{DWConv}_{k,d}(p)_t,\\
x'_t&=x_t+\gamma_{\rm layer}\odot
\operatorname{Dropout}\left(W_{\rm out}\operatorname{SiLU}(\operatorname{RMS}(c_t))\right).
\end{aligned}
$$

Depthwise convolution filters each hidden channel separately through time, reading offsets $0,d,\ldots,(k-1)d$. Padding is exclusively on the left. The projections mix hidden coordinates at the same time.

The learned LayerScale vector $\gamma_{\rm layer}$ begins at $0.01$. The combined stem receptive field is

$$
R_{\rm conv}=1+(5-1)\cdot1+(9-1)\cdot2=21\text{ stored samples}.
$$

The oldest center lies $20\times4=80$ seconds behind the newest. Counting sample widths instead gives $84$ seconds; these are different conventions.

### 8.4 Six causal Transformer blocks

Each target Transformer block contains four attention heads, head width $128/4=32$, and a SwiGLU feed-forward width of $512$.

For head $m$, normalized input states produce queries, keys, and values:

$$
q_t^m=W_Q^m\operatorname{RMS}(x_t),\quad
k_s^m=W_K^m\operatorname{RMS}(x_s),\quad
v_s^m=W_V^m\operatorname{RMS}(x_s).
$$

Queries and keys receive rotary positional encoding. For coordinate pair $r$, the rotation angle at position $t$ is

$$
\theta_{t,r}=t\,10000^{-2r/d_{\rm head}},\qquad
R(\theta)=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}.
$$

This lets a query–key dot product depend on relative position. Values are not rotated.

Attention weights are normalized only over the causal prefix:

$$
\alpha_{t,s}^m=
\frac{\exp(\langle R_tq_t^m,R_sk_s^m\rangle/\sqrt{32})}
{\sum_{r=0}^{t}\exp(\langle R_tq_t^m,R_rk_r^m\rangle/\sqrt{32})},
\qquad 0\le s\le t.
$$

The attention result is $\sum_{s\le t}\alpha_{t,s}^m v_s^m$, concatenated across heads and projected back to width $128$. The kernel itself uses no attention-weight dropout; dropout is applied after the output projection.

The full block is

$$
\begin{aligned}
x'&=x+\gamma_A\odot\operatorname{AttentionBranch}(x),\\
x''&=x'+\gamma_F\odot\operatorname{FFN}(\operatorname{RMS}(x')),\\
\operatorname{FFN}(r)&=\operatorname{Dropout}\left[
W_o\big(\operatorname{SiLU}(W_gr)\odot W_vr\big)\right].
\end{aligned}
$$

Both LayerScale vectors begin at $0.01$. After six blocks, a final RMSNorm gives $h_t^Y$.

The target Transformer has no finite attention window: it can access the entire available segment prefix. The 21-sample convolution reach is therefore **not** the complete target encoder reach. There is no target LSTM in this model.

## 9. The metadata clock and target-only prior

### 9.1 A source-independent clock

For $k=1,\ldots,d_m/2$, define

$$
\chi_{t,2k-2}=\sin\left(\frac{2\pi kt}{T}\right),\qquad
\chi_{t,2k-1}=\cos\left(\frac{2\pi kt}{T}\right).
$$

The shared conditioning state is

$$
h_t=h_t^Y+W_A\operatorname{LN}(\chi_t).
$$

The clock is a deterministic buffer, identical for all recordings with the same geometry. $W_A$ is a learned, bias-free $128\times128$ projection initialized to zero. Its LayerNorm is learnable. The prior head is constructed without a second internal clock projection.

Why supply this clock? Source availability varies with stored position because channels warm up at different times. Providing position to the base as well as the full branch reduces the chance that a deterministic availability schedule alone is credited as new source information. This is a representational provision, not a proof that the fitted prior has perfectly learned the schedule.

The clock carries no actual UA values, GUID, outcome label, or segment-start clinical time. If source availability later becomes recording-dependent, this fixed clock is insufficient to communicate that new metadata to both branches.

### 9.2 Two full-width prior heads

Separate mean and log-variance heads each apply their own input LayerNorm and residual MLP. At production widths, each MLP body follows

$$
128\longrightarrow111\longrightarrow97\longrightarrow84\longrightarrow74\longrightarrow64.
$$

The body normalizes its input again, applies LayerNorm/GELU/dropout after intermediate linear maps, and leaves its last layer linear. A learned $128\to64$ projection of the normalized input is added as a skip. Mean and log-variance heads do not share these weights.

Writing their raw outputs as $u_t^p$ and $\eta_t^p$,

$$
\mu_t^p=5\tanh(u_t^p/5),\qquad
\lambda_t^p=-5+8\operatorname{sigmoid}(\eta_t^p),\qquad
\sigma_t^p=e^{\lambda_t^p/2}.
$$

The resulting diagonal Gaussian is $P_t$. The mean bound is in latent units. The log-variance interval is approximately $[-5,3]$, with open endpoints mathematically and possible endpoint rounding numerically.

## 10. Preserving source values and gathering lags

### 10.1 Pointwise source encoding

The recommended source encoder has no trainable parameters. For an available standardized coefficient, it emits

$$
e_{s,j}=[x^{\rm safe}_{s,j},m^U_{s,j}],\qquad
x^{\rm safe}_{s,j}=\begin{cases}U_{s,j},&m^U_{s,j}=1,\\0,&m^U_{s,j}=0.\end{cases}
$$

Its shape is $(B,T,46,2)$. An observed coefficient equal to zero is represented as $[0,1]$; an unavailable coefficient is $[0,0]$.

No source temporal convolution, recurrence, attention, pooling, time normalization, or dropout occurs in this recommended encoding. With metadata fixed,

$$
\frac{\partial e_{s,j}}{\partial U_{r,k}}=0
\quad\text{whenever }(r,k)\ne(s,j).
$$

This preserves each available stored coefficient exactly. It does not undo the raw-history mixing already inside that coefficient.

### 10.2 Lag gathering

Anchor $t$ and lag $\ell$ read source step

$$
s_{t,\ell}=t-\ell.
$$

The main model allows $\ell=0$, so it may use the source coefficient **at** the anchor. It predicts strictly later target steps.

The gathered channel mask is

$$
m_{t,\ell,j}
=\mathbb 1[F_u\le t-\ell<T]\,
\mathbb 1[t-\ell\ge W'_j+d_j^U].
$$

The encoder has already enforced finiteness on available source positions. The gather computes range validity before accessing memory, clamps illegal indices to a safe surrogate, then zeros every coordinate ruled out by the combined mask. This prevents negative indices from wrapping around to future values at the end of the sequence.

Define

$$
E_{t,\ell}=(e_{t-\ell,j})_{j=1}^{46},\qquad
v_{t,\ell}=\mathbb 1\left[\sum_jm_{t,\ell,j}>0\right].
$$

$E$ has shape $(B,A,L,46,2)$ and flattens to width $92$ per anchor–lag pair. $v$ says whether at least one source channel is available. A lag can be in range while most of its channels are still cold.

### 10.3 Optional scalar lift

With `source_scalar_lift: true`, each channel appends its own two learned features:

$$
e_{s,j}=[x^{\rm safe}_{s,j},m^U_{s,j},\phi_j(x^{\rm safe}_{s,j})],
\qquad \phi_j:\mathbb R\to\mathbb R^2.
$$

Each $\phi_j$ is a separate $1\to8\to2$ GELU MLP. No channel or time mixing occurs in this lift. Identity and mask remain, so the original value is still recoverable. Source width becomes $184$ per lag, and the proposal input width becomes $128+184+8=320$.

The main configuration disables this lift.

## 11. Local proposals and Gaussian residual updates

### 11.1 One shared network, one evaluation per lag

Each lag has a learned embedding $\zeta_\ell\in\mathbb R^8$, initialized with standard deviation $0.02$. It identifies lag position, not source content.

The shared proposal MLP computes

$$
\begin{aligned}
f_1&=\operatorname{GELU}\left(W_1[h_t;\operatorname{vec}(E_{t,\ell});\zeta_\ell]+b_1\right),\\
f_2&=\operatorname{GELU}(W_2f_1+b_2),\\
o_{t,\ell}&=W_3f_2+b_3.
\end{aligned}
$$

Its production widths are

$$
228\longrightarrow128\longrightarrow128\longrightarrow128.
$$

There is no normalization or dropout in this MLP. The implementation computes the first projection as separate target, source, and embedding projections that are added together. This is algebraically the same affine map and avoids materializing a large concatenation.

Split $o_{t,\ell}$ into two 64-vectors. For an intervention selector $s_{t,\ell}$, normally one,

$$
(r^\mu_{t,\ell},r^\sigma_{t,\ell})
=s_{t,\ell}v_{t,\ell}\,\operatorname{split}(o_{t,\ell}).
$$

The mask is applied after the output projection. A completely unavailable lag must contribute exactly zero even if the MLP has a nonzero bias.

Each proposal can combine all available UA channels at its own source step and can interact with FHR context. It cannot read a second source lag. There is one shared set of MLP weights, not 91 independent MLPs.

### 11.2 Sum first, bound second

With configured lag count $L$, the default scale is

$$
c_L=L^{-1/2}=91^{-1/2}\approx0.104828.
$$

The raw summed updates and bounded updates are

$$
\begin{aligned}
\bar a_t&=c_L\sum_{\ell=0}^{L-1}r^\mu_{t,\ell},
&a_t&=a_{\max}\tanh(\bar a_t/a_{\max}),\\
\bar b_t&=c_L\sum_{\ell=0}^{L-1}r^\sigma_{t,\ell},
&b_t&=b_{\max}\tanh(\bar b_t/b_{\max}),
\end{aligned}
$$

where $a_{\max}=3$ and $b_{\max}=1$.

The scale is fixed by the configured lag count; it is **not** recomputed from the number of currently available lags or surviving intervention selectors. Otherwise unchanged evidence would receive a different scale whenever another lag disappeared.

The inverse-square-root rule is a numerical convention. It stabilizes sums of independent equal-variance terms, but actual proposals share weights and source histories are correlated:

$$
\operatorname{Var}\left(c_L\sum_\ell r_\ell\right)
=c_L^2\sum_{\ell,k}\operatorname{Cov}(r_\ell,r_k).
$$

It does not eliminate reinforcement or cancellation.

### 11.3 Residuals are measured relative to the prior

The full Gaussian parameters are

$$
\boxed{
\mu_t^q=\mu_t^p+\sigma_t^p\odot a_t,
\qquad
\lambda_t^q=\lambda_t^p+2b_t.
}
$$

Equivalently,

$$
\sigma_t^q=\sigma_t^p\odot e^{b_t}.
$$

$a$ is a mean shift in **prior standard deviations**. $b$ is a change in **log standard deviation**, explaining the factor two when updating log-variance.

The bounds imply

$$
|\mu_{t,d}^q-\mu_{t,d}^p|\le3\sigma_{t,d}^p,
\qquad e^{-1}\le\frac{\sigma_{t,d}^q}{\sigma_{t,d}^p}\le e,
\qquad -7\le\lambda_{t,d}^q\le5.
$$

The full log-variance is not passed through another $[-5,3]$ sigmoid bound. Doing that would define a different model and generally break equality when the update is zero.

The scale correction can be positive or negative. Useful source evidence can increase conditional uncertainty for a particular example; the architecture does not require uncertainty to shrink pointwise.

### 11.4 Exact absence semantics

If all selectors are zero, or all lags are unavailable,

$$
a_t=b_t=0\quad\Longrightarrow\quad Q_t=P_t.
$$

This also holds at initialization because the proposal output projection is zeroed.

An observed zero source stream is different: its availability bits remain one, and its target-conditioned MLP can emit nonzero updates. The model deliberately does not force a standardized zero to mean “no evidence.”

### 11.5 Where interactions are possible

Before the limiter, the parameter update is a sum of functions local to source lag. That is a real restriction compared with an unrestricted network over the entire source window. However, the limiter and shared nonlinear decoder can combine coordinates containing different lag summaries. Therefore it would be too strong to say the entire model can never represent multi-lag interactions or a product of lagged variables. The architecture restricts the route and capacity of such interactions; synthetic tests measure the practical consequences.

## 12. Sampling and the shared forecast decoder

### 12.1 Reparameterization with common noise

Draw $\epsilon_t\sim\mathcal N(0,I_{64})$ and form

$$
z_t^p=\mu_t^p+\sigma_t^p\odot\epsilon_t,
\qquad
z_t^q=\mu_t^q+\sigma_t^q\odot\epsilon_t.
$$

This rewrites Gaussian sampling as a differentiable function of parameters and parameter-independent noise. Gradients can pass through $\mu$ and $\sigma$.

The same noise is used for both branches. Their sample difference is exactly

$$
z_t^q-z_t^p
=\sigma_t^p\odot\left[a_t+(e^{b_t}-1)\odot\epsilon_t\right].
$$

With $b=0$, this difference is deterministic for fixed inputs. Otherwise it is noise-dependent. It is not an independent random “UA code” added to an independent FHR code.

Setting `model.eval()` disables dropout but **does not disable latent sampling**. Repeated evaluation forwards still draw noise. The offline scorer uses an explicit shared draw loop to make comparisons controlled.

### 12.2 Project the code to decoder width

`BaselineFutureDecoder.proj` maps the 64-dimensional code to $256$ dimensions through a residual MLP:

$$
64\longrightarrow91\longrightarrow128\longrightarrow181\longrightarrow256.
$$

It has input LayerNorm, intermediate LayerNorm/GELU, a projected skip from normalized input, and a final GELU after adding the skip. Decoder dropout is zero.

Call the projected vector $d_t$. Learned horizon embeddings $p_h\in\mathbb R^{256}$ initialize the future tokens:

$$
f_{t,h}^{(0)}=d_t+p_h,\qquad h=1,\ldots,10.
$$

The embeddings begin with standard deviation $0.8$. They give different forecast offsets different initial representations even though each starts from the same latent code.

### 12.3 Four horizon convolution blocks with FiLM

The horizon core reshapes each anchor into its own sequence of $H$ generated tokens. It uses ordinary convolutions across the **forecast-offset axis**, with kernel $3$, dilations $1,2,4,8$, and symmetric padding.

One block computes

$$
\begin{aligned}
g^{(r)}&=\operatorname{GELU}\left(\operatorname{GroupNorm}
\left(\operatorname{Conv}^{(r)}_H(f^{(r)})\right)\right),\\
(\gamma_t^{(r)},\delta_t^{(r)})&=A_r d_t+b_r,\\
f_{t,h}^{(r+1)}&=f_{t,h}^{(r)}+
(1+\gamma_t^{(r)})\odot g_{t,h}^{(r)}+\delta_t^{(r)}.
\end{aligned}
$$

FiLM means feature-wise linear modulation: the latent supplies a scale and offset for each hidden coordinate in a block. Production GroupNorm uses eight groups and normalizes over group coordinates and the generated horizon positions within an anchor.

The nominal convolution receptive field is $1+2(1+2+4+8)=31$ horizon positions, wider than $H=10$. GroupNorm and the following attention also communicate across generated positions.

These decoder operations need not be causal across $h$: all their tokens are generated from information at the anchor. They do not receive future observed FHR or UA. This is simultaneous block prediction, not teacher forcing or autoregressive rollout of ground-truth future samples.

### 12.4 Two horizon attention blocks and the outer skip

Two four-head attention blocks operate over all $H$ generated positions. Each uses pre-LayerNorm, bias-free query/key/value/output projections, standard scaled dot-product attention, and a learned scalar residual gain initialized at $0.01$:

$$
f\leftarrow f+g_AW_O\operatorname{MHA}(\operatorname{LN}(f)).
$$

These particular horizon blocks have no additional Transformer feed-forward sublayer. They differ from the target encoder's RMSNorm/SwiGLU blocks.

The core then adds its saved initial token tensor and applies output LayerNorm:

$$
f^{\rm out}_{t,h}=\operatorname{LN}\left(f^{\rm refined}_{t,h}+f^{(0)}_{t,h}\right).
$$

The saved tensor is added after the internal residual convolutions and attention. This outer skip is part of the implemented computation.

### 12.5 Observation distribution and persistence

Two linear heads produce the observation parameters:

$$
\widehat\mu_{t,h,c}(z)=W_{\mu,c}f^{\rm out}_{t,h}+b_{\mu,c}
+\omega_{h,c}Y_{t,c},
$$

$$
\nu_{t,h,c}(z)=-5+8\operatorname{sigmoid}
\left(W_{\nu,c}f^{\rm out}_{t,h}+b_{\nu,c}\right).
$$

The learned persistence coefficients start at

$$
\omega_{h,c}=2^{-(h-1)/5}.
$$

They are trainable unconstrained parameters after initialization. They are separate from the normalized horizon loss weights, despite sharing an initial exponential shape.

The default persistence input is the kept standardized FHR vector at the anchor, gathered before input-channel alignment. With a negative forecast shift, the persistence gather follows that delayed scored clock. Positive target-label advances do not introduce a future-valued persistence input.

Conditional on a code, the forecast density factorizes:

$$
p_\psi(V_t\mid z,Y_t)
=\prod_{h=1}^H\prod_{c=1}^{C_Y}
\mathcal N\left(V_{t,h,c};\widehat\mu_{t,h,c}(z),e^{\nu_{t,h,c}(z)}\right).
$$

Diagonal observation noise does not imply marginal independence after integrating out the shared code. Different future coordinates can co-vary through their common dependence on $z$.

The source has no decoder bypass. FHR does have the explicit persistence route, so the default is not a strict all-information-through-latent model. Setting `persistence_residual: false` removes this route and requires matching the baseline accordingly.

## 13. The exact KL divergence

### 13.1 Derivation

For two diagonal Gaussians,

$$
\operatorname{KL}(Q_t\Vert P_t)
=\frac12\sum_{d=1}^{d_z}\left[
\log\frac{(\sigma_{t,d}^p)^2}{(\sigma_{t,d}^q)^2}
+\frac{(\sigma_{t,d}^q)^2+(\mu_{t,d}^q-\mu_{t,d}^p)^2}{(\sigma_{t,d}^p)^2}-1
\right].
$$

Substitute the residual definitions:

$$
\frac{\sigma^q}{\sigma^p}=e^b,\qquad
\frac{\mu^q-\mu^p}{\sigma^p}=a.
$$

Then

$$
\boxed{K_t=\frac12\sum_d\left[a_{t,d}^2+e^{2b_{t,d}}-1-2b_{t,d}\right].}
$$

The code evaluates the scale contribution as `expm1(2*b) - 2*b`, promoting to at least FP32. This is better behaved near zero than forming $e^{2b}$ and subtracting one separately.

Because $e^x\ge1+x$, $K_t\ge0$. In exact arithmetic, equality holds precisely when $a=b=0$. Near zero,

$$
K_t=\frac12\sum_da_{t,d}^2+\sum_db_{t,d}^2+O(\|b_t\|^3).
$$

The derivatives with respect to the bounded updates are

$$
\frac{\partial K}{\partial a_d}=a_d,
\qquad
\frac{\partial K}{\partial b_d}=e^{2b_d}-1.
$$

They vanish at the zero update. Prediction gradients, rather than a KL gradient, initially move the source output projection away from zero.

The simplified KL has no direct dependence on $\mu^p$ or $\sigma^p$ when $a,b$ are fixed. It still trains shared upstream parameters through the target-conditioned proposal network. It should not be assumed to independently fit the prior to an aggregate posterior.

### 13.2 What is actually decomposable

The total KL is a sum over **latent dimensions**. Latent dimensions are not lag labels.

Even in an unbounded mean-only thought experiment,

$$
\frac12\left\|c_L\sum_\ell r_\ell\right\|^2
=\frac{c_L^2}{2}\sum_\ell\|r_\ell\|^2
+c_L^2\sum_{\ell<k}\langle r_\ell,r_k\rangle.
$$

The cross-terms can cancel. Two scalar proposals $+1$ and $-1$ sum to zero and give zero total KL, although each has a nonzero isolated norm. The actual limiter and scale updates do not restore an additive lag decomposition.

The model therefore emits `kld_per_anchor_dim` and `kld_per_anchor`, and does not emit an exact per-lag KL map.

### 13.3 Prior-scale regularization

The separate prior penalty is

$$
R_{p,t}=\frac12\sum_d\left[e^{\lambda_{t,d}^p}-1-\lambda_{t,d}^p\right].
$$

It encourages unit prior variance. It does not penalize the prior mean. In particular,

$$
\operatorname{KL}(P_t\Vert\mathcal N(0,I))
=R_{p,t}+\frac12\|\mu_t^p\|^2,
$$

so calling $R_p$ the full KL to a standard normal would be incorrect. Its role is to oppose collapse of the base variance toward a deterministic code; it is not a direct measure of FHR information.

## 14. The training objective and its gradients

### 14.1 Conditional Gaussian negative log likelihood

For branch $r\in\{p,q\}$ and one latent sample, each target coordinate contributes

$$
d^r_{b,a,h,c}=\frac12\left[
\log(2\pi)+\nu^r_{b,a,h,c}
+(V_{b,a,h,c}-\widehat\mu^r_{b,a,h,c})^2e^{-\nu^r_{b,a,h,c}}
\right].
$$

This rewards accurate means and appropriately calibrated uncertainty. Larger variance can soften a large residual, but costs a larger log-variance term. A continuous-density NLL need not be nonnegative.

The implementation also supports squared error, $d=(V-\widehat\mu)^2$, but the shipped production and tiny configurations both use `gaussian_nll`. MSE does not train the observation-variance head through reconstruction and does not define the predictive density used by Gaussian evaluation.

### 14.2 Channel and horizon weights

Relative target weights are $1$ for kept scattering channels and $0.1$ for phase channels. They are normalized to sum to $C_Y$:

$$
w_c=C_Y\frac{\widetilde w_c}{\sum_{c'}\widetilde w_{c'}},\qquad
w_{\rm st}=\frac{76}{32+4.4}\approx2.087912,\quad
w_{\rm ph}\approx0.208791.
$$

The phase block receives $4.4/36.4\approx12.09\%$ of total channel weight. The split is computed from declared kept indices, not by taking the first 36 decoder outputs as scattering; only 32 scattering channels survive.

The horizon half-life is five stored steps:

$$
w_h=H\frac{2^{-(h-1)/5}}{\sum_{r=1}^{H}2^{-(r-1)/5}},\qquad \sum_hw_h=H.
$$

The weighted block score is

$$
D^w_{r,b,a}=\sum_{h,c}M_{b,a,h}w_hw_cd^r_{b,a,h,c}.
$$

Weights are fixed for the geometry. They are not renormalized when a window has a missing step.

### 14.3 The objective

Let $K_{b,a,d}$ be each coordinate's raw KL. If free bits $f>0$ are configured, the training penalty is

$$
K^{\rm train}_{b,a}=\sum_d\max(K_{b,a,d},f).
$$

The raw reported KL remains $K_{b,a}=\sum_dK_{b,a,d}$. The main configuration uses $f=0$.

The minimized loss is

$$
\boxed{
\mathcal L=\frac1N\sum_{b,a}\left[
\lambda_{\rm full}D^w_{q,b,a}
+\lambda_{\rm base}D^w_{p,b,a}
+C_{b,a}\left(\beta(e)K^{\rm train}_{b,a}+\beta_pR_{p,b,a}\right)
\right].
}
$$

The main settings are

$$
\lambda_{\rm full}=\lambda_{\rm base}=1,\qquad
\beta_p=0.1,\qquad
\beta(e)=\min(e/50,1),
$$

where $e$ is the zero-based current training epoch. Each forward uses one shared-noise draw for the two reconstruction terms.

Scores are **summed over coefficients and averaged over contributing anchors**. A complete production block has $10\times76=760$ coefficients. Dividing reconstruction by 760 without similarly changing the other terms would increase the effective KL pressure by that factor.

There is no UA reconstruction term, no clinical classification loss, no explicit proposal-norm penalty, and no orthogonality or independence constraint between the correction and target state. Raw-waveform multiscale, derivative, and boundary losses are not part of this objective; nonzero shape-term arguments are refused at the model loss API.

### 14.4 Distributed reduction

With $R$ distributed ranks, each rank computes its local numerator. A packed all-reduce obtains the global contributing-anchor count $N_{\rm global}$ and detached totals for objective reporting.

Because DDP averages gradients across ranks, rank $r$ backpropagates

$$
\mathcal L_r^{\rm backward}
=\frac{R}{N_{\rm global}}\,\operatorname{numerator}_r.
$$

The averaged gradient then equals the gradient of the global anchor mean. Taking a separate mean on each rank would incorrectly give equal weight to ranks with unequal valid-anchor counts.

If no rank has scored support, the loss returns a graph-connected zero. Core score and rate totals are reduced globally. Some diagnostic fractions, saturation measures, and the rank-local `total_loss` representation have different reduction paths; their names should not be assumed to imply the same packed global statistic.

### 14.5 Where gradients flow

The base score trains the target adapter, target encoder, clock, prior heads, and shared decoder. The full score trains these same components plus the source proposal pathway. The KL trains the residual computation and its conditioning path. The prior-scale penalty trains prior variance and upstream components.

There is no stop-gradient freezing the base during joint training. As a result, the base is a moving comparison, not a permanently fixed baseline. An external independently trained target-only predictor is needed to detect whether an apparent improvement in the internal gap came from base degradation.

## 15. Initialization and training orchestration

### 15.1 Why the initial branches are identical

After generic initialization, the implementation repairs specific layers:

- The source proposal output weight and bias are exactly zero.
- The explicit clock projection starts at zero.
- FiLM generator weights and biases are zeroed.
- Depthwise-convolution weights are redrawn with standard deviation $1/\sqrt{k}$.
- Horizon embeddings are redrawn at the configured standard deviation, $0.8$.

Fresh-model head calibration sets prior log-variance to zero by zeroing the final variance-body weight and its skip projection, then setting the final body bias to

$$
\log\frac{0-(-5)}{3-0}=\log(5/3).
$$

The smooth bound maps this bias to zero, so the initial prior standard deviation is one. The decoder mean-head weight is multiplied by $0.02$; the observation log-variance-head weight is multiplied by $0.1$ and its bias set to the same preimage of zero.

Zero FiLM means modulation starts at scale one and offset zero. It does **not** make the entire horizon core an identity: its convolution, normalization, residual, and attention computations still run.

At initialization, $P=Q$, and shared noise makes $z^p=z^q$. The deterministic shared decoder then produces identical base/full outputs, including in training mode. Target/prior dropout is computed once and shared.

The zero source output layer still receives prediction gradients through a nonconstant decoder. Earlier proposal layers may have zero gradients on the first step because that output weight is zero. Once it moves, gradients can propagate farther into the source head. The regression suite checks that this escape from the zero start occurs on a nondegenerate example.

### 15.2 Target-only pretraining and warm start

`target_only.yaml` sets `source_disabled: true`, building no source encoder or proposal head. Its full distribution equals its prior, and both reconstruction terms remain enabled. Therefore its training reconstruction is twice the same sampled branch score, rather than a single score with one term silently removed.

`joint.yaml` starts from a target-only checkpoint through `target_warm_start_checkpoint`, and sets `head_init_calibration: false`. The intent is to preserve learned target/decoder weights and initialize source outputs at zero.

The actual transfer allowlist includes `target_gate`, `target_adapter`, `target_encoder`, `prior_head`, `clock_proj`, `horizon_core`, and `decoder`. **It currently omits the separate `clock_norm` prefix**, so those normalization parameters remain freshly initialized. Missing transferable tensors are reported, and mismatched tensor shapes raise. Transfer does not prove that equal-shaped channel tensors have identical semantic ordering; matching data/configuration provenance is still necessary.

This is a partial weight transfer, not optimizer/scheduler-state resume. `core_model_checkpoint` uses the stricter model-reconstruction/weight-load path and restores the source pathway too; it is not the target-only warm-start mechanism. Supplying both checkpoint keys is refused.

### 15.3 Actual optimizer and learning-rate schedule

The inherited optimizer builder uses AdamW with

$$
\eta_0=3\times10^{-4},\quad
\text{weight decay}=10^{-4},\quad
(\beta_1,\beta_2)=(0.9,0.95),\quad\epsilon=10^{-8}.
$$

At zero-based scheduler step $k$, the implemented positive-warm-up schedule is

$$
\eta(k)=\eta_0\min\left(1,\frac{k+1}{2000}\right)
\,0.1^{\#\{m\in\mathcal M_{\rm step}:m\le k\}}.
$$

Epoch milestones $400$ and $800$ are converted to steps using the trainer's estimated steps per epoch. This formula is the executed `LambdaLR`, rather than the “start at a tenth of LR” wording in some configuration/design comments. At the initial scheduler index its factor is $1/2000$.

Other supplied operational settings include FP32, batch size 128 per device, up to 5000 epochs, norm clipping at 3500, seven listed CUDA devices, and the inherited spike breaker. These are configuration choices, not demonstrated optimal settings or hardware guarantees.

### 15.4 What validation actually selects

The task decodes densely for validation and test, but still evaluates the same **one-draw weighted objective** used by `compute_loss`. The default early-stopping monitor and primary checkpoint monitor are `val/total_loss`; the secondary checkpoint monitor is `val/nll_full_block`.

The latter is also the weighted, single-draw conditional score in this task. It is not the unweighted, marginalized $K$-draw predictive likelihood defined in Section 16. The design's proposed predictive-validation selection policy has not replaced this task's validation step.

## 16. Predictive evaluation

### 16.1 Integrating over latent uncertainty

For branch $r$, the predictive density is

$$
\widehat p_r(V_t\mid\text{inputs})
=\int p_\psi(V_t\mid z,Y_t)\,P_r(dz\mid\text{inputs}),
\qquad P_p=P,\quad P_q=Q.
$$

This is a continuous Gaussian observation mixture. Its latent integral is approximated by Monte Carlo.

For draw $k$, compute the **unweighted** conditional block NLL on the shared mask:

$$
D^{(k)}_{r,t}=\sum_{h,c}M_{t,h}d^{r,(k)}_{t,h,c}.
$$

The predictive mixture NLL is

$$
\boxed{
D^{(K)}_{r,t}
=-\log\left(\frac1K\sum_{k=1}^Ke^{-D^{(k)}_{r,t}}\right)
=-\operatorname{logsumexp}_k(-D^{(k)}_{r,t})+\log K.
}
$$

The offline scorer gives every branch and intervention the same $\epsilon^{(k)}$, persistence values, target coordinates, and mask. It operates directly on anchor-indexed latent parameters; it does not gather them as if they were dense stored-time tensors.

### 16.2 Three different scores

The following are distinct:

$$
\text{score of }\operatorname{decoder}(\mu^r),\qquad
\frac1K\sum_kD^{(k)}_{r,t},\qquad
D^{(K)}_{r,t}.
$$

The first decodes a point code; the second estimates expected conditional NLL; the third estimates predictive mixture NLL. Jensen's inequality gives

$$
-\log\mathbb E_Zp_\psi(V\mid Z,Y_t)
\le\mathbb E_Z[-\log p_\psi(V\mid Z,Y_t)].
$$

Also, the main training weights make its score a composite weighted criterion. Raising density factors to unequal powers does not preserve normalization without additional parameter-dependent normalizers. The weighted training gap is therefore not an unweighted log predictive-density ratio.

### 16.3 Predictive gain and Monte Carlo uncertainty

The reported predictive gain is base minus full:

$$
G^{(K)}=\operatorname{average}_t\left[D^{(K)}_{p,t}-D^{(K)}_{q,t}\right].
$$

A positive value means the full branch assigns greater predictive density to the observed target blocks on average. A negative value means the base branch scores better under this estimator.

The sample mean likelihood is unbiased for model likelihood; taking its negative logarithm introduces upward finite-$K$ bias. The biases of the two branches need not cancel. The committed offline default is $K=8$; the acceptance plan calls for a primary $K=32$ and stability checks at $8,32,128$.

With normalized likelihood weights

$$
\alpha_k=\frac{e^{-D^{(k)}}}{\sum_re^{-D^{(r)}}},
\qquad K_{\rm eff}=\frac1{\sum_k\alpha_k^2},
$$

the scorer reports draw concentration. $K_{\rm eff}$ near one means a small number of draws dominate. These weights are unrelated to source attention and do not prove convergence.

### 16.4 Calibration

For one forecast coordinate, the predictive mean and variance obey

$$
\mathbb E[V]=\mathbb E_Z\widehat\mu(Z),\qquad
\operatorname{Var}(V)=\mathbb E_Ze^{\nu(Z)}+\operatorname{Var}_Z(\widehat\mu(Z)).
$$

The implemented calibration loop averages Gaussian component CDFs:

$$
\widehat F(v)=\frac1K\sum_k
\Phi\left(\frac{v-\widehat\mu(z^{(k)})}{e^{\nu(z^{(k)})/2}}\right).
$$

For each observed coefficient, $u=\widehat F(V)$ is its probability-integral transform. Under an ideal calibrated continuous forecast, $u$ is uniform, with mean $1/2$ and variance $1/12$. Central coverage at probability $q$ checks whether

$$
(1-q)/2\le u\le(1+q)/2.
$$

Averaging conditional standard deviations would omit latent variation. Even a Gaussian interval constructed from total variance is not generally an exact interval of a non-Gaussian mixture.

### 16.5 Recording-level aggregation

Dense anchor windows overlap heavily. They are not independent patients or recordings. The evaluator aggregates scored sums and counts by GUID, writes `per_recording.csv`, and bootstraps complete recordings, by default with 2000 resamples.

It reports equal-recording and anchor-weighted summaries separately. Equal-recording summaries give each recording equal influence; anchor-weighted summaries give longer or more valid recordings greater influence. Means of arbitrary batch means would implement neither reliably.

If one scores horizon/channel subsets, their marginal mixture NLLs generally do not add to the joint block mixture NLL:

$$
\log\mathbb E_Z\prod_i p(V_i\mid Z)
\ne\sum_i\log\mathbb E_Zp(V_i\mid Z).
$$

Conditional per-draw scores do add. A mixture log likelihood is nonlinear in those sums.

## 17. Source controls and lag interpretation

### 17.1 Suppression of lag proposals

For a lag band $\mathcal B$, set its selectors to zero and keep the other proposals, masks, target context, and $c_L$ fixed. Recompute the summed update, bounds, Gaussian parameters, samples, and decoder outputs.

The suppression margin is

$$
J_{\mathcal B}=\operatorname{average}_t
\left[D^{(K)}_{q\setminus\mathcal B,t}-D^{(K)}_{q,t}\right].
$$

Positive $J_{\mathcal B}$ means this fitted model predicts worse when those proposals are removed. Negative values are possible and meaningful. Margins are not normalized to sum to either total predictive gain or total KL.

The configured inclusive lag bands are:

| Band | Lag indices | Anchor-relative center offsets |
|---|---|---|
| `anchor` | $0$–$14$ | $0$–$56$ s |
| `near` | $15$–$44$ | $60$–$176$ s |
| `mid` | $45$–$67$ | $180$–$268$ s |
| `far` | $68$–$90$ | $272$–$360$ s |

`suppress:none` reproduces the matched full branch. `silence` disables every selector and reproduces the base. `suppress:all` removes the union of declared bands, which equals all lags for this partition. Exact endpoint handling avoids tiny subtraction residues being mistaken for source information.

### 17.2 Values, masks, and pairing controls

| Control | Operation | What it probes |
|---|---|---|
| Silence | Set all selectors to zero | Structural equality to the base |
| Observed zeros | Replace source values by zero, keep availability and selectors | Response to real standardized-zero observations and mask/target capacity |
| Observed constant | Replace each channel by its per-segment temporal mean | Dependence on temporal detail rather than that summary |
| Permutation | Pair complete source segments with different GUIDs | Sensitivity to correct recording pairing |
| Trained capacity control | Train with source values withheld from the start | Whether added target-conditioned capacity explains the gain |
| External target-only reference | Independently train and freeze a baseline | Whether the internal base became weak |

The constant replacement is an offline diagnostic: the implementation averages over the **whole supplied source segment**, including times after a particular anchor. It is not a causal online imputation law or a source sample from a fitted conditional distribution.

Permutation preserves within-source time order and rejects same-recording pairings. If a batch cannot be deranged across GUIDs, it records a skip. The implementation does not additionally enforce all acquisition/clinical compatibility strata proposed in the design.

Conditional source replacement from $p(U_{\mathcal B}\mid\mathcal H,U_{-\mathcal B})$ and exclusion refits are discussed as stronger scientific comparisons in the design. They are not implemented by these zero/constant/permutation controls.

### 17.3 Availability and cancellation

For a scored support $C_{b,a}$, exposure includes

$$
N_\ell^{\rm anchors}=\sum_{b,a}C_{b,a}v_{b,a,\ell},
\qquad
N_\ell^{\rm channels}=\sum_{b,a,j}C_{b,a}m_{b,a,\ell,j}.
$$

The second count reveals whether a nominally exposed lag carries nearly all source channels or just a few. Unsupported bands are reported as missing measurements, not measured zero effects.

For mean or scale proposals, cancellation is summarized by

$$
\kappa_t=\frac{\left\|\sum_\ell r_{t,\ell}\right\|_2}
{\sum_\ell\|r_{t,\ell}\|_2+10^{-8}}.
$$

A small ratio may mean large opposing proposals or simply tiny proposals. The diagnostic retains numerator and denominator to distinguish these cases. It is computed before the final limiter.

### 17.4 Why local proposals are not uniquely identified effects

On a fixed all-valid support, suppose functions $k_\ell(h)$ sum to zero. Replacing

$$
r_{t,\ell}\longmapsto r_{t,\ell}+k_\ell(h_t),
\qquad\sum_\ell k_\ell(h_t)=0,
$$

leaves the summed update, full Gaussian, KL, and ordinary predictions unchanged. Removing one of these altered proposals can nevertheless change its suppression margin.

Thus locality specifies which stored source time a proposal may read. It does not establish a unique decomposition of the predictive function over lags. Source autocorrelation creates further redundancy between candidate times.

### 17.5 Stored lag versus physical delay

For source position $t-\ell$ and target position $t+h$, two exact stored-grid separations are

$$
L_{\rm anchor}=\Delta\ell,\qquad
L_{\rm endpoint}=\Delta(\ell+h).
$$

For nominal source and target feature-content delays $\delta_j^U,\delta_c^Y$, an approximate content separation is

$$
L_{j,c}^{\rm approx}(\ell,h)=\Delta(\ell+h)+\delta_j^U-\delta_c^Y.
$$

It depends on the channel pair and horizon offset. Filters, nonlinear envelopes, phase operations, and autocorrelation distribute information over time. The build-time UA shift is already embedded in the stored source convention. There is no universal scalar lag correction that recovers a physiological delay from every proposal.

For a simple stored-feature process with planted source delay $d_0$,

$$
Y_s=f(Y_{<s},U_{s-d_0})+\eta_s,
$$

the source directly used by label $Y_{t+h}$ corresponds to lag $\ell=d_0-h$. Its pooled direct support is

$$
[d_0-H,d_0-1]\cap[0,L-1].
$$

For example, $d_0=20$ with $H=10$ gives direct lags $10$–$19$, not lag 20. Autoregression and correlated source history may broaden the predictive profile beyond that direct structural support.

## 18. What the information-theoretic quantities mean

### 18.1 The latent KL is an upper bound with a mismatch term

Fix trained parameters and disable dropout. Define the aggregate full encoder under the data's conditional source distribution:

$$
Q_{\rm agg}(z\mid\mathcal H)
=\int Q(z\mid\mathcal H,u)p_{\rm data}(u\mid\mathcal H)\,du.
$$

Insert $Q_{\rm agg}$ into the logarithm defining KL and take expectations:

$$
\begin{aligned}
\mathbb E_{\mathcal H,U}\operatorname{KL}(Q\Vert P)
&=\mathbb E\log\frac{Q(Z\mid\mathcal H,U)}{Q_{\rm agg}(Z\mid\mathcal H)}
+\mathbb E\log\frac{Q_{\rm agg}(Z\mid\mathcal H)}{P(Z\mid\mathcal H)}\\
&=I_Q(Z;U\mid\mathcal H)
+\mathbb E_{\mathcal H}\operatorname{KL}(Q_{\rm agg}\Vert P).
\end{aligned}
$$

The second term is nonnegative, so expected source KL upper-bounds source information encoded in $Z$ conditional on target history. It can also pay for aggregate-prior mismatch or source details irrelevant to the target. The aggregate is generally a mixture, even though every individual encoder Gaussian is diagonal.

Because the encoder does not observe future labels and uses independent sampling noise, conditional data processing gives, under this declared information set,

$$
I(V;Z\mid\mathcal H)
\le\min\{I(V;U\mid\mathcal H),I_Q(Z;U\mid\mathcal H)\}
\le\mathbb E K.
$$

These statements concern the fixed evaluation model and a specified data population. Marginalizing dropout would create additional mixtures; average conditional Gaussian KL is not generally their marginal KL.

### 18.2 Predictive gain includes approximation error

Let $p_0^\star(V\mid\mathcal H)$ and $p_1^\star(V\mid\mathcal H,U)$ be true conditional densities on a fixed scored population. Let $\widehat p_0,\widehat p_1$ be normalized model predictive densities with latent uncertainty integrated out. Define their expected approximation errors $\mathcal E_0$ and $\mathcal E_1$ by KL from the corresponding true conditionals.

Adding and subtracting the true-density log ratio gives

$$
\boxed{
\mathbb E\log\frac{\widehat p_1(V\mid\mathcal H,U)}{\widehat p_0(V\mid\mathcal H)}
=I(V;U\mid\mathcal H)+\mathcal E_0-\mathcal E_1.
}
$$

A positive model gain may therefore reflect improved approximation by the larger source-enabled computation even when the source contains no additional information. The capacity-control arm tests exactly this concern. Conversely, poor optimization or misspecification can hide genuine conditional predictive information.

Finite-$K$ estimation adds another layer of error. Weighted conditional training scores do not satisfy this normalized predictive-density identity as written.

### 18.3 Conditional information, synergy, and causality

Source evidence need not be independent of FHR to be useful. For independent fair bits $H,U$ and $V=H\mathbin{\mathrm{XOR}}U$, neither input alone predicts $V$, but

$$
I(V;U\mid H)=\log2.
$$

Target-conditioned proposal heads can retain this kind of synergy. There is no justification here for forcing their corrections to be orthogonal to the target representation.

For $H>1$, the scientific quantity concerns a future **block**, not automatically one-step transfer entropy conditioned on the immediately preceding observed target history at every future step. No latent coordinate is defined to equal TE. Predictive relevance can also arise from a common driver or reverse-direction dependence; the architecture does not identify an unconfounded biological intervention effect.

When target coordinates are missing, identities must be stated for the selected population and fixed observed-coordinate pattern, or conditional on that pattern. Pooling scores under variable masks does not automatically recover the complete-block information quantity.

## 19. Configuration arms and alternatives

### 19.1 The committed configuration family

The configuration loader recursively deep-merges `base:` files. `default.yaml` is standalone, avoiding accidental inheritance of incompatible source-attention settings from older packages.

| File | Base | Defining change |
|---|---|---|
| `default.yaml` | None | Full mean/scale local-residual candidate from fresh construction |
| `target_only.yaml` | `default.yaml` | Removes the source pathway |
| `joint.yaml` | `default.yaml` | Target-only warm start; disables fresh head calibration |
| `mean_only.yaml` | `joint.yaml` | Omits the scale-update output parameters |
| `capacity_control.yaml` | `joint.yaml` | Keeps proposal capacity but withholds source values |
| `pointwise_attention.yaml` | `joint.yaml` | Replaces local fusion with lag attention |
| `attention_reference.yaml` | `pointwise_attention.yaml` | Replaces pointwise source representation with a causal convolution stem |
| `tiny.yaml` | `default.yaml` | Fixture-scale network, one epoch, small batches |

The seed values also differ across the supplied arm files. These files define experimental arms; an empirical mechanism comparison still needs matched training budgets, multiple seeds, data support, and appropriate uncertainty.

### 19.2 Mean-only residual

With `mean_only_residual: true`, the proposal output has width $64$ instead of $128$. No scale proposal parameters are built:

$$
b=0,\quad\lambda^q=\lambda^p,\quad
K=\frac12\|a\|^2,\quad z^q-z^p=\sigma^p\odot a.
$$

Observation variance can still change because the decoder variance head reads a changed latent. “Mean-only” describes the encoder's latent update, not a requirement that the final forecast variance remain unchanged.

### 19.3 Capacity control

With `source_values_withheld: true`, the pointwise encoder emits $[0,m]$ rather than $[x^{\rm safe},m]$. The proposal head, lag embeddings, dimensions, and parameter budget remain those of the candidate. It trains from the beginning without source values.

This differs from replacing values during evaluation of a model trained with actual source data. The trained arm asks whether additional FHR-conditioned capacity and availability/lag metadata can account for an apparent gain. Scalar lift plus value withholding is refused.

### 19.4 Attention fusion comparator

Let $e_{t,\ell}$ be the flattened source representation. The comparator normalizes the target state and source vector separately, then forms, for each of four heads,

$$
q_t^m=W_Q^m\operatorname{LN}(h_t)+b_Q^m,
\quad k_{t,\ell}^m=W_K^m\operatorname{LN}(e_{t,\ell})+b_K^m,
\quad v_{t,\ell}^m=W_V^m\operatorname{LN}(e_{t,\ell})+b_V^m.
$$

A learned vector $b_\ell^m$ is added to each lag's key in the score:

$$
\alpha_{t,\ell}^m
=\operatorname{softmax}_{\ell\in\mathcal A_t}
\left(\frac{\langle q_t^m,k_{t,\ell}^m+b_\ell^m\rangle}{\sqrt{d_m/4}}\right).
$$

The attended head summaries are concatenated, passed through an output projection, then a zero-initialized linear projection into the mean/scale residual outputs. The downstream bounds, prior-relative Gaussian equations, paired sampling, decoder, and objective stay the same. The lag-scale multiplier is one on this arm.

If no lag is admissible, the update is forced to zero. The full latent is still one unpartitioned $64$-dimensional space. Attention heads partition the attention calculation, not latent dimensions into lag-specific blocks.

Removing a lag changes the softmax denominator, so surviving weights renormalize. Local-proposal suppression leaves other proposals unchanged. Each margin must be interpreted against its own fitted fusion mechanism.

Lag chunking is refused for attention fusion; splitting and independently normalizing its lag axis would change the model. Anchor chunking is allowed. The public model/evaluation contract does not export attention weights as physiological lag attribution.

### 19.5 Convolution source comparator

The `conv` source stem first uses the pointwise availability checker, then a source availability adapter and the same two causal gated convolution blocks used by the target stem. It emits width-$128$ source states, with 21 stored samples of additional neural receptive field. It has no source self-attention stack.

The convolution blocks themselves use dropout zero. In the current construction, the supplied source adapter receives the model's `dropout` value, which is $0.1$ in production. Thus statements in comments that every source comparator is entirely dropout-free are too broad. Offline evaluation disables this dropout; the recommended pointwise/local pathway has no such stochastic source adapter.

### 19.6 Incompatible and unimplemented options

Legacy keys such as `num_heads`, `d_head`, `source_attention_blocks`, `query_uses_logvar`, `posterior_logvar_mode`, `delta_mu_scale`, `delta_logvar_scale`, `base_decode`, and `prior_availability_input` are explicitly refused when non-null. Their old semantics do not describe this model. In particular, `residual_mu_scale` is in prior-standard-deviation units, unlike an old raw-latent-unit delta.

The design discusses Student-$t$ observations, conditional replacements, exclusion refits, alternative losses, and other future experiments. The present objective accepts only Gaussian NLL or MSE. Those discussions should not be read as available production switches.

## 20. Outputs, diagnostics, and computational cost

### 20.1 Forward output contract

| Key | Shape | Meaning |
|---|---|---|
| `anchor_index`, `anchor_valid` | $(B,A)$ | Stored positions and genuine/padded anchor flags |
| `target_state` | $(B,T,128)$ | Full target-encoder stream |
| `conditioning_state` | $(B,A,128)$ | Clock-augmented state at anchors |
| `mu_prior`, `logvar_prior`, `raw_logvar_prior` | $(B,A,64)$ | Base Gaussian parameters and raw variance-head output |
| `mu_post`, `logvar_post` | $(B,A,64)$ | Full Gaussian parameters |
| `z_prior`, `z_post` | $(B,A,64)$ | Paired latent samples |
| `raw_update_mean`, `update_mean` | $(B,A,64)$ | Summed/scaled raw update and bounded $a$ |
| `raw_update_logsigma`, `update_logsigma` | $(B,A,64)$ | Raw/bounded $b$; absent when no scale update exists |
| `mu_base`, `logvar_base`, `mu_full`, `logvar_full` | $(B,A,10,76)$ | Conditional forecast means and observation log-variances |
| `kld_per_anchor_dim` | $(B,A,64)$ | Raw KL per coordinate |
| `kld_per_anchor` | $(B,A)$ | Coordinate-summed raw KL |
| `lag_valid` | $(B,A,91)$ | At least one available channel per lag |
| `persistence` | $(B,A,76)$ | Optional shared target persistence input |
| `mean_proposals`, `scale_proposals` | $(B,A,91,64)$ each | Local contributions, only when requested and applicable |
| `source_channel_mask` | $(B,A,91,46)$ | Gathered channel availability, only when requested and applicable |

Additional outputs include scalar saturation fractions and per-anchor cancellation quantities. The mean-only and source-disabled arms omit quantities that do not exist rather than creating a fake scale pathway. Attention fusion omits local-proposal cancellation quantities.

Mapping dense outputs back to time requires `anchor_index`. For example, entry zero on the dense latent axis describes time 134, not time zero. A padded training entry is not another time observation.

### 20.2 Reading loss metrics

`nll_full_block` and `nll_base_block` in the **training task** are weighted conditional scores per contributing anchor. `pred_gap` is their difference. `nll_*_sample` divides by the nominal $HC_Y$ coefficient count, not by each anchor's actual valid-coordinate count. `scored_coefficients` similarly reports nominal block size.

`source_conditioned_kl_raw` is the mean raw latent KL on scored anchors. `source_conditioned_kl_train` includes any configured free-bits floor. `prior_rate` is the variance-only prior regularizer. Saturation and scale diagnostics reveal whether bounds are frequently active.

The forward's mean/update saturation indicators use $99\%$ of the corresponding absolute bound. Log-variance floor/ceiling diagnostics use a margin of $5\%$ of the configured observation/prior interval width. A latent coordinate is counted active when its scored-anchor mean KL exceeds $10^{-2}$ nats. These are diagnostic thresholds, not physiological categories.

Task `mu_post_prior_gap_rms` is a root mean **squared vector norm** over anchors:

$$
\sqrt{\frac{\sum_{b,a}C_{b,a}\|\mu^q_{b,a}-\mu^p_{b,a}\|^2}{\sum_{b,a}C_{b,a}}}.
$$

The objective's `delta_mu_rms` averages squared differences over both anchors and coordinates before taking the root. On identical support they differ by a factor $\sqrt{d_z}$.

### 20.3 Diagnostic pages are another estimator

`plotting.py` and `sample_page.py` build model-specific pages with raw context, feature forecasts, input panels, latent dimensions, total/per-coordinate KL, proposal magnitude, exposure, single-lag suppression, and cancellation.

The pages use a sampled forward and matched diagnostic calculations. Their weighted forecast curves illustrate the training score, not the offline $K$-draw unweighted predictive mixture estimator. Their displayed conditional uncertainty bands should not be mistaken for the mixture calibration analysis.

Specifically, the page's single-lag suppression heatmap is $K_t-K_t^{\setminus\ell}$: the change in **latent divergence** after removing that lag's proposals. It does not decode and rescore a predictive likelihood for each pixel. The offline band margin $J_{\mathcal B}$ instead measures a change in predictive NLL. These two suppression readouts have different definitions and signs and must not be interchanged.

Plots keep model quantities at stored anchor positions. The raw row is contextual; there is no reconstructed raw FHR output hidden in the feature forecast. Overlapping forecasts may be tiled or stitched for display, while scoring still concerns the explicit anchor–horizon blocks.

The rank-zero plotting callback avoids calling this model's collective-reducing loss; otherwise a single-rank figure could wait for collectives that other ranks never enter. The current configuration enables the dedicated plotting callback, despite older comments saying the per-epoch page was disabled.

### 20.4 Parameters and memory

A construction using the production network dimensions and the fixture's verified production channel plan has **3,941,316** unique parameters: **63,064** in `source_encoder` plus `proposal_head`, and **3,878,252** in the remaining target/clock/decoder pathway. The identity source encoder contributes zero of those source parameters.

The local proposal count can be checked directly:

$$
91\cdot8+(228\cdot128+128)
+(128\cdot128+128)+(128\cdot128+128)=63064.
$$

The main local source computation scales with $BAL$ proposal evaluations. A single FP32 proposal array at $(B,A,L,d_z)=(128,32,91,64)$ occupies

$$
128\cdot32\cdot91\cdot64\cdot4=95{,}420{,}416\text{ bytes}=91\text{ MiB}.
$$

Mean and scale arrays together occupy 182 MiB before accounting for hidden activations, gradients, optimizer state, target attention, and the decoder. Dense decoding increases $A$ from 32 to 156.

`anchor_chunk` and `lag_chunk` evaluate smaller local proposal windows while retaining gradient connectivity. They can change floating-point summation order. They are useful for inference memory, but accumulated autograd graphs still retain training activations: chunking is not a guarantee of reduced peak training memory. The repository's recorded production-geometry measurements found batch-size reduction substantially more effective for training. Those historical GPU measurements were not rerun for this document.

## 21. A worked example

Take one segment and training phase $\varphi=0$.

1. The loader turns four coefficient arrays into $(300,36)$ FHR scattering, $(300,44)$ FHR phase, $(300,36)$ UA scattering, and $(300,10)$ UA phase.
2. The task assembles target width 80 and source width 46. The target gate keeps 76 channels. Raw traces and clinical labels do not enter these encoders.
3. The training anchors are $134,139,\ldots,289$: 32 positions. Anchor $134$ corresponds to $536$ seconds after the trimmed segment start, or $596$ seconds after the original untrimmed segment start.
4. Its labels are kept FHR features at steps $135,\ldots,144$, corresponding to $4,8,\ldots,40$ seconds after the anchor on the stored clock.
5. Its source lag 90 reads step $44$. A channel with $W'_j=41$ is available there; one with $W'_j=51$ is not. A source channel with $W'_j=278$ is unavailable at every lag of this anchor.
6. The target adapter and causal encoder produce a width-128 state. Adding the metadata-clock projection gives $h_{134}$. The prior heads produce 64 means and 64 log-variances.
7. Each of 91 lag positions supplies a width-92 source vector and an eight-dimensional lag embedding. Together with $h_{134}$, these form the proposal MLP's width-228 input. Entirely unavailable lags contribute zero.
8. The mean/scale proposal sums are multiplied by $1/\sqrt{91}$ and bounded, yielding $a,b\in\mathbb R^{64}$.
9. Suppose one latent coordinate has $\mu^p=0.4$, $\lambda^p=0$, $a=0.5$, and $b=\log(1.2)$. Then

$$
\sigma^p=1,\quad\mu^q=0.9,\quad\sigma^q=1.2,\quad
\lambda^q=2\log(1.2),
$$

and this coordinate's divergence is

$$
K_d=\tfrac12\left[0.5^2+1.2^2-1-2\log(1.2)\right]
\approx0.162678\text{ nats}.
$$

For shared $\epsilon=-0.3$, its two sampled coordinates are $z^p=0.1$ and $z^q=0.54$. Their difference is $0.44$, matching $0.5+(1.2-1)(-0.3)$.

10. Each whole 64-dimensional code generates ten width-256 horizon tokens, which pass through the shared convolution/FiLM/attention decoder. Four output arrays have shape $(10,76)$ at this anchor: base/full mean and base/full observation log-variance.
11. If the anchor and all ten future steps have valid weights, all 760 coordinates are scored. If exactly one future step is invalid, 684 coordinates are scored. If two future steps are invalid, coverage is 0.8 and the entire anchor contributes no reconstruction or rate term.
12. The optimizer updates the shared target pathway and decoder as well as the source proposal head. Offline evaluation later resamples latent codes repeatedly and compares average **likelihoods** on common masks.

## 22. Implementation qualifications and evidence

### 22.1 What follows structurally

For the recommended pointwise/local arm, the implementation enforces:

- The prior has no path from actual source values.
- A source encoding reads one coefficient; a proposal reads one stored source time.
- Target convolutions and attention read no stored input after the anchor.
- Future validity and labels do not enter prediction.
- Source-off and all-unavailable updates reproduce the prior.
- Both decoder calls share weights, persistence, and sampling noise.
- UA reaches the forecast through the latent only.
- Raw KL is computed from the actual residual Gaussian parameterization.

These are architecture and indexing facts. They do not establish predictive superiority, physiological lag recovery, or unconfounded causal effect.

### 22.2 How the documentation was checked

The committed integer-causal fixture metadata was resolved through the actual configuration and warm-up functions. It confirmed target width $76$, source width $46$, target warm-up range $0$–$134$, and source range $0$–$278$.

A dense fixture-geometry model construction/forward confirmed 156 anchor positions, output width 76, the anchor-indexed proposal contract, exact initial base/full forecast equality, and zero initial KL. The production-width parameter count above was obtained by constructing the model with the verified channel plan; it does not depend on training a production checkpoint.

An additional end-to-end check loaded an actual fixture sample through `CombinedHDF5Dataset`, including its paired statistics, trimming, normalization, and channel transpose. The resulting input shapes were $(1,300,80)$ and $(1,300,46)$; the forecast shape was $(1,156,10,76)$. Predictions and loss were finite, and all 156 anchors of that sample contributed to scoring. The fixture contains four samples.

The following existing suites were run during preparation: `test_residual_kl.py`, `test_forward_contract.py`, `test_causality.py`, `test_invariants.py`, and `test_objective.py`. **All 70 tests passed.** This was a focused mathematical/structural check, not a full production training or evaluation campaign.

### 22.3 Research validation already supported by the package

The instrument package includes source-free autoregression, redundant source, single/several delays, broad kernels, state-dependent effects, multi-lag interactions, informative zeros, constants, common drivers, and a raw process passed through the actual causal feature operator. Its fit/validation/test splits separate optimization, stopping, and final scoring. Its reduced training setup is not identical to production optimization.

The frozen-probe evaluator fits ridge regressions from six readouts: prior/full means, standard deviations, and paired samples. Targets are complete-coverage future blocks relative to the anchor's stored values, not relative to learned persistence weights. Recordings are deterministically split for fitting and scoring. An out-of-sample $R^2$ measures linearly accessible future dynamics; it does not prove what the nonlinear decoder uses or measure source information exactly.

The acceptance machinery groups runs by arm and training seed, checks score provenance and structural invariants, compares an external target-only reference, reports recording-bootstrap differences, handles declared band multiplicity, and checks recording overlap between development and confirmation sets. The plan asks for at least three training seeds. Repeated scorings of one checkpoint are not independent training seeds.

The roadmap records synthetic and integration work and leaves real-data arm/acceptance runs outstanding. This document does not claim that the revised model has already met the scientific acceptance criteria.

### 22.4 Important differences from older prose

| Older or planned description | Current implementation |
|---|---|
| A stochastic Gaussian slot per lag | One shared latent with deterministic per-lag parameter proposals |
| Separate target and source latent partitions | Full-width target prior and full-width residual update |
| All future information passes through the latent | Target persistence is enabled by default; source influence remains latent-mediated |
| Source KL can be allocated by attention | No exact per-lag KL allocation is emitted |
| Prior penalty is full KL to a standard normal | It regularizes variance only |
| Learning-rate warm-up starts at factor 0.1 | Executed step factor is $\min(1,(k+1)/2000)$ |
| Early stopping on marginalized predictive NLL | Default stops on the weighted validation total loss |
| Dedicated diagnostic pages are disabled | Current default enables the residual-model plotting callback |
| Warm start copies the entire clock | `clock_proj` transfers; separate `clock_norm` is not allowlisted |
| Every source comparator has no dropout | The convolution comparator's adapter inherits configured dropout |
| Zero FiLM makes the decoder core an identity | It makes modulation an identity, not the whole decoder |
| One-sided filters certify the entire raw pipeline | Resampling, interpolation, acquisition shifts, and selection need their own interpretation |

These distinctions are recorded to keep the mathematical explanation faithful to the executed model. No implementation or configuration changes were made as part of this documentation task.

## 23. Source map and reproduction

The explanations above do not require opening these files. This map provides the implementation trace for readers who want to verify or extend a particular stage. Paths are relative to this document.

| Responsibility | Primary implementation |
|---|---|
| Current architecture and full forward | [nets/model.py](nets/model.py), [nets/core.py](nets/core.py) |
| Source representation and lag indexing | [nets/pointwise_source.py](nets/pointwise_source.py) |
| Proposal MLP, bounds, residual parameters, KL | [nets/lag_updates.py](nets/lag_updates.py) |
| Model-specific objective and DDP reduction | [nets/objective.py](nets/objective.py) |
| Source controls and comparator mechanisms | [nets/controls.py](nets/controls.py), [nets/lag_attention.py](nets/lag_attention.py), [nets/conv_source.py](nets/conv_source.py) |
| Task integration and warm-start transfer | [task.py](task.py), [trainer.py](trainer.py) |
| Main configuration and inheritance | [configs/default.yaml](configs/default.yaml), [../lag_attn/config.py](../lag_attn/config.py) |
| Causal channel metadata and constructor kwargs | [../lag_attn_cfs/causal_warmup.py](../lag_attn_cfs/causal_warmup.py), [../lag_attn_cfs/model_kwargs.py](../lag_attn_cfs/model_kwargs.py) |
| Availability, anchors, label clocks | [../lag_attn_cfs/nets/causal_inputs.py](../lag_attn_cfs/nets/causal_inputs.py), [../lag_attn_cfs/nets/causal_feature_target.py](../lag_attn_cfs/nets/causal_feature_target.py) |
| Feature-target gathering | [../lag_attn_fs/nets/feature_target.py](../lag_attn_fs/nets/feature_target.py), [../lag_attn_fs/task.py](../lag_attn_fs/task.py) |
| Tiling phase and task inheritance | [../lag_attn_cfs/task.py](../lag_attn_cfs/task.py), [../lag_attn_rws/task.py](../lag_attn_rws/task.py) |
| Target conv–Transformer and RoPE/RMSNorm/SwiGLU | [../lag_attn_transformer_rws/nets/encoders.py](../lag_attn_transformer_rws/nets/encoders.py), [../lag_attn_transformer_rws/nets/blocks.py](../lag_attn_transformer_rws/nets/blocks.py) |
| Input adapter, residual MLP, bounds | [../lag_attn/nets/encoders.py](../lag_attn/nets/encoders.py), [../lag_attn/nets/blocks.py](../lag_attn/nets/blocks.py) |
| Channel gathering and delays | [../lag_attn/nets/delays.py](../lag_attn/nets/delays.py) |
| Prior heads and geometry | [../lag_attn_rws/nets/heads.py](../lag_attn_rws/nets/heads.py), [../lag_attn_rws/nets/geometry.py](../lag_attn_rws/nets/geometry.py) |
| Mask construction and elementary scoring | [../lag_attn_rws/nets/raw_masks.py](../lag_attn_rws/nets/raw_masks.py), [../lag_attn_rws/nets/losses.py](../lag_attn_rws/nets/losses.py) |
| Shared horizon decoder | [../lag_attn/nets/decoders.py](../lag_attn/nets/decoders.py) |
| Data module and HDF5 loading | [../../train/data_module.py](../../train/data_module.py), [../../hdf5_dataset/hdf5_dataset.py](../../hdf5_dataset/hdf5_dataset.py) |
| Recording preparation and causal feature execution | [../../hdf5_dataset/new_pipeline/create_new_pipeline.py](../../hdf5_dataset/new_pipeline/create_new_pipeline.py), [../../hdf5_dataset/causal_scattering.py](../../hdf5_dataset/causal_scattering.py), [../../hdf5_dataset/causal_scattering_torch.py](../../hdf5_dataset/causal_scattering_torch.py) |
| Raw adaptor and normalization-statistics construction | [../../mimo/EarlyMaestra/early_maestra/adaptor/mimo_adaptor.py](../../mimo/EarlyMaestra/early_maestra/adaptor/mimo_adaptor.py), [../../hdf5_dataset/calculate_dataset_stats.py](../../hdf5_dataset/calculate_dataset_stats.py) |
| Optimizer, scheduler, and checkpoint utilities | [../../train/pl_model_base.py](../../train/pl_model_base.py), [../lag_attn_transformer_rws/task.py](../lag_attn_transformer_rws/task.py), [../../train/graph_models_utils.py](../../train/graph_models_utils.py) |
| Shared trainer/data/checkpoint orchestration | [../lag_attn_cfs/trainer.py](../lag_attn_cfs/trainer.py), [../lag_attn_rws/trainer.py](../lag_attn_rws/trainer.py), [../../train/graph_model_base.py](../../train/graph_model_base.py) |
| Offline model binding and score loop | [eval/binding.py](eval/binding.py), [eval/run.py](eval/run.py), [eval/predictive.py](eval/predictive.py) |
| Lag summaries and verification | [eval/lag_metrics.py](eval/lag_metrics.py), [eval/verify.py](eval/verify.py) |
| Probes, multi-run acceptance, memory measurement | [eval/latent_probes.py](eval/latent_probes.py), [eval/acceptance.py](eval/acceptance.py), [eval/memory.py](eval/memory.py) |
| Synthetic processes and detection rules | [instruments/generators.py](instruments/generators.py), [instruments/raw_process.py](instruments/raw_process.py), [instruments/criteria.py](instruments/criteria.py), [instruments/campaign.py](instruments/campaign.py) |
| Diagnostic pages | [sample_page.py](sample_page.py), [plotting.py](plotting.py) |
| Design rationale, implementation history, evaluation contract | [DESIGN.md](DESIGN.md), [SPEC_AND_SPRINTS.md](SPEC_AND_SPRINTS.md), [eval/EVAL.md](eval/EVAL.md) |
| Superseded architecture, for historical distinction only | [DESIGN_V1_SUPERSEDED.md](DESIGN_V1_SUPERSEDED.md) |

From the repository root, the existing entry points include:

```powershell
# Fixture-scale integration training; this is not held-out research evidence.
.venv/Scripts/python.exe -m teb_vae.lag_slot_transformer_cfs.trainer --config teb_vae/lag_slot_transformer_cfs/configs/tiny.yaml

# Scoring a real checkpoint requires a valid resolved training config and evaluation paths.
.venv/Scripts/python.exe -m teb_vae.lag_slot_transformer_cfs.eval.run --checkpoint <checkpoint.ckpt> --overrides <evaluation_delta.yaml>

# Read back the produced summary's structural checks.
.venv/Scripts/python.exe -m teb_vae.lag_slot_transformer_cfs.eval.verify --summary <summary.json>
```

The essential computation is a target-only Gaussian, a sum of source-lag proposals that changes its parameters, paired samples, and one shared future-feature decoder. Understanding the availability masks and clocks determines what evidence each proposal actually contains; understanding the score definitions determines what can be concluded from the resulting forecasts.
