# Why the UP → FHR Transfer Entropy Reads Null: An Objective-, Estimator- and Representation-Level Analysis

**Status.** This revision supersedes the previous version of this document, whose central claim — that the scattering / phase-harmonic representation is *blind* to the contraction → deceleration coupling — is **not supported** by direct measurement of the transform and is contradicted by a known-answer simulation reported here. The coupling is retained by the representation. Three other mechanisms, two of them decisive and both fixable, account for the null. Section 12 lists every claim that changed and why.

---

## Abstract

A causal sequential variational model (`SeqVaeLagAttn`) measures the transfer entropy (TE) from a uterine-pressure (UP) source $U$ to a fetal-heart-rate (FHR) target $Y$ as the per-step Kullback–Leibler divergence $K_t$ between a source-conditioned posterior and a target-only prior. On real recordings it reports a near-null coupling, and a pooled feature-space predictive-uplift probe agrees. A strictly-index-causal, event-locked analysis of the **raw** $4$ Hz signals nevertheless finds a strong time-locked FHR deceleration after contractions ($D^\star = -4.43$ bpm, $z = -43.8$ against a count-matched random-trigger null). We resolve the contradiction, and find that the representation is the **least** important of the causes.

Three results, each measured rather than asserted:

1. **The training objective cannot report a calibrated TE.** With the shipped configuration the KL is charged per *latent dimension per step* while the forecast is scored per *predicted element*, so the exchange rate between latent rate and predictive gain is fixed by geometry. Opening the source channel pays only if $\beta < \lambda_{\mathrm{full}}\, d_z/(H_d c_y) = 24/3270 = 7.34\times 10^{-3}$. The shipped $\beta_{\mathrm{end}} = 0.1$ is $13.6\times$ that threshold and crosses it at epoch $\approx 4$ of $50$. Above the free-bits floor no amount of transfer entropy in the data can pay for the rate; below it the KL term is exactly flat. The reported $K_t$ is therefore hard-capped at $d_z \cdot \mathrm{free\_bits} = 2.4$ nats/step and is otherwise unregularised — a hyperparameter-determined quantity, not an estimate.

2. **The pooled uplift probe's null is an artefact of design width.** In a known-answer simulation with a coupling calibrated to the measured $-4.4$ bpm, the *source-specific* uplift is constant at $2.1\%$ across every design, while the *measured* uplift falls from $+2.14\%$ to $+0.13\%$ as nuisance source columns are added from $8$ to $808$, tracking a variance floor $\approx p_{\mathrm{extra}}/N_{\mathrm{fit}}$ to within $0.1$ percentage points. At the probe's own width ($p_{\mathrm{extra}} = 580$–$790$, $N_{\mathrm{fit}} \approx 33{,}600$) the floor is $1.7$–$2.4\%$, which **exceeds the probe's declared $1\%$ detection threshold**. The probe cannot resolve a coupling at the size the raw-signal probe measured, and its reported negatives ($-2.3\%$, $-7.5\%$, $-3.9\%$, $-14.7\%$) match that floor in sign, magnitude and ordering.

3. **The representation retains the coupling; it is not causal.** Measured on the production filter bank ($J{=}11$, $Q{=}4$, $T{=}16$, $f_s{=}4$ Hz): the low-pass $\phi$ retains **96.9 %** of a realistic deceleration's depth with **zero** timing shift, and passes the deceleration band's envelopes at gain $\ge 0.988$ with no aliasing on the 4 s grid. Consistent with the phase-retrieval literature, the modulus is not what destroys information. What *is* wrong is causality: the wavelets are two-sided, so **$28.4\%$ of the target block and $50\%$ of the source block read raw signal beyond the $120$ s forecast horizon**, and a deceleration is already $81.5\%$ visible in the $0.0082$ Hz channel $120$ s *before* its nadir. This corrupts the estimand — the "target-only prior" conditions on the target's future — rather than merely weakening it.

The coupling is real, the representation admits it, and the measurement apparatus does not currently deliver a calibrated number. Section 11 gives the minimal changes.

---

## 1. The question, and three candidate answers

Uterine contractions transiently reduce placental perfusion; the fetus responds with a heart-rate deceleration. The estimand is the directional information flow $U \to Y$ beyond what $Y$'s own past predicts — Schreiber's transfer entropy [1].

`SeqVaeLagAttn` forms, at each decimated step $t$ ($4$ s per step), a target-only prior $p(z_t \mid Y_{\le t})$ and a source-conditioned posterior $q(z_t \mid Y_{\le t}, U_{\le t})$, and reports

$$K_t \;=\; \mathrm{KL}\!\big(q(z_t \mid Y_{\le t}, U_{\le t}) \,\big\|\, p(z_t \mid Y_{\le t})\big)$$

as a TE surrogate, alongside a $120$ s forecast split into a target-only baseline and a source-driven correction. It consumes **not** the raw signals but their scattering and phase-harmonic features.

The model reports that the source contributes essentially nothing. Four explanations are available, and they are not exclusive:

* **(H0) No coupling.** Rejected in §3.
* **(H1) Representation.** The features do not carry the coupling. Rejected in §6–7.
* **(H2) Estimator.** The measurement is too insensitive. Established in §8.
* **(H3) Objective.** The training objective's optimum is a near-null $K_t$ whatever the data. Established in §5, and decisive for the model.

The previous version of this document argued H1 and treated H2 as a secondary confound. That ordering is wrong. The evidence below puts H3 first, H2 second, and finds H1 to be a **validity** problem (the features are non-causal) rather than an information-loss problem.

---

## 2. Transfer entropy and what $K_t$ actually bounds

For stationary processes with histories $Y_t^{(k)}$, $U_t^{(\ell)}$, transfer entropy is the conditional mutual information

$$T_{U \to Y} \;=\; I\big(Y_{t+1}\,;\,U_t^{(\ell)} \,\big|\, Y_t^{(k)}\big).
\tag{1}$$

It vanishes when $Y$'s future is determined by its own past and is positive only when $U$'s history reduces uncertainty about $Y_{t+1}$ beyond $Y$'s own. For jointly Gaussian processes it coincides with Granger causality [2] — the bridge that licenses importing the filtering-and-causality literature (§7.3) into a TE-framed argument, and which holds *exactly* only under that Gaussianity condition.

**$K_t$ is a rate, and an upper bound — not a lower bound on TE.** For any conditional $q(z\mid y,u)$ and any $y$-measurable $p(z\mid y)$, the standard variational-marginal identity gives

$$\mathbb{E}_{U\mid Y=y}\big[\mathrm{KL}\big(q(z\mid y,U)\,\|\,p(z\mid y)\big)\big]
\;=\; I\big(z; U \mid Y = y\big) \;+\; \mathrm{KL}\big(\bar q(z\mid y)\,\|\,p(z\mid y)\big),
\tag{2}$$

with $\bar q(z\mid y) = \mathbb{E}_{U\mid y}[q(z\mid y,U)]$. Both terms are non-negative, so

$$\mathbb{E}[K_t] \;\ge\; I\big(z_t; U_{\le t} \mid Y_{\le t}\big),
\tag{3}$$

with equality exactly when the prior equals the posterior's source-marginal. So $K_t$ measures the **rate** the latent spends on the source, and *over*-states the source information the latent carries whenever the prior is a poor fit to $\bar q$. It is a TE surrogate only in combination with the decoder: rate is worthless unless the forecast converts it. Section 5 shows the objective is configured so that it cannot.

**The predictive-gain identity.** For any baseline model $p_{\mathrm{base}}(y^+\mid y)$ and full model $p_{\mathrm{full}}(y^+\mid y,u)$,

$$\underbrace{\mathbb{E}\Big[\log \tfrac{p_{\mathrm{full}}}{p_{\mathrm{base}}}\Big]}_{\text{measured gain}}
=\; T_{U\to Y} \;-\; \underbrace{\mathbb{E}\big[\mathrm{KL}(p^\ast(\cdot\mid y,u)\,\|\,p_{\mathrm{full}})\big]}_{\text{full-model misfit}}
\;+\; \underbrace{\mathbb{E}\big[\mathrm{KL}(p^\ast(\cdot\mid y)\,\|\,p_{\mathrm{base}})\big]}_{\text{baseline misfit}}.
\tag{4}$$

A **negative** measured gain therefore does not imply $T_{U\to Y}=0$; it implies the full model's misfit exceeds the true TE plus the baseline's misfit. Section 8 shows that in the pooled probe this excess misfit is exactly the added-column estimation variance, and quantifies it.

---

## 3. The coupling is real (H0 rejected), with three corrections to how it was reported

`stage2_contraction_deceleration_probe.py` detects contractions on the raw $4$ Hz UP trace and forms the baseline-corrected contraction-triggered average (CTA)

$$\bar d(\tau) \;=\; \frac{1}{N}\sum_{i=1}^{N}\Big(Y(c_i+\tau) - \mu_i^{-}\Big),
\qquad \mu_i^{-}=\tfrac{1}{|W^-|}\!\!\sum_{\tau'\in W^-}\!\!Y(c_i+\tau'),
\tag{5}$$

with $W^- = [-30,0)$ s, and takes $D^\star = \min_{\tau\in[10,150]\,\mathrm{s}} \bar d(\tau)$ against a **count-matched, per-recording** random-trigger null over $300$ draws. Measured:

$$D^\star = -4.428\ \text{bpm at}\ \tau = 29\ \text{s},\qquad
\bar D^\star_{\text{null}} = -0.167 \pm 0.097\ \text{bpm},\qquad z = -43.8 .$$

**The result stands: FHR is time-locked to uterine activity far beyond chance.** Three statements in the previous version were wrong or unsupported, and matter for interpretation.

**3.1 The min-over-window selection bias is absorbed — say so, it is a strength.** Taking a minimum over $560$ samples of a noisy mean curve is biased downward. Because the null applies the *identical* operator to the *same* traces with the *same* per-recording event counts, that bias is measured directly: it is $\bar D^\star_{\text{null}} = -0.167$ bpm, i.e. $3.8\%$ of the effect, and $z$ is correctly centred on it.

**3.2 $p \approx 0$ is not supported; the lag is not a physiological latency.** With $n_{\text{shuffle}} = 300$ the empirical p-value is resolution-limited at $p < 1/300 \approx 3\times10^{-3}$; the printed $0.0000$ means *no null draw was as extreme*, and $z$ is a Gaussian extrapolation beyond the sampled tail. Separately, $\tau^\star$ is **not** a contraction-onset-to-nadir delay, for two compounding reasons. First, the detector's documented rising-edge walk-back (`while grad[idx] >= grad_thresh`, entered at the peak where $\mathrm{grad}\approx 0$) almost always terminates immediately, so the trigger is in practice the **contraction peak**. Second, the pipeline stores UP advanced by $20$ s (`up_shift_secs=-20`), so a stored trigger at index $c$ is true time $c+20$ s while FHR is unshifted. The measured dip at stored $\tau = 29$ s therefore sits $\approx 9$ s **after the true contraction peak** — a timing more compatible with an early (head-compression) deceleration, or with a shared mechanical artefact, than with the $20$–$120$ s hypoxic late-deceleration mechanism the previous version invoked. Report $\tau^\star$ as a detected-index quantity with an explicit convention caveat.

**3.3 "1,200 recordings" is "1,200 segments", and the pooling is by event.** Each HDF5 row is a $22$-minute segment; a GUID contributes up to $\sim37$ of them, and the $1{,}200$ rows are the *head* of `test_dataset_cs.hdf5` (the healthy-with-blood-gas, caesarean split of a $45{,}196$-segment shard, i.e. $2.7\%$, selected by prefix rather than at random). Pooling is over all contractions with no GUID-level clustering or aggregation, so a few contraction-rich segments can dominate $D^\star$. The null preserves within-recording structure, so the test of "no time-locking" is valid; the **population effect size is not established**, and $z=-43.8$ should not be read as a population-level $z$.

Two further caveats belong in any use of this result. The statistic is $Y(t+\tau) - \overline{Y}[t-30,t)$, so a **baseline elevation** at detected contraction times — accelerations accompanying contractions are routine — produces a negative CTA by pure mean reversion; the random-trigger null cannot cancel this, because random triggers carry no such elevation by construction. The CTA waveform is never saved, so this cannot be checked post hoc; it is the single most important unaddressed alternative explanation. And the companion coincidence statistic, omitted from the previous version, is nearly saturated: $71.6\%$ observed vs $65.5\%$ null, **lift $1.09\times$**, on a null that is drawn without the edge restriction the real events obey and is therefore biased low. It is not independent confirmation.

---

## 4. What the model consumes, measured

All features come from `KymatioPhaseScattering1D` at $J=11$, $Q=4$, $T=16$, `shape` $=5280$, `max_order=1`, reflection padding: $42$ analytic Morlet wavelets $\{\psi_i\}$ plus a low-pass $\phi$, decimated by $16$ (one step $=4$ s), trimmed to $T=300$ steps ($20$ min).

$$S_0 x(t) = (x \star \phi)(t), \qquad S_1 x(t,i) = \big(\,|x \star \psi_i|\,\star \phi\big)(t).
\tag{6}$$

Target: `fhr_st` $\oplus$ `fhr_ph`, $c_y = 43+66 = 109$. Source: `up_st` $\oplus$ `up_ph`, $c_u = 43+15 = 58$. Horizon $H_d = 30$ steps $=120$ s; latent $d_z = 24$.

Every number in §6–7 was measured by rebuilding this exact filter bank (`scattering_filter_factory`, $J_{\mathrm{pad}}$ from `compute_minimum_support_to_pad`). The rebuild is validated: it reproduces $903$ wavelet pairs, and the documented band/harmonic selection rule yields exactly $66$ `fhr_ph` and $15$ `up_ph` channels, matching the pipeline.

| Band | Frequency | Filters | $\sigma_t$ (measured) | $\pm2\sigma_t$ span |
|---|---|---|---|---|
| Beat-to-beat | $0.25$–$1.5$ Hz | $0$–$10$ | $0.7$–$4.1$ s | $3$–$16$ s |
| LF / MF variability | $0.04$–$0.25$ Hz | $11$–$20$ | $4.9$–$23.3$ s | $20$–$93$ s |
| **Decelerations** | $\mathbf{0.008}$–$\mathbf{0.04}$ Hz | $\mathbf{21}$–$\mathbf{30}$ | $27.7$–$131.6$ s | $111$–$526$ s |
| Baseline trend | $< 0.008$ Hz | $31$–$41$ | $156.5$–$568.5$ s | $626$–$2274$ s |

(The previous version's "$25$–$744$ s" for the deceleration band is superseded by the measured $111$–$526$ s.) The bank is constant-$Q$ with $\xi_i/\sigma_i = 9.63$ and $\sigma_t\,\xi = 0.922$ throughout.

---

## 5. Result 1 — the objective cannot report a calibrated TE

This is the decisive mechanism for the model, it is pure arithmetic, and it is independent of the physiology, the data and the representation.

### 5.1 The two loss terms are normalised on incompatible axes

From `compute_loss` and `_kld_loss`:

$$\mathcal{L} = \lambda_{\mathrm{full}} L_{\mathrm{feat}} + \lambda_{\mathrm{base}} L_{\mathrm{base}}
+ \beta L_{\mathrm{KL}} + \lambda_{\mathrm{lag}} L_{\mathrm{smooth}},$$

where — and this is the whole point —

$$L_{\mathrm{feat}} = \frac{1}{A\,H_d\,c_y}\sum_{t\in\mathcal A}\sum_{h,c} \mathrm{nll}_{t,h,c},
\qquad
L_{\mathrm{KL}} = \frac{1}{A\,d_z}\sum_{t\in\mathcal A}\sum_{j} \big[K_{t,j}\big]_{\ge \mathrm{fb}} .$$

$L_{\mathrm{feat}}$ is a mean **per predicted element** over $H_d \times c_y = 30\times109 = 3270$ elements per anchor; $L_{\mathrm{KL}}$ is a mean **per latent dimension per step** over $d_z = 24$. Under the production `kld_support='anchor'` the two supports are the same set $\mathcal{A}$, so the anchor count $A$ cancels and only the per-anchor axis lengths remain.

### 5.2 The critical $\beta$

**Proposition 1.** *Let $z_t$ be the latent and $s_t$ the target-only decoder state. Because $z_t$ is, given $Y_{\le t}$, a function of $U_{\le t}$ and independent noise, the chain $z_t - U_{\le t} - Y^+_t$ is Markov given $Y_{\le t}$, and the maximum achievable reduction in expected **total** forecast NLL at anchor $t$ from conditioning the decoder on $z_t$ is*

$$\Delta\mathrm{NLL}_t \;\le\; I\big(Y^+_t; z_t \mid Y_{\le t}\big) \;\le\; \min\big(T_{U\to Y}(t),\ \mathbb{E}[K_t]\big).$$

*Hence the change in the objective from opening the source channel obeys*

$$\Delta\mathcal{L} \;\ge\; \Big(\beta - \lambda_{\mathrm{full}}\frac{d_z}{H_d\,c_y}\Big)\cdot
\frac{1}{A\,d_z}\sum_{t\in\mathcal A} K_t ,
\tag{7}$$

*so if $\beta > \beta^\star := \lambda_{\mathrm{full}}\, d_z/(H_d c_y)$ then any positive KL strictly increases the loss, whatever the data.*

*Proof.* The gain term is $-\lambda_{\mathrm{full}}\Delta\mathrm{NLL}_t/(A H_d c_y)$ summed over $\mathcal A$, bounded below by $-\lambda_{\mathrm{full}} K_t/(A H_d c_y)$; the cost term is $+\beta K_t/(A d_z)$. Factor $K_t/A$. $\square$

At production values $\lambda_{\mathrm{full}} = 1$, $d_z = 24$, $H_d = 30$, $c_y = 109$:

$$\boxed{\;\beta^\star = \frac{24}{3270} = 7.34\times 10^{-3}, \qquad
\frac{\beta_{\mathrm{end}}}{\beta^\star} = \frac{0.1}{7.34\times10^{-3}} = 13.6\;}$$

The shipped schedule is `linear_warmup` from $10^{-4}$ to $0.1$ over $50$ epochs, so $\beta$ crosses $\beta^\star$ at

$$e^\star = 50\cdot\frac{7.34\times10^{-3} - 10^{-4}}{0.1 - 10^{-4}} = 3.6,$$

i.e. **from epoch 4 of 50 onward, marginal latent rate costs $13.6\times$ more than the maximum predictive gain it could possibly buy.**

### 5.3 What free bits do, precisely

`free_bits` $=0.1$ clamps each per-dimension per-step KL *upward* before masking, so a dimension carrying less than $0.1$ nats contributes a constant and receives **exactly zero KL gradient**. Two consequences follow, and only the second is a hard bound:

* Below the floor the KL term is flat: the reported `kld_raw` is set purely by the reconstruction gradient, i.e. by the (small) predictive benefit of the source. It is *unregularised*, not *suppressed*.
* Above the floor Proposition 1 binds. **The objective therefore caps the reportable TE surrogate at $d_z \cdot \mathrm{free\_bits} = 2.4$ nats/step, independent of the data.**

So `kld_raw` is not a calibrated estimate of anything: its scale is set by $(\beta, \mathrm{free\_bits}, d_z, H_d, c_y)$, and any true coupling above $2.4$ nats/step is truncated. The model documentation's note that `beta_schedule.end` "is a starting value to retune against `kld_raw`" is correct but understates the problem: there is a *critical value*, it is computable in closed form, and the shipped value is on the wrong side of it.

**This alone explains the model's null.** It requires no claim about physiology, representation or estimator sensitivity.

---

## 6. Result 2 — the representation retains the coupling

### 6.1 The low-pass keeps the deceleration essentially intact

$\phi$ is a Gaussian low-pass with measured RMS time width $4.50$ s and $\sigma_f = 0.025$ Hz. Its gain across the physiological range:

| $f$ (Hz) | $0.002$ | $0.005$ | $0.008$ | $0.02$ | $0.04$ | $0.0667$ | $0.125$ |
|---|---|---|---|---|---|---|---|
| $\lvert\hat\phi(f)\rvert/\lvert\hat\phi(0)\rvert$ | $0.997$ | $0.981$ | $0.952$ | $0.726$ | $0.277$ | $0.028$ | $4\times10^{-6}$ |

A realistic deceleration — Gaussian, $20$ bpm deep, $\sigma = 25$ s — has its energy well below $0.0064$ Hz. Passed through the *actual* $\phi$:

$$\text{depth raw } -20.000\ \text{bpm} \;\longrightarrow\;
\text{after } \phi:\ -19.381\ \text{bpm } (\mathbf{96.9\%}\ \text{retained}),\quad
\text{nadir shift } \mathbf{+0.00\ s}.$$

$S_0$ is stored unmasked as `fhr_st` channel $0$ and, per the pipeline's normalisation table, receives **no** log/asinh transform — only standardisation. It is therefore an affine function of the $4$ s locally-averaged FHR in bpm. The same holds for `up_st` channel $0$ and the contraction. **The deceleration and the contraction are both present in the feature vector, at $4$ s resolution, with no timing bias.**

### 6.2 Nor does the decimation or the low-pass damage the band envelopes

The envelope of band $i$ has bandwidth $\approx\sigma_i$. Measured against $\phi$'s response and the decimated-grid Nyquist ($0.125$ Hz):

| Band | Envelope bandwidth | min $\phi$ gain at that bandwidth | Channels aliased by the $\times16$ decimation |
|---|---|---|---|
| Beat-to-beat | $0.0274$–$0.1548$ Hz | $\approx 0$ | $2/11$ |
| LF/MF variability | $0.0048$–$0.0230$ Hz | $0.656$ | $0/10$ |
| **Deceleration** | $0.00086$–$0.00407$ Hz | $\mathbf{0.988}$ | $\mathbf{0/10}$ |
| Baseline trend | $0.0002$–$0.0007$ Hz | $1.000$ | $0/11$ |

In the deceleration band the low-pass passes the envelope at gain $\ge 0.988$ and nothing is aliased. Only the top two of $42$ channels lose envelope structure to the decimation. The generic "invariance / averaging destroys the transient" argument does not apply at the frequencies where this coupling lives.

### 6.3 The modulus is not the culprit — and the literature says so explicitly

The previous version's mechanism 5.1 ("the modulus discards the analytic phase, therefore the information is gone") is wrong as an *information* statement. Mallat & Waldspurger [4] prove that for Cauchy wavelets the modulus of the wavelet transform determines the function **uniquely up to a global phase**; Andén & Mallat [6] state it directly: *"a wavelet modulus operator removes the complex phase, [but] it does not lose information because the temporal variation of the multiscale envelopes is kept."* What those same sources identify as the lossy step is the **averaging**: *"this time averaging removes fine-scale information"*, and Waldspurger [5] gives the sharp form — the scattering transform is invertible *up to a constant time shift of at most $T$*.

What *is* true, and is worth keeping, is a narrower pair of statements:

* **Exact sign blindness above $S_0$.** $S_1(-x) = S_1(x)$ identically; measured on the real bank, $\max_t\big||x\star\psi_i| - |(-x)\star\psi_i|\big| = 0$ exactly. Only $S_0$ distinguishes a deceleration from an equal-amplitude acceleration — and $S_0$ is present.
* **Reconstruction is unstable in the delay direction.** Mallat & Waldspurger's counterexample construction produces signals far apart in $L^2$ whose wavelet moduli are nearly identical, by a phase modulation slowly varying in time and frequency — i.e. precisely a time-shift perturbation. Timing is the fragile coordinate, which is why $S_0$ (which never takes a modulus) matters so much here.

Note also that `max_order=1`: the second-order coefficients that Andén & Mallat show recover the amplitude modulation discarded by $\phi$ are simply not computed. Per §6.2 this costs little in the deceleration band and more in the beat-to-beat band.

### 6.4 The phase-harmonic blocks answer a different question

The stored self- and cross-phase features are $\phi$-averaged, real-part-only correlations

$$C_{i,j,p}(t) = \Big(\phi \star \big([\,x\star\psi_i\,]^{p}\ \overline{(x\star\psi_j)}\big)\Big)(t),
\qquad p = \xi_j/\xi_i ,
\tag{8}$$

and their cross-channel analogue. Three structural facts limit what they can contribute to a *directional, target-past-conditioned* quantity, and none is a defect of implementation:

1. They are **covariances of a stationary process** [8]: time-translation-invariant second-order statistics with no privileged time origin, capturing "dependencies across scales, which specify the geometry of local coherent structures" — within-signal structure, not cross-signal absolute lag.
2. They measure **marginal** association and never subtract $Y_{\le t}$, so they are not transfer entropies. Coherence can be large with zero TE (common drive), and TE positive with negligible coherence.
3. Only $\mathrm{Re}\,C$ is stored; the discarded quadrature component is where the **sign of the lag** between the two bands lives.

The invertibility result of Mallat–Zhang–Rochette [7] is proved for the phase-harmonic *operator*, **not** for the covariances, and the recovery-from-correlations result is numerical and conditional on wavelet sparsity — a condition physiological processes treated as stationary realisations do not satisfy. There is **no** completeness or sufficiency theorem for phase-harmonic covariances in either direction; do not assert one.

---

## 7. Result 3 — the features are not causal, and that corrupts the estimand

### 7.1 The bound the previous version wrote is not the right bound

Because $S_{t+1}=\Phi_Y(Y)_{t+1}$ is a deterministic function of $Y$ over a two-sided window, conditioning on $Y_{\le t}$ leaves it a function of $Y_{>t}$ alone, so $U_{\le t} - Y_{>t} - S_{t+1}$ is Markov given $Y_{\le t}$ and the data-processing inequality gives, with $\Delta$ the filter's forward reach,

$$I\big(S_{t+1};U_{\le t}\mid Y_{\le t}\big) \;\le\; I\big(Y_{t+1:t+\Delta};U_{\le t}\mid Y_{\le t}\big),
\tag{9}$$

the **block** TE over the forward support — not the one-step TE of Eq. (1). Since $\Delta$ reaches $966$ s (§7.2), this bound is very loose and cannot support the previous version's "processing can only reduce TE, therefore the residual is small".

**Worse, the feature-space quantity is not a lower bound at all.** The model conditions on $S_{\le t}$, not $Y_{\le t}$, and coarsening a conditioning set is not monotone. Writing $A=S_{t+1}$, $B=\tilde U_{\le t}$, $C=Y_{\le t}$ and expanding $I(A;B,C\mid S_{\le t})$ two ways,

$$I\big(S_{t+1};\tilde U_{\le t}\mid S_{\le t}\big)
= I\big(S_{t+1};\tilde U_{\le t}\mid Y_{\le t}\big)
+ \underbrace{\Big[I\big(S_{t+1};Y_{\le t}\mid S_{\le t}\big) - I\big(S_{t+1};Y_{\le t}\mid \tilde U_{\le t}, S_{\le t}\big)\Big]}_{\text{conditioning deficit, sign indefinite}} .
\tag{10}$$

The bracket can be positive: an imperfectly observed driver/target state **inflates** measured TE. This is exactly Smirnov's result that unobserved state variables, low temporal resolution and observation error all produce spurious TE as special cases of imperfect state observation [11]. That the measured feature-space TE is null despite being exposed to this inflation is a genuinely *stronger* null than the previous version claimed — but it is a null about a quantity that is not the physiological TE.

### 7.2 The measured forward reach

Define the forward reach $L_{95}$ of a filter as the smallest $D$ with $95\%$ of its $t'>t$ energy inside $[0,D]$. Measured on the production bank, against the $H_d = 120$ s forecast horizon:

| Block | Channels reading past $+120$ s | Reach range |
|---|---|---|
| `fhr_st` (43) | $16/43$ ($37.2\%$) | $2$–$966$ s |
| `up_st` (43) | $16/43$ ($37.2\%$) | $2$–$966$ s |
| `fhr_ph` (66) | $15/66$ ($22.7\%$) | $14$–$267$ s |
| `up_ph` (15) | $13/15$ ($\mathbf{86.7\%}$) | $100$–$267$ s |
| **Target $c_y=109$** | $\mathbf{31/109 = 28.4\%}$ | |
| **Source $c_u=58$** | $\mathbf{29/58 = 50.0\%}$ | |

$\phi$ itself is nearly causal ($L_{95} = 8.75$ s, $0.0000\%$ of energy beyond $120$ s), which is why $S_0$ is the trustworthy channel. By band: $0/11$ beat-to-beat, $0/10$ LF/MF, $5/10$ deceleration and $11/11$ baseline-trend channels read past the horizon.

The consequence is concrete. Injecting a deceleration with nadir at $t_0$ and asking when each channel first exceeds $10\%$ of its peak:

| Channel | $0.0392$ Hz | $0.0233$ Hz | $0.0196$ Hz | $0.0139$ Hz | $0.0082$ Hz | $S_0$ |
|---|---|---|---|---|---|---|
| lead before deceleration onset | $50$ s | $101$ s | $126$ s | $193$ s | $353$ s | $5$ s |
| fraction of response already present at $t_0-120$ s | $0.035$ | $0.234$ | $0.344$ | $0.571$ | $\mathbf{0.815}$ | $\mathbf{0.000}$ |

**A deceleration is $81.5\%$ visible in the $0.0082$ Hz channel two minutes before its nadir.** The "target-only prior" $p(z_t\mid Y_{\le t})$ therefore conditions on the target's own future, and the $120$ s "forecast" is, in $28.4\%$ of target channels, partly a read-off of already-observed data. This is a **validity** failure, not a power failure: whatever `kld_raw` measures, it is not $I(Y_{t+1};U_{\le t}\mid Y_{\le t})$.

The irony is worth stating plainly. The model goes to considerable lengths to eliminate a feature-space causality leak — `CausalGroupNorm` swaps all $18$ encoder `GroupNorm`s to remove a measured $3.4\%$ leak at $T=300$ — while the feature construction upstream leaks hundreds of seconds of raw future into every step. The small leak was fixed; the large one is unaddressed.

### 7.3 This is the operation the TE literature warns about

Weber et al. [12] test precisely low-pass filtering plus downsampling on three coupled systems with a nearest-neighbour TE estimator: $80$ Hz low-pass gave up to $86\%$ false-negative connections; downsampling by $10$ gave up to $100\%$ false-negative *direct* connections; and filtering **consistently caused underestimation of interaction delays**. Their recommendation is explicit: refrain from low-pass filtering and downsampling when inferring causality via TE, and downsample only by a factor smaller than the smallest assumed interaction delay. The scattering front end is a per-channel low-pass followed by $\times16$ downsampling.

Two cautions on how far this transfers. Barnett & Seth's invariance theorem [10] — G-causality is invariant under an arbitrary **invertible** filter applied to a **stationary VAR** process — does **not** apply here: scattering is nonlinear, non-invertible, averaged and downsampled. What transfers is their *practical* mechanism (model-order inflation making inference weak) and Weber et al.'s empirical result. Separately, Daube et al. [13] show classic TE fails on **narrowband** signals through target self-predictability, with "flawed recovery of effect sizes and interaction delays" — and every first-order scattering channel *is* a narrowband envelope. That is an independent mechanism pointing the same way.

---

## 8. Result 4 — the pooled uplift probe's null is a variance artefact

### 8.1 What the probe does

`stage0_source_uplift_probe.py` fits a multi-output ridge predicting $Y_{t+h}$ for $h \in \{1,5,15,29\}$ steps ($4$, $20$, $60$, $116$ s) across all $c_y = 109$ target channels — $436$ outputs jointly — from a baseline design of the target's own lagged features ($7$ lags $\times\,109 = 763$ columns, including lag $0$), versus a full design that appends the source's lagged features ($10$ lags $\times\,58 = 580$ columns for `up`, or $\times\,79 = 790$ for the cross-phase screen). Uplift is the pooled MSE ratio

$$\mathcal{U} = \frac{\mathrm{MSE}_{\text{base}} - \mathrm{MSE}_{\text{full}}}{\mathrm{MSE}_{\text{base}}},
\tag{11}$$

over all eval rows $\times$ $436$ columns. Standardisation and target centring are computed on the fit split only, and $\lambda$ is chosen on a separate validation split — both correct. Row counts: $N_{\text{fit}} \approx 33{,}600$, $N_{\text{eval}} \approx 18{,}000$. Reported:

| Stratification | Source $=$ `up_st` $\oplus$ `up_ph` | Source $=$ cross-phase `fhr_up_ph` |
|---|---|---|
| Pooled | $-2.3\%$ | $-7.5\%$ |
| Per-horizon ($4$–$116$ s) | all $\approx -2.4\%$ | all negative |
| Best single target channel | $-0.9\%$ | all negative |
| Near-contraction anchors (refit) | $-3.9\%$ | $-14.7\%$ |
| Nonlinear estimator | $-0.02\%$ | $\approx -1\%$ |

### 8.2 The masking law, measured on known-answer data

We generated $600$ synthetic $22$-minute recordings at $4$ Hz with a slow baseline, LF variability, beat-to-beat noise, contractions every $\sim150$ s, and an injected deceleration response with jittered lag ($29 \pm 12$ s) firing on $45\%$ of contractions. The injection is calibrated against the real measurement: its own contraction-triggered average is $D^\star = -4.32$ bpm at $\tau = 32$ s (real: $-4.43$ bpm at $29$ s). We then computed the **real** scattering features and ran one ridge estimator with $5$-fold grouped CV by recording, predicting $S_0^Y$ at $120$ s ahead, varying only the width of the source block — a lean causal source ($S_0^U$ at $8$ lags) padded with $p$ columns of pure noise. The decoupled arm pairs each target with a stranger's source, so its uplift must be zero in expectation for an unbiased estimator.

| $p_{\text{extra}}$ | coupled $\mathcal{U}$ (%) | decoupled $\mathcal{U}$ (%) = floor | source-specific (%) | $p_{\text{extra}}/N_{\text{train}}$ (%) |
|---:|---:|---:|---:|---:|
| $8$ | $+2.14 \pm 0.17$ | $+0.03 \pm 0.04$ | $2.11 \pm 0.15$ | $0.02$ |
| $33$ | $+2.08 \pm 0.15$ | $-0.04 \pm 0.04$ | $2.11 \pm 0.15$ | $0.08$ |
| $58$ | $+2.04 \pm 0.15$ | $-0.07 \pm 0.03$ | $2.12 \pm 0.15$ | $0.14$ |
| $108$ | $+1.94 \pm 0.12$ | $-0.19 \pm 0.06$ | $2.13 \pm 0.15$ | $0.26$ |
| $208$ | $+1.65 \pm 0.12$ | $-0.48 \pm 0.08$ | $2.14 \pm 0.16$ | $0.50$ |
| $408$ | $+1.20 \pm 0.14$ | $-0.95 \pm 0.17$ | $2.15 \pm 0.16$ | $0.98$ |
| $808$ | $\mathbf{+0.13 \pm 0.19}$ | $-2.03 \pm 0.22$ | $2.16 \pm 0.15$ | $1.93$ |

($N_{\text{train}} = 41{,}760$ per fold.) Three things to read off:

1. **The source-specific uplift is constant at $2.11$–$2.16\%$ across the entire sweep.** The coupling is fully present and fully recoverable at every design width. Nothing about the representation changes.
2. **The measured uplift collapses from $+2.14\%$ to $+0.13\%$** as the design widens, and the decoupled floor tracks $-p_{\text{extra}}/N_{\text{train}}$ to within $0.1$ percentage points at every point. This is textbook out-of-sample optimism for added regressors, and Eq. (4) names it: the added columns are full-model misfit, not absent TE.
3. At $p_{\text{extra}} = 808$ — essentially the real probe's cross-phase width of $790$ — the measured uplift is $+0.13\%$, **below the probe's own `_UPLIFT_REL_FLOOR` $= 1\%$**, so the probe would return `data_ceiling` ("no extractable coupling") on data containing a fully recoverable $2.1\%$ coupling.

### 8.3 Applying the law to the real probe

With $N_{\text{fit}} \approx 33{,}600$ the floors are

$$\frac{580}{33{,}600} = 1.7\%\ (\texttt{up}), \qquad \frac{790}{33{,}600} = 2.4\%\ (\texttt{cross}),$$

rising to $5.8\%$ and $7.8\%$ in the near-contraction refit ($N_{\text{fit}} \approx 10{,}080$). The observed negatives match in **sign, magnitude and ordering**: more negative for the wider cross design, more negative in the smaller refit.

Repeating the sweep at $N_{\text{train}} = 13{,}920$ per fold — comparable to that near-contraction refit — reproduces the real probe's *negatives* outright, on data whose coupling we injected ourselves:

| $p_{\text{extra}}$ | coupled $\mathcal{U}$ (%) | floor (%) | source-specific (%) |
|---:|---:|---:|---:|
| $8$ | $+2.22 \pm 0.15$ | $-0.06 \pm 0.26$ | $2.29 \pm 0.27$ |
| $108$ | $+1.64 \pm 0.14$ | $-0.60 \pm 0.28$ | $2.23 \pm 0.29$ |
| $408$ | $\mathbf{-0.33 \pm 0.32}$ | $-2.62 \pm 0.38$ | $2.29 \pm 0.28$ |
| $808$ | $\mathbf{-3.43 \pm 0.51}$ | $-5.78 \pm 0.64$ | $2.35 \pm 0.30$ |

A pooled uplift of $-3.4\%$ — squarely inside the range the probe reported — is here produced by a coupling that the same estimator recovers at $+2.3\%$ the moment the nuisance columns are removed. **A negative pooled uplift is not evidence of absent coupling.** The declared $1\%$ detection threshold sits *below* the floor in every cell. Adding back the floor leaves nothing distinguishable from zero, and the `up` near-contraction cell becomes nominally positive ($-3.9 + 5.8 = +1.9\%$, close to the simulation's $+2.1\%$) — a hypothesis-generating coincidence, not a result: that stratum was refit on a different eval split with a different $\lambda$.

Effective sample size is smaller still than these numbers suggest, in two ways the probe does not account for: anchors are stride-1 within a segment and strongly autocorrelated, and the grouping variable is the **22-minute segment, not the GUID** (`group_id = enumerate(samples)`; `Sample.guid` is loaded and never used), so the same subject appears on both sides of every split and a "stranger's source" may be the same mother twenty minutes later.

Four further defects mean the reported table cannot bear the weight placed on it:

* **The shuffle control imposes no constraint in the observed regime.** The verdict rule `uplift_rel >= 2.0 * max(uplift_shuffled_rel, 0.0)` degenerates to `uplift_rel >= 0` whenever the shuffled uplift is negative — which is exactly what happened. The control is also biased toward "coupling": weights are fit on *matched* data where the source block absorbs its collinearity with the history block, so permuting the source at eval time raises MSE even under H0.
* **`fhr_up_ph` is never normalised.** It is absent from `normalize_fields`, so under the cross screen the source block enters as raw, signed, heavy-tailed coherence covariances while everything else is log/asinh + z-scored. The larger negative for `cross` is therefore **not** cleanly attributable to representation; column count and conditioning both differ.
* **No uncertainty quantification.** One seed, one split, one permutation; no CI, no permutation distribution, no p-value.
* **The nonlinear check is weak.** It collapses the $436$-dimensional target to a single scalar (channel-mean of the $116$ s block), which averages away exactly the channel-localised effect the per-channel stratification exists to find, and uses sklearn's default random-row early stopping with `random_state=None`.

One genuinely strong result survives and should be stated positively: the maximum uplift over all $109$ target channels was still **negative**. That is a stronger null than the pooled figure — though it too sits inside the same variance floor.

### 8.4 The conditioning-set effect is real but second-order

The same simulation varied the target conditioning set with the source held fixed. Adding the $16$ anticipating channels lifted the source-blind baseline from $R^2 = 0.719$ to $R^2 = 0.799$ — the anticipation of §7 buying $8$ points of *spurious*, non-causal predictability. But the source-specific uplift barely moved ($2.24 \pm 0.40$ with the full $43$-channel set, $2.13 \pm 0.42$ causal-only, $2.17 \pm 0.40$ with $S_0$ alone). **Feature non-causality inflates the baseline and invalidates the causal reading; it does not, by itself, destroy the measurable uplift.** We had expected otherwise, and record the correction.

---

## 9. What the §8-of-the-previous-version "isolating experiment" would show

The previous version proposed applying the event-locked estimator to the $S_0$ feature channel to decide whether the deceleration survives the representation, and left it as future work. It does not need to be run to be answered, and the answer is not ambiguous: $S_0$ is a $4.5$ s Gaussian-weighted local mean of FHR with forward reach $8.75$ s, and it retains a realistic deceleration at $96.9\%$ of its depth with zero timing shift (§6.1). The aligned-$S_0$ test **must** show the dip, attenuated by at most a few percent.

Running it remains worthwhile as a positive control on real data — the probe would need a `load_fields` change plus a second sampling rate for the response geometry ($f_{\text{response}} = 0.25$ Hz, $\text{pre} = 7$ steps, $\text{post} = 43$ steps, lag window $[2,37]$ steps), with the trigger detector left on the raw $4$ Hz UP so the event definition is held fixed. But it is a confirmation, not a discriminator, and the previous version's framing of it as *the* test that separates representation from estimator was mistaken: it varies the representation while leaving the estimator aligned in both arms, so it could never have isolated the estimator. The experiment that does isolate them is §8.2 — same data, same estimator, same representation, only the design width varying.

---

## 10. The corrected account

| Cause | Status | Evidence | Fixable by |
|---|---|---|---|
| **H3 Objective**: $\beta = 13.6\,\beta^\star$; reported $K_t$ capped at $2.4$ nats/step and otherwise unregularised | **Decisive for the model** | Closed-form, §5 | One config value |
| **H2 Estimator**: variance floor $1.7$–$7.8\%$ exceeds the $1\%$ detection threshold | **Decisive for Diagnostic I** | Known-answer sweep, §8.2 | Estimator design |
| **H1b Non-causal features**: $28.4\%$ of target / $50\%$ of source read past the horizon; deceleration $81.5\%$ visible $120$ s early | **Invalidates the causal reading; second-order for power** | Measured, §7.2; simulation, §8.4 | Channel restriction or a causal front end |
| **H1a Information loss**: modulus / low-pass / decimation destroy the coupling | **Not supported** | $96.9\%$ retention, gain $\ge0.988$, $0/10$ aliased, §6 | — |
| **H0 No coupling** | **Rejected** | $z = -43.8$, §3 | — |

The honest one-line summary: *the coupling is real and the representation admits it; the model's near-null $K_t$ is what its own objective is configured to return, and the probe's null is what its own design width guarantees.*

---

## 11. What to change

**Ordered by ratio of decisiveness to effort.**

1. **Set $\beta$ below $\beta^\star$, or renormalise the KL.** Either $\beta_{\mathrm{end}} \ll 7.34\times10^{-3}$, or — better, because it removes the geometry dependence entirely — score the KL as a *total per anchor* ($\times d_z$) against a *total per anchor* forecast NLL ($\times H_d c_y$), so that $\beta$ is a genuine nats-for-nats exchange rate and $\beta = 1$ means "one nat of rate must buy one nat of forecast". Recompute $\beta^\star$ whenever $d_z$, $H_d$ or $c_y$ changes: at $c_y = 87$ (the pre-migration width) $\beta^\star$ was $9.2\times10^{-3}$. Report `kld_raw` in nats/step ($\times d_z$), never the per-dimension mean, and state the free-bits cap $d_z\cdot\mathrm{free\_bits}$ alongside it.

2. **Make the conditioning set causal.** At a $120$ s horizon, $27$ of $43$ `fhr_st` channels have forward reach under the horizon. Restricting the target and source blocks to those channels costs $8$ points of (spurious) baseline $R^2$ and makes $K_t$ a transfer entropy again. The alternative — and the better long-run answer — is a causal (one-sided) analytic front end, which also removes the group-delay bias Weber et al. document. Note this bound is horizon-dependent: at a $30$ s horizon far fewer channels qualify.

3. **Rebuild the uplift probe around its own noise floor.** Report the decoupled/permuted uplift as the floor in every cell and quote *source-specific* uplift (matched minus permuted); prefer a lean, physiologically-targeted source design ($S_0^U$ at a few lags) over the full $580$–$790$-column block, or spend the degrees of freedom smoothly (spline-Granger-style [14] cuts parameters $\sim5\times$ and materially improves detection of small effects); group by GUID, not by segment; and give a permutation distribution rather than a single draw. Do not compare `up` against `cross` until `fhr_up_ph` is normalised.

4. **Fix the two Stage-2 reporting defects.** Make the contraction-onset walk-back actually walk back (the loop condition is entered at a point where it is already false), and save the CTA waveform plus the pre-window absolute level for real and null triggers, so the baseline-elevation confound of §3 can be tested. Raise `n_shuffle` above $300$ if a p-value smaller than $3\times10^{-3}$ is to be quoted, and report a GUID-clustered interval.

5. **Match the estimator to the effect.** The BPRSA literature is the modality-matched precedent: event-locked phase-rectified averaging detects UP–FHR coupling in $\approx90\%$ of cases where cross-spectral methods detect it in $24$–$48\%$ [15, 16]. The lag-attention divergence readout is the right architecture; the objective should reward the deceleration response where the signal-to-noise is highest rather than a pooled $109$-channel horizon forecast.

6. **Compare against the one direct antecedent.** Warrick & Hamilton [3] computed MI and TE between UP and FHR with a Kraskov estimator on raw signals and separated normal from pathological fetuses $110$–$160$ min before delivery. Any claim that TE here is unmeasurable has to explain that result.

---

## 12. Changes from the previous version

| Previous claim | Status | Replacement |
|---|---|---|
| The representation is invariant to the coupling's carrier; TE is not extractable from scattering/phase | **Withdrawn** | $\phi$ retains $96.9\%$ of the deceleration with zero timing shift; band envelopes pass at gain $\ge0.988$; known-answer uplift recovers the coupling from scattering features (§6, §8.2) |
| "The modulus discards the analytic phase" as the mechanism of information loss | **Corrected** | The modulus is invertible up to a global phase [4,6]; the lossy operations are averaging and downsampling, and the fragile coordinate is timing [5] |
| Deceleration-band wavelets span $25$–$744$ s | **Corrected** | Measured $\pm2\sigma_t$ span $111$–$526$ s; $\sigma_t = 27.7$–$131.6$ s |
| Only the cross-phase block is non-causal (§5.4) | **Generalised** | $28.4\%$ of the target block and $50\%$ of the source block read past the $120$ s horizon; the "target-only prior" sees the target's future (§7.2) |
| Eq. (7): feature processing can only reduce TE, so the feature TE lower-bounds the truth | **Corrected** | The bound is over the *block* future $Y_{t+1:t+\Delta}$; and coarsening the conditioning set makes the feature TE neither an upper nor a lower bound (Eq. 10, [11]) |
| The pooled/aligned confound cannot be separated without the aligned-$S_0$ test | **Superseded** | The aligned-$S_0$ test is answerable analytically and cannot isolate the estimator; §8.2 isolates it by varying design width alone |
| $\tau^\star \approx 29$ s is a rising-edge-onset-to-nadir delay | **Corrected** | The trigger is effectively the contraction *peak*, and after the $20$ s pre-shift the dip is $\approx 9$ s after the true peak (§3.2) |
| $p \approx 0$; "1,200 recordings" | **Corrected** | $p < 1/300$; $1{,}200$ prefix-selected 22-min segments from one cohort shard, pooled by event without clustering (§3.3) |
| Estimator sensitivity is a $\sqrt{N}$ story about alignment | **Sharpened** | The binding term is added-column optimism $\approx p_{\text{extra}}/N_{\text{fit}}$, measured and validated (§8.2) |
| — | **New** | The objective's critical $\beta^\star = \lambda_{\text{full}}d_z/(H_dc_y)$ and the $13.6\times$ overshoot (§5) |

**What did not change:** the coupling is real; $S_1$ is exactly sign-blind; the phase-harmonic blocks measure marginal cross-scale coherence rather than a target-past-conditioned directional quantity; the cross-phase block cannot enter the source pathway without destroying source-purity; and the architecture's causal, source-pure divergence readout is the right design.

---

## 13. Reproducibility

The measurements in §4, §6 and §7.2 come from rebuilding the production filter bank directly and are exactly reproducible from the pipeline constants; the rebuild is self-validating (it reproduces $903$ pairs and the $66$/$15$ phase-channel counts). §8.2 and §8.4 are synthetic known-answer experiments and make no claim about the real cohort beyond the calibration of the injected effect size to the measured $D^\star$. The real dataset is not present on the development machine — the configured shards are `/data1/...REPOINT_ME.../` placeholders — so neither production probe was re-run here; their numbers are taken from the recorded run output and were verified line-by-line against the code that produced them.

**Two provenance gaps worth closing.** Both probes default to `RUN_JSON_OUT = None`, so no machine-readable artefact of either run exists anywhere in the repository; every published number survives only in prose. And `default.yaml`'s shard paths are deliberate placeholders, so the exact data behind the published table cannot be identified. Set `--json-out` and record the resolved shard paths before the next run.

---

## Notation

| Symbol | Meaning |
|---|---|
| $Y,\ U$ | raw FHR (target), UP (source) at $f_s=4$ Hz |
| $T_{U\to Y}$ | transfer entropy, Eq. (1) |
| $z_t,\ K_t$ | latent; per-step posterior–prior KL (the reported surrogate) |
| $\psi_i,\ \phi$ | analytic Morlet wavelet $i$ (centre $\xi_i$, width $\sigma_i$); low-pass ($T=16$) |
| $S_0,\ S_1$ | scattering channels, Eq. (6); $1+42=43$ per stream |
| $\sigma_t,\ L_{95}$ | filter RMS time width; forward reach (95% of $t'>t$ energy) |
| $C_{i,j,p}$ | phase-harmonic correlation, Eq. (8) |
| $c_y=109,\ c_u=58$ | target and source stream widths |
| $H_d=30,\ d_z=24$ | forecast horizon (steps); latent width |
| $\beta,\ \beta^\star$ | KL weight; its critical value $\lambda_{\mathrm{full}}d_z/(H_dc_y)$ |
| $\mathcal{U}$ | out-of-sample predictive uplift, Eq. (11) |
| $p_{\text{extra}},\ N_{\text{fit}}$ | added source columns; fit-split rows |
| $\bar d(\tau),\ D^\star$ | contraction-triggered average and dip statistic, Eq. (5) |

## References

1. T. Schreiber. *Measuring information transfer.* Physical Review Letters **85**(2):461–464, 2000.
2. L. Barnett, A. B. Barrett, A. K. Seth. *Granger causality and transfer entropy are equivalent for Gaussian variables.* Physical Review Letters **103**:238701, 2009.
3. P. A. Warrick, E. F. Hamilton. *Information theoretic measures of perinatal cardiotocography synchronization.* Mathematical Biosciences and Engineering **17**(3):2179–2192, 2020.
4. S. Mallat, I. Waldspurger. *Phase retrieval for the Cauchy wavelet transform.* Journal of Fourier Analysis and Applications **21**:1251–1309, 2015 (arXiv:1404.1183).
5. I. Waldspurger. *Wavelet transform modulus: phase retrieval and scattering.* Journées équations aux dérivées partielles, Exp. No. 10, 2017.
6. J. Andén, S. Mallat. *Deep scattering spectrum.* IEEE Transactions on Signal Processing **62**(16):4114–4128, 2014.
7. S. Mallat, S. Zhang, G. Rochette. *Phase harmonic correlations and convolutional neural networks.* Information and Inference **9**(3):721–747, 2020 (arXiv:1810.12136).
8. S. Zhang, S. Mallat. *Maximum entropy models from phase harmonic covariances.* Applied and Computational Harmonic Analysis **53**:199–230, 2021.
9. S. Mallat. *Group invariant scattering.* Communications on Pure and Applied Mathematics **65**(10):1331–1398, 2012.
10. L. Barnett, A. K. Seth. *Behaviour of Granger causality under filtering: theoretical invariance and practical application.* Journal of Neuroscience Methods **201**(2):404–419, 2011.
11. D. A. Smirnov. *Spurious causalities with transfer entropy.* Physical Review E **87**(4):042917, 2013.
12. I. Weber, E. Florin, M. von Papen, L. Timmermann. *The influence of filtering and downsampling on the estimation of transfer entropy.* PLOS ONE **12**(11):e0188210, 2017.
13. C. Daube, J. Gross, R. A. A. Ince. *A whitening approach for transfer entropy permits the application to narrow-band signals.* arXiv:2201.02461, 2022.
14. E. Spencer et al. *A procedure to increase the power of Granger-causal analysis through temporal smoothing.* Journal of Neuroscience Methods **308**:48–61, 2018.
15. D. Casati, T. Stampalija, K. Rizas, et al. *Assessment of coupling between trans-abdominally acquired fetal ECG and uterine activity by bivariate phase-rectified signal averaging analysis.* PLOS ONE **9**(4):e94557, 2014.
16. J. E. Montero-Nava, A. C. Pliego-Carrillo, C. I. Ledesma-Ramírez, et al. *Analysis of the fetal cardio-electrohysterographic coupling at the third trimester of gestation in healthy women by bivariate phase-rectified signal averaging.* PLOS ONE **15**(7):e0236123, 2020.
17. J. Vargas-Calixto, Y. Wu, M. Kuzniewicz, et al. *The nonlinear dynamic response of intrapartum fetal heart rate to uterine pressure.* Computing in Cardiology **49**, 2022.
18. T. M. Cover, J. A. Thomas. *Elements of Information Theory*, 2nd ed. Wiley, 2006 (data-processing inequality; conditional mutual information).
19. A. A. Alemi, I. Fischer, J. V. Dillon, K. Murphy. *Deep variational information bottleneck.* ICLR, 2017.

> **Citation hygiene.** References 4–8 were checked against primary abstracts; the specific quotations attributed to [4] and [6] are verbatim. Reference 8's volume/pages should be confirmed against DOI 10.1016/j.acha.2021.01.003 before external submission. The phase-harmonic reconstruction-failure conditions cited in §6.4 are at abstract-plus-discussion level in [7]; read §6–7 of arXiv:1810.12136 before quoting them verbatim. No published study computes UP→FHR transfer entropy on scattering or phase-harmonic features, so [3] is the only direct antecedent for the raw-signal case and there is no literature baseline effect size in bits.
