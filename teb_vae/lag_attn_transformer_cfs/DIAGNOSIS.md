# Diagnosis of `lag_attn_trf_cfs_baseline`

> **This document describes a run at $H = 15$, and the shipped horizon is now $30$.** It is a record
> of what that run did, not a description of the current configuration, and it is deliberately not
> rewritten: rewriting a measurement to match a later config is how a record stops being one. Read
> every number here as belonging to a $1470$-coefficient block over $\approx 10.1$ decoded anchors
> per step. Nothing in the *diagnosis* is invalidated by the change — the source-path generalisation
> failure, the `base_decode: mean` estimator asymmetry, the availability-clock share of the KL and
> the latent-variance pinning are all properties of the objective and the split rather than of the
> horizon — but §8.2's persistence table is the part the change bears on most directly, and it bears
> on it in the **unfavourable** direction: the loss now weights fifteen further horizon steps, all of
> them beyond the $\tau$ at which persistence $R^2$ has already gone negative. See the note at the
> end of §8.2.

This document reviews the model in `teb_vae/lag_attn_transformer_cfs`, its relationship to
`teb_vae/lag_attn_cfs`, and the trained run in
`output/metrics_history_lag_attn_trf_cfs_baseline`.

Run artifacts reviewed:

- `metrics_history.csv`: 710 validation epochs, epochs 0 through 709.
- `resolved_config.yaml`: the configuration actually used by the run.
- `lag-attn-trf-cfs-epoch=336.ckpt`: the checkpoint selected by minimum `val/total_loss`.
- Checkpoint metadata and state dict, loaded strictly through the repository's checkpoint-loading
  convention.

The main conclusion is that the result is **not explained by insufficient model size**. The strongest
evidence supports a combination of:

1. source-path generalization failure or source-distribution shift;
2. a `pred_gap` estimator that compares a deterministic prior-mean decode to a stochastic posterior
   decode;
3. a source-null KL dominated by source-path behavior that does not depend on source variation;
4. severe latent variance pinning and concentration of useful KL in very few dimensions; and
5. a forecast objective that makes it rational to discard fast, weakly predictable feature channels.

The first item explains the opposite train/validation signs. Items 2 and 3 make the reported
coupling statistics harder to interpret. Items 4 and 5 limit the quality and scientific clarity of
the learned representation. The available evidence does **not** establish that `d_z: 64`, transformer
depth, or the total 5.05 million parameter count is the limiting resource.

---

## 1. Executive findings

| Priority | Finding | Evidence strength | Consequence |
|---|---|---|---|
| **P0** | The source-conditioned branch generalizes much worse than the target-only branch. | Strong, from the run history. | Positive train `pred_gap` and negative validation `pred_gap` are consistent with source-path overfitting, split shift, or both. |
| **P0** | `pred_gap` is not a like-for-like predictive comparison under `base_decode: mean`. | Certain, from code/config and checkpoint inference. | Its sign mixes source value with unequal latent sampling noise. |
| **P0** | The run continued long after the selected validation optimum. | Certain. | Epoch 336 is the best saved checkpoint, but training to epoch 709 wasted compute and worsened validation metrics. |
| **P1** | About 86% of the checkpoint KL survives source-mean/null replacement on the probe cohort. | Strong on the external probe, not yet confirmed on the production split. | `source_conditioned_kl_raw` is not a clean source-information measurement. |
| **P1** | Most latent variances are pinned to the lower bound, with useful KL concentrated in about two dimensions. | Strong, reproduced in metrics and checkpoint probes. | This is dimensional and variance collapse, although not complete VAE posterior collapse. |
| **P1** | Many fast phase-harmonic targets are suppressed at the input adapter. | Strong on the external probe. | Much of the NLL comes from channels whose values become nearly unpredictable over the uniformly weighted 15-step horizon. |
| **P2** | A shrunk source displacement helps on the external cohort, while the trained full displacement hurts. | Mechanistically useful but exploratory. | The learned direction contains signal, but its scale does not transfer. This does not prove a production-split improvement or imply a specific new $\beta$. |
| **P2** | Low model capacity is not demonstrated. | Strong negative evidence against the current diagnosis. | Widening `d_z` or adding transformer blocks is a low-priority experiment, not the first fix. |

### Direct answers

**Is the model overfitting?** Yes, the histories are strongly consistent with overfitting localized
primarily to the source-conditioned branch. However, the artifacts do not allow a clean separation
of ordinary parameter overfitting from recording-level distribution shift. The production HDF5
shards are not present locally, so recording overlap, cohort composition, and per-recording confidence
intervals could not be audited.

**Why is train `pred_gap` positive and validation `pred_gap` negative?** The full branch learns a
large training advantage that does not transfer. In epochs 600--709, the sampled train rows average
$+56.15$ nats/anchor while validation averages $-41.60$. The difference, about $97.75$ nats/anchor,
is exactly the excess generalization gap of the full branch over the base branch. The source path is
therefore the main owner of the sign reversal.

**Is the transformer too small?** There is no evidence for that conclusion. The model has 5,051,920
parameters. A latent activation probe has participation ratio 23.7 and requires about 40 of 64
principal components for 90% of variance; it is not filling all 64 dimensions uniformly. The target
encoder state also retains substantial unused linear dimensionality. Capacity can still be tested,
but the current evidence points first to objective, regularization, and split issues.

**Should `d_z` be increased?** Not as the first intervention. The discarded target-channel
information is already largely removed by the first target adapter, before the latent bottleneck.
Increasing `d_z` cannot restore information the optimized adapter chooses not to pass. A target
residual/persistence path or horizon-aware objective is a more direct test.

**Are hyperparameters involved?** Yes. The absence of early stopping, no dropout on the attended
source summary, the KL/prior-scale balance, and `base_decode: mean` all matter. No single artifact
proves that changing only $\beta$ will solve the problem.

---

## 2. Model and training path

### 2.1 Architecture

`SeqVaeLagAttnTrfCfs` composes:

```text
CausalWarmupInputs
    + CausalFeatureForecastTarget
    + SeqVaeLagAttnTrfRws
```

At each admitted anchor $t$, the model reads causal target features $Y_{\le t}$ and causal source
features $U_{\le t}$. It forecasts the next 15 decimated steps for 98 retained target channels, or
$15 \times 98 = 1470$ Gaussian coefficients per anchor.

The two forecast branches are:

$$
p(z_t\mid Y_{\le t}) \quad\text{and}\quad
q(z_t\mid Y_{\le t}, U_{\le t}).
$$

The target-only prior head consumes the target transformer state. The posterior head consumes the
same target state plus a lag-attended source-transformer summary and produces a bounded residual
relative to the prior mean. Both latent paths use one shared horizon decoder. There is no learned
decoder path around $z$.

The loss is:

$$
\mathcal L = D_1 + D_0 + \beta(e)\,\mathrm{KL}(q\Vert p) + \beta_p R_p,
$$

where $D_0$ is `nll_base_block`, $D_1$ is `nll_full_block`, $\beta$ ramps linearly from 0 to 1 over
50 epochs, and $\beta_p=0.1$. The three waveform-specific auxiliary losses are zero. The reported
gap is:

$$
\mathrm{pred\_gap}=D_0-D_1,
$$

so a positive value means the source-conditioned branch has lower NLL.

### 2.2 Transformer versus `lag_attn_cfs`

The causal-feature target, forecast block, VAE heads, lag attention, decoder, objective, seed, batch
size, and principal hyperparameters are intended to match `lag_attn_cfs`. The meaningful
architectural difference is the history encoder:

- `lag_attn_cfs`: gated/dilated convolution plus LSTM, with causal normalization required;
- `lag_attn_transformer_cfs`: gated causal convolution stem plus pre-normalized causal
  self-attention with RoPE and RMSNorm.

The transformer target encoder has 6 blocks and the source encoder has 3 blocks, at `d_model: 128`
and feed-forward width 512. The transformer run also uses a 2,000-optimizer-step learning-rate
warmup and its own gradient-clip threshold.

No completed `lag_attn_cfs` checkpoint or metric history was found locally. Therefore this review
can compare the two implementations and resolved configurations, but it cannot claim that the
transformer is empirically better or worse than the LSTM causal model. That comparison requires the
paired run on the same recording split.

### 2.3 Parameter inventory

The checkpoint constructs strictly with 5,051,920 parameters, of which 5,035,408 are trainable.
`lag_attn.W_o` is frozen by design.

| Component | Parameters | Share |
|---|---:|---:|
| Shared horizon decoder | 2,001,590 | 39.6% |
| Target transformer encoder | 1,676,928 | 33.2% |
| Source transformer encoder | 888,960 | 17.6% |
| Posterior head | 116,484 | 2.3% |
| Prior head | 108,010 | 2.1% |
| Input adapters | 173,056 | 3.4% |
| Lag attention and query projection | 86,892 | 1.7% |

This is not a small network relative to the feature dimensions. Parameter count alone cannot prove
adequate statistical capacity, but it rules out the simple claim that the run failed because the
network is tiny.

---

## 3. Evidence and limitations

The conclusions use three evidence sources, which must not be conflated.

### 3.1 Production metric history

All `val/` columns are epoch aggregates. The unsuffixed `train/` columns in this CSV are **one final
optimizer-step value per epoch on rank 0**, not full epoch means. This was verified against the
Lightning 2.5 callback behavior with a minimal four-batch experiment: at validation end,
`trainer.callback_metrics["train/x"]` held the last step value, while `train/x_epoch` was not yet
available to the callback.

Consequences:

- validation levels and minima can be read directly;
- individual train rows must not be treated as epoch means;
- train curves below use rolling or long-window averages;
- precise train/validation crossing epochs are approximate.

The current `RESULTS.md` description of train values as epoch means should be corrected.

### 3.2 Production checkpoint

The epoch-336 checkpoint was loaded strictly through
`train.graph_models_utils.load_checkpoint_strict`. Its class, model kwargs, parameter inventory, and
state-dict keys are consistent. The checkpoint is the minimum of the monitored `val/total_loss`, not
the final epoch-709 weights.

### 3.3 External causal-feature probe

Additional forward probes used 339 windows from
`output/causal_scattering/hie_cs_causal.hdf5`, producing 51,516 dense anchors. The geometry matches
the model, but this is a different dataset build and not the production validation split. Statistics
were regenerated for this shard before normalization.

**Which shard, and in which alignment state.** `hie_cs_causal.hdf5` is the *legacy* causal build: it
predates both the envelope leg-alignment operator and the cross-channel input alignment, and the
model these 339 windows were pushed through was built with `causal_align_reference` **unset** — the
key did not exist when the run was made. So every number in this section is an **unaligned**
measurement on an **unaligned** shard, and none of it is a measurement of the shipped
`causal_align_reference: target_max` geometry. The numbers are left exactly as they were taken; what
they must not be used for is a before/after against an aligned run, because the shard differs as
well as the model. That comparison is the pre-registration in §9.2, and it has not been run.

The probe reproduces several production-validation directions:

| Metric | Production validation, epochs 600--709 | External probe |
|---|---:|---:|
| `pred_gap` | -41.60 | -43.44 |
| `pred_gap_st` | about -5.96 | -7.94 |
| `pred_gap_ph` | about -35.82 | -35.50 |
| `source_conditioned_kl_raw` | 3.796 | 3.960 |
| `kld_source_null` | 3.131 | 3.395 |
| `logvar_prior_floor_frac` | 0.955 | 0.953 |

This agreement makes the probe useful for mechanism discovery. It does **not** make its absolute
NLL, post-hoc optimal scale, or channel-level effect sizes production-validation estimates. Every
probe result below remains exploratory until rerun on the fixed production train and validation
recordings with per-recording uncertainty.

---

## 4. Training-history diagnosis

### 4.1 Run mechanics are healthy

The run does not show a geometry or numerical failure:

- `target_warm_frac` is exactly 1.0 for train and validation on every row;
- train `anchors_per_sample` is 10.052--10.247 and validation is exactly 152;
- no non-finite metric rows and no spike-breaker skips occurred;
- anchor coverage is at least 0.998;
- the sampled gradient norm exceeded 6500 once, at epoch 336;
- the forecast block decompositions agree to numerical tolerance.

These observations eliminate warmup admission, tiling, gross instability, and a continuously binding
gradient clip as primary explanations.

### 4.2 Validation optima

| Quantity | Best epoch | Best value |
|---|---:|---:|
| `val/total_loss` | **336** | **585.528** |
| `val/nll_base_block` | **336** | **270.483** |
| `val/nll_full_block` | **278** | **297.054** |

At the shipped checkpoint, validation `pred_gap` is $270.483-298.994=-28.511$ nats/anchor. By epoch
709 it is -40.620, and validation total loss is 627.839. Training beyond epoch 336 did not improve
the monitored objective. The checkpoint callback protected the selected artifact, but the disabled
early stopping spent another 373 epochs without producing a better selected model.

The learning-rate decay at epoch 400 did not reverse overfitting. Mean validation total loss is
607.76 over epochs 300--399 and 630.92 over 400--709, while the sampled train loss decreases. This
comparison is time-confounded: it shows the decay failed to rescue generalization, not that the
decay itself caused a 23-nat regression.

### 4.3 Source-path generalization failure

Long-window averages over epochs 600--709 are:

| Metric | Train sampled rows | Validation epoch mean |
|---|---:|---:|
| `nll_base_block` | 220.68 | 287.86 |
| `nll_full_block` | 164.53 | 329.46 |
| `pred_gap` | **+56.15** | **-41.60** |
| `total_loss` | 401.04 | 633.22 |

The branch generalization gaps are approximately:

$$
G_0 = 287.86-220.68=67.18,
$$

$$
G_1 = 329.46-164.53=164.93.
$$

Thus the source-conditioned branch has an excess gap of:

$$
G_1-G_0=97.75
=\mathrm{pred\_gap}_{\rm train}-\mathrm{pred\_gap}_{\rm val}.
$$

Using centered rolling-25 means, the full-branch gap first becomes positive near epoch 99,
validation `pred_gap` becomes negative near epoch 113, and the base-branch gap first becomes positive
near epoch 223. These dates are approximate because train points are sampled steps, but their order
is stable: source-specific failure appears materially earlier than target-only failure.

This is strong evidence that the sign reversal is not caused only by a weak base model. The full
branch learns additional training-set structure that does not transfer. Plausible causes are:

1. excessive source-path flexibility relative to the number of independent recordings;
2. recording or cohort shift in source-target relationships;
3. source availability/bias shortcuts;
4. training/validation recording leakage or duplication in either direction; or
5. a displacement scale that is optimized for training likelihood but poorly calibrated out of
   sample.

The artifacts cannot distinguish items 1 and 2 because they contain window-level steps but not a
recording-level split audit. The nominal $565\times128\times7\approx506{,}000$ windows per epoch are
not 506,000 independent examples.

---

## 5. `pred_gap` does not compare equivalent estimands

The resolved configuration uses:

```yaml
base_decode: mean
posterior_logvar_mode: independent
```

The base branch is decoded at $z_p=\mu_p$. The full branch is decoded at
$z_q=\mu_q+\sigma_q\epsilon$. Consequently,

$$
D_0-D_1
=D(y;\mu_p)-D(y;z_q)
$$

mixes a deterministic point prediction with a stochastic draw. It is not solely the value of
conditioning on $U$.

On the external probe:

| Evaluation | NLL, nats/anchor |
|---|---:|
| Base at prior mean | 399.02 |
| Full at posterior sample | 442.46 |
| Full at posterior mean | 429.93 |
| Base at prior sample, four draws | 476.26 |

Therefore:

- shipped `pred_gap`: $399.02-442.46=-43.44$;
- both branches at their means: $399.02-429.93=-30.91$;
- the posterior sample costs about 12.52 nats relative to its mean on this probe.

Sampling both distributions with common random numbers gives a positive gap on this cohort, but that
is not an automatic replacement: the prior is much wider than the posterior in the two dominant
dimensions, and a single-sample difference remains noisy. Likewise, the both-means gap measures
conditional mean-path performance, not predictive distribution quality.

### Required reporting change

Report at least three separate quantities:

1. **Mean-path source gain:** $D(y;\mu_p)-D(y;\mu_q)$.
2. **Monte Carlo predictive log likelihood:** estimate each branch's marginal predictive density
   with the same number of latent samples and stable `logsumexp` aggregation.
3. **Common-random-number paired sample difference:** report mean and per-recording interval as a
   variance-reduced diagnostic, not as the marginal likelihood.

The existing evaluation package already contains MC `pred_gap` support; the unresolved eval config
should be bound to the production shards and run. Multi-sample likelihood estimation is consistent
with the motivation of importance-weighted evaluation, but sample count and uncertainty must be
reported.

---

## 6. What the source-null and intervention probes show

### 6.1 Most KL is not driven by source variation

On the external probe:

| KL component | Nats/anchor | Share |
|---|---:|---:|
| Total $\mathrm{KL}(q\Vert p)$ | 3.960 | 100% |
| Mean-displacement term | 1.335 | 33.7% |
| Variance-ratio term | 2.625 | 66.3% |
| Source-null KL | 3.395 | 85.7% |
| Difference, matched minus null | 0.565 | 14.3% |

Replacing the normalized source values with their mean/zero leaves the source availability pattern,
adapter biases, source-encoder response to a flat trajectory, and posterior-head biases. Therefore
the result means:

> About 85.7% of the KL survives removal of source **variation**.

It does **not** prove that 85.7% is literally an availability clock, and the remaining 0.565 nats is
not a mutual-information estimate. Isolating the availability announcement requires a separate
ablation of the source adapter's announcement parameters, with all other source-path computations
unchanged — `source_adapter.mask_proj` when this run was made, and `mask_proj` together with
`start_embed` under the configuration that ships now, for the reason the next paragraphs give.

The suspicion is nevertheless concrete. `AvailabilityInputAdapter.forward`, in
`teb_vae/lag_attn/nets/encoders.py`, computes:

```python
if self.mask_proj is not None:
    available = self._slice(self.availability, seq_len)
    embedded = self.linear(x * available) + self.mask_proj(available - 1.0)
else:
    embedded = self.linear(x)
if self.start_embed is not None:
    embedded = embedded + self._slice(self.start_indicator, seq_len) * self.start_embed
```

**There are three terms, and the third is newer than this run.** The masked projection and the
per-channel announcement $W_m(m_t - \mathbf 1)$ are built together whenever $\max_c \delta_c > 0$.
The start-of-record term $\mathbb 1[\sum_c m_{t,c} = 0]\, e_{\mathrm{start}}$ is built only when
$\min_c \delta_c > 0$ as well: one learned $d_{\mathrm{model}}$-wide vector, added at every step at
which *no* channel of the stream has yet become a function of the recording, and absent as a
parameter otherwise.

That minimum was zero in every configuration this run saw, so no such vector existed here. It is not
zero now. In the causal cells the adapter is told $\delta^{\mathrm{adapter}}_c = W'_c + d_c$ — the
channel's warm-up plus its cross-channel alignment shift, combined in
`CausalWarmupInputs._build_adapter` rather than either term alone — and under the shipped
`causal_align_reference: target_max` the minimum of that sum is $80$ steps on **both** streams, where
the warm-up alone reaches $0$. ($80$ rather than the $91$ this section first carried: the shifts are
now scaled by the impulse-response centroid factor $\kappa = 0.875$ before quantisation, which
shortens every shift without moving the keep-index or the anchor floor. `DESIGN.md` §12 is the
record.) The start-of-record vector is therefore now constructed on the source
adapter *and* on the target adapter of this cell. The parameter arithmetic is the cleanest
confirmation: the alignment moves this cell's total by $-768$, which factorises exactly as four
channels lost from two $d_{\mathrm{model}}$-wide source linears ($-4 \times 128 \times 2$) plus two
start-of-record vectors ($+2 \times 128$), one per stream.

**What that does to the residue this section measures.** The $85.7\%$ above is defined by what
survives replacing source *values* with their mean, and every announcement term survives that
replacement by construction, being a function of $t$ alone. Under the alignment the source path
carries one more such term than it did when this number was measured, and carries it over a longer
leading region — the warm fraction of the source's (anchor, lag) grid falls from $0.943$ at the
unaligned floor $F = 133$ to $0.863$ at the aligned floor $F = 134$, both taken over the same $47$
kept channels and the same $\ell \le 90$. (The aligned figure was $0.828$ before the shifts were
scaled by $\kappa$; shorter shifts leave more of the grid warm.) So the residue re-measured on an
aligned run is not comparable to $85.7\%$ as
a level, and the ablation that would name the announcement's share has to remove both terms rather
than one; see §10 P1, arms D1 and D2.

The availability vector is a deterministic time-dependent staircase. It enters the posterior through
the source path but does not enter the prior through that same path. It can therefore create a
non-source-content difference between $q$ and $p$. The target adapter also has availability
information, but its target state is shared by both branches.

### 6.2 The learned source direction contains signal but is over-scaled on the probe

A post-hoc intervention decoded both branches at their means:

$$
z(\alpha)=\mu_p+\alpha(\mu_q-\mu_p).
$$

Gain is $D(\mu_p)-D(z(\alpha))$, so positive is helpful.

| $\alpha$ | Matched source | Random same-norm direction | Source-null component | Matched-minus-null component |
|---:|---:|---:|---:|---:|
| 0.0 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.1 | +5.28 | -0.41 | -1.82 | +5.68 |
| 0.2 | +8.71 | -1.59 | -6.12 | +9.26 |
| 0.3 | **+10.17** | -3.56 | -13.18 | +11.03 |
| 0.4 | +9.61 | -6.31 | -23.22 | **+11.24** |
| 0.5 | +7.07 | -9.84 | -36.40 | +10.06 |
| 0.7 | -3.50 | -19.23 | -72.30 | +4.05 |
| 1.0 | -30.91 | -39.16 | -145.65 | -12.44 |

This is useful mechanistic evidence:

- a small displacement in the learned direction improves this external cohort;
- random directions of the same norm do not;
- the trained full displacement overshoots the useful region;
- removing the variation-independent displacement improves the curve.

It is **not** a production result. The cohort is different, $\alpha$ was selected after observing its
outcomes, and no per-recording interval was calculated. It also does not imply `kld_beta: 3.0`.
Changing $\beta$ changes both latent means, both variances, encoders, decoder, and their co-adaptation;
it is not equivalent to multiplying a frozen displacement by $1/3$.

### 6.3 Lag attention is not the obvious failure

`source_lag_warmth_frac_st` is 1.000 and the phase counterpart is about 0.980. Probe attention heads
are non-degenerate: two emphasize recent lags while other heads assign substantial mass farther back.
This argues against `lag_floor` or source warmup validity as the first experiment. Attention weights
remain model attributions over stored-feature time, not identified physiological delays.

---

## 7. Latent geometry and variance pinning

### 7.1 What collapsed and what did not

Late validation metrics show:

- `logvar_prior_floor_frac` about 0.955;
- `kld_active_frac` about 0.109, or about 7 active dimensions by that threshold;
- checkpoint probes find 10 of 64 dimensions above 0.01 nats;
- two dimensions contribute about 82% of total KL.

This is not complete posterior collapse in the canonical VAE sense, because $q$ is not equal to $p$
and the decoder uses a small active subset. The more accurate diagnosis is **dimensional collapse plus
variance pinning**.

The distinction matters. Work on posterior collapse defines the degenerate case as an approximate
posterior that mimics the prior and a decoder that ignores $z$. Here the problem is concentrated use
and badly behaved scales, not total absence of source-conditioned latent activity.

### 7.2 Dimension-level probe

The dominant external-probe dimensions are:

| Dimension | $\log\sigma_p^2$ | $\log\sigma_q^2$ | RMS $\Delta\mu$ | KL |
|---:|---:|---:|---:|---:|
| 30 | -0.845 | -4.176 | 0.790 | 1.629 |
| 15 | -0.628 | -3.770 | 0.890 | 1.601 |
| 47 | -4.315 | -4.999 | 0.092 | 0.316 |
| 32 | -4.722 | -5.000 | 0.040 | 0.094 |
| 37 | -4.798 | -5.000 | 0.027 | 0.052 |

Most other dimensions sit near the lower log-variance bound. The prior is unusually wide precisely
on the two dimensions with the largest source displacement, while the posterior remains narrow.
This explains why sampling the prior is much more costly than decoding its mean.

`base_decode: mean` removes base-reconstruction pressure to make the prior samples themselves useful.
It is therefore plausible that the objective permits this configuration. The evidence shows an
association and an exploitable objective path; it does not, by itself, prove that the model
deliberately “games” the rate or that changing `base_decode` alone will repair training.

### 7.3 Training dynamics

The posterior log variance reaches about -4.76 by epoch 1. The prior then follows during the
$\beta$ warmup: prior-floor fraction is about 0.55 by epoch 16, 0.92 by epoch 25, and 0.97 by epoch
40. This is consistent with the KL gradient pulling the prior toward a much narrower posterior:

$$
\frac{\partial\mathrm{KL}}{\partial \ell_p}
=\frac{1}{2}\left[1-(e^{\ell_q}+\Delta\mu^2)e^{-\ell_p}\right].
$$

`posterior_logvar_mode: independent` removes parameter sharing between prior and posterior variance
heads, but it cannot remove this coupling through the KL itself. The prior-scale anchor with
`beta_prior: 0.1` is not sufficient to prevent floor pinning in this run.

### 7.4 Implication for hyperparameters

Do not tune $\beta$ alone against `pred_gap`. A higher $\beta$ may shrink the source displacement, but it
may also reduce the active dimension count or pull the prior more strongly toward the narrow
posterior. A defensible sweep must jointly monitor:

- mean-path and MC predictive gaps;
- active dimensions and per-dimension KL;
- prior and posterior floor fractions;
- source-null KL and matched-minus-null KL;
- base and full NLL separately; and
- per-recording train/validation intervals.

`free_bits` can prevent small dimensions from being penalized, but it is not automatically desirable:
it can also subsidize variation-independent source-path KL. It should only be tested after the null
component is controlled.

---

## 8. Target-channel behavior and capacity

### 8.1 The model suppresses fast channels

On the external probe, pooled over 15 forecast steps:

| Group | Channels | Base NLL | Model $R^2$ |
|---|---:|---:|---:|
| Live | 40 | -28.53 | 0.763 |
| Dead by aggregate $R^2$ | 58 | 427.55 | 0.001 |
| Scattering block | 32 | 58.08 | 0.732 |
| Phase-harmonic block | 66 | 340.94 | 0.128 |

The naming “dead” means the forecast mean has approximately no aggregate predictive $R^2$ on this
probe. It does not mean the raw channel is constant or scientifically irrelevant.

Target-adapter column norms are strongly suppressed for roughly 56 phase-harmonic channels. Linear
probes also show that their instantaneous values are mostly absent immediately after the target
adapter, before the transformer and latent bottleneck. This is learned input selection, not direct
evidence that the transformer or `d_z` physically cannot encode them.

### 8.2 The objective makes suppression rational

The dead channels have high near-term persistence and then decorrelate quickly:

| Forecast step $\tau$ | 0 | 1 | 2 | 3 | 4 | 7 | 14 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence $R^2$ | 0.877 | 0.582 | 0.247 | near 0 | negative | negative | negative |

The loss weights all 15 horizon steps and all retained channels uniformly. For a channel whose
current value helps only the first 2--3 steps, a shared latent/decoder can reduce average Gaussian
NLL by predicting conditional variance rather than preserving the level. This is an objective and
target-definition issue before it is a capacity issue.

An oracle that selects the better of the trained model and a persistence forecast separately for
each channel and horizon step would reduce optimized NLL by 151.43 nats/anchor on the probe, with
75.11 at the first step, 34.08 at the second, and 16.85 at the third. This is an **upper bound**, not
the expected gain of adding one fixed persistence skip. The selection uses outcomes post hoc.

> **What the move to $H = 30$ does to this finding.** The table above is measured over $\tau \in
> [0, 15)$ and persistence $R^2$ is already negative from $\tau = 4$. Doubling the horizon appends
> fifteen further steps, every one of them deeper into that regime, and the objective weights them
> equally with $\tau = 0$. So the incentive this section identifies — suppress a fast channel at the
> input adapter and forecast its conditional variance instead, because preserving its level pays for
> only two or three steps out of fifteen — is *strengthened*, not relieved: it now pays for two or
> three steps out of thirty. Expect the dead-channel fraction to rise rather than fall, and read
> `pred_gap` at the new horizon knowing that. The remedy §8.4 proposes — a horizon weighting
> $w_\tau$ that decays, or a target-only persistence residual — is correspondingly more
> load-bearing at this horizon than it was at the one measured here, and neither ships.

> **The measurement that has since shipped.** The question this section keeps circling — is a good
> score on a channel a forecast, or the model inverting its own differently-delayed history — now has
> a tracked readout rather than only an argument. `pred_gap_novel_lo`, `pred_gap_novel_mid` and
> `pred_gap_novel_hi` are registered in `teb_vae/lag_attn_cfs/trainer.py` and inherited unchanged by
> this cell, whose tracked surface is asserted identical to the conv-LSTM causal cell's. They split
> the same block score by the **novelty fraction** $\nu_c$ — the share of a scored coefficient drawn
> from raw samples the anchor has not seen, measured on the composed kernel at the shipped horizon
> and read off the shard rather than assumed. Three properties make the split worth reading rather
> than guessing at. It is not the warm-up split renamed: the novelty partition and the warm-up
> partition disagree on $67$ of the $98$ kept channels, so `pred_gap_novel_hi` and `pred_gap_warm_hi`
> are answering different questions about different sets. The tertiles are $33/33/32$ channels, so no
> group is a rounding artefact of a lopsided cut. And $\nu$ over the kept set runs $0.026$ to
> $1.000$, so the kept set really does span from a channel whose scored value is almost entirely
> already-seen history to one whose scored value lies almost entirely ahead of the anchor. Both
> splits recompose to `pred_gap_st + pred_gap_ph`, so reading either costs nothing and neither
> replaces the pooled number. What ships is the **measurement**; the remedies named above still do
> not.

### 8.3 Why low capacity is not established

Additional checkpoint probes give:

- latent PCA participation ratio: 23.7;
- about 40 of 64 PCs explain 90% of latent-mean variance;
- target encoder state participation ratio: 47.5 of 128;
- linear $R^2$ from $\mu_p$ to target state: 0.783;
- linear $R^2$ from target state to $\mu_p$: 0.985.

These statistics do not prove that the representation is optimal, but they do not look like a hard
64-dimensional saturation. More importantly, the fast-channel information is suppressed before the
latent. Widening the latent without changing the loss can let the optimizer make the same selection
in a larger space.

### 8.4 Better architectural test

If near-term accuracy on these channels matters scientifically, test a target-only residual path:

$$
\mu_{t,\tau}=w_\tau y_t + f_\theta(z_t)_\tau,
$$

where $w_\tau$ is fixed or strongly regularized and decays with horizon. The source must still reach
the forecast only through $z$, preserving the source-coupling constraint. This directly tests whether
the model is wasting latent capacity on target levels.

Other defensible target experiments are:

- weight early horizons more heavily if early prediction is the scientific objective;
- report per-channel and per-horizon normalized NLL rather than only a block sum;
- exclude channels only after a domain decision that they are not meaningful targets;
- forecast residuals relative to persistence; and
- compare the current uniform objective to a predictability-aware weighting fixed before training.

---

## 9. Ranked root-cause assessment

### 9.1 High confidence

1. **Source-specific generalization failure:** directly visible in the branch gaps.
2. **Estimator asymmetry:** certain from `base_decode: mean` and confirmed numerically.
3. **Training duration/checkpoint mismatch:** certain; the run continues well beyond all relevant
   NLL minima.
4. **Latent variance pinning:** present in both run history and checkpoint inference.
5. **Target-objective mismatch:** supported by channel weights, persistence, and horizon behavior.

### 9.2 Likely but not yet isolated

1. **Availability/bias shortcut in the source path:** source-null KL is large, but an ablation of
   the source adapter's announcement terms is needed to name the clock's exact contribution — both
   of them now, since the alignment gave the source adapter a start-of-record vector as well (§6.1;
   §10 P1 arms D1 and D2).

   A second probe of the same hazard already ships, and it is a configuration rather than a code
   change. The cross-channel alignment is shipped as `causal_align_reference: target_max` with
   `null`, the unaligned arm, as its named comparison, and the pair is **pre-registered** against
   exactly this hazard. Alignment discards recency — the freshest uterine activity any source
   channel reports moves from $13.3$ s before the anchor onto the common aligned clock, which in
   *realised* delay is $\kappa\,\tau_{\mathrm{ref}} = 0.875 \times 402.1604 \approx 351.9$ s (the
   $402.2$ s the configuration names is the reference channel's *reported* envelope mean) — and
   it equalises the per-channel delay spread that an availability clock would be reading. So it *should not* improve
   `pred_gap`, and it *should* improve lag-profile concentration if there is a real lag to find. An
   aligned arm that improved `pred_gap` while leaving concentration flat would be evidence that the
   encoder was using the delay spread as a clock, which is a named revert trigger.

   **This is a pre-registration and not a result.** The comparison has not been run, no fit exists
   on either arm, and nothing here reports an outcome for either. It is written down now precisely
   so that the reading is fixed before any number is in view, which is the only condition under
   which either outcome would mean anything.
2. **Insufficient source-specific regularization:** the posterior's attended-source dropout is zero
   when `source_dropout: null`, although the source adapter and encoder inherit general dropout 0.1.
3. **Recording-level distribution shift or too few independent recordings:** highly plausible, but
   impossible to quantify without the production shards and GUID metadata.
4. **Mis-calibrated source displacement:** clearly present on the external cohort, not yet confirmed
   on the production split.

### 9.3 Not supported as a primary cause

1. Too few total parameters.
2. `d_z: 64` as a demonstrated hard bottleneck.
3. Transformer depth or attention-head count.
4. `lag_floor` or causal warmup coverage.
5. Gradient clipping, spike handling, DDP, precision, or `torch.compile`.
6. Decoder observation-variance floor: standardized residual checks are close to calibrated overall.

---

## 10. Recommended next work

The experiments below are ordered so that measurement defects and data questions are resolved before
another expensive architecture sweep.

### P0: evaluate the existing checkpoint correctly

1. Bind `eval/configs/eval_overrides.yaml` to the exact production train and validation shards.
2. Evaluate the epoch-336 checkpoint per recording, not only per window.
3. Report mean-path, paired-sample, and MC marginal predictive gaps separately.
4. Bootstrap recordings for confidence intervals; never bootstrap overlapping windows as if they
   were independent.
5. Run source-null, source-permutation, and source-availability ablations on both splits.
6. Audit train/validation GUID intersection, duplicate windows, cohort labels, recording duration,
   source missingness, and source/target normalization statistics.

This is the highest-value step because it distinguishes ordinary overfit from split shift and tells
whether the external-cohort shrinkage result transfers.

### P1: fix training control and source regularization

Use early stopping and keep multiple checkpoint criteria:

- checkpoint minimum `val/nll_full_block` for best conditioned forecasting;
- checkpoint minimum `val/total_loss` for the composite objective;
- record, but do not optimize directly against, noisy single-sample `val/pred_gap`;
- use patience around 40--60 validation epochs as an initial value.

Run a compact, seeded ablation matrix:

| Arm | Change | Question |
|---|---|---|
| A | baseline reproduction with early stopping | Is the result reproducible? |
| B | `source_dropout: 0.2` | Does source-specific regularization reduce the excess gap? |
| C | `source_dropout: 0.3` | Is stronger source regularization beneficial or destructive? |
| D1 | remove source `mask_proj`, keep `start_embed` | How much of null KL is the per-channel availability announcement? |
| D2 | remove source `start_embed`, keep `mask_proj` | How much of it is the single start-of-record vector? |
| E | explicit source-displacement gain/penalty | Does direct shrinkage transfer better than changing the whole VAE rate? |
| F | paired `base_decode` evaluation/training arm | Does prior sampling behavior improve without damaging mean forecasts? |

**Why arm D became two arms.** As written for this run it was one — remove only source
`mask_proj` — and that was the whole announcement then, because the source adapter carried no other
availability-shaped parameter. It does now (§6.1): under the shipped alignment a learned
$d_{\mathrm{model}}$-wide start-of-record vector is added at every step at which no source channel is
yet available, so removing `mask_proj` alone leaves part of the announcement standing and silently
attributes its share to something else. Splitting is chosen over widening the single arm to "remove
both" because the two terms are different objects and a widened arm cannot tell them apart: one is a
$C_{\mathrm{keep}} \to d_{\mathrm{model}}$ projection of a per-channel staircase that keeps moving
throughout the leading region, the other is one vector on a single indicator that switches once.
Their shares can differ by an order of magnitude, and so do the remedies. Nothing is lost by
separating them — run both on the same seeds and their sum is the widened arm — while a single number
cannot be taken back apart afterwards.

Each arm should use at least three seeds if the production cost permits. Select using
recording-level validation likelihood and report source gain as a secondary metric.

### P2: sweep the rate/variance controls jointly

After the null component is controlled, test a small factorial design rather than assuming
`kld_beta: 3.0`:

- final $\beta$ in $\{0.5,1,2,3\}$;
- `beta_prior` in a small range around $\{0.1,0.3,1.0\}$ or replace it with an explicit, inspected
  prior-variance constraint;
- optionally `free_bits` in $\{0,0.01,0.05\}$ nats/dimension only if null KL is no longer dominant.

Reject arms that improve `pred_gap` by destroying the base forecast, collapsing active dimensions,
or increasing source-null KL.

Read the novelty split beside the pooled number in every arm. `pred_gap_novel_lo`, `_mid` and `_hi`
ship in this cell (§8.2) and cost nothing to log, and they separate the two ways an arm can move
`pred_gap` in the favourable direction: a gain concentrated in `pred_gap_novel_hi` is a gain on the
coefficients genuinely ahead of the anchor, while one concentrated in `pred_gap_novel_lo` is the
model becoming better at reconstructing history it has already seen through a differently-delayed
channel. Those are not the same result, and the pooled score cannot tell them apart. Read them
*beside* `pred_gap_warm_lo/mid/hi` rather than instead of them — the two partitions disagree on most
of the kept set, so an arm can move one and leave the other where it was.

### P3: address the target objective

Compare:

1. current uniform 15-step objective;
2. a fixed persistence-residual decoder;
3. early-horizon weighting fixed from the scientific use case; and
4. a target set justified by domain relevance and held-out predictability.

Only after these tests should capacity be swept. If it remains useful, test `d_z` in
$\{32,64,128\}$ while holding everything else fixed. The smaller arm is important: if 32 performs
similarly, it further refutes capacity limitation; if 128 helps only after the target fix, it shows
that the original objective, not raw width, was masking the need.

### P4: compare encoder families

Train `lag_attn_cfs` and `lag_attn_transformer_cfs` on the identical GUID split, seeds, objective,
and evaluation pipeline. Compare:

- base and full per-recording likelihood;
- source excess generalization gap;
- source-null KL;
- active latent dimensions;
- channel/horizon errors; and
- wall-clock and memory cost.

Without this paired run, attributing the current result to the transformer is not defensible.

---

## 11. Code and documentation issues found

1. **Training-history semantics are easy to misread.** `MetricsLoggingCallback` captures the bare
   train key before its epoch aggregate is exposed. The CSV writer should explicitly collect
   `train/..._epoch` after aggregation, or label the current fields as sampled steps. `RESULTS.md`
   should not call them epoch means.

2. **Warm-tertile labels were documented backward — resolved.** `_resolve_warm_tertiles` sorts
   ascending warm-up and maps rank $0$ to `lo`, so `warm_lo` is the low-warmup/fast third and not
   the slowest third, while the docstring at the time of this review said the opposite. The source
   now states the implemented convention directly: `_resolve_warm_tertiles` in
   `teb_vae/lag_attn_cfs/nets/causal_feature_target.py` opens *"Assign each kept target channel to a
   warm-up tertile: group $0$ is the shortest wait."* The item is kept as a closed one rather than
   deleted, so that a reader of any earlier note written under the reversed reading can see that it
   was reversed.

3. **The source-null name is stronger than its intervention.** It removes source variation but
   retains availability, network biases, and the flat-input encoder response. Documentation should
   call this out wherever `kld_source_null` is interpreted.

4. **The guarantee attached to `posterior_logvar_mode: independent` is too broad.** It removes a
   shared-parameter route but not the KL-gradient route by which the prior can follow the posterior
   variance downward.

5. **`perm_forward_outputs` needs explicit anchors for tiled models.** Omitting them can create an
   anchor-shape mismatch; the helper should require anchors or validate the returned time axis.

6. **The registered evaluation config is not runnable as shipped.** Its data path still contains
   `REPOINT_ME`, so the run has no completed reproducible production evaluation artifact.

---

## 12. Literature context

The literature supports the proposed tests but does not replace model-specific evidence.

- He et al., [Lagging Inference Networks and Posterior Collapse in Variational
  Autoencoders](https://openreview.net/pdf?id=rylDfnCqF7), define posterior collapse through an
  approximate posterior that mimics the prior and emphasize early optimization dynamics. This is
  why the present run is described more narrowly as dimensional/variance collapse.
- Dai et al., [The Usual Suspects? Reassessing Blame for VAE Posterior
  Collapse](https://proceedings.mlr.press/v119/dai20c.html), show that deep VAE loss surfaces can
  contain information-discarding local minima. That is compatible with the adapter suppressing fast
  channels and cautions against attributing everything to $\beta$.
- Higgins et al., [$\beta$-VAE](https://openreview.net/forum?id=Sy2fzU9gl), establish $\beta$ as a
  tradeoff between reconstruction and latent-channel constraints. They do not imply that a frozen
  displacement scale maps linearly to a new training $\beta$.
- Srivastava et al., [Dropout: A Simple Way to Prevent Neural Networks from
  Overfitting](https://www.jmlr.org/papers/v15/srivastava14a.html), motivate dropout against
  co-adaptation. In this model, applying it specifically to the attended source summary is a direct
  test of source-path overfit rather than a generic recommendation.
- Burda et al., [Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519), motivate
  multi-sample bounds and likelihood estimates. This supports MC predictive evaluation instead of
  comparing one deterministic decode to one stochastic decode.
- Dieng et al., [Avoiding Latent Variable Collapse with Generative Skip
  Models](https://proceedings.mlr.press/v89/dieng19a.html), show that skip structure can change latent
  use. The proposed target persistence residual is more limited: it must not create a source bypass,
  and it needs its own ablation because the paper's setting is not this conditional forecast model.

---

## 13. Reproduction notes

The checkpoint and metric probes were run with the project's `.venv` and repository code. The
checkpoint was loaded strictly, not with permissive missing/unexpected keys. The external probe used
the causal HIE shard and regenerated causal normalization statistics; the similarly named top-level
two-sided statistics file is not schema-compatible.

The exploratory scripts currently reside outside the package in the session scratch area. They are
not production evaluation entry points and should not be cited as permanent reproducibility
artifacts. Before publishing results, promote the needed diagnostics into `teb_vae/lag_attn_cfs/eval`
or the transformer package, add the repository's `RUN_ARGS` convention, tests, resolved input paths,
and machine-readable per-recording output.

---

## 14. Final diagnosis

The training run is numerically and geometrically valid. Its weak validation performance is not best
explained by a transformer that is too small. The dominant observed failure is that the
source-conditioned branch learns substantially more training advantage than it can transfer. That
failure is compounded by a non-equivalent `pred_gap` estimator and by a KL readout that largely
survives removal of source variation.

Separately, the model concentrates latent use in a small number of dimensions and suppresses many
rapidly decorrelating target features at its first adapter. The latter behavior is consistent with
the uniformly weighted 15-step Gaussian objective: preserving those values is useful for only a few
near horizons, so the optimizer chooses variance forecasts instead. This is an objective-design
problem, not demonstrated proof of a 64-dimensional capacity ceiling.

The next defensible action is not a larger transformer. It is a production-shard, per-recording
evaluation of the existing checkpoint with equivalent predictive estimators and explicit source-null
ablations, followed by source-specific regularization and early stopping. Only after the split,
metric, source shortcut, and target-objective questions are controlled should $\beta$, latent width, or
transformer depth be interpreted as model limitations.
