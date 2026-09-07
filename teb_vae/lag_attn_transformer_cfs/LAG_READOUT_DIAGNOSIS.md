# Why the lag readout peaks at zero — diagnosis of the `argmax_lag` failure

> **Which run this documents.** The $H = 30$ evaluation of
> `/data/deid/isilon/MS_model/q3_2026/lag_attn_transformer_cfs/2026-08-20--[06-49]-lag_attn_trf_cfs_baseline`,
> whose `eval_results/summary.json` was written 2026-08-24 by `teb_vae.lag_attn_cfs.eval.run`
> through this package's binding. It is a companion to `DIAGNOSIS.md`, which reviews the earlier
> $H = 15$ training run; nothing here restates that document, and the two failures are different:
> that one is about the source pathway not generalising, this one is about the **lag attribution
> pinning at $\ell = 0$** and what that number can and cannot mean. Written 2026-08-26.
>
> **Revised the same day, after the run's arm was resolved.** The first draft flagged a
> discrepancy — the eval reported $\delta = 0$ while the shipped config says
> `causal_align_reference: target_max` — and left it open. The operator has since confirmed that
> the training machine's config carried **`causal_align_reference: null`**: this run is the
> **unaligned arm**, the eval's delay reporting is correct, and §2 records the resolution. §3 is
> ordered accordingly — the unaligned geometry first, as the operative analysis for this run, and
> the aligned geometry second, as the prediction for the arm that has not yet been trained.
>
> **Extended the same day, after the scope was widened.** The first draft kept its architectural
> proposals deliberately conservative, because this cell's documented contract fixes it as one
> corner of the encoder-by-target square and an edited cell reads on neither edge. The operator
> has since ruled that **no trained run is final and the architecture may change**. §10 now
> carries the full architecture revision program; §9 P3 summarises it and points there; §11
> records the one structural cost the freedom has (the square) and how to pay it deliberately.

The evaluation finished cleanly and then failed one sanity check:

```
sanity check FAILED [argmax_lag]: the argmax lag is 0, so the attribution never
looks back and the lag window is inert
```

with, in the headline block:

| Metric | Value |
|---|---:|
| `kl_argmax_lag_step` | 0 |
| `kl_argmax_lag_step_support_corrected` | 0 |
| `kl_lag_compensated_seconds` | 0 |
| `attention_argmax_lag_step` | 0 |
| `attention_argmax_lag_step_untruncated` | 0 |
| `attention_entropy_nats` | 2.664 (attainable $\ln 91 = 4.511$) |

The question investigated: is this the data, the model, or something else — and what should
change.

---

## Executive verdict

**The $\mathrm{argmax} = 0$ result is overdetermined.** It is not a single bug, and it is not
evidence that the model "failed to find the lag". Four independent causes each predict an argmax
of zero on their own, and all four are active in this run:

| # | Cause | Kind | Confidence |
|---|---|---|---|
| A | The run is the **unaligned arm** (`causal_align_reference: null` on the training machine, confirmed by the operator). `source_delay_steps = 0` is correct reporting, the eval's lag axis is right — and the repo's shipped `default.yaml`, which says `target_max`, no longer describes this run. The pre-registered aligned/unaligned pair is half-run. | run identity — **resolved** | certain |
| B | On the unaligned arm the "correct" lag is a channel-**pair** quantity smeared over a $\approx 1167$ s bias range, censored at both edges of the $91$-lag window, while the freshest source channel is only $13.3$ s stale at $\ell = 0$ — so reading lag $0$ is the **information-optimal** strategy, not an inert one. On the aligned arm (§3.2) the pooled argmax of even a *perfect* lag detector is also predicted at $0$, by horizon censoring. | data / representation | certain — arithmetic on the model's own lag identity |
| C | The lag attention's keys and values are **deep causal encoder states**, so for $\approx 76\%$ of the lag window the lag-0 state already contains everything the lag-$\ell$ state knows; the lag axis is informationally degenerate by construction | model architecture | certain from the code; magnitude from the shipped receptive field |
| D | The KL being attributed is $67.5\%$ availability **clock** and only $0.160$ nats/anchor of clock-exceeding content, concentrated in $2$ latent dimensions ($95.4\%$ in one), with the prior variance pinned — so the attribution weighs a deterministic function of $t$ that is readable at every lag, and the `alibi_decay` initialisation resolves that indifference to $\ell = 0$ and is never overcome | training signal | strong, from the run's own controls |
| E | `check_argmax_lag` treats a near-edge pin as "the machinery is inert". In this cell the near edge is a **censoring edge exactly like the far edge** (finding B), which the check's *other* arm correctly treats as an artifact. The FAIL is right to block quoting a lag; its stated cause is wrong for this domain | evaluation check | certain from the check's source |

The machinery itself is intact: the two structural identities pass on this run
(`lag_map_sums_to_kl`, `per_head_kl_sums_to_kl`, both to $10^{-4}$ nats on the worst anchor),
`source_margin_positive` passes with $61.7$ nats — the pathway reads *this* recording's source
rather than any source — and the orientation of the lag axis is pinned by the architecture
parent's suite. The readout is empty because of geometry and signal, not breakage.

**Nothing needs rebuilding from scratch — but the model does need deliberate revision.** The
grid, the eval pipeline and its controls are precisely what made this diagnosable, and none of
them is implicated. What the findings do implicate is three specific pieces of the model — the
lag-attention memory (C), the asymmetric clock conditioning (D), and the horizon-uniform decoding
the earlier diagnosis already indicted — and the operator has ruled the architecture open, so §10
specifies the revision rather than hedging it. One check needs re-scoping (E), and the deeper
blocker remains the one `DIAGNOSIS.md` already names: the source pathway carries almost nothing
that generalises, so no lag mechanism — current or revised — has a signal to learn from until the
training-control program runs with it.

---

## 1. What the failing number is, mechanically

The chain, verified in source:

1. The model computes the attribution
   $\widetilde K_{t,\ell} = \sum_m K_t^{(m)}\,\alpha^{(m)}_{t,\ell}$ — per-head KL times that
   head's attention distribution over lags, head-structured so the split is an additive
   decomposition (`teb_vae/lag_attn/nets/heads.py::TEAnalysisHead`; invoked with
   `head_structured=True` in `teb_vae/lag_attn_cfs/nets/causal_inputs.py::forward`).
2. The eval pools it over the dense anchor support (the $136$ anchors $t \in [134, 270)$), per
   recording and then across the $1{,}633$ recordings, into the three per-lag profiles
   (`teb_vae/lag_attn_cfs/eval/metrics.py::lag_profiles`, `lag_summary`). At this geometry the
   raw, support-corrected and untruncated profiles coincide (every lag is valid at every anchor:
   $F = 134 > L - 1 = 90$), which the run confirms — all argmaxes agree at $0$.
3. `check_argmax_lag` (`teb_vae/lag_attn_cfs/eval/report_seam.py:640`) fails the run when the
   pooled argmax sits at either end of the attainable window. It sat at the near end.

Two properties of the attention module matter later:

- **Lag 0 is the anchor's own step**: keys are $H^u_{t-\ell}$ for $\ell = 0 \dots 90$, at
  $\Delta = 4$ s per step, so the window spans $0$–$360$ s
  (`teb_vae/lag_attn/nets/attention.py`, `max_lag: 90`).
- The shipped normaliser is `entmax15` (exact zeros possible) and the shipped score-bias init is
  `alibi_decay`: every head starts with a negative slope in $\ell$ — biased toward lag 0, made to
  *earn* a long-lag reading (`configs/default.yaml`; `attention.py::alibi_slopes`).

---

## 2. Finding A — the run is the unaligned arm (resolved)

The summary prints `kl_lag_compensated_seconds = 0`. That value is
$\Delta(\ell + \delta) = 4 \cdot (0 + \delta)$ with $\delta$ read from
`task.orig_model.source_delay_steps` (`teb_vae/lag_attn_cfs/eval/run.py:1241`), which is
`source_gate.max_delay` (`teb_vae/lag_attn_transformer_rws/nets/model.py:640`). Under
`causal_align_reference: target_max` the fastest kept source channel ($\tau_c = 13.3$ s) would be
shifted by $(402.16 - 13.3)/4 \approx 97$ steps, so an aligned checkpoint would report
$\delta \approx 97$ and `kl_lag_compensated_seconds` $\approx 388$ — not $0$.

**Resolution (2026-08-26, confirmed by the operator):** the training machine's config carried
`causal_align_reference: null`. The checkpoint is the **unaligned arm**, `source_delay_steps = 0`
is the correct value for it, and every lag-resolved artifact of this evaluation is on the right
axis. There is no delay-reporting bug. The eval summary is internally consistent with that:
`source_lag_warmth_frac_st = 1.000` / `_ph = 0.995` match the unaligned warm-up geometry, and the
unaligned build constructs no `start_embed` on either adapter
(`tests/test_construct.py::test_omitting_the_alignment_keywords_builds_todays_model_bitwise`),
which matters for §5's reading of the clock.

Three consequences follow, and each lands somewhere in this document:

1. **Every number in this summary belongs to the unaligned arm's column.** The run stamp
   (2026-08-20 06:49) postdates the alignment commit (`800effd`, 2026-08-19 22:52) but the
   production box's config did not carry it, so the repo's shipped `default.yaml` — which says
   `target_max` — no longer describes the run directory named "baseline". The run's own
   `resolved_config.yaml` is the authority, as it is designed to be; the hazard is a *reader*
   pairing this summary with the shipped config. §9 P0 carries the operational fix.
2. **The pre-registered aligned/unaligned pair is now half-run.** This evaluation is the
   unaligned half, at $H = 30$, with full per-recording intervals; the aligned arm has never been
   trained. The pair was pre-registered in `default.yaml` against exactly the availability-clock
   hazard, with the reading fixed in advance: alignment should *not* improve `pred_gap` and
   *should* improve lag-profile concentration if there is a real lag. §3.2 refines what
   "concentration" can mean there — the pooled argmax is not the quantity that can move.
3. **The operative geometry for this run is §3.1**, the unaligned one — the per-pair bias and
   the retained recency — not the aligned censoring arithmetic, which becomes the *prediction*
   for the missing arm.

---

## 3. Finding B — the geometry predicts argmax 0, on both arms, for different reasons

The lag axis is stored-coefficient time, and the model's own arithmetic
(`teb_vae/lag_attn/nets/lag_report.py::physical_lag_seconds`) gives the exact identity between a
lag index and the physical lead time between the epochs two coefficients summarise:

$$
\tau^{\mathrm{phys}}_{\ell,h} \;=\; \Delta\,(\ell + 1 + h)
\;+\; \kappa\bigl(\tau^{u}_{c} - \tau^{y}_{c'}\bigr),
$$

with $\Delta = 4$ s, $h \in [0, 29]$ the horizon step, $\kappa = 0.875$ the approximate
content-delay convention, **no dataset-shift term** (the stored timeline is canonical; the
$-20$ s this identity once subtracted is superseded), and — on an unaligned run — the bias term indexed by the channel
**pair**, not by a per-stream constant.

### 3.1 This run — the unaligned arm

Unaligned, the composed group delays retain their natural spread: $13.3$–$791.0$ s across the
source channels (all $51$ kept — the warm-up budget gates the target stream only), and up to
$402.16$ s across the $98$ kept target channels. The pair bias
$\tau^u_c - \tau^y_{c'}$ therefore spans roughly $[-389, +778]$ s — the $\approx 1167$ s
pair-dependent range this package's own `DESIGN.md` §14 lean-limit names. Three regimes, and none
of them puts a coherent peak inside the window:

- **Fast source against slow target** ($\tau^u \approx 13$ s, $\tau^y \approx 402$ s): a
  $\Delta_{\mathrm{physio}} \approx 40$ s physiological delay needs
  $4(\ell + 1 + h) \approx 449$ s, i.e. $\ell \approx 111 - h$ — **beyond the far edge of the
  $91$-lag window for the entire horizon.**
- **Slow source against fast target** ($\tau^u - \tau^y \gtrsim +389$ s): the required
  $4(\ell + 1 + h)$ is negative — **censored below $\ell = 0$ for every $(\ell, h)$.**
- **Delay-matched pairs** ($\tau^u \approx \tau^y$): behave like the aligned case of §3.2 — a
  thin ramp out to $\ell \approx 14$ plus a censored spike at $0$.

The pooled attribution is a single profile over $\ell$; a mixture of mutually inconsistent
per-pair optima censored at both edges has no meaningful pooled peak, and its argmax is decided
by whatever else shapes the profile — which findings C and D supply.

**And on this arm, lag 0 is genuinely where the information is.** Unaligned, recency is *not*
discarded: the fastest source channel reports uterine activity only $13.3$ s before the anchor,
and a one-sided coefficient is itself a causal integral of its past — the newest coefficient of
each channel summarises everything its older coefficients report, plus more. The
information-optimal strategy for the posterior is therefore to read the freshest source state and
let the causal filters' own memory do the "looking back". An argmax at $0$ on this arm is the
*rational* read, not an inert mechanism — which is precisely why it cannot be interpreted as "no
lagged coupling exists".

### 3.2 The aligned arm — the prediction for the run not yet made

Under `target_max` both streams sit on the same $402.16$ s clock, so the pair bias collapses to
$\tau^{u}_{\mathrm{ref}} - \tau^{y}_{\mathrm{ref}} = 0$ and

$$\tau^{\mathrm{phys}}_{\ell,h} = 4(\ell + 1 + h) - 20\ \mathrm{s}.$$

Take a genuine UA$\to$FHR deceleration latency of $\Delta_{\mathrm{physio}} \approx 40$ s (late
decelerations run roughly $20$–$60$ s). Solving $\tau^{\mathrm{phys}} = \Delta_{\mathrm{physio}}$
for the informative lag:

$$\ell^\*(h) \;=\; \frac{\Delta_{\mathrm{physio}} + 20}{4} - 1 - h \;=\; 14 - h .$$

- Only $h \le 14$ admits a non-negative lag at all.
- The **entire second half of the horizon needs $\ell < 0$** — source content that lies in the
  *future* of the anchor — and is censored at $\ell = 0$.

The KL and the attention are per-anchor quantities: $z_t$ conditions the whole $30$-step block,
so the profile pools over $h$. The pooled ideal profile of a *perfect* lag detector at this
geometry is therefore a censored spike at $\ell = 0$ (all the $h > 14$ mass) plus a thin ramp out
to $\ell \approx 14$ — **its argmax is also 0.** On top of that, the intra-band group-delay
dispersion — $\pm 16\%$ of $402$ s $\approx \pm 64$ s $\approx \pm 16$ lag steps, this package's
own lean-limit in `DESIGN.md` §14 — smears the ramp into the spike. And the `target_max`
reference discards all recency: the freshest uterine activity any aligned channel reports moves
from $13.3$ s to $402.2$ s before the anchor, so the contraction that will shape the next two
minutes of FHR is largely not in the observable window at any lag.

**Consequence for the pre-registered pair.** "Alignment should improve lag-profile concentration
if there is a real lag" cannot be read off the pooled argmax — that is predicted to sit at $0$ on
*both* arms. The reading must be made on the shape vocabulary (peak width, mass above half peak,
secondary peaks, degeneracy — `lag_shape.py`) and on the **per-head** profiles, where a
concentration change is actually expressible. §9 P0 restates this as the pair's protocol.

**Conclusion of finding B.** In the causal-feature domain, the pooled stored-coefficient lag
argmax cannot land at a physiological delay at this geometry — smeared and edge-censored without
alignment, near-edge-censored with it. That is a property of the transform, the reference choice
and the horizon pooling — not of the encoder, and not of this training run.

---

## 4. Finding C — the lag axis is informationally degenerate by construction

The lag attention's keys and values are **causal encoder states**, not local source measurements:
`causal_inputs.py::forward` hands `h_u = source_encoder(source_adapter(source))` to `lag_attn`.
$H^u_{t-\ell}$ is a function of $U_{\le t-\ell}$ — a strict subset of what $H^u_t$ sees. By the
data-processing inequality, any information available at lag $\ell$ within the encoder's
receptive field is also available at lag $0$.

The shipped source-state receptive field:

| Component | Config | Reach |
|---|---|---:|
| conv stem | kernels $[5, 9]$, dilations $[1, 2]$, causal | $(5-1)\cdot 1 + (9-1)\cdot 2 = 20$ steps |
| attention | `source_attention_blocks: 3` $\times$ band $W_U = 16$ | $48$ steps |
| **total** | | $\approx 68$ steps $\approx 272$ s |

So for every lag $\ell \lesssim 68$ — **$\approx 76\%$ of the $91$-lag window** — the lag-0 state
already contains everything the lag-$\ell$ state knows. A trained model has no
information-theoretic reason to attend anywhere but $\ell = 0$ in that range; only
$\ell \in [\sim 69, 90]$ offers content the lag-0 state has forgotten, and those are precisely
the lags `alibi_decay` penalises hardest at initialisation.

The causal *transform* compounds this, and on the unaligned arm it compounds it maximally: the
stored coefficients under the lag-0 state's window reach a further $13.3$–$791$ s back into the
raw signal, so the freshest state already holds both the newest uterine activity ($13.3$ s stale,
via the fastest channel) and a long causal summary of the older activity (via the slow channels).
Attending deeper only re-reads staler summaries of the same past. The conv-LSTM raw cells got a
meaningful lag axis semi-accidentally — LSTM forgetting created effective locality in the memory
the attention reads. The transformer encoders plus causal features remove it.

---

## 5. Finding D — what the attribution is weighting: a clock, with almost no signal beside it

The attribution is $K^{(m)}_t \alpha^{(m)}_{t,\ell}$, so its shape is decided by what the KL
*contains* and which head carries it. On this run:

| Quantity | Value | Reading |
|---|---:|---|
| `kl_total_nats` | $0.494$ | the whole coupling readout |
| `kld_source_null_nats` | $0.333$ | **$67.5\%$ of the KL survives zeroing the source** — availability clock plus flat-input response |
| `coupling_minus_clock_nats` | $0.160$ $[0.157, 0.164]$ | the *entire* clock-exceeding signal, per anchor, over a $2{,}940$-coefficient block |
| `kl_active_dims` | $2$ | dimensional collapse persists at $H = 30$ |
| `kl_top_dimension_share` | $0.954$ | the attribution is essentially one latent group $\approx$ one head's attention profile |
| `logvar_prior_floor_frac` | $0.984$ | prior variance pinned on its clamp floor |
| `attention_entropy_nats` | $2.664$ vs attainable $4.511$ | effective support $\approx 14$ of $91$ lags — close to the `alibi_decay` init shape |

What the clock *is* on this arm, concretely: with no alignment there is no `start_embed` on
either adapter (§2), so the announcement is the per-channel `mask_proj` staircase alone — and it
is a live one. The source stream is not budget-gated, so its slowest channels
($\tau_c$ up to $791$ s) come warm only around step $\approx 260$, **inside the scored anchor
range $[134, 270)$**: the availability pattern keeps switching across the very region the KL is
measured on, which makes a $67.5\%$ null share entirely plausible as literal clock-reading. The
figure is also legitimately comparable to `DIAGNOSIS.md` §6.1's $85.7\%$ external-probe figure —
both belong to unaligned configurations — in a way an aligned run's null share would not be (that
document's own §6.1 note).

The availability staircase is a deterministic function of $t$, readable identically from the
source state at **any** lag — each state knows its own position — so a clock-dominated KL leaves
the attention indifferent over lags, and the ALiBi initialisation resolves that indifference to
$\ell = 0$. With $\approx 0.16$ nats of genuine signal against a $2{,}940$-coefficient
reconstruction, `entmax15` plus `alibi_decay` never had to earn a long-lag reading; the profile
stays near its initialisation. (`alibi_slope_scale` and `lag_bias_init` are constructor defaults
in the shipped config — no arm has varied them.)

---

## 6. Finding E — the check is mis-scoped for the causal cells

`check_argmax_lag` (`teb_vae/lag_attn_cfs/eval/report_seam.py:640`) has two arms:

- argmax at the **far** edge → "censoring artifact: the true maximum may lie beyond the window";
- argmax at **0** → "the attribution never looks back and the lag window is inert".

That second reading is correct in the raw cells, where lag $0$ is the instantaneous sample and a
pin there means the machinery is unused. In this cell, §3 shows the near edge is a **censoring
edge exactly like the far edge** — on this arm part of the pair mass is censored below $\ell = 0$
and part beyond $\ell = 90$, and on the aligned arm the true optimum sits at negative lags for
most of the horizon. The check's FAIL is still doing its job — *do not quote a lag from this
run* — but its stated cause is wrong for this domain, and the check structurally cannot pass at
this geometry: even the ideal model fails it.

---

## 7. Corroborating evidence from the same summary

- **`spectral_gap_deceleration_nats = -4.18`.** The one clinical band where contraction-driven
  coupling should live is the band source-conditioning *hurts* on holdout, while
  `variability` ($+11.16$) and `beat_to_beat` ($+13.53$) gain; the five bands sum to
  `pred_gap_train_path_nats` ($+20.75$), confirming the estimator they are computed under.
  Whatever the source pathway contributes, it is not deceleration-band coupling.
- **`pred_gap_mc_nats = -29.42`** ($-0.99\%$ likelihood) reproduces the known source-path
  generalisation failure at $H = 30$ on the held-out cohort — `DIAGNOSIS.md` P0, unchanged.
  The train-path and MC estimators still disagree in sign ($+20.75$ vs $-29.42$), the estimator
  asymmetry that document names.
- **`source_margin_positive` passes ($61.7$ nats)** while `source_specificity` fails
  ($D_{\mathrm{full}} > D_{\mathrm{base}}$): the pathway reads *this* recording's source rather
  than any source, but its conditioning costs more than it delivers — the FAIL/PASS/FAIL triple
  the verdict registry documents as a real state.
- **`prior_variance_not_pinned` FAIL** ($0.984$ floor fraction) and
  **`calibration_near_nominal` FAIL** (mean standardised square $0.838$; PIT deviation $0.101$)
  persist from the $H = 15$ diagnosis.
- **`coupling_exceeds_availability_clock` is INCONCLUSIVE only because `clock_margin_min_nats`
  ships unset** — the pre-registered open item. The measurement itself now exists, with a
  bootstrap interval, on the unaligned arm.
- **`lag_clocks`: 5 significant windows across 4 Holm families** — weak but present temporal
  structure in the per-segment lag shape. The machinery can register structure when there is any.

---

## 8. What this run's artifacts can still answer, without retraining

Before any code changes, two readings are sitting in the run directory:

1. **Per-head profiles.** `lag_kl/lag_kl_stratified_profile.csv` carries `attention_head_0..3`
   at full lag resolution. The pooled head-mean buries the shallow-ALiBi heads; at $H = 15$ the
   external probe demonstrably found heads with substantial long-lag mass. Whether any head still
   looks back is answerable from this file alone — and it is the profile on which the coming
   aligned/unaligned comparison is actually expressible (§3.2).
2. **Degeneracy columns.** `lag_kl/lag_kl_summary.csv` and `lag_kl_stratified_peaks.csv` carry
   `degenerate`, `peak_to_median`, `zero_fraction`, `mass_above_half_peak`, `peak_width_seconds`
   per profile — the shape vocabulary that says whether the $\ell = 0$ peak is a spike, a
   shoulder, or a smear. An argmax is not a reading of a profile; these are. Recording this run's
   values now fixes the unaligned arm's baseline for the pair before the aligned numbers exist.

---

## 9. Recommendations

Ordered so that measurement questions are resolved before architecture is touched, and cheap
changes before expensive ones.

### P0 — complete the pre-registered pair, and close the config-drift hole that hid the arm

1. **Train and evaluate the aligned arm.** This run *is* the unaligned half of the
   pre-registered pair, at $H = 30$, with full per-recording intervals; the `target_max` arm has
   never been trained. Same seeds, same split, same eval binding. *Superseded in part by §10.6:*
   under the operator's ruling that the architecture is open, the pair completes within the
   **revised** family rather than by training the aligned arm of the current model — this run
   stands as the current family's unaligned record.
2. **Fix the pair's reading protocol before the aligned numbers exist.** §3.2 shows the pooled
   argmax is predicted at $0$ on both arms, so "improved lag-profile concentration" must be
   judged on the shape vocabulary (peak width, mass above half peak, degeneracy) and the
   per-head profiles — quantities this run's artifacts already record (§8). Writing that down
   now keeps the comparison pre-registered rather than chosen after the fact.
3. **Close the drift hole.** The production box trained `null` while the repo's shipped default
   says `target_max`, and only the eval's $\delta = 0$ exposed it. Two cheap guards: put the arm
   in the run name (`_unaligned` / `_aligned_tmax`) rather than overloading "baseline"; and have
   the eval's summary always print `causal_align_reference` beside `source_delay_steps` in the
   console block, so the arm is on the page a reader actually looks at rather than only in
   `resolved_config.yaml`.
4. **Set `clock_margin_min_nats`** in the causal parent's override delta from this run's
   observed spread ($\Delta_{\mathrm{clock}} = 0.160$, interval $[0.157, 0.164]$), as
   `RESULTS.md`'s open item specifies — turning the INCONCLUSIVE tenth verdict into a real gate
   for both cfs cells. One caution to record with it: the *null's composition* differs across
   arms (the aligned adapter adds `start_embed` and a longer cold region — `DIAGNOSIS.md` §6.1),
   so while $\Delta_{\mathrm{clock}}$ remains the right gated quantity on both, the margin's
   provenance is the unaligned arm and should be stated as such where it is set.

### P1 — fix the readout before the model (eval-side, cheap)

1. **Re-scope `check_argmax_lag` for the causal cells.** A near-edge pin under this geometry is
   *censoring*, not inertness. Report it as censored (INCONCLUSIVE, or a FAIL whose message
   states the $\tau^{\mathrm{phys}}$ arithmetic of §3), and judge "machinery alive" by the
   profile-shape vocabulary already computed in `teb_vae/lag_attn_cfs/eval/lag_shape.py` —
   degeneracy, mass above half peak, secondary peaks, per-head entropies — rather than by argmax
   position. As written, the ideal model fails the check at this geometry, on either arm.
2. **Read and report per-head** before concluding nothing looks back (§8.1).
3. **Add an interventional lag readout.** `teb_vae/lag_attn/eval/analyses/lag_ablation.py`
   (keep-mask sufficiency over lag bands, common random numbers, shared anchor support) exists in
   the shared eval and is **not in the cfs pipeline's step list**. Port it — but note its limit
   here: with encoder-state K/V, every kept state still summarises everything older (§4), so band
   sufficiency will read short-band-sufficient partly by construction. The stronger instrument
   for this cell is **input-level occlusion in stored time**: zero the stored source steps in a
   band $[t - b_{\mathrm{hi}},\, t - b_{\mathrm{lo}}]$ via the availability mask, re-encode, and
   measure the per-horizon-step NLL change. That answers "when did the source matter" without
   attention weights at all, and resolves by $h$ — which §3 shows is the only axis on which a
   physiological peak can sit inside the window.
4. **Resolve the lag readout per horizon step** wherever it is predictive rather than
   KL-attributional. The KL cannot be split by $h$ (one $z_t$ conditions the whole block), but
   occlusion and forecast-error readouts can, and $\tau^{\mathrm{phys}}$ is a function of
   $(\ell, h)$ jointly.

### P2 — remove the clock from the KL (small model change, large interpretive win)

The availability pattern is a deterministic function of $t$ and the configuration — it is not
source *content* — yet it enters $q(z \mid Y, U)$ and not $p(z \mid Y)$, so two-thirds of the
coupling readout is a clock. **Condition both branches on it**: feed the source-availability
staircase (or an embedding of it) into the prior head's input as well, and the clock cancels out
of the KL by construction instead of being measured and subtracted. This preserves the source-purity
invariant in the information sense — the prior gains no information about $U$'s *values*, only a
config-determined function of $t$ — but it does change the "prior never sees the source" contract
textually, so it should be its own reviewed arm with the invariant's tests updated deliberately.

The measurement-only alternative is the pre-registered ablation of the announcement terms
(`DIAGNOSIS.md` §10 P1). On **this run's arm** that is a single ablation: the unaligned adapter
carries `mask_proj` only, so arm D1 *is* the whole announcement here, and arm D2 (`start_embed`)
does not exist to remove. The D1/D2 split becomes meaningful on the aligned arm, where both terms
are constructed and removing one alone silently attributes the other's share elsewhere.

### P3 — revise the model architecture (§10 is the program)

The operator has ruled that no trained run is final and the model may change, so this is no
longer an "only if" item. The revision has four components, specified in §10 with their
rationale, their acceptance criteria and the one new test that gates all of them:

1. **localise the lag-attention memory** — K *and* V from a local source representation, the
   deep source transformer dropped from that path (§10.1);
2. **clock-symmetric conditioning** — the prior receives the same deterministic availability
   embedding the posterior does, cancelling the clock out of the KL by construction (§10.2);
3. **horizon-aware decoding** — the target-only persistence residual and a decaying horizon
   weighting (§10.3);
4. **a flat, learnable lag bias** replacing `alibi_decay` as the shipped default (§10.4);

plus the data-side decision that pairs with them — the reference offset, which decides where in
the lag window a physiological delay *can* appear at all (§10.5) — and the versioning rule that
keeps the six-cell square's record intact while freeing the design (§11).

### P4 — the deeper blocker: restore a trainable source pathway first

With $0.160$ nats of clock-exceeding coupling, a pinned prior, two active dimensions and
`pred_gap_mc = -29.4`, there is no signal for **any** lag machinery — current or redesigned — to
learn from. The lag readout failing is downstream of the source pathway carrying almost nothing
that generalises. That program is already written and pre-registered in `DIAGNOSIS.md` §10 and is
unchanged by this document:

- early stopping and multiple checkpoint criteria (this family demonstrably trains hundreds of
  epochs past its validation optimum);
- `source_dropout` $0.2/0.3$ arms — the one seam that regularises the source map without touching
  the target pathway;
- the $\beta$ / prior-variance factorial **after** the clock is controlled (P2), never $\beta$
  alone against `pred_gap`;
- the horizon weighting $w_\tau$ or a target-only persistence residual — which the $H = 30$ move
  made *more* load-bearing, per `DIAGNOSIS.md` §8.2's own note: the uniform objective now pays
  for suppressing fast channels over thirty steps instead of fifteen;
- read `pred_gap_novel_lo/mid/hi` beside every arm, to separate genuine forecast gains from
  better inversion of already-seen history.

### The causal chain, in one line

Weak, non-generalising source signal (P4) $\to$ clock-dominated KL (P2) $\to$ attribution over a
degenerate encoder-state memory (P3) $\to$ pooled over a censored, pair-smeared geometry (P1/§3)
$\to$ a check that reads the inevitable result as inertness (P1/§6).

---

## 10. The architecture revision program

Written under the operator's ruling that no trained run is final and the model may change. Each
component below names the finding it answers, what changes, what deliberately does not, and the
measurement that decides whether it worked. The order within this section is logical; §10.6
carries the build order.

### 10.1 Localise the lag-attention memory — answers finding C

**What changes.** The lag attention's keys **and** values are built from a *local* source
representation instead of the deep encoder state:

- `lag_kv_source: conv_stem` (proposed default) — the availability adapter followed by the causal
  conv stem alone, receptive field $\approx 20$ steps $\approx 80$ s;
- `lag_kv_source: adapter` (the sharp arm) — the adapter output directly, receptive field one
  step, $4$ s lag resolution, at the cost of purely linear per-step content.

The three-block band-attention stack leaves the K/V path — and since nothing else consumes
$h_u$, it leaves the model: most of the source encoder's $\approx 889$K parameters
(`DIAGNOSIS.md` §2.3) are removed rather than repurposed. Content aggregation across lags moves
into the attention itself — the heads already span scales — and into the head-structured
posterior fusion, which both remain as built. So do `entmax15`, the Shaw per-lag key bias, the
frozen `W_o` convention, and the warm-up masking and announcement inside the adapter.

**Why both K and V must be local.** If only the keys were localised and the values stayed deep,
$V$ at lag $0$ would still contain everything every other lag's value contains, and the queries
could settle on recent lags regardless of where the content lives — the §4 degeneracy would
survive with a more plausible-looking attention map on top of it. The incentive to look back
exists only when the content has to be *fetched from where it is*.

**What this cannot remove.** The K/V representation's receptive field is the resolution floor of
the lag readout — $\approx 5$ lag steps at the conv-stem arm, $1$ at the adapter arm — and
underneath both sits the transform's own memory: a stored coefficient is a causal integral with
its channel's group delay and kernel width, and no encoder change reaches that. Removing the
encoder degeneracy shrinks the ambiguity from $\approx 272$ s to the transform's intrinsic
smearing, which is finding B's territory and §10.5's problem.

**The test that gates this — the lag-identifiability fixture.** No test in the family ever
checked that the lag machinery can recover a *known* lag; the §4 degeneracy would have failed
exactly such a test before any production run was trained. Build a synthetic shard in which the
source determines a target feature at a planted delay of $\delta$ stored steps, with
$\delta \in (H,\, L - 1)$ so the informative lag $\ell^\*(\tau) = \delta - \tau$ lies strictly
inside the window for every horizon step (e.g. $\delta = 45$, $H = 30$: $\ell^\* \in [15, 44]$).
Train the tiny geometry a few epochs; assert the support-corrected profile peaks inside that band
and not at $0$. The current architecture is expected to *fail* this fixture — which is the
demonstration that the redesign is load-bearing, and the reason the fixture lands in the same
sprint as the redesign rather than after it.

### 10.2 Clock-symmetric conditioning — answers finding D

**What changes.** The source-availability announcement — a deterministic function of $t$ and the
resolved configuration, nothing else — is conditioned on by **both** branches instead of only the
posterior. Concretely: build the announcement tensor once from the resolved delay vectors (the
per-channel staircase; plus the start-of-record indicator on aligned arms, where it exists),
register it as a non-persistent buffer like every other budget-shaped tensor, and add its
projection to the prior head's input through its own `LayerNorm`. The projection is the prior's
own parameter, deliberately **not** shared with the source adapter's `mask_proj`: sharing would
couple the two pathways' gradients, and the *pattern* is what must be shared, not the map.

**Why this is cancellation rather than measurement.** Any deterministic function of $t$ that
enters $q$ but not $p$ can push the posterior off the prior with no source information in it —
the hazard `kld_source_null` measures and the permutation control cannot see. Conditioning both
branches on it makes the clock term cancel out of $\mathrm{KL}(q \| p)$ *by construction*. The
pre-registered removal arms (D1/D2) subtract the clock; this makes it never enter.

**The invariant, restated rather than weakened.** "The prior never sees the source" becomes "the
prior sees no function of the source **values**" — the availability pattern carries zero
information about $U$'s content. `tests/test_invariants.py`'s source-purity assertions are
updated to state exactly that, in the same change, so the contract moves deliberately rather
than eroding.

**Acceptance, pre-registered.** On the revised model, `kld_source_null` $\approx 0$ (against
$0.333$ of $0.494$ here). The source-null control stays in the eval unchanged — it becomes the
check that the cancellation worked, rather than the estimate of the contamination.

### 10.3 Horizon-aware decoding — answers the objective finding `DIAGNOSIS.md` §8 already made

**What changes.** Two mechanisms, separable, both fixed before training:

1. **A target-only persistence residual** in the decoder head:
   $\mu_{t,\tau,c} = w_{\tau,c}\, y_{t,c} + f_\theta(z_t)_{\tau,c}$, with $w$ fixed or strongly
   regularised and decaying in $\tau$. It is target-only — $y_t$ is input to both branches
   already — so it opens no source bypass and moves `pred_gap` only through what it frees:
   the latent stops spending capacity carrying the *levels* of fast channels whose persistence
   pays for two or three steps of thirty.
2. **A decaying horizon weighting** $w_\tau$ in the reconstruction, replacing the uniform
   thirty-step sum whose incentive to kill fast channels the $H = 30$ move doubled
   (`DIAGNOSIS.md` §8.2's own note).

**Acceptance.** The fast warm-tertile (`pred_gap_warm_hi`, $-4.05$ here) and the novelty split:
a gain concentrated in `pred_gap_novel_hi` is a forecast improving, one in `_lo` is history
inversion improving, and the dead-channel fraction of `DIAGNOSIS.md` §8.1 is the direct readout.

### 10.4 The lag bias default — answers the initialisation half of finding D

`alibi_decay` was built so a randomly-initialised head could not lock onto spurious long-lag
structure. On this family the observed failure is the opposite one: with $0.16$ nats of genuine
signal the profile never leaves the short-lag prior. The revised default is a **flat, learnable**
per-(head, lag) bias (`alibi_slope_scale: 0`, or `lag_bias_init: normal`), with `alibi_decay`
kept as the named comparison arm — one config key, and the identifiability fixture of §10.1 runs
under both, which separates "the data says lag 0" from "the init says lag 0" on ground truth
rather than argument. A lag-band dropout (masking random lag bands during training so no single
lag can carry everything) is a plausible further regulariser; it is speculative, and is noted as
an arm rather than a default.

### 10.5 The reference offset — the data-side half of the pair

The readable physical window is fixed by geometry before any training run:

$$\tau^{\mathrm{phys}} \in \bigl[\, 4(1{+}h) - 20 + (\tau^u_{\mathrm{ref}} -
\tau^y_{\mathrm{ref}}),\;\; 4(91{+}h) - 20 + (\tau^u_{\mathrm{ref}} - \tau^y_{\mathrm{ref}}) \,\bigr].$$

With equal references (`target_max`) it starts at $-16$ s and censors physiological delays for
most of the horizon (§3.2). Aligning the source onto a **faster** clock than the target's — drop
source channels slower than some $\tau^u_{\mathrm{ref}} \approx \tau^y_{\mathrm{ref}} -
2\ \mathrm{min}$ — re-centres a $20$–$60$ s physiological delay mid-window across all $h$, and
*reduces* the recency cost of alignment (the freshest UA becomes $\tau^u_{\mathrm{ref}}$ old
instead of $402$ s). The cost is measurable before committing: how many of the $51$ UP channels
survive the lower reference is a property of the shards' stored `causal_delay_s`, not a guess. A
shift can only delay — a causal stream cannot be advanced — so the source reference is bounded
below by the slowest *kept* source channel; the knob is which channels are kept.

### 10.6 Build order and what decides each step

| Order | Component | Cost | Gate |
|---|---|---|---|
| 1 | the identifiability fixture (§10.1), run against the **current** architecture first | a synthetic shard and one test | documents the degeneracy on ground truth; every later step re-runs it |
| 2 | clock-symmetric conditioning (§10.2) | small, additive | `kld_source_null` $\approx 0$ on the tiny fixture |
| 3 | local K/V (§10.1) | the one structural change | the fixture recovers the planted lag |
| 4 | persistence residual + horizon weighting (§10.3) | moderate | fast-tertile and novelty splits |
| 5 | flat lag bias default (§10.4) | one key | fixture under both inits |

Then train the revised model under P4's training-control program (early stopping,
`source_dropout`) and evaluate it under P1's revised readout. **One supersession to record:**
under the operator's ruling, the pre-registered aligned/unaligned pair of P0 completes **within
the revised family** rather than by training the aligned arm of the current model — the pair is
worth its cost once, on the model whose lag readout can express the answer, and this run stands
as the current family's unaligned record.

---

## 11. The cost of the freedom, and what is explicitly *not* supported

**The one structural cost: the square.** The six-cell grid's whole value is that its cells
differ by exactly one axis, pinned leaf-for-leaf by the config tests. A revised architecture
reads on neither edge, so it must not mutate this cell in place: it lands as a **new, versioned
package** with its own pre-registered `DESIGN.md`/`RESULTS.md` written before its first
production run — the same discipline every existing cell followed. The existing cells' records
stay intact, and the revised model is still comparable to *this run* through the shared eval
pipeline (same shards, same readouts, same verdicts); what changes is that the comparison is a
family edge, stated as such, rather than an encoder edge.

**And what remains unsupported by the evidence:**

- **Not a delay-reporting bug.** `source_delay_steps = 0` is the correct value for this
  checkpoint — the run is the unaligned arm — and the eval's lag axis was right.
- **Not a capacity problem.** Nothing here changes `DIAGNOSIS.md` §8.3's negative finding;
  widening $d_z$ or deepening the encoders addresses none of the five links above.
- **Not `lag_floor` or warm-up coverage.** `source_lag_warmth_frac_st = 1.000` and
  `_ph = 0.995`; the floor ships at $0$ and no evidence implicates it.
- **Not a broken attribution.** Both lag identities hold to $10^{-4}$ nats on the worst anchor;
  the per-anchor recombination and the KL identity pass; the axis orientation is pinned by the
  architecture parent's suite.
- **Not evidence that no lagged coupling exists.** On this arm an argmax at $0$ is the
  information-optimal read of a representation whose filters have already integrated the past
  (§3.1, §4); absence of a peak here says nothing about the physiology.
- **Not a reason to rebuild the family from scratch.** The eval pipeline's controls
  (source-null, permutation, clock margin, the pre-registered alignment pair), the target
  transformer encoder, the shared decoder body, the latent width, the tiled anchor geometry and
  the warm-up/budget machinery are all unimplicated and carry forward unchanged into §10's
  revision. What §10 replaces is the lag-attention memory, one conditioning asymmetry, the
  decoder's horizon handling and one initialisation default — a revision, not a restart.

---

## Appendix — code path index

| What | Where |
|---|---|
| Attribution identity $\widetilde K_{t,\ell} = \sum_m K^{(m)}_t \alpha^{(m)}_{t,\ell}$ | `teb_vae/lag_attn/nets/heads.py::TEAnalysisHead` |
| Lag attention, `alibi_decay` init, `entmax15`, window-as-view | `teb_vae/lag_attn/nets/attention.py` |
| K/V from deep causal source state; `head_structured=True` | `teb_vae/lag_attn_cfs/nets/causal_inputs.py::forward` |
| Pooled profiles, four argmaxes, `lag_summary` | `teb_vae/lag_attn_cfs/eval/metrics.py` |
| The failing check | `teb_vae/lag_attn_cfs/eval/report_seam.py::check_argmax_lag` |
| $\delta$ resolution in the eval | `teb_vae/lag_attn_cfs/eval/run.py:1241` (`task.orig_model.source_delay_steps`) |
| `source_delay_steps` = `source_gate.max_delay`, and why it is not $\tau_{\mathrm{ref}}$ | `teb_vae/lag_attn_transformer_rws/nets/model.py:640`; `teb_vae/lag_attn_cfs/causal_warmup.py::WarmupBudget.reference_delay_s` |
| The unaligned build: no shifts, no `start_embed`, $\delta = 0$ | `teb_vae/lag_attn_transformer_cfs/tests/test_construct.py::test_omitting_the_alignment_keywords_builds_todays_model_bitwise` |
| Lag-to-seconds arithmetic, all three quantities | `teb_vae/lag_attn/nets/lag_report.py` |
| Profile-shape vocabulary (degeneracy, peak width, mass) | `teb_vae/lag_attn_cfs/eval/lag_shape.py` |
| Interventional lag readout, not yet in the cfs pipeline | `teb_vae/lag_attn/eval/analyses/lag_ablation.py` |
| Alignment mechanics, adapter announces $W'_c + d_c$ | `teb_vae/lag_attn_cfs/nets/causal_inputs.py::_build_adapter` |
| Pre-registered aligned/unaligned reading; recency cost | `teb_vae/lag_attn_transformer_cfs/configs/default.yaml` (`causal_align_reference` block) |
| The $H = 15$ diagnosis this document extends | `teb_vae/lag_attn_transformer_cfs/DIAGNOSIS.md` |
