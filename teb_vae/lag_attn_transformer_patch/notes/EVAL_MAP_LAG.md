# Eval map — lag, coupling and control analyses (patch cell)

Written 2026-10-03. Scope: how the lag, coupling and control half of `teb_vae/lag_attn_cfs/eval`
carries over to `SeqVaeLagAttnTrfPatch`, and which new lag analyses the raw signals make possible.
Core infrastructure (collection, preflight, `metrics`, binding), Captum and attribution, and the
forecast and latent analyses are mapped elsewhere. They appear here only where a module in this
slice depends on them. Paths are relative to `teb_vae/lag_attn_cfs/eval/` unless stated.

Status key: **as-is** = runs unchanged once the collection pass produces its tables. **hook** = needs
one small binding value. **port** = needs a patch-specific rewrite (line estimate given). **N/A** =
has no meaning for patches.

## 1. Facts about the patch model that decide the mapping

Each fact below was checked on a built model (`tests/conftest.build(max_lag=37)`).

| Fact | Value | Consequence |
|---|---|---|
| `source_delay_steps` | 0 | `lag_axis.compensated_seconds_axis(L, 0)` gives τ_ℓ = 4ℓ s exactly. There is no group delay: lag ℓ means the UP patch whose last sample lies exactly 4ℓ s before the anchor patch's last sample, quantised to 4 s. |
| `source_gate`, `target_gate` | `None` | The `gated = u if model.source_gate is None else ...` branches in `occlusion` and `time_shift` already handle this. |
| Source adapter | `PatchEmbedding` (`nets/patching.py:358`); no `availability` buffer, no positional embedding | **There is no availability clock.** `occlusion._announcement` returns `None`, so the invariance is NaN, which is correct. On a zero stream every K/V token is identical, so the null-arm attention is the attention's learned per-lag key bias (`lag_attn/nets/attention.py:68`), not a clock. |
| K/V reach | 1 token (`lag_kv_source: adapter`) | Occluding a band of lags changes K/V at exactly those lags, and nowhere else (the query is prior-only). The intervention is lag-local. Only the entmax renormalisation spreads its effect to other lags. |
| Lag-support margin | F − (L−1) − F_u = 30 − 37 − 0 = **−7** | Unlike CFS, the support correction, the untruncated profile and the entropy ceiling < log L now **do work**. Anchors 30–36 (7 of 240 dense) are truncated. `lag_kl`, `attention` and `lag_clocks` were written for this case. They measure it, and they need preflight's `causality.lag_support` block (core infra). |
| Where informative lags sit | Anchor t forecasts tokens t+1…t+30 | A UP→FHR delay of d tokens is informative at lags [d−H, d−1] = [d−30, d−1] (the planted-band rule, `lag_recovery_check.py`). Physiological delays of 20–60 s (5–15 tokens) therefore land at the **near edge**, lags 0–14. The lag map smears a delay over up to H lags, so the delay is d = ℓ+1+τ and needs the **horizon** axis (§6, N4). |
| Token layout | `[value(16), delta(16), m_t − 1]`, validity 0 = valid | Zeroing a whole token turns an invalid token into "valid and flat at the z-mean". On a valid token it zeroes only the values (§4). |
| Units | UP and FHR are loader z-scored; the target level is `(mean − loc)/scale` | The z-mean of UP is **above** resting tone, because contractions skew UP positive. "Zero" is therefore a weak tonic plateau, not "no contraction". |
| Step | 16 samples at 4 Hz = 4 s = `SECONDS_PER_STEP` | `lag_shape.SECONDS_PER_LAG_STEP`, the 0.5 h clinical grids and `events` endpoints (`16·(a+1) − 1`, `collect.py:776`) are all correct as they stand. |

**What every offline analysis below depends on (core infra, mapped elsewhere).** The collection
pass's lag block (`metrics.lag_summary`, `metrics.py:3111`), the per-sample `vectors` sidecar
(`lag_profile_untruncated`, `attention_profile_support_corrected`, `lag_profile_per_head`), the
per-anchor `anchor_vectors` (`kl_lag_map`, `attention_lag_map`), the per-anchor columns (`kld_per_t`,
`argmax_lag`, `seconds_since_contraction`, the gain columns), the retained `attn_weights` and
`up_raw`, and `kld_source_null`. Once collection runs on `(y_patch, u_patch, phase, stride)`, every
offline analysis in this slice is **as-is**.

## 2. Shared hooks (binding values, used by several modules)

| Id | What | Why | Size |
|---|---|---|---|
| H1 | Lag-axis wording: `COEFFICIENT_LAG_AXIS_LABEL`, `GROUP_DELAY_CAVEAT`, `GROUP_DELAY_NOTE` (`lag_axis.py:72,87,99`) | All three are false for patches. They speak of stored-coefficient time and a 791 s group delay. Patch wording: *"lag (s): UP patch before the anchor patch, 4 s quantised; no filter delay. The UP→FHR delay a lag explains is 4(ℓ+1+τ) s."* The constants are imported **by name** in 6 analyses (`attention`, `lag_clocks`, `lag_high_kl`, `lag_kld_scaled`, `lag_kl`, `source_null`) and in 9 other modules. | Have consumers read `lag_axis.X` at call time and let the binding rebind it: ~15 one-line edits. A local copy of each analysis would be worse. |
| H2 | `occlusion_bands` within `max_lag` 37 | The CFS override's `far: [68, 90]` is refused at config load. Suggested override: `{a0_28s: [0,7], a32_60s: [8,15], a64_104s: [16,26], a108_148s: [27,37]}`, or `partition_width: 2` for N4. On the planted shard (`max_lag` 60), use `planted: [15, 44]`. | Override YAML |
| H3 | `clock_margin_min_nats: null` | 0.15 was calibrated on the CFS unaligned run's Δ_clock, and there is no clock here. Leave the verdict INCONCLUSIVE until a patch run measures the spread. | Override YAML |
| H4 | Input/target seam: `metrics.model_inputs` (`metrics.py:249`) returns the CFS 5-tuple, and the forward takes 3 streams. The patch model has **no `_build_forecast_target`** (checked on the MRO). | Patch equivalent: `task._build_forward_inputs(batch)` → `(y_patch, u_patch, phase, stride)`, and target = `model.summary_target(batch.fhr, batch.weight)` gathered at `a+1+τ` (as in `patch_target.py:525-532`). `time_shift`, `occlusion` and N1/N4/N6 use it. | One `patch_seam.py`, ~25 lines |

## 3. Per-module map

| Module (entry) | Computes | Reads | Status | What a raw-signal version could add |
|---|---|---|---|---|
| `lag_axis.py` (layer 0; `compensated_seconds_axis` :102, `read_lag_support` :208) | The τ_ℓ axis, NaN padding, segment stride, preflight lag-support reader | `lag_report.SECONDS_PER_STEP`, `delay_steps`, `preflight.json causality.lag_support` | **hook** H1 (wording only; the numbers are right with δ = 0) | The axis becomes physical seconds. Add an optional second axis, d = 4(ℓ+1+τ̄), for "UP→FHR delay". |
| `lag_shape.py` (`profile_statistics` :445) | 16 shape statistics, degeneracy guard, band restriction, rectification | Profiles and seconds only | **as-is** | — |
| `lag_hist.py` (`cell_distances` :221) | Normalisation, JS distance, W1 in seconds, centroid, quantiles | Profiles and seconds only | **as-is** | — |
| `events.py` (layer 0; `detect_contractions` :265) | Contraction onset, peak and end from **raw `batch.up`** (Savitzky–Golay, σ-relative prominence, level-crossing flanks, gap-dropping via `weight`) | Raw UP at 4 Hz, `weight` expanded ×16 | **as-is**. Collection keeps only the **onset** age (`collect.py:732-781`). | Store peak and end ages plus prominence and duration per anchor (+~10 lines in `_contraction_ages`). Port `detect_decelerations` (it exists at `lag_attn_rws/eval/events.py:423`, but the layer rule forbids importing `lag_attn_rws.eval`; ~80 lines) on raw FHR in bpm. |
| `analyses/events.py` (`run_events_analysis` :480) | `mc_pred_gap` and `kld_per_t` at anchors ≤ `event_lag_window_s` (120 s) after onset, against count-matched same-recording controls; detection figure on retained `up_raw` | per_anchor `seconds_since_contraction`, per_sample, retained `up_raw`, `traces.raw_signal_scales` | **as-is** | `REMOVED_READOUTS` (:92) no longer applies, because the level channel maps back to bpm. Restore the contraction-triggered response and deceleration skill (forecast slice), and add event-locked **lag** analyses (§5, N2/N3). |
| `analyses/lag_kl.py` (`run_lag_kl_analysis` :752) | Raw, support-corrected and untruncated KL-by-lag profiles; peak/degeneracy; identity check Σ_ℓ K̃ = K; class/subgroup/time-window strata | `results.lag` (`kl_lag_profile*`, `kl_lag_anchor_counts`, `delay_steps`, `n_lags`, `num_heads`, per head), `vectors`, preflight margin | **as-is** + H1. The margin is −7, so the three profiles now genuinely differ. | A data-side reference on the same axis: the UP patch-mean autocorrelation (the smear floor) and the CCF of UP level at t−ℓ against FHR level at t+1+τ, averaged over τ, from retained raw signals. A model profile that differs from the data CCF is the finding. |
| `analyses/attention.py` (`run_attention_analysis` :535) | Head-averaged and per-head attention profiles, per-anchor entropy against the attainable ceiling, (t, ℓ) heatmap of one retained segment | `results.lag` attention keys, `geometry.anchor_floor`/`t_valid`, retained `attn_weights (N,T,M,L)` (`caps.attention`) | **as-is** + H1 | Overlay the detected contraction peaks on the heatmap as diagonals ℓ = t − t_peak. A head that tracks contractions follows them. |
| `analyses/lag_clocks.py` (`run_lag_clocks_analysis` :1587) | 14 shape statistics × {KL, attention} per segment, on the two **clinical** clocks; Holm-tested centroids | per_sample, `vectors` (`lag_profile_untruncated`, `attention_profile_support_corrected`), `results.lag` | **as-is** + H1 | A contraction-frequency covariate per window (contractions per 10 min, from raw UP), to ask whether the lag centroid tracks uterine activity rather than the clock. |
| `analyses/lag_kld_scaled.py` (`run_lag_kld_scaled_analysis` :832) | Band-restricted, soft-weighted, full-support and per-head nats-scale lag statistics on both clinical clocks; untested | `vectors` (+`lag_profile_per_head`), `results.lag.kl_lag_profile_clock_excess`, `occlusion_bands` | **as-is** + H1 + H2. The "clock-excess" soft weight (`soft_weight` :302) reads here as **flat-UP excess** (§4.2). | Bands defined in seconds before a detected contraction peak instead of fixed lags (event-relative bands). |
| `analyses/lag_high_kl.py` (`run_lag_high_kl_analysis` :4391) | High-, top-, rest- and gain-anchor lag profiles, hot lags, histograms and distances, drift, usefulness, occlusion consistency, contraction enrichment (structure in §4.3) | per_anchor (`kld_per_t`, `argmax_lag`, `seconds_since_contraction`, gain), `anchor_vectors` (`kl_lag_map`, `attention_lag_map`), per_sample, `occlusion/occlusion_per_recording.csv`, `event_lag_window_s` | **as-is** + H1 | Enrichment as a curve over time-since-**peak** (not one 120 s window). Gain-by-argmax against the lag of the nearest contraction peak. |
| `analyses/source_null.py` (`run_source_null_analysis` :606) | Δ_clock = `source_conditioned_kl_raw − kld_source_null` per recording, bootstrapped; signed clock-excess lag profile; delta mask | per_sample columns, `results.lag` (`kl_lag_profile`, `_null`, `_clock_excess`), `occlusion_bands`, `clock_margin_min_nats` | **as-is** numerically + H3. The "availability clock" reading is **N/A**: the null arm is zero K/V content, so Δ = **KL beyond the flat-UP response** (the lag prior). | Two more nulls in collection: a **baseline-tone** null (constant resting UP, not the z-mean) and a **missing** null (validity −1 → the learned `missing` token, rarely trained under `finite`, so out of distribution). |
| `analyses/perm_control.py` (`run_perm_control_analysis` :283) | Base, full and shuffled branch scores; penalties; `source_margin`; outcome class | per_sample `mc_nll_*_block`, `source_conditioned_kl(_shuffled)_raw` | **as-is** (the derangement permutes `source_state` rows, which works for any representation) | — (time_shift and N1 cover timing) |
| `analyses/coupling.py` (`run_coupling_analysis` :844) | Per-recording `pred_gap` (MC, mean, train), KL, error- and likelihood-space percentages, estimator agreement | per_sample, `record.likelihood_structure.scored_cells` (= 60) | **as-is** (the "C_keep budget-local" note strings are stale wording only) | Stratify `pred_gap` by UP quality: the fraction of flat runs in valid UP. Under `finite`, a disconnected UP reads as valid (DESIGN §4). |
| `analyses/time_shift.py` (`run_time_shift_analysis` :332) | Source swapped for the nearest non-overlapping segment of the same recording (≥ T·4 + 2·trim s apart); ΔK, Δg (mean-decoded); verdict `coupling_is_time_specific` | `model_inputs`, 3-stream forward, `model._build_forecast_target`, `model.geometry.t`, `traces.LOADER_TRIM_S`, `controls.occluded_forward_outputs`, `model.kld_tensor`, `model.decoder` | **port** `score_pairs` (:141) via H4, ~30 lines. Pairing, summary and figure are as-is. | A sub-token and token **delay sweep** (§4.4). |
| `analyses/occlusion.py` (`run_occlusion_analysis` :1042) | Per band: zero the source at lags [lo, hi] of one seeded anchor per segment, re-encode, re-pose, re-decode; Δ block NLL by horizon step; live fraction; clinical-clock placement | `model_inputs`, 3-stream forward, `_build_forecast_target`, `scored_weight`, `forecast_mask`, `persistence`, `source_gate`, adapter `availability` | **port** `collect_batch` (:313) + `_horizon_scores` (:210), ~60 lines + an edit callback. `build_frames`, summary, clocks and figures are as-is. Fix the live fraction (§4.1). | Raw-domain "no contraction" fill; occlusion bands placed on detected contractions; a (band × τ) delay map (N4). |
| `analyses/band_partition.py` (`run_band_partition_analysis` :546) | Channel → band map from the shards' `sel_*` attributes, for ST/PH channels | The shards' `fhr_ph`/`up_ph` attributes, keep index | **N/A**. There are no spectral channels; a token is 16 values, 16 deltas and 1 validity. Exclude it in the binding (and `spectral_skill`, which joins its CSV; forecast slice). | — |

## 4. Deep dives

### 4.1 Occlusion: is zeroing the right intervention?

**What the helper does.** `controls.occluded_forward_outputs` (`lag_attn_rws/nets/controls.py:401`)
computes `source * keep[:, :, None]`, which zeroes **all 33 channels** of every token in the mask, and
then re-runs `encode_source_kv` → `_attend_and_pose` → the decoder.

**What that means for a patch token.**
- **Valid token.** The validity channel is already 0, so zeroing the token zeroes value and delta
  only. The result is "valid UP, flat at the population z-mean".
- **Invalid token.** The validity channel goes from −1 to 0, so a gap becomes "valid and flat". This
  is rare under `finite` validity, but the edit is wrong in kind.
- **Boundary delta.** The first token after the band keeps `delta[0]`, the real cross-boundary
  difference. A raw-domain edit followed by re-`patchify` gets this right.
- **"Zero" is not "no contraction".** The z-mean of UP sits above resting tone, so this arm asks
  "what if UP had been a flat, mildly raised plateau here". That answer is comparable with CFS (its
  zero is also a channel mean), but it is not the clinical intervention.

**Proposed arms.** Score all three under the same anchor and the same latent noise:

| Arm | Edit | Question it answers |
|---|---|---|
| `zero` (CFS-comparable) | `u[..., :32] = 0` on band tokens; validity untouched | Value is replaced by the mean; the existing headline |
| **`baseline`** (primary) | On **raw** `batch.up`: band samples ← the segment's resting tone (10th percentile of valid UP over the preceding 10 min, held flat), then `patchify` | "No contraction at these lags." Delta is 0 inside the band and consistent at its edges |
| `missing` | Band tokens' validity ← −1 (→ the learned `missing`) | "UP unknown here". Out of distribution under `finite`; report as a curiosity, not a readout |

**Can the existing analysis do it with a different callback?** Yes, without touching the helper.
`occluded_forward_outputs(model, outputs, source, occlusion=None, ...)` re-encodes whatever stream
it is given, so the patch `_arm` passes `edit(u_patch or raw, band_mask)` with `occlusion=None`. The
edit callback is ~10 lines.

The work is in `collect_batch`, not the callback:
- the 3-stream forward and the 5-tuple from `model_inputs` → H4;
- `_build_forecast_target` → `summary_target` + gather;
- `persistence` is absent (`persistence_residual: false`), so it is already handled;
- the **live fraction** (`occlusion.py:409`, `gated != 0`) counts channels, so a fully valid patch
  reads 32/33 and an invalid one counts as live. Use the validity channel, `u[..., -1] + 1`, over the
  band instead (one line).

`band_mask` (:158), `choose_anchors` (:183), the frames, summary, cost block, clock join and figures
are representation-agnostic.

Estimate: **port ~60 lines + 10-line callback**. Add the arm name to the per-band column key.

**Raw-signal additions.** Event-anchored occlusion: mask the tokens spanning a detected contraction
[onset, end] instead of a fixed lag band. Score the anchors 0–148 s after its peak, against an
equal-length contraction-free band of the same segment. This gives "what one contraction is worth",
measured directly.

### 4.2 `source_null`, `lag_kld_scaled`, `lag_clocks`: clock and group-delay machinery vs reusable statistics

There are three different "clocks" in this family. Only two have nothing to measure in the patch
model.

| Thing | Where | Patch |
|---|---|---|
| Availability/announcement clock (warm-up staircase m^u_{t,c}) | `source_null` framing (:1-60), `occlusion._announcement` (:260), `perm_control` docstring, `lag_kld_scaled` soft-weight rationale | **N/A**. There is no staircase or announcement buffer. The code paths self-skip (`None`/NaN). Read every "clock" as **flat-UP response**. |
| Alignment and forecast clocks, group delay, `source_delay_is_max_over_channels`, `causal_delay_s` | `lag_axis` caveat, `metrics.lag_summary` flag, `band_partition` columns | **N/A**. δ = 0, so the compensated axis is the identity ×4 s. Only the H1 wording changes. |
| **Clinical** clocks (time to delivery, second stage; `cohort.add_time_bins`, `TRAJECTORY_BIN_HOURS`) | `lag_clocks.CLOCKS` (:423), `lag_kld_scaled.Clock` (:202), `lag_high_kl.Clock` (:572), occlusion clock page | **Reusable as-is.** Nothing representation-specific. |

**What carries over unchanged.**
- In `lag_clocks`, everything does: `add_feature_columns` (:477), `clock_rows`, `analyse_windows`
  (Kruskal–Wallis/Holm), `window_profiles` and the figures. All are `lag_shape` statistics on any
  per-segment profile.
- In `lag_kld_scaled`, the band sources (`occlusion_bands`), the full support, the per-head sources
  (`build_sources` :360) and both clocks.

**What needs reinterpretation, not code.** `soft_weight` (:302) uses ω_ℓ = Δ⁺_ℓ / max Δ⁺, with
Δ = matched − null attribution. On patches this is "KL beyond what a flat UP would elicit". That is
arguably a cleaner selector than on CFS, because the null contains no clock leak.

### 4.3 `lag_high_kl.py` (4.9k lines): structure

| Lines | Block | Reusable on any `source_kl_lag_map`? |
|---|---|---|
| 130–300 | Filenames, column names, `AnchorBand` (high 30 %, top 10 %, rest, gain), `PROFILE_SOURCES` | yes |
| 641–780 | `AnchorPopulation` join (per_anchor ⋈ per_sample on `sample_index`), pooled thresholds, band masks | yes |
| 788–875 | `restricted_profiles`, `hot_lag_set` (pooled upper 30 % lags), `share_on` | yes |
| 877–1034 | Per-segment features via `lag_shape` | yes |
| 1037–1300 | `argmax_by_quantile`, `gain_by_kl_quantile`, `gain_by_argmax`, `band_overlap`, `usefulness_test` (Wilcoxon, paired within recording) | yes |
| 1300–1415 | `occlusion_consistency`: Spearman between band KL share and occlusion Δ (reads the occlusion CSV) | yes, once occlusion is ported |
| 1415–1515 | `contraction_enrichment`: high share ≤ 120 s after onset vs outside, per recording | yes. The only event-related block, and it is scalar, not lag-resolved |
| 1518–1807 | Clinical clocks: windows, Kruskal–Wallis/Holm, recording and window profiles | yes |
| 1810–2709 | Per-recording normalised histograms, JS/W1 distances, drift slopes and tests | yes |
| 2709–4298 | 12 figures | yes (H1 wording) |
| 4298–4931 | Recordings table (feeds `cross_subgroup`), headline, entry | yes |

Nothing in it is clock or group-delay machinery beyond `compensated_seconds_axis(n, delay_steps)`
(:4421), which is the identity here, and the caveat strings.

### 4.4 `time_shift`: what it shifts, and a raw delay sweep

**Today.** Only the **source** stream moves. The whole source is replaced by the nearest
non-overlapping segment of the same recording (≥ ~20 min away). Everything target-side is kept. The
verdict is `coupling_is_time_specific` on ΔK. This is a recording-level timing control, not a lag
probe.

**Raw delay sweep** (new; reuses `occluded_forward_outputs(..., occlusion=None)`):
- **Shifts.** s ∈ {±1, 2, 4, 8, 15} samples (sub-token, 0.25–3.75 s) and {±16, 32, 64, 160} (1–10
  tokens).
- **Edit.** Roll raw `batch.up` by s. Pad the vacated samples with NaN, so `finite` validity marks the
  pad token `missing`, then `patchify`. The target and weight are untouched.
- **Readouts per s.** ΔK(s), Δg(s) (mean-decoded, as in `score_pairs`), and the per-head attention
  centroid shift Δℓ̄(s).
- **Figure.** Three panels against s in seconds: Δg(s), ΔK(s), and Δℓ̄(s) per head with the identity
  line s/16 tokens. Token boundaries are ticked.
- **Interpretation.**
  - The width of Δg(s) is the timing precision the forecast relies on. Flat out to ±40 s means the
    model reads slow UP state, not contraction timing.
  - Δℓ̄(s) ≈ s/16 means the attention tracks content. If Δℓ̄(s) ≈ 0, the attention sits at fixed lags.
  - Structure periodic in s mod 16 is a patch-grid artefact.
  - s < 0 hands the model ≤ s of future UP. Label it as a non-causal probe, not a forecast.
- **Cost.** 18 arms × (re-encode + attention + posterior + dense decode) per batch. At cap 256
  segments, this is about 18 dense forwards' worth of segments, a few minutes on one GPU.
- **Size.** ~40 lines on top of the H4 seam.

### 4.5 Events: detection today, and what is event-locked

**Detection today.** Contractions come from **raw `batch.up`**, not from coefficients
(`events.detect_contractions`, called in `collect._contraction_ages` at `collect.py:779`). Only
`seconds_since_contraction` (the age of the latest **onset** at the anchor's last raw sample) reaches
the tables. Peak and end are recomputed only for the detection figure, on retained `up_raw`.
Decelerations are **not detected anywhere** in the CFS eval (`events.py:15-23`).

**Event-related analyses today.** There are two, and both are scalar in lag:
1. `analyses/events` conditioned coupling: KL and gap within 120 s after onset, against matched
   controls.
2. `lag_high_kl.contraction_enrichment`: the high-KL anchor share within 120 s after onset.

**No analysis resolves attention, the lag map, KL or forecast error against time-from-event.** For
the patch cell this is the main gap. The UP tokens are 4 s raw patches and K/V reach is one token,
so "which contraction did the model look at" is now a well-posed question. N2 and N3 below answer
it.

## 5. New lag analyses enabled by raw signals and the 1-token K/V reach

Shared conventions:
- every statistic is reduced per recording, then bootstrapped over recordings (the house rule);
- figures follow the memory conventions (mask warm-up cells, log scales where magnitudes span
  decades);
- **validate each one on `planted.yaml` first**: the shard couples UP to FHR at δ = 45 tokens = 180 s
  (`RESULTS.md`), which gives a known answer.

**N1. Effective FIR kernel: synthetic contraction injection (plan A.3).**
- *Question.* What forecast response does one contraction elicit, as a function of delay? This is
  the model's learned UP→FHR impulse response.
- *Computation.*
  - Add a raised-cosine bump to raw UP: duration 60–90 s; amplitude at the 25/50/75 % quantiles of
    detected prominence; injected at t₀.
  - Use two backgrounds: (a) the segment's resting tone, held flat; (b) real UP in a stretch with no
    contraction.
  - Run the dense forward with and without the bump, under the same latent noise.
  - Record, for every anchor t with t − tok(t₀) ≤ 37:
    - Δ attention mass on the injected tokens, per head;
    - ΔK_t and Δ`source_kl_lag_map`[t, t − tok(t₀)];
    - Δ`mu_full` level channel at (t, τ), de-standardised to bpm.
  - Collapse onto d = 4(t + 1 + τ) − t₀ s to get the kernel k(d), with an interval over recordings.
- *Figure.* Three panels: k(d) in bpm against d (0–268 s), one curve per amplitude; attention on the
  injection against t − t₀ per head; ΔK against t − t₀.
- *Cost.* 3 amplitudes × 2 backgrounds × ~5 injection times = 30 dense forwards per batch. At 128
  segments this takes minutes. ~120 lines.
- *Interpretation.*
  - A negative dip in k(d) with its minimum at d* is the learned contraction→deceleration latency.
    The late-deceleration literature puts d* at about 20–60 s after the peak.
  - Attention moves while k ≈ 0: the model looked but did not use what it saw.
  - k scaling with amplitude: the model reads magnitude (see N5).
  - On `planted`, k must peak at d = 180 s.

**N2. Contraction-locked lag maps.**
- *Question.* Does the attention follow a contraction as it recedes into the past (a diagonal), or
  sit at fixed lags (horizontal bands)? Do the KL and forecast error rise after contractions?
- *Computation.*
  - From the raw-UP detections, assign each anchor its time since the latest peak, s (also since
    onset and since end). Bin s in 4 s bins from −60 s to +200 s.
  - Average per bin:
    - the per-head attention (M × L);
    - `source_kl_lag_map` (L);
    - `kld_per_t` and `kld_per_t_per_head`;
    - `mean_pred_gap`;
    - the per-τ level error of base and full.
  - The **tracking index** of head m is its attention mass within ±1 token of the peak token,
    ℓ = s/4, minus the same quantity for shuffled pseudo-events in the same recording.
  - Accumulate the sums in the collection pass (or a dedicated pass) as n_bins × M × L floats, so no
    new retention is needed.
- *Figure.* One (s, ℓ) heatmap per head, with the diagonal ℓ = s/4 overlaid; below it, KL(s), gap(s)
  and the base/full level error (s, τ).
- *Cost.* No extra forward if it rides on collection; otherwise one dense forward per segment. About
  80 lines, plus +10 in `_contraction_ages` to emit peak and end ages.
- *Interpretation.* A positive tracking index means the head is a contraction tracker. A gap(s) that
  rises after the peak while KL(s) stays flat means the source helps through the prior-side path, not
  by moving the latent.

**N3. Deceleration-locked attribution: does the model find the causing contraction?**
- *Question.* Before an FHR nadir, does attention and KL mass sit on the contraction that preceded
  it? Does the attended lag track the measured contraction→nadir latency event by event?
- *Computation.*
  - Detect decelerations on raw FHR in bpm (port `detect_decelerations`; the loader inverse comes
    from `traces.raw_signal_scales`).
  - Pair each nadir n with the latest contraction peak p in [n − 120 s, n]; the measured latency is
    Δ = n − p.
  - At the anchors whose horizon contains the nadir, t ∈ [n_tok − 30, n_tok − 1], read:
    - attention and lag-map mass at ℓ_p = t − p_tok, divided by the mean mass at the other lags
      (enrichment);
    - the model's implied latency, d̂ = 4(ℓ̄_attn + 1 + τ_n).
  - Controls: nadirs with no preceding contraction, and shuffled peaks.
- *Figure.* Enrichment violins per class, and a scatter of d̂ against Δ per event with the
  per-recording Spearman ρ.
- *Cost.* Same pass as N2. About 120 lines including the detector port.
- *Interpretation.* Enrichment > 1 together with ρ > 0 means the model localises the causal
  contraction and adapts its latency per event. Enrichment > 1 with ρ ≈ 0 means a fixed-latency
  heuristic.

**N4. Lag × horizon interventional delay map.**
- *Question.* At which UP→FHR delay d does removing UP cost forecast?
- *Computation.*
  - Run occlusion (§4.1, `baseline` arm) with `partition_width: 2` (19 bands). It already emits
    Δ(band, τ) in `occlusion_per_horizon.csv`.
  - Re-plot as a (ℓ, τ) heatmap, then sum along the diagonals d = ℓ_mid + 1 + τ to get Δ(d).
  - Run the same with KL in place of NLL, using the matched-vs-occluded `kld_tensor`.
- *Figure.* The (ℓ, τ) heatmap with iso-d diagonals, and Δ(d) with a recording interval.
- *Cost.* 20 arms × one single-anchor decode per segment. At cap 512 this is about 10k arm-encodes,
  roughly 10–20 min. About 40 lines of figure code on top of §4.1.
- *Interpretation.* This is the interventional counterpart of N1, on real data. The peak d* is the
  delay the forecast uses. The lag map alone cannot show this, because every d is smeared across H
  lags. On `planted`, Δ(d) must peak at 45 tokens; CFS needed exactly this readout to find its plant
  (+14.94 nats, `RESULTS.md`).

**N5. Lag and coupling against contraction strength (dose–response).**
- *Question.* Does the coupling scale with contraction magnitude and duration, or does it fire on
  timing alone?
- *Computation.*
  - Per detected contraction: prominence (σ, and mmHg via `raw_signal_scales`), duration (end − onset)
    and area.
  - For the anchors 0–148 s after the peak: KL_t, attention mass on the contraction's tokens
    [onset, end], the lag-map mass there, and gain.
  - Fit a per-recording slope against prominence (and against duration). Test with a Wilcoxon test on
    the slopes; stratify by class.
- *Figure.* Binned means by prominence quintile, one line per class; slope violins.
- *Cost.* Free on top of the N2 pass (key the accumulator by strength bin). About 50 lines.
- *Interpretation.* A positive slope means the model reads magnitude; a flat slope with positive N2
  tracking means it reads timing only. A steeper slope in acidosis/HIE is a clinically relevant
  coupling difference.

**N6. Per-head lag specialisation and head ablation.**
- *Question.* Do heads split the work (a contraction tracker, a fixed-latency head, a tonic or
  baseline head)? Which head carries the forecast benefit?
- *Computation.* For each head:
  - the N2 tracking index and the slope of attended lag against s;
  - event selectivity: mass on contraction tokens divided by mass on contraction-free tokens;
  - KL share (`kld_per_t_per_head`);
  - **ablation**: swap the head's `attended_source_heads[..., m, :]` for its null-arm value, re-pose
    through `posterior_head`, decode, and record Δ gap and ΔK. This is a ~40-line variant of
    `controls._attend_and_pose` (`controls.py:104`).
  - Also emit per-head, per-class lag histograms from the existing `lag_profile_per_head` vectors
    (`lag_high_kl` pools the heads).
- *Figure.* A heads × {tracking, selectivity, KL share, ablation Δnats} panel, plus per-head lag
  histograms by class.
- *Cost.* M = 4 ablation arms per batch. About 90 lines.
- *Interpretation.* A head with high tracking, a large ablation Δ and a modest KL share is the useful
  contraction reader. A head with a high KL share and an ablation Δ near zero is moving the belief
  without helping the forecast (compare the usefulness test in `lag_high_kl`).

## 6. Summary of work for a patch eval binding (lag slice only)

| Item | Kind | Lines |
|---|---|---|
| H1 wording rebind (consumers read `lag_axis.X` at call time) | hook | ~15 edits |
| H2/H3 override values | hook | YAML |
| H4 patch input/target seam | port | ~25 |
| `occlusion.collect_batch` + edit callback + validity live fraction | port | ~70 |
| `time_shift.score_pairs` | port | ~30 |
| Exclude `band_partition` (and its `spectral_skill` join) | N/A | binding |
| Peak/end/prominence ages in `_contraction_ages` | hook | ~10 |
| `detect_decelerations` port (needed by N3) | port | ~80 |
| Raw delay sweep (§4.4) | new | ~40 |
| N1–N6 | new | ~500 total |

Everything else in this slice (`lag_kl`, `attention`, `lag_clocks`, `lag_kld_scaled`, `lag_high_kl`,
`source_null`, `perm_control`, `coupling`, `events`, `lag_shape`, `lag_hist`) is as-is once
collection produces the CFS table and sidecar schema from the patch forward.
