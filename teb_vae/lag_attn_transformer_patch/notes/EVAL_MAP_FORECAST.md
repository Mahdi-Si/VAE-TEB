# Eval map: forecast, latent, clinical and visual analyses (patch cell)

Written 2026-10-03, read-only mapping of `teb_vae/lag_attn_cfs/eval` for `SeqVaeLagAttnTrfPatch`.
Scope: `analyses/{forecast, calibration, residual, spectral_skill, sufficiency, samples,
recording_traces, latent, trajectory, distributions, cross_subgroup, second_stage, time_to_delivery,
warmup}.py`, `traces.py`, `traces_html.py`, the `metrics.py` functions they consume, and the
matching `EVAL.md` / `FIGURE_GUIDE.md` sections. Paths below are relative to
`teb_vae/lag_attn_cfs/eval/` unless stated. Core infra, Captum and lag analyses are mapped elsewhere.

Status key: **as-is** (runs unchanged once the core hooks H0–H4 exist), **hook** (a small binding
value or a few lines), **port** (patch-specific rewrite, line estimate), **N/A** (meaningless here).

## 0. Facts measured on the tiny patch model (not assumed)

- Forward keys: `mu_prior/logvar_prior/mu_post/logvar_post (B,300,d_z)`, `mu_base/logvar_base/mu_full/
  logvar_full (B,240,30,2)`, `kld_per_t`, `kld_per_t_per_head`, `source_kl_lag_map`, `attn_weights`,
  `target_state`, `source_state`, `anchor_index`, `anchor_valid`, saturation scalars. No `persistence`
  key (shipped `persistence_residual: false`).
- Model attributes the CFS eval reads: `target_gate=None`, `target_forecast_shift=None`,
  `target_warmup_steps=None`, `c_y=33` (token width, **not** a channel count), `decoder_out_channels=2`,
  `anchor_ceiling=270`, `warmup_period=30`, `expected_anchors_per_sample=240`,
  `forecast_likelihood_kwargs() = {cell_mask: None, ar_coef: (2,)}`.
- **Absent** (AttributeError): `_build_forecast_target`, `TARGET_BLOCK_SPLIT`, `warm_tertile_id`,
  `target_warm_frac`, `source_block_warm_st/_ph`.
- The task inherits RWS `_build_target_streams` (reads `batch.fhr_st/fhr_ph`, width-checked against
  `c_y` → raises), `_build_raw_target` (returns raw `(fhr (B,4800), weight)`), and CRWS
  `forecast_rows` / CFS `input_stream_panels` (feature/raw-row page seams, wrong for a `(B,A,H,2)` block).
- Shipped density: `gaussian_nll`, `forecast_ar_residual: true`, `base_decode: mean`, `d_z: 64`,
  `max_lag: 37`. Shipped `target_summary_loc/scale = [0,0]/[1,1]` (provisional, `DESIGN.md` §3).

## 1. Core hooks every module below assumes (owned by the core-infra map, listed for dependency)

| ID | Site | Today | Patch |
|---|---|---|---|
| H0 | `metrics.model_inputs` 249; forward calls at `metrics.py:2263`, `samples.py:487`, `recording_traces.py:326`, `oracle.py:465` | `(y_st, y_ph, u_stream, target_features, weight)`; `model(y_st, y_ph, u_stream, anchor_phase=, anchor_stride=)` | `y_patch, u_patch, _, _ = task._build_forward_inputs(batch)` (dense outside a step); `fhr, weight = task._build_raw_target(batch)`; `target_features = model.summary_target(fhr, weight)` `(B,T,2)`; `model(y_patch, u_patch, 0, 1)`. One seam fixes all four call sites. |
| H1 | `model._build_forecast_target(target_features, anchors)` at `metrics.py:2276`, `oracle.py:616` | gathers kept channels at `a+1+τ` | 6 lines = `patch_target.py:182-189`: `summaries[b, anchors[:,:,None] + 1 + arange(H)]` → `(B,A,30,2)`. Put it on the model or the binding. |
| H2 | `evaluate_batch` 2534–2560 | `pred_gap_st/_ph` via `target_block_membership` (needs `TARGET_BLOCK_SPLIT`); `pred_gap_warm_*` via `warm_tertile_id`; `target_warm_frac`; `source_lag_warmth_per_sample` (needs `source_block_warm_*`) | Replace by `pred_gap_level = gap_per_channel[:,0]`, `pred_gap_variability = gap_per_channel[:,1]` (+ per-anchor pair); drop warm tertiles, warm frac, warmth. Also add `sq_error_{base,full}_{level,variability}` columns from `_per_channel_sq_error` (already computed, line 2799). ~20 lines. |
| H3 | `likelihood_structure_record` 474 (line 502 calls `target_block_membership`) | per-block mean φ for st/ph | `ar_coef_level`, `ar_coef_variability`; keep `scored_horizon_per_channel=[30,30]`, `ar_coef_per_channel`. ~8 lines. |
| H4 | eval override `load_fields`; `probe.REQUIRED_BATCH_FIELDS` 101 | CFS list incl. `fhr_st/fhr_ph/up_st/up_ph` | `[fhr, up, weight, guid, epoch, target, cs_label, bg_label, time_from_labor_onset, second_stage_onset]`; drop the four feature fields from the required list. |
| H5 | `lag_axis.COEFFICIENT_LAG_AXIS_LABEL` (used `traces.py:1185`, `traces_html.py:257`), `lag_axis.GROUP_DELAY_CAVEAT` (`recording_traces.py:483`) | "stored-coefficient time" + group-delay caveat | "lag (s, patch time)"; a patch token has no filter group delay, lag ℓ = source patch covering `[4(a−ℓ), 4(a−ℓ)+4)` s. Binding constant. |

## 2. Forecast target and baselines: CFS today → patch

| Item | CFS today (file:line) | Patch equivalent |
|---|---|---|
| Target stream | `_build_raw_target` → stored `fhr_st‖fhr_ph` `(B,T,80)` z-scored coefficients (`metrics.py:2263`) | `model.summary_target(fhr_raw, weight)` `(B,T,2)` = standardized `[level, variability]` per 4 s token (`patch_target.py:144`) |
| Gathered block | `_build_forecast_target`: `index_select(keep_index)` then steps `a+1+τ` → `(B,A,H,C_keep=76)` | `summaries[a+1+τ]` → `(B,240,30,2)` (H1) |
| Channel structure | 76 kept of 80 declared, ST block (32) / PH block (44) split by `TARGET_BLOCK_SPLIT`; names and bands from `band_channel_map_kept.csv` | 2 named channels: `0 = level`, `1 = variability`. No gate, no keep-index, no band, no ST/PH. |
| Scored horizon | per-channel `H_c` cell mask (fast PH channels scored < H) | `cell_mask=None`: both channels scored over all 30 steps |
| Warm-up | per-channel staircase `W'_c`; anchor floor `F = max(B−1, max_c(W'_c+d_c))` | none; `F = warmup_period = 30`, dense anchors 30..269 (240) |
| Mask | `forecast_mask(model.scored_weight(weight), …)` (`metrics.py:2156`) | identical; `scored_weight` is identity; a target patch is valid iff `weight[t] ≥ 1` (`VALID_THRESHOLD`) |
| AR(1) | `φ_c = tanh(a_c)`, 76 values | 2 values (`target_ar_logit` (2,)) |
| **persistence** | `metrics.baseline_forecasts` 891: coefficient at the last *observed* step ≤ `p = a + min(s_c,0)`, per kept channel, read from stored `target_features` | same function, unchanged, on `target_features = summary_target`: the anchor patch's own `[level, variability]` (last valid token ≤ a when the anchor patch is a gap). Equals the decoder's persistence input `_anchor_target_values` (`patch_target.py:149`) except that fallback. |
| **climatology** | exactly 0 = z-scored population mean (stats accumulated outside warm-up) | exactly 0 = `target_summary_loc`, i.e. the training-fold recording-weighted mean **only after** `summary_stats.py` constants are pasted. With the shipped `[0,0]/[1,1]` it means level = FHR stats mean (fine) but variability = log(1 z-unit) per sample (meaningless). Hook: refuse/flag climatology when loc/scale are the identity. |
| **segment_mean** | running mean over observed, warm steps `W'_c ≤ u ≤ p` | running mean over valid tokens `≤ a` (`target_warmup_steps=None` → all warm) |
| Baseline σ | `BASELINE_LOGVAR = 0` (σ = 1 z) under the model's φ | same; σ = 1 standardized unit = `scale_c` in summary units |

**Units** (stats from `traces.raw_signal_scales(config)` 241 → `(μ_FHR, σ_FHR)` for `fhr`; the
summary affine from `model_config.VAE_model.target_summary_{loc,scale}` in the dumped config, so all
of it is available offline):

- level: `bpm = (ŷ·scale₀ + loc₀)·σ_FHR + μ_FHR`; an error or σ: `Δbpm = Δŷ·scale₀·σ_FHR`; RMSE and
  CRPS scale the same way (AR(1) marginal σ from `forecast.marginal_variance` 648 first).
- variability: `rms Δ (bpm per 0.25 s) = (exp(ŷ·scale₁ + loc₁) − eps)·σ_FHR`; an error `Δŷ·scale₁` is
  a log-ratio, report it as the factor `exp(|Δ|·scale₁)`. No additive bpm error exists for it.
- nats stay in standardized units (a bpm density adds `log(scale₀σ_FHR)` per level cell; do not mix).

## 3. Module map

| Module (entry) | Computes | Reads | CFS-specific dependence | Status / patch action |
|---|---|---|---|---|
| `forecast` (`run_forecast_analysis` 937) | per-recording skill of base/full vs 3 baselines in MSE and nats (`build_skill_rows` 144), MAE/RMSE/bias (200), horizon curve `D(τ)` (251), anchor profile (326), overlay (`_emit_overlay` 1029) | `per_sample` `nll_{branch}_block`, `sq_error_*`, `abs/signed_error_*`; `results.horizon` sums; `per_anchor`; `record.geometry`, `record.likelihood_structure`; `retained` `target/mu_*/logvar_*/fhr_raw/up_raw/weight/anchor_index`; `band_channel_map_kept.csv` | overlay picks "middle scattering channel per band + one PH" (`overlay_channels` 526), labels from kept map (`channel_label` 583), shades past `H_c`; docstring and `NORMALISED_UNIT` claim no clinical unit exists | **as-is** for skill/horizon/profile (the 2-channel block flows through unchanged; without a kept map `overlay_channels` draws both channels). **hook** (~25 lines): labels `level`/`variability`; draw the level row in bpm; add `rmse_level_bpm`, `bias_level_bpm` rows from the per-channel columns (H2). Unit sentence in docstring/EVAL.md becomes wrong for level. |
| `calibration` (`run_calibration_analysis` 432) | PIT, 1/2/3σ coverage, CRPS, gain over homoscedastic MLE, logvar clamp fractions and clamp recommendation | `results.calibration` (built by `metrics.calibration_sums` 1696 / `calibration_report` 1818 over all scored cells, on the AR(1) innovation), `record.bounds`, `per_sample` `mean_logvar_full`, `logvar_full_*_frac` | pools all C_keep cells; "per coefficient" wording | **as-is** (pools 2 channels). Recommended **port** (~40 lines in `calibration_sums` + frame): keep the channel axis so PIT/coverage/CRPS are reported per channel; level CRPS also in bpm. Level and log-variability have different error shapes, so one pooled PIT hides which one is miscalibrated. |
| `residual` (`run_residual_analysis` 171) | RMS of `mu_full−mu_base` per scored cell; `delta_mu_rms`, `mu_post_prior_gap_rms`, roots once | `per_sample` `forecast_difference_sq`, `delta_mu_sq`, `mu_post_prior_gap_sq` (`RMS_METRICS` 79) | wording only ("wavelet modulus") | **as-is**. Optional hook: level forecast difference in bpm. |
| `spectral_skill` (`run_spectral_skill_analysis` 570) | pred_gap and MSE skill per clinical frequency band, recomposition to pred_gap, 5 coverage counts | vectors `gap_per_channel`, `sq_error_per_channel_{base,full}` (`_vector` 264); `band_channel_map_kept.csv` (`read_channel_maps` 125, `band_positions` 150 validates names against `CLINICAL_BANDS`) | the channel axis *is* a filter-frequency axis; join through the persisted kept map | **N/A** as a band analysis (no filter, no band). **port** (~100 lines) as `channel_skill`: the same two vectors are `(B,2)` already; report per channel (level, variability) gap, MSE skill vs persistence, recomposition `gap_level + gap_variability = pred_gap`, level in bpm. Keep the column name `pred_gap_variability` so `cross_subgroup`'s registered metric resolves. |
| `sufficiency` (`run_sufficiency_analysis` 327) | Δ_suff = D_base − D_oracle, oracle decoder on `target_state`, GUID-split fit | `oracle.run_oracle` (caches `target_state`, `target_features`, `anchor_index`; `oracle.py:409,465,616`); `model.horizon_core`, `model.decoder.out_channels` (= 2, verified) | none in the module; oracle needs H0 + H1; empty-cache shape uses `model.c_y` (`oracle.py:483`, 33 ≠ 2) | **as-is** after H0/H1; **hook** 1 line at `oracle.py:483` (`decoder_out_channels`). Cheap here: the probe emits 60 values per anchor, not 2280. |
| `samples` (`run_samples_analysis` 797, `render_pages` 421) | stratified, class-balanced and extreme-metric diagnostic pages, full + compact, via `lag_attn_rws.sample_page.build_diagnostic_figure` and the task's 3 seams (`page_seams` 340) | task, loader, `model_inputs`, forward, `kld_tensor`; seams `forecast_rows`, `forecast_extra_rows`, `input_stream_panels`; `per_sample` | page rows are CFS feature lanes (kept channels, warm-up staircase, ST/PH input heatmaps, `lag_attn_cfs/sample_page.py`); the patch task inherits CRWS `forecast_rows` (draws `(B,A,H,R)` raw rows) and CFS `input_stream_panels` | **port** (~150–200 lines): a patch `forecast_rows` seam (level forecast in bpm with ±σ band over the raw FHR, variability lane, tiled like `causal_forecast_rows`), `input_stream_panels` returning the two patch-value streams (`y_patch[...,:16]` flattened is the raw trace, so one row each), `forecast_extra_rows=()`. Selection, identity checks and manifest are **as-is**. See proposal P1. |
| `recording_traces` (`run_recording_traces_analysis` 387) | re-reads every segment of class-balanced recordings, per-anchor latent/KL/lag trace, segment summary, PDF + HTML dashboard | task, loader, forward keys `mu_*`, `logvar_*`, `kld_per_t(_per_head)`, `source_kl_lag_map`, `attn_weights`, `anchor_index/valid` (`gather_segment_traces` 212); `anchor_support`; raw `fhr/up` via `traces.attach_raw_signals` | none beyond H0 and the lag caveat H5; no forecast tensor is traced | **as-is** after H0/H5. Optional hook (~30 lines): add the level forecast at τ=0 and τ=H−1 in bpm as a `LinePanel` beside the raw FHR (the trace currently shows no forecast). |
| `latent` (`run_latent_analysis` 235) | per-dim KL spectrum, active dims, top-dim share, prior/post logvar, floor fractions, saturation, `prior_rate` | `results.latent_health`, `per_sample` `DIAGNOSTIC_COLUMNS` (67), `results.lag.num_heads`, verdicts, `record.bounds` | "coefficients" wording only | **as-is** |
| `trajectory` (`run_trajectory_analysis` 483) | within-segment profile from F (152), whole-delivery assembly on `t_abs = epoch + 60 + 4t` (204), delivery summary | `per_anchor` (`pred_gap`, `mc_pred_gap`, `mean_pred_gap`, `kld_per_t`), `record.geometry.anchor_floor` (125), `traces.absolute_seconds` | docstring ties F to the warm-up budget | **as-is** (patch token = 4 s = the CFS step; `trim_minutes: 1.0` matches `LOADER_TRIM_S`). Profile starts at step 30 (120 s), not ~134. |
| `distributions` (`run_distributions_analysis` 670) | per-segment densities with per-recording strips, by class and subgroup, no tests | `per_sample` `METRICS` (163): `sq_error_*`, `nll_full_block`, `mc/mean_pred_gap`, KL, `delta_mu_rms`, `mean_logvar_full`, `attention_entropy_nats` | axis label "normalised"; "H × C_keep" wording | **as-is**. **hook** (~6 lines): add `rmse_level_bpm` (from `sq_error_full_level`, H2) to `METRICS`; the docstring's "no bpm exists" reason no longer holds for level. |
| `cross_subgroup` (`run_cross_subgroup_analysis` 614) | Kruskal per metric → Holm across metrics → pairwise MWU + Cliff δ, per recording | per-recording CSVs via `METRIC_SOURCES` (123–195); a missing source is recorded, not raised | 3 `warmup` tertile entries; `spectral_skill` `pred_gap_variability` (the CFS *band*) | **hook** (~15 lines): a patch `METRIC_SOURCES` (drop the warmup three; point the variability entry at `channel_skill`, optionally add `pred_gap_level`) and a `sources=` pass-through, because the run function reads the module constant (`analyse_metrics` 305). Otherwise the Holm family silently shrinks by three missing sources. |
| `time_to_delivery` (`run_time_to_delivery_analysis` 654) | 0.5 h windows of hours-before-delivery, per-recording medians, Holm per window, pairwise, violins | `per_sample` `epoch`, `clinical_class`, `mc_pred_gap`, `source_conditioned_kl_raw` (`cohort.CLOCK_READOUTS` 115), `max_hours_before_delivery` | none | **as-is** (needs H4 fields) |
| `second_stage` (`run_second_stage_analysis` 744) | same on signed hours from second-stage onset, eligibility census, `capped` | `per_sample` `second_stage_onset`, `epoch`, `clinical_class`, clock columns; `cohort.add_second_stage_bins` 280, `second_stage_eligibility` 573 | none | **as-is** (needs H4 fields) |
| `warmup` (`run_warmup_analysis` 454) | pred_gap by warm-up tertile + recomposition, source-lag warmth, guards `anchors_per_sample`/`target_warm_frac`, staircase and budget figures (`write_budget_figures` 331 imports `lag_attn_cfs.causal_warmup`) | `pred_gap_warm_*`, `source_lag_warmth_frac_*`, `anchors_per_sample`, `target_warm_frac`, shards' warm-up vectors | entirely the causal filter bank's warm-up | **N/A**: a patch has no per-channel warm-up. The one transferable piece, `anchors_per_sample == 240` (dense) and `== 16` (train), is already covered by `metrics.anchor_geometry_verdict` 3586 and the `RESULTS.md` health check. Exclude from the binding. |

### Shared helpers

| File | What | Status |
|---|---|---|
| `traces.py` | `SegmentTrace`/`RecordingTrace`, `absolute_seconds` 219, `raw_signal_scales` 241 (μ/σ of `fhr`, `up` from the loader or the stats file), `physical_raw` 291, `attach_raw_signals` 330, `select_recordings` 440, `assemble_recording` 641, figures 1320/1484 | **as-is**; model-free. Imports `preflight.REQUIRED_TRIM_MINUTES` (1.0 = patch `trim_minutes`) and the lag label (H5). `raw_signal_scales` is the bpm source for every conversion in §2. |
| `traces_html.py` | plotly dashboard (`build_recording_dashboard` 203), raw FHR/UA panels + the trace panels | **as-is**; lag label H5 at 257. |
| `probe.py` | **not** a linear-probe analysis: the loader population census plus the forward-contract printout | core infra (H4 touches `REQUIRED_BATCH_FIELDS` 101, and `forward_contract` 779 needs H0). No linear probe exists in this eval; the closest is `oracle.py` (sufficiency). |
| `metrics.py` (consumed here) | `baseline_forecasts` 891, `masked_raw_error_sums` 1013, `branch_channel_scores` 1066, `horizon_block_sums` 1149 / `horizon_residual_sums` 1197 (pool channels, dim 3), `_per_channel_sq_error` 1477, `calibration_sums` 1696, `calibration_report` 1818, `latent_health` 3018 | **as-is** with H1–H3; every function is channel-count agnostic. Per-channel horizon curves would need the channel axis kept in the two horizon sums (~10 lines, P6). |

## 4. Clinical batch fields

| Field | Consumer | Patch task loads? | Provided by the eval? |
|---|---|---|---|
| `guid`, `epoch` | per-recording aggregation, anchor phase, every clock, traces axis | yes | yes |
| `target` (class code × weight) | `collect._class_column` 463 → `clinical_class` | no | yes, if the patch override lists it (H4) |
| `cs_label`, `bg_label` | identity columns, subgroup cuts (`collect.py:629`) | no | yes, via H4 |
| `source_file_basename` | `subgroup` (`labels.subgroup_of`) | added by `hdf5_dataset.py:1858` automatically | yes |
| `time_from_labor_onset` | `cohort.labor_onset_readout` 541 | no | yes, via H4 |
| `second_stage_onset` | `second_stage`, traces clocks | no | yes, via H4 |
| `fhr`, `up`, `weight` | model input and target; contraction detector; retained `fhr_raw/up_raw` | yes | yes |

The patch `load_fields` are `[fhr, up, weight, guid, epoch]`. The eval's own override is merged over
the checkpoint's resolved config and replaces `load_fields` wholesale, so the patch override supplies
the clinical five. The holdout shards (CFS causal k-fold) carry them. No patch task change is needed:
`_build_forward_inputs` reads only `fhr/up/weight`.

## 5. Documentation sections to rewrite

- `EVAL.md`: *forecast* (the no-unit paragraph, baselines on stored coefficients, warm steps),
  *calibration* (per-coefficient wording), *residual*, *distributions* (no-bpm reason), *trajectory*
  (floor from the warm-up budget), *samples* (fifteen-row page), *spectral_skill* (whole section →
  `channel_skill`), *warmup* (drop), *cross_subgroup* table (warmup rows, spectral row), the
  "percentage is budget-local" and "frequency statement has no timing half" bullets of *How the output
  is misread* (H·C = 60, and the patch target has a timing half: see P2).
- `FIGURE_GUIDE.md`: *What the model predicts*, *Warm-up, bands and geometry*, *Geometry reference*
  (C_keep → 2, F → 30, no H_c), `forecast/forecast_overlay.pdf`, `forecast/horizon_skill.pdf`,
  `calibration/*`, `spectral_skill/spectral_skill_bands.pdf` (→ channel skill), the three
  `warmup/*.pdf` entries (drop), `recording_traces/*` (lag caveat).

## 6. Proposals: raw-signal forecast and latent analyses the patch representation enables

Each is offline unless stated. "Retained" means `eval_config.caps.waveforms` retention
(`collect.RETAINED_QUANTITIES` 204), which already carries `target`, mean-decoded `mu/logvar_{base,full}`,
`fhr_raw`, `up_raw`, `weight`, `anchor_index` at `(N,240,30,2)`. At C = 2 this costs about 115 KB per
sample, against about 4.4 MB per sample in CFS, so the cap can be raised to hundreds of segments.

**P1. Raw-trace forecast page (level in bpm over the raw FHR).**
- *Question:* what does the model forecast, in clinical units, against the trace a clinician reads?
- *Computation:* for a few anchors per segment (for example every 15th, the training tiling), map
  `mu_{base,full}[a,:,0]` to bpm (§2) and draw it on the 4 s grid `[4(a+1+τ), 4(a+2+τ))` over the 4 Hz
  raw FHR, with the AR(1) marginal ±1σ/±2σ band (`forecast.marginal_variance`). Add a variability lane
  as rms-Δ bpm against the realised per-patch rms-Δ, and UP below it with `events.detect_contractions`
  shading. Use the class-balanced selection from `samples`. This is also the `samples` port (P1 is the
  patch `forecast_rows` seam), and an offline version reads the retained arrays.
- *Figure:* `forecast/raw_forecast_page_<guid>.pdf`, 3 rows (FHR + forecasts, variability, UP), one
  column per class.
- *Cost:* about 150 lines; seconds offline.
- *Interpretation:* an illustration, not evidence. It shows whether the full branch bends the level
  forecast after a contraction where base does not, and whether the band widens over gaps.

**P2. Deceleration-centred forecast skill: does UP history let the model forecast decels?**
- *Question:* is the source gain concentrated on anchors whose horizon contains a deceleration, and
  is that gain larger when a contraction preceded the anchor?
- *Computation:* run a deceleration detector on the raw FHR in bpm. `lag_attn_rws/eval/events.py`
  has `detect_decelerations`, which must be moved into layer-0 `events.py`, because production modules
  may not import `lag_attn_rws.eval`. In the collection pass, add per-anchor
  `seconds_to_next_decel_nadir` and `decel_depth_bpm`, beside `seconds_since_contraction`. Offline,
  stratify anchors by "a nadir lands in `[4(a+1), 4(a+H)]`" × "a contraction onset in
  `[a−event_lag_window, a]`". Per stratum and per recording, compute `pred_gap`, full and base level
  RMSE in bpm restricted to the steps around the nadir (retained arrays, or a per-anchor
  nadir-window level error added in the pass), and the persistence level RMSE. Compare count-matched
  decel and control anchors from the same recordings, as `events` already does.
- *Figure:* skill vs `seconds_to_nadir` (−H·4 … 0 s) for base, full and persistence, faceted by
  contraction-preceded or not, per class; and the nadir-aligned mean forecast vs truth in bpm.
- *Cost:* about 200 lines, plus one detector run per segment in the pass (cheap, numpy).
- *Interpretation:* full beating base before the nadir, only in the contraction-preceded stratum,
  is the forecastable UP→FHR coupling the model exists to find. Gain only after the nadir is
  persistence of a decel already in progress. This revives `REMOVED_READOUTS["deceleration_skill"]`
  (`analyses/events.py:93`), which CFS dropped because coefficients have no bpm.

**P3. Latent vs raw-signal descriptors (what the latent encodes).**
- *Question:* which clinical descriptors of the recent raw trace do `mu_prior`, `mu_post` and
  `delta_mu` carry linearly?
- *Computation:* per anchor, from the raw window before `a` (60 s and 5 min), compute: FHR baseline
  (median bpm), STV (mean |Δ| of 3.75 s epoch means, Dawes–Redman style; patch level diffs are the 4 s
  version), LTV (range of 1 min means), min-minus-baseline (decel depth), FHR gap fraction, UP mean,
  UP peak and contraction count. Inputs are `mu_prior`/`mu_post` per anchor from the
  `recording_traces` full npz (already saved with raw `fhr/up`), or a dedicated capped pass. Fit ridge
  regressions with GUID-grouped 5-fold CV, giving CV R² per descriptor and target `{mu_prior, mu_post,
  delta_mu}`, and Spearman of `‖delta_mu‖` and `K_t` vs each descriptor.
- *Figure:* grouped bar chart of CV R² (descriptor × latent), plus `‖delta_mu‖` vs UP-peak and
  vs decel-depth hexbins.
- *Cost:* about 150 lines; CPU minutes.
- *Interpretation:* high R² for baseline from `mu_prior` means the level sits in the latent (the
  `sweep_persistence` prediction in `RESULTS.md`). `delta_mu` predicting UP descriptors but not FHR
  ones means the source writes contraction state into the latent. `delta_mu` predicting decel depth
  is the coupling read in latent space. A probe, not an attribution: it is linear, and the descriptors
  correlate with each other.

**P4. Variability-channel calibration against raw STV.**
- *Question:* does the forecast of the variability channel track a clinically defined variability,
  and is its uncertainty calibrated?
- *Computation:* per anchor, back-transform the predicted variability over the horizon to rms-Δ bpm
  (§2). Compare it with the realised rms-Δ, and with STV/LTV computed on the raw FHR over the same
  4H = 120 s window. Run PIT and coverage on channel 1 alone (needs the per-channel calibration port
  in §3). Draw a reliability curve (predicted-variability decile vs realised mean, with the 0.25 bpm
  quantisation floor and `variability_eps` marked) and the Spearman ρ of predicted vs realised STV
  per recording.
- *Figure:* reliability plot plus PIT histogram for channel 1, per class.
- *Cost:* about 100 lines; retained arrays only.
- *Interpretation:* the channel is `log(rms Δ + eps)` at 4 Hz inside 4 s, so it is dominated by
  beat-to-beat noise and the 0.25 bpm monitor resolution. Poor ρ against clinical STV/LTV says this
  target is not the clinically meaningful variability (consider a longer-window variability target).
  A PIT ∪ shape on this channel alone says σ is too small where the quantisation floor bites.

**P5. Effect of FHR signal loss (`missing` tokens) on the latent and the forecast.**
- *Question:* how do gaps move the latent and the scores, and does the model hallucinate across them?
- *Computation:*
  - (a) Observational, offline: per anchor, the fraction of `missing` target tokens in the last 15
    and 37 patches (from `weight`; add `history_gap_frac` and `steps_since_gap` to `per_anchor` in
    the pass). Bin `K_t`, `mean_logvar_prior`, `nll_full_block` and `pred_gap` by them, per recording.
  - (b) Interventional, capped pass: on clean segments, set `weight = 0` over g ∈ {1, 4, 15, 37}
    tokens ending at d ∈ {0, 15} before the anchor. Keep the target and the scored future unchanged.
    Re-forward and measure Δ`mu_prior` norm, Δ`logvar_prior`, ΔNLL_base/full, Δ`K_t` and the
    level-forecast shift in bpm. With `source_validity: finite`, UP stays visible across an FHR gap,
    so also compare the full branch with UP zeroed over the same window.
- *Figure:* response curves of each readout vs gap length, one line per gap-end distance, base vs full.
- *Cost:* (a) about 60 lines, free. (b) about 120 lines, plus 8 extra forwards per segment on about
  200 segments (minutes on one GPU).
- *Interpretation:* `logvar_prior` and decoder σ should widen with gap length, with NLL degrading
  smoothly. A `K_t` jump at gaps means the source fills in missing FHR, which supports the
  `fhr_weight` validity arm. A flat response means `missing` is untrained. The committed fixtures
  have no gaps (`DESIGN.md` §4), so check the training gap rate first.

**P6. Per-channel horizon skill in bpm and the forecastable lead time.**
- *Question:* up to what lead time does the level forecast beat persistence, in bpm, and does the
  source extend it?
- *Computation:* keep the channel axis in `horizon_residual_sums` and `horizon_block_sums` (dim 3,
  about 10 lines in `metrics.py`), and add per-τ per-channel persistence sums. Offline, compute
  `RMSE_level(τ)` in bpm for base, full and persistence, and the crossover lead `τ*` where
  `RMSE_full ≥ RMSE_persist`. Bootstrap `τ*` over recordings, per class.
- *Figure:* `forecast/horizon_skill_bpm.pdf`: RMSE (bpm) vs lead (4–120 s), three lines, with the
  `τ*` CI marked; a second panel for variability as a ratio factor.
- *Cost:* about 60 lines; free once the pass keeps the axis.
- *Interpretation:* `τ*` is the first clinically readable number this cell can report: "the model
  forecasts FHR level to ±x bpm up to y s". A full-branch `τ*` beyond base is the source's
  contribution in time rather than in nats.
