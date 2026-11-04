# EVAL_MAP_FORECAST: forecast, latent and clinical analyses on the patch cell

This note covers how the forecast, latent, clinical and visual half of `teb_vae/lag_attn_cfs/eval`
reads on `SeqVaeLagAttnTrfPatch`: the target and baselines, the clinical units, the shared modules,
and the proposals P1–P6 that the analysis modules cite. `../eval/EVAL.md` describes the built
analyses.

## 1. Forecast target and baselines: CFS and patch

| Item | CFS | Patch |
|---|---|---|
| Target stream | Stored `fhr_st‖fhr_ph` `(B,T,80)` z-scored coefficients | `model.summary_target(fhr_raw, weight)` `(B,T,2)` = standardized `[level, variability]` per 4 s token |
| Gathered block | `_build_forecast_target`: kept channels at steps $a+1+\tau$ → `(B,A,H,76)` | `summaries[a+1+τ]` → `(B,A,30,2)` |
| Channel structure | 76 kept of 80 declared, ST block / PH block split by `TARGET_BLOCK_SPLIT` | 2 named channels: 0 = `level`, 1 = `variability`. No gate, no keep-index, no band. The eval view sets `TARGET_BLOCK_SPLIT = 1`, so the `*_st` / `*_ph` columns mean level / variability |
| Scored horizon | Per-channel cell mask | `cell_mask=None`: both channels are scored over all 30 steps |
| Warm-up | Per-channel staircase | None; $F$ = `warmup_period` = 30, dense anchors 30..269 (240) |
| Mask | `forecast_mask(model.scored_weight(weight), …)` | Identical; `scored_weight` is the identity, and a target patch is valid when `weight[t] ≥ 1` |
| AR(1) | $\phi_c = \tanh(a_c)$, 76 values | 2 values (`target_ar_logit` `(2,)`), when `forecast_ar_residual` is on |
| Persistence | `metrics.baseline_forecasts`: the coefficient at the last observed step | The same function on `summary_target`: the anchor patch's own `[level, variability]`, or the last valid token before it when the anchor patch is a gap |
| Climatology | Exactly 0 = the z-scored population mean | Exactly 0 = `target_summary_loc`. This is the training-fold mean only after the `summary_stats.py` constants replace the identity placeholders; until then `channel_skill` drops climatology |
| Segment mean | Running mean over observed, warm steps | Running mean over valid tokens up to $a$ |

## 2. Clinical units

`eval/raw.py::SummaryUnits` converts the standardized summaries. $\mu$ and $\sigma$ are the loader
FHR statistics (`traces.raw_signal_scales`); `loc` and `scale` are `target_summary_loc` and
`target_summary_scale`.

- **Level:** $\mathrm{bpm} = (\hat y \cdot \mathrm{scale}_0 + \mathrm{loc}_0)\,\sigma + \mu$. An
  error or a standard deviation scales as $\Delta\mathrm{bpm} = \Delta\hat y \cdot \mathrm{scale}_0
  \, \sigma$.
- **Variability:** $\mathrm{rms}\,\Delta$ (bpm per 0.25 s) $= \big(\exp(\hat y \cdot \mathrm{scale}_1
  + \mathrm{loc}_1) - \varepsilon\big)\,\sigma$. A difference $\Delta\hat y \cdot \mathrm{scale}_1$
  is a log-ratio; report it as the factor $\exp(|\Delta \hat y| \cdot \mathrm{scale}_1)$, never as
  bpm.
- **Nats** stay in standardized units. A bpm density would add
  $\log(\mathrm{scale}_0\,\sigma)$ per level cell.

## 3. Shared modules

| Module | Status on the patch cell |
|---|---|
| `forecast`, `residual`, `latent`, `trajectory`, `distributions`, `time_to_delivery`, `second_stage`, `sufficiency`, `recording_traces` | Reused. The two-channel block flows through unchanged. The trajectory profile starts at step 30 (120 s) |
| `calibration` | Replaced in place by the patch port (per-channel PIT, coverage and CRPS; variability against raw STV) |
| `cross_subgroup` | Replaced in place: it reads the patch `METRIC_SOURCES` (no `warmup` tertiles; `channel_skill` in place of `spectral_skill`) |
| `samples` | Ported: the patch page seams |
| `spectral_skill` | Excluded; `channel_skill` replaces it |
| `warmup` | Excluded: a patch has no per-channel warm-up. `anchors_per_sample` and `target_warm_frac` stay guarded by the headline and `metrics.anchor_geometry_verdict` |

The eval override (`eval/configs/eval_overrides.yaml`) lists the clinical batch fields that these
modules read: `target`, `cs_label`, `bg_label`, `time_from_labor_onset` and `second_stage_onset`,
beside `fhr`, `up`, `weight`, `guid` and `epoch`. The override replaces `load_fields` wholesale. The
patch task reads only `fhr`, `up` and `weight`.

**Open items (not implemented).** These shared modules still use CFS wording or units on the patch
cell:
- `forecast`: the channel labels and the overlay are not patch-specific, and there are no
  `rmse_level_bpm` or `bias_level_bpm` rows.
- `distributions`: no `rmse_level_bpm` metric.
- `recording_traces`: no level-forecast panel in bpm.
- `oracle.py`: the empty-cache target shape uses `model.c_y` (33) in place of
  `decoder_out_channels` (2).
- `metrics.horizon_block_sums` and `horizon_residual_sums` pool the channel axis, so there are no
  per-channel horizon curves from the collection pass.

## 4. Proposal index

| # | Question | Implemented in |
|---|---|---|
| P1 | What does the model forecast, in clinical units, against the raw trace? | `samples` (the patch page) |
| P2 | Is the source gain concentrated on anchors whose horizon contains a deceleration? | `decelerations` |
| P3 | Which clinical descriptors of the recent raw trace does the latent carry linearly? | `latent_descriptors` |
| P4 | Does the variability channel track a clinical variability, and is it calibrated? | `calibration` (the patch port) |
| P5 | How does FHR signal loss move the latent and the forecast? | `signal_loss` |
| P6 | Up to what lead does the level forecast beat persistence, in bpm? | `channel_skill` |
