# Evaluating the patch-token lag-attention VAE

Short by design. `SeqVaeLagAttnTrfPatch` is the transformer-CFS cell over raw 4 s patches, so it is
evaluated by the shared CFS pipeline (`teb_vae/lag_attn_cfs/eval`) through one `ModelBinding`, not
by a copy of it. This page says only what is true of this package: what it supplies, how the shared
columns read on this model, and the fourteen analyses it adds or ports. The rest is in:

| Document | What it covers |
|---|---|
| `teb_vae/lag_attn_cfs/eval/EVAL.md` | **The contract**: the run, the output layout, the configuration reference, every shared analysis, the verdicts, the guard recovery table. |
| `teb_vae/lag_attn_cfs/eval/FIGURE_GUIDE.md` | Every shared figure and how it is misread. |
| `teb_vae/lag_attn_cfs/eval/ATTRIBUTION.md` | The shared Captum pass (`attribution`), which runs here unchanged. |
| `../notes/EVAL_PLAN.md` | The decisions D1–D9 this package was built under, and the registry. |

## What this package supplies

| File | What it carries |
|---|---|
| `view.py` | `SeqVaeLagAttnTrfPatch`: the training class's name and weights, no new parameter, the CFS forward `(y_patch, empty, u_patch, φ, S)`, and three eval constants true by construction (`TARGET_BLOCK_SPLIT = 1`, `target_warm_frac = 1.0`, all-True source warmth). `SeqVaeLagAttnTrfPatchEvalTask`: every shared builder returns patch streams; the target is `summary_target(fhr, weight)`. |
| `binding.py` | `TRF_PATCH_BINDING`: the classes, 27 geometry keys, the encoder disclosure (+ `raw_per_step`, `source_validity`), the registry, the exclusions, the headline scalars, the raw-length shard guard, target fields `fhr, up`, the probe's field list and the causality wording. The only place the registry is written. |
| `raw.py` | The raw-signal substrate (D6): raw → patch → forward with edits, units (`SummaryUnits`), the resting-tone estimator, the occlusion fills, segment draws, `RawReadout` for Captum over raw `(B, T, R)`, and re-exports of the shared selection and event detectors. |
| `figures.py` | Three helpers on top of the shared drawing primitives: `spans_decades`, `share_axis`, `stacked_page`. |
| `analyses/` | Ports: `samples`, `occlusion`, `time_shift`, `calibration`. New: `channel_skill`, `latent_descriptors`, `event_locked`, `decelerations`, `signal_loss`, `delay_map`, `raw_shift`, `impulse_response`, `raw_attribution`, `fhr_drivers`. |
| `configs/eval_overrides.yaml` | The production delta: the CFS causal holdout split, `load_fields` without the ST/PH blocks, `clock_margin_min_nats: null`, the transformer-CFS `occlusion_bands` (inside `max_lag` 37), and a cap per new analysis with its production cost. |
| `configs/planted_overrides.yaml` | The planted-delay instrument: the committed 8-segment planted shard (holdout = training shard, by design), CPU-sized caps, bands `near [0, 14]`, `planted [15, 44]`, `far [45, 60]`. |
| `run.py`, `verify.py` | The command lines. Both delegate to the shared ones; `run.py` owns the parser so `--only`/`--skip` name this registry. `verify` emits no cross-cell table. |

**Registry, in run order.** The shared table analyses, with `calibration` replaced in place; the CFS
cell's extras with `samples`, `occlusion` and `time_shift` replaced in place (`samples`,
`recording_traces` and `attribution` draw unlabelled recordings under the class `unlabelled`); the
ten new analyses, table readers first and IG passes last; `cross_subgroup` over this registry's
metric sources. **Excluded:** `warmup` and `spectral_skill` (a filter-bank warm-up and frequency
channels a patch does not have) and the unskippable `band_partition` (an ST/PH channel map).
`python -m teb_vae.lag_attn_transformer_patch.eval.run --help` lists the registry.

## Running

From the repository root:

```bash
# Production holdout (repoint the REPOINT_ME paths in configs/eval_overrides.yaml first).
python -m teb_vae.lag_attn_transformer_patch.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt
python -m teb_vae.lag_attn_transformer_patch.eval.verify <out>/eval_results/summary.json
python -m teb_vae.lag_attn_transformer_patch.eval.verify --runs <dir-of-runs> --out RESULTS_arms.md

# The planted instrument (local; about 9 min on a GTX 1660 Ti, 5 of them event_locked).
python -m teb_vae.lag_attn_transformer_patch.trainer --config teb_vae/lag_attn_transformer_patch/configs/planted.yaml
python -m teb_vae.lag_attn_transformer_patch.eval.run \
    --checkpoint output/teb_vae_trf_patch_planted/<run>/model_checkpoints/<name>.ckpt \
    --overrides teb_vae/lag_attn_transformer_patch/eval/configs/planted_overrides.yaml

# Re-run analyses offline against a finished directory (no checkpoint; tables are reused).
python -m teb_vae.lag_attn_transformer_patch.eval.run --output-dir <out> --only channel_skill
```

Or with no command line: `run.py` and `verify.py` each carry a `RUN_ARGS` dictionary at the bottom.
Fill `checkpoint` (or a finished `output_dir`) and press the IDE's Run button; a flag on the command
line wins, and the console prints where each value came from. Run settings belong in the override
file, which is dumped into the run directory, not in `RUN_ARGS`.

An offline re-run of an analysis that re-forwards (every new one except `channel_skill`) records a
skip: it needs `task` and `loader`, which only a checkpoint run builds.

## What resolves to the shared pipeline

Everything not listed above: the preflight guards, the probe, the collection pass, the readouts, the
verdicts, the headline registry, the sanity block, the figure seam and the gate. The shared table
analyses run unchanged on the patch tables.

**Shared edits made for this package.** Each leaves CFS behaviour unchanged (its default is the
previous code path). After any further shared edit run both CFS eval suites
(`pytest teb_vae/lag_attn_cfs/tests -k eval -m "not slow"` and the same on
`teb_vae/lag_attn_transformer_cfs/tests`).

| Where | Edit | On a CFS binding |
|---|---|---|
| `binding.py` | Five `ModelBinding` fields: `replaced_analyses`, `target_fields`, `shard_guard`, `required_batch_fields`, `causality_text` | Defaults: `{}`, `None` ×3, `{}` |
| `run.py` | `replaced_analyses` swapped in at the same position (an unknown name raises); `unskippable_for(binding)` drops an excluded unskippable step | Nothing excluded, nothing replaced |
| `preflight.py` | `check_target_normalized(fields)`; `shard_guard or check_declared_widths`; `causality_text` replaces only `statement`, `lag_axis`, `group_delay_seconds` | `TARGET_FIELDS`, `check_declared_widths`, the shared wording |
| `preflight.objective_channel_weights` | `{"applies": False}` for a model with no `target_channel_weight` | The CFS cells carry the buffer |
| `metrics.batch_size_of` | Falls back to `fhr` when `fhr_st` is absent | `fhr_st` is present |
| `metrics.target_block_membership` | `arange(decoder_out_channels)` in place of `arange(c_y)` | Equal on the feature cells |
| `metrics.evaluate_batch` | No `warm_tertile_id` → NaN `pred_gap_warm_*` columns | The CFS models carry it |
| `analyses/cross_subgroup.py` | Sources from `context.metric_sources`, default `METRIC_SOURCES` | The default |
| `events.py` | `detect_decelerations` on bpm FHR, ported from `lag_attn_rws` (which the layer rule forbids importing) | Nothing in CFS calls it |

D4 is the one model-side edit: `PatchSummaryTarget._build_forecast_target` is the `a + 1 + τ`
gather that both `compute_loss` and the shared collection call, so an evaluated `nll_*` is the
training loss by construction.

## How the shared columns read on this model

| Column or readout | Reads as |
|---|---|
| `*_st` / `*_ph` (`pred_gap_st`, `ar_coef_mean_ph`, …) | **level** / **variability** (`TARGET_BLOCK_SPLIT = 1`: channel 0 is level). |
| `pred_gap_warm_{lo,mid,hi}` | NaN. There is no warm-up partition, and no split is fabricated. |
| `target_warm_frac`, `source_lag_warmth_frac_*` | 1.0 by construction (a headline guard here, beside `anchors_per_sample` = 240). |
| Every block score in nats | H × 2 = 60 standardized summaries per anchor. Not comparable with any CFS cell. |
| `pred_gap` (per-sample column) | The **train-path** gap (base at the prior mean, full at one draw). The gate's gap is `pred_gap_mc_nats`; the two can differ in sign. |
| `source_null`, `coupling_minus_clock_nats`, `kl_lag_profile_clock_excess` | There is no availability clock. The null arm is zero K/V content, where the attention is its learned per-lag key bias, so the "clock" is **the KL a flat UP elicits** and the excess is KL beyond a flat UP. `clock_margin_min_nats: null` keeps `coupling_exceeds_availability_clock` INCONCLUSIVE until a patch run measures its own spread. |
| `time_shift` | A **swap control**: the source is the nearest non-overlapping segment of the same recording, at least one stored segment away. It asks whether timing matters at the recording scale. It is not a lag probe; `raw_shift` is. |
| `occlusion_delta_<band>_nats` | The `baseline` (resting-tone) arm. The other arms are `…__zero_nats` and `…__missing_nats`. `announcement_invariance` is NaN. |
| Lag ℓ, `kl_lag_compensated_seconds` | Patch time: lag ℓ is the UP patch whose last sample is exactly 4ℓ s before the anchor patch's, with no group delay. A delay d informs lags [d − H, d − 1]. The shared figures still label this axis "stored-coefficient time" (Lean limits). |
| `causality.lag_support` margin | Negative (−7 at the shipped geometry, −30 on the planted one), so the support-corrected and untruncated profiles differ from the raw one and are the ones to read. |

## The analyses

Each has the shared protocol (`run_<name>_analysis(context, *, eval_config, output_dir, probe)`,
failure-isolated), writes under `<out>/eval_results/<name>/`, and states its caps in
`eval_overrides.yaml` under `caps.<name>_*`. Costs are on one GTX 1660 Ti at the production caps.
Unless a section says otherwise, rows reduce per recording, then across recordings, and intervals
resample recordings.

### `samples` (port)

**Question.** What does the model forecast, in clinical units, against the trace a clinician reads?
**Method.** The shared selection, re-forward and page variants, with the task's three page seams
swapped for patch ones: abutting level-forecast tiles in bpm with the AR(1) marginal ±2σ band over
raw FHR, a variability lane (rms-Δ bpm per 0.25 s, log axis, 0.25 bpm resolution and `eps` floor
marked), UP with contractions shaded, the two patch input streams, then the shared latent/KL/lag rows.
**Outputs.** The shared `samples/` layout (`stratified/`, `by_class/`, the metric extremes,
`sample_pages.csv`). No headline.
**Reading.** `full` is one posterior draw; `base` is decoded at the prior mean. **Misread:** a tile
is one latent's 120 s forecast of 4 s patch means, not a beat-to-beat trace.
**Cost.** The shared pages' (`caps.pages` 24, `pages_per_class` 10); 18 s on the planted run.

### `occlusion` (port, D5)

**Question.** What is the forecast worth without the UP at a band of lags?
**Method.** Raw UP of the band's tokens is replaced and re-patchified, under three arms:
`baseline` (**primary**: the segment's resting tone, "no contraction here"), `zero` (the z-mean,
CFS-comparable, above resting tone) and `missing` (NaN → the learned `missing` token). Gaps stay
gaps. The shared pairing, per-horizon scoring, frames and figures run unchanged.
**Outputs.** `occlusion_{per_segment,per_recording,per_horizon,summary,clocks}.csv`, two figures.
Headline: the shared `occlusion_peak_band*` (baseline arm) plus `occlusion_zero_peak_band(_delta_nats)`.
**Reading.** Positive = the forecast needed that UP. The live fraction is read off the validity
channel. **Misread:** one anchor per segment; a band delta whose interval spans 0 says nothing.
The `missing` arm is flagged `reliable: False`.
**Cost.** 512 segments × 13 arms (reference + 3 arms × 4 bands), about 3 ms per arm-segment: under a minute.

### `time_shift` (port)

**Question.** Is the coupling about the UP at the right moment, or about this patient's UP at any moment?
**Method.** No code change was needed: the shared `score_pairs` runs on the eval view (patch
inputs, CFS forward, the D4 gather). ΔK and the mean-decoded Δgap, verdict `coupling_is_time_specific`.
**Outputs.** The shared `time_shift_*.csv` and `time_shift_control.pdf`. No headline of its own.
**Misread:** a recording-scale timing control, never a lag readout. It skips on any shard with one
segment per recording (the planted one).
**Cost.** Two mean-decoded forwards per pair, `caps.time_shift` 512.

### `calibration` (port, replaces the shared one in place)

**Question.** Is the predicted uncertainty right **per channel**, and does the variability channel
track a clinical variability measured on the raw trace?
**Method.** The shared pooled PIT/coverage/CRPS unchanged; the same arithmetic per channel on the
retained segments (level CRPS also in bpm); predicted vs realised rms-Δ against a Dawes–Redman-style
raw STV (3.75 s epochs, bpm), as decile reliability and a per-recording Spearman ρ.
**Outputs.** The shared files plus `calibration_channels.csv/.pdf`, `calibration_channel_pit.csv`,
`variability_reliability.csv/.pdf`, `variability_per_recording.csv`. Headline: `calibration_level_coverage_2sigma`,
`calibration_variability_coverage_2sigma`, `calibration_level_crps_bpm`, `calibration_variability_stv_spearman`.
**Misread:** the per-channel reading uses the retained **mean-decoded** pair, the pooled one a
posterior draw; they differ by design. Below the 0.25 bpm and `eps` floors the variability target
measures quantisation, not physiology. The per-recording ρ is the statistic, not the bins.
**Cost.** No forward; retained waveforms (`caps.waveforms` 512, ~170 MB).

### `channel_skill` (new, replaces `spectral_skill`)

**Question.** How much of the source gain does each channel carry, how good is each in clinical
units, and up to what lead does the level forecast beat persistence?
**Method.** Per channel, whole split: the collection's `gap_per_channel` and squared errors, per
recording, bootstrapped; asserted to recompose to `pred_gap`. Per channel and lead (retained):
RMSE of base, full, persistence, segment mean and climatology; skill `1 − MSE/MSE_persist`; the
**crossover lead** where the level skill changes sign, bootstrapped and by class.
**Outputs.** `channel_skill_{per_recording,channels,horizon,crossover}.csv`, two figures.
Headline: `channel_skill_pred_gap_{level,variability}_nats`, `…level_rmse_{full,persistence}_bpm`,
`…level_crossover_s`, `…level_skill_vs_persistence_4s`, `…level_frac_leads_beating_persistence`.
**Misread:** `pred_gap_level/_variability` decompose the **train-path** `pred_gap`, not
`pred_gap_mc_nats`. A null crossover means no sign change (`kind` says whether it always or never
beats). The variability RMSE is a factor on rms-Δ, never bpm. Climatology is dropped while the
summary constants are the identity placeholders (`summary_constants_identity`).
**Cost.** No forward; seconds.

### `raw_attribution` (new)

**Question.** Which raw UP samples, at 0.25 s resolution over the lag window, move `kld`, the
mean-decoded `pred_gap` and the UP-driven level shift `delta_level`, and is that where the model's
own `source_kl_lag_map` puts it?
**Method.** IG from the source-null baseline (raw UP = 0), FHR held, through `RawReadout`; one
`LayerIntegratedGradients` on the `source_adapter` input gives both patch IG and raw IG exactly
(`patchify` is affine). Also a key/value split through `lag_attn.W_k/W_v`, value vs delta channels,
within-patch position, and each head's own KL against its attention.
**Outputs.** `raw_attribution_{rows,recordings,kl_gain}.csv`, `raw_attribution_vectors.npz`,
`…_{lag_profile,mechanism,within_patch,checks}.pdf`, `maps/`. Headline: `raw_attr_kld_lag_corr`,
`raw_attr_kld_centroid_s`, `raw_attr_completeness_max`.
**Reading.** High `lag_corr` / low `lag_js` = the attention's lag is where UP content moves the KL.
`after_anchor_max_abs`, `outside_window_max_abs` and `validity_channel_max_abs` must be exactly 0.
**Misread:** model sensitivity along a path from zero UP, not a causal effect. Zero is the z-mean,
above resting tone, so resting stretches carry attribution too (`event_locked` uses a tone baseline).
**Cost.** 24 segments × 4 anchors × 10 IG calls at 128 steps: ~6 min.

### `fhr_drivers` (new)

**Question.** How far back, and where, does raw FHR history drive the target-only branch (base level,
base variability, `mu_prior`)? Which FHR history makes the UP matter (`src_effect` = K(fhr, up) − K(fhr, 0))?
**Method.** IG over raw FHR through `RawReadout` (UP held, validity fixed), from `pop_mean` (raw 0:
level included) and `seg_median` (flat at the segment's causal median: pattern only). `src_effect`
under `pop_mean` only.
**Outputs.** `fhr_drivers_{rows,recordings}.csv`, `fhr_drivers_vectors.npz`,
`fhr_drivers_{history,checks}.pdf`, `maps/`. Headline: `fhr_drivers_base_level_lag90_s`,
`fhr_drivers_src_effect_lag50_s`, `fhr_drivers_completeness_max`.
**Reading.** `lag50_s`/`lag90_s` hold half and 90% of |IG|; 90% inside the conv-stem reach
(`share_in_stem`) is near-persistence. `seg_median` against `pop_mean` separates level from pattern.
**Misread:** a long `lag90_s` under `pop_mean` can be the level alone; read `seg_median` beside it.
**Cost.** 24 segments × 4 anchors × 8 IG-equivalents at 128 steps: ~7.5 min.

### `impulse_response` (new)

**Question.** What forecast does one contraction elicit, as a function of the delay
d = 4(t + 1 + τ) − t₀ s? This is the model's effective FIR kernel.
**Method.** A raised-cosine bump of the segment's median contraction amplitude and width is added to
raw UP at known tokens, on two backgrounds paired with themselves: `segment` (real UP) and `rest`
(UP flat at resting tone, the cleaner probe). Mean-decoded, so both arms share the latent.
**Outputs.** `impulse_response_kernel.csv/.pdf`, `impulse_response_per_segment.npz`,
`impulse_response_summary.json`, `impulse_response_example.pdf`. Headline: `impulse_level_peak_delay_s`,
`impulse_level_peak_bpm` (both on `rest`), `impulse_causality_max_abs`.
**Reading.** Δlevel in bpm binned on d; ΔKL and Δattention on the lag pointing at the peak.
`causality_max_abs` must be 0. **Misread:** the headline is the kernel's peak, which may sit far from
the delay the forecast uses; read the whole kernel, and its size against the data's own
contraction-triggered FHR change (`decelerations_triggered_fhr.csv`).
**Cost.** 128 segments × 2 backgrounds × (1 + 5 injections) = 1536 segment-forwards: ~1 min.

### `raw_shift` (new)

**Question.** Does the lag readout move with the UP content?
**Method.** Raw UP shifted against FHR by ±1..15 samples (sub-patch) and ±1..10 patches, vacated
samples at resting tone, re-patchified. ΔK, Δ`pred_gap`, and the shift of the attention centroid
(per head and head mean) and of the KL-map centroid/argmax, regressed on the applied shift per recording.
**Outputs.** `raw_shift_{per_segment,summary}.csv`, `raw_shift.pdf`; `fits.{sub,patch}` in the record.
Headline: `raw_shift_attention_centroid_{slope,r2}`, `raw_shift_klmap_centroid_slope` (whole-patch fits).
**Reading.** Slope 1 (both axes in seconds) = a content tracker; 0 = fixed lags. **Misread:** `s > 0`
hands the model future UP: a non-causal probe, not a forecast.
**Cost.** 256 segments × 51 arms = 13 056 segment-forwards: ~6 min.

### `delay_map` (new)

**Question.** At which UP→FHR delay does removing UP cost the forecast?
**Method.** The `baseline` occlusion arm applied to one lag at a time over 16 anchors per segment
(each in its own row, whole lag window inside the segment), scored per horizon step: a field Δ(ℓ, τ).
The **delay curve** Δ(d) is its mean over the diagonal ℓ + 1 + τ = d, with the diagonal sum and
cell count beside it.
**Outputs.** `delay_map_{field,curve,per_recording}.csv`, `delay_map.pdf`. Headline:
`delay_map_argmax_steps`, `delay_map_argmax_s`, `delay_map_peak_nats`.
**Reading.** The argmax d* is the delay the forecast uses; a flat curve means the forecast does not
use UP timing. No verdict. **Misread:** corner diagonals hold few cells (see `n_cells`); a lone
peak there is noise. Fewer anchors per segment left the planted delay in the noise.
**Cost.** 512 segments × 16 anchors × 39 arms (the reference + 38 lags): ~5 min.

### `event_locked` (new)

**Question.** Does the attention follow a contraction as it recedes (a diagonal ℓ = s/4 on a lag ×
time-since-peak map) or sit at fixed lags? Which raw UP around a peak drives K and the gap?
**Method.** One dense forward per segment at the latent means; each anchor is indexed by tokens since
the latest peak. Per head: attention, the KL lag map, K, `pred_gap`, level RMSE. **Tracking index**:
mean on the band |ℓ − k| ≤ 1 over mean off it (a fixed-lag preference cancels). Raw-UP IG from a
resting-tone baseline at peak + 6 offsets, in seconds from the peak, with rise/fall/outside shares.
**Outputs.** `event_locked_{events,recordings,summary,curves,gain_by_target_time,ig_phase}.csv`,
`event_locked_maps.npz`, `event_locked.pdf`. Headline: `event_locked_tracking_attention(_best_head)`,
`…tracking_lag_map`, `…gain_peak_target_s`, `…ig_completeness_max`.
**Reading.** 1 = no tracking, > 1 = the head follows the contraction. Per head in the summary CSV.
**Misread:** the head-mean index can sit below 1 while one head tracks; read the heads. The tone path
is rough: read `ig_completeness.*.n_rows_over_tolerance`, not only the median.
**Cost.** 256 dense forwards + 64 events × 6 offsets × 2 readouts at 512 IG steps: ~20 min (the
override comment; the planted run did 396 IG rows in 294 s).

### `decelerations` (new)

**Question.** Is the forecast better around decelerations, does the attention sit on the causal
contraction, and does the model's implied delay track the measured one?
**Method.** Decelerations on raw FHR in bpm (`events.detect_decelerations`), contractions on raw UP;
each contraction claims the deepest nadir within `caps.decelerations_pair_window_s` (120 s; 240 s on
planted). Readouts: skill around the nadir, attention enrichment on the causal contraction (control:
delays permuted within recording), implied delay d̂ = 4(ℓ̄ + 1 + τ_n) against the measured one
(Spearman ρ), dose–response against amplitude and duration, contraction-triggered FHR.
**Outputs.** `decelerations_{events,contractions,recordings,nadir_curves,dose_slopes,summary,triggered_fhr}.csv`,
two figures. Headline: `decelerations_measured_delay_median_s` (pooled over paired events; the per-recording median is the `recording_delay_median_s` column), `…triggered_fhr_min_s`,
`…implied_delay_spearman`, `…implied_delay_attn_median_s`, `…attention_enrichment(_control)`.
**Reading.** A fixed-lag reader gives a constant d̂; a tracker gives d̂ ≈ delay with ρ > 0.
**Misread:** a d̂ median near the measured median is not tracking (a constant can match a median);
ρ and the enrichment against its control are the test. The headline delay median pools events; the
summary CSV's is a mean of per-recording medians.
**Cost.** 256 dense forwards: about a minute.

### `latent_descriptors` (new)

**Question.** Which clinical descriptors of the recent raw trace does the latent carry linearly?
**Method.** Grouped-CV (by recording) ridge R² from `mu_prior`, `mu_post` and `delta_mu` at an anchor
to descriptors of the 300 s ending there: FHR baseline, STV (4 s epochs), LTV, deceleration depth and
area, gap fraction, contraction count, UP tone, time since the last peak.
**Outputs.** `latent_descriptors_{r2,correlation}.csv/.pdf`, `latent_descriptors_rows.csv`,
`latent_descriptors_latents.npz`. Headline: `latent_r2_{baseline,stv}_mu_prior`,
`latent_r2_decel_depth_delta_mu`, `latent_r2_up_tone_delta_mu`.
**Misread:** a linear probe, not an attribution; the descriptors correlate with each other.
Negative R² is worse than the fold mean, not a signed effect.
**Cost.** 512 dense forwards + ridge: ~1 min.

### `signal_loss` (new)

**Question.** How does FHR signal loss move the latent and the forecast: does the model fall back
to the prior (KL down) or inflate its variance?
**Method.** Observational: per anchor, gap fraction and `missing`-token fraction over the lag window
against K, the log-variances and NLL (binned, Spearman). Interventional, on gap-free segments:
`weight = 0` over a 4–120 s FHR span ending 0, 60 or 180 s before the anchor; edited minus clean.
**Outputs.** `signal_loss_{anchors,binned,injection,injection_summary}.csv`,
`signal_loss_{observational,injection}.pdf`. Headline: `signal_loss_missing_token_frac`,
`signal_loss_{d_kld,d_logvar_prior,level_shift_bpm}_120s_at_anchor`.
**Reading.** `missing_occurrence` says how often the `missing` token appears at all; the
observational half is empty without it. **Misread:** under `source_validity: finite` the UP stays
visible across an FHR gap; the injection probes the FHR stream only.
**Cost.** 1024 forwards (~30 s) + 64 segments × 4 anchors × 13 forwards (~1.5 min).

## Lean limits

> lean-limit: the shared figures label the lag axis "stored-coefficient time". Here it is patch
> time: lag ℓ = 4ℓ s exactly, with no group delay. The correct wording is in `preflight.json`
> `causality.lag_axis`. Fixing the figures needs the ~15 modules that import
> `lag_axis.COEFFICIENT_LAG_AXIS_LABEL`, `GROUP_DELAY_CAVEAT` and `GROUP_DELAY_NOTE` by name to read
> them at call time, so a binding can rebind them.

> lean-limit: the `missing` token is rarely trained. Under `source_validity: finite` only non-finite
> UP is invalid, and the committed fixtures have no FHR gap. The `missing` occlusion arm, the
> `signal_loss` injection and any `missing`-heavy recording read an embedding the fit barely saw.

> lean-limit: climatology is meaningful only once `summary_stats.py` constants replace the identity
> placeholders in `configs/default.yaml`. Until then `channel_skill` drops it and records
> `summary_constants_identity: true`.

> lean-limit: `clock_margin_min_nats` is null, so `coupling_exceeds_availability_clock` is
> INCONCLUSIVE on every run until a patch run measures its own spread.

> lean-limit: the resting tone is the 10th percentile of the whole 20-min segment, and the event
> detectors smooth over the whole segment, so the `baseline` arm, the tone IG baseline and the
> `latent_descriptors` event counts read the segment's future. Each is an intervention or a probe,
> never a forecast input.

> lean-limit: deceleration pairing is greedy. A nadir claimed by two contractions keeps the later one,
> and the loser does not re-claim its second-deepest nadir.

> lean-limit: no cross-cell table. A `pred_gap` here is over a 60-cell patch block; the comparison
> with CFS goes through the classifier and the source share of the KL (`../RESULTS.md`).

> lean-limit: on the planted shard `time_shift` and `cross_subgroup` record skips (one segment per
> recording, one cohort), and every class contrast is empty (no labels).
