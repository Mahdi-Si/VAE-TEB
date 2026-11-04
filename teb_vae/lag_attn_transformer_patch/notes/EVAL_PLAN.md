# Eval and analysis pipeline for `lag_attn_transformer_patch` — design decisions

The patch evaluation binds the patch model to the existing CFS evaluation pipeline
(`teb_vae/lag_attn_cfs/eval`) instead of forking it. It adds the analyses that only a raw-signal
model allows, aimed at two questions: *where do the lags come from*, and *what drives the model*.
`../eval/EVAL.md` describes the built package and how to run it. The proposal IDs that the analysis
modules cite are indexed in `EVAL_MAP_CAPTUM.md` (P1–P8), `EVAL_MAP_LAG.md` (N1–N6) and
`EVAL_MAP_FORECAST.md` (P1–P6).

## 1. Decisions

- **D1. One package, no fork.** The code is in `teb_vae/lag_attn_transformer_patch/eval/`. It
  binds to the shared pipeline through one `ModelBinding`, as `lag_attn_transformer_cfs/eval`
  does. Analyses that read only the collection tables are reused unchanged.
- **D2. Shared edits stay small and backward compatible.** An edit to `teb_vae/lag_attn_cfs/eval/`
  is made only where a patch need cannot be met from the patch package. Each one leaves CFS
  behaviour unchanged. Typical forms are a new `ModelBinding` field with the previous behaviour as
  its default, a `getattr` fallback, or `arange(decoder_out_channels)` in place of `arange(c_y)`.
  The patch package never monkeypatches a shared module global. After any shared edit, run the CFS
  eval tests (`teb_vae/lag_attn_cfs/tests -k eval -m "not slow"` and
  `teb_vae/lag_attn_transformer_cfs/tests -k eval -m "not slow"`). `../eval/EVAL.md` lists the
  shared edits.
- **D3. An eval view of the model and task** (`eval/view.py`).
  - `class SeqVaeLagAttnTrfPatch(nets.model.SeqVaeLagAttnTrfPatch)` keeps the same class name, so
    the checkpoint check passes, and adds no parameters. It does two things:
    - its `forward` is the CFS five-argument form `(y_st, y_ph, u_stream, anchor_phase=, anchor_stride=)`,
      that is `CausalWarmupInputs.forward`, which every shared call site uses;
    - it declares the eval-only constants that a patch model makes true by construction:
      `TARGET_BLOCK_SPLIT = 1` (the "st"/"ph" columns then mean level/variability),
      `target_warm_frac = 1.0` and all-True `source_block_warm_*`.
  - `SeqVaeLagAttnTrfPatchEvalTask(SeqVaeLagAttnTrfPatchTask)` overrides `_build_target_streams`,
    `_build_source_stream`, `_build_raw_target` and `_build_forward_inputs`, so every shared
    builder returns patch streams in the five-argument layout.
  - The training model and task are not changed, except for D4.
- **D4. One target definition.** `PatchSummaryTarget._build_forecast_target(...)` is the gather at
  $a + 1 + \tau$ that the shared collection calls. `compute_loss` uses it as well, so the eval's
  `nll_*` equals the training loss by construction.
- **D5. Occlusion has three arms.**
  - `zero` keeps comparability with CFS; it means "flat at mean UP".
  - `baseline` is the primary arm: the raw UP in the band is replaced with the segment's resting
    tone, then re-patchified. This is the "no contraction" intervention.
  - `missing` sets validity to −1. It is reported but flagged `reliable: False`, because `missing`
    is rarely trained.
- **D6. A shared raw-signal substrate** (`eval/raw.py`). It holds everything the new analyses share:
  - a raw → patch → forward helper with optional UP/FHR edits;
  - unit conversion (standardized level → bpm, variability → rms-Δ bpm) through
    `traces.raw_signal_scales` and the model's `target_summary_loc/scale` (`SummaryUnits`);
  - a resting-tone UP estimator;
  - the Captum `RawReadout` wrapper (inputs `(B, 300, 16)` raw samples, `patchify` inside, the
    validity channel held fixed);
  - re-exports of the shared anchor selection and the event detectors.

  New analyses import these and do not re-implement them.
- **D7. The import rule.** The patch eval obeys `teb_vae/lag_attn_cfs/tests/test_eval_self_contained.py`:
  no `teb_vae.lag_attn_rws.eval`, no `model/*`, no Lightning in analyses. The deceleration detector
  is ported into the shared layer-0 `teb_vae/lag_attn_cfs/eval/events.py`, beside
  `detect_contractions`, as an added function.
- **D8. Tests.** The patch eval has three tests: an end-to-end eval run (`tests/test_eval_run.py`),
  a binding contract check (`tests/test_eval_binding.py`) and the raw forward and raw-Captum checks
  (`tests/test_eval_raw.py`).
- **D9. The analysis contract** is the shared one:
  `run_<name>_analysis(context, *, eval_config, output_dir, probe) -> dict`, with `n_samples`,
  `composition` and `plan`, run failure-isolated through `report.step`. Every figure follows these
  conventions:
  - cold or masked cells are blanked;
  - log or symmetric-log scales are used for magnitudes that span decades;
  - one segment's figures are stacked on one shared time axis in the samples-page layout;
  - `masked_field`, `signed_log_norm`, `symlog_legend` and `build_example_figure` from
    `lag_attn_cfs/eval/attributions.py` are reused.

Every analysis states its caps (segments, anchors, IG steps) in `eval/configs/eval_overrides.yaml`.
IG uses 128 steps by default and reports its completeness error.

## 2. Registered analyses

| Name | Status | Question |
|---|---|---|
| Shared table analyses (`forecast`, `residual`, `latent`, `trajectory`, `distributions`, `cross_subgroup`, `lag_kl`, `attention`, `lag_shape`, `lag_clocks`, `lag_kld_scaled`, `lag_high_kl`, `source_null`, `perm_control`, `coupling`, `events`, `second_stage`, `time_to_delivery`, `sufficiency`, `recording_traces`, `attribution`) | reused | As in CFS |
| `warmup`, `spectral_skill`, `band_partition` | excluded | They read a filter-bank warm-up, frequency channels and an ST/PH channel map, which a patch does not have |
| `samples` | port | Raw-trace page: level forecast in bpm with a ±σ band over raw FHR, a variability lane, UP with contraction shading |
| `occlusion` | port | The D5 arms over the patch lag bands |
| `time_shift` | port | The swap control on patch inputs |
| `calibration` | port | Per-channel PIT, coverage and CRPS; variability calibration against raw STV |
| `channel_skill` | new (replaces `spectral_skill`) | Per-channel horizon skill in bpm, and the lead time where level stops beating persistence |
| `raw_attribution` | new | A raw-UP IG lag map (0.25 s resolution) of `kld`, `pred_gap` and the UP-driven level shift, against `source_kl_lag_map`. Also within-patch position, value vs delta, a key vs value split, and per-head profiles |
| `fhr_drivers` | new | IG of the base forecast (level, variability) and of `mu_prior` to raw FHR history, plus the FHR context of the UP effect |
| `impulse_response` | new | Inject a synthetic contraction at a known time and read the forecast-level and KL response against the delay $d = 4(t + 1 + \tau) - t_0$. This is the model's effective FIR kernel |
| `raw_shift` | new | Shift raw UP by ±1..15 samples and ±1..10 patches; read ΔKL, Δgap and the movement of the attention centroid |
| `delay_map` | new | Single-lag `baseline` occlusion over lag × horizon, re-plotted along the delay $d = \ell + 1 + \tau$: an interventional delay curve |
| `event_locked` | new | Contraction-locked maps of per-head attention, `source_kl_lag_map`, KL, gap and error against time since the peak (with a diagonal tracking index), and contraction-locked raw-UP attribution |
| `decelerations` | new | Deceleration-centred skill and attribution: decelerations detected on raw FHR and paired with a contraction; the model's implied delay against the measured one; and dose–response against contraction amplitude and duration |
| `latent_descriptors` | new | Grouped-CV ridge R² from `mu_prior`, `mu_post` and `delta_mu` to raw descriptors: baseline, STV, LTV, deceleration depth, gap fraction and UP descriptors |
| `signal_loss` | new | Observational gap fraction against KL, logvar and NLL, plus an interventional gap-injection sweep |

The source modules cite their owner tags (E1-C, E1-I, E1-O, E1-E, E1-F, E1-L). Each tag names one
group of analyses: C = `raw_attribution`, `fhr_drivers`; I = `impulse_response`, `raw_shift`;
O = `occlusion`, `time_shift`, `delay_map`; E = `event_locked`, `decelerations`; F = `samples`,
`calibration`, `channel_skill`; L = `latent_descriptors`, `signal_loss`.
