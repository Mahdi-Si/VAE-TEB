# Eval and analysis pipeline for `lag_attn_transformer_patch` — work order

Written 2026-10-03 by the orchestrator. It synthesises four read-only maps in this folder:
`EVAL_MAP_CORE.md` (run flow, binding, the feature-specific touch points),
`EVAL_MAP_CAPTUM.md` (attribution machinery, an IG dry run, the raw-signal proposals),
`EVAL_MAP_LAG.md` (lag, coupling and control analyses) and `EVAL_MAP_FORECAST.md` (forecast,
latent and clinical analyses). Read the map for your slice before you start.

**Goal.** Bind the patch model to the existing CFS evaluation pipeline (`teb_vae/lag_attn_cfs/eval`)
instead of forking it. Add the analyses that only a raw-signal model allows, aimed at two
questions: *where do the lags come from*, and *what drives the model*.

## 1. Decisions (binding on every task)

- **D1. One package, no fork.** The work goes in `teb_vae/lag_attn_transformer_patch/eval/`. It
  binds to the shared pipeline through one `ModelBinding`, as `lag_attn_transformer_cfs/eval`
  does. Analyses that read only the collection tables are reused unchanged.
- **D2. Shared edits stay small and backward compatible.** Edits to `teb_vae/lag_attn_cfs/eval/` are
  allowed only where a patch need cannot be met from the patch package. Each one must leave CFS
  behaviour unchanged, and together they should come to about 60 lines or less. Typical forms are a
  new `ModelBinding` field with the current behaviour as its default, a `getattr` fallback, or
  `arange(decoder_out_channels)` in place of `arange(c_y)`. The patch package never monkeypatches a
  shared module global. After any shared edit, run the CFS eval tests
  (`teb_vae/lag_attn_cfs/tests -k eval -m "not slow"` and `teb_vae/lag_attn_transformer_cfs/tests -k eval -m "not slow"`).
- **D3. An eval view of the model and task** (`eval/view.py`).
  - `class SeqVaeLagAttnTrfPatch(nets.model.SeqVaeLagAttnTrfPatch)` keeps the same class name, so
    the checkpoint check passes, and adds no parameters. It does three things:
    - its `forward` is the CFS five-argument form `(y_st, y_ph, u_stream, anchor_phase=, anchor_stride=)`,
      that is `CausalWarmupInputs.forward`, which every shared call site uses;
    - it declares the eval-only constants a patch model makes true by construction:
      `TARGET_BLOCK_SPLIT = 1` (the "st"/"ph" columns then mean level/variability),
      `target_warm_frac = 1.0` and all-True `source_block_warm_*`;
    - it adds anything else the core map lists.
  - `SeqVaeLagAttnTrfPatchEvalTask(SeqVaeLagAttnTrfPatchTask)` overrides `_build_target_streams`,
    `_build_source_stream`, `_build_raw_target` and `_build_forward_inputs`, so every shared
    builder returns patch streams in the five-argument layout.
  - The training model and task are not changed, except for D4.
- **D4. One target definition.** Add `PatchSummaryTarget._build_forecast_target(...)`, the gather at
  `a + 1 + τ` that the shared collection calls. `compute_loss` uses it as well, so the eval's
  `nll_*` equals the training loss by construction.
- **D5. Occlusion has three arms.**
  - `zero` keeps comparability with CFS; it means "flat at mean UP".
  - `baseline` is the primary arm: replace the raw UP in the band with the segment's resting tone,
    then re-patchify. This is the "no contraction" intervention.
  - `missing` sets validity to −1. It is reported but flagged as unreliable, because `missing` is
    rarely trained.
- **D6. A shared raw-signal substrate** (`eval/raw.py`), written once, before the analyses. It holds
  everything the new analyses share:
  - a raw → patch → forward helper with optional UP/FHR edits;
  - unit conversion (standardized level → bpm, variability → rms-Δ bpm) through
    `traces.raw_signal_scales` and the model's `target_summary_loc/scale`;
  - a resting-tone UP estimator;
  - the Captum `RawReadout` wrapper (inputs `(B, 300, 16)` raw samples, `patchify` inside, the
    validity channel held fixed);
  - re-exports of the shared anchor selection and the event detectors.

  New analyses import these and do not re-implement them.
- **D7. The import rule.** The patch eval obeys `teb_vae/lag_attn_cfs/tests/test_eval_self_contained.py`:
  no `teb_vae.lag_attn_rws.eval`, no `model/*`, no Lightning in analyses. The deceleration detector
  is ported into the shared layer-0 `teb_vae/lag_attn_cfs/eval/events.py`, beside
  `detect_contractions`, as an added function.
- **D8. Few tests** (the user's standing rule): an end-to-end eval run, a binding contract check and
  one raw-Captum check. Each must be able to fail on a real regression. The shared CFS suites must
  stay green.
- **D9. The analysis contract** is the shared one:
  `run_<name>_analysis(context, *, eval_config, output_dir, probe) -> dict`, with `n_samples`,
  `composition` and `plan`, run failure-isolated through `report.step`. Every figure follows the
  user's figure conventions:
  - cold or masked cells are blanked;
  - log or symmetric-log scales for magnitudes that span decades;
  - one segment's figures are stacked on one shared time axis in the samples-page layout;
  - reuse `masked_field`, `signed_log_norm`, `symlog_legend` and `build_example_figure` from
    `lag_attn_cfs/eval/attributions.py`.

## 2. Registered analyses

| Name | Status | Source | Question |
|---|---|---|---|
| shared table analyses (`forecast`, `residual`, `latent`, `trajectory`, `distributions`, `cross_subgroup`, `lag_kl`, `attention`, `lag_shape`, `lag_clocks`, `lag_kld_scaled`, `lag_high_kl`, `source_null`, `perm_control`, `coupling`, `events`, `second_stage`, `time_to_delivery`, `sufficiency`, `recording_traces`, `attribution`) | reused | shared | as in CFS |
| `warmup`, `spectral_skill`, `band_partition` | excluded | — | meaningless for patches |
| `samples` | port | E1-F | Raw-trace page: level forecast in bpm with a ±σ band over raw FHR, a variability lane, UP with contraction shading |
| `occlusion` | port | E1-O | The D5 arms over the patch lag bands |
| `time_shift` | port | E1-O | The swap control on patch inputs |
| `calibration` | port | E1-F | Per-channel PIT, coverage and CRPS; variability calibration against raw STV |
| `channel_skill` | new (replaces `spectral_skill`) | E1-F | Per-channel horizon skill in bpm, and the lead time where level stops beating persistence |
| `raw_attribution` | new | E1-C | A raw-UP IG lag map (0.25 s resolution) of `kld`, `pred_gap` and the UP-driven level shift, against `source_kl_lag_map`. Also within-patch position, value vs delta, a key vs value split, and per-head profiles |
| `fhr_drivers` | new | E1-C | IG of the base forecast (level, variability) and of `mu_prior` to raw FHR history, plus the FHR context of the UP effect |
| `impulse_response` | new | E1-I | Inject a synthetic contraction at a known time and read the forecast-level and KL response against the delay `d = 4(t + 1 + τ) − t₀`. This is the model's effective FIR kernel |
| `raw_shift` | new | E1-I | Shift raw UP by ±1..15 samples and ±1..10 patches; read ΔKL, Δgap and the movement of the attention centroid |
| `delay_map` | new | E1-O | Fine-band `baseline` occlusion over lag × horizon, re-plotted along the delay `d = ℓ + 1 + τ`: an interventional delay curve |
| `event_locked` | new | E1-E | Contraction-locked maps of per-head attention, `source_kl_lag_map`, KL, gap and error against time since the peak (with a diagonal tracking index), and contraction-locked raw-UP attribution |
| `decelerations` | new | E1-E | Deceleration-centred skill and attribution: decelerations detected on raw FHR and paired with the preceding contraction; the model's implied delay against the measured one; and dose–response against contraction prominence and duration |
| `latent_descriptors` | new | E1-L | Grouped-CV ridge R² from `mu_prior`, `mu_post` and `delta_mu` to raw descriptors: baseline, STV, LTV, deceleration depth, gap fraction and UP descriptors |
| `signal_loss` | new | E1-L | Observational gap fraction against KL, logvar and NLL, plus an interventional gap-injection sweep |

## 3. Phases and file ownership

```
E0  foundation (one agent)                                   → gate: the eval runs end to end on the planted checkpoint
E1  C | I | O | E | F | L (parallel, disjoint files)         → gate: every step of the run is ok on the planted checkpoint
E2  tests, EVAL.md, the planted instrument reading, DESIGN.md
```

**E0 — foundation.** It owns the shared edits, D4 in `nets/patch_target.py`, and these files:
`eval/{__init__,binding,view,raw,run,verify}.py`, `eval/configs/eval_overrides.yaml`,
`eval/configs/planted_overrides.yaml` (local instrument runs on the fixture), `eval/EVAL.md` (stub)
and `eval/analyses/__init__.py`. It also writes **a stub module for every new or ported analysis**
in §2, with the exact signature, a docstring holding the spec row and a body that returns a skip
record. `binding.py` registers all of them once, so no E1 agent touches `binding.py`.
- **Acceptance:** `python -m teb_vae.lag_attn_transformer_patch.eval.run` completes on a 40-epoch
  `planted.yaml` checkpoint with `planted_overrides.yaml`. `summary.json` is written, and every
  reused shared step is `ok` or an explained skip. The CFS eval fast suites stay green.

**E1 — analyses.** Each agent owns only its own `eval/analyses/<name>.py` modules (and a figure
helper of its own if needed). It reads `raw.py`, `view.py` and its map. It may ask the orchestrator
for a `raw.py` addition but must not edit `raw.py` itself.
- C: `raw_attribution`, `fhr_drivers`
- I: `impulse_response`, `raw_shift`
- O: `occlusion`, `time_shift`, `delay_map`
- E: `event_locked`, `decelerations` (plus the shared `events.detect_decelerations` port, D7)
- F: `samples`, `calibration`, `channel_skill`
- L: `latent_descriptors`, `signal_loss`

**Validation.** Every agent validates on the planted checkpoint, whose delay is known to be 45
steps = 180 s, and reports what its analysis reads there. The pass/fail reading is the user's.

**Cost caps.** Every analysis states its caps in `eval_overrides.yaml` (segments, anchors, IG
steps), and its default cost on the production holdout must be stated. IG uses 128 steps and
reports its completeness error.
