# Session 7 report: `lag_attn_transformer_cfs`, epoch 424 (s6) to rung s7

Date: 2026-11-10. Decision log with every pre-registration and result: `DECISIONS.md`.

## Recommendation

**Launch s7** with `teb_vae/lag_attn_transformer_cfs/configs/sweep_next_s7_tolerant_ua_tails.yaml`
and the warm start `lag-attn-trf-cfs-epoch=424-s7init.ckpt`.

The rung makes four changes to s6. Each one passed its pre-registered test on the select
recordings (split code 2), and the stack passed on the test half (codes 1 and 3, 159 recordings,
opened once):

| Test-half result, s7 stack against the control fine-tune | Value |
|---|---|
| Forecast NLL of the full branch | -197.3 nats per anchor [-210.2, -184.6] |
| Gain at contractions with a short invalid UA run | +1.51 nats [0.51, 2.55] (unseen cohorts +1.01 [-0.10, 2.26]) |
| Calibration tail error (tolerance 0.5) | 0.89 against 1.18 |
| Decoder cells on the variance floor | 0.02 against 0.34 |
| KL minus source-null KL | 0.41 [0.35, 0.46] against 0.21 [0.16, 0.25] nats |
| Source gain over all anchors | 2.57 [2.15, 2.97] against 3.27 nats |
| Scattering MSE against own-channel AR(6) | 0.931 against 0.884 (a cost of 5.3%) |
| Official verdicts | all ten keep their status: 8 PASS, calibration FAIL, clock INCONCLUSIVE |

The control is the epoch-424 checkpoint fine-tuned with the same data, order and budget and no
change. Intervals are recording-level paired bootstraps (2000 resamples). The statistics file is
a stand-in, so read the sizes as relative effects.

**To launch.** (1) On the training machine run
`python -m teb_vae.neural_surgery_trf_cfs.session7.init_s7 --mode build --checkpoint <epoch-424 file> --output <warm-start file>`;
it needs no data and checks the result by digest. (2) Set `core_model_checkpoint` in the s7 config
to that file. (3) Start the trainer with the s7 config. A one-epoch run of the real driver on the
local HIE shard completed with this config and warm start.

## The four changes and their evidence

| Key | Hypothesis | Select-code evidence (arm minus control) |
|---|---|---|
| `source_running_gap_keys: remove` | H1, causal part | Removal alone costs nothing (-0.00 [-0.04, 0.03] nats). It removes a look-ahead path: with all flags valid, anchors inside a running gap gained 15.1 nats against 9.7 on clean anchors. |
| `source_valid_tolerance_s: 16.0` | H1 (yours), in a causal form | Gain at contractions with a short invalid run +3.19 nats [1.24, 5.41]; unseen cohorts +3.59 [1.46, 6.32]; no cost over all anchors. On the stack: NLL -0.11 [-0.20, -0.03]. |
| `decoder_logvar_clamp: [-9, 3]` | H5, first half | NLL -133 nats [-156, -110]; phase floor share 0.50 to 0.00. |
| `forecast_student_nu: 8.0` | H5, second half | On top of the floor: NLL -54 nats [-59, -49]; tail error 2.60 to 0.99; scattering 1-sigma coverage 0.684. |

Your interpolation hypothesis holds within limits (Phase 1, synthetic gaps in clean UA rebuilt
with the builder's own bridge): bridged UA with valid flags costs 1.0% [-2.2, 4.6] of the source
gain at 16 s, against 11.7% with the strict flags; 4.6% at 32 s; 12.6% at 64 s. The bridge cannot
draw a crest (peak error -10.9% at 16 s, -28.8% at 32 s) and is weak on the rise beyond 4-8 s.
83% of invalid runs are at most 16 s.

## Prior findings, re-verified

| Finding | Verdict at epoch 424 |
|---|---|
| 1. Gate on at 43% of anchors with 97.7% of the KL; lag 0 holds 48% | Confirmed (43.4%, 97.7%, 48.1%). Calibration FAILs (3-sigma tail). |
| 2. Slow channels stay contaminated for minutes after short gaps | Refuted for gaps up to 16 s. The fast and middle channels (st0-st18) keep the strict rule closed for a median 48 s after an 8-s gap. True for gaps of 32 s or more. |
| 3. `event_weight_ua_valid` drops 11.9% of contractions, 54% with at most 8 s invalid | Confirmed in size: 14.1%, 51%. |
| 4. The bridge pulls toward the first valid sample after the gap | Confirmed in the builder code: it reads that sample plus 4 s of slope. |
| 5. Source adapter RMS 9.6 against 0.98; phase block gets nothing from the latent; floor binds; heavy tails | The first two are refuted (RMS 10.05 against 11.48; 28% of the phase forecast variance is latent). The last two are confirmed. |
| 6. `logvar_clamp` is shared; `named_parameters()` dedups shared tensors | Confirmed; s7 gives the decoder its own bound, and `init_s7.py` copies with `remove_duplicate=False`. |
| Soft-mask parameters | Only the adapter's flag input learned. The per-head key bias (at most +0.29) and the gate's invalid-mass weight (at most 0.26) stayed near zero. |

## Killed

| Hypothesis | Why |
|---|---|
| H2, tolerant reward (`event_weight_ua_tolerance_s`) | No rise of the gain at the contractions it adds, alone (+0.04) and on the stack (-0.02). Key stays inert. |
| H3, per-band key rule | The slow third of the channels carries 5% of the flag effect; the per-channel flags of s6 already beat the all-channel rule. |
| H4, normalise the source adapter | No scale mismatch at epoch 424. |
| H6, phase weight 3, or phase off the latent path | Phase MSE unchanged, NLL slightly worse; the latent carries 28% of the phase forecast. |
| H7, new contraction detector | The shipped peak finder is sound (onset spread 5 s, overlap agreement 0.985); its weakness is the drop rule. The session-7 detector made the forecast worse (+0.14 nats). |
| H8, source-null weight 1 to 3 | Identical to the control to four decimals: the reconstruction gradient swamps the KL terms. A null-referenced posterior failed as an intervention (rewarded gain keeps 68%). |
| Step-level tolerance rule | Recovered 26% of the effect: it kept the keys inside an ended gap flagged for ever. Replaced by per-anchor key removal. |
| Reward without the UA rule | Worse over all anchors (+0.11 nats). |

The deceleration detector is weak (12% of its events meet the NICHD definition), but it feeds
evaluation tables only.

## Remaining risks

- Local data (329 recordings), stand-in statistics, and fine-tunes of about 4% of one epoch. The
  directions are tested; the sizes over a full run are predictions.
- The lower floor costs about 5% scattering MSE against AR(6) on both halves.
- The source gain in nats falls under the Student-t density (to 79% on the test half).
- About two thirds of the KL still needs no UA values; the clock verdict stays INCONCLUSIVE.
- The warm start is not function-preserving in the decoder variance (mean log-variance change
  0.51); the first epochs start about 14 nats worse than epoch 424.
- The reward still drops contractions with a short invalid run; a long run could unlearn what the
  tolerant flags allow.
- The model reads UA almost only at lags 0 and 10 s, and its KL peaks at the contraction peak
  (the early-deceleration pattern). s7 does not address late responses.
- Training membership of the healthy shard is unknown; the unseen cohorts agree with it.
- 15 unit tests of the selection fail on the last commit too (see `DECISIONS.md`, E6).

## Pre-registered predictions for the full run

Relative to the rung-6 run at epoch 424, same validation data.

| # | Metric | Direction and size | By | Refuted when |
|---|---|---|---|---|
| 1 | `val/nll_full_block` | lower by 150-220 nats | epoch 10 | lower by less than 100 at epoch 20 |
| 2 | `logvar_full_floor_frac` | about 0.34 to below 0.05 | epoch 5 | above 0.15 at epoch 20 |
| 3 | calibration | tail error 0.6-1.0, 1-sigma coverage at most 0.73, still FAIL | first full evaluation | tail error above 1.2 |
| 4 | `val/pred_gap_mean` | positive, 65-85% of rung 6 | epoch 60 | at or below 0, or below 40% |
| 5 | gate at contractions with a short invalid run | at least 0.30 (rung 6: 0.16-0.18); gain at least 1.5 times rung 6 | epoch 60 | gate below 0.22: set `event_weight_ua_tolerance_s: 16.0` |
| 6 | scattering MSE against AR(6) | ratio at most 0.95 | epoch 60 | above 1.0: set `decoder_logvar_clamp: [-8.0, 3.0]`, rebuild the warm start |
| 7 | `coupling_minus_clock` | 0.3-0.5 nats, still INCONCLUSIVE | epoch 60 | below 0.15 |
| 8 | `source_conditioned_kl_raw` | rises in the KL warm-up, back to 1.1-1.5 | epoch 80 | above 3 at epoch 100 |
| 9 | official verdicts | the rung-6 statuses | first full evaluation | any PASS turns FAIL |
| 10 | phase MSE against AR(6) | unchanged (0.75-0.76) | epoch 60 | worse by more than 3% |

## Where things are

- Config: `teb_vae/lag_attn_transformer_cfs/configs/sweep_next_s7_tolerant_ua_tails.yaml`.
- Warm start: `tmp/checkpoint-2026-11-09/lag-attn-trf-cfs-epoch=424-s7init.ckpt` with its `.json`
  report; builder `session7/init_s7.py`; asset `assets/s7init_fitted.npz`.
- Production code (new keys, inert by default): `lag_attn_cfs/ua_tolerance.py`,
  `lag_attn_cfs/nets/causal_inputs.py`, `lag_attn_cfs/task.py`, `lag_attn_cfs/trainer.py`,
  `lag_attn_rws/nets/losses.py`, `lag_attn_rws/nets/controls.py`, the two cell constructors, the
  evaluation's calibration and metrics modules, and `configs/default.yaml`.
- Probes: `session7/p0_*.py` to `p5_*.py`, shared code in `s7common.py`, `s7_interventions.py`,
  `s7_tolerance.py`, `s7_pilot.py`, `detectors7.py`. Outputs under `session7/out/`.
- Evaluation: official test-half runs in `session7/out/phase5/official/`; the epoch-424 run
  directory has a new calibration-only pass and a rebuilt `dashboard.html`.
