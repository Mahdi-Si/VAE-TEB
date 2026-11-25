# `coupling_irf`: UA → FHR coupling and delay as an impulse response

## 1. Goal

1. Measure the coupling between uterine activity (UA, toco) and fetal heart rate (FHR): how much the
   recording's own UA tells about the FHR beyond the FHR's own past.
2. Report the delay of the FHR response to a contraction, in seconds.

This package replaces the UA path of `lag_attn_transformer_cfs` for these two goals. The lag study of
checkpoint 369 (`tmp/lag_analysis_369/REPORT.md`) showed that the old UA path gives neither:

| Finding of the lag study | Consequence here |
|---|---|
| A UA-blind attention profile keeps 95–100 % of the gain (F4); a lag index is not a delay (F10) | No attention over lags. The delay is the location of the minimum of an explicit impulse response. |
| KL, gate and expected lag follow any UA that is fed (F8) | The only coupling readout is forecast gain, and the UA path must stay shut for a UA that does not belong to the FHR. |
| An injected contraction has no defined delay; a sham moves the forecast more (R13) | Acceptance test: a response planted at a known delay must be read back within 4 s. |
| The gain peak cannot be separated from the 60-s reward window (F12) | No event weighting. Every valid cell is scored the same. |
| Scattering channels have 12–790 s of filter latency | The model works on the FHR level in bpm and the UA in toco units, on the 4-s grid. |
| Two of 64 latent dimensions carry the gain; one query serves all horizons (F11) | No latent and no network in the UA path. |

It follows the system-identification line of work on UA → FHR (Warrick, Hamilton, Precup and Kearney,
IEEE TBME 2009 and 2010; Warrick and Hamilton, Math Biosci Eng 2020): a rectified UA input, an impulse
response (IRF) on a 4-s grid, surrogate UA as the null. What is new here is that the IRF is estimated
causally at every step, gated by its own out-of-sample record, and scored as forecast gain.

## 2. Data

`data.py` builds a step cache from the shards once (`python -m teb_vae.coupling_irf.data`, 17 s, 38 MB for
the local split). Per 4-s step: mean FHR (bpm), SD within the step, valid fraction; mean UA (toco units),
valid fraction. A step is valid with the loader's own rule (`_quality_masks`). `stitch` joins the segments
of a recording on one timeline (segments tile at multiples of 120 s; missing stretches are invalid steps),
so one batch row is one recording of up to 4 h.

Clock: everything is on the stored clock. The dataset stores UA 20 s earlier than it was recorded, so
**recorded delay = stored delay − 20 s**. The model's delay range, 0–120 s stored, is −20 to +100 s
recorded, the range used in the literature.

## 3. Model

```
FHR level ──► causal TCN ──► base forecast μ0(t, τ), σ0(t, τ)        τ = 1…30 steps (4…120 s)
UA ──► minus causal resting tone, rectified ──► u ──► Z(t) = u convolved with 29 hat functions
FHR and Z, same causal high-pass ──► ridge regression on the last 60 min ──► IRF h_t   (29 knots, 4 s apart)
h_t fitted up to step s − 60 s predicts FHR at s ──► calibration slope, standard error ──► gate a_t
full forecast(t, τ) = μ0 + a_t · h_t · (X(t, τ) − ρ_τ · Z(t))
```

- **Base.** A causal TCN (6 blocks, receptive field 8.5 min) on the FHR level, its within-step SD and its
  validity. It never sees UA. Student-t likelihood, ν = 8.
- **UA input.** `u` = UA above its running 10th percentile of the last 5 min, floored at 0, divided by 25.
  The floor is the half-wave rectifier of the Hammerstein models of this system.
- **IRF.** `h_t` has one value per delay knot, 4 s apart from 4 to 116 s, and is zero at 0 and at 120 s.
  It is the ridge solution (ridge and smoothness penalties, both per row of data and trained) of the
  regression of the FHR on `Z` over the last `context_steps` = 900 steps (60 min) of the same recording.
  FHR and `Z` both lose their trailing 5-min mean first; the same linear high-pass on both sides leaves
  the regression unbiased.
- **Gate.** For every past step `s`, the IRF fitted on steps ≤ `s` − 60 s predicts the FHR at `s`.
  The slope of the observed FHR on these out-of-sample predictions, minus `k` standard errors and cut
  to [0, 1], is `a_t`. With a UA that does not belong to the FHR the predictions do not correlate
  with the FHR and `a_t` is 0. `k` is set after training so that another recording's UA opens the
  gate at 5 % of the anchors of the training recordings (`calibrate_k`; `k` = 4.4 for run v2).
- **Two modes.** `X(t, τ)` is `Z` at the target step. In mode **fore** it uses UA up to the anchor `t`
  only (delays shorter than τ are not available); in mode **expl** it uses UA up to the target.
  `ρ_τ · Z(t)` removes the part of the UA response that is already in the FHR level at the anchor.
- **Trained numbers in the UA path:** two penalties, `ρ_τ` and a scale `η_τ` for the fore mode; `k` is
  calibrated. Everything else is closed form.

## 4. Readouts

| Readout | Definition |
|---|---|
| Coupling, fore / expl | Base NLL minus full NLL, nats per valid cell (`gains`). Report own UA, wrong UA and the difference. |
| Gate `a` | 0 = no evidence that UA explains FHR in this recording so far. "Not estimable" is a valid answer. |
| `skill` | Signed out-of-sample R² of the IRF on the recording's past. |
| IRF | `h_t`, in units of 20 bpm per unit of `u`; `irf_delay(h)` gives the delay of its minimum and its depth. Read only where `a > 0`. |
| Per contraction | Predicted response `h_t · Z(s)` around the contraction: delay from UA peak to predicted nadir, and the measured amplitude of the observed FHR on that template (`evaluate.py`, A5). |

## 5. Training

`python -m teb_vae.coupling_irf.train --out <dir>`: one GPU, data in memory, about 2.5 min per epoch on a
GTX 1660 Ti. Each batch is used twice: with the own UA (the response may lower the loss) and with another
recording's UA (any response raises it). The base forecast is detached in the full loss, so the UA path
cannot change it and the gain is a pure mean shift. Validation recordings are one of five GUID folds.
After the last epoch `k` is calibrated (`--false-open`, default 0.05); `--calibrate-only` redoes only that.

## 6. Acceptance tests

`python -m teb_vae.coupling_irf.evaluate --ckpt <dir>/model.pt --cache <steps npz> [--fold 0] --out <dir>`

| Test | Pass when |
|---|---|
| A1 own UA | Gain above 0 in both modes (95 % interval over recordings) |
| A2 wrong UA | Gain with another recording's UA not below −0.1 × the own gain |
| A3 planted delay | A response planted on a wrong UA at 8…88 s is read back with a median error ≤ 4 s |
| A4 shift | Shifting the own UA by s moves the read delay by −s (slope −1.2…−0.8) |
| A5 real contractions | Reported: predicted against observed peak-to-nadir delay; response against "a deceleration follows" |
| A6 reliability | Split-half reliability of the gain per recording above 0.45 (the old model's value) |
| A7 description | Gain, gate and delay by time to delivery and by class |

Unit tests (`tests/`, 2 s): the cache build, a planted delay read within one step on synthetic signals,
and causality (nothing at an anchor changes when FHR or UA after the anchor change; in the expl mode
future UA changes the output, future FHR does not).

## 7. Results of run v2

Trained on 3,806 recordings of the local train split (8 epochs, 19 min), `k` calibrated on training
recordings, then evaluated once on the test split: 1,186 recordings that were used for nothing else.
The held-out fold of the train split (923 recordings) gives the same verdicts and close numbers.
Brackets are 95 % bootstrap intervals over recordings. Seconds are on the stored clock.

| Test | Result on the test split | Verdict |
|---|---|---|
| A1 own UA | Gain expl 0.0164 [0.0143, 0.0185], fore 0.0043 [0.0036, 0.0050] nats per cell. Gate open at 41 % of anchors. | Pass |
| A2 wrong UA | Gain expl −0.0005 [−0.0007, −0.0002], fore −0.0003 [−0.0004, −0.0001]: 3 % and 6 % of the own gain. Gate open at 4.7 % of anchors. | Pass |
| A3 planted delay | Planted 8, 24, 40, 56, 72, 88 s at 20 bpm: median error 1.0, 2.7, 2.4, 2.3, 2.3, 2.4 s. At 40 bpm 1.3 s; at 10 bpm 4.4 s; at 5 bpm 12 s (detected at 11 % of anchors). A response to 30 % of the contractions at 25 bpm: 5.3 s, detected at 15 %. | Pass |
| A4 shift | Slope −1.00 [−1.00, −1.00]; a shift of ±8 s moves the delay by ∓8.0 s. | Pass |
| A5 real contractions | Predicted against observed peak-to-nadir delay: Spearman 0.48 [0.40, 0.56] over recordings, 0.38 over contractions; median error 15.6 s. Medians by type, predicted / observed: early 21 / 6 s, variable 43 / 35 s, late 50 / 62 s. | Reported |
| A6 reliability | First against second half of a recording: gain expl 0.42 [0.29, 0.53], IRF delay 0.57 [0.49, 0.64], gate 0.50 [0.45, 0.55]. | Fail for the gain (limit 0.45) |
| A7 description | Gain expl rises from 0.013 at 120–240 min to 0.027 in the last 30 min before delivery. Median IRF delay 38–44 s (18–24 s recorded) in every time bin. | Reported |

Reading:

- **Coupling** is specific: another recording's UA of the same subgroup costs a few percent of what the
  own UA gains.
- **Delay** is an instrument reading now: planted delays come back within one step when the response is
  10 bpm or more per large contraction, and a shifted UA moves the reading one for one.
- **Per contraction** the model predicts the recording's average response, so the ordering early <
  variable < late is right but compressed, and single contractions are off by a median of 16 s.
- **Reliability.** The gain of the first half of a recording predicts the gain of the second half with
  r = 0.42, about the old model's 0.45; labour is not stationary, so this is a strict split. The delay
  (0.57) is far more stable than any timing of the old model (0.10–0.38).
- **Classes** (descriptive, `ACCEPTANCE.md` A7): the gain is larger in HIE and acidosis than in healthy
  recordings without a blood-gas sample, but healthy recordings with a blood-gas sample are as high as
  acidosis, so most of this is how recordings were selected. A longer delay in HIE appears in one
  delivery mode per split and not the same one; it is not a finding.

Outputs: `tmp/coupling_irf/runs/v2/` (`model.pt`, `log.json`, `eval_val/`, `eval_test/` with
`ACCEPTANCE.md`, `acceptance.json` and two figures).

## 8. Limits

- **The IRF is an average over the last 60 min.** A response that follows only some contractions is
  diluted: planted on 30 % of the contractions it is detected at 15 % of the anchors and read 5 s off.
  About 29 % of real contractions are followed by a deceleration. A per-contraction delay needs a
  response model per contraction on top of this one; A5 is the baseline it has to beat.
- **Responses below about 10 bpm per large contraction are not detected.**
- **Linear in `u`.** One IRF for all contraction sizes. Toco amplitude is uncalibrated and saturates near 100.
- **Delays below about 8 s stored** (−12 s recorded) are squeezed against the zero at delay 0.
- **FHR level only.** Coupling to FHR variability is not measured.
- **Independent cells.** The likelihood treats the 30 horizon steps of an anchor as independent; gains are
  for comparison between conditions, not absolute information rates. Use recording-level bootstrap.
- **The 20-s UA shift is taken as given.** If it does not correct a real toco lag, read the recorded column.
- **The small linear solves run on the CPU** (see the `ponytail:` note in `model.py`).
