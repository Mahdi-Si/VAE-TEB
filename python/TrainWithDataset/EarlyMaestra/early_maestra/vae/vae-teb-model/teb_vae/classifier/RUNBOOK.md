# Classifier runbook: what to run, in which order, and what to send back

**Written:** 2026-10-08. **For:** the project owner running the experiments, and a fresh Claude session picking the work up without the earlier conversation. **Companion documents:** `SPEC.md` (the normative design, §13.3 lists every shipped config, §17 the experiment plan), `RESEARCH.md` (the literature). Numbers below are quoted from two finished runs and are dated; re-read them from the runs' own `summary.md` before relying on them.

---

## 0. How to use this file

1. Read §1 (where things stand) and §2 (how to read a result) once.
2. Follow the steps of §3 in order. Each step says what to run, what it produces, and what to send back.
3. For every finished run, send the files listed in §4. The next session starts from those files and this document.
4. §5 lists the decisions still open and §6 the session log, so a new session knows what was done and why.

---

## 1. Where things stand (2026-10-08)

### 1.1 The two finished 10-fold runs

Both used `default.yaml` plus overrides, frozen trf_cfs latents (`mu_prior` + `delta_mu`, `kld_per_t` as attention cue), sequence scope, binary `adverse_vs_healthy`, one seed. Results zips sit in `tmp/` (figures stripped).

| Run | Settings beyond `default.yaml` | Test GUID AUROC fold mean / pooled | Probe | Notes |
|---|---|---|---|---|
| `2026-10-07--13-30-04-trf_cfs_adverse_seq` | `horizon_decay` (H 1 h, half-life 0.5 h), train window last 6 h, `delta_t` off, lr 1e-4, 1000 epochs, logloss monitor | 0.713 ± 0.028 / 0.671 | 0.662 / 0.639 | verify PASS; best epochs 3–10; train AUROC 0.98 by epoch 30 |
| `2026-10-07--17-52-25-trf_cfs_adverse_seq` | as above but the full 12.4 h training window | 0.698 / 0.654 | 0.662 / 0.639 | verify **FAIL** on criterion 7 (shuffled control), explained in §1.3; logloss monitor picked epoch 0–2 in every fold while the AUROC-best epoch (6–26) was 0.02–0.06 higher on val |

### 1.2 What the 13:30 run taught (the diagnosis the plan rests on)

- **The input representation is the ceiling, not the head.** Model minus probe is +0.05 fold mean, +0.03 pooled. 60 of the 64 `delta_mu` channels sit under the scaler floor (std ≤ 0.08 against ≈ 2 for `mu_prior`): the UP pathway reaches the classifier through about two dimensions. Context alone (time from onset, stage, time in second stage at the last segment; logistic fit on val) gives test AUROC 0.60. The sequence aggregator adds +0.05 over the segment-local head (0.713 vs 0.666).
- **The healthy comparator differs between train/val and test.** Train and val healthy GUIDs are ≈ 85 % blood-gas (BG+); test healthy GUIDs are ≈ 80 % no-gas (BG−), including 906 shared GUIDs present in every fold's test split. AUROC against BG+ healthy is 0.64 (test and val alike), against BG− healthy 0.73. This explains the val-vs-test gap and the pooled-vs-fold-mean gap. The headline must say which population it means.
- **Overfit within ten epochs.** Val log loss bottoms at epoch ≈ 6, val AUROC peaks at ≈ 10 (0.657 vs 0.650 at the log-loss optimum in that run), train AUROC reaches 0.98. Fold SD 0.028 is about the evaluation SE for 270 positives per fold.
- **The signal is late.** Snapshot AUROC 0.76 in the last 30 min, 0.70 at 1–2 h, 0.63 at 4–5 h, 0.54 at 11–12 h. 43 % of detections and 41 % of false alarms fire in the last hour. Sensitivity 2 h before delivery is 0.26.
- **Alarm rules and score smoothing are not levers.** k-of-n, EMA, alarms restricted to the last 3–6 h, mean/max over last positions: all land at sensitivity 0.50 ± 0.01 and FPR 0.19–0.21 once their thresholds are re-selected on val.
- **The model is largely a second-stage detector.** np30 sensitivity 0.545 in second stage vs 0.218 in first; healthy second-stage segments score ≈ 0.3 logits above first-stage ones. The 17:52 run repeated this (0.566 vs 0.127; healthy specificity 0.673 vs 0.955).
- **The 6 h training window did not cause early false alarms** (8 % at position 0; healthy scores at 6–12 h extrapolate low), but train sequences were 8–31 segments against 10–66 at test.
- **Severity is not learned**: HIE-vs-acidosis AUROC 0.549; the auxiliary 3-class head never predicted HIE.
- The NP umbrella (`np30`) costs 4.5 sensitivity points against the empirical cap (`emp30`) at 272 validation negatives per fold: 0.458 vs 0.503.

### 1.3 What the 17:52 run taught

- **Criterion 7 FAIL was a selection artefact, not a leak.** The shuffled control early-stopped on the real val labels, so it kept the epoch whose near-random function best matched the real signal, and that match carried to test (pooled test AUROC 0.524 [0.510, 0.539]; per-fold test AUROCs 0.41–0.64, SD 0.072, while the pooled bootstrap CI covers test sampling only, ±0.015). `shuffle_labels` itself was correct.
- **Log-loss selection is too early.** It picked epoch 0–2 in every fold; selecting on val AUROC would have given +0.02 to +0.06 val AUROC. This contradicts the 13:30 reading that the monitor choice did not matter.

### 1.4 What changed in the code (2026-10-07 and 2026-10-08)

Everything is in `teb_vae/classifier/`. SPEC.md documents each item; its CHANGELOG has the dated entries.

**Committed (`becfae3`, `95652d9`):**
- `source.kind: vae+hdf5` (`CombinedSource`: VAE keys and ST/PH fields side by side, frozen only).
- `source.drop_floor_channels` (channels under the scale floor are always warned and counted; with `true` they are dropped, probe scaler too).
- `train.loss_weights.bag_head: segment | position` (the bag term can pool the per-position head, whose running max the alarm reads).
- `eval.stage_offsets`: a `<kind>_stage` prediction row per unit with per-stratum offsets (in / not in second stage) fitted on val negatives and every policy re-selected on val; locked with the unit (`stage_offsets.json`, `thresholds_stage.json`).
- Subgroup family `healthy_comparator` (`all_healthy`, `bg_healthy`, `no_bg_healthy`), in the R11 ROC figures and as a pre-specified table in `summary.md` §3; `summary.md` §5 states the NP cost against the empirical cap.
- `train.unfreeze` accepts `<module list>.last` (the top block of the checkpoint).
- `SCHEMA_VERSION` 6: run directories made before it cannot be resumed.
- `regularised.yaml` became the ablation baseline; 17 one-delta ablation configs; `sweep.py` (runs a config list, then `compare`), `feature_rank.py` (per-channel spread and effective rank of a cache).
- `default.yaml`: the `labels.task` comment lists every task with its class-code mapping.

**Staged, not committed (the CORN head, 2026-10-07 later):**
- `labels.head: corn` with `train.loss.name: corn`: K−1 free conditional logits (adverse-vs-healthy on every GUID, HIE-vs-acidosis on the adverse GUIDs), chain-rule probabilities, prior-initialised biases, one temperature per logit. Configs `three_class_corn.yaml` and its CORAL twin `three_class_ordinal.yaml`. The two CORN tasks are pooled under the row weights, not normalised per task (marked `ponytail:` in `losses.py`).

**In the working tree, neither staged nor committed (a second session, 2026-10-08):**
- The shuffled control now permutes the **val** labels too (`seed + 1`), so `best.ckpt` is no longer chosen on the real signal; calibration, thresholds and prediction rows keep the real labels.
- Verify criterion 7 tests the fold-mean test AUROC with a 95 % t interval over the folds (`metrics.shuffled_interval`), the pooled bootstrap CI reported beside it.
- `default.yaml` early stopping monitors `val/guid_auroc` (max) instead of `val/guid_logloss`; `cotrain.yaml`'s comment follows.
- `labels.time_matched` (default false): the `horizon*` weight schedule applies to negatives too, so early healthy segments stop teaching "early means healthy" while early adverse segments are silent. Config `lab_time_matched.yaml`, appended to the batch-1 list in `sweep.py`.
- Tests updated accordingly (`test_cohort.py`, `test_data.py`, `test_metrics.py`).

**Not done:** the test suite has not been run since any of these changes (project policy: run it before committing); `CombinedSource` is import-checked only; no real-data run has used the regularised baseline yet.

---

## 2. How to read a result

- **Noise floor.** One fold's test AUROC has an SE of about 0.02 (270 positives). Pooled over 10 folds the SE is about 0.007. Treat a pooled AUROC difference under 0.015 as noise. Use `compare` (paired GUID bootstrap, DeLong, Nadeau–Bengio) rather than eyeballing two summaries.
- **Which healthy population.** Read the "Healthy comparator (pre-specified)" table in `summary.md` §3 first: `all_healthy` is the augmented test population, `bg_healthy` the clinically concerning one, `no_bg_healthy` the easy one. Report `all_healthy` and `bg_healthy` together; never one alone.
- **Pooled vs fold mean.** Pooled OOF counts the 906 shared BG− healthy GUIDs once; the fold mean counts them in every fold. Expect pooled < fold mean on this cohort.
- **Stage.** Compare `model` with `model_stage` in §3 and §9. If `model_stage` keeps the overall sensitivity and equalises the two stages, the stage dependence is a threshold problem; if it loses sensitivity, it is in the features.
- **Early detection.** The §8 time-resolved table at 2 h and 1 h before delivery is the number the label ablations should move; the final AUROC may not.
- **Verify.** `evaluation/verify.json` must say `passed: true`. Criterion 7 (shuffled control) is the one that has failed; with the val-label fix and the fold interval it should pass. If it fails again, send the per-fold shuffled AUROCs from `summary.md` §10 (H1) before anything else.
- **Baseline floor.** Every neural number is read against the probe row (B1). A neural model under the probe means a wiring or training fault, not a weak idea.

---

## 3. The order of things to run

Every run needs four path overrides. They are the ones of the 13:30 run; change them if the data or checkpoint moved:

```
classifier.data.kfold_root=/data1/fetal-heart-tracing/HDF5_Datasets/one-sided-causal-p5-overlap-aligned-phase-2026-09-15/k_fold_cross_validation_dataset
classifier.source.vae.checkpoint=/data/deid/isilon/MS_model/q3_2026/lag_attn_transformer_cfs/2026-09-23--[06-46]-lag_attn_trf_cfs_revised/model_checkpoints/lag-attn-trf-cfs-epoch=1510.ckpt
classifier.source.hdf5.stats_path=/data1/fetal-heart-tracing/HDF5_Datasets/one-sided-causal-p5-overlap-aligned-phase-2026-09-15/stats.hdf5
classifier.run.out_root=/data/deid/isilon/MS_model/q3_2026/classifier
```

plus `classifier.run.num_workers=4` and the device list `cuda:0,...,cuda:7`.

### Step 1. Close the code state (before any GPU time)

1. Run the quick tests from the repo root: `pytest teb_vae/classifier/tests -m "not slow"`.
2. Run the end-to-end test once: `pytest teb_vae/classifier/tests/test_e2e.py` (a few minutes, CPU).
3. If both pass, stage the second session's changes and commit everything (the CORN set is already staged). If a test fails, send its output.
4. Sync the repo to the server.

### Step 2. Batch 1: the baseline and the first-order ablations

Edit `RUN_ARGS` in `teb_vae/classifier/sweep.py`: keep the six configs already listed (`regularised`, `src_mu_prior`, `src_mu_prior_target_state`, `src_st_ph`, `ctx_off`, `lab_time_matched`), set `overrides` to the five override strings above, `devices` to the eight GPUs, `tag` to `batch1`. Press Run, or:

```
python -m teb_vae.classifier.sweep --configs <the six yamls> --set <each override> --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 --tag batch1
```

What happens: each config runs `--stage all` in its own process into `<out_root>/<stamp>-<run.name>`; the cache is re-extracted once per new key set (the fingerprint covers `sources.py`, so the baseline re-extracts too; budget about an hour each at the spec's measured rate); the finished runs are compared into `<out_root>/compare-<stamp>-batch1/`.

What each run answers:

| Config | Question | Read |
|---|---|---|
| `regularised` | the new baseline: 1 transformer layer at d 64, causal-conv step encoder, dropout 0.2–0.3, segment dropout 0.3, weight decay 5e-2, lr 3e-4, 60 epochs, AUROC monitor, seeds 42–44 with an `ens` row, `stage_offsets` on | `ens` row AUROC vs the 13:30 run's 0.713 / 0.671; H2 seed spread |
| `src_mu_prior` | does dropping the dead `delta_mu` cost anything? | ΔAUROC vs baseline (expect ≈ 0) |
| `src_mu_prior_target_state` | is the 128-d encoder state before the bottleneck richer than the latent? | ΔAUROC; if clearly positive, the VAE bottleneck is the limit (→ Step 6) |
| `src_st_ph` | the model-free reference: ST/PH straight from the shards | ΔAUROC vs baseline; also read the kept-steps line of its extract log (`min_step: auto`) |
| `ctx_off` | what do the latents give without time-from-onset, stage and time-in-second-stage? | ΔAUROC; context alone was 0.60 |
| `lab_time_matched` | does weighting the negatives by the same schedule remove the stage shortcut? | §9 `stage_last` sensitivities; `model` vs `model_stage` gap |

### Step 3. While batch 1 runs: the rank check

After the baseline's extract stage, read `cache_dir` from its `manifest.json` (`source` block) and run:

```
python -m teb_vae.classifier.feature_rank --cache-dir <cache_dir> --out <cache_dir>/feature_rank.json
```

Read `n_under_floor` and `effective_rank` for `mu_prior` and `delta_mu`. A `mu_prior` effective rank far below 64 confirms the bottleneck hypothesis. Send the JSON.

### Step 4. Read batch 1

1. `compare-<stamp>-batch1/comparison.md`: ΔAUROC and Δsensitivity per run with p-values.
2. Each run's `summary.md` §3 (headline and comparator table), §5 (thresholds, NP cost), §8 (time-resolved), §9 (`stage_last`, `healthy_comparator`), §10 (H1 fold spread, H2 seed spread).
3. Each run's `evaluation/verify.json`.
4. Send the files of §4 below, then we pick the baseline for batch 2 (the regularised config, or `lab_time_matched` if it removed the stage shortcut without losing sensitivity).

### Step 5. Batch 2: labels, representation, windows, the 3-class heads

Same sweep with `tag` `batch2`, the reference first, then: `lab_halflife_1h`, `lab_halflife_2h`, `lab_horizon_2h`, `lab_propagate_kwarm3`, `lab_mil_bag`, `src_drop_floor`, `src_states`, `src_st_ph_mu_prior`, `reg_pool4`, `full_window`, `ctx_stage_only`, `three_class_corn`, `three_class_ordinal`. `src_st_ph_mu_prior` refuses to start unless the checkpoint's own stats file equals the `stats_path` above. The 3-class runs pair with the binary ones in `compare` (same cohort digest), and their §7 gives QWK, RPS and the per-class OvR metrics; HIE recall is the number to watch (it was 0 for the auxiliary head).

### Step 6. Batch 3: adaptation (only if Step 2 pointed at the bottleneck)

`adapt_partial_top_block.yaml` then `adapt_lpft_top_block.yaml`, folds 1–3, one seed. They run the VAE online (about 50 segments/s per GPU). Read them against their own `frozen` unit (`summary.md` §4, analysis VF), not against batch 1.

### Step 7. Decide the final configuration

With batches 1–3 read: fix the source keys, the label schedule, the context set and the head; then one final run with 5 seeds (`run.seeds: [42, 43, 44, 45, 46]`) and `eval.bootstrap.refit_threshold: true`. That run's `summary.md` is the one the write-up quotes.

---

## 4. What to send back after each run or batch

From each run directory (zip without `evaluation/figures/` as before):
- `summary.md`, `config.resolved.yaml`, `manifest.json`, `kfold_progress.log`, `kfold_summary.json`
- `evaluation/verify.json`, `evaluation/summary.json`, `evaluation/tables/*.parquet`, `evaluation/tables/inclusion.csv`
- `predictions/guids.parquet`, `predictions/segments.parquet`, `predictions/thresholds.json`
- `folds/fold_*/seed_*/{setup.json, scaler.json, calibration.json, thresholds.json, stage_offsets.json, thresholds_stage.json, fold_results.json}` and `train_results/{metrics_history.csv, epoch_summary.jsonl, full.log}`
- the `ens/` directories of each fold when there are several seeds

From each batch: the `compare-<stamp>-<tag>/` directory (`comparison.md`, `comparison.parquet`, `sweep.json`, figures).

From Step 3: `feature_rank.json`.

If a run fails: `run.log` and the failing unit's `fold_results.json` and `full.log`.

---

## 5. Open decisions

1. **`labels.time_matched` as the baseline?** Decide from batch 1. If it removes the stage shortcut at equal sensitivity, every later run should carry it.
2. **Drop the floor channels by default?** Decide from `src_drop_floor` (batch 2) and the rank check.
3. **CORN task normalisation.** The two tasks are pooled under the row weights. If HIE recall stays near zero in `three_class_corn`, switch to per-task means (one change in `losses.py`, marked `ponytail:`).
4. **Shuffled-control criterion.** The val-label permutation and the fold t interval are in place but have not been exercised on real data. Criterion 7 must pass on the batch-1 runs; if it does not, the next suspect is the per-fold threshold selection of the control.
5. **The NP cap.** `np30` stays primary, with the 4.5-point cost stated in §5 of every summary. `emp15` is the literature anchor.
6. **Stage-stratified operating points** exist as the `_stage` view only. Promoting them to a policy family is a larger change to the threshold engine; do it only if the view shows a clear gain.
7. **Evaluate time.** `stage_offsets` adds one neural model row per unit; with three seeds plus the ensemble, `evaluate` roughly doubles against the 13:30 run. If it becomes the bottleneck, set `classifier.eval.bootstrap.resamples=500` for the ablation batches and keep 2000 for the final run.

---

## 6. Session log

- **2026-10-07, session A.** Read the package and the 13:30 run; profiled the prediction tables (training curves, population shift, alarm-rule counterfactuals, context-only baseline); wrote the diagnosis of §1.2; implemented the committed items of §1.4, then the CORN head. Smoke-checked every change against the 13:30 tables (stage offsets, comparator family, NP cost line, bag loss, CORN probabilities and calibration); did not run the test suite. Memory notes: `classifier-run-2026-10-07-diagnosis`.
- **2026-10-07 to 2026-10-08, session B.** Read the 17:52 run; found the criterion 7 artefact and the log-loss selection problem; changed the shuffled control, criterion 7, the default monitor, and added `labels.time_matched`. Memory note: `classifier-run-2026-10-07-1752-shuffled-fail`.
- **2026-10-08, session A.** Wrote this runbook; added the missing `lab_time_matched.yaml` that session B's sweep list names.

A new session should: read this file, SPEC §13.3 and §17, the two memory notes, then the latest files sent under §4.
