# Classifier runbook

This file gives the procedure to run the classifier. [`SPEC.md`](SPEC.md) is the design: §13.3 lists the shipped configs, §14 the CLI and the run directory.

---

## 1. Paths every real run needs

`default.yaml` holds server paths. When the data, the checkpoint or the output root are elsewhere, override these four keys with `--set` (or the `overrides` list of `RUN_ARGS`):

```
classifier.data.kfold_root=<.../k_fold_cross_validation_dataset>
classifier.source.vae.checkpoint=<.../model_checkpoints/<checkpoint>.ckpt>
classifier.source.hdf5.stats_path=<.../stats.hdf5>
classifier.run.out_root=<classifier output root>
```

Two execution settings are often set with them: `classifier.run.num_workers=4` and a device list such as `cuda:0,cuda:1,...,cuda:7` (fold-parallel train and predict, SPEC §14.4). Neither is part of the config digest.

---

## 2. Run the tests

Run the tests before a commit, from the repository root.

1. Run the quick tests: `pytest teb_vae/classifier/tests -m "not slow"`.
2. Run the end-to-end test: `pytest teb_vae/classifier/tests/test_e2e.py`.

The end-to-end test runs `smoke.yaml` on a generated 3-fold fixture on CPU (SPEC §15).

---

## 3. Run one config

Edit `RUN_ARGS` at the bottom of `teb_vae/classifier/run.py` and press Run, or use the command line:

```
python -m teb_vae.classifier.run --config teb_vae/classifier/configs/<config>.yaml --stage all \
       --set <override> ... --devices cuda:0,cuda:1,...
```

- A new run goes to `<out_root>/<YYYY-mm-dd--HH-MM-SS>-<run.name>/`.
- `--run-dir <existing run>` resumes a run. Finished stages and locked units are skipped. The config digest must match.
- `--stage evaluate`, `report` or `verify` re-runs that stage on an existing run. Any stage that runs marks the later stages pending.
- `python -m teb_vae.classifier.run compare --runs RUN_A RUN_B ... --out DIR` compares 2–6 finished runs that share a cohort digest.

---

## 4. Run a sweep of configs

`teb_vae/classifier/sweep.py` runs a list of configs one after another, then compares the finished runs.

1. Open `RUN_ARGS` at the bottom of `sweep.py`.
2. Set `configs` to the configs in run order. The first config is the reference of the comparison.
3. Set `overrides` to the override strings of §1.
4. Set `devices` to the device slots.
5. Set `tag` to a short name for the comparison directory.
6. Press Run.

The command-line form is:

```
python -m teb_vae.classifier.sweep --configs <yaml> <yaml> ... --set <override> ... \
       --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 --tag <tag>
```

What the sweep does:
- Each config runs `run.py --stage all` in its own process, into `<out_root>/<stamp>-<run.name>`.
- A run re-extracts the feature cache when its fingerprint changes (source keys, checkpoint, code digest of the extraction path; SPEC §8.4).
- The finished runs are compared into `<out_root>/compare-<stamp>-<tag>/`, at most six runs per comparison. A longer list is compared in groups that share the reference.
- `sweep.json` in that directory maps every config to its run directory, exit code and wall time.

The shipped `RUN_ARGS` list is `regularised`, `src_mu_prior`, `src_mu_prior_target_state`, `src_st_ph`, `ctx_off` and `lab_time_matched`. Any other config of SPEC §13.3 can be added. Two constraints apply:
- `src_st_ph_mu_prior` (`source.kind: vae+hdf5`) refuses to start unless the checkpoint's own stats file is the `stats_path` of the run (L13).
- `adapt_partial_top_block` and `adapt_lpft_top_block` run the VAE online, so they are slow. Read each against its own `frozen` unit (`summary.md` §4, analysis VF).

---

## 5. Check the spread and rank of a feature cache

`feature_rank.py` reads a frozen feature cache and reports, per key, the per-channel spread and the effective rank.

1. After the `extract` stage, read `source.cache_dir` from the run's `manifest.json`.
2. Run `python -m teb_vae.classifier.feature_rank --cache-dir <cache_dir> --out <cache_dir>/feature_rank.json`.

The report holds, per key: `n_channels`; `std_min`, `std_median`, `std_max`; `n_under_floor` (channels under the scaler floor $\max(10^{-3}, 0.1 \cdot \text{median positive std})$, SPEC §8.5); `n_under_10pct_of_key_median`; and, for the correlation and the covariance spectrum, `effective_rank` ($\exp$ of the entropy of the normalised eigenvalues), `n_90`, `n_99` and `top_share`. Optional flags are `--sample-segments`, `--sample-steps` and `--seed`.

---

## 6. Settings for a seed ensemble and refit intervals

- Set `run.seeds` to several seeds, e.g. `[42, 43, 44, 45, 46]`. Each fold then gets an `ens` unit that averages the seeds' logits and is calibrated once (SPEC §10.7).
- Set `eval.bootstrap.refit_threshold: true` to re-select the thresholds inside every bootstrap replicate (SPEC §11.7).
- `eval.stage_offsets: true` adds one `<kind>_stage` model row per unit, so `evaluate` takes longer. `classifier.eval.bootstrap.resamples=500` shortens it.

---

## 7. Files of a finished run

Archive these files from each run directory (`evaluation/figures/` can be left out):
- `summary.md`, `config.resolved.yaml`, `manifest.json`, `kfold_progress.log`, `kfold_summary.json`
- `evaluation/verify.json`, `evaluation/summary.json`, `evaluation/tables/*.parquet`, `evaluation/tables/inclusion.csv`
- `predictions/guids.parquet`, `predictions/segments.parquet`, `predictions/thresholds.json`
- `folds/fold_*/seed_*/{setup.json, scaler.json, calibration.json, thresholds.json, stage_offsets.json, thresholds_stage.json, fold_results.json}` and `train_results/{metrics_history.csv, epoch_summary.jsonl, full.log}`
- the `ens/` directories of each fold when there are several seeds

From each sweep, archive the `compare-<stamp>-<tag>/` directory (`comparison.md`, `comparison.parquet`, `sweep.json`, figures). From §5, archive `feature_rank.json`.

If a run fails, collect `run.log`, and the `fold_results.json` and `full.log` of the failing unit.

Where to read a run:
- `evaluation/verify.json` holds the gate verdicts (SPEC §11.15). `passed: true` means no criterion failed.
- `summary.md` §3 holds the headline table and the pre-specified healthy-comparator table (`all_healthy`, `bg_healthy`, `no_bg_healthy`). §5 holds the thresholds, §8 the time-resolved table, §9 the subgroups (`stage_last`, `healthy_comparator`) and §10 the fold and seed spread (H1, H2).
- `comparison.md` holds the paired differences between runs (paired GUID bootstrap, DeLong, Nadeau–Bengio).
