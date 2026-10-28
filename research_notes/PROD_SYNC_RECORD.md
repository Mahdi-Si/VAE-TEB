# Production box: files to copy by hand

This record lists every change to a file that git does not track here, and every untracked file that
the production box needs. Add one dated section per change. Tracked files reach the box through git
and are listed only for completeness.

## 2026-10-25: UA quality mask, Phases 0 and 1

**`mimo/` (untracked): no change.** The profile turns `MimoSequence.interpolate_input` off by
replacing the method on its class while it reads a record, and it restores the method afterwards
(`mimo_fill` in `profile_signal_quality.py`). Nothing under `mimo/` changes on disk. Phase 2 will use
the same patch from the builder, so it needs no `mimo/` change either.

| File | State in git here | Needed on the production box |
|---|---|---|
| `hdf5_dataset/signal_quality.py` | new, staged (not committed) | yes: the profile imports it |
| `hdf5_dataset/new_pipeline/profile_signal_quality.py` | new, staged (not committed) | yes: the script to run |
| `teb_vae/lag_attn_rws/tests/test_eval_launch.py` | tracked, modified (one `ENTRY_POINTS` line) | no: test only |
| `research_notes/UA_QUALITY_MASK_DESIGN.md` | new, staged (not committed) | no: design notes |
| `research_notes/PROD_SYNC_RECORD.md` | new, staged (not committed) | no: this record |

If you copy the two code files instead of committing them, put them at the same paths below the
repository root on the production box (`.../seqvae_teb/hdf5_dataset/...`).

Python packages: the profile uses `scipy`, `matplotlib`, `pandas` and `torch`, which the builder
already needs. `tabulate` is optional (a nicer table in `REPORT.md`).

To run:

1. Open `hdf5_dataset/new_pipeline/profile_signal_quality.py` on the production box.
2. Check `output_dir` and `screening_csv_path` in `RUN_ARGS`.
3. Set `workers` to the number of free CPU cores.
4. Press Run.

A second run with `eligible_only` set to `False` is not needed: the folds fix the record pool of the
fold build.

The outputs hold clinical traces. Keep them on the production box.

## 2026-10-25, later: quality rules version 2

The first production profile (`tmp/new-k-fold-docs/signal_quality/` here) found three rule defects.
`QUALITY_VERSION` is now 2: no UA level correction, a stricter step test, ceiling runs decided as a
whole, and a new `ISLAND` bit (bit 7, which was `SHIFTED`). The design note, section 2.7, has the
evidence.

**`mimo/` (untracked): no change.**

| File | State in git here | Needed on the production box |
|---|---|---|
| `hdf5_dataset/signal_quality.py` | tracked, modified and staged (not committed) | yes: replace the copy there |
| `hdf5_dataset/new_pipeline/profile_signal_quality.py` | tracked, modified and staged (not committed) | yes: replace the copy there |
| `research_notes/UA_QUALITY_MASK_DESIGN.md` | tracked, modified and staged (not committed) | no |
| `research_notes/PROD_SYNC_RECORD.md` | tracked, modified and staged (not committed) | no |

To run the second profile:

1. Copy the two code files to the same paths on the production box.
2. Open `profile_signal_quality.py` and set `output_dir` in `RUN_ARGS` to a new, empty folder.
3. Keep every other `RUN_ARGS` value of the first run, so the record draw is the same.
4. Press Run.
5. Check that `run_config.json` in the new folder says `"quality_version": 2`.

A new folder is necessary because the script does not delete old example pages. The new `island`
category also changes the page numbers, so old and new pages would mix.

## 2026-10-25, later: quality rules version 3 and the builder (Phase 2)

The second production profile (`tmp/new-k-fold-docs/quality_proflie_2/` here) found one new defect:
a regular breathing wave on real contractions was masked. `QUALITY_VERSION` is now 3, with a new
`OSCILLATION` bit (bit 8, valid). The UA bit field is therefore `uint16`. The builder now writes the
quality fields when `signal_quality` is on. The design note, sections 2.8, 4.3 and 4.7, has the details.

**`mimo/` (untracked): no change.** `mimo_fill` moved from the profile to `create_new_pipeline.py`,
and both scripts use it from there. It still replaces `MimoSequence.interpolate_input` in memory only.

**The flat folder.** The production box keeps the scripts below in one folder. Each import tries the
flat name first (`import signal_quality`) and then the package path (`hdf5_dataset.signal_quality`).
The fallback runs only when the flat module itself is absent, so an error inside a module is not hidden.

| File here | Name in the flat folder | State in git here | Copy to the box |
|---|---|---|---|
| `hdf5_dataset/signal_quality.py` | `signal_quality.py` | tracked, modified, staged | yes: version 3; `raw_fhr_repaired` moved here from the profile |
| `hdf5_dataset/new_pipeline/create_new_pipeline.py` | `create_new_pipeline.py` | tracked, modified, staged | yes: the quality step, `mimo_fill`, flat imports |
| `hdf5_dataset/new_pipeline/build_fold_datasets.py` | `build_fold_datasets.py` | tracked, modified, staged | yes: `signal_quality`, `quality_examples`, the report step, flat imports |
| `hdf5_dataset/new_pipeline/profile_signal_quality.py` | `profile_signal_quality.py` | tracked, modified, staged | yes: the `oscillation` pages; the report before the pages |
| `hdf5_dataset/new_pipeline/quality_report.py` | `quality_report.py` | new, staged | yes: `build_fold_datasets.py` imports it |
| `hdf5_dataset/new_pipeline/fold_eda_analysis.py` | `fold_eda_analysis.py` | tracked, unchanged | only if it is not there yet |
| `teb_vae/lag_attn_rws/eval/events.py` | `events.py` | tracked, unchanged | no: the profile already uses it there |
| `teb_vae/lag_attn_rws/eval/launch.py` | `launch.py` | tracked, unchanged | only if `teb_vae` is not importable on the box |
| `research_notes/UA_QUALITY_MASK_DESIGN.md`, this record | — | tracked, modified, staged | no |

The three scattering modules (`kymatio_phase_scattering.py`, `causal_scattering.py`,
`causal_scattering_torch.py`) stay at `Variational_AutoEncoder/seqvae_teb/hdf5_dataset/`. They use
package imports and cannot run from a flat folder. They did not change.

To run the third profile:

1. Copy the five files marked "yes" into the flat folder, over the old copies.
2. Open `profile_signal_quality.py`.
3. Check that `output_dir` in `RUN_ARGS` is `quality_profile_3`, a new folder.
4. Keep every other `RUN_ARGS` value, so the record draw is the same as before.
5. Press Run.
6. Check that `run_config.json` in the new folder says `"quality_version": 3`.
7. Look at the `oscillation` pages and at the `hie_no_cs` page that version 2 masked (section 2.8).

To build the fold datasets with the quality fields:

1. Open `build_fold_datasets.py`.
2. Set `fold_dir` and `tlo_csv_path` in `RUN_ARGS`.
3. Check that `output_base_path` is a new folder and that `signal_quality` is `True`.
4. Keep `dry_run` at `True` and press Run.
5. Check the GUID counts in the printed table and in `manifest.json`.
6. Set `dry_run` to `False` and set `devices`.
7. Press Run.
8. Read `quality_report/REPORT.md` in the output folder.
9. Compute the dataset statistics for this build into a new file.

Expect slightly more segments than a build without the quality step. The build applies its flat gates
to the signals before the MIMO fill; in the first production profile, this kept 172 more of 19,644
segments.

The outputs, `quality_report/` included, hold clinical traces. Keep them on the production box.

## 2026-10-25, later: loader, statistics and viewer (Phase 3)

Phase 3 makes the quality build usable for training. The loader reads the quality fields on request,
the sequence collate pads them as invalid, and the statistics skip bridged samples and contaminated
coefficients. The dataset viewer draws the quality fields. The design note, section 4.9, has the
behaviour.

**`mimo/` (untracked): no change.**

| File | State in git here | Needed on the production/training box |
|---|---|---|
| `hdf5_dataset/hdf5_dataset.py` | tracked, modified (not staged) | yes: opt-in quality fields, `emit_quality_mask`, the `quality_version` checks |
| `hdf5_dataset/guid_hdf5_dataset.py` | tracked, modified (not staged) | yes: the pad values of the quality fields |
| `hdf5_dataset/calculate_dataset_stats.py` | tracked, modified (not staged) | yes: the masked statistics |
| `hdf5_dataset/plot_dataset_samples.py` | tracked, modified (not staged) | only to view pages there; it imports `teb_vae.lag_attn_rws.eval.launch` |
| `hdf5_dataset/tests/test_quality_loader.py`, `test_quality_stats.py` | new, untracked | no: tests only |
| `teb_vae/lag_attn_rws/tests/test_eval_launch.py` | tracked, modified (one `ENTRY_POINTS` line) | no: test only |
| `neurotome_teb/` | untracked; text edits only | no |
| `research_notes/UA_QUALITY_MASK_DESIGN.md`, this record, `hdf5_dataset/dataset_explained_research.md`, `teb_vae/lag_attn_cfs/DESIGN.md`, `teb_vae/lag_attn_transformer_cfs/DESIGN.md` | tracked, modified (not staged) | no: documentation |
| `hdf5_dataset/PIPELINE.md` | new, untracked | no: documentation |

These four modules use package imports (`from hdf5_dataset.hdf5_dataset import ...`). Put them at the
same paths below the repository root (`.../seqvae_teb/hdf5_dataset/`), not in the flat folder.

**Old statistics files are refused with quality data.** A statistics file without `quality_version`
does not describe the bridged signals. The loader raises at construction, names both files, and asks
you to recompute the statistics for this build.

To compute the statistics for the quality build:

1. Copy the three `hdf5_dataset/` files marked "yes" to the same paths on the box.
2. Open `pre_training_dataset/dataset_files.yaml` in the output folder of the quality build.
3. Open `hdf5_dataset/calculate_dataset_stats.py`.
4. Set `input_files` in `RUN_ARGS` to every file of both lists, train and test.
5. Set `output_file` to a new path.
6. Keep `trim_minutes` at 1.0, the loader's value.
7. Keep `quality_mask` and `quality_rho_epsilon` at `None` (mask on, $\varepsilon$ = 0.1).
8. Press Run.
9. Check that the summary prints `quality_version=3, mask rho <= 0.1`.
10. Set `stat_path` in the training config to the new file.

The comment at the top of `dataset_files.yaml` gives the reason for step 4: `stat_path` must be
computed over all of these files, train and test alike.

The model configs do not name a quality field yet (Phase 4). With their current `load_fields`, a model
trains on the bridged `fhr`/`up` and the coefficients computed from them, with the new statistics.

## 2026-10-28: quality rules version 4 and the report pages

A diagnosis of the version-3 build (`diagnose_signal_quality.py`, committed in 512ba66) found six
defects. `QUALITY_VERSION` is now 4. `CENSORED` is a ceiling crest only. Three new bits come in:
`REST` (a clamped rest, valid), `PUSH` (a second-stage push, valid) and `FLAT` (a flat line after
300 s, invalid). The UA $\rho$ also counts level shifts, and a new per-segment field,
`fhr_raw_match`, stores the raw FHR channel's match share. Each report page now shows the UA
coefficients of `up_raw` before the mask and the bridge. The design note, sections 2.9, 4.3, 4.5 and
4.8, has the details.

**`mimo/` (untracked): no change.**

| File here | Name in the flat folder | State in git here | Copy to the box |
|---|---|---|---|
| `hdf5_dataset/signal_quality.py` | `signal_quality.py` | tracked, modified, staged | yes: version 4 |
| `hdf5_dataset/new_pipeline/create_new_pipeline.py` | `create_new_pipeline.py` | tracked, modified, staged | yes: the second-stage mask, the step term of $\rho$, `fhr_raw_match` |
| `hdf5_dataset/new_pipeline/build_fold_datasets.py` | `build_fold_datasets.py` | tracked, modified, staged | yes: hands the transform to the report |
| `hdf5_dataset/new_pipeline/quality_report.py` | `quality_report.py` | tracked, modified, staged | yes: the new pages, columns and raw-coefficient row |
| `hdf5_dataset/hdf5_dataset.py` | — (package path) | tracked, modified, staged | yes, on the training box: `fhr_raw_match` is a quality field |
| `teb_vae/lag_attn_transformer_cfs/configs/default.yaml` | — | tracked, modified, staged | yes, on the training box: comment only |
| `hdf5_dataset/tests/test_signal_quality_rules.py` | — | new, staged | no: tests only |
| `hdf5_dataset/tests/test_quality_loader.py` | — | untracked, modified | no: tests only |
| `hdf5_dataset/PIPELINE.md`, `hdf5_dataset/dataset_explained_research.md`, `research_notes/UA_QUALITY_MASK_DESIGN.md`, `teb_vae/lag_attn_transformer_cfs/DESIGN.md`, this record | — | documentation | no |

The report's raw-coefficient row needs the causal bank. The build computes the masks again on the CPU
when the transform ran in worker processes, so the report step takes a little longer. If the bank
cannot be built, the pages leave the row out and the log says why.

To build the fold datasets and the pretraining shards with version 4:

1. Copy the four flat-folder files marked "yes" over the old copies.
2. Open `build_fold_datasets.py`.
3. Set `output_base_path` in `RUN_ARGS` to a new folder.
4. Check that `signal_quality` is `True`.
5. Keep `dry_run` at `True` and press Run.
6. Check the GUID counts in the printed table and in `manifest.json`.
7. Set `dry_run` to `False` and set `devices`.
8. Press Run.
9. Check that `manifest.json` says `"quality_version": 4` under `settings`.
10. Read `quality_report/REPORT.md`, and look at the `rest`, `push` and `flat` pages.
11. Check that no page title names a mismatch of the recomputed coefficients.
12. Compute the dataset statistics for this build into a new file (section "Phase 3" above).
13. Check that the statistics summary prints `quality_version=4, mask rho <= 0.1`.

A version-3 statistics file is refused with version-4 data, and the two versions cannot be mixed in
one file list. The outputs, `quality_report/` included, hold clinical traces. Keep them on the
production box.

## 2026-10-28, later: the model reads the quality mask (Phase 4, second part)

The dataset files and the build do not change in this entry; the version-4 build of the previous
entry is the one to use. The loader, the training task, the model inputs, the evaluation and the
classifier cache change. `research_notes/UA_QUALITY_MASK_DESIGN.md`, sections 4.10 and 8.2, has the
details.

| File here | State in git here | Copy to the box |
|---|---|---|
| `hdf5_dataset/hdf5_dataset.py` | tracked, modified, staged | yes, on the training box: `fhr_step_mask`, the $\varepsilon$ refusal, the legacy append refusal |
| `hdf5_dataset/guid_hdf5_dataset.py` | tracked, modified, staged | yes, on the training box: the NaN pad of `fhr_raw_match` |
| `hdf5_dataset/new_pipeline/quality_report.py` | tracked, modified, staged | yes, to the flat folder: the `usable_ua_steps` column |
| `teb_vae/lag_attn_rws/event_weight.py`, `task.py`, `trainer.py` | tracked, modified, staged | yes, on the training box |
| `teb_vae/lag_attn_fs/task.py`, `teb_vae/lag_attn_cfs/task.py`, `trainer.py`, `nets/causal_inputs.py` | tracked, modified, staged | yes, on the training box |
| `teb_vae/lag_attn_cfs/eval/` (`collect.py`, `metrics.py`, `traces.py`, `analyses/events.py`, `analyses/forecast.py`), `teb_vae/lag_slot_transformer_cfs/eval/collect.py`, `teb_vae/lag_attn_transformer_cfs/eval/verify.py` | tracked, modified, staged | yes, on the training box |
| `teb_vae/classifier/sources.py` | tracked, modified, staged | yes, on the training box: the cache fingerprint |
| `teb_vae/lag_attn_transformer_cfs/configs/` (`default.yaml`, `tiny.yaml`, `smoke_hie.yaml`, `sweep_no_event_weight.yaml`, new `sweep_source_event_mask.yaml`) | tracked, modified or new, staged | yes, on the training box |
| tests | new or modified, staged | no: tests only |
| documentation | — | no |

The classifier cache key now includes each shard's `quality_version`, so the first classifier run on
the version-4 build computes its cache again.

To train the transformer cfs model on the version-4 build:

1. Build the version-4 datasets and their statistics (previous entry).
2. Copy the files marked "yes" to the training box.
3. Point `vae_train_datasets`, `vae_test_datasets` and `stat_path` in `default.yaml` to the new build.
4. Start the run.
5. In the first epoch, compare `source_mask_fraction` with `usable_ua_steps` in `quality_report/REPORT.md`.
6. Check that `event_anchor_frac` is above zero.

The loader refuses a statistics file whose `quality_rho_epsilon` differs from the step masks'
`rho_epsilon` (0.1 in `default.yaml`). If you compute the statistics with another $\varepsilon$,
change `ua_step_mask.rho_epsilon` to the same value.

**The mask decision for the final build (2026-10-28).** The build does not depend on which mask the
model reads, because the files store the raw bits and $\rho$ side by side. The final choice, with
the numbers from the version-3 diagnosis, is in `research_notes/UA_QUALITY_MASK_DESIGN.md` section
4.11: $\rho \le 0.1$ for the UA lag keys, `up_valid` for the contraction detector, and the raw FHR
bits for the target. `hdf5_dataset/dataset_explained_research.md` section 11.4 has the build, report,
statistics and config steps for the final files. Only the current quality version (4) is supported.

## 2026-10-28, later: no more copies; the box runs from the git clone

From this entry on, nothing is copied by hand. The box runs the build from its clone of this
repository, and `git pull` brings every change. The old flat folder is no longer used; rename it
so that none of its old copies (`signal_quality.py`, `launch.py`, `create_new_pipeline.py`) can
be imported by accident.

**What changed in the imports.** `create_new_pipeline.py` now imports the scattering modules as
`hdf5_dataset.*`, from the clone, and keeps `Variational_AutoEncoder.seqvae_teb.hdf5_dataset.*`
only as a fallback. It imports them before `early_maestra`, because `hdf5_dataset` pins the
vendored kymatio and `early_maestra` imports kymatio itself. Then
`hdf5_dataset/new_pipeline/external_paths.py` puts the two outside folders on `sys.path`:
the folder that holds `early_maestra/` and the folder that holds `mimo/`. They go just after the
repository root, so an older copy elsewhere on the path cannot win. The spawned worker processes
inherit `sys.path`.

`external_paths.py` finds the folders in this order: the environment variable
`TEB_EXTERNAL_ROOTS` (folders separated by `:`), the list `EXTERNAL_ROOTS` in the file, and then a
search for `EarlyMaestra/` and `MIMO_Sequence_Trainer/` in `<repo>/mimo`, `<repo>/tmp/TrainWithDataset`,
`<repo>/../TrainWithDataset`, `<repo>/..` and `~/tmp/TrainWithDataset`.

To set up the box:

1. Run `git pull` in the clone.
2. Rename the old flat folder.
3. Open `hdf5_dataset/new_pipeline/external_paths.py`.
4. Set `EXTERNAL_ROOTS` to your two folders, for example `.../tmp/TrainWithDataset/EarlyMaestra` and `.../tmp/TrainWithDataset/MIMO_Sequence_Trainer`.
5. Open `hdf5_dataset/new_pipeline/prod_import_probe.py` with the build's interpreter.
6. Press Run.
7. Check in `prod_import_report.txt` that `early_maestra` and `mimo` resolve to your folders.
8. Check that both build modules print `OK` and that `QUALITY_VERSION=4`.

If the folders are under one of the searched places, step 4 is not needed; the report says what was
found. Step 4 is a local edit on the box, so keep it out of your commits, or set
`TEB_EXTERNAL_ROOTS` in the PyCharm run configuration instead.
