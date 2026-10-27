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
