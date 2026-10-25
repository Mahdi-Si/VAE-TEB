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
