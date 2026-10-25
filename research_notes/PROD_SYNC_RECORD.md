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
5. Run it a second time with `eligible_only` set to `False` and a new `output_dir`.

The outputs hold clinical traces. Keep them on the production box.
