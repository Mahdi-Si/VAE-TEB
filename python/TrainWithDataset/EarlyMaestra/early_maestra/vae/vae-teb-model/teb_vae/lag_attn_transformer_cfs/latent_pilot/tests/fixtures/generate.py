r"""Write the small non-clinical fixtures the smoke scenario runs on.

**Nothing here is clinical and nothing here is committed.** The coefficients come from the
repository's own tiny-shard generator, the identities are invented, the times are invented, and the
whole tree is ignored by version control. A smoke run that finishes proves the stages connect, the
artifacts round-trip and the report renders. It is not evidence about latents, outcomes or
separation, and no number it produces may be quoted.

**Reused rather than rebuilt**, and for one reason each:

* The shards come from ``scripts/make_tiny_shard.py`` in its ``causal_cohort`` mode, through the
  same helper the sibling suites use. That script owns the schema, the per-block warm-up attributes,
  the channel plan and the ``target = code * weight`` convention -- a second writer here would be a
  second description of a warm-up boundary, free to disagree with the one the loader reads.
* The statistics file is the one that generator computed from those shards with the real
  calculator. A hand-rolled stats file is the single shortcut in this tree with a genuinely bad
  failure mode: every shape stays right and every number becomes wrong.
* The checkpoint is a **real one-epoch fit** driven through ``trainer.main``, not a blob saved from
  a freshly constructed model. What the pilot reads out of a checkpoint -- ``model_kwargs`` carrying
  the warm-up tuples the budget resolved against these shards, ``model_class``, and the
  ``resolved_config.yaml`` written beside it -- is exactly what the driver puts there, and a
  hand-assembled blob would carry the same keys while proving none of it.

**Two things are rewritten** on the way out, because the cohort generator answers a different
question than this pilot does:

* **Identities.** The same eight subgroup shards are copied into three splits, so without a rewrite
  every GUID would appear in all three and the split-disjointness assertion -- the one that stops a
  model from recognising an individual instead of generalising -- would be checking nothing. Each
  split's copy carries a split prefix on every GUID.
* **Times.** The cohort shards sit around eleven hours before delivery, which is outside this
  pilot's three-hour window entirely; every anchor would be filtered out and an empty cohort would
  be reported as a clean run. Each recording is given one segment inside the final hour and one in
  the early window, so late eligibility, the six trajectory bins and the paired early/late
  comparison all have something to act on. ``time_from_labor_onset`` and ``second_stage_onset``
  move by the same offset, so the two clinical clocks stay consistent with the segment start rather
  than being left describing the old one.

Generation is not an import side effect. Run it explicitly, once, on the machine that runs the
smoke stage::

    python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.tests.fixtures.generate

It writes into :data:`GENERATED_ROOT`, which is the directory ``configs/smoke.yaml`` names and which
``.gitignore`` excludes. The fit it runs is small but it is a real fit: expect it to take minutes,
not seconds.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

#: Where the artifacts land. The path ``configs/smoke.yaml`` names, and one ``.gitignore`` covers.
GENERATED_ROOT = Path(__file__).resolve().parent / "generated"

#: Prefix of the build's scratch directory under :data:`GENERATED_ROOT`. Named so a leftover is
#: recognisable as scratch and can be swept by the next build.
_BUILD_PREFIX = "_build_"

#: The three splits, in the order the pilot reads them.
SPLITS: Sequence[str] = ("train", "val", "test")

#: The subgroups each split carries. Two, one per binary class, because a split holding one class
#: cannot produce an AUROC at all -- and the canonical basenames are how a shard's subgroup is
#: recovered, so they are the generator's names rather than names of this file's choosing.
SMOKE_SUBGROUPS: Sequence[str] = ("healthy_bg_no_cs", "acidosis_no_cs")

#: Filenames ``configs/smoke.yaml`` names for the checkpoint and the statistics file.
CHECKPOINT_FILENAME = "tiny_checkpoint.ckpt"
STATISTICS_FILENAME = "tiny_stats.hdf5"

#: The configuration the training driver writes beside its checkpoints, and which the strict loader
#: looks for beside the checkpoint it is handed. Copied out together with the checkpoint: a
#: checkpoint separated from it has lost the record of what it was trained on.
RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"

#: Segment start, in seconds before delivery, of each recording's **late** segment -- the one inside
#: the supervised final hour. One per recording in a shard, staggered so the six half-hour
#: trajectory bins are not all filled from one place.
#:
#: Every value is far enough from zero that the whole trimmed segment, and the forecast horizon
#: scored from its last anchor, still end before delivery: a 330-step stored segment trims to
#: 300 steps of 4 seconds, so a start at $-1560$ s puts its last anchor at $-364$ s with room left
#: for the horizon. A start nearer than that would put scored coefficients at or after delivery and
#: the pilot would -- correctly -- drop them.
LATE_EPOCHS: Sequence[float] = (-1560.0, -1960.0, -2360.0)

#: Segment start of each recording's **early** segment, inside the $(2, 3]$ h window the paired
#: temporal comparison reads. The last of them reaches past three hours on purpose, so the window
#: boundary is exercised by a real segment rather than only by a unit test.
EARLY_EPOCHS: Sequence[float] = (-8000.0, -9500.0, -11000.0)


def split_guid(split: str, guid: str) -> str:
    """The identity one source recording is written under in one split.

    Args:
        split: The split.
        guid: The GUID the cohort generator wrote.

    Returns:
        The prefixed GUID. Prefixed rather than renumbered so a reader can still see which source
        recording a fixture row came from, while no GUID appears in two splits.
    """
    return f"{str(split).upper()}-{guid}"


def segment_epochs(n_samples: int, *, segments_per_guid: int) -> List[float]:
    """The segment starts one rewritten shard carries, in row order.

    The cohort generator writes a shard's rows as consecutive segments of consecutive recordings,
    so row $i$ belongs to recording $i // s$ and is that recording's segment $i \\bmod s$. Segment
    zero of each recording goes late, segment one goes early, and a shard written with more
    segments per recording cycles through both tables rather than failing.

    Args:
        n_samples: Rows in the shard.
        segments_per_guid: Segments each recording contributes.

    Returns:
        One start per row, in seconds relative to delivery.
    """
    epochs: List[float] = []
    for index in range(int(n_samples)):
        recording = index // int(segments_per_guid)
        position = index % int(segments_per_guid)
        table = LATE_EPOCHS if position % 2 == 0 else EARLY_EPOCHS
        epochs.append(float(table[recording % len(table)]))
    return epochs


def _rewrite_identity(handle: Any, *, split: str, segments_per_guid: int) -> None:
    """Give one copied shard this split's identities and this pilot's times.

    Args:
        handle: The open HDF5 file, in append mode.
        split: The split this copy belongs to.
        segments_per_guid: Segments each recording contributes.
    """
    import numpy as np

    guids = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in handle["guid"][:]
    ]
    handle["guid"][:] = [split_guid(split, guid) for guid in guids]

    old = np.asarray(handle["epoch"][:], dtype="f8")
    new = np.asarray(
        segment_epochs(len(old), segments_per_guid=segments_per_guid), dtype="f8"
    )
    handle["epoch"][:] = new.astype("f4")

    # The two clinical clocks are stored relative to the segment start, so moving the start without
    # moving them would leave the file describing a labour that ended before it began.
    shift = new - old
    for name in ("time_from_labor_onset", "second_stage_onset"):
        if name in handle:
            values = np.asarray(handle[name][:], dtype="f8")
            handle[name][:] = (values + shift).astype("f4")


def write_split_shards(source_directory: Any, out_root: Any) -> Dict[str, List[str]]:
    """Copy the generated subgroup shards into three splits, rewriting identity and time.

    Args:
        source_directory: Where the cohort generator wrote its eight shards.
        out_root: The fixture root; one subdirectory per split is created under it.

    Returns:
        Split -> the shard paths written, in :data:`SMOKE_SUBGROUPS` order.
    """
    from scripts.make_tiny_shard import COHORT_SEGMENTS_PER_GUID
    from teb_vae.lag_attn_cfs.tests.conftest import write_variant

    source = Path(source_directory)
    root = Path(out_root)
    written: Dict[str, List[str]] = {}
    for split in SPLITS:
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        paths: List[str] = []
        for subgroup in SMOKE_SUBGROUPS:
            destination = write_variant(
                source / f"{subgroup}.hdf5",
                directory / f"{subgroup}.hdf5",
                lambda handle, split=split: _rewrite_identity(
                    handle, split=split, segments_per_guid=COHORT_SEGMENTS_PER_GUID
                ),
            )
            paths.append(str(destination))
        written[split] = paths
    return written


def write_checkpoint(
    shards: Sequence[str], statistics: str, out_root: Any, *, work_directory: Any
) -> Dict[str, str]:
    """Run one real tiny fit and copy its checkpoint and resolved config into the fixture root.

    The base is this package's shipped ``tiny.yaml``. Four leaves are moved because the fixture
    chose them -- both shard lists, the statistics file, and the output directory, since the
    shipped one is a path inside the repository and a fixture must not write there -- and three
    more are **read off the shard**, because they are not choices at all.

    Those three are the channel widths and the phase operator. The shipped configuration declares
    the *integer* operator (44 ``fhr_ph`` + 10 ``up_ph``, so ``c_y`` 80 and ``c_u`` 46), while
    ``write_causal_cohort_shards`` writes the *legacy* one (66 + 15, so 102 and 51) and takes no
    operator argument. Declaring the config's numbers over these shards is refused by the trainer
    before the first batch, and rightly: a width is a property of the HDF5. Reading them here
    rather than pinning the legacy triple keeps this correct if the cohort generator ever gains an
    operator switch -- the fixture then fits whatever it was handed.

    Args:
        shards: The shards to fit on. The **source** cohort shards: the fit only has to produce a
            checkpoint whose geometry matches the fixture's channel and clock contract, and it must
            not be the thing that decides what the pilot's splits contain.
        statistics: The statistics file those shards were generated with.
        out_root: The fixture root the artifacts are copied into.
        work_directory: Where the fit itself writes. Discarded afterwards.

    Returns:
        The written checkpoint and resolved-config paths.

    Raises:
        FileNotFoundError: If the fit left no checkpoint, or no configuration beside it. Either
            means the driver's layout changed and the copy would silently produce an unloadable
            fixture.
    """
    import h5py
    import yaml

    from hdf5_dataset.causal_scattering import PHASE_OPERATOR_LEGACY
    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_cfs.tests.conftest import absolutize_dataset_paths
    from teb_vae.lag_attn_transformer_cfs import trainer as trainer_module

    root = Path(out_root)
    work = Path(work_directory)
    # Created here rather than assumed: the driver makes its own run directory underneath, but the
    # configuration written just below is the first thing to land in this one.
    work.mkdir(parents=True, exist_ok=True)
    tiny = Path(__file__).resolve().parents[3] / "configs" / "tiny.yaml"
    config = absolutize_dataset_paths(load_config(str(tiny)))
    dataset = config["dataset_config"]
    dataset["vae_train_datasets"] = list(shards)
    dataset["vae_test_datasets"] = list(shards)
    dataset["stat_path"] = str(statistics)
    config["general_config"]["folders_config"]["out_dir_base"] = str(work)

    # The shard's own geometry, in the config's three corresponding leaves. Stored layout is
    # ``(N, C, T)``, and an absent operator attribute means the legacy one, which is the same
    # reading the loader and the trainer's own pre-flight check apply.
    vae = config["model_config"]["VAE_model"]
    with h5py.File(str(shards[0]), "r") as handle:
        widths = {
            name: int(handle[name].shape[1])
            for name in ("fhr_st", "fhr_ph", "up_st", "up_ph")
        }
        operator = str(handle.attrs.get("causal_phase_operator", PHASE_OPERATOR_LEGACY))
    vae["c_y"] = widths["fhr_st"] + widths["fhr_ph"]
    vae["c_u"] = (
        widths["up_st"] + widths["up_ph"] if bool(vae.get("use_up_st", True))
        else widths["up_ph"]
    )
    vae["causal_phase_operator"] = operator

    config_path = work / "fixture_fit.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    trainer_module.main(str(config_path))

    # The driver names its run directory from a tag and a timestamp, so it is found rather than
    # predicted; exactly one, or the fit wrote somewhere this function does not know about.
    checkpoint_dirs = sorted(work.rglob("model_checkpoints"))
    if len(checkpoint_dirs) != 1:
        raise FileNotFoundError(
            f"the fixture fit left {len(checkpoint_dirs)} checkpoint directory/ies under {work}; "
            f"expected exactly one."
        )
    checkpoints = sorted(checkpoint_dirs[0].glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"the fixture fit wrote no checkpoint into {checkpoint_dirs[0]}.")
    resolved = checkpoint_dirs[0] / RESOLVED_CONFIG_FILENAME
    if not resolved.is_file():
        resolved = checkpoint_dirs[0].parent / RESOLVED_CONFIG_FILENAME
    if not resolved.is_file():
        raise FileNotFoundError(
            f"no {RESOLVED_CONFIG_FILENAME} beside the fixture checkpoint in {checkpoint_dirs[0]}. "
            f"The strict loader reads the architecture from it, and a checkpoint copied without it "
            f"cannot be rebuilt."
        )

    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / CHECKPOINT_FILENAME
    shutil.copyfile(checkpoints[-1], checkpoint)
    shutil.copyfile(resolved, root / RESOLVED_CONFIG_FILENAME)
    return {
        "checkpoint": str(checkpoint),
        "resolved_config": str(root / RESOLVED_CONFIG_FILENAME),
        "source_checkpoint": str(checkpoints[-1]),
    }


def generate(out_root: Any = None) -> Dict[str, Any]:
    """Write every artifact ``configs/smoke.yaml`` names.

    Args:
        out_root: Destination. ``None`` uses :data:`GENERATED_ROOT`, which is what the smoke
            configuration points at. The build's scratch directory is made underneath it and
            removed again, so nothing is written outside this tree.

    Returns:
        The manifest: the checkpoint, the statistics file and the three split shard lists, plus the
        statement that none of it is clinical -- carried in the record so that a run reading this
        manifest cannot present the result as anything else.
    """
    from teb_vae.lag_attn_cfs.tests.conftest import COHORT_STATS_FILENAME, write_cohort_shards

    root = Path(GENERATED_ROOT if out_root is None else out_root)
    root.mkdir(parents=True, exist_ok=True)

    # Anything a previous build could not remove. Swept on the way in rather than left to
    # accumulate, which is the price of the tolerant cleanup below.
    for stale in root.glob(f"{_BUILD_PREFIX}*"):
        shutil.rmtree(stale, ignore_errors=True)

    # Scratch under the fixture root rather than under the system temp. Three reasons, in order of
    # how much they cost when ignored: an operator watching a run should not have to guess that the
    # minutes are being spent in /tmp; the source shards and the fit can be large, and a system
    # temp is the likeliest filesystem to be small or noexec; and the copies below then stay on one
    # filesystem.
    #
    # ``ignore_cleanup_errors`` because the fit leaves its own log sink open, and on Windows an
    # open file cannot be unlinked -- without it a fit that finished and wrote every artifact
    # raises PermissionError on the way out and reports itself as a failure. The sweep above is
    # what keeps that tolerance from turning into litter.
    with tempfile.TemporaryDirectory(
        prefix=_BUILD_PREFIX, dir=str(root), ignore_cleanup_errors=True
    ) as work:
        source = Path(work) / "shards"
        source.mkdir(parents=True, exist_ok=True)
        shards = list(write_cohort_shards(source))
        statistics = source / COHORT_STATS_FILENAME
        shutil.copyfile(statistics, root / STATISTICS_FILENAME)
        splits = write_split_shards(source, root)
        checkpoint = write_checkpoint(
            shards, str(statistics), root, work_directory=Path(work) / "fit"
        )

    manifest = {
        "root": str(root),
        "checkpoint": checkpoint["checkpoint"],
        "resolved_config": checkpoint["resolved_config"],
        "statistics": str(root / STATISTICS_FILENAME),
        "splits": splits,
        "n_recordings_per_split": len(SMOKE_SUBGROUPS) * len(LATE_EPOCHS),
        "clinical": False,
        "note": (
            "artificial identities, artificial times and coefficients from the repository's tiny "
            "shard generator; a smoke run over these files is a wiring check and no number it "
            "produces is evidence about latents, outcomes or separation"
        ),
    }
    return manifest


def manifest_matches(manifest: Mapping[str, Any], settings: Mapping[str, Any]) -> List[str]:
    """Which paths a smoke configuration names that this manifest did not write.

    The one check that keeps the fixture generator and ``configs/smoke.yaml`` from drifting apart:
    a renamed subgroup or a moved directory would otherwise surface as a missing-file error during
    a run, long after the thing that moved.

    Args:
        manifest: The record :func:`generate` returned.
        settings: Resolved pilot settings, whose ``paths`` block names what the run will read.

    Returns:
        The configured paths that are absent from the manifest, as strings. Empty when the two
        agree.
    """
    written = {
        str(Path(manifest["checkpoint"]).resolve()),
        str(Path(manifest["statistics"]).resolve()),
    }
    for paths in dict(manifest.get("splits") or {}).values():
        written.update(str(Path(path).resolve()) for path in paths)

    paths = dict(dict(settings).get("paths") or {})
    configured: List[str] = []
    for key in ("checkpoint", "statistics"):
        if paths.get(key) is not None:
            configured.append(str(Path(paths[key]).resolve()))
    for key in ("train_shards", "val_shards", "test_shards"):
        configured.extend(str(Path(path).resolve()) for path in paths.get(key) or [])
    return [path for path in configured if path not in written]


def main() -> int:
    """Generate the fixtures into :data:`GENERATED_ROOT` and print what was written.

    Returns:
        A process exit code; zero on success.
    """
    manifest = generate()
    print(f"fixtures written under {manifest['root']}")
    print(f"  checkpoint: {manifest['checkpoint']}")
    print(f"  statistics: {manifest['statistics']}")
    for split, paths in manifest["splits"].items():
        print(f"  {split}: {len(paths)} shard(s)")
    print(f"  {manifest['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_FILENAME",
    "EARLY_EPOCHS",
    "GENERATED_ROOT",
    "LATE_EPOCHS",
    "RESOLVED_CONFIG_FILENAME",
    "SMOKE_SUBGROUPS",
    "SPLITS",
    "STATISTICS_FILENAME",
    "generate",
    "main",
    "manifest_matches",
    "segment_epochs",
    "split_guid",
    "write_checkpoint",
    "write_split_shards",
]
