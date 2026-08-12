r"""Smoke check for the phase-harmonic channel selection in the new pipeline.

Verifies the selection logic, the HDF5 geometry it produces, the per-channel
provenance attrs, and the fail-fast guard — without needing a single real
sample. Run from the repository root::

    python hdf5_dataset/smoke_check_channel_selection.py

Exits non-zero on the first failure.

Why a script and not pytest: it needs no fixtures and no data, so it stays
runnable straight from the Run button with nothing installed but the package's
own dependencies. ``hdf5_dataset/tests/`` now holds the pytest suite, and
:func:`_import_pipeline` is shared with it rather than duplicated there.

Expected counts at the production geometry ($J=11$, $Q=4$, $T=16$,
``shape=5280``, $f_s = 4$ Hz) with the shipped constants:
``fhr_ph`` = 66, ``fhr_up_ph`` = 79, ``up_ph`` = 15.

It also prints the resolved channel layout for **both** transform variants — the same lines a
build logs before it commits — so an operator can read the causal widths, the dropped channels and
the warm-up and delay ranges without starting one.
"""

import os
import sys
import types
import shutil
import tempfile
import importlib
import traceback
from typing import Any, Callable, List, Tuple

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Import shim
# ---------------------------------------------------------------------------
def _import_pipeline() -> Any:
    """Import ``create_new_pipeline`` outside the production environment.

    The pipeline imports ``early_maestra.adaptor.mimo_adaptor`` (a prod-only
    package) and reaches ``kymatio_phase_scattering`` through the
    ``Variational_AutoEncoder.seqvae_teb`` path that only exists on the
    production box. Neither is needed for the selection logic, so both are
    stubbed / aliased here.

    Returns:
        The imported ``create_new_pipeline`` module.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Stub the prod-only adaptor: the pipeline imports the symbol but the
    # selection code path never constructs one.
    early_maestra = types.ModuleType("early_maestra")
    early_maestra.__path__ = []  # type: ignore[attr-defined]
    adaptor = types.ModuleType("early_maestra.adaptor")
    adaptor.__path__ = []  # type: ignore[attr-defined]
    mimo = types.ModuleType("early_maestra.adaptor.mimo_adaptor")

    class EarlyMaestraMimoAdaptor:  # pragma: no cover - stub
        """Placeholder for the production adaptor."""

    mimo.EarlyMaestraMimoAdaptor = EarlyMaestraMimoAdaptor  # type: ignore[attr-defined]
    sys.modules.update({
        "early_maestra": early_maestra,
        "early_maestra.adaptor": adaptor,
        "early_maestra.adaptor.mimo_adaptor": mimo,
    })

    # Alias the prod import path onto the in-repo package. Each submodule the pipeline imports is
    # registered explicitly as well as the package: without that, `from
    # Variational_AutoEncoder.seqvae_teb.hdf5_dataset.X import Y` imports a *second* copy of X
    # under the aliased name, and a class from one copy is not the class from the other.
    import hdf5_dataset
    import hdf5_dataset.causal_scattering as causal
    import hdf5_dataset.causal_scattering_torch as causal_torch
    import hdf5_dataset.kymatio_phase_scattering as kps

    vae = types.ModuleType("Variational_AutoEncoder")
    vae.__path__ = []  # type: ignore[attr-defined]
    seqvae = types.ModuleType("Variational_AutoEncoder.seqvae_teb")
    seqvae.__path__ = []  # type: ignore[attr-defined]
    prefix = "Variational_AutoEncoder.seqvae_teb.hdf5_dataset."
    sys.modules.update({
        "Variational_AutoEncoder": vae,
        "Variational_AutoEncoder.seqvae_teb": seqvae,
        "Variational_AutoEncoder.seqvae_teb.hdf5_dataset": hdf5_dataset,
        prefix + "kymatio_phase_scattering": kps,
        prefix + "causal_scattering": causal,
        prefix + "causal_scattering_torch": causal_torch,
    })

    return importlib.import_module(
        "hdf5_dataset.new_pipeline.create_new_pipeline"
    )


# ---------------------------------------------------------------------------
# Tiny check harness
# ---------------------------------------------------------------------------
_CHECKS: List[Tuple[str, Callable[[Any], None]]] = []


def check(name: str) -> Callable:
    """Register a check function under a human-readable name."""

    def decorator(fn: Callable[[Any], None]) -> Callable[[Any], None]:
        _CHECKS.append((name, fn))
        return fn

    return decorator


def _require(condition: bool, message: str) -> None:
    """Raise ``AssertionError`` with ``message`` when ``condition`` is false."""
    if not condition:
        raise AssertionError(message)


# Expected widths with the shipped constants. Kept as literals on purpose: the
# point of the check is to catch an unintended change to the selection, so
# deriving them from the same constants under test would be circular.
EXPECTED_FHR_PH = 66
EXPECTED_UP_PH = 15
EXPECTED_CROSS = 79
EXPECTED_ST = 43
SIGNAL_LENGTH = 5280
SEQUENCE_LENGTH = SIGNAL_LENGTH // 16


def _masks(cnp: Any) -> dict:
    """Compute the selections at the production geometry."""
    return cnp.compute_scattering_masks(SIGNAL_LENGTH, scattering_T=16)


# ---------------------------------------------------------------------------
# Selection-level checks
# ---------------------------------------------------------------------------
@check("channel counts match the documented selection")
def _check_counts(cnp: Any) -> None:
    m = _masks(cnp)
    fhr, up = m["fhr_ph_selection"], m["up_ph_selection"]
    _require(
        fhr.n_channels == EXPECTED_FHR_PH,
        f"fhr_ph: expected {EXPECTED_FHR_PH}, got {fhr.n_channels}",
    )
    _require(
        up.n_channels == EXPECTED_UP_PH,
        f"up_ph: expected {EXPECTED_UP_PH}, got {up.n_channels}",
    )
    _require(
        m["n_cross"] == EXPECTED_CROSS,
        f"fhr_up_ph: expected {EXPECTED_CROSS}, got {m['n_cross']}",
    )


@check("diagonal (k=0) is excluded from both self-phase blocks")
def _check_no_diagonal(cnp: Any) -> None:
    m = _masks(cnp)
    for field, sel in (("fhr_ph", m["fhr_ph_selection"]),
                       ("up_ph", m["up_ph_selection"])):
        n_diag = int((sel.i == sel.j).sum())
        _require(n_diag == 0, f"{field}: {n_diag} diagonal channels leaked in")
        # k=4 is the lowest admitted step, so every power must be near 2 or above.
        _require(
            float(sel.power.min()) > 1.9,
            f"{field}: min power {sel.power.min():.3f} implies a k<4 channel",
        )


@check("selected frequencies stay inside the configured Hz band")
def _check_band_containment(cnp: Any) -> None:
    # Guards against an fs-conversion regression — comparing a Hz threshold
    # against normalised xi is the exact defect this selection replaces.
    m = _masks(cnp)
    for field, sel in (("fhr_ph", m["fhr_ph_selection"]),
                       ("up_ph", m["up_ph_selection"])):
        lo, hi = sel.band_hz
        _require(
            float(sel.xi_i_hz.min()) >= lo - 1e-9,
            f"{field}: xi_i {sel.xi_i_hz.min():.5f} Hz below floor {lo}",
        )
        _require(
            float(sel.xi_j_hz.max()) <= hi + 1e-9,
            f"{field}: xi_j {sel.xi_j_hz.max():.5f} Hz above ceiling {hi}",
        )
        # i is the lower-frequency filter of the pair, by construction.
        _require(
            bool(np.all(sel.xi_i_hz <= sel.xi_j_hz + 1e-9)),
            f"{field}: i/j frequency ordering violated",
        )


@check("metadata arrays are the same length as the mask population")
def _check_metadata_invariant(cnp: Any) -> None:
    m = _masks(cnp)
    for field, sel in (("fhr_ph", m["fhr_ph_selection"]),
                       ("up_ph", m["up_ph_selection"])):
        n_mask = int(sel.mask.sum().item())
        for attr in ("i", "j", "xi_i_hz", "xi_j_hz", "power"):
            got = len(getattr(sel, attr))
            _require(
                got == n_mask,
                f"{field}.{attr}: length {got} != mask population {n_mask}",
            )
        _require(
            sel.n_channels == n_mask,
            f"{field}: n_channels {sel.n_channels} != mask population {n_mask}",
        )


@check("keep-diagonal variant is a one-constant switch")
def _check_keep_diagonal_switch(cnp: Any) -> None:
    original = cnp.PHASE_HARMONIC_K_STEPS
    try:
        cnp.PHASE_HARMONIC_K_STEPS = (0, 4, 6, 8)
        m = cnp.compute_scattering_masks(SIGNAL_LENGTH, scattering_T=16)
        fhr, up = m["fhr_ph_selection"].n_channels, m["up_ph_selection"].n_channels
        _require(fhr == 94, f"keep-diagonal fhr_ph: expected 94, got {fhr}")
        _require(up == 26, f"keep-diagonal up_ph: expected 26, got {up}")
    finally:
        cnp.PHASE_HARMONIC_K_STEPS = original


# ---------------------------------------------------------------------------
# HDF5-level checks
# ---------------------------------------------------------------------------
def _build_file(cnp: Any, path: str, masks: dict) -> None:
    """Create an HDF5 with the production schema and append one zero sample."""
    cnp.create_initial_hdf5(
        path=path,
        len_signal=SIGNAL_LENGTH,
        len_sequence=SEQUENCE_LENGTH,
        fhr_ph_selection=masks["fhr_ph_selection"],
        n_fhr_st_channels=EXPECTED_ST,
        n_cross_phase_channels=masks["n_cross"],
        n_up_st_channels=EXPECTED_ST,
        up_ph_selection=masks["up_ph_selection"],
    )
    k = 2
    z = lambda c: np.zeros((k, c, SEQUENCE_LENGTH), dtype=np.float32)
    cnp.append_samples_batch(
        path=path,
        fhr_batch=np.zeros((k, SIGNAL_LENGTH), dtype=np.float32),
        up_batch=np.zeros((k, SIGNAL_LENGTH), dtype=np.float32),
        fhr_st_batch=z(EXPECTED_ST),
        fhr_ph_batch=z(masks["fhr_ph_selection"].n_channels),
        fhr_up_ph_batch=z(masks["n_cross"]),
        target_batch=np.zeros((k, SEQUENCE_LENGTH), dtype=np.float32),
        weight_batch=np.zeros((k, SEQUENCE_LENGTH), dtype=np.float32),
        guid_batch=["SMOKE0", "SMOKE1"],
        epoch_batch=np.zeros(k, dtype=np.float32),
        cs_label_batch=np.zeros(k, dtype=np.uint8),
        bg_label_batch=np.ones(k, dtype=np.uint8),
        tlo_batch=np.zeros(k, dtype=np.float32),
        second_stage_batch=np.zeros(k, dtype=np.float32),
        up_st_batch=z(EXPECTED_ST),
        up_ph_batch=z(masks["up_ph_selection"].n_channels),
    )


@check("HDF5 write round-trip produces mask-derived widths")
def _check_write_round_trip(cnp: Any) -> None:
    # This is the regression test for the hardcoded fhr_ph=44 landmine: on the
    # pre-change code the append below raises an h5py broadcast error.
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "smoke.hdf5")
        _build_file(cnp, path, _masks(cnp))
        with h5py.File(path, "r") as f:
            for field, expected in (
                ("fhr_st", EXPECTED_ST),
                ("fhr_ph", EXPECTED_FHR_PH),
                ("fhr_up_ph", EXPECTED_CROSS),
                ("up_st", EXPECTED_ST),
                ("up_ph", EXPECTED_UP_PH),
            ):
                _require(field in f, f"{field} missing from the written file")
                got = int(f[field].shape[1])
                _require(
                    got == expected,
                    f"{field}: stored width {got}, expected {expected}",
                )
                _require(
                    int(f[field].shape[0]) == 2,
                    f"{field}: expected 2 appended samples",
                )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("per-channel provenance attrs round-trip")
def _check_attrs_round_trip(cnp: Any) -> None:
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "smoke.hdf5")
        masks = _masks(cnp)
        _build_file(cnp, path, masks)
        expected_bands = {
            "fhr_ph": (cnp.FHR_PHASE_BAND_HZ, EXPECTED_FHR_PH),
            "up_ph": (cnp.UP_PHASE_BAND_HZ, EXPECTED_UP_PH),
        }
        with h5py.File(path, "r") as f:
            for field, (band, n_expected) in expected_bands.items():
                attrs = f[field].attrs
                for key in ("sel_i", "sel_j", "sel_xi_i_hz", "sel_xi_j_hz",
                            "sel_power", "sel_band_hz", "sel_k_steps"):
                    _require(key in attrs, f"{field}: attr {key} missing")
                for key in ("sel_i", "sel_j", "sel_xi_i_hz", "sel_xi_j_hz",
                            "sel_power"):
                    got = len(attrs[key])
                    _require(
                        got == n_expected,
                        f"{field}.{key}: length {got} != {n_expected}",
                    )
                _require(
                    tuple(attrs["sel_k_steps"]) == tuple(cnp.PHASE_HARMONIC_K_STEPS),
                    f"{field}: sel_k_steps {tuple(attrs['sel_k_steps'])} "
                    f"!= {tuple(cnp.PHASE_HARMONIC_K_STEPS)}",
                )
                _require(
                    np.allclose(attrs["sel_band_hz"], np.asarray(band, np.float32)),
                    f"{field}: sel_band_hz {attrs['sel_band_hz']} != {band}",
                )
                # Provenance must describe the data actually stored.
                _require(
                    bool(np.all(attrs["sel_xi_i_hz"] <= attrs["sel_xi_j_hz"] + 1e-9)),
                    f"{field}: stored i/j frequency ordering violated",
                )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("channel-count mismatch fails loudly, not as an h5py broadcast error")
def _check_assertion_fires(cnp: Any) -> None:
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "wrong.hdf5")
        masks = _masks(cnp)
        # Build the file with a deliberately stale selection (the diagonal
        # variant), then apply the shipped selection to it.
        stale_k = cnp.PHASE_HARMONIC_K_STEPS
        try:
            cnp.PHASE_HARMONIC_K_STEPS = (0, 4, 6, 8)
            stale = cnp.compute_scattering_masks(SIGNAL_LENGTH, scattering_T=16)
        finally:
            cnp.PHASE_HARMONIC_K_STEPS = stale_k
        cnp.create_initial_hdf5(
            path=path,
            len_signal=SIGNAL_LENGTH,
            len_sequence=SEQUENCE_LENGTH,
            fhr_ph_selection=stale["fhr_ph_selection"],  # 94 wide
            n_fhr_st_channels=EXPECTED_ST,
            n_cross_phase_channels=stale["n_cross"],
            n_up_st_channels=EXPECTED_ST,
            up_ph_selection=stale["up_ph_selection"],
        )
        try:
            # Empty records list: the guard is the only thing that runs.
            cnp.create_hdf5_dataset_from_records_list(
                hdf5_path=path,
                records_list=[],
                cs_label=False,
                bg_label=True,
                pre_defined_target=1,
                precomputed_masks=masks,  # 66 / 15 wide
                labor_onset_map={},
                second_stage_map={},
                base_block_size=cnp.BASE_BLOCK_SIZE,
                overlap_percentage=cnp.OVERLAP_PERCENTAGE,
                run_guid_analysis=False,
                verbose=False,
            )
        except ValueError as exc:
            _require(
                "Channel-count mismatch" in str(exc),
                f"wrong error text: {exc}",
            )
            return
        raise AssertionError("expected a ValueError, none was raised")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_writer(cnp: Any, path: str, masks: dict) -> None:
    """Invoke the writer with an empty record list, so only the guard runs."""
    cnp.create_hdf5_dataset_from_records_list(
        hdf5_path=path,
        records_list=[],
        cs_label=False,
        bg_label=True,
        pre_defined_target=1,
        precomputed_masks=masks,
        labor_onset_map={},
        second_stage_map={},
        base_block_size=cnp.BASE_BLOCK_SIZE,
        overlap_percentage=cnp.OVERLAP_PERCENTAGE,
        run_guid_analysis=False,
        verbose=False,
    )


@check("a missing coefficient dataset is rejected, not silently skipped")
def _check_missing_dataset_rejected(cnp: Any) -> None:
    # append_samples_batch drops optional blocks whose dataset is absent, so a
    # file built without up_ph would silently lose it for the whole run.
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "no_up_ph.hdf5")
        masks = _masks(cnp)
        cnp.create_initial_hdf5(
            path=path,
            len_signal=SIGNAL_LENGTH,
            len_sequence=SEQUENCE_LENGTH,
            fhr_ph_selection=masks["fhr_ph_selection"],
            n_fhr_st_channels=EXPECTED_ST,
            n_cross_phase_channels=masks["n_cross"],
            n_up_st_channels=EXPECTED_ST,
            up_ph_selection=None,  # the mistake being guarded against
        )
        try:
            _run_writer(cnp, path, masks)
        except ValueError as exc:
            _require("up_ph" in str(exc), f"error does not name up_ph: {exc}")
            return
        raise AssertionError("expected a ValueError for the missing up_ph")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("a mask built for a different filter bank is rejected")
def _check_pair_axis_mismatch(cnp: Any) -> None:
    # The real desync risk: masks built against one filter bank, applied to
    # another. Left unchecked this raises IndexError inside the per-record
    # handler, so every record fails identically, the run reports success and
    # an empty dataset ships as if validated.
    #
    # Simulated by building the masks at Q=8 (3081 pairs) and running the
    # writer at Q=4 (903 pairs) — exactly what a half-applied SCATTERING_Q
    # change would produce.
    tmp = tempfile.mkdtemp()
    original_q = cnp.SCATTERING_Q
    try:
        cnp.SCATTERING_Q = 8
        stale = cnp.compute_scattering_masks(SIGNAL_LENGTH, scattering_T=16)
        cnp.SCATTERING_Q = original_q  # writer now builds a Q=4 bank

        path = os.path.join(tmp, "wrong_bank.hdf5")
        cnp.create_initial_hdf5(
            path=path,
            len_signal=SIGNAL_LENGTH,
            len_sequence=SEQUENCE_LENGTH,
            fhr_ph_selection=stale["fhr_ph_selection"],
            n_fhr_st_channels=EXPECTED_ST,
            n_cross_phase_channels=stale["n_cross"],
            n_up_st_channels=EXPECTED_ST,
            up_ph_selection=stale["up_ph_selection"],
        )
        try:
            _run_writer(cnp, path, stale)
        except ValueError as exc:
            _require(
                "pair axis" in str(exc).lower(),
                f"error does not describe a pair-axis mismatch: {exc}",
            )
            return
        raise AssertionError("expected a ValueError for the pair-axis mismatch")
    finally:
        cnp.SCATTERING_Q = original_q
        shutil.rmtree(tmp, ignore_errors=True)


@check("an empty band selection is rejected with a readable message")
def _check_empty_selection_rejected(cnp: Any) -> None:
    # A band narrower than the widest harmonic step matches no pair; left
    # unchecked h5py rejects the zero-width chunk with a message naming
    # neither the field nor the band.
    original = cnp.UP_PHASE_BAND_HZ
    try:
        cnp.UP_PHASE_BAND_HZ = (0.02, 0.03)  # spans < one octave
        try:
            cnp.compute_scattering_masks(SIGNAL_LENGTH, scattering_T=16)
        except ValueError as exc:
            _require("up_ph" in str(exc), f"error does not name up_ph: {exc}")
            _require("empty" in str(exc).lower(), f"unclear message: {exc}")
            return
        raise AssertionError("expected a ValueError for the empty selection")
    finally:
        cnp.UP_PHASE_BAND_HZ = original


@check("stats calculator adapts to the new widths")
def _check_stats_agnostic(cnp: Any) -> None:
    from hdf5_dataset.calculate_dataset_stats import DatasetStatsCalculator

    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "smoke.hdf5")
        _build_file(cnp, path, _masks(cnp))
        stats = DatasetStatsCalculator(device="cpu").calculate_stats(
            [path], batch_size=2, progress_bar=False
        )
        for field, expected in (
            ("fhr_st", EXPECTED_ST),
            ("fhr_ph", EXPECTED_FHR_PH),
            ("fhr_up_ph", EXPECTED_CROSS),
            ("up_st", EXPECTED_ST),
            ("up_ph", EXPECTED_UP_PH),
        ):
            _require(field in stats, f"{field} missing from computed stats")
            got = int(stats[field]["n_channels"])
            _require(
                got == expected,
                f"{field}: stats report {got} channels, expected {expected}",
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    """Run every registered check and report a summary.

    Returns:
        Process exit code: 0 when all checks pass, 1 otherwise.
    """
    cnp = _import_pipeline()
    print(f"selection: k_steps={cnp.PHASE_HARMONIC_K_STEPS} "
          f"fhr_band={cnp.FHR_PHASE_BAND_HZ} Hz "
          f"up_band={cnp.UP_PHASE_BAND_HZ} Hz\n")

    # The layout a build would log, for both variants, before any check runs. Printed and not
    # asserted: the pytest suite already asserts the layout dict against the channel plan, and a
    # third copy of those numbers here would be one more place for them to go stale. What this
    # gives the operator is the same lines a production run prints, without a production run.
    for transform in cnp.TRANSFORMS:
        layout = cnp.describe_layout(
            cnp.compute_scattering_masks(SIGNAL_LENGTH, scattering_T=16, transform=transform)
        )
        for line in cnp.format_layout(layout):
            print(line)
        print()

    failures = 0
    for name, fn in _CHECKS:
        try:
            fn(cnp)
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL  {name}\n      {type(exc).__name__}: {exc}")
            if not isinstance(exc, AssertionError):
                traceback.print_exc()
        else:
            print(f"ok    {name}")

    total = len(_CHECKS)
    print(f"\n{total - failures}/{total} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
