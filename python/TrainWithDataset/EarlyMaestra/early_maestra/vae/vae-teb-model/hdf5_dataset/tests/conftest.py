"""Shared fixtures for the dataset tests: the two banks, real raw signals, and the pipeline module.

Why a committed fixture
-----------------------
``output/`` is git-ignored, so ``output/hie_cs.hdf5`` exists on the machine it was built on and
nowhere else -- not on a clean checkout, and not on the production box, which is precisely where a
build gate matters. A suite whose data-dependent tests all skip there is green and proves nothing.

``data/causal_fixture.hdf5`` therefore holds eight real segments of raw ``fhr``/``up`` at $5280$
float32 samples, extracted verbatim from a production shard -- one segment per GUID, spread across
the file, every one at mean weight $1.0$ so no padded or interpolated region is included. It is
$336$ KB and is tracked (``.gitignore`` un-ignores this ``data/`` directory the same way it does
``teb_vae/lag_attn_rws/tests/data/``). Its source shard and row indices are recorded in its own
root attributes, so it can be regenerated from those two facts alone.

:func:`raw_segments` **raises** rather than skips when the fixture is missing. That is the guard
against the failure mode above: data-dependent coverage cannot quietly vanish, because the data it
depends on is in the repository. The full shard remains available as a higher-fidelity input
through :data:`requires_shard`, which is the only thing here that may skip.

Paths anchor on ``__file__``, never on the working directory, so pytest can be invoked from
anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import pytest
import torch

# hdf5_dataset/tests/conftest.py -> parents[0]=tests, [1]=hdf5_dataset, [2]=repo root. The absolute
# ``hdf5_dataset.*`` imports below resolve from the repository root whichever directory pytest was
# invoked from.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hdf5_dataset.causal_scattering import (  # noqa: E402
    DECIMATION,
    N_RAW,
    SOURCE_PHASE_BAND_HZ,
    TARGET_PHASE_BAND_HZ,
    CausalBank,
    CausalChannelPlan,
    FilterBank,
    build_causal_bank,
    build_channel_plan,
    build_filter_bank,
    selected_pairs,
)
from hdf5_dataset.smoke_check_channel_selection import _import_pipeline  # noqa: E402

#: The committed raw signals. Present by construction; see the module docstring.
FIXTURE_PATH = Path(__file__).resolve().parent / "data" / "causal_fixture.hdf5"

#: The full 339-segment shard, when the machine happens to have it. Optional by design.
SHARD_PATH = Path(_REPO_ROOT) / "output" / "hie_cs.hdf5"

#: The published per-channel causal measurements, when the machine that produced them is this one.
#: Also git-ignored, so every pin taken from it is duplicated as a hand-computed value.
MEASUREMENTS_PATH = Path(_REPO_ROOT) / "output" / "causal_scattering" / "per_channel.csv"

#: Gate for the higher-fidelity tests that want the whole shard rather than eight segments. The
#: only skips in this package: everything else runs off the committed fixture.
requires_shard = pytest.mark.skipif(
    not SHARD_PATH.exists(), reason=f"no shard at {SHARD_PATH}"
)

requires_measurements = pytest.mark.skipif(
    not MEASUREMENTS_PATH.exists(), reason=f"no measurements at {MEASUREMENTS_PATH}"
)

#: The build runs on a GPU and its round-off is not the CPU's, so what agreement the device
#: actually delivers is worth measuring where a device exists -- which includes the production box.
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


def scale_relative_errors(
    produced: np.ndarray, expected: np.ndarray
) -> Tuple[float, float]:
    r"""Agreement between two coefficient blocks, normalised by the **block's** own scale.

    $$E_{\infty} = \frac{\max_t |a_t - b_t|}{\max(\max_t |b_t|,\ 10^{-30})},
      \qquad
      E_{2} = \frac{\lVert a - b \rVert_2}{\max(\lVert b \rVert_2,\ 10^{-30})}$$

    A **pointwise** relative error $\max_t |a_t - b_t| / |b_t|$ is the wrong instrument here and
    would make any gate built on it meaningless: the phase blocks are signed and cross zero
    constantly, so a coefficient that is numerically zero produces an unbounded ratio from a
    difference of no consequence. Normalising by the block's scale gives a near-zero coefficient a
    small numerator instead of a large ratio. $E_2$ is the norm-based companion that catches a
    diffuse error $E_\infty$ would miss.

    $E_\infty$ is the form ``compare_causal_scattering.py`` already measures with; this is the
    shared implementation, so the gate and the published comparison cannot mean different things
    by "relative error".

    Note:
        The two are normalised by **different** denominators -- a maximum against a norm -- so
        they are not on a common scale and $E_2$ is not systematically the smaller. On the stored
        blocks, whose energy is spread over hundreds of steps rather than concentrated at the
        maximum, $E_2 \approx 1.3\,E_\infty$.

    Args:
        produced: The block under test.
        expected: The reference block, same shape.

    Returns:
        ``(E_inf, E_2)``.
    """
    difference = np.asarray(produced, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    reference = np.asarray(expected, dtype=np.float64)
    e_inf = float(np.abs(difference).max() / max(float(np.abs(reference).max()), 1e-30))
    e_2 = float(
        np.linalg.norm(difference.ravel()) / max(float(np.linalg.norm(reference.ravel())), 1e-30)
    )
    return e_inf, e_2


@pytest.fixture(scope="module")
def bank() -> FilterBank:
    """The production two-sided filter bank, rebuilt."""
    return build_filter_bank()


@pytest.fixture(scope="module")
def causal_bank(bank: FilterBank) -> CausalBank:
    """The causal gammatone bank at the shipped kernel length, matched to *bank*."""
    return build_causal_bank(bank)


@pytest.fixture(scope="module")
def phase_pairs(bank: FilterBank) -> Dict[str, np.ndarray]:
    """The two stored phase-harmonic selections, rebuilt from the documented rule.

    Returns:
        ``{'fhr_ph': (66, 2), 'up_ph': (15, 2)}`` of $(i, j)$ filter indices.
    """
    return {
        "fhr_ph": selected_pairs(TARGET_PHASE_BAND_HZ, bank),
        "up_ph": selected_pairs(SOURCE_PHASE_BAND_HZ, bank),
    }


@pytest.fixture(scope="module")
def channel_plan(
    causal_bank: CausalBank, phase_pairs: Dict[str, np.ndarray]
) -> Dict[str, CausalChannelPlan]:
    """The stored causal channel plan: kept channels, warm-up and delay, per block."""
    return build_channel_plan(causal_bank, phase_pairs["fhr_ph"], phase_pairs["up_ph"])


@pytest.fixture(scope="module")
def raw_segments() -> Dict[str, np.ndarray]:
    """Eight real ``fhr``/``up`` segments, ``(8, 5280)`` float32 each, exactly as stored.

    Kept at float32 because that is what the pipeline transforms and what the dataset stores; the
    numpy reference casts to float64 itself, which is lossless in that direction.

    Returns:
        ``{'fhr': (8, 5280) float32, 'up': (8, 5280) float32}``.

    Raises:
        FileNotFoundError: If the committed fixture is missing, which would otherwise turn every
            data-dependent test in this package into a silent skip.
    """
    if not FIXTURE_PATH.exists():
        raise FileNotFoundError(
            f"the committed raw-signal fixture is missing from {FIXTURE_PATH}. It is tracked in "
            f"git; restore it rather than skipping, or the data-dependent tests test nothing."
        )
    with h5py.File(FIXTURE_PATH, "r") as handle:
        return {
            "fhr": np.asarray(handle["fhr"][:], dtype=np.float32),
            "up": np.asarray(handle["up"][:], dtype=np.float32),
        }


@pytest.fixture(scope="module")
def fixture_provenance() -> Dict[str, Any]:
    """The fixture's own record of where it came from: source shard path and row indices."""
    with h5py.File(FIXTURE_PATH, "r") as handle:
        return dict(handle.attrs)


@pytest.fixture(scope="module")
def pipeline() -> Any:
    """The dataset writer module, imported outside the production environment.

    Reuses ``smoke_check_channel_selection._import_pipeline``, which stubs the prod-only
    ``early_maestra`` adaptor and aliases the ``Variational_AutoEncoder.seqvae_teb`` import path
    onto this package. One shim, two callers -- a second copy here would be free to drift from the
    layout the production box actually has.
    """
    return _import_pipeline()


@pytest.fixture(scope="module")
def masks(pipeline: Any) -> Dict[str, Any]:
    """The shipped channel selections at the production geometry, on the CPU.

    Built through the pipeline's own resolver rather than restated, so a test's idea of the widths
    is the one a production run would use rather than a second set that happens to agree today.
    """
    return pipeline.compute_scattering_masks(
        N_RAW, scattering_T=DECIMATION, device=torch.device("cpu")
    )


@pytest.fixture(scope="module")
def causal_masks(pipeline: Any) -> Dict[str, Any]:
    """The same selections, resolved for the causal variant, with its channel plan."""
    return pipeline.compute_scattering_masks(
        N_RAW, scattering_T=DECIMATION, device=torch.device("cpu"), transform="causal"
    )


@pytest.fixture(scope="module")
def st_model(pipeline: Any) -> Any:
    """The two-sided transform the writer builds, at the geometry the masks were built against."""
    return pipeline.KymatioPhaseScattering1D(
        J=11, Q=pipeline.SCATTERING_Q, T=DECIMATION, shape=N_RAW,
        device=torch.device("cpu"), tukey_alpha=None, max_order=1,
    )
