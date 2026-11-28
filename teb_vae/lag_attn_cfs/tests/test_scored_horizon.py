r"""The scored-horizon and scored-start rules read every configured shard, not the first one only."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pytest

from teb_vae.lag_attn_cfs.scored_horizon import (
    resolve_target_scored_horizon,
    resolve_target_scored_start,
)

HORIZON = 4


def _shard(path: Path, xi: np.ndarray, curve: Optional[np.ndarray] = None) -> str:
    """Write a shard holding only the two target blocks and the attributes the rules read."""
    with h5py.File(path, "w") as handle:
        st = handle.create_dataset("fhr_st", data=np.zeros((1, 2), dtype=np.float32))
        ph = handle.create_dataset("fhr_ph", data=np.zeros((1, xi.size), dtype=np.float32))
        ph.attrs["sel_xi_i_hz"] = xi
        if curve is not None:
            st.attrs["causal_novelty_curve"] = curve
    return str(path)


def _config(paths, **vae) -> dict:
    """A run configuration naming *paths* as its training shards."""
    return {
        "model_config": {"VAE_model": {"horizon": HORIZON, "c_y": 4, **vae}},
        "dataset_config": {"vae_train_datasets": list(paths)},
    }


def test_shards_that_agree_resolve_as_one_shard_does(tmp_path) -> None:
    xi = np.asarray([0.01, 0.2])
    paths = [_shard(tmp_path / f"{i}.hdf5", xi) for i in range(2)]
    config = _config(paths, target_phase_fast_cutoff_hz=0.1, target_phase_fast_horizon=1)

    assert resolve_target_scored_horizon(config) == (HORIZON, HORIZON, HORIZON, 1)


def test_a_later_shard_with_other_phase_legs_is_refused(tmp_path) -> None:
    paths = [
        _shard(tmp_path / "a.hdf5", np.asarray([0.01, 0.2])),
        _shard(tmp_path / "b.hdf5", np.asarray([0.2, 0.2])),
    ]
    config = _config(paths, target_phase_fast_cutoff_hz=0.1, target_phase_fast_horizon=1)

    with pytest.raises(ValueError, match="b.hdf5.*sel_xi_i_hz"):
        resolve_target_scored_horizon(config)


def test_a_later_shard_with_another_novelty_curve_is_refused(tmp_path) -> None:
    xi = np.asarray([0.01, 0.2])
    curve = np.linspace(0.0, 1.0, 2 * (HORIZON + 1)).reshape(2, HORIZON + 1)
    paths = [
        _shard(tmp_path / "a.hdf5", xi, curve),
        _shard(tmp_path / "b.hdf5", xi, curve[::-1]),
    ]
    config = _config(paths, target_min_novelty={"fhr_st": 0.5})

    with pytest.raises(ValueError, match="b.hdf5.*causal_novelty_curve"):
        resolve_target_scored_start(config)
    assert resolve_target_scored_start(_config(paths[:1], target_min_novelty={"fhr_st": 0.5}))


def test_a_scored_start_the_config_does_not_resolve_to_is_refused() -> None:
    """``target_scored_start`` has no config key of its own, so the preflight re-resolves
    ``target_min_novelty`` and compares vectors; a checkpoint built with a start against a config
    that sets no rule is refused, and two unset sides compare nothing."""
    from teb_vae.lag_attn_cfs.eval import preflight

    no_rule = {"model_config": {"VAE_model": {}}, "dataset_config": {}}
    with pytest.raises(preflight.EvalPreconditionUnmet, match="target_scored_start"):
        preflight.reconcile_with_checkpoint(
            no_rule, model_kwargs={"target_scored_start": [0, 2]}, hyper_parameters={},
            geometry_keys=(),
        )
    record = preflight.reconcile_with_checkpoint(
        no_rule, model_kwargs={}, hyper_parameters={}, geometry_keys=()
    )
    assert "target_scored_start" not in record["compared"]
