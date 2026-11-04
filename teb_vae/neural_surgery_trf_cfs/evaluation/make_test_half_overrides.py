r"""Write an evaluation override file that keeps only the test-half recordings.

The edits were fitted on the recordings with $\operatorname{crc32}(\mathrm{guid}) \bmod 4 = 0$ and
selected on those with $\bmod 4 = 2$. The official evaluation must therefore see only
$\bmod 4 \in \{1, 3\}$; on ``acidosis_no_cs.hdf5`` that is 71 of 157 recordings. The dataset reads
the list through its ``allowed_guids`` filter.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import h5py  # noqa: E402

from teb_vae.neural_surgery_trf_cfs import common  # noqa: E402

RUN_ARGS = {
    "data": common.DATA,
    "name": "eval_overrides_test_half.yaml",
    "codes": (1, 3),
}


def test_guids(data, codes=(1, 3)):
    """Read every GUID of a shard and keep those whose split code is in ``codes``.

    Args:
        data: Path of the HDF5 shard.
        codes: Split codes to keep.

    Returns:
        ``(kept, n_total)``: the sorted kept GUIDs and the number of recordings in the shard.
    """
    with h5py.File(data, "r") as f:
        guids = sorted({g.decode() if isinstance(g, bytes) else str(g) for g in f["guid"][:]})
    return [g for g in guids if common.guid_hash(g) in codes], len(guids)


def main(a):
    """Write the override file into ``results/generated``.

    Args:
        a: The run arguments; see :data:`RUN_ARGS`.

    Returns:
        Path of the written file.
    """
    kept, total = test_guids(a["data"], a["codes"])
    path = common.write_overrides(a["name"], allowed_guids=kept)
    print(f"{total} recordings; {len(kept)} kept; wrote {path}")
    return path


if __name__ == "__main__":
    main(RUN_ARGS)
