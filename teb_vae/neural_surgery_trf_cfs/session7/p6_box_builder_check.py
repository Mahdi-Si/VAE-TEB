r"""E8.4: the builder's valid-box rule equals the box rule of the probes.

The probes :mod:`p6_box_events`, :mod:`p6_box_model` and :mod:`p6_box_pilot` select the box events
from ``up_quality`` bits with :func:`p6_box_events.box_runs` and remove them from the invalid mask
by hand. The dataset builder has the same rule as an option, ``QualityConfig.box_valid``. This
check runs the builder's ``ua_quality`` twice on every segment of split codes 0 and 2, with the
option off and on and with the same inputs, and compares:

1. **Rule.** The runs that the option turns valid are exactly the runs that
   :func:`p6_box_events.box_runs` selects from the bits of the run with the option off: those
   samples lose ``TRANSIENT`` and gain ``BOX``, and every other bit is unchanged.
2. **Level shifts.** The level shifts with the option on against the shifts with the option off
   that lie outside those runs. A difference is expected in a few segments and is reported, not
   failed: the samples of a valid box count as data in the builder's return test, so a
   neighbouring invalid span whose level the box comes back to is no longer a level shift. The
   probes kept that shift, which gives them slightly more contamination there.
3. **Stored events.** The share of the samples of the stored box events
   (``box_events.parquet``) that the option turns valid.

A re-run of ``ua_quality`` on a stored segment does not reproduce the stored bits in every
segment, because the file keeps the raw UA as ``float16`` and the builder's own FHR repair mask
is not stored. Comparison 3 therefore reports a share, and ``off_bits_differ_from_stored`` counts
the segments in which the re-run differs. Comparisons 1 and 2 use one set of inputs for both
runs. Comparison 1 decides ``passed``.

Outputs go to ``out/phase6/box_builder_check.json``.

Edit :data:`RUN_ARGS` and run this file from the IDE.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import p1_common as C  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7.p6_box_events import OUT6, box_runs  # noqa: E402

#: Run-button inputs. ``max_rows`` limits the rows per shard for a smoke test (``None`` = all).
RUN_ARGS: Dict[str, Any] = {"max_rows": None}


def main(max_rows: Any) -> Dict[str, Any]:
    """Run the comparisons and write the summary.

    Args:
        max_rows: Limit on rows per shard (``None`` = all).

    Returns:
        The summary dictionary.
    """
    from hdf5_dataset import signal_quality as sq

    off = sq.DEFAULT_CONFIG
    on = dataclasses.replace(off, box_valid=True)
    events = pd.read_parquet(OUT6 / "box_events.parquet")
    by_segment = {key: part for key, part in events.groupby(["cohort", "row"])}
    box_bit, transient = int(sq.UaBit.BOX), int(sq.UaBit.TRANSIENT)
    out: Dict[str, Any] = {"segments": 0, "off_bits_differ_from_stored": 0, "rule_differs": 0, "shifts_differ": 0,
                           "runs_turned_valid": 0, "stored_event_samples": 0, "stored_event_samples_turned_valid": 0,
                           "examples": []}
    for cohort in S.SHARDS:
        rows = C.split_rows(cohort)
        if max_rows is not None:
            rows = rows[: int(max_rows)]
        with h5py.File(S.SHARDS[cohort], "r") as handle:
            for row in rows:
                row = int(row)
                x = handle["up_raw"][row].astype(np.float64)
                stored = handle["up_quality"][row].astype(np.int64)
                fhr_raw = handle["fhr_raw"][row].astype(np.float64)
                present = np.repeat(handle["weight"][row] > 0, S.RAW_PER_STEP)[: x.size] & (fhr_raw != 0)
                hold = (handle["fhr_quality"][row] & int(sq.FhrBit.HOLD)) != 0
                epoch = float(handle["epoch"][row])
                onset = epoch - float(handle["second_stage_onset"][row])
                stage2 = (epoch + np.arange(x.size) / off.fs >= onset) if np.isfinite(onset) else None
                bits_off, steps_off = sq.ua_quality(x, present, hold, off, stage2)
                bits_on, steps_on = sq.ua_quality(x, present, hold, on, stage2)
                bits_off, bits_on = bits_off.astype(np.int64), bits_on.astype(np.int64)
                out["segments"] += 1
                out["off_bits_differ_from_stored"] += int(not np.array_equal(bits_off, stored))
                runs = [(int(r["start"]), int(r["stop"])) for r in box_runs(x, bits_off)]
                out["runs_turned_valid"] += len(runs)
                expected = bits_off.copy()
                for a, b in runs:
                    expected[a:b] = (expected[a:b] & ~transient) | box_bit
                if not np.array_equal(bits_on, expected):
                    out["rule_differs"] += 1
                    if len(out["examples"]) < 5:
                        out["examples"].append({"cohort": cohort, "row": row,
                                                "samples_differ": int((bits_on != expected).sum())})
                kept = [int(e.at) for e in steps_off if not any(a - S.FS <= e.at <= b + S.FS for a, b in runs)]
                if kept != [int(e.at) for e in steps_on]:
                    out["shifts_differ"] += 1
                    if len(out["examples"]) < 5:
                        out["examples"].append({"cohort": cohort, "row": row, "shifts_off": [int(e.at) for e in steps_off],
                                                "shifts_on": [int(e.at) for e in steps_on], "runs": runs})
                if (cohort, row) in by_segment:
                    part = by_segment[(cohort, row)]
                    for a, b in zip(part["start"], part["stop"]):
                        out["stored_event_samples"] += int(b - a)
                        out["stored_event_samples_turned_valid"] += int(((bits_on[int(a):int(b)] & box_bit) != 0).sum())
    out["stored_event_share_turned_valid"] = out["stored_event_samples_turned_valid"] / max(out["stored_event_samples"], 1)
    out["passed"] = bool(out["rule_differs"] == 0)
    OUT6.mkdir(parents=True, exist_ok=True)
    (OUT6 / "box_builder_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-rows", type=int, default=RUN_ARGS["max_rows"])
    args = parser.parse_args()
    main(args.max_rows)
