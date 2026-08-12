r"""What one causal segment-transform costs, per stage, in milliseconds and bytes.

Throughput was never the risk for a causal build; memory is. The whole chain runs in tens of
milliseconds per segment, so a full ten-fold build is under an hour of GPU time -- but the shipped
``scatter_batch_size`` multiplies the per-segment footprint by $128$, and whether that fits is a
property of the box, not of the transform.

**This module therefore reports per-segment figures and refuses to recommend a batch size.** The
dev box is a $12$ GB laptop GPU and the production box has $48$ GB A6000s; a batch size derived on
either one is wrong for the other, whereas GiB-per-segment lets whoever is running the build
divide by the memory they actually have. The one fixed cost -- the cached kernel spectra, paid once
per output file rather than once per batch -- is reported separately for the same reason.

Run it from the IDE's Run button: edit the constants below and press Run. It writes a JSON
artefact as well as printing, because a number that exists only in a terminal scrollback cannot be
compared against the next run's.

    .venv/Scripts/python.exe hdf5_dataset/benchmark_causal_torch.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

# Launched as a script, this file's own directory goes on sys.path instead of the repository root,
# so the absolute imports below would not resolve. Every other module in this package uses relative
# intra-package imports precisely to stay importable under the production package name; this one is
# an entry point, is not on the pipeline's import path, and takes the repo-root guard instead.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hdf5_dataset.causal_scattering import (  # noqa: E402
    DECIMATION,
    N_RAW,
    SOURCE_PHASE_BAND_HZ,
    TARGET_PHASE_BAND_HZ,
    build_causal_bank,
    build_channel_plan,
    build_filter_bank,
    selected_pairs,
)
from hdf5_dataset.causal_scattering_torch import CausalTorchBank  # noqa: E402

# =================================================================================================
# Run-button configuration
# =================================================================================================
#: Device to measure. ``'cuda:0'`` on a box with a GPU; ``'cpu'`` works and is slower. Explicit,
#: because the number this produces describes one device and is meaningless about another.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

#: Segments per forward pass. Several sizes are measured so the per-segment figure can be seen to
#: be flat -- if it is not, the batch is amortising a fixed cost and dividing by it would mislead.
BATCH_SIZES = (1, 4, 16)

#: Timed repeats per batch size, after one untimed warm-up pass.
REPEATS = 3

#: Where the artefact lands. Under ``output/``, which is git-ignored: this is a measurement of one
#: machine, not a fact about the repository.
OUTPUT_PATH = os.path.join("output", "causal_scattering", "torch_cost.json")


def _synchronise(device: torch.device) -> None:
    """Wait for the device to finish, so a timing measures work rather than queueing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_bytes(device: torch.device) -> Optional[int]:
    """Peak allocator watermark since it was last reset, or ``None`` off CUDA.

    The CPU allocator keeps no watermark, so this is honestly absent there rather than reported as
    zero -- a zero would read as "this stage is free".
    """
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None


def measure(
    device: str = DEVICE,
    batch_sizes: Sequence[int] = BATCH_SIZES,
    repeats: int = REPEATS,
    n_signal: int = N_RAW,
) -> Dict[str, Any]:
    r"""Time and measure each stage of the chain, per segment.

    The stages are the four the chain actually spends its time in: the $\psi$ convolution shared by
    both blocks, the two $\phi$ smoothings of the scattering block, and the two phase blocks, whose
    cost differs by an order of magnitude between $66$ pairs and $15$. They are measured
    individually and then as one pass, because the total is **not** their sum -- intermediates are
    freed between stages, so the peak of the whole chain is lower than the sum of the peaks.

    Signals are random rather than read from a fixture: cost depends on shape alone, and a
    benchmark that needs data cannot be run on a box that has none.

    Note:
        ``peak_bytes_per_segment`` divides the allocator's watermark by the batch, and the
        watermark includes the cached spectra, which are paid **once per file** and not once per
        segment. That is why the per-segment figure falls as the batch grows -- the fixed cost is
        being amortised, not the transform getting cheaper -- and why ``spectra_bytes`` is reported
        on its own. Read the largest batch's figure as the marginal cost of a segment.

    Args:
        device: Torch device string.
        batch_sizes: Segments per forward pass to measure.
        repeats: Timed repeats after one warm-up pass.
        n_signal: Raw segment length.

    Returns:
        A record with the fixed cost, one entry per batch size, and the environment it was taken in.
    """
    torch_device = torch.device(device)
    bank = build_filter_bank(n_signal)
    causal = build_causal_bank(bank)
    target_pairs = selected_pairs(TARGET_PHASE_BAND_HZ, bank)
    source_pairs = selected_pairs(SOURCE_PHASE_BAND_HZ, bank)
    plan = build_channel_plan(
        causal, target_pairs, source_pairs, sequence_length=n_signal // DECIMATION
    )
    torch_bank = CausalTorchBank(causal, torch_device, n_signal=n_signal)

    generator = torch.Generator().manual_seed(0)
    records: List[Dict[str, Any]] = []
    for batch_size in batch_sizes:
        fhr = 140.0 + 5.0 * torch.randn((batch_size, n_signal), generator=generator)
        up = 20.0 + 5.0 * torch.randn((batch_size, n_signal), generator=generator)
        fhr, up = fhr.to(torch_device), up.to(torch_device)

        stages = {
            "wavelet_responses": lambda: torch_bank.wavelet_responses(fhr),
            "scattering_block": lambda: torch_bank.scattering_block(fhr),
            "phase_block_fhr": lambda: torch_bank.phase_block(fhr, fhr, target_pairs),
            "phase_block_up": lambda: torch_bank.phase_block(up, up, source_pairs),
            "whole_chain": lambda: torch_bank.transform_batch(
                fhr, up, target_pairs, source_pairs, plan=plan
            ),
        }
        measured: Dict[str, Dict[str, Optional[float]]] = {}
        for name, stage in stages.items():
            with torch.no_grad():
                stage()  # warm-up: the first call allocates workspaces the timed ones reuse
                _synchronise(torch_device)
                if torch_device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(torch_device)
                start = time.perf_counter()
                for _ in range(repeats):
                    stage()
                _synchronise(torch_device)
                elapsed = time.perf_counter() - start
            peak = _peak_bytes(torch_device)
            measured[name] = {
                "ms_per_segment": 1000.0 * elapsed / (repeats * batch_size),
                "peak_bytes_per_segment": None if peak is None else peak / batch_size,
            }
        records.append({"batch_size": batch_size, "stages": measured})

    return {
        "device": str(torch_device),
        "device_name": (
            torch.cuda.get_device_name(torch_device) if torch_device.type == "cuda" else "cpu"
        ),
        "dtype": str(torch_bank.dtype),
        "n_signal": n_signal,
        "fft_length": torch_bank.fft_length,
        "n_filters": torch_bank.n_filters,
        "spectra_bytes": torch_bank.spectra_bytes,
        "widths": {name: block.n_channels for name, block in plan.items()},
        "repeats": repeats,
        "batches": records,
    }


def format_report(record: Dict[str, Any]) -> str:
    """Render a measurement as the operator-facing table.

    Args:
        record: Output of :func:`measure`.

    Returns:
        The report, ending in a newline.
    """
    lines = [
        f"device        {record['device']} ({record['device_name']})",
        f"dtype         {record['dtype']}, fft length {record['fft_length']}",
        f"fixed cost    cached spectra {record['spectra_bytes'] / 2 ** 20:.1f} MiB, "
        f"once per output file",
        f"widths        " + ", ".join(f"{k}={v}" for k, v in record["widths"].items()),
        "",
        f"{'batch':>6}  {'stage':<20} {'ms/segment':>12} {'GiB/segment':>13}",
    ]
    for batch in record["batches"]:
        for name, values in batch["stages"].items():
            peak = values["peak_bytes_per_segment"]
            gib = "n/a" if peak is None else f"{peak / 2 ** 30:.4f}"
            lines.append(
                f"{batch['batch_size']:>6}  {name:<20} {values['ms_per_segment']:>12.2f} "
                f"{gib:>13}"
            )
    lines.append("")
    lines.append(
        "GiB/segment is the allocator watermark divided by the batch, so it still carries the "
        "once-per-file spectra; read the largest batch's row as the marginal cost."
    )
    lines.append(
        "No batch size is recommended: divide the memory of the box you are building on by the "
        "whole-chain GiB/segment above."
    )
    return "\n".join(lines) + "\n"


def main(
    device: str = DEVICE,
    batch_sizes: Sequence[int] = BATCH_SIZES,
    repeats: int = REPEATS,
    output_path: str = OUTPUT_PATH,
) -> int:
    """Measure, print and write the artefact.

    Args:
        device: Torch device string.
        batch_sizes: Segments per forward pass.
        repeats: Timed repeats per batch size.
        output_path: Where the JSON artefact is written.

    Returns:
        Process exit code.
    """
    record = measure(device=device, batch_sizes=batch_sizes, repeats=repeats)
    print(format_report(record), end="")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(f"\nwrote {output_path}")
    return 0


def _cli() -> int:
    """Entry point: config paths are repo-root-relative, so run from there whatever the cwd is."""
    os.chdir(_REPO_ROOT)
    return main()


if __name__ == "__main__":
    sys.exit(_cli())
