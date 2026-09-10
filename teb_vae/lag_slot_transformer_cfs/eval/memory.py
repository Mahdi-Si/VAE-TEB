r"""Peak memory and reassociation tolerance at the production geometry, measured rather than argued.

Run from the repository root:

.. code-block:: bash

    python -m teb_vae.lag_slot_transformer_cfs.eval.memory \
        --config teb_vae/lag_slot_transformer_cfs/configs/default.yaml

From an IDE's Run button, with no command line: fill in ``RUN_ARGS`` at the bottom of this file.
Nothing is required; the shipped production configuration is the default subject.

**This pass reads no data.** Every input is a seeded tensor of the configured shape, because both
quantities it measures are properties of the **geometry** -- the batch, the anchor set, the lag
window, the latent width -- and none of them is a property of what the coefficients contain. That is
what makes the measurement runnable before a single shard is in place, which is the order it is
needed in: the production batch size has to be settled before a multi-day fit is launched, not
after it dies.

**Two numbers, and both are owed rather than optional.**

*Peak allocated memory* per step. The architecture's largest tensors are the two proposal arrays,
$(B, A, L, d_z)$ each, and the specification is explicit that removing an attention does not
automatically make a model cheaper. Three step shapes are measured because they are genuinely
different workloads: a training step holds gradients and optimizer state; a dense evaluation step
holds neither but decodes five times as many anchors; and a dense evaluation step *retaining the
proposals* -- which is what the scoring pass does, since the suppression arms are built from them --
holds the largest array this architecture ever materialises.

*The reassociation tolerance*. Chunking changes the order the per-lag proposals are summed in, so it
changes the last digits of every downstream number. The tolerance that belongs in a comparison is
therefore a measurement at the widths a run will use, not a constant carried over from a fixture
geometry a hundred times smaller.

**An out-of-memory result is a measurement, not a failure.** It is recorded with the setting that
produced it and the sweep continues, because "the configured batch does not fit and this one does"
is precisely the finding the pass exists to produce.

**Without the configured shards the model is built ungated**, at the declared channel widths rather
than the resolved surviving ones, and every record says which it was. That is an upper bound on the
gated model rather than a different measurement: the gate only ever removes channels, and the
arrays that dominate the peak are indexed by anchor, lag and latent width, none of which the gate
touches.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Repository root: ``teb_vae/lag_slot_transformer_cfs/eval/memory.py`` -> up four.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Launched as a script -- which is what an IDE's Run button does -- this file's own directory goes
# on sys.path instead of the repository root, and every absolute import below fails before
# ``__main__`` is reached.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
from loguru import logger  # noqa: E402

from teb_vae.lag_attn.config import load_config  # noqa: E402
from teb_vae.lag_attn.eval.numerics import configure_numerics  # noqa: E402
from teb_vae.lag_attn.eval.report import json_safe  # noqa: E402
from teb_vae.lag_attn_cfs.eval.launch import resolve_launch_args  # noqa: E402
from teb_vae.lag_attn_cfs.eval.probe import resolve_device  # noqa: E402
from teb_vae.lag_attn_cfs.model_kwargs import warmup_model_kwargs  # noqa: E402
from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget  # noqa: E402
from teb_vae.lag_slot_transformer_cfs.nets.core import (  # noqa: E402
    pathway_parameter_counts,
)
from teb_vae.lag_slot_transformer_cfs.nets.model import SeqVaeLagResidualTrfCfs  # noqa: E402

#: The chunk grids swept for every step shape, as ``(anchor_chunk, lag_chunk)``.
#:
#: ``(None, None)`` is the shipped setting and the reference every other row is read against: it
#: builds the whole anchor-by-lag proposal array at once. The rest halve one axis at a time and then
#: both, so a reader can see which axis actually bounds the peak rather than inferring it. Fractions
#: of the resolved counts rather than absolute sizes, because the counts come from the configuration
#: and a grid of literals would describe a geometry no configuration has.
CHUNK_FRACTIONS: Tuple[Tuple[Optional[float], Optional[float]], ...] = (
    (None, None),
    (0.5, None),
    (None, 0.5),
    (0.25, 0.25),
)

#: The step shapes measured, and what each one holds.
#:
#: They are not variations of one workload. ``train`` holds activations for the backward pass plus
#: two optimizer state tensors per parameter; ``eval_dense`` holds neither but decodes the dense
#: anchor set, five times the training stride's; ``eval_dense_proposals`` additionally materialises
#: the per-lag arrays, which is what the scoring pass does because its suppression arms are built
#: from them and which is the largest tensor this architecture ever holds.
STEP_SHAPES: Tuple[str, ...] = ("train", "eval_dense", "eval_dense_proposals")

#: Bytes per mebibyte, for the reported figures.
_MIB = float(1024**2)

#: Seed for the pinned numeric environment. A constant rather than a setting: this pass measures a
#: geometry, and a peak that moved with a seed would be reporting something other than the geometry.
_NUMERICS_SEED = 20260909


def resolve_model_kwargs(config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Build the constructor keywords, gated against the configured shards where they exist.

    Args:
        config: The loaded run configuration.

    Returns:
        ``(kwargs, gated)``. *gated* is ``False`` when the shards could not be read, in which case
        the model is built at the declared channel widths -- an upper bound on the gated one rather
        than a different subject, since the gate only ever removes channels and the arrays that
        dominate the peak are indexed by anchor, lag and latent width.
    """
    kwargs = dict((config.get("model_config", {}) or {}).get("VAE_model", {}) or {})
    accepted = set(SeqVaeLagResidualTrfCfs.__init__.__code__.co_varnames)
    kwargs = {name: value for name, value in kwargs.items() if name in accepted}

    try:
        budget = resolve_warmup_budget(config)
        resolved = warmup_model_kwargs(budget, SeqVaeLagResidualTrfCfs)
    except Exception as error:  # noqa: BLE001 - an unreadable shard is a normal state here.
        logger.warning(
            f"could not resolve the warm-up budget ({type(error).__name__}: {error}); building "
            f"UNGATED at the declared channel widths. Every record below says so, and the figure "
            f"is an upper bound on the gated model rather than a different measurement."
        )
        return kwargs, False

    if not resolved:
        logger.warning("no warm-up budget is configured; building ungated at the declared widths")
        return kwargs, False
    kwargs.update(resolved)
    return kwargs, True


def synthetic_inputs(
    model: Any, batch: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Seeded tensors of the configured shape, standing in for one loader batch.

    The two target block widths come from the model's own declared boundary rather than from
    literals, so a run whose stored blocks were written at other widths is measured at those.

    The validity signal is all ones on purpose. It makes every anchor score, which is the **worst**
    case for memory and for the objective's reduction alike; a realistic mask would drop anchors and
    report a peak no run should plan against.

    Args:
        model: The constructed net.
        batch: Samples in the synthetic batch.
        device: Device to build them on.

    Returns:
        ``(y_st, y_ph, u_stream, target_features, weight)``.
    """
    steps, split = int(model.sequence_length), int(model.TARGET_BLOCK_SPLIT)
    generator = torch.Generator(device="cpu").manual_seed(20260909)

    def draw(*shape: int) -> torch.Tensor:
        """Draw one seeded tensor on the measurement device."""
        return torch.randn(*shape, generator=generator).to(device)

    y_st = draw(batch, steps, split)
    y_ph = draw(batch, steps, int(model.c_y) - split)
    u_stream = draw(batch, steps, int(model.c_u))
    return y_st, y_ph, u_stream, torch.cat([y_st, y_ph], dim=-1), torch.ones(
        batch, steps, device=device
    )


def _peak_mib(device: torch.device) -> Optional[float]:
    """Peak allocated memory since the last reset, in mebibytes, or ``None`` off CUDA.

    Args:
        device: The measurement device.

    Returns:
        The peak, or ``None`` where the allocator reports nothing.
    """
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device)) / _MIB


def measure_step(
    model: Any,
    inputs: Tuple[torch.Tensor, ...],
    *,
    shape: str,
    device: torch.device,
) -> Dict[str, Any]:
    r"""Run one step of the named shape and report what it peaked at.

    Args:
        model: The constructed net, already on ``device``.
        inputs: The synthetic batch.
        shape: One of :data:`STEP_SHAPES`.
        device: The measurement device.

    Returns:
        ``{'peak_mib', 'seconds', 'anchors', 'ok'}``, or ``{'ok': False, 'error': ...}`` on an
        allocator failure -- which is a measurement rather than a crash, and is what settles the
        batch size.

        The wall time is one step and is not a benchmark: it is the **compute** column a comparison
        of arms needs beside their parameter counts and their peaks, because two arms whose gaps
        differ by a little and whose step times differ by a lot are not the same offer. It is
        measured on the first and only run of each setting, so it carries whatever warm-up the
        device was in; read differences of a factor, never of a few percent.

    Raises:
        ValueError: On an unknown step shape.
    """
    if shape not in STEP_SHAPES:
        raise ValueError(f"unknown step shape {shape!r}; the shapes are {list(STEP_SHAPES)}")

    y_st, y_ph, u_stream, target_features, weight = inputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    try:
        if shape == "train":
            model.train()
            # An optimizer with real state, because two moment tensors per parameter are part of
            # what a training step holds and a measurement without them under-reports every run.
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            optimizer.zero_grad(set_to_none=True)
            phase = torch.zeros(y_st.shape[0], dtype=torch.long, device=device)
            outputs = model(y_st, y_ph, u_stream, anchor_phase=phase)
            loss = model.compute_loss(outputs, target_features, weight=weight)["metrics"][
                "total_loss"
            ]
            loss.backward()
            optimizer.step()
            anchors = int(outputs["anchor_index"].shape[1])
            del outputs, loss, optimizer
        else:
            model.eval()
            with torch.no_grad():
                outputs = model(
                    y_st,
                    y_ph,
                    u_stream,
                    anchor_phase=0,
                    anchor_stride=1,
                    return_proposals=shape == "eval_dense_proposals",
                )
            anchors = int(outputs["anchor_index"].shape[1])
            del outputs
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        return {"ok": False, "error": "out of memory", "detail": str(error).splitlines()[0]}

    # Synchronised before the clock is read, or the figure is how long it took to *queue* the work.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    peak = _peak_mib(device)
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"ok": True, "peak_mib": peak, "seconds": seconds, "anchors": anchors}


def wake_source_pathway(model: Any, *, seed: int = 20260909) -> bool:
    r"""Give the fusion head weights that are not zero, in place.

    **Without this the reassociation measurement is vacuous.** A constructed model's final source
    projection is exactly zero, so every update is exactly zero, and a sum of zeros is the same in
    any order -- the pass would report a tolerance of zero and a run at trained weights would
    exceed it on its first batch.

    The draw is seeded, so the tolerance is a property of the geometry rather than of whatever
    random state preceded the call. It addresses the head by the role every arm gives it, so the
    two fusions wake the same way; on a target-only arm there is no head to wake and there is also
    no summation whose order could matter, which the return value says rather than raising.

    Args:
        model: The constructed net, modified in place.
        seed: Seed applied immediately before the draw.

    Returns:
        Whether a source pathway was woken. ``False`` on a target-only arm, where the reassociation
        figure that follows is a measurement of nothing and is reported as such.
    """
    head = getattr(model, "proposal_head", None)
    if head is None:
        logger.warning(
            "this arm builds no source pathway, so there is no fusion to wake and no summation "
            "over lags whose order could move a result. The reassociation figures below are "
            "measurements of the target half alone."
        )
        return False
    torch.manual_seed(int(seed))
    projection = head.output_proj
    torch.nn.init.normal_(projection.weight, std=0.3)
    torch.nn.init.normal_(projection.bias, std=0.3)
    return True


def reassociation_tolerance(
    model: Any,
    inputs: Tuple[torch.Tensor, ...],
    *,
    anchor_chunk: int,
    lag_chunk: Optional[int],
) -> Dict[str, Any]:
    r"""How far a chunked dense forward moves from an unchunked one, at these widths.

    Chunking changes the order the per-lag proposals are summed in and nothing else, so the
    difference ought to be pure reassociation. It is measured rather than assumed because the figure
    scales with the number of terms in the sum: a tolerance taken at a fixture geometry with a
    handful of lags says nothing about one with the production window.

    **The repeat-run floor is measured first, and the two are reported together.** Two identical
    forwards can differ for reasons that have nothing to do with chunking -- a GPU picking a
    convolution algorithm per call, or, far larger here, the forward drawing a fresh latent -- and a
    chunking tolerance quoted without that floor beside it attributes all of it to the chunk size.
    The draw is pinned inside each run for exactly that reason, so a floor that is not zero is a
    device property and is reported as one.

    **The relative figure is taken against the reference's own scale rather than elementwise.** The
    forecast passes through zero, so an elementwise relative difference is unbounded there and says
    nothing; a test comparing two chunk settings wants an absolute tolerance, and this is the
    measurement that sets it.

    Args:
        model: The constructed net, with a source pathway that is not at its zero start.
        inputs: The synthetic batch.
        anchor_chunk: Anchors per chunk in the chunked arm.
        lag_chunk: Lags per chunk in the chunked arm, or ``None`` to hold that axis whole. It is
            ``None`` on an arm whose aggregation over lags is normalised rather than summed, where
            a chunk would compute a different model rather than the same one in a different order.

    Returns:
        The chunked-versus-unchunked differences, the repeat-run noise floor beside them, and the
        chunk sizes that produced them.
    """
    y_st, y_ph, u_stream = inputs[0], inputs[1], inputs[2]
    model.eval()

    def run() -> Dict[str, torch.Tensor]:
        """One dense forward under the model's current chunk settings, at a pinned latent draw.

        The seed is reset immediately before the forward because the forward **samples**: it draws
        one shared epsilon per anchor and both branches use it. Two forwards at different draws
        differ in every decoded coefficient, by far more than any summation order does, so an
        unpinned comparison would measure the reparameterisation and report it as a chunking
        tolerance.
        """
        torch.manual_seed(_NUMERICS_SEED)
        with torch.no_grad():
            return model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1)

    def differences(left: Dict[str, torch.Tensor], right: Dict[str, torch.Tensor]):
        """The update and forecast gaps between two forwards, and the forecast's own scale."""
        update = float((left["update_mean"] - right["update_mean"]).abs().max())
        forecast = float((left["mu_full"] - right["mu_full"]).abs().max())
        scale = float(left["mu_full"].abs().max())
        return update, forecast, scale

    original = (model.anchor_chunk, model.lag_chunk)
    try:
        model.anchor_chunk, model.lag_chunk = None, None
        whole = run()
        # The floor: the same computation twice, nothing changed. Anything the chunked arm shows
        # below this is the device rather than the summation order.
        repeat = run()
        model.anchor_chunk, model.lag_chunk = (
            int(anchor_chunk),
            None if lag_chunk is None else int(lag_chunk),
        )
        chunked = run()
    finally:
        model.anchor_chunk, model.lag_chunk = original

    floor_update, floor_forecast, scale = differences(whole, repeat)
    update, forecast, _scale = differences(whole, chunked)
    return {
        "anchor_chunk": int(anchor_chunk),
        "lag_chunk": None if lag_chunk is None else int(lag_chunk),
        "update_max_abs": update,
        "forecast_max_abs": forecast,
        "forecast_scale_max_abs": scale,
        "forecast_rel_to_scale": forecast / scale if scale > 0.0 else None,
        "repeat_run_update_max_abs": floor_update,
        "repeat_run_forecast_max_abs": floor_forecast,
        "source_pathway_awake": float(whole["update_mean"].abs().max()) > 0.0,
    }


def sweep(
    config: Dict[str, Any], device: torch.device, *, batch_sizes: Sequence[int]
) -> Dict[str, Any]:
    """Measure every step shape over the chunk grid, at each batch size, and the tolerance once.

    Args:
        config: The loaded run configuration.
        device: The measurement device.
        batch_sizes: Batch sizes to sweep, largest first is the natural order.

    Returns:
        The full record, ready for serialisation.
    """
    kwargs, gated = resolve_model_kwargs(config)
    model = SeqVaeLagResidualTrfCfs(**kwargs).to(device)
    n_lags = int(model.n_lags)
    dense_anchors = int(model.sequence_length) - int(model.warmup_period) - int(model.horizon)

    # The lag axis is chunkable only where the aggregation over it is a sum. An attention fusion
    # normalises over the whole axis, so a lag chunk would renormalise inside each chunk and
    # measure a different model rather than the same one in a different order -- which the
    # constructor refuses outright. The sweep therefore holds that axis whole on those arms and
    # every row says so, rather than dropping half the grid without a word.
    lag_chunkable = not model.source_disabled and str(model.lag_fusion) == "local"

    def chunk_pair(anchor_fraction, lag_fraction, anchors: int):
        """Turn a fraction of each axis into a chunk size, or ``None`` for the whole axis."""
        return (
            None if anchor_fraction is None else max(1, int(anchors * anchor_fraction)),
            None
            if lag_fraction is None or not lag_chunkable
            else max(1, int(n_lags * lag_fraction)),
        )

    geometry = {
        "gated": gated,
        "sequence_length": int(model.sequence_length),
        "declared_target_channels": int(model.c_y),
        "declared_source_channels": int(model.c_u),
        "decoder_out_channels": int(model.decoder_out_channels),
        "horizon": int(model.horizon),
        "latent_width": int(model.d_z),
        "model_width": int(model.d_model),
        "candidate_lags": n_lags,
        "training_stride": int(model.anchor_stride),
        "dense_anchors": dense_anchors,
        # Which arm this whole record describes, and its budget split. Without them a directory of
        # measurements from a mechanism-separating comparison is a set of numbers with no subjects:
        # the peak, the step time and the parameter count are all properties of an arm, and the arm
        # is exactly what a comparison is trying to attribute a difference to.
        "arm": {
            "source_stem": None if model.source_disabled else str(model.source_stem),
            "lag_fusion": None if model.source_disabled else str(model.lag_fusion),
            "source_disabled": bool(model.source_disabled),
            "mean_only_residual": bool(model.mean_only_residual),
            "source_values_withheld": bool(model.source_values_withheld),
            "source_scalar_lift": bool(model.source_scalar_lift),
        },
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "parameters_by_pathway": pathway_parameter_counts(model),
    }
    logger.info(f"measuring at {geometry}")

    rows: List[Dict[str, Any]] = []
    for batch in batch_sizes:
        inputs = synthetic_inputs(model, int(batch), device)
        for shape in STEP_SHAPES:
            for anchor_fraction, lag_fraction in CHUNK_FRACTIONS:
                anchor_chunk, lag_chunk = chunk_pair(anchor_fraction, lag_fraction, dense_anchors)
                model.anchor_chunk, model.lag_chunk = anchor_chunk, lag_chunk
                record = measure_step(model, inputs, shape=shape, device=device)
                rows.append(
                    {
                        "batch": int(batch),
                        "step": shape,
                        "anchor_chunk": anchor_chunk,
                        "lag_chunk": lag_chunk,
                        **record,
                    }
                )
                logger.info(f"{rows[-1]}")
        del inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model.anchor_chunk, model.lag_chunk = None, None
    # The tolerance is a property of the widths rather than of the batch, so it is measured once
    # and at the smallest swept batch -- the arithmetic per anchor is identical and the smallest
    # batch is the one that fits wherever this runs. The source pathway is woken first: at its zero
    # start every update is zero and a sum of zeros has no order, so the measurement would report
    # a tolerance no trained run could meet.
    awake = wake_source_pathway(model)
    tolerance_inputs = synthetic_inputs(model, int(min(batch_sizes)), device)
    tolerance = reassociation_tolerance(
        model,
        tolerance_inputs,
        anchor_chunk=max(1, dense_anchors // 4),
        lag_chunk=max(1, n_lags // 4) if lag_chunkable else None,
    )
    # What the figure above is a figure OF, recorded beside it. On an arm with no summation over
    # lags the number is a target-half measurement and reads as a suspiciously small tolerance
    # unless the record says why.
    tolerance["source_pathway_woken"] = awake
    tolerance["lag_axis_chunked"] = lag_chunkable
    logger.info(f"reassociation at the production widths: {tolerance}")

    return {"geometry": geometry, "peak_memory": rows, "reassociation": tolerance}


def main(
    config: Optional[str] = None,
    device: Optional[str] = None,
    output: Optional[str] = None,
    batch_sizes: Optional[str] = None,
    sources: Optional[Dict[str, str]] = None,
) -> int:
    """Measure and record.

    Args:
        config: Path to the configuration whose geometry is measured. ``None`` uses the shipped
            production configuration, which is the subject this pass exists for.
        device: Device string, or ``None`` to choose automatically.
        output: Where to write the record, or ``None`` to print it only.
        batch_sizes: Comma-separated batch sizes, or ``None`` for the configured training batch and
            its halves -- which is the sweep that settles a batch size rather than confirming one.
        sources: Where each launch value came from, recorded with the measurement.

    Returns:
        ``0``; an out-of-memory row is a measurement rather than a failure.
    """
    config_path = config or str(
        os.path.join(_REPO_ROOT, "teb_vae", "lag_slot_transformer_cfs", "configs", "default.yaml")
    )
    loaded = load_config(config_path)
    # Pinned before anything is built: without it the repeat-run floor below is the device picking a
    # convolution algorithm per call, and the chunking tolerance would inherit that noise.
    numerics = configure_numerics(_NUMERICS_SEED)
    resolved_device = resolve_device(device)
    if resolved_device.type != "cuda":
        logger.warning(
            "no CUDA device: the peak-memory columns will be empty. The geometry, the anchor "
            "counts and the reassociation tolerance are still measured, and those are device "
            "independent."
        )

    if batch_sizes is None:
        configured = int(
            ((loaded.get("general_config", {}) or {}).get("batch_size", {}) or {}).get("train", 1)
        )
        swept = [size for size in (configured, configured // 2, configured // 4) if size >= 1]
    else:
        swept = [int(piece) for piece in str(batch_sizes).split(",") if piece.strip()]

    record = sweep(loaded, resolved_device, batch_sizes=swept)
    record["run"] = {
        "config": config_path,
        "device": str(resolved_device),
        "device_name": (
            torch.cuda.get_device_name(resolved_device)
            if resolved_device.type == "cuda"
            else "cpu"
        ),
        "batch_sizes": swept,
        # What was actually in force, read back from global state rather than echoed, so a reader
        # can tell a tolerance measured under a pinned environment from one that was not.
        "numerics": numerics,
        "argument_sources": dict(sources or {}),
    }

    text = json.dumps(json_safe(record), indent=2)
    if output is not None:
        with open(str(output), "w", encoding="utf-8") as handle:
            handle.write(text)
        logger.info(f"wrote {output}")
    else:
        print(text)
    return 0


#: Values used when the module is launched with no command line -- i.e. an IDE's Run button. Keyed
#: by argparse ``dest``; a flag always wins over the entry here, per key.
#:
#: **Nothing here is required.** The shipped production configuration is the subject this pass
#: exists to measure, so an operator who wants exactly that types nothing and presses Run.
RUN_ARGS: Dict[str, Any] = {
    # The configuration whose geometry is measured. None uses the shipped production one.
    "config": None,
    # 'cuda:0', 'cpu', or None to choose automatically.
    "device": None,
    # Where to write the record, or None to print it.
    "output": None,
    # Comma-separated batch sizes, or None for the configured training batch and its halves.
    "batch_sizes": None,
}


def build_parser() -> argparse.ArgumentParser:
    """Build this entry point's own parser.

    No ``required=True`` and no non-``None`` default, for the reasons the scoring entry point's
    parser records: the first fires before the launch dict is read, and the second would make a
    dict entry unreachable while the operator edited it.

    Returns:
        The parser, whose ``dest`` set is also the valid key set for :data:`RUN_ARGS`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_slot_transformer_cfs.eval.memory",
        description="Measure peak memory and reassociation tolerance at a configured geometry.",
    )
    parser.add_argument("--config", default=None, help="Configuration to measure.")
    parser.add_argument("--device", default=None, help="'cuda:0', 'cpu', or omit to choose.")
    parser.add_argument("--output", default=None, help="Where to write the record.")
    parser.add_argument(
        "--batch-sizes", default=None, help="Comma-separated batch sizes to sweep."
    )
    return parser


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    """Parse, merge with :data:`RUN_ARGS`, and measure.

    Args:
        argv: Command-line arguments, or ``None`` for ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    values, sources = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        logger.info(f"changing working directory to the repo root: {_REPO_ROOT}")
        os.chdir(_REPO_ROOT)
    logger.info(f"argument sources: {sources}")
    return main(**values, sources=sources)


if __name__ == "__main__":
    sys.exit(_cli())
