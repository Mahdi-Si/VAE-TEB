r"""Run every instrument at several seeds and report power and a false-positive rate.

Run from the repository root:

.. code-block:: bash

    python -m teb_vae.lag_slot_transformer_cfs.instruments.campaign --output instruments.json

From an IDE's Run button, with no command line: fill in ``RUN_ARGS`` at the bottom of this file.
Nothing is required; the shipped generators at the shipped seeds are the campaign this pass exists
to run.

**What one run does.** It builds one generator's segments at a seed and splits them three ways. It
fits a small model of this exact architecture on the first split, stops it on the second under the
matched full-branch block score, and scores the predictive gap and every declared lag window's
suppression margin on the third through **this package's own** readouts. Then it applies the
criteria that were declared with the generator, and does the whole thing again at the next seed.
The rates are reported over all of them -- the runs that found nothing included.

**Three splits, and each one closes a way the harness could answer its own question.** A gap read on
the segments a model was fitted to measures memorisation, and the source branch has more capacity to
memorise with than the target-only branch -- so an in-sample instrument reports a gain on a
generator whose source carries nothing. A fit stopped at whichever step scored best on the split it
is then reported on has had its stopping point chosen by the number it reports. Neither is a subtle
effect at this scale: both were measured here before the splits existed, and the first turned every
control into a detection.

**The fit is deliberately reduced, and that is the honest limitation.** The forward, the objective,
its global reduction, the paired sampling, the shared decoder and every readout are the production
ones. The optimizer is plain AdamW at a fixed rate with a linear divergence ramp and no learning-rate
schedule, and there is no spike breaker and no loader. A rate measured here is a rate for that fit.
What it establishes is whether a readout can find a planted dependence at all and whether it claims
one where none exists, which no production run can answer about itself.

**Why the fit runs dense at stride one.** A generator produces a few dozen segments, and at a
training stride those give a few dozen supervised anchors per pass -- not enough for a source
pathway to leave its zero start in the seconds an instrument is allowed. Dense also removes the
training-against-evaluation stride difference from the reading, since the scoring pass decodes
densely regardless.

**Why the horizon weighting is uniform here.** The production objective decays its horizon weights,
which is right for a forecast whose far steps are the hardest -- and wrong for an instrument, where
the far steps are exactly where a delayed source is readable. A decayed weight would measure the
weighting as well as the plant.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: Repository root: ``teb_vae/lag_slot_transformer_cfs/instruments/campaign.py`` -> up four.
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

from teb_vae.lag_attn.eval.numerics import configure_numerics  # noqa: E402
from teb_vae.lag_attn.eval.report import json_safe  # noqa: E402
from teb_vae.lag_attn_cfs.eval.launch import resolve_launch_args  # noqa: E402
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask  # noqa: E402
from teb_vae.lag_slot_transformer_cfs.eval.predictive import (  # noqa: E402
    matched_predictive_scores,
)
from teb_vae.lag_slot_transformer_cfs.instruments import criteria as criteria_module  # noqa: E402
from teb_vae.lag_slot_transformer_cfs.instruments.generators import (  # noqa: E402
    GeneratedBatch,
    Generator,
    InstrumentGeometry,
    stored_feature_generators,
)
from teb_vae.lag_slot_transformer_cfs.nets import controls  # noqa: E402
from teb_vae.lag_slot_transformer_cfs.nets.model import SeqVaeLagResidualTrfCfs  # noqa: E402

#: The anchor geometry every fit and every score runs at: phase zero, stride one.
DENSE_ANCHOR_GEOMETRY: Tuple[int, int] = (0, 1)

#: Widths of the instrument model. Small in every dimension a plant does not live in, and left
#: alone in the two it does: the lag window and the horizon come from the generator's geometry.
MODEL_WIDTHS: Dict[str, Any] = {
    "d_model": 32,
    "d_z": 16,
    "decoder_hidden": 32,
    "encoder_conv_kernels": (3, 3),
    "encoder_conv_dilations": (1, 2),
    "encoder_num_heads": 4,
    "encoder_d_ff": 64,
    "target_attention_blocks": 2,
    "horizon_depth": 2,
    "horizon_kernel": 3,
    "horizon_attention_blocks": 1,
    "dropout": 0.0,
}

#: Optimizer steps per fit, the learning rate, and the fraction of them the divergence ramp climbs
#: over.
#:
#: The ramp matters more than either of the others: with the penalty at full weight from the first
#: step the cheapest way to satisfy it is an update of exactly zero, and every instrument would
#: report a source pathway that never left its start for a reason about the schedule.
FIT_STEPS = 500
FIT_LR = 3e-3
BETA_WARMUP_FRACTION = 0.3

#: Steps between selection checks, once the ramp has finished.
#:
#: Frequent enough that the stopping point is not itself a coarse grid, and rare enough that the
#: checks are a small share of a fit: each is one forward over the selection split against a
#: training step over a larger one.
SELECT_EVERY = 25

#: The divergence weight the ramp climbs to, and the prior scale rate beside it.
BETA_END = 1.0
BETA_PRIOR = 0.1

#: Seeds every generator is repeated at. Written out rather than a count, so a campaign is
#: reproducible from its own record and two campaigns are comparable run for run.
DEFAULT_SEEDS: Tuple[int, ...] = (11, 23, 37)


def build_model(generator: Generator, geometry: InstrumentGeometry) -> Any:
    """Construct one instrument's model at the campaign's widths and the generator's geometry.

    Args:
        generator: The instrument, for the extra constructor keywords its own data requires.
        geometry: The stored grid this instrument runs at.

    Returns:
        The constructed net, in training mode.
    """
    kwargs: Dict[str, Any] = {
        "sequence_length": geometry.sequence_length,
        "horizon": geometry.horizon,
        "warmup_period": geometry.warmup_period,
        "max_lag": geometry.max_lag,
        "c_y": geometry.n_target,
        "c_u": geometry.n_source,
        # Dense, for the reason this module's docstring gives: a training stride would leave a few
        # dozen supervised anchors per pass and the source pathway would not leave its zero start.
        "anchor_stride": 1,
        # Uniform over the horizon: the far steps are where a delayed source is readable, and a
        # decayed weight would measure the weighting as well as the plant.
        "horizon_weight_halflife_steps": None,
        # Fresh models, so the decoder and the prior scale start on the trivial predictor.
        "head_init_calibration": True,
        **MODEL_WIDTHS,
    }
    kwargs.update(generator.model_kwargs or {})
    return SeqVaeLagResidualTrfCfs(**kwargs)


def block_scores(model: Any, batch: GeneratedBatch) -> Dict[str, float]:
    """Both branches' unweighted block scores on one batch, without touching the graph.

    Used to stop a fit and to trace it. Unweighted, so the number is comparable across generators
    and across steps whatever the training weights were.

    Args:
        model: The net.
        batch: The streams to score on.

    Returns:
        ``{'base', 'full', 'divergence'}`` in nats per anchor.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(
            batch.y_st,
            batch.y_ph,
            batch.u_stream,
            anchor_phase=torch.zeros(batch.y_st.shape[0], dtype=torch.long),
            anchor_stride=1,
        )
        metrics = model.compute_loss(
            outputs,
            torch.cat([batch.y_st, batch.y_ph], dim=-1),
            weight=batch.weight,
            beta=1.0,
            beta_prior=BETA_PRIOR,
            likelihood="gaussian_nll",
        )["metrics"]
    model.train(was_training)
    return {
        "base": float(metrics["nll_base_block"]),
        "full": float(metrics["nll_full_block"]),
        "divergence": float(metrics["source_conditioned_kl_raw"]),
    }


def fit(
    model: Any,
    batch: GeneratedBatch,
    selection: GeneratedBatch,
    *,
    steps: int,
    seed: int,
) -> Dict[str, Any]:
    r"""Fit one model, stopping it on a split it is not scored on.

    The whole fit batch is one step's worth: a generator produces a few dozen segments and the
    instruments are small enough that mini-batching would add a sampling policy without changing
    what is measured.

    **The stopping rule is the reason this function holds three splits between it and the caller.**
    These models memorise: at this width, on this many segments, a fit run to a fixed step count
    drives its observation log-variance to the floor and both branches with it, and the branch with
    the extra latent degree of freedom overfits further. A fixed step count therefore measures the
    step count. The state kept here is the one that scored best on the **selection** split under the
    matched full-branch block score, which is the production selection criterion, and the verdicts
    are read on a third split that had no part in either.

    **The divergence ramp is what stops the source at zero from being the answer.** With the penalty
    at full weight from the first step, the cheapest update is exactly zero and every instrument
    would report a source pathway that never moved, for a reason about the schedule.

    Args:
        model: The constructed net.
        batch: The streams to fit on.
        selection: The streams to stop on, disjoint from both other splits.
        steps: The step budget.
        seed: Seed applied before the fit, so a run is a function of its seed.

    Returns:
        The fit's own trace: where it stopped, and what the two branches were doing at the start,
        at the stop and at the end of the budget.
    """
    torch.manual_seed(int(seed))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=FIT_LR, betas=(0.9, 0.95))
    target_features = torch.cat([batch.y_st, batch.y_ph], dim=-1)
    phase = torch.zeros(batch.y_st.shape[0], dtype=torch.long)
    warmup = max(int(steps * BETA_WARMUP_FRACTION), 1)

    best_step, best_full = 0, float("inf")
    best_state = copy.deepcopy(model.state_dict())
    first: Dict[str, float] = {}
    last: Dict[str, float] = {}
    for step in range(int(steps)):
        beta = BETA_END * min((step + 1) / warmup, 1.0)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch.y_st, batch.y_ph, batch.u_stream, anchor_phase=phase, anchor_stride=1)
        metrics = model.compute_loss(
            outputs,
            target_features,
            weight=batch.weight,
            beta=beta,
            beta_prior=BETA_PRIOR,
            likelihood="gaussian_nll",
        )["metrics"]
        metrics["total_loss"].backward()
        optimizer.step()

        # Checked only once the ramp has finished: before that the objective is still changing
        # under the fit, so two steps' selection scores are not scores of one criterion.
        if (step + 1) % SELECT_EVERY == 0 and beta >= BETA_END:
            scores = block_scores(model, selection)
            last = scores
            first = first or scores
            if scores["full"] < best_full:
                best_step, best_full = step + 1, scores["full"]
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return {
        "steps": int(steps),
        "selected_step": int(best_step),
        "selection_every": int(SELECT_EVERY),
        "selection_criterion": "matched full-branch block score on the selection split",
        "selection_first": first,
        "selection_last": last,
        "selection_best_full": None if best_full == float("inf") else float(best_full),
        "selected": block_scores(model, selection),
    }


def score(
    model: Any,
    batch: GeneratedBatch,
    windows: Mapping[str, Tuple[int, int]],
    settings: criteria_module.Criteria,
) -> Dict[str, Any]:
    """Score the matched gap and every declared window's suppression margin, per segment.

    Every arm is handed to one draw loop, so two arms with identical latent parameters give
    bitwise identical scores and a margin is a difference of predictions rather than of noise.

    Args:
        model: The fitted net.
        batch: The generated streams.
        windows: The declared lag partition.
        settings: The declared criteria, for the draw count.

    Returns:
        ``{'gap': [...], 'margins': {window: [...]}, 'kld': float}``, the first two per segment.
    """
    model.eval()
    phase, stride = DENSE_ANCHOR_GEOMETRY
    target_features = torch.cat([batch.y_st, batch.y_ph], dim=-1)
    local_fusion = str(getattr(model, "lag_fusion", "local")) == "local"

    with torch.no_grad():
        outputs = model(
            batch.y_st,
            batch.y_ph,
            batch.u_stream,
            anchor_phase=phase,
            anchor_stride=stride,
            return_proposals=True,
        )
        anchors, anchor_valid = outputs["anchor_index"], outputs["anchor_valid"]
        target = model._build_forecast_target(target_features, anchors)
        mask, _coverage = forecast_mask(
            model.scored_weight(batch.weight),
            model.geometry,
            coverage_floor=model.coverage_floor,
            anchors=anchors,
            anchor_valid=anchor_valid,
        )

        branches: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {
            "base": (outputs["mu_prior"], outputs["logvar_prior"]),
            "full": (outputs["mu_post"], outputs["logvar_post"]),
        }
        segments, n_anchors = outputs["mu_prior"].shape[0], outputs["mu_prior"].shape[1]
        for name, (low, high) in windows.items():
            removed = controls.band_lag_mask(model.n_lags, low, high)
            if local_fusion:
                suppressed = controls.suppressed_parameters(model, outputs, removed)
            else:
                suppressed = model(
                    batch.y_st,
                    batch.y_ph,
                    batch.u_stream,
                    anchor_phase=phase,
                    anchor_stride=stride,
                    selector=controls.band_selector(
                        removed, segments, n_anchors, dtype=batch.u_stream.dtype
                    ),
                )
            branches[f"suppress:{name}"] = (
                suppressed["mu_post"],
                suppressed["logvar_post"],
            )

        scored = matched_predictive_scores(
            model,
            branches,
            target,
            mask,
            likelihood="gaussian_nll",
            num_samples=int(settings.num_mc_samples),
            generator=torch.Generator().manual_seed(int(settings.seed)),
            persistence=outputs.get("persistence"),
        )

    contributing = scored["base"].contributing.to(torch.float64)
    per_segment_anchors = contributing.sum(dim=1)

    def per_segment(values: torch.Tensor) -> List[float]:
        """Average a per-anchor quantity within each segment, over its scored anchors."""
        pooled = (values.to(torch.float64) * contributing).sum(dim=1)
        return (pooled / per_segment_anchors.clamp_min(1.0)).tolist()

    base = per_segment(scored["base"].marginal)
    full = per_segment(scored["full"].marginal)
    margins: Dict[str, List[float]] = {}
    for name in windows:
        suppressed = per_segment(scored[f"suppress:{name}"].marginal)
        margins[name] = [value - matched for value, matched in zip(suppressed, full)]
    return {
        "gap": [first - second for first, second in zip(base, full)],
        "margins": margins,
        "scored_anchors": float(per_segment_anchors.sum()),
        "kld_per_anchor": float(
            (outputs["kld_per_anchor"].to(torch.float64) * contributing).sum()
            / per_segment_anchors.sum().clamp_min(1.0)
        ),
    }


def _slice(batch: GeneratedBatch, start: int, stop: int) -> GeneratedBatch:
    """Take a contiguous range of segments out of one generated batch.

    Args:
        batch: The generated streams.
        start: First segment, inclusive.
        stop: Last segment, exclusive.

    Returns:
        The sliced batch.
    """
    return GeneratedBatch(*(tensor[int(start) : int(stop)] for tensor in batch))


def run_once(
    generator: Generator,
    seed: int,
    settings: criteria_module.Criteria,
    *,
    geometry: InstrumentGeometry,
    steps: int,
) -> Dict[str, Any]:
    """Build, fit, score and decide one instrument at one seed.

    Args:
        generator: The instrument.
        seed: The run's seed, used for the build and the fit alike.
        settings: The declared criteria.
        geometry: The stored grid to run at.
        steps: Optimizer steps.

    Returns:
        The run's record, carrying the verdicts, the fit's own trace and the declared truth.
    """
    rows = generator.build(geometry, seed)
    # Fitted on one set of segments and scored on a disjoint one, drawn from the same process.
    # **This split is what makes a rate here a property of the readout rather than of the harness.**
    # A gap read on the segments a model was fitted to measures how well it memorised them, and the
    # source branch has strictly more capacity to memorise with than the target-only branch does --
    # so an in-sample instrument reports a positive gap on a generator whose source carries nothing,
    # which is exactly the false positive it exists to detect.
    fit_batch = _slice(rows, 0, geometry.segments)
    select_batch = _slice(
        rows, geometry.segments, geometry.segments + geometry.selection
    )
    score_batch = _slice(rows, geometry.segments + geometry.selection, geometry.rows)

    model = build_model(generator, geometry)
    trace = fit(model, fit_batch, select_batch, steps=steps, seed=seed)
    windows = criteria_module.lag_windows(geometry.n_lags, settings.window_width)
    measured = score(model, score_batch, windows, settings)

    relevance = criteria_module.relevance_verdict(measured["gap"], settings)
    # The band is passed in whether or not it is a criterion, which is the whole point of the
    # filter-bank instrument: the peak, the plant and the **distance** between them are the
    # measurement there, and a record that withheld the band could not state it. What the flag
    # decides is only whether a pass or a fail is recorded.
    recovery = criteria_module.recovery_verdict(
        measured["margins"],
        windows,
        generator.truth.direct_support,
        settings,
        graded=generator.truth.recovery_expected,
    )
    if not generator.truth.recovery_expected:
        recovery["not_a_criterion"] = generator.truth.note
    return {
        "generator": generator.name,
        "seed": int(seed),
        "checks": generator.checks,
        "truth": {
            "source_informative": bool(generator.truth.source_informative),
            "direct_support": (
                None
                if generator.truth.direct_support is None
                else list(generator.truth.direct_support)
            ),
            "causal": bool(generator.truth.causal),
            "recovery_expected": bool(generator.truth.recovery_expected),
            "note": generator.truth.note,
        },
        "geometry": {
            "sequence_length": geometry.sequence_length,
            "horizon": geometry.horizon,
            "warmup_period": geometry.warmup_period,
            "n_lags": geometry.n_lags,
            "fit_segments": geometry.segments,
            "selection_segments": geometry.selection,
            "scored_segments": geometry.holdout,
            "target_channels": geometry.n_target,
            "source_channels": geometry.n_source,
        },
        "windows": {name: list(span) for name, span in windows.items()},
        "fit": trace,
        "relevance": relevance,
        "recovery": recovery,
        "kld_per_anchor": measured["kld_per_anchor"],
        "scored_anchors": measured["scored_anchors"],
    }


def all_generators(geometry: InstrumentGeometry, *, include_raw: bool) -> Dict[str, Generator]:
    """Every instrument the campaign runs.

    Args:
        geometry: The stored grid the synthetic instruments run at.
        include_raw: Whether to resolve the filter-bank instrument, which imports the feature
            pipeline and runs it once for its channel plan. Separable because that import is by
            far the heaviest thing here and a campaign over the synthetic instruments alone is a
            legitimate thing to want.

    Returns:
        ``{name: Generator}``.
    """
    generators = dict(stored_feature_generators(geometry))
    if include_raw:
        from teb_vae.lag_slot_transformer_cfs.instruments.raw_process import raw_instrument

        instrument = raw_instrument()
        generators[instrument.name] = instrument
    return generators


def run_campaign(
    geometry: InstrumentGeometry,
    settings: criteria_module.Criteria,
    *,
    seeds: Sequence[int],
    steps: int,
    include_raw: bool,
    only: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run every instrument at every seed and aggregate the rates.

    Args:
        geometry: The stored grid the synthetic instruments run at.
        settings: The declared criteria.
        seeds: Seeds each instrument is repeated at.
        steps: Optimizer steps per fit.
        include_raw: Whether to run the filter-bank instrument.
        only: Run these instruments alone, or ``None`` for all of them.

    Returns:
        The campaign record: the criteria, every run, and the rates.

    Raises:
        ValueError: If ``only`` names an instrument that does not exist, which would otherwise run
            a smaller campaign than the caller asked for and report its rates as the campaign's.
    """
    generators = all_generators(geometry, include_raw=include_raw)
    if only is not None:
        unknown = sorted(set(only) - set(generators))
        if unknown:
            raise ValueError(
                f"no instrument named {unknown}; the campaign holds {sorted(generators)}. A "
                f"selection naming nothing would report a rate over the instruments that happened "
                f"to match."
            )
        generators = {name: generators[name] for name in only}

    records: List[Dict[str, Any]] = []
    for name, generator in generators.items():
        # The filter-bank instrument's grid is the bank's rather than a choice, so it runs at its
        # own; every other instrument runs at the campaign's.
        run_geometry = generator.geometry or geometry
        for seed in seeds:
            record = run_once(
                generator, int(seed), settings, geometry=run_geometry, steps=steps
            )
            records.append(record)
            logger.info(
                f"{name} seed {seed}: detected={record['relevance']['detected']} "
                f"recovered={record['recovery']['recovered']} "
                f"divergence={record['fit']['selected']['divergence']:.4f} "
                f"stopped at {record['fit']['selected_step']}"
            )
    return {
        "criteria": settings.as_record(),
        "seeds": [int(seed) for seed in seeds],
        "fit": {
            "steps": int(steps),
            "learning_rate": FIT_LR,
            "beta_warmup_fraction": BETA_WARMUP_FRACTION,
            "select_every": SELECT_EVERY,
            "optimizer": (
                "AdamW at a fixed rate, no scheduler, stopped on a selection split under the "
                "matched full-branch block score"
            ),
            "note": (
                "The forward, the objective, its global reduction, the paired sampling, the shared "
                "decoder and every readout are the production ones; the schedule is not. A rate "
                "reported here is a rate for this fit."
            ),
        },
        "runs": records,
        "rates": criteria_module.campaign_rates(records),
    }


def main(
    output: Optional[str] = None,
    seeds: Optional[str] = None,
    steps: Optional[int] = None,
    only: Optional[str] = None,
    skip_raw: Optional[bool] = None,
    sources: Optional[Mapping[str, str]] = None,
) -> int:
    """Run the campaign and write or print its record.

    Args:
        output: Where to write the record, or ``None`` to print it.
        seeds: Comma-separated seeds, or ``None`` for the shipped ones.
        steps: Optimizer steps per fit, or ``None`` for the shipped count.
        only: Comma-separated instrument names, or ``None`` for all of them.
        skip_raw: Skip the filter-bank instrument, or ``None`` to run it.
        sources: Where each launch value came from, recorded with the campaign so a run's
            provenance is recoverable from its own output rather than from a shell history.

    Returns:
        The process exit code, which is ``0`` whatever the rates are: a campaign that failed when
        an instrument found nothing would be a gate rather than a measurement.
    """
    settings = criteria_module.Criteria()
    numerics = configure_numerics(int(settings.seed))
    geometry = InstrumentGeometry()
    chosen = (
        DEFAULT_SEEDS
        if seeds is None
        else tuple(int(piece) for piece in str(seeds).split(",") if piece.strip())
    )
    selected = (
        None
        if only is None
        else tuple(piece.strip() for piece in str(only).split(",") if piece.strip())
    )

    record = run_campaign(
        geometry,
        settings,
        seeds=chosen,
        steps=int(FIT_STEPS if steps is None else steps),
        include_raw=not bool(skip_raw),
        only=selected,
    )
    record["run"] = {"numerics": numerics, "argument_sources": dict(sources or {})}

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
#: **Nothing here is required.** The shipped generators at the shipped seeds are the campaign this
#: pass exists to run, so an operator who wants exactly that types nothing and presses Run.
#:
#: This dict is a launch convenience and not a second declaration surface. Everything that decides
#: a **verdict** -- the interval, the resamples, the window width, the draw count -- lives in
#: ``criteria.Criteria`` and is written into the record, because a criterion injected from here
#: would appear in no artifact and could not be told from one chosen after the fact.
RUN_ARGS: Dict[str, Any] = {
    # Where to write the campaign record, or None to print it.
    "output": None,
    # Comma-separated seeds, or None for the shipped ones. More seeds is a tighter rate.
    "seeds": None,
    # Optimizer steps per fit, or None for the shipped count. Fewer is a faster and blunter
    # instrument, and the record says which was used.
    "steps": None,
    # Comma-separated instrument names to run alone, or None for all of them.
    "only": None,
    # True skips the filter-bank instrument, which is the only one importing the feature pipeline.
    "skip_raw": None,
}


def build_parser() -> argparse.ArgumentParser:
    """Build this entry point's own parser.

    No ``required=True`` and no non-``None`` default, for the reasons the evaluation entry points'
    parsers record: the first fires before the launch dict is read, and the second would make a
    dict entry unreachable while the operator edited it.

    Returns:
        The parser, whose ``dest`` set is also the valid key set for :data:`RUN_ARGS`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_slot_transformer_cfs.instruments.campaign",
        description="Run the synthetic instruments and report power and a false-positive rate.",
    )
    parser.add_argument("--output", default=None, help="Where to write the record.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds.")
    parser.add_argument("--steps", type=int, default=None, help="Optimizer steps per fit.")
    parser.add_argument("--only", default=None, help="Comma-separated instrument names.")
    parser.add_argument(
        "--skip-raw",
        dest="skip_raw",
        action="store_true",
        default=None,
        help="Skip the filter-bank instrument.",
    )
    return parser


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    """Parse, merge with :data:`RUN_ARGS`, and run.

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
